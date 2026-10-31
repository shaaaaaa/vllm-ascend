# SPDX-License-Identifier: Apache-2.0
"""Replay a saved D receive through an existing D HTTP service and Mooncake."""

from __future__ import annotations

import argparse
import base64
import glob
import gzip
import json
import math
import re
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from decode_kv_compare import request_key
from multi_turn_kv_reuse_concurrent import Client, save_json
from pd_tensor_collect import password_auth, run_ssh
from prefill_kv_replay import child_argv, format_failure, http_json, seeded, validate_response_counts
from prefill_kv_seed import live_contract, token_digest

REPO = Path(__file__).resolve().parents[1]
PREFIX = "[D_KV_REPLAY]"
SAMPLING_FIELDS = frozenset(
    [
        "n",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "stop_token_ids",
        "ignore_eos",
        "max_tokens",
        "min_tokens",
        "logprobs",
        "prompt_logprobs",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "include_stop_str_in_output",
        "structured_outputs",
        "logit_bias",
        "allowed_token_ids",
        "repetition_detection",
    ]
)


def metadata(options: dict) -> dict:
    """Read only manifests and startup logs on the snapshot host; no device imports."""
    from decode_kv_replay import build_plan

    plan = build_plan([Path(options["snapshot_root"])], options["request_id"], Path("."), validate_bytes=False)
    paths = sorted({Path(p) for pattern in options["d_logs"] for p in glob.glob(pattern)})
    if not paths:
        raise ValueError("No current D startup logs found; set --d-logs to paths on the snapshot host/container")
    contract = live_contract(paths)
    skip_counts = set()
    for path in paths:
        for line in path.open(encoding="utf-8", errors="replace"):
            skip_counts.update(int(v) for v in re.findall(r"skip_last_n_tokens[\"']?\s*[:=]\s*(\d+)", line))
    if len(skip_counts) > 1:
        raise ValueError("D logs contain different skip_last_n_tokens settings; select the current D logs")
    # The connector's default is zero when the extra setting is omitted.
    return dict(
        request=plan["request"],
        runtime=plan["runtime"],
        received_tokens=plan["received_tokens"],
        tp_size=len(plan["workers"]),
        contract=contract,
        skip_last_n_tokens=next(iter(skip_counts), 0),
        logs=[str(p.resolve()) for p in paths],
    )


def collect_metadata(args, output: Path) -> dict:
    remote_repo = PurePosixPath(args.workspace) if args.snapshot_host else Path(args.workspace).resolve()
    options = dict(
        request_id=request_key(args.request_id),
        snapshot_root=args.snapshot_root or str(remote_repo / "decode-kv-snapshots"),
        d_logs=args.d_logs or [str(remote_repo / "tp4_dp4_logs" / "D_dp*.log")],
    )
    if not args.snapshot_host:
        result = metadata(options)
    else:
        encoded = base64.b64encode(json.dumps(options).encode()).decode("ascii")
        command = child_argv(args, "--metadata-child", encoded, script_name="decode_kv_replay.py")
        target = output / "metadata.json.gz"
        with target.open("wb") as stream, (output / "metadata.log").open("wb") as errors:
            child = run_ssh(args, args.snapshot_host, command, stdout=stream, stderr=errors, timeout=args.seed_timeout)
        if child.returncode:
            raise RuntimeError(f"Cannot read snapshot/D metadata; inspect {output / 'metadata.log'}")
        result = json.loads(gzip.decompress(target.read_bytes()))
    result["snapshot_root"] = options["snapshot_root"]
    return result


def request_payload(record: dict, model: str, tag: str, output_tokens: int | None) -> dict:
    """Use actual D token IDs and its saved bootstrap, never rebuild a text prompt."""
    sampling = dict(record["sampling"])
    if sampling.get("n", 1) != 1:
        raise ValueError("Replay requires n=1")
    defaults = dict(detokenize=True, flat_logprobs=False, bad_words=[])
    for name, default in defaults.items():
        value = sampling.pop(name, default)
        if value != default and not (name == "bad_words" and value is None):
            raise ValueError(f"Saved sampling option cannot be replayed by the completion API: {name}={value!r}")
    skip_cache = sampling.pop("skip_reading_prefix_cache", None)
    if skip_cache is not None and skip_cache != (sampling.get("prompt_logprobs") is not None):
        raise ValueError("Saved skip_reading_prefix_cache differs from the API's prompt-logprob policy")
    extra = dict(sampling.pop("extra_args", None) or {})
    transfer = dict(extra.pop("kv_transfer_params", None) or {})
    if extra or transfer.keys() - {"bootstrap_final_hidden"}:
        raise ValueError("Snapshot contains unsupported sampling extras or stale transport addresses")
    if sampling.keys() - SAMPLING_FIELDS:
        raise ValueError(f"Unsupported saved sampling fields: {sorted(sampling.keys() - SAMPLING_FIELDS)}")
    if output_tokens is not None:
        sampling["max_tokens"] = output_tokens
        sampling["min_tokens"] = min(sampling.get("min_tokens", 0), output_tokens)
    if not isinstance(sampling.get("max_tokens"), int) or sampling["max_tokens"] < 1:
        raise ValueError("Saved request has no finite output limit; pass --output-tokens")
    transfer["lmcache.tag.decode_replay"] = tag
    return dict(
        sampling,
        model=model,
        prompt=record["prompt_token_ids"],
        request_id="d-replay-" + tag,
        cache_salt="d-replay-" + tag,
        kv_transfer_params=transfer,
        add_special_tokens=False,
        stream=True,
        stream_options={"include_usage": True},
        return_token_ids=True,
    )


def prepare(args, meta: dict, models: dict, output: Path) -> tuple[dict, dict, int]:
    saved_model = meta["runtime"]["engine_options"]["model"]
    candidates = [m for m in models["data"] if m.get("root") == saved_model]
    if args.model_name:
        candidates = [m for m in candidates if m["id"] == args.model_name]
    if len(candidates) != 1:
        raise ValueError("D must serve the saved model path; use --model-name when it has multiple aliases")
    contract = meta["contract"]
    if contract.get("tp_size") != meta["tp_size"]:
        raise ValueError("Current D logs and snapshot TP sizes differ")
    master = args.master or contract.get("master")
    if not master or (contract.get("master") and master != contract["master"]):
        raise ValueError("Cannot determine the current D Mooncake master, or --master disagrees with D logs")
    received, prompt = meta["received_tokens"], meta["request"]["prompt_token_ids"]
    # A different D truncation policy changes the partial page's hash/key. Do
    # not turn that mismatch into a silent full-prefill replay.
    if len(prompt) - meta["skip_last_n_tokens"] != len(received):
        raise ValueError(
            f"D lookup frontier differs from snapshot: prompt={len(prompt)} "
            f"skip_last_n_tokens={meta['skip_last_n_tokens']} received={len(received)}; "
            "use the original D launch configuration"
        )
    tag = uuid.uuid4().hex
    payload = request_payload(meta["request"], candidates[0]["id"], tag, args.output_tokens)
    limit = candidates[0].get("max_model_len")
    if limit and len(prompt) + payload["max_tokens"] > limit:
        raise ValueError("Saved prompt plus requested output exceeds live D max_model_len")
    remote_repo = PurePosixPath(args.workspace) if args.snapshot_host else Path(args.workspace).resolve()
    plan = dict(
        contract=contract,
        model_path=saved_model,
        master=master,
        tp_size=meta["tp_size"],
        previous=request_key(args.request_id),
        cached_tokens=len(received),
        seed_mode="decode_receive",
        replay_tag=tag,
        prefix_digest=token_digest(received),
        previous_prompt_tokens=len(received),
        previous_prompt_digest=token_digest(received),
        snapshot_root=meta["snapshot_root"],
        workers=args.workers,
        seed_device=args.seed_device,
        local_hostname=args.local_hostname,
        store_bytes=None if args.store_gb is None else int(args.store_gb * 1024**3),
        release_file=str(remote_repo / "prefill-kv-replays" / "leases" / (tag + ".release")),
        hold_seconds=args.timeout + 120,
    )
    save_json(output / "request.json", payload)
    save_json(output / "seed_plan.json", plan)
    save_json(output / "source.json", meta)
    return payload, plan, min(len(received), len(prompt) - 1)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--request-id")
    cli.add_argument("--d-url", help="Already-running D API URL; not the PD proxy or P URL")
    cli.add_argument("--model-name", help="Served model alias; auto-select the alias for the saved model")
    cli.add_argument("--snapshot-host", help="SSH host containing the snapshot and current D startup logs")
    cli.add_argument("--snapshot-root", help="Defaults to WORKSPACE/decode-kv-snapshots on snapshot host")
    cli.add_argument("--workspace", default=str(REPO), help="vllm-ascend root inside the snapshot host/container")
    cli.add_argument("--d-logs", nargs="+", help="Current D log globs on snapshot host; default tp4_dp4_logs/D_dp*.log")
    cli.add_argument("--container")
    cli.add_argument("--remote-python", default=sys.executable)
    cli.add_argument("--master", help="Infer Mooncake master from current D logs")
    password = cli.add_mutually_exclusive_group()
    password.add_argument("--password", dest="ssh_password", help="Native OpenSSH askpass; no interactive prompt")
    password.add_argument("--password-env")
    cli.add_argument("--skip-host-key-check", action="store_true")
    cli.add_argument("--ssh-port", type=int)
    cli.add_argument("--identity-file", type=Path)
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--seed-device", type=int, default=0, help="Transport context only; no model is loaded")
    cli.add_argument("--local-hostname", help="Snapshot host transport IP; infer via route to master")
    cli.add_argument("--store-gb", type=float, help="Temporary seed CPU segment capacity; infer from full KV size")
    cli.add_argument("--timeout", type=float, default=7200)
    cli.add_argument("--seed-timeout", type=float, default=7200)
    cli.add_argument("--output-tokens", type=int, help="Keep saved D output limit unless overridden")
    cli.add_argument("--prepare-only", action="store_true", help="Read metadata/check API, without injection/inference")
    cli.add_argument("--metadata-child", help=argparse.SUPPRESS)
    return cli


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.metadata_child:
        options = json.loads(base64.b64decode(args.metadata_child))
        sys.stdout.buffer.write(gzip.compress(json.dumps(metadata(options)).encode()))
        return 0
    if not args.request_id or not args.d_url:
        raise ValueError("--request-id and --d-url are required; use the existing D API endpoint")
    if any(not math.isfinite(n) or n <= 0 for n in (args.timeout, args.seed_timeout, args.workers)):
        raise ValueError("Timeouts and workers must be positive and finite")
    if args.output_tokens is not None and args.output_tokens <= 0:
        raise ValueError("output-tokens must be positive")
    if args.store_gb is not None and (not math.isfinite(args.store_gb) or args.store_gb <= 0):
        raise ValueError("store-gb must be positive and finite")
    args.hosts = [args.snapshot_host] if args.snapshot_host else []
    args.password = args.password_per_host = False
    parent = REPO / "decode-kv-replays"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=request_key(args.request_id) + "-", dir=parent))
    report = dict(
        status="preparing",
        source_request=request_key(args.request_id),
        d_url=args.d_url,
        accuracy_verdict="not_assessed",
        scope="Saved received KV through existing D; original concurrent schedule is not reproduced.",
    )
    print(f"{PREFIX} artifacts={output}; using existing D; no model/server startup", flush=True)
    try:
        # Check HTTP before touching the large snapshot or creating a seed segment.
        models = http_json(args.d_url, "/v1/models", timeout=min(args.timeout, 30))
        with password_auth(args):
            meta = collect_metadata(args, output)
            payload, plan, expected = prepare(args, meta, models, output)
            report.update(expected_cached_tokens=expected, request_id=payload["request_id"])
            print(
                f"{PREFIX} prompt={len(payload['prompt'])} received={plan['cached_tokens']} "
                f"expected_cached={expected}; direct D={args.d_url}",
                flush=True,
            )
            if args.prepare_only:
                report["status"] = "prepared_not_run"
                return 0
            with seeded(args, plan, output) as owner:
                text, metrics = Client(args.d_url, args.timeout).complete(payload, output)
                print(f"{PREFIX} OUTPUT BEGIN\n{text}\n{PREFIX} OUTPUT END", flush=True)
                report.update(request_completed=True, metrics=metrics)
                if owner.poll() is not None:
                    raise RuntimeError("Seed owner exited during D request; replay is invalid")
                validation = validate_response_counts(metrics, len(payload["prompt"]), expected)
                report.update(count_validation=validation)
                report["status"] = "replay_unverified" if validation["missing"] else "replay_complete"
                print(
                    f"{PREFIX} status={report['status']} cached={metrics.get('cached_tokens')}/{expected} "
                    f"response_id={metrics.get('response_id')}",
                    flush=True,
                )
                if validation["missing"]:
                    print(f"{PREFIX} usage omitted cache/token counts; output saved, reuse is UNVERIFIED", flush=True)
        return 2 if report["status"] == "replay_unverified" else 0
    except BaseException as error:
        report.update(status="failed", error=format_failure(error))
        raise
    finally:
        save_json(output / "report.json", report)
