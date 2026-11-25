#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read previous raw KV files and replay one turn through an existing PD service.

No model/server is started. Replay requires the opt-in LMCache file reader. Run on the client/P host and use
--snapshot-hosts to find the previous request's raw KV across D hosts.
"""

from __future__ import annotations

import argparse
import base64
import glob
import gzip
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath

from decode_kv_compare import request_key
from kv_replay_discovery import candidate_hosts, select_snapshot
from kv_replay_files import attach_source, configure_plan, require_enabled, source_stats
from kv_replay_snapshot import PREFIX as FILE_PREFIX
from kv_replay_snapshot import live_contract, token_digest
from multi_turn_chat import CHAT_FORMAT, NATIVE_FORMAT, TEMPLATE_KWARGS, read_assistant, restore_native_tokens
from multi_turn_kv_reuse_concurrent import Client, read_text, save_json
from pd_tensor_collect import ASKPASS_PASSWORD_ENV, password_auth, remote_command, run_ssh

REPO = Path(__file__).resolve().parents[1]
PREFIX = "[PREFILL_REPLAY]"
PROXY_CONNECT_TIMEOUT_SECONDS = 10


def restore_turn(root: Path, session: int, turn: int) -> dict:
    """Reconstruct the exact text originally submitted, preserving whitespace."""
    if turn < 1:
        raise ValueError("Replay requires a follow-up round >= 1")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    sessions = {s["session_id"]: s for s in summary["sessions"]}
    if session not in sessions:
        raise ValueError(f"Session {session} does not exist in the run")
    directory = (root / sessions[session]["directory"]).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("Session directory is outside the run")
    history = read_text(directory / "session_prefix.txt") + read_text(root / "initial_prompt.txt")
    messages = [dict(role="user", content=history)]
    previous_messages = None
    measurements = {(r["session_id"], r["round"]): r for r in summary["warmups"] + summary["rounds"]}
    previous_text = None
    for current in range(turn + 1):
        folder = directory / "turns" / f"{current:03d}"
        request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        if current:
            previous_text = history
            answer = read_text(directory / "turns" / f"{current - 1:03d}" / "output.txt")
            question = read_text(folder / "added_input.txt")
            history += answer + question
            previous_messages = list(messages)
            assistant = (
                read_assistant(directory / "turns" / f"{current - 1:03d}")
                if request.get("prompt_format") == CHAT_FORMAT
                else dict(role="assistant", content=answer)
            )
            messages.extend((assistant, dict(role="user", content=question)))
        if len(history) != request["prompt_chars"]:
            raise ValueError(f"Reconstructed prompt length differs at round {current}; run files changed")
        row = measurements[session, current]
        if row["request_id"] != request["request_id"]:
            raise ValueError("Request metadata does not match run summary")
    cached = row.get("cached_tokens")
    if type(cached) is not int or cached <= 0:
        raise ValueError("Original follow-up must have a recorded, positive cached_tokens value")
    previous = measurements[session, turn - 1]
    payload = {
        key: value
        for key, value in request.items()
        if key not in ("session_id", "round", "prompt_chars", "prompt_source", "prompt_format", "prompt_token_count")
    }
    payload.update(prompt=history, request_id="replay-" + uuid.uuid4().hex)
    native = request.get("prompt_format") == NATIVE_FORMAT
    chat = request.get("prompt_format") == CHAT_FORMAT
    if native:
        payload["prompt"] = restore_native_tokens(directory, turn)
    elif chat:
        del payload["prompt"]
        payload["messages"] = messages
    return dict(
        payload=payload,
        previous_text=previous_text,
        native_messages=messages if native or chat else None,
        previous_messages=previous_messages if native or chat else None,
        previous_tokens=restore_native_tokens(directory, turn - 1) if native else None,
        previous=request_key(previous["response_id"]),
        current=request_key(row["response_id"]),
        original_metrics=row,
        previous_metrics=previous,
        original_output=read_text(directory / "turns" / f"{turn:03d}" / "output.txt"),
        cached_tokens=cached,
    )


def http_json(base: str, path: str, body: dict | None = None, timeout: float = 120) -> dict:
    base = base.rstrip("/").removesuffix("/v1")
    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def check_proxy_connection(base_url: str, timeout: float) -> None:
    """Check the listener before reading large KV files, without issuing inference.

    Proxy health/models routes differ between deployments. A TCP connection
    checks this failure mode without requiring any optional HTTP endpoint.
    It does not promise that P/D are healthy or that the proxy will stay up.
    """
    target = urllib.parse.urlsplit(base_url)
    if target.scheme not in ("http", "https") or not target.hostname:
        raise ValueError("--base-url must be an http(s) PD proxy URL")
    port = target.port or (443 if target.scheme == "https" else 80)
    try:
        with socket.create_connection((target.hostname, port), timeout=min(timeout, PROXY_CONNECT_TIMEOUT_SECONDS)):
            pass
    except OSError as error:
        raise RuntimeError(
            f"Cannot connect to PD proxy {target.hostname}:{port}; check --base-url and the proxy listener. "
            "KV file reading has not started."
        ) from error
    print(f"{PREFIX} PD proxy TCP reachable: {target.hostname}:{port}", flush=True)


def validate_response_counts(metrics: dict, prompt_tokens: int, cached_tokens: int) -> dict:
    """Missing usage is unverified, never a zero cache hit or a matching count."""
    expected = dict(prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
    missing = []
    for name, value in expected.items():
        actual = metrics.get(name)
        if actual is None:
            missing.append(name)
        elif type(actual) is not int or actual != value:
            label = "Cache" if name == "cached_tokens" else "Prompt token count"
            raise RuntimeError(f"{label} mismatch: got {actual}, expected {value}")
    return dict(status="unverified" if missing else "matched", missing=missing, expected=expected)


def prepare(args, output: Path) -> tuple[dict, dict]:
    restored = restore_turn(args.run_dir.resolve(), args.session, args.round)
    files = sorted({Path(p) for pattern in args.p_logs for p in glob.glob(pattern)})
    if not files:
        raise ValueError("No P startup logs found; --p-logs must name the NEW running P's logs")
    contract = live_contract(files)
    require_enabled(files, "P")
    model = restored["payload"]["model"]
    models = http_json(args.p_url, "/v1/models")["data"]
    selected = next((m for m in models if m["id"] == model), None)
    if not selected or not selected.get("root"):
        raise ValueError(f"Live P does not serve the original model alias {model!r}")
    if restored["native_messages"] is not None:

        def tokenize_messages(messages):
            return http_json(
                args.p_url,
                "/tokenize",
                dict(
                    model=model,
                    messages=messages,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                    chat_template_kwargs=restored["payload"].get("chat_template_kwargs", dict(TEMPLATE_KWARGS)),
                ),
            )["tokens"]

        tokens = tokenize_messages(restored["native_messages"])
        previous_tokens = tokenize_messages(restored["previous_messages"])
        if restored["previous_tokens"] is not None and (
            tokens != restored["payload"]["prompt"] or previous_tokens != restored["previous_tokens"]
        ):
            raise ValueError("Live native template/tokenizer differs from the archived input token IDs")
    else:
        tokens = http_json(args.p_url, "/tokenize", dict(model=model, prompt=restored["payload"]["prompt"]))["tokens"]
        previous_tokens = http_json(args.p_url, "/tokenize", dict(model=model, prompt=restored["previous_text"]))[
            "tokens"
        ]
    for ids, metrics in ((tokens, restored["original_metrics"]), (previous_tokens, restored["previous_metrics"])):
        if len(ids) != metrics["prompt_tokens"]:
            raise ValueError(
                "Live tokenization length differs from the original run; do not replay with this tokenizer"
            )
    cached = restored["cached_tokens"]
    if cached > len(previous_tokens) or tokens[:cached] != previous_tokens[:cached]:
        raise ValueError("Original cached frontier is not inside the unchanged previous prompt")
    tp = args.tp_size or contract.get("tp_size")
    if not tp:
        raise ValueError("Cannot infer TP size from P log labels; pass --tp-size")
    if args.output_tokens is not None:
        restored["payload"]["max_tokens"] = args.output_tokens
    save_json(output / "request.json", restored["payload"])
    save_json(output / "input_tail_tokens.json", dict(start=cached, end=len(tokens), token_ids=tokens[cached:]))
    save_json(
        output / "original.json",
        {
            k: v
            for k, v in restored.items()
            if k
            not in (
                "payload",
                "previous_text",
                "native_messages",
                "previous_messages",
                "previous_tokens",
            )
        },
    )
    remote_repo = PurePosixPath(args.workspace) if args.snapshot_host else REPO
    snapshot_root = args.snapshot_root or str(remote_repo / "decode-kv-snapshots")
    lease = uuid.uuid4().hex
    plan = dict(
        contract=contract,
        model_path=selected["root"],
        tp_size=tp,
        previous=restored["previous"],
        cached_tokens=cached,
        prefix_digest=token_digest(tokens[:cached]),
        previous_prompt_digest=token_digest(previous_tokens),
        previous_prompt_tokens=len(previous_tokens),
        snapshot_root=snapshot_root,
        workers=args.workers,
        release_file=str(remote_repo / "prefill-kv-replays" / "leases" / (lease + ".release")),
        hold_seconds=args.timeout + 120,
    )
    configure_plan(args, plan)
    save_json(output / "file_plan.json", plan)
    print(
        f"{PREFIX} session={args.session} round={args.round} prompt={len(tokens)} "
        f"expected_cached={cached} recompute={len(tokens) - cached} "
        f"previous={restored['previous']} current={restored['current']}",
        flush=True,
    )
    return restored, plan


def child_argv(args, *parts: str, script_name: str = "prefill_kv_replay.py") -> list[str]:
    if args.snapshot_host:
        return remote_command(
            args,
            args.snapshot_host,
            [
                args.remote_python,
                str(PurePosixPath(args.workspace) / "tools" / script_name),
                *parts,
            ],
        )
    return [sys.executable, str(Path(__file__).resolve().with_name(script_name)), *parts]


def child_environment(args) -> dict:
    env = dict(os.environ)
    if args.snapshot_host in args._passwords:
        env.update(
            SSH_ASKPASS=args._askpass,
            SSH_ASKPASS_REQUIRE="force",
            DISPLAY=env.get("DISPLAY") or "pd-replay:0",
        )
        env[ASKPASS_PASSWORD_ENV] = args._passwords[args.snapshot_host]
    return env


def wait_file_source(process, path: Path, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    failure = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        while time.monotonic() < deadline:
            line = stream.readline()
            if line:
                if line.startswith(FILE_PREFIX):
                    item = json.loads(line[len(FILE_PREFIX) :])
                    print(line.rstrip(), flush=True)
                    if item["status"] == "ready":
                        if process.poll() is not None:
                            raise RuntimeError("File source process exited before request submission")
                        return item
                elif line.startswith(PREFIX + " FAILED: "):
                    failure = line.removeprefix(PREFIX + " FAILED: ").strip()
                continue
            if process.poll() is not None:
                if failure:
                    raise RuntimeError(f"File source failed: {failure}; log: {path}")
                raise RuntimeError(f"File source process exited ({process.returncode}); inspect {path}")
            time.sleep(0.25)
    raise TimeoutError(f"File source setup timed out; inspect {path}")


@contextmanager
def file_source(args, plan: dict, output: Path):
    encoded = base64.b64encode(gzip.compress(json.dumps(plan).encode())).decode("ascii")
    command = child_argv(args, "--file-child", encoded)
    log_path = output / "file_source.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_environment(args),
            start_new_session=os.name != "nt",
        )
        try:
            info = wait_file_source(process, log_path, args.source_timeout)
            process.file_info = info
            save_json(output / "file_source.json", info)
            yield process
        finally:
            release = child_argv(args, "--release-child", plan["release_file"])
            try:
                if args.snapshot_host:
                    result = run_ssh(
                        args, args.snapshot_host, release, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
                    )
                else:
                    result = subprocess.run(release, capture_output=True, timeout=30, check=False)
                if result.returncode:
                    print(f"{PREFIX} file source release failed; its lease will expire automatically", flush=True)
                process.wait(timeout=45)
            except (OSError, subprocess.TimeoutExpired):
                # This owns only the local file source/SSH process, never a P/D process.
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"{PREFIX} file source cleanup pending; no serving process was stopped", flush=True)


def release_child(path: str) -> None:
    target = Path(path).resolve()
    parent = (REPO / "prefill-kv-replays" / "leases").resolve()
    if target.parent != parent or len(target.stem) != 32 or any(c not in "0123456789abcdef" for c in target.stem):
        raise ValueError("Invalid file source release path")
    parent.mkdir(parents=True, exist_ok=True)
    target.touch()


def format_failure(error: BaseException) -> str:
    """Keep the loader's underlying cause in the parent's compact error line."""
    chain, seen = [], set()
    while error is not None and id(error) not in seen and len(chain) < 4:
        seen.add(id(error))
        message = " ".join(str(error).splitlines())[:1200]
        chain.append(f"{type(error).__name__}: {message}")
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return " <- ".join(chain)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", type=Path, default=Path("multi-turn-concurrent"))
    cli.add_argument("--file-port", type=int, default=0, help="Snapshot HTTP port; default: choose a free port")
    cli.add_argument("--file-child", help=argparse.SUPPRESS)
    cli.add_argument("--session", type=int, default=0)
    cli.add_argument("--round", type=int, default=1)
    cli.add_argument("--base-url", help="Existing PD proxy URL, e.g. http://7.150.2.72:8000")
    cli.add_argument("--p-url", help="Existing P API URL for tokenization, e.g. http://7.150.2.72:7910")
    cli.add_argument("--p-logs", nargs="+", default=[str(REPO / "tp4_dp4_logs" / "P_dp*.log")])
    cli.add_argument("--snapshot-host", help="SSH host containing the raw previous D snapshot; omit for local files")
    cli.add_argument(
        "--snapshot-hosts",
        nargs="+",
        default=[],
        help="Search these D hosts for the previous request's complete snapshot",
    )
    cli.add_argument("--snapshot-root", help="Snapshot root on its host; default: WORKSPACE/decode-kv-snapshots")
    cli.add_argument("--workspace", default=str(REPO), help="vllm-ascend path on the snapshot host/container")
    cli.add_argument("--container", help="Optional Docker container on the snapshot host")
    cli.add_argument(
        "--remote-python", default=sys.executable, help="Python on snapshot host; no Torch or NPU SDK required"
    )
    password = cli.add_mutually_exclusive_group()
    password.add_argument(
        "--password", dest="ssh_password", help="SSH password, via native OpenSSH askpass; never prompt"
    )
    password.add_argument("--password-env")
    cli.add_argument("--skip-host-key-check", action="store_true")
    cli.add_argument("--ssh-port", type=int)
    cli.add_argument("--identity-file", type=Path)
    cli.add_argument("--tp-size", type=int, help="Infer from current P log labels unless overridden")
    cli.add_argument(
        "--local-hostname", help="Snapshot file service IP reachable from all P/D nodes; default: snapshot-host"
    )
    cli.add_argument("--workers", type=int, default=8, help="Offline validation threads")
    cli.add_argument("--timeout", type=float, default=7200, help="Request timeout seconds")
    cli.add_argument("--source-timeout", type=float, default=7200)
    cli.add_argument("--output-tokens", type=int, help="Keep the original limit unless explicitly overridden")
    cli.add_argument(
        "--prepare-only",
        action="store_true",
        help="Check request/tokenization and locate complete snapshot files; do not read KV bytes or generate",
    )
    cli.add_argument("--release-child", help=argparse.SUPPRESS)
    return cli


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.file_child:
        from kv_replay_files import serve

        serve(json.loads(gzip.decompress(base64.b64decode(args.file_child))))
        return 0
    if args.release_child:
        release_child(args.release_child)
        return 0
    if not args.base_url or not args.p_url or min(args.timeout, args.source_timeout, args.workers) <= 0:
        raise ValueError("--base-url, --p-url and positive timeouts/workers are required")
    if args.output_tokens is not None and args.output_tokens < 1:
        raise ValueError("output-tokens must be positive")
    if args.session < 0 or (args.tp_size is not None and args.tp_size < 1):
        raise ValueError("session must be nonnegative and tp-size must be positive")
    if not all(math.isfinite(t) for t in (args.timeout, args.source_timeout)):
        raise ValueError("Timeouts must be finite")
    args.hosts = candidate_hosts(args)
    if args.hosts:
        args.snapshot_host = args.hosts[0]
    args.password = args.password_per_host = False
    parent = REPO / "prefill-kv-replays"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"s{args.session}-r{args.round}-", dir=parent))
    report = dict(
        status="preparing",
        output=str(output),
        accuracy_verdict="not_assessed",
        scope=(
            "Original single request through existing PD, reading saved history; "
            "original concurrency/timing not reproduced."
        ),
    )
    print(f"{PREFIX} artifacts={output}; using existing P/D; no server startup/restart", flush=True)
    try:
        if not args.prepare_only:
            check_proxy_connection(args.base_url, args.timeout)
        restored, plan = prepare(args, output)
        report.update(
            previous=restored["previous"], current=restored["current"], expected_cached_tokens=plan["cached_tokens"]
        )
        with password_auth(args), ExitStack() as stack:
            selected = select_snapshot(args, plan, output)
            report["snapshot_host"] = selected["host"]
            if args.prepare_only:
                report["status"] = "prepared_not_run"
                return 0
            owner = stack.enter_context(file_source(args, plan, output))
            attach_source(restored["payload"], owner, output)
            client = Client(args.base_url, args.timeout)
            print(f"{PREFIX} source=files ready; submitting original turn to {args.base_url}", flush=True)
            text, metrics = client.complete(restored["payload"], output)
            if owner.poll() is not None:
                raise RuntimeError("File source process exited during the request; replay is invalid")
            report.update(
                request_completed=True, metrics=metrics, text_equals_original=text == restored["original_output"]
            )
            print(f"{PREFIX} OUTPUT BEGIN\n{text}\n{PREFIX} OUTPUT END", flush=True)
            report["file_reads"] = source_stats(owner, output)
            validation = validate_response_counts(
                metrics, restored["original_metrics"]["prompt_tokens"], plan["cached_tokens"]
            )
            report["count_validation"] = validation
            report["status"] = "replay_unverified" if validation["missing"] else "replay_complete"
            ttft_label = f"{metrics['ttft_s']:.3f}s" if metrics["ttft_s"] is not None else "n/a"
            print(
                f"{PREFIX} cached={metrics['cached_tokens']} uncached={metrics['uncached_prompt_tokens']} "
                f"ttft={ttft_label} response_id={metrics['response_id']}",
                flush=True,
            )
            if validation["missing"]:
                print(
                    f"{PREFIX} UNVERIFIED: P/D response completed, but usage omitted "
                    f"{', '.join(validation['missing'])}; inspect this request's P logs to confirm reuse. "
                    f"Output retained; no inference retry. artifacts={output}",
                    flush=True,
                )
        return 2 if report["status"] == "replay_unverified" else 0
    except BaseException as error:
        report.update(status="failed", error=format_failure(error))
        raise
    finally:
        save_json(output / "report.json", report)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        traceback.print_exception(type(error), error, error.__traceback__)
        print(f"{PREFIX} FAILED: {format_failure(error)}", file=sys.stderr, flush=True)
        raise SystemExit(1)
