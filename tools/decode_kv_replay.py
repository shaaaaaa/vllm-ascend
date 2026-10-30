#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Restart one D instance and generate from its archived receive payloads."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from decode_kv_compare import read_workers, request_key

REPO = Path(__file__).resolve().parents[1]


def equal_payload(left: Path, right: Path) -> bool:
    """Verify the TP-shared latent source without hashing or loading whole files."""
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            first, second = a.read(4 * 1024 * 1024), b.read(4 * 1024 * 1024)
            if first != second:
                return False
            if not first:
                return True


def build_plan(roots: list[Path], request: str, output: Path) -> dict:
    """Reject incomplete/incompatible captures before loading model weights."""
    workers = read_workers(roots, request_key(request))
    plan = dict(schema=1, request_id=request_key(request), output=str(output.resolve()), workers={})
    reference = None
    for rank, (meta, tokens, rows) in workers.items():
        directory = next(iter(rows.values()))[1].parent
        if meta.get("replay_file") != "replay.json":
            raise ValueError("This capture predates replay metadata; capture once with the updated D snapshot code")
        replay = json.loads((directory / "replay.json").read_text(encoding="utf-8"))
        req, runtime = replay.get("request"), replay.get("runtime")
        if not req or not runtime or req.get("error") or runtime.get("error"):
            raise ValueError(f"Unusable replay metadata: {replay}")
        prompt = req["prompt_token_ids"]
        if req.get("schema") != 1 or not tokens or prompt[: len(tokens)] != tokens:
            raise ValueError("Received KV does not match the archived D prompt")
        if req["sampling"].get("n", 1) != 1:
            raise ValueError("Replay requires one output sequence")
        if meta["byteorder"] != sys.byteorder or len(meta["groups"]) != 2:
            raise ValueError("Replay requires native-endian, two-group DSA snapshots")
        if reference is not None and replay != reference:
            raise ValueError("TP ranks disagree on request/runtime metadata")
        reference = replay
        files = []
        for key, (row, path) in sorted(rows.items()):
            if row["row_bytes"] <= 0 or path.is_symlink():
                raise ValueError(f"Invalid payload: {path}")
            files.append(dict(row, absolute_path=str(path.resolve())))
        for group, count in enumerate(meta["groups"]):
            for layer in range(count):
                components = [row["component"] for row in files if row["group"] == group and row["layer"] == layer]
                if components != list(range(len(components))):
                    raise ValueError("Missing or non-contiguous KV components")
        plan["workers"][str(rank)] = dict(groups=meta["groups"], layer_names=meta["layer_names"], files=files)
    for rank, (_, _, rows) in workers.items():
        if rank == 0:
            continue
        print(f"[D_KV_REPLAY] validating shared latent bytes: TP {rank} versus TP 0", flush=True)
        if rows.keys() != workers[0][2].keys():
            raise ValueError("TP ranks have different component inventories")
        for key, (row, path) in rows.items():
            baseline, baseline_path = workers[0][2][key]
            if (row["row_bytes"], row["dtype"]) != (baseline["row_bytes"], baseline["dtype"]):
                raise ValueError(f"TP rank {rank} component layout mismatch: {key}")
            # Production group 0 is rank0-owned and shared. Replaying divergent
            # copies as one shared source would erase evidence of corruption.
            if key[0] == 0 and not equal_payload(baseline_path, path):
                raise ValueError(
                    f"TP rank {rank} latent bytes differ from rank 0 at {key}; "
                    "cannot faithfully replay shared CPU source"
                )
    plan.update(reference)
    plan["received_tokens"] = next(iter(workers.values()))[1]
    return plan


def configure(plan: dict, args) -> tuple[dict, dict]:
    """Use recorded compute choices and an isolated, read-only file transport."""
    from layerwise_prefill_mooncake_check import config_environment, deployment_config, stage_config
    from layerwise_prefill_profile import case_environment

    options = dict(plan["runtime"]["engine_options"])
    tp = len(plan["workers"])
    if tp != options["tensor_parallel_size"]:
        raise ValueError("Recorded TP size disagrees with captured ranks")
    devices = args.devices or ",".join(str(i) for i in range(tp))
    if len(devices.split(",")) != tp:
        raise ValueError(f"Replay requires {tp} devices, matching the original TP size")
    # CPU backing is shared by the TP group. Include room for receive staging.
    size = sum(row["bytes"] for row in plan["workers"]["0"]["files"])
    cpu_gb = args.cpu_cache_gb or math.ceil(size * 1.3 / 1024**3 + 4)
    settings = SimpleNamespace(devices=devices, cpu_cache_gb=cpu_gb, store_gb=0)
    env = case_environment(settings, "10k_off")
    env = {key: value for key, value in env.items() if not key.startswith(("LMCACHE_", "MOONCAKE_"))}
    env.update(plan["runtime"]["environment"])
    env.update(
        VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE="false",
        VLLM_ASCEND_BALANCE_SCHEDULING="0",
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=str(args.timeout),
        VLLM_ENGINE_READY_TIMEOUT_S=str(args.timeout),
    )
    config = stage_config(deployment_config("127.0.0.1:1", "127.0.0.1"), settings, "decode")
    config["chunk_size"] = plan["runtime"]["chunk_size"]
    config["blocking_timeout_secs"] = args.timeout
    config["extra_config"]["transfer_timeout"] = args.timeout
    config["extra_config"]["decode_snapshot_replay"] = {"plan": str(Path(plan["output"]) / "plan.json")}
    env.update(config_environment(config))
    options.update(
        model=args.model or options["model"],
        data_parallel_size=1,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        gpu_memory_utilization=args.gpu_memory_utilization or options.get("gpu_memory_utilization", 0.90),
        kv_transfer_config=dict(
            kv_connector="LMCacheAscendConnectorV1Dynamic",
            kv_role="kv_consumer",
            kv_connector_module_path="lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
            kv_connector_extra_config={
                "skip_last_n_tokens": len(plan["request"]["prompt_token_ids"]) - len(plan["received_tokens"]),
            },
        ),
    )
    if args.eager:
        options["enforce_eager"] = True
    if args.model:
        options["tokenizer"] = args.model
    # Balanced DP scheduling assumes peer replicas; this replay has only one.
    options["additional_config"] = dict(options.get("additional_config") or {}, recompute_scheduler_enable=False)
    return options, env


def run_child(path: Path) -> None:
    """Generate with real D kernels; a cache miss cannot be reported as a successful replay."""
    import msgspec
    from decode_kv_replay_store import install

    install()
    from vllm import LLM, SamplingParams

    plan = json.loads(path.read_text(encoding="utf-8"))
    output = Path(plan["output"])
    sampling = dict(plan["request"]["sampling"])
    sampling["max_tokens"] = plan["output_tokens"]
    sampling["min_tokens"] = min(sampling.get("min_tokens", 0), plan["output_tokens"])
    # Transport addresses were removed when recording. The optional final-hidden
    # bootstrap remains a pure payload and is accepted by the normal Request.
    params = msgspec.convert(sampling, type=SamplingParams, strict=False)
    llm = LLM(**plan["engine_options"])
    try:
        prompt = {"prompt_token_ids": plan["request"]["prompt_token_ids"]}
        if plan["request"].get("cache_salt"):
            prompt["cache_salt"] = plan["request"]["cache_salt"]
        (result,) = llm.generate(prompt, params, use_tqdm=False)
        completion = result.outputs[0]
        expected = min(len(plan["received_tokens"]), len(prompt["prompt_token_ids"]) - 1)
        report = dict(
            request_id=plan["request_id"],
            text=completion.text,
            token_ids=list(completion.token_ids),
            num_cached_tokens=result.num_cached_tokens,
            expected_cached_tokens=expected,
            finish_reason=completion.finish_reason,
            status="complete" if result.num_cached_tokens == expected else "cache_mismatch",
            scope="Single D replay of saved received KV; original batching/timing are not reproduced.",
        )
        (output / "output.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "output.txt").write_text(completion.text, encoding="utf-8")
        print(f"[D_KV_REPLAY] cached={result.num_cached_tokens}/{expected}\n{completion.text}", flush=True)
        if result.num_cached_tokens != expected:
            raise RuntimeError("Replay did not load the full recorded prefix; inspect output.json and server.log")
    finally:
        llm.llm_engine.engine_core.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id")
    parser.add_argument(
        "--local", nargs="+", type=Path, help="Collected archive roots; defaults to this repo's decode-kv-snapshots"
    )
    parser.add_argument("--devices", help="NPU IDs; defaults to 0 through recorded TP size minus one")
    parser.add_argument("--model", help="Optional relocated path to the same model weights")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--cpu-cache-gb", type=float)
    parser.add_argument("--gpu-memory-utilization", type=float, help="Defaults to the captured value")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--eager", action="store_true", help="Disable graphs for this replay; otherwise keep recorded eager choice"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Validate and write launch plan without loading the model"
    )
    parser.add_argument("--child-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_plan:
        run_child(args.child_plan)
        return 0
    if not args.request_id or args.output_tokens < 1 or args.timeout < 1:
        parser.error("--request-id and positive output-tokens/timeout are required")
    if (args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization < 1) or (
        args.cpu_cache_gb is not None and args.cpu_cache_gb <= 0
    ):
        parser.error("Invalid memory capacity")
    parent = REPO / "decode-kv-replays"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=request_key(args.request_id) + "-", dir=parent))
    print(f"[D_KV_REPLAY] validating archive; artifacts={output}", flush=True)
    plan = build_plan(args.local or [REPO / "decode-kv-snapshots"], args.request_id, output)
    options, env = configure(plan, args)
    plan.update(engine_options=options, output_tokens=args.output_tokens)
    if len(plan["request"]["prompt_token_ids"]) + args.output_tokens > options["max_model_len"]:
        raise ValueError("Prompt plus replay output exceeds the captured max_model_len")
    path = output / "plan.json"
    path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    bootstrap = output / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "import os, traceback\ntry:\n from decode_kv_replay_store import install\n install()\n"
        "except BaseException:\n traceback.print_exc()\n os._exit(1)\n",
        encoding="utf-8",
    )
    env["PYTHONPATH"] = str(bootstrap) + os.pathsep + env["PYTHONPATH"]
    print(f"[D_KV_REPLAY] artifacts={output}; TP={len(plan['workers'])}; no P/Mooncake required", flush=True)
    if args.prepare_only:
        return 0
    from layerwise_prefill_mooncake_check import finish_child, start_logged_process

    process = start_logged_process(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--child-plan", str(path)],
        env,
        output / "server.log",
        "D replay",
        prefix="[D_KV_REPLAY]",
    )
    try:
        return process.wait()
    finally:
        finish_child(process)


if __name__ == "__main__":
    raise SystemExit(main())
