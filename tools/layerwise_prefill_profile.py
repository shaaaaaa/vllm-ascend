#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture full-model TP8 prefill profiles for the P/D comparison.

Local LMCache CPU storage only: no Mooncake, file SDK shim or KV probes. The
``*_off`` case keeps the D-node/original transfer path; ``*_on`` enables the
P-node layerwise path with ``VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=true``. Each
case uses a fresh model process and executes one request through its first
output token. 80k captures only the first/last three compute-prefill chunks;
10k captures the whole request. Model startup is outside capture.

Compute/runtime settings follow the serving P node, with single-host TP8/DP1
and local CPU cache for this benchmark. Keep host IP/NIC settings in the caller
environment; deployment paths, Mooncake and API-server options are not needed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from layerwise_prefill_check import DEFAULT_PROMPT_FILE, normalize_prompt_token_ids, write_json
from layerwise_prefill_mooncake_check import finish_child, start_logged_process
from layerwise_prefill_profile_worker import (
    make_capture_plan,
    validate_capture,
)

CASES = ("10k_off", "10k_on", "80k_off", "80k_on")
LONG_CASES = ("80k_off", "80k_on")
DEFAULT_LONG_PROMPT_FILE = DEFAULT_PROMPT_FILE.with_name("article_summary_80k.txt")
MAX_PROMPT_FIT_ATTEMPTS = 3
MIN_PROMPT_FRACTION = 0.95
CACHE_CHUNK_TOKENS = 1024
COMPUTE_CHUNK_TOKENS = 4096
SHORT_MAX_MODEL_LEN = 16384
LONG_MAX_MODEL_LEN = 80000 + COMPUTE_CHUNK_TOKENS
PREFIX = "[PREFILL_PROFILE]"


def clear_shm(shm_dir: Path) -> int:
    """Clear /dev/shm contents before this dedicated-machine profile run."""
    root = shm_dir.resolve(strict=True)
    removed = 0
    for path in root.iterdir():
        # Match the shell's /dev/shm/* glob: hidden entries are not included.
        if path.name.startswith("."):
            continue
        try:
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                if path.resolve(strict=True).parent != root:
                    raise RuntimeError(f"Refusing to remove a directory outside {root}: {path}")
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            continue
        removed += 1
    print(f"{PREFIX} cleared {removed} /dev/shm entries", flush=True)
    return removed


def check_shm_capacity(shm_dir: Path, cache_gb: float) -> None:
    usage = os.statvfs(shm_dir)
    free_bytes = usage.f_bavail * usage.f_frsize
    total_bytes = usage.f_blocks * usage.f_frsize
    required_bytes = int(cache_gb * 1024**3)
    print(
        f"{PREFIX} /dev/shm total={total_bytes / 1024**3:.2f} GiB, "
        f"free={free_bytes / 1024**3:.2f} GiB, "
        f"requested CPU slab={cache_gb:g} GiB",
        flush=True,
    )
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Not enough /dev/shm after stale LMCache cleanup: "
            f"free={free_bytes / 1024**3:.2f} GiB, required={cache_gb:g} GiB. "
            "Increase the container's /dev/shm capacity; deleting files cannot "
            "fix a mount whose total size is too small."
        )


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", default="/workspace/models/GLM-5.1-w4a8")
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--prompt-file", type=Path, help="Override the fixed 10k/80k example article")
    cli.add_argument("--cpu-cache-gb", type=float, default=24, help="Requires this much free /dev/shm and host RAM")
    selection = cli.add_mutually_exclusive_group()
    selection.add_argument(
        "--case",
        choices=("all", *CASES),
        default="80k_on",
        help="Default: 80k ON; all: 80k OFF then ON",
    )
    selection.add_argument("--include-off", action="store_const", dest="case", const="all", help="Run 80k OFF then ON")
    cli.add_argument("--run-dir", type=Path, help="New, empty results directory")
    cli.add_argument(
        "--analyse-only", type=Path, help="Export an existing run's raw profiles without loading the model"
    )
    cli.add_argument("--child", choices=CASES, help=argparse.SUPPRESS)
    return cli


def build_prompt(tokenizer, article: str, target_tokens: int):
    """Bound the fixed article's body, then apply the intact chat template.

    Never grow input in a loop or binary-search the entire long text. Template
    boundary tokenization can vary, so allow a bounded number of body trims.
    Reject unexpectedly short tokenization instead of duplicating indefinitely.
    """

    def encode(text):
        return normalize_prompt_token_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )

    body_ids = tokenizer.encode(article, add_special_tokens=False)
    body_budget = target_tokens - len(encode(""))
    for _ in range(MAX_PROMPT_FIT_ATTEMPTS):
        if body_budget <= 0:
            break
        text = (
            article
            if len(body_ids) <= body_budget
            else tokenizer.decode(body_ids[:body_budget], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        )
        ids = encode(text)
        if len(ids) <= target_tokens:
            if len(ids) < target_tokens * MIN_PROMPT_FRACTION or len(ids) <= 4096:
                raise ValueError(
                    f"Fixed article tokenized to only {len(ids)} tokens for target={target_tokens}; "
                    "check the input file/tokenizer. No text expansion was attempted."
                )
            return text, ids
        body_budget -= len(ids) - target_tokens
    raise ValueError(f"Could not fit the fixed article within {target_tokens} tokens; no unbounded retry")


def prepare_inputs(args, root, cases):
    started = time.perf_counter()
    print(f"{PREFIX} loading tokenizer: {args.model}", flush=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"{PREFIX} tokenizer loaded in {time.perf_counter() - started:.3f}s", flush=True)
    for name, target in (("10k", 10000), ("80k", 80000)):
        if not any(case.startswith(name + "_") for case in cases):
            continue
        source = args.prompt_file or (DEFAULT_LONG_PROMPT_FILE if name == "80k" else DEFAULT_PROMPT_FILE)
        article = source.read_text(encoding="utf-8")
        if not article.strip():
            raise ValueError(f"Empty article: {source}")
        (root / f"{name}_article_source.txt").write_text(article, encoding="utf-8")
        started = time.perf_counter()
        print(f"{PREFIX} {name}: tokenizing fixed file {source}; chars={len(article)}", flush=True)
        text, ids = build_prompt(tokenizer, article, target)
        (root / f"{name}_input.txt").write_text(text, encoding="utf-8")
        write_json(root / f"{name}_prompt.json", {"target_tokens": target, "length": len(ids), "token_ids": ids})
        print(
            f"{PREFIX} {name}: input ready in {time.perf_counter() - started:.3f}s; "
            f"actual prompt_tokens={len(ids)}; saved text and token IDs",
            flush=True,
        )


def case_environment(args, case):
    # Never inherit file-shim/debug hooks or another deployment's remote config.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VLLM_", "LMCACHE_", "MOONCAKE_"))}
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": args.devices,
            "PYTHONHASHSEED": "0",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_INTRA_ROCE_ENABLE": "1",
            "HCCL_BUFFSIZE": "200",
            "MSMONITOR_USE_DAEMON": "0",
            "OMP_PROC_BIND": "false",
            "OMP_NUM_THREADS": "10",
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": "/usr/local/lib:" + env.get("LD_LIBRARY_PATH", ""),
            "ASCEND_BUFFER_POOL": "4:8",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "VLLM_LOG_STATS_INTERVAL": "1",
            "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": str(case.endswith("_on")).lower(),
            "VLLM_ASCEND_DSA_UNBUNDLE": "1",
            "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
            "VLLM_ASCEND_DSA_SHARED_POOL": "1",
            "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
            "VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE": "0",
            "VLLM_ASCEND_DSA_DISABLE_TARGET_SLOT_MAPPING": "0",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
            "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "0",
            "VLLM_ASCEND_BALANCE_SCHEDULING": "1",
            "TASK_QUEUE_ENABLE": "1",
            "CPU_AFFINITY_CONF": "1",
            "ASCEND_AGGREGATE_ENABLE": "1",
            "ASCEND_TRANSPORT_PRINT": "1",
            "ACL_OP_INIT_MODE": "1",
            "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": "600",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "PD_SERVING_PERF": "detail",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
            "LMCACHE_ASCEND_SPARSE_TRANSFER_TOPK": "2048",
            "LMCACHE_CHUNK_SIZE": str(CACHE_CHUNK_TOKENS),
            "LMCACHE_LOCAL_CPU": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(args.cpu_cache_gb),
            "LMCACHE_USE_LAYERWISE": "true",
            "LMCACHE_ENABLE_SPARSE_ATTENTION": "true",
            "LMCACHE_DSA_TWO_GROUPS": "true",
            "LMCACHE_STORE_ASYNC": "true",
            "LMCACHE_STORE_ASYNC_MAX_QUEUE_SIZE": "2",
            "LMCACHE_SAVE_DECODE_CACHE": "false",
            "LMCACHE_SAVE_UNFULL_CHUNK": "true",
            "LMCACHE_SAVE_FULL_CHUNK_IN_DECODE": "false",
            "LMCACHE_ENABLE_SHARED_CPU_CACHE": "true",
            "LMCACHE_SHARED_CPU_CACHE_STRICT": "true",
            "LMCACHE_SHARED_CPU_CACHE_NUMA_POLICY": "interleave",
            "LMCACHE_SHARED_CPU_CACHE_PASSIVE_WRITABLE": "true",
            "LMCACHE_LOOKUP_TIMEOUT_MS": "30000",
            "LMCACHE_EXPERIMENTAL_SAMPLED_LAYERWISE_LOOKUP": "true",
            "LMCACHE_PIN_TIMEOUT_SEC": "1800",
            "LMCACHE_ENABLE_NPU_CONTENT_DIAGNOSTICS": "false",
            "LMCACHE_EXTRA_CONFIG": json.dumps({"save_only_first_rank": True}),
        }
    )
    # The P-node marker is the single switch for the new path.  Do not add the
    # obsolete LMCACHE_LAYERWISE_PREFILL_DMA variable: D must retain the
    # original single-layer paged transfer path when this marker is false.
    return env


def engine_options(args, case_dir, prompt_len):
    # The 80k input cases reserve another 4k tokens of sequence length.
    # Keep the OFF/ON capacity identical even if the tokenized input is a little short.
    max_len = LONG_MAX_MODEL_LEN if prompt_len > SHORT_MAX_MODEL_LEN else SHORT_MAX_MODEL_LEN
    return {
        "model": args.model,
        "trust_remote_code": True,
        "load_format": "safetensors",
        "quantization": "ascend",
        "tensor_parallel_size": len(args.devices.split(",")),
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "worker_extension_cls": "layerwise_prefill_profile_worker.ChunkProfileWorkerExtension",
        "enable_expert_parallel": True,
        "gpu_memory_utilization": 0.97,
        "max_model_len": max_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": COMPUTE_CHUNK_TOKENS,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "async_scheduling": None,  # Use the same automatic selection as vllm serve.
        "enforce_eager": True,  # Same P-only execution mode for OFF and ON.
        "seed": 1024,
        "speculative_config": {"method": "deepseek_mtp", "num_speculative_tokens": 1},
        "additional_config": {
            "recompute_scheduler_enable": False,
            "multistream_overlap_shared_expert": False,
            "fuse_muls_add": True,
            "fuse_qknorm_rope": False,
            "enable_npugraph_ex": True,
            "layer_sharding": ["q_b_proj"],
        },
        "kv_transfer_config": {
            "kv_connector": "LMCacheAscendConnectorV1Dynamic",
            "kv_role": "kv_both",
            "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
        },
        "profiler_config": {
            "profiler": "torch",
            "torch_profiler_dir": str((case_dir / "profile").resolve()),
            "ignore_frontend": True,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_record_shapes": False,
        },
    }


def capture_request(llm, token_ids, params, case, case_dir=None):
    plan = make_capture_plan(len(token_ids), COMPUTE_CHUNK_TOKENS) if case in LONG_CASES else None
    if plan:
        write_json(case_dir / "capture_plan.json", plan)
        print(f"{PREFIX} {case}: capture windows={plan['windows']}; middle chunks still compute", flush=True)
    print(f"{PREFIX} {case}: profiler start begin", flush=True)
    if plan:
        llm.collective_rpc("install_chunk_profile", args=(case, plan))
    else:
        llm.start_profile(profile_prefix=case)
    print(f"{PREFIX} {case}: profiler {'armed' if plan else 'started'}; generate begin", flush=True)
    try:
        start = time.perf_counter()
        results = llm.generate({"prompt_token_ids": token_ids}, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        print(f"{PREFIX} {case}: generate complete in {elapsed:.3f}s (prefill/first token finished)", flush=True)
        return results, elapsed
    finally:
        error_in_flight = sys.exc_info()[0] is not None
        stopped = time.perf_counter()
        print(f"{PREFIX} {case}: profiler stop begin", flush=True)
        try:
            if plan:
                reports = llm.collective_rpc("finish_chunk_profile")
                write_json(case_dir / "capture_windows.json", {"plan": plan, "workers": reports})
                if not error_in_flight:
                    validate_capture(plan, reports)
            else:
                llm.stop_profile()
            print(f"{PREFIX} {case}: profiler stop complete in {time.perf_counter() - stopped:.3f}s", flush=True)
        except Exception as exc:
            if not error_in_flight:
                raise
            print(f"{PREFIX} profiler stop also failed: {exc}", file=sys.stderr, flush=True)


def run_child(args):
    from vllm import LLM, SamplingParams

    case_dir = args.run_dir / args.child
    prompt = json.loads((args.run_dir / f"{args.child.split('_')[0]}_prompt.json").read_text(encoding="utf-8"))
    options = engine_options(args, case_dir, prompt["length"])
    write_json(case_dir / "engine_options.json", options)
    print(f"{PREFIX} {args.child}: loading full model; capture starts AFTER startup", flush=True)
    llm = LLM(**options)
    try:
        print(f"{PREFIX} {args.child}: capturing {prompt['length']} input tokens -> first output token", flush=True)
        results, elapsed = capture_request(
            llm,
            prompt["token_ids"],
            SamplingParams(temperature=0, seed=1024, max_tokens=1),
            args.child,
            case_dir,
        )
        result = results[0]
        completion = result.outputs[0]
        report = {
            "output_valid_for_correctness": True,
            "case": args.child,
            "prompt_tokens": prompt["length"],
            "num_cached_tokens": result.num_cached_tokens,
            "text": completion.text,
            "token_ids": list(completion.token_ids),
            "request_seconds_with_profiler": elapsed,
            "scope": "local CPU KV offload/reload + P forward incl. MTP + first-token sampling; no remote store",
        }
        write_json(case_dir / "result.json", report)
        # OFF/ON must both compute the whole prompt, never measure a cache hit.
        if result.num_cached_tokens:
            raise RuntimeError("Unexpected cache hit; this trace is not a fresh prefill")
        print(f"{PREFIX} {args.child}: captured; request_seconds_with_profiler={elapsed:.3f}", flush=True)
    finally:
        print(f"{PREFIX} {args.child}: model shutdown begin", flush=True)
        llm.llm_engine.engine_core.shutdown()
        print(f"{PREFIX} {args.child}: model shutdown complete", flush=True)


def analyse_case(case_dir):
    started = time.perf_counter()
    print(f"{PREFIX} exporting {case_dir.name} traces (model has exited)", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            "from torch_npu.profiler.profiler import analyse; import sys; analyse(sys.argv[1], max_process_number=2)",
            str(case_dir / "profile"),
        ],
        check=True,
    )
    traces = sorted(str(path.resolve()) for path in (case_dir / "profile").rglob("trace_view.json"))
    expected = json.loads((case_dir / "engine_options.json").read_text(encoding="utf-8"))["tensor_parallel_size"]
    plan_path = case_dir / "capture_plan.json"
    windows = json.loads(plan_path.read_text(encoding="utf-8"))["windows"] if plan_path.exists() else None
    if windows:
        expected *= len(windows)
    write_json(case_dir / "traces.json", traces)
    if len(traces) != expected:
        raise RuntimeError(f"Expected {expected} worker trace_view.json files, found {len(traces)} in {case_dir}")
    if windows:
        ranks = expected // len(windows)
        for window in windows:
            marker = f"{case_dir.name}_{window['name']}_"
            matched = [p for p in traces if marker in Path(p).relative_to(case_dir.resolve() / "profile").as_posix()]
            if len(matched) != ranks:
                raise RuntimeError(f"Expected {ranks} {window['name']} traces, found {len(matched)} in {case_dir}")
    print(
        f"{PREFIX} {case_dir.name}: export complete in {time.perf_counter() - started:.3f}s; "
        f"{len(traces)} MindStudio traces; paths in {case_dir / 'traces.json'}",
        flush=True,
    )


def run_cases(args, root, cases):
    for index, case in enumerate(cases):
        if index:
            clear_shm(Path("/dev/shm"))
        case_dir = root / case
        case_dir.mkdir()
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--child",
            case,
            "--run-dir",
            str(root),
            "--model",
            args.model,
            "--devices",
            args.devices,
            "--cpu-cache-gb",
            str(args.cpu_cache_gb),
        ]
        env = case_environment(args, case)
        write_json(
            case_dir / "environment.json",
            {
                k: v
                for k, v in env.items()
                if k.startswith(("LMCACHE_", "VLLM_", "HCCL_", "ASCEND_", "OMP_", "PYTORCH_NPU_"))
                or k
                in {
                    "TASK_QUEUE_ENABLE",
                    "CPU_AFFINITY_CONF",
                    "ACL_OP_INIT_MODE",
                    "PD_SERVING_PERF",
                    "MSMONITOR_USE_DAEMON",
                    "GLOO_SOCKET_IFNAME",
                    "TP_SOCKET_IFNAME",
                    "PYTHONHASHSEED",
                }
            },
        )
        proc = start_logged_process(command, env, case_dir / "server.log", case, prefix=PREFIX)
        try:
            if proc.wait():
                raise RuntimeError(f"{case} failed; see {case_dir / 'server.log'}; remaining cases not run")
        finally:
            finish_child(proc)
        analyse_case(case_dir)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.child:
        run_child(args)
        return
    if args.analyse_only:
        for case in CASES:
            case_dir = args.analyse_only.resolve() / case
            if (case_dir / "engine_options.json").is_file():
                analyse_case(case_dir)
        return
    if os.name != "posix":
        cli.error("Run on the Linux Ascend server")
    devices = args.devices.split(",")
    if not all(d.isdigit() for d in devices) or len(devices) != len(set(devices)) or args.cpu_cache_gb <= 0:
        cli.error("Specify distinct NPU device IDs and a positive CPU cache size")
    clear_shm(Path("/dev/shm"))
    check_shm_capacity(Path("/dev/shm"), args.cpu_cache_gb)
    root = (
        args.run_dir.resolve()
        if args.run_dir
        else Path(tempfile.mkdtemp(prefix="layerwise-profile-", dir=".")).resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        cli.error("--run-dir must be empty (nothing was deleted)")
    cases = LONG_CASES if args.case == "all" else (args.case,)
    print(f"{PREFIX} results: {root}; full model, TP={len(devices)}, gpu=0.97, eager P only, MTP1", flush=True)
    print(f"{PREFIX} local CPU cache={args.cpu_cache_gb} GiB; no Mooncake/file shim/KV dumps", flush=True)
    prepare_inputs(args, root, cases)
    run_cases(args, root, cases)
    print(f"{PREFIX} done: {root}", flush=True)


if __name__ == "__main__":
    main()
