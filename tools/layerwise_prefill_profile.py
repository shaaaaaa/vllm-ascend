#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture full-model TP8 P-node prefill: 80k ON by default.

Local LMCache CPU storage only: no Mooncake, file SDK shim, D node or KV probes.
Each case uses a fresh model process and executes one request through its first
output token. 80k captures only the first/last three compute-prefill chunks;
10k captures the whole request. Model startup is outside capture.
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
LONG_MAX_MODEL_LEN = 84000
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
    cli.add_argument(
        "--dummy-dma-bind", action="store_true", help="Skip only per-layer DMA address binding; implies dummy DMA"
    )
    cli.add_argument(
        "--dummy-submit-load",
        action="store_true",
        help="Keep submit/cursor state but disable native load DMA; outputs invalid",
    )
    cli.add_argument(
        "--dummy-prepare",
        action="store_true",
        help="Skip LMCache worker preparation and transfers; implies dummy DMA; outputs invalid",
    )
    cli.add_argument(
        "--dummy-prefill-store",
        action="store_true",
        help="Compatibility alias for --dummy-prefill-store-stage 0",
    )
    cli.add_argument(
        "--dummy-prefill-store-stage",
        type=int,
        choices=range(11),
        help=(
            "Cumulatively run the P-node store path through stage 0..10; "
            "stage 8 stops before storer finalization, stage 9 finalizes "
            "storers, stage 10 adds publish/release; "
            "all stages imply dummy DMA and produce invalid output"
        ),
    )
    cli.add_argument(
        "--dummy-dma", action="store_true", help="Skip layerwise KV DMA only; outputs are invalid (diagnostic run)"
    )
    cli.add_argument("--model", default="/workspace/models/GLM-5.2-w4a8c8-0723")
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--prompt-file", type=Path, help="Override the fixed 10k/80k example article")
    cli.add_argument("--cpu-cache-gb", type=float, default=24, help="Requires this much free /dev/shm and host RAM")
    cli.add_argument(
        "--diagnose-chunk-start",
        action="store_true",
        help="Log host setup stages at each prefill chunk; opt-in profiling overhead",
    )
    selection = cli.add_mutually_exclusive_group()
    selection.add_argument(
        "--case", choices=("all", *CASES), default="80k_on", help="Default: 80k ON; all: 80k OFF then ON"
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
            "HCCL_DETERMINISTIC": "strict",
            "HCCL_BUFFSIZE": "200",
            "MSMONITOR_USE_DAEMON": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", ""),
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": str(case.endswith("_on")).lower(),
            "VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE": "false",
            "VLLM_ASCEND_DSA_UNBUNDLE": "1",
            "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
            "VLLM_ASCEND_DSA_SHARED_POOL": "1",
            "VLLM_ASCEND_DSA_SHRINK_LATENT": "0",
            "VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE": "0",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
            "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "0",
            "VLLM_ASCEND_SFA_STAGED_GRAPH": "0",
            "VLLM_ASCEND_SFA_FULL_GRAPH": "0",
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
            "LMCACHE_LAYERWISE_PREFILL_DMA": "1" if case.endswith("_on") else "0",
            "LMCACHE_PREFILL_START_TIMING": "1" if args.diagnose_chunk_start else "0",
            "LMCACHE_ENABLE_SHARED_CPU_CACHE": "true",
            "LMCACHE_SHARED_CPU_CACHE_STRICT": "true",
            "LMCACHE_SHARED_CPU_CACHE_PASSIVE_WRITABLE": "true",
            "LMCACHE_EXTRA_CONFIG": json.dumps({"save_only_first_rank": True}),
        }
    )
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
        "async_scheduling": False,
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
            "kv_role": "kv_producer",
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


def capture_request(
    llm, token_ids, params, case, case_dir=None, dummy_dma=False,
    dummy_prepare=False, dummy_dma_bind=False, dummy_submit_load=False,
    dummy_prefill_store=False, dummy_prefill_store_stage=None,
):
    if dummy_prepare and (dummy_dma_bind or dummy_submit_load):
        raise ValueError("Use --dummy-dma-bind without --dummy-prepare to isolate address construction")
    if dummy_prefill_store and dummy_prefill_store_stage is not None:
        raise ValueError(
            "Use either --dummy-prefill-store or "
            "--dummy-prefill-store-stage, not both"
        )
    if dummy_prefill_store:
        dummy_prefill_store_stage = 0
    dummy_dma = (
        dummy_dma
        or dummy_prepare
        or dummy_dma_bind
        or dummy_submit_load
        or dummy_prefill_store_stage is not None
    )
    plan = make_capture_plan(len(token_ids), COMPUTE_CHUNK_TOKENS) if case in LONG_CASES or dummy_dma else None
    if dummy_prepare:
        plan["dummy_prepare"] = True
    if dummy_dma_bind:
        plan["dummy_dma_bind"] = True
    if dummy_submit_load:
        plan["dummy_submit_load"] = True
    if dummy_prefill_store_stage is not None:
        plan["dummy_prefill_store_stage"] = dummy_prefill_store_stage
    if dummy_dma:
        plan["dummy_dma"] = True
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
            dummy_dma=args.dummy_dma,
            dummy_prepare=args.dummy_prepare,
            dummy_dma_bind=args.dummy_dma_bind,
            dummy_submit_load=args.dummy_submit_load,
            dummy_prefill_store=args.dummy_prefill_store,
            dummy_prefill_store_stage=args.dummy_prefill_store_stage,
        )
        result = results[0]
        completion = result.outputs[0]
        report = {
            "dummy_dma": (
                args.dummy_dma
                or args.dummy_prepare
                or args.dummy_dma_bind
                or args.dummy_submit_load
                or args.dummy_prefill_store
                or args.dummy_prefill_store_stage is not None
            ),
            "dummy_dma_bind": args.dummy_dma_bind,
            "dummy_submit_load": args.dummy_submit_load,
            "dummy_prefill_store": args.dummy_prefill_store,
            "dummy_prefill_store_stage": (
                0 if args.dummy_prefill_store
                else args.dummy_prefill_store_stage
            ),
            "dummy_prepare": args.dummy_prepare,
            "output_valid_for_correctness": not (
                args.dummy_dma or args.dummy_prepare or args.dummy_dma_bind
                or args.dummy_submit_load or args.dummy_prefill_store
                or args.dummy_prefill_store_stage is not None
            ),
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
        if args.dummy_dma:
            command.append("--dummy-dma")
        if args.dummy_prepare:
            command.append("--dummy-prepare")
        if args.dummy_dma_bind:
            command.append("--dummy-dma-bind")
        if args.dummy_submit_load:
            command.append("--dummy-submit-load")
        if args.dummy_prefill_store:
            command.append("--dummy-prefill-store")
        if args.dummy_prefill_store_stage is not None:
            command.extend([
                "--dummy-prefill-store-stage",
                str(args.dummy_prefill_store_stage),
            ])
        write_json(case_dir / "environment.json", {k: v for k, v in env.items() if k.startswith(("LMCACHE_", "VLLM_"))})
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
