#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture full-model TP8 P-node prefill: 10k ON (add --include-off for OFF/ON).

Local LMCache CPU storage only: no Mooncake, file SDK shim, D node or KV probes.
Each case uses a fresh model process and profiles one request through its first
output token. Model startup is outside capture; real-request cold costs remain.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from layerwise_prefill_check import DEFAULT_PROMPT_FILE, normalize_prompt_token_ids, write_json
from layerwise_prefill_mooncake_check import finish_child, start_logged_process

CASES = ("10k_off", "10k_on")
CACHE_CHUNK_TOKENS = 1024
SHORT_MAX_MODEL_LEN = 16384
PREFIX = "[PREFILL_PROFILE]"


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", default="/workspace/models/GLM-5.2-w4a8c8-0723")
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    cli.add_argument("--cpu-cache-gb", type=float, default=16, help="Requires this much free /dev/shm and host RAM")
    selection = cli.add_mutually_exclusive_group()
    selection.add_argument("--case", choices=("all", *CASES), default="10k_on")
    selection.add_argument(
        "--include-off", action="store_const", dest="case", const="all", help="Run 10k OFF then ON instead of ON only"
    )
    cli.add_argument("--run-dir", type=Path, help="New, empty results directory")
    cli.add_argument(
        "--analyse-only", type=Path, help="Export an existing run's raw profiles without loading the model"
    )
    cli.add_argument("--child", choices=CASES, help=argparse.SUPPRESS)
    return cli


def build_prompt(tokenizer, article: str, target_tokens: int):
    """Repeat/crop article TEXT, then apply the intact chat template.

    Tokenization need not be strictly monotonic in character count. Keep the
    best fitting candidate encountered, report its real length, and never cut
    encoded chat delimiters just to claim an exact token count.
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

    copies = max(1, target_tokens // len(encode(article)))
    text = (article + "\n\n") * copies
    while len(encode(text)) < target_tokens:
        copies *= 2
        text = (article + "\n\n") * copies
    low, high = 1, len(text)
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle]
        ids = encode(candidate)
        if len(ids) <= target_tokens:
            if best is None or len(ids) > len(best[1]):
                best = candidate, ids
            if len(ids) == target_tokens:
                break
            low = middle + 1
        else:
            high = middle - 1
    if best is None or len(best[1]) <= 4096:
        raise ValueError("Article input must span multiple 4096-token prefill chunks")
    return best


def prepare_inputs(args, root, cases):
    print(f"{PREFIX} loading tokenizer and preparing article inputs", flush=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    article = args.prompt_file.read_text(encoding="utf-8")
    if not article.strip():
        raise ValueError(f"Empty article: {args.prompt_file}")
    (root / "article_source.txt").write_text(article, encoding="utf-8")
    for name, target in (("10k", 10000),):
        if not any(case.startswith(name + "_") for case in cases):
            continue
        text, ids = build_prompt(tokenizer, article, target)
        (root / f"{name}_input.txt").write_text(text, encoding="utf-8")
        write_json(root / f"{name}_prompt.json", {"target_tokens": target, "length": len(ids), "token_ids": ids})
        print(f"{PREFIX} {name}: actual prompt_tokens={len(ids)}; saved text and token IDs", flush=True)


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
            "LMCACHE_STORE_ASYNC": "false",
            "LMCACHE_SAVE_DECODE_CACHE": "false",
            "LMCACHE_SAVE_UNFULL_CHUNK": "true",
            "LMCACHE_SAVE_FULL_CHUNK_IN_DECODE": "false",
            "LMCACHE_ENABLE_SHARED_CPU_CACHE": "true",
            "LMCACHE_SHARED_CPU_CACHE_STRICT": "true",
            "LMCACHE_SHARED_CPU_CACHE_PASSIVE_WRITABLE": "true",
            "LMCACHE_EXTRA_CONFIG": json.dumps({"save_only_first_rank": True}),
        }
    )
    return env


def engine_options(args, case_dir, prompt_len):
    # Reserve the same KV capacity for the OFF/ON pair.
    max_len = max(SHORT_MAX_MODEL_LEN, (prompt_len // CACHE_CHUNK_TOKENS + 1) * CACHE_CHUNK_TOKENS)
    return {
        "model": args.model,
        "trust_remote_code": True,
        "load_format": "safetensors",
        "quantization": "ascend",
        "tensor_parallel_size": len(args.devices.split(",")),
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "enable_expert_parallel": True,
        "gpu_memory_utilization": 0.96,
        "max_model_len": max_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 4096,
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


def capture_request(llm, token_ids, params, case):
    # Capture all compute-prefill chunks, not a fixed number of engine steps.
    llm.start_profile(profile_prefix=case)
    try:
        start = time.perf_counter()
        results = llm.generate({"prompt_token_ids": token_ids}, params, use_tqdm=False)
        return results, time.perf_counter() - start
    finally:
        error_in_flight = sys.exc_info()[0] is not None
        try:
            llm.stop_profile()
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
            llm, prompt["token_ids"], SamplingParams(temperature=0, seed=1024, max_tokens=1), args.child
        )
        result = results[0]
        completion = result.outputs[0]
        report = {
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
        llm.llm_engine.engine_core.shutdown()


def analyse_case(case_dir):
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
    write_json(case_dir / "traces.json", traces)
    if len(traces) != expected:
        raise RuntimeError(f"Expected {expected} worker trace_view.json files, found {len(traces)} in {case_dir}")
    print(f"{PREFIX} {case_dir.name}: {len(traces)} MindStudio traces; paths in {case_dir / 'traces.json'}", flush=True)


def run_cases(args, root, cases):
    for case in cases:
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
    root = (
        args.run_dir.resolve()
        if args.run_dir
        else Path(tempfile.mkdtemp(prefix="layerwise-profile-", dir=".")).resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        cli.error("--run-dir must be empty (nothing was deleted)")
    cases = CASES if args.case == "all" else (args.case,)
    print(f"{PREFIX} results: {root}; full model, TP={len(devices)}, gpu=0.96, eager P only, MTP1", flush=True)
    print(f"{PREFIX} local CPU cache={args.cpu_cache_gb} GiB; no Mooncake/file shim/KV dumps", flush=True)
    prepare_inputs(args, root, cases)
    run_cases(args, root, cases)
    print(f"{PREFIX} done: {root}", flush=True)


if __name__ == "__main__":
    main()
