#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.2 prefill request on 8 local NPUs, with layerwise CPU offload.

Run from the matching vllm-ascend checkout in the server container:
  python tools/glm52_prefill_profile.py --model /path/to/GLM-5.2 2>&1 | tee log.log

Uses 8 layers / dummy weights / TP8, preserves the checkpoint's indexer sharing,
and profiles ALL prefill chunks, not just decode. This is a P-node diagnostic,
not a two-server Mooncake PD benchmark or a model-quality test.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

NUM_LAYERS = 8
NUM_DEVICES = 8
CHUNK_SIZE = 256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local GLM-5.2 model/config directory")
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--prompt-tokens", type=int, default=30000)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    parser.add_argument("--profile-dir", type=Path, default=Path("profile"))
    parser.add_argument("--no-profile", action="store_true", help="Run the same one request without profiler overhead")
    parser.add_argument("--load-format", choices=("dummy", "auto"), default="dummy")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print configuration without importing vLLM"
    )
    args = parser.parse_args(argv)
    devices = args.devices.split(",")
    if len(devices) != NUM_DEVICES or len(set(devices)) != NUM_DEVICES or not all(d.isdigit() for d in devices):
        parser.error("--devices must name exactly 8 distinct local NPU ids")
    if args.chunk_tokens < CHUNK_SIZE or args.chunk_tokens % CHUNK_SIZE:
        parser.error("--chunk-tokens must be a positive multiple of 256")
    if args.prompt_tokens <= args.chunk_tokens:
        parser.error("--prompt-tokens must exceed --chunk-tokens to exercise history KV reload")
    return args


def model_overrides(model):
    config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    total = config.get("num_hidden_layers", 0)
    types = config.get("indexer_types")
    if total < NUM_LAYERS or not isinstance(types, list) or len(types) != total:
        raise ValueError("Expected a GLM-5.2 config with indexer_types for every hidden layer")
    if types[0] != "full" or any(kind not in ("full", "shared") for kind in types):
        raise ValueError("indexer_types must start with 'full' and contain only 'full'/'shared'")
    selected = types[:NUM_LAYERS]
    if "shared" not in selected:
        raise ValueError("The first 8 layers have no shared indexer; this would not test the GLM-5.2 port")
    overrides = {"num_hidden_layers": NUM_LAYERS, "indexer_types": selected}
    pattern = config.get("index_topk_pattern")
    if isinstance(pattern, list):
        if len(pattern) != total or any((p == "S") != (t == "shared") for p, t in zip(pattern, types)):
            raise ValueError("index_topk_pattern disagrees with indexer_types")
        overrides["index_topk_pattern"] = pattern[:NUM_LAYERS]
    vocab_size = config.get("vocab_size", 0)
    if vocab_size < 512:
        raise ValueError("Expected GLM-5.2 vocab_size >= 512")
    return overrides


def environment(devices):
    # Do not inherit an external LMCache/Mooncake/DP deployment. This changes
    # only this process and its children, not the caller's shell or any files.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LMCACHE_", "VLLM_", "MOONCAKE_"))}
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": devices,
            "PYTHONHASHSEED": "0",
            "HCCL_DETERMINISTIC": "strict",
            "MSMONITOR_USE_DAEMON": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": "true",
            "VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE": "false",
            "VLLM_ASCEND_DSA_UNBUNDLE": "1",
            "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
            "VLLM_ASCEND_DSA_SHARED_POOL": "1",
            "VLLM_ASCEND_DSA_SHRINK_LATENT": "0",
            "VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE": "0",
            "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "0",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
            "VLLM_ASCEND_SFA_STAGED_GRAPH": "0",
            "VLLM_ASCEND_SFA_FULL_GRAPH": "0",
            "LMCACHE_CHUNK_SIZE": str(CHUNK_SIZE),
            "LMCACHE_LOCAL_CPU": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "4",
            "LMCACHE_USE_LAYERWISE": "true",
            "LMCACHE_ENABLE_SPARSE_ATTENTION": "true",
            "LMCACHE_DSA_TWO_GROUPS": "true",
            "LMCACHE_STORE_ASYNC": "false",
            "LMCACHE_SAVE_DECODE_CACHE": "false",
            "LMCACHE_SAVE_UNFULL_CHUNK": "true",
            "LMCACHE_SAVE_FULL_CHUNK_IN_DECODE": "false",
            "LMCACHE_ENABLE_SHARED_CPU_CACHE": "true",
            "LMCACHE_SHARED_CPU_CACHE_STRICT": "true",
            "LMCACHE_EXTRA_CONFIG": '{"save_only_first_rank": true}',
        }
    )
    return env


def engine_options(args, overrides):
    options = dict(
        model=str(Path(args.model).resolve()),
        hf_overrides=overrides,
        load_format=args.load_format,
        quantization="ascend",
        trust_remote_code=True,
        skip_tokenizer_init=True,
        tensor_parallel_size=NUM_DEVICES,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        enable_expert_parallel=False,
        max_model_len=args.prompt_tokens + CHUNK_SIZE,
        max_num_seqs=1,
        max_num_batched_tokens=args.chunk_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.9,
        seed=0,
        # P-only: no MTP/draft or decode graph. Only produce the first token.
        enforce_eager=True,
        additional_config={"recompute_scheduler_enable": False},
        kv_transfer_config={
            "kv_connector": "LMCacheAscendConnectorV1Dynamic",
            "kv_role": "kv_producer",
            "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
        },
    )
    if not args.no_profile:
        options["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir.resolve()),
            "ignore_frontend": True,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": False,
        }
    return options


def run_request(llm, sampling_params, prompt_tokens, profile):
    started = False
    try:
        if profile:
            llm.start_profile(profile_prefix="glm52_prefill")
            started = True
        begin = time.perf_counter()
        output = llm.generate(
            {"prompt_token_ids": [100 + i % 256 for i in range(prompt_tokens)]},
            sampling_params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - begin
        if len(output) != 1 or len(output[0].outputs[0].token_ids) != 1:
            raise RuntimeError("Expected exactly one completed prefill request and its first output token")
        return elapsed
    finally:
        try:
            if started:
                # Preserve a model/transfer error if profiler finalization also
                # fails. Never leave worker processes alive after a successful run.
                error_in_flight = sys.exc_info()[0] is not None
                try:
                    llm.stop_profile()
                except Exception as exc:
                    if not error_in_flight:
                        raise
                    print(f"[PREFILL_PROFILE] profiler stop also failed: {exc}", file=sys.stderr)
        finally:
            error_in_flight = sys.exc_info()[0] is not None
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception as exc:
                if not error_in_flight:
                    raise
                print(f"[PREFILL_PROFILE] worker shutdown also failed: {exc}", file=sys.stderr)


def analyse(profile_dir):
    # Worker profiler uses analyse_flag=False. Parse only after all workers
    # have stopped, to avoid competing with the measured workload.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from torch_npu.profiler.profiler import analyse; import sys; analyse(sys.argv[1], max_process_number=2)",
            str(profile_dir.resolve()),
        ],
        check=True,
    )
    traces = sorted(profile_dir.rglob("trace_view.json"))
    if len(traces) != NUM_DEVICES:
        raise RuntimeError(f"Expected 8 worker traces, found {len(traces)} under {profile_dir}")
    print(f"[PREFILL_PROFILE] {len(traces)} trace_view.json files under {profile_dir.resolve()}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    overrides = model_overrides(args.model)
    options = engine_options(args, overrides)
    if args.dry_run:
        print(json.dumps(options, indent=2))
        return
    if not args.no_profile:
        if args.profile_dir.exists() and any(args.profile_dir.iterdir()):
            raise ValueError(
                f"{args.profile_dir} is not empty; use --profile-dir with a new path (nothing was deleted)"
            )
        args.profile_dir.mkdir(parents=True, exist_ok=True)
    child_env = environment(args.devices)
    os.environ.clear()
    os.environ.update(child_env)
    # Deliberately import after selecting the isolated P-node environment.
    from vllm import LLM, SamplingParams

    print(
        f"[PREFILL_PROFILE] TP8 layers=8 indexers={overrides['indexer_types'].count('full')} "
        f"prompt={args.prompt_tokens} chunk={args.chunk_tokens} weights={args.load_format} "
        "local_cpu_offload=true MTP=false",
        flush=True,
    )
    llm = LLM(**options)
    params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False)
    elapsed = run_request(llm, params, args.prompt_tokens, not args.no_profile)
    label = "request_seconds_with_profiler_overhead" if not args.no_profile else "request_seconds"
    print(f"[PREFILL_PROFILE] {label}={elapsed:.6f} (prefill + first-token sampling/IPC; excludes startup)", flush=True)
    if not args.no_profile:
        analyse(args.profile_dir)


if __name__ == "__main__":
    main()
