#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run LoCoMo against fresh local OFF/ON API servers and capture NPU profiles.

Default workload: python /workspace/dataset/benchmark-new/locomo/test_advanced.py
--vllm_port 8000 --vllm_ip 127.0.0.1. No synthetic requests are submitted.
TP8/DP1, eager, MTP1, FlashComm1=1, 4096 compute chunk and local LMCache only.
All model/cache options are inline; no YAML, Mooncake or tensor-dump probes.
Startup is excluded; all benchmark requests (prefill and decode) are profiled.
Prompt construction helpers below are also imported by the correctness tools.
"""

import argparse
import faulthandler
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from layerwise_prefill_check import DEFAULT_PROMPT_FILE, normalize_prompt_token_ids, write_json
from layerwise_prefill_mooncake_check import finish_child, start_logged_process

LEGACY_CASES = ("10k_off", "10k_on", "80k_off", "80k_on")
DEFAULT_LONG_PROMPT_FILE = DEFAULT_PROMPT_FILE.with_name("article_summary_80k.txt")
MAX_PROMPT_FIT_ATTEMPTS = 3
MIN_PROMPT_FRACTION = 0.95
REQUEST_SUBMISSION_TIMEOUT_SECONDS = 120
CACHE_CHUNK_TOKENS = 1024
COMPUTE_CHUNK_TOKENS = 4096
SHORT_MAX_MODEL_LEN = 16384
LONG_MAX_MODEL_LEN = 80000 + COMPUTE_CHUNK_TOKENS
PREFIX = "[PREFILL_PROFILE]"
DEFAULT_MODEL = "/workspace/models/GLM-5.2-w4a8c8-0723"
DEFAULT_BENCHMARK_SCRIPT = Path("/workspace/dataset/benchmark-new/locomo/test_advanced.py")


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
        "--model",
        default=DEFAULT_MODEL,
        help="Checkpoint directory; defaults to GLM-5.2-w4a8c8-0723",
    )
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--benchmark-script", type=Path, default=DEFAULT_BENCHMARK_SCRIPT)
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--startup-timeout", type=float, default=1800)
    cli.add_argument("--benchmark-timeout", type=float, default=7200)
    cli.add_argument("--cpu-cache-gb", type=float, default=24, help="Requires this much free /dev/shm and host RAM")
    selection = cli.add_mutually_exclusive_group()
    selection.add_argument(
        "--case",
        choices=("all", "off", "on"),
        default="all",
        help="Default: LoCoMo OFF then ON, each with a fresh API server",
    )
    selection.add_argument("--include-off", action="store_const", dest="case", const="all", help="Run OFF then ON")
    cli.add_argument("--run-dir", type=Path, help="New, empty results directory")
    cli.add_argument(
        "--analyse-only", type=Path, help="Export an existing run's raw profiles without loading the model"
    )
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


def record_model_identity(model, root):
    """Record the checkpoint configuration before starting expensive workers."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    text_config = getattr(config, "text_config", None) or config
    indexer_types = getattr(text_config, "indexer_types", None)
    info = {
        "model": model,
        "model_type": getattr(text_config, "model_type", None),
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "indexer_types": indexer_types,
        "index_topk_pattern": getattr(text_config, "index_topk_pattern", None),
    }
    write_json(root / "model_info.json", info)
    if indexer_types is None:
        indexer_description = "indexer_types absent; checkpoint declares no shared-indexer schedule"
    else:
        producers = [i for i, kind in enumerate(indexer_types) if kind == "full"]
        shared = sum(kind == "shared" for kind in indexer_types)
        indexer_description = f"indexer producer layers={producers}; shared layers={shared}"
    print(
        f"{PREFIX} checkpoint={model}; layers={info['num_hidden_layers']}; {indexer_description}",
        flush=True,
    )
    return info


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
            "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": str(case.rsplit("_", 1)[-1] == "on").lower(),
            "VLLM_ASCEND_DSA_UNBUNDLE": "1",
            "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
            "VLLM_ASCEND_DSA_SHARED_POOL": "1",
            "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
            "VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE": "0",
            "VLLM_ASCEND_DSA_DISABLE_TARGET_SLOT_MAPPING": "0",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
            "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "0",
            # This benchmark has DP=1 and max_num_seqs=1. The cross-DP
            # scheduler reserves one slot, so enabling it admits no requests.
            "VLLM_ASCEND_BALANCE_SCHEDULING": "0",
            "TASK_QUEUE_ENABLE": "1",
            "CPU_AFFINITY_CONF": "1",
            "ASCEND_AGGREGATE_ENABLE": "1",
            "ASCEND_TRANSPORT_PRINT": "1",
            "ACL_OP_INIT_MODE": "1",
            "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": "600",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            # Compact rank-1 reuse counters replace the verbose timing stream.
            "PD_SERVING_PERF": "0",
            "LMCACHE_PREFILL_START_TIMING": "0",
            "LMCACHE_PREFILL_REUSE_DEBUG_RANK": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
            "LMCACHE_ASCEND_SPARSE_TRANSFER_TOPK": "2048",
            "LMCACHE_CHUNK_SIZE": str(CACHE_CHUNK_TOKENS),
            "LMCACHE_LOCAL_CPU": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(args.cpu_cache_gb),
            "LMCACHE_USE_LAYERWISE": "true",
            "LMCACHE_ENABLE_SPARSE_ATTENTION": "true",
            "LMCACHE_DSA_TWO_GROUPS": "true",
            # The original local layerwise path does not support async stores.
            "LMCACHE_STORE_ASYNC": str(case.rsplit("_", 1)[-1] == "on").lower(),
            "LMCACHE_STORE_ASYNC_MAX_QUEUE_SIZE": "2",
            "LMCACHE_ENABLE_ASYNC_LOADING": "false",
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


@contextmanager
def trace_request_submission(llm, case, case_dir):
    """Diagnose preprocessing/enqueue stalls without tracing the compute loop."""
    engine = llm.llm_engine
    original_add_request = engine.add_request
    stack_dir = case_dir / "startup-stacks"
    stack_dir.mkdir(parents=True, exist_ok=True)
    with (stack_dir / f"frontend-{os.getpid()}.log").open("w", encoding="utf-8") as stack_file:
        stack_file.write(f"{PREFIX} {case}: frontend request submission; pid={os.getpid()}\n")
        stack_file.flush()
        faulthandler.dump_traceback_later(REQUEST_SUBMISSION_TIMEOUT_SECONDS, repeat=False, file=stack_file)

        def add_request(*args, **kwargs):
            print(f"{PREFIX} {case}: engine add_request begin", flush=True)
            result = original_add_request(*args, **kwargs)
            print(f"{PREFIX} {case}: engine add_request returned; waiting for execution", flush=True)
            faulthandler.cancel_dump_traceback_later()
            return result

        engine.add_request = add_request
        try:
            yield
        finally:
            engine.add_request = original_add_request
            faulthandler.cancel_dump_traceback_later()


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


def api_request(port, path, *, method="GET", timeout=10):
    # Local benchmark control must not go through a shell's HTTP proxy.
    request = Request(f"http://127.0.0.1:{port}{path}", method=method)
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return response.read()


def check_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Match the server: allow TIME_WAIT sockets from the previous OFF run,
        # while an existing listener still prevents binding on Linux.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def wait_for_server(proc, port, timeout):
    deadline = time.monotonic() + timeout
    next_notice = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"API server exited with code {proc.returncode}; see server.log")
        try:
            api_request(port, "/health", timeout=2)
            return
        except (URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() >= next_notice:
            print(f"{PREFIX} waiting for API startup on 127.0.0.1:{port}; see server.log", flush=True)
            next_notice = time.monotonic() + 30
        time.sleep(1)
    raise TimeoutError(f"API server was not ready after {timeout:g}s; see server.log")


def server_command(args, options):
    command = [
        sys.executable,
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    for name, value in options.items():
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            command.extend((flag, json.dumps(value) if isinstance(value, (dict, list)) else str(value)))
    return command


def benchmark_command(args):
    return [
        sys.executable,
        str(args.benchmark_script.resolve()),
        "--vllm_port",
        str(args.port),
        "--vllm_ip",
        "127.0.0.1",
    ]


def run_benchmark(args, server, env, case_dir):
    command = benchmark_command(args)
    write_json(case_dir / "benchmark_command.json", command)
    proc = start_logged_process(
        command,
        env,
        case_dir / "benchmark.log",
        "LoCoMo",
        prefix=PREFIX,
        cwd=str(args.benchmark_script.resolve().parent),
    )
    deadline = time.monotonic() + args.benchmark_timeout
    try:
        while proc.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("API server exited during LoCoMo; see server.log")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"LoCoMo exceeded {args.benchmark_timeout:g}s; see benchmark.log")
            time.sleep(1)
        if proc.returncode:
            raise RuntimeError(f"LoCoMo exited with code {proc.returncode}; see benchmark.log")
    finally:
        finish_child(proc)


def run_cases(args, root, cases):
    for index, case in enumerate(cases):
        if index:
            clear_shm(Path("/dev/shm"))
        case_dir = root / case
        case_dir.mkdir()
        check_port_available(args.port)
        # Retain the previous long-profile capacity, irrespective of LoCoMo's
        # request lengths. The external benchmark controls prompts/generation.
        options = engine_options(args, case_dir, LONG_MAX_MODEL_LEN)
        options.pop("worker_extension_cls")  # No synthetic head/tail capture plan.
        command = server_command(args, options)
        write_json(case_dir / "engine_options.json", options)
        write_json(case_dir / "server_command.json", command)
        env = case_environment(args, case)
        env["PYTHONUNBUFFERED"] = "1"
        for name in ("NO_PROXY", "no_proxy"):
            env[name] = ",".join(filter(None, (env.get(name), "127.0.0.1", "localhost")))
        reuse_log = case_dir / "reuse.log"
        reuse_log.touch()
        env["LMCACHE_PREFILL_REUSE_DEBUG_FILE"] = str(reuse_log.resolve())
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
        report = {
            "case": case,
            "status": "starting",
            "workload": "LoCoMo",
            "scope": "local CPU cache; prefill and decode profiles; no PD transfer",
        }
        write_json(case_dir / "result.json", report)
        profiling = False
        failure = None
        try:
            wait_for_server(proc, args.port, args.startup_timeout)
            print(f"{PREFIX} {case}: server ready; profiler start", flush=True)
            api_request(args.port, "/start_profile", method="POST", timeout=300)
            profiling = True
            started = time.monotonic()
            print(f"{PREFIX} {case}: running LoCoMo; stdout/stderr -> benchmark.log", flush=True)
            run_benchmark(args, proc, env, case_dir)
            report.update(status="benchmark_complete", benchmark_seconds_with_profiler=time.monotonic() - started)
        except BaseException as error:
            failure = error
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            try:
                if profiling and proc.poll() is None:
                    print(f"{PREFIX} {case}: profiler stop/export raw data", flush=True)
                    api_request(args.port, "/stop_profile", method="POST", timeout=1800)
            except Exception as error:
                report["profile_stop_error"] = str(error)
                report["status"] = "failed"
                if failure is None:
                    failure = error
            finally:
                write_json(case_dir / "result.json", report)
                finish_child(proc)
        if failure is not None:
            raise failure
        try:
            analyse_case(case_dir)
            report["status"] = "complete"
        except Exception as error:
            report.update(status="failed", profile_export_error=str(error))
            raise
        finally:
            write_json(case_dir / "result.json", report)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.analyse_only:
        for case in ("off", "on", *LEGACY_CASES):
            case_dir = args.analyse_only.resolve() / case
            if (case_dir / "engine_options.json").is_file():
                analyse_case(case_dir)
        return
    if os.name != "posix":
        cli.error("Run on the Linux Ascend server")
    devices = args.devices.split(",")
    if not all(d.isdigit() for d in devices) or len(devices) != len(set(devices)) or args.cpu_cache_gb <= 0:
        cli.error("Specify distinct NPU device IDs and a positive CPU cache size")
    if not 0 < args.port < 65536 or args.startup_timeout <= 0 or args.benchmark_timeout <= 0:
        cli.error("Specify a valid port and positive timeouts")
    if not args.benchmark_script.is_file():
        cli.error(f"LoCoMo script does not exist: {args.benchmark_script}")
    check_port_available(args.port)
    root = (
        args.run_dir.resolve()
        if args.run_dir
        else Path(tempfile.mkdtemp(prefix="layerwise-profile-", dir=".")).resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        cli.error("--run-dir must be empty (nothing was deleted)")
    clear_shm(Path("/dev/shm"))
    check_shm_capacity(Path("/dev/shm"), args.cpu_cache_gb)
    cases = ("off", "on") if args.case == "all" else (args.case,)
    print(f"{PREFIX} model: {args.model}", flush=True)
    print(f"{PREFIX} results: {root}; full model, TP={len(devices)}, gpu=0.97, eager P only, MTP1", flush=True)
    print(f"{PREFIX} local CPU cache={args.cpu_cache_gb} GiB; no Mooncake/file shim/KV dumps", flush=True)
    print(f"{PREFIX} LoCoMo: {' '.join(benchmark_command(args))}", flush=True)
    record_model_identity(args.model, root)
    run_cases(args, root, cases)
    print(f"{PREFIX} done: {root}", flush=True)


if __name__ == "__main__":
    main()
