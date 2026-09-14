#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce staged/full SFA serving with one active request and an idle DP peer.

This intentionally differs from ``sfa_graph_benchmark.py``: it launches the
online server as TP4 x DP2 with expert parallelism, production request buckets
and the recompute scheduler.  The eight-layer dummy model keeps it runnable on
one eight-NPU host; it does not claim to reproduce real-weight kernel time or
cross-node DP latency.
"""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import mkdtemp

from sfa_full_graph_parity import DEFAULT_DEVICES, DEFAULT_MODEL, parse_devices
from sfa_graph_benchmark import benchmark_environment

SERVED_MODEL = "sfa-serving-repro"
SERVER_READY_TIMEOUT = 1800
SERVER_STOP_TIMEOUT = 120


def serving_environment(mode: str, devices: str, *, diagnose: bool) -> dict[str, str]:
    environment = benchmark_environment(mode, devices)
    environment.update(
        {
            "VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": "4,8,12,16",
            "VLLM_ASCEND_DSA_DISABLE_TARGET_SLOT_MAPPING": "0",
            "VLLM_ASCEND_BALANCE_SCHEDULING": "1",
            "LMCACHE_ENABLE_DSA_COLD_COMPACT_LOAD": "true",
            # The offline parity fixture uses independent per-rank CPU caches.
            # Serving cold-compact loading instead needs a shared rank0 store
            # with passive TP readers. Override the inherited policy together.
            "LMCACHE_ENABLE_SHARED_CPU_CACHE": "true",
            "LMCACHE_SHARED_CPU_CACHE_STRICT": "true",
            "LMCACHE_EXTRA_CONFIG": '{"save_only_first_rank": true}',
            "LMCACHE_ASCEND_SPARSE_TRANSFER_TOPK": "2048",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "8",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_BUFFSIZE": "256",
            "OMP_PROC_BIND": "false",
            "OMP_NUM_THREADS": "10",
            "ASCEND_BUFFER_POOL": "4:8",
            "TASK_QUEUE_ENABLE": "1",
            "ASCEND_AGGREGATE_ENABLE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": str(SERVER_READY_TIMEOUT),
            # Keep the clean run aligned with the deployment. Diagnostic hooks
            # own the rejection-stage recorder, so isolate them from PD timing.
            "PD_SERVING_PERF": "0" if diagnose else "1",
        }
    )
    return environment


def client_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        environment.pop(name, None)
    return environment


def server_command(args: argparse.Namespace, mode: str) -> list[str]:
    additional = {
        "recompute_scheduler_enable": True,
        "multistream_overlap_shared_expert": False,
        "fuse_qknorm_rope": False,
        "fuse_muls_add": True,
        "enable_npugraph_ex": True,
        "sfa_benchmark": True,
        "sfa_benchmark_serving": True,
    }
    kv_transfer = {
        "kv_connector": "LMCacheAscendConnectorV1Dynamic",
        "kv_role": "kv_both",
        "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
        "engine_id": "sfa-serving-repro",
    }
    return [
        "vllm",
        "serve",
        args.model,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--load-format",
        "dummy",
        "--hf-overrides",
        json.dumps({"num_hidden_layers": 8}),
        "--quantization",
        "ascend",
        "--gpu-memory-utilization",
        "0.9",
        "--tensor-parallel-size",
        "4",
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "2",
        "--enable-expert-parallel",
        "--max-model-len",
        str(max(140000, args.prompt_tokens + args.output_tokens + 32)),
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "4096",
        "--seed",
        "1024",
        "--speculative-config",
        json.dumps({"num_speculative_tokens": 1, "method": "deepseek_mtp"}),
        "--compilation-config",
        json.dumps({"cudagraph_mode": "PIECEWISE"}),
        "--additional-config",
        json.dumps(additional),
        "--worker-cls",
        "vllm_ascend.worker.sfa_benchmark_worker.SFABenchmarkWorker",
        "--no-enable-prefix-caching",
        "--kv-transfer-config",
        json.dumps(kv_transfer),
    ]


def client_command(args: argparse.Namespace, mode_dir: Path) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        SERVED_MODEL,
        # The API alias is not a Hugging Face model/tokenizer identifier.
        # Load the same local tokenizer and custom code as the server.
        "--tokenizer",
        args.model,
        "--trust-remote-code",
        "--header",
        "X-data-parallel-rank=0",
        "--dataset-name",
        "random",
        "--num-prompts",
        "1",
        "--random-input-len",
        str(args.prompt_tokens),
        "--random-output-len",
        str(args.output_tokens),
        "--random-range-ratio",
        "0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        "1",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--ready-check-timeout-sec",
        "0",
        "--disable-tqdm",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(mode_dir),
        "--result-filename",
        "client.json",
        "--seed",
        "0",
    ]


def request_json(url: str, body: dict | None = None, timeout: float = 10) -> dict | None:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload) if payload else None


def wait_until_ready(process: subprocess.Popen, port: int, log_path: Path) -> None:
    deadline = time.monotonic() + SERVER_READY_TIMEOUT
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            raise RuntimeError("Server exited during startup:\n" + "\n".join(tail))
        try:
            request_json(url, timeout=2)
            return
        except (OSError, urllib.error.HTTPError):
            time.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {SERVER_READY_TIMEOUT}s: {log_path}")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=SERVER_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=30)


def server_log_tail(log_path: Path) -> str:
    """Keep errors self-contained without reading a potentially huge server log."""
    try:
        with log_path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 65536))
            return "\n".join(log.read().decode("utf-8", errors="replace").splitlines()[-80:])
    except OSError as exc:
        return f"Unable to read {log_path}: {exc}"


def print_stall_snapshots(mode: str, log_path: Path) -> None:
    """Forward bounded worker snapshots even after engine death breaks the RPC."""
    marker = "[SFA_SERVING_STALL] "
    try:
        with log_path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                if marker in line:
                    print(f"[SFA_SERVING_STALL] mode={mode} " + line.split(marker, 1)[1].rstrip(), flush=True)
    except OSError as exc:
        print(f"[SFA_SERVING] Unable to read stall snapshots from {log_path}: {exc}", flush=True)


def timing_rpc(args: argparse.Namespace, method: str, *, server_log: Path | None = None) -> dict | None:
    body: dict = {"method": method, "timeout": SERVER_READY_TIMEOUT}
    if method == "benchmark_start_decode_timing":
        body["args"] = [str(args.prompt_tokens)]
    try:
        return request_json(
            f"http://127.0.0.1:{args.port}/collective_rpc",
            body,
            timeout=SERVER_READY_TIMEOUT,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read(8192).decode("utf-8", errors="replace")
        message = f"{method} failed: HTTP {exc.code} {exc.reason}\nResponse: {detail}"
        if server_log is not None:
            message += f"\nServer log: {server_log}\n{server_log_tail(server_log)}"
        raise RuntimeError(message) from exc


def run_mode(args: argparse.Namespace, mode: str, root: Path) -> dict:
    mode_dir = root / mode
    mode_dir.mkdir()
    log_path = mode_dir / "server.log"
    command = server_command(args, mode)
    (mode_dir / "server-command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=serving_environment(mode, args.devices, diagnose=args.diagnose),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_until_ready(process, args.port, log_path)
            if args.diagnose:
                timing_rpc(args, "benchmark_start_decode_timing", server_log=log_path)
            subprocess.run(client_command(args, mode_dir), check=True, env=client_environment())
            result = json.loads((mode_dir / "client.json").read_text(encoding="utf-8"))
            if (
                result.get("completed") != 1
                or result.get("failed")
                or result.get("output_lens") != [args.output_tokens]
            ):
                # Some bench clients mark an interrupted HTTP-200 stream as
                # successful after its first token. Do not report it as TPOT=0
                # or hide the incomplete generation behind a follow-up RPC 500.
                raise RuntimeError(
                    f"{mode} incomplete generation: expected one request with {args.output_tokens} output tokens; "
                    f"completed={result.get('completed')} failed={result.get('failed')} "
                    f"output_lens={result.get('output_lens')}.\nServer log: {log_path}\n{server_log_tail(log_path)}"
                )
            if args.diagnose:
                timing = timing_rpc(args, "benchmark_stop_decode_timing", server_log=log_path)
                (mode_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
                print_timing(mode, timing)
        finally:
            try:
                stop_server(process)
            finally:
                if args.diagnose:
                    print_stall_snapshots(mode, log_path)
    return result


def print_timing(mode: str, response: dict | None) -> None:
    workers = response.get("results", []) if isinstance(response, dict) else []
    if not workers:
        raise RuntimeError(f"{mode} diagnostic RPC returned no worker timings")
    if all("async_scheduling" in worker for worker in workers):
        async_workers = sum(bool(worker["async_scheduling"]) for worker in workers)
        print(
            f"[SFA_SERVING_TIMING] {mode} async_workers={async_workers}/{len(workers)} "
            "(async output-token histograms are unavailable; client TPOT is unchanged)",
            flush=True,
        )
    stage_names = (
        "dp.batch_sync",
        "target.forward",
        "source.prepare",
        "source.bind",
        "full.prepare_agreement",
        "root.run",
        "mtp.propose",
        "sampling",
    )
    for stage in stage_names:
        values = []
        for worker in workers:
            forwards = worker.get("decode_steps", 0)
            metric = worker.get("stages", {}).get(stage, {}).get("wall", {})
            if forwards and metric.get("count"):
                values.append(metric["total_ms"] / forwards)
        if values:
            print(
                f"[SFA_SERVING_TIMING] {mode} {stage} "
                f"ms/forward mean={sum(values) / len(values):.3f} "
                f"max_rank={max(values):.3f}",
                flush=True,
            )


def compare_results(staged: dict, full: dict) -> dict:
    fields = ("mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "mean_e2el_ms")
    if staged.get("input_lens") != full.get("input_lens") or staged.get("output_lens") != full.get("output_lens"):
        raise ValueError("Staged/full serving requests did not have identical token lengths")
    staged_outputs = staged.get("generated_texts")
    full_outputs = full.get("generated_texts")
    if not isinstance(staged_outputs, list) or not isinstance(full_outputs, list):
        raise ValueError("Serving benchmark must save detailed generated outputs")
    comparison = {
        "staged": {name: staged.get(name) for name in fields},
        "full": {name: full.get(name) for name in fields},
        "input_lens": staged.get("input_lens"),
        "output_lens": staged.get("output_lens"),
        "outputs_equal": staged_outputs == full_outputs,
    }
    staged_tpot, full_tpot = staged.get("mean_tpot_ms"), full.get("mean_tpot_ms")
    if not isinstance(staged_tpot, int | float) or not isinstance(full_tpot, int | float) or staged_tpot <= 0:
        raise ValueError("Serving benchmark did not return valid TPOT measurements")
    comparison["tpot_reduction_percent"] = (1 - full_tpot / staged_tpot) * 100
    comparison["decode_speedup"] = staged_tpot / full_tpot
    return comparison


def validate_args(args: argparse.Namespace) -> None:
    devices = parse_devices(args.devices)
    if len(devices) != 8:
        raise ValueError("Serving reproduction requires exactly eight distinct NPU devices for TP4 x DP2")
    if not Path(args.model, "config.json").is_file():
        raise FileNotFoundError(f"Missing model configuration: {args.model}/config.json")
    if args.prompt_tokens <= 4096 or args.output_tokens < 2:
        raise ValueError("Use prompt_tokens > 4096 and output_tokens >= 2")
    if not (1 <= args.port <= 65535):
        raise ValueError("Port must be between 1 and 65535")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--devices", default=DEFAULT_DEVICES)
    parser.add_argument("--prompt-tokens", type=int, default=30000)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--output-dir", type=Path, default=Path("profile"))
    parser.add_argument("--order", choices=("staged,full", "full,staged", "staged", "full"), default="staged,full")
    parser.add_argument("--diagnose", action="store_true", help="Add request-bounded host timings; not a clean run")
    args = parser.parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix="sfa-serving-", dir=args.output_dir.resolve()))
    print(f"[SFA_SERVING] results: {root}", flush=True)
    results = {mode: run_mode(args, mode, root) for mode in args.order.split(",")}
    if len(results) == 1:
        mode, result = next(iter(results.items()))
        print(
            f"[SFA_SERVING] {mode} completed: output_lens={result['output_lens']} "
            f"TPOT={result.get('mean_tpot_ms')}ms; single-mode run, no speedup comparison",
            flush=True,
        )
        return
    comparison = compare_results(results["staged"], results["full"])
    comparison["config"] = {
        "model": args.model,
        "devices": args.devices,
        "topology": "TP4xDP2+EP; one active request and one idle DP rank",
        "layers": 8,
        "weights": "dummy",
        "capture_request_sizes": [4, 8, 12, 16],
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "diagnose": args.diagnose,
    }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(
        "[SFA_SERVING] TPOT "
        f"staged={comparison['staged']['mean_tpot_ms']:.3f}ms "
        f"full={comparison['full']['mean_tpot_ms']:.3f}ms "
        f"reduction={comparison['tpot_reduction_percent']:.2f}% "
        f"speedup={comparison['decode_speedup']:.3f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
