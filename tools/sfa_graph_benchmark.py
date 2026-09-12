#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP8/eight-layer staged vs full graph: TPOT, optional diagnostics or traces.

No server/client is needed. Each mode has a fresh engine and ordinary prefill;
this is a performance test, NOT the one-prefill numerical-parity test.
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

from sfa_full_graph_parity import (
    DEFAULT_DEVICES,
    DEFAULT_MODEL,
    ENGINE_SHUTDOWN_TIMEOUT,
    FIXED_TOKEN,
    child_environment,
    engine_options,
    parse_devices,
    preflight_dependencies,
    shutdown_engine,
    track_workers,
)
from sfa_graph_trace import analyse_traces

PROFILE_SKIP_TOKENS = 8
BENCHMARK_PROMPT_TOKENS = 30000


def benchmark_environment(mode: str, devices: str) -> dict[str, str]:
    environment = child_environment("graph", devices)
    # Both modes keep the original staged/PIECEWISE compilation. Only this
    # production switch differs; comparing to enforce_eager overstates gains.
    environment["VLLM_ASCEND_SFA_FULL_GRAPH"] = str(int(mode == "full"))
    environment["LMCACHE_LOG_LEVEL"] = "WARNING"
    for name in ("ASCEND_LAUNCH_BLOCKING", "CUDA_LAUNCH_BLOCKING"):
        environment.pop(name, None)
    return environment


def benchmark_options(args) -> dict:
    options = engine_options(
        SimpleNamespace(child="graph", devices=args.devices, model=args.model, reference=None, atol=0, rtol=0)
    )
    options.update(
        worker_cls="vllm_ascend.worker.sfa_benchmark_worker.SFABenchmarkWorker",
        additional_config={"sfa_benchmark": True},
        max_model_len=args.prompt_tokens + max(args.output_tokens, args.profile_tokens + PROFILE_SKIP_TOKENS) + 32,
        disable_log_stats=True,
    )
    if args.profile:
        options["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(Path(args.run_dir) / args.child),
            "ignore_frontend": True,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": False,
        }
    return options


def prompt_ids(length: int, ordinal: int) -> list[int]:
    # Different first token prevents warmup/other measured requests from
    # reusing an LMCache prefix. Identical ordinals match across both engines.
    # These IDs remain in the same ordinary-token range as the parity fixture.
    return [FIXED_TOKEN + ordinal] + [FIXED_TOKEN + (i + ordinal) % 257 for i in range(length - 1)]


def request_metrics(tokens: list[int], arrivals: list[tuple[int, float]], started: float) -> dict:
    if len(arrivals) < 2 or arrivals[-1][0] != len(tokens):
        raise RuntimeError("Need multiple token emissions to measure decode TPOT")
    first_count, first_time = arrivals[0]
    last_count, last_time = arrivals[-1]
    count = last_count - first_count
    elapsed = last_time - first_time
    if count <= 0 or not math.isfinite(elapsed) or elapsed <= 0 or first_time < started:
        raise RuntimeError("Invalid decode timing interval")
    return {
        "token_ids": tokens,
        "first_emission_tokens": first_count,
        "decode_tokens": count,
        "emissions": len(arrivals),
        "ttft_ms": (first_time - started) * 1000,
        "decode_ms": elapsed * 1000,
        "tpot_ms": elapsed * 1000 / count,
        "decode_tokens_per_second": count / elapsed,
    }


def generate_request(llm, args, ordinal: int, *, profile: bool = False) -> dict:
    # Bypass LLM.generate's FINAL_ONLY setting, but use the real input
    # processor, scheduler, sampling/MTP and output processor unchanged.
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    count = args.profile_tokens + PROFILE_SKIP_TOKENS if profile else args.output_tokens
    engine = llm.llm_engine
    params = SamplingParams(
        temperature=0,
        seed=0,
        min_tokens=count,
        max_tokens=count,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    if engine.has_unfinished_requests():
        raise RuntimeError("Benchmark requires an idle single-request engine")
    tokens, arrivals = [], []
    attempted_profile = False
    finished = False
    started = time.perf_counter()
    request_id = f"sfa-benchmark-{ordinal}"
    # add_request returns a randomized INTERNAL ID; RequestOutput deliberately
    # restores the caller's external ID. Do not compare those two namespaces.
    engine.add_request(request_id, {"prompt_token_ids": prompt_ids(args.prompt_tokens, ordinal)}, params)
    try:
        while engine.has_unfinished_requests():
            outputs = engine.step()
            now = time.perf_counter()
            for output in outputs:
                if output.request_id != request_id or len(output.outputs) != 1:
                    raise RuntimeError("Unexpected concurrent benchmark output")
                current = list(output.outputs[0].token_ids)
                if current[: len(tokens)] != tokens:
                    raise RuntimeError("Output processor did not return cumulative committed tokens")
                if len(current) > len(tokens):
                    arrivals.append((len(current), now))
                tokens = current
                finished = output.finished
            if profile and not attempted_profile and len(tokens) >= PROFILE_SKIP_TOKENS and not finished:
                attempted_profile = True
                llm.start_profile(profile_prefix=f"sfa_{args.child}")
    finally:
        if attempted_profile:
            # One attempt only: torch_npu profiler stop is not idempotent.
            error = sys.exc_info()[1]
            try:
                llm.stop_profile()
            except Exception as stop_error:
                if error is None:
                    raise
                error.add_note(f"Profiler stop also failed: {stop_error}")
    if not finished or len(tokens) != count:
        raise RuntimeError(f"Incomplete generation: {len(tokens)}/{count}")
    if profile:
        if not attempted_profile:
            raise RuntimeError("Generation finished before decode profiling could start")
        # Never turn instrumented timings into a performance result.
        return {"profiled": True, "output_tokens": len(tokens)}
    return request_metrics(tokens, arrivals, started)


def worker_state(llm, args) -> list[dict]:
    reports = llm.collective_rpc("benchmark_state", timeout=ENGINE_SHUTDOWN_TIMEOUT)
    size = len(parse_devices(args.devices))
    if len(reports) != size or {r["rank"] for r in reports} != set(range(size)):
        raise RuntimeError("Missing/duplicate benchmark worker ranks")
    for report in reports:
        if report["layers"] != 8 or not report["staged"] or report["full"] != (args.child == "full"):
            raise RuntimeError(f"Incorrect benchmark worker configuration: {report}")
        if args.child == "full" and (not report["root_sealed"] or report["root_keys"] < 1):
            raise RuntimeError(f"Full target graph was not captured: {report}")
    return sorted(reports, key=lambda r: r["rank"])


def replay_delta(before: list[dict], after: list[dict], mode: str) -> list[int]:
    if [(r["rank"], r["pid"]) for r in before] != [(r["rank"], r["pid"]) for r in after]:
        raise RuntimeError("Worker identities changed during generation")
    delta = [b["root_replays"] - a["root_replays"] for a, b in zip(before, after)]
    if mode == "full" and (not delta or min(delta) <= 0 or len(set(delta)) != 1):
        raise RuntimeError(f"Missing/inconsistent target root replays: {delta}")
    if mode == "staged" and any(delta):
        raise RuntimeError("Staged baseline unexpectedly used full graph")
    return delta


def distribution(values: list[float]) -> dict:
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Timing samples must be finite and positive")
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def diagnose_request(llm, args, ordinal: int) -> dict:
    """One separate request; worker gating excludes ALL chunked prefill steps.

    Arm before submission, stop after completion: no mid-request control RPCs
    and no engine/client progress race when selecting the decode interval.
    """
    attempted = False
    workers = None
    try:
        attempted = True
        llm.collective_rpc("benchmark_start_decode_timing", args=(args.prompt_tokens,), timeout=ENGINE_SHUTDOWN_TIMEOUT)
        request = generate_request(llm, args, ordinal)
    finally:
        if attempted:
            error = sys.exc_info()[1]
            try:
                workers = llm.collective_rpc("benchmark_stop_decode_timing", timeout=ENGINE_SHUTDOWN_TIMEOUT)
            except Exception as stop_error:
                if error is None:
                    raise
                error.add_note(f"Decode timing stop also failed: {stop_error}")
    expected = set(range(len(parse_devices(args.devices))))
    if len(workers) != len(expected) or {w["rank"] for w in workers} != expected:
        raise RuntimeError("Missing/duplicate decode timing ranks")
    for worker in workers:
        steps = worker["decode_steps"]
        target_calls = worker["stages"].get("target.forward", {}).get("wall", {}).get("count", 0)
        replays = worker["root_replays"]
        if steps <= 0 or target_calls != steps:
            raise RuntimeError(f"Decode timing missed target forwards on rank {worker['rank']}")
        if replays != (steps if args.child == "full" else 0):
            raise RuntimeError(f"Decode timing root coverage mismatch on rank {worker['rank']}")
        stages = worker["stages"]
        if args.child == "full" and stages.get("root.replay_submit", {}).get("wall", {}).get("count", 0) != steps:
            raise RuntimeError(f"Root replay timing hook was not reached on rank {worker['rank']}")
        for index in range(8):
            expected_stage = f"metadata.L{index}" if args.child == "full" else f"retrieve.L{index}"
            if stages.get(expected_stage, {}).get("wall", {}).get("count", 0) != steps:
                raise RuntimeError(f"Incomplete {expected_stage} timing on rank {worker['rank']}")
            if args.child == "full" and stages.get(f"retrieve.L{index}", {}).get("wall", {}).get("count", 0):
                raise RuntimeError("Full target replay unexpectedly executed a Python layer retrieval")
    if len({w["decode_steps"] for w in workers}) != 1:
        raise RuntimeError("Decode step counts differ across TP ranks")
    return {
        "mode": args.child,
        "scope": "separate instrumented decode request; not a performance sample; no profiler",
        "request": request,
        "workers": sorted(workers, key=lambda w: w["rank"]),
    }


def print_decode_timing(report: dict) -> None:
    mode, workers, request = report["mode"], report["workers"], report["request"]
    steps = workers[0]["decode_steps"]
    print(
        f"[SFA_TIMING] {mode} diagnostic-only: decode={request['decode_ms']:.3f}ms "
        f"committed={request['decode_tokens']} forwards={steps} "
        f"committed/forward={request['decode_tokens'] / steps:.3f}; excluded from TPOT comparison",
        flush=True,
    )
    for worker in workers:
        print(
            f"[SFA_TIMING] {mode} rank={worker['rank']} forwards={worker['decode_steps']} "
            f"roots={worker['root_replays']} source_updates={worker['source_binding_updates']} "
            f"Q_hist={worker['query_tokens_histogram']} sampled_hist={worker['sampled_tokens_histogram']} "
            f"prefill_excluded={worker['prefill_steps_excluded']} "
            f"event_drops={worker['device_intervals_dropped']}",
            flush=True,
        )
    print(
        f"[SFA_TIMING] {mode} ms/forward: wall/self=rank mean(max rank); "
        "cpu=exclusive thread CPU; call_max=slowest call; stream=mean(max rank) current-stream span. "
        "Nested wall times and stream spans MUST NOT be added together.",
        flush=True,
    )
    names = sorted({name for w in workers for name in w["stages"]})
    for name in names:
        metrics = [w["stages"].get(name, {}) for w in workers]

        def normalized(key, metrics=metrics):
            values = [m.get(key, {}).get("total_ms", 0.0) / w["decode_steps"] for m, w in zip(metrics, workers)]
            return f"{statistics.mean(values):.3f}({max(values):.3f})"

        calls = [m.get("wall", {}).get("count", 0) / w["decode_steps"] for m, w in zip(metrics, workers)]
        call_max = max(m.get("wall", {}).get("max_ms", 0.0) for m in metrics)
        device = f" stream={normalized('stream_span')}" if any("stream_span" in m for m in metrics) else ""
        print(
            f"[SFA_TIMING] {mode} {name} calls/fwd={min(calls):.2f}..{max(calls):.2f} "
            f"wall={normalized('wall')} self={normalized('self_wall')} "
            f"cpu={normalized('self_cpu')} call_max={call_max:.3f}{device}",
            flush=True,
        )
    # Same decode interval, but this residual also includes result transport,
    # worker idle gaps and boundary skew. Do not label it pure scheduler time.
    worker_times = [
        sum(
            w["stages"].get(name, {}).get("wall", {}).get("total_ms", 0.0)
            for name in ("worker.execute", "worker.sample")
        )
        for w in workers
    ]
    residuals = [(request["decode_ms"] - value) / steps for value in worker_times]
    print(
        f"[SFA_TIMING] {mode} engine_minus_worker ms/forward="
        f"{min(residuals):.3f}..{max(residuals):.3f} (IPC/scheduling/idle + boundary skew; not pure scheduler). "
        "root.run self is mostly the existing completion fence plus bookkeeping, NOT proof of fence overhead.",
        flush=True,
    )


def run_child(args) -> None:
    from vllm import LLM

    llm = LLM(**benchmark_options(args))
    workers = []
    try:
        identities = llm.collective_rpc("benchmark_process_info", timeout=ENGINE_SHUTDOWN_TIMEOUT)
        workers = track_workers(identities, len(parse_devices(args.devices)))
        state = worker_state(llm, args)
        for index in range(args.warmups):
            generate_request(llm, args, index)
        samples = []
        for index in range(args.repeats):
            before = worker_state(llm, args)  # RPC/fence OUTSIDE the timed request.
            sample = generate_request(llm, args, args.warmups + index)
            after = worker_state(llm, args)
            sample["root_replays_per_rank"] = replay_delta(before, after, args.child)
            sample["source_binding_updates_per_rank"] = [
                b["source_binding_updates"] - a["source_binding_updates"] for a, b in zip(before, after)
            ]
            samples.append(sample)
            binding_note = (
                f" source_updates(rank0)={sample['source_binding_updates_per_rank'][0]}"
                f" root_replays(rank0)={sample['root_replays_per_rank'][0]}"
                if args.child == "full"
                else ""
            )
            print(
                f"[SFA_BENCH] {args.child} {index + 1}/{args.repeats} "
                f"TPOT={sample['tpot_ms']:.3f} ms/token decode={sample['decode_tokens_per_second']:.2f} token/s"
                f"{binding_note}",
                flush=True,
            )
        report = {
            "mode": args.child,
            "samples": samples,
            "tpot_ms": distribution([s["tpot_ms"] for s in samples]),
            "workers": state,
            "profiled_measurements": False,
        }
        # Save measurements even if the subsequent, separate profiling fails.
        Path(args.run_dir, f"{args.child}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.profile:
            generate_request(llm, args, args.warmups + args.repeats, profile=True)
        if args.diagnose:
            diagnostic = diagnose_request(llm, args, args.warmups + args.repeats)
            Path(args.run_dir, f"{args.child}-timing.json").write_text(
                json.dumps(diagnostic, indent=2), encoding="utf-8"
            )
            print_decode_timing(diagnostic)
        released = llm.collective_rpc("benchmark_release_resources", timeout=ENGINE_SHUTDOWN_TIMEOUT)
        if sorted((r["rank"], r["pid"]) for r in released) != [(r["rank"], r["pid"]) for r in state]:
            raise RuntimeError("Some benchmark workers did not acknowledge resource release")
    finally:
        error = sys.exc_info()[1]
        try:
            shutdown_engine(llm, workers)
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Benchmark cleanup failed: {cleanup_error}")
    print(f"[SFA_BENCH] {args.child} cleanup complete: all {len(workers)} workers exited", flush=True)


def compare_results(staged: dict, full: dict) -> dict:
    a, b = staged["samples"], full["samples"]
    if not a or len(a) != len(b) or staged["profiled_measurements"] or full["profiled_measurements"]:
        raise ValueError("Need equally sized, unprofiled measurements")
    if any(len(x["token_ids"]) != len(y["token_ids"]) for x, y in zip(a, b)):
        raise ValueError("Output lengths differ between modes")
    baseline = distribution([s["tpot_ms"] for s in a])
    optimized = distribution([s["tpot_ms"] for s in b])
    return {
        "staged_tpot_ms": baseline,
        "full_tpot_ms": optimized,
        "tpot_reduction_percent": (1 - optimized["mean"] / baseline["mean"]) * 100,
        "decode_speedup": baseline["mean"] / optimized["mean"],
        "tokens_equal": all(x["token_ids"] == y["token_ids"] for x, y in zip(a, b)),
        "scope": "offline engine decode incl. scheduling/IPC/MTP/sampling; excludes prefill and profiler",
    }


def validate_args(args) -> None:
    if args.profile and args.diagnose:
        raise ValueError("Choose --diagnose (no profiler) OR --profile")
    parse_devices(args.devices)
    if not Path(args.model, "config.json").is_file():
        raise FileNotFoundError(f"Missing model configuration: {args.model}/config.json")
    if args.prompt_tokens <= 4096:
        raise ValueError("Prompt must exceed the 4096-token MTP scratch prefix")
    if args.output_tokens < 2 or args.profile_tokens < 2 or args.warmups < 1 or args.repeats < 2:
        raise ValueError("Need output/profile tokens >= 2, warmups >= 1, repeats >= 2")
    if args.warmups + args.repeats >= 257:
        raise ValueError("Keep warmups + repeats below 257 distinct fixture prompts")


def run_pair(args) -> None:
    validate_args(args)
    root = args.profile_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    args.run_dir = mkdtemp(prefix="sfa-", dir=root)
    print(f"[SFA_BENCH] results/traces: {args.run_dir}", flush=True)
    print("[SFA_BENCH] staged vs full; performance BEFORE separate diagnostics/profile; no parity probes", flush=True)
    for mode in ("preflight", *args.order.split(",")):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                mode,
                "--model",
                args.model,
                "--devices",
                args.devices,
                "--run-dir",
                args.run_dir,
                "--prompt-tokens",
                str(args.prompt_tokens),
                "--output-tokens",
                str(args.output_tokens),
                "--profile-tokens",
                str(args.profile_tokens),
                "--warmups",
                str(args.warmups),
                "--repeats",
                str(args.repeats),
                *(["--profile"] if args.profile else []),
                *(["--diagnose"] if args.diagnose else []),
            ],
            env=benchmark_environment("full" if mode == "preflight" else mode, args.devices),
            check=True,
        )
    reports = [json.loads(Path(args.run_dir, f"{mode}.json").read_text()) for mode in ("staged", "full")]
    result = compare_results(*reports)
    result["config"] = {
        k: getattr(args, k)
        for k in ("model", "devices", "prompt_tokens", "output_tokens", "warmups", "repeats", "order")
    }
    Path(args.run_dir, "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for mode in ("staged", "full"):
        timing = result[f"{mode}_tpot_ms"]
        print(
            f"[SFA_BENCH] {mode} TPOT mean={timing['mean']:.3f} median={timing['median']:.3f} "
            f"std={timing['std']:.3f} ms/token",
            flush=True,
        )
    print(
        f"[SFA_BENCH] mean TPOT: staged={result['staged_tpot_ms']['mean']:.3f}, "
        f"full={result['full_tpot_ms']['mean']:.3f} ms/token; "
        f"reduction={result['tpot_reduction_percent']:.2f}%; speedup={result['decode_speedup']:.3f}x; "
        f"tokens_equal={result['tokens_equal']}",
        flush=True,
    )
    if not result["tokens_equal"]:
        print(
            "[SFA_BENCH] WARNING: outputs differ; MTP acceptance/routing may confound the speed comparison", flush=True
        )
    if args.profile:
        for mode in ("staged", "full"):
            # Separate non-daemon process, AFTER both engines have exited.
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from torch_npu.profiler.profiler import analyse; "
                    "import sys; analyse(sys.argv[1], max_process_number=2)",
                    str(Path(args.run_dir, mode)),
                ],
                check=True,
            )
            summary = analyse_traces(Path(args.run_dir, mode), mode, len(parse_devices(args.devices)))
            Path(args.run_dir, f"{mode}-trace-check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[SFA_PROFILE] {mode}: {summary['status']}; {summary['ranks']} rank traces", flush=True)
            if summary["status"] == "SPLIT_DETECTED":
                raise RuntimeError("Multiple target submissions or Python retrieval detected inside a root scope")
        print(f"[SFA_PROFILE] Open {args.run_dir} in MindStudio Insight; see full/staged subdirectories", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--devices", default=DEFAULT_DEVICES)
    parser.add_argument("--prompt-tokens", type=int, default=BENCHMARK_PROMPT_TOKENS)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument(
        "--profile", action="store_true", help="Also capture separate decode-only traces in both modes"
    )
    diagnostic.add_argument(
        "--diagnose", action="store_true", help="Separate decode timing in the log, without profiling"
    )
    parser.add_argument("--profile-tokens", type=int, default=32)
    parser.add_argument("--profile-dir", type=Path, default=Path("profile"))
    parser.add_argument("--order", choices=("staged,full", "full,staged"), default="staged,full")
    parser.add_argument("--child", choices=("preflight", "staged", "full"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child == "preflight":
        preflight_dependencies()
    elif args.child:
        validate_args(args)
        if not args.run_dir:
            parser.error("Internal child requires --run-dir")
        run_child(args)
    else:
        run_pair(args)


if __name__ == "__main__":
    main()
