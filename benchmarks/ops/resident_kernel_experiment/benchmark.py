"""Compare resident variants using device-task profiling or legacy event timing."""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch
from profile_timing import measure_profile
from resident_experiment import HERE, STAGES, VARIANTS, assert_result, load_library, make_case, reference


def summary(samples):
    ordered = sorted(samples)
    return {
        "mean_us": statistics.fmean(samples),
        "p50_us": statistics.median(samples),
        "p95_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "min_us": ordered[0],
        "n": len(samples),
    }


def stage_snapshot(cpu, device, stage):
    case = cpu.clone(device)
    if stage in ("finalize", "update"):
        case.run(False, "union")
    if stage == "update":
        case.run(False, "finalize")
    torch.npu.synchronize()
    return case


def measure_pair(snapshot, stage, iterations, warmup, mode, experiment="optimized"):
    cases = [snapshot.clone(), snapshot.clone()]
    for _ in range(warmup):
        for optimized, case in enumerate(cases):
            case.reset_from(snapshot)
            case.run(experiment if optimized else "baseline", stage)
    torch.npu.synchronize()
    graphs = []
    if mode == "graph":
        for optimized, case in enumerate(cases):
            case.reset_from(snapshot)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                case.run(experiment if optimized else "baseline", stage)
            graphs.append(graph)
        torch.npu.synchronize()
    events = [
        [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(iterations)]
        for _ in cases
    ]
    for iteration in range(iterations):
        # Alternating order reduces systematic clock/thermal/order bias.
        for variant in (0, 1) if iteration % 2 == 0 else (1, 0):
            case = cases[variant]
            # Restore ALL mutable state and top-k outside the timed interval.
            # Otherwise repeated measurements converge to all-hit residency.
            case.reset_from(snapshot)
            begin, end = events[variant][iteration]
            begin.record()
            if mode == "graph":
                graphs[variant].replay()
            else:
                case.run(experiment if variant else "baseline", stage)
            end.record()
    torch.npu.synchronize()
    return [summary([begin.elapsed_time(end) * 1000 for begin, end in row]) for row in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=HERE / "build")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument(
        "--mtp", type=int, choices=(1, 2), default=2, help="query rows per request; 2 means one speculative token"
    )
    parser.add_argument("--shards-per-row", type=int, choices=(1, 2, 4), nargs="+", default=[4])
    parser.add_argument("--hit-rates", type=float, nargs="+", default=[0.0, 0.9, 1.0])
    parser.add_argument("--overlap", type=int, choices=(0, 1024, 2048), default=1024)
    parser.add_argument("--variants", choices=tuple(VARIANTS), nargs="+",
                        default=["baseline", "compact_remap", "sharded_finalize", "combined"])
    parser.add_argument(
        "--scenario", choices=("normal", "cold", "one_shard_miss", "subset", "zero_boundary"), default="normal"
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mode", choices=("profile", "graph", "eager"), default="profile")
    parser.add_argument("--stage", choices=("all", *STAGES), default="full")
    parser.add_argument("--trace-dir", type=Path, default=HERE / "profiles")
    parser.add_argument("--json", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1 or min(args.requests) < 1:
        parser.error("iterations, warmup and request counts must be positive")
    if any(not 0 <= rate <= 1 for rate in args.hit_rates):
        parser.error("hit rates must be in [0, 1]")
    variants = list(dict.fromkeys(["baseline", *args.variants]))
    run_dir = args.trace_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    build = load_library(args.build_dir)
    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    import torch_npu

    report = {
        "build": build,
        "device": torch.npu.get_device_name(args.device),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "mode": args.mode,
        "timing": ("Profiler device-task durations from graph replay; host gaps excluded"
                   if args.mode == "profile" else "Legacy event intervals; may include host submission gaps"),
        "overlap": args.overlap,
        "results": [],
    }
    stages = tuple(STAGES) if args.stage == "all" else (args.stage,)
    for requests in args.requests:
        for shards in args.shards_per_row:
            for rate in args.hit_rates:
                cpu = make_case(requests, args.mtp, shards, rate, args.scenario, overlap=args.overlap)
                expected, stats = reference(cpu)
                actual_hit = 1 - stats["misses"] / stats["selected"] if stats["selected"] else None
                # Correctness is a prerequisite to timing each input case.
                for optimized in variants:
                    actual = cpu.clone(device)
                    actual.run(optimized)
                    torch.npu.synchronize()
                    assert_result(actual, expected)
                for stage in stages:
                    snapshot = stage_snapshot(cpu, device, stage)
                    measurements = {}
                    if args.mode == "profile":
                        for variant in variants:
                            trace = run_dir / f"case_{len(report['results'])}_{variant}_{stage}.json"
                            measurements[variant] = measure_profile(
                                snapshot, variant, stage, args.iterations, args.warmup, trace)
                        old = measurements["baseline"]["kernel_sum"]
                    else:
                        print("WARNING: event timing includes submission effects; use --mode profile for kernel cost.")
                        for variant in variants[1:] or ["baseline"]:
                            old, new = measure_pair(snapshot, stage, args.iterations, args.warmup, args.mode, variant)
                            measurements[variant] = {"event_interval": new, "paired_baseline": old}
                        measurements["baseline"] = {"event_interval": old}
                    record = {
                        "requests": requests,
                        "mtp": args.mtp,
                        "shards_per_row": shards,
                        "input_hit_fraction": rate,
                        "actual_hit_fraction": actual_hit,
                        "scenario": args.scenario,
                        "stage": stage,
                        "counts": stats,
                        "measurements": measurements,
                    }
                    report["results"].append(record)
                    hit_label = f"{actual_hit:.3f}" if actual_hit is not None else "n/a"
                    for variant in variants:
                        values = measurements[variant]
                        elapsed = values["kernel_sum" if args.mode == "profile" else "event_interval"]
                        print(f"R={requests} M={args.mtp} S={args.mtp * shards} hit={hit_label} "
                              f"{stage} {variant}: {elapsed['mean_us']:.2f}us "
                              f"speedup={old['mean_us'] / elapsed['mean_us']:.3f}x "
                              f"miss={stats['misses']}", flush=True)
                        if args.mode == "profile":
                            print("  kernel means:", {k: round(v['mean_us'], 3) for k, v in values['kernels'].items()},
                                  "chain_span_us=", round(values['chain_span']['mean_us'], 3), flush=True)
                    args.json.parent.mkdir(parents=True, exist_ok=True)
                    args.json.write_text(json.dumps(report, indent=2))
                    del snapshot
    print(f"Results: {args.json.resolve()}")


if __name__ == "__main__":
    main()
