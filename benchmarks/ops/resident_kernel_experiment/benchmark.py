"""Compare resident variants using device-task profiling or legacy event timing."""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch
from profile_timing import measure_profile
from resident_experiment import (
    HERE,
    MEMBERSHIP_VARIANTS,
    STAGES,
    VARIANTS,
    assert_result,
    load_library,
    make_case,
    membership_reference,
    reference,
)


def summary(samples):
    ordered = sorted(samples)
    return {
        "mean_us": statistics.fmean(samples),
        "p50_us": statistics.median(samples),
        "p95_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "min_us": ordered[0],
        "n": len(samples),
    }


def stage_snapshot(cpu, device, stage, variant):
    case = cpu.clone(device)
    if stage in ("finalize", "update", "state_update", "remap"):
        case.run(variant, "union")
    if stage in ("update", "state_update", "remap"):
        case.run(variant, "finalize")
    torch.npu.synchronize()
    return case


def measure_events(snapshots, stage, iterations, warmup, mode):
    variants = tuple(snapshots)
    cases = {variant: snapshot.clone() for variant, snapshot in snapshots.items()}
    for _ in range(warmup):
        for variant, case in cases.items():
            case.reset_from(snapshots[variant])
            case.run(variant, stage)
    torch.npu.synchronize()
    graphs = {}
    if mode == "graph":
        for variant, case in cases.items():
            case.reset_from(snapshots[variant])
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                case.run(variant, stage)
            graphs[variant] = graph
        torch.npu.synchronize()
    events = {
        variant: [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                  for _ in range(iterations)]
        for variant in variants
    }
    for iteration in range(iterations):
        first = iteration % len(variants)
        # Rotate order to reduce systematic clock/thermal/order bias.
        for variant in variants[first:] + variants[:first]:
            case = cases[variant]
            # Restore ALL mutable state and top-k outside the timed interval.
            # Otherwise repeated measurements converge to all-hit residency.
            case.reset_from(snapshots[variant])
            begin, end = events[variant][iteration]
            begin.record()
            if mode == "graph":
                graphs[variant].replay()
            else:
                case.run(variant, stage)
            end.record()
    torch.npu.synchronize()
    return {
        variant: {"event_interval": summary([begin.elapsed_time(end) * 1000 for begin, end in row])}
        for variant, row in events.items()
    }


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
    parser.add_argument("--live-tokens", type=int, default=0, help="live entries per query row")
    parser.add_argument("--variants", choices=tuple(VARIANTS), nargs="+")
    parser.add_argument("--compare-to", choices=tuple(VARIANTS),
                        help="reference variant (defaults to exact_combined in the membership build)")
    parser.add_argument(
        "--scenario", choices=("normal", "cold", "one_shard_miss", "subset", "zero_boundary", "skewed"),
        default="normal",
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
    if not 0 <= args.live_tokens <= 2048 or (
        args.live_tokens and args.scenario in ("subset", "zero_boundary")
    ):
        parser.error("--live-tokens must be in [0, 2048] and requires a scenario without a boundary override")
    build = load_library(args.build_dir)
    membership_build = build.get("profile") == "membership"
    comparison = args.compare_to or ("exact_combined" if membership_build else "baseline")
    defaults = MEMBERSHIP_VARIANTS if membership_build else ("baseline", "vector_union")
    variants = list(dict.fromkeys([comparison, *(args.variants or defaults)]))
    if membership_build and (
        any(v not in MEMBERSHIP_VARIANTS for v in variants) or STAGES.get(args.stage, 0) > 3
    ):
        parser.error(
            "membership build supports only exact_combined/query_membership and full/union/finalize/update/all"
        )
    if args.stage in ("union_sort", "union_dedup") and any(v not in ("baseline", "vector_union") for v in variants):
        parser.error("union prefix diagnostics support only baseline/vector_union")
    if args.stage in ("state_update", "remap") and any(
        v not in ("baseline", "vector_intersection", "vector_state_update", "exact_combined")
        for v in variants
    ):
        parser.error("state/remap probes support baseline and exact variants only")
    run_dir = args.trace_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
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
        "live_tokens_per_row": args.live_tokens,
        "comparison": comparison,
        "results": [],
    }
    stages = ("full", "union", "finalize", "update") if args.stage == "all" else (args.stage,)
    for requests in args.requests:
        for shards in args.shards_per_row:
            for rate in args.hit_rates:
                cpu = make_case(requests, args.mtp, shards, rate, args.scenario,
                                overlap=args.overlap, live_tokens=args.live_tokens)
                expected, stats = reference(cpu)
                actual_hit = 1 - stats["misses"] / stats["selected"] if stats["selected"] else None
                # Correctness is a prerequisite to timing each input case.
                for variant in variants:
                    membership = variant == "query_membership"
                    expected_variant = membership_reference(cpu, expected) if membership else expected
                    actual = cpu.clone(device)
                    actual.run(variant)
                    torch.npu.synchronize()
                    assert_result(actual, expected_variant, membership=membership)
                    del actual
                for stage in stages:
                    measurements = {}
                    if args.mode == "profile":
                        index = len(report["results"])
                        first = index % len(variants)
                        for variant in variants[first:] + variants[:first]:
                            snapshot = stage_snapshot(cpu, device, stage, variant)
                            trace = run_dir / f"case_{index}_{variant}_{stage}.json"
                            measurements[variant] = measure_profile(
                                snapshot, variant, stage, args.iterations, args.warmup, trace)
                            del snapshot
                    else:
                        print("WARNING: event timing includes submission effects; use --mode profile for kernel cost.")
                        snapshots = {v: stage_snapshot(cpu, device, stage, v) for v in variants}
                        measurements = measure_events(snapshots, stage, args.iterations, args.warmup, args.mode)
                        del snapshots
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
                    metric = "kernel_sum" if args.mode == "profile" else "event_interval"
                    base = measurements[comparison][metric]
                    for variant in variants:
                        values = measurements[variant]
                        elapsed = values[metric]
                        speedup = base["p50_us"] / elapsed["p50_us"] if elapsed["p50_us"] > 0 else None
                        values["p50_speedup_vs_comparison"] = speedup
                        ratio = f"{speedup:.3f}x" if speedup is not None else "n/a"
                        print(f"R={requests} M={args.mtp} S={args.mtp * shards} hit={hit_label} "
                              f"{stage} {variant}: p50={elapsed['p50_us']:.2f}us p95={elapsed['p95_us']:.2f}us "
                              f"vs {comparison}={ratio} "
                              f"miss={stats['misses']}", flush=True)
                        if args.mode == "profile":
                            print("  kernel means:", {k: round(v['mean_us'], 3) for k, v in values['kernels'].items()},
                                  "chain_span_us=", round(values['chain_span']['mean_us'], 3), flush=True)
                    args.json.parent.mkdir(parents=True, exist_ok=True)
                    args.json.write_text(json.dumps(report, indent=2))
    print(f"Results: {args.json.resolve()}")


if __name__ == "__main__":
    main()
