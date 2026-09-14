"""Paired device-event timing of the baseline and experimental resident kernels."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from resident_experiment import HERE, STAGES, assert_result, load_library, make_case, reference


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


def measure_pair(snapshot, stage, iterations, warmup, mode):
    cases = [snapshot.clone(), snapshot.clone()]
    for _ in range(warmup):
        for optimized, case in enumerate(cases):
            case.reset_from(snapshot)
            case.run(bool(optimized), stage)
    torch.npu.synchronize()
    graphs = []
    if mode == "graph":
        for optimized, case in enumerate(cases):
            case.reset_from(snapshot)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                case.run(bool(optimized), stage)
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
                case.run(bool(variant), stage)
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
    parser.add_argument(
        "--scenario", choices=("normal", "cold", "one_shard_miss", "subset", "zero_boundary"), default="normal"
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--json", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1 or min(args.requests) < 1:
        parser.error("iterations, warmup and request counts must be positive")
    if any(not 0 <= rate <= 1 for rate in args.hit_rates):
        parser.error("hit rates must be in [0, 1]")
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
        "timing": "NPU events; reset/predecessor work excluded; paired alternating order",
        "results": [],
    }
    stages = tuple(STAGES) if args.stage == "all" else (args.stage,)
    for requests in args.requests:
        for shards in args.shards_per_row:
            for rate in args.hit_rates:
                cpu = make_case(requests, args.mtp, shards, rate, args.scenario)
                expected, stats = reference(cpu)
                actual_hit = 1 - stats["misses"] / stats["selected"] if stats["selected"] else None
                # Correctness is a prerequisite to timing each input case.
                for optimized in (False, True):
                    actual = cpu.clone(device)
                    actual.run(optimized)
                    torch.npu.synchronize()
                    assert_result(actual, expected)
                for stage in stages:
                    snapshot = stage_snapshot(cpu, device, stage)
                    old, new = measure_pair(snapshot, stage, args.iterations, args.warmup, args.mode)
                    record = {
                        "requests": requests,
                        "mtp": args.mtp,
                        "shards_per_row": shards,
                        "input_hit_fraction": rate,
                        "actual_hit_fraction": actual_hit,
                        "scenario": args.scenario,
                        "stage": stage,
                        "counts": stats,
                        "baseline": old,
                        "optimized": new,
                        "speedup": old["mean_us"] / new["mean_us"],
                    }
                    report["results"].append(record)
                    hit_label = f"{actual_hit:.3f}" if actual_hit is not None else "n/a"
                    print(
                        f"R={requests:2d} M={args.mtp} S={args.mtp * shards} hit={hit_label} "
                        f"{stage:8s} old={old['mean_us']:.2f}us new={new['mean_us']:.2f}us "
                        f"speedup={record['speedup']:.3f}x "
                        f"miss={stats['misses']} unchanged_shards={stats['unchanged_shards']}",
                        flush=True,
                    )
                    args.json.parent.mkdir(parents=True, exist_ok=True)
                    args.json.write_text(json.dumps(report, indent=2))
                    del snapshot
    print(f"Results: {args.json.resolve()}")


if __name__ == "__main__":
    main()
