# SPDX-License-Identifier: Apache-2.0
"""Compare old/new DCP residency over identical multi-step selection traces.

Times the planner only. Real HBM scratch is reserved equally, but this benchmark
does not execute H2D, attention, or HCCL. Miss-byte savings are estimates.
"""

import argparse
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path

import torch
from dcp_workload import CAPACITY, cold_case, dcp_reference, make_trace, step_stats, summarize, trace_metadata
from profile_timing import parse_trace
from resident_experiment import HERE, assert_result, load_library, source_digest

HISTORY = ("topk", "miss_counts", "miss_tokens", "target_slots", "state_counts")


def run(case, world, rank, policy):
    torch.ops.resident_experiment.run_dcp_(case.tensors, case.dummy_base, world, rank, policy)


def sequence(case, inputs, world, rank, policy, history=None):
    for step, values in enumerate(inputs):
        case["topk"].copy_(values)
        run(case, world, rank, policy)
        if history is not None:
            for name in HISTORY:
                history[name][step].copy_(case[name])
            history["selected_counts"][step].copy_(case.tensors[21])


def check_history(history, expected, step):
    for name in ("topk", "state_counts"):
        torch.testing.assert_close(history[name][step], expected[name], rtol=0, atol=0)
    torch.testing.assert_close(history["selected_counts"][step, :, 0], expected.tensors[21][:, 0], rtol=0, atol=0)
    for r in range(expected.requests):
        count = int(expected["miss_counts"][r, 0])
        assert int(history["miss_counts"][step, r, 0]) == count
        for name in ("miss_tokens", "target_slots"):
            torch.testing.assert_close(history[name][step, r, :count], expected[name][r, :count], rtol=0, atol=0)


def validate(snapshot, trace, world, rank, policy, device=None):
    """Check each step's valid outputs, then the complete final resident map."""
    graph = None
    if device is not None:
        actual, initial = snapshot.clone(device), snapshot.clone(device)
        inputs = trace.to(device)
        history = {
            name: torch.empty((len(trace), *actual[name].shape), dtype=actual[name].dtype, device=device)
            for name in HISTORY
        }
        history["selected_counts"] = torch.empty(
            (len(trace), *actual.tensors[21].shape), dtype=torch.int32, device=device
        )
        try:
            actual["topk"].copy_(inputs[0])
            run(actual, world, rank, policy)
            torch.npu.synchronize()
            actual.reset_from(initial)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                sequence(actual, inputs, world, rank, policy, history)
            torch.npu.synchronize()
            actual.reset_from(initial)
            graph.replay()
            torch.npu.synchronize()
            host_history = {name: value.cpu() for name, value in history.items()}
            host_actual = actual.clone("cpu")
        finally:
            if graph is not None:
                torch.npu.synchronize()
                graph.reset()
    expected = snapshot.clone("cpu")
    records = []
    for step, values in enumerate(trace):
        expected["topk"].copy_(values)
        expected = dcp_reference(expected, world, rank, free_first=bool(policy))
        if device is not None:
            try:
                check_history(host_history, expected, step)
            except (AssertionError, RuntimeError) as error:
                raise AssertionError(f"policy={policy} rank={rank} step={step}: {error}") from error
        records.append(step_stats(expected))
    if device is not None:
        assert_result(host_actual, expected, membership=True)
    return summarize(records)


def measure(snapshot, trace, world, rank, device, repeats, warmup, profile_dir=None):
    """Reset once per independent trace, never between its steps."""
    initial, inputs = snapshot.clone(device), trace.to(device)
    cases = [snapshot.clone(device), snapshot.clone(device)]
    graphs = []
    samples = [[], []]
    profiling = {}
    try:
        for policy, case in enumerate(cases):
            case["topk"].copy_(inputs[0])
            run(case, world, rank, policy)
            torch.npu.synchronize()
            case.reset_from(initial)
            graph = torch.npu.NPUGraph()
            graphs.append(graph)
            with torch.npu.graph(graph):
                sequence(case, inputs, world, rank, policy)
        torch.npu.synchronize()
        for _ in range(warmup):
            for case, graph in zip(cases, graphs):
                case.reset_from(initial)
                graph.replay()
        torch.npu.synchronize()
        events = [
            [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(repeats)]
            for _ in cases
        ]
        for repeat in range(repeats):
            for policy in (repeat % 2, 1 - repeat % 2):
                cases[policy].reset_from(initial)
                begin, end = events[policy][repeat]
                begin.record()
                graphs[policy].replay()
                end.record()
        torch.npu.synchronize()
        for policy in (0, 1):
            samples[policy] = [begin.elapsed_time(end) * 1000 / len(trace) for begin, end in events[policy]]
            if any(not math.isfinite(t) or t <= 0 for t in samples[policy]):
                raise RuntimeError("NPU event timing returned invalid intervals; no timing result accepted")
        if profile_dir is not None:
            import torch_npu

            profile_dir.mkdir(parents=True, exist_ok=True)
            for policy, variant in enumerate(("dcp_old", "dcp_new")):
                path = profile_dir / f"{variant}.json"
                if path.exists():
                    raise FileExistsError(f"Refusing to overwrite profiler trace: {path}")
                cases[policy].reset_from(initial)
                torch.npu.synchronize()
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as prof:
                    graphs[policy].replay()
                    torch.npu.synchronize()
                prof.export_chrome_trace(str(path.resolve()))
                profiling[variant] = parse_trace(
                    json.loads(path.read_text(), parse_float=Decimal), variant, "full", len(trace)
                )
        return {
            name: {
                "mean_us_per_step": statistics.fmean(values),
                "median_us_per_step": statistics.median(values),
                "samples_us_per_step": values,
                "device_profile": profiling.get(f"dcp_{name}"),
            }
            for name, values in zip(("old", "new"), samples)
        }
    finally:
        torch.npu.synchronize()
        for graph in graphs:
            graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=HERE / "build-dcp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true", help="validate the reference/trace; no native timings")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--requests", type=int, nargs="+", default=[1])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.9, help="adjacent-step Top-K intersection / 2048, per query")
    parser.add_argument("--query-overlap", type=float, default=0.5, help="target/MTP intersection / 2048 within a step")
    parser.add_argument("--dcp-size", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[0], help="simulate ranks sequentially on one NPU; no HCCL"
    )
    parser.add_argument("--history", type=int, default=65536)
    parser.add_argument("--working-set", type=int, default=32768, help="global token universe for the reuse workload")
    parser.add_argument("--workload", choices=("reuse", "fresh"), default="reuse")
    parser.add_argument("--distribution", choices=("uniform", "owner-skew", "shard-skew"), default="uniform")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--profile-dir", type=Path, help="optional actual per-kernel device-task timing")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    if args.repeats < 1 or args.warmup < 0 or len(set(args.requests)) != len(args.requests):
        parser.error("positive repeats, nonnegative warmup and unique request counts required")
    if len(set(args.ranks)) != len(args.ranks):
        parser.error("rank list must be unique")
    if args.cpu_only and args.profile_dir is not None:
        parser.error("device profiling requires native execution")
    torch.set_num_threads(1)  # CPU reference bookkeeping only; never part of the captured device interval.
    device, stamp = None, None
    if not args.cpu_only:
        stamp = load_library(args.build_dir)
        if stamp.get("profile") != "dcp" or stamp.get("torch") != torch.__version__:
            raise RuntimeError("rebuild the dcp profile with the current PyTorch environment")
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
    report = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "reserved_slots_per_request": CAPACITY,
        "source_sha256": source_digest(),
        "build": stamp,
        "timing_scope": "planner trace including input D2D copies; no H2D, attention, HCCL or model",
        "validation": "CPU reference only" if args.cpu_only else "every graph step and final state vs CPU oracle",
        "results": [],
    }
    for requests in args.requests:
        # Each block-table destination belongs to this equal fixed-size HBM
        # reservation. The planner only produces addresses; no KV is copied.
        reservation = (
            torch.empty((requests * CAPACITY + 128, 576), dtype=torch.bfloat16, device=device)
            if device is not None
            else None
        )
        for rank in args.ranks:
            trace = make_trace(
                requests,
                args.steps,
                args.overlap,
                args.query_overlap,
                args.working_set,
                args.history,
                args.seed,
                args.workload,
                args.distribution,
                args.dcp_size,
                rank,
            )
            metadata = trace_metadata(trace, args.dcp_size, rank)
            snapshot = cold_case(requests, args.history)
            stats = {
                name: validate(snapshot, trace, args.dcp_size, rank, policy, device)
                for policy, name in enumerate(("old", "new"))
            }
            profiles = args.profile_dir / f"requests-{requests}-rank-{rank}" if args.profile_dir else None
            timings = (
                measure(snapshot, trace, args.dcp_size, rank, device, args.repeats, args.warmup, profiles)
                if device is not None
                else None
            )
            reserved_bytes = reservation.numel() * reservation.element_size() if reservation is not None else 0
            row = {
                "requests": requests,
                "rank": rank,
                "trace": metadata,
                "policies": stats,
                "timings": timings,
                "reserved_kv_bytes_including_null": reserved_bytes,
            }
            if timings:
                ratio = timings["new"]["mean_us_per_step"] / timings["old"]["mean_us_per_step"]
                row["latency_change_percent"] = (ratio - 1) * 100
            report["results"].append(row)
            print(
                json.dumps(
                    {
                        "requests": requests,
                        "rank": rank,
                        "old_misses": stats["old"]["total_misses"],
                        "new_misses": stats["new"]["total_misses"],
                        "new_occupancy": stats["new"]["final_resident_per_request"],
                        "timings": timings,
                        "latency_change_percent": row.get("latency_change_percent"),
                    }
                ),
                flush=True,
            )
        del reservation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
