"""Extract actual resident device tasks, never host enqueue spans, from a trace."""
import json
import statistics
from decimal import Decimal
from pathlib import Path

import torch


def summary(samples):
    ordered = sorted(samples)
    return {"mean_us": statistics.fmean(samples), "p50_us": statistics.median(samples),
            "p95_us": ordered[min(len(ordered) - 1, int(len(ordered) * .95))], "n": len(samples)}


def kernel_names(variant):
    union = "optimized" if variant == "optimized" else "baseline"
    finalize = ("dsa_resident_sharded_finalize_worker_kernel_sharded"
                if variant in ("sharded_finalize", "combined")
                else f"dsa_resident_sorted_finalize_kernel_{union}")
    update = "compact" if variant in ("compact_remap", "combined") else union
    if variant == "vector_union":
        union = "vector"
    return {"union": f"dsa_resident_sharded_union_kernel_{union}",
            "finalize": finalize, "update": f"dsa_resident_sorted_update_kernel_{update}"}


def parse_trace(document, variant, stage, iterations):
    events = document if isinstance(document, list) else document.get("traceEvents", [])
    hardware_pids = {e.get("pid") for e in events if e.get("ph") == "M"
                     and e.get("name") == "process_name"
                     and "ascend hardware" in str(e.get("args", {}).get("name", "")).lower()}
    expected = kernel_names(variant)
    if stage in ("union_sort", "union_dedup"):
        expected = {stage: expected["union"] + "_" + stage.removeprefix("union_")}
    elif stage != "full":
        expected = {stage: expected[stage]}
    grouped = {name: [] for name in expected}
    for event in events:
        if event.get("ph") != "X" or "dur" not in event or "ts" not in event:
            continue
        args = event.get("args", {})
        category = str(event.get("cat", "")).lower()
        task = str(args.get("Task Type", args.get("task_type", ""))).upper()
        # CANN's device lane, kernel category or explicit hardware task type
        # must identify a device task; matching a function name alone is unsafe.
        if not (event.get("pid") in hardware_pids or category in ("kernel", "aicore", "ai_core")
                or task in ("AI_CORE", "AI_VECTOR_CORE", "AIV", "AIC")):
            continue
        labels = [str(event.get("name", ""))]
        labels += [str(value) for key, value in args.items() if "kernel" in key.lower() and "name" in key.lower()]
        for name, symbol in expected.items():
            if any(symbol in label for label in labels):
                grouped[name].append(event)
                break
    counts = {name: len(values) for name, values in grouped.items()}
    if any(count != iterations for count in counts.values()):
        raise RuntimeError(f"Expected {iterations} device tasks per stage, got {counts}. "
                           "Inspect the saved trace; refusing to report host spans as kernel time.")
    for values in grouped.values():
        values.sort(key=lambda e: Decimal(str(e["ts"])))
    sums, spans = [], []
    for i in range(iterations):
        tasks = [values[i] for values in grouped.values()]
        if len({(t.get("pid"), t.get("tid")) for t in tasks}) != 1:
            raise RuntimeError("Resident stages were not on one device stream")
        # Absolute microsecond timestamps can be too large for float to retain
        # submicrosecond gaps. Convert only durations/differences to float.
        starts = [Decimal(str(t["ts"])) for t in tasks]
        durations = [Decimal(str(t["dur"])) for t in tasks]
        if any(not d.is_finite() or d <= 0 for d in durations):
            raise RuntimeError("Device task has an invalid duration")
        if any(not start.is_finite() for start in starts):
            raise RuntimeError("Device task has an invalid timestamp")
        ends = [start + duration for start, duration in zip(starts, durations)]
        if any(ends[j] > starts[j + 1] + Decimal("0.01") for j in range(len(tasks) - 1)):
            raise RuntimeError("Resident stage order overlaps or is invalid")
        sums.append(float(sum(durations)))
        spans.append(float(max(ends) - min(starts)))
    return {"kernel_sum": summary(sums), "chain_span": summary(spans),
            "kernels": {name: summary([float(t["dur"]) for t in values]) for name, values in grouped.items()}}


def measure_profile(snapshot, variant, stage, iterations, warmup, trace_path: Path):
    import torch_npu

    case = snapshot.clone()
    for _ in range(warmup):
        case.reset_from(snapshot)
        case.run(variant, stage)
    torch.npu.synchronize()
    case.reset_from(snapshot)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        case.run(variant, stage)
    torch.npu.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        raise FileExistsError(f"Refusing stale/overwritten profiler output: {trace_path}")
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
    ) as prof:
        for _ in range(iterations):
            case.reset_from(snapshot)
            graph.replay()
            prof.step()
        torch.npu.synchronize()
    prof.export_chrome_trace(str(trace_path.resolve()))
    result = parse_trace(json.loads(trace_path.read_text(), parse_float=Decimal), variant, stage, iterations)
    result["trace"] = str(trace_path.resolve())
    return result
