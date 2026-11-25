#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure GLM decode layer and graph-prefetch windows in a MindStudio trace.

Times in output are us.
Each sample is a complete replay of the selected target graph on one device.
"""

import argparse
import bisect
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict, namedtuple
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from summarize_prefill_profile import complete_events

Task = namedtuple("Task", "start stop stream model task name kind")
CONTROL = {
    "CAPTURE_WAIT",
    "CAPTURE_RECORD",
    "NOTIFY_WAIT",
    "NOTIFY_RECORD",
    "MEM_WRITE_VALUE",
    "MEM_WAIT_VALUE",
    "EVENT_WAIT",
    "EVENT_RECORD",
    "PROFILING_ENABLE",
    "PROFILING_DISABLE",
    "MODEL_EXECUTE",
}
NONCORE = {"KERNEL_AICPU", "AI_CPU", "SDMA_SQE", "PCIE_DMA_SQE", "RDMA_DB_SEND", "DSA_SQE"}


def ns(value):
    # Sub-microsecond gaps matter here. Epoch-sized float timestamps lose bits.
    text = str(value)
    whole, _, fraction = text.partition(".")
    if whole.isdigit() and (not fraction or fraction.isdigit()):
        return int(whole) * 1000 + int((fraction + "000")[:3])
    return int(Decimal(text) * 1000)


@lru_cache(maxsize=8192)
def resource(kind, name):
    if kind in {"AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"} or name.lower().startswith("aivkernel"):
        return "vector_or_mixed"
    if kind == "AI_CORE" and any(s in name.lower() for s in ("matmul", "batch_matmul", "aclnnmm")):
        return "cube_matmul"
    if kind in NONCORE or kind in CONTROL:
        return "noncore"
    return "unknown"


def read_tasks(path):
    devices, labels, unknown = defaultdict(list), {}, defaultdict(Counter)
    for ev in complete_events(path):
        a = ev.get("args") or {}
        pid = str(ev.get("pid"))
        if ev.get("ph") == "M" and ev.get("name") == "process_name":
            labels[pid] = a.get("name")
        if ev.get("ph") != "X" or ("Physic Stream Id" not in a and labels.get(pid) != "Ascend Hardware"):
            continue
        kind, name = a.get("Task Type", ""), ev.get("name", "")
        if kind in CONTROL or kind in NONCORE:
            continue
        width = ns(ev.get("dur", 0))
        if width <= 0:
            continue
        t = ns(ev["ts"])
        row = Task(
            t,
            t + width,
            int(a.get("Physic Stream Id", ev["tid"])),
            int(a.get("Model Id", -1)),
            int(a.get("Task Id", -1)),
            sys.intern(name),
            sys.intern(kind),
        )
        devices[pid].append(row)
        if resource(kind, name) == "unknown":
            unknown[pid][kind + "/" + name] += 1
    return devices, unknown


def is_sfa(row):
    return "sparseflashattention" in row.name.lower().replace("_", "")


def is_copy(row):
    return "sfacopy" in row.name.lower().replace("_", "")


def graph_choice(devices, wanted_pid, wanted_model, wanted_stream):
    choices = Counter()
    for pid, rows in devices.items():
        for r in rows:
            if (
                is_sfa(r)
                and r.model not in (-1, 4294967295)
                and (wanted_pid is None or pid == wanted_pid)
                and (wanted_model is None or r.model == wanted_model)
                and (wanted_stream is None or r.stream == wanted_stream)
            ):
                choices[pid, r.stream, r.model] += 1
    if not choices:
        raise ValueError("No sparse-attention target graph matches the selector")
    return choices.most_common(1)[0][0]


def layout(main, expected):
    sfas = [r for r in main if is_sfa(r)]
    if len(sfas) != expected:
        return None
    starts = []
    times = [r.start for r in main]
    lo = 0
    for i, sfa in enumerate(sfas):
        hi = bisect.bisect_left(times, sfa.start)
        before = main[lo:hi]
        kv = [k for k, r in enumerate(before) if r.name.lower().startswith("kvrmsnormropecache")]
        if len(kv) != 1:
            raise ValueError(f"layer {i}: need exactly one KV/RoPE anchor before SFA, got {len(kv)}")
        if i == 0:
            norms = [r for r in before[: kv[0]] if r.name.lower().startswith("rmsnorm")]
            start = norms[0] if norms else None
        else:
            norms = [r for r in before[: kv[0]] if r.name.lower().startswith("addrmsnormbias")]
            start = norms[-1] if norms else None
        if start is None:
            raise ValueError(f"layer {i}: input norm anchor absent for this graph layout")
        starts.append(start)
        lo = hi + 1
    tail = [r for r in main[lo:] if r.name.lower().startswith("addrmsnormbias")]
    if not tail:
        raise ValueError("final model norm missing after final SFA")
    # Dense/MoE layers end at the next input norm; the final norm belongs to the tail.
    last = tail[-1]
    ff = [r for r in main[lo:] if "moedistributecombine" in r.name.lower() or "swiglu" in r.name.lower()]
    if not ff or ff[-1].stop > last.start:
        raise ValueError("last layer truncated before the final norm")
    return starts, sfas, last


class Timeline:
    def __init__(self, pairs):
        self.spans = []
        for start, stop in sorted(pairs):
            if self.spans and start <= self.spans[-1][1]:
                a, b = self.spans[-1]
                self.spans[-1] = a, max(b, stop)
            else:
                self.spans.append((start, stop))
        self.ends = [b for _, b in self.spans]

    def gaps(self, start, stop):
        cursor, gaps = start, []
        i = bisect.bisect_right(self.ends, start)
        while i < len(self.spans) and self.spans[i][0] < stop:
            a, b = self.spans[i]
            if a > cursor:
                gaps.append(a - cursor)
            cursor = max(cursor, b)
            i += 1
        if cursor < stop:
            gaps.append(stop - cursor)
        return gaps

    def overlap(self, parts):
        return sum(r.stop - r.start - sum(self.gaps(r.start, r.stop)) for r in parts)


def read_controls(path, pid, stream, stages, wanted):
    starts = [s[0] for s in stages]
    result = defaultdict(dict)
    for ev in complete_events(path):
        if ev.get("ph") != "X" or str(ev.get("pid")) != pid:
            continue
        a = ev.get("args") or {}
        kind = a.get("Task Type")
        if (
            kind not in {"CAPTURE_WAIT", "CAPTURE_RECORD", "NOTIFY_WAIT", "NOTIFY_RECORD"}
            or int(a.get("Physic Stream Id", ev["tid"])) != stream
        ):
            continue
        task = int(a.get("Task Id", -1))
        if (kind, task) not in wanted:
            continue
        start = ns(ev["ts"])
        step = bisect.bisect_right(starts, start) - 1
        if step < 0 or start >= stages[step][1]:
            continue
        stop = start + ns(ev.get("dur", 0))
        key = kind, task
        previous = result[step].get(key)
        result[step][key] = (min(start, previous[0]), max(stop, previous[1])) if previous else (start, stop)
    return result


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    return sorted_values[left] + (sorted_values[right] - sorted_values[left]) * (position - left)


def sample_summary(samples, name):
    values = sorted(s[name] / 1000 for s in samples if name in s)
    if not values:
        return {}
    return {
        "n": len(values),
        "mean_us": round(statistics.mean(values), 2),
        "p10_us": round(percentile(values, 0.1), 2),
        "p50_us": round(percentile(values, 0.5), 2),
        "p95_us": round(percentile(values, 0.95), 2),
    }


def analyze(path, wanted_pid=None, wanted_model=None, wanted_stream=None, expected=78):
    print("[LAYER_GAPS] scanning device kernels", file=sys.stderr, flush=True)
    devices, unknown = read_tasks(path)
    pid, stream, model = graph_choice(devices, wanted_pid, wanted_model, wanted_stream)
    tasks = sorted(devices[pid])
    main = [r for r in tasks if r.stream == stream]
    times = [r.start for r in main]
    # Graphs may start with stream-control tasks. Anchor each model on its
    # first compute task, also for adjacent graphs.
    first_task = {}
    for r in main:
        if r.task >= 0 and r.model not in (-1, 4294967295):
            first_task[r.model] = min(first_task.get(r.model, r.task), r.task)
    anchors = [(r.start, r.model) for r in main if first_task.get(r.model) == r.task]
    if not anchors:
        raise ValueError("target replay has no device task anchor")
    stages, wanted = [], set()
    incomplete = 0
    moe = [r for r in main if "moedistributedispatch" in r.name.lower()]
    moe_times = [r.start for r in moe]
    combines = [r for r in main if "moedistributecombine" in r.name.lower()]
    combine_times = [r.start for r in combines]
    wanted.update((kind, r.task - 1) for r in moe if r.model == model for kind in ("NOTIFY_WAIT", "NOTIFY_RECORD"))
    for i, (start, m) in enumerate(anchors):
        if m != model:
            continue
        end = anchors[i + 1][0] if i + 1 < len(anchors) else max(r.stop for r in main) + 1
        step_tasks = main[bisect.bisect_left(times, start) : bisect.bisect_left(times, end)]
        valid = layout([r for r in step_tasks if r.model in {model, 4294967295, -1}], expected)
        if valid is None:
            incomplete += 1
            continue
        norms, sfas, final = valid
        wanted.update(("CAPTURE_WAIT", n.task + 1) for n in norms[1:])
        wanted.update(("CAPTURE_RECORD", n.task + 1) for n in norms[1:])
        wanted.update(("CAPTURE_WAIT", n.task + 2) for n in norms[1:])
        wanted.update(("CAPTURE_RECORD", r.task + 1) for r in sfas[:-1])
        stages.append((start, end, norms, sfas, final))
    if not stages:
        raise ValueError(f"no full replay with {expected} layers on stream {stream}, model {model}")
    print(
        f"[LAYER_GAPS] {len(stages)} complete replay(s); resolving main-stream fork/join events",
        file=sys.stderr,
        flush=True,
    )
    controls = read_controls(path, pid, stream, stages, wanted)
    copies = [r for r in tasks if r.stream != stream and is_copy(r)]
    copy_times = [r.start for r in copies]
    by_edge, background = {}, set()
    for step, (_, _, norms, sfas, final) in enumerate(stages):
        events = controls[step]
        for i, sfa in enumerate(sfas[:-1]):
            a = bisect.bisect_left(copy_times, sfa.stop)
            b = bisect.bisect_left(copy_times, sfas[i + 1].start)
            parts = copies[a:b]
            next_norm = norms[i + 1]
            late_release = events.get(("CAPTURE_RECORD", next_norm.task + 1))
            late_join = events.get(("CAPTURE_WAIT", next_norm.task + 2))
            at_consumer = bool(parts) and late_release is not None and late_join is not None
            if at_consumer:
                release, join = late_release, late_join
            else:
                release = events.get(("CAPTURE_RECORD", sfa.task + 1))
                join = events.get(("CAPTURE_WAIT", next_norm.task + 1))
            if not release or not join:
                # Some layers use main-stream retrieval. A side copy without
                # a matched fork/join means this graph needs a different mapping.
                if parts:
                    raise ValueError(f"unmapped asynchronous copy in replay {step} layer {i}")
                continue
            ready, deadline = release[1], join[0]
            predecessor = next_norm.stop if at_consumer else sfa.stop
            if not predecessor <= ready <= deadline or deadline < next_norm.stop:
                raise ValueError(f"inconsistent fork/join timestamps in replay {step} layer {i}")
            # MindStudio streams can differ by a few timestamp ticks at the fork/join.
            # Pair by the surrounding SFA interval, then check the control timestamps.
            if (
                not parts
                or len({r.stream for r in parts}) != 1
                or parts[0].start < ready - 1000
                or max(r.stop for r in parts) > join[1] + 1000
            ):
                raise ValueError(f"side copy disagrees with the fork/join in replay {step} layer {i}")
            by_edge[step, i] = ready, deadline, parts, at_consumer
            background.update(parts)
    rtypes = {r: resource(r.kind, r.name) for r in tasks}
    all_busy = Timeline((r.start, r.stop) for r in tasks)
    foreground = [r for r in tasks if r not in background]
    core = Timeline((r.start, r.stop) for r in foreground)
    vector = Timeline((r.start, r.stop) for r in foreground if rtypes[r] != "cube_matmul")
    root_core = Timeline((r.start, r.stop) for r in main)
    side_compute = [r for r in foreground if r.stream != stream]
    comm_tasks = {r for r in side_compute if r.kind == "COMMUNICATION" and "aivkernel" in r.name.lower()}
    comm_aiv = Timeline((r.start, r.stop) for r in comm_tasks)
    other_compute = Timeline((r.start, r.stop) for r in side_compute if r not in comm_tasks)
    dispatch = Timeline((r.start, r.stop) for r in moe)
    samples = []
    for step, (_, _, norms, sfas, final) in enumerate(stages):
        for i, (left, right) in enumerate(zip(norms, norms[1:] + [final])):
            a, b = left.start, right.start
            s = {
                "replay": step,
                "layer": i,
                "layer_duration": b - a,
                "layer_no_ai_including_copy": sum(all_busy.gaps(a, b)),
                "layer_no_foreground_ai": sum(core.gaps(a, b)),
                "layer_no_foreground_aiv": sum(vector.gaps(a, b)),
            }
            edge = by_edge.get((step, i))
            if edge:
                ready, deadline, parts, at_consumer = edge
                lengths = vector.gaps(ready, deadline)
                copy = sum(r.stop - r.start for r in parts)
                s.update(
                    edge_window=deadline - ready,
                    edge_no_foreground_ai=sum(core.gaps(ready, deadline)),
                    edge_no_foreground_aiv=sum(lengths),
                    longest_aiv_gap=max(lengths, default=0),
                    copy_device=copy,
                    margin_to_join=deadline - max(r.stop for r in parts),
                    max_piece_device=max(r.stop - r.start for r in parts),
                    copy_span=max(r.stop for r in parts) - parts[0].start,
                    join_wait_for_copy=max(0, max(r.stop for r in parts) - deadline),
                    uncovered_if_only_aiv_gaps=max(0, copy - sum(lengths)),
                    dispatch_overlap=dispatch.overlap(parts),
                    copy_overlap_main_compute=root_core.overlap(parts),
                    copy_overlap_comm_aiv=comm_aiv.overlap(parts),
                    copy_overlap_other_device_compute=other_compute.overlap(parts),
                    copy_before_consumer_norm_end=sum(
                        max(0, min(r.stop, norms[i + 1].stop) - r.start) for r in parts
                    ),
                    released_at_consumer=at_consumer,
                    pieces=len(parts),
                    side_stream=parts[0].stream,
                    fit=copy <= sum(lengths),
                    fit_contiguous=copy <= max(lengths, default=0),
                )
                # The protected layer still contains MoE when the actual fork
                # has moved past it. Report zero overlap for those dispatches.
                ds = moe[bisect.bisect_left(moe_times, sfas[i].stop) : bisect.bisect_left(moe_times, deadline)]
                if len(ds) == 1:
                    d = ds[0]
                    s["dispatch_device"] = d.stop - d.start
                    s["copy_left_at_dispatch_start"] = max(
                        (r.stop - d.start for r in parts if r.start <= d.start < r.stop), default=0
                    )
                    s["pieces_started_during_dispatch"] = sum(d.start <= r.start < d.stop for r in parts)
                    first = bisect.bisect_left(combine_times, d.stop)
                    last = bisect.bisect_left(combine_times, deadline)
                    if last == first + 1 and combines[first].stop <= deadline:
                        end = combines[first].stop
                        s["dispatch_to_combine_copy_overlap"] = sum(
                            max(0, min(r.stop, end) - max(r.start, d.start)) for r in parts
                        )
                        s["pieces_started_dispatch_to_combine"] = sum(d.start <= r.start < end for r in parts)
                        s["combine_to_join_arrival"] = deadline - end
                        if not at_consumer:
                            s["fork_to_dispatch"] = d.start - ready
                            outside = s["fork_to_dispatch"] + s["combine_to_join_arrival"]
                            s["copy_excess_outside_dispatch_to_combine"] = max(0, copy - outside)
                    preceding_notify = controls[step].get(("NOTIFY_WAIT", d.task - 1)) or controls[step].get(
                        ("NOTIFY_RECORD", d.task - 1)
                    )
                    if preceding_notify and ready <= preceding_notify[1] <= d.start:
                        # Observable predecessor on the main queue. Other hardware or
                        # cross-rank prerequisites may still delay actual eligibility.
                        s["notify_end_to_dispatch_start"] = d.start - preceding_notify[1]
                        s["pieces_started_after_notify_before_dispatch"] = sum(
                            preceding_notify[1] <= r.start < d.start for r in parts
                        )
            samples.append(s)
    layer_fields = ["layer_duration", "layer_no_ai_including_copy", "layer_no_foreground_ai", "layer_no_foreground_aiv"]
    edge_fields = [
        "edge_window",
        "edge_no_foreground_ai",
        "edge_no_foreground_aiv",
        "longest_aiv_gap",
        "copy_device",
        "copy_overlap_main_compute",
        "copy_overlap_comm_aiv",
        "copy_overlap_other_device_compute",
        "copy_before_consumer_norm_end",
        "margin_to_join",
        "uncovered_if_only_aiv_gaps",
        "dispatch_overlap",
        "dispatch_to_combine_copy_overlap",
        "fork_to_dispatch",
        "combine_to_join_arrival",
        "copy_excess_outside_dispatch_to_combine",
        "dispatch_device",
        "copy_left_at_dispatch_start",
        "notify_end_to_dispatch_start",
        "max_piece_device",
        "copy_span",
        "join_wait_for_copy",
    ]
    layers = []
    for i in range(expected):
        group = [s for s in samples if s["layer"] == i]
        row = {
            "layer": i,
            "replays": len(group),
            "async_to_next": sum("fit" in s for s in group),
            "released_at_consumer": sum(s.get("released_at_consumer", False) for s in group),
            "fits_total_gap": sum(s.get("fit", False) for s in group),
        }
        for name in layer_fields + edge_fields:
            st = sample_summary(group, name)
            row[name + "_mean_us"], row[name + "_p50_us"] = st.get("mean_us"), st.get("p50_us")
        layers.append(row)
    edge_samples = [s for s in samples if "fit" in s]
    summary = {
        "file": str(path),
        "bytes": path.stat().st_size,
        "device_pid": pid,
        "main_stream": stream,
        "model_id": model,
        "complete_replays": len(stages),
        "skipped_incomplete_replays": incomplete,
        "layers_per_replay": expected,
        "prefetch_layers_zero_based": [r["layer"] for r in layers if r["async_to_next"]],
        "side_streams": sorted({s["side_stream"] for s in edge_samples}),
        "edges": len(edge_samples),
        "consumer_release_edges": sum(s["released_at_consumer"] for s in edge_samples),
        "edges_overlapping_main_compute": sum(s["copy_overlap_main_compute"] > 0 for s in edge_samples),
        "edges_overlapping_comm_aiv": sum(s["copy_overlap_comm_aiv"] > 0 for s in edge_samples),
        "fit_total_aiv_gap": sum(s["fit"] for s in edge_samples),
        "fit_largest_aiv_gap": sum(s["fit_contiguous"] for s in edge_samples),
        "copy_pieces_per_edge": dict(Counter(s["pieces"] for s in edge_samples)),
        "copy_fits_outside_dispatch_to_combine": sum(
            s.get("copy_excess_outside_dispatch_to_combine") == 0 for s in edge_samples
        ),
        "individual_copy_piece": sample_summary([{"duration": r.stop - r.start} for r in background], "duration"),
        "pieces_started_during_dispatch_histogram": dict(
            Counter(s["pieces_started_during_dispatch"] for s in edge_samples if "pieces_started_during_dispatch" in s)
        ),
        "pieces_started_dispatch_to_combine_histogram": dict(
            Counter(
                s["pieces_started_dispatch_to_combine"]
                for s in edge_samples
                if "pieces_started_dispatch_to_combine" in s
            )
        ),
        "pieces_started_after_notify_before_dispatch_histogram": dict(
            Counter(
                s["pieces_started_after_notify_before_dispatch"]
                for s in edge_samples
                if "pieces_started_after_notify_before_dispatch" in s
            )
        ),
        "finished_by_join_arrival": sum(s["margin_to_join"] >= 0 for s in edge_samples),
        "conservatively_busy_unknown": dict(unknown[pid]),
        "all_layers": {name: sample_summary(samples, name) for name in layer_fields},
        "prefetch_edges": {name: sample_summary(edge_samples, name) for name in edge_fields},
    }
    return summary, layers, samples


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace", type=Path)
    cli.add_argument("--pid")
    cli.add_argument("--model-id", type=int)
    cli.add_argument("--stream", type=int)
    cli.add_argument("--layers", type=int, default=78, help="Expected number of target transformer layers")
    cli.add_argument("--csv", type=Path, default=Path("decode-layer-gaps.csv"))
    cli.add_argument("--summary", type=Path, help="Also save the short JSON printed to stdout")
    args = cli.parse_args()
    before = args.trace.stat()
    summary, layers, _ = analyze(args.trace, args.pid, args.model_id, args.stream, args.layers)
    after = args.trace.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("trace changed during analysis")
    with args.csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(layers[0]))
        writer.writeheader()
        writer.writerows(layers)
    summary["per_layer_csv"] = str(args.csv)
    output = json.dumps(summary, indent=2)
    if args.summary:
        args.summary.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
