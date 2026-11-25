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
from itertools import permutations
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
    if kind == "AI_CORE" and any(s in name.lower().replace("_", "") for s in ("matmul", "aclnnmm")):
        return "cube_matmul"
    if kind in NONCORE or kind in CONTROL:
        return "noncore"
    return "unknown"


def read_tasks(path, *, include_controls=False):
    devices, labels, unknown = defaultdict(list), {}, defaultdict(Counter)
    for ev in complete_events(path):
        a = ev.get("args") or {}
        pid = str(ev.get("pid"))
        if ev.get("ph") == "M" and ev.get("name") == "process_name":
            labels[pid] = a.get("name")
        if ev.get("ph") != "X" or ("Physic Stream Id" not in a and labels.get(pid) != "Ascend Hardware"):
            continue
        name = ev.get("name", "")
        kind = a.get("Task Type") or (name if name in CONTROL else "")
        if (kind in CONTROL and not include_controls) or kind in NONCORE:
            continue
        width = ns(ev.get("dur", 0))
        if width < 0 or (width == 0 and kind not in CONTROL):
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


def _grouped_copy_replays(path, devices, replay_limits, layers, pieces):
    if layers < 2 or pieces < 2 or not replay_limits:
        raise ValueError("expected a sharded prefetch group and at least one replay")
    copies_by_device = {pid: [task for task in tasks if is_copy(task)] for pid, tasks in devices.items()}
    captures = [(pid, tasks) for pid, tasks in copies_by_device.items() if tasks]
    if len(captures) != 1:
        raise ValueError(f"{path}: expected SFA copy tasks on one device, found {len(captures)}")
    pid, tasks = captures[0]
    copies = sorted(tasks, key=lambda task: task.start)
    tasks_per_replay = 1 + (layers - 1) * pieces
    expected = len(replay_limits) * tasks_per_replay
    if len(copies) != expected:
        raise ValueError(f"{path}: expected {expected} device copy tasks in the timed replays, found {len(copies)}")

    for replay, limit in enumerate(replay_limits):
        offset = replay * tasks_per_replay
        root = copies[offset]
        side = copies[offset + 1 : offset + tasks_per_replay]
        # L0 is a single main-stream kernel. Its V-up opens the gate. A, B and C
        # each submit their pieces in order on one FIFO, with a join before the next replay.
        if (
            "piece" in root.name.lower()
            or any("piece" not in t.name.lower() for t in side)
            or len({t.stream for t in side}) != 1
            or side[0].stream == root.stream
        ):
            raise ValueError(f"{path}: replay {replay} does not match the main/side copy schedule")
        next_offset = offset + tasks_per_replay
        end = copies[next_offset].start if next_offset < len(copies) else float("inf")
        batches = [side[start : start + pieces] for start in range(0, len(side), pieces)]
        yield pid, limit, root, batches, end


def grouped_copy_layer_times(path, replay_limits, layers, pieces):
    """Median kernel execution and first-to-last span (us) per layer of the gated benchmark."""
    devices, _ = read_tasks(path)
    samples = defaultdict(list)
    for _, limit, root, batches, _ in _grouped_copy_replays(path, devices, replay_limits, layers, pieces):
        samples[limit, 0].append((root.stop - root.start, root.stop - root.start))
        for layer, batch in enumerate(batches, 1):
            active = sum(task.stop - task.start for task in batch)
            span = batch[-1].stop - batch[0].start
            samples[limit, layer].append((active, span))
    medians = {}
    for key, values in samples.items():
        active_ns, span_ns = zip(*values)
        medians[key] = statistics.median(active_ns) / 1000, statistics.median(span_ns) / 1000
    return medians


def _coalesce_capture_waits(main):
    """Fold profiler polling records into the hardware wait instruction on this replay's main stream."""
    ordered = []
    positions = {}
    for task in main:
        if task.kind in {"CAPTURE_WAIT", "EVENT_WAIT"} and 0 <= task.task < 0xFFFFFFFF:
            key = (task.model, task.task, task.kind)
            index = positions.get(key)
            if index is not None:
                first = ordered[index]
                ordered[index] = first._replace(stop=max(first.stop, task.stop))
                continue
            positions[key] = len(ordered)
        ordered.append(task)
    return ordered


def grouped_copy_waits(path, replay_limits, layers, pieces):
    """Pair consumer wait arrival with its own final copy on the side FIFO, per replay (ns)."""
    devices, _ = read_tasks(path, include_controls=True)
    samples = defaultdict(list)
    for pid, limit, root, batches, end in _grouped_copy_replays(path, devices, replay_limits, layers, pieces):
        main = sorted(
            (t for t in devices[pid] if t.stream == root.stream and root.start <= t.start < end),
            key=lambda t: (t.start, t.task),
        )
        main = _coalesce_capture_waits(main)
        all_waits = [t for t in main if t.kind in {"CAPTURE_WAIT", "EVENT_WAIT"}]
        # Each consumer opens the gate (write word, record Notify) immediately before its wait.
        # Capture infrastructure may add other waits on this same stream.
        gated = [
            t for write, signal, t in zip(main, main[1:], main[2:])
            if write.kind == "MEM_WRITE_VALUE" and signal.kind == "NOTIFY_RECORD"
            and t.kind in {"CAPTURE_WAIT", "EVENT_WAIT"}
        ]
        waits = gated or all_waits
        if len(waits) != len(batches):
            raise ValueError(f"{path}: expected {len(batches)} consumer waits, found {len(all_waits)} wait tasks "
                             f"and {len(gated)} matched gate openings on stream {root.stream}")
        for layer, (batch, wait) in enumerate(zip(batches, waits), 1):
            finished = batch[-1].stop
            if finished > wait.stop + 1000:  # Allow up to 1 us of cross-stream profiling skew.
                raise ValueError(f"{path}: final copy for consumer {layer} falls past its wait")
            remaining = max(0, min(finished, wait.stop) - wait.start)
            samples[limit, layer].append((remaining, wait.stop - wait.start))
    return samples


def cube_window_type(row):
    if row.kind == "AI_CORE":
        name = row.name.lower().replace("_", "")
        if name.startswith(("batchmatmultranspose", "transposebatchmatmul")):
            return "vup"
        if name.startswith("matmulv2"):
            return "router"
    return None


def copy_window_controls(cubes, events, norm, consumer_sfa):
    """Find device open/close pairs. Empty windows are valid on any replay."""
    tail_open = events.get(("NOTIFY_RECORD", norm.task + 2))
    join = events.get(("CAPTURE_WAIT", norm.task + 3))
    if not tail_open or not join:
        return {}, None
    windows = {"tail": (tail_open[0], join[1], join[0])}
    # Group capture holds the same FIFO grant across the wait and the fused QKV.
    # The adjacent-layer tail closes immediately after the wait.
    for tid in range(norm.task + 4, consumer_sfa.task):
        close = events.get(("MEM_WRITE_VALUE", tid))
        acquired = events.get(("NOTIFY_WAIT", tid + 1))
        if close and acquired:
            if tid > norm.task + 4:
                windows = {"qkv": (tail_open[0], acquired[1], close[0])}
            break
    for name, op in cubes:
        grant = events.get(("NOTIFY_RECORD", op.task - 1))
        close = events.get(("MEM_WRITE_VALUE", op.task + 1))
        acquired = events.get(("NOTIFY_WAIT", op.task + 2))
        if grant and acquired:
            windows[name] = (grant[0], acquired[1], close[0] if close else None)
    return windows, join


def window_piece_counts(parts, intervals):
    counts = dict.fromkeys((name for name, _ in intervals), 0)
    for part in parts:
        matches = [name for name, (start, stop, _) in intervals
                   if start - 1000 <= part.start and part.stop <= stop + 1000]
        if len(matches) == 1:
            counts[matches[0]] += 1
    return counts


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


def read_controls(path, pid, stream, stages, wanted, copy_streams):
    starts = [s[0] for s in stages]
    result = defaultdict(dict)
    done_records = defaultdict(dict)
    for ev in complete_events(path):
        if ev.get("ph") != "X" or str(ev.get("pid")) != pid:
            continue
        a = ev.get("args") or {}
        kind = a.get("Task Type")
        queue = int(a.get("Physic Stream Id", ev["tid"]))
        side_done = kind == "CAPTURE_RECORD" and queue in copy_streams
        if kind not in CONTROL or (queue != stream and not side_done):
            continue
        task = int(a.get("Task Id", -1))
        if not side_done and (kind, task) not in wanted and kind not in {
            "CAPTURE_RECORD", "MEM_WRITE_VALUE", "NOTIFY_WAIT", "NOTIFY_RECORD",
        }:
            continue
        start = ns(ev["ts"])
        step = bisect.bisect_right(starts, start) - 1
        if step < 0 or start >= stages[step][1]:
            continue
        stop = start + ns(ev.get("dur", 0))
        bucket = done_records[step] if side_done else result[step]
        key = (queue, task) if side_done else (kind, task)
        previous = bucket.get(key)
        bucket[key] = (min(start, previous[0]), max(stop, previous[1])) if previous else (start, stop)
    return result, done_records


def indexer_trains(norms, sfas, root_copies, side_copies, events, done_records, windows):
    """Pair device FIFO completion records with consuming layers, even when several finish in one window."""
    joins, forks = {}, {}
    for i, (sfa, norm) in enumerate(zip(sfas, norms[1:])):
        _, window_join = windows[i]
        join = window_join or events.get(("CAPTURE_WAIT", norm.task + 1))
        if join is None:
            continue
        joins[i] = join
        if ("CAPTURE_RECORD", sfa.task + 1) in events:  # Adjacent post-SFA fork.
            continue
        own = next((r for r in root_copies if norms[i].stop <= r.start < sfa.start), None)
        if own is not None:
            # The indexer forks after publishing the bridge, before retrieving
            # its own KV. Bound the search to this producer's pre-compute.
            records = [stop for (kind, tid), (_, stop) in events.items()
                       if kind == "CAPTURE_RECORD" and tid < own.task and norms[i].stop <= stop <= own.start + 1000]
            if records:
                forks[i] = max(records)
    if not forks:
        return {}
    if len({r.stream for r in side_copies}) != 1:
        raise ValueError("shared indexer trace requires one copy FIFO for this replay")
    record_distance = 2 if any(window_join for _, window_join in windows.values()) else 1
    trains, pieces = [], []
    for copy in side_copies:
        pieces.append(copy)
        # The ungated record follows the kernel. A gated kernel first returns the Notify.
        done = done_records.get((copy.stream, copy.task + record_distance))
        if done and copy.stop <= done[1] + 1000:
            trains.append((pieces, done))
            pieces = []
    if pieces or len(trains) != len(joins):
        raise ValueError(
            f"shared indexer trace has {len(trains)} complete layer trains for {len(joins)} consumer waits"
        )
    edges = {}
    origin = None
    for (i, join), (parts, done) in zip(joins.items(), trains):
        if i in forks:
            origin, ready = i, forks[i]
        elif release := events.get(("CAPTURE_RECORD", sfas[i].task + 1)):
            origin, ready = i, release[1]
        elif i - 1 not in edges or origin not in forks:
            raise ValueError(f"shared consumer layer {i + 1} has no indexer fork")
        if ready > parts[0].start + 1000 or done[1] > join[1] + 1000 or join[0] < norms[i + 1].stop:
            raise ValueError(f"shared consumer layer {i + 1} disagrees with its device fork or completion record")
        edges[i] = (origin, origin in forks, ready, join[0], parts)
    return edges


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
    stages, wanted, cube_tasks = [], set(), {}
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
        for layer, (sfa, norm) in enumerate(zip(sfas[:-1], norms[1:])):
            lo, hi = bisect.bisect_left(times, sfa.stop), bisect.bisect_left(times, norm.start)
            cubes = [(kind, op) for op in main[lo:hi] if (kind := cube_window_type(op))]
            cube_tasks[len(stages), layer] = cubes
            for _, op in cubes:
                wanted.update((("NOTIFY_RECORD", op.task - 1), ("MEM_WRITE_VALUE", op.task + 1),
                               ("NOTIFY_WAIT", op.task + 2)))
        wanted.update(("CAPTURE_WAIT", n.task + 1) for n in norms[1:])
        wanted.update(("CAPTURE_RECORD", n.task + 1) for n in norms[1:])
        wanted.update(("CAPTURE_WAIT", n.task + 2) for n in norms[1:])
        wanted.update(("NOTIFY_RECORD", n.task + 2) for n in norms[1:])
        wanted.update(("CAPTURE_WAIT", n.task + 3) for n in norms[1:])
        wanted.update(("CAPTURE_RECORD", r.task + 1) for r in sfas[:-1])
        stages.append((start, end, norms, sfas, final))
    if not stages:
        raise ValueError(f"no full replay with {expected} layers on stream {stream}, model {model}")
    print(
        f"[LAYER_GAPS] {len(stages)} complete replay(s); resolving main-stream fork/join events",
        file=sys.stderr,
        flush=True,
    )
    copies = [r for r in tasks if r.stream != stream and is_copy(r)]
    copy_times = [r.start for r in copies]
    controls, done_records = read_controls(path, pid, stream, stages, wanted, {r.stream for r in copies})
    main_copies = [r for r in main if is_copy(r)]
    main_copy_times = [r.start for r in main_copies]
    by_edge, background = {}, set()
    origin_layers, layer_windows, qkv_ops = {}, {}, {}
    for step, (start, end, norms, sfas, final) in enumerate(stages):
        events = controls[step]
        control_windows = {i: copy_window_controls(cube_tasks[step, i], events, norm, sfas[i + 1])
                           for i, norm in enumerate(norms[1:])}
        for i, (windows, join) in control_windows.items():
            layer_windows[step, i] = windows
            if "qkv" in windows:
                close = windows["qkv"][2]
                qkv_ops[step, i] = main[bisect.bisect_left(times, join[1]):bisect.bisect_left(times, close)]
        group_edges = indexer_trains(
            norms, sfas,
            main_copies[bisect.bisect_left(main_copy_times, start):bisect.bisect_left(main_copy_times, end)],
            copies[bisect.bisect_left(copy_times, start):bisect.bisect_left(copy_times, end)],
            events, done_records[step], control_windows,
        )
        if group_edges:
            for i, (origin, from_indexer, ready, deadline, parts) in group_edges.items():
                by_edge[step, i] = ready, deadline, parts, False, layer_windows[step, i]
                if from_indexer:
                    origin_layers[step, i] = origin
                background.update(parts)
            continue
        for i, sfa in enumerate(sfas[:-1]):
            a = bisect.bisect_left(copy_times, sfa.stop)
            b = bisect.bisect_left(copy_times, sfas[i + 1].start)
            parts = copies[a:b]
            next_norm = norms[i + 1]
            late_release = events.get(("CAPTURE_RECORD", next_norm.task + 1))
            late_join = events.get(("CAPTURE_WAIT", next_norm.task + 2))
            at_consumer = bool(parts) and late_release is not None and late_join is not None
            windows, window_join = control_windows[i]
            if windows:
                release, join = events.get(("CAPTURE_RECORD", sfa.task + 1)), window_join
                at_consumer = False
            elif at_consumer:
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
            by_edge[step, i] = ready, deadline, parts, at_consumer, windows
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
                ready, deadline, parts, at_consumer, windows = edge
                lengths = vector.gaps(ready, deadline)
                copy = sum(r.stop - r.start for r in parts)
                origin = origin_layers.get((step, i), i)
                if (step, i) in origin_layers:
                    s["indexer_layer"] = origin
                s["consumer_layer"] = i + 1
                cube_overlaps = {
                    name: Timeline(
                        (op.start, op.stop) for layer in range(origin, i + 1) for kind, op in cube_tasks[step, layer]
                        if kind == name and (not windows or (name in layer_windows[step, layer] and
                            layer_windows[step, layer][name][0] <= op.start <= layer_windows[step, layer][name][1]))
                    ).overlap(parts)
                    for name in ("vup", "router")
                }
                root_overlap = root_core.overlap(parts)
                qkv_overlap = Timeline(
                    (op.start, op.stop) for layer in range(origin, i + 1) for op in qkv_ops.get((step, layer), ())
                ).overlap(parts)
                if windows:
                    # A later transfer may use an earlier consumer's QKV grant
                    # while that consumer waits or computes.
                    allowed = [(name, span) for layer in range(origin, i + 1)
                               for name, span in layer_windows[step, layer].items() if name != "tail" or layer == i]
                    counts = window_piece_counts(parts, allowed)
                    s["window_pieces"] = counts
                    s["unassigned_pieces"] = len(parts) - sum(counts.values())
                    for name, (_, acquired, close) in windows.items():
                        if name != "tail" and close is not None:
                            s[f"{name}_close_to_acquire"] = acquired - close
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
                    copy_overlap_main_compute=root_overlap,
                    copy_overlap_vup=cube_overlaps["vup"],
                    copy_overlap_router=cube_overlaps["router"],
                    copy_overlap_qkv=qkv_overlap,
                    copy_overlap_protected_main_compute=root_overlap - sum(cube_overlaps.values()) - qkv_overlap,
                    cube_windows=[name for name in windows if name in {"vup", "router"}],
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
                        if not at_consumer and not windows:
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
        "copy_overlap_vup",
        "copy_overlap_router",
        "copy_overlap_protected_main_compute",
        "vup_close_to_acquire",
        "router_close_to_acquire",
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
    window_names = ("vup", "router", "tail")
    if qkv_ops:
        window_names += ("qkv",)
        edge_fields.extend(["copy_overlap_qkv", "qkv_close_to_acquire"])
    for i in range(expected):
        group = [s for s in samples if s["layer"] == i]
        row = {
            "layer": i,
            "replays": len(group),
            "async_to_next": sum("fit" in s for s in group),
            "indexer_layer_for_next": next((s["indexer_layer"] for s in group if "indexer_layer" in s), None),
            "released_at_consumer": sum(s.get("released_at_consumer", False) for s in group),
            "fits_total_gap": sum(s.get("fit", False) for s in group),
        }
        for name in layer_fields + edge_fields:
            st = sample_summary(group, name)
            row[name + "_mean_us"], row[name + "_p50_us"] = st.get("mean_us"), st.get("p50_us")
        for name in window_names:
            counts = [s["window_pieces"][name] for s in group if name in s.get("window_pieces", {})]
            row[name + "_pieces_mean"] = round(statistics.mean(counts), 2) if counts else None
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
        "indexer_prefetch_edges": len(origin_layers),
        "indexer_fork_layers_zero_based": sorted(set(origin_layers.values())),
        "consumer_release_edges": sum(s["released_at_consumer"] for s in edge_samples),
        "cube_windows": dict(Counter(name for s in edge_samples for name in s["cube_windows"])),
        "pieces_per_window_histogram": {
            name: dict(Counter(s["window_pieces"][name] for s in edge_samples if name in s.get("window_pieces", {})))
            for name in window_names
        },
        "pieces_outside_or_ambiguous_window": sum(s.get("unassigned_pieces", 0) for s in edge_samples),
        "edges_overlapping_protected_main_compute": sum(
            s["copy_overlap_protected_main_compute"] > 0 for s in edge_samples
        ),
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
    cli.add_argument("--grouped-410-waits", action="store_true",
                     help="Read the saved test_grouped_layer_copy_times_410 trace and print consumer waits")
    cli.add_argument("--pid")
    cli.add_argument("--model-id", type=int)
    cli.add_argument("--stream", type=int)
    cli.add_argument("--layers", type=int, default=78, help="Expected number of target transformer layers")
    cli.add_argument("--csv", type=Path, default=Path("decode-layer-gaps.csv"))
    cli.add_argument("--summary", type=Path, help="Also save the short JSON printed to stdout")
    args = cli.parse_args()
    if args.grouped_410_waits:
        limits = (12, 24, 0)
        schedule = [limit for order in permutations(limits) for limit in order]
        waits = grouped_copy_waits(args.trace, schedule, layers=4, pieces=16)
        print("consumer  cores  copy_wait_p50_us  wait_task_p50_us  ready_at_arrival/replays")
        for layer, name in enumerate(("A", "B", "C"), 1):
            for limit in limits:
                pending, elapsed = zip(*waits[limit, layer])
                print(f"{name:>8}  {str(limit or 'auto'):>5}  {statistics.median(pending) / 1000:>16.2f}"
                      f"  {statistics.median(elapsed) / 1000:>16.2f}  {pending.count(0)}/{len(pending)}")
        return
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
