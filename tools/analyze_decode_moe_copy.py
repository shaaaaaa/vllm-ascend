#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline analysis of one MindStudio decode trace, using only Python's standard library.

python analyze_decode_moe_copy.py trace_view.json --output summary.json --layers-csv layers.csv

All durations are device wall time. The threshold for a long MoE dispatch defaults to 200 us.
The JSON reader streams both bare event arrays and Chrome traceEvents objects. Only device
kernels and selected, coalesced stream controls are retained across the two passes.
"""

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict, namedtuple
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

Kernel = namedtuple("Kernel", "start end stream model task op")
CORE = {
    "AI_CORE",
    "AI_VECTOR_CORE",
    "MIX_AIC",
    "MIX_AIV",
    "KERNEL_AICORE",
    "KERNEL_AIVEC",
    "KERNEL_MIX_AIC",
    "KERNEL_MIX_AIV",
}
CONTROLS = {
    "CAPTURE_WAIT": "wait",
    "EVENT_WAIT": "wait",
    "CAPTURE_RECORD": "record",
    "EVENT_RECORD": "record",
    "MEM_WRITE_VALUE": "write",
    "NOTIFY_WAIT": "take",
    "NOTIFY_RECORD": "give",
}
COPY = {"copy", "copy_piece"}
MOE = {"dispatch", "combine", "expert_gemm", "fused_moe"}
BAD_ID = {-1, 0xFFFFFFFF}
SKEW_NS = 1000


def event_array_position(buf):
    """Locate a traceEvents array and retain a partial header across read boundaries."""
    key = '"traceEvents"'
    cursor = 0
    while (start := buf.find(key, cursor)) != -1:
        cursor = start + len(key)
        for expected in ":[":
            while cursor < len(buf) and buf[cursor].isspace():
                cursor += 1
            if cursor == len(buf):
                return None, buf[start:]
            if buf[cursor] != expected:
                break
            cursor += 1
        else:
            return cursor, ""
    return None, buf[-(len(key) - 1) :]


def events(path):
    # Incremental Chrome/MindStudio event reader adapted from analyze_decode_trace.py.
    # Keep the original decimal spelling of JSON numbers with epoch-sized timestamps.
    decoder = json.JSONDecoder(parse_float=str)
    with path.open(encoding="utf-8-sig") as f:
        buf = ""
        wrapped = False
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                raise ValueError("Missing trace event array")
            buf += chunk
            if not wrapped:
                beginning = buf.lstrip()
                if not beginning:
                    buf = ""
                    continue
                if beginning.startswith("["):
                    pos = buf.index("[") + 1
                    break
                if not beginning.startswith("{"):
                    raise ValueError("Expected a MindStudio JSON array or a Chrome traceEvents object")
                wrapped = True
            pos, tail = event_array_position(buf)
            if pos is not None:
                break
            buf = tail
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                item, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError as error:
                if not error.msg.startswith("Unterminated string") and error.pos + 8 < len(buf):
                    raise ValueError(f"Malformed JSON inside trace event: {error.msg}") from error
                chunk = f.read(1 << 20)
                if not chunk:
                    raise ValueError("Incomplete or invalid trace event") from error
                buf, pos = buf[pos:] + chunk, 0
                continue
            pos = end
            if isinstance(item, dict):
                yield item


def ns(value):
    whole, _, fraction = str(value).partition(".")
    if whole.isdigit() and (not fraction or fraction.isdigit()):
        return int(whole) * 1000 + int((fraction + "000")[:3])
    return int(Decimal(str(value)) * 1000)


def device_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def device_events(path, wanted, hardware, lanes=None):
    """Yield X or matched B/E hardware intervals. Ignore host launches and derived tracks."""
    stacks = defaultdict(list)
    for e in events(path):
        phase = e.get("ph")
        if phase not in ("X", "B", "E", "M"):
            continue
        a = e.get("args") or {}
        pid = e.get("pid")
        if phase == "M":
            if e.get("name") == "process_name" and a.get("name") == "Ascend Hardware":
                hardware.add(pid)
            continue
        if phase == "E":
            stack = stacks.get((pid, e.get("tid")))
            if stack:
                begin = stack.pop()
                if begin is not None:
                    yield (*begin, ns(e["ts"]))
            continue
        kind = a.get("Task Type") or e.get("name")
        physical = a.get("Physic Stream Id")
        # On the kernel pass also accept explicit device-task candidates. Some
        # exports put process metadata after the timed events; load_kernels
        # confirms their PID after this pass when no physical stream was given.
        selected = kind in wanted and (physical is not None or pid in hardware or (lanes is None and "Task Type" in a))
        if selected:
            stream = device_id(physical if physical is not None else e.get("tid"))
            selected = stream not in BAD_ID
            if lanes is not None:
                selected = selected and kind in lanes.get((pid, stream), ())
        if phase == "B":
            stack = stacks[pid, e.get("tid")]
            if selected:
                stack.append((pid, stream, kind, e.get("name", ""), a, ns(e["ts"])))
            else:
                stack.append(None)
            continue
        if selected:
            start = ns(e["ts"])
            yield pid, stream, kind, e.get("name", ""), a, start, start + ns(e.get("dur", 0))


@lru_cache(maxsize=8192)
def operator(name):
    n = name.lower().replace("_", "")
    if "sfacopy" in n:
        return "copy_piece" if "piece" in n else "copy"
    if "dispatchgmmcombine" in n or "megamoe" in n:
        return "fused_moe"
    if "moedistributedispatch" in n:
        return "dispatch"
    if "moedistributecombine" in n:
        return "combine"
    if "groupedmatmul" in n or "groupedgemm" in n:
        return "expert_gemm"
    if "sparseflashattention" in n:
        return "sfa"
    if n.startswith("kvrmsnormropecache"):
        return "kv_norm"
    if n.startswith("addrmsnormbias"):
        return "input_add_norm"
    if n.startswith("rmsnorm"):
        return "rms_norm"
    if n.startswith(("transposebatchmatmul", "batchmatmultranspose")):
        return "v_up"
    if n.startswith("matmulv2") and "fp32" in n:
        return "fp32_matmul"
    if "quantbatchmatmul" in n:
        return "quant_matmul"
    if "dynamicquant" in n:
        return "dynamic_quant"
    return "other_compute"


class Intervals:
    """Union of half-open nanosecond intervals with O(log n) intersection queries."""

    def __init__(self, pairs):
        self.starts, self.ends, self.prefix = [], [], [0]
        for start, end in sorted(pairs):
            if end <= start:
                continue
            if self.ends and start <= self.ends[-1]:
                self.ends[-1] = max(self.ends[-1], end)
            else:
                self.starts.append(start)
                self.ends.append(end)
        for a, b in zip(self.starts, self.ends):
            self.prefix.append(self.prefix[-1] + b - a)

    def before(self, t):
        i = bisect.bisect_right(self.ends, t)
        tail = max(0, t - self.starts[i]) if i < len(self.starts) else 0
        return self.prefix[i] + tail

    def overlap(self, start, end):
        return self.before(end) - self.before(start)

    def includes(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        return i >= 0 and t < self.ends[i]

    def intersect(self, other):
        return sum(other.overlap(a, b) for a, b in zip(self.starts, self.ends))


def span(rows):
    return Intervals((r.start, r.end) for r in rows)


def quantile(ordered, q):
    pos = (len(ordered) - 1) * q
    i = int(pos)
    return ordered[i] + (ordered[min(i + 1, len(ordered) - 1)] - ordered[i]) * (pos - i)


def us(t):
    return round(t / 1000, 3)


def stats(values, threshold=None):
    ordered = sorted(values)
    out = {"n": len(ordered), "sum_us": us(sum(ordered)) if ordered else None}
    for name, q in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        out[name + "_us"] = us(quantile(ordered, q)) if ordered else None
    out["mean_us"] = us(statistics.mean(ordered)) if ordered else None
    out["max_us"] = us(ordered[-1]) if ordered else None
    if threshold is not None:
        out["long_n"] = len(ordered) - bisect.bisect_left(ordered, threshold)
        out["long_pct"] = pct(out["long_n"], len(ordered))
        out["at_least_1ms_n"] = len(ordered) - bisect.bisect_left(ordered, 1_000_000)
    return out


def pct(num, denom):
    return round(100 * num / denom, 3) if denom else None


class Replay:
    def __init__(self, stream, model, mode, rows, layer_ids, expected, side, first_moe_id):
        self.stream, self.model, self.mode = stream, model, mode
        self.main = rows
        self.start, self.end = rows[0].start, max(r.end for r in rows)
        self.side = side
        self.controls = {}
        self.first_moe_id = first_moe_id
        self.sfas = [r for r in rows if r.op == "sfa"]
        self.sfa_times = [r.start for r in self.sfas]
        self.layer_ids = layer_ids
        self.expected = expected
        self.vups = [r for r in rows if r.op == "v_up"]
        self.complete = len(self.sfas) == expected and [layer_ids.get(r.task) for r in self.sfas] == list(
            range(expected)
        )
        if mode == "projection_test":
            self.complete = len(self.vups) == expected and sum(r.op in COPY for r in rows) == 1
        self.index = None
        self.parallel = False
        self.crossing_copy = False

    def layer(self, t):
        i = bisect.bisect_right(self.sfa_times, t) - 1
        return self.layer_ids.get(self.sfas[i].task) if i >= 0 else None


def load_kernels(path):
    devices = defaultdict(lambda: defaultdict(list))
    hardware = set()
    unconfirmed = defaultdict(list)
    for pid, stream, _, name, a, start, end in device_events(path, CORE, hardware):
        if end <= start:
            continue
        row = Kernel(start, end, stream, device_id(a.get("Model Id")), device_id(a.get("Task Id")), operator(name))
        if a.get("Physic Stream Id") is None and pid not in hardware:
            unconfirmed[pid].append(row)
        else:
            devices[pid][stream].append(row)
    for pid in hardware.intersection(unconfirmed):
        for row in unconfirmed[pid]:
            devices[pid][row.stream].append(row)
    for streams in devices.values():
        for rows in streams.values():
            rows.sort(key=lambda r: (r.start, r.task))
    return devices


def discover(streams, layers):
    """Root streams have SFA or the known projection-only smoke graph. Task IDs repeat on replay."""
    plans = defaultdict(lambda: defaultdict(set))
    for stream, rows in streams.items():
        for r in rows:
            if r.model not in BAD_ID and r.task not in BAD_ID:
                plans[stream, r.model][r.op].add(r.task)
    target = {key for key, roles in plans.items() if len(roles.get("sfa", ())) > 1}
    if target:
        expected = layers or max(len(plans[key]["sfa"]) for key in target)
        mode = "target_decode"
    else:
        pieces = any(any(r.op == "copy_piece" for r in rows) for rows in streams.values())
        target = {
            k
            for k, roles in plans.items()
            if pieces and len(roles.get("v_up", ())) >= 2 and any(r.op == "copy" for r in streams[k[0]])
        }
        expected = layers or max((len(plans[key]["v_up"]) for key in target), default=0)
        mode = "projection_test"
    copies = sorted((r for rows in streams.values() for r in rows if r.op in COPY), key=lambda r: r.start)
    copy_times = [r.start for r in copies]
    max_copy_ends = []
    for r in copies:
        max_copy_ends.append(max(max_copy_ends[-1], r.end) if max_copy_ends else r.end)
    first_task = {key: min(t for tasks in roles.values() for t in tasks) for key, roles in plans.items()}
    replays = []
    for stream in {key[0] for key in target}:
        rows = streams[stream]
        anchors = [
            (i, r.model)
            for i, r in enumerate(rows)
            if first_task.get((stream, r.model)) == r.task and r.model not in BAD_ID
        ]
        for number, (i, model) in enumerate(anchors):
            key = stream, model
            if key not in target:
                continue
            j = anchors[number + 1][0] if number + 1 < len(anchors) else len(rows)
            own = [r for r in rows[i:j] if r.model == model or r.model in BAD_ID]
            # Cut off unrelated work between graph launches after its last known model kernel.
            tail = next(pos for pos in range(len(own) - 1, -1, -1) if own[pos].model == model)
            own = own[: tail + 1]
            lo, hi = own[0].start, max(r.end for r in own)
            left = bisect.bisect_right(max_copy_ends, lo)
            right = bisect.bisect_left(copy_times, hi)
            sides = [r for r in copies[left:right] if r.stream != stream and r.end > lo]
            ids = {t: layer for layer, t in enumerate(sorted(plans[key].get("sfa", ())))}
            first_moe = min(plans[key].get("dispatch", ()), default=None)
            rep = Replay(stream, model, mode, own, ids, expected, sides, first_moe)
            rep.crossing_copy = any(r.start < lo or r.end > hi for r in sides)
            replays.append(rep)
    replays.sort(key=lambda rep: rep.start)
    # Physical lanes may run independent graphs concurrently. Without a model ID on
    # custom copy kernels, a shared interval is insufficient to assign one to a replay.
    active = []
    for i, rep in enumerate(replays):
        rep.index = i
        active = [old for old in active if old.end > rep.start]
        if active:
            for old in active:
                old.parallel = True
            rep.parallel = True
        active.append(rep)
    return replays


def load_controls(path, by_device):
    # Coalesce polling samples by instruction inside a replay, never across replays.
    schedules = defaultdict(list)
    lanes = defaultdict(set)
    for pid, replays in by_device.items():
        for rep in replays:
            if rep.parallel or rep.crossing_copy or not rep.complete:
                continue
            for stream in {rep.stream} | {k.stream for k in rep.side}:
                schedules[pid, stream].append(rep)
                side_kinds = ("CAPTURE_RECORD", "EVENT_RECORD", "NOTIFY_RECORD")
                kinds = CONTROLS if stream == rep.stream else side_kinds
                lanes[pid, stream].update(kinds)
    starts = {key: [rep.start for rep in val] for key, val in schedules.items()}
    if not schedules:
        return
    for pid, stream, kind, _, a, begin, end in device_events(path, CONTROLS, set(by_device), lanes):
        key = pid, stream
        if key not in starts:
            continue
        idx = bisect.bisect_right(starts[key], begin) - 1
        if idx < 0:
            continue
        rep = schedules[key][idx]
        code = CONTROLS[kind]
        if begin >= rep.end or (stream != rep.stream and code not in {"record", "give"}):
            continue
        task_id = device_id(a.get("Task Id"))
        if task_id in BAD_ID or end < begin:
            continue
        address = stream, task_id, code
        previous = rep.controls.get(address)
        if previous is None:
            rep.controls[address] = (begin, end)
        else:
            rep.controls[address] = (min(previous[0], begin), max(previous[1], end))


def joins(rep):
    """Locate only copy-consumer waits, using the emitted GLM or test graph instructions."""
    main = rep.stream
    controls = rep.controls
    if rep.mode == "projection_test":
        ordered = sorted(
            (
                v
                for (s, tid, kind), v in controls.items()
                if s == main
                and kind == "wait"
                and (s, tid - 1, "give") in controls
                and (s, tid - 2, "write") in controls
            )
        )
        if len(ordered) != rep.expected - 1:
            return None, "projection_wait_signature_missing"
        return list(enumerate(ordered, 1)), None
    result = []
    low = 0
    for layer, sfa in enumerate(rep.sfas):
        before = []
        while low < len(rep.main) and rep.main[low].start < sfa.start:
            before.append(rep.main[low])
            low += 1
        low += 1
        if layer == 0:
            continue
        kv = [i for i, r in enumerate(before) if r.op == "kv_norm"]
        if len(kv) != 1:
            return None, "KV_RoPE_anchor_missing"
        norm = next((r for r in reversed(before[: kv[0]]) if r.op == "input_add_norm"), None)
        if norm is None:
            return None, "input_norm_anchor_missing"
        if any(r.op in COPY and r.start >= norm.end for r in before):
            continue  # This layer loads its KV directly on the main stream.
        t = norm.task
        wait = None
        if (main, t + 1, "write") in controls and (main, t + 2, "give") in controls:
            wait = controls.get((main, t + 3, "wait"))
        elif (main, t + 1, "record") in controls:
            wait = controls.get((main, t + 2, "wait"))
        else:
            wait = controls.get((main, t + 1, "wait"))
        if wait:
            if wait[0] < norm.end - SKEW_NS or wait[1] > sfa.start + SKEW_NS:
                return None, "wait_outside_consumer_precompute"
            result.append((layer, wait))
    return result, None


def pair_copies(rep):
    found, issue = joins(rep)
    if issue:
        return [], issue
    if not rep.side and not found:
        return [], None
    if len({r.stream for r in rep.side}) != 1:
        return [], "missing_or_ambiguous_side_FIFO"
    # A gated piece returns its Notify before the done event. Only the last
    # piece of this consumer emits a completion record on the side FIFO.
    trains, batch = [], []
    for piece in rep.side:
        batch.append(piece)
        s, tid = piece.stream, piece.task
        delta = 2 if (s, tid + 1, "give") in rep.controls else 1
        record = rep.controls.get((s, tid + delta, "record"))
        if record:
            if record[0] + SKEW_NS < piece.end:
                return [], "done_record_precedes_copy"
            if delta == 2:
                give = rep.controls[s, tid + 1, "give"]
                if give[0] < piece.end - SKEW_NS or give[1] > record[0] + SKEW_NS:
                    return [], "gate_return_and_done_out_of_order"
            trains.append((batch, record))
            batch = []
    if batch or len(trains) != len(found):
        return [], "side_completions_and_consumer_waits_disagree"
    pairs = []
    for (layer, wait), (pieces, done) in zip(found, trains):
        last = pieces[-1].end
        if done[1] > wait[1] + SKEW_NS or last > wait[1] + SKEW_NS:
            return [], "consumer_wait_ends_before_its_copy"
        pending = max(0, min(last, wait[1]) - wait[0])
        pairs.append((layer, wait, pieces, pending))
    return pairs, None


def overlap_sample(rep):
    main = [r for r in rep.main if r.op not in COPY]
    foreground, copies = span(main), span(rep.side)
    by_op = defaultdict(list)
    for r in main:
        by_op[r.op].append((r.start, r.end))
    both = copies.intersect(foreground)
    inside = foreground.overlap(copies.starts[0], copies.ends[-1]) if copies.starts else 0
    sample = {
        "pieces": len(rep.side),
        "copy": copies.prefix[-1],
        "main": foreground.prefix[-1],
        "main_in_copy_span": inside,
        "both": both,
        "inline": span(r for r in rep.main if r.op in COPY).prefix[-1],
        "by_op": {op: copies.intersect(Intervals(ranges)) for op, ranges in by_op.items()},
    }
    return foreground, copies, sample


def overlap_report(samples):
    sums = Counter()
    by_op = Counter()
    for s in samples:
        sums.update({k: n for k, n in s.items() if k != "by_op"})
        by_op.update(s["by_op"])
    return {
        "windows": len(samples),
        "side_copy_kernels": sums["pieces"],
        "side_copy_wall_us": us(sums["copy"]) if samples else None,
        "main_compute_wall_us": us(sums["main"]) if samples else None,
        "main_compute_inside_copy_span_us": us(sums["main_in_copy_span"]) if samples else None,
        "overlap_us": us(sums["both"]) if samples else None,
        "pct_of_side_copy": pct(sums["both"], sums["copy"]),
        "pct_of_main_compute": pct(sums["both"], sums["main"]),
        "pct_of_main_compute_inside_copy_span": pct(sums["both"], sums["main_in_copy_span"]),
        "main_stream_inline_copy_us": us(sums["inline"]) if samples else None,
        "per_window_overlap": stats(s["both"] for s in samples),
        "per_window_side_copy": stats(s["copy"] for s in samples),
        "overlap_by_main_operator_us": {k: us(v) for k, v in sorted(by_op.items())},
    }


def exposure(copies, r):
    share = copies.overlap(r.start, r.end)
    if copies.includes(r.start):
        return "copy_at_entry", share
    return ("copy_started_during" if share else "clear"), share


def worst(rows, top):
    return sorted(rows, key=lambda x: x["duration_us"], reverse=True)[:top]


def wait_report(pairs, totals, full_totals, closes, failures, eligible):
    pending = [p[0] for p in pairs]
    if len(totals) < eligible:
        status = "partial" if totals else "unavailable"
    elif pending:
        status = "matched"
    else:
        status = "no_async_joins_observed" if eligible else "no_complete_replay"
    return {
        "status": status,
        "eligible_replays": eligible,
        "verified_replays": len(totals),
        "skipped_replays_by_reason": dict(failures),
        "joined_layers": len(pairs),
        "ready_at_arrival": pending.count(0) if pending else None,
        "still_pending_at_arrival": sum(t > 0 for t in pending) if pending else None,
        "copy_stall_per_join": stats(pending),
        "copy_stall_when_pending": stats(t for t in pending if t),
        "copy_stall_sum_per_replay": stats(totals),
        "full_event_wait_per_join": stats(full for _, full in pairs),
        "full_event_wait_sum_per_replay": stats(full_totals),
        "gate_close_notify_wait_sum_per_gated_replay": stats(closes),
    }


def fallback_replay(streams):
    # Eager/different graphs can still expose physical overlap without copy-completion markers.
    counts = {stream: sum(r.op in {"sfa", "dispatch", "fused_moe"} for r in rows) for stream, rows in streams.items()}
    if not any(counts.values()):
        return []
    stream = max(counts, key=counts.get)
    rows = streams[stream]
    lo, hi = rows[0].start, max(r.end for r in rows)
    side = sorted(
        (
            r
            for s, kernels in streams.items()
            if s != stream
            for r in kernels
            if r.op in COPY and lo <= r.start < r.end <= hi
        ),
        key=lambda r: r.start,
    )
    rep = Replay(stream, -1, "unsegmented", rows, {}, 0, side, None)
    rep.index, rep.complete = 0, False
    return [rep]


def device_report(pid, streams, replays, threshold, top):
    origin = min(rows[0].start for rows in streams.values())
    pools, stage_times = defaultdict(list), defaultdict(list)
    slow, slow_moe, slow_stalls = defaultdict(list), defaultdict(list), []
    overlap_samples, all_pairs, totals, full_totals, closes = [], [], [], [], []
    graphs, layers = defaultdict(list), defaultdict(lambda: defaultdict(list))
    reasons, pair_failures = Counter(), Counter()
    seen_moe = set()
    eligible = 0

    for rep in replays:
        key = rep.stream, rep.model
        entry = {
            "replay": rep.index,
            "complete": rep.complete,
            "mode": rep.mode,
            "expected_layers": rep.expected if rep.mode != "unsegmented" else None,
        }
        graphs[key].append(entry)
        foreground, copies, sample = overlap_sample(rep)
        if (rep.complete or rep.mode == "unsegmented") and not rep.parallel and not rep.crossing_copy:
            overlap_samples.append(sample)
            entry["overlap"] = sample

        # MoE latency is available even where fork/join or the end of a replay is missing.
        layer_moe = defaultdict(lambda: defaultdict(list))
        for r in rep.main:
            if r.op not in MOE:
                continue
            seen_moe.add(r)
            duration = r.end - r.start
            layer = rep.layer(r.start)
            stage_times[r.op].append(duration)
            layer_moe[layer][r.op].append(r)
            if r.op != "dispatch":
                continue
            if rep.mode != "target_decode" or rep.first_moe_id is None:
                scope = "target_order_unknown"
            else:
                scope = "target_first" if r.task == rep.first_moe_id else "target_regular"
            state, share = exposure(copies, r)
            if rep.parallel:
                state, share = "ambiguous_replay", None
            pools[scope].append(duration)
            if scope == "target_regular":
                pools["regular_" + state].append(duration)
            if layer is not None:
                layers[key, layer]["dispatch"].append(duration)
            slow[scope].append(
                {
                    "stream": rep.stream,
                    "model": rep.model,
                    "replay": rep.index,
                    "layer": layer,
                    "start_ms_from_first_device_kernel": round((r.start - origin) / 1e6, 3),
                    "duration_us": us(duration),
                    "copy": state,
                    "copy_overlap_us": us(share) if share is not None else None,
                }
            )
        for layer, ops in layer_moe.items():
            if layer is None or not ops["dispatch"]:
                continue
            if len(ops["dispatch"]) == len(ops["combine"]) == 1:
                first, last = ops["dispatch"][0], ops["combine"][0]
                if first.end <= last.start:
                    scope = "first" if first.task == rep.first_moe_id else "regular"
                    elapsed = last.end - first.start
                    stage_times["dispatch_to_combine_wall_" + scope].append(elapsed)
                    layers[key, layer]["moe_wall"].append(elapsed)
                    active_experts = [r for r in ops["expert_gemm"] if first.end <= r.start and r.end <= last.start]
                    slow_moe[scope].append(
                        {
                            "stream": rep.stream,
                            "model": rep.model,
                            "replay": rep.index,
                            "layer": layer,
                            "start_ms_from_first_device_kernel": round((first.start - origin) / 1e6, 3),
                            "duration_us": us(elapsed),
                            "dispatch_us": us(first.end - first.start),
                            "expert_gemm_wall_us": us(span(active_experts).prefix[-1]),
                            "combine_us": us(last.end - last.start),
                            "copy_active_inside_span_us": None
                            if rep.parallel
                            else us(copies.overlap(first.start, last.end)),
                        }
                    )
                    continue
            pair_failures["layer_without_one_ordered_dispatch_combine_pair"] += 1

        if rep.parallel:
            reasons["concurrent_graph_ownership_ambiguous"] += 1
            continue
        if rep.crossing_copy:
            reasons["copy_crosses_observed_replay_boundary"] += 1
            continue
        if not rep.complete:
            reasons["partial_or_unsegmented_replay"] += 1
            continue
        eligible += 1
        for idx in range(rep.expected):
            layers[key, idx]["replays"].append(rep.index)
        inline = defaultdict(list)
        for r in rep.main:
            if r.op in COPY:
                idx = bisect.bisect_left(rep.sfa_times, r.end) if rep.mode == "target_decode" else 0
                if idx < rep.expected:
                    inline[idx].append(r)
        for idx, kernels in inline.items():
            layers[key, idx]["inline"].append(span(kernels).prefix[-1])
        pairs, issue = pair_copies(rep)
        if issue:
            reasons[issue] += 1
            entry["wait_issue"] = issue
            continue
        stalls, event_times = [], []
        for layer, wait, pieces, pending in pairs:
            full = wait[1] - wait[0]
            stalls.append(pending)
            event_times.append(full)
            all_pairs.append((pending, full))
            bucket = layers[key, layer]
            batch_span = span(pieces)
            bucket["copy"].append(batch_span.prefix[-1])
            bucket["overlap"].append(batch_span.intersect(foreground))
            bucket["wait"].append(pending)
            bucket["event"].append(full)
            if pending:
                slow_stalls.append(
                    {
                        "stream": rep.stream,
                        "model": rep.model,
                        "replay": rep.index,
                        "consumer_layer": layer,
                        "start_ms_from_first_device_kernel": round((wait[0] - origin) / 1e6, 3),
                        "duration_us": us(pending),
                        "event_wait_us": us(full),
                        "copy_pieces": len(pieces),
                    }
                )
        totals.append(sum(stalls))
        full_totals.append(sum(event_times))
        entry["wait_total"] = sum(stalls)
        if any((r.stream, r.task + 1, "give") in rep.controls for r in rep.side):
            gate_waits = [
                (a, b)
                for (s, tid, k), (a, b) in rep.controls.items()
                if s == rep.stream and k == "take" and (s, tid - 1, "write") in rep.controls
            ]
            if gate_waits:
                closes.append(Intervals(gate_waits).prefix[-1])

    # Keep first and other streams (including potential MTP/eager work) out of regular-layer comparisons.
    outside = defaultdict(list)
    for sid, rows in streams.items():
        for r in rows:
            if r.op == "dispatch" and r not in seen_moe:
                outside[sid].append(r.end - r.start)
                slow["outside_target"].append(
                    {
                        "stream": sid,
                        "model": r.model,
                        "replay": None,
                        "layer": None,
                        "start_ms_from_first_device_kernel": round((r.start - origin) / 1e6, 3),
                        "duration_us": us(r.end - r.start),
                    }
                )

    graph_results = []
    for (stream, model), entries in sorted(graphs.items()):
        graph_results.append(
            {
                "main_stream": stream,
                "model_id": model,
                "scope": entries[0]["mode"],
                "expected_layers": entries[0]["expected_layers"],
                "observed_replays": len(entries),
                "complete_replays": sum(e["complete"] for e in entries),
                "overlap": overlap_report([e["overlap"] for e in entries if "overlap" in e]),
                "copy_stall_sum_per_replay": stats(e["wait_total"] for e in entries if "wait_total" in e),
            }
        )
    csv_rows = []
    for ((stream, model), layer), data in sorted(layers.items()):
        wait, event, dispatch = stats(data["wait"]), stats(data["event"]), stats(data["dispatch"], threshold)
        copy, share = stats(data["copy"]), stats(data["overlap"])
        moe = stats(data["moe_wall"])
        csv_rows.append(
            {
                "device_pid": pid,
                "main_stream": stream,
                "model_id": model,
                "layer_0based": layer,
                "dispatch_calls": dispatch["n"],
                "dispatch_long": dispatch["long_n"],
                "dispatch_mean_us": dispatch["mean_us"],
                "dispatch_p95_us": dispatch["p95_us"],
                "dispatch_max_us": dispatch["max_us"],
                "moe_wall_p95_us": moe["p95_us"],
                "moe_wall_max_us": moe["max_us"],
                "matched_joins": wait["n"],
                "complete_replays": len(data["replays"]),
                "inline_copy_p50_us": stats(data["inline"])["p50_us"],
                "ready_at_arrival": data["wait"].count(0),
                "copy_p50_us": copy["p50_us"],
                "overlap_p50_us": share["p50_us"],
                "overlap_pct_of_copy": pct(sum(data["overlap"]), sum(data["copy"])),
                "wait_sum_us": wait["sum_us"],
                "wait_mean_us": wait["mean_us"],
                "wait_p50_us": wait["p50_us"],
                "wait_p95_us": wait["p95_us"],
                "wait_max_us": wait["max_us"],
                "full_event_p50_us": event["p50_us"],
            }
        )
    stats_by_scope = {
        k: stats(pools[k], threshold)
        for k in (
            "target_regular",
            "regular_copy_at_entry",
            "regular_copy_started_during",
            "regular_clear",
            "target_first",
            "target_order_unknown",
        )
    }
    if pools["regular_ambiguous_replay"]:
        stats_by_scope["regular_ambiguous_replay"] = stats(pools["regular_ambiguous_replay"], threshold)
    stats_by_scope["outside_target_by_stream"] = {s: stats(v, threshold) for s, v in sorted(outside.items())}
    out = {
        "hardware_pid": pid,
        "device_origin_us": str(Decimal(origin) / 1000),
        "graphs": graph_results,
        "observed_copy_kernels_by_stream": {
            s: n for s, rows in streams.items() if (n := sum(r.op in COPY for r in rows))
        },
        "dispatch": stats_by_scope,
        "slowest_dispatch": {k: worst(v, top) for k, v in slow.items()},
        "moe_stage_wall": {k: stats(v) for k, v in sorted(stage_times.items())},
        "slowest_dispatch_to_combine_wall": {k: worst(v, top) for k, v in slow_moe.items()},
        "unpaired_dispatch_to_combine_layers": sum(pair_failures.values()),
        "overlap": overlap_report(overlap_samples),
        "consumer_wait": wait_report(all_pairs, totals, full_totals, closes, reasons, eligible),
        "largest_consumer_copy_stalls": worst(slow_stalls, top),
    }
    return out, csv_rows


def analyze(path, threshold, layers, top):
    before = path.stat()
    print("[DECODE] 1/2 reading NPU kernels and graph replay boundaries", file=sys.stderr, flush=True)
    devices = load_kernels(path)
    if not devices:
        raise ValueError("No physical Ascend AI-core/vector kernel intervals found in this trace")
    replays = {pid: discover(streams, layers) or fallback_replay(streams) for pid, streams in devices.items()}
    print("[DECODE] 2/2 matching layer completion records and waits", file=sys.stderr, flush=True)
    load_controls(path, replays)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Trace changed during analysis; use the completed export")
    report = {
        "trace": path.name,
        "trace_bytes": before.st_size,
        "long_threshold_us": us(threshold),
        "units": "us unless labeled otherwise; zero-based layers",
        "moe_scope": "The first target dispatch, other target dispatches, and outside streams have "
        "separate distributions. Dispatch-to-combine includes expert kernels and waits.",
        "overlap_definition": "Device side-stream SFA copy union intersected with main-stream AI kernels. "
        "CPU submission, waits, SDMA and communication-only tracks are excluded.",
        "stall_definition": "From arrival at the layer's device wait to its last copy piece finishing, "
        "floored at zero. Full event and gate closing durations are separate metrics.",
        "devices": [],
    }
    rows = []
    for pid, streams in devices.items():
        result, layers_data = device_report(pid, streams, replays[pid], threshold, top)
        report["devices"].append(result)
        rows.extend(layers_data)
    return report, rows


CSV_FIELDS = [
    "device_pid",
    "main_stream",
    "model_id",
    "layer_0based",
    "dispatch_calls",
    "dispatch_long",
    "dispatch_mean_us",
    "dispatch_p95_us",
    "dispatch_max_us",
    "moe_wall_p95_us",
    "moe_wall_max_us",
    "complete_replays",
    "inline_copy_p50_us",
    "matched_joins",
    "ready_at_arrival",
    "copy_p50_us",
    "overlap_p50_us",
    "overlap_pct_of_copy",
    "wait_sum_us",
    "wait_mean_us",
    "wait_p50_us",
    "wait_p95_us",
    "wait_max_us",
    "full_event_p50_us",
]


def print_compact(report):
    threshold = report["long_threshold_us"]

    def value(n):
        return "n/a" if n is None else f"{n:,.2f}"

    def timing(label, sample):
        tail = (
            f"  >={threshold:g} us: {sample['long_n']}/{sample['n']}  >=1 ms: {sample['at_least_1ms_n']}"
            if "long_n" in sample and sample["n"]
            else ""
        )
        print(
            f"  {label}: n={sample['n']}  mean={value(sample['mean_us'])}  "
            f"p50={value(sample['p50_us'])}  p95={value(sample['p95_us'])}  "
            f"p99={value(sample['p99_us'])}  max={value(sample['max_us'])} us{tail}"
        )

    for dev in report["devices"]:
        print(f"[DECODE] hardware pid={dev['hardware_pid']}")
        print("MoE dispatch (device kernels)")
        for key in (
            "target_regular",
            "regular_copy_at_entry",
            "regular_copy_started_during",
            "regular_clear",
            "target_first",
            "target_order_unknown",
        ):
            sample = dev["dispatch"][key]
            if sample["n"] or key == "target_regular":
                timing(key, sample)
        for stream, sample in dev["dispatch"]["outside_target_by_stream"].items():
            timing(f"outside_target stream {stream}", sample)
        for key in (
            "dispatch_to_combine_wall_regular",
            "dispatch_to_combine_wall_first",
            "expert_gemm",
            "combine",
            "fused_moe",
        ):
            if key in dev["moe_stage_wall"]:
                timing(key, dev["moe_stage_wall"][key])
        o = dev["overlap"]
        print(
            f"Copy/compute: {value(o['overlap_us'])} us simultaneous / {value(o['side_copy_wall_us'])} us "
            f"background copy = {value(o['pct_of_side_copy'])}%"
        )
        if o["overlap_by_main_operator_us"]:
            stages = ", ".join(f"{op}={value(t)}" for op, t in o["overlap_by_main_operator_us"].items())
            print(f"  overlap per main operator (us): {stages}")
        wr = dev["consumer_wait"]
        layer, step = wr["copy_stall_per_join"], wr["copy_stall_sum_per_replay"]
        print(
            f"Consumer wait: {wr['status']}  "
            f"verified replays={wr['verified_replays']}/{wr['eligible_replays']}  "
            f"joins={wr['joined_layers']}  ready={wr['ready_at_arrival']}  "
            f"pending={wr['still_pending_at_arrival']}"
        )
        print(
            f"  remaining copy at join: total={value(layer['sum_us'])}  mean={value(layer['mean_us'])}  "
            f"p95={value(layer['p95_us'])}  max={value(layer['max_us'])} us"
        )
        print(f"  sum per replay: p50={value(step['p50_us'])}  p95={value(step['p95_us'])} us")
        close = wr["gate_close_notify_wait_sum_per_gated_replay"]
        if close["n"]:
            print(f"  gate-close Notify wait (separate): p50={value(close['p50_us'])} us per gated replay")
        if wr["skipped_replays_by_reason"]:
            print("  excluded:", json.dumps(wr["skipped_replays_by_reason"], sort_keys=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("trace", type=Path, help="One rank's trace_view.json")
    cli.add_argument("--long-us", type=float, default=200, help="Count dispatch kernels at or above this duration (us)")
    cli.add_argument("--layers", type=int, help="Expected layer count (default: infer from SFA task IDs)")
    cli.add_argument("--top", type=int, default=5, help="Number of worst examples per scope")
    cli.add_argument("--output", type=Path, help="Save full JSON here and print a short terminal report")
    cli.add_argument("--layers-csv", type=Path, help="Save timing statistics for each consumer/MoE layer")
    args = cli.parse_args()
    if (
        not math.isfinite(args.long_us)
        or args.long_us <= 0
        or args.top < 0
        or (args.layers is not None and args.layers < 1)
    ):
        cli.error("Use a positive duration/layer count and a nonnegative example count")
    paths = [p.resolve() for p in (args.trace, args.output, args.layers_csv) if p is not None]
    if len(paths) != len(set(paths)):
        cli.error("Input, summary and CSV must have distinct paths")
    try:
        report, rows = analyze(args.trace, max(1, round(args.long_us * 1000)), args.layers, args.top)
        content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output:
            args.output.write_text(content + "\n", encoding="utf-8")
        if args.layers_csv:
            with args.layers_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        if args.output:
            print_compact(report)
            print(f"[DECODE] summary: {args.output.resolve()}")
            if args.layers_csv:
                print(f"[DECODE] per-layer CSV: {args.layers_csv.resolve()}")
        else:
            print(content)
    except (OSError, ValueError) as e:
        cli.exit(1, f"[DECODE] {e}\n")


if __name__ == "__main__":
    main()
