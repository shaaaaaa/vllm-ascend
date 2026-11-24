#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read a saved check_decode_creation_priority capture and compare A/B kernel progress."""

import argparse
import csv
import json
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from statistics import median

from analyze_decode_layer_gaps import Timeline, ns
from check_decode_creation_priority import REPLAYS, STEPS
from summarize_prefill_profile import events


def is_abs(name):
    words = "".join(c if c.isalpha() else " " for c in str(name).lower()).split()
    return any(word in {"abs", "absd", "absaicore", "aclnnabs", "aclnnabsd", "aclnnabsaicore"} for word in words)


def csv_abs(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = {
            "".join(c for c in s.lower().replace("μ", "u").replace("µ", "u") if c.isalnum()): s
            for s in reader.fieldnames or ()
        }
        required = ("streamid", "taskstarttimeus", "taskdurationus")
        if not all(k in columns for k in required) or not ({"opname", "optype"} & columns.keys()):
            raise ValueError(f"{path.name} lacks per-operator stream, name or device timing columns")
        stream, start, duration = (columns[k] for k in required)
        for row in reader:
            if is_abs(row.get(columns.get("opname"))) or is_abs(row.get(columns.get("optype"))):
                yield row[stream], row[start], row[duration], row.get(columns.get("blockdim"))


def timeline_abs(path, requested):
    hardware = set()
    for ev in events(path):
        a = ev.get("args") or {}
        if ev.get("ph") == "M" and ev.get("name") == "process_name" and a.get("name") == "Ascend Hardware":
            hardware.add(ev.get("pid"))
        if ev.get("ph") != "X" or ("Physic Stream Id" not in a and ev.get("pid") not in hardware):
            continue
        if is_abs(ev.get("name")) or is_abs(a.get("Op Type")):
            ids = [str(value).strip() for value in (a.get("Stream Id"), ev.get("tid"), a.get("Physic Stream Id"))]
            stream = next((s for s in ids if s in requested), ids[-1])
            yield stream, ev["ts"], ev["dur"], a.get("Block Dim")


def exported_files(path):
    if path.is_file():
        return [[path]]
    if not path.is_dir():
        raise ValueError(f"missing capture directory {path}")
    captures = sorted(p for p in path.glob("PROF_*") if p.is_dir())
    if len(captures) > 1:
        raise ValueError(f"{path} contains {len(captures)} captures; select one PROF directory")
    root = captures[0] if captures else path
    exported = root / "mindstudio_profiler_output"
    if exported.is_dir():
        root = exported
    return [sorted(root.glob(pattern)) for pattern in ("op_summary*.csv", "msprof*.json", "trace_view.json")]


def replay_progress(a_rows, b_rows):
    a_ends, b_ends = (sorted(end for _, end in rows) for rows in (a_rows, b_rows))
    half = STEPS // 2
    a_mid, b_mid = a_ends[half - 1], b_ends[half - 1]
    when = min(a_mid, b_mid)
    a, b = Timeline(a_rows), Timeline(b_rows)
    overlap = sum(end - start - sum(b.gaps(start, end)) for start, end in a.spans)
    shorter_busy = min(sum(y - x for x, y in lane.spans) for lane in (a, b))
    return {
        "a_lead_at_half_kernels": bisect_right(a_ends, when) - bisect_right(b_ends, when),
        "b_minus_a_half_us": (b_mid - a_mid) / 1000,
        "b_minus_a_last_us": (b_ends[-1] - a_ends[-1]) / 1000,
        "interval_overlap_pct": 100 * overlap / shorter_busy,
    }


def count_ready_starts(a_rows, b_rows, counts):
    # Both chains were enqueued at graph build time. Count the next observed
    # start only while each lane still has work and both predecessors have ended.
    # Skip the initial launches, which also include activation of the two heads.
    i = j = 0
    while i < len(a_rows) and j < len(b_rows):
        a_start, b_start = a_rows[i][0], b_rows[j][0]
        if i and j and max(a_rows[i - 1][1], b_rows[j - 1][1]) <= min(a_start, b_start):
            if a_start == b_start:
                counts["tie"] += 1
            else:
                counts["a" if a_start < b_start else "b"] += 1
        if a_start <= b_start:
            i += 1
        if b_start <= a_start:
            j += 1


def summarize(files, streams):
    lanes = defaultdict(set)
    dims = set()
    for path in files:
        rows = csv_abs(path) if path.suffix.lower() == ".csv" else timeline_abs(path, streams)
        for stream, start, duration, blocks in rows:
            stream = str(stream).strip()
            t, width = ns(str(start).strip()), ns(str(duration).strip())
            if width > 0:
                lanes[stream].add((t, t + width))
                if stream in streams and blocks not in (None, "", "N/A"):
                    dims.add(str(blocks))

    rounds = REPLAYS + 1
    required = 6 * rounds * STEPS
    if any(len(lanes[s]) != required for s in streams):
        counts = {s: len(rows) for s, rows in lanes.items() if rows}
        raise ValueError(f"need {required} Abs rows on each of {streams}; found {counts}")
    first, second = (sorted(lanes[s]) for s in streams)
    cases = []
    previous_end = None
    for order in ("AB", "BA"):
        for priority in ("equal", "A_high", "B_high"):
            samples = []
            ready_starts = {"a": 0, "b": 0, "tie": 0}
            for replay in range(rounds):
                i = (len(cases) * rounds + replay) * STEPS
                a, b = first[i : i + STEPS], second[i : i + STEPS]
                if previous_end is not None and min(a[0][0], b[0][0]) < previous_end:
                    raise ValueError("Abs rows cross a synchronized replay boundary; check the capture and stream IDs")
                previous_end = max(end for _, end in a + b)
                if replay:
                    if order == "BA":
                        a, b = b, a
                    samples.append(replay_progress(a, b))
                    count_ready_starts(a, b, ready_starts)
            cases.append(
                {
                    "order": order,
                    "priority": priority,
                    "ready_start_counts": ready_starts,
                    **{k: round(median(row[k] for row in samples), 2) for k in samples[0]},
                }
            )
    return {
        "sources": [p.name for p in files],
        "streams_in_creation_order": streams,
        "block_dim": sorted(dims),
        "measured_replays_per_case": REPLAYS,
        "abs_per_lane_per_replay": STEPS,
        "cases": cases,
    }


def analyze(path, streams):
    if len(set(streams)) != 2:
        raise ValueError("supply two distinct stream IDs from the first probe case")
    failures = []
    for files in exported_files(path):
        if not files:
            continue
        try:
            return summarize(files, streams)
        except (ValueError, ArithmeticError, KeyError) as error:
            failures.append(f"{files[0].name}: {error}")
    raise ValueError(" / ".join(failures) or f"no exported op_summary CSV or msprof timeline in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture", type=Path, help="one saved msprof capture, its export directory, or an exported file"
    )
    parser.add_argument(
        "--streams",
        nargs=2,
        required=True,
        metavar=("FIRST", "SECOND"),
        help="stream_ids from the first case, for a capture reusing this pair in creation order",
    )
    args = parser.parse_args()
    try:
        report = analyze(args.capture, args.streams)
    except (OSError, ValueError) as error:
        print("[PRIORITY_KERNELS]", json.dumps({"error": str(error)}))
        return 1
    print("[PRIORITY_KERNELS]", json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
