# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative CPU-scope audit of MindStudio traces, not KV correctness proof."""

import argparse
import json
from collections import defaultdict
from decimal import Decimal, DecimalException, InvalidOperation
from pathlib import Path

import regex as re

ROOT = "sfa_full_graph::target_replay"
RETRIEVE = "sfa_cross_layer::lmcache_retrieve"
REPLAY_APIS = {"aclmdlRIExecuteAsync", "aclmdlExecuteAsync", "aclmdlExecuteAsyncV2"}


def time_value(event: dict, field: str) -> Decimal:
    """CANN can mix JSON numbers and numeric strings within one trace.

    Decimal also avoids rounding a short interval across a root boundary when
    its absolute timestamp is large. Never replace invalid timing with zero.
    """
    raw = event.get(field)
    try:
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str, Decimal)):
            raise ValueError("not a number or numeric string")
        value = Decimal(str(raw))
        if not value.is_finite() or (field == "dur" and value < 0):
            raise ValueError("nonfinite time or negative duration")
        return value
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Invalid {field}={raw!r} in trace event {event.get('name', event.get('ph'))!r}") from error


def intervals(events: list[dict]) -> list[dict]:
    """Handle complete events and paired Chrome B/E events; ignore metadata."""
    result, stacks = [], defaultdict(list)
    for event in events:
        key = (event.get("pid"), event.get("tid"))
        phase = event.get("ph")
        if phase == "X":
            result.append({**event, "ts": time_value(event, "ts"), "dur": time_value(event, "dur")})
        elif phase == "B":
            stacks[key].append({**event, "ts": time_value(event, "ts")})
        elif phase == "E" and stacks[key]:
            begin = stacks[key].pop()
            duration = time_value(event, "ts") - begin["ts"]
            if duration < 0:
                raise ValueError(f"End precedes begin in trace event {begin.get('name')!r}")
            result.append({**begin, "ph": "X", "dur": duration})
    if any(e.get("name") in {ROOT, RETRIEVE, *REPLAY_APIS} for stack in stacks.values() for e in stack):
        raise ValueError("Unclosed target/retrieval/execute scope in trace")
    return result


def contains(parent: dict, child: dict) -> bool:
    return (
        parent.get("pid") == child.get("pid")
        and parent.get("tid") == child.get("tid")
        and parent["ts"] <= child["ts"]
        # Compare offsets, not large absolute timestamps plus tiny durations.
        and child["ts"] - parent["ts"] <= parent["dur"] - child["dur"]
    )


def audit_events(events: list[dict], mode: str) -> dict:
    try:
        spans = intervals(events)
    except (ValueError, DecimalException) as error:
        # Skipping an invalid event could hide an extra execute/retrieval.
        return {"status": "UNVERIFIED", "reason": str(error)}
    roots = [e for e in spans if e.get("name") == ROOT]
    copies = [e for e in spans if e.get("name") == RETRIEVE]
    calls = [e for e in spans if e.get("name") in REPLAY_APIS]
    result = {"root_scopes": len(roots), "retrieve_scopes": len(copies), "acl_execute_events": len(calls)}
    if mode == "staged":
        result["status"] = "STAGED_TRACE_PRESENT" if copies and calls and not roots else "UNVERIFIED"
        return result
    if not roots:
        return {**result, "status": "UNVERIFIED", "reason": "No target root scopes recorded"}
    if any(e.get("pid") is None or e.get("tid") is None or e["dur"] <= 0 for e in roots):
        return {**result, "status": "UNVERIFIED", "reason": "Missing root thread identity or duration"}
    counts = []
    for root in roots:
        nested = [call for call in calls if contains(root, call)]
        # Some runtimes nest one API wrapper around another. Count outer
        # submissions, not both aliases; disjoint submissions still fail.
        outer = [
            call
            for call in nested
            if not any(
                other is not call and contains(other, call) and (other["ts"] < call["ts"] or other["dur"] > call["dur"])
                for other in nested
            )
        ]
        counts.append(len(outer))
    result["executes_per_root"] = counts
    if any(count > 1 for count in counts) or any(contains(root, copy) for root in roots for copy in copies):
        return {**result, "status": "SPLIT_DETECTED"}
    if any(count == 0 for count in counts):
        # CANN versions may remap runtime pid/tid into separate timeline lanes.
        # Never infer a match merely from overlapping timestamps across lanes.
        return {**result, "status": "UNVERIFIED", "reason": "Missing same-thread ACL events; inspect runtime lanes"}
    return {**result, "status": "ONE_EXECUTE_PER_ROOT", "scope": "CPU submission only; inspect device timeline too"}


def analyse_traces(root: Path, mode: str, ranks: int) -> dict:
    reports = {}
    for path in sorted(root.rglob("trace_view.json")):
        match = re.search(r"(?:^|_)rank(\d+)(?:_|/|\\)", str(path.relative_to(root)))
        if not match:
            raise ValueError(f"Cannot identify rank from trace path: {path}")
        rank = int(match.group(1))
        if rank in reports:
            raise ValueError(f"Duplicate trace for rank {rank}: {path}")
        data = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
        events = data["traceEvents"] if isinstance(data, dict) else data
        reports[rank] = {"path": str(path), **audit_events(events, mode)}
    if set(reports) != set(range(ranks)):
        raise ValueError(f"Incomplete trace ranks: {sorted(reports)}, expected {list(range(ranks))}")
    statuses = {r["status"] for r in reports.values()}
    status = next(iter(statuses)) if len(statuses) == 1 else "UNVERIFIED"
    if "SPLIT_DETECTED" in statuses:
        status = "SPLIT_DETECTED"
    return {"status": status, "ranks": ranks, "workers": reports}


def latest_run(root: Path) -> Path:
    candidates = [
        path
        for path in root.glob("sfa-*")
        if (path / "comparison.json").is_file() and all((path / mode).is_dir() for mode in ("staged", "full"))
    ]
    if not candidates:
        raise ValueError(f"No completed benchmark with staged/full profile directories under {root}")
    return max(candidates, key=lambda path: (path / "comparison.json").stat().st_mtime_ns)


def recheck_run(directory: Path, ranks: int) -> int:
    """CPU-only recovery: don't load a model, recapture or re-export raw data."""
    split = False
    print(f"[SFA_PROFILE] rechecking existing traces: {directory}", flush=True)
    for mode in ("staged", "full"):
        summary = analyse_traces(directory / mode, mode, ranks)
        (directory / f"{mode}-trace-check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[SFA_PROFILE] {mode}: {summary['status']}; {summary['ranks']} rank traces", flush=True)
        for rank, report in summary["workers"].items():
            if report["status"] in ("UNVERIFIED", "SPLIT_DETECTED"):
                print(f"[SFA_PROFILE] {mode} rank={rank}: {report.get('reason', report['status'])}", flush=True)
        split |= summary["status"] == "SPLIT_DETECTED"
    print(f"[SFA_PROFILE] Existing measurements unchanged: {directory / 'comparison.json'}", flush=True)
    return int(split)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recheck existing SFA traces on CPU without rerunning the model")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir", type=Path, help="Existing profile/sfa-... run directory")
    selection.add_argument("--latest", type=Path, help="Select the latest profiled run under this directory")
    parser.add_argument("--ranks", type=int, default=8)
    args = parser.parse_args(argv)
    if args.ranks <= 0:
        parser.error("--ranks must be positive")
    try:
        directory = args.run_dir if args.run_dir is not None else latest_run(args.latest)
        return recheck_run(directory.resolve(), args.ranks)
    except (ValueError, OSError) as error:
        parser.exit(1, f"[SFA_PROFILE] {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
