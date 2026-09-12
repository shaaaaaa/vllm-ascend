# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative CPU-scope audit of MindStudio traces, not KV correctness proof."""

import json
from collections import defaultdict
from pathlib import Path

import regex as re

ROOT = "sfa_full_graph::target_replay"
RETRIEVE = "sfa_cross_layer::lmcache_retrieve"
REPLAY_APIS = {"aclmdlRIExecuteAsync", "aclmdlExecuteAsync", "aclmdlExecuteAsyncV2"}


def intervals(events: list[dict]) -> list[dict]:
    """Handle complete events and paired Chrome B/E events; ignore metadata."""
    result, stacks = [], defaultdict(list)
    for event in events:
        key = (event.get("pid"), event.get("tid"))
        phase = event.get("ph")
        if phase == "X" and "ts" in event and "dur" in event:
            result.append(event)
        elif phase == "B":
            stacks[key].append(event)
        elif phase == "E" and stacks[key]:
            begin = stacks[key].pop()
            result.append({**begin, "ph": "X", "dur": event["ts"] - begin["ts"]})
    return result


def contains(parent: dict, child: dict) -> bool:
    return (
        parent.get("pid") == child.get("pid")
        and parent.get("tid") == child.get("tid")
        and parent["ts"] <= child["ts"]
        and child["ts"] + child["dur"] <= parent["ts"] + parent["dur"]
    )


def audit_events(events: list[dict], mode: str) -> dict:
    spans = intervals(events)
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
        data = json.loads(path.read_text(encoding="utf-8"))
        events = data["traceEvents"] if isinstance(data, dict) else data
        reports[rank] = {"path": str(path), **audit_events(events, mode)}
    if set(reports) != set(range(ranks)):
        raise ValueError(f"Incomplete trace ranks: {sorted(reports)}, expected {list(range(ranks))}")
    statuses = {r["status"] for r in reports.values()}
    status = next(iter(statuses)) if len(statuses) == 1 else "UNVERIFIED"
    if "SPLIT_DETECTED" in statuses:
        status = "SPLIT_DETECTED"
    return {"status": status, "ranks": ranks, "workers": reports}
