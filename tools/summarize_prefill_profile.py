#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare collective timings across ranks in existing Ascend profiles.

Interactive: python tools/summarize_prefill_profile.py
Default directory: python tools/summarize_prefill_profile.py 162830_20261109154305160
List captures: python tools/summarize_prefill_profile.py /path/to/profiles --list
Select raw directory: python tools/summarize_prefill_profile.py /path/to/profiles 162830_20261109154305160
Compare ranks without reading traces: python tools/summarize_prefill_profile.py 5
Include detailed trace summaries: python tools/summarize_prefill_profile.py 5 --full
Locate late-rank submission delays: python tools/summarize_prefill_profile.py 5 --locate
PID_timestamp selects exactly one raw directory. Legacy run-number selection is also supported.
By default, reads communication.json only. Full mode also streams trace_view.json with bounded memory.
"""

import argparse
import heapq
import json
import math
import re
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

DEFAULT_PROFILE_DIRECTORY = Path(__file__).resolve().parents[1] / "vllm_profile" / "P_dp0"
TRACE_TIMESTAMP_TOLERANCE_US = 1.0
TRACE_SLICE_MARGIN_US = 100.0


def number(value):
    try:
        result = float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def events(path, chunk_size=1024 * 1024):
    """Iterate complete Chrome trace events without loading the full file."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as stream:
        buffer = ""
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError(f"No traceEvents array found in {path}")
            buffer += chunk
            match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if match:
                position = match.end()
                break
            if buffer.lstrip().startswith("["):
                position = buffer.index("[") + 1
                break
            if len(buffer) > 16 * 1024 * 1024:
                raise ValueError(f"Trace header exceeds 16 MiB in {path}")
        while True:
            while position < len(buffer) and buffer[position] in " \r\n\t,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as error:
                chunk = stream.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Incomplete trace event in {path}") from error
                buffer = buffer[position:] + chunk
                position = 0
                if len(buffer) > 16 * 1024 * 1024:
                    raise ValueError(f"Single trace event exceeds 16 MiB in {path}") from error
                continue
            position = end
            if isinstance(event, dict):
                yield event


def compact(value):
    if isinstance(value, dict):
        return {str(key): compact(item) for key, item in list(value.items())[:24]}
    if isinstance(value, list):
        return [compact(item) for item in value[:12]]
    if isinstance(value, str):
        return value[:240]
    return value


def longest(heap, item, duration, serial, limit):
    entry = (duration, serial, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif duration > heap[0][0]:
        heapq.heapreplace(heap, entry)


def complete_events(path):
    """Normalize paired CPU ranges and complete device events to durations."""
    stacks = {}
    for event in events(path):
        phase = event.get("ph")
        key = (event.get("pid"), event.get("tid"))
        if phase == "B":
            stacks.setdefault(key, []).append(event)
        elif phase == "E":
            stack = stacks.get(key)
            if stack:
                begin = stack.pop()
                duration = number(event.get("ts")) - number(begin.get("ts"))
                if duration >= 0:
                    yield dict(begin, ph="X", dur=duration)
        else:
            yield event


def trace_summary(path, top):
    processes, threads = {}, {}
    counts = {"allreduce": 0, "collective": 0, "gloo": 0, "pad_reduce": 0, "copy": 0}
    selected = {}
    serial = 0
    for event in complete_events(path):
        name = str(event.get("name", ""))
        pid, tid = str(event.get("pid", "")), str(event.get("tid", ""))
        args = event.get("args") or {}
        if event.get("ph") == "M":
            if name == "process_name":
                processes[pid] = args.get("name")
            elif name == "thread_name":
                threads[(pid, tid)] = args.get("name")
            continue
        normalized = name.lower().replace("_", "").replace(" ", "")
        categories = []
        if "allreduce" in normalized:
            categories.append("allreduce")
        if "gloo" in normalized:
            categories.append("gloo")
        elif any(
            marker in normalized
            for marker in ("hcom", "hccl", "allreduce", "allgather", "reducescatter", "alltoall", "broadcast")
        ):
            categories.append("collective")
        if "maybepadandreduce" in normalized:
            categories.append("pad_reduce")
        if "memcpy" in normalized or "hosttodevice" in normalized or "h2d" in normalized:
            categories.append("copy")
        if not categories or event.get("ph") != "X":
            continue
        duration = number(event.get("dur"))
        serial += 1
        item = {
            "name": name,
            "category": event.get("cat"),
            "pid": pid,
            "tid": tid,
            "start_us": number(event.get("ts")),
            "duration_ms": duration / 1000,
            "args": compact(args),
        }
        # Keep CPU submission ranges and device tasks independently.
        for category in categories:
            counts[category] += 1
            heap = selected.setdefault((category, pid), [])
            longest(heap, item, duration, serial, top)
    result = {"file": str(path), "counts": counts}
    for category in counts:
        retained = [entry for (kind, _), heap in selected.items() if kind == category for entry in heap]
        rows = [row for _, _, row in sorted(retained, reverse=True)]
        for row in rows:
            row["process"] = processes.get(row["pid"])
            row["thread"] = threads.get((row["pid"], row["tid"]))
        result[category] = rows
    return result


def communication_summary(path, top):
    """Keep HCCL time breakdowns and link totals, excluding packet histograms."""
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    selected, all_collectives = [], []
    counts = {"allreduce": 0, "collective": 0}
    serial = 0

    def visit(value, location):
        nonlocal serial
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, [*location, str(index)])
            return
        if not isinstance(value, dict):
            return
        info = value.get("Communication Time Info")
        if isinstance(info, dict) and location and not location[-1].lower().startswith("total"):
            duration = max((number(item) for key, item in info.items() if "elap" in key.lower()), default=0)
            links = {
                link: {key: item for key, item in details.items() if "distribution" not in key.lower()}
                for link, details in value.get("Communication Bandwidth Info", {}).items()
                if isinstance(details, dict)
            }
            item = {"operation": "/".join(location), "time": info, "links": links}
            is_allreduce = "allreduce" in item["operation"].lower().replace("_", "")
            serial += 1
            item = compact(item)
            counts["collective"] += 1
            longest(all_collectives, item, duration, serial, top)
            if is_allreduce:
                counts["allreduce"] += 1
                longest(selected, item, duration, serial, top)
            return
        for key, item in value.items():
            visit(item, [*location, str(key)])

    visit(data, [])
    return {
        "file": str(path),
        "counts": counts,
        "top_level_keys": list(data)[:20] if isinstance(data, dict) else [],
        "allreduce": [row for _, _, row in sorted(selected, reverse=True)],
        "collective": [row for _, _, row in sorted(all_collectives, reverse=True)],
    }


def profile_rank(path):
    worker = next((parent.name for parent in path.parents if parent.name.endswith("_ascend_pt")), path.parent.name)
    match = re.search(r"(?:^|_)rank(\d+)(?:_|$)", worker)
    return int(match.group(1)) if match else worker


def collective_alignment(summaries, top):
    """Compare every rank before selecting the slowest collective instances."""
    grouped = {}
    for summary in summaries:
        rank = profile_rank(Path(summary["file"]))
        for operation in summary["collective"]:
            timing = operation["time"]
            if "Start Timestamp(us)" not in timing:
                continue
            start = number(timing["Start Timestamp(us)"])
            duration = number(timing.get("Elapse Time(ms)"))
            grouped.setdefault(operation["operation"], []).append(
                {"rank": rank, "start_us": start, "end_us": start + duration * 1000, "duration_ms": duration}
            )
    result = []
    for operation, ranks in grouped.items():
        ranks.sort(key=lambda row: (row["start_us"], str(row["rank"])))
        first_start = ranks[0]["start_us"]
        last_start = ranks[-1]["start_us"]
        first_end = min(row["end_us"] for row in ranks)
        last_end = max(row["end_us"] for row in ranks)
        result.append(
            {
                "operation": operation,
                "matched_profiles": len(ranks),
                "available_profiles": len(summaries),
                "first_start_us": first_start,
                "start_skew_ms": (last_start - first_start) / 1000,
                "end_skew_ms": (last_end - first_end) / 1000,
                "tail_after_last_start_ms": (last_end - last_start) / 1000,
                "ranks": [
                    {
                        "rank": row["rank"],
                        "start_offset_ms": (row["start_us"] - first_start) / 1000,
                        "end_offset_ms": (row["end_us"] - first_start) / 1000,
                        "duration_ms": row["duration_ms"],
                    }
                    for row in ranks
                ],
            }
        )
    result.sort(key=lambda row: max(rank["duration_ms"] for rank in row["ranks"]), reverse=True)
    return result[:top]


def trace_dependencies(path, connections, trace_slice=None):
    """Read selected Host APIs and optionally export a JSON trace window."""
    launches = {}
    count = 0
    output, start_us, end_us = trace_slice if trace_slice else (None, 0, 0)
    with output.open("w", encoding="utf-8") if output else nullcontext() as stream:
        if stream is not None:
            stream.write('{"traceEvents":[')
        for event in complete_events(path):
            name = str(event.get("name", ""))
            args = event.get("args") or {}
            connection = args.get("connection_id")
            # Device IDs can refer to low-level Runtime APIs. Match all Host
            # families by ID while retaining only the requested connections.
            if (
                event.get("ph") == "X"
                and connection is not None
                and str(connection) in connections
                and ("level" in args or "@" in name)
                and "Physic Stream Id" not in args
                and "task type" not in args
                and "model id" not in args
            ):
                launches.setdefault(str(connection), []).append(event)
            if stream is None:
                continue
            start = number(event.get("ts"))
            end = start + number(event.get("dur"))
            linked = str(connection) in connections or str(event.get("id")) in connections
            if event.get("ph") != "M" and not linked and (end < start_us or start > end_us):
                continue
            if count:
                stream.write(",")
            json.dump(event, stream, ensure_ascii=False, separators=(",", ":"))
            count += 1
        if stream is not None:
            stream.write("]}")
    artifact = None
    if output:
        artifact = {
            "file": str(output.resolve()),
            "start_us": start_us,
            "end_us": end_us,
            "events": count,
            "bytes": output.stat().st_size,
            "includes_linked_context": True,
        }
    return launches, artifact


def collective_delay_details(alignments, communication_files, top, trace_slice_dir=None):
    """Correlate late-rank device calls and Host launches with bounded memory."""
    traces_by_rank = {}
    for path in communication_files:
        rank = profile_rank(path)
        traces_by_rank.setdefault(rank, []).append(path.with_name("trace_view.json"))
    grouped = {}
    results = []
    for alignment in alignments:
        late = max(alignment["ranks"], key=lambda row: row["start_offset_ms"])
        detail = {
            "operation": alignment["operation"],
            "rank": late["rank"],
            "device_start_offset_ms": late["start_offset_ms"],
            "host_launch": None,
            "launch_end_to_device_start_ms": None,
            "long_host_copies": [],
            "long_device_tasks": [],
        }
        results.append(detail)
        candidates = traces_by_rank[late["rank"]]
        if len(candidates) != 1:
            detail["status"] = "multiple_profiles_for_rank"
            continue
        path = candidates[0]
        if not path.is_file():
            detail["status"] = "trace_missing"
            continue
        grouped.setdefault(path, []).append((alignment, detail))
    for path, targets in grouped.items():
        device_calls = {}
        slice_connections = set()
        host_copies = [[] for _ in targets]
        device_tasks = [[] for _ in targets]
        slice_start = targets[0][0]["first_start_us"] - TRACE_SLICE_MARGIN_US
        slice_end = (
            targets[0][0]["first_start_us"] + targets[0][1]["device_start_offset_ms"] * 1000 + TRACE_SLICE_MARGIN_US
        )
        serial = 0
        for event in complete_events(path):
            if event.get("ph") != "X":
                continue
            name = str(event.get("name", ""))
            args = event.get("args") or {}
            start = number(event.get("ts"))
            duration = number(event.get("dur"))
            end = start + duration
            serial += 1
            connection = args.get("connection_id")
            if (
                trace_slice_dir is not None
                and connection is not None
                and ("Physic Stream Id" in args or "model id" in args)
                and start <= slice_end
                and end >= slice_start
            ):
                slice_connections.add(str(connection))
            for index, (alignment, detail) in enumerate(targets):
                first_start = alignment["first_start_us"]
                device_start = first_start + detail["device_start_offset_ms"] * 1000
                trace_name = alignment["operation"].rsplit("/", 1)[-1].split("@", 1)[0]
                if name == trace_name:
                    previous = device_calls.get(index)
                    if previous is None or abs(start - device_start) < abs(number(previous.get("ts")) - device_start):
                        device_calls[index] = event
                if start >= device_start or end <= first_start:
                    continue
                overlap = min(end, device_start) - max(start, first_start)
                item = {
                    "name": name,
                    "thread": str(event.get("tid", "")),
                    "stream": args.get("Physic Stream Id"),
                    "connection_id": connection,
                    "start_offset_ms": (start - first_start) / 1000,
                    "end_offset_ms": (end - first_start) / 1000,
                    "duration_ms": duration / 1000,
                    "overlap_ms": overlap / 1000,
                }
                if name == "AscendCL@aclrtMemcpyAsync":
                    longest(host_copies[index], item, overlap, serial, top)
                if "Physic Stream Id" in args:
                    longest(device_tasks[index], item, overlap, serial, top)
        connections = {str((event.get("args") or {}).get("connection_id")) for event in device_calls.values()}
        connections.update(str(item["connection_id"]) for heap in device_tasks for _, _, item in heap)
        connections.update(slice_connections)
        trace_slice = None
        if trace_slice_dir is not None:
            alignment, detail = targets[0]
            start = alignment["first_start_us"]
            end = start + detail["device_start_offset_ms"] * 1000
            worker = next(
                (parent.name for parent in path.parents if parent.name.endswith("_ascend_pt")), path.parent.name
            )
            output = trace_slice_dir / f"prefill-delay-{worker}.json"
            trace_slice = (output, start - TRACE_SLICE_MARGIN_US, end + TRACE_SLICE_MARGIN_US)
        launches, artifact = trace_dependencies(path, connections, trace_slice)
        if artifact is not None:
            targets[0][1]["trace_slice"] = artifact
        for index, (alignment, detail) in enumerate(targets):
            detail["long_host_copies"] = [row for _, _, row in sorted(host_copies[index], reverse=True)]
            detail["long_device_tasks"] = [row for _, _, row in sorted(device_tasks[index], reverse=True)]
            for task in detail["long_device_tasks"]:
                task["host_api_candidates"] = [
                    {
                        "name": candidate["name"],
                        "start_offset_ms": (number(candidate.get("ts")) - alignment["first_start_us"]) / 1000,
                        "duration_ms": number(candidate.get("dur")) / 1000,
                        "args": compact(candidate.get("args") or {}),
                    }
                    for candidate in launches.get(str(task["connection_id"]), [])
                ]
            device = device_calls.get(index)
            if device is None:
                detail["status"] = "device_call_missing"
                continue
            connection = (device.get("args") or {}).get("connection_id")
            detail["device_connection_id"] = connection
            first_start = alignment["first_start_us"]
            device_start = number(device.get("ts"))
            detail["device_match_error_us"] = device_start - (first_start + detail["device_start_offset_ms"] * 1000)
            if abs(detail["device_match_error_us"]) > TRACE_TIMESTAMP_TOLERANCE_US:
                detail["status"] = "device_timestamp_mismatch"
                continue
            candidates = launches.get(str(connection), []) if connection is not None else []
            if len(candidates) != 1:
                detail["status"] = "host_launch_missing" if not candidates else "host_launch_ambiguous"
                continue
            launch = candidates[0]
            launch_start = number(launch.get("ts"))
            launch_duration = number(launch.get("dur"))
            launch_end = launch_start + launch_duration
            detail["host_launch"] = {
                "name": launch["name"],
                "thread": str(launch.get("tid", "")),
                "connection_id": connection,
                "start_offset_ms": (launch_start - first_start) / 1000,
                "end_offset_ms": (launch_end - first_start) / 1000,
                "duration_ms": launch_duration / 1000,
            }
            detail["launch_end_to_device_start_ms"] = (device_start - launch_end) / 1000
            detail["status"] = "matched"
    return results


def discover_runs(directory, runs_dir=None):
    """Match client manifests to the exact worker captures under a shared root."""
    captures = {path.resolve() for path in directory.rglob("*_ascend_pt") if path.is_dir()}
    if directory.name.endswith("_ascend_pt"):
        captures.add(directory.resolve())
    by_name = {}
    for path in sorted(captures):
        by_name.setdefault(path.name, []).append(path)
    roots = {Path.cwd(), directory, directory.parent, directory.parent.parent}
    if runs_dir is not None:
        roots.add(runs_dir)
    manifests = set()
    for root in roots:
        if (root / "result.json").is_file():
            manifests.add(root / "result.json")
        manifests.update(root.glob("prefill-profile-*/result.json"))
    if runs_dir is not None:
        manifests.update(runs_dir.glob("*/result.json"))
    runs, assigned, seen = [], set(), set()
    for manifest in sorted(manifests):
        try:
            with manifest.open(encoding="utf-8-sig") as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                continue
            references = data.get("raw_profiles") or []
            if not references and (manifest.parent / "raw_profiles.json").is_file():
                references = json.loads((manifest.parent / "raw_profiles.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            print(f"Skipping unreadable capture record {manifest}: {error}", file=sys.stderr)
            continue
        if not isinstance(references, list) or any(not isinstance(path, str) for path in references):
            print(f"Skipping invalid profile paths in {manifest}", file=sys.stderr)
            continue
        paths = set()
        for reference in references:
            path = Path(reference).resolve()
            if path in captures:
                paths.add(path)
            elif len(by_name.get(path.name, [])) == 1:
                # Preserve associations after moving a complete profile directory.
                paths.add(by_name[path.name][0])
        identity = tuple(sorted(paths))
        if not paths or identity in seen:
            continue
        seen.add(identity)
        assigned.update(paths)
        match = re.search(r"prefill-profile-(\d{8}-\d{6})", manifest.parent.name)
        timestamp = (
            datetime.strptime(match.group(1), "%Y%m%d-%H%M%S").timestamp() if match else manifest.stat().st_mtime
        )
        runs.append(
            {
                "run_id": data.get("run_id") or manifest.parent.name,
                "label": manifest.parent.name,
                "timestamp": timestamp,
                "prefix_tokens": data.get("prefix_tokens"),
                "new_prompt_tokens": data.get("new_prompt_tokens"),
                "status": data.get("status"),
                "manifest": str(manifest.resolve()),
                "paths": sorted(paths),
                "expected_profiles": len(set(references)),
            }
        )
    for path in sorted(captures - assigned):
        runs.append(
            {
                "run_id": path.name,
                "label": path.name,
                "timestamp": path.stat().st_mtime,
                "prefix_tokens": None,
                "new_prompt_tokens": None,
                "status": "raw",
                "manifest": None,
                "paths": [path],
                "expected_profiles": 1,
            }
        )
    return sorted(runs, key=lambda run: (run["timestamp"], run["label"]))


def select_run(runs, selector):
    if not selector:
        raise ValueError("Enter a profile identifier")
    if selector.isdecimal():
        index = int(selector) - 1
        if not 0 <= index < len(runs):
            raise ValueError(f"Capture number must be between 1 and {len(runs)}")
        return index + 1, runs[index]
    matches = [
        (index, run)
        for index, run in enumerate(runs, 1)
        if selector == run["label"] or str(run["run_id"]).startswith(selector)
    ]
    if len(matches) != 1:
        raise ValueError(f"Capture ID {selector!r} matched {len(matches)} records; use --list")
    return matches[0]


def is_profile_identifier(value):
    return bool(re.fullmatch(r"\d+_\d{14,20}", value)) or (
        value.endswith("_ascend_pt") and "/" not in value and "\\" not in value
    )


def select_profile_directory(directory, identifier):
    """Select one raw capture by its exact PID_timestamp suffix."""
    candidates = list(directory.rglob("*_ascend_pt"))
    if directory.name.endswith("_ascend_pt"):
        candidates.append(directory)
    matches = {
        path.resolve()
        for path in candidates
        if path.is_dir()
        and (
            path.name == identifier
            or path.name == identifier + "_ascend_pt"
            or path.name.endswith("_" + identifier + "_ascend_pt")
        )
    }
    if len(matches) != 1:
        raise ValueError(f"Profile identifier {identifier!r} matched {len(matches)} directories under {directory}")
    return matches.pop()


def list_runs(runs):
    print("序号\t采集时间\t前缀token\t新增token\tprofile数\t状态\t记录/ID\t原始目录编号")
    for index, run in enumerate(runs, 1):
        timestamp = datetime.fromtimestamp(run["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        prefix = run["prefix_tokens"] if run["prefix_tokens"] is not None else "?"
        new = run["new_prompt_tokens"] if run["new_prompt_tokens"] is not None else "?"
        count = f"{len(run['paths'])}/{run['expected_profiles']}"
        identifiers = []
        for path in run["paths"]:
            match = re.search(r"(\d+_\d{14,20})_ascend_pt$", path.name)
            identifiers.append(match.group(1) if match else path.name)
        print(
            f"{index}\t{timestamp}\t{prefix}\t{new}\t{count}\t{run['status']}\t{run['label']}\t{','.join(identifiers)}"
        )


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "paths", nargs="*", type=Path, help="Profile directories, files, or PID_timestamp in the default directory"
    )
    cli.add_argument("--list", action="store_true", help="List captures under one common profile directory")
    cli.add_argument("--interactive", action="store_true", help="Prompt for a raw directory's PID_timestamp")
    cli.add_argument("--run", help="Select PID_timestamp, a raw directory name, or a legacy run number/ID")
    cli.add_argument("--runs-dir", type=Path, help="Additional directory containing client result directories")
    cli.add_argument(
        "--output", type=Path, help="Summary file; interactive/default-directory selection uses ./profile-summary.json"
    )
    cli.add_argument(
        "--top", type=int, default=3, help="Slowest collective instances, or events of each kind in full mode"
    )
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument(
        "--full",
        action="store_true",
        help="Include detailed per-file communication and trace summaries",
    )
    mode.add_argument(
        "--locate",
        action="store_true",
        help="Locate late-rank delays and export the relevant trace window as JSON",
    )
    args = cli.parse_args(argv)
    if args.top < 1:
        cli.error("--top must be positive")
    default_directory = not args.paths or (
        len(args.paths) == 1
        and (
            str(args.paths[0]).isdecimal() or (is_profile_identifier(str(args.paths[0])) and not args.paths[0].exists())
        )
    )
    if default_directory:
        if args.paths:
            if args.run is not None:
                cli.error("Supply the capture identifier once")
            args.run = str(args.paths[0])
        args.paths = [DEFAULT_PROFILE_DIRECTORY]
        args.interactive = args.interactive or (args.run is None and not args.list)
    if args.interactive and (args.list or args.run is not None):
        cli.error("Use interactive selection, --list, or a capture number separately")
    # Accept DIRECTORY IDENTIFIER alongside --run and legacy paths.
    if len(args.paths) == 2 and args.run is None and (str(args.paths[1]).isdecimal() or not args.paths[1].exists()):
        args.run = str(args.paths.pop())
    if args.list and args.run is not None:
        cli.error("Use --list or a capture identifier separately")
    selection = None
    if args.list or args.interactive or args.run is not None:
        if len(args.paths) != 1 or not args.paths[0].is_dir():
            cli.error("Capture selection requires one existing profile directory")
        if args.interactive:
            print(f"Profile 根目录 {args.paths[0].resolve()}")
            try:
                args.run = input("请输入目录编号（PID_时间戳）：").strip()
            except EOFError:
                cli.error("No profile identifier was entered")
        if args.run is not None and is_profile_identifier(args.run):
            try:
                path = select_profile_directory(args.paths[0].resolve(), args.run)
            except ValueError as error:
                cli.error(str(error))
            selection = {"profile_identifier": args.run, "profiles": [str(path)]}
            args.paths = [path]
        else:
            if args.runs_dir is not None and not args.runs_dir.is_dir():
                cli.error("--runs-dir must be an existing client-results directory")
            try:
                runs = discover_runs(args.paths[0].resolve(), args.runs_dir)
            except (OSError, ValueError) as error:
                cli.error(str(error))
            if not runs:
                cli.error("No *_ascend_pt captures found in this directory")
            if args.list:
                list_runs(runs)
                return
            try:
                index, run = select_run(runs, args.run)
            except ValueError as error:
                cli.error(str(error))
            if len(run["paths"]) != run["expected_profiles"]:
                cli.error(f"Capture {index} has {len(run['paths'])}/{run['expected_profiles']} worker profiles")
            selection = {key: value for key, value in run.items() if key not in {"paths", "timestamp"}}
            selection["number"] = index
            selection["profiles"] = list(map(str, run["paths"]))
            args.paths = run["paths"]
    if args.output is None and (default_directory or args.interactive or args.locate):
        args.output = Path("profile-summary.json")
    files = set()
    for path in args.paths:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            for name in ("communication.json", "trace_view.json") if args.full else ("communication.json",):
                files.update(item.resolve() for item in path.rglob(name))
        else:
            cli.error(f"Path does not exist: {path}")
    if not files:
        cli.error("No communication.json or trace_view.json found" if args.full else "No communication.json found")
    if not args.full:
        communication_files = sorted(path for path in files if path.name == "communication.json")
        if not communication_files:
            cli.error("Rank alignment requires communication.json")
        # Rank-local top-N filtering would discard the late rank's short call.
        summaries = [communication_summary(path, sys.maxsize) for path in communication_files]
        report = {"collective_alignment": collective_alignment(summaries, args.top)}
        if args.locate:
            report["collective_delay_details"] = collective_delay_details(
                report["collective_alignment"],
                communication_files,
                args.top,
                trace_slice_dir=args.output.parent if args.output else Path.cwd(),
            )
    else:
        report = {"communication": [], "traces": []}
        for path in sorted(files):
            if path.name == "communication.json":
                report["communication"].append(communication_summary(path, args.top))
            elif path.name == "trace_view.json":
                report["traces"].append(trace_summary(path, args.top))
            else:
                cli.error(f"Expected communication.json or trace_view.json: {path}")
    if selection is not None:
        report["selection"] = selection
    summary = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(summary)
    else:
        args.output.write_text(summary + "\n", encoding="utf-8")
        print(f"摘要已保存到 {args.output.resolve()}")
    if args.locate:
        for detail in report["collective_delay_details"]:
            if "trace_slice" in detail:
                print(f"诊断片段已保存到 {detail['trace_slice']['file']}")


if __name__ == "__main__":
    main()
