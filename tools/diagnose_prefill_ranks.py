#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read existing profiles and Linux CPU/NUMA placement for a late TP rank.

python tools/diagnose_prefill_ranks.py --run 8fdedb5f64aa4ef08bf531fb6e439a35 \
    --output /tmp/rank2-diagnostic.json

Reads the same capture registry as summarize_prefill_profile.py. It does not
import torch, launch a model, allocate device memory or change CPU affinity.
Current /proc observations are labelled separately from historical profiles.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import suppress
from pathlib import Path

import regex as re
import summarize_prefill_profile as summary


def event_row(event, origin):
    args = event.get("args") or {}
    start = summary.number(event.get("ts"))
    duration = summary.number(event.get("dur"))
    return {
        "name": event.get("name"),
        "stream": args.get("Physic Stream Id"),
        "task_id": args.get("Task Id"),
        "connection_id": args.get("connection_id"),
        "start_ms": (start - origin) / 1000,
        "end_ms": (start + duration - origin) / 1000,
        "duration_ms": duration / 1000,
    }


def union_duration(intervals):
    end = float("-inf")
    total = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(end, start))
        end = max(end, stop)
    return total


def profile_pid(path):
    match = re.search(r"_rank(\d+)_(\d+)_", str(path))
    return int(match[2]) if match else None


def trace_report(path, alignment):
    origin = alignment["first_start_us"]
    end = origin + max(row["start_offset_ms"] for row in alignment["ranks"]) * 1000
    window = (origin - 250000, end + 20000)
    streams = defaultdict(list)
    host_copies = []
    connection_ids = set()
    for event in summary.complete_events(path):
        if event.get("ph") != "X":
            continue
        start = summary.number(event.get("ts"))
        stop = start + summary.number(event.get("dur"))
        if stop < window[0] or start > window[1]:
            continue
        args = event.get("args") or {}
        if "Physic Stream Id" in args:
            row = event_row(event, origin)
            row["task_type"] = args.get("Task Type", "")
            streams[row["stream"]].append(row)
            if row["connection_id"] is not None:
                connection_ids.add(str(row["connection_id"]))
        if event.get("name") == "AscendCL@aclrtMemcpyAsync":
            host_copies.append(event_row(event, origin))
    hosts = defaultdict(list)
    for event in summary.complete_events(path):
        args = event.get("args") or {}
        connection = str(args.get("connection_id"))
        if (
            event.get("ph") == "X"
            and connection in connection_ids
            and "level" in args
            and "Physic Stream Id" not in args
        ):
            hosts[connection].append(event_row(event, origin))
    origin_window = ((window[0] - origin) / 1000, (window[1] - origin) / 1000)
    all_tasks = [row for rows in streams.values() for row in rows]
    copies_by_stream = []
    waits = []
    experts = []
    for stream, tasks in sorted(streams.items()):
        tasks.sort(key=lambda row: (row["start_ms"], row["task_id"] or 0))
        copies = [row for row in tasks if row["name"] == "MEMCPY_ASYNC"]
        for row in tasks:
            if row["name"] == "EVENT_WAIT":
                waits.append(row)
            if "groupedmatmul" in row["name"].lower() or "grouped_matmul" in row["name"].lower():
                experts.append(row)
        gaps = []
        for before, after in zip(copies, copies[1:]):
            gap_start, gap_end = before["end_ms"], after["start_ms"]
            if gap_end - gap_start < 1:
                continue
            activity = defaultdict(list)
            for row in all_tasks:
                clipped = (max(gap_start, row["start_ms"]), min(gap_end, row["end_ms"]))
                if clipped[1] <= clipped[0] or row["task_type"] in ("EVENT_WAIT", "EVENT_RECORD"):
                    continue
                category = "other_dma" if row["name"] == "MEMCPY_ASYNC" else "compute_or_communication"
                activity[category].append(clipped)
            controls = [
                row
                for row in tasks
                if row["name"] in ("EVENT_WAIT", "EVENT_RECORD")
                and row["start_ms"] < gap_end
                and row["end_ms"] > gap_start
            ]
            gaps.append(
                {
                    "duration_ms": gap_end - gap_start,
                    "before": before,
                    "after": after,
                    "before_host_calls": hosts.get(str(before["connection_id"]), []),
                    "after_host_calls": hosts.get(str(after["connection_id"]), []),
                    "same_stream_control_tasks": controls,
                    "other_activity_ms": {key: union_duration(value) for key, value in activity.items()},
                }
            )
        if copies:
            intervals = [
                (max(row["start_ms"], origin_window[0]), min(row["end_ms"], origin_window[1])) for row in copies
            ]
            copies_by_stream.append(
                {
                    "stream": stream,
                    "copy_count_in_window": len(copies),
                    "copy_active_ms": union_duration(intervals),
                    "first_copy": copies[0],
                    "last_copy": copies[-1],
                    "largest_gaps": sorted(gaps, key=lambda gap: gap["duration_ms"], reverse=True)[:3],
                }
            )
    return {
        "rank": summary.profile_rank(path),
        "profile_pid": profile_pid(path),
        "trace": str(path.resolve()),
        "window_start_offset_ms": origin_window[0],
        "window_end_offset_ms": origin_window[1],
        "copy_streams": copies_by_stream,
        "longest_host_copy_calls": sorted(host_copies, key=lambda row: row["duration_ms"], reverse=True)[:5],
        "longest_device_waits": sorted(waits, key=lambda row: row["duration_ms"], reverse=True)[:5],
        "expert_kernels": experts,
    }


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=10)
        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout[:100000],
            "stderr": result.stderr[:2000],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def status_fields(path):
    allowed = {
        "Name",
        "Pid",
        "PPid",
        "Tgid",
        "NSpid",
        "Threads",
        "Cpus_allowed_list",
        "Mems_allowed_list",
        "VmRSS",
        "VmLck",
    }
    return {
        key: value.strip()
        for line in path.read_text().splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
        if key in allowed
    }


def process_snapshot(pid, proc_root=Path("/proc")):
    root = proc_root / str(pid)
    try:
        status = status_fields(root / "status")
    except OSError as error:
        return {"pid": pid, "status": "unavailable", "error": str(error)}
    result = {"pid": pid, "status": "current_observation", "process": status}
    with suppress(OSError):
        result["command"] = [
            field.decode(errors="replace") for field in (root / "cmdline").read_bytes().split(b"\0") if field
        ]
    try:
        wanted = {"RANK", "LOCAL_RANK", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "CPU_AFFINITY_CONF"}
        environment = {}
        for field in (root / "environ").read_bytes().split(b"\0"):
            key, separator, value = field.partition(b"=")
            name = key.decode(errors="replace")
            if separator and name in wanted:
                environment[name] = value.decode(errors="replace")
        result["device_environment"] = environment
    except OSError:
        pass
    threads = []
    for task in sorted((root / "task").glob("*")):
        try:
            fields = status_fields(task / "status")
            threads.append({key: fields.get(key) for key in ("Name", "Pid", "Cpus_allowed_list")})
        except OSError:
            continue
    result["threads"] = threads
    try:
        pages = Counter()
        mappings = []
        for line in (root / "numa_maps").read_text().splitlines():
            nodes = {int(node): int(count) for node, count in re.findall(r"\bN(\d+)=(\d+)\b", line)}
            size = re.search(r"kernelpagesize_kB=(\d+)", line)
            page_kib = int(size[1]) if size else 4
            kib = {node: count * page_kib for node, count in nodes.items()}
            pages.update(kib)
            if kib:
                mappings.append(
                    {
                        "address": line.split()[0],
                        "policy": line.split()[1],
                        "size_MiB": sum(kib.values()) / 1024,
                        "numa_MiB": {str(node): value / 1024 for node, value in kib.items()},
                    }
                )
        result["memory_numa_MiB"] = {str(node): value / 1024 for node, value in pages.items()}
        result["largest_mappings"] = sorted(mappings, key=lambda row: row["size_MiB"], reverse=True)[:8]
    except OSError as error:
        result["memory_error"] = str(error)
    return result


def machine_snapshot(profile_pids):
    commands = [
        ["npu-smi", "info"],
        ["npu-smi", "info", "-m"],
        ["npu-smi", "info", "-t", "topo"],
        ["lscpu", "-e=CPU,NODE"],
    ]
    outputs = [command_output(command) for command in commands]
    pids = set(profile_pids)
    # Also inspect current worker processes when the profiled workers have exited.
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes()
            if b"VLLM::Worker" in command or b"Worker_TP" in command or b"Worker_DP" in command:
                pids.add(int(path.parent.name))
        except OSError:
            continue
    interrupts = []
    try:
        for line in Path("/proc/interrupts").read_text().splitlines():
            if not re.search(r"sq_send|cq_update|devdrv|ascend|davinci|npu", line, re.IGNORECASE):
                continue
            row = {"counters": line.strip()}
            match = re.match(r"\s*(\d+):", line)
            if match:
                affinity = Path("/proc/irq") / match[1] / "smp_affinity_list"
                with suppress(OSError):
                    row["affinity"] = affinity.read_text().strip()
            interrupts.append(row)
    except OSError:
        pass
    return {
        "observed_at_unix": time.time(),
        "commands": outputs,
        "profile_pids": sorted(profile_pids),
        "npu_interrupts": interrupts,
        "processes": [process_snapshot(pid) for pid in sorted(pids)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Existing profile run ID or capture number")
    parser.add_argument("--profile-root", type=Path, default=summary.DEFAULT_PROFILE_DIRECTORY)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("rank2-diagnostic.json"))
    args = parser.parse_args(argv)
    runs = summary.discover_runs(args.profile_root, args.runs_dir)
    _, selected = summary.select_run(runs, args.run)
    files = [path for root in selected["paths"] for path in root.rglob("communication.json")]
    if len({summary.profile_rank(path) for path in files}) != len(files) or len(files) < 2:
        raise ValueError("Select a complete multi-rank run with one communication.json per rank")
    communications = [summary.communication_summary(path, 100000) for path in files]
    alignments = summary.collective_alignment(communications, 3)
    if not alignments or any(row["matched_profiles"] != len(files) for row in alignments):
        raise ValueError("The selected capture lacks matching collectives on every rank")
    profile_pids = set()
    for path in files:
        pid = profile_pid(path)
        if pid is not None:
            profile_pids.add(pid)
    current_machine = machine_snapshot(profile_pids)
    reports = []
    for path in files:
        trace = path.with_name("trace_view.json")
        print(f"Reading rank {summary.profile_rank(path)} from {trace}", file=sys.stderr, flush=True)
        reports.append(trace_report(trace, alignments[0]))
    report = {
        "run_id": selected["run_id"],
        "collective_alignment": alignments,
        "ranks": reports,
        "current_machine": current_machine,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
