#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile foreground AIV kernels becoming ready behind queued background work.

Both modes use the installed CANN runtime. Explicit mode binds the two streams
to an ACL model. Capture mode exercises the automatic graph path used by vLLM.
"""

import argparse
import csv
import ctypes as C
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from statistics import median

from analyze_decode_layer_gaps import ns
from check_decode_creation_priority import HEAD, NOTIFY_TIMEOUT_S, PERSISTENT, SYNC_TIMEOUT_MS, AbsWork, CreationRuntime
from check_decode_graph_priority import EVENT_SYNC, I32, U32, Ptr, Ref, one
from summarize_creation_priority import is_abs

FOREGROUND = "foreground"
BACKGROUND = "background"
ROLES = (FOREGROUND, BACKGROUND)
ELEMENTS = {FOREGROUND: 2**21, BACKGROUND: 2**23}
LEAD_IN = 4
KERNELS = {FOREGROUND: 12, BACKGROUND: 64}
REPLAYS = 6  # First replay binds the SQs; analyze the following five.


def capture_case(
    rt,
    mode,
    execution,
    work,
    host,
    root,
    priorities,
    *,
    kernels=KERNELS,
    lead_in=LEAD_IN,
    replays=REPLAYS,
    on_bound=None,
):
    other = next(role for role in ROLES if role != root)
    order = (root, other)
    with ExitStack() as stack:
        explicit = mode == "explicit"
        flags = PERSISTENT if explicit else 0
        streams = {
            role: rt.manage(stack, "aclrtCreateStreamWithConfig", "aclrtDestroyStream", priorities[role], flags)
            for role in order
        }
        if explicit:
            gate, join = (rt.manage(stack, "aclrtCreateNotify", "aclrtDestroyNotify", 0) for _ in range(2))
            graph = rt.manage(stack, "aclmdlRIBuildBegin", "aclmdlRIDestroy", 0)
            for role in order:
                rt.call("aclmdlRIBindStream", graph, streams[role], HEAD)
                stack.callback(rt.call, "aclmdlRIUnbindStream", graph, streams[role])

            def signal(handle, sender, receiver):
                rt.call("aclrtRecordNotify", handle, streams[sender])
                rt.call("aclrtWaitAndResetNotify", handle, streams[receiver], NOTIFY_TIMEOUT_S)
        else:
            fork, gate, join = (
                rt.manage(stack, "aclrtCreateEventExWithFlag", "aclrtDestroyEvent", EVENT_SYNC) for _ in range(3)
            )

            def signal(handle, sender, receiver):
                rt.call("aclrtRecordEvent", handle, streams[sender])
                rt.call("aclrtStreamWaitEvent", streams[receiver], handle)

            rt.call("aclmdlRICaptureBegin", streams[root], 0)
            active, state = Ptr(), I32()
            rt.call("aclmdlRICaptureGetInfo", streams[root], C.byref(state), C.byref(active))
            if state.value != 1 or not active.value:
                raise RuntimeError(f"unexpected capture state {state.value}")
            rt.call("aclrtRecordEvent", fork, streams[root])
            root_queue = one(rt.streams(active.value), "capture root before fork")
            rt.call("aclrtStreamWaitEvent", streams[other], fork)

        # A graph dependency makes foreground arrive after background has already started.
        # All foreground kernels after its first are consecutive, with no wait in between.
        for _ in range(lead_in):
            work[BACKGROUND].enqueue(streams[BACKGROUND])
        signal(gate, BACKGROUND, FOREGROUND)
        for role in ROLES:
            count = kernels[role] - (lead_in if role == BACKGROUND else 0)
            for _ in range(count):
                work[role].enqueue(streams[role])
        signal(join, other, root)

        if explicit:
            rt.call("aclmdlRIEndTask", graph, streams[root])
            rt.call("aclmdlRIBuildEnd", graph, None)
            logical = streams
        else:
            result = Ptr()
            rt.call("aclmdlRICaptureEnd", streams[root], C.byref(result))
            graph = result.value
            stack.callback(rt.call, "aclmdlRIDestroy", graph)
            bound = rt.streams(graph)
            if graph != active.value or root_queue not in bound:
                raise RuntimeError("capture root changed at graph finalization")
            logical = {root: root_queue, other: one(bound - {root_queue}, "second capture queue")}

        creation_fields = {}
        for role, queue in logical.items():
            priority = U32()
            rt.call("aclrtStreamGetPriority", queue, C.byref(priority))
            creation_fields[role] = priority.value
        for _ in range(replays):
            for op in work.values():
                rt.call("aclrtMemset", op.output, op.nbytes, 0, op.nbytes)
            rt.call("aclmdlRIExecuteAsync", graph, execution)
            rt.call("aclrtSynchronizeStreamWithTimeout", execution, SYNC_TIMEOUT_MS)
            if on_bound is not None:
                on_bound(logical)
            for op in work.values():
                op.check(host)
        return {
            "root": root,
            "priorities": priorities,
            "logical_ids": {r: rt.stream_id(logical[r]) for r in ROLES},
            "source_ids": {r: rt.stream_id(streams[r]) for r in ROLES},
            "graph_stream_creation_fields": creation_fields,
        }


def collect(device, mode, report):
    rt = CreationRuntime(capture_functions=(mode == "capture"))
    if mode == "explicit":
        rt.bind({"aclrtStreamGetPriority": [Ptr, Ref(U32)]})
    with ExitStack() as stack:
        rt.call("aclInit", None)
        stack.callback(rt.call, "aclFinalize")
        rt.call("aclrtSetDevice", device)
        stack.callback(rt.call, "aclrtResetDevice", device)
        least, greatest = I32(), I32()
        rt.call("aclrtDeviceGetStreamPriorityRange", C.byref(least), C.byref(greatest))
        low, high = least.value, greatest.value
        if low <= high:
            raise RuntimeError(f"expected two distinct priorities, got {high}..{low}")
        execution = rt.manage(stack, "aclrtCreateStream", "aclrtDestroyStream")
        host = rt.manage(stack, "aclrtMallocHost", "aclrtFreeHost", 32)
        work = {r: AbsWork(rt, stack, byte, execution, ELEMENTS[r]) for r, byte in zip(ROLES, (0xBC, 0xC0))}
        cases = []
        for root in ROLES:
            for label, fg, bg in (
                ("equal", high, high),
                ("foreground_high", high, low),
                ("background_high", low, high),
            ):
                case = capture_case(rt, mode, execution, work, host, root, {FOREGROUND: fg, BACKGROUND: bg})
                cases.append(dict(case, case=label))
        report.write_text(json.dumps({"mode": mode, "cases": cases, "warmup_stream": rt.stream_id(execution)}))


def read_rows(path):
    result = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["Task Type"].strip().upper() != "AI_VECTOR_CORE" or not is_abs(r["OP Type"]):
                continue
            start = ns(r["Task Start Time(us)"].strip())
            duration = ns(r["Task Duration(us)"].strip())
            size = r["Input Shapes"].strip().strip('"')
            role = next((r for r in ROLES if str(ELEMENTS[r]) == size), None)
            if role is None or duration <= 0 or int(r["Block Num"]) <= 0:
                raise ValueError(f"unexpected AIV task shape or duration in {path.name}: {size}, {duration}")
            result.append((start, start + duration, role, int(r["Stream ID"])))
    return sorted(result)


def replay_gaps(fg, bg, lead_in=LEAD_IN):
    # A completed foreground predecessor is a device-side readiness marker. At least one
    # background kernel must still be queued. An already started kernel is allowed to finish.
    gaps, ahead = [], []
    for (_, ready), (start, _) in zip(fg, fg[1:]):
        if start < ready or not bg[0][0] < ready < bg[-1][0]:
            continue
        gaps.append(start - ready)
        ahead.append(sum(ready < bg_start < start for bg_start, _ in bg))
    # Initial arrival includes the event/notify handoff. Keep it out of the arbitration verdict.
    gate_latency = fg[0][0] - bg[lead_in - 1][1]
    if gate_latency < 0:
        raise ValueError("foreground started before the background gate")
    return gaps, ahead, gate_latency


def analyze(rows, report, *, kernels=KERNELS, lead_in=LEAD_IN, replays=REPLAYS):
    warmup = report["warmup_stream"]
    if [(r[2], r[3]) for r in rows[:2]] != [(r, warmup) for r in ROLES]:
        raise ValueError("missing or out-of-order eager warm-up tasks")
    cursor, width = 2, sum(kernels.values())
    result = []
    for case in report["cases"]:
        all_gaps, all_ahead, arrivals = [], [], []
        eligible_replays, overtaken_replays, drained_before_foreground = 0, 0, 0
        for replay in range(replays):
            lanes = defaultdict(list)
            for start, end, role, stream in rows[cursor : cursor + width]:
                valid_ids = {case["logical_ids"][role], case["source_ids"][role]}
                if stream not in valid_ids:
                    raise ValueError(f"unexpected profiled stream {stream} for {role}; expected {sorted(valid_ids)}")
                lanes[role].append((start, end))
            cursor += width
            if any(len(lanes[r]) != kernels[r] for r in ROLES):
                raise ValueError(f"incomplete replay for {case['case']}")
            gaps, ahead, latency = replay_gaps(lanes[FOREGROUND], lanes[BACKGROUND], lead_in)
            if replay:
                all_gaps.extend(gaps)
                all_ahead.extend(ahead)
                arrivals.append(latency)
                eligible_replays += bool(gaps)
                overtaken_replays += any(ahead)
                # With the background lane preferred, its entire tail can start before
                # foreground enters. Then there is no contested gap inside the foreground
                # burst. Keep this handoff observation separate from overtakes of ready work.
                gate_end = lanes[BACKGROUND][lead_in - 1][1]
                tail = lanes[BACKGROUND][lead_in:]
                first_fg = lanes[FOREGROUND][0][0]
                if tail[-1][0] < first_fg and sum(gate_end < t < first_fg for t, _ in tail) > len(tail) // 2:
                    drained_before_foreground += 1
        metrics = {
            "eligible_gaps": len(all_gaps),
            "gaps_with_background_ahead": sum(n > 0 for n in all_ahead),
            "eligible_replays": eligible_replays,
            "overtaken_replays": overtaken_replays,
            "background_tail_started_before_foreground_replays": drained_before_foreground,
            "max_background_ahead": max(all_ahead, default=0),
            "ready_to_start_p95_us": sorted(all_gaps)[math.ceil(0.95 * len(all_gaps)) - 1] / 1000 if all_gaps else None,
            "gate_to_first_foreground_median_us": median(arrivals) / 1000,
        }
        result.append(dict(case, **metrics))
    if cursor != len(rows):
        raise ValueError(f"profile contains {len(rows) - cursor} extra AIV tasks")
    return result


def verdict(rows):
    by_order = {root: {r["case"]: r for r in rows if r["root"] == root} for root in ROLES}
    if any(c["foreground_high"]["gaps_with_background_ahead"] for c in by_order.values()):
        return "foreground_overtaken"
    if any(c["foreground_high"]["eligible_replays"] != REPLAYS - 1 for c in by_order.values()):
        return "insufficient_contention"
    if all(c["background_high"]["overtaken_replays"] > (REPLAYS - 1) // 2 for c in by_order.values()):
        return "foreground_preferred_in_both_orders"
    if all(
        c["background_high"]["background_tail_started_before_foreground_replays"] > (REPLAYS - 1) // 2
        for c in by_order.values()
    ):
        return "priority_reversal_delays_foreground_entry_in_both_orders"
    return "no_directional_priority_effect"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--mode", choices=("explicit", "capture"), default="explicit")
    p.add_argument("--output", type=Path, help="new directory for the MindStudio capture")
    p.add_argument("--worker-report", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker_report is not None:
        collect(args.device, args.mode, args.worker_report)
        return 0
    if not shutil.which("msprof"):
        p.error("msprof was not found in the current CANN environment")
    if args.output is None:
        directory = Path(tempfile.mkdtemp(prefix="queue-arbitration-", dir=".")).resolve()
    else:
        directory = args.output.resolve()
        directory.mkdir(parents=True, exist_ok=False)
    report = directory / "cases.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--device",
        str(args.device),
        "--mode",
        args.mode,
        "--worker-report",
        str(report),
    ]
    print(f"[QUEUE_ARBITRATION] MindStudio output: {directory}", flush=True)
    with (directory / "msprof.log").open("w") as log:
        proc = subprocess.run(["msprof", f"--output={directory}", *command], stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode or not report.is_file():
        raise RuntimeError((directory / "msprof.log").read_text(errors="replace")[-3500:])
    files = list(directory.glob("PROF_*/mindstudio_profiler_output/op_summary*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one device op_summary CSV under {directory}, found {len(files)}")
    cases = analyze(read_rows(files[0]), json.loads(report.read_text()))
    status = verdict(cases)
    (directory / "summary.json").write_text(json.dumps({"mode": args.mode, "status": status, "cases": cases}, indent=2))
    for c in cases:
        print(
            f"  {c['root']} first, {c['case']}, creation fields={c['graph_stream_creation_fields']}, "
            f"background ahead in {c['gaps_with_background_ahead']}/{c['eligible_gaps']} ready gaps, "
            f"background tail started before foreground on {c['background_tail_started_before_foreground_replays']} "
            f"replays, p95={c['ready_to_start_p95_us']} us"
        )
    print(f"[QUEUE_ARBITRATION] {status} ({args.mode})", flush=True)
    return 0 if status == "foreground_preferred_in_both_orders" else 2


if __name__ == "__main__":
    sys.exit(main())
