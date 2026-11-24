#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise device-side AIV admission in one captured two-stream graph per case."""

import argparse
import ctypes as C
import faulthandler
import statistics
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from check_decode_graph_priority import D2H, EVENT_SYNC, I32, U32, Ptr, Ref, Runtime
from check_stream_priority import LINE, TICK_NS, prepare_kernel

CAPACITY, SHORTS, READY_AFTER = 40, 60, 5
SHORT_TICKS, LONG_TICKS = 100_000, 2_000_000  # 2 and 40 ms on 910B3.
RAW, FOREGROUND, PIECE = range(3)
TRAIN_CASES = ("raw_train", "open_train", "long_high", "train_high")
SHARED_CASES = ("raw_28_12", "allow_12", "defer_13")
CONTROLS = ("raw_train", "raw_28_12")
STRIDE = CAPACITY * LINE.size


class Gate(C.Structure):
    _fields_ = [("word", Ptr), ("capacity", U32)]


class Batch(C.Structure):
    _fields_ = [
        ("kernel", Ptr),
        ("stamps", Ptr),
        ("ticks", C.c_uint64),
        ("ready", Ptr),
        ("blocks", U32),
        ("count", U32),
        ("signal_after", U32),
    ]


class Probe:
    def __init__(self, build, kernel, word, stamps):
        self.kernel, self.gate, self.stamps = kernel, Gate(word, CAPACITY), stamps
        self.lib = C.CDLL(str(build / "libstream_priority_host.so"), mode=C.RTLD_GLOBAL)
        self.lib.aiv_probe_initialize.argtypes = [Ref(Gate), Ptr]
        self.lib.aiv_probe_initialize.restype = I32
        self.lib.aiv_probe_pulses.argtypes = [Ref(Gate), Ptr, I32, U32, Ref(Batch)]
        self.lib.aiv_probe_pulses.restype = I32
        self.lib.aiv_probe_kernel_width.argtypes = [Ptr, U32]
        self.lib.aiv_probe_kernel_width.restype = U32

    def initialize(self, stream):
        rc = self.lib.aiv_probe_initialize(C.byref(self.gate), stream)
        if rc:
            raise RuntimeError(f"capture gate initialization rc={rc}")

    def pulse(self, stream, slot, blocks, ticks, *, mode=RAW, charge=None, count=1, ready=None, signal_after=0):
        batch = Batch(self.kernel, self.stamps + slot * STRIDE, ticks, ready, blocks, count, signal_after)
        rc = self.lib.aiv_probe_pulses(
            C.byref(self.gate), stream, mode, blocks if charge is None else charge, C.byref(batch)
        )
        if rc:
            raise RuntimeError(f"capture pulse slot={slot}, mode={mode}, blocks={blocks}, count={count}, rc={rc}")


def record(case, probe, rt, root, side, ready, join):
    if case not in CONTROLS:
        probe.initialize(root)
    if case in TRAIN_CASES:
        if case in ("open_train", "long_high"):
            for slot in range(SHORTS):
                probe.pulse(root, slot, CAPACITY, SHORT_TICKS, mode=PIECE)
                if slot + 1 == READY_AFTER:
                    rt.call("aclrtRecordEvent", ready, root)
        else:
            mode = FOREGROUND if case == "train_high" else RAW
            probe.pulse(root, 0, CAPACITY, SHORT_TICKS, mode=mode, count=SHORTS, ready=ready, signal_after=READY_AFTER)
        rt.call("aclrtStreamWaitEvent", side, ready)
        mode = {"long_high": FOREGROUND, "train_high": PIECE}.get(case, RAW)
        probe.pulse(side, SHORTS, CAPACITY, LONG_TICKS, mode=mode)
    else:
        # Signal from inside the protected batch after its allowance is visible.
        # The same physical 28+12 launch fits in all three cases, including defer_13.
        gated = case != "raw_28_12"
        probe.pulse(root, 0, 28, LONG_TICKS, mode=FOREGROUND if gated else RAW, ready=ready)
        rt.call("aclrtStreamWaitEvent", side, ready)
        probe.pulse(side, 1, 12, LONG_TICKS, mode=PIECE if gated else RAW, charge=13 if case == "defer_13" else 12)
    rt.call("aclrtRecordEvent", join, side)
    rt.call("aclrtStreamWaitEvent", root, join)


def measured_span(raw, slot, blocks, hold):
    start, stop = slot * STRIDE, slot * STRIDE + blocks * LINE.size
    stamps = list(LINE.iter_unpack(raw[start:stop]))
    if len(stamps) != blocks or any(a == 0 or b - a < hold for a, b in stamps):
        raise ValueError(f"slot {slot} missing or invalid stamps for {blocks} AIVs")
    starts, ends = zip(*stamps)
    if min(ends) <= max(starts):
        raise ValueError(f"slot {slot} lacks a common busy interval across all {blocks} AIVs")
    return min(starts), max(ends), max(starts), min(ends)


def assess(case, raw):
    failures = []
    if case in TRAIN_CASES:
        shorts = [measured_span(raw, i, CAPACITY, SHORT_TICKS) for i in range(SHORTS)]
        start, end, _, _ = measured_span(raw, SHORTS, CAPACITY, LONG_TICKS)
        if any(a[1] > b[0] for a, b in zip(shorts, shorts[1:])) or any(a < end and start < b for a, b, _, _ in shorts):
            failures.append("40-AIV tasks executed out of order or overlapped")
        done = sum(b <= start for _, b, _, _ in shorts)
        latency = start - shorts[READY_AFTER - 1][1]
        if latency < 0:
            failures.append("long started before the ready edge")
        if case == "long_high" and done not in (READY_AFTER, READY_AFTER + 1):
            failures.append(f"long took over after {done} shorts (target {READY_AFTER} or {READY_AFTER + 1})")
        if case == "long_high" and latency > SHORT_TICKS * 1.25:
            failures.append("ready-to-long handoff exceeded 2.5 ms")
        if case == "train_high" and done != SHORTS:
            failures.append(f"long started before the protected train finished ({done}/{SHORTS})")
        return {
            "shorts_before_long": done,
            "long_start_ms": (start - shorts[0][0]) * TICK_NS / 1e6,
            "ready_to_long_ms": latency * TICK_NS / 1e6,
        }, failures

    main = measured_span(raw, 0, 28, LONG_TICKS)
    side = measured_span(raw, 1, 12, LONG_TICKS)
    shared = max(0, min(main[3], side[3]) - max(main[2], side[2]))
    if case in ("raw_28_12", "allow_12") and not shared:
        failures.append("no measured simultaneous execution of all 28+12 AIVs")
    if case == "defer_13" and side[0] < main[1]:
        failures.append("the 13-credit piece started before the foreground ended")
    return {"all_40_overlap_ms": shared * TICK_NS / 1e6}, failures


def kernel_widths(rt, probe, graph):
    widths = []
    streams = rt.streams(graph)
    if len(streams) < 2:
        raise RuntimeError(f"capture contains {len(streams)} streams, expected both branches")
    for stream in streams:
        count = U32()
        rt.call("aclmdlRIGetTasksByStream", stream, None, C.byref(count))
        if not count.value:
            continue
        tasks = (Ptr * count.value)()
        rt.call("aclmdlRIGetTasksByStream", stream, tasks, C.byref(count))
        for task in tasks[: count.value]:
            kind = I32()
            rt.call("aclmdlRITaskGetType", task, C.byref(kind))
            if kind.value == 1:  # ACL_MODEL_RI_TASK_KERNEL
                widths.append(probe.lib.aiv_probe_kernel_width(task, CAPACITY))
    return sorted(widths)


def allocate(rt, stack, create, destroy, *args):
    handle = rt.new(create, *args)
    stack.callback(rt.call, destroy, handle)
    return handle


def capture(rt, stack, case, probe, root, side, ready, join):
    graph = Ptr()
    rt.call("aclmdlRICaptureBegin", root, 0)
    try:
        record(case, probe, rt, root, side, ready, join)
    except BaseException:
        # End the capture before unwinding its streams. Preserve the first error.
        if rt.functions["aclmdlRICaptureEnd"](root, C.byref(graph)) == 0 and graph.value:
            stack.callback(rt.call, "aclmdlRIDestroy", graph.value)
        raise
    rt.call("aclmdlRICaptureEnd", root, C.byref(graph))
    if not graph.value:
        raise RuntimeError("capture returned an empty graph")
    stack.callback(rt.call, "aclmdlRIDestroy", graph.value)
    return graph.value


def run_case(rt, probe, host, case, replays):
    failures, rows = [], []
    with ExitStack() as stack:
        root, side = (allocate(rt, stack, "aclrtCreateStreamWithConfig", "aclrtDestroyStream", 0, 0) for _ in range(2))
        ready, join = (
            allocate(rt, stack, "aclrtCreateEventExWithFlag", "aclrtDestroyEvent", EVENT_SYNC) for _ in range(2)
        )
        graph = capture(rt, stack, case, probe, root, side, ready, join)
        if case == "allow_12":
            try:
                widths = kernel_widths(rt, probe, graph)
                status = "verified" if widths == [12, 28] else "use_explicit_widths"
                print(f"[AIV_GATE] captured_kernel_widths={widths} {status}", flush=True)
            except RuntimeError as exc:
                # The workload uses explicit block counts; check metadata independently of scheduling.
                print(f"[AIV_GATE] captured_kernel_widths=unavailable use_explicit_widths ({exc})", flush=True)
        size = (SHORTS + 1 if case in TRAIN_CASES else 2) * STRIDE
        for n in range(replays):
            # Poison both outputs and the admission word before every execution.
            # Initialization, ready, permission checks and join execute in the graph.
            rt.call("aclrtMemset", probe.stamps, size, 0, size)
            rt.call("aclrtMemset", probe.gate.word, 8, 0, 8)
            before = time.perf_counter_ns()
            rt.call("aclmdlRIExecuteAsync", graph, root)
            rt.call("aclrtSynchronizeStreamWithTimeout", root, 5000)
            elapsed = (time.perf_counter_ns() - before) / 1e6
            rt.call("aclrtMemcpy", host, size, probe.stamps, size, D2H)
            metrics, bad = assess(case, C.string_at(host, size))
            if case not in CONTROLS:
                rt.call("aclrtMemcpy", host, 8, probe.gate.word, 8, D2H)
                if C.c_uint64.from_address(host).value != CAPACITY:
                    bad.append("foreground allowance was not restored")
            rows.append(dict(metrics, graph_ms=elapsed))
            failures.extend(f"replay {n + 1}: {reason}" for reason in bad)
    return rows, failures


def measure(device, replays):
    rt = Runtime()
    rt.bind(
        {
            "aclrtGetDeviceInfo": [U32, I32, Ref(C.c_int64)],
            "aclrtBinaryLoadFromFile": [C.c_char_p, Ptr, Ref(Ptr)],
            "aclrtBinaryGetFunction": [Ptr, C.c_char_p, Ref(Ptr)],
            "aclrtBinaryUnLoad": [Ptr],
            "aclrtMemset": [Ptr, C.c_size_t, I32, C.c_size_t],
            "aclrtSynchronizeStreamWithTimeout": [Ptr, I32],
            "aclmdlRIGetTasksByStream": [Ptr, Ref(Ptr), Ref(U32)],
            "aclmdlRITaskGetType": [Ptr, Ref(I32)],
        }
    )
    failed, durations = [], {}
    with ExitStack() as outer:
        rt.call("aclInit", None)
        outer.callback(rt.call, "aclFinalize")
        rt.call("aclrtSetDevice", device)
        outer.callback(rt.call, "aclrtResetDevice", device)
        soc = (rt.functions["aclrtGetSocName"]() or b"").decode()
        cores = C.c_int64()
        rt.call("aclrtGetDeviceInfo", device, 201, C.byref(cores))
        if soc != "Ascend910B3" or cores.value != CAPACITY:
            raise RuntimeError(f"this 40-AIV test targets Ascend910B3, detected {soc} / {cores.value} AIV")
        build = Path(outer.enter_context(tempfile.TemporaryDirectory(prefix="aiv-gate-")))
        kernel = prepare_kernel(rt, outer, soc, build, gate_probe=True)
        size = (SHORTS + 1) * STRIDE
        word = allocate(rt, outer, "aclrtMalloc", "aclrtFree", 8, 2)
        stamps = allocate(rt, outer, "aclrtMalloc", "aclrtFree", size, 2)
        host = allocate(rt, outer, "aclrtMallocHost", "aclrtFreeHost", size)
        probe = Probe(build, kernel, word, stamps)
        for case in (*TRAIN_CASES, *SHARED_CASES):
            print(f"[AIV_GATE] {case} capturing", flush=True)
            # Also bounds a stalled capture or graph destructor in this standalone process.
            faulthandler.dump_traceback_later(20 + 6 * replays, exit=True)
            try:
                rows, bad = run_case(rt, probe, host, case, replays)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(f"{case}: {exc}") from exc
            durations[case] = statistics.median(row["graph_ms"] for row in rows)
            values = " ".join(f"{k}={[round(row[k], 3) for row in rows]}" for k in rows[0] if k != "graph_ms")
            print(f"[AIV_GATE] {case} {'FAIL' if bad else 'PASS'} {values} graph_ms={durations[case]:.3f}", flush=True)
            if bad:
                print("[AIV_GATE] " + " / ".join(bad), flush=True)
                failed.append(case)
    faulthandler.cancel_dump_traceback_later()
    print(f"[AIV_GATE] open_train_minus_raw_ms={durations['open_train'] - durations['raw_train']:+.3f} (60 pieces)")
    print(f"[AIV_GATE] defer_13_minus_allow_12_ms={durations['defer_13'] - durations['allow_12']:+.3f}")
    print(f"[AIV_GATE] {'failed: ' + ', '.join(failed) if failed else 'gate_graph_verified'}")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--replays", type=int, default=3)
    args = parser.parse_args()
    if args.replays < 2:
        parser.error("--replays must be at least 2 to check graph reuse")
    try:
        return measure(args.device, args.replays)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[AIV_GATE] ERROR {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    sys.exit(main())
