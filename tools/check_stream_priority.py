#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""When does one late long kernel actually start behind 60 already-submitted short kernels?"""

import argparse
import ctypes as C
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, suppress
from functools import partial
from pathlib import Path
from statistics import median

SHORTS, DELAY_UNITS, LONG_UNITS = 60, 5, 20
TICK_NS = 20  # Ascend 910B / Atlas A2 system counter, 50 MHz.
LINE = struct.Struct("=QQ48x")  # Entry and exit ticks, one cache line per AIV block.
U32, I32, Ptr = C.c_uint32, C.c_int32, C.c_void_p


class Runtime:
    def __init__(self):
        libs = [C.CDLL("libascendcl.so", mode=C.RTLD_GLOBAL)]
        with suppress(OSError):
            libs.append(C.CDLL("libacl_rt.so", mode=C.RTLD_GLOBAL))
        self.libs = libs
        self.fn = {}
        for name, args in {
            "aclInit": [C.c_char_p],
            "aclFinalize": [],
            "aclrtSetDevice": [I32],
            "aclrtResetDevice": [I32],
            "aclrtGetSocName": [],
            "aclrtGetDeviceInfo": [U32, I32, C.POINTER(C.c_int64)],
            "aclrtBinaryLoadFromFile": [C.c_char_p, Ptr, C.POINTER(Ptr)],
            "aclrtBinaryGetFunction": [Ptr, C.c_char_p, C.POINTER(Ptr)],
            "aclrtBinaryUnLoad": [Ptr],
            "aclrtDeviceGetStreamPriorityRange": [C.POINTER(I32), C.POINTER(I32)],
            "aclrtCreateStreamWithConfig": [C.POINTER(Ptr), U32, U32],
            "aclrtStreamGetPriority": [Ptr, C.POINTER(U32)],
            "aclrtDestroyStream": [Ptr],
            "aclrtSynchronizeStreamWithTimeout": [Ptr, I32],
            "aclrtMalloc": [C.POINTER(Ptr), C.c_size_t, I32],
            "aclrtFree": [Ptr],
            "aclrtMallocHost": [C.POINTER(Ptr), C.c_size_t],
            "aclrtFreeHost": [Ptr],
            "aclrtMemcpy": [Ptr, C.c_size_t, Ptr, C.c_size_t, I32],
            "aclrtMemset": [Ptr, C.c_size_t, I32, C.c_size_t],
        }.items():
            f = next((getattr(lib, name) for lib in libs if hasattr(lib, name)), None)
            if f is None:
                raise RuntimeError(f"loaded CANN runtime lacks {name}")
            f.argtypes, f.restype = args, C.c_char_p if name == "aclrtGetSocName" else I32
            self.fn[name] = f

    def call(self, name, *args):
        rc = self.fn[name](*args)
        if rc:
            raise RuntimeError(f"{name} rc={rc}")

    def own(self, stack, create, destroy, *args):
        ptr = Ptr()
        self.call(create, C.byref(ptr), *args)
        if not ptr.value:
            raise RuntimeError(f"{create} returned a null handle")
        stack.callback(self.call, destroy, ptr)
        return ptr.value


def toolkit():
    explicit = os.environ.get("ASCENDC_CMAKE_DIR")
    roots = [os.environ.get(k) for k in ("ASCEND_HOME_PATH", "ASCEND_INSTALL_PATH", "ASCEND_CANN_PACKAGE_PATH")]
    roots += [
        "/usr/local/Ascend/cann",
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/cann-9.1.0",
        "/usr/local/Ascend/cann-8.5.1",
    ]
    for location in filter(None, roots):
        base = Path(location)
        paths = [
            base / s
            for s in (
                f"{platform.machine()}-linux/tikcpp/ascendc_kernel_cmake",
                "tikcpp/ascendc_kernel_cmake",
                "tools/tikcpp/ascendc_kernel_cmake",
                "compiler/tikcpp/ascendc_kernel_cmake",
                "ascendc_devkit/tikcpp/samples/cmake",
            )
        ]
        if explicit and Path(explicit).resolve().is_relative_to(base.resolve()):
            paths.insert(0, Path(explicit))
        for directory in paths:
            if (directory / "ascendc.cmake").is_file():
                return base.resolve(), directory.resolve()
    raise RuntimeError("ascendc.cmake was not found under the active CANN toolkit; source its set_env.sh")


def prepare_kernel(rt, stack, soc, build, *, gate_probe=False):
    base, cmake = toolkit()
    source = Path(__file__).with_name("stream_priority_unit")
    label = "AIV_GATE" if gate_probe else "STREAM_PRIORITY"
    print(f"[{label}] building one AIV kernel for {soc}", flush=True)
    for cmd in (
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DSOC_VERSION={soc}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={build}",
            f"-DASCEND_CANN_PACKAGE_PATH={base}",
            f"-DASCENDC_CMAKE_DIR={cmake}",
            *(["-DBUILD_AIV_GATE_PROBE=ON"] if gate_probe else []),
        ],
        ["cmake", "--build", str(build), "--parallel", "2"],
    ):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        if proc.returncode:
            raise RuntimeError(f"building the standalone kernel failed\n{proc.stdout[-4500:]}")
    binary = build / "fatbin" / "stream_priority_pulse" / "stream_priority_pulse.o"
    if not binary.is_file():
        raise RuntimeError(f"CANN build did not produce {binary}")
    handle, kernel = Ptr(), Ptr()
    rt.call("aclrtBinaryLoadFromFile", os.fsencode(binary), None, C.byref(handle))
    stack.callback(rt.call, "aclrtBinaryUnLoad", handle)
    rt.call("aclrtBinaryGetFunction", handle, b"stream_priority_pulse_kernel", C.byref(kernel))
    return kernel


def build_kernel(rt, stack, soc, build):
    kernel = prepare_kernel(rt, stack, soc, build)
    lib = C.CDLL(str(build / "libstream_priority_host.so"), mode=C.RTLD_GLOBAL)
    submit = lib.stream_priority_submit
    submit.argtypes, submit.restype = [Ptr, U32, Ptr, Ptr, U32, C.c_uint64], I32
    return partial(submit, kernel)


def enqueue_train(short_batch, long_task, unit_ns, clock=time.monotonic_ns, sleep=time.sleep):
    t0 = clock()
    short_batch()
    train_queued = clock()
    deadline = t0 + DELAY_UNITS * unit_ns
    arrival = train_queued
    while arrival < deadline:
        sleep((deadline - arrival) / 1e9)
        arrival = clock()
    submitted = clock()
    long_task()
    return t0, train_queued, submitted, clock()


class Work:
    def __init__(self, rt, stack, submit, blocks, unit_ticks):
        self.rt, self.submit, self.blocks, self.ticks = rt, submit, blocks, unit_ticks
        self.unit_ns = unit_ticks * TICK_NS
        self.timeout_ms = max(15000, int(2000 + 2 * (SHORTS + LONG_UNITS) * self.unit_ns / 1e6))
        self.stride = blocks * LINE.size
        self.size = (SHORTS + 1) * self.stride
        self.memory = rt.own(stack, "aclrtMalloc", "aclrtFree", self.size + LINE.size, 2)
        self.host = rt.own(stack, "aclrtMallocHost", "aclrtFreeHost", self.size)
        self.marker = self.memory + self.size

    def launch(self, stream, ptr, blocks, count, ticks):
        rc = self.submit(blocks, stream, ptr, count, ticks)
        if rc:
            raise RuntimeError(f"AIV kernel submission rc={rc}")

    def sync(self, stream):
        self.rt.call("aclrtSynchronizeStreamWithTimeout", stream, self.timeout_ms)

    def read(self, ptr, size):
        self.rt.call("aclrtMemcpy", self.host, self.size, ptr, size, 2)
        return C.string_at(self.host, size)

    def clock_bounds(self, stream):
        # The marker executes between the two host clock reads. Only used outside each trial.
        intervals = []
        for _ in range(3):
            before = time.monotonic_ns()
            self.launch(stream, self.marker, 1, 1, 50)
            self.sync(stream)
            after = time.monotonic_ns()
            start, end = LINE.unpack(self.read(self.marker, LINE.size))
            if not start or end < start + 50:
                raise RuntimeError("device clock calibration kernel did not record its timestamps")
            intervals.append((before - start * TICK_NS, after - end * TICK_NS))
        low, high = max(a for a, _ in intervals), min(b for _, b in intervals)
        if low > high:
            raise RuntimeError("host/device clock brackets disagree; check the 910B system-counter frequency")
        return low, high

    def run(self, long_stream, short_stream):
        # Warm the two user streams before defining the time origin.
        self.launch(long_stream, self.marker, 1, 1, 50)
        self.sync(long_stream)
        before = self.clock_bounds(short_stream)
        self.rt.call("aclrtMemset", self.memory, self.size, 0, self.size)
        host_times = enqueue_train(
            lambda: self.launch(short_stream, self.memory, self.blocks, SHORTS, self.ticks),
            lambda: self.launch(
                long_stream, self.memory + SHORTS * self.stride, self.blocks, 1, LONG_UNITS * self.ticks
            ),
            self.unit_ns,
        )
        self.sync(long_stream)
        self.sync(short_stream)
        after = self.clock_bounds(short_stream)
        offset = min(before[0], after[0]), max(before[1], after[1])
        raw = self.read(self.memory, self.size)
        tasks = []
        for i in range(SHORTS + 1):
            rows = list(LINE.iter_unpack(raw[i * self.stride : (i + 1) * self.stride]))
            duration = self.ticks * (LONG_UNITS if i == SHORTS else 1)
            if any(not a or b < a + duration for a, b in rows):
                raise RuntimeError(f"task {i} has a missing or invalid per-block timestamp")
            tasks.append([(a * TICK_NS, b * TICK_NS) for a, b in rows])
        return timing(tasks, host_times, offset, self.unit_ns)


def timing(tasks, host_times, offset, unit_ns):
    spans, full_width_ns = [], []
    for blocks in tasks:
        starts, ends = zip(*blocks)
        spans.append((min(starts), max(ends)))
        # Intersection of the per-block device intervals, independent of host clock alignment.
        full_width_ns.append(max(0, min(ends) - max(starts)))
    short, (entry, end) = spans[:-1], spans[-1]
    cross_stream_overlaps = sum(a < end and entry < b for a, b in short)
    short_stream_overlaps = sum(prev_end > start for (_, prev_end), (start, _) in zip(short, short[1:]))
    full_width_tasks = sum(ns > 0 for ns in full_width_ns)
    t0, queued, submitted, returned = host_times
    low, high = offset
    start_low, start_high = entry + low, entry + high
    finished = sum(b <= entry for _, b in short)
    pending = sum(a > entry for a, _ in short)
    pending_at_submit = sum(a + low > returned for a, _ in short)
    issues = []
    if queued > t0 + DELAY_UNITS * unit_ns:
        issues.append("short train submission missed the deadline")
    if submitted > t0 + (DELAY_UNITS + 0.5) * unit_ns or returned - submitted > unit_ns:
        issues.append("host delivered the long kernel late")
    if high - low > unit_ns / 2 or start_high < submitted:
        issues.append("host/device clock alignment is too imprecise")
    if min(a for a, _ in short) + high > submitted or pending_at_submit < SHORTS // 2:
        issues.append("long submission did not meet a running short train with a substantial backlog")
    return {
        "long_submitted_ms": (submitted - t0) / 1e6,
        "short_train_submitted_ms": (queued - t0) / 1e6,
        "long_started_ms": ((start_low + start_high) / 2 - t0) / 1e6,
        "long_started_bounds_ms": [(start_low - t0) / 1e6, (start_high - t0) / 1e6],
        "wait_bounds_ms": [max(0, start_low - returned) / 1e6, max(0, start_high - submitted) / 1e6],
        "long_from_first_short_device_ms": (entry - min(a for a, _ in short)) / 1e6,
        "long_device_span_ms": (end - entry) / 1e6,
        "shorts_done_at_long_start": finished,
        "shorts_running_at_long_start": SHORTS - finished - pending,
        "shorts_queued_at_long_start": pending,
        "min_shorts_queued_at_long_submit": pending_at_submit,
        "full_width_tasks": full_width_tasks,
        "min_full_width_us": min(full_width_ns) / 1000,
        "shorts_overlapping_long": cross_stream_overlaps,
        "adjacent_shorts_overlapping": short_stream_overlaps,
        "resource_check_passed": full_width_tasks == SHORTS + 1 and cross_stream_overlaps == short_stream_overlaps == 0,
        "issues": issues,
    }


def verdict(pairs, unit_ms):
    if any(not r["resource_check_passed"] for pair in pairs for r in pair.values()):
        return "inconclusive_full_grid_contention"
    if any(r["issues"] for pair in pairs for r in pair.values()):
        return "inconclusive_arrival_or_timing"
    high = [p["S1_high"] for p in pairs]
    low = [p["S2_high"] for p in pairs]
    # Strict user scenario: hand off within one short kernel (allow 25% timing/dispatch slack)
    # in one direction, drain all 60 in the reverse direction. Check every repetition.
    early = all(r["wait_bounds_ms"][1] <= 1.25 * unit_ms and r["shorts_queued_at_long_start"] for r in high)
    drained = all(r["shorts_done_at_long_start"] == SHORTS for r in low)
    if early and drained:
        return "strict_priority_observed" if len(pairs) >= 2 else "strict_pattern_one_pair"
    separated = max(r["wait_bounds_ms"][1] for r in high) + unit_ms < min(r["wait_bounds_ms"][0] for r in low)
    if separated and max(r["shorts_done_at_long_start"] for r in high) < min(
        r["shorts_done_at_long_start"] for r in low
    ):
        return "directional_priority_observed" if len(pairs) >= 2 else "directional_pattern_one_pair"
    if drained and all(r["shorts_done_at_long_start"] == SHORTS for r in high):
        return "both_started_after_short_train"
    if all(r["shorts_queued_at_long_start"] for r in high + low):
        return "both_started_with_shorts_still_queued"
    return "no_repeatable_directional_start_difference"


def measure(args):
    rt = Runtime()
    with ExitStack() as stack:
        rt.call("aclInit", None)
        stack.callback(rt.call, "aclFinalize")
        rt.call("aclrtSetDevice", args.device)
        stack.callback(rt.call, "aclrtResetDevice", args.device)
        soc = (rt.fn["aclrtGetSocName"]() or b"").decode()
        if not soc.startswith("Ascend910B"):
            raise RuntimeError(f"this probe uses the 50 MHz Atlas A2 counter; current device is {soc}")
        blocks, least, greatest = C.c_int64(), I32(), I32()
        rt.call("aclrtGetDeviceInfo", args.device, 201, C.byref(blocks))  # ACL_DEV_ATTR_VECTOR_CORE_NUM
        rt.call("aclrtDeviceGetStreamPriorityRange", C.byref(least), C.byref(greatest))
        if blocks.value < 1 or least.value <= greatest.value:
            raise RuntimeError(f"invalid AIV count or priority range: {blocks.value}, {greatest.value}..{least.value}")
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="stream-priority-"))
        ticks = round(args.unit_ms * 1e6 / TICK_NS)
        work = Work(rt, stack, build_kernel(rt, stack, soc, Path(tmp)), blocks.value, ticks)
        unit_ms = work.unit_ns / 1e6
        print(
            f"[STREAM_PRIORITY] {soc}, {blocks.value} AIV, batch mode, high={greatest.value} low={least.value}; "
            f"S2 queues {SHORTS} x {unit_ms:g} ms at t=0; S1 submits 1 x {LONG_UNITS * unit_ms:g} ms "
            f"at {DELAY_UNITS * unit_ms:g} ms",
            flush=True,
        )
        pairs = []
        for n in range(args.repeat):
            pair = {}
            roles = ("S1", "S2") if n % 2 == 0 else ("S2", "S1")
            high_roles = ("S1", "S2") if n % 4 in (0, 3) else ("S2", "S1")
            for high_role in high_roles:
                streams = {}
                with ExitStack() as owned:
                    for role in roles:
                        prio = greatest.value if role == high_role else least.value
                        s = rt.own(owned, "aclrtCreateStreamWithConfig", "aclrtDestroyStream", prio, 0)
                        actual = U32()
                        rt.call("aclrtStreamGetPriority", s, C.byref(actual))
                        if actual.value != prio:
                            raise RuntimeError(f"{role}: creation priority {actual.value}, requested {prio}")
                        streams[role] = s
                    pair[high_role + "_high"] = dict(work.run(streams["S1"], streams["S2"]), created_first=roles[0])
            pairs.append(pair)
        status = verdict(pairs, unit_ms)
        trials = [r for pair in pairs for r in pair.values()]
        resource_check = "pass" if all(r["resource_check_passed"] for r in trials) else "failed"
        print(
            f"[STREAM_PRIORITY] resource_check={resource_check}, "
            f"all {blocks.value} AIV active together in at least {min(r['full_width_tasks'] for r in trials)}"
            f"/{SHORTS + 1} tasks per trial "
            f"(shortest shared interval {min(r['min_full_width_us'] for r in trials):.2f} us), "
            f"max shorts overlapping S1={max(r['shorts_overlapping_long'] for r in trials)}, "
            f"max adjacent S2 overlaps={max(r['adjacent_shorts_overlapping'] for r in trials)}"
        )
        for label in ("S1_high", "S2_high"):
            rows = [p[label] for p in pairs]
            starts = [r["long_started_ms"] for r in rows]
            uncertainty = max((b - a) / 2 for a, b in (r["long_started_bounds_ms"] for r in rows))
            print(
                f"[STREAM_PRIORITY] {label}: S1 submitted {median(r['long_submitted_ms'] for r in rows):.2f} ms, "
                f"S1 started {median(starts):.2f} ms (trials {[round(t, 2) for t in starts]}, "
                f"clock +/-{uncertainty:.3f} ms), "
                f"shorts done {[r['shorts_done_at_long_start'] for r in rows]}/{SHORTS}, "
                f"still queued {[r['shorts_queued_at_long_start'] for r in rows]}"
            )
        problems = sorted({reason for pair in pairs for r in pair.values() for reason in r["issues"]})
        if problems:
            print("[STREAM_PRIORITY] " + " / ".join(problems))
        print(f"[STREAM_PRIORITY] {status}")
        if args.json:
            args.json.write_text(
                json.dumps(
                    dict(
                        status=status,
                        resource_check=resource_check,
                        scheduling_mode="batch",
                        soc=soc,
                        aiv=blocks.value,
                        unit_ms=unit_ms,
                        high=greatest.value,
                        low=least.value,
                        pairs=pairs,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
        return 0 if status in {"strict_priority_observed", "directional_priority_observed"} else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--unit-ms", type=float, default=2, help="duration of one short kernel in milliseconds")
    p.add_argument("--repeat", type=int, default=4, help="matched priority swaps (default: 4)")
    p.add_argument("--json", type=Path, help="optional per-trial timing report")
    args = p.parse_args()
    if not math.isfinite(args.unit_ms) or not 0.001 <= args.unit_ms <= 1000 or args.repeat < 1:
        p.error("use --unit-ms in [0.001, 1000] and --repeat >= 1")
    return measure(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"[STREAM_PRIORITY] error: {exc}")
