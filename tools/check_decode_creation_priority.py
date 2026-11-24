#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure AIV scheduling on two explicitly bound queues in a replayed ACL model.

Run in the CANN 9.1 container with an idle NPU. Requires Python, CANN and the
910B math ops package. The serving NPUGraph uses a separate capture path.
"""

import argparse
import ctypes as C
import json
import sys
from contextlib import ExitStack, suppress
from statistics import median

from check_decode_graph_priority import D2H, I32, U32, Ptr, Ref, Runtime, Size

U64 = C.c_uint64
I64 = C.c_int64
PERSISTENT = 4
TIME_LINE = 8
HEAD = 0
FLOAT16 = 1
FORMAT_ND = 2
ELEMENTS = 2**21
STEPS = 48
REPLAYS = 5
SYNC_TIMEOUT_MS = 15000
NOTIFY_TIMEOUT_S = 15


class CreationRuntime(Runtime):
    def __init__(self, *, capture_functions=False):
        super().__init__(capture_functions=capture_functions)
        for library in ("libnnopbase.so", "libopapi_math.so", "libopapi.so"):
            with suppress(OSError):
                self.libs.append(C.CDLL(library, mode=C.RTLD_GLOBAL))
        self.bind(
            {
                "aclrtMemset": [Ptr, Size, I32, Size],
                "aclrtCreateNotify": [Ref(Ptr), U64],
                "aclrtRecordNotify": [Ptr, Ptr],
                "aclrtWaitAndResetNotify": [Ptr, Ptr, U32],
                "aclrtDestroyNotify": [Ptr],
                "aclrtEventGetTimestamp": [Ptr, Ref(U64)],
                "aclrtSynchronizeStreamWithTimeout": [Ptr, I32],
                "aclmdlRIBuildBegin": [Ref(Ptr), U32],
                "aclmdlRIBindStream": [Ptr, Ptr, U32],
                "aclmdlRIEndTask": [Ptr, Ptr],
                "aclmdlRIBuildEnd": [Ptr, Ptr],
                "aclmdlRIUnbindStream": [Ptr, Ptr],
                "aclmdlRIGetId": [Ptr, Ref(U32)],
                "aclmdlRISetName": [Ptr, C.c_char_p],
                "aclCreateTensor": [Ref(I64), U64, I32, Ref(I64), I64, I32, Ref(I64), U64, Ptr],
                "aclDestroyTensor": [Ptr],
                "aclnnAbsGetWorkspaceSize": [Ptr, Ptr, Ref(U64), Ref(Ptr)],
                "aclnnAbs": [Ptr, U64, Ptr, Ptr],
            },
            {"aclCreateTensor": Ptr},
        )

    def manage(self, stack, create, destroy, *args):
        handle = self.new(create, *args)
        stack.callback(self.call, destroy, handle)
        return handle

    def tensor(self, stack, address, elements=ELEMENTS):
        shape, stride = (I64 * 1)(elements), (I64 * 1)(1)
        tensor = self.functions["aclCreateTensor"](shape, 1, FLOAT16, stride, 0, FORMAT_ND, shape, 1, address)
        if not tensor:
            raise RuntimeError("aclCreateTensor returned a null handle")
        stack.callback(self.call, "aclDestroyTensor", tensor)
        return tensor


class AbsWork:
    def __init__(self, rt, stack, byte, warmup_stream, elements=None):
        self.rt = rt
        elements = ELEMENTS if elements is None else elements
        self.nbytes = elements * 2
        self.byte = byte
        src, self.output = (rt.manage(stack, "aclrtMalloc", "aclrtFree", self.nbytes, 2) for _ in range(2))
        rt.call("aclrtMemset", src, self.nbytes, byte, self.nbytes)
        self.src, self.dst = (rt.tensor(stack, pointer, elements) for pointer in (src, self.output))
        self.capacity, executor = self.plan()
        self.workspace = rt.manage(stack, "aclrtMalloc", "aclrtFree", self.capacity, 2) if self.capacity else None
        rt.call("aclnnAbs", self.workspace, self.capacity, executor, warmup_stream)
        rt.call("aclrtSynchronizeStreamWithTimeout", warmup_stream, SYNC_TIMEOUT_MS)

    def plan(self):
        size, executor = U64(), Ptr()
        self.rt.call("aclnnAbsGetWorkspaceSize", self.src, self.dst, C.byref(size), C.byref(executor))
        if not executor.value:
            raise RuntimeError("aclnnAbsGetWorkspaceSize returned a null executor")
        return size.value, executor.value

    def enqueue(self, stream):
        size, executor = self.plan()
        if size > self.capacity:
            raise RuntimeError(f"Abs workspace changed from {self.capacity} to {size} bytes at build")
        self.rt.call("aclnnAbs", self.workspace, size, executor, stream)

    def check(self, host):
        # Each FP16 input has both bytes set. Abs clears its sign bit.
        expected = bytes((self.byte, self.byte & 0x7F)) * 16
        for offset in (0, self.nbytes - len(expected)):
            self.rt.call("aclrtMemcpy", host, len(expected), self.output + offset, len(expected), D2H)
            if C.string_at(host, len(expected)) != expected:
                raise RuntimeError(f"Abs replay output differs at byte {offset} for input {self.byte:#x}")


def run_case(rt, execution, observers, work, host, priorities, order):
    with ExitStack() as stack:
        streams = {}
        for lane in order:
            streams[lane] = rt.manage(
                stack, "aclrtCreateStreamWithConfig", "aclrtDestroyStream", priorities[lane], PERSISTENT
            )
        done = [rt.manage(stack, "aclrtCreateNotify", "aclrtDestroyNotify", 0) for _ in work]
        join = rt.manage(stack, "aclrtCreateNotify", "aclrtDestroyNotify", 0)
        model = rt.manage(stack, "aclmdlRIBuildBegin", "aclmdlRIDestroy", 0)
        order_name = "ab" if order[0] == 0 else "ba"
        model_name = f"priority_a{priorities[0]}_b{priorities[1]}_{order_name}"
        rt.call("aclmdlRISetName", model, model_name.encode())
        for lane in order:
            rt.call("aclmdlRIBindStream", model, streams[lane], HEAD)
            stack.callback(rt.call, "aclmdlRIUnbindStream", model, streams[lane])
        for _ in range(STEPS):
            for lane in order:
                work[lane].enqueue(streams[lane])
        for lane in order:
            rt.call("aclrtRecordNotify", done[lane], streams[lane])
        # The end task on A completes after B. Each completion marker above
        # measures only its own branch, before the model's join.
        rt.call("aclrtRecordNotify", join, streams[1])
        rt.call("aclrtWaitAndResetNotify", join, streams[0], NOTIFY_TIMEOUT_S)
        rt.call("aclmdlRIEndTask", model, streams[0])
        rt.call("aclmdlRIBuildEnd", model, None)
        model_id = U32()
        rt.call("aclmdlRIGetId", model, C.byref(model_id))
        deltas = []
        for replay in range(REPLAYS + 1):
            timers = [rt.manage(stack, "aclrtCreateEventExWithFlag", "aclrtDestroyEvent", TIME_LINE) for _ in work]
            for op in work:
                rt.call("aclrtMemset", op.output, op.nbytes, 0, op.nbytes)
            # Arm observers before submitting the graph. Both timestamps use
            # the same device clock. All host waits follow graph submission.
            for lane in order:
                rt.call("aclrtWaitAndResetNotify", done[lane], observers[lane], NOTIFY_TIMEOUT_S)
                rt.call("aclrtRecordEvent", timers[lane], observers[lane])
            rt.call("aclmdlRIExecuteAsync", model, execution)
            rt.call("aclrtSynchronizeStreamWithTimeout", execution, SYNC_TIMEOUT_MS)
            for observer in observers:
                rt.call("aclrtSynchronizeStreamWithTimeout", observer, SYNC_TIMEOUT_MS)
            stamps = []
            for lane, op in enumerate(work):
                stamp = U64()
                rt.call("aclrtEventGetTimestamp", timers[lane], C.byref(stamp))
                if not stamp.value:
                    raise RuntimeError("device observer returned an empty timestamp")
                stamps.append(stamp.value)
                op.check(host)
            if replay:
                deltas.append(stamps[1] - stamps[0])
        return {
            "priorities": list(priorities),
            "creation_order": list(order),
            "model_id": model_id.value,
            "model_name": model_name,
            "stream_ids": [rt.stream_id(streams[lane]) for lane in range(2)],
            "b_minus_a_us": deltas,
            "median_b_minus_a_us": median(deltas),
        }


def priority_effect(equal, a_high, b_high):
    # Positive delta means A finished first. For each queue order, both
    # directions must beat the lead and jitter of that order's control.
    center = median(equal)
    jitter = max(1, 3 * median(abs(x - center) for x in equal))
    a_wins = sum(x > max(0, center) + jitter for x in a_high)
    b_wins = sum(x < min(0, center) - jitter for x in b_high)
    return a_wins * 4 >= len(a_high) * 3 and b_wins * 4 >= len(b_high) * 3


def run(rt, device, report):
    with ExitStack() as stack:
        rt.call("aclInit", None)
        stack.callback(rt.call, "aclFinalize")
        rt.call("aclrtSetDevice", device)
        stack.callback(rt.call, "aclrtResetDevice", device)
        soc = rt.functions["aclrtGetSocName"]()
        report["soc"] = soc.decode() if soc else None
        least, greatest = I32(), I32()
        rt.call("aclrtDeviceGetStreamPriorityRange", C.byref(least), C.byref(greatest))
        lo, hi = least.value, greatest.value
        report["range"] = {"least": lo, "greatest": hi}
        if lo <= hi:
            raise RuntimeError(f"expected two distinct priority levels, got {hi}..{lo}")
        execution, *observers = (rt.manage(stack, "aclrtCreateStream", "aclrtDestroyStream") for _ in range(3))
        host = rt.manage(stack, "aclrtMallocHost", "aclrtFreeHost", 32)
        work = [AbsWork(rt, stack, byte, execution) for byte in (0xBC, 0xC0)]
        cases = report["cases"] = []
        effects = []
        for order in ((0, 1), (1, 0)):
            samples = []
            for priorities in ((hi, hi), (hi, lo), (lo, hi)):
                report["active_case"] = {"priorities": list(priorities), "creation_order": list(order)}
                case = run_case(rt, execution, observers, work, host, priorities, order)
                cases.append(case)
                samples.append(case["b_minus_a_us"])
            effects.append(priority_effect(*samples))
        del report["active_case"]
        return "creation_priority_effect_observed" if all(effects) else "no_clear_creation_priority_effect"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    report = {"device": args.device, "steps_per_queue": STEPS, "measured_replays": REPLAYS, "status": "failed"}
    try:
        report["status"] = run(CreationRuntime(), args.device, report)
    except (OSError, RuntimeError) as error:
        report["error"] = str(error)
    print("[CREATE_PRIORITY]", json.dumps(report), flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    sys.exit(main())
