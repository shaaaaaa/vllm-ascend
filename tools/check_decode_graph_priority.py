#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check CANN queue priorities on the actual queues of a two-stream ACL graph.

Run in its own process inside the D container. Requires only Python and CANN.
One small D2D on each branch plus a D2H result check, independent of vLLM.
"""

import argparse
import ctypes as C
import json
import sys
from contextlib import suppress

Ptr = C.c_void_p
U32 = C.c_uint32
I32 = C.c_int32
Size = C.c_size_t
Ref = C.POINTER
PRIORITY = 5  # ACL_STREAM_ATTR_PRIORITY, CANN 9.1 acl/acl_rt.h
EVENT_SYNC = 1
H2D, D2H, D2D = 1, 2, 3


class StreamAttribute(C.Union):
    _fields_ = [("priority", U32), ("reserve", U32 * 4)]


class Runtime:
    def __init__(self):
        self.libs = [C.CDLL("libascendcl.so", mode=C.RTLD_GLOBAL)]
        with suppress(OSError):
            self.libs.append(C.CDLL("libacl_rt.so", mode=C.RTLD_GLOBAL))
        signatures = {
            "aclInit": [C.c_char_p],
            "aclFinalize": [],
            "aclrtSetDevice": [I32],
            "aclrtResetDevice": [I32],
            "aclrtDeviceGetStreamPriorityRange": [Ref(I32), Ref(I32)],
            "aclrtCreateStream": [Ref(Ptr)],
            "aclrtDestroyStream": [Ptr],
            "aclrtCreateEventExWithFlag": [Ref(Ptr), U32],
            "aclrtDestroyEvent": [Ptr],
            "aclrtMalloc": [Ref(Ptr), Size, I32],
            "aclrtFree": [Ptr],
            "aclrtMallocHost": [Ref(Ptr), Size],
            "aclrtFreeHost": [Ptr],
            "aclrtMemcpy": [Ptr, Size, Ptr, Size, I32],
            "aclrtMemcpyAsync": [Ptr, Size, Ptr, Size, I32, Ptr],
            "aclrtRecordEvent": [Ptr, Ptr],
            "aclrtStreamWaitEvent": [Ptr, Ptr],
            "aclmdlRICaptureBegin": [Ptr, I32],
            "aclmdlRICaptureEnd": [Ptr, Ref(Ptr)],
            "aclmdlRICaptureGetInfo": [Ptr, Ref(I32), Ref(Ptr)],
            "aclmdlRIGetStreams": [Ptr, Ref(Ptr), Ref(U32)],
            "aclrtStreamGetId": [Ptr, Ref(I32)],
            "aclrtSetStreamAttribute": [Ptr, I32, Ref(StreamAttribute)],
            "aclrtGetStreamAttribute": [Ptr, I32, Ref(StreamAttribute)],
            "aclmdlRIExecuteAsync": [Ptr, Ptr],
            "aclrtSynchronizeStream": [Ptr],
            "aclmdlRIDestroy": [Ptr],
        }
        self.functions = {}
        for name, args in signatures.items():
            f = next((getattr(lib, name) for lib in self.libs if hasattr(lib, name)), None)
            if f is None:
                raise RuntimeError(f"loaded CANN runtime lacks {name}")
            f.argtypes, f.restype = args, I32
            self.functions[name] = f

    def call(self, name, *args):
        rc = self.functions[name](*args)
        if rc:
            raise RuntimeError(f"{name} rc={rc}")

    def new(self, name, *args):
        handle = Ptr()
        self.call(name, C.byref(handle), *args)
        if not handle.value:
            raise RuntimeError(f"{name} returned a null handle")
        return handle.value

    def streams(self, graph):
        n = U32()
        self.call("aclmdlRIGetStreams", graph, None, C.byref(n))
        handles = (Ptr * n.value)()
        self.call("aclmdlRIGetStreams", graph, handles, C.byref(n))
        return set(handles[: n.value])

    def priority(self, stream):
        # Read the device attribute updated by the setter. StreamGetPriority
        # may report the priority used on creation on some runtime versions.
        value = StreamAttribute()
        self.call("aclrtGetStreamAttribute", stream, PRIORITY, C.byref(value))
        return I32(value.priority).value

    def describe(self, stream):
        stream_id = I32()
        self.call("aclrtStreamGetId", stream, C.byref(stream_id))
        return {"id": stream_id.value, "priority": self.priority(stream)}


def one(handles, what):
    if len(handles) != 1 or None in handles:
        raise RuntimeError(f"{what}: expected one bound stream, got {len(handles)}")
    return next(iter(handles))


def run(rt, device, report):
    size = 256
    call = rt.call
    call("aclInit", None)
    call("aclrtSetDevice", device)
    low, high = I32(), I32()
    call("aclrtDeviceGetStreamPriorityRange", C.byref(low), C.byref(high))
    report["range"] = {"least": low.value, "greatest": high.value}
    if low.value == high.value:
        raise RuntimeError("device reports one priority level")
    root, side = (rt.new("aclrtCreateStream") for _ in range(2))
    fork, join = (rt.new("aclrtCreateEventExWithFlag", EVENT_SYNC) for _ in range(2))
    src, left, right = (rt.new("aclrtMalloc", size, 2) for _ in range(3))
    host_in, host_out = (rt.new("aclrtMallocHost", size) for _ in range(2))

    call("aclmdlRICaptureBegin", root, 0)
    live, status = Ptr(), I32()
    call("aclmdlRICaptureGetInfo", root, C.byref(status), C.byref(live))
    if status.value != 1 or not live.value:
        raise RuntimeError(f"unexpected capture state {status.value}")
    call("aclrtMemcpyAsync", left, size, src, size, D2D, root)
    call("aclrtRecordEvent", fork, root)
    before = rt.streams(live)
    main_queue = one(before, "main capture before fork")
    call("aclrtStreamWaitEvent", side, fork)
    call("aclrtMemcpyAsync", right, size, left, size, D2D, side)
    side_queue = one(rt.streams(live) - before, "new capture queue after side joins")
    call("aclrtRecordEvent", join, side)
    call("aclrtStreamWaitEvent", root, join)
    call("aclrtMemcpyAsync", host_out, size, right, size, D2H, root)
    graph = Ptr()
    call("aclmdlRICaptureEnd", root, C.byref(graph))
    if graph.value != live.value or not {main_queue, side_queue} <= rt.streams(graph):
        raise RuntimeError("captured queue ownership changed on capture end")

    report["before"] = {"main": rt.describe(main_queue), "side": rt.describe(side_queue)}
    for queue, requested in ((main_queue, high), (side_queue, low)):
        attr = StreamAttribute(priority=U32(requested.value).value)
        call("aclrtSetStreamAttribute", queue, PRIORITY, C.byref(attr))

    def check():
        actual = {"main": rt.describe(main_queue), "side": rt.describe(side_queue)}
        if [actual["main"]["priority"], actual["side"]["priority"]] != [high.value, low.value]:
            raise RuntimeError(f"graph priority readback mismatch: {actual}")
        return actual

    report["after_set"] = check()

    for value in (1, 7, 2):
        C.memset(host_in, value, size)
        C.memset(host_out, 0, size)
        call("aclrtMemcpy", src, size, host_in, size, H2D)
        call("aclmdlRIExecuteAsync", graph, root)
        call("aclrtSynchronizeStream", root)
        if C.string_at(host_out, size) != bytes([value]) * size:
            raise RuntimeError(f"incorrect fork/join replay result for byte {value}")
    report["after_replay"] = check()
    report["verified_replays"] = 3
    call("aclmdlRIDestroy", graph)
    for event in (join, fork):
        call("aclrtDestroyEvent", event)
    for stream in (side, root):
        call("aclrtDestroyStream", stream)
    for buf in (right, left, src):
        call("aclrtFree", buf)
    for buf in (host_out, host_in):
        call("aclrtFreeHost", buf)
    call("aclrtResetDevice", device)
    call("aclFinalize")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--device", type=int, default=0)
    args = cli.parse_args()
    report = {"device": args.device, "status": "failed"}
    try:
        run(Runtime(), args.device, report)
        report["status"] = "graph_priority_verified"
    except (OSError, RuntimeError) as exc:
        report["error"] = str(exc)
    print("[GRAPH_PRIORITY]", json.dumps(report), flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    sys.exit(main())
