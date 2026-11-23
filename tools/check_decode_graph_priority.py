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
from pathlib import Path

Ptr = C.c_void_p
U32 = C.c_uint32
I32 = C.c_int32
Size = C.c_size_t
Ref = C.POINTER
PRIORITY = 5  # ACL_STREAM_ATTR_PRIORITY, CANN 9.1 acl/acl_rt.h
EVENT_SYNC = 1
H2D, D2H, D2D = 1, 2, 3


class StreamAttribute(C.Union):
    _fields_ = [("priority", U32), ("failure_mode", C.c_uint64), ("reserve", U32 * 4)]


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
            "aclrtGetSocName": [],
            "aclrtDeviceGetStreamPriorityRange": [Ref(I32), Ref(I32)],
            "aclrtCreateStream": [Ref(Ptr)],
            "aclrtCreateStreamWithConfig": [Ref(Ptr), U32, U32],
            "aclrtStreamGetPriority": [Ptr, Ref(U32)],
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
            f.argtypes = args
            f.restype = C.c_char_p if name == "aclrtGetSocName" else I32
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

    def device_priority(self, stream):
        value = StreamAttribute()
        rc = self.functions["aclrtGetStreamAttribute"](stream, PRIORITY, C.byref(value))
        return {"rc": rc, "priority": value.priority} if rc == 0 else {"rc": rc}

    def set_priority(self, stream, priority):
        value = StreamAttribute(priority=priority)
        rc = self.functions["aclrtSetStreamAttribute"](stream, PRIORITY, C.byref(value))
        return {"requested": priority, "rc": rc}

    def stream_id(self, stream):
        stream_id = I32()
        rc = self.functions["aclrtStreamGetId"](stream, C.byref(stream_id))
        return {"rc": rc} if rc else stream_id.value

    def ordinary_controls(self, original_side, least):
        # Capture has ended and the graph has been destroyed. Use its original
        # ordinary stream for the same update as the graph's internal copy queue.
        controls = {
            "id": self.stream_id(original_side),
            "set_side": self.set_priority(original_side, least),
            "after_set": self.device_priority(original_side),
        }
        stream = Ptr()
        rc = self.functions["aclrtCreateStreamWithConfig"](C.byref(stream), least, 0)
        created = {"requested": least, "rc": rc}
        controls["create_low"] = created
        if rc == 0:
            try:
                # StreamGetPriority reports the creation field in CANN 9.1.
                # Only GetStreamAttribute queries the device-side scheduler.
                priority = U32()
                rc = self.functions["aclrtStreamGetPriority"](stream, C.byref(priority))
                created["creation_field"] = {"rc": rc, "priority": priority.value} if rc == 0 else {"rc": rc}
                created["device_read"] = self.device_priority(stream)
            finally:
                self.call("aclrtDestroyStream", stream)
        return controls


def one(handles, what):
    if len(handles) != 1 or None in handles:
        raise RuntimeError(f"{what}: expected one bound stream, got {len(handles)}")
    return next(iter(handles))


def check_graph_queues(rt, main, side, least, greatest, report, replay):
    # Match the D worker: change only the graph-owned copy queue. Record each
    # driver result separately so a failed query cannot hide the update result.
    report["set_side"] = rt.set_priority(side, least)
    report["queues"] = {"main": rt.stream_id(main), "side": rt.stream_id(side)}

    def read_queues():
        return {"main": rt.device_priority(main), "side": rt.device_priority(side)}

    report["after_set"] = read_queues()
    report["verified_replays"] = 0
    for value in (1, 7, 2):
        try:
            replay(value)
        except RuntimeError as exc:
            report["replay_error"] = str(exc)
            return "graph_replay_failed"
        report["verified_replays"] += 1
    report["after_replay"] = read_queues()

    if report["set_side"]["rc"]:
        return "graph_priority_set_failed"
    for snapshot in (report["after_set"], report["after_replay"]):
        main_read, side_read = snapshot["main"], snapshot["side"]
        if main_read["rc"] or side_read["rc"]:
            return "graph_priority_readback_failed"
        if side_read["priority"] != least or not greatest <= main_read["priority"] < least:
            return "graph_priority_mismatch"
    return "graph_priority_verified"


def run(rt, device, report):
    size = 256
    call = rt.call
    call("aclInit", None)
    call("aclrtSetDevice", device)
    soc = rt.functions["aclrtGetSocName"]()
    if soc:
        report["soc"] = soc.decode(errors="replace")
    for package in ("driver", "firmware"):
        version_file = Path("/usr/local/Ascend") / package / "version.info"
        with suppress(OSError):
            for line in version_file.read_text(encoding="utf-8", errors="replace").splitlines():
                name, sep, version = line.partition("=")
                if sep and name.strip().lower() == "version":
                    report[f"{package}_version"] = version.strip()
                    break
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

    def replay(value):
        C.memset(host_in, value, size)
        C.memset(host_out, 0, size)
        call("aclrtMemcpy", src, size, host_in, size, H2D)
        call("aclmdlRIExecuteAsync", graph, root)
        call("aclrtSynchronizeStream", root)
        if C.string_at(host_out, size) != bytes([value]) * size:
            raise RuntimeError(f"incorrect fork/join replay result for byte {value}")

    verdict = check_graph_queues(rt, main_queue, side_queue, low.value, high.value, report, replay)
    call("aclmdlRIDestroy", graph)
    if verdict != "graph_priority_verified":
        report["ordinary"] = rt.ordinary_controls(side, low.value)
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
    return verdict


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--device", type=int, default=0)
    args = cli.parse_args()
    report = {"device": args.device, "status": "failed"}
    try:
        report["status"] = run(Runtime(), args.device, report)
    except (OSError, RuntimeError) as exc:
        report["error"] = str(exc)
    print("[GRAPH_PRIORITY]", json.dumps(report), flush=True)
    return int(report["status"] != "graph_priority_verified")


if __name__ == "__main__":
    sys.exit(main())
