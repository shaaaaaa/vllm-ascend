# SPDX-License-Identifier: Apache-2.0
"""Identify and set the captured retrieval queue once, before graph replay."""

import ctypes as c
import logging
from contextlib import suppress

logger = logging.getLogger(__name__)
Handle = c.c_void_p
UInt32 = c.c_uint32
Int32 = c.c_int32
Pointer = c.POINTER
ACL_STREAM_ATTR_PRIORITY = 5
ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE = 1
ACL_MODEL_RI_TASK_KERNEL = 1


class StreamAttr(c.Union):
    _fields_ = [("priority", UInt32), ("failure_mode", c.c_uint64), ("reserved", UInt32 * 4)]


class CapturedCopyPriority:
    def __init__(self, main_stream):
        # Loaded only for the opt-in CANN 9.1 experiment.
        self.libraries = [c.CDLL("libascendcl.so", mode=c.RTLD_GLOBAL)]
        with suppress(OSError):
            self.libraries.append(c.CDLL("libacl_rt.so", mode=c.RTLD_GLOBAL))
        signatures = {
            "aclmdlRICaptureGetInfo": [Handle, Pointer(Int32), Pointer(Handle)],
            "aclmdlRIGetStreams": [Handle, Pointer(Handle), Pointer(UInt32)],
            "aclmdlRIGetTasksByStream": [Handle, Pointer(Handle), Pointer(UInt32)],
            "aclmdlRITaskGetType": [Handle, Pointer(Int32)],
            "aclrtDeviceGetStreamPriorityRange": [Pointer(Int32), Pointer(Int32)],
            "aclrtSetStreamAttribute": [Handle, Int32, Pointer(StreamAttr)],
            "aclrtGetStreamAttribute": [Handle, Int32, Pointer(StreamAttr)],
            "aclrtStreamGetId": [Handle, Pointer(Int32)],
        }
        self.functions = {}
        for name, args in signatures.items():
            fn = next((getattr(lib, name) for lib in self.libraries if hasattr(lib, name)), None)
            if fn is None:
                raise RuntimeError(f"SFA prefetch priority: missing CANN symbol {name}")
            fn.argtypes, fn.restype = args, Int32
            self.functions[name] = fn
        self.graph = self.capture(main_stream)
        initial = self.handles("aclmdlRIGetStreams", self.graph)
        if len(initial) != 1:
            raise RuntimeError(f"expected one initial graph queue, found {len(initial)}")
        self.main = initial[0]
        self.side = None

    def call(self, name, *args, operation=None):
        rc = self.functions[name](*args)
        if rc:
            where = f" ({operation})" if operation else ""
            raise RuntimeError(f"SFA prefetch priority: {name}{where} rc={rc}")

    def capture(self, stream):
        status, graph = Int32(), Handle()
        # torch-npu's npu_stream accessor drains the host submission queue.
        # All introspection here is inside startup capture, never live replay.
        self.call("aclmdlRICaptureGetInfo", stream.npu_stream, c.byref(status), c.byref(graph))
        if status.value != ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE or not graph.value:
            raise RuntimeError(f"SFA prefetch stream is outside the target capture: {status.value}")
        return graph.value

    def handles(self, name, obj):
        n = UInt32()
        self.call(name, obj, None, c.byref(n))
        handles = (Handle * n.value)()
        self.call(name, obj, handles, c.byref(n))
        return handles[: n.value]

    def before_first_fork(self, main):
        if self.capture(main) != self.graph:
            raise RuntimeError("SFA fork belongs to a different graph")
        return set(self.handles("aclmdlRIGetStreams", self.graph))

    def after_first_fork(self, side, before):
        if self.capture(side) != self.graph:
            raise RuntimeError("SFA side stream joined a different graph")
        added = set(self.handles("aclmdlRIGetStreams", self.graph)) - before
        if len(added) != 1 or self.main in added:
            raise RuntimeError(f"could not uniquely identify SFA side queue, new queues={len(added)}")
        self.side = added.pop()

    def priority(self, stream, role):
        attr = StreamAttr()
        self.call("aclrtGetStreamAttribute", stream, ACL_STREAM_ATTR_PRIORITY, c.byref(attr), operation=role)
        return Int32(attr.priority).value

    def stream_id(self, stream):
        value = Int32()
        self.call("aclrtStreamGetId", stream, c.byref(value))
        return value.value

    def finish(self, transfers, pieces):
        if self.side is None or self.side not in self.handles("aclmdlRIGetStreams", self.graph):
            raise RuntimeError("SFA side queue missing after capture")
        # Verify every copy piece stayed on this queue. An automatic queue
        # split or an unexpected colocated kernel invalidates this experiment.
        kernels = 0
        for task in self.handles("aclmdlRIGetTasksByStream", self.side):
            kind = Int32()
            self.call("aclmdlRITaskGetType", task, c.byref(kind))
            kernels += kind.value == ACL_MODEL_RI_TASK_KERNEL
        if kernels != transfers * pieces:
            raise RuntimeError(f"SFA side queue contains {kernels} kernels, expected {transfers * pieces}")
        least, greatest = Int32(), Int32()
        self.call("aclrtDeviceGetStreamPriorityRange", c.byref(least), c.byref(greatest))
        if least.value == greatest.value:
            raise RuntimeError(f"no lower priority for SFA: least=greatest={least.value}")
        attr = StreamAttr(priority=UInt32(least.value).value)
        self.call("aclrtSetStreamAttribute", self.side, ACL_STREAM_ATTR_PRIORITY, c.byref(attr), operation="set copy")
        foreground = self.priority(self.main, "read main after copy set")
        copy = self.priority(self.side, "read copy after set")
        if copy != least.value or not greatest.value <= foreground < copy:
            raise RuntimeError(
                f"SFA graph priority readback mismatch: main={foreground}, copy={copy}, requested={least.value}"
            )
        logger.info(
            "SFA prefetch graph main_stream=%d priority=%d copy_stream=%d priority=%d "
            "transfers=%d pieces=%d kernels=%d",
            self.stream_id(self.main),
            foreground,
            self.stream_id(self.side),
            copy,
            transfers,
            pieces,
            kernels,
        )
