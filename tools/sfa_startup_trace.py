# SPDX-License-Identifier: Apache-2.0
"""Test-only startup tracing. Importing this module never imports Torch/NPU."""

import functools
import inspect
import json
import os
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

TRACE_PREFIX = "[SFA_STARTUP] "
STACK_DEPTH = 8
COMM_NAME_METHOD = "get_hccl_comm_name"


class StartupTrace:
    """Observe existing calls; never query a communicator name to log it.

    get_hccl_comm_name is a pybind method on supported torch_npu builds. A
    Python C-call profiler observes it without replacing the extension type
    or executing another HCCL call. Its return event does NOT imply that a new
    native communicator/socket was created (the native backend may cache it).
    """

    def __init__(self, directory, *, rank=None, local_rank=None, stage="worker"):
        self.directory = Path(directory)
        self.rank = rank
        self.local_rank = local_rank
        self.stage = stage
        self.sequence = 0
        self.torch = None
        self.group_name = None
        self.patches = ExitStack()
        self.previous_profile = None
        self.profile_installed = False
        self.comm_calls = 0

    def device(self):
        # current_device() can initialize a context: do not call it beforehand.
        torch = self.torch if self.torch is not None else sys.modules.get("torch")
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_initialized():
            return None
        return npu.current_device()

    def emit(self, event, **fields):
        self.sequence += 1
        record = dict(
            event=event,
            pid=os.getpid(),
            ppid=os.getppid(),
            rank=self.rank,
            local_rank=self.local_rank,
            stage=self.stage,
            sequence=self.sequence,
            time_ns=time.time_ns(),
            **fields,
        )
        encoded = json.dumps(record, default=str, ensure_ascii=True)
        # One file per PID; parent reads these even if executor initialization
        # fails before collective_rpc can return a report.
        with (self.directory / f"trace-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        print(TRACE_PREFIX + encoded, flush=True)

    @contextmanager
    def span(self, operation, **fields):
        started = time.monotonic()
        self.emit("begin", operation=operation, device=self.device(), **fields)
        try:
            yield
        except BaseException as exc:
            self.emit("error", operation=operation, error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self.emit("end", operation=operation, device=self.device(), seconds=time.monotonic() - started)

    def _profile(self, frame, event, function):
        if event in ("c_call", "c_return", "c_exception") and getattr(function, "__name__", "") == COMM_NAME_METHOD:
            if event == "c_call":
                self.comm_calls += 1
            self.emit(
                "hccl_comm_name",
                action=event,
                call=self.comm_calls,
                device=self.device(),
                backend_id=id(getattr(function, "__self__", None)),
                process_group_id=id(frame.f_locals["device_group"]) if "device_group" in frame.f_locals else None,
                caller_local_rank=frame.f_locals.get("local_rank"),
                stack=traceback.format_stack(frame, limit=STACK_DEPTH) if event == "c_call" else [],
            )
        if self.previous_profile is not None:
            self.previous_profile(frame, event, function)

    def install(self, torch):
        self.torch = torch
        self.emit("trace_installed", device=self.device(), visible_devices=os.getenv("ASCEND_RT_VISIBLE_DEVICES"))
        for owner, name in (
            (torch.npu, "set_device"),
            (torch.distributed, "init_process_group"),
            (torch.distributed, "new_group"),
        ):
            original = getattr(owner, name)
            signature = inspect.signature(original)

            def wrap(original, signature, name):
                @functools.wraps(original)
                def traced(*args, **kwargs):
                    bound = signature.bind(*args, **kwargs).arguments
                    fields = {
                        key: bound[key]
                        for key in ("device", "rank", "world_size", "ranks", "backend", "init_method")
                        if key in bound
                    }
                    # Use requested_device to avoid shadowing the observed device.
                    if "device" in fields:
                        fields["requested_device"] = fields.pop("device")
                    if "rank" in fields:
                        fields["requested_rank"] = fields.pop("rank")
                    options = bound.get("pg_options")
                    fields["hccl_config"] = getattr(options, "hccl_config", None)
                    with self.span(
                        name,
                        group_name=self.group_name,
                        arguments=fields,
                        stack=traceback.format_stack(limit=STACK_DEPTH),
                    ):
                        result = original(*args, **kwargs)
                    if name == "new_group":
                        self.emit("group_created", group_name=self.group_name, group_id=id(result), arguments=fields)
                    return result

                return traced

            self.patches.enter_context(patch.object(owner, name, wrap(original, signature, name)))
        self.observe_native_calls()

    def observe_native_calls(self):
        # Can run before any Torch import, preserving the original preflight
        # import order. No NPU context is created just to install the observer.
        self.previous_profile = sys.getprofile()
        sys.setprofile(self._profile)
        self.profile_installed = True

    def trace_coordinator(self, coordinator):
        # Install after NPUWorker.__init__ applies the Ascend coordinator patch.
        original = coordinator.__init__
        signature = inspect.signature(original)

        @functools.wraps(original)
        def traced(instance, *args, **kwargs):
            bound = signature.bind(instance, *args, **kwargs).arguments
            previous = self.group_name
            self.group_name = bound.get("group_name")
            try:
                with self.span("GroupCoordinator", group_name=self.group_name, ranks=bound.get("group_ranks")):
                    original(instance, *args, **kwargs)
            finally:
                self.group_name = previous

        self.patches.enter_context(patch.object(coordinator, "__init__", traced))

    def close(self):
        if self.profile_installed:
            sys.setprofile(self.previous_profile)
            self.profile_installed = False
        self.patches.close()

    def runtime_libraries(self):
        """Report already-loaded libraries only; never import one to inspect it."""
        modules = {}
        for name in ("torch", "torch_npu", "lmcache_ascend.c_ops"):
            module = sys.modules.get(name)
            if module is not None:
                modules[name] = {
                    "path": getattr(module, "__file__", None),
                    "version": str(getattr(module, "__version__", "unknown")),
                }
        libraries = set()
        maps = Path("/proc/self/maps")
        if maps.is_file():
            for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and any(
                    name in fields[5] for name in ("libhccl", "libhcomm", "libascendcl", "libtorch_npu")
                ):
                    libraries.add(fields[5])
        self.emit("runtime_libraries", modules=modules, libraries=sorted(libraries))


def mapping_errors(records, tp_size, *, require_complete):
    """Check per-rank mappings without introducing another HCCL collective."""
    mappings = [record for record in records if record.get("event") == "device_mapping"]
    ranks = [record["rank"] for record in mappings]
    errors = []
    if len(ranks) != len(set(ranks)):
        errors.append("multiple worker mappings for the same rank")
    if require_complete and set(ranks) != set(range(tp_size)):
        errors.append(f"incomplete rank coverage: {sorted(ranks)}")
    for field in ("pid", "visible_device", "uuid"):
        values = [record.get(field) for record in mappings if record.get(field) is not None]
        if len(values) != len(set(values)):
            errors.append(f"duplicate {field} across workers")
    for record in mappings:
        if record["device"] != record["local_rank"]:
            errors.append(f"rank={record['rank']}: current device differs from assigned local rank")
    return errors
