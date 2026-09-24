# SPDX-License-Identifier: Apache-2.0
"""Small host-only phase logs for PD worker startup.

This module deliberately imports neither vLLM nor torch. It can run before
distributed initialization and never creates a group or reads a device tensor.
"""

import os
import socket
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps

MAX_HOST_LENGTH = 24
MAX_LABEL_LENGTH = 48
MAX_DETAIL_KEY_LENGTH = 20


def _attribute(owner, name, default=None):
    try:
        return getattr(owner, name, default)
    except Exception:
        return default


def _integer(value, default=None):
    # Do not coerce tensors or invoke caller-owned conversion methods.
    return value if type(value) is int else default


def _label(value: str, limit: int = MAX_LABEL_LENGTH) -> str:
    return "".join(character if character.isalnum() or character in "._:-" else "_" for character in value)[:limit]


def _identity(owner) -> str:
    config = _attribute(owner, "vllm_config")
    parallel = _attribute(config, "parallel_config")
    dp_rank = _integer(_attribute(parallel, "data_parallel_rank"), 0)
    tp_size = max(1, _integer(_attribute(parallel, "tensor_parallel_size"), 1))
    tp_rank = _integer(_attribute(owner, "tp_rank"))
    if tp_rank is None:
        rank = _integer(_attribute(owner, "rank"))
        tp_rank = rank % tp_size if rank is not None else "?"
    try:
        hostname = _label(socket.gethostname(), MAX_HOST_LENGTH)
    except Exception:
        hostname = "?"
    result = f"h={hostname} p={os.getpid()} d={dp_rank} t={tp_rank}"
    # Inspect only an already initialized group; no imports, group creation,
    # collective calls, or rank queries against the distributed runtime.
    module = sys.modules.get("vllm.distributed.parallel_state")
    group = _attribute(module, "_WORLD")
    global_rank = _integer(_attribute(group, "rank"))
    if global_rank is not None:
        result += f" g={global_rank}"
    return result


def _detail(value):
    if type(value) is bool:
        return str(int(value))
    if type(value) in (int, float):
        return str(value)
    if type(value) is str:
        return _label(value)
    if type(value) in (tuple, list) and value and all(type(rank) is int for rank in value):
        if all(right == left + 1 for left, right in zip(value, value[1:])):
            return str(value[0]) if len(value) == 1 else f"{value[0]}-{value[-1]}"
        return ",".join(map(str, value))[:MAX_LABEL_LENGTH]
    # Do not format configurations, exceptions, tensors, or arbitrary objects.
    return None


def _emit(identity, stage, event, elapsed_ms, details, error=None):
    try:
        fields = []
        for key, value in details.items():
            formatted = _detail(value)
            if formatted is not None:
                fields.append(f"{_label(key, MAX_DETAIL_KEY_LENGTH)}={formatted}")
        if error is not None:
            fields.append(f"exc={_label(type(error).__name__)}")
        suffix = " " + " ".join(fields) if fields else ""
        sys.stderr.write(f"[PD_INIT] {identity} {stage} {event} ms={elapsed_ms:.1f}{suffix}\n")
        sys.stderr.flush()
    except Exception:
        # Diagnostics must never turn a successful phase into a failure or
        # replace the model/connector exception that led to this log.
        pass


@contextmanager
def startup_phase(owner, stage: str, **details) -> Iterator[None]:
    """Log one PD startup boundary and preserve the wrapped operation exactly."""
    config = _attribute(owner, "vllm_config")
    if _attribute(config, "kv_transfer_config") is None:
        yield
        return
    identity = _identity(owner)
    stage = _label(stage)
    started = time.perf_counter()
    _emit(identity, stage, "begin", 0.0, details)
    try:
        yield
    except BaseException as error:
        _emit(identity, stage, "error", (time.perf_counter() - started) * 1000, details, error)
        raise
    else:
        _emit(identity, stage, "end", (time.perf_counter() - started) * 1000, details)


def startup_stage(stage: str):
    """Decorate a bound startup method, retaining its signature and result."""

    def decorate(method):
        @wraps(method)
        def traced(owner, *args, **kwargs):
            with startup_phase(owner, stage):
                return method(owner, *args, **kwargs)

        return traced

    return decorate
