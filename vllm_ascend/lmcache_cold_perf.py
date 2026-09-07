# SPDX-License-Identifier: Apache-2.0
"""Request-scoped LMCache cold-path performance logging."""

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from vllm.logger import logger

_FALSE_VALUES = ("", "0", "false", "no", "off")
_COLD_PERF_MODE = os.environ.get("LMCACHE_COLD_START_PERF", "0").strip().lower()
_COLD_PERF_ENABLED = _COLD_PERF_MODE not in _FALSE_VALUES
# Ordinary perf logging is host-only. Device timing requires explicit opt-in;
# it records stream events and must not be treated as an overhead-free timer.
_COLD_PERF_DEVICE_TIMING_ENABLED = _COLD_PERF_MODE == "device"
_cold_perf_request_ids: set[str] = set()
_cold_perf_emitted: dict[str, set[str]] = {}


def _clock_domain() -> tuple[str, str]:
    host = socket.gethostname()
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = str(round(time.time() - time.monotonic()))
    return host, f"{host}:{boot}"


_HOST, _CLOCK_DOMAIN = _clock_domain()


def cold_perf_clock_fields() -> dict[str, Any]:
    """Return clock-safe correlation fields for one performance event."""
    return {
        "wall_time_ns": time.time_ns(),
        "host": _HOST,
        "clock_domain": _CLOCK_DOMAIN,
    }


def cold_perf_enabled() -> bool:
    return _COLD_PERF_ENABLED


def cold_perf_device_timing_enabled() -> bool:
    """Whether explicit device-event diagnostics were selected at startup."""
    return _COLD_PERF_DEVICE_TIMING_ENABLED


def _non_json_field(_value: Any) -> str:
    # Never invoke repr/str on tensors or other caller-owned objects: formatting
    # a device tensor can read it back and synchronize the serving thread.
    return "<non-JSON value>"


def mark_cold_perf_requests(request_ids: Any) -> None:
    if not cold_perf_enabled():
        return
    if isinstance(request_ids, str):
        _cold_perf_request_ids.add(request_ids)
        return
    _cold_perf_request_ids.update(str(req_id) for req_id in request_ids)


def mark_cold_perf_connector_requests(metadata: Any) -> None:
    """Mark LMCache cold-compact resumes carried by connector metadata."""
    if not cold_perf_enabled():
        return
    mark_cold_perf_requests(
        request.req_id
        for request in getattr(metadata, "requests", ())
        if getattr(
            getattr(request, "load_spec", None),
            "dsa_cold_compact_resume",
            False,
        )
    )


def is_cold_perf_request(request_id: str) -> bool:
    return cold_perf_enabled() and request_id in _cold_perf_request_ids


def forget_cold_perf_request(request_id: str) -> None:
    if not cold_perf_enabled():
        return
    _cold_perf_request_ids.discard(request_id)
    _cold_perf_emitted.pop(request_id, None)


def log_cold_perf_event(
    event: str,
    *,
    request_id: str | None = None,
    request_ids: Any = None,
    once: bool = False,
    require_active: bool = True,
    **fields: Any,
) -> None:
    if not cold_perf_enabled():
        return
    ids = [request_id] if request_id is not None else list(request_ids or ())
    ids = [str(req_id) for req_id in ids if req_id is not None]
    if require_active:
        ids = [req_id for req_id in ids if req_id in _cold_perf_request_ids]
    if once:
        ids = [req_id for req_id in ids if event not in _cold_perf_emitted.get(req_id, ())]
        for req_id in ids:
            _cold_perf_emitted.setdefault(req_id, set()).add(event)
    if not ids:
        return
    payload = {
        "schema": 1,
        "event": event,
        "pid": os.getpid(),
        "monotonic_ms": round(time.perf_counter() * 1000, 3),
        **cold_perf_clock_fields(),
        **fields,
    }
    if request_id is not None and len(ids) == 1:
        payload["req_id"] = ids[0]
    else:
        payload["request_ids"] = ids
    logger.info(
        "[LMCACHE_COLD_PERF] %s",
        json.dumps(payload, default=_non_json_field, separators=(",", ":")),
    )


def log_cold_perf_process_event(event: str, **fields: Any) -> None:
    """Log a process-level anomaly that is not tied to a marked request."""
    if not cold_perf_enabled():
        return
    payload = {
        "schema": 1,
        "event": event,
        "pid": os.getpid(),
        "monotonic_ms": round(time.perf_counter() * 1000, 3),
        **cold_perf_clock_fields(),
        **fields,
    }
    logger.info(
        "[LMCACHE_COLD_PERF] %s",
        json.dumps(payload, default=_non_json_field, separators=(",", ":")),
    )
