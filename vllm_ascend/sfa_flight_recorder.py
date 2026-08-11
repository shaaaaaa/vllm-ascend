# SPDX-License-Identifier: Apache-2.0
"""Low-perturbation flight recording for staged SFA failures."""

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from vllm.logger import logger


_SFA_FLIGHT_SCHEMA_VERSION = 1
_SFA_FLIGHT_CAPACITY = 16384


class SFAFlightRecorder:
    """Keep recent host-side SFA phase markers and dump them on failure.

    Recording deliberately performs no device reads, synchronization, or file
    I/O. Events are written only after an exception or a watchdog-detected
    stall has already occurred.

    Args:
        output_dir: Directory used for failure dumps.
        identity: Static JSON-compatible worker identity fields.
        stall_timeout_sec: Seconds without a host-side phase transition while
            an operation is active before dumping. Non-positive disables it.
    """

    def __init__(
        self,
        output_dir: str,
        identity: dict[str, Any] | None = None,
        stall_timeout_sec: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.identity = dict(identity or {})
        self.events: deque[dict[str, Any]] = deque(
            maxlen=_SFA_FLIGHT_CAPACITY,
        )
        self._lock = threading.Lock()
        self._active_depth = 0
        self._last_progress_ns = 0
        self._stall_dumped = False
        self._stall_timeout_sec = max(0, int(stall_timeout_sec))
        if self._stall_timeout_sec:
            threading.Thread(
                target=self._watchdog_loop,
                name="sfa-flight-watchdog",
                daemon=True,
            ).start()
        logger.warning(
            "[SFA_FLIGHT] enabled: directory=%s stall_timeout_sec=%d",
            self.output_dir,
            self._stall_timeout_sec,
        )

    def record(self, phase: str, **fields: Any) -> None:
        """Append one host-only phase marker to the in-memory ring."""
        now_ns = time.monotonic_ns()
        event = {
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": now_ns,
            "phase": phase,
            **fields,
        }
        with self._lock:
            self.events.append(event)
            self._last_progress_ns = now_ns
            if phase == "dummy_enter" or phase.endswith("_begin"):
                if self._active_depth == 0:
                    self._stall_dumped = False
                self._active_depth += 1
            elif phase == "dummy_exit" or phase.endswith("_end"):
                self._active_depth = max(0, self._active_depth - 1)

    def _watchdog_loop(self) -> None:
        """Dump the ring when an active host-side phase stops progressing."""
        timeout_ns = self._stall_timeout_sec * 1_000_000_000
        poll_interval_sec = min(5.0, max(0.25, self._stall_timeout_sec / 4))
        while True:
            time.sleep(poll_interval_sec)
            now_ns = time.monotonic_ns()
            with self._lock:
                if (
                    self._active_depth == 0
                    or self._stall_dumped
                    or self._last_progress_ns == 0
                ):
                    continue
                stalled_for_ns = now_ns - self._last_progress_ns
                if stalled_for_ns < timeout_ns:
                    continue
                self._stall_dumped = True
                last_phase = (
                    self.events[-1].get("phase")
                    if self.events
                    else None
                )
                active_depth = self._active_depth

            stalled_for_sec = stalled_for_ns / 1_000_000_000
            self.dump(
                "watchdog_stall",
                TimeoutError(
                    "No SFA flight-recorder progress for "
                    f"{stalled_for_sec:.1f}s; last_phase={last_phase}; "
                    f"active_depth={active_depth}"
                ),
            )

    def dump(self, reason: str, error: BaseException) -> Path | None:
        """Persist the current ring after a failure without touching the NPU."""
        try:
            with self._lock:
                events = list(self.events)
                self._stall_dumped = True
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp_ns = time.time_ns()
            path = self.output_dir / (
                f"sfa-flight-pid{os.getpid()}-{timestamp_ns}.jsonl"
            )
            header = {
                "schema_version": _SFA_FLIGHT_SCHEMA_VERSION,
                "kind": "header",
                "pid": os.getpid(),
                "reason": reason,
                "error_type": type(error).__qualname__,
                "error": str(error),
                "event_count": len(events),
                **self.identity,
            }
            with path.open("w", encoding="utf-8") as output:
                output.write(json.dumps(header, separators=(",", ":")))
                output.write("\n")
                for event in events:
                    output.write(
                        json.dumps(
                            {
                                "schema_version": (
                                    _SFA_FLIGHT_SCHEMA_VERSION
                                ),
                                "kind": "event",
                                **event,
                            },
                            separators=(",", ":"),
                        )
                    )
                    output.write("\n")
            logger.error("[SFA_FLIGHT] failure trace saved to %s", path)
            return path
        except (OSError, RuntimeError, TypeError, ValueError):
            logger.exception("[SFA_FLIGHT] failed to persist failure trace")
            return None


def record_sfa_flight_event(
    recorder: SFAFlightRecorder | None,
    phase: str,
    **fields: Any,
) -> None:
    """Record an event when diagnostics are enabled."""
    if recorder is not None:
        recorder.record(phase, **fields)
