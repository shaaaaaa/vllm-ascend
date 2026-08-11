# SPDX-License-Identifier: Apache-2.0
"""Low-perturbation flight recording for staged SFA failures."""

import json
import os
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
    I/O. Events are written only after an exception has already occurred.

    Args:
        output_dir: Directory used for failure dumps.
        identity: Static JSON-compatible worker identity fields.
    """

    def __init__(
        self,
        output_dir: str,
        identity: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.identity = dict(identity or {})
        self.events: deque[dict[str, Any]] = deque(
            maxlen=_SFA_FLIGHT_CAPACITY,
        )

    def record(self, phase: str, **fields: Any) -> None:
        """Append one host-only phase marker to the in-memory ring."""
        self.events.append(
            {
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "phase": phase,
                **fields,
            }
        )

    def dump(self, reason: str, error: BaseException) -> Path | None:
        """Persist the current ring after a failure without touching the NPU."""
        try:
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
                "event_count": len(self.events),
                **self.identity,
            }
            with path.open("w", encoding="utf-8") as output:
                output.write(json.dumps(header, separators=(",", ":")))
                output.write("\n")
                for event in self.events:
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
        except (OSError, TypeError, ValueError):
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
