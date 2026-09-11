# SPDX-License-Identifier: Apache-2.0
"""Bounded, opt-in model-runner timing; serving operations stay in the runner.

This mixin has no constructor or serving overrides. Host timing never creates
device events. Device timing only queries completed events and retains the
existing queue bound and failure handling.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.serving_perf import (
    cold_perf_device_timing_enabled,
    log_cold_perf_event,
)

_COLD_PERF_SAMPLE_TRACE_CALLS = 2


_COLD_PERF_SLOW_SAMPLE_MS = 500.0


_COLD_PERF_SLOW_NPU_INTERVAL_MS = 100.0


_COLD_PERF_MAX_PENDING_NPU_INTERVALS = 32


@dataclass
class _ColdPerfNPUInterval:
    request_ids: tuple[str, ...]
    stage: str
    start_event: Any
    end_event: Any
    host_wall_ms: float
    host_thread_cpu_ms: float
    host_process_cpu_ms: float
    force_emit: bool = False


def _record_sample_stage(stages: dict[str, float], name: str, started: float) -> None:
    stages[name] = (time.perf_counter() - started) * 1000


def _log_slow_sample_invocation(
    request_ids: tuple[str, ...],
    elapsed_ms: float,
    thread_cpu_ms: float,
    process_cpu_ms: float,
    stages: dict[str, float],
) -> None:
    if elapsed_ms < _COLD_PERF_SLOW_SAMPLE_MS:
        return
    log_cold_perf_event(
        "decoder_sample_invocation_slow",
        request_ids=request_ids,
        require_active=False,
        total_wall_ms=round(elapsed_ms, 3),
        total_thread_cpu_ms=round(thread_cpu_ms, 3),
        total_process_cpu_ms=round(process_cpu_ms, 3),
        unattributed_wall_ms=round(max(0.0, elapsed_ms - sum(stages.values())), 3),
        **{name: round(value, 3) for name, value in stages.items()},
    )


class ServingPerfMixin:
    """Optional device timing methods shared by model-runner trace sites."""

    def _cold_perf_npu_error(self, interval, operation: str, exc: Exception) -> None:
        request_ids, stage = (
            (interval.request_ids, interval.stage) if isinstance(interval, _ColdPerfNPUInterval) else interval
        )
        log_cold_perf_event(
            "decoder_npu_interval_error",
            request_ids=request_ids,
            require_active=False,
            once=True,
            stage=stage,
            operation=operation,
            error_type=type(exc).__name__,
        )

    def _run_cold_perf_npu_stage(
        self,
        stage: str,
        request_ids: tuple[str, ...],
        operation: Callable[..., Any],
        *args,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ):
        # The ordinary cold-perf knob must not create device events. Optional
        # device diagnostics are also bounded when completion is delayed.
        if (
            not cold_perf_device_timing_enabled()
            or len(getattr(self, "_cold_perf_pending_npu_intervals", ())) >= _COLD_PERF_MAX_PENDING_NPU_INTERVALS
        ):
            return operation(*args, **kwargs)
        try:
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            start_event.record()
        except Exception as exc:
            self._cold_perf_npu_error((request_ids, stage), "start", exc)
            return operation(*args, **kwargs)

        wall_start = time.perf_counter()
        thread_start = time.thread_time_ns()
        process_start = time.process_time_ns()
        try:
            return operation(*args, **kwargs)
        finally:
            interval = _ColdPerfNPUInterval(
                request_ids,
                stage,
                start_event,
                end_event,
                (time.perf_counter() - wall_start) * 1000,
                (time.thread_time_ns() - thread_start) / 1e6,
                (time.process_time_ns() - process_start) / 1e6,
            )
            if metrics is not None:
                metrics.update(
                    {
                        f"{stage}_wall_ms": interval.host_wall_ms,
                        f"{stage}_thread_cpu_ms": interval.host_thread_cpu_ms,
                        f"{stage}_process_cpu_ms": interval.host_process_cpu_ms,
                    }
                )
            try:
                end_event.record()
            except Exception as exc:
                self._cold_perf_npu_error(interval, "end", exc)
            else:
                self.__dict__.setdefault("_cold_perf_pending_npu_intervals", []).append(interval)
                current = getattr(self, "_cold_perf_current_sample_npu_intervals", None)
                if current is not None:
                    current.append(interval)
                self._cold_perf_last_npu_interval = interval

    def _drain_cold_perf_npu_intervals(self) -> None:
        pending = getattr(self, "_cold_perf_pending_npu_intervals", ())
        if not pending:
            return
        remaining = []
        for interval in pending:
            try:
                if not interval.end_event.query():
                    remaining.append(interval)
                    continue
                device_ms = interval.start_event.elapsed_time(interval.end_event)
            except Exception as exc:
                self._cold_perf_npu_error(interval, "query", exc)
                continue
            if interval.force_emit or device_ms >= _COLD_PERF_SLOW_NPU_INTERVAL_MS:
                log_cold_perf_event(
                    "decoder_npu_interval_slow",
                    request_ids=interval.request_ids,
                    require_active=False,
                    stage=interval.stage,
                    device_elapsed_ms=round(device_ms, 3),
                    host_wall_ms=round(interval.host_wall_ms, 3),
                    host_thread_cpu_ms=round(interval.host_thread_cpu_ms, 3),
                    host_process_cpu_ms=round(interval.host_process_cpu_ms, 3),
                    forced_by_sample_stall=interval.force_emit,
                )
        self._cold_perf_pending_npu_intervals = remaining
