# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detailed, opt-in timing for the SFA path inside one model forward."""

import os
import time
from collections import defaultdict
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

import torch

_TRACE_ENABLED = os.environ.get("VLLM_PD_STAGE_TRACE", "0").lower() in ("1", "true", "yes", "on")


@dataclass
class _PhaseStats:
    total_ns: int = 0
    count: int = 0
    max_ns: int = 0
    slowest_layer: str = "none"


@dataclass
class _ForwardTrace:
    step: int
    requests: str
    sync_npu: bool
    started_ns: int
    last_sfa_boundary_ns: int
    phases: dict[str, _PhaseStats] = field(default_factory=lambda: defaultdict(_PhaseStats))
    non_sfa_segments: _PhaseStats = field(default_factory=_PhaseStats)
    current_layer: str = "none"
    previous_sfa_layer: str = "forward_start"
    open_sfa: "_SpanToken | None" = None


@dataclass
class _ForwardToken:
    trace: _ForwardTrace
    context_token: Token


@dataclass
class _SpanToken:
    trace: _ForwardTrace
    name: str
    layer_name: str
    started_ns: int


_ACTIVE_TRACE: ContextVar[_ForwardTrace | None] = ContextVar("vllm_ascend_pd_forward_trace", default=None)


def _sync(trace: _ForwardTrace) -> None:
    if trace.sync_npu:
        torch.npu.synchronize()


def _record(stats: _PhaseStats, elapsed_ns: int, layer_name: str) -> None:
    elapsed_ns = max(0, elapsed_ns)
    stats.total_ns += elapsed_ns
    stats.count += 1
    if elapsed_ns > stats.max_ns:
        stats.max_ns = elapsed_ns
        stats.slowest_layer = layer_name


def start_forward(
    step: int,
    requests: str,
    sync_npu: bool,
) -> _ForwardToken | None:
    """Start an aggregate trace spanning the same region as worker forward."""
    if not _TRACE_ENABLED:
        return None

    existing = _ACTIVE_TRACE.get()
    if existing is not None:
        return None

    if sync_npu:
        torch.npu.synchronize()
    started_ns = time.perf_counter_ns()
    trace = _ForwardTrace(
        step=step,
        requests=requests,
        sync_npu=sync_npu,
        started_ns=started_ns,
        last_sfa_boundary_ns=started_ns,
    )
    return _ForwardToken(trace, _ACTIVE_TRACE.set(trace))


def set_step_kind(is_decode_only: bool) -> None:
    # Kept for the existing call sites. The PD trace intentionally includes
    # prefill and mixed steps as well as decode steps.
    return None


class _NoopSection:
    __slots__ = ()

    def __enter__(self) -> "_NoopSection":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _TimedSection:
    __slots__ = ("name", "token")

    def __init__(self, name: str) -> None:
        self.name = name
        self.token: _SpanToken | None = None

    def __enter__(self) -> "_TimedSection":
        self.token = begin(self.name)
        return self

    def __exit__(self, *exc: Any) -> None:
        end(self.token)


_NOOP_SECTION = _NoopSection()


def section(name: str) -> _NoopSection | _TimedSection:
    if not _TRACE_ENABLED or _ACTIVE_TRACE.get() is None:
        return _NOOP_SECTION
    return _TimedSection(name)


def begin(name: str, layer_name: str | None = None) -> _SpanToken | None:
    if not _TRACE_ENABLED:
        return None
    trace = _ACTIVE_TRACE.get()
    if trace is None:
        return None

    _sync(trace)
    started_ns = time.perf_counter_ns()
    resolved_layer = layer_name or trace.current_layer
    token = _SpanToken(trace, name, resolved_layer, started_ns)
    if name == "sfa_fwd":
        segment = f"{trace.previous_sfa_layer}->{resolved_layer}"
        _record(
            trace.non_sfa_segments,
            started_ns - trace.last_sfa_boundary_ns,
            segment,
        )
        trace.current_layer = resolved_layer
        trace.open_sfa = token
    return token


def end(token: _SpanToken | None) -> None:
    if token is None or _ACTIVE_TRACE.get() is not token.trace:
        return

    trace = token.trace
    _sync(trace)
    ended_ns = time.perf_counter_ns()
    _record(
        trace.phases[token.name],
        ended_ns - token.started_ns,
        token.layer_name,
    )
    if token.name == "sfa_fwd":
        trace.last_sfa_boundary_ns = ended_ns
        trace.previous_sfa_layer = token.layer_name
        trace.current_layer = "none"
        trace.open_sfa = None


def _phase_summary(trace: _ForwardTrace) -> tuple[str, int]:
    child_total_ns = 0
    parts = []
    for name, stats in sorted(
        trace.phases.items(),
        key=lambda item: item[1].total_ns,
        reverse=True,
    ):
        if name == "sfa_fwd":
            continue
        child_total_ns += stats.total_ns
        parts.append(
            f"{name}:{stats.total_ns / 1_000_000:.3f}/"
            f"{stats.count}/{stats.max_ns / 1_000_000:.3f}"
            f"@{stats.slowest_layer}"
        )
    return "|".join(parts) or "none", child_total_ns


def finish_forward(token: _ForwardToken | None) -> dict[str, Any] | None:
    if token is None or _ACTIVE_TRACE.get() is not token.trace:
        return None

    trace = token.trace
    try:
        _sync(trace)
        ended_ns = time.perf_counter_ns()

        # Preserve useful partial timing when the model forward exits by error.
        if trace.open_sfa is not None:
            open_sfa = trace.open_sfa
            _record(
                trace.phases["sfa_fwd"],
                ended_ns - open_sfa.started_ns,
                open_sfa.layer_name,
            )
            trace.last_sfa_boundary_ns = ended_ns
            trace.previous_sfa_layer = open_sfa.layer_name
            trace.open_sfa = None

        tail_label = f"{trace.previous_sfa_layer}->forward_end"
        _record(
            trace.non_sfa_segments,
            ended_ns - trace.last_sfa_boundary_ns,
            tail_label,
        )

        total_ns = max(0, ended_ns - trace.started_ns)
        sfa = trace.phases.get("sfa_fwd", _PhaseStats())
        phase_summary, child_total_ns = _phase_summary(trace)
        non_sfa_ns = max(0, total_ns - sfa.total_ns)
        sfa_unattributed_ns = max(0, sfa.total_ns - child_total_ns)
        return {
            "step": trace.step,
            "requests": trace.requests,
            "timing_mode": ("sync_npu_detail" if trace.sync_npu else "host_no_sync_detail"),
            "total_ms": total_ns / 1_000_000,
            "sfa_total_ms": sfa.total_ns / 1_000_000,
            "sfa_calls": sfa.count,
            "sfa_max_ms": sfa.max_ns / 1_000_000,
            "sfa_slowest_layer": sfa.slowest_layer,
            "sfa_child_total_ms": child_total_ns / 1_000_000,
            "sfa_unattributed_ms": sfa_unattributed_ns / 1_000_000,
            "non_sfa_total_ms": non_sfa_ns / 1_000_000,
            "non_sfa_segments": trace.non_sfa_segments.count,
            "non_sfa_max_ms": trace.non_sfa_segments.max_ns / 1_000_000,
            "non_sfa_slowest_segment": trace.non_sfa_segments.slowest_layer,
            "phases": phase_summary,
        }
    finally:
        _ACTIVE_TRACE.reset(token.context_token)


def log_topk_padding(topk_row: Any, invalid: int) -> None:
    return None


def step() -> None:
    return None
