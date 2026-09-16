# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only captured event probes; no Python layer callbacks on replay.

Events in a graph are reused, so these are LAST-replay observations, never
averages over a request. Read them only after the request-wide completion fence
and reject stale/capture-only timestamps using live target boundary events.
"""

import math
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
from typing import Any
from unittest.mock import patch


@dataclass
class CapturedInterval:
    name: str
    start: Any
    end: Any


class CapturedTimingUnavailable(RuntimeError):
    """Optional event timestamps are unsupported; not a model/kernel failure."""


def _captured_timestamp(event):
    read = getattr(event, "recorded_time", None)
    if read is None:
        raise CapturedTimingUnavailable("torch_npu events have no recorded_time API")
    try:
        return read()
    except RuntimeError as error:
        # A captured event can report query()==True without having a host-
        # readable recorder. Do not turn this known readback limitation into
        # a worker startup failure, or swallow unrelated NPU execution errors.
        message = str(error)
        if "507000" not in message or "event recorder null" not in message:
            raise
        raise CapturedTimingUnavailable(
            "captured Event.recorded_time is unavailable: event recorder null (507000)"
        ) from error


def verify_captured_timing_events(torch) -> None:
    """Small startup check, before loading model weights. No profiler needed."""
    value = torch.ones(1, device="npu")
    value.add_(1)  # Warm the tiny op before entering capture on its stream.
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        start.record()
        value.add_(1)
        end.record()
    graph.replay()
    torch.npu.synchronize()

    def timestamps():
        if not start.query() or not end.query():
            raise CapturedTimingUnavailable("SFA captured timing events did not complete after replay")
        return _captured_timestamp(start), _captured_timestamp(end)

    first = timestamps()
    graph.replay()
    torch.npu.synchronize()
    second = timestamps()
    if not (0 < first[0] <= first[1] < second[0] <= second[1]):
        raise CapturedTimingUnavailable(
            "SFA diagnostics require timing events refreshed by NPUGraph replay; "
            "this torch_npu/CANN combination returned stale captured timestamps"
        )
    elapsed = start.elapsed_time(end)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise CapturedTimingUnavailable("SFA captured event timing returned an invalid duration")


def probe_captured_timing_support(torch) -> dict:
    """Only a known optional timestamp limitation permits reduced diagnostics.

    Capture/replay/synchronize failures still escape and stop startup. They may
    indicate a poisoned device context and must never be treated as support
    detection. No profiler or eager model fallback is started here.
    """
    try:
        verify_captured_timing_events(torch)
    except CapturedTimingUnavailable as error:
        return {"status": "unavailable", "reason": str(error), "stages": {}}
    return {"status": "supported"}


def agree_captured_timing_support(torch, group, local: dict) -> dict:
    """Startup only: all TP ranks either install markers or omit them."""
    statuses = [local]
    if group.world_size > 1:
        statuses = [None] * group.world_size
        torch.distributed.all_gather_object(statuses, local, group=group.cpu_group)
    if any(
        not isinstance(value, dict) or value.get("status") not in ("supported", "unavailable") for value in statuses
    ):
        raise RuntimeError("Incomplete captured-timing capability agreement")
    unavailable = [(rank, value) for rank, value in enumerate(statuses) if value["status"] == "unavailable"]
    if unavailable:
        reasons = sorted({value["reason"] for _, value in unavailable})
        return {
            "status": "unavailable",
            "reason": "; ".join(reasons),
            "unsupported_ranks": [rank for rank, _ in unavailable],
            "stages": {},
        }
    return {"status": "supported"}


class GraphPhaseTiming:
    def __init__(self, *, event_factory, is_capturing, get_context):
        self.event_factory = event_factory
        self.is_capturing = is_capturing
        self.get_context = get_context
        self.intervals: dict[Any, list[CapturedInterval]] = {}
        self.patches = ExitStack()

    def observe(self, owner, method: str, name: str) -> None:
        original = getattr(owner, method)

        @wraps(original)
        def call(*args, **kwargs):
            # These methods live inside existing opaque SFA/collective ops.
            # Do not run clocks/events during Dynamo tracing, eager prefill,
            # or a staged layer's live Python retrieval.
            if not self.is_capturing():
                return original(*args, **kwargs)
            context = self.get_context()
            key = getattr(context, "staged_sfa_graph_key", None)
            if key is None:
                return original(*args, **kwargs)
            start, end = self.event_factory(), self.event_factory()
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.intervals.setdefault(key, []).append(CapturedInterval(name, start, end))
            return result

        self.patches.enter_context(patch.object(owner, method, call))

    def close(self) -> None:
        # Keep events alive as long as graphs: restoring Python hooks must NOT
        # free handles that the captured graph still references.
        self.patches.close()

    def report(self, bounds: tuple[Any, Any] | None, *, full: bool) -> dict:
        if bounds is None:
            return {"status": "unavailable", "reason": "no live decode boundary events", "stages": {}}
        left, right = (event.recorded_time() for event in bounds)
        if not (0 < left <= right):
            raise RuntimeError("Invalid SFA live decode timing boundaries")
        stages: dict[str, dict] = {}
        live: dict[str, list[CapturedInterval]] = {}
        stale = 0

        def add(name, start, end):
            duration = float(start.elapsed_time(end))
            if not math.isfinite(duration) or duration < 0:
                raise RuntimeError(f"Invalid captured duration: {name}")
            stage = stages.setdefault(name, {"calls": 0, "total_ms": 0.0, "max_ms": 0.0})
            stage["calls"] += 1
            stage["total_ms"] += duration
            stage["max_ms"] = max(stage["max_ms"], duration)

        for intervals in self.intervals.values():
            for interval in intervals:
                if not interval.start.query() or not interval.end.query():
                    stale += 1
                    continue
                start, end = interval.start.recorded_time(), interval.end.recorded_time()
                # Other graph keys, MTP reuse, and capture-only events are not
                # observations of the final target forward. Never report them.
                if not (left <= start <= end <= right):
                    stale += 1
                    continue
                add(interval.name, interval.start, interval.end)
                live.setdefault(interval.name, []).append(interval)

        # Reuse existing boundaries to expose work outside the opaque SFA ops:
        # bridge/retrieval gaps, then FFN/MoE, residual/norm and TP before the
        # next attention. These are elapsed gaps, NOT isolated FFN kernel times.
        def endpoint(name, side):
            matches = live.get(name, [])
            return getattr(matches[0], side) if len(matches) == 1 else None

        for layer in range(8):
            pre_end = endpoint(f"L{layer}.pre", "end")
            post_start = endpoint(f"L{layer}.post", "start")
            post_end = endpoint(f"L{layer}.post", "end")
            next_start = endpoint(f"L{layer + 1}.pre", "start") if layer < 7 else bounds[1]
            for phase, start, end in (
                ("pre_to_post", pre_end, post_start),
                ("after_post", post_end, next_start),
            ):
                if start is not None and end is not None and start.recorded_time() <= end.recorded_time():
                    add(f"L{layer}.{phase}", start, end)
        required = [f"L{i}.{phase}" for i in range(8) for phase in ("pre", "indexer", "select", "attention", "post")]
        required.extend(f"L{i}.{phase}" for i in range(8) for phase in ("pre_to_post", "after_post"))
        if full:
            required.extend(f"L{i}.transfer" for i in range(8))
        missing = [name for name in required if stages.get(name, {}).get("calls") != 1]
        return {
            "status": "complete" if not missing else "incomplete",
            "scope": "last target decode only; nested spans include dependencies, not isolated kernel time",
            "missing": missing,
            "excluded_stale_intervals": stale,
            "target_span_ms": float(bounds[0].elapsed_time(bounds[1])),
            "stages": stages,
        }


def install_graph_phase_timing(runner, torch, get_context, tp_group) -> GraphPhaseTiming:
    """Install before startup capture, only for the dedicated benchmark worker."""
    timing = GraphPhaseTiming(
        event_factory=lambda: torch.npu.Event(enable_timing=True),
        is_capturing=torch.npu.is_current_stream_capturing,
        get_context=get_context,
    )
    try:
        impls = runner._collect_staged_sfa_impls()
        if len(impls) != 8:
            raise RuntimeError(f"SFA graph timing expects eight target layers, got {len(impls)}")
        for index, (_, impl) in enumerate(impls):
            for method, phase in (
                ("_cross_layer_pre_compute", "pre"),
                ("indexer_select_post_process", "indexer"),
                ("_prepare_decode_sparse_indices", "select"),
                ("cross_layer_lmcache_retrieve", "transfer"),
                ("_execute_sparse_flash_attention_process", "attention"),
                ("_cross_layer_post_compute", "post"),
            ):
                timing.observe(impl, method, f"L{index}.{phase}")
        # Group methods execute within existing custom ops; no new graph splits.
        for method in ("all_reduce", "all_gather", "reduce_scatter"):
            timing.observe(tp_group, f"_{method}_out_place", f"TP.{method}")
    except BaseException:
        timing.close()
        raise
    return timing
