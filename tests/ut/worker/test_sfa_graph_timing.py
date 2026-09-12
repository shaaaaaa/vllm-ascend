# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for captured probes, not proof of NPU timing support."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def module(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/sfa_graph_timing.py"
    spec = importlib.util.spec_from_file_location("tested_sfa_graph_timing", path)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, result)
    spec.loader.exec_module(result)
    return result


class Event:
    def __init__(self, stamp=0, ready=True):
        self.stamp, self.ready = stamp, ready
        self.records = 0

    def record(self):
        self.records += 1

    def query(self):
        return self.ready

    def recorded_time(self):
        assert self.ready, "Never read unrecorded capture-only events"
        return self.stamp * 1_000_000

    def elapsed_time(self, end):
        return (end.recorded_time() - self.recorded_time()) / 1_000_000


def collector(module, *, capturing=False, key="key"):
    state = SimpleNamespace(capturing=capturing, staged_sfa_graph_key=key)
    timing = module.GraphPhaseTiming(
        event_factory=Event, is_capturing=lambda: state.capturing, get_context=lambda: state
    )
    return timing, state


def populate(module, timing, *, full=True):
    intervals = []
    for i in range(8):
        phases = {"pre": (10, 40), "indexer": (15, 20), "select": (25, 30), "attention": (45, 50), "post": (44, 60)}
        if full:
            phases["transfer"] = (41, 43)
        for phase, (start, end) in phases.items():
            intervals.append(module.CapturedInterval(f"L{i}.{phase}", Event(i * 100 + start), Event(i * 100 + end)))
    timing.intervals["key"] = intervals
    return Event(1), Event(800)


def test_records_only_inside_existing_capture_and_keeps_original_result(module):
    timing, state = collector(module)
    owner = SimpleNamespace(operation=Mock(return_value=object()))
    original = owner.operation
    timing.observe(owner, "operation", "L0.pre")
    try:
        assert owner.operation(7, named=3) is original.return_value
        assert not timing.intervals
        state.capturing, state.staged_sfa_graph_key = True, None
        owner.operation()
        assert not timing.intervals
        state.staged_sfa_graph_key = "key"
        assert owner.operation(7, named=3) is original.return_value
        interval = timing.intervals["key"][0]
        assert interval.start.records == interval.end.records == 1
        original.assert_called_with(7, named=3)
    finally:
        timing.close()
    assert owner.operation is original
    # Restoring Python patches must not drop handles referenced by live graphs.
    assert timing.intervals["key"][0] is interval


def test_capture_exception_is_not_swallowed_and_hooks_restore(module):
    timing, _ = collector(module, capturing=True)
    error = ValueError("original")
    owner = SimpleNamespace(operation=Mock(side_effect=error))
    original = owner.operation
    timing.observe(owner, "operation", "L0.pre")
    try:
        with pytest.raises(ValueError) as caught:
            owner.operation()
        assert caught.value is error
        assert not timing.intervals
    finally:
        timing.close()
    assert owner.operation is original


@pytest.mark.parametrize("full", [False, True])
def test_reports_only_last_live_decode_and_exposes_non_attention_gaps(module, full):
    timing, _ = collector(module)
    bounds = populate(module, timing, full=full)
    timing.intervals["unused"] = [
        module.CapturedInterval("L0.pre", Event(-100), Event(-90)),
        module.CapturedInterval("L0.post", Event(901), Event(902)),
        module.CapturedInterval("L0.select", Event(20, ready=False), Event(30, ready=False)),
    ]
    report = timing.report(bounds, full=full)
    assert report["status"] == "complete"
    assert report["excluded_stale_intervals"] == 3
    assert report["stages"]["L0.pre"] == {"calls": 1, "total_ms": 30, "max_ms": 30}
    assert report["stages"]["L0.pre_to_post"]["total_ms"] == 4
    assert report["stages"]["L0.after_post"]["total_ms"] == 50
    assert report["stages"]["L7.after_post"]["total_ms"] == 40
    assert "last target decode only" in report["scope"]
    assert ("L0.transfer" in report["stages"]) is full


@pytest.mark.parametrize("failure", ["missing", "duplicate", "stale", "ordering"])
def test_missing_or_duplicate_layers_are_not_a_success(module, failure):
    timing, _ = collector(module)
    bounds = populate(module, timing)
    interval = next(value for value in timing.intervals["key"] if value.name == "L3.pre")
    if failure == "missing":
        timing.intervals["key"].remove(interval)
    elif failure == "duplicate":
        timing.intervals["key"].append(interval)
    elif failure == "stale":
        interval.start.stamp, interval.end.stamp = -20, -10
    else:
        interval.end.stamp = 350  # Past L3.post entry at 344.
    result = timing.report(bounds, full=True)
    assert result["status"] == "incomplete"
    assert "L3.pre_to_post" in result["missing"]


def test_full_requires_graph_internal_transfer_and_no_bounds_is_unavailable(module):
    timing, _ = collector(module)
    bounds = populate(module, timing, full=False)
    assert timing.report(None, full=True)["status"] == "unavailable"
    result = timing.report(bounds, full=True)
    assert result["missing"] == [f"L{i}.transfer" for i in range(8)]


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_invalid_durations_rejected(module, duration):
    timing, _ = collector(module)
    bounds = populate(module, timing)
    timing.intervals["key"][0].start.elapsed_time = lambda end: duration
    with pytest.raises(RuntimeError, match="Invalid captured duration"):
        timing.report(bounds, full=True)


def test_hooks_collectives_inside_opaque_op_not_dynamo_visible_public_method(module):
    impls = []
    methods = (
        "_cross_layer_pre_compute",
        "indexer_select_post_process",
        "_prepare_decode_sparse_indices",
        "cross_layer_lmcache_retrieve",
        "_execute_sparse_flash_attention_process",
        "_cross_layer_post_compute",
    )
    for i in range(8):
        impls.append((str(i), SimpleNamespace(**{method: Mock() for method in methods})))
    group = SimpleNamespace(
        **{f"_{method}_out_place": Mock() for method in ("all_reduce", "all_gather", "reduce_scatter")}
    )
    group.all_reduce = public = Mock()
    original = group._all_reduce_out_place
    torch = SimpleNamespace(npu=SimpleNamespace(Event=lambda **kw: Event(), is_current_stream_capturing=lambda: True))
    runner = SimpleNamespace(_collect_staged_sfa_impls=lambda: impls)
    timing = module.install_graph_phase_timing(
        runner, torch, lambda: SimpleNamespace(staged_sfa_graph_key="key"), group
    )
    try:
        assert group.all_reduce is public
        group._all_reduce_out_place("tensor")
        assert timing.intervals["key"][0].name == "TP.all_reduce"
        original.assert_called_once_with("tensor")
    finally:
        timing.close()
    assert group._all_reduce_out_place is original
    # Fail late in installation: all earlier patches must still be undone.
    del group._reduce_scatter_out_place
    with pytest.raises(AttributeError):
        module.install_graph_phase_timing(runner, torch, lambda: None, group)
    assert group._all_reduce_out_place is original
    assert isinstance(impls[0][1]._cross_layer_pre_compute, Mock)


@pytest.mark.parametrize("refresh,ready", [(False, True), (True, True), (True, False)])
def test_startup_smoke_detects_events_that_do_not_refresh_on_replay(module, refresh, ready):
    state = SimpleNamespace(events=[], stamp=0, replays=0, fenced=False)

    class CapturedEvent(Event):
        def record(self):
            state.events.append(self)

        def recorded_time(self):
            assert state.fenced, "Startup timestamps may only be read after a fence"
            return super().recorded_time()

    def replay():
        state.fenced = False
        if refresh or state.replays == 0:
            for event in state.events:
                state.stamp += 1
                event.stamp = state.stamp
        state.replays += 1

    @contextmanager
    def capture(graph):
        yield

    torch = SimpleNamespace(
        ones=lambda *a, **kw: SimpleNamespace(add_=Mock()),
        npu=SimpleNamespace(
            Event=lambda **kw: CapturedEvent(ready=ready),
            NPUGraph=lambda: SimpleNamespace(replay=replay),
            graph=capture,
            synchronize=lambda: setattr(state, "fenced", True),
        ),
    )
    if not ready:
        with pytest.raises(RuntimeError, match="did not complete"):
            module.verify_captured_timing_events(torch)
        assert state.replays == 1
        return
    if refresh:
        module.verify_captured_timing_events(torch)
    else:
        with pytest.raises(RuntimeError, match="stale captured timestamps"):
            module.verify_captured_timing_events(torch)
    assert state.replays == 2
