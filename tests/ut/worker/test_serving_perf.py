# SPDX-License-Identifier: Apache-2.0
"""Exercise optional runner timing without submitting device work."""

from types import SimpleNamespace

import pytest

from vllm_ascend.worker import serving_perf as perf


def _forbidden(*args, **kwargs):
    raise AssertionError("host mode must not time or inspect device work")


def test_host_stage_delegates_without_device_or_clock_work(monkeypatch):
    monkeypatch.setattr(perf, "cold_perf_device_timing_enabled", lambda: False)
    monkeypatch.setattr(perf, "torch", SimpleNamespace(npu=SimpleNamespace(Event=_forbidden)))
    monkeypatch.setattr(perf.time, "perf_counter", _forbidden)
    runner = perf.ServingPerfMixin()
    value = object()
    assert runner._run_cold_perf_npu_stage("test", (), lambda: value) is value
    assert not vars(runner)
    with pytest.raises(ValueError, match="original failure"):
        runner._run_cold_perf_npu_stage("test", (), _raise_original)


def _raise_original():
    raise ValueError("original failure")


def test_full_device_queue_delegates_without_allocating_events(monkeypatch):
    monkeypatch.setattr(perf, "cold_perf_device_timing_enabled", lambda: True)
    monkeypatch.setattr(perf, "torch", SimpleNamespace(npu=SimpleNamespace(Event=_forbidden)))
    runner = perf.ServingPerfMixin()
    runner._cold_perf_pending_npu_intervals = [object()] * perf._COLD_PERF_MAX_PENDING_NPU_INTERVALS
    assert runner._run_cold_perf_npu_stage("test", (), lambda: 7) == 7
    assert len(runner._cold_perf_pending_npu_intervals) == perf._COLD_PERF_MAX_PENDING_NPU_INTERVALS


def test_pending_device_interval_is_not_read_or_synchronized():
    runner = perf.ServingPerfMixin()
    interval = SimpleNamespace(
        end_event=SimpleNamespace(query=lambda: False, synchronize=_forbidden),
        start_event=SimpleNamespace(elapsed_time=_forbidden),
    )
    runner._cold_perf_pending_npu_intervals = [interval]
    runner._drain_cold_perf_npu_intervals()
    assert runner._cold_perf_pending_npu_intervals == [interval]


def test_completed_device_interval_emits_existing_fields(monkeypatch):
    records = []
    monkeypatch.setattr(perf, "log_cold_perf_event", lambda event, **fields: records.append((event, fields)))
    runner = perf.ServingPerfMixin()
    interval = perf._ColdPerfNPUInterval(
        ("r",),
        "forward",
        SimpleNamespace(elapsed_time=lambda end: 125.0),
        SimpleNamespace(query=lambda: True, synchronize=_forbidden),
        130.0,
        12.0,
        15.0,
    )
    runner._cold_perf_pending_npu_intervals = [interval]
    runner._drain_cold_perf_npu_intervals()
    assert runner._cold_perf_pending_npu_intervals == []
    event, fields = records[0]
    assert event == "decoder_npu_interval_slow"
    assert fields["request_ids"] == ("r",)
    assert fields["device_elapsed_ms"] == 125.0
    assert fields["host_wall_ms"] == 130.0


def test_mtp_snapshot_helpers_use_shared_implementation():
    from vllm_ascend import diagnostic_utils
    from vllm_ascend.spec_decode import mtp_draft_diagnostics

    for name in ("cpu_snapshot", "atomic_torch_save", "tensor_layout", "snapshot_cache_components"):
        assert getattr(mtp_draft_diagnostics, name) is getattr(diagnostic_utils, name)


def test_shared_snapshot_preserves_metadata_and_cycles():
    from vllm_ascend.diagnostic_utils import cpu_snapshot

    value = {"value": [1, 2]}
    value["cycle"] = value
    assert cpu_snapshot(value) == {"value": [1, 2], "cycle": {"__cycle__": "dict"}}
