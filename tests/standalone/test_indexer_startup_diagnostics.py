# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/worker/dsa_shared_pool.py"
    spec = importlib.util.spec_from_file_location("diagnostic_mapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    perf = ModuleType("vllm_ascend.serving_perf")
    perf.cold_perf_enabled = lambda: True
    perf.cold_perf_device_timing_enabled = lambda: True
    perf.log_cold_perf_process_event = lambda name, **fields: events.append((name, fields))
    monkeypatch.setitem(sys.modules, perf.__name__, perf)
    stream = SimpleNamespace(synchronize=Mock())
    npu = SimpleNamespace(is_current_stream_capturing=lambda: False,
                          current_stream=lambda device: stream)
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    obj = module.MixedIndexerMetadata(2, 4, 4, "cpu", block_map=tuple(range(8)))
    obj._map_device_metadata = Mock()
    return module, obj, perf, events, stream, npu


@pytest.mark.parametrize("mode", ["off", "detail", "device", "capture"])
def test_modes_and_restoration(runtime, monkeypatch, mode):
    module, obj, perf, events, stream, npu = runtime
    perf.cold_perf_enabled = lambda: mode != "off"
    perf.cold_perf_device_timing_enabled = lambda: mode in ("device", "capture")
    npu.is_current_stream_capturing = lambda: mode == "capture"
    original = module.MixedIndexerMetadata.update
    if mode != "device":
        monkeypatch.setattr(torch.Tensor, "cpu", lambda self: pytest.fail("forbidden readback"))
    with module.diagnose_indexer_metadata("capture"):
        if mode == "off":
            assert module.MixedIndexerMetadata.update is original
        obj.update(torch.tensor([[0, 7, -1]]), torch.tensor([0, 1023, -1]))
    assert module.MixedIndexerMetadata.update is original
    obj._map_device_metadata.assert_called_once()
    assert stream.synchronize.call_count == (mode == "device")
    assert bool(events) == (mode != "off")
    if mode == "device":
        bounds = next(fields for _, fields in events if fields["stage"] == "bounds")
        assert bounds["table_range"] == [-1, 7] and not bounds["invalid"]


@pytest.mark.parametrize("table,slots", [([[8]], [0]), ([[0]], [1024])])
def test_bad_indices_fail_before_native_launch_and_restore(runtime, table, slots):
    module, obj, _, events, stream, _ = runtime
    original = module.MixedIndexerMetadata.update
    with pytest.raises(ValueError, match="exceeds mapping"):
        with module.diagnose_indexer_metadata("profiling"):
            obj.update(torch.tensor(table), torch.tensor(slots))
    assert module.MixedIndexerMetadata.update is original
    obj._map_device_metadata.assert_not_called()
    stream.synchronize.assert_not_called()
    assert any(fields.get("invalid") for _, fields in events)


def test_completion_failure_reports_stage_and_restores(runtime):
    module, obj, _, events, stream, _ = runtime
    original = module.MixedIndexerMetadata.update
    stream.synchronize.side_effect = RuntimeError("device fault")
    with pytest.raises(RuntimeError, match="device fault"):
        with module.diagnose_indexer_metadata("capture"):
            obj.update(torch.tensor([[0]]), torch.tensor([0]))
    assert module.MixedIndexerMetadata.update is original
    assert events[-1][1]["failed_stage"] == "completion"


def test_prior_device_failure_does_not_get_attributed_to_mapper(runtime, monkeypatch):
    module, obj, _, events, _, _ = runtime
    def fail_readback(self):
        raise RuntimeError("prior device fault")
    monkeypatch.setattr(torch.Tensor, "cpu", fail_readback)
    with pytest.raises(RuntimeError, match="prior device fault"):
        with module.diagnose_indexer_metadata("capture"):
            obj.update(torch.tensor([[0]]), torch.tensor([0]))
    obj._map_device_metadata.assert_not_called()
    assert events[0][1]["stage"] == "before_readback"
    assert events[-1][1]["failed_stage"] == "before_readback"
