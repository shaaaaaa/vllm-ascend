# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the serving-only stall observer; no NPU waits or inference."""

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def stall_module(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/sfa_serving_stall.py"
    spec = importlib.util.spec_from_file_location("tested_sfa_serving_stall", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def diagnostic(stall_module, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(stall_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    worker = SimpleNamespace(
        rank=5,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=1, tensor_parallel_size=4),
        ),
        model_runner=SimpleNamespace(_sfa_full_graph=SimpleNamespace(replay_count=10)),
        execute_model=Mock(return_value="hidden"),
        sample_tokens=Mock(return_value="output"),
        execute_dummy_batch=Mock(),
    )
    timing = SimpleNamespace(decode_steps=0)
    return stall_module.ServingStallDiagnostic(worker, timing), clock


def test_silent_before_any_rpc_and_before_threshold(diagnostic):
    observer, clock = diagnostic
    clock.now = 100
    assert observer.snapshot_if_stalled() is None
    observer.progress = ("execute_model.enter", clock.now)
    clock.now += 29
    assert observer.snapshot_if_stalled() is None


def test_idle_dp_stall_has_identity_stack_and_only_one_snapshot(diagnostic):
    observer, clock = diagnostic
    observer.progress = ("execute_dummy_batch.enter", 0)
    observer.worker.model_runner._sfa_full_graph.replay_count += 1
    clock.now = 31
    report = observer.snapshot_if_stalled()
    assert report["dp_rank"] == 1 and report["tp_rank"] == 1
    assert report["last_phase"] == "execute_dummy_batch.enter"
    assert report["decode_steps"] == 0  # Idle DP still needs a stall snapshot.
    assert report["root_replays_submitted"] == 1  # Not proof of device completion.
    assert any("test_idle_dp_stall" in line for thread in report["threads"] for line in thread["stack"])
    clock.now += 100
    assert observer.snapshot_if_stalled() is None


def test_async_return_is_not_assumed_to_be_device_complete(diagnostic):
    observer, clock = diagnostic

    class DeferredOutput:
        def get_output(self):
            raise AssertionError("The observer must not materialize async outputs")

        @property
        def sampled_token_ids(self):
            raise AssertionError("The observer must not inspect tensors")

    output = DeferredOutput()
    observer.worker.sample_tokens = Mock(return_value=output)
    observer.observe("sample_tokens")
    try:
        assert observer.worker.sample_tokens() is output
        clock.now = 31
        assert observer.snapshot_if_stalled()["last_phase"] == "sample_tokens.return"
    finally:
        observer.close()


def test_snapshot_includes_async_output_thread(stall_module, diagnostic, monkeypatch):
    observer, clock = diagnostic
    observer.progress = ("sample_tokens.return", 0)
    clock.now = 31

    def async_output_busy_loop():
        return sys._getframe()

    async_frame = async_output_busy_loop()
    monkeypatch.setattr(stall_module.sys, "_current_frames", lambda: {123456: async_frame})
    report = observer.snapshot_if_stalled()
    assert any("async_output_busy_loop" in line for line in report["threads"][0]["stack"])


def test_progress_while_collecting_stack_discards_snapshot(stall_module, diagnostic, monkeypatch):
    observer, clock = diagnostic
    observer.progress = ("execute_model.enter", 0)
    clock.now = 31
    original = stall_module.traceback.extract_stack

    def extract(*args, **kwargs):
        observer.progress = ("sample_tokens.enter", clock.now)
        return original(*args, **kwargs)

    monkeypatch.setattr(stall_module.traceback, "extract_stack", extract)
    assert observer.snapshot_if_stalled() is None
    assert not observer.reported


def test_original_error_propagates_and_methods_restore(diagnostic):
    observer, clock = diagnostic
    error = ValueError("original worker error")
    original = observer.worker.execute_model = Mock(side_effect=error)
    observer.observe("execute_model")
    with pytest.raises(ValueError) as caught:
        observer.worker.execute_model("scheduler")
    assert caught.value is error
    observer.close()
    observer.close()
    assert observer.worker.execute_model is original
    clock.now = 31
    assert observer.snapshot_if_stalled() is None


def test_background_loop_reports_without_rpc_or_device_calls(stall_module, diagnostic, monkeypatch, capsys):
    observer, clock = diagnostic
    observer.progress = ("execute_model.enter", 0)
    clock.now = 31
    observer.stop = SimpleNamespace(wait=Mock(side_effect=[False, True]), is_set=lambda: False)
    observer.watch()
    assert "[SFA_SERVING_STALL]" in capsys.readouterr().out
    observer.worker.execute_model.assert_not_called()
    observer.worker.sample_tokens.assert_not_called()
    observer.worker.execute_dummy_batch.assert_not_called()


def test_start_close_stops_background_thread_and_restores_methods(diagnostic):
    observer, _ = diagnostic
    original = observer.worker.execute_dummy_batch
    observer.start()
    assert observer.thread.daemon and observer.thread.is_alive()
    observer.worker.execute_dummy_batch()
    observer.close()
    assert not observer.thread.is_alive()
    assert observer.worker.execute_dummy_batch is original
    assert observer.owner_thread == threading.get_ident()


def test_partial_install_rolls_back(diagnostic):
    observer, _ = diagnostic
    original = observer.worker.execute_model
    del observer.worker.execute_dummy_batch
    with pytest.raises(AttributeError):
        observer.start()
    assert observer.worker.execute_model is original
