# SPDX-License-Identifier: Apache-2.0
"""Exercise tool-only startup diagnostics without importing torch or an NPU."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "layerwise_prefill_profile_worker.py"
SPEC = importlib.util.spec_from_file_location("prefill_profile_startup_test_worker", MODULE_PATH)
capture_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_tool)


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    events = []
    streams = []

    def arm(timeout, *, repeat, file):
        assert timeout == 120
        assert repeat is False
        assert not file.closed
        streams.append(file)
        events.append("arm")

    def cancel():
        assert not streams[-1].closed
        events.append("cancel")

    monkeypatch.setattr(
        capture_tool,
        "faulthandler",
        SimpleNamespace(dump_traceback_later=arm, cancel_dump_traceback_later=cancel),
    )
    monkeypatch.setattr(capture_tool.os, "getpid", lambda: 5678)
    monkeypatch.setattr(capture_tool, "synchronize_boundary", lambda: events.append("sync"))
    monkeypatch.setattr(
        capture_tool,
        "install_transfer_attribution",
        lambda: SimpleNamespace(restore=lambda: events.append("restore_attribution")),
    )
    output = object()

    def execute(scheduler_output):
        events.append("execute")
        return output

    def profile(*, is_start, profile_prefix=None):
        events.append("profile_start" if is_start else "profile_stop")

    worker = SimpleNamespace(rank=2, execute_model=execute, profile=profile, profiler=None)
    plan = capture_tool.make_capture_plan(8, 4)
    plan["startup_diagnostic_dir"] = str(tmp_path / "startup-stacks")
    return SimpleNamespace(worker=worker, plan=plan, events=events, streams=streams, output=output)


def test_first_execution_logs_before_sync_and_cancels_after_return(runtime, monkeypatch, capsys):
    original_execute = runtime.worker.execute_model
    capture_tool.install_chunk_profile(runtime.worker, "case", runtime.plan)
    stream = runtime.streams[0]
    assert Path(stream.name).name == "worker-2-5678.log"

    def sync():
        visible = capsys.readouterr().out
        assert "first execute_model received; scheduled_tokens=4" in visible
        assert "rank=2 pid=5678" in visible
        assert "boundary sync begin; action=start" in visible
        assert "boundary sync end" not in visible
        assert "profiler start; chunk=1" not in visible
        runtime.events.append("sync")

    monkeypatch.setattr(capture_tool, "synchronize_boundary", sync)
    assert runtime.worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=4)) is runtime.output
    remaining = capsys.readouterr().out
    assert remaining.index("boundary sync end") < remaining.index("profiler start; chunk=1")
    assert runtime.events == ["arm", "sync", "profile_start", "execute", "cancel"]
    assert stream.closed

    # The second chunk neither arms nor cancels the process-wide watchdog again.
    runtime.worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))
    assert "first execute_model received" not in capsys.readouterr().out
    assert runtime.events.count("arm") == runtime.events.count("cancel") == 1

    def stop_sync():
        assert "boundary sync begin; action=stop" in capsys.readouterr().out

    monkeypatch.setattr(capture_tool, "synchronize_boundary", stop_sync)
    report = capture_tool.finish_chunk_profile(runtime.worker)
    assert "boundary sync end; action=stop" in capsys.readouterr().out
    assert report["windows"] == ["all"]
    assert runtime.worker.execute_model is original_execute
    assert runtime.events.count("cancel") == 1


def test_zero_token_first_execution_is_visible_and_cleans_watchdog(runtime, capsys):
    capture_tool.install_chunk_profile(runtime.worker, "case", runtime.plan)
    runtime.worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=0))
    assert "first execute_model received; scheduled_tokens=0" in capsys.readouterr().out
    assert runtime.events == ["arm", "execute", "cancel"]
    assert runtime.streams[0].closed
    capture_tool.finish_chunk_profile(runtime.worker)


@pytest.mark.parametrize("failure_point", ["sync", "profile", "execute"])
def test_first_execution_failure_cancels_and_closes_watchdog(runtime, monkeypatch, failure_point):
    def fail(*args, **kwargs):
        raise RuntimeError(failure_point)

    if failure_point == "sync":
        monkeypatch.setattr(capture_tool, "synchronize_boundary", fail)
    elif failure_point == "profile":
        runtime.worker.profile = fail
    else:
        runtime.worker.execute_model = fail
    capture_tool.install_chunk_profile(runtime.worker, "case", runtime.plan)
    with pytest.raises(RuntimeError, match=failure_point):
        runtime.worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))
    assert runtime.events.count("cancel") == 1
    assert runtime.streams[0].closed
    capture_tool.finish_chunk_profile(runtime.worker)
    assert runtime.events.count("cancel") == 1


def test_finish_before_first_execution_cleans_watchdog(runtime):
    original_execute = runtime.worker.execute_model
    capture_tool.install_chunk_profile(runtime.worker, "case", runtime.plan)
    capture_tool.finish_chunk_profile(runtime.worker)
    assert runtime.events == ["arm", "cancel"]
    assert runtime.streams[0].closed
    assert runtime.worker.execute_model is original_execute


def test_finish_failure_still_cleans_watchdog_and_restores_worker(runtime):
    original_execute = runtime.worker.execute_model
    capture_tool.install_chunk_profile(runtime.worker, "case", runtime.plan)

    def fail_stop():
        raise RuntimeError("stop failed")

    runtime.worker._prefill_chunk_profile_capture.stop_window = fail_stop
    with pytest.raises(RuntimeError, match="stop failed"):
        capture_tool.finish_chunk_profile(runtime.worker)
    assert runtime.events == ["arm", "cancel"]
    assert runtime.streams[0].closed
    assert runtime.worker.execute_model is original_execute
    assert not hasattr(runtime.worker, "_prefill_chunk_profile_capture")


def test_watchdog_arm_failure_closes_file_and_preserves_worker(runtime, monkeypatch):
    original_execute = runtime.worker.execute_model

    def fail_arm(timeout, *, repeat, file):
        runtime.streams.append(file)
        raise RuntimeError("watchdog unavailable")

    monkeypatch.setattr(capture_tool.faulthandler, "dump_traceback_later", fail_arm)
    with pytest.raises(RuntimeError, match="watchdog unavailable"):
        capture_tool.install_chunk_profile(runtime.worker, "case", runtime.plan)
    assert runtime.streams[0].closed
    assert runtime.worker.execute_model is original_execute
    assert not hasattr(runtime.worker, "_prefill_chunk_profile_capture")


def test_legacy_plan_does_not_arm_watchdog(runtime):
    del runtime.plan["startup_diagnostic_dir"]
    capture_tool.install_chunk_profile(runtime.worker, "case", runtime.plan)
    runtime.worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))
    capture_tool.finish_chunk_profile(runtime.worker)
    assert not runtime.streams
    assert "arm" not in runtime.events
    assert "cancel" not in runtime.events
