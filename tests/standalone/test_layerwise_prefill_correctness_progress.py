# SPDX-License-Identifier: Apache-2.0
"""Deterministic watchdog checks without devices or long-running timers."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "correctness_progress_test", ROOT / "tools/layerwise_prefill_correctness_progress.py"
)
assert SPEC and SPEC.loader
progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progress)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _monitor(tmp_path, rank=1):
    clock, logs, dumps = Clock(), [], []

    def dump(*, file, all_threads):
        dumps.append(all_threads)
        file.write("all Python thread stacks\n")

    monitor = progress.ProgressMonitor(tmp_path / f"rank{rank}", rank, clock=clock, log=logs.append, stack_dumper=dump)
    return monitor, clock, logs, dumps


def test_rank_local_progress_and_one_stack_per_stuck_operation(tmp_path):
    monitor, clock, logs, dumps = _monitor(tmp_path)
    monitor.begin_call("execute_model")
    monitor.begin_record(step=2, layer=47, kind="kv_loaded", name="nope")
    clock.now = 30
    monitor.tick()
    assert not logs and not dumps
    state = json.loads((monitor.directory / "progress.json").read_text())
    assert state["phase"] == "copy_cpu" and state["phase_elapsed_seconds"] == 30
    assert state["operation"] == "execute_model" and state["step"] == 2 and state["layer"] == 47
    clock.now = 90
    monitor.tick()
    assert dumps == [True] and len(logs) == 1 and "stacks.txt" in logs[0]
    assert "kv_loaded/nope" in logs[0] and len(logs[0]) <= 100
    original_stack = (monitor.directory / "stacks.txt").read_text()
    clock.now = 180
    monitor.tick()
    assert len(logs) == len(dumps) == 1
    assert (monitor.directory / "stacks.txt").read_text() == original_stack
    monitor.phase("save")
    clock.now = 270
    monitor.tick()
    assert len(dumps) == len(logs) == 2
    text = (monitor.directory / "stacks.txt").read_text()
    assert '"phase":"save"' in text and '"phase":"copy_cpu"' not in text


def test_rank_zero_periodic_messages_are_short_and_report_counters(tmp_path):
    monitor, clock, logs, _ = _monitor(tmp_path, rank=0)
    monitor.begin_record(step=0, layer=1, kind="decoder", name="output")
    monitor.record_done(14, 14, 123456789)
    monitor.tick()
    assert not logs
    clock.now = 30
    monitor.tick()
    monitor.tick()
    assert len(logs) == 1 and "n=14" in logs[0] and "b=123456789" in logs[0]
    assert len(logs[0]) <= 100
    state = json.loads((monitor.directory / "progress.json").read_text())
    assert state["records"] == state["files"] == 14 and state["bytes"] == 123456789
    assert state["phase"] == "compute"


def test_stall_clock_resets_on_reentry_to_same_phase(tmp_path):
    monitor, clock, logs, dumps = _monitor(tmp_path)
    monitor.phase("compute")
    clock.now = 89
    monitor.phase("compute")
    clock.now = 90
    monitor.tick()
    assert not logs and not dumps
    clock.now = 179
    monitor.tick()
    assert len(dumps) == 1


def test_failure_preserves_inner_phase_even_when_outer_call_also_fails(tmp_path):
    monitor, clock, _, _ = _monitor(tmp_path)
    monitor.begin_call("execute_model")
    monitor.begin_record(step=0, layer=3, kind="attention", name="output")
    monitor.phase("save")
    clock.now = 17
    monitor.fail(OSError("disk full"), records=10, files=9, size=900)
    monitor.fail(RuntimeError("outer forward aborted"))
    monitor.end_call("execute_model")
    monitor.finish()
    clock.now = 500
    state = monitor.snapshot()
    assert state["status"] == "failed" and state["phase"] == "save"
    assert state["error"] == "disk full" and state["phase_elapsed_seconds"] == 17
    assert (state["records"], state["files"], state["bytes"]) == (10, 9, 900)
    assert json.loads((monitor.directory / "progress.json").read_text()) == state


def test_execute_and_sample_boundaries_remain_visible_outside_records(tmp_path):
    monitor, _, _, _ = _monitor(tmp_path)
    monitor.begin_call("execute_model")
    assert monitor.snapshot()["phase"] == "compute"
    monitor.end_call("execute_model")
    assert monitor.snapshot()["phase"] == "await_sample"
    monitor.begin_call("sample_tokens")
    assert monitor.snapshot()["operation"] == "sample_tokens"
    monitor.end_call("sample_tokens")
    assert monitor.snapshot()["operation"] == "sample_tokens_done"
    assert monitor.snapshot()["phase"] == "running"


def test_monitor_thread_stops_without_waiting_for_poll_interval(tmp_path):
    monitor, _, _, _ = _monitor(tmp_path)
    monitor.start()
    assert monitor._thread.is_alive()
    with pytest.raises(RuntimeError, match="already started"):
        monitor.start()
    monitor.finish()
    assert not monitor._thread.is_alive()
    assert monitor.snapshot()["status"] == "finished"
    monitor.stop()  # Idempotent cleanup from archive.close / finish RPC.


def test_real_faulthandler_writes_all_thread_snapshot_without_global_watchdog(tmp_path):
    clock = Clock()
    monitor = progress.ProgressMonitor(tmp_path, 1, clock=clock, log=lambda _: None)
    monitor.begin_call("sample_tokens")
    clock.now = 90
    monitor.tick()
    text = (tmp_path / "stacks.txt").read_text()
    assert "test_real_faulthandler" in text and '"operation":"sample_tokens"' in text


def test_diagnostic_io_failure_does_not_mask_model_failure(tmp_path, monkeypatch):
    monitor, _, logs, _ = _monitor(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("diagnostic disk is unavailable")

    monkeypatch.setattr(progress.os, "replace", fail)
    monitor.begin_call("sample_tokens")
    monitor.fail(ValueError("model error"))
    monitor.publish()
    assert monitor.snapshot()["error"] == "model error"
    assert len(logs) == 1 and "diagnostic I/O failed" in logs[0]


def test_failed_progress_file_and_closed_stdout_preserve_original_error(tmp_path, monkeypatch):
    def closed_log(_):
        raise BrokenPipeError("worker stdout closed")

    def failed_replace(*args):
        raise OSError("diagnostic disk unavailable")

    monitor = progress.ProgressMonitor(tmp_path, 1, log=closed_log)
    monkeypatch.setattr(progress.os, "replace", failed_replace)
    monitor.begin_call("sample_tokens")
    monitor.fail(ValueError("original model error"))
    assert monitor.snapshot()["status"] == "failed"
    assert monitor.snapshot()["error"] == "original model error"


@pytest.mark.parametrize("ending", ["finished", "failed"])
def test_monitor_publish_cannot_overwrite_terminal_state_with_older_snapshot(tmp_path, monkeypatch, ending):
    monitor, clock, logs, dumps = _monitor(tmp_path)
    monitor.begin_record(step=0, layer=3, kind="decoder", name="input")
    original_replace = progress.os.replace
    interleaved = False

    def replace(source, destination):
        nonlocal interleaved
        if not interleaved:
            interleaved = True
            if ending == "finished":
                monitor.finish()
            else:
                monitor.fail(ValueError("original failure"))
        original_replace(source, destination)

    monkeypatch.setattr(progress.os, "replace", replace)
    clock.now = 100
    monitor.tick()
    state = json.loads((monitor.directory / "progress.json").read_text())
    assert state["status"] == ending
    if ending == "failed":
        assert state["error"] == "original failure" and state["phase"] == "copy_cpu"
    assert not dumps and not logs
