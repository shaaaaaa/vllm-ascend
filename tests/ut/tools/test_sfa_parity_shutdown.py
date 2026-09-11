# SPDX-License-Identifier: Apache-2.0
"""Test engine handoff/owned-process cleanup, not simulated NPU correctness."""

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest
from test_sfa_full_graph_parity import driver as driver


@pytest.fixture
def lifecycle(driver, monkeypatch):
    events = []
    identities = [{"rank": rank, "pid": 10000 + rank} for rank in range(8)]
    llm = Mock()

    def rpc(method, **kwargs):
        events.append(method)
        return identities

    llm.collective_rpc.side_effect = rpc
    stub = ModuleType("vllm")
    stub.LLM = Mock(return_value=llm)
    monkeypatch.setitem(sys.modules, "vllm", stub)
    monkeypatch.setattr(driver, "track_workers", Mock(return_value=["owned-processes"]))
    monkeypatch.setattr(driver, "run_generation", Mock(side_effect=lambda *args: events.append("generate")))
    monkeypatch.setattr(driver, "shutdown_engine", Mock(side_effect=lambda *args: events.append("shutdown")))
    args = SimpleNamespace(child="eager", devices=driver.DEFAULT_DEVICES)
    monkeypatch.setattr(driver, "engine_options", lambda args: {})
    return args, llm, identities, events


def test_success_releases_resources_and_waits_before_return(driver, lifecycle, capsys):
    args, llm, identities, events = lifecycle
    driver.run_child(args)
    assert events == ["parity_process_info", "generate", "parity_release_resources", "shutdown"]
    driver.track_workers.assert_called_once_with(identities, 8)
    driver.shutdown_engine.assert_called_once_with(llm, ["owned-processes"])
    assert "eager cleanup complete" in capsys.readouterr().out


@pytest.mark.parametrize("phase", ["track_workers", "run_generation"])
def test_failure_still_closes_engine_and_never_reports_cleanup_success(driver, lifecycle, capsys, phase):
    args, llm, _, events = lifecycle
    getattr(driver, phase).side_effect = RuntimeError("primary failure")
    with pytest.raises(RuntimeError, match="primary failure"):
        driver.run_child(args)
    assert events[-1] == "shutdown"
    assert "parity_release_resources" not in events
    driver.shutdown_engine.assert_called_once_with(llm, [] if phase == "track_workers" else ["owned-processes"])
    assert "cleanup complete" not in capsys.readouterr().out


def test_missing_release_ack_is_not_a_successful_handoff(driver, lifecycle):
    args, llm, identities, _ = lifecycle
    llm.collective_rpc.side_effect = [identities, identities[:-1]]
    with pytest.raises(RuntimeError, match="acknowledged"):
        driver.run_child(args)
    driver.shutdown_engine.assert_called_once()


def test_cleanup_failure_aborts_successful_generation(driver, lifecycle):
    args, _, _, _ = lifecycle
    driver.shutdown_engine.side_effect = RuntimeError("workers still alive")
    with pytest.raises(RuntimeError, match="workers still alive"):
        driver.run_child(args)


def test_cleanup_failure_preserves_original_generation_error(driver, lifecycle):
    args, _, _, _ = lifecycle
    driver.run_generation.side_effect = ValueError("bad generation")
    driver.shutdown_engine.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(ValueError, match="bad generation") as error:
        driver.run_child(args)
    assert "cleanup failed" in error.value.__notes__[0]


@pytest.mark.parametrize(
    "reports", [[], [{"rank": 0, "pid": -1}], [{"rank": 0, "pid": True}], [{"rank": 1, "pid": 123}]]
)
def test_invalid_worker_identities_fail_before_process_lookup(driver, monkeypatch, reports):
    lookup = Mock()
    monkeypatch.setattr(driver.psutil, "Process", lookup)
    with pytest.raises(RuntimeError, match="identities"):
        driver.track_workers(reports, 1)
    lookup.assert_not_called()


def test_real_owned_process_is_gone_before_cleanup_returns(driver):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        workers = driver.track_workers([{"rank": 0, "pid": process.pid}], 1)

        def shutdown(*, timeout):
            process.terminate()
            process.wait(timeout=timeout)

        llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown)))
        driver.shutdown_engine(llm, workers)
        assert not workers[0].is_running()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def test_real_live_process_blocks_handoff_without_killing_it(driver, monkeypatch):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        workers = driver.track_workers([{"rank": 0, "pid": process.pid}], 1)
        monkeypatch.setattr(driver, "ENGINE_SHUTDOWN_TIMEOUT", 0.05)
        with pytest.raises(RuntimeError, match="still alive"):
            driver.shutdown_engine(Mock(), workers)
        assert psutil.Process(process.pid).is_running()
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_worker_wait_runs_even_when_engine_shutdown_raises(driver, monkeypatch):
    llm = Mock()
    llm.llm_engine.engine_core.shutdown.side_effect = RuntimeError("engine error")
    wait = Mock(return_value=([], []))
    monkeypatch.setattr(driver.psutil, "wait_procs", wait)
    with pytest.raises(RuntimeError, match="engine error"):
        driver.shutdown_engine(llm, [])
    wait.assert_called_once_with([], timeout=driver.ENGINE_SHUTDOWN_TIMEOUT)


def test_exited_zombie_or_reused_pid_does_not_block_handoff(driver):
    zombie = Mock()
    zombie.is_running.return_value = True
    zombie.status.return_value = psutil.STATUS_ZOMBIE
    reused = Mock()
    reused.is_running.return_value = False
    vanished = Mock()
    vanished.is_running.return_value = True
    vanished.status.side_effect = psutil.NoSuchProcess(123)
    live = Mock()
    live.is_running.return_value = True
    live.status.return_value = psutil.STATUS_SLEEPING
    assert driver.live_workers([zombie, reused, vanished, live]) == [live]
    reused.status.assert_not_called()


def test_worker_becoming_zombie_during_wait_is_not_reported_alive(driver, monkeypatch):
    worker = Mock()
    worker.is_running.return_value = True
    worker.status.side_effect = [psutil.STATUS_SLEEPING, psutil.STATUS_ZOMBIE]
    monkeypatch.setattr(driver.psutil, "wait_procs", Mock(return_value=([], [worker])))
    driver.shutdown_engine(Mock(), [worker])
