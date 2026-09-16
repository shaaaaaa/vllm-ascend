# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU process-failure tests, including the real sibling vLLM RPC/monitor code.

No NPU/HCCL behavior is simulated as an inference pass. A blocked peer here is
an OS process waiting on an Event; the tests verify supervision and exit scope.
"""

import ast
import importlib.util
import logging
import multiprocessing
import multiprocessing.connection
import threading
import time
import traceback
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def fail_stop_module():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/compilation/sfa_fail_stop.py"
    spec = importlib.util.spec_from_file_location("tested_sfa_fail_stop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def executor_methods():
    path = Path(__file__).resolve().parents[4] / "vllm/vllm/v1/executor/multiproc_executor.py"
    if not path.is_file():
        pytest.skip("Requires the matching sibling vllm checkout")
    wanted = {"start_worker_monitor", "_ensure_worker_termination", "worker_busy_loop"}
    methods = [
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in methods} == wanted
    for method in methods:
        method.decorator_list = []
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    ns = dict(
        multiprocessing=multiprocessing,
        weakref=weakref,
        Thread=threading.Thread,
        time=time,
        traceback=traceback,
        logger=logging.getLogger("sfa_fail_stop_test"),
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize(
    "changes,enabled",
    [
        ({}, True),
        ({"distributed_executor_backend": "uni"}, False),
        ({"distributed_executor_backend": "ray"}, False),
        ({"distributed_executor_backend": "external_launcher"}, False),
        ({"distributed_executor_backend": object}, False),
        ({"nnodes": 2}, False),
        ({"data_parallel_size": 2}, False),
        ({"pipeline_parallel_size": 2}, False),
    ],
)
def test_only_one_local_supervised_cohort_uses_fail_stop(changes, enabled, monkeypatch):
    config = dict(distributed_executor_backend="mp", nnodes=1, data_parallel_size=1, pipeline_parallel_size=1)
    config.update(changes)
    module = fail_stop_module()
    monkeypatch.setattr(module, "parent_process", lambda: object())
    assert module.uses_local_sfa_fail_stop(SimpleNamespace(**config)) is enabled
    assert not module.uses_local_sfa_fail_stop(SimpleNamespace())


def test_main_process_cannot_opt_into_worker_termination():
    module = fail_stop_module()
    config = SimpleNamespace(
        distributed_executor_backend="mp", nnodes=1, data_parallel_size=1, pipeline_parallel_size=1
    )
    assert multiprocessing.parent_process() is None
    assert not module.uses_local_sfa_fail_stop(config)


@pytest.mark.parametrize("timer_error", [False, True])
def test_error_only_timer_and_system_exit_preserve_original_failure(monkeypatch, timer_error):
    module = fail_stop_module()
    hard_exit = Mock()
    monkeypatch.setattr(module.os, "_exit", hard_exit)
    timer = Mock()
    if timer_error:
        timer.start.side_effect = RuntimeError("cannot start new thread")
    factory = Mock(return_value=timer)
    monkeypatch.setattr(module, "Timer", factory)
    original = ValueError("rank 7 invalid source pointer")
    with pytest.raises(SystemExit) as failure:
        module.exit_failed_sfa_worker(original)
    assert failure.value.code == 1 and failure.value.__cause__ is original
    factory.assert_called_once_with(5.0, module.os._exit, args=(1,))
    assert timer.daemon is True
    timer.start.assert_called_once()
    if timer_error:
        hard_exit.assert_called_once_with(1)
    else:
        hard_exit.assert_not_called()


def test_real_rpc_loop_swallows_exception_but_not_fatal_exit():
    loop = executor_methods()["worker_busy_loop"]
    calls = []

    def execute():
        calls.append("execute")
        raise ValueError("ordinary RPC error")

    worker = SimpleNamespace(
        rank=0,
        worker=SimpleNamespace(execute=execute),
        rpc_broadcast_mq=SimpleNamespace(dequeue=Mock(side_effect=[("execute", (), {}, None), SystemExit(0)])),
        handle_output=Mock(),
    )
    with pytest.raises(SystemExit):
        loop(worker)
    assert calls == ["execute"]
    assert worker.rpc_broadcast_mq.dequeue.call_count == 2
    assert isinstance(worker.handle_output.call_args.args[0], ValueError)

    worker.worker.execute = Mock(side_effect=SystemExit(1))
    worker.rpc_broadcast_mq.dequeue = Mock(return_value=("execute", (), {}, None))
    worker.handle_output.reset_mock()
    with pytest.raises(SystemExit) as failure:
        loop(worker)
    assert failure.value.code == 1
    worker.rpc_broadcast_mq.dequeue.assert_called_once()
    worker.handle_output.assert_not_called()


def blocked_worker(rank, failing_rank, ready, go, never, stall_cleanup):
    # Only CPU imports; run actual catch-and-continue RPC handling, then actual
    # fail-stop on the failed rank while its peers are deliberately blocked.
    loop = executor_methods()["worker_busy_loop"]
    module = fail_stop_module()
    module.SFA_FATAL_CLEANUP_TIMEOUT_SECONDS = 0.2  # Test clock, production is 5s.
    ready.set()
    if not go.wait(20):
        raise RuntimeError("test did not start")

    def execute():
        if rank == failing_rank:
            module.exit_failed_sfa_worker(ValueError(f"injected bad source on rank {rank}"))
        never.wait()

    worker = SimpleNamespace(
        rank=rank,
        worker=SimpleNamespace(execute=execute),
        rpc_broadcast_mq=SimpleNamespace(dequeue=lambda **kw: ("execute", (), {}, 0)),
        handle_output=lambda output: None,
    )
    try:
        loop(worker)
    finally:
        if rank == failing_rank and stall_cleanup:
            never.wait()  # Models a runtime/connector finalizer that cannot finish.


def unrelated_worker(ready, release):
    ready.set()
    release.wait()


@pytest.mark.parametrize("workers,stall_cleanup", [(2, False), (8, False), (8, True)])
def test_real_supervisor_stops_owned_blocked_peers_on_nonzero_rank_failure(workers, stall_cleanup):
    methods = executor_methods()
    context = multiprocessing.get_context("spawn")
    go, never = context.Event(), context.Event()
    ready = [context.Event() for _ in range(workers)]
    processes = [
        context.Process(target=blocked_worker, args=(rank, workers - 1, ready[rank], go, never, stall_cleanup))
        for rank in range(workers)
    ]
    unrelated_ready, unrelated_release = context.Event(), context.Event()
    unrelated = context.Process(target=unrelated_worker, args=(unrelated_ready, unrelated_release))
    stopped = threading.Event()

    class Supervisor:
        shutting_down = False
        is_failed = False
        failure_callback = staticmethod(stopped.set)

        def shutdown(self):
            self.shutting_down = True
            methods["_ensure_worker_termination"](processes)

    supervisor = Supervisor()
    supervisor.workers = [SimpleNamespace(proc=process) for process in processes]
    try:
        unrelated.start()
        for process in processes:
            process.start()
        assert unrelated_ready.wait(15)
        deadline = time.monotonic() + 30
        assert all(event.wait(max(0, deadline - time.monotonic())) for event in ready)
        methods["start_worker_monitor"](supervisor)
        go.set()
        assert stopped.wait(20), "failed rank did not wake the existing supervisor"
        for process in processes:
            process.join(timeout=3)
        assert supervisor.is_failed and all(not process.is_alive() for process in processes)
        assert processes[-1].exitcode == 1
        assert unrelated.is_alive(), "supervisor must not terminate unrelated processes"
    finally:
        # Do not touch synchronization objects used by forcibly terminated
        # peers: a killed process can leave their shared locks acquired.
        unrelated_release.set()
        for process in [*processes, unrelated]:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
