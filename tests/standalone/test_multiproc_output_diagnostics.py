# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests using real output types and worker queue methods.

Load those definitions from vLLM source with AST to avoid initializing a device
runtime. Only the response transport and performance logger are substituted.
"""

import ast
import importlib.util
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from queue import Queue
from types import ModuleType
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "vllm_ascend/patch/platform/patch_multiproc_executor.py"


def compile_nodes(nodes, path, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)


def load_worker(monkeypatch, *, perf=True, scheduling=False, rank=0):
    vllm = ROOT.parent / "vllm/vllm"
    if not vllm.is_dir():
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            pytest.skip("requires sibling vLLM checkout or installed vLLM source")
        vllm = Path(spec.origin).parent
    module = ModuleType("pd_output_diagnostics_under_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = module.__dict__
    events = []
    ns.update(
        dataclass=dataclass,
        field=field,
        ClassVar=ClassVar,
        ABC=ABC,
        abstractmethod=abstractmethod,
        Enum=Enum,
        auto=auto,
        time=time,
        suppress=suppress,
        InvalidStateError=InvalidStateError,
        FutureWrapper=type("FutureWrapper", (Future,), {"wait_for_response": lambda *args: None}),
        cold_perf_enabled=lambda: perf,
        log_cold_perf_event=lambda event, **fields: events.append((event, fields)),
        log_cold_perf_process_event=lambda event, **fields: events.append((event, fields)),
    )
    outputs = vllm / "v1/outputs.py"
    tree = ast.parse(outputs.read_text(encoding="utf-8"))
    compile_nodes(
        [
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name in ("ModelRunnerOutput", "AsyncModelRunnerOutput")
        ],
        outputs,
        ns,
    )
    executor = vllm / "v1/executor/multiproc_executor.py"
    tree = ast.parse(executor.read_text(encoding="utf-8"))
    worker_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WorkerProc")
    worker_class.body = [
        n
        for n in worker_class.body
        if getattr(n, "name", None)
        in (
            "ResponseStatus",
            "handle_output",
            "enqueue_output",
        )
    ]
    worker_class.bases = []
    worker_class.decorator_list = []
    compile_nodes([worker_class], executor, ns)
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id.startswith(("_COLD_PERF_", "_SLOW_", "_worker_")) for t in node.targets
            )
            or isinstance(node, ast.FunctionDef)
            and node.name
            in (
                "_handle_output",
                "_enqueue_output",
                "_wait_for_response",
            )
            or isinstance(node, ast.If)
            and ast.unparse(node.test) == "cold_perf_enabled()"
        ):
            selected.append(node)
    compile_nodes(selected, PATCH, ns)
    worker = ns["WorkerProc"]()
    worker.rank = rank
    worker.use_async_scheduling = scheduling
    worker.async_output_queue = Queue()
    worker.worker_response_mq = Queue()
    worker.worker_response_mq.enqueue = worker.worker_response_mq.put
    return worker, ns, events


def runner_output(ns, *, wrapped=False, trace=False):
    result = ns["ModelRunnerOutput"](
        req_ids=["request"],
        req_id_to_index={"request": 0},
        sampled_token_ids=[[42]],
        kv_connector_output=object(),
    )

    class AsyncResult(ns["AsyncModelRunnerOutput"]):
        calls = 0

        def get_output(self):
            self.calls += 1
            assert self.calls == 1, "an async output may only be resolved once"
            return result

    output = AsyncResult() if wrapped else result
    if trace:
        for target in (output, result):
            setattr(target, ns["_COLD_PERF_REQUEST_IDS"], ("request",))
            setattr(target, ns["_COLD_PERF_SAMPLE_RETURN_NS"], time.perf_counter_ns() - 200_000_000)
    return output, result


def deliver(worker, output):
    worker.handle_output(output)
    if worker.use_async_scheduling:
        assert worker.worker_response_mq.empty()
        if hasattr(output, "calls"):
            assert output.calls == 0  # No premature device wait on the worker thread.
        worker.enqueue_output(worker.async_output_queue.get_nowait())
    status, result = worker.worker_response_mq.get_nowait()
    assert worker.worker_response_mq.empty()
    return status, result


@pytest.mark.parametrize("scheduling", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("rank", [0, 3])
def test_output_type_controls_unwrapping_not_trace_tags_or_scheduler_mode(
    monkeypatch, scheduling, wrapped, trace, rank
):
    worker, ns, events = load_worker(monkeypatch, scheduling=scheduling, rank=rank)
    output, payload = runner_output(ns, wrapped=wrapped, trace=trace)
    status, result = deliver(worker, output)
    assert status is worker.ResponseStatus.SUCCESS
    assert result is payload
    assert result.sampled_token_ids == [[42]]
    assert hasattr(payload, ns["_COLD_PERF_WORKER_TIMING"]) is (wrapped and trace)
    assert bool(events) is (wrapped and trace)
    if wrapped:
        assert output.calls == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_disabled_diagnostics_retain_original_methods(monkeypatch, wrapped):
    worker, ns, events = load_worker(monkeypatch, perf=False, scheduling=True)
    assert worker.handle_output.__func__ is ns["_worker_handle_output"]
    assert worker.enqueue_output.__func__ is ns["_worker_enqueue_output"]
    output, payload = runner_output(ns, wrapped=wrapped, trace=True)
    assert deliver(worker, output) == (worker.ResponseStatus.SUCCESS, payload)
    assert not events
    assert not hasattr(payload, ns["_COLD_PERF_WORKER_TIMING"])


def test_plain_output_with_existing_queue_tag_is_still_not_unwrapped(monkeypatch):
    worker, ns, events = load_worker(monkeypatch)
    output, _ = runner_output(ns, trace=True)
    setattr(output, ns["_COLD_PERF_QUEUED_NS"], time.perf_counter_ns())
    worker.enqueue_output(output)
    status, result = worker.worker_response_mq.get_nowait()
    assert status is worker.ResponseStatus.SUCCESS
    assert result is output
    assert not events


@pytest.mark.parametrize("scheduling", [False, True])
@pytest.mark.parametrize("kind", ["none", "dict", "exception"])
def test_other_rpc_results_keep_original_success_or_failure_handling(monkeypatch, scheduling, kind):
    worker, ns, events = load_worker(monkeypatch, scheduling=scheduling)
    output = {"none": None, "dict": {"ready": True}, "exception": RuntimeError("original failure")}[kind]
    if isinstance(output, Exception):
        setattr(output, ns["_COLD_PERF_REQUEST_IDS"], ("request",))
    status, result = deliver(worker, output)
    if kind == "exception":
        assert status is worker.ResponseStatus.FAILURE
        assert result == "original failure"
    else:
        assert status is worker.ResponseStatus.SUCCESS
        assert result is output
    assert not events
