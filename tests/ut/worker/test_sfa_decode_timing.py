# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for opt-in timing. These tests do not claim NPU performance."""

import ast
import importlib.util
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def timing_module(monkeypatch):
    path = ROOT / "vllm_ascend/worker/sfa_decode_timing.py"
    spec = importlib.util.spec_from_file_location("tested_sfa_decode_timing", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def schedule(computed, outputs, query=2, *, new=False):
    return SimpleNamespace(
        num_scheduled_tokens={"request": query},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[] if new else ["request"], num_computed_tokens=[computed], num_output_tokens=[outputs]
        ),
    )


def test_moments(timing_module):
    values = timing_module.Moments()
    for value in (0, 1, 2, 3):
        values.add(value)
    assert values.report() == {"count": 4, "total_ms": 6, "mean_ms": 1.5, "std_ms": 1.25**0.5, "max_ms": 3}
    for value in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            values.add(value)


def test_exclusive_wall_and_cpu_do_not_double_count(timing_module, monkeypatch):
    ticks = iter([0, 1_000_000, 4_000_000, 10_000_000])
    cpu = iter([0, 1_000_000, 2_000_000, 3_000_000])
    monkeypatch.setattr(timing_module.time, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(timing_module.time, "thread_time_ns", lambda: next(cpu))
    timing = timing_module.DecodeTiming()
    timing.active = True
    with timing.scope("parent"), timing.scope("child"):
        pass
    stages = timing.report()["stages"]
    assert stages["parent"]["wall"]["total_ms"] == 10
    assert stages["parent"]["self_wall"]["total_ms"] == 7
    assert stages["child"]["self_wall"]["total_ms"] == 3
    assert stages["parent"]["self_cpu"]["total_ms"] == 2


def test_disabled_and_other_threads_never_time_or_record_events(timing_module, monkeypatch):
    clock, factory = Mock(side_effect=AssertionError("No clocks")), Mock(side_effect=AssertionError("No events"))
    timing = timing_module.DecodeTiming(factory)
    monkeypatch.setattr(timing_module.time, "perf_counter_ns", clock)
    with timing.scope("prefill", device=True):
        pass
    timing.active = True
    called = []

    def background():
        with timing.scope("background", device=True):
            called.append(True)

    thread = threading.Thread(target=background)
    thread.start()
    thread.join(timeout=5)
    assert called == [True] and not thread.is_alive()
    clock.assert_not_called()
    factory.assert_not_called()
    assert not timing.scopes


def test_event_intervals_are_bounded_read_only_after_request(timing_module, monkeypatch):
    monkeypatch.setattr(timing_module, "MAX_DEVICE_INTERVALS", 2)
    events = []
    ready = False

    class Event:
        def __init__(self):
            self.recorded = False
            events.append(self)

        def record(self):
            self.recorded = True

        def elapsed_time(self, end):
            assert ready and self.recorded and end.recorded
            return 3

    timing = timing_module.DecodeTiming(Event)
    timing.active = True
    for _ in range(4):
        with timing.scope("target", device=True):
            pass
    assert len(events) == 4
    assert "stream_span" not in timing.scopes["target"]
    ready = True
    report = timing.report()
    assert report["device_intervals_dropped"] == 2
    assert report["stages"]["target"]["stream_span"]["total_ms"] == 6
    assert timing.report() == report  # Event intervals are drained only once.


@pytest.mark.parametrize(
    "computed,outputs,query,new,active",
    [
        (0, 0, 512, True, False),
        (4999, 0, 1, False, False),
        (5000, 0, 1, False, False),
        (5000, 1, 1, False, True),
        (5001, 2, 2, False, True),
    ],
)
def test_decode_gate_uses_scheduler_history_not_query_shape(timing_module, computed, outputs, query, new, active):
    timing = timing_module.DecodeTiming()
    timing.begin_step(schedule(computed, outputs, query, new=new), 5000)
    assert timing.active is active
    assert timing.decode_steps == int(active)
    assert timing.prefill_steps == int(not active)
    timing.begin_step(SimpleNamespace(num_scheduled_tokens={}), 5000)
    assert not timing.active


def test_multi_request_gate_rejected(timing_module):
    timing = timing_module.DecodeTiming()
    with pytest.raises(RuntimeError, match="exactly one"):
        timing.begin_step(SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 2}), 5000)


def actual_root_run(events):
    """Exercise the real replay/validate/fence implementation, without NPU imports."""
    path = ROOT / "vllm_ascend/compilation/sfa_full_graph.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SFAFullGraph")
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    namespace = {
        "get_forward_context": lambda: SimpleNamespace(staged_sfa_graph_key="key", cudagraph_runtime_mode="full"),
        "CUDAGraphMode": SimpleNamespace(NONE="none"),
        "torch": SimpleNamespace(
            profiler=SimpleNamespace(record_function=lambda name: nullcontext()),
            npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: events.append("fence"))),
        ),
    }
    module = ast.parse("from __future__ import annotations")
    module.body.append(run)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["run"]


def fake_worker(events, *, full=True):
    connector = SimpleNamespace(
        **{
            name: Mock()
            for name in ("prepare_sparse_graph_step", "start_load_kv", "wait_for_layer_load", "wait_for_save")
        }
    )
    graph = SimpleNamespace(
        entries={
            "key": SimpleNamespace(graph=SimpleNamespace(replay=lambda: events.append("replay")), output="hidden")
        },
        bind_sources=Mock(),
        validate_inputs=Mock(),
        replay_count=0,
    )
    graph.run = actual_root_run(events).__get__(graph)
    impls = tuple(
        (
            f"layer.{i}",
            SimpleNamespace(
                prepare_full_graph_layer=Mock(), cross_layer_lmcache_retrieve=lambda: connector.wait_for_layer_load()
            ),
        )
        for i in range(8)
    )
    runner = SimpleNamespace(
        _sfa_full_graph=graph,
        _staged_sfa_impls=impls,
        **{
            name: Mock()
            for name in (
                "_prepare_inputs",
                "_build_attention_metadata",
                "_sample",
                "_bookkeeping_sync",
                "_copy_draft_token_ids_to_cpu",
                "finalize_kv_connector",
            )
        },
    )

    def forward():
        if full:
            connector.prepare_sparse_graph_step()
            for _, impl in impls:
                impl.prepare_full_graph_layer()
            graph.bind_sources()
            graph.validate_inputs()
            return graph.run(None)
        for _, impl in impls:
            impl.cross_layer_lmcache_retrieve()
        return "hidden"

    def execute(scheduler):
        connector.start_load_kv()
        runner._prepare_inputs()
        runner._build_attention_metadata()
        return runner._model_forward()

    def sample():
        runner._sample()
        runner.propose_draft_token_ids()
        runner._copy_draft_token_ids_to_cpu()
        runner._bookkeeping_sync()
        connector.wait_for_save()
        runner.finalize_kv_connector()
        return SimpleNamespace(sampled_token_ids=[[10, 11, -1]])

    runner._model_forward = forward
    runner.propose_draft_token_ids = lambda: connector.wait_for_layer_load()
    return SimpleNamespace(model_runner=runner, execute_model=execute, sample_tokens=sample), connector


@pytest.mark.parametrize("full", [True, False])
def test_installation_calls_original_code_and_restores_all_handles(timing_module, full):
    events = []
    worker, connector = fake_worker(events, full=full)
    graph = worker.model_runner._sfa_full_graph
    original_graph = graph.entries["key"].graph
    original_execute, original_forward = worker.execute_model, worker.model_runner._model_forward
    original_validate = graph.validate_inputs
    timing = timing_module.install_decode_timing(worker, connector, prompt_tokens=5000)
    # A final prefill chunk with Q1 is still excluded, including MTP sampling.
    worker.execute_model(schedule(4999, 0, 1))
    worker.sample_tokens()
    assert not timing.scopes
    events.clear()
    for _ in range(3):
        assert worker.execute_model(schedule(5001, 2)) == "hidden"
        worker.sample_tokens()
    stages = timing.report()["stages"]
    assert stages["target.forward"]["wall"]["count"] == 3
    assert stages["kv.wait.mtp"]["wall"]["count"] == 3
    assert timing.sampled_tokens == {2: 3}
    if full:
        assert events == ["replay", "fence"] * 3
        assert stages["root.replay_submit"]["wall"]["count"] == 3
        assert stages["signature.validate"]["wall"]["count"] == 6
        assert "retrieve.L0" not in stages
        assert all(stages[f"metadata.L{i}"]["wall"]["count"] == 3 for i in range(8))
    else:
        assert not events
        assert "root.replay_submit" not in stages
        assert stages["kv.wait.target"]["wall"]["count"] == 24
        assert all(stages[f"retrieve.L{i}"]["wall"]["count"] == 3 for i in range(8))
    timing.close()
    timing.close()
    assert worker.execute_model is original_execute
    assert worker.model_runner._model_forward is original_forward
    assert graph.entries["key"].graph is original_graph
    assert graph.validate_inputs is original_validate


def test_failed_install_restores_previously_patched_methods(timing_module):
    worker, connector = fake_worker([])
    del worker.model_runner._sample
    original = worker.execute_model
    with pytest.raises(AttributeError):
        timing_module.install_decode_timing(worker, connector, prompt_tokens=5000)
    assert worker.execute_model is original


def test_scope_failure_keeps_original_exception_and_unwinds_stack(timing_module):
    timing = timing_module.DecodeTiming()
    timing.active = True
    with pytest.raises(ValueError, match="original"), timing.scope("parent"), timing.scope("child"):
        raise ValueError("original")
    assert not timing.stack
    assert set(timing.report()["stages"]) == {"parent", "child"}


def benchmark_rpc_methods(namespace):
    path = ROOT / "vllm_ascend/worker/sfa_benchmark_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SFABenchmarkWorker")
    methods = [
        n
        for n in cls.body
        if getattr(n, "name", "") in ("benchmark_start_decode_timing", "benchmark_stop_decode_timing")
    ]
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["benchmark_start_decode_timing"], namespace["benchmark_stop_decode_timing"]


@pytest.mark.parametrize("report_failure", [False, True])
def test_worker_rpcs_only_fence_outside_measurement_and_always_restore(monkeypatch, report_failure):
    events = []
    timing = SimpleNamespace(
        close=lambda: events.append("restore"),
        report=lambda: (events.append("report"), {"decode_steps": 3})[1],
    )
    if report_failure:
        timing.report = Mock(side_effect=ValueError("event error"))
    install = Mock(side_effect=lambda *a, **kw: (events.append("install"), timing)[1])
    for name, fields in {
        "vllm.distributed.kv_transfer": {"get_kv_transfer_group": lambda: "connector"},
        "vllm_ascend.worker.sfa_decode_timing": {"install_decode_timing": install},
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, stub)
    npu = SimpleNamespace(synchronize=lambda: events.append("sync"), Event=Mock())
    start, stop = benchmark_rpc_methods({"torch": SimpleNamespace(npu=npu)})
    graph = SimpleNamespace(replay_count=100, source_binding_count=5)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(_sfa_full_graph=graph, use_async_scheduling=False),
        benchmark_process_info=lambda: {"rank": 1, "pid": 123},
    )
    assert start(worker, 5000) == {"rank": 1, "pid": 123}
    assert events == ["sync", "install"]
    assert install.call_args.kwargs["prompt_tokens"] == 5000
    assert worker._decode_timing is timing
    with pytest.raises(RuntimeError, match="already active"):
        start(worker, 5000)
    graph.replay_count += 3
    graph.source_binding_count += 1
    if report_failure:
        with pytest.raises(ValueError, match="event error"):
            stop(worker)
    else:
        assert stop(worker) == {
            "decode_steps": 3,
            "rank": 1,
            "pid": 123,
            "root_replays": 3,
            "source_binding_updates": 1,
        }
    assert worker._decode_timing is None
    assert events[2:4] == ["restore", "sync"]
    assert events[-1] == "restore"


def test_actual_worker_wrapper_dispatches_to_installed_probe(timing_module):
    path = ROOT.parent / "vllm/vllm/v1/worker/worker_base.py"
    if not path.is_file():
        pytest.skip("Requires sibling vllm checkout")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WorkerWrapperBase")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    worker, connector = fake_worker([])
    timing = timing_module.install_decode_timing(worker, connector, prompt_tokens=5000)
    wrapper = SimpleNamespace(worker=worker, _apply_mm_cache=lambda scheduler: None)
    try:
        namespace["execute_model"](wrapper, schedule(5001, 2))
        assert timing.scopes["target.forward"]["wall"].count == 1
    finally:
        timing.close()
