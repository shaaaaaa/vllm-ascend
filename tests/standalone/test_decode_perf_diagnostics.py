# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for bounded decode tracing and host-only stage timing."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def forbidden(*args, **kwargs):
    raise AssertionError("disabled/host diagnostics must not touch the device")


@pytest.fixture
def perf(monkeypatch):
    events = []
    logger = ModuleType("vllm_ascend.serving_perf")
    logger.cold_perf_enabled = lambda: True
    logger.cold_perf_device_timing_enabled = lambda: False
    logger.log_cold_perf_event = lambda event, **fields: events.append((event, fields))
    monkeypatch.setitem(sys.modules, logger.__name__, logger)
    path = ROOT / "vllm_ascend/worker/serving_perf.py"
    spec = importlib.util.spec_from_file_location("decode_perf_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.torch = SimpleNamespace(npu=SimpleNamespace(Event=forbidden, synchronize=forbidden))
    clock = SimpleNamespace(wall=100.0, thread_ns=0, process_ns=0)
    module.time = SimpleNamespace(
        perf_counter=lambda: clock.wall,
        thread_time_ns=lambda: clock.thread_ns,
        process_time_ns=lambda: clock.process_ns,
    )
    return module, clock, events


def test_chunked_prefill_does_not_consume_first_decode_trace(perf):
    module, _, events = perf
    runner = module.ServingPerfMixin()
    request = SimpleNamespace(num_prompt_tokens=16464, num_computed_tokens=0)
    runner.requests = {"r": request}
    runner._cold_perf_sample_trace_budget = {"r": 2}
    scheduled = SimpleNamespace(num_scheduled_tokens={"r": 4096}, finished_req_ids=set())
    for computed in (0, 4096, 8192, 12288, 16463):
        request.num_computed_tokens = computed
        # Even a one-token final prefill chunk is not decode.
        scheduled.num_scheduled_tokens["r"] = min(4096, 16464 - computed)
        runner._arm_cold_perf_decode_trace(scheduled)
        runner._cold_perf_sample_trace_budget["r"] = max(0, runner._cold_perf_sample_trace_budget["r"] - 1)
    assert runner._cold_perf_sample_trace_budget["r"] == 0
    assert events == []

    request.num_computed_tokens = 16464
    scheduled.num_scheduled_tokens["r"] = 2
    for remaining in (2, 1, 0, 0):
        runner._arm_cold_perf_decode_trace(scheduled)
        assert runner._cold_perf_sample_trace_budget["r"] == remaining
        assert runner._cold_perf_sample_trace_req_ids == (("r",) if remaining else ())
        runner._cold_perf_sample_trace_budget["r"] = max(0, remaining - 1)
    assert events == []


def test_decode_trace_handles_mixed_batches_pauses_and_request_retirement(perf):
    module, _, events = perf
    runner = module.ServingPerfMixin()
    runner.requests = {
        "decode": SimpleNamespace(num_prompt_tokens=100, num_computed_tokens=100),
        "prefill": SimpleNamespace(num_prompt_tokens=100, num_computed_tokens=0),
    }
    runner._cold_perf_sample_trace_budget = {}
    runner._arm_cold_perf_decode_trace(
        SimpleNamespace(num_scheduled_tokens={"decode": 1, "prefill": 80}, finished_req_ids=set())
    )
    assert runner._cold_perf_sample_trace_req_ids == ("decode",)
    runner._cold_perf_sample_trace_budget = {}  # Unscheduled batches prune the old budget.
    runner._arm_cold_perf_decode_trace(SimpleNamespace(num_scheduled_tokens={"prefill": 20}, finished_req_ids=set()))
    runner._arm_cold_perf_decode_trace(SimpleNamespace(num_scheduled_tokens={"decode": 1}, finished_req_ids=set()))
    assert runner._cold_perf_sample_trace_req_ids == ()
    runner.requests.clear()
    runner._arm_cold_perf_decode_trace(SimpleNamespace(num_scheduled_tokens={}, finished_req_ids={"decode", "prefill"}))
    assert runner._cold_perf_decode_traced_requests == set()
    assert not events


def test_disabled_trace_does_not_read_request_state_or_clocks(perf):
    module, _, events = perf
    module.cold_perf_enabled = lambda: False
    module.time = SimpleNamespace(perf_counter=forbidden, thread_time_ns=forbidden, process_time_ns=forbidden)
    runner = module.ServingPerfMixin()
    runner._arm_cold_perf_decode_trace(None)
    metrics = {}
    value = object()
    assert runner._run_cold_perf_npu_stage("test", ("r",), lambda: value, metrics=metrics) is value
    assert not vars(runner) and not metrics and not events


@pytest.mark.parametrize("failure", [False, True])
def test_host_stage_collects_wall_time_without_logging_cpu_counters_or_device_objects(perf, failure):
    module, clock, events = perf
    module.time.thread_time_ns = forbidden
    module.time.process_time_ns = forbidden
    module._ColdPerfNPUInterval = forbidden
    runner = module.ServingPerfMixin()
    metrics = {}
    error = ValueError("original failure")
    result = object()

    def operation():
        assert events == []
        clock.wall += 16
        clock.thread_ns += 240_000_000
        clock.process_ns += 1_600_000_000
        if failure:
            raise error
        return result

    if failure:
        with pytest.raises(ValueError) as caught:
            runner._run_cold_perf_npu_stage("rejection_total", ("r",), operation, metrics=metrics)
        assert caught.value is error
    else:
        assert runner._run_cold_perf_npu_stage("rejection_total", ("r",), operation, metrics=metrics) is result
    assert metrics == {"rejection_total_wall_ms": 16000}
    assert events == []
    assert not vars(runner)  # No pending device events in host mode.


def test_nested_stage_metrics_do_not_hide_unattributed_time(perf):
    module, _, events = perf
    module._log_slow_sample_invocation(
        ("r",),
        650,
        30,
        60,
        {
            "target_sampling_ms": 300,
            "rejection_total_wall_ms": 300,
            "rejection_total_thread_cpu_ms": 30,
            "mtp_draft_ms": 100,
            "mtp_graph_wall_ms": 90,
        },
    )
    assert events[-1][1]["unattributed_wall_ms"] == 250


def test_fast_samples_skip_summary_formatting(perf):
    module, _, events = perf

    class NoFormatting(dict):
        def items(self):
            forbidden()

    module._log_slow_sample_invocation(("r",), 499, 0, 0, NoFormatting())
    assert not events


def test_slow_summary_has_only_three_substages_and_no_per_stage_cpu_counters(perf):
    module, _, events = perf
    stages = {
        "target_sampling_ms": 16000,
        "rejection_total_wall_ms": 16000,
        "rejection_total_thread_cpu_ms": 240,
        "rejection_total_process_cpu_ms": 1600,
    }
    stages.update({f"rejection_part_{index}_wall_ms": 100 * index for index in range(10)})
    module._log_slow_sample_invocation(("r",), 16100, 240, 1600, stages)
    assert len(events) == 1
    event, fields = events[0]
    assert event == "decoder_sample_invocation_slow"
    assert fields["slow_substages_ms"] == {
        "rejection_part_9": 900,
        "rejection_part_8": 800,
        "rejection_part_7": 700,
    }
    assert fields["target_sampling_ms"] == 16000
    assert fields["unattributed_wall_ms"] == 100
    assert not any(name.startswith("rejection_") for name in fields)


@pytest.mark.parametrize("device_detail", [False, True])
def test_host_sampling_preserves_normal_rejection_forward(perf, device_detail):
    module, clock, events = perf
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    sample = next(n for n in cls.body if getattr(n, "name", None) == "_sample")
    recorder = []
    ns = dict(
        lmhead_tp_enable=lambda: False,
        cold_perf_device_timing_enabled=lambda: device_detail,
        set_stage_recorder=lambda value: recorder.append(value),
        reset_stage_recorder=lambda token: recorder.pop(),
    )
    exec(compile(ast.Module(body=[sample], type_ignores=[]), str(path), "exec"), ns)
    runner = module.ServingPerfMixin()
    runner.input_batch = SimpleNamespace(sampling_metadata=object())
    runner._cold_perf_sample_trace_req_ids = ("r",)
    runner._cold_perf_active_sample_stages = {}
    result = object()

    def rejection(*args):
        def kernel():
            clock.wall += 16
            return result

        if device_detail:
            return recorder[-1]("rejection_kernel", kernel, (), {})
        assert not recorder
        return kernel()

    runner.rejection_sampler = rejection
    assert ns["_sample"](runner, object(), object()) is result
    assert not recorder
    assert not events
    if device_detail:
        assert runner._cold_perf_active_sample_stages["rejection_rejection_kernel_wall_ms"] == 16000
    else:
        assert runner._cold_perf_active_sample_stages == {}


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("duration", [0.01, 16])
def test_worker_rpc_carries_decode_trace_to_async_output(perf, traced, duration):
    module, clock, events = perf
    path = ROOT / "vllm_ascend/worker/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sample_tokens")
    method.decorator_list = []
    constants = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id.startswith("_COLD_PERF_") for t in n.targets)
    ]
    module.time.perf_counter_ns = lambda: int(clock.wall * 1e9)
    if not traced:
        module.time.perf_counter = forbidden
        module.time.perf_counter_ns = forbidden
    ns = dict(time=module.time, log_cold_perf_event=module.log_cold_perf_event)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    extracted = ast.fix_missing_locations(ast.Module(body=[future, *constants, method], type_ignores=[]))
    exec(compile(extracted, str(path), "exec"), ns)
    payload = SimpleNamespace()
    output = SimpleNamespace(_model_runner_output=payload)

    def sample(_grammar):
        clock.wall += duration
        return output

    worker = SimpleNamespace(
        _raise_if_remote_fill_restart_required=lambda: None,
        model_runner=SimpleNamespace(_cold_perf_sample_trace_req_ids=("r",) if traced else (), sample_tokens=sample),
    )
    assert ns["sample_tokens"](worker, None) is output
    if traced:
        assert [event for event, _ in events] == (["decoder_sample_rpc_slow"] if duration >= 0.5 else [])
        if events:
            assert events[-1][1]["elapsed_ms"] == 16000
        for result in (output, payload):
            assert result._ascend_cold_perf_request_ids == ("r",)
            assert result._ascend_cold_perf_sample_return_ns == int(clock.wall * 1e9)
    else:
        assert not events
        assert not vars(payload)
