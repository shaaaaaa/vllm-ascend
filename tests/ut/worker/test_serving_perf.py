# SPDX-License-Identifier: Apache-2.0
"""Exercise optional runner timing without submitting device work."""

import ast
from contextlib import nullcontext
from pathlib import Path
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


def _prefill_harness(monkeypatch, enabled=True, tokens=4096, mtp=True, asynchronous=False):
    """Run the actual sampling method with device/serving boundaries substituted."""
    path = Path(perf.__file__).with_name("model_runner_v1.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    method = next(n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == "sample_tokens")
    method.decorator_list = []
    clock = SimpleNamespace(now=100.0, calls=0)
    calls, records = [], []

    def now():
        clock.calls += 1
        if not enabled or tokens <= 2:
            _forbidden()
        return clock.now

    def step(name, duration, result):
        def run(*args, **kwargs):
            calls.append(name)
            clock.now += duration
            return result

        return run

    fake_time = SimpleNamespace(perf_counter=now, thread_time_ns=_forbidden, process_time_ns=_forbidden)
    monkeypatch.setattr(perf, "time", fake_time)
    monkeypatch.setattr(perf, "log_cold_perf_event", lambda event, **fields: records.append((event, fields)))
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"r": tokens}, total_num_scheduled_tokens=tokens, finished_req_ids=set()
    )
    state = (scheduler, object(), None, object(), object(), object(), None, {}, object(), None, None, None, None)
    sampler = SimpleNamespace(sampled_token_ids=[[1]], logprobs_tensors=None)

    def output(**kwargs):
        calls.append("output")
        clock.now += 0.01
        return SimpleNamespace(**kwargs)

    namespace = dict(
        time=fake_time,
        _PREFILL_PERF_INTERVAL_SECONDS=perf._PREFILL_PERF_INTERVAL_SECONDS,
        _log_prefill_sample=perf._log_prefill_sample,
        envs_ascend=SimpleNamespace(VLLM_ASCEND_MTP_DRAFT_DEBUG=False),
        record_function_or_nullcontext=lambda *a: nullcontext(),
        nullcontext=nullcontext,
        _mtp_dw_for_requests=lambda *a, **k: None,
        _mtp_dw_diag_enabled=lambda: False,
        has_kv_transfer_group=lambda: True,
        npu_content_diagnostics_enabled=lambda: False,
        ModelRunnerOutput=output,
        AsyncGPUModelRunnerOutput=lambda **k: SimpleNamespace(**k),
        get_pp_group=lambda: SimpleNamespace(world_size=1),
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    runner = SimpleNamespace(
        kv_connector_output=None,
        execute_model_state=state,
        _prefill_sample_perf=enabled,
        _prefill_sample_perf_next=0.0,
        uniform_decode_query_len=2,
        need_accepted_tokens=False,
        input_batch=SimpleNamespace(vocab_size=100, sampling_metadata=object()),
        speculative_config=SimpleNamespace(use_eagle=lambda: False, uses_draft_model=lambda: False) if mtp else None,
        drafter=SimpleNamespace(method="mtp", mtp_draft_diagnostic_scope=nullcontext),
        _sample=step("sampling", 0.1, sampler),
        _bookkeeping_sync=step("bookkeeping", 0.2, (None, [[1]], {}, ["r"], {"r": 0}, [])),
        propose_draft_token_ids=step("mtp", 0.3, [[2]]),
        _copy_draft_token_ids_to_cpu=step("readback", 0.04, None),
        finalize_kv_connector=step("finalize", 0.05, SimpleNamespace(is_empty=lambda: False)),
        model_config=SimpleNamespace(enable_return_routed_experts=False),
        supports_mm_inputs=False,
        dynamic_eplb=False,
        debugger=None,
        use_async_scheduling=asynchronous,
        async_output_copy_stream=object(),
        _run_cold_perf_npu_stage=_forbidden,
    )

    def invoke():
        runner.execute_model_state = state
        return namespace["sample_tokens"](runner, None)

    return runner, invoke, clock, calls, records


@pytest.mark.parametrize("asynchronous", [False, True])
def test_prefill_summary_preserves_sampling_order_and_measures_phases(monkeypatch, asynchronous):
    runner, invoke, clock, calls, records = _prefill_harness(monkeypatch, asynchronous=asynchronous)
    result = invoke()
    assert calls == ["sampling", "bookkeeping", "mtp", "readback", "finalize", "output"]
    assert runner.execute_model_state is None
    output = result.model_runner_output if asynchronous else result
    assert output.sampled_token_ids == [[1]]
    event, fields = records[0]
    assert event == "prefiller_sample_slow"
    assert fields["require_active"] is False
    assert fields["sample_started_monotonic_ms"] == 100000
    assert fields["total_wall_ms"] == 700
    assert [
        fields[k]
        for k in (
            "grammar_sampling_wall_ms",
            "bookkeeping_wall_ms",
            "mtp_proposal_readback_wall_ms",
            "connector_finalize_wall_ms",
            "output_tail_wall_ms",
        )
    ] == [100, 200, 340, 50, 10]
    assert fields["num_scheduled_tokens"] == 4096
    assert fields["async_scheduling"] is asynchronous
    assert clock.calls == 6


@pytest.mark.parametrize("enabled,tokens", [(False, 4096), (True, 2)])
def test_prefill_off_or_decode_adds_no_clock_work(monkeypatch, enabled, tokens):
    _, invoke, clock, calls, records = _prefill_harness(monkeypatch, enabled=enabled, tokens=tokens)
    invoke()
    assert len(calls) == 6
    assert clock.calls == 0
    assert records == []


def test_prefill_sampling_is_rate_limited_and_fast_calls_are_silent(monkeypatch):
    _, invoke, clock, _, records = _prefill_harness(monkeypatch)
    invoke()
    invoke()
    assert len(records) == 1
    clock.now = 105.0
    invoke()
    assert len(records) == 2
    _, invoke, _, calls, records = _prefill_harness(monkeypatch, mtp=False)
    invoke()
    assert calls == ["sampling", "bookkeeping", "output"]
    assert records == []


@pytest.mark.parametrize(
    "operation", ["_sample", "_bookkeeping_sync", "propose_draft_token_ids", "finalize_kv_connector"]
)
def test_prefill_timing_preserves_operation_failures(monkeypatch, operation):
    runner, invoke, _, _, records = _prefill_harness(monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("original failure")

    setattr(runner, operation, fail)
    with pytest.raises(ValueError, match="original failure"):
        invoke()
    assert records == []


def test_prefill_summary_bounds_request_ids(monkeypatch):
    records = []
    monkeypatch.setattr(perf, "time", SimpleNamespace(perf_counter=lambda: 1.0))
    monkeypatch.setattr(perf, "log_cold_perf_event", lambda event, **fields: records.append(fields))
    scheduler = SimpleNamespace(
        num_scheduled_tokens=dict.fromkeys(map(str, range(100)), 1), total_num_scheduled_tokens=100
    )
    perf._log_prefill_sample(scheduler, [0.0, 0.1, 0.2, 0.3, 0.4], False)
    assert records[0]["request_ids"] == tuple(map(str, range(8)))
    assert records[0]["num_reqs"] == 100


@pytest.mark.parametrize(
    "enabled,producer,rank", [(False, True, 0), (True, False, 0), (True, True, 1), (True, True, 0)]
)
def test_prefill_gate_is_cached_for_tp0_producers(enabled, producer, rank):
    path = Path(perf.__file__).with_name("model_runner_v1.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_prefill_sample_perf" for t in n.targets)
    )
    runner = SimpleNamespace(is_kv_producer=producer)
    namespace = dict(
        self=runner,
        cold_perf_enabled=lambda: enabled,
        get_tp_group=(lambda: SimpleNamespace(rank_in_group=rank)) if enabled and producer else _forbidden,
    )
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), namespace)
    assert runner._prefill_sample_perf is (enabled and producer and rank == 0)
