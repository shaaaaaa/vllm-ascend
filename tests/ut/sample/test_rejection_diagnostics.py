from unittest.mock import ANY, patch

import vllm_ascend.worker.model_runner_v1 as model_runner_module
from vllm_ascend.sample.rejection_diagnostics import (
    diagnostic_stage,
    reset_stage_recorder,
    set_stage_recorder,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_rejection_stage_recorder_is_scoped():
    calls = []

    def recorder(name, operation, args, kwargs):
        calls.append(name)
        return operation(*args, **kwargs)

    @diagnostic_stage("constraint")
    def operation(value):
        return value + 1

    token = set_stage_recorder(recorder)
    try:
        assert operation(1) == 2
    finally:
        reset_stage_recorder(token)

    assert operation(2) == 3
    assert calls == ["constraint"]


def test_rejection_stage_recorder_preserves_exception():
    error = RuntimeError("failure")

    def operation():
        raise error

    token = set_stage_recorder(lambda _name, function, args, kwargs: function(*args, **kwargs))
    try:
        try:
            diagnostic_stage("failure")(operation)()
        except RuntimeError as exc:
            assert exc is error
        else:
            raise AssertionError("expected operation failure")
    finally:
        reset_stage_recorder(token)


def test_deferred_npu_timing_queries_without_synchronizing():
    class FakeEvent:
        def __init__(self):
            self.recorded = False

        def record(self):
            self.recorded = True

        def query(self):
            return True

        def elapsed_time(self, end):
            assert self.recorded and end.recorded
            return 12.5

        def synchronize(self):
            raise AssertionError("diagnostics must not synchronize")

    events = []

    def make_event(**_kwargs):
        event = FakeEvent()
        events.append(event)
        return event

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._cold_perf_current_sample_npu_intervals = []
    with (
        patch.object(model_runner_module.torch.npu, "Event", side_effect=make_event),
        patch.object(model_runner_module, "log_cold_perf_event") as log_event,
    ):
        result = runner._run_cold_perf_npu_stage(
            "rejection_total", ("request",), lambda: "result"
        )
        runner._cold_perf_pending_npu_intervals[0].force_emit = True
        runner._drain_cold_perf_npu_intervals()

    assert result == "result"
    assert len(events) == 2
    assert all(event.recorded for event in events)
    assert runner._cold_perf_pending_npu_intervals == []
    log_event.assert_called_once_with(
        "decoder_npu_interval_slow",
        request_ids=("request",),
        require_active=False,
        stage="rejection_total",
        device_elapsed_ms=12.5,
        host_wall_ms=ANY,
        host_thread_cpu_ms=ANY,
        host_process_cpu_ms=ANY,
        forced_by_sample_stall=True,
    )
