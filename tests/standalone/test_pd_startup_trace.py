# SPDX-License-Identifier: Apache-2.0
"""CPU checks for startup boundaries; no torch or vLLM runtime is imported."""

import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "vllm_ascend" / "worker" / "startup_trace.py"
SPEC = importlib.util.spec_from_file_location("pd_startup_trace_test", MODULE_PATH)
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


def owner(*, dp=2, rank=9, tp_size=4, kv=True, **fields):
    return SimpleNamespace(
        rank=rank,
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_both") if kv else None,
            parallel_config=SimpleNamespace(data_parallel_rank=dp, tensor_parallel_size=tp_size),
        ),
        **fields,
    )


@pytest.fixture
def output(monkeypatch):
    class FlushedOutput(io.StringIO):
        flush_count = 0

        def flush(self):
            self.flush_count += 1
            return super().flush()

    stream = FlushedOutput()
    monkeypatch.setattr(trace, "sys", SimpleNamespace(stderr=stream, modules=sys.modules))
    monkeypatch.setattr(trace.socket, "gethostname", lambda: "worker-host")
    monkeypatch.setattr(trace.os, "getpid", lambda: 4321)
    monkeypatch.delitem(sys.modules, "vllm.distributed.parallel_state", raising=False)
    return stream


def test_phase_records_duration_and_flushes_without_initialized_group(output, monkeypatch):
    timestamps = iter((10.0, 10.025))
    monkeypatch.setattr(trace.time, "perf_counter", lambda: next(timestamps))
    with trace.startup_phase(owner(), "kv_register", layers=80):
        assert "kv_register begin ms=0.0" in output.getvalue()
        assert " end " not in output.getvalue()
    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    assert "h=worker-host p=4321 d=2 t=1" in lines[0]
    assert "g=" not in lines[0]
    assert lines[1].endswith("kv_register end ms=25.0 layers=80")
    assert output.flush_count == 2


@pytest.mark.parametrize("value", (SimpleNamespace(), owner(kv=False)))
def test_no_kv_config_is_silent_and_does_not_query_host(value, output, monkeypatch):
    monkeypatch.setattr(trace.socket, "gethostname", lambda: pytest.fail("disabled probe queried hostname"))
    with trace.startup_phase(value, "disabled"):
        pass
    assert output.getvalue() == ""


def test_exception_identity_preserved_without_message_or_config_dump(output):
    failure = RuntimeError("secret connector detail")
    with (
        pytest.raises(RuntimeError) as caught,
        trace.startup_phase(owner(), "connector", config={"password": "secret"}),
    ):
        raise failure
    assert caught.value is failure
    assert "connector error" in output.getvalue()
    assert "exc=RuntimeError" in output.getvalue()
    assert "secret" not in output.getvalue()
    assert "password" not in output.getvalue()
    assert " end " not in output.getvalue()


def test_base_exception_is_reraised(output):
    failure = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught, trace.startup_phase(owner(), "startup"):
        raise failure
    assert caught.value is failure
    assert "exc=KeyboardInterrupt" in output.getvalue()


def test_existing_rank_read_without_runtime_calls_and_identity_reused(output, monkeypatch):
    group = SimpleNamespace(rank=9)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", SimpleNamespace(_WORLD=group))
    worker = owner(tp_rank=3)
    with trace.startup_phase(worker, "ep_barrier", ep_rank=9, ep_peers=list(range(16))):
        group.rank = 12
        worker.tp_rank = 0
        worker.vllm_config.parallel_config.data_parallel_rank = 3
    lines = output.getvalue().splitlines()
    assert all("d=2 t=3 g=9" in line for line in lines)
    assert all("ep_rank=9 ep_peers=0-15" in line for line in lines)


def test_decorator_preserves_bound_arguments_return_and_metadata(output):
    result = object()

    class Worker:
        vllm_config = owner().vllm_config
        rank = 2

        @trace.startup_stage("load_model")
        def load(self, a, *, b):
            """Original method documentation."""
            assert (a, b) == (3, 4)
            return result

    assert Worker().load(3, b=4) is result
    assert Worker.load.__name__ == "load"
    assert Worker.load.__doc__ == "Original method documentation."
    assert hasattr(Worker.load, "__wrapped__")
    assert len(output.getvalue().splitlines()) == 2


@pytest.mark.parametrize("failed_method", ("write", "flush"))
def test_broken_stderr_does_not_change_success_or_original_failure(output, monkeypatch, failed_method):
    def fail(*args):
        raise BrokenPipeError("closed diagnostic pipe")

    monkeypatch.setattr(output, failed_method, fail)
    with trace.startup_phase(owner(), "success"):
        pass
    original = ValueError("original failure")
    with pytest.raises(ValueError) as caught, trace.startup_phase(owner(), "failure"):
        raise original
    assert caught.value is original


def test_unknown_detail_objects_are_never_formatted(output):
    class DeviceLike:
        def __str__(self):
            pytest.fail("diagnostic must not format device-like object")

        def __repr__(self):
            pytest.fail("diagnostic must not repr device-like object")

    with trace.startup_phase(owner(), "kv_init\nunsafe", tensor=DeviceLike(), note="two\nlines"):
        pass
    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    assert all("kv_init_unsafe" in line and "note=two_lines" in line for line in lines)
    assert "tensor=" not in output.getvalue()


def test_owner_without_rank_and_hostname_failure_remains_safe(output, monkeypatch):
    def fail():
        raise OSError("hostname unavailable")

    monkeypatch.setattr(trace.socket, "gethostname", fail)
    worker = SimpleNamespace(vllm_config=SimpleNamespace(kv_transfer_config=object()))
    with trace.startup_phase(worker, "connector"):
        pass
    assert "h=? p=4321 d=0 t=?" in output.getvalue()
