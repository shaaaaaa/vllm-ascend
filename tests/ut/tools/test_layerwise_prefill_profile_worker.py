# SPDX-License-Identifier: Apache-2.0
"""CPU worker/RPC boundary tests, with device/profiler APIs replaced by fakes."""

import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    return importlib.import_module("layerwise_prefill_profile_worker")


class Worker:
    rank = 0

    def __init__(self):
        self.profiler = None
        self.active = None
        self.events = []

    def profile(self, is_start=True, profile_prefix=None):
        if is_start:
            # Match real Worker.profile: restart reuses the old name unless
            # the tool actually clears the real worker's profiler attribute.
            self.profiler = self.profiler or profile_prefix
            self.active = self.profiler
            self.events.append(("start", self.active))
        else:
            self.events.append(("stop", self.active))
            self.active = None

    def execute_model(self, scheduler_output, **kwargs):
        self.events.append(("compute", self.active, scheduler_output.total_num_scheduled_tokens, kwargs))
        return "result"


class Wrapper:
    """The real multiproc executor passes a read-forwarding WorkerWrapperBase."""

    def __init__(self, worker):
        self.worker = worker
        self.mm_cache_calls = 0

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def execute_model(self, scheduler_output, **kwargs):
        self.mm_cache_calls += 1
        return self.worker.execute_model(scheduler_output, **kwargs)


@pytest.mark.parametrize("length", [95000, 100000, 24 * 4096, 4096, 6 * 4096, 6 * 4096 + 1])
@pytest.mark.parametrize("wrapped", [False, True])
def test_capture_first_and_last_chunks_only(module, monkeypatch, length, wrapped):
    worker = Worker()
    receiver = Wrapper(worker) if wrapped else worker
    original = worker.execute_model
    fences = []
    monkeypatch.setattr(module, "synchronize_boundary", lambda: fences.append(len(worker.events)))
    plan = module.make_capture_plan(length, 4096)
    module.install_chunk_profile(receiver, "100k_on", plan)
    assert worker.profiler is None  # no startup capture
    assert not (wrapped and "profiler" in vars(receiver))
    # Idle scheduler iterations must not consume chunk indices.
    receiver.execute_model(NS(total_num_scheduled_tokens=0))
    for start in range(0, length, 4096):
        result = receiver.execute_model(NS(total_num_scheduled_tokens=min(4096, length - start)), test_arg=True)
        assert result == "result"
        worker.events.append(("sample_mtp", worker.active))
    report = module.finish_chunk_profile(receiver)
    module.validate_capture(plan, [report])
    assert worker.execute_model == original
    assert worker.profiler is None and worker.active is None
    assert not hasattr(worker, "_prefill_chunk_profile_capture")
    if wrapped:
        assert receiver.mm_cache_calls == plan["total_chunks"] + 1
    selected = [c["chunk"] for c in report["chunks"] if c["window"]]
    total = plan["total_chunks"]
    assert selected == sorted(set(range(1, min(3, total) + 1)) | set(range(max(1, total - 2), total + 1)))
    compute = [e for e in worker.events if e[0] == "compute" and e[2] > 0]
    sample = [e for e in worker.events if e[0] == "sample_mtp"]
    for i, (execution, sampling) in enumerate(zip(compute, sample), start=1):
        assert bool(execution[1]) == (i in selected)
        assert sampling[1] == execution[1]  # includes final chunk sampling/MTP
        assert execution[3] == {"test_arg": True}
    assert [e[1] for e in worker.events if e[0] == "start"] == [f"100k_on_{w['name']}" for w in plan["windows"]]
    assert len(fences) == 2 * len(plan["windows"])  # never one sync per chunk
    if length == 100000:
        assert selected == [1, 2, 3, 23, 24, 25]
        assert report["chunks"][-3]["token_start"] == 90112
        assert report["chunks"][-1]["token_end"] == 100000


def test_finish_restores_execute_even_if_profiler_stop_fails(module, monkeypatch):
    worker = Worker()
    original = worker.execute_model
    monkeypatch.setattr(module, "synchronize_boundary", lambda: None)
    module.install_chunk_profile(worker, "100k_on", module.make_capture_plan(100000, 4096))
    worker.execute_model(NS(total_num_scheduled_tokens=4096))

    def fail(**kwargs):
        raise RuntimeError("stop failed")

    worker.profile = fail
    with pytest.raises(RuntimeError, match="stop failed"):
        module.finish_chunk_profile(worker)
    assert worker.execute_model == original
    assert not hasattr(worker, "_prefill_chunk_profile_capture")


def test_changed_chunk_schedule_is_reported_not_silently_mislabelled(module, monkeypatch):
    worker = Worker()
    monkeypatch.setattr(module, "synchronize_boundary", lambda: None)
    plan = module.make_capture_plan(100000, 4096)
    module.install_chunk_profile(worker, "100k_on", plan)
    worker.execute_model(NS(total_num_scheduled_tokens=2048))
    report = module.finish_chunk_profile(worker)
    with pytest.raises(RuntimeError, match="actual prefill chunk layout differs"):
        module.validate_capture(plan, [report])
