# SPDX-License-Identifier: Apache-2.0
"""CPU worker/RPC boundary tests, with device/profiler APIs replaced by fakes."""

import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    module = importlib.import_module("layerwise_prefill_profile_worker")
    monkeypatch.setattr(module, "install_transfer_attribution", lambda: NS(restore=lambda: None))
    return module


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


@pytest.mark.parametrize("length", [80000, 95000, 100000, 24 * 4096, 4096, 6 * 4096, 6 * 4096 + 1])
@pytest.mark.parametrize("wrapped", [False, True])
def test_capture_first_and_last_chunks_only(module, monkeypatch, length, wrapped):
    worker = Worker()
    receiver = Wrapper(worker) if wrapped else worker
    original = worker.execute_model
    fences = []
    monkeypatch.setattr(module, "synchronize_boundary", lambda: fences.append(len(worker.events)))
    plan = module.make_capture_plan(length, 4096)
    module.install_chunk_profile(receiver, "80k_on", plan)
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
    assert [e[1] for e in worker.events if e[0] == "start"] == [f"80k_on_{w['name']}" for w in plan["windows"]]
    assert len(fences) == 2 * len(plan["windows"])  # never one sync per chunk
    if length == 80000:
        assert selected == [1, 2, 3, 18, 19, 20]
        assert report["chunks"][-3]["token_start"] == 69632
        assert report["chunks"][-1]["token_end"] == 80000


def test_finish_restores_execute_even_if_profiler_stop_fails(module, monkeypatch):
    worker = Worker()
    original = worker.execute_model
    monkeypatch.setattr(module, "synchronize_boundary", lambda: None)
    module.install_chunk_profile(worker, "80k_on", module.make_capture_plan(80000, 4096))
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
    plan = module.make_capture_plan(80000, 4096)
    module.install_chunk_profile(worker, "80k_on", plan)
    worker.execute_model(NS(total_num_scheduled_tokens=2048))
    report = module.finish_chunk_profile(worker)
    with pytest.raises(RuntimeError, match="actual prefill chunk layout differs"):
        module.validate_capture(plan, [report])


def test_attribution_preserves_results_errors_and_restores(module):
    from contextlib import contextmanager

    events = []

    @contextmanager
    def record(label):
        events.append(("begin", label))
        try:
            yield
        finally:
            events.append(("end", label))

    def copy(copies, device_to_host):
        events.append(("copy", copies, device_to_host))
        return 42

    owner = NS(copy=copy)
    ranges = module.TransferAttribution(record)
    ranges.wrap(owner, "copy", module.dma_range_label)
    copies = [(1, 2, 1024), (3, 4, 2048)]
    assert owner.copy(copies, device_to_host=False) == 42
    assert events[0] == ("begin", "PREFILL_ATTR/dma_submit/H2D/segments=2/bytes=3072")
    assert events[1] == ("copy", copies, False)
    assert events[2][0] == "end"
    ranges.restore()
    assert owner.copy is copy
    ranges.restore()

    def fail():
        raise ValueError("original error")

    owner.fail = fail
    ranges.wrap(owner, "fail", "failure")
    with pytest.raises(ValueError, match="original error"):
        owner.fail()
    assert events[-1] == ("end", "PREFILL_ATTR/failure")
    ranges.restore()
    assert owner.fail is fail


def test_attribution_only_installed_in_capture_windows(module, monkeypatch):
    events = []
    monkeypatch.setattr(module, "synchronize_boundary", lambda: None)

    def install():
        events.append("install")
        return NS(restore=lambda: events.append("restore"))

    monkeypatch.setattr(module, "install_transfer_attribution", install)
    worker = Worker()
    module.install_chunk_profile(worker, "80k_on", module.make_capture_plan(80000, 4096))
    for chunk in range(20):
        worker.execute_model(NS(total_num_scheduled_tokens=min(4096, 80000 - chunk * 4096)))
        if chunk == 3:
            assert events == ["install", "restore"]
    module.finish_chunk_profile(worker)
    assert events == ["install", "restore", "install", "restore"]


def test_real_installer_wraps_dma_and_restores_on_failure(monkeypatch):
    import sys
    from contextlib import nullcontext

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    tool = importlib.import_module("layerwise_prefill_profile_worker")
    labels = []

    def record(label):
        labels.append(label)
        return nullcontext()

    def operation(*args, **kwargs):
        return "ok"

    class Impl:
        pass

    methods = (
        "forward",
        "exec_kv",
        "indexer_select_pre_process",
        "indexer_select_post_process",
        "_update_indexcache_topk_indices",
        "_get_indexcache_topk_indices",
    )
    for name in methods:
        setattr(Impl, name, operation)
    npu = NS(npu_scatter_nd_update_=operation, npu_lightning_indexer=operation)
    sfa = NS(
        AscendSFAImpl=Impl,
        torch_npu=npu,
        maybe_submit_layerwise_prefill_load=operation,
        wait_for_kv_layer_from_connector=operation,
    )
    ops = NS(npu_lightning_indexer=operation, npu_lightning_indexer_quant=operation)
    dma = NS(
        layerwise_prefill_dma_copy=operation,
        layerwise_prefill_dma_copy_diagnose=lambda copies, direction: (12.0, 1234, [2000.0] * len(copies)),
    )
    monkeypatch.setitem(sys.modules, "torch", NS(profiler=NS(record_function=record), ops=NS(_C_ascend=ops)))
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.sfa_v1", sfa)
    monkeypatch.setitem(sys.modules, "lmcache_ascend.v1.npu_connector.npu_connectors", NS(lmc_ops=dma))
    ranges = tool.install_transfer_attribution()
    assert Impl().forward(layer_name="layer0") == "ok"
    assert dma.layerwise_prefill_dma_copy([(1, 2, 64)], True) is None
    assert labels == ["PREFILL_ATTR/mla/layer0", "PREFILL_ATTR/dma_submit/D2H/segments=1/bytes=64"]
    ranges.restore()
    assert dma.layerwise_prefill_dma_copy is operation
    assert Impl.forward is operation
    assert npu.npu_scatter_nd_update_ is operation
    # Installation failure must not leave partially wrapped production APIs.
    del dma.layerwise_prefill_dma_copy
    with pytest.raises(AttributeError):
        tool.install_transfer_attribution()
    assert Impl.forward is operation
    assert ops.npu_lightning_indexer is operation


def test_dma_diagnostic_context_and_slow_calls(module, capsys):
    import json
    from contextlib import nullcontext

    calls = []

    def diagnose(copies, direction):
        calls.append((copies, direction))
        return 10.0, 4096, [5.0, 2000000.0, 7.0]

    original = lambda *a: None
    ops = NS(layerwise_prefill_dma_copy=original, layerwise_prefill_dma_copy_diagnose=diagnose)
    ranges = module.TransferAttribution(lambda name: nullcontext())
    module.install_dma_diagnostics(ranges, ops)

    def submit():
        layer_id, kv_group, bank = 7, 0, 1  # noqa: F841 - inspected by diagnostic wrapper
        ops.layerwise_prefill_dma_copy([(100, 200, 16), (300, 400, 32), (500, 600, 64)], False)

    submit()
    report = json.loads(capsys.readouterr().out.split("[PREFILL_DMA] ")[1])
    assert (report["layer_id"], report["kv_group"], report["bank"]) == (7, 0, 1)
    assert report["slowest"][0] == dict(index=1, us=2000000.0, src="0x190", dst="0x12c", bytes=32)
    assert report["memcpy_sum_us"] == 2000012.0
    assert report["calls_over_1ms"] == 1
    assert report["bytes"] == 112
    assert len(calls) == 1
    ranges.restore()
    assert ops.layerwise_prefill_dma_copy is original


def test_dummy_dma_skips_native_for_whole_request_and_restores(module, monkeypatch):
    import sys
    from contextlib import nullcontext

    native_calls = []
    original = lambda *args: native_calls.append(args)
    ops = NS(layerwise_prefill_dma_copy=original)
    monkeypatch.setitem(sys.modules, "lmcache_ascend", NS(c_ops=ops))
    monkeypatch.setitem(sys.modules, "lmcache_ascend.c_ops", ops)
    monkeypatch.setattr(module, "synchronize_boundary", lambda: None)

    def attribution():
        ranges = module.TransferAttribution(lambda name: nullcontext())
        # No diagnostic C++ binary needed in dummy mode.
        module.install_dma_diagnostics(ranges, ops)
        return ranges

    monkeypatch.setattr(module, "install_transfer_attribution", attribution)
    worker = Worker()
    plan = module.make_capture_plan(80000, 4096)
    plan["dummy_dma"] = True
    module.install_chunk_profile(worker, "80k_on", plan)
    for chunk in range(20):
        worker.execute_model(NS(total_num_scheduled_tokens=min(4096, 80000 - chunk * 4096)))
        ops.layerwise_prefill_dma_copy([(1, 2, 64)], False)
        ops.layerwise_prefill_dma_copy([(1, 2, 64)], True)
    assert not native_calls
    module.finish_chunk_profile(worker)
    assert ops.layerwise_prefill_dma_copy is original
