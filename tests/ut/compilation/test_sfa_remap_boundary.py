# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU comparison against the original remap helper, including cache lifetime."""

import importlib.util
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sfa_test_support import definitions, extract
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("remap_under_test", ROOT / "vllm_ascend/attention/sfa_remap_boundary.py")
remap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remap)


@pytest.fixture
def original():
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    names = {"_prepare_sfa_remap_boundary", "_validate_dsa_scratch_capacity"}
    config = SimpleNamespace(window=256)
    namespace = {
        "torch": torch,
        "np": np,
        "_decode_window_save_window_size": lambda: config.window,
        "get_lmcache_sparse_cached_tokens": Mock(side_effect=AssertionError("unexpected connector lookup")),
    }
    definitions(path, names, namespace)
    prepare = namespace["_prepare_sfa_remap_boundary"]
    return partial(prepare, reuse_unchanged=True), config


def metadata(rows, prompts, lengths, buffer=None, capacity=4096):
    tensor = torch.zeros(len(rows), dtype=torch.int32) if buffer is None else buffer.tensor[: len(rows)]
    return SimpleNamespace(
        decode_remap_boundary=tensor,
        decode_remap_boundary_ready=False,
        decode_remap_boundary_buffer=buffer,
        decode_req_indices_cpu=np.asarray(rows, dtype=np.int32),
        prompt_lens_cpu_rows=np.asarray(prompts, dtype=np.int32),
        seq_lens_cpu=torch.tensor(lengths),
        decode_scratch_capacity=capacity,
        decode_scratch_base_cpu=None,
    )


@pytest.mark.parametrize("window", [0, 256, 512])
@pytest.mark.parametrize("requests", [1, 4, 16])
def test_random_layouts_exactly_match_uncached_helper(original, window, requests):
    prepare, config = original
    config.window = window
    rng = np.random.default_rng(19)
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(requests * 2 + 3, dtype=torch.int32))
    for _ in range(40):
        # Includes non-contiguous request indices, Q1/Q2, lane reorder and pads.
        rows = [row for row in range(requests) for _ in range(int(rng.integers(0, 3)))]
        rows += [-1] * 3
        rng.shuffle(rows)
        prompts = [int(rng.choice((0, 4096, 8192))) for _ in rows]
        lengths = rng.integers(8193, 30000, size=requests).tolist()
        frontiers = tuple(int(rng.choice((0, 8192, 12288))) for _ in sorted(set(rows) - {-1}))
        old = metadata(rows, prompts, lengths)
        new = metadata(rows, prompts, lengths, buffer)
        expected = prepare(old, None, is_dummy_run=False, index_topk=2048, cached_tokens=frontiers)
        actual = prepare(new, None, is_dummy_run=False, index_topk=2048, cached_tokens=frontiers)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_511_decode_steps_upload_only_three_window_changes(original):
    prepare, _ = original
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32))
    address = buffer.tensor.data_ptr()
    for seq in range(5001, 5512):
        item = metadata([0, 0], [5000, 5000], [seq], buffer)
        prepare(item, None, is_dummy_run=False, index_topk=2048, cached_tokens=(30000,))
        assert item.decode_remap_boundary.tolist() == [(seq - 1) // 256 * 256] * 2
        assert item.decode_remap_boundary_ready
    assert buffer.upload_count == 3
    assert buffer.tensor.data_ptr() == address


def test_warm_update_uses_no_tensor_operators_or_copy():
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32))
    lengths = torch.tensor([5001])
    args = ([0, 0], [5000, 5000], lengths, (5000,), 256, 2048, 4096)
    buffer.update(*args)

    class NoTensorOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            pytest.fail(f"Unchanged boundary issued a tensor operation: {func}")

    with NoTensorOps():
        buffer.update(*args)
    assert buffer.upload_count == 1


def test_mode_handoff_invalidates_shadow_before_external_write(original):
    prepare, _ = original
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32))
    live = metadata([0, 0], [5000, 5000], [5200], buffer)
    prepare(live, None, is_dummy_run=False, index_topk=2048, cached_tokens=(5120,))
    assert buffer.tensor.tolist() == [5120, 5120]
    dummy = metadata([0, 0], [5000, 5000], [7000], buffer)
    prepare.func(dummy, None, is_dummy_run=True, index_topk=2048)
    assert buffer.tensor.tolist() == [6912, 6912]
    live.decode_remap_boundary_ready = False
    prepare(live, None, is_dummy_run=False, index_topk=2048, cached_tokens=(5120,))
    assert buffer.tensor.tolist() == [5120, 5120]
    assert buffer.upload_count == 2


def test_frontier_shrink_padding_prompts_and_capacity_switches():
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(4, dtype=torch.int32))
    buffer.update([0, 0, 1, -1], [10000] * 4, [15000, 17000], (12288, 16384), 256, 2048, 4096)
    assert buffer.tensor.tolist() == [12288, 12288, 16384, 10000]
    buffer.update([0], [10000], [15000], (8192,), 256, 2048, 4096)
    buffer.update([0, 0, 1, -1], [10000, 10000, 10000, 123], [15000, 17000], (8192, 0), 256, 2048, 4096)
    assert buffer.tensor.tolist() == [8192, 8192, 0, 123]
    assert buffer.upload_count == 3


@pytest.mark.parametrize("change", ["overlap", "too_many_rows", "scratch_growth", "frontier_count", "missing_request"])
def test_invalid_dynamic_inputs_cannot_hit_previous_value_cache(change):
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(3, dtype=torch.int32))
    args = [[0, 0], [5000, 5000], [5200], (5120,), 256, 2048, 4096]
    buffer.update(*args)
    if change == "overlap":
        args[3] = (2048,)
    elif change == "too_many_rows":
        args[0], args[1] = [0, 0, 0], [5000] * 3
    elif change == "scratch_growth":
        args[-1] = 8192
    elif change == "frontier_count":
        args[3] = ()
    else:
        args[2] = []
    with pytest.raises(RuntimeError):
        buffer.update(*args)
    assert buffer.tensor[:2].tolist() == [5120, 5120]


def test_partial_failed_upload_invalidates_old_shadow():
    class Destination:
        device = torch.device("cpu")

        def __init__(self):
            self.fail = False
            self.writes = 0

        def numel(self):
            return 2

        def __getitem__(self, key):
            return self

        def copy_(self, value):
            self.writes += 1
            if self.fail:
                raise RuntimeError("copy failed")

    destination = Destination()
    buffer = remap.SFARemapBoundaryBuffer(destination)
    args = ([0, 0], [5000, 5000], [5200], (5120,), 256, 2048, 4096)
    buffer.update(*args)
    destination.fail = True
    with pytest.raises(RuntimeError, match="copy failed"):
        buffer.update([0, 0], [5000, 5000], [6000], (5632,), 256, 2048, 4096)
    destination.fail = False
    buffer.update(*args)
    assert destination.writes == 3 and buffer.upload_count == 2


def test_original_path_still_updates_each_step_and_invalidates_full_cache(original):
    prepare, _ = original
    # Use the underlying function, without the fixture's full-graph opt-in.
    uncached = prepare.func
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32))
    item = metadata([0, 0], [5000, 5000], [5200], buffer)
    prepare(item, None, is_dummy_run=False, index_topk=2048, cached_tokens=(5120,))
    version = buffer.tensor._version
    for _ in range(2):
        item.decode_remap_boundary_ready = False
        uncached(item, None, is_dummy_run=False, index_topk=2048, cached_tokens=(5120,))
    assert buffer.tensor._version == version + 2
    item.decode_remap_boundary_ready = False
    prepare(item, None, is_dummy_run=False, index_topk=2048, cached_tokens=(5120,))
    assert buffer.upload_count == 2


@pytest.fixture
def deferred_uploads(monkeypatch):
    """Delay DMA reads so premature source reuse changes the observed bytes."""
    operations, events, waits = [], [], []
    cursor = 0

    def complete(end):
        nonlocal cursor
        while cursor < end:
            operations[cursor]()
            cursor += 1

    stream = SimpleNamespace()
    current_stream = [stream]

    class Event:
        def __init__(self):
            self.end = None
            self.queries = 0
            self.fail_record = False
            events.append(self)

        def record(self, actual_stream):
            assert actual_stream is stream
            if self.fail_record:
                raise RuntimeError("record failed")
            self.end = len(operations)

        def query(self):
            self.queries += 1
            assert self.end is not None
            return cursor >= self.end

        def synchronize(self):
            assert self.end is not None
            waits.append(self.end)
            complete(self.end)

    class DeviceBuffer:
        device = SimpleNamespace(type="npu")

        def __init__(self, data):
            self.data = data

        def numel(self):
            return self.data.numel()

        def __getitem__(self, index):
            return DeviceBuffer(self.data[index])

        def copy_(self, source, *, non_blocking):
            assert non_blocking is True
            # Keep the original view; deliberately do not snapshot source bytes.
            operations.append(lambda: self.data.copy_(source))

    empty_like = torch.empty_like

    def allocate(destination, *, pin_memory, **kwargs):
        assert pin_memory and kwargs == {"device": "cpu"}
        return empty_like(destination.data)

    monkeypatch.setattr(remap.torch, "empty_like", allocate)
    monkeypatch.setattr(
        remap.torch,
        "npu",
        SimpleNamespace(
            Event=Event,
            current_stream=lambda device: current_stream[0],
        ),
        raising=False,
    )
    target = DeviceBuffer(torch.zeros(2, dtype=torch.int32))
    buffer = remap.SFARemapBoundaryBuffer(target)

    def update(frontier):
        buffer.update([0, 0], [10000, 10000], [20000], (frontier,), 0, 2048, 4096)

    return SimpleNamespace(
        buffer=buffer,
        update=update,
        target=target,
        events=events,
        waits=waits,
        operations=operations,
        drain=lambda: complete(len(operations)),
        current_stream=current_stream,
    )


def test_async_boundary_preserves_each_pending_snapshot_and_bounds_staging(deferred_uploads):
    state = deferred_uploads
    observed = []
    for frontier in (4096, 8192):
        state.update(frontier)
        state.operations.append(lambda: observed.append(state.target.data.tolist()))
    assert state.waits == [] and state.target.data.tolist() == [0, 0]
    queries = sum(event.queries for event in state.events)
    state.update(8192)
    assert sum(event.queries for event in state.events) == queries
    assert state.buffer.upload_count == 2

    state.update(12288)  # Recycle only after the first H2D, not its later consumer.
    state.operations.append(lambda: observed.append(state.target.data.tolist()))
    assert len(state.events) == remap.MAX_PENDING_BOUNDARY_UPLOADS
    assert state.waits == [1]
    state.drain()
    assert observed == [[4096, 4096], [8192, 8192], [12288, 12288]]
    state.update(16384)
    assert state.waits == [1]  # Completed slots need no host wait.
    state.drain()
    assert state.target.data.tolist() == [16384, 16384]


def test_async_boundary_rejects_stream_change_even_for_cached_values(deferred_uploads):
    state = deferred_uploads
    state.update(8192)
    state.current_stream[0] = object()
    with pytest.raises(RuntimeError, match="one stream"):
        state.update(8192)
    assert state.buffer.upload_count == 1
    state.drain()


def test_async_failed_record_retains_source_and_refuses_reuse(deferred_uploads):
    state = deferred_uploads
    state.update(4096)
    state.update(8192)
    state.drain()
    state.events[0].fail_record = True
    with pytest.raises(RuntimeError, match="record failed"):
        state.update(12288)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        state.update(4096)
    assert len(state.events) == 2
    state.drain()
    assert state.target.data.tolist() == [12288, 12288]


@pytest.fixture
def native_preparation():
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    names = {"_prepare_sfa_remap_boundary", "_resolve_sparse_cached_tokens_by_request"}
    frontiers = {"a": 8192, "b": 12288}
    lookup = Mock(side_effect=lambda ids: [frontiers[key] for key in ids])
    ns = {
        "np": np,
        "torch": torch,
        "get_lmcache_sparse_cached_tokens": lookup,
        "_decode_window_save_window_size": lambda: 256,
    }
    definitions(path, names, ns)
    prepare = extract(path, "prepare_native_sparse_boundary", ns)
    impl_type = type("NativeImpl", (), {"prepare_native_sparse_boundary": prepare})
    impl = impl_type()
    impl.dsa_shrink_latent, impl.index_topk = 2, 2048
    return impl, frontiers, lookup


def test_native_preparation_survives_builder_reset_and_reused_draft_metadata(native_preparation):
    impl, frontiers, lookup = native_preparation
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(4, dtype=torch.int32))
    first = metadata([1, 0, 1, -1], [5000, 6000, 5000, 0], [20000, 22000], buffer)
    first.req_ids = ["a", "b"]
    first.num_decode_tokens, first.need_sparse_lmcache_payload = 3, True
    first.split_boundary = torch.tensor([5000, 6000, 5000, 0], dtype=torch.int32)
    second = metadata([0], [6000], [20000, 22000], buffer)
    second.req_ids = ["a", "b"]
    second.num_decode_tokens, second.need_sparse_lmcache_payload = 1, True
    second.split_boundary = torch.tensor([6000], dtype=torch.int32)

    impl.prepare_native_sparse_boundary(first)
    assert first.decode_split_boundary.tolist() == [12288, 8192, 12288, 0]
    first.split_boundary.fill_(99)  # Generic builder storage is independent.
    first.decode_split_boundary = None  # New native metadata readiness marker.
    impl.prepare_native_sparse_boundary(first)
    assert buffer.upload_count == 1
    impl.prepare_native_sparse_boundary(second)
    impl.prepare_native_sparse_boundary(first)
    assert buffer.upload_count == 3
    assert first.decode_split_boundary.tolist() == [12288, 8192, 12288, 0]
    frontiers["b"] = 0  # Recovery/changed frontier cannot reuse old values.
    impl.prepare_native_sparse_boundary(first)
    assert first.decode_split_boundary.tolist() == [0, 8192, 0, 0]
    assert lookup.call_args.args == (["a", "b"],)


def test_native_preparation_deduplicates_shared_metadata_and_skips_other_backends():
    impl = SimpleNamespace(prepare_native_sparse_boundary=Mock())
    item = object()
    remap.prepare_native_sparse_boundaries(
        [("l0", impl), ("l1", impl), ("other", object()), ("absent", impl)],
        {"l0": item, "l1": item, "other": object()},
    )
    impl.prepare_native_sparse_boundary.assert_called_once_with(item)
    remap.prepare_native_sparse_boundaries([], None)


@pytest.mark.parametrize("mode", ["NONE", "PIECEWISE", "FULL"])
@pytest.mark.parametrize("enabled", [False, True])
def test_draft_hook_prepares_each_step_before_native_model_only(mode, enabled):
    path = ROOT / "vllm_ascend/spec_decode/eagle_proposer.py"
    calls = []
    ns = {
        "envs_ascend": SimpleNamespace(VLLM_ASCEND_SFA_FULL_GRAPH=enabled),
        "get_forward_context": lambda: SimpleNamespace(cudagraph_runtime_mode=mode),
        "CUDAGraphMode": SimpleNamespace(FULL="FULL"),
        "prepare_native_sparse_boundaries": remap.prepare_native_sparse_boundaries,
    }
    run_draft = extract(path, "_run_mtp_draft_layer_with_diagnostics", ns)
    impl = SimpleNamespace(prepare_native_sparse_boundary=lambda item: calls.append(("prepare", item)))
    runner = SimpleNamespace(
        method="mtp",
        _draft_attn_layers={"draft": SimpleNamespace(impl=impl)},
        model=lambda **kwargs: calls.append(("model", kwargs["step"])),
    )
    for step in (0, 1):
        run_draft(
            runner, {"step": step}, draft_step=step, per_layer_attn_metadata={"draft": step}, runtime_inputs={}
        )
    expected = [("prepare", 0), ("model", 0), ("prepare", 1), ("model", 1)]
    assert calls == (expected if enabled and mode != "FULL" else [("model", 0), ("model", 1)])


def test_target_fallback_prepares_after_context_binding_before_model():
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    calls = []
    context = SimpleNamespace(
        attn_metadata={"target": "bound"}, cudagraph_runtime_mode="NONE", flash_comm_v1_enabled=False
    )
    ns = {
        "envs_ascend": SimpleNamespace(VLLM_ASCEND_SFA_FULL_GRAPH=True),
        "get_forward_context": lambda: context,
        "CUDAGraphMode": SimpleNamespace(FULL="FULL"),
        "prepare_native_sparse_boundaries": remap.prepare_native_sparse_boundaries,
        "_capture_live_source_event_handoff": lambda: None,
    }
    forward = extract(path, "_model_forward", ns)
    impl = SimpleNamespace(prepare_native_sparse_boundary=lambda item: calls.append(item))
    runner = SimpleNamespace(_staged_sfa_impls=[("target", impl)], model=lambda **kwargs: calls.append("model"))
    forward(runner, 1)
    assert calls == ["bound", "model"]


@pytest.mark.parametrize("prepared", [False, True])
def test_boundary_upload_replay_on_real_npu(prepared):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires an NPU")
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32, device="npu"))
    output = torch.empty_like(buffer.tensor)
    buffer.update([0, 0], [10000, 10000], [20000], (8192,), 0, 2048, 4096)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output.copy_(buffer.tensor)
    torch.npu.synchronize()
    observed = []
    frontiers = (4096, 8192, 12288, 0) * 16
    for frontier in frontiers:
        if prepared:
            buffer.update_prepared((frontier, frontier))
        else:
            buffer.update([0, 0], [10000, 10000], [20000], (frontier,), 0, 2048, 4096)
        graph.replay()
        observed.append(output.clone())
    torch.npu.synchronize()
    assert torch.stack(observed).cpu().tolist() == [[v, v] for v in frontiers]
    assert len(buffer._uploads) <= remap.MAX_PENDING_BOUNDARY_UPLOADS


@pytest.mark.parametrize("window", [0, 256])
def test_cached_dummy_boundaries_match_legacy_across_live_and_capacity_changes(original, window):
    prepare, config = original
    config.window = window
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(8, dtype=torch.int32))
    for n in (4, 1, 4):
        rows = np.repeat(np.arange(n), 2)
        for dummy in (True, True, False, False, True):
            prompts, lengths = [5000] * (2 * n), [5200] * n
            frontiers = None if dummy else (4096,) * n
            expected = prepare.func(metadata(rows, prompts, lengths), None, is_dummy_run=dummy,
                                    index_topk=2048, cached_tokens=frontiers)
            old_values, old_count = buffer._values, buffer.upload_count
            actual = prepare(metadata(rows, prompts, lengths, buffer), None, is_dummy_run=dummy,
                             index_topk=2048, cached_tokens=frontiers)
            torch.testing.assert_close(actual, expected)
            if old_values == tuple(expected.tolist()):
                assert buffer.upload_count == old_count


def test_prepared_dummy_upload_keeps_snapshot_and_reuses_owned_staging(deferred_uploads, original):
    state = deferred_uploads
    prepare, config = original
    config.window = 0
    observed = []
    for prompt in (5000, 6000, 6000, 7000):
        item = metadata([0, 0], [prompt, prompt], [prompt + 2], state.buffer)
        prepare(item, None, is_dummy_run=True, index_topk=2048)
        item.prompt_lens_cpu_rows[:] = 9000  # Pending copies must own their source.
        state.operations.append(lambda: observed.append(state.target.data.tolist()))
    assert state.buffer.upload_count == 3 and len(state.events) == 2
    assert state.waits == [1]  # Reuse waits for an upload, not the later consumer.
    state.drain()
    assert observed == [[5000, 5000], [6000, 6000], [6000, 6000], [7000, 7000]]
    state.current_stream[0] = object()
    with pytest.raises(RuntimeError, match="one stream"):
        state.buffer.update_prepared((7000, 7000))


def test_prepared_dummy_upload_refuses_reuse_after_record_failure(deferred_uploads):
    state = deferred_uploads
    state.buffer.update_prepared((5000, 5000))
    state.buffer.update_prepared((6000, 6000))
    state.drain()
    state.events[0].fail_record = True
    with pytest.raises(RuntimeError, match="record failed"):
        state.buffer.update_prepared((7000, 7000))
    with pytest.raises(RuntimeError, match="cannot be reused"):
        state.buffer.update_prepared((5000, 5000))
    state.drain()


def test_prepared_dummy_values_do_not_inherit_live_layout_validation(original):
    prepare, config = original
    config.window = 0
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32))
    buffer.update([0, 0], [12000, 12000], [14000], (8192,), 0, 2048, 8192)
    # These values are valid for the dummy's smaller scratch reservation.
    dummy = metadata([0, 0], [4096, 4096], [4100], buffer, capacity=4096)
    prepare(dummy, None, is_dummy_run=True, index_topk=2048)
    assert buffer.tensor.tolist() == [4096, 4096]
    # Matching device values do not prove safety for the older live layout.
    with pytest.raises(RuntimeError, match="alias live KV positions"):
        buffer.update([0, 0], [12000, 12000], [14000], (4096,), 0, 2048, 8192)


def test_unchanged_prepared_boundary_has_no_tensor_operations():
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32))
    buffer.update_prepared((4096, 4096))

    class NoTensorOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            pytest.fail(f"Unchanged prepared boundary issued a tensor operation: {func}")

    with NoTensorOps():
        buffer.update_prepared((4096, 4096))
    assert buffer.upload_count == 1
