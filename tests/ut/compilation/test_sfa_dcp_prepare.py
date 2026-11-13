# SPDX-License-Identifier: Apache-2.0
"""Addressing and ownership tests for shared SFA and DCP preparation kernels."""

import sys
from contextlib import nullcontext
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
from sfa_test_support import ROOT, HostTL, Pointer, extract
from test_sfa_dcp_forward import TopkTL
from test_sfa_event_handoff import handoff  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode

OPS = ROOT / "vllm_ascend/ops/triton/sfa_dcp.py"
QUERY_OPS = ROOT / "vllm_ascend/ops/triton/sfa_query.py"


class Kernel:
    def __init__(self, name, calls):
        self.tl = TopkTL() if name == "_prepare_topk" else HostTL()
        self.body = extract(QUERY_OPS if name == "_copy_query_bridge" else OPS, name, {"tl": self.tl})
        self.name, self.calls = name, calls

    def __getitem__(self, grid):
        def launch(*args):
            self.calls.append((self.name, grid))
            self.tl.programs = grid[0]
            for self.tl.pid in range(grid[0]):
                self.body(*(Pointer(a) if isinstance(a, torch.Tensor) else a for a in args))

        return launch


@pytest.fixture
def ops():
    calls = []
    ns = {
        "torch": torch,
        "init_device_properties_triton": lambda: None,
        "get_vectorcore_num": lambda: 7,
        "triton": SimpleNamespace(
            cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (n - 1).bit_length()
        ),
    }
    for name in ("_mask_topk", "_prepare_topk", "_copy_query_bridge"):
        ns[name] = Kernel(name, calls)
    return (
        extract(OPS, "prepare_sfa_dcp_topk", ns),
        extract(QUERY_OPS, "copy_sfa_query_bridge", ns),
        calls,
    )


@pytest.mark.parametrize("rows,width", [(0, 2048), (1, 2048), (2, 2048), (9, 7), (32, 2048)])
@pytest.mark.parametrize("active", [0, 1, 32])
def test_mask_tiles_preserve_global_ids_causality_and_counts(ops, rows, width, active):
    prepare, _, calls = ops
    active = min(rows, active)
    raw = (torch.arange(rows * width, dtype=torch.int32) % 1027 - 2).reshape(rows, 1, width)
    original = raw.clone()
    # Short position storage checks that inactive rows never read it.
    positions = torch.arange(active, dtype=torch.int64) * 128
    output = torch.full_like(raw, -99)
    counts = torch.full((rows, 16), -77, dtype=torch.int32)
    prepare(raw, positions, output, counts, active, 8, 3, 128, local=False)
    expected = torch.full_like(raw, -1)
    for row in range(active):
        expected[row] = torch.where((raw[row] >= 0) & (raw[row] <= positions[row]), raw[row], -1)
    assert torch.equal(output, expected) and torch.equal(raw, original)
    assert torch.all(counts == -77)
    assert not calls if rows == 0 else calls == [("_mask_topk", (min((rows * width + 255) // 256, 7),))]


@pytest.mark.parametrize("local,mixed", [(True, False), (False, True), (True, True)])
def test_local_and_mixed_rows_keep_existing_compaction(ops, local, mixed):
    prepare, _, calls = ops
    raw = torch.tensor([[[0, 127, 128, 129, 511, 512, -1, 1024]]] * 2, dtype=torch.int32)
    output = torch.empty_like(raw)
    counts = torch.full((2, 16), -77, dtype=torch.int32)
    owners = torch.tensor([-1, 0]) if mixed else None
    prepare(raw, torch.tensor([512, 512]), output, counts, 2, 4, 1, 128, local=local, row_requests=owners)
    assert calls == [("_prepare_topk", (2,))]
    assert output[0, 0].tolist() == [0, 1, -1, -1, -1, -1, -1, -1]
    if mixed:
        assert output[1, 0].tolist() == [0, 127, 128, 129, 511, 512, -1, -1]
        assert counts[1, 0] == -77
    assert counts[0, 0] == 2 and torch.all(counts[:, 1:] == -77)


@pytest.mark.parametrize("rows", [0, 1, 2, 9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_query_bridge_preserves_bits_strides_and_unused_capacity(ops, rows, dtype, strided):
    _, copy_queries, calls = ops
    heads = 8
    bits = ((torch.arange(heads * (rows + 2) * 1152) * 97) % 65536 - 32768).to(torch.int16)
    storage = bits.view(dtype).reshape(heads, rows + 2, 1152)
    query = storage[:, 1 : rows + 1, :1024:2].transpose(0, 1)
    rope = storage[:, 1 : rows + 1, 1024::2].transpose(0, 1)
    if not strided:
        query, rope = query.contiguous(), rope.contiguous()
    destinations = [torch.full((rows + 3, heads, d), 7, dtype=dtype) for d in (512, 64)]
    copy_queries(query, rope, *destinations)
    for source, destination in zip((query, rope), destinations):
        assert torch.equal(source.view(torch.int16), destination[:rows].view(torch.int16))
        assert torch.all(destination[rows:] == 7)
    assert torch.equal(storage.view(torch.int16).flatten(), bits)
    assert len(calls) == (rows > 0)


@pytest.mark.parametrize("enabled,dtype", [(True, torch.bfloat16), (False, torch.bfloat16), (True, torch.float32)])
def test_actual_bridge_keeps_buffers_and_other_outputs(ops, monkeypatch, enabled, dtype):
    _, copy_queries, calls = ops
    module = ModuleType("vllm_ascend.ops.triton.sfa_query")
    module.copy_sfa_query_bridge = copy_queries
    monkeypatch.setitem(sys.modules, module.__name__, module)
    copy_bridge = extract(
        ROOT / "vllm_ascend/attention/sfa_v1.py", "_copy_to_staged_sfa_bridge", {"torch": torch, "HAS_TRITON": enabled}
    )
    query = torch.arange(2 * 8 * 512).to(dtype).reshape(8, 2, 512).transpose(0, 1)
    outputs = (query, torch.ones(2, 8, 64, dtype=dtype), *(torch.arange(4).reshape(2, 2) for _ in range(4)))
    buffers = tuple(torch.full((4, *t.shape[1:]), -9, dtype=t.dtype) for t in outputs)
    owner = SimpleNamespace(_ensure_staged_sfa_bridge_buffers=lambda hidden: buffers)
    for _ in range(2):
        assert copy_bridge(owner, query, outputs) is buffers
    assert len(calls) == (2 if enabled and dtype == torch.bfloat16 else 0)
    for source, destination in zip(outputs, buffers):
        assert torch.equal(source, destination[:2]) and torch.all(destination[2:] == -9)
    with pytest.raises(RuntimeError, match="arity"):
        copy_bridge(owner, query, outputs[:-1])
    with pytest.raises(RuntimeError, match="fixed storage"):
        copy_bridge(owner, query, (query.repeat(3, 1, 1), *outputs[1:]))


@pytest.mark.parametrize("dcp_size", [1, 8])
def test_graph_caller_fuses_queries_in_both_dcp_modes(ops, request, monkeypatch, dcp_size):
    _, copy_queries, calls = ops
    module = ModuleType("vllm_ascend.ops.triton.sfa_query")
    module.copy_sfa_query_bridge = copy_queries
    monkeypatch.setitem(sys.modules, module.__name__, module)
    copy_bridge = extract(
        ROOT / "vllm_ascend/attention/sfa_v1.py", "_copy_to_staged_sfa_bridge", {"torch": torch, "HAS_TRITON": True}
    )
    h = request.getfixturevalue("handoff")
    h.metadata.sfa_dcp = None if dcp_size == 1 else SimpleNamespace()
    query = torch.arange(2 * 8 * 512).to(torch.bfloat16).reshape(8, 2, 512).transpose(0, 1)
    outputs = (query, torch.ones(2, 8, 64, dtype=query.dtype), *h.bridge[2:])
    buffers = tuple(torch.empty_like(t, memory_format=torch.contiguous_format) for t in outputs)
    h.impl._cross_layer_pre_compute.return_value = outputs
    h.impl._ensure_staged_sfa_bridge_buffers = lambda hidden: buffers
    h.impl._copy_to_staged_sfa_bridge = MethodType(copy_bridge, h.impl)
    h.pre()
    assert [name for name, _ in calls] == ["_copy_query_bridge"]
    assert all(torch.equal(source, target) for source, target in zip(outputs, buffers))


@pytest.mark.parametrize("layout", ["different_dtype", "strided_destination", "different_rows"])
def test_bridge_keeps_ordinary_copy_for_nonmatching_layouts(layout):
    copy_bridge = extract(
        ROOT / "vllm_ascend/attention/sfa_v1.py", "_copy_to_staged_sfa_bridge", {"torch": torch, "HAS_TRITON": True}
    )
    query = torch.arange(2 * 8 * 512).to(torch.bfloat16).reshape(2, 8, 512)
    rope = torch.ones(1 if layout == "different_rows" else 2, 8, 64, dtype=query.dtype)
    outputs = (query, rope, *(torch.arange(4).reshape(2, 2) for _ in range(4)))
    buffers = [torch.full((4, *t.shape[1:]), -9, dtype=t.dtype) for t in outputs]
    if layout == "different_dtype":
        buffers[0] = buffers[0].float()
    elif layout == "strided_destination":
        buffers[0] = torch.full((4, 8, 1024), -9, dtype=query.dtype)[:, :, ::2]
    owner = SimpleNamespace(_ensure_staged_sfa_bridge_buffers=lambda hidden: buffers)
    copy_bridge(owner, query, outputs)
    for source, target in zip(outputs, buffers):
        rows = source.shape[0]
        assert torch.equal(source.to(target.dtype), target[:rows]) and torch.all(target[rows:] == -9)


@pytest.mark.parametrize("values", [[], [1, 2, 3, 4], [1, -1, 3, -1], [-1, -1, -1, -1]])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("logits_mode", [False, True])
def test_logprobs_preserve_placeholders_without_boolean_index_expansion(values, dtype, logits_mode):
    method = extract(
        ROOT.parent / "vllm/vllm/v1/sample/rejection_sampler.py",
        "_get_logprobs_tensors",
        {"torch": torch, "PLACEHOLDER_TOKEN_ID": -1, "record_function_or_nullcontext": nullcontext},
    )
    sampled = torch.tensor(values, dtype=dtype).reshape(-1, 2)
    snapshot = sampled.clone()
    count = sampled.numel()
    logits = torch.arange(count * 8).float().reshape(count, 8)
    metadata = SimpleNamespace(
        cu_num_sampled_tokens=(torch.arange(count // 2) + 1) * 2,
        target_logits_indices=torch.arange(0, count, 2),
        bonus_logits_indices=torch.arange(count // 2) * 2 + 1,
    )
    sampler = SimpleNamespace(
        compute_logprobs=lambda t: t.log_softmax(-1),
        gather_logprobs=lambda t, n, ids: (t, ids),
    )
    owner = SimpleNamespace(sampler=sampler, is_logits_logprobs_mode=logits_mode)

    class NoBooleanIndex(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.index_put_.default, torch.ops.aten.index_put.default):
                assert not any(t is not None and t.dtype == torch.bool for t in args[1]), "Boolean index expansion"
            return func(*args, **(kwargs or {}))

    with NoBooleanIndex():
        actual, ids = method(owner, 0, metadata, logits, logits[::2], logits[1::2], sampled)
    assert torch.equal(sampled, snapshot)
    assert ids.tolist() == [0 if v == -1 else v for v in values]
    assert torch.equal(actual, logits if logits_mode else logits.log_softmax(-1))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("kind", ["accepted_token_counts", "input_staging"])
def test_wait_scope_obeys_existing_profiling_switch_and_keeps_dependency(enabled, kind):
    import contextlib

    events = []
    scope = extract(
        ROOT.parent / "vllm/vllm/v1/utils.py",
        "record_function_or_nullcontext",
        {
            "_PROFILER_FUNC": None,
            "contextlib": contextlib,
            "record_function": torch.profiler.record_function,
            "envs": SimpleNamespace(VLLM_CUSTOM_SCOPES_FOR_PROFILING=enabled, VLLM_NVTX_SCOPES_FOR_PROFILING=False),
        },
    )
    wait = extract(
        ROOT.parent / "vllm/vllm/v1/worker/gpu_model_runner.py",
        "_get_valid_sampled_token_count" if kind == "accepted_token_counts" else "synchronize_input_prep",
        {"record_function_or_nullcontext": scope},
    )
    owner = SimpleNamespace(
        input_batch=SimpleNamespace(prev_sampled_token_ids=torch.ones(2, 1)),
        valid_sampled_token_count_event=SimpleNamespace(synchronize=lambda: events.append("wait")),
        valid_sampled_token_count_cpu=torch.tensor([1, 2, 99]),
    )
    owner.prepare_inputs_event = owner.valid_sampled_token_count_event
    owner.prepare_inputs_event.record = lambda: events.append("record")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        if kind == "input_staging":
            with contextlib.contextmanager(wait)(owner):
                events.append("body")
        else:
            assert wait(owner) == [1, 2]
    expected_events = ["wait", "body", "record"] if kind == "input_staging" else ["wait"]
    assert events == expected_events
    assert (f"decode_wait::{kind}" in {e.key for e in profile.key_averages()}) == enabled
    owner.input_batch.prev_sampled_token_ids = None
    owner.prepare_inputs_event = None
    if kind == "input_staging":
        with contextlib.contextmanager(wait)(owner):
            pass
    else:
        assert wait(owner) == []
    assert events == expected_events
