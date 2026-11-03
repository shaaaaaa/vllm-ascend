# SPDX-License-Identifier: Apache-2.0
"""CPU checks of host overhead and optional-mask consumers; no NPU emulation."""

import gc
import math
import sys
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import definitions, extract
from test_sfa_independent import independent

ROOT = Path(__file__).resolve().parents[3]
CONTEXT = ROOT / "vllm_ascend/ascend_forward_context.py"
METHODS = ROOT / "vllm_ascend/ops/fused_moe/moe_comm_method.py"
PREPARE = ROOT / "vllm_ascend/ops/fused_moe/prepare_finalize.py"
DISPATCH = ROOT / "vllm_ascend/ops/fused_moe/token_dispatcher.py"


@pytest.mark.parametrize("rank", range(4))
def test_packet_reuses_storage_without_scalar_tensor_writes(monkeypatch, rank):
    monkeypatch.setattr(independent.dist, "new_group", lambda **kw: "routing")
    policy = independent.IndependentDP(NS(rank_in_group=rank, world_size=4, ranks=list(range(4))), 4, 24)
    packet_ptr, count_ptrs = policy.packet.data_ptr(), [c.data_ptr() for c in policy.counts]
    observed = []

    def reduce(packet, *, group):
        assert group == "routing" and packet.data_ptr() == packet_ptr
        values = packet.numpy()
        observed.append(values[:, rank].copy().tolist())
        assert not values[:, [r for r in range(4) if r != rank]].any()
        values[:] = values[:, rank : rank + 1].copy()

    monkeypatch.setattr(independent.dist, "all_reduce", reduce)
    writes = Mock(side_effect=AssertionError("scalar Tensor write in routing packet"))
    monkeypatch.setattr(torch.Tensor, "__setitem__", writes)
    enabled = gc.isenabled()
    gc.disable()
    try:
        target = policy.agree(24, 1, -1, True, requests=12)
        draft = policy.agree(12, 0, 0, False, draft=True, requests=12)
        assert target.tolist() == [24] * 4 and draft.tolist() == [12] * 4
        assert [target.data_ptr(), draft.data_ptr()] == count_ptrs
        assert policy.agree(2, 1, 1, True, requests=0).tolist() == [2] * 4
        assert observed == [[24, 1, -1, 1, 0, 1, 12], [12, 0, 0, 0, 1, 2, 12], [2, 1, 1, 1, 0, 3, 0]]
        owner = weakref.ref(policy)
        del policy
        assert owner() is None
    finally:
        if enabled:
            gc.enable()
    writes.assert_not_called()


@pytest.mark.parametrize("field", range(7))
def test_every_tp_proposal_field_is_still_checked(monkeypatch, field):
    monkeypatch.setattr(independent.dist, "new_group", lambda **kw: "routing")
    policy = independent.IndependentDP(NS(rank_in_group=2, world_size=4, ranks=list(range(4))), 4, 24)

    def reduce(packet, **kwargs):
        values = packet.numpy()
        values[:] = values[:, 2:3].copy()
        values[field, 3] += 1

    monkeypatch.setattr(independent.dist, "all_reduce", reduce)
    with pytest.raises(RuntimeError, match="within TP"):
        policy.agree(8, 1, 1, True, requests=4)


@pytest.fixture
def forward_context(monkeypatch):
    context, mask = NS(), torch.zeros(32, dtype=torch.bool)
    methods = {"masked": NS(), "MoECommMethod": object}
    definitions(METHODS, ["MC2CommImpl", "FusedMC2CommImpl"], methods)
    return_context = NS(metadata=True, peer_max=16, mask=mask, selected="masked", scalar_reads=0)

    class Maximum:
        def item(self):
            return_context.scalar_reads += 1
            return return_context.peer_max

    @contextmanager
    def base(**kwargs):
        context.dp_metadata = NS(max_tokens_across_dp_cpu=Maximum()) if return_context.metadata else None
        yield

    methods["mc2"] = object.__new__(methods["MC2CommImpl"])
    methods["fused"] = object.__new__(methods["FusedMC2CommImpl"])
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_comm_method",
        NS(
            get_moe_comm_method=lambda mode: methods[mode],
        ),
    )
    selector = Mock(side_effect=lambda *args: return_context.selected)
    namespace = dict(
        CUDAGraphMode=NS(NONE=0),
        set_forward_context=base,
        get_forward_context=lambda: context,
        vllm_version_is=lambda _: False,
        select_moe_comm_method=selector,
        get_tensor_model_parallel_world_size=lambda: 4,
        is_moe_model=lambda _: True,
        is_drafter_moe_model=lambda _: True,
        enable_sp=lambda _: True,
        flashcomm2_enable=lambda: False,
        has_layer_idx=lambda _: False,
        get_dp_group=lambda: NS(world_size=4 if return_context.metadata else 1),
        get_mc2_mask=lambda: return_context.mask,
        math=math,
    )
    return_context.open = contextmanager(extract(CONTEXT, "set_ascend_forward_context", namespace))
    return_context.context, return_context.selector = context, selector
    return return_context


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("metadata,counts", [(True, True), (True, False), (False, True), (False, False)])
def test_context_reuses_maximum_and_preserves_local_fallback(monkeypatch, forward_context, draft, metadata, counts):
    state = forward_context
    state.metadata = metadata
    values = torch.tensor([8, 16, 4, 12]) if counts else None
    maximum = Mock(wraps=torch.Tensor.max)
    monkeypatch.setattr(torch.Tensor, "max", lambda tensor: maximum(tensor))
    with state.open(None, NS(), num_tokens=8, num_actual_tokens=5, num_tokens_across_dp=values, is_draft_model=draft):
        assert state.selector.call_args.args[0] == (16 if counts else 8)
        assert state.selector.call_args.args[2] is draft
        assert state.context.max_tokens_across_dp == (16 if metadata else 8)
        assert state.context.pad_size == (8 if metadata else 0)
        assert state.context.mc2_mask.tolist() == [True] * 5 + [False] * ((16 if metadata else 8) - 5)
    assert state.scalar_reads == int(metadata)
    assert maximum.call_count == int(counts and not metadata)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("actual", [0, 3, 8])
def test_maskless_mc2_then_masked_fallback(forward_context, mode, actual):
    state = forward_context
    values = torch.tensor([8, 16, 4, 12])
    for selected in ("mc2", "masked", "mc2", "fused"):
        state.selected = selected
        before = state.mask.clone()
        with state.open(
            None,
            NS(),
            num_tokens=8,
            num_actual_tokens=actual,
            num_tokens_across_dp=values,
            aclgraph_runtime_mode=mode,
            staged_sfa_graph_dummy_run=actual == 0,
        ):
            if selected == "mc2":
                assert state.context.mc2_mask is None and torch.equal(state.mask, before)
            else:
                assert state.context.mc2_mask.tolist() == [True] * actual + [False] * (16 - actual)
            assert state.context.padded_num_tokens == 16


def test_context_without_reserved_mask_or_explicit_token_count(forward_context):
    state = forward_context
    state.metadata, state.mask = False, None
    with state.open(NS(num_actual_tokens=3), NS(), num_tokens=None):
        assert state.selector.call_args.args[0] is None
        assert state.context.max_tokens_across_dp == 3
        assert state.context.padded_num_tokens == 4 and state.context.mc2_mask is None


@pytest.mark.parametrize("width,rank", [(1, 0), *[(4, rank) for rank in range(4)]])
@pytest.mark.parametrize("actual", [1, 4, 7, 8])
@pytest.mark.parametrize("shared,replace", [(False, False), (True, False), (False, True)])
def test_optional_mask_preserves_mc2_padding_and_tp_slices(width, rank, actual, shared, replace):
    context = NS(padded_num_tokens=8, mc2_mask=None)
    namespace = dict(torch=torch, nn=torch.nn, _EXTRA_CTX=context, QuantType=NS(NONE=0), MoEPrepareOutput=NS)
    definitions(PREPARE, ["prepare"], namespace, class_name="PrepareAndFinalizeWithMC2")
    prepare = namespace["prepare"]
    owner = NS(tp_size=width, tp_rank=rank)
    hidden = torch.arange(actual * 16).reshape(actual, 16).float()
    router = torch.arange(actual * 4).reshape(actual, 4).float()
    without = prepare(owner, hidden, router, shared, replace)
    context.mc2_mask = torch.arange(8) < actual
    with_mask = prepare(owner, hidden, router, shared, replace)
    for field in ("hidden_states", "router_logits"):
        assert torch.equal(getattr(without, field), getattr(with_mask, field))
    assert without.padded_hidden_states_shape == with_mask.padded_hidden_states_shape
    assert without.mc2_mask is None
    assert torch.equal(with_mask.mc2_mask, context.mc2_mask.chunk(width)[rank])


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize(
    "extra,a5,hierarchy", [(False, False, False), (True, False, False), (True, False, True), (True, True, False)]
)
def test_regular_mc2_native_arguments_do_not_depend_on_mask(quant, extra, a5, hierarchy):
    builder = extract(DISPATCH, "get_dispatch_mc2_kwargs", dict(torch=torch))
    owner = NS(
        a5_need_extra_args=a5,
        need_extra_args=extra,
        need_expert_scale=hierarchy,
        need_comm_alg=hierarchy,
        global_bs=96,
        moe_all_to_all_group_name="ep",
        ep_world_size=16,
        ep_rank_id=0,
    )
    inputs = NS(
        hidden_states=torch.zeros(6, 16),
        topk_ids=torch.zeros(6, 4, dtype=torch.int32),
        topk_weights=torch.zeros(6, 4),
        routing=NS(expert_map=list(range(256)), global_redundant_expert_num=0),
        quant=NS(comm_quant_mode=None, dispatch_with_quant=quant, is_mxfp=a5, mxfp=None),
    )
    outputs = []
    for mask in (None, torch.zeros(6, dtype=torch.bool), torch.ones(6, dtype=torch.bool)):
        inputs.routing.mc2_mask = mask
        outputs.append(builder(owner, inputs))
    assert outputs[0] == outputs[1] == outputs[2]
    assert not any("mask" in key for key in outputs[0])
