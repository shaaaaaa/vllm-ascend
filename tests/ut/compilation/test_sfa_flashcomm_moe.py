# SPDX-License-Identifier: Apache-2.0
"""Production MoE methods on CPU; collective stand-ins do not qualify NPU ordering."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from sfa_test_support import definitions, extract

ROOT = Path(__file__).resolve().parents[3]
MOE = ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py"
PREPARE = ROOT / "vllm_ascend/ops/fused_moe/prepare_finalize.py"
COMM = NS(MC2="mc2", ALLTOALL="alltoall", FUSED_MC2="fused", ALLGATHER="gather")


def shared_methods(context, shared_dp, ops, all_reduce=lambda x: x):
    namespace = dict(
        torch=NS(ops=NS(vllm=ops)),
        _EXTRA_CTX=context,
        shared_expert_dp_enabled=lambda: shared_dp,
        MoECommType=COMM,
        tensor_model_parallel_all_reduce=all_reduce,
    )
    methods = ("_prepare_shared_expert_input", "_finalize_shared_expert_output")
    definitions(MOE, methods, namespace, class_name="AscendFusedMoE")
    return type("SharedMethods", (), {name: namespace[name] for name in methods})()


@pytest.mark.parametrize("tokens", [1, 2, 5, 12, 24])
def test_tp_shared_mlp_reconstructs_distinct_tokens_with_padding(tokens):
    """Gather before partial MLPs, sum partials before restoring local token rows."""
    width = 4
    x = torch.arange(tokens * 3, dtype=torch.float64).reshape(tokens, 3) / 100
    first = torch.arange(3 * 8, dtype=x.dtype).reshape(3, 8) / 20
    second = torch.arange(8 * 3, dtype=x.dtype).reshape(8, 3) / 30
    padded = torch.nn.functional.pad(x, (0, 0, 0, -tokens % width))
    shards = padded.chunk(width)
    partials, owners, calls = [], [], []
    for rank in range(width):

        def gather(local, label, rank=rank):
            assert label and torch.equal(local, shards[rank])
            calls.append((rank, "gather"))
            return torch.cat(shards)[:tokens]

        def reduce_scatter(partial, rank=rank):
            assert torch.equal(partial, partials[rank])
            calls.append((rank, "reduce_scatter"))
            total = torch.stack(partials).sum(0)
            return torch.nn.functional.pad(total, (0, 0, 0, -tokens % width)).chunk(width)[rank]

        owner = shared_methods(
            NS(flash_comm_v1_enabled=True),
            False,
            NS(maybe_all_gather_and_maybe_unpad=gather, maybe_pad_and_reduce=reduce_scatter),
        )
        local_first, local_second = first.chunk(width, dim=1)[rank], second.chunk(width)[rank]
        full = owner._prepare_shared_expert_input(shards[rank])
        partials.append(torch.nn.functional.silu(full @ local_first) @ local_second)
        owners.append(owner)
    output = torch.cat([owner._finalize_shared_expert_output(partials[r]) for r, owner in enumerate(owners)])
    reference = torch.nn.functional.silu(x @ first) @ second
    torch.testing.assert_close(output[:tokens], reference)
    assert not output[tokens:].any()
    assert calls == [(r, "gather") for r in range(width)] + [(r, "reduce_scatter") for r in range(width)]


@pytest.mark.parametrize("flash,shared_dp", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("mode", vars(COMM).values())
def test_ordinary_and_replicated_shared_paths_keep_existing_reductions(flash, shared_dp, mode):
    calls = []
    owner = shared_methods(
        NS(flash_comm_v1_enabled=flash, moe_comm_type=mode),
        shared_dp,
        NS(),
        lambda x: calls.append("all_reduce") or x,
    )
    x = torch.ones(3, 2)
    assert owner._prepare_shared_expert_input(x) is x
    assert owner._finalize_shared_expert_output(x) is x
    assert calls == (["all_reduce"] if not shared_dp and mode != COMM.ALLGATHER else [])


@pytest.mark.parametrize("mode", ["MC2", "All2All"])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("tokens", [1, 3, 8])
@pytest.mark.parametrize("mask", [False, True])
def test_routed_layout_does_not_depend_on_shared_weight_placement(mode, replace, tokens, mask):
    context = NS(padded_num_tokens=8, mc2_mask=torch.arange(8) < tokens if mask else None)
    namespace = dict(torch=torch, nn=torch.nn, _EXTRA_CTX=context, QuantType=NS(NONE=0), MoEPrepareOutput=NS)
    definitions(PREPARE, ["prepare"], namespace, class_name="PrepareAndFinalizeWith" + mode)
    prepare = namespace["prepare"]
    hidden = torch.arange(tokens * 3).reshape(tokens, 3).float()
    router = torch.arange(tokens * 2).reshape(tokens, 2).float()
    for rank in range(4):
        owner = NS(tp_size=4, tp_rank=rank)
        outputs = [prepare(owner, hidden, router, shared, replace) for shared in (False, True)]
        extent = tokens if replace else max(tokens, 8 if mode == "MC2" else 4)
        for field, source in (("hidden_states", hidden), ("router_logits", router)):
            expected = source if replace else torch.nn.functional.pad(source, (0, 0, 0, extent - tokens)).chunk(4)[rank]
            for result in outputs:
                torch.testing.assert_close(getattr(result, field), expected)
        assert outputs[0].padded_hidden_states_shape == outputs[1].padded_hidden_states_shape
        for output in outputs:
            if mode == "MC2" and mask:
                torch.testing.assert_close(output.mc2_mask, context.mc2_mask.chunk(4)[rank])
            else:
                assert output.mc2_mask is None


@pytest.mark.parametrize("shared_dp", [False, True])
@pytest.mark.parametrize("replace", [False, True])
def test_routed_finalize_gathers_only_when_input_was_split(shared_dp, replace):
    full = torch.arange(8 * 3).reshape(8, 3).float()
    local = full.chunk(4)[2]
    calls = []

    def gather(outputs, value, group):
        assert value is local and group == "tp"
        calls.append("gather")
        for destination, source in zip(outputs, full.chunk(4)):
            destination.copy_(source)

    namespace = dict(torch=torch, dist=NS(all_gather=gather))
    definitions(PREPARE, ["finalize"], namespace, class_name="PrepareAndFinalizeWithAll2All")
    owner = NS(
        replace_allreduce=replace,
        enable_shared_expert_dp=shared_dp,
        tp_size=4,
        num_tokens=5,
        moe_config=NS(tp_group=NS(device_group="tp")),
    )
    result = namespace["finalize"](owner, local, False, full.shape)
    torch.testing.assert_close(result, local if replace else full[:5])
    assert calls == ([] if replace else ["gather"])


@pytest.mark.parametrize("shared_dp", [False, True])
def test_allgather_finalize_reduces_dp_independently_of_shared_placement(shared_dp):
    calls, full = [], torch.arange(8).reshape(4, 2).float()
    namespace = dict(
        get_dp_group=lambda: NS(reduce_scatter=lambda x, dim: calls.append("dp_reduce") or x[:2]),
        prefill_context_parallel_enable=lambda: False,
    )
    finish = extract(PREPARE, "_finalize_with_dp_group", namespace)
    owner = NS(moe_config=NS(dp_size=2), num_tokens=1, enable_shared_expert_dp=shared_dp)
    torch.testing.assert_close(finish(owner, full, False), full[:1])
    assert calls == ["dp_reduce"]


@pytest.mark.parametrize("shared_dp", [False, True])
def test_legacy_flashcomm_allgather_keeps_ep_path(shared_dp):
    namespace = dict(enable_sp=lambda: True, QuantType=NS(NONE=0))
    definitions(PREPARE, ["prepare", "finalize"], namespace, class_name="PrepareAndFinalizeWithAllGather")
    calls = []
    owner = NS(
        _prepare_with_ep_group=lambda *args: calls.append("ep_prepare") or args,
        _finalize_with_ep_group=lambda x: calls.append("ep_finalize") or x,
    )
    hidden, router = torch.ones(2, 3), torch.ones(2, 5)
    assert namespace["prepare"](owner, hidden, router, shared_dp, True) == (hidden, router, 0)
    assert namespace["finalize"](owner, hidden, False) is hidden
    assert calls == ["ep_prepare", "ep_finalize"]


@pytest.mark.parametrize("overlap", [False, True])
def test_shared_stream_waits_precede_gather_and_join_precedes_finalize(overlap):
    calls = []
    stream = NS(wait_event=lambda event: calls.append(event), wait_stream=lambda _: calls.append("join"))
    namespace = dict(
        torch=NS(npu=NS(current_stream=lambda: stream)),
        shared_experts_calculation_stream=lambda: stream,
        npu_stream_switch=lambda *args, **kwargs: nullcontext(),
    )
    run = extract(MOE, "_forward_shared_experts", namespace)
    owner = NS(
        _shared_experts=object(),
        multistream_overlap_shared_expert=overlap,
        _prepare_shared_expert_input=lambda x: calls.append("gather") or x,
        _shared_experts_part1=lambda x: calls.append("part1") or x,
        _shared_experts_part2=lambda x, y: calls.append("part2") or x,
        _finalize_shared_expert_output=lambda x: calls.append("reduce_scatter") or x,
    )
    x = torch.ones(2, 3)
    events = NS(before_routed_experts="producer", before_dispatch="dispatch", before_combine="combine")
    assert run(owner, x, events) is x
    assert calls == ["producer", "gather", "dispatch", "part1", "combine", "part2"] + (["join"] if overlap else []) + [
        "reduce_scatter"
    ]


def test_gate_overlap_preserves_router_token_shard():
    class Prepared(Exception):
        pass

    local, full, calls = torch.ones(2, 3), torch.ones(8, 3), []
    stream = NS(wait_stream=lambda _: calls.append("wait"))
    context = NS(flash_comm_v1_enabled=True, moe_comm_type=COMM.MC2, in_profile_run=False)
    ops = NS(
        maybe_all_gather_and_maybe_unpad=lambda x, label: calls.append("gather") or full,
        maybe_pad_and_reduce=lambda x: calls.append("reduce_scatter") or local,
    )
    owner = shared_methods(context, False, ops)

    def shared_mlp(x):
        assert x is full
        calls.append("shared_mlp")
        return x

    def select(**kwargs):
        assert kwargs["hidden_states"] is local
        calls.append("router")
        return None, None

    def prepare(**kwargs):
        assert kwargs["hidden_states"] is local and kwargs["replace_allreduce"]
        calls.append("prepare")
        raise Prepared

    context.moe_comm_method = NS(prepare=prepare)
    namespace = dict(
        torch=NS(npu=NS(current_stream=lambda: stream)),
        _EXTRA_CTX=context,
        get_forward_context=lambda: NS(),
        AscendFusedMoE=NS(gate_stream=stream),
        npu_stream_switch=lambda *args, **kwargs: nullcontext(),
        get_flash_common3_context=lambda: NS(shared_experts=shared_mlp),
        set_flash_common3_context=lambda **kwargs: None,
        select_experts=select,
        AllGatherCommImpl=type("AllGather", (), {}),
    )
    definitions(MOE, ["forward_impl"], namespace, class_name="AscendFusedMoE")
    owner.quant_method = object()
    owner.enable_npugraph_ex_static_kernel = False
    owner.multistream_overlap_gate = True
    for name in (
        "top_k",
        "use_grouped_topk",
        "renormalize",
        "topk_group",
        "num_expert_group",
        "custom_routing_function",
        "scoring_func",
        "routed_scaling_factor",
        "e_score_correction_bias",
        "global_num_experts",
        "enable_shared_expert_dp",
        "quant_type",
    ):
        setattr(owner, name, None)
    with pytest.raises(Prepared):
        namespace["forward_impl"](owner, local, torch.ones(2, 5))
    assert calls == ["wait", "gather", "shared_mlp", "reduce_scatter", "router", "prepare"]
