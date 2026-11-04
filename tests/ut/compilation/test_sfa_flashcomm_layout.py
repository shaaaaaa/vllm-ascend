# SPDX-License-Identifier: Apache-2.0
"""CPU execution of production layout methods; collectives stand in for NPU TP."""

import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from sfa_test_support import ROOT, extract


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 6, 10, 24])
@pytest.mark.parametrize("rank", range(4))
def test_tp_fake_shapes_match_padded_collectives(tokens, rank):
    context = NS(flash_comm_v1_enabled=True, pad_size=(-tokens) % 4, dp_metadata=None, is_draft_model=False)
    full = torch.arange(tokens * 3).reshape(tokens, 3).float()
    ns = dict(
        torch=torch,
        F=F,
        _EXTRA_CTX=context,
        get_forward_context=lambda: context,
        enable_sp_by_pass=lambda: False,
        is_vl_model=lambda: False,
        get_tensor_model_parallel_world_size=lambda: 4,
        tensor_model_parallel_reduce_scatter=lambda x, _: x.chunk(4)[rank],
        tensor_model_parallel_all_gather=lambda x, _: F.pad(full, (0, 0, 0, context.pad_size)),
    )
    path = ROOT / "vllm_ascend/ops/register_custom_ops.py"
    reduce = extract(path, "_maybe_pad_and_reduce_impl", ns)
    fake_reduce = extract(path, "_maybe_pad_and_reduce_fake", ns)
    gather = extract(path, "_maybe_all_gather_and_maybe_unpad_impl", ns)
    fake_gather = extract(path, "_maybe_all_gather_and_maybe_unpad_fake", ns)
    local = reduce(full)
    assert local.shape == fake_reduce(full).shape == ((tokens + 3) // 4, 3)
    assert gather(local, True).shape == fake_gather(local, True).shape == full.shape
    torch.testing.assert_close(gather(local, True), full)
    assert fake_gather(local, False) is local
    context.flash_comm_v1_enabled = False
    assert fake_reduce(full) is full
    assert fake_gather(local, True) is local


def test_dynamic_compilation_preserves_non_aligned_collective_shapes():
    # CPU AOT validates the fake contracts; it does not compile Ascend kernels.
    context = NS(flash_comm_v1_enabled=True, pad_size=0, dp_metadata=None)
    ns = dict(
        torch=torch,
        _EXTRA_CTX=context,
        get_forward_context=lambda: context,
        get_tensor_model_parallel_world_size=lambda: 4,
        enable_sp_by_pass=lambda: False,
    )
    path = ROOT / "vllm_ascend/ops/register_custom_ops.py"
    fake_reduce = extract(path, "_maybe_pad_and_reduce_fake", ns)
    fake_gather = extract(path, "_maybe_all_gather_and_maybe_unpad_fake", ns)

    @torch.library.custom_op("sfa_flashcomm_layout_test::reduce", mutates_args=())
    def reduce(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 0, 0, context.pad_size)).chunk(4)[0].clone()

    @torch.library.custom_op("sfa_flashcomm_layout_test::gather", mutates_args=())
    def gather(x: torch.Tensor) -> torch.Tensor:
        result = x.repeat(4, 1)
        return result[: -context.pad_size].clone() if context.pad_size else result.clone()

    reduce.register_fake(fake_reduce)
    gather.register_fake(lambda x: fake_gather(x, True))

    def forward(x):
        result = gather(reduce(x))
        return result + torch.ones(result.shape, dtype=result.dtype)

    compiled = torch.compile(forward, backend="aot_eager", dynamic=True)
    for tokens in (8, 6, 2, 10, 24, 3, 1, 8) * 2:
        context.pad_size = (-tokens) % 4
        values = torch.ones(tokens, 3)
        torch.testing.assert_close(compiled(values), forward(values))


@pytest.mark.parametrize("tokens", [1, 2, 3, 6, 10, 24])
@pytest.mark.parametrize("prefix,cp", [("attn.o_proj", False), ("mlp.down_proj", False), ("mlp.down_proj", True)])
def test_projection_fake_matches_real_row_parallel_padding(monkeypatch, tokens, prefix, cp):
    for module_name, class_name in (
        ("vllm.model_executor.layers.linear", "UnquantizedLinearMethod"),
        ("vllm_ascend.quantization.method_adapters", "AscendLinearMethod"),
        ("vllm_ascend.quantization.methods", "AscendW8A8LinearMethod"),
    ):
        monkeypatch.setitem(sys.modules, module_name, NS(**{class_name: type(class_name, (), {})}))
    ctx = NS(flash_comm_v1_enabled=True, pad_size=(-tokens) % 4, mmrs_fusion=False)
    backend = NS(get_hccl_comm_name=lambda _: "cpu-stand-in")
    group = NS(device_group=NS(_get_backend=lambda _: backend))
    weight = torch.arange(15).reshape(3, 5).float()
    layer = NS(
        prefix=prefix,
        tp_size=4,
        tp_rank=0,
        output_size_per_partition=5,
        quant_method=NS(apply=lambda layer, x, bias: x @ weight),
    )
    ctx.no_compile_layers = {prefix: layer}
    ns = dict(
        torch=torch,
        F=F,
        _EXTRA_CTX=ctx,
        enable_dsa_cp=lambda: cp,
        get_forward_context=lambda: ctx,
        get_tp_group=lambda: group,
        tensor_model_parallel_reduce_scatter=lambda x, _: x.chunk(4)[0],
    )
    # torch.device('npu') is queried only for the fake communicator, not computation.
    torch_cpu = NS(**{name: getattr(torch, name) for name in ("Tensor", "empty", "int8")}, device=lambda _: "npu")
    ns["torch"] = torch_cpu
    real = extract(ROOT / "vllm_ascend/ops/linear_op.py", "matmul_and_reduce", ns)
    fake = extract(ROOT / "vllm_ascend/ops/register_custom_ops.py", "_matmul_and_reduce_impl_fake", ns)
    values = torch.ones(tokens, 3)
    actual = real(NS(layer=layer, quant_method=layer.quant_method), values, None)
    assert actual.shape == fake(values, prefix).shape == ((tokens + 3) // 4, 5)
    # Legacy CP o_proj already carries a divisible all-head/TP layout; it must not be padded again.
    if cp:
        layer.prefix = "attn.o_proj"
        padded = torch.ones((tokens + 3) // 4 * 4, 3)
        actual = real(NS(layer=layer, quant_method=layer.quant_method), padded, None)
        assert actual.shape == fake(padded, prefix).shape == (padded.shape[0] // 4, 5)


@pytest.mark.parametrize(
    "fc,cp,staged,pass_sp",
    [
        (True, False, True, False),
        (True, True, True, False),
        (True, False, False, False),
        (False, False, True, False),
        (False, False, True, True),
    ],
)
def test_only_fc_staged_preserves_logical_capacities(fc, cp, staged, pass_sp):
    ns = dict(
        enable_sp=lambda _: fc,
        enable_sp_by_pass=lambda: pass_sp,
        enable_dsa_cp=lambda: cp,
        enable_dsa_cp_decoder=lambda: False,
        round_up=lambda n, m: (n + m - 1) // m * m,
    )
    pad = extract(ROOT / "vllm_ascend/worker/model_runner_v1.py", "_pad_for_sequence_parallelism", ns)
    runner = NS(
        vllm_config=NS(parallel_config=NS(tensor_parallel_size=4)),
        _staged_sfa_graph_capture_sizes=(2, 6) if staged else (),
    )
    for tokens in (1, 2, 3, 6, 24):
        unchanged = not (fc or pass_sp) or (fc and not cp and staged)
        assert pad(runner, tokens) == (tokens if unchanged else (tokens + 3) // 4 * 4)


@pytest.mark.parametrize("placement", [False, True])
@pytest.mark.parametrize("fc", [False, True])
def test_mtp_tail_gather_follows_token_layout(monkeypatch, placement, fc):
    events = []
    gather = lambda x, label: events.append(label) or x.repeat_interleave(4, dim=0)
    monkeypatch.setattr(torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", gather, raising=False)
    fn = extract(
        ROOT / "vllm_ascend/spec_decode/eagle_proposer.py",
        "maybe_all_gather_and_unpad",
        dict(torch=torch, _EXTRA_CTX=NS(flash_comm_v1_enabled=fc)),
    )
    local = torch.ones(2, 3)
    out, pos, hidden = fn(NS(method="mtp", enable_shared_expert_dp=placement), local, torch.zeros(2), local)
    assert out.shape[0] == pos.shape[0] == hidden.shape[0] == (8 if fc else 2)
    assert events == ([True, True] if fc else [])


@pytest.mark.parametrize("tokens", [1, 2, 3, 6, 10, 24])
@pytest.mark.parametrize("rank", range(4))
def test_mtp_replicated_inputs_are_sliced_without_sum_or_source_mutation(monkeypatch, tokens, rank):
    context = NS(flash_comm_v1_enabled=True)
    ns = dict(
        torch=torch,
        _EXTRA_CTX=context,
        enable_dsa_cp=lambda: False,
        get_tp_group=lambda: NS(world_size=4, rank=rank + 8, rank_in_group=rank),
    )
    path = ROOT / "vllm_ascend/spec_decode/eagle_proposer.py"
    ns["split_inputs_tp_to_sp"] = extract(path, "split_inputs_tp_to_sp", ns)
    prepare = extract(path, "maybe_pad_and_reduce", ns)
    forbidden = Mock(side_effect=AssertionError("replicated inputs must not use SUM reduce-scatter"))
    monkeypatch.setattr(torch.ops.vllm, "maybe_pad_and_reduce", forbidden, raising=False)
    hidden = torch.arange(tokens * 3).reshape(tokens, 3).float()
    positions = torch.arange(100, 100 + tokens)
    original = hidden.clone(), positions.clone()
    local_hidden, local_positions = prepare(NS(method="mtp", is_multimodal_model=False), hidden, positions)
    padded = (-tokens) % 4
    torch.testing.assert_close(local_hidden, F.pad(hidden, (0, 0, 0, padded)).chunk(4)[rank])
    torch.testing.assert_close(local_positions, F.pad(positions, (0, padded)).chunk(4)[rank])
    torch.testing.assert_close(hidden, original[0])
    torch.testing.assert_close(positions, original[1])
    forbidden.assert_not_called()


@pytest.mark.parametrize("cp,fc,multimodal", [(True, True, False), (False, False, False), (False, True, True)])
def test_existing_mtp_paths_preserve_preparation_behavior(monkeypatch, cp, fc, multimodal):
    reduce = Mock(side_effect=lambda x: x.clone())
    monkeypatch.setattr(torch.ops.vllm, "maybe_pad_and_reduce", reduce, raising=False)
    fn = extract(
        ROOT / "vllm_ascend/spec_decode/eagle_proposer.py",
        "maybe_pad_and_reduce",
        dict(torch=torch, _EXTRA_CTX=NS(flash_comm_v1_enabled=fc), enable_dsa_cp=lambda: cp,
             enable_dsa_cp_decoder=lambda: False),
    )
    x, positions = torch.ones(4, 3), torch.arange(4)
    out, pos = fn(NS(method="mtp", is_multimodal_model=multimodal), x, positions)
    torch.testing.assert_close(out, x)
    torch.testing.assert_close(pos, positions)
    assert reduce.call_count == (2 if cp and fc and not multimodal else 0)


def test_existing_explicit_split_output_uses_local_tp_rank():
    fn = extract(
        ROOT / "vllm_ascend/spec_decode/eagle_proposer.py",
        "split_inputs_tp_to_sp",
        dict(get_tp_group=lambda: NS(world_size=4, rank=9, rank_in_group=1)),
    )
    x = torch.arange(24).reshape(8, 3)
    out = torch.zeros_like(x)
    local = fn(x, out)
    assert local.data_ptr() == out.data_ptr()
    torch.testing.assert_close(local, x[2:4])


@pytest.fixture
def attention(monkeypatch):
    events = []
    full = torch.arange(18).reshape(6, 3).float()
    local = full[:2]
    context = NS(staged_sfa_graph_key=None, staged_sfa_route=None)

    def gather(x, label):
        events.append(("gather", tuple(x.shape)))
        return full

    monkeypatch.setattr(torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", gather, raising=False)

    class ReachedProjection(Exception):
        pass

    def projection(**kwargs):
        events.append(("projection", tuple(kwargs["hidden_states"].shape)))
        raise ReachedProjection

    ns = dict(
        torch=torch,
        get_forward_context=lambda: context,
        AscendAttentionState=NS(DecodeOnly="decode", SpecDecoding="spec"),
        StagedSFARouteAction=NS(STAGED="staged"),
        _dsa_prof=NS(set_step_kind=lambda _: None, begin=lambda _: None),
        npu_content_diagnostics_enabled=lambda: False,
        MLAPO_MAX_SUPPORTED_TOKENS=128,
    )
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    forward = extract(path, "forward", ns)
    pre = extract(path, "cross_layer_graph_pre", ns)
    subject = NS(
        _layerwise_prefill_p_node=False,
        dsa_offload_unbundle=False,
        has_indexer=True,
        enable_dsa_cp=False,
        dsa_cp_decoder=False,
        enable_dsa_cp_with_o_proj_tp=False,
        enable_mlapo=True,
        use_sparse_c8_indexer=False,
        _sfa_preprocess_with_mlapo=projection,
        _cross_layer_empty_outputs=lambda _: (),
        _cross_layer_kv_cache=lambda *args: (args[1], None, False),
    )
    subject.forward = lambda *args, **kwargs: forward(subject, *args, **kwargs)
    metadata = NS(
        attn_state="decode",
        cos=None,
        sin=None,
        slot_mapping=None,
        indexer_slot_mapping=None,
        req_ids=[],
        cum_query_lens=None,
        seq_lens=None,
        num_input_tokens=6,
    )
    return subject, metadata, context, pre, events, local, full, ReachedProjection


@pytest.mark.parametrize("gather_needed", [False, True])
def test_native_fallback_gathers_once_before_projection(attention, gather_needed):
    subject, metadata, _, pre, events, local, full, reached = attention
    hidden = local if gather_needed else full
    with pytest.raises(reached):
        pre(subject, "layer", hidden, (), metadata, gather_needed, torch.empty_like(hidden))
    assert events == ([("gather", (2, 3))] if gather_needed else []) + [("projection", (6, 3))]


def test_staged_gathers_global_rows_before_eligibility(attention):
    subject, metadata, context, pre, events, local, _, _ = attention
    context.staged_sfa_graph_key = NS(token_capacity=6)
    context.staged_sfa_route = NS(action="staged", graph_key=context.staged_sfa_graph_key)

    def validate(x, *args):
        events.append(("validate", tuple(x.shape)))
        return "test stops before device work"

    subject._cross_layer_ineligible_reason = validate
    with pytest.raises(RuntimeError, match="test stops"):
        pre(subject, "layer", local, (), metadata, True, torch.empty_like(local))
    assert events == [("gather", (2, 3)), ("validate", (6, 3))]


def test_profile_retains_zero_output_without_gather(attention):
    subject, _, _, pre, events, local, _, _ = attention
    subject.enable_dsa_cp_with_layer_shard = False
    output = torch.ones_like(local)
    assert pre(subject, "layer", local, (), None, True, output) == ()
    assert events == []
    assert not torch.count_nonzero(output)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("tokens", [2, 6, 24])
def test_post_attention_keeps_global_queries_and_local_diagnostic_owners(rank, tokens):
    key = NS(token_capacity=tokens)
    context = NS(staged_sfa_graph_key=key, cudagraph_runtime_mode="eager")
    # Graph keys are hashable in production; use a minimal hashable stand-in.
    key = type("Key", (), {"token_capacity": tokens})()
    context.staged_sfa_graph_key = key
    captures = []
    ns = dict(
        torch=torch,
        get_forward_context=lambda: context,
        enable_sp=lambda: True,
        CUDAGraphMode=NS(NONE="eager"),
        queue_staged_graph_stage_fingerprint=lambda **kw: captures.append(kw),
    )
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    post = extract(path, "cross_layer_graph_post", ns)
    queue = extract(path, "queue_staged_graph_post_diagnostic", ns)
    local_rows = (tokens + 3) // 4
    output = torch.empty(local_rows, 12)
    compute = Mock(side_effect=lambda *args, **kwargs: args[-1].fill_(rank + 1))
    subject = NS(
        tp_size=4,
        tp_rank=rank,
        diagnostic_num_cache_layers=78,
        dsa_cp_decoder=False,
        _cp_query_rows=lambda n: n,
        _cross_layer_kv_cache=lambda *args: (args[1], None, False),
        _cross_layer_post_compute=compute,
        _staged_graph_content_diagnostic_enabled=lambda _: True,
        _staged_sfa_capture_state=NS(post_diagnostic_buffers={}),
    )
    owners = torch.arange(tokens) // 2
    owners[0] = -1  # Bounded-layout hole: valid rows need not be a prefix.
    metadata = NS(
        cum_query_lens=None,
        seq_lens=None,
        block_table=None,
        num_actual_tokens=tokens - 1,
        decode_req_indices_cpu=owners,
        decode_request_ids_compact=["r"],
        seq_lens_cpu=None,
        num_decode_tokens=tokens - 1,
        attn_state="decode",
    )
    q = torch.ones(tokens, 4)
    post(subject, "layer", q, q, q, (None, None), metadata, output)
    assert compute.call_args.args[0].shape[0] == tokens
    snapshot = subject._staged_sfa_capture_state.post_diagnostic_buffers[key]
    assert snapshot.shape == (min(local_rows, 4), 12)
    queue(subject, "layer", key, metadata)
    count = min(snapshot.shape[0], max(tokens - rank * local_rows, 0))
    assert captures[0]["components"]["attention_output"].shape == (count, 12)
    torch.testing.assert_close(
        captures[0]["row_request_indices"], owners[rank * local_rows : rank * local_rows + count]
    )


def test_disabled_diagnostics_do_not_query_layout():
    forbidden = Mock(side_effect=AssertionError("disabled diagnostics queried layout"))
    fn = extract(
        ROOT / "vllm_ascend/attention/sfa_v1.py", "queue_staged_graph_post_diagnostic", dict(enable_sp=forbidden)
    )
    fn(NS(_staged_graph_content_diagnostic_enabled=lambda _: False), "layer", None, None)
    forbidden.assert_not_called()
