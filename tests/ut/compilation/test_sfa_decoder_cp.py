# SPDX-License-Identifier: Apache-2.0
"""Production CP arithmetic and attention methods with CPU collective stand-ins."""

import gc
import os
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier, local
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import ROOT, extract, load_module
from test_sfa_async_mtp import setup as async_setup  # noqa: F401
from test_sfa_metadata_validation import checks as metadata_checks
from test_sfa_row_layout_cache import builder, common, snapshot


@pytest.fixture
def layout(monkeypatch):
    return load_module(ROOT / "vllm_ascend/attention/sfa_graph_layout.py", "decoder_cp_layout", monkeypatch)


@pytest.mark.parametrize("world", [1, 4])
@pytest.mark.parametrize("widths", [(1,), (2,), (1, 2, 1), (2,) * 12, (1, 1, 0), (2, 2)])
def test_local_causal_lengths_and_padding(layout, world, widths):
    ends = torch.tensor(widths, dtype=torch.int32).cumsum(0).to(torch.int32)
    lengths = torch.tensor([100 + n for n in widths], dtype=torch.int32)
    lengths[-1] = 0  # also covers a nonempty, zero-KV graph-padding sequence
    rows = int(ends[-1])
    local_width = (rows + world - 1) // world
    tensor = torch.arange(rows * 3).reshape(rows, 3)
    pieces = []
    for rank in range(world):
        query, key = torch.empty_like(ends), torch.empty_like(lengths)
        pointers = query.data_ptr(), key.data_ptr()
        layout.cp_sequence_lengths(ends, lengths, rank * local_width, local_width, query, key)
        expected_q, expected_k, previous, count = [], [], 0, 0
        for end, seq in zip(ends.tolist(), lengths.tolist()):
            owned = max(0, min(end, (rank + 1) * local_width) - max(previous, rank * local_width))
            count += owned
            expected_q.append(count)
            expected_k.append(seq - max(end - (rank + 1) * local_width, 0) if owned and seq else 0)
            previous = end
        assert query.tolist() == expected_q
        assert key.tolist() == expected_k
        assert pointers == (query.data_ptr(), key.data_ptr())
        pieces.append(layout.cp_slice(tensor, rank, world, -1))
    gathered = torch.cat(pieces)
    torch.testing.assert_close(gathered[:rows], tensor)
    assert torch.all(gathered[rows:] == -1)


def test_invalid_real_length_is_not_concealed(layout):
    query, key = torch.empty(1, dtype=torch.int32), torch.empty(1, dtype=torch.int32)
    layout.cp_sequence_lengths(torch.tensor([4]), torch.tensor([1]), 0, 1, query, key)
    assert key.item() == -2  # malformed real metadata remains invalid, not clamped


@pytest.mark.parametrize("has_indexer", [False, True])
def test_disabled_cp_releases_unused_kv_before_query_projection(has_indexer):
    refs = []

    def cache_op(*args, **kwargs):
        key, rope = torch.empty(2, 1, 1, 512), torch.empty(2, 1, 1, 64)
        refs.extend((weakref.ref(key), weakref.ref(rope)))
        return None, None, rope, key

    ns = dict(
        torch=torch,
        torch_npu=NS(npu_kv_rmsnorm_rope_cache=cache_op),
        get_weight_prefetch_method=lambda: NS(maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kw: None),
    )
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    pre = extract(path, "_cross_layer_pre_compute", ns)
    exec_kv = extract(path, "exec_kv", ns)

    class ReachedQuery(Exception):
        pass

    def query(x):
        assert len(refs) == 2 and all(ref() is None for ref in refs)
        raise ReachedQuery

    class Projection:
        weight = torch.empty(1)

        def __call__(self, hidden):
            return (torch.zeros(2, 577),)

    subject = NS(
        dsa_cp_decoder=False,
        enable_dsa_cp=False,
        q_lora_rank=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        q_a_layernorm=lambda x: x,
        fused_qkv_a_proj=Projection(),
        has_indexer=has_indexer,
        num_kv_heads=1,
        kv_a_layernorm=NS(weight=torch.ones(512), variance_epsilon=1e-6),
        _q_proj_and_k_up_proj=query,
        indexer_select_pre_process=lambda **kw: (torch.empty(2, 128), None),
        use_sparse_c8_indexer=False,
    )
    subject.exec_kv = lambda *args, **kwargs: exec_kv(subject, *args, **kwargs)
    args = [
        torch.empty(2, 577),
        torch.empty(1, 16, 1, 512),
        torch.empty(1, 16, 1, 64),
        torch.empty(1, 16, 1, 128) if has_indexer else None,
        torch.zeros(2, 1, 1, 64),
        torch.zeros(2, 1, 1, 64),
        torch.arange(2),
        torch.arange(2),
    ] + [None] * 15
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with pytest.raises(ReachedQuery):
            pre(subject, *args)
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("full", [False, True])
def test_real_builder_preserves_global_cache_rows(layout, rank, full):
    subject, ns = builder()
    subject.enable_dsa_cp = subject.dsa_cp_decoder = True
    subject.actual_seq_lengths_query = torch.zeros(9, dtype=torch.int32)
    subject.actual_seq_lengths_key = torch.zeros(9, dtype=torch.int32)
    ns.update(
        cp_sequence_lengths=layout.cp_sequence_lengths,
        cp_slice=layout.cp_slice,
        DSACPContext=NS,
        get_tp_group=lambda: NS(world_size=4, rank_in_group=rank),
        _round_up=lambda n, m: (n + m - 1) // m * m,
    )
    previous = None
    for widths in ((1, 2, 1), (2, 1, 2), (2, 2, 2), (1, 1, 1)):
        item = common(widths, padded=6)
        item.sfa_full_graph = full
        item.indexer_block_table_tensor = item.block_table_tensor + 100
        item.indexer_slot_mapping = item.slot_mapping + 100
        reference, _ = builder()
        expected = reference.build(0, item)
        actual = subject.build(0, item)
        for name, value in snapshot(actual).items():
            if value is not None:
                torch.testing.assert_close(value, snapshot(expected)[name])
        assert actual.cos.shape[0] == 6
        assert actual.slot_mapping.data_ptr() == item.slot_mapping.data_ptr()
        cp = actual.dsa_cp_context
        assert cp.slot_mapping_cp.tolist() == layout.cp_slice(item.slot_mapping, rank, 4, -1).tolist()
        pointers = cp.actual_seq_lengths_query.data_ptr(), cp.actual_seq_lengths_key.data_ptr()
        assert previous is None or previous == pointers
        previous = pointers
        assert cp.actual_seq_lengths_query.numel() == actual.seq_lens.numel()
        if full:
            assert cp.actual_seq_lengths_key[-1] == 0


@pytest.mark.parametrize("tokens", [4, 6])
@pytest.mark.parametrize("rank", range(4))
def test_legacy_cp_slot_views_share_one_padded_allocation(layout, tokens, rank):
    subject, ns = builder()
    subject.enable_dsa_cp = True
    subject.actual_seq_lengths_query = torch.zeros(9, dtype=torch.int32)
    subject.actual_seq_lengths_key = torch.zeros(9, dtype=torch.int32)
    ns.update(
        cp_sequence_lengths=layout.cp_sequence_lengths,
        cp_slice=layout.cp_slice,
        DSACPContext=NS,
        nn=torch.nn,
        get_tp_group=lambda: NS(world_size=4, rank_in_group=rank),
        _round_up=lambda n, m: (n + m - 1) // m * m,
    )
    item = common(padded=tokens)
    result = subject.build(0, item)
    assert (
        result.slot_mapping.untyped_storage().data_ptr()
        == result.dsa_cp_context.slot_mapping_cp.untyped_storage().data_ptr()
    )
    if tokens % 4 == 0:
        assert result.slot_mapping.data_ptr() == item.slot_mapping.data_ptr()


@pytest.mark.parametrize("rank", range(4))
def test_graph_eligibility_checks_local_queries_and_memo_layout(rank):
    make_impl, make_metadata, _, context = metadata_checks.__wrapped__()
    impl, metadata = make_impl(), make_metadata()
    impl.dsa_cp_decoder = impl.enable_dsa_cp = impl.enable_dsa_cp_with_o_proj_tp = True
    impl.tp_rank = rank
    metadata.dsa_cp_context = NS(
        num_tokens=2,
        local_start=rank,
        local_end_with_pad=rank + 1,
        actual_seq_lengths_query=torch.zeros(2, dtype=torch.int32),
        actual_seq_lengths_key=torch.zeros(2, dtype=torch.int32),
    )
    caches = tuple(torch.zeros(5, 4, 1, dim, dtype=torch.float16) for dim in (4, 2, 3))
    memo = {}
    assert impl._cross_layer_ineligible_reason(torch.zeros(1, 4), caches, metadata, metadata_checks=memo) is None
    assert "CP query rows" in impl._cross_layer_ineligible_reason(torch.zeros(2, 4), caches, metadata)
    peer = make_impl()
    peer.dsa_cp_decoder = peer.enable_dsa_cp = peer.enable_dsa_cp_with_o_proj_tp = True
    peer.tp_rank = (rank + 1) % 4
    assert "CP attention metadata" in peer._cross_layer_ineligible_reason(
        torch.zeros(1, 4), caches, metadata, metadata_checks=memo
    )


@pytest.mark.parametrize("rank", range(4))
def test_async_cp_refresh_precedes_replay_and_count_readback(async_setup, layout, rank):  # noqa: F811
    runner, scheduled, shape, events, ns = async_setup
    ns.update(
        cp_sequence_lengths=layout.cp_sequence_lengths, cp_slice=layout.cp_slice, enable_dsa_cp_decoder=lambda: True
    )
    item = runner.item
    item.cum_query_lens = torch.tensor([2, 4, 6, 6, 8], dtype=torch.int32)
    cp = item.dsa_cp_context = NS(
        num_tokens=8,
        num_tokens_pad=8,
        local_start=2 * rank,
        local_end_with_pad=2 * rank + 2,
        slot_mapping_cp=item.slot_mapping[rank * 2 : rank * 2 + 2],
        actual_seq_lengths_query=torch.zeros(5, dtype=torch.int32),
        actual_seq_lengths_key=torch.zeros(5, dtype=torch.int32),
    )
    pointers = cp.actual_seq_lengths_query.data_ptr(), cp.actual_seq_lengths_key.data_ptr()
    runner._update_states(scheduled)
    metadata, _ = runner._build_attention_metadata(**shape)
    expected_q, expected_k = torch.empty_like(item.cum_query_lens), torch.empty_like(item.cum_query_lens)
    layout.cp_sequence_lengths(item.cum_query_lens, metadata["l0"].seq_lens, rank * 2, 2, expected_q, expected_k)
    torch.testing.assert_close(cp.actual_seq_lengths_query, expected_q)
    torch.testing.assert_close(cp.actual_seq_lengths_key, expected_k)
    assert pointers == (cp.actual_seq_lengths_query.data_ptr(), cp.actual_seq_lengths_key.data_ptr())
    assert "sync" not in events and events.count("kernel") == 1
    runner._model_forward()
    assert events.index("replay") < events.index("sync")
    assert runner._async_snapshot is not None
    clear = extract(ROOT / "vllm_ascend/worker/model_runner_v1.py", "_clear_sfa_idle_metadata", {})
    metadata["l0"].decode_remap_boundary = torch.zeros(8)
    clear(metadata["l0"])
    assert cp.actual_seq_lengths_key.eq(0).all() and cp.slot_mapping_cp.eq(-1).all()


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("tokens", [1, 2, 6, 24])
def test_attention_initializes_undeclared_tp_padding(monkeypatch, layout, rank, tokens):
    width = (tokens + 3) // 4
    ns = dict(
        torch=torch,
        get_forward_context=lambda: NS(staged_sfa_graph_key=NS(token_capacity=tokens)),
        AscendAttentionState=NS(DecodeOnly="decode", SpecDecoding="spec"),
    )
    execute = extract(ROOT / "vllm_ascend/attention/sfa_v1.py", "_execute_sparse_flash_attention_process", ns)
    # Emulate TND's unwritten suffix with poison; the caller must initialize it.
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_sparse_flash_attention", lambda **kw: torch.full((width, 4, 2), 123.0), raising=False
    )
    subject = NS(dsa_cp_decoder=True, dsa_shrink_latent=2, tp_rank=rank, scale=1.0)
    output = execute(subject, None, None, (None, None), None, None, None, None, block_table_override=torch.zeros(1, 1))
    valid = max(0, min(width, tokens - rank * width))
    assert output[:valid].eq(123).all()
    assert output[valid:].eq(0).all()


@pytest.mark.parametrize("explicit", [None, False, True])
@pytest.mark.parametrize("role", ["producer", "consumer", "both", "none"])
@pytest.mark.parametrize("p_node", [False, True])
def test_explicit_decoder_role_preserves_prefiller(monkeypatch, explicit, role, p_node):
    kv = (
        None
        if role == "none"
        else NS(is_kv_producer=role in ("producer", "both"), is_kv_consumer=role in ("consumer", "both"))
    )
    config = NS(additional_config={} if explicit is None else {"enable_dsa_cp": explicit}, kv_transfer_config=kv)
    monkeypatch.setitem(sys.modules, "vllm.config", NS(get_current_vllm_config=lambda: config))
    ns = dict(enable_dsa_cp=lambda: explicit is not False, envs_ascend=NS(VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=p_node))
    for name in (
        "dsa_cp_decoder_configured",
        "enable_dsa_cp_decoder",
        "enable_dsa_cp_with_layer_shard",
        "enable_dsa_cp_with_o_proj_tp",
    ):
        extract(ROOT / "vllm_ascend/utils.py", name, ns)
    decoder = explicit is True and role in ("consumer", "both") and not p_node
    assert ns["enable_dsa_cp_decoder"]() == decoder
    assert ns["enable_dsa_cp_with_layer_shard"]() == (
        explicit is not False and role in ("producer", "both") and not decoder
    )
    assert ns["enable_dsa_cp_with_o_proj_tp"]() == (explicit is not False and (role == "none" or decoder))


@pytest.mark.parametrize("tokens", [1, 2, 3, 6, 24])
@pytest.mark.parametrize("rank", range(4))
def test_decoder_mtp_does_not_reduce_replicated_values(monkeypatch, tokens, rank):
    ns = dict(
        torch=torch,
        _EXTRA_CTX=NS(flash_comm_v1_enabled=True),
        enable_dsa_cp=lambda: True,
        enable_dsa_cp_decoder=lambda: True,
        get_tp_group=lambda: NS(world_size=4, rank_in_group=rank),
    )
    path = ROOT / "vllm_ascend/spec_decode/eagle_proposer.py"
    extract(path, "split_inputs_tp_to_sp", ns)
    prepare = extract(path, "maybe_pad_and_reduce", ns)
    forbidden = Mock(side_effect=AssertionError("replicated inputs cannot use SUM"))
    monkeypatch.setattr(torch.ops.vllm, "maybe_pad_and_reduce", forbidden, raising=False)
    hidden, positions = torch.ones(tokens, 3), torch.arange(10, 10 + tokens)
    before = hidden.clone(), positions.clone()
    out, pos = prepare(NS(method="mtp", is_multimodal_model=False), hidden, positions)
    pad = (-tokens) % 4
    torch.testing.assert_close(out, torch.nn.functional.pad(hidden, (0, 0, 0, pad)).chunk(4)[rank])
    torch.testing.assert_close(pos, torch.nn.functional.pad(positions, (0, pad)).chunk(4)[rank])
    torch.testing.assert_close(hidden, before[0])
    torch.testing.assert_close(positions, before[1])
    forbidden.assert_not_called()


@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("capacity", [2, 6, 24])
def test_bridge_fake_keeps_global_transfer_rows(cp, capacity):
    fake = extract(
        ROOT / "vllm_ascend/ops/mla.py",
        "sfa_forward_pre_fake",
        dict(torch=torch, enable_dsa_cp_decoder=lambda: cp, get_tensor_model_parallel_world_size=lambda: 4),
    )
    out = fake(torch.zeros(1, 8), True, torch.zeros(1, 8), "layer", 8, 4, 2, 8, capacity, capacity // 2, 16)
    query_rows = (capacity + 3) // 4 if cp else capacity
    assert [x.shape[0] for x in out] == [query_rows, query_rows, capacity] + [capacity // 2] * 3


@pytest.mark.parametrize("rank", range(4))
def test_cp_diagnostic_distinguishes_query_and_cache_rows(monkeypatch, rank):
    saved = []
    ns = dict(
        torch=torch,
        os=os,
        logger=Mock(),
        TARGET_SFA_DIAG_SCHEMA_VERSION=1,
        target_sfa_session=lambda *a, **kw: (NS(step_id=1, rank=rank), 0),
        topk_physical_block_ids=lambda *a, **kw: [],
        target_sfa_path=lambda *a: "unused",
        atomic_torch_save=lambda payload, path: saved.append(payload),
        target_metadata_snapshot=lambda m: {},
        active_resident_state_snapshot=lambda *a: None,
        cpu_snapshot=lambda x: x,
    )
    monkeypatch.setattr(torch, "npu", NS(synchronize=Mock()), raising=False)
    diagnostic = extract(ROOT / "vllm_ascend/attention/sfa_v1.py", "_target_sfa_diag_pre_retrieve", ns)
    cp = NS(local_start=rank * 2, local_end_with_pad=rank * 2 + 2)
    metadata = NS(
        num_actual_tokens=6,
        dsa_cp_context=cp,
        decode_req_indices=torch.arange(6) // 2,
        block_table=torch.zeros(3, 2),
        decode_union_mapping_workspace=torch.arange(12).reshape(3, 4),
    )
    impl = NS(
        _target_sfa_debug_is_live=lambda _: True,
        tp_rank=rank,
        block_size=1,
        index_topk=2,
        _staged_sfa_bridge_buffers=(torch.ones(2, 4, 3), torch.ones(2, 4, 2), torch.zeros(6, 1, 2)),
        _staged_sfa_capture_state=NS(runtime=None, remap_boundary=torch.zeros(6)),
        _sorted_resident_workspace_views={},
        _sorted_resident_state=None,
    )
    diagnostic(impl, "layer", torch.zeros(3, 4), torch.zeros(3), torch.zeros(3, 4), metadata, NS(), 3)
    assert saved[0]["raw_topk"].shape == (6, 1, 2)
    assert saved[0]["topk_indices"].shape[0] == 6
    assert saved[0]["ql_nope"].shape[0] == (2 if rank < 3 else 0)


class CPUCollectives:
    """Threaded TP rendezvous; checks ordering/shapes, not HCCL/device behavior."""

    def __init__(self):
        self.barrier, self.values, self.kinds = Barrier(4, timeout=10), [None] * 4, [None] * 4

    def exchange(self, rank, tensor, kind):
        self.values[rank], self.kinds[rank] = tensor.clone(), kind
        self.barrier.wait()
        assert len(set(self.kinds)) == 1
        assert len({tuple(x.shape) for x in self.values}) == 1
        result = torch.cat(self.values) if kind == "gather" else torch.cat([x.chunk(4)[rank] for x in self.values])
        self.barrier.wait()
        return result


@pytest.mark.parametrize("tokens", [2, 6, 24])
def test_cp_post_reconstructs_tp_projection_without_weight_switch(monkeypatch, tokens):
    thread, comm = local(), CPUCollectives()
    width = (tokens + 3) // 4
    rows = width * 4
    heads = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
    heads[tokens:].zero_()
    weights = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
    ns = dict(
        torch=torch,
        get_tp_group=lambda: NS(device_group=thread.rank),
        MAX_O_PROJ_PREFETCH_SIZE=0,
        get_weight_prefetch_method=lambda: NS(maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kw: None),
    )
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    names = ("_cross_layer_post_compute", "_handle_o_proj_weight_switch_and_forward")
    impl_type = type("ProductionCPPost", (), {name: extract(path, name, ns) for name in names})

    def all_to_all(output, send, group):
        output.copy_(comm.exchange(group, send, "all_to_all"))

    monkeypatch.setattr(torch.distributed, "all_to_all_single", all_to_all)

    def forward(rank):
        thread.rank = rank
        subject = impl_type()
        subject.dsa_cp_decoder, subject.tp_size, subject.num_heads, subject.v_head_dim = True, 4, 1, 2
        source = heads[rank * width : (rank + 1) * width].clone()
        subject._execute_sparse_flash_attention_process = lambda *a, **kw: source
        subject._v_up_proj = lambda x: x

        class Projection:
            weight = weights[rank * 2 : (rank + 1) * 2]

            def __call__(self, x):
                partial = x @ self.weight
                total = comm.exchange(rank, partial, "gather").reshape(4, rows, 3).sum(0)
                return (total.chunk(4)[rank],)

        subject.o_proj = Projection()
        before = subject.o_proj.weight.data_ptr()
        result = subject._cross_layer_post_compute(
            None, None, None, None, None, None, None, None, torch.empty(width, 3), trace_label="test"
        )
        torch.testing.assert_close(result, (heads @ weights).chunk(4)[rank])
        assert subject.o_proj.weight.data_ptr() == before

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(forward, range(4)))


@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("tokens", [2, 6, 24])
@pytest.mark.parametrize("native_state", [None, "spec", "mixed"])
@pytest.mark.parametrize("paired", [False, True])
def test_graph_pre_cp_publication_and_shared_consumer(monkeypatch, layout, c8, tokens, native_state, paired):
    thread, comm = local(), CPUCollectives()
    key = NS(query_profile="bounded", request_capacity=tokens // 2, max_query_len=2, token_capacity=tokens)
    ns = dict(
        torch=torch,
        cp_slice=layout.cp_slice,
        get_forward_context=lambda: NS(staged_sfa_graph_key=key),
        get_tp_group=lambda: NS(all_gather=lambda x, dim=0: comm.exchange(thread.rank, x, "gather")),
        get_weight_prefetch_method=lambda: NS(maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kw: None),
        StagedSFAQueryProfile=NS(DECODE_BOUNDED="bounded"),
        pack_decode_lanes=layout.pack_decode_lanes,
        unpack_decode_lanes=layout.unpack_decode_lanes,
    )

    def scatter(cache, slots, updates):
        cache[slots.flatten().long()] = updates

    ns["torch_npu"] = NS(npu_scatter_nd_update_=scatter)
    ns.update(
        _dsa_prof=NS(set_step_kind=lambda _: None, begin=lambda _: None, section=lambda _: nullcontext()),
        npu_content_diagnostics_enabled=lambda: False,
        AscendAttentionState=NS(DecodeOnly="decode", SpecDecoding="spec"),
        _dsa_indexer_layer_name=lambda name: name + ".indexer",
        _dsa_index_lmcache_enabled=lambda: True,
        wait_for_kv_layer_from_connector=lambda *a, **kw: None,
        _decode_window_save_window_size=lambda: 0,
        _mtp_dw_diag_enabled=lambda: False,
        _fixed_staged_decode_mtp=lambda *a, **kw: 2 if kw["pure_decode"] else None,
        all_gather_async=lambda x, group, **kw: (comm.exchange(thread.rank, x, "gather"), None),
        DeviceOperator=NS(
            reshape_and_cache=lambda **kw: (
                scatter(kw["key_cache"].view(-1, 1), kw["slot_mapping"], kw["key"].view(-1, 1)),
                scatter(kw["value_cache"].view(-1, 1), kw["slot_mapping"], kw["value"].view(-1, 1)),
            )
        ),
    )
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    names = (
        "_cp_local",
        "_cp_query_rows",
        "_cp_global_topk",
        "_cross_layer_pre_compute",
        "_mask_staged_index_scatter_padding",
        "_get_indexcache_topk_indices",
        "_update_indexcache_topk_indices",
        "forward",
    )
    methods = {name: extract(path, name, ns) for name in names}
    methods["_mask_staged_index_scatter_padding"] = staticmethod(methods["_mask_staged_index_scatter_padding"])
    impl_type = type("ProductionCPPre", (), methods)
    global_hidden = torch.arange(tokens, dtype=torch.float32)[:, None].expand(-1, 3).clone() + 1
    ends = torch.arange(2, tokens + 1, 2, dtype=torch.int32)
    owners = torch.arange(tokens, dtype=torch.int32) // 2
    slots = torch.arange(tokens, dtype=torch.int64)
    requests = tokens // 2
    observed = []

    def rank_forward(rank):
        thread.rank = rank
        impl = impl_type()
        impl.dsa_cp_decoder, impl.tp_size, impl.tp_rank = True, 4, rank
        impl.q_lora_rank, impl.kv_lora_rank, impl.qk_rope_head_dim = 1, 1, 1
        impl.head_dim = 1
        impl.fused_qkv_a_proj = NS(weight=torch.zeros(1))

        class Projection:
            weight = torch.zeros(1)

            def __call__(self, x):
                return (x,)

        impl.fused_qkv_a_proj = Projection()
        impl.q_a_layernorm = lambda x: x
        impl.has_indexer, impl.use_sparse_c8_indexer = True, c8
        impl.skip_topk, impl.index_cache_enabled = False, True
        impl.layer_name, impl._indexcache_topk_staging = "producer", None
        impl.topk_indices_buffer = torch.empty(tokens, 2, dtype=torch.int32)
        impl.indexer_select_pre_process = lambda x, **kw: (
            x[:, :1].to(torch.int8 if c8 else torch.float32),
            x[:, :1].to(torch.float16) if c8 else None,
        )
        impl.exec_kv = lambda x, *args, **kwargs: (x[:, 1:], x[:, :1])
        impl._q_proj_and_k_up_proj = lambda q: (q[:, None, :], q[:, None, :])
        impl.rope_single = lambda q, *args: q
        impl.indexer_select_post_process = lambda x, **kw: x[:, :1].int()[:, None, :].expand(-1, 1, 2).clone()
        impl._prepare_decode_sparse_indices = (
            lambda topk, boundary, row_ids, table, selected, counts, targets, *a, **kw: (
                topk + 100,
                selected,
                counts,
                targets,
            )
        )
        storage = torch.zeros(tokens + 1, 1, 1, 2)
        latent = (
            [storage[..., :1], storage[..., 1:]] if paired else [torch.zeros(tokens + 1, 1, 1, 1) for _ in range(2)]
        )
        index = torch.zeros(tokens + 1, 1, 1, 1, dtype=torch.int8 if c8 else torch.float32)
        scales = torch.zeros(tokens + 1, 1, 1, 1, dtype=torch.float16) if c8 else None
        lengths = torch.full_like(ends, 100)
        local_ends, local_keys = torch.empty_like(ends), torch.empty_like(ends)
        width = (tokens + 3) // 4
        layout.cp_sequence_lengths(ends, lengths, rank * width, width, local_ends, local_keys)
        local_hidden = layout.cp_slice(global_hidden, rank, 4)
        args = (
            local_hidden,
            *latent,
            index,
            torch.zeros(tokens, 1),
            torch.zeros(tokens, 1),
            slots,
            slots,
            ends,
            lengths,
            torch.zeros(requests, 1, dtype=torch.int32),
            torch.ones(tokens),
            owners,
            torch.zeros(requests, 1, dtype=torch.int32),
            torch.zeros(requests, 4, dtype=torch.int32),
            torch.zeros(requests, dtype=torch.int32),
            torch.zeros(requests, 4, dtype=torch.int64),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if native_state is not None:
            impl._layerwise_prefill_p_node = impl.enable_mlapo = impl.enable_dsa_cp_with_layer_shard = False
            impl.is_kv_producer = False
            impl.enable_dsa_cp = impl.enable_dsa_cp_with_o_proj_tp = impl.dsa_offload_unbundle = True
            impl.dsa_shrink_latent = 2
            metadata = NS(
                attn_state=native_state,
                cos=args[4],
                sin=args[5],
                slot_mapping=slots,
                indexer_slot_mapping=slots,
                indexer_c8_slot_mapping=None,
                indexer_block_table=args[10],
                num_actual_tokens=tokens,
                num_input_tokens=tokens,
                num_decode_tokens=tokens,
                req_ids=[],
                dsa_cp_context=NS(
                    slot_mapping_cp=layout.cp_slice(slots, rank, 4, -1),
                    actual_seq_lengths_query=local_ends,
                    actual_seq_lengths_key=local_keys,
                ),
                split_boundary=args[11],
                decode_split_boundary=args[11],
                need_sparse_lmcache_payload=True,
                decode_req_indices=owners,
                decode_req_indices_cpu=owners.tolist(),
                block_table=args[13],
                decode_selected_tokens=args[14],
                decode_selected_counts=args[15],
                decode_target_slot_mapping=args[16],
                resident_state_indices=None,
                resident_state_generations=None,
                decode_union_mapping_workspace=None,
                decode_shard_packed_workspace=None,
                decode_shard_mapping_workspace=None,
                decode_shard_counts_workspace=None,
            )

            class PlanObserved(Exception):
                pass

            def check_plan(raw, boundaries, row_ids, *a, **kw):
                torch.testing.assert_close(raw[:, 0, 0], global_hidden[:, 0].int())
                torch.testing.assert_close(row_ids, owners)
                raise PlanObserved

            impl._prepare_decode_sparse_indices = check_plan
            caches = (*latent, index, scales) if c8 else (*latent, index)
            with pytest.raises(PlanObserved):
                impl.forward("layer", local_hidden, caches, metadata, True, torch.empty_like(local_hidden))
            for cache in latent:
                torch.testing.assert_close(cache[:tokens].flatten(), global_hidden[:, 0])
            assert index[:tokens].flatten().tolist() == global_hidden[:, 0].tolist()
            if c8:
                assert scales[:tokens].flatten().tolist() == global_hidden[:, 0].tolist()
            return True
        producer = impl._cross_layer_pre_compute(
            *args, indexer_scale_cache=scales, cp_query_lens=local_ends, cp_key_lens=local_keys
        )
        assert producer[0].shape[0] == width
        assert producer[2].shape[0] == tokens
        torch.testing.assert_close(impl.topk_indices_buffer[:, 0], global_hidden[:, 0].int())
        torch.testing.assert_close(producer[2][:, 0, 0], global_hidden[:, 0].int() + 100)
        for cache in latent:
            torch.testing.assert_close(cache[:tokens].flatten(), global_hidden[:, 0])
        assert index[:tokens].flatten().tolist() == global_hidden[:, 0].tolist()
        if c8:
            assert scales[:tokens].flatten().tolist() == global_hidden[:, 0].tolist()
        impl.skip_topk, impl.has_indexer = True, False
        impl.shared_resident_plan = NS(active=True, attention_plan=lambda *args: producer[2:])
        # Shared consumer still gathers its current latent, but does not select/gather top-k.
        impl.indexer_select_post_process = Mock(side_effect=AssertionError("shared consumer recomputed top-k"))
        consumer_args = (*args[:3], None, *args[4:])
        consumer = impl._cross_layer_pre_compute(*consumer_args, cp_query_lens=local_ends, cp_key_lens=local_keys)
        assert consumer[2] is producer[2]
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        observed = list(pool.map(rank_forward, range(4)))
    assert all(observed)
