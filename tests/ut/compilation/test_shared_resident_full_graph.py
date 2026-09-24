# SPDX-License-Identifier: Apache-2.0
"""Execute bounded producer/consumer pre-compute with real CPU layout helpers."""

from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from tests.ut.distributed.kv_transfer.test_shared_resident_plan import api as resident_api
from tests.ut.distributed.kv_transfer.test_shared_resident_plan import grouped, load_source, methods


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("c8", [False, True])
def test_bounded_group_packs_and_restores_once_per_invocation(explicit, c8):
    api = resident_api.__wrapped__()
    _, (producer, consumer), group, _ = grouped(api, mtp=2, bounded=True)
    layout = load_source("vllm_ascend/attention/sfa_graph_layout.py", "shared_bounded_layout")
    key = NS(query_profile="bounded", request_capacity=2, max_query_len=2)
    ns = dict(
        torch=torch,
        get_forward_context=lambda: NS(staged_sfa_graph_key=key),
        StagedSFAQueryProfile=NS(DECODE_BOUNDED="bounded"),
        get_weight_prefetch_method=lambda: NS(maybe_prefetch_mla_or_sla_weight_in_current_stream=Mock()),
        torch_npu=NS(npu_scatter_nd_update_=Mock()),
        pack_decode_lanes=Mock(wraps=layout.pack_decode_lanes),
        unpack_decode_lanes=Mock(wraps=layout.unpack_decode_lanes),
    )
    methods("vllm_ascend/attention/sfa_v1.py", {"_cross_layer_pre_compute"}, ns)
    frame = {}
    producer.use_sparse_c8_indexer = c8
    for layer, obj in enumerate((producer, consumer)):
        obj._cross_layer_pre_compute = MethodType(ns["_cross_layer_pre_compute"], obj)
        obj.fused_qkv_a_proj = Mock(
            side_effect=lambda hidden, layer=layer: (torch.full((hidden.shape[0], 4), float(layer)),)
        )
        obj.fused_qkv_a_proj.weight = torch.empty(1)
        obj.q_lora_rank, obj.kv_lora_rank, obj.qk_rope_head_dim = 2, 1, 1
        obj.q_a_layernorm = lambda x: x
        obj._q_proj_and_k_up_proj = lambda q: (q, q)
        obj.rope_single = lambda q, *args: q
        obj.exec_kv = Mock()
        obj.index_cache_enabled = True
        obj.indexer_select_pre_process = Mock(
            side_effect=lambda x, **kwargs: (
                torch.empty(x.shape[0], 4, dtype=torch.int8 if c8 else torch.bfloat16),
                torch.empty(x.shape[0], 1, dtype=torch.float16) if c8 else None,
            )
        )
        obj._mask_staged_index_scatter_padding = lambda slots, values, *args: (slots, values)
        obj.indexer_select_post_process = Mock(side_effect=lambda **kwargs: frame["raw"].clone())
    consumer._get_indexcache_topk_indices = Mock(side_effect=AssertionError("consumer staged raw top-k"))
    writes = [t.clone() for t in group.writes] if explicit else None
    reads = [writes[0], writes[-4], writes[-3][:, 0], writes[-2], writes[-1]] if explicit else None

    # Mix Q1/Q2, then change row order and include an entirely idle DP lane.
    for step, widths in enumerate(((1, 2), (2, 1), (1,), (2, 2), (2,), (1, 1), (0, 0)), 1):
        key.request_capacity = len(widths)
        tokens = 2 * len(widths)
        ends = torch.tensor(widths).cumsum(0)
        rows = torch.tensor([r for r, width in enumerate(widths) for _ in range(width)] + [-1] * (tokens - sum(widths)))
        frame["raw"] = torch.arange(tokens * 2048, dtype=torch.int32).view(tokens, 1, 2048) + step * 10
        kwargs = dict(
            hidden_states=torch.empty(tokens, 4),
            kv_cache_nope=torch.empty(1),
            kv_cache_pe=torch.empty(1),
            cos=None,
            sin=None,
            slot_mapping=torch.arange(tokens),
            indexer_slot_mapping=torch.arange(tokens),
            actual_seq_lengths_query=torch.cat((ends, torch.tensor([tokens]))),
            actual_seq_lengths_key=torch.tensor([10000] * len(widths) + [0]),
            indexer_block_table=torch.zeros(len(widths) + 1, 32),
            remap_boundary=torch.where(rows >= 0, 10000, 0),
            row_req_indices=rows,
            request_block_table=torch.zeros(len(widths) + 1, 32),  # Attention padding is not a planner request.
            selected_packed=group.workspace.miss_tokens,
            selected_counts=group.workspace.miss_counts,
            target_slot_mapping=group.workspace.target_slots,
            local_to_union_workspace=None,
            shard_packed_workspace=None,
            shard_mapping_workspace=None,
            shard_counts_workspace=None,
            request_state_indices=torch.arange(len(widths)),
            request_state_generations=torch.ones(len(widths), dtype=torch.int64),
        )
        first = producer._cross_layer_pre_compute(
            indexer_cache=torch.empty(tokens, 4, dtype=torch.int8 if c8 else torch.bfloat16),
            indexer_scale_cache=torch.empty(tokens, 1, dtype=torch.float16) if c8 else None,
            resident_writes=writes,
            **kwargs,
        )
        second = consumer._cross_layer_pre_compute(indexer_cache=None, resident_reads=reads, **kwargs)
        expected = torch.where((rows >= 0)[:, None, None], frame["raw"] + 100, -1)
        assert torch.equal(first[2], expected) and torch.equal(second[2], expected)
        assert torch.equal(producer.topk_indices_buffer[:tokens], frame["raw"].squeeze(1))
        assert first[2].data_ptr() == (writes[-1] if explicit else group.attention_topk).data_ptr()
        assert all(torch.equal(a, b) for a, b in zip(first[2:], second[2:]))
        assert first[4].stride() == (16,)
        assert producer.exec_kv.call_count == consumer.exec_kv.call_count == step
        assert len(producer.indexer_select_post_process.call_args.kwargs["kv_cache"]) == (4 if c8 else 3)
        assert ns["torch_npu"].npu_scatter_nd_update_.call_count == step * (2 if c8 else 1)
        assert not torch.equal(first[0], second[0])  # Queries remain layer-local.
        assert ns["pack_decode_lanes"].call_count == ns["unpack_decode_lanes"].call_count == step
        assert api[1]["prepare_resident_sharded_union_"].call_count == step
    consumer.indexer_select_pre_process.assert_not_called()
    consumer.indexer_select_post_process.assert_not_called()
    consumer._get_indexcache_topk_indices.assert_not_called()


def test_bounded_storage_is_optional_and_fixed_plan_stays_independent():
    api = resident_api.__wrapped__()
    _, _, fixed, _ = grouped(api)
    _, _, bounded, _ = grouped(api, mtp=2, bounded=True)
    assert fixed.attention_topk is None and len(fixed.writes) == 15
    assert len(bounded.writes) == 16 and bounded.attention_topk.data_ptr() != bounded.topk.data_ptr()
    bounded.attention_topk.fill_(123)
    assert bounded.attention_plan(1, 2)[0].shape == (2, 1, 2048)
    assert bounded.plans[1][0].data_ptr() == bounded.topk.data_ptr()
    with pytest.raises(RuntimeError, match="storage"):
        fixed.attention_plan(1, 1)
