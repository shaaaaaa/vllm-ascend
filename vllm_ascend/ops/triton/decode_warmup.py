# SPDX-License-Identifier: Apache-2.0
"""Compile live decode variants before readiness, without executing a model.

JIT warmup accepts dtypes in place of tensor pointers. It compiles the same
specializations without allocating KV, consuming RNG, launching collectives or
creating temporary graphs. Batch extents are runtime arguments; layout, dtype
and optional-pointer variants still need explicit coverage.
"""

import torch
from vllm.triton_utils import HAS_TRITON, triton


def _compile(kernel, *args, **kwargs):
    # JIT warmup converts positional dtypes to mock pointers, not keywords.
    kernel.warmup(*args, grid=(1,), **kwargs)


def warmup_owner_topk(width, publication, strides, target_position, draft, *, layout=None):
    from vllm_ascend.ops.triton.sfa_dcp import _prepare_topk, _restore_owner_topk

    i32 = torch.int32
    variants = {(None, False, False), (target_position, True, False)}
    if draft and layout is None:  # Standalone restore reference. Live MTP is prepared by the planner below.
        variants.update(((i32, True, False), (i32, True, True), (i32, False, True)))
    for position, causal, reject in variants:
        _compile(_restore_owner_topk, i32, torch.int64, position, i32, publication, 1, 1, 1, 1,
                 width, *strides, causal, reject, 256)
    if layout is not None:
        native_positions = {(target_position, False)}
        if draft:
            native_positions.update(((i32, False), (i32, True)))
        for position, reject in native_positions:
            for local, requests in ((True, None), (False, i32), (False, None)):
                _compile(_prepare_topk, i32, position, i32, i32, requests, 1, width, 1, *layout, local,
                         torch.int64, 1, 1, publication, *strides, reject)


def warmup_dcp(width, heads, dtype, world, rank, block, *, draft=False):
    from vllm_ascend.ops.triton.sfa_dcp import _combine, _mask_topk, _pack, _prepare_topk, _unpack_queries
    from vllm_ascend.ops.triton.spec_decode.utils import prepare_dcp_mtp_draft_kernel

    i32, i64, f32 = torch.int32, torch.int64, torch.float32
    for position in (i32, i64):
        _compile(_mask_topk, i32, position, i32, 1, width, 1, 256)
        for local, requests in ((True, None), (False, i32)):
            _compile(_prepare_topk, i32, position, i32, i32, requests, 1, width, 1, world, rank, block, local)
    _compile(_unpack_queries, torch.int16, torch.int16, torch.int16, 1, heads, world, 512, 64, 512, 64)
    _compile(_pack, dtype, f32, f32, dtype, 1, heads * world, heads, 512)
    _compile(_combine, dtype, dtype, heads, world, 512)
    if not draft:
        return
    for rejected in (None, i32):
        for slot0 in (i32, i64):
            for slot1 in (i32, i64):
                _compile(
                    prepare_dcp_mtp_draft_kernel,
                    i32,
                    i32,
                    rejected,
                    i32,
                    i32,
                    slot0,
                    slot1,
                    i32,
                    i32,
                    i32,
                    i32,
                    1,
                    WORLD=world,
                    RANK=rank,
                    BLOCK_SIZE=block,
                    TILE=64,
                )


def warmup_query(heads, query_dim, rope_dim, projected_dim):
    from vllm_ascend.ops.triton.sfa_query import _copy_query_bridge

    # BMM produces head-major Q; MLAPO produces contiguous Q. RoPE normally
    # returns contiguous storage; retain the projected-plane layout as well.
    for qs0 in (query_dim, heads * query_dim):
        for rs1 in (rope_dim, projected_dim):
            for packet in (False, True):
                qo1 = ro1 = query_dim + rope_dim
                if not packet:
                    qo1, ro1 = query_dim, rope_dim
                _compile(
                    _copy_query_bridge,
                    *([torch.int16] * 4),
                    heads,
                    heads,
                    query_dim,
                    rope_dim,
                    qs0,
                    query_dim,
                    1,
                    heads * rs1,
                    rs1,
                    1,
                    heads * qo1,
                    qo1,
                    heads * ro1,
                    ro1,
                    triton.next_power_of_2(query_dim),
                    triton.next_power_of_2(rope_dim),
                )


def warmup_indexer_rope(heads, head_dim, rope_dim, neox, dtype):
    from vllm_ascend.ops.triton.rope import _triton_rope_siso

    _compile(
        _triton_rope_siso,
        dtype,
        heads * head_dim,
        dtype,
        rope_dim,
        dtype,
        rope_dim,
        None,
        None,
        None,
        1,
        heads,
        head_dim,
        rope_dim,
        triton.next_power_of_2(heads),
        triton.next_power_of_2(rope_dim),
        BLOCK_SIZE=triton.next_power_of_2(heads),
        IS_NEOX_STYLE=neox,
        USE_COS_SIN=False,
    )


def warmup_indexer_metadata(width, stride):
    from vllm_ascend.ops.triton.spec_decode.indexer_c8_metadata import _map_indexer_metadata

    for slot_dtype in (torch.int32, torch.int64):
        for source_stride in {width, stride}:
            _compile(
                _map_indexer_metadata,
                torch.int32,
                slot_dtype,
                torch.int32,
                torch.int32,
                torch.int64,
                width,
                1,
                width,
                source_stride,
                BLOCK=256,
            )


def warmup_spec_decode(vocab_size, mtp):
    from vllm_ascend.ops.triton.spec_decode.utils import (
        PREPARE_INPUTS_BLOCK_SIZE,
        prepare_inputs_padded_kernel,
        prepare_next_mtp_tokens_kernel,
    )

    i32, i64 = torch.int32, torch.int64
    _compile(prepare_inputs_padded_kernel, i32, i64, i32, i32, i32, 1, BLOCK_SIZE=PREPARE_INPUTS_BLOCK_SIZE)
    if mtp:
        for width in (1, 2):
            for fixed in (False, True):
                _compile(
                    prepare_next_mtp_tokens_kernel,
                    i32,
                    i32,
                    i32,
                    i64,
                    1,
                    vocab_size,
                    width,
                    1,
                    width,
                    PREPARE_INPUTS_BLOCK_SIZE,
                    i32 if fixed else None,
                    i32 if fixed else None,
                    fixed,
                )


def warmup_sampling(vocab_size, max_requests, spec_tokens):
    from vllm_ascend.ops.triton.bincount import token_bin_counts_and_mask_kernel
    from vllm_ascend.ops.triton.penalty import apply_all_penalties_kernel

    i32, i64, f32, boolean = torch.int32, torch.int64, torch.float32, torch.bool
    # Prompt length and growing output history must not add specializations.
    for token_dtype in (i32, i64):
        _compile(
            token_bin_counts_and_mask_kernel, token_dtype, 1, 1, 1, 1, vocab_size, i32, vocab_size, 1, SEQ_BLOCK=256
        )
    _compile(
        apply_all_penalties_kernel,
        f32,
        boolean,
        boolean,
        i32,
        f32,
        f32,
        f32,
        1,
        vocab_size,
        vocab_size,
        1,
        vocab_size,
        1,
        vocab_size,
        1,
        vocab_size,
        1,
        BLOCK_SIZE=2048,
    )
    if not spec_tokens:
        return
    from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN

    from vllm_ascend.ops.triton.reject_sample import (
        cal_grid_and_block_size,
        expand_kernel,
        rejection_greedy_sample_spec_len_1_triton,
        rejection_greedy_sample_triton,
        rejection_random_sample_kernel,
        sample_recovered_tokens_kernel,
    )

    # Tiling changes above the vector-core count; warming only the maximum
    # would miss the small-batch tile. Ascend's sampler supplies no draft probs.
    for tile in sorted({cal_grid_and_block_size(n)[1] for n in range(1, max_requests + 1)}):
        _compile(rejection_greedy_sample_spec_len_1_triton, i32, i32, i64, i32, 1, BLOCK_SIZE=tile)
        for greedy in (None, boolean):
            _compile(rejection_greedy_sample_triton, i32, i32, i32, i64, i32, greedy, 1, spec_tokens, BLOCK_SIZE=tile)
        _compile(
            rejection_random_sample_kernel,
            i32,
            i32,
            i32,
            None,
            f32,
            i32,
            i32,
            f32,
            boolean,
            spec_tokens,
            vocab_size,
            1,
            NO_DRAFT_PROBS=True,
            BLOCK_SIZE=tile,
        )
        for dtype in (i32, f32):
            _compile(expand_kernel, dtype, dtype, i32, 0, 1, 1, MAX_NUM_TOKENS=MAX_SPEC_LEN, BLOCK_SIZE=tile)
    _compile(
        sample_recovered_tokens_kernel,
        i32,
        i32,
        i32,
        None,
        f32,
        f32,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=True,
        SUB_BLOCK=4096,
        multibuffer=False,
    )


def warmup_decode_kernels(runner):
    """Common V1 startup, regardless of graph, async-MTP or DP policy flags."""
    if not HAS_TRITON:
        return
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    spec_tokens = int(getattr(runner.speculative_config, "num_speculative_tokens", 0))
    if not runner.is_pooling_model:
        warmup_sampling(runner.input_batch.vocab_size, runner.max_num_reqs, spec_tokens)
    drafter = getattr(runner, "drafter", None)
    if spec_tokens and callable(getattr(drafter, "prepare_inputs_padded", None)):
        warmup_spec_decode(runner.input_batch.vocab_size, drafter.method == "mtp" and spec_tokens == 1)
    if not runner.use_sparse:
        return
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    seen_queries, seen_rope, seen_owner_keys = set(), set(), set()
    owner_publications = set()
    for layer in get_layers_from_vllm_config(runner.vllm_config, AttentionLayerBase).values():
        impl = getattr(layer, "impl", None)
        if not isinstance(impl, AscendSFAImpl):
            continue
        geometry = (impl.local_num_heads, impl.kv_lora_rank, impl.qk_rope_head_dim, impl.qk_head_dim)
        if geometry not in seen_queries:
            warmup_query(*geometry)
            if runner.dsa_decoder_dcp and getattr(impl, "_dcp_packet_io", "off") == "fused":
                from vllm_ascend.ops.triton.sfa_query import _rope_query_packet

                heads, dim, rope_dim, projected = geometry
                dtype = runner.dtype
                _compile(_rope_query_packet, torch.int16, dtype, dtype, dtype, torch.int16, dtype,
                         1, heads, dim, dim, heads * projected, projected, rope_dim, rope_dim, enable_fp_fusion=False)
            seen_queries.add(geometry)
        if impl.has_indexer:
            if getattr(runner, "_dcp_indexer_workspace", None) is not None:
                from vllm_ascend.ops.triton.sfa_dcp import _store_owner_keys

                raw = impl.topk_indices_buffer if impl.index_cache_enabled else None
                signature = (raw.dtype, (raw.stride(0), raw.stride(-1))) if raw is not None else (None, (0, 1))
                owner_publications.add((impl.index_topk, *signature))
                key = impl.c8_k_cache_dtype if impl.use_sparse_c8_indexer else runner.dtype
                scale = impl.c8_k_scale_cache_dtype if impl.use_sparse_c8_indexer else None
                if (key, scale, impl.head_dim) not in seen_owner_keys:
                    for slots in (torch.int32, torch.int64):
                        _compile(_store_owner_keys, torch.int64, torch.bool, slots, key, scale, key, scale,
                                 1, impl.head_dim, impl.head_dim, 1, 1)
                    seen_owner_keys.add((key, scale, impl.head_dim))
            for heads in {1, impl.n_head}:
                geometry = (heads, impl.head_dim, impl.qk_rope_head_dim, impl.is_rope_neox_style, runner.dtype)
                if geometry not in seen_rope:
                    warmup_indexer_rope(*geometry)
                    seen_rope.add(geometry)
    for width, publication, strides in owner_publications:
        warmup_owner_topk(width, publication, strides, runner.positions.gpu.dtype, bool(spec_tokens),
                          layout=(runner.dcp_size, runner.dcp_rank, runner.block_size))
    if runner.ascend_config.indexer_hbm_block_map is not None:
        table = runner.input_batch.block_table[1].get_device_tensor()
        warmup_indexer_metadata(table.shape[1], table.stride(0))
    if runner.dsa_decoder_dcp:
        warmup_dcp(
            runner.dsa_index_topk,
            runner.model_config.get_num_attention_heads(runner.parallel_config),
            runner.dtype,
            runner.dcp_size,
            runner.dcp_rank,
            runner.block_size,
            draft=bool(spec_tokens),
        )
