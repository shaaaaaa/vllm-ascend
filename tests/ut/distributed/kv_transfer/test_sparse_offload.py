# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the DSA latent KV offload index logic.

Covers the NPU-independent core: the in-memory offload backend, the A1 gather
planning (prefill/decode split + compact remapping), the growing decode-latent pool,
and the manager gather that reads prefill latent from the backend (LMCache) and decode
latent from the pool. The on-NPU kernel wiring is verified by the parity run.
"""

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.decode_latent_pool import (
    GrowingDecodeLatentPool,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_backend import (
    InMemoryLatentOffloadBackend,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_manager import (
    INVALID_TOKEN_INDEX,
    SparseOffloadConfig,
    build_gather_plan,
    resolve_scratch_gather,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.runner_integration import (
    build_manager,
    compute_reserved_bytes,
)

LAYER_NAMES = ["L0", "L1"]


def _cpu_config(**overrides):
    kwargs = dict(
        num_layers=2,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        block_size=4,
        max_num_seqs=2,
        topk_tokens=4,
        pool_num_blocks=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    kwargs.update(overrides)
    return SparseOffloadConfig(**kwargs)


def _build(cfg):
    return build_manager(cfg, LAYER_NAMES)


# ---------------------------------------------------------------- backend
def test_in_memory_backend_save_load_roundtrip():
    backend = InMemoryLatentOffloadBackend(device="cpu")
    latent_dim = 8
    positions = torch.tensor([0, 1, 2, 3, 4])
    latent = torch.arange(5 * latent_dim, dtype=torch.float32).reshape(5, latent_dim)
    backend.save_layer("L0", "r0", positions, latent)

    backend.save_layer("L0", "r1", positions, latent + 100)
    load_buffer = torch.zeros((10, latent_dim))
    backend.register_load_buffer(load_buffer)
    backend.set_load_req_ids(["r0", "r1"])
    backend.wait_for_layer_load("L0", torch.tensor([4, 0, 2]), [0, 2])  # r0:[4,0], r1:[2]
    assert torch.equal(load_buffer[0], latent[4])
    assert torch.equal(load_buffer[1], latent[0])
    assert torch.equal(load_buffer[2], latent[2] + 100)


# --------------------------------------------------------------- gather plan
def test_gather_plan_splits_prefill_and_decode():
    # prompt_len=10: <10 prefill (abs pos, LMCache); >=10 decode (relative pos, pool).
    topk = torch.tensor([[2, 11, 5], [12, 0, INVALID_TOKEN_INDEX]])
    prompt_lens = torch.tensor([10, 10])
    plan = build_gather_plan(topk, prompt_lens, block_size=4, scratch_blocks_per_req=1)

    assert plan.seq_lens_kv.tolist() == [3, 2]
    assert plan.prefill_positions[0].tolist() == [2, INVALID_TOKEN_INDEX, 5]
    assert plan.decode_positions[0].tolist() == [INVALID_TOKEN_INDEX, 11, INVALID_TOKEN_INDEX]
    assert plan.decode_positions[1].tolist() == [12, INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]
    assert plan.prefill_positions[1].tolist() == [INVALID_TOKEN_INDEX, 0, INVALID_TOKEN_INDEX]
    assert plan.sparse_indices[0].tolist() == [0, 1, 2]
    assert plan.dest_slot[1].tolist() == [4, 5, INVALID_TOKEN_INDEX]
    assert plan.scratch_block_table.tolist() == [[0], [1]]


def test_gather_plan_all_invalid_row():
    plan = build_gather_plan(
        torch.tensor([[INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]]),
        torch.tensor([5]), block_size=4, scratch_blocks_per_req=1,
    )
    assert plan.seq_lens_kv.tolist() == [0]
    assert plan.dest_slot[0].tolist() == [INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]


def test_compute_reserved_bytes():
    cfg = _cpu_config()
    # scratch 192; load 192; pool: 2 layers*16 blocks*4*6*4 = 3072.
    assert compute_reserved_bytes(cfg) == 192 + 192 + 3072


# --------------------------------------------------------------- decode pool
def test_decode_pool_append_gather_roundtrip():
    pool = GrowingDecodeLatentPool(
        num_layers=2, block_size=4, kv_lora_rank=4, qk_rope_head_dim=2,
        dtype=torch.float32, device="cpu", chunk_blocks=2,
    )
    latents = torch.randn(6, pool.latent_dim)
    for d in range(6):
        pool.append_token("r0", 1, d, latents[d])
    got = pool.gather("r0", 1, torch.tensor([5, 0, 3]))
    assert torch.equal(got, latents[torch.tensor([5, 0, 3])])
    assert torch.count_nonzero(pool.gather("r0", 0, torch.tensor([0]))) == 0  # layer 0 untouched


def test_decode_pool_grows_on_demand_and_recycles():
    pool = GrowingDecodeLatentPool(
        num_layers=1, block_size=2, kv_lora_rank=2, qk_rope_head_dim=2,
        dtype=torch.float32, device="cpu", chunk_blocks=2,
    )
    assert pool.num_allocated_blocks == 0  # nothing pre-allocated
    one = torch.ones(pool.latent_dim)
    for d in range(5):  # 5 tokens / block_size 2 -> 3 blocks -> 2 chunks of 2 = 4
        pool.append_token("r0", 0, d, one)
    assert pool.num_allocated_blocks == 4
    pool.free_request("r0")
    for d in range(3):
        pool.append_token("r1", 0, d, one)
    assert pool.num_allocated_blocks == 4  # reused recycled blocks, no growth


# --------------------------------------------------------------- manager gather
def test_manager_gather_mixed_sources_roundtrip():
    cfg = _cpu_config()
    mgr = _build(cfg)

    prompt_len = 5
    k_nope = torch.arange(prompt_len * cfg.kv_lora_rank, dtype=torch.float32).reshape(
        prompt_len, cfg.kv_lora_rank
    )
    k_pe = torch.arange(prompt_len * cfg.qk_rope_head_dim, dtype=torch.float32).reshape(
        prompt_len, cfg.qk_rope_head_dim
    )
    mgr.store_prefill_layer("r0", "L1", torch.arange(prompt_len), k_nope, k_pe)

    # one decode token at abs pos 5 (rel 0) stored in the pool.
    dnope = torch.full((1, cfg.kv_lora_rank), 99.0)
    dpe = torch.full((1, cfg.qk_rope_head_dim), 88.0)
    mgr.store_decode_token("r0", "L1", 5, dnope, dpe)

    topk = torch.tensor([[3, 5, 1, INVALID_TOKEN_INDEX]])  # prefill 3, decode 5, prefill 1
    plan = build_gather_plan(topk, torch.tensor([prompt_len]), cfg.block_size, cfg.scratch_blocks_per_req)
    s_knope, s_kpe, sparse_indices, block_tbl, seq_lens_kv = mgr.gather_decode_layer("L1", ["r0"], plan)

    knope_flat = s_knope.view(-1, cfg.kv_lora_rank)
    kpe_flat = s_kpe.view(-1, cfg.qk_rope_head_dim)
    assert torch.equal(knope_flat[0], k_nope[3])   # prefill 3 (backend)
    assert torch.equal(knope_flat[1], dnope[0])    # decode 5 -> pool rel 0
    assert torch.equal(kpe_flat[1], dpe[0])
    assert torch.equal(knope_flat[2], k_nope[1])   # prefill 1 (backend)
    assert sparse_indices[0].tolist() == [0, 1, 2, INVALID_TOKEN_INDEX]


def _toy_attention(q, k_full, v_full):
    scores = (q @ k_full.t()) / (q.shape[-1] ** 0.5)
    return torch.softmax(scores, dim=-1) @ v_full


def test_offload_path_attends_to_exactly_the_selected_tokens():
    """Pre-NPU correctness: gather selected latent into the A1 scratch (prefill from
    backend, decode from pool) and resolve it the way the kernel will -> identical
    attended K/V and attention output vs directly indexing the full latent."""
    torch.manual_seed(0)
    cfg = _cpu_config(topk_tokens=6, block_size=4, max_num_seqs=1)
    mgr = _build(cfg)

    prompt_len = 12
    k_nope = torch.randn(prompt_len, cfg.kv_lora_rank)
    k_pe = torch.randn(prompt_len, cfg.qk_rope_head_dim)
    mgr.store_prefill_layer("r0", "L0", torch.arange(prompt_len), k_nope, k_pe)

    dec_nope = torch.randn(2, cfg.kv_lora_rank)
    dec_pe = torch.randn(2, cfg.qk_rope_head_dim)
    for i in range(2):  # decode tokens at abs 12,13 -> pool rel 0,1
        mgr.store_decode_token("r0", "L0", 12 + i, dec_nope[i : i + 1], dec_pe[i : i + 1])

    full_nope = torch.cat([k_nope, dec_nope], dim=0)
    full_pe = torch.cat([k_pe, dec_pe], dim=0)

    topk = torch.tensor([[9, 13, 2, 12, 0, INVALID_TOKEN_INDEX]])
    plan = build_gather_plan(topk, torch.tensor([prompt_len]), cfg.block_size, cfg.scratch_blocks_per_req)
    s_knope, s_kpe, sparse_indices, block_tbl, seq_lens_kv = mgr.gather_decode_layer("L0", ["r0"], plan)

    got_nope, got_pe = resolve_scratch_gather(
        s_knope, s_kpe, sparse_indices, block_tbl, cfg.block_size, seq_lens_kv
    )[0]
    valid_positions = torch.tensor([9, 13, 2, 12, 0])
    exp_nope = full_nope.index_select(0, valid_positions)
    exp_pe = full_pe.index_select(0, valid_positions)
    assert torch.allclose(got_nope, exp_nope)
    assert torch.allclose(got_pe, exp_pe)

    q = torch.randn(1, cfg.latent_dim)
    out_offload = _toy_attention(q, torch.cat([got_nope, got_pe], -1), torch.cat([got_nope, got_pe], -1))
    out_full = _toy_attention(q, torch.cat([exp_nope, exp_pe], -1), torch.cat([exp_nope, exp_pe], -1))
    assert torch.allclose(out_offload, out_full)


def test_manager_free_request_delegates():
    cfg = _cpu_config()
    mgr = _build(cfg)
    mgr.store_prefill_layer(
        "r0", "L0", torch.arange(3),
        torch.randn(3, cfg.kv_lora_rank), torch.randn(3, cfg.qk_rope_head_dim),
    )
    mgr.store_decode_token("r0", "L0", 0, torch.randn(1, cfg.kv_lora_rank), torch.randn(1, cfg.qk_rope_head_dim))
    mgr.free_request("r0")
    assert ("r0", "L0") not in mgr.backend._store
    assert "r0" not in mgr._paged_latent_pool._req_blocks


# --------------------------------------------------------------- hooks
def test_paged_latent_pool_write_read_and_free():
    from vllm_ascend.distributed.kv_transfer.sparse_offload.paged_latent_pool import (
        PagedLatentPool,
    )

    pool = PagedLatentPool(
        num_layers=2, num_blocks=4, block_size=2, kv_lora_rank=3, qk_rope_head_dim=2,
        dtype=torch.float32, device="cpu",
    )
    # reserve for 3 tokens -> 2 blocks (size 2).
    pool.reserve("r0", 3)
    assert pool.num_free_blocks == 2

    positions = torch.tensor([0, 1, 2])
    slots = pool.slot_mapping("r0", positions)
    # simulate the op writing latent at those slots for layer 1.
    knope, kpe = pool.layer_caches(1)
    vals_n = torch.randn(3, 3)
    vals_p = torch.randn(3, 2)
    knope.view(-1, 3)[slots] = vals_n
    kpe.view(-1, 2)[slots] = vals_p

    # read back via block_table + position resolution (mirrors the kernel).
    bt = pool.block_table("r0", width=4)  # [4] padded
    for i, p in enumerate(positions.tolist()):
        slot = int(bt[p // pool.block_size]) * pool.block_size + p % pool.block_size
        assert torch.equal(knope.view(-1, 3)[slot], vals_n[i])
        assert torch.equal(kpe.view(-1, 2)[slot], vals_p[i])

    # layer 0 untouched.
    assert torch.count_nonzero(pool.layer_caches(0)[0]) == 0
    pool.free_request("r0")
    assert pool.num_free_blocks == 4  # recycled


def test_manager_populate_pool_and_attn_args():
    cfg = _cpu_config(block_size=2, pool_num_blocks=8)
    mgr = _build(cfg)
    # prefill one request of 3 tokens for layer 1; latent from "exec_kv return".
    qsl = torch.tensor([0, 3])
    ctx = torch.tensor([0])
    kn = torch.randn(3, cfg.kv_lora_rank)
    kp = torch.randn(3, cfg.qk_rope_head_dim)
    mgr.populate_pool_layer(["r0"], "L1", qsl, ctx, kn, kp)

    knope, kpe, bt = mgr.pool_attn_args("L1", ["r0"], max_blocks=4)
    assert bt.shape == (1, 4)
    # read back each position via the pool block_table (mirrors the kernel).
    knope_flat = knope.reshape(-1, cfg.kv_lora_rank)
    for p in range(3):
        slot = int(bt[0][p // cfg.block_size]) * cfg.block_size + p % cfg.block_size
        assert torch.equal(knope_flat[slot], kn[p])
    # layer 0 of the pool is untouched.
    knope0, _, _ = mgr.pool_attn_args("L0", ["r0"], max_blocks=4)
    assert torch.count_nonzero(knope0) == 0


def test_hooks_store_prefill_splits_requests_by_csr():
    from vllm_ascend.distributed.kv_transfer.sparse_offload.sfa_hooks import store_prefill

    cfg = _cpu_config()
    mgr = _build(cfg)
    qsl = torch.tensor([0, 3, 5])
    ctx = torch.tensor([0, 10])  # r1 had 10 prior tokens (chunked prefill)
    k_nope = torch.randn(5, cfg.kv_lora_rank)
    k_pe = torch.randn(5, cfg.qk_rope_head_dim)
    store_prefill(mgr, "L0", ["r0", "r1"], qsl, ctx, k_nope, k_pe)

    r1 = mgr.backend._store[("r1", "L0")]
    assert r1.shape[0] == 12
    assert torch.equal(r1[10], torch.cat([k_nope[3], k_pe[3]], -1))


def test_hooks_gather_decode_full_step():
    from vllm_ascend.distributed.kv_transfer.sparse_offload.sfa_hooks import (
        gather_decode,
        store_prefill,
    )

    cfg = _cpu_config(topk_tokens=4, block_size=4, max_num_seqs=1)
    mgr = _build(cfg)
    prompt_len = 6
    kn = torch.randn(prompt_len, cfg.kv_lora_rank)
    kp = torch.randn(prompt_len, cfg.qk_rope_head_dim)
    store_prefill(mgr, "L0", ["r0"], torch.tensor([0, prompt_len]), torch.tensor([0]), kn, kp)

    # current decode token at abs pos 6 (rel 0); indexer picks prefill 2 + this token.
    cur_nope = torch.randn(1, cfg.kv_lora_rank)
    cur_pe = torch.randn(1, cfg.qk_rope_head_dim)
    topk = torch.tensor([[[2, 6, INVALID_TOKEN_INDEX, INVALID_TOKEN_INDEX]]])  # 3-D
    sk, skp, si, bt, sl = gather_decode(
        mgr, "L0", ["r0"], topk, torch.tensor([prompt_len]),
        torch.tensor([6]), cfg.block_size, cur_nope, cur_pe,
    )
    assert torch.equal(sk.view(-1, cfg.kv_lora_rank)[0], kn[2])     # prefill 2 (backend)
    assert torch.equal(sk.view(-1, cfg.kv_lora_rank)[1], cur_nope[0])  # decode 6 (pool)
    assert si[0].tolist()[:2] == [0, 1]
    assert sl[0] == 2


class TestPrepareSparseIndices:
    """Step B2: prepare decode top-k for compact scratch and LMCache."""

    def test_remap_splits_prefill_compact_and_decode_absolute(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as prepare_sparse_indices,
        )

        # req0: prompt 100; selected mixes prefill (5,7,99) and decode (100,103)
        # req1: prompt 200; all prefill
        topk = torch.tensor(
            [[[5, 100, 7, 103, 99]],
             [[10, 11, 12, 13, 14]]], dtype=torch.int32)
        plen = torch.tensor([100, 200])
        new_idx, packed = prepare_sparse_indices(
            topk,
            plen,
            scratch_base=torch.zeros(2, dtype=torch.int32),
            valid_row_indices=torch.arange(2, dtype=torch.int32),
        )

        # prefill entries -> compact ranks in topk order; decode stay absolute
        assert new_idx.tolist() == [[[0, 100, 1, 103, 2]],
                                    [[0, 1, 2, 3, 4]]]
        # packed rows: front-packed prefill positions (LMCache scatter order)
        assert packed.tolist() == [[5, 7, 99, 0, 0],
                                   [10, 11, 12, 13, 14]]
        assert packed.dtype == torch.int32
        # the FA kernel requires int32 sparse indices — dtype must be preserved
        assert new_idx.dtype == topk.dtype

    def test_remap_padding_entries_untouched(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as prepare_sparse_indices,
        )

        topk = torch.tensor([[[3, -1, 8, -1, 4]]], dtype=torch.int32)
        plen = torch.tensor([10])
        new_idx, packed = prepare_sparse_indices(
            topk,
            plen,
            scratch_base=torch.zeros(1, dtype=torch.int32),
            valid_row_indices=torch.zeros(1, dtype=torch.int32),
        )
        assert new_idx.tolist() == [[[0, -1, 1, -1, 2]]]
        assert packed.tolist() == [[3, 8, 4, 0, 0]]

    def test_remap_shape_2d_input(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as prepare_sparse_indices,
        )

        topk = torch.tensor([[2, 50, 1]], dtype=torch.int32)
        new_idx, packed = prepare_sparse_indices(
            topk,
            torch.tensor([40]),
            scratch_base=torch.zeros(1, dtype=torch.int32),
            valid_row_indices=torch.zeros(1, dtype=torch.int32),
        )
        assert new_idx.shape == topk.shape
        assert new_idx.tolist() == [[0, 50, 1]]
        assert packed.tolist() == [[2, 1, 0]]

    def test_remap_mixed_rows_plen_zero_untouched(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as prepare_sparse_indices,
        )

        # row0: decode row (plen 100) -> remapped; row1: prefill row (plen 0) -> untouched
        topk = torch.tensor([[[5, 100, 7]],
                             [[5, 100, 7]]], dtype=torch.int32)
        plen = torch.tensor([100, 0])
        new_idx, packed = prepare_sparse_indices(
            topk,
            plen,
            scratch_base=torch.zeros(2, dtype=torch.int32),
            valid_row_indices=torch.arange(2, dtype=torch.int32),
        )
        assert new_idx.tolist() == [[[0, 100, 1]],
                                    [[5, 100, 7]]]      # prefill row unchanged
        assert packed.tolist()[0] == [5, 7, 0]

    def test_remap_compacts_only_explicit_decode_rows(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as prepare_sparse_indices,
        )

        topk = torch.tensor(
            [
                [[5, 100, 7, -1]],
                [[20, 21, 22, 23]],
                [[3, 200, 4, 201]],
            ],
            dtype=torch.int32,
        )
        split_boundary = torch.tensor([100, 0, 200], dtype=torch.int32)
        valid_rows = torch.tensor([0, 2], dtype=torch.int32)
        scratch_base = torch.tensor([0, 0, 4], dtype=torch.int32)

        new_idx, packed = prepare_sparse_indices(
            topk,
            split_boundary,
            scratch_base=scratch_base,
            valid_row_indices=valid_rows,
        )

        assert new_idx.tolist() == [
            [[0, 100, 1, -1]],
            [[20, 21, 22, 23]],
            [[4, 200, 5, 201]],
        ]
        assert packed.tolist() == [[5, 7, 0, 0], [3, 4, 0, 0]]
        assert packed.shape == (2, 4)

    def test_remap_compact_without_payload(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as prepare_sparse_indices,
        )

        topk = torch.tensor(
            [[5, 100, 7], [10, 11, 12]], dtype=torch.int32
        )
        new_idx, packed = prepare_sparse_indices(
            topk,
            torch.tensor([100, 0], dtype=torch.int32),
            need_packed=False,
            scratch_base=torch.zeros(2, dtype=torch.int32),
            valid_row_indices=torch.tensor([0], dtype=torch.int32),
        )

        assert new_idx.tolist() == [[0, 100, 1], [10, 11, 12]]
        assert packed is None

    def test_prepare_zeroes_graph_padding_when_request_rows_are_provided(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as prepare_sparse_indices,
        )

        topk = torch.arange(4 * 64, dtype=torch.int32).reshape(4, 1, 64)
        new_idx, packed = prepare_sparse_indices(
            topk,
            torch.tensor([128, 128, 0, 0], dtype=torch.int32),
            scratch_base=torch.tensor([0, 64, 0, 0], dtype=torch.int32),
            valid_row_indices=torch.tensor([0, 1], dtype=torch.int32),
            row_req_indices=torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        )

        assert torch.count_nonzero(new_idx[2:]).item() == 0
        assert packed.shape == (2, 64)

    def test_prepare_keeps_mixed_prefill_row_without_request_rows(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as prepare_sparse_indices,
        )

        topk = torch.tensor(
            [[[5, 100, 7, -1]], [[20, 21, 22, 23]]], dtype=torch.int32
        )
        prefill_row = topk[1].clone()
        new_idx, packed = prepare_sparse_indices(
            topk,
            torch.tensor([100, 0], dtype=torch.int32),
            scratch_base=torch.zeros(2, dtype=torch.int32),
            valid_row_indices=torch.tensor([0], dtype=torch.int32),
            row_req_indices=None,
        )

        assert new_idx[0].tolist() == [[0, 100, 1, -1]]
        assert torch.equal(new_idx[1], prefill_row)
        assert packed.tolist() == [[5, 7, 0, 0]]

    def test_public_prepare_does_not_fall_back_to_torch_on_cpu(self):
        import pytest

        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            prepare_sparse_indices,
        )

        with pytest.raises(RuntimeError, match="requires the NPU custom op"):
            prepare_sparse_indices(
                torch.zeros((1, 64), dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                scratch_base=torch.zeros(1, dtype=torch.int32),
                valid_row_indices=torch.zeros(1, dtype=torch.int32),
            )
