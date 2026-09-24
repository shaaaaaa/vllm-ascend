# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device metadata for unchanged, request-major Q2 batches (no CP)."""

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs", "common_capacity", "target_capacity"])
def prepare_async_mtp_kernel(
    bases,
    counts,
    positions,
    seq_common,
    seq_target,
    table0,
    table1,
    slots0,
    slots1,
    num_reqs,
    token_capacity: tl.constexpr,
    common_capacity,
    target_capacity,
    stride0: tl.constexpr,
    stride1: tl.constexpr,
    block0: tl.constexpr,
    block1: tl.constexpr,
    BLOCK: tl.constexpr,
    slots_c8=None,
    MIXED_C8: tl.constexpr = False,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = rows < num_reqs
    base = tl.load(bases + rows, active, other=0)
    count = tl.load(counts + rows, active, other=0).to(tl.int32)
    current = base + count
    tl.store(bases + rows, current, active)
    length = tl.where(active, current + 2, 0)
    tl.store(seq_common + rows, length, rows < common_capacity)
    tl.store(seq_target + rows, length, rows < target_capacity)
    for column in tl.static_range(2):
        token = rows * 2 + column
        position = current + column
        block_id0 = tl.load(table0 + rows * stride0 + position // block0, active, other=0)
        block_id1 = tl.load(table1 + rows * stride1 + position // block1, active, other=0)
        tl.store(positions + token, tl.where(active, position, 0), token < token_capacity)
        tl.store(slots0 + token, tl.where(active, block_id0 * block0 + position % block0, -1), token < token_capacity)
        tl.store(slots1 + token, tl.where(active, block_id1 * block1 + position % block1, -1), token < token_capacity)
        if MIXED_C8:
            tl.store(
                slots_c8 + token,
                tl.where(active, block_id1 * 2 * block1 + position % block1, -1),
                token < token_capacity,
            )


@triton.jit(do_not_specialize=["num_reqs"])
def prepare_async_mtp_tokens_kernel(sampled, draft, input_ids, draft_ids, num_reqs, BLOCK: tl.constexpr):
    # Contiguous output avoids Ascend's masked strided-store interleave pass.
    tokens = tl.program_id(0) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    active_tokens = tokens < 2 * num_reqs
    sample = tl.load(sampled + tokens // 2, active_tokens, other=0)
    proposal = tl.load(draft + tokens // 2, active_tokens, other=0).to(tl.int32)
    tl.store(input_ids + tokens, tl.where(tokens % 2 == 0, sample, proposal), active_tokens)
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = rows < num_reqs
    proposal = tl.load(draft + rows, active, other=0).to(tl.int32)
    tl.store(draft_ids + rows, proposal, active)
