# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device metadata for unchanged, request-major Q2 batches."""

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs", "common_capacity", "target_capacity", "local_capacity"])
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
    table_c8=None,
    stride_c8: tl.constexpr = 0,
    live_origins=None,
    seq_local=None,
    local_capacity=0,
    DCP_WORLD: tl.constexpr = 1,
    DCP_RANK: tl.constexpr = 0,
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
    if DCP_WORLD > 1:
        # The scratch prefix is already measured in rank-local slots.
        origins = tl.load(live_origins + tl.where(active, rows, 0), True, other=0).to(tl.int32)
        span = block0 * DCP_WORLD
        tail = tl.minimum(block0, tl.maximum(0, length % span - DCP_RANK * block0))
        local_length = origins + length // span * block0 + tail
        tl.store(seq_local + rows, tl.where(active, local_length, 0), rows < local_capacity)
    # As in token packing below, contiguous stores avoid Ascend lowering paired
    # masked stride-2 stores into unmasked reads/writes of the entire tile.
    token = tl.program_id(0) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    token_row = token // 2
    active_token = token_row < num_reqs
    # Keep the Q2 expansion one-dimensional: CANN 8.5.1 cannot flatten a
    # [BLOCK, 2] UB temporary after its short rows receive alignment padding.
    # Triton-Ascend 3.2.0 gather rejects int32. Move the same 32-bit words
    # through its FP32 path, then restore integers before any arithmetic.
    # Numeric float conversion would lose position bits above 2**24.
    lane = tl.arange(0, 2 * BLOCK) // 2
    position = tl.cast(
        tl.gather(tl.cast(current, tl.float32, bitcast=True), lane, 0),
        tl.int32,
        bitcast=True,
    ) + token % 2
    # The installed scalar-gather lowering loads before applying a mask. Give
    # inactive lanes a valid address and use unmasked loads so the address
    # selection cannot be discarded as redundant with a load mask. Tables are
    # allocated even for padding-only programs; active addresses are unchanged.
    column0 = position // block0
    owns_token = active_token
    if DCP_WORLD > 1:
        origin = tl.cast(
            tl.gather(tl.cast(origins, tl.float32, bitcast=True), lane, 0),
            tl.int32,
            bitcast=True,
        )
        owns_token = active_token & (column0 % DCP_WORLD == DCP_RANK)
        column0 = column0 // DCP_WORLD + origin // block0
    offset0 = tl.where(owns_token, token_row * stride0 + column0, 0)
    offset1 = tl.where(active_token, token_row * stride1 + position // block1, 0)
    block_id0 = tl.load(table0 + offset0, True, other=0)
    block_id1 = tl.load(table1 + offset1, True, other=0)
    tl.store(positions + token, tl.where(active_token, position, 0), token < token_capacity)
    tl.store(slots0 + token, tl.where(owns_token, block_id0 * block0 + position % block0, -1), token < token_capacity)
    tl.store(slots1 + token, tl.where(active_token, block_id1 * block1 + position % block1, -1), token < token_capacity)
    if MIXED_C8:
        physical = (
            block_id1 * 2
            if table_c8 is None
            else tl.load(
                table_c8 + tl.where(active_token, token_row * stride_c8 + position // block1, 0),
                True,
                other=0,
            )
        )
        tl.store(
            slots_c8 + token,
            tl.where(active_token, physical * block1 + position % block1, -1),
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
