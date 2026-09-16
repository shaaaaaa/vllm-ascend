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
