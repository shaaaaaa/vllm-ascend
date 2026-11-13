# SPDX-License-Identifier: Apache-2.0
"""Query layout copies shared by ordinary and DCP SFA graphs."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _copy_query_bridge(
    query,
    rope,
    query_out,
    rope_out,
    items: tl.constexpr,
    heads: tl.constexpr,
    query_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    rs0: tl.constexpr,
    rs1: tl.constexpr,
    rs2: tl.constexpr,
    query_block: tl.constexpr,
    rope_block: tl.constexpr,
):
    q_col, r_col = tl.arange(0, query_block), tl.arange(0, rope_block)
    for item in range(tl.program_id(0), items, tl.num_programs(0)):
        row, head = item // heads, item % heads
        tl.store(
            query_out + item * query_dim + q_col,
            tl.load(query + row * qs0 + head * qs1 + q_col * qs2, q_col < query_dim, 0),
            q_col < query_dim,
        )
        tl.store(
            rope_out + item * rope_dim + r_col,
            tl.load(rope + row * rs0 + head * rs1 + r_col * rs2, r_col < rope_dim, 0),
            r_col < rope_dim,
        )


def copy_sfa_query_bridge(query, rope, query_out, rope_out):
    """Copy strided halfword query planes into the existing contiguous graph bridge."""
    rows, heads, query_dim = query.shape
    rope_dim = rope.shape[-1]
    if rows == 0:
        return
    init_device_properties_triton()
    _copy_query_bridge[(min(rows * heads, get_vectorcore_num()),)](
        query.view(torch.int16),
        rope.view(torch.int16),
        query_out.view(torch.int16),
        rope_out.view(torch.int16),
        rows * heads,
        heads,
        query_dim,
        rope_dim,
        *query.stride(),
        *rope.stride(),
        triton.next_power_of_2(query_dim),
        triton.next_power_of_2(rope_dim),
    )
