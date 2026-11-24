# SPDX-License-Identifier: Apache-2.0
"""Query layout copies shared by ordinary and DCP SFA graphs."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(do_not_specialize=["items", "qs1"])
def _copy_query_bridge(
    query,
    rope,
    query_out,
    rope_out,
    items,
    heads: tl.constexpr,
    query_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    qs0: tl.constexpr,
    qs1,
    qs2: tl.constexpr,
    rs0: tl.constexpr,
    rs1: tl.constexpr,
    rs2: tl.constexpr,
    qo0: tl.constexpr,
    qo1: tl.constexpr,
    ro0: tl.constexpr,
    ro1: tl.constexpr,
    query_block: tl.constexpr,
    rope_block: tl.constexpr,
):
    q_col, r_col = tl.arange(0, query_block), tl.arange(0, rope_block)
    for item in range(tl.program_id(0), items, tl.num_programs(0)):
        row, head = item // heads, item % heads
        q_offset = item * query_dim if qo0 == heads * query_dim and qo1 == query_dim else row * qo0 + head * qo1
        r_offset = item * rope_dim if ro0 == heads * rope_dim and ro1 == rope_dim else row * ro0 + head * ro1
        tl.store(
            query_out + q_offset + q_col,
            tl.load(query + row * qs0 + head * qs1 + q_col * qs2, q_col < query_dim, 0),
            q_col < query_dim,
        )
        tl.store(
            rope_out + r_offset + r_col,
            tl.load(rope + row * rs0 + head * rs1 + r_col * rs2, r_col < rope_dim, 0),
            r_col < rope_dim,
        )


def copy_sfa_query_bridge(query, rope, query_out, rope_out):
    """Copy halfword planes into a graph bridge or disjoint packet views."""
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
        *query_out.stride()[:2],
        *rope_out.stride()[:2],
        triton.next_power_of_2(query_dim),
        triton.next_power_of_2(rope_dim),
    )


def pack_sfa_dcp_queries(query, rope):
    """Construct the collective packet directly from strided projection outputs."""
    packet = query.new_empty((*query.shape[:2], query.shape[-1] + rope.shape[-1]))
    copy_sfa_query_bridge(query, rope, packet[..., :query.shape[-1]], packet[..., query.shape[-1]:])
    return packet


@triton.jit(do_not_specialize=["items", "qs0", "qs1", "rs0", "rs1", "cs", "ss"])
def _rope_query_packet(q, r, cos, sin, bits_out, values_out, items, heads: tl.constexpr,
                       qs0, qs1, rs0, rs1, cs, ss, CONTIGUOUS: tl.constexpr = False):
    q_col = tl.arange(0, 512)
    col = tl.arange(0, 64)
    pair = (col % 32) * 2
    for item in range(tl.program_id(0), items, tl.num_programs(0)):
        row, head = item // heads, item % heads
        tl.store(bits_out + item * 576 + q_col, tl.load(q + row * qs0 + head * qs1 + q_col))
        base = r + row * rs0 + head * rs1
        # Match native InterleaveRope's multiply by -1, including the sign of zero.
        if CONTIGUOUS:
            # Keep the interleaved pairs in one flat tile for the A2 on-chip gathers.
            raw = tl.load(base + col).to(tl.float32)
            half = col // 32
            x = tl.gather(raw, pair + half, 0)
            partner = tl.gather(raw, pair + (1 - half), 0)
            rotated = tl.where(col < 32, partner * -1.0, partner)
        else:
            even = tl.load(base + pair).to(tl.float32)
            odd = tl.load(base + pair + 1).to(tl.float32)
            x = tl.where(col < 32, even, odd)
            rotated = tl.where(col < 32, odd * -1.0, even)
        c = tl.load(cos + row * cs + col).to(tl.float32)
        s = tl.load(sin + row * ss + col).to(tl.float32)
        value = x * c + rotated * s
        tl.store(values_out + item * 576 + 512 + col, value)


def rope_sfa_dcp_query_packet(query, rope, cos, sin, *, contiguous=False):
    """Apply InterleaveRope to the projected Q slice while writing the gather packet."""
    rows, heads, dim = query.shape
    if (
        dim != 512 or rope.shape != (rows, heads, 64) or cos.shape != (rows, 1, 1, 64) or sin.shape != cos.shape
        or query.dtype not in (torch.float16, torch.bfloat16)
        or any(t.dtype != query.dtype or t.device != query.device or t.stride(-1) != 1 for t in (query, rope, cos, sin))
    ):
        raise ValueError("unsupported fused DCP query geometry, dtype or stride")
    packet = query.new_empty((rows, heads, 576))
    init_device_properties_triton()
    _rope_query_packet[(min(rows * heads, get_vectorcore_num()),)](
        query.view(torch.int16), rope, cos, sin, packet.view(torch.int16), packet,
        rows * heads, heads, query.stride(0), query.stride(1), rope.stride(0), rope.stride(1),
        cos.stride(0), sin.stride(0), CONTIGUOUS=contiguous, enable_fp_fusion=False,
    )
    return packet
