# SPDX-License-Identifier: Apache-2.0
"""Head-parallel exchange for counted, rank-local SFA."""

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element, get_vectorcore_num, init_device_properties_triton


@triton.jit
def _pack(partial, maximum, total, send, rows: tl.constexpr, heads: tl.constexpr,
          local_heads: tl.constexpr, value_dim: tl.constexpr):
    features = tl.arange(0, value_dim)
    bytes_ = tl.arange(0, 16)
    for row in range(tl.program_id(0), rows * heads, tl.num_programs(0)):
        token, head = row // heads, row % heads
        peer = head // local_heads
        dest = ((peer * rows + token) * local_heads + head % local_heads) * (value_dim + 16)
        tl.store(send + dest + features, tl.load(partial + row * value_dim + features))
        exp_sum = tl.load(total + row)
        max_logit = tl.where(exp_sum > 0, tl.load(maximum + row), -float("inf"))
        max_bits = tl.cast(max_logit, tl.uint32, bitcast=True)
        sum_bits = tl.cast(tl.where(exp_sum > 0, exp_sum, 0.0), tl.uint32, bitcast=True)
        # Keep both FP32 statistics in the existing aligned BF16/FP16 packet.
        # Adding log(exp_sum) to max_logit here can round away the local mass.
        bits = tl.where(bytes_ < 4, max_bits, sum_bits)
        digits = (bits >> (8 * (bytes_ % 4))) & 255
        tl.store(send + dest + value_dim + bytes_, tl.where(bytes_ < 8, digits, 0))


@triton.jit
def _unpack_stat(ptr):
    bits = tl.load(ptr).to(tl.uint32)
    for byte in tl.static_range(1, 4):
        bits |= tl.load(ptr + byte).to(tl.uint32) << (8 * byte)
    return tl.cast(bits, tl.float32, bitcast=True)


@triton.jit
def _combine(recv, output, local_rows: tl.constexpr, world: tl.constexpr, value_dim: tl.constexpr):
    features = tl.arange(0, value_dim)
    peers = tl.arange(0, world)
    stride = value_dim + 16
    for row in range(tl.program_id(0), local_rows, tl.num_programs(0)):
        stats = recv + (peers * local_rows + row) * stride + value_dim
        maxima = _unpack_stat(stats)
        sums = _unpack_stat(stats + 4)
        best = tl.max(maxima, axis=0)
        best = tl.where(best == -float("inf"), 0.0, best)
        weights = sums * tl.exp(maxima - best)
        total = tl.sum(weights, axis=0)
        # Normalize before multiplying partials so finite BF16 values stay in range.
        weights = weights / tl.where(total > 0, total, 1.0)
        result = tl.zeros((value_dim,), tl.float32)
        for peer in tl.static_range(world):
            offset = (peer * local_rows + row) * stride
            partial = tl.load(recv + offset + features).to(tl.float32)
            result += get_element(weights, (peer,)) * partial
        tl.store(output + row * value_dim + features, result)


def merge_sfa_dcp(
    partial: torch.Tensor,
    maximum: torch.Tensor,
    total: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Exchange only the partials needed by each TP head owner."""
    rows, heads, value_dim = partial.shape
    world = dist.get_world_size(group)
    if world not in (2, 4, 8) or heads % world or not rows or value_dim != 512:
        raise ValueError("invalid head-parallel DCP attention geometry")
    if (
        partial.dtype not in (torch.float16, torch.bfloat16)
        or not partial.is_contiguous()
        or any(
            stat.shape != partial.shape[:2] or stat.dtype != torch.float32
            or stat.device != partial.device or not stat.is_contiguous()
            for stat in (maximum, total)
        )
    ):
        raise ValueError("DCP attention requires contiguous partials and FP32 statistics")
    local_heads = heads // world
    send = partial.new_empty((world, rows, local_heads, value_dim + 16))
    recv = torch.empty_like(send)
    output = partial.new_empty((rows, local_heads, value_dim))
    init_device_properties_triton()
    cores = get_vectorcore_num()
    _pack[(min(rows * heads, cores),)](partial, maximum, total, send, rows, heads, local_heads, value_dim)
    dist.all_to_all_single(recv, send, group=group)
    _combine[(min(rows * local_heads, cores),)](recv, output, rows * local_heads, world, value_dim)
    return output
