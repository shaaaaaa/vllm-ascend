# SPDX-License-Identifier: Apache-2.0
"""Head-parallel exchange for counted, rank-local SFA."""

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element, get_vectorcore_num, init_device_properties_triton


@triton.jit
def _prepare_topk(raw, positions, output, counts, row_requests, rows: tl.constexpr,
                  width: tl.constexpr, actual_rows, world: tl.constexpr,
                  rank: tl.constexpr, block: tl.constexpr, local: tl.constexpr):
    col = tl.arange(0, width)
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        token = tl.load(raw + row * width + col)
        position = tl.where(row < actual_rows, tl.load(positions + row), -1)
        valid = (token >= 0) & (token <= position)
        if row_requests is None:
            local_row = local
        else:
            owner = tl.load(row_requests + tl.where(row < actual_rows, row, 0))
            local_row = (row >= actual_rows) | (owner < 0)
        if local_row:
            stripe = token // block
            valid = valid & (stripe % world == rank)
            prefix = tl.cumsum(valid.to(tl.int32), axis=0)
            count = tl.sum(valid.to(tl.int32), axis=0)
            # The two partitions together address each output column once.
            dst = tl.where(valid, prefix - 1, count + col - prefix)
            value = (stripe // world) * block + token % block
            tl.store(counts + row * 16, count)
        else:
            dst, value = col, token
        tl.store(output + row * width + dst, tl.where(valid, value, -1))


def prepare_sfa_dcp_topk(
    topk: torch.Tensor, positions: torch.Tensor, output: torch.Tensor,
    counts: torch.Tensor, actual_rows: int, world: int, rank: int, block: int,
    *, local: bool, row_requests: torch.Tensor | None = None,
) -> None:
    """Keep causal global IDs for the planner and compact dense rows locally."""
    rows, width = topk.shape[0], topk.shape[-1]
    init_device_properties_triton()
    _prepare_topk[(min(rows, get_vectorcore_num()),)](
        topk, positions, output, counts, row_requests, rows, width, actual_rows, world, rank, block, local
    )


def execute_sfa_dcp(
    query: torch.Tensor, rope: torch.Tensor, kv_cache: tuple[torch.Tensor, ...],
    topk: torch.Tensor, counts: torch.Tensor, block_table: torch.Tensor,
    query_ends: torch.Tensor, local_seq_lens: torch.Tensor, scale: float, group,
    *, prefetch: tuple | None = None,
) -> torch.Tensor:
    """Gather only query heads, then return the merged output for this head owner."""
    packed = group.all_gather(torch.cat((query, rope), dim=-1), dim=1)
    query, rope = packed.split((query.shape[-1], rope.shape[-1]), dim=-1)
    requests = local_seq_lens.shape[0]
    partial = torch.ops._C_ascend.npu_sparse_flash_attention_dcp(
        query.contiguous(), rope.contiguous(), kv_cache[0], kv_cache[1], topk, counts,
        block_table[:requests], query_ends[:requests], local_seq_lens, scale,
    )
    if prefetch is not None:
        edge, selected, miss_counts, slots, destinations = prefetch
        edge.launch(selected, miss_counts, slots, destinations)
    return merge_sfa_dcp(*partial, group.device_group)


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
        max_bits = tl.cast(max_logit, tl.int32, bitcast=True)
        sum_bits = tl.cast(tl.where(exp_sum > 0, exp_sum, 0.0), tl.int32, bitcast=True)
        # Keep both FP32 statistics in the existing aligned BF16/FP16 packet.
        # Adding log(exp_sum) to max_logit here can round away the local mass.
        # CANN 8.5.1 cannot lower the uint32 -> float conversion at the store.
        # Signed shifts preserve each byte after masking, and 0..255 is exact
        # in both packet dtypes. Use signed conversion on unpack as well.
        bits = tl.where(bytes_ < 4, max_bits, sum_bits)
        digits = (bits >> (8 * (bytes_ % 4))) & 255
        tl.store(send + dest + value_dim + bytes_, tl.where(bytes_ < 8, digits, 0))


@triton.jit
def _unpack_stat(ptr):
    bits = tl.load(ptr).to(tl.int32)
    for byte in tl.static_range(1, 4):
        bits |= tl.load(ptr + byte).to(tl.int32) << (8 * byte)
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
