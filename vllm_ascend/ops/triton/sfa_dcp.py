# SPDX-License-Identifier: Apache-2.0
"""Head-parallel exchange for counted, rank-local SFA."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element, get_vectorcore_num, init_device_properties_triton


@triton.jit(do_not_specialize=["rows"])
def _store_owner_keys(take, valid, slots, keys, scales, cache, scale_cache, rows,
                      DIM: tl.constexpr, ROW_STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr, SCALE_STRIDE: tl.constexpr):
    col = tl.arange(0, DIM)
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        if tl.load(valid + row):
            index = tl.load(take + row)
            slot = tl.load(slots + index).to(tl.int64)
            if slot >= 0:
                tl.store(cache + slot * ROW_STRIDE + col * COL_STRIDE, tl.load(keys + row * DIM + col))
                if scales is not None:
                    tl.store(scale_cache + slot * SCALE_STRIDE, tl.load(scales + row))


def store_owner_keys(take, valid, slots, keys, scales, cache):
    rows = take.numel()
    init_device_properties_triton()
    key_cache = cache[2].view(-1, keys.shape[-1])
    scale_cache = cache[3].view(-1) if scales is not None else None
    _store_owner_keys[(min(rows, get_vectorcore_num()),)](
        take, valid, slots, keys.contiguous(), scales, key_cache, scale_cache, rows,
        DIM=keys.shape[-1], ROW_STRIDE=key_cache.stride(0), COL_STRIDE=key_cache.stride(1),
        SCALE_STRIDE=scale_cache.stride(0) if scale_cache is not None else 1,
    )


@triton.jit(do_not_specialize=["rows", "actual_rows"])
def _mask_topk(raw, positions, output, rows, width: tl.constexpr,
               actual_rows, TILE: tl.constexpr):
    col = tl.arange(0, TILE)
    for start in range(tl.program_id(0) * TILE, rows * width, tl.num_programs(0) * TILE):
        offset = start + col
        row = offset // width
        token = tl.load(raw + offset, offset < rows * width, -1)
        position = tl.load(positions + row, (row < rows) & (row < actual_rows), -1)
        tl.store(output + offset, tl.where((token >= 0) & (token <= position), token, -1),
                 offset < rows * width)


@triton.jit(do_not_specialize=["rows", "actual_rows"])
def _prepare_topk(raw, positions, output, counts, row_requests, rows,
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
    _causal_global_ids: bool = False,
) -> None:
    """Keep causal global IDs for the planner and compact dense rows locally.

    Only audited target indexer callers may assert ``_causal_global_ids``.
    Draft rejection can invalidate rows beyond the indexer's causal mask.
    """
    rows, width = topk.shape[0], topk.shape[-1]
    if not local and row_requests is None:
        if _causal_global_ids:
            output.copy_(topk)  # The resident planner mutates this private copy.
            return
        init_device_properties_triton()
        # Graph planners need causal global IDs, not row-wise compaction.
        tile = 256
        if rows:
            _mask_topk[(min(triton.cdiv(rows * width, tile), get_vectorcore_num()),)](
                topk, positions, output, rows, width, actual_rows, tile,
            )
        return
    init_device_properties_triton()
    _prepare_topk[(min(rows, get_vectorcore_num()),)](
        topk, positions, output, counts, row_requests, rows, width, actual_rows, world, rank, block, local
    )


def warmup_sfa_dcp_mask(max_rows: int, width: int, device: torch.device) -> None:
    """Warm eager draft row counts before serving, including non-graph buckets."""
    raw = torch.full((max_rows, 1, width), -1, dtype=torch.int32, device=device)
    output = torch.empty_like(raw)
    counts = raw.new_empty(0)
    position_buffers = [torch.zeros(max_rows, dtype=dtype, device=device) for dtype in (torch.int32, torch.int64)]
    for positions in position_buffers:
        for rows in range(1, max_rows + 1):
            # Global-ID masking ignores counts and DCP ownership. Reuse the
            # live launch path with private buffers, without any collectives.
            prepare_sfa_dcp_topk(raw[:rows], positions[:rows], output[:rows], counts, rows, 1, 0, 1, local=False)
    # Finish private writes before releasing their storage or reporting ready.
    torch.npu.current_stream().synchronize()


@triton.jit(do_not_specialize=["rows"])
def _unpack_queries(packed, query, rope, rows, heads: tl.constexpr,
                    world: tl.constexpr, query_dim: tl.constexpr, rope_dim: tl.constexpr,
                    query_block: tl.constexpr, rope_block: tl.constexpr):
    q_col = tl.arange(0, query_block)
    r_col = tl.arange(0, rope_block)
    for item in range(tl.program_id(0), rows * heads * world, tl.num_programs(0)):
        token, head = item // (heads * world), item % (heads * world)
        source = ((head // heads * rows + token) * heads + head % heads) * (query_dim + rope_dim)
        tl.store(query + item * query_dim + q_col,
                 tl.load(packed + source + q_col, q_col < query_dim, 0), q_col < query_dim)
        tl.store(rope + item * rope_dim + r_col,
                 tl.load(packed + source + query_dim + r_col, r_col < rope_dim, 0), r_col < rope_dim)


def _sfa_dcp_query_packet(query: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    if query.is_contiguous() and rope.is_contiguous():
        return torch.cat((query, rope), dim=-1)
    from vllm_ascend.ops.triton.sfa_query import pack_sfa_dcp_queries

    return pack_sfa_dcp_queries(query, rope)


def _unpack_sfa_dcp_queries(packed, rows, heads, query_dim, rope_dim, world):
    query = packed.new_empty((rows, heads * world, query_dim))
    rope = packed.new_empty((rows, heads * world, rope_dim))
    init_device_properties_triton()
    # Copy raw halfwords so BF16/FP16 NaN payloads and signed zero stay intact.
    _unpack_queries[(min(rows * heads * world, get_vectorcore_num()),)](
        packed.view(torch.int16), query.view(torch.int16), rope.view(torch.int16),
        rows, heads, world, query_dim, rope_dim,
        triton.next_power_of_2(query_dim), triton.next_power_of_2(rope_dim),
    )
    return query, rope


def _gather_sfa_dcp_queries(query: torch.Tensor, rope: torch.Tensor, group) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack rank-major collective storage into contiguous TND query planes."""
    packet = _sfa_dcp_query_packet(query, rope)
    return _unpack_sfa_dcp_queries(
        group.all_gather(packet, dim=0), *query.shape, rope.shape[-1], group.world_size,
    )


@dataclass
class SfaDcpQueryGather:
    # Keep BOTH DMA operands alive until the communication stream rejoins.
    packet: torch.Tensor
    gathered: torch.Tensor
    work: Any
    group: Any

    def wait(self) -> None:
        if self.work is not None:
            self.work.wait()
            self.work = None


def start_sfa_dcp_query_gather(query: torch.Tensor, rope: torch.Tensor, group) -> SfaDcpQueryGather:
    """Submit the existing single gather without joining the current stream."""
    packet = _sfa_dcp_query_packet(query, rope)
    gathered = packet.new_empty((packet.shape[0] * group.world_size, *packet.shape[1:]))
    work = dist.all_gather_into_tensor(gathered, packet, group=group.device_group, async_op=True)
    return SfaDcpQueryGather(packet, gathered, work, group.device_group)


def execute_sfa_dcp(
    query: torch.Tensor, rope: torch.Tensor, kv_cache: tuple[torch.Tensor, ...],
    topk: torch.Tensor, counts: torch.Tensor, block_table: torch.Tensor,
    query_ends: torch.Tensor, local_seq_lens: torch.Tensor, scale: float, group,
    *, prefetch: tuple | None = None, query_gather: SfaDcpQueryGather | None = None,
    prefetch_after_pack: bool = False,
) -> torch.Tensor:
    """Gather only query heads, then return the merged output for this head owner."""
    if query_gather is None:
        query, rope = _gather_sfa_dcp_queries(query, rope, group)
    else:
        query_gather.wait()
        if (query_gather.group is not group.device_group
                or query_gather.packet.shape != (*query.shape[:2], query.shape[-1] + rope.shape[-1])):
            raise ValueError("DCP query gather does not match this attention call")
        query, rope = _unpack_sfa_dcp_queries(
            query_gather.gathered, *query.shape, rope.shape[-1], group.world_size,
        )
    requests = local_seq_lens.shape[0]
    partial = torch.ops._C_ascend.npu_sparse_flash_attention_dcp(
        query, rope, kv_cache[0], kv_cache[1], topk, counts,
        block_table[:requests], query_ends[:requests], local_seq_lens, scale,
    )
    if prefetch is not None and prefetch_after_pack:
        return merge_sfa_dcp(*partial, group.device_group, prefetch=prefetch)
    if prefetch is not None:
        edge, selected, miss_counts, slots, destinations = prefetch
        edge.launch(selected, miss_counts, slots, destinations)
    return merge_sfa_dcp(*partial, group.device_group)


@triton.jit(do_not_specialize=["rows"])
def _pack(partial, maximum, total, send, rows, heads: tl.constexpr,
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


@triton.jit(do_not_specialize=["local_rows"])
def _combine(recv, output, local_rows, world: tl.constexpr, value_dim: tl.constexpr):
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
    *, prefetch: tuple | None = None,
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
    if prefetch is None:
        dist.all_to_all_single(recv, send, group=group)
    else:
        # Submit before recording the copy-stream start, but do not join yet:
        # RetrievalEdge's event must depend on _pack, not all-to-all completion.
        work = dist.all_to_all_single(recv, send, group=group, async_op=True)
        try:
            edge, selected, miss_counts, slots, destinations = prefetch
            edge.launch(selected, miss_counts, slots, destinations)
        finally:
            # Keep both packet buffers alive and order _combine, also if the
            # transfer launch fails. This is the HCCL stream dependency.
            work.wait()
    _combine[(min(rows * local_heads, cores),)](recv, output, rows * local_heads, world, value_dim)
    return output
