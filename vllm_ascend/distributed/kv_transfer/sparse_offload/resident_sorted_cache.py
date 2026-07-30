"""Experimental sorted-shard sparse-cache planner.

This path is deliberately separate from production sparse-index preparation:

* MTP=1 partitions and sorts the already-unique row without deduplication.
* MTP=2 partitions, sorts, and deduplicates the two rows shard-locally.
* no pre-resident slot mapping is calculated.
* each sort/union AIV also intersects exactly one resident token shard.

All persistent state and launch workspaces are allocated by Python before
graph capture. The hot path only invokes custom NPU operators.
"""

from dataclasses import dataclass

import torch

RESIDENT_COUNT_CACHELINE_INTS = 16
RESIDENT_GENERATION_CACHELINE_LONGS = 8
INDEX_TOPK = 2048


def resident_shard_count(mtp: int) -> int:
    """Return the strictly next power of two: MTP=1 -> 2, MTP=2 -> 4."""
    if mtp not in (1, 2):
        raise ValueError("sorted resident path supports only MTP=1 or MTP=2")
    return 1 << mtp.bit_length()


@dataclass
class SortedResidentState:
    """Single-copy sorted ``(token, slot)`` state for graph-safe decode."""

    tokens: torch.Tensor
    slots: torch.Tensor
    counts: torch.Tensor
    generations: torch.Tensor
    dummy_state_base: int


@dataclass
class SortedResidentWorkspace:
    """Every temporary and output used by the sorted resident path."""

    shard_packed: torch.Tensor
    shard_mapping: torch.Tensor
    shard_counts: torch.Tensor
    prior_slots: torch.Tensor
    overwritten_slots: torch.Tensor
    miss_tokens: torch.Tensor
    miss_counts: torch.Tensor
    target_slots: torch.Tensor


def allocate_sorted_resident_state(
    state_capacity: int,
    max_active_requests: int,
    mtp: int,
    *,
    device: torch.device,
) -> SortedResidentState:
    """Allocate persistent sorted state plus request-private dummy rows."""
    if state_capacity < max_active_requests:
        raise ValueError("state capacity must cover every active request")
    shard_count = resident_shard_count(mtp)
    scratch_capacity = mtp * INDEX_TOPK
    state_rows = state_capacity + max_active_requests
    return SortedResidentState(
        tokens=torch.empty(
            (state_rows, shard_count, scratch_capacity),
            dtype=torch.int32,
            device=device,
        ),
        slots=torch.empty(
            (state_rows, shard_count, scratch_capacity),
            dtype=torch.int16,
            device=device,
        ),
        counts=torch.zeros(
            (
                state_rows,
                shard_count,
                RESIDENT_COUNT_CACHELINE_INTS,
            ),
            dtype=torch.int32,
            device=device,
        ),
        generations=torch.full(
            (state_rows, RESIDENT_GENERATION_CACHELINE_LONGS),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        dummy_state_base=state_capacity,
    )


def allocate_sorted_resident_workspace(
    request_count: int,
    mtp: int,
    *,
    device: torch.device,
) -> SortedResidentWorkspace:
    """Allocate fixed-address workspaces for MTP=1 or MTP=2."""
    if request_count <= 0:
        raise ValueError("request count must be positive")
    shard_count = resident_shard_count(mtp)
    capacity = mtp * INDEX_TOPK
    shard_shape = (request_count, shard_count, capacity)
    count_shape = (
        request_count,
        shard_count,
        RESIDENT_COUNT_CACHELINE_INTS,
    )
    return SortedResidentWorkspace(
        shard_packed=torch.empty(shard_shape, dtype=torch.int32, device=device),
        shard_mapping=torch.empty(shard_shape, dtype=torch.int16, device=device),
        shard_counts=torch.zeros(count_shape, dtype=torch.int32, device=device),
        prior_slots=torch.empty(shard_shape, dtype=torch.int16, device=device),
        overwritten_slots=torch.empty(
            (request_count, capacity),
            dtype=torch.uint8,
            device=device,
        ),
        miss_tokens=torch.empty(
            (request_count, capacity),
            dtype=torch.int32,
            device=device,
        ),
        miss_counts=torch.zeros(
            (request_count, RESIDENT_COUNT_CACHELINE_INTS),
            dtype=torch.int32,
            device=device,
        ),
        target_slots=torch.empty(
            (request_count, capacity),
            dtype=torch.int64,
            device=device,
        ),
    )


def prepare_resident_sharded_union_(
    topk_indices: torch.Tensor,
    split_boundary: torch.Tensor,
    row_req_indices: torch.Tensor,
    request_state_indices: torch.Tensor,
    request_state_generations: torch.Tensor,
    state: SortedResidentState,
    workspace: SortedResidentWorkspace,
    *,
    mtp: int,
) -> None:
    """Create sorted shards and intersect them with resident state."""
    resident_shard_count(mtp)
    try:
        op = torch.ops._C_ascend.npu_dsa_resident_sharded_union_
    except AttributeError as error:
        raise RuntimeError(
            "vllm_ascend_C does not expose npu_dsa_resident_sharded_union_; rebuild the extension"
        ) from error
    op(
        topk_indices,
        split_boundary,
        row_req_indices,
        workspace.shard_packed,
        workspace.shard_mapping,
        workspace.shard_counts,
        request_state_indices,
        request_state_generations,
        state.tokens,
        state.slots,
        state.counts,
        state.generations,
        workspace.prior_slots,
        mtp,
        state.dummy_state_base,
    )


def prepare_sorted_resident_cache_(
    topk_indices: torch.Tensor,
    request_block_table: torch.Tensor,
    request_state_indices: torch.Tensor,
    request_state_generations: torch.Tensor,
    state: SortedResidentState,
    workspace: SortedResidentWorkspace,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign miss slots and linearly update the sorted resident state."""
    try:
        op = torch.ops._C_ascend.npu_dsa_resident_sorted_plan_
    except AttributeError as error:
        raise RuntimeError(
            "vllm_ascend_C does not expose the sorted resident planner; rebuild the extension"
        ) from error
    op(
        topk_indices,
        workspace.shard_packed,
        workspace.shard_mapping,
        workspace.shard_counts,
        request_block_table,
        request_state_indices,
        request_state_generations,
        state.tokens,
        state.slots,
        state.counts,
        state.generations,
        workspace.prior_slots,
        workspace.overwritten_slots,
        workspace.miss_tokens,
        workspace.miss_counts,
        workspace.target_slots,
        block_size,
        state.dummy_state_base,
    )
    return (
        workspace.miss_tokens,
        workspace.miss_counts[:, 0],
        workspace.target_slots,
    )
