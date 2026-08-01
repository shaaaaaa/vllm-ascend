"""Sorted-shard sparse-cache planner for decode-only compact scratch.

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
MAX_INT16_SCRATCH_CAPACITY = 1 << 15
RESIDENT_READ_PROBE_DEBUG_INTS = 32
RESIDENT_FINALIZE_DEBUG_INTS = 16


def resident_shard_count(mtp: int, shard_count: int | None = None) -> int:
    """Resolve value shards: defaults to MTP=1 -> 2 and MTP=2 -> 4."""
    if mtp not in (1, 2):
        raise ValueError("sorted resident path supports only MTP=1 or MTP=2")
    default_shard_count = 1 << mtp.bit_length()
    if shard_count is None or shard_count == default_shard_count:
        return default_shard_count
    if mtp == 1 and shard_count == 4:
        return shard_count
    raise ValueError("resident shard count supports MTP=1 with 2/4 and MTP=2 with 4")


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
    shard_miss_tokens: torch.Tensor
    shard_miss_positions: torch.Tensor
    shard_evictable_slots: torch.Tensor
    miss_tokens: torch.Tensor
    miss_counts: torch.Tensor
    target_slots: torch.Tensor


def allocate_sorted_resident_state(
    state_capacity: int,
    max_active_requests: int,
    mtp: int,
    *,
    device: torch.device,
    shard_count: int | None = None,
) -> SortedResidentState:
    """Allocate persistent sorted state plus request-private dummy rows."""
    if state_capacity < max_active_requests:
        raise ValueError("state capacity must cover every active request")
    shard_count = resident_shard_count(mtp, shard_count)
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
    shard_count: int | None = None,
) -> SortedResidentWorkspace:
    """Allocate fixed-address workspaces for MTP=1 or MTP=2."""
    if request_count <= 0:
        raise ValueError("request count must be positive")
    shard_count = resident_shard_count(mtp, shard_count)
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
        shard_miss_tokens=torch.empty(shard_shape, dtype=torch.int32, device=device),
        shard_miss_positions=torch.empty(shard_shape, dtype=torch.int16, device=device),
        shard_evictable_slots=torch.empty(shard_shape, dtype=torch.int16, device=device),
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


def sorted_resident_workspace_prefix(
    workspace: SortedResidentWorkspace,
    request_count: int,
) -> SortedResidentWorkspace:
    """Return aligned contiguous prefix views for the active request count."""
    if request_count <= 0 or request_count > workspace.shard_packed.shape[0]:
        raise ValueError(
            "active sorted-resident request count is out of range: "
            f"{request_count} not in [1, {workspace.shard_packed.shape[0]}]"
        )
    return SortedResidentWorkspace(
        shard_packed=workspace.shard_packed[:request_count],
        shard_mapping=workspace.shard_mapping[:request_count],
        shard_counts=workspace.shard_counts[:request_count],
        prior_slots=workspace.prior_slots[:request_count],
        shard_miss_tokens=workspace.shard_miss_tokens[:request_count],
        shard_miss_positions=workspace.shard_miss_positions[:request_count],
        shard_evictable_slots=workspace.shard_evictable_slots[:request_count],
        miss_tokens=workspace.miss_tokens[:request_count],
        miss_counts=workspace.miss_counts[:request_count],
        target_slots=workspace.target_slots[:request_count],
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
        workspace.shard_miss_tokens,
        workspace.shard_miss_positions,
        workspace.shard_evictable_slots,
        mtp,
        state.dummy_state_base,
    )


def _run_sorted_resident_plan_(
    op,
    topk_indices: torch.Tensor,
    request_block_table: torch.Tensor,
    request_state_indices: torch.Tensor,
    request_state_generations: torch.Tensor,
    state: SortedResidentState,
    workspace: SortedResidentWorkspace,
    *,
    block_size: int,
) -> None:
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
        workspace.shard_miss_tokens,
        workspace.shard_miss_positions,
        workspace.shard_evictable_slots,
        workspace.miss_tokens,
        workspace.miss_counts,
        workspace.target_slots,
        block_size,
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
    """Run state planning followed by the standalone vector remap."""
    prepare_sorted_resident_cache_no_remap_(
        topk_indices,
        request_block_table,
        request_state_indices,
        request_state_generations,
        state,
        workspace,
        block_size=block_size,
    )
    remap_sorted_resident_cache_(topk_indices, workspace)
    return (
        workspace.miss_tokens,
        workspace.miss_counts[:, 0],
        workspace.target_slots,
    )


def prepare_sorted_resident_cache_no_remap_(
    topk_indices: torch.Tensor,
    request_block_table: torch.Tensor,
    request_state_indices: torch.Tensor,
    request_state_generations: torch.Tensor,
    state: SortedResidentState,
    workspace: SortedResidentWorkspace,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign misses and update resident state without remapping top-k."""
    try:
        plan_op = torch.ops._C_ascend.npu_dsa_resident_sorted_plan_no_remap_
    except AttributeError as error:
        raise RuntimeError(
            "vllm_ascend_C does not expose the no-remap sorted resident planner; rebuild the extension"
        ) from error
    _run_sorted_resident_plan_(
        plan_op,
        topk_indices,
        request_block_table,
        request_state_indices,
        request_state_generations,
        state,
        workspace,
        block_size=block_size,
    )
    return (
        workspace.miss_tokens,
        workspace.miss_counts[:, 0],
        workspace.target_slots,
    )


def remap_sorted_resident_cache_(
    topk_indices: torch.Tensor,
    workspace: SortedResidentWorkspace,
) -> None:
    """Apply the standalone shard-local-rank to resident-slot remap."""
    try:
        op = torch.ops._C_ascend.npu_dsa_resident_sorted_remap_
    except AttributeError as error:
        raise RuntimeError(
            "vllm_ascend_C does not expose the standalone sorted resident remap; rebuild the extension"
        ) from error
    op(
        topk_indices,
        workspace.shard_mapping,
        workspace.shard_counts,
        workspace.prior_slots,
    )


def prepare_sorted_resident_cache_fused_(
    topk_indices: torch.Tensor,
    request_block_table: torch.Tensor,
    request_state_indices: torch.Tensor,
    request_state_generations: torch.Tensor,
    state: SortedResidentState,
    workspace: SortedResidentWorkspace,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the production fused state update and top-k remap."""
    try:
        op = torch.ops._C_ascend.npu_dsa_resident_sorted_plan_
    except AttributeError as error:
        raise RuntimeError(
            "vllm_ascend_C does not expose the fused sorted resident planner; rebuild the extension"
        ) from error
    _run_sorted_resident_plan_(
        op,
        topk_indices,
        request_block_table,
        request_state_indices,
        request_state_generations,
        state,
        workspace,
        block_size=block_size,
    )
    return (
        workspace.miss_tokens,
        workspace.miss_counts[:, 0],
        workspace.target_slots,
    )


def coordinate_sorted_resident_finalize_(
    workspace: SortedResidentWorkspace,
) -> None:
    """Compute only request-global shard prefixes and allocation counts."""
    try:
        op = torch.ops._C_ascend.npu_dsa_resident_finalize_coordinator_
    except AttributeError as error:
        raise RuntimeError(
            "vllm_ascend_C does not expose the resident finalize coordinator; rebuild the extension"
        ) from error
    op(workspace.shard_counts, workspace.miss_counts)


def run_sharded_resident_finalize_(
    request_block_table: torch.Tensor,
    workspace: SortedResidentWorkspace,
    *,
    block_size: int,
) -> None:
    """Self-coordinate, assign slots, and emit LMCache payloads by shard."""
    try:
        op = torch.ops._C_ascend.npu_dsa_resident_sharded_finalize_worker_
    except AttributeError as error:
        raise RuntimeError(
            "vllm_ascend_C does not expose the sharded resident finalize worker; rebuild the extension"
        ) from error
    op(
        workspace.shard_counts,
        workspace.prior_slots,
        workspace.shard_miss_tokens,
        workspace.shard_miss_positions,
        workspace.shard_evictable_slots,
        workspace.miss_tokens,
        workspace.miss_counts,
        workspace.target_slots,
        request_block_table,
        block_size,
    )


def prepare_sorted_resident_cache_coordinated_(
    topk_indices: torch.Tensor,
    request_block_table: torch.Tensor,
    request_state_indices: torch.Tensor,
    request_state_generations: torch.Tensor,
    state: SortedResidentState,
    workspace: SortedResidentWorkspace,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the experimental self-coordinating two-kernel planner."""
    run_sharded_resident_finalize_(
        request_block_table,
        workspace,
        block_size=block_size,
    )
    debug_sorted_resident_update_only_(
        topk_indices,
        request_state_indices,
        request_state_generations,
        state,
        workspace,
    )
    return (
        workspace.miss_tokens,
        workspace.miss_counts[:, 0],
        workspace.target_slots,
    )


def probe_sorted_resident_reads_(
    workspace: SortedResidentWorkspace,
    debug_info: torch.Tensor,
    prior_readback: torch.Tensor,
) -> None:
    """Test-only: publish the finalize kernel's GM-to-UB input view."""
    try:
        op = torch.ops._C_ascend.npu_dsa_resident_sorted_read_probe_
    except AttributeError as error:
        raise RuntimeError(
            "vllm_ascend_C does not expose the sorted resident read probe; rebuild the extension"
        ) from error
    op(
        workspace.shard_counts,
        workspace.prior_slots,
        debug_info,
        prior_readback,
    )


def debug_sorted_resident_finalize_only_(
    request_block_table: torch.Tensor,
    workspace: SortedResidentWorkspace,
    *,
    block_size: int,
    debug_info: torch.Tensor | None = None,
    debug_stage: int = 0,
) -> None:
    """Test-only: run the exact finalize kernel without state update/remap."""
    try:
        op = torch.ops._C_ascend.npu_dsa_resident_sorted_finalize_debug_
    except AttributeError as error:
        raise RuntimeError(
            "vllm_ascend_C does not expose the sorted resident finalize debug op; rebuild the extension"
        ) from error
    if debug_info is None:
        debug_info = workspace.miss_counts
    op(
        workspace.shard_packed,
        workspace.shard_counts,
        workspace.prior_slots,
        workspace.shard_miss_tokens,
        workspace.shard_miss_positions,
        workspace.shard_evictable_slots,
        workspace.miss_tokens,
        workspace.miss_counts,
        workspace.target_slots,
        request_block_table,
        debug_info,
        block_size,
        debug_stage,
    )


def debug_sorted_resident_update_only_(
    topk_indices: torch.Tensor,
    request_state_indices: torch.Tensor,
    request_state_generations: torch.Tensor,
    state: SortedResidentState,
    workspace: SortedResidentWorkspace,
) -> None:
    """Benchmark-only: run the exact production state-update/remap kernel."""
    try:
        op = torch.ops._C_ascend.npu_dsa_resident_sorted_update_debug_
    except AttributeError as error:
        raise RuntimeError(
            "vllm_ascend_C does not expose the sorted resident update debug op; rebuild the extension"
        ) from error
    op(
        topk_indices,
        workspace.shard_packed,
        workspace.shard_mapping,
        workspace.shard_counts,
        workspace.prior_slots,
        request_state_indices,
        request_state_generations,
        state.tokens,
        state.slots,
        state.counts,
        state.generations,
        state.dummy_state_base,
    )
