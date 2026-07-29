"""Persistent compact-scratch residency for sparse DSA decode.

The request-union operator produces one unique token row per request and a
position-to-union mapping.  This module turns that transient union into a
persistent scratch-cache plan:

* tokens already present in the request's scratch stay in their old slots;
* only misses are emitted in the LMCache payload;
* misses replace the lowest-numbered slots not protected by this step's union;
* scratch contents that are neither selected nor overwritten remain resident.

Persistent reverse entries use int16 because they contain scratch-slot
indices, not absolute token positions.  Absolute token ids remain int32.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

INVALID_SCRATCH_SLOT = -1
MAX_INT16_SCRATCH_CAPACITY = 1 << 15


class ResidentRequestStateRegistry:
    """Assign stable resident-state rows to scheduler request ids.

    An unscheduled request keeps its row.  Finished requests release it.  If
    more unfinished requests have accumulated than ``max_states``, binding a
    new request evicts the least-recently-used request that is not in the
    current batch.  Eviction only loses a cache hit opportunity; it does not
    affect correctness.
    """

    def __init__(self, max_states: int) -> None:
        if max_states <= 0:
            raise ValueError(f"max_states must be positive, got {max_states}")
        self.max_states = int(max_states)
        self._request_to_state: dict[str, int] = {}
        self._state_to_request: list[str | None] = [None] * self.max_states
        self._signatures: list[Hashable | None] = [None] * self.max_states
        self._generations = np.zeros(self.max_states, dtype=np.int64)
        self._last_used = np.zeros(self.max_states, dtype=np.int64)
        self._clock = 0

    def release(self, request_ids: Sequence[str]) -> None:
        for request_id in request_ids:
            state = self._request_to_state.pop(request_id, None)
            if state is None:
                continue
            self._state_to_request[state] = None
            self._signatures[state] = None
            # A later owner of this row must invalidate each layer's small
            # forward map before accepting reverse-map hits.
            self._generations[state] += 1

    def _allocate(self, request_id: str, protected: set[str]) -> int:
        for state, owner in enumerate(self._state_to_request):
            if owner is None:
                break
        else:
            candidates = [
                state
                for state, owner in enumerate(self._state_to_request)
                if owner not in protected
            ]
            if not candidates:
                raise RuntimeError(
                    "resident request-state capacity is smaller than the "
                    f"active batch: max_states={self.max_states}, "
                    f"active={len(protected)}"
                )
            state = min(candidates, key=self._last_used.__getitem__)
            old_owner = self._state_to_request[state]
            assert old_owner is not None
            del self._request_to_state[old_owner]

        self._state_to_request[state] = request_id
        self._request_to_state[request_id] = state
        self._signatures[state] = None
        self._generations[state] += 1
        return state

    def bind(
        self,
        request_ids: Sequence[str],
        scratch_signatures: Sequence[Hashable],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(request_ids) != len(scratch_signatures):
            raise ValueError(
                "request ids and scratch signatures must have equal length: "
                f"{len(request_ids)} != {len(scratch_signatures)}"
            )
        if len(request_ids) > self.max_states:
            raise RuntimeError(
                "resident request-state capacity is smaller than the active "
                f"batch: max_states={self.max_states}, active={len(request_ids)}"
            )

        protected = set(request_ids)
        states = np.empty(len(request_ids), dtype=np.int32)
        generations = np.empty(len(request_ids), dtype=np.int64)
        for row, (request_id, signature) in enumerate(
            zip(request_ids, scratch_signatures, strict=True)
        ):
            state = self._request_to_state.get(request_id)
            if state is None:
                state = self._allocate(request_id, protected)
            if self._signatures[state] != signature:
                self._signatures[state] = signature
                self._generations[state] += 1
            self._clock += 1
            self._last_used[state] = self._clock
            states[row] = state
            generations[row] = self._generations[state]
        return states, generations


def validate_resident_shapes(
    token_to_slot: torch.Tensor,
    slot_to_token: torch.Tensor,
    scratch_capacity: int,
) -> None:
    if scratch_capacity <= 0 or scratch_capacity >= MAX_INT16_SCRATCH_CAPACITY:
        raise ValueError(
            "resident sparse scratch requires "
            f"0 < scratch_capacity < {MAX_INT16_SCRATCH_CAPACITY}, got "
            f"{scratch_capacity}"
        )
    if token_to_slot.dtype != torch.int16:
        raise TypeError(
            "resident token_to_slot must use int16 storage, got "
            f"{token_to_slot.dtype}"
        )
    if slot_to_token.dtype != torch.int32:
        raise TypeError(
            "resident slot_to_token must use int32 storage, got "
            f"{slot_to_token.dtype}"
        )
    if token_to_slot.dim() != 2 or slot_to_token.dim() != 2:
        raise ValueError("resident maps must both be rank-two tensors")
    # Each row owns one extra sentinel element.  Invalid fixed-shape scatter
    # positions target that element and never alias another request's row.
    if slot_to_token.shape[1] < scratch_capacity + 1:
        raise ValueError(
            "slot_to_token needs one request-local sentinel entry: "
            f"shape={tuple(slot_to_token.shape)}, capacity={scratch_capacity}"
        )


@dataclass
class ResidentSparseWorkspace:
    """Caller-owned fixed-address workspaces for graph-safe resident planning."""

    union_tokens: torch.Tensor
    valid_union: torch.Tensor
    old_slots_i16: torch.Tensor
    old_slots: torch.Tensor
    candidate_tokens: torch.Tensor
    hit_mask: torch.Tensor
    protected: torch.Tensor
    available_mask: torch.Tensor
    available_prefix: torch.Tensor
    available_by_rank: torch.Tensor
    miss_mask: torch.Tensor
    miss_prefix: torch.Tensor
    union_to_slot: torch.Tensor
    miss_payload: torch.Tensor
    miss_slot_payload: torch.Tensor
    state_token_indices: torch.Tensor
    state_slot_indices: torch.Tensor
    gather_indices: torch.Tensor
    safe_states: torch.Tensor
    current_generations: torch.Tensor
    generation_matches: torch.Tensor
    valid_states: torch.Tensor
    slot_ids: torch.Tensor
    row_ids: torch.Tensor
    int_sources: torch.Tensor
    short_sources: torch.Tensor


def allocate_resident_workspace(
    max_requests: int,
    scratch_capacity: int,
    *,
    device: torch.device,
) -> ResidentSparseWorkspace:
    """Allocate every resident-planning temporary at a stable address."""
    shape = (max_requests, scratch_capacity)
    sentinel_stride = (
        (scratch_capacity + 1 + 15) // 16
    ) * 16
    sentinel_shape = (max_requests, sentinel_stride)
    return ResidentSparseWorkspace(
        union_tokens=torch.empty(shape, dtype=torch.int32, device=device),
        valid_union=torch.empty(shape, dtype=torch.bool, device=device),
        old_slots_i16=torch.empty(shape, dtype=torch.int16, device=device),
        old_slots=torch.empty(shape, dtype=torch.int32, device=device),
        candidate_tokens=torch.empty(shape, dtype=torch.int32, device=device),
        hit_mask=torch.empty(shape, dtype=torch.bool, device=device),
        protected=torch.empty(
            sentinel_shape, dtype=torch.int32, device=device
        ),
        available_mask=torch.empty(shape, dtype=torch.bool, device=device),
        available_prefix=torch.empty(shape, dtype=torch.int32, device=device),
        available_by_rank=torch.empty(
            sentinel_shape, dtype=torch.int32, device=device
        ),
        miss_mask=torch.empty(shape, dtype=torch.bool, device=device),
        miss_prefix=torch.empty(shape, dtype=torch.int32, device=device),
        union_to_slot=torch.empty(shape, dtype=torch.int32, device=device),
        miss_payload=torch.empty(
            sentinel_shape, dtype=torch.int32, device=device
        ),
        miss_slot_payload=torch.empty(
            sentinel_shape, dtype=torch.int32, device=device
        ),
        state_token_indices=torch.empty(
            shape, dtype=torch.long, device=device
        ),
        state_slot_indices=torch.empty(
            shape, dtype=torch.long, device=device
        ),
        gather_indices=torch.empty(shape, dtype=torch.long, device=device),
        safe_states=torch.empty(
            max_requests, dtype=torch.long, device=device
        ),
        current_generations=torch.empty(
            max_requests, dtype=torch.long, device=device
        ),
        generation_matches=torch.empty(
            max_requests, dtype=torch.bool, device=device
        ),
        valid_states=torch.empty(
            max_requests, dtype=torch.bool, device=device
        ),
        slot_ids=torch.arange(
            scratch_capacity, dtype=torch.int32, device=device
        ).expand(max_requests, -1),
        row_ids=torch.arange(
            max_requests, dtype=torch.long, device=device
        ).reshape(-1, 1),
        int_sources=torch.empty(shape, dtype=torch.int32, device=device),
        short_sources=torch.empty(shape, dtype=torch.int16, device=device),
    )


def _flat_gather(
    source: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    torch.gather(
        source.reshape(-1),
        0,
        indices.reshape(-1),
        out=out.reshape(-1),
    )


def prepare_resident_sparse_cache_(
    topk_indices: torch.Tensor,
    position_to_union: torch.Tensor,
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slot_mapping: torch.Tensor,
    request_block_table: torch.Tensor,
    request_state_indices: torch.Tensor,
    request_state_generations: torch.Tensor,
    token_to_slot: torch.Tensor,
    slot_to_token: torch.Tensor,
    state_generations: torch.Tensor,
    workspace: ResidentSparseWorkspace,
    *,
    block_size: int,
    scratch_capacity: int,
    parallel_map: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a request union into a persistent scratch-cache miss plan.

    This function is intentionally fixed-shape and mutates only caller-owned
    tensors.  Every operation is an NPU-native tensor operation, so the path is
    graph-capturable without a host-side token loop. ``selected_packed`` and
    ``selected_counts`` enter with the union and leave with only cache misses.
    """
    validate_resident_shapes(
        token_to_slot, slot_to_token, scratch_capacity
    )
    request_count = int(selected_packed.shape[0])
    if request_count == 0:
        return (
            topk_indices,
            selected_packed,
            selected_counts,
            target_slot_mapping,
        )
    if int(selected_packed.shape[1]) != scratch_capacity:
        raise ValueError(
            "resident selected width must equal scratch capacity: "
            f"{selected_packed.shape[1]} != {scratch_capacity}"
        )
    if position_to_union.numel() != request_count * scratch_capacity:
        raise ValueError(
            "resident position-to-union mapping must have one entry per "
            "request scratch position"
        )
    if topk_indices.numel() != request_count * scratch_capacity:
        raise ValueError(
            "resident top-k must contain MTP * index_topk positions per "
            "request"
        )

    union_tokens = workspace.union_tokens[:request_count]
    valid_union = workspace.valid_union[:request_count]
    old_slots_i16 = workspace.old_slots_i16[:request_count]
    old_slots = workspace.old_slots[:request_count]
    candidate_tokens = workspace.candidate_tokens[:request_count]
    hit_mask = workspace.hit_mask[:request_count]
    protected = workspace.protected[:request_count]
    available_mask = workspace.available_mask[:request_count]
    available_prefix = workspace.available_prefix[:request_count]
    available_by_rank = workspace.available_by_rank[:request_count]
    miss_mask = workspace.miss_mask[:request_count]
    miss_prefix = workspace.miss_prefix[:request_count]
    union_to_slot = workspace.union_to_slot[:request_count]
    miss_payload = workspace.miss_payload[:request_count]
    miss_slot_payload = workspace.miss_slot_payload[:request_count]
    state_token_indices = workspace.state_token_indices[:request_count]
    state_slot_indices = workspace.state_slot_indices[:request_count]
    gather_indices = workspace.gather_indices[:request_count]
    safe_states = workspace.safe_states[:request_count]
    current_generations = workspace.current_generations[:request_count]
    generation_matches = workspace.generation_matches[:request_count]
    valid_states = workspace.valid_states[:request_count]
    slot_ids = workspace.slot_ids[:request_count]
    row_ids = workspace.row_ids[:request_count]
    int_sources = workspace.int_sources[:request_count]
    short_sources = workspace.short_sources[:request_count]

    counts = (
        selected_counts[:, 0]
        if selected_counts.dim() == 2
        else selected_counts
    )
    token_stride = int(token_to_slot.shape[1])
    slot_stride = int(slot_to_token.shape[1])
    dummy_base = int(token_to_slot.shape[0]) - int(
        workspace.slot_ids.shape[0]
    )
    if dummy_base < request_count:
        raise ValueError(
            "resident persistent maps need one cacheline-private dummy row "
            "per maximum batch request"
        )
    if slot_to_token.shape[0] != token_to_slot.shape[0]:
        raise ValueError("resident persistent maps have different state rows")
    if (
        state_generations.dim() != 2
        or state_generations.shape[0] != token_to_slot.shape[0]
        or state_generations.shape[1] < 8
    ):
        raise ValueError("resident generation table has different state rows")

    union_tokens.copy_(selected_packed)
    torch.lt(slot_ids, counts.reshape(-1, 1), out=valid_union)

    safe_states.copy_(request_state_indices)
    current_generations.copy_(row_ids.reshape(-1))
    current_generations.add_(dummy_base)
    torch.ge(request_state_indices, 0, out=valid_states)
    torch.where(
        valid_states,
        safe_states,
        current_generations,
        out=safe_states,
    )
    gather_indices.reshape(-1)[:request_count].copy_(safe_states)
    gather_indices.reshape(-1)[:request_count].mul_(
        int(state_generations.shape[1])
    )
    torch.gather(
        state_generations.reshape(-1),
        0,
        gather_indices.reshape(-1)[:request_count],
        out=current_generations,
    )
    torch.eq(
        current_generations,
        request_state_generations,
        out=generation_matches,
    )
    generation_matches.logical_and_(valid_states)

    # Materialize and logically clear every slot row whose owner generation
    # changed. This prevents an unoverwritten stale slot from becoming a hit
    # on a later step of the new request.
    state_slot_indices.copy_(safe_states.reshape(-1, 1))
    state_slot_indices.mul_(slot_stride)
    state_slot_indices.add_(slot_ids)
    _flat_gather(slot_to_token, state_slot_indices, candidate_tokens)
    int_sources.fill_(-1)
    torch.where(
        generation_matches.reshape(-1, 1),
        candidate_tokens,
        int_sources,
        out=int_sources,
    )
    slot_to_token.reshape(-1).scatter_(
        0,
        state_slot_indices.reshape(-1),
        int_sources.reshape(-1),
    )

    # Reverse lookup token -> candidate slot. Invalid fixed-shape positions
    # use token zero only for the gather; valid_union rejects them below.
    gather_indices.copy_(union_tokens)
    torch.logical_not(valid_union, out=miss_mask)
    gather_indices.masked_fill_(miss_mask, 0)
    state_token_indices.copy_(safe_states.reshape(-1, 1))
    state_token_indices.mul_(token_stride)
    state_token_indices.add_(gather_indices)
    _flat_gather(token_to_slot, state_token_indices, old_slots_i16)
    old_slots.copy_(old_slots_i16)

    # Validate the reverse map through the forward map. This makes stale
    # int16 reverse entries harmless and avoids clearing the 130K-token row.
    gather_indices.copy_(old_slots)
    gather_indices.clamp_(min=0, max=scratch_capacity - 1)
    state_slot_indices.copy_(safe_states.reshape(-1, 1))
    state_slot_indices.mul_(slot_stride)
    state_slot_indices.add_(gather_indices)
    _flat_gather(slot_to_token, state_slot_indices, candidate_tokens)
    torch.eq(candidate_tokens, union_tokens, out=hit_mask)
    hit_mask.logical_and_(valid_union)
    torch.ge(old_slots, 0, out=miss_mask)
    hit_mask.logical_and_(miss_mask)
    torch.lt(old_slots, scratch_capacity, out=miss_mask)
    hit_mask.logical_and_(miss_mask)
    hit_mask.logical_and_(generation_matches.reshape(-1, 1))

    # protected[request, slot] is private to one request row; all invalid
    # entries scatter to that same row's sentinel at index capacity.
    protected.zero_()
    gather_indices.copy_(old_slots)
    torch.logical_not(hit_mask, out=available_mask)
    gather_indices.masked_fill_(available_mask, scratch_capacity)
    int_sources.copy_(hit_mask)
    protected.scatter_add_(1, gather_indices, int_sources)
    torch.eq(protected[:, :scratch_capacity], 0, out=available_mask)
    torch.cumsum(
        available_mask,
        dim=1,
        dtype=torch.int32,
        out=available_prefix,
    )

    # Invert the ascending free-slot prefix without a scalar/token loop.
    available_by_rank.fill_(scratch_capacity)
    gather_indices.copy_(available_prefix)
    gather_indices.sub_(1)
    torch.logical_not(available_mask, out=miss_mask)
    gather_indices.masked_fill_(miss_mask, scratch_capacity)
    available_by_rank.scatter_(1, gather_indices, slot_ids)

    torch.logical_not(hit_mask, out=miss_mask)
    miss_mask.logical_and_(valid_union)
    torch.cumsum(
        miss_mask,
        dim=1,
        dtype=torch.int32,
        out=miss_prefix,
    )
    gather_indices.copy_(miss_prefix)
    gather_indices.sub_(1)
    torch.logical_not(miss_mask, out=available_mask)
    gather_indices.masked_fill_(available_mask, scratch_capacity)
    state_slot_indices.copy_(row_ids)
    state_slot_indices.mul_(int(available_by_rank.shape[1]))
    state_slot_indices.add_(gather_indices)
    _flat_gather(
        available_by_rank,
        state_slot_indices,
        int_sources,
    )
    torch.where(hit_mask, old_slots, int_sources, out=union_to_slot)
    torch.logical_not(valid_union, out=available_mask)
    union_to_slot.masked_fill_(available_mask, -1)

    # Compact misses and their assigned slots in union order.
    miss_payload.zero_()
    miss_slot_payload.zero_()
    miss_payload.scatter_(1, gather_indices, union_tokens)
    miss_slot_payload.scatter_(1, gather_indices, int_sources)
    selected_packed.copy_(miss_payload[:, :scratch_capacity])
    counts.copy_(miss_prefix[:, -1])

    # Invalidate overwritten reverse entries, then publish both directions.
    gather_indices.copy_(int_sources)
    gather_indices.clamp_(min=0, max=scratch_capacity - 1)
    state_slot_indices.copy_(safe_states.reshape(-1, 1))
    state_slot_indices.mul_(slot_stride)
    state_slot_indices.add_(gather_indices)
    _flat_gather(slot_to_token, state_slot_indices, candidate_tokens)

    state_token_indices.copy_(safe_states.reshape(-1, 1))
    state_token_indices.mul_(token_stride)
    gather_indices.copy_(candidate_tokens)
    gather_indices.clamp_(min=0, max=token_stride - 1)
    state_token_indices.add_(gather_indices)
    # Non-miss and empty old slots target this request's token sentinel.
    gather_indices.copy_(safe_states.reshape(-1, 1))
    gather_indices.mul_(token_stride)
    gather_indices.add_(token_stride - 1)
    torch.ge(candidate_tokens, 0, out=hit_mask)
    hit_mask.logical_and_(miss_mask)
    torch.where(
        hit_mask,
        state_token_indices,
        gather_indices,
        out=state_token_indices,
    )
    short_sources.fill_(INVALID_SCRATCH_SLOT)
    token_to_slot.reshape(-1).scatter_(
        0,
        state_token_indices.reshape(-1),
        short_sources.reshape(-1),
    )

    # token -> slot
    state_token_indices.copy_(safe_states.reshape(-1, 1))
    state_token_indices.mul_(token_stride)
    gather_indices.copy_(union_tokens)
    torch.logical_not(valid_union, out=available_mask)
    gather_indices.masked_fill_(available_mask, token_stride - 1)
    state_token_indices.add_(gather_indices)
    # union_to_slot was already set to -1 at every invalid union position.
    # Copying it performs the checked int32 -> int16 conversion without an
    # int16 masked_fill_, which aclnn does not support on Ascend.
    short_sources.copy_(union_to_slot)
    token_to_slot.reshape(-1).scatter_(
        0,
        state_token_indices.reshape(-1),
        short_sources.reshape(-1),
    )

    # slot -> token
    state_slot_indices.copy_(safe_states.reshape(-1, 1))
    state_slot_indices.mul_(slot_stride)
    gather_indices.copy_(union_to_slot)
    gather_indices.masked_fill_(available_mask, scratch_capacity)
    state_slot_indices.add_(gather_indices)
    int_sources.copy_(union_tokens)
    int_sources.masked_fill_(available_mask, -1)
    slot_to_token.reshape(-1).scatter_(
        0,
        state_slot_indices.reshape(-1),
        int_sources.reshape(-1),
    )
    gather_indices.reshape(-1)[:request_count].copy_(safe_states)
    gather_indices.reshape(-1)[:request_count].mul_(
        int(state_generations.shape[1])
    )
    state_generations.reshape(-1).scatter_(
        0,
        gather_indices.reshape(-1)[:request_count],
        request_state_generations,
    )

    # Build physical LMCache targets for compacted misses.
    gather_indices.copy_(miss_slot_payload[:, :scratch_capacity])
    int_sources.copy_(gather_indices)
    gather_indices.div_(block_size, rounding_mode="floor")
    state_slot_indices.copy_(row_ids)
    state_slot_indices.mul_(int(request_block_table.shape[1]))
    state_slot_indices.add_(gather_indices)
    _flat_gather(request_block_table, state_slot_indices, candidate_tokens)
    candidate_tokens.mul_(block_size)
    int_sources.remainder_(block_size)
    candidate_tokens.add_(int_sources)
    target_slot_mapping.copy_(candidate_tokens)

    # Map every selected original top-k position through union rank -> actual
    # persistent scratch slot. Live-cache positions retain absolute indices.
    if parallel_map:
        try:
            remap_op = (
                torch.ops._C_ascend.npu_dsa_resident_remap_rows_
            )
        except AttributeError as exc:
            raise RuntimeError(
                "vllm_ascend_C does not expose "
                "npu_dsa_resident_remap_rows_; rebuild the custom-op "
                "extension"
            ) from exc
        remap_op(topk_indices, position_to_union, union_to_slot)
    else:
        topk_2d = topk_indices.reshape(request_count, scratch_capacity)
        mapping = position_to_union.reshape(
            request_count, scratch_capacity
        )
        torch.ge(mapping, 0, out=valid_union)
        gather_indices.copy_(mapping)
        gather_indices.clamp_min_(0)
        torch.gather(
            union_to_slot,
            1,
            gather_indices,
            out=int_sources,
        )
        torch.where(valid_union, int_sources, topk_2d, out=topk_2d)
    return topk_indices, selected_packed, counts, target_slot_mapping


def _resident_sparse_cache_reference(
    union_tokens: torch.Tensor,
    union_counts: torch.Tensor,
    request_state_indices: torch.Tensor,
    token_to_slot: torch.Tensor,
    slot_to_token: torch.Tensor,
    *,
    scratch_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small CPU oracle; mutates the two persistent maps like production."""
    validate_resident_shapes(
        token_to_slot, slot_to_token, scratch_capacity
    )
    request_count = int(union_tokens.shape[0])
    union_to_slot = torch.full_like(union_tokens, -1, dtype=torch.int32)
    misses = torch.zeros_like(union_tokens, dtype=torch.int32)
    miss_counts = torch.zeros(request_count, dtype=torch.int32)

    for request in range(request_count):
        state = int(request_state_indices[request])
        count = int(union_counts[request])
        if state < 0:
            # Graph-capture/dummy cold path: produce a valid transient plan but
            # never touch persistent request state.
            union_to_slot[request, :count] = torch.arange(
                count, dtype=torch.int32
            )
            misses[request, :count] = union_tokens[request, :count]
            miss_counts[request] = count
            continue

        protected: set[int] = set()
        missing_positions: list[int] = []
        for pos in range(count):
            token = int(union_tokens[request, pos])
            slot = int(token_to_slot[state, token])
            if (
                0 <= slot < scratch_capacity
                and int(slot_to_token[state, slot]) == token
            ):
                union_to_slot[request, pos] = slot
                protected.add(slot)
            else:
                missing_positions.append(pos)

        available = [
            slot for slot in range(scratch_capacity) if slot not in protected
        ]
        for miss_rank, pos in enumerate(missing_positions):
            token = int(union_tokens[request, pos])
            slot = available[miss_rank]
            old_token = int(slot_to_token[state, slot])
            if old_token >= 0:
                token_to_slot[state, old_token] = INVALID_SCRATCH_SLOT
            slot_to_token[state, slot] = token
            token_to_slot[state, token] = slot
            union_to_slot[request, pos] = slot
            misses[request, miss_rank] = token
        miss_counts[request] = len(missing_positions)

    return union_to_slot, misses, miss_counts


def remap_union_positions_(
    topk_indices: torch.Tensor,
    position_to_union: torch.Tensor,
    union_to_slot: torch.Tensor,
) -> torch.Tensor:
    """Map selected original positions to persistent scratch slots in-place."""
    topk_2d = topk_indices.reshape(position_to_union.shape)
    valid = position_to_union >= 0
    ranks = position_to_union.clamp_min(0).to(torch.long)
    mapped = torch.gather(union_to_slot, 1, ranks)
    topk_2d.copy_(torch.where(valid, mapped, topk_2d))
    return topk_indices
