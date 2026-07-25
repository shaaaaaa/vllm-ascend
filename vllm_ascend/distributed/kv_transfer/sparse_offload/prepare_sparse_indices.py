"""Device-only sparse-index preparation for DSA latent scratch (Step B2).

Decode reads the latent through two disjoint index spaces resolved by the SAME
per-request block table:

  * LMCache-selected positions (< cache boundary) -> request-level bitmap union
    shared by all MTP rows for that request. The baseline operator compacts
    this union into [0..n_unique); the reuse operator preserves hits in their
    existing physical scratch slots and fills misses into evictable slots;
  * live-cache positions (>= cache boundary) -> kept ABSOLUTE, read in
    place from their tail blocks. No copy, no [retrieve|decode] assembly.

A zero boundary selects nothing from LMCache and leaves every index absolute.

Everything is fixed-shape tensor math: no D2H sync, graph-mode friendly.
"""

import torch


def _prepare_sparse_indices_torch(
    topk_indices: torch.Tensor,
    split_boundary: torch.Tensor,
    row_req_indices: torch.Tensor | None = None,
    request_block_table: torch.Tensor | None = None,
    block_size: int = 1,
    need_packed: bool = True,
    clear_invalid_rows: bool = False,
    scratch_base: torch.Tensor | None = None,
    valid_row_indices: torch.Tensor | None = None,
):
    """Request-level sorted bitmap-union reference used as a test oracle."""
    orig_shape = topk_indices.shape
    sel = topk_indices.reshape(orig_shape[0], -1)
    if request_block_table is None:
        boundary = split_boundary.reshape(-1, 1).to(sel)
        base = (
            torch.zeros((sel.shape[0], 1), dtype=sel.dtype, device=sel.device)
            if scratch_base is None
            else scratch_base.reshape(-1, 1).to(sel)
        )
        selected = (sel >= 0) & (sel < boundary)
        rank = torch.cumsum(selected, dim=1, dtype=sel.dtype) - 1
        remapped = torch.where(selected, base + rank, sel)
        if row_req_indices is not None:
            remapped[row_req_indices[: sel.shape[0]] < 0] = 0
        if not need_packed:
            return remapped.reshape(orig_shape), None
        packed = sel.new_zeros((sel.shape[0], sel.shape[1] + 1))
        dst = torch.where(selected, rank, torch.full_like(rank, sel.shape[1]))
        packed.scatter_(1, dst.long(), sel)
        packed = packed[:, : sel.shape[1]]
        if valid_row_indices is not None:
            packed = packed.index_select(0, valid_row_indices.long())
        return remapped.reshape(orig_shape), packed

    assert row_req_indices is not None
    request_count = int(request_block_table.shape[0])
    capacity = sel.shape[1] * max(
        1,
        max(
            (
                int((row_req_indices == req).sum())
                for req in range(request_count)
            ),
            default=1,
        ),
    )
    packed = sel.new_zeros((request_count, capacity))
    counts = torch.zeros(request_count, dtype=torch.int32, device=sel.device)
    targets = torch.zeros(
        (request_count, capacity), dtype=torch.long, device=sel.device
    )
    new_indices = sel.clone()
    for req in range(request_count):
        selected_tokens = sorted(
            {
                int(sel[row, col])
                for row in range(sel.shape[0])
                if int(row_req_indices[row]) == req
                for col in range(sel.shape[1])
                if 0 <= int(sel[row, col]) < int(split_boundary[row])
            }
        )
        inverse = {token: slot for slot, token in enumerate(selected_tokens)}
        for token, slot in inverse.items():
            packed[req, slot] = token
            block_id = int(request_block_table[req, slot // block_size])
            targets[req, slot] = block_id * block_size + slot % block_size
        for row in range(sel.shape[0]):
            if int(row_req_indices[row]) != req:
                continue
            boundary = int(split_boundary[row])
            for col in range(sel.shape[1]):
                token = int(sel[row, col])
                if 0 <= token < boundary:
                    new_indices[row, col] = inverse[token]
        counts[req] = len(inverse)
    if clear_invalid_rows:
        new_indices[row_req_indices[: sel.shape[0]] < 0] = 0
    return (
        new_indices.reshape(orig_shape),
        packed if need_packed else None,
        counts if need_packed else None,
        targets if need_packed else None,
    )


def _prepare_sparse_indices_reuse_torch(
    topk_indices: torch.Tensor,
    split_boundary: torch.Tensor,
    row_req_indices: torch.Tensor,
    request_block_table: torch.Tensor,
    request_state_indices: torch.Tensor,
    request_generations: torch.Tensor,
    resident_token_ids: torch.Tensor,
    resident_generations: torch.Tensor,
    block_size: int,
    clear_invalid_rows: bool = False,
):
    """Reference for the stateful scratch-reuse custom operator.

    ``resident_token_ids[state_row, scratch_slot]`` records the absolute
    sequence position whose KV currently occupies that physical scratch slot.
    The state row is stable across compact-batch reorderings and is selected by
    ``request_state_indices``. A generation mismatch invalidates the complete
    state row before it is used. If every row for a request has a zero
    boundary, state is not used and generation validation is deferred.

    Only misses are emitted in ``selected_packed``. Their corresponding
    ``target_slot_mapping`` entries may therefore be non-contiguous even
    though the payload itself is compact.
    """
    if topk_indices.dtype != torch.int32:
        raise ValueError("topk_indices must be int32")
    if row_req_indices.dtype != torch.int32:
        raise ValueError("row_req_indices must be int32")
    if request_state_indices.dtype != torch.int32:
        raise ValueError("request_state_indices must be int32")
    if request_generations.dtype != torch.int64:
        raise ValueError("request_generations must be int64")
    if resident_token_ids.dtype != torch.int32:
        raise ValueError("resident_token_ids must be int32")
    if resident_generations.dtype != torch.int64:
        raise ValueError("resident_generations must be int64")
    if resident_token_ids.dim() != 2:
        raise ValueError("resident_token_ids must be two-dimensional")
    if resident_generations.dim() != 2 or resident_generations.shape[1] < 8:
        raise ValueError(
            "resident_generations must be 2D with a padded stride of at least 8"
        )

    orig_shape = topk_indices.shape
    selected = topk_indices.reshape(orig_shape[0], -1)
    remapped = selected.clone()
    request_count = int(request_block_table.shape[0])
    scratch_capacity = int(resident_token_ids.shape[1])
    packed = torch.zeros(
        (request_count, scratch_capacity),
        dtype=torch.int32,
        device=selected.device,
    )
    counts = torch.zeros(
        request_count, dtype=torch.int32, device=selected.device
    )
    targets = torch.zeros(
        (request_count, scratch_capacity),
        dtype=torch.int64,
        device=selected.device,
    )

    if request_state_indices.numel() < request_count:
        raise ValueError("request_state_indices must cover every request")
    if request_generations.numel() < request_count:
        raise ValueError("request_generations must cover every request")
    if request_block_table.shape[1] * block_size < scratch_capacity:
        raise ValueError("request block table cannot address every scratch slot")
    token_capacity = int(request_block_table.shape[1]) * int(block_size)

    for req in range(request_count):
        generation = int(request_generations[req])
        if generation <= 0:
            # Graph-padding requests do not own stable state and must not read
            # or mutate a resident row.
            counts[req] = 0
            continue
        request_rows = [
            row
            for row in range(selected.shape[0])
            if int(row_req_indices[row]) == req
        ]
        if not any(int(split_boundary[row]) > 0 for row in request_rows):
            # State is irrelevant while every row reads only live NPU cache.
            # Defer generation validation/reset until scratch is first used.
            counts[req] = 0
            continue
        state_row = int(request_state_indices[req])
        if not 0 <= state_row < resident_token_ids.shape[0]:
            raise ValueError(
                f"request {req} has invalid stable state row {state_row}"
            )
        if int(resident_generations[state_row, 0]) != generation:
            resident_token_ids[state_row].fill_(-1)
            resident_generations[state_row, 0] = generation

        desired_tokens: list[int] = []
        desired_set: set[int] = set()
        for row in request_rows:
            boundary = int(split_boundary[row])
            for col in range(selected.shape[1]):
                token = int(selected[row, col])
                if 0 <= token < boundary:
                    if token >= token_capacity:
                        raise ValueError(
                            f"request {req} selected LMCache token {token}, "
                            f"but its block-table capacity is {token_capacity}"
                        )
                    if token in desired_set:
                        continue
                    desired_set.add(token)
                    desired_tokens.append(token)
        if len(desired_tokens) > scratch_capacity:
            raise ValueError(
                f"request {req} needs {len(desired_tokens)} scratch slots, "
                f"but capacity is {scratch_capacity}"
            )

        token_to_slot: dict[int, int] = {}
        for slot, token_tensor in enumerate(resident_token_ids[state_row]):
            token = int(token_tensor)
            if token in desired_set:
                # The state invariant is one slot per token. Using the last
                # slot matches the device kernel defensively if corrupt state
                # contains a duplicate.
                token_to_slot[token] = slot
        retained_slots = set(token_to_slot.values())
        free_slots = (
            slot
            for slot in range(scratch_capacity)
            if slot not in retained_slots
        )

        miss_count = 0
        for token in desired_tokens:
            if token in token_to_slot:
                continue
            try:
                slot = next(free_slots)
            except StopIteration as exc:
                raise ValueError(
                    f"request {req} has no evictable scratch slot"
                ) from exc
            token_to_slot[token] = slot
            resident_token_ids[state_row, slot] = token
            packed[req, miss_count] = token
            physical_block = int(
                request_block_table[req, slot // block_size]
            )
            if physical_block <= 0:
                raise ValueError(
                    f"request {req} scratch slot {slot} maps to invalid "
                    f"physical block {physical_block}; block 0 is reserved "
                    "as vLLM's null block"
                )
            targets[req, miss_count] = (
                physical_block * block_size + slot % block_size
            )
            miss_count += 1

        for row in request_rows:
            boundary = int(split_boundary[row])
            for col in range(selected.shape[1]):
                token = int(selected[row, col])
                if 0 <= token < boundary:
                    remapped[row, col] = token_to_slot[token]
        counts[req] = miss_count

    if clear_invalid_rows:
        remapped[row_req_indices[: selected.shape[0]] < 0] = 0
    return remapped.reshape(orig_shape), packed, counts, targets


def prepare_sparse_indices(
    topk_indices: torch.Tensor,
    split_boundary: torch.Tensor,
    row_req_indices: torch.Tensor,
    request_block_table: torch.Tensor,
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slot_mapping: torch.Tensor,
    block_size: int,
    need_packed: bool = True,
    clear_invalid_rows: bool = False,
    request_state_indices: torch.Tensor | None = None,
    request_generations: torch.Tensor | None = None,
    resident_token_ids: torch.Tensor | None = None,
    resident_generations: torch.Tensor | None = None,
):
    """Remap absolute top-k indices for the compact-scratch decode path.

    Args:
        topk_indices: [bs, 1, k] (or [bs, k]) absolute token positions selected
            by the indexer; negative entries are padding.
        split_boundary: [bs] cache split boundary per decode request. Zero
            means the whole prefix is resident in NPU cache. A positive value
            is the LMCache-committed frontier; selected positions below it are
            remapped through the request-level union scratch prefix.
        need_packed: whether to build the LMCache selected-token payload.
        row_req_indices: [bs] request index for each row; negative entries are
            zeroed in the same kernel. Pass this only for pure
            decode/spec-decode; a mixed prefill row also has a negative request
            index but is real.
        request_state_indices: optional [num_requests] mapping from compact
            request rows to stable resident-state rows. Positive-generation
            requests must map to distinct rows because one AIV owns each
            request. Supplying it enables reuse and requires all three other
            state tensors.
        request_generations: optional positive int64 lifetime generation per
            compact request. Non-positive values denote graph padding and
            never read or mutate resident state.
        resident_token_ids: optional per-layer
            [max_state_rows, scratch_capacity] int32 slot ownership.
        resident_generations: optional
            [max_state_rows, padded_stride] int64 lifetime ownership.

    Returns:
        new_indices: same shape as topk_indices. LMCache-selected entries are
            replaced by a scratch slot. Baseline uses the request-union rank;
            reuse uses the retained or newly assigned physical slot.
            Live-cache and padding entries stay unchanged.
        selected_packed: [num_requests, scratch_capacity] int32. Baseline
            contains the complete union in ascending absolute-token order;
            reuse contains only misses in deterministic row-major first-seen
            order. None when need_packed=False.
    """
    if topk_indices.device.type != "npu":
        raise RuntimeError(
            "prepare_sparse_indices requires the NPU custom op; use "
            "_prepare_sparse_indices_torch only as a test reference"
        )
    reuse_state = (
        request_state_indices,
        request_generations,
        resident_token_ids,
        resident_generations,
    )
    reuse_enabled = any(tensor is not None for tensor in reuse_state)
    if reuse_enabled and not all(tensor is not None for tensor in reuse_state):
        raise ValueError(
            "scratch reuse requires request_state_indices, "
            "request_generations, resident_token_ids, and "
            "resident_generations together"
        )
    if reuse_enabled and not need_packed:
        raise ValueError(
            "scratch reuse requires need_packed=True so resident state is "
            "updated only when an LMCache payload is prepared"
        )

    op_name = (
        "npu_dsa_prepare_sparse_indices_reuse_"
        if reuse_enabled
        else "npu_dsa_prepare_sparse_indices_"
    )
    try:
        fused_op = getattr(torch.ops._C_ascend, op_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"vllm_ascend_C does not expose {op_name}; rebuild the "
            "custom-op extension"
        ) from exc

    if reuse_enabled:
        fused_op(
            topk_indices,
            split_boundary,
            row_req_indices,
            request_block_table,
            selected_packed,
            selected_counts,
            target_slot_mapping,
            request_state_indices,
            request_generations,
            resident_token_ids,
            resident_generations,
            block_size,
            need_packed,
            clear_invalid_rows,
        )
    else:
        fused_op(
            topk_indices,
            split_boundary,
            row_req_indices,
            request_block_table,
            selected_packed,
            selected_counts,
            target_slot_mapping,
            block_size,
            need_packed,
            clear_invalid_rows,
        )
    return (
        topk_indices,
        selected_packed if need_packed else None,
        selected_counts[:, 0] if need_packed else None,
        target_slot_mapping if need_packed else None,
    )
