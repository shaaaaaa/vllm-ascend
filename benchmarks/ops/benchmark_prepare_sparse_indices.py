import argparse
import statistics

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    _sparse_index_op_name,
    prepare_sparse_indices,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sorted_cache import (
    allocate_sorted_resident_state,
    allocate_sorted_resident_workspace,
    coordinate_sorted_resident_finalize_,
    debug_sorted_resident_finalize_only_,
    debug_sorted_resident_update_only_,
    prepare_resident_sharded_union_,
    prepare_sorted_resident_cache_coordinated_,
    prepare_sorted_resident_cache_fused_,
    run_sharded_resident_finalize_,
)
from vllm_ascend.utils import enable_custom_op


def _mtp_rows_with_half_overlap(topk: int, request_batch: int, mtp: int, device: str) -> torch.Tensor:
    if topk % 2:
        raise ValueError("topk must be even for an exact 0.5 row overlap")
    if mtp < 1:
        raise ValueError("the MTP benchmark requires at least one row")
    shared = torch.arange(topk // 2, dtype=torch.int32, device=device)
    request_rows = torch.stack(
        tuple(
            torch.cat(
                (
                    shared,
                    torch.arange(
                        topk // 2 + row * topk // 2,
                        topk // 2 + (row + 1) * topk // 2,
                        dtype=torch.int32,
                        device=device,
                    ),
                )
            )
            for row in range(mtp)
        )
    )
    return request_rows.repeat(request_batch, 1).unsqueeze(1)


def _measure_npu_ms(run, reset, warmups: int, iterations: int) -> list[float]:
    for _ in range(warmups):
        reset()
        run()
    torch.npu.synchronize()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        reset()
        start.record()
        run()
        end.record()
    torch.npu.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _seed_sorted_resident_state_for_hit_rate(
    state,
    *,
    request_batch: int,
    valid_token_count: int,
    capacity: int,
    shard_count: int,
    hit_rate: float,
) -> tuple[int, int]:
    """Fill every scratch slot while retaining an exact fraction of input."""
    hit_count = int(round(valid_token_count * hit_rate))
    miss_count = valid_token_count - hit_count
    stale_count = capacity - hit_count
    if not 0 < hit_count < valid_token_count:
        raise ValueError("resident stage benchmark requires partial hits")
    if stale_count < miss_count:
        raise ValueError("resident seed does not have enough evictable slots")

    token_seed = torch.full(tuple(state.tokens.shape), -1, dtype=torch.int32)
    slot_seed = torch.full(tuple(state.slots.shape), -1, dtype=torch.int16)
    count_seed = torch.zeros(tuple(state.counts.shape), dtype=torch.int32)
    generation_seed = torch.full(tuple(state.generations.shape), -1, dtype=torch.int64)
    resident_tokens = list(range(hit_count)) + list(range(65_536, 65_536 + stale_count))
    shard_entries = [[] for _ in range(shard_count)]
    for slot, token in enumerate(resident_tokens):
        shard_entries[token % shard_count].append((token, slot))
    for shard in range(shard_count):
        shard_entries[shard].sort()
    for request in range(request_batch):
        generation_seed[request, 0] = 1
        for shard, entries in enumerate(shard_entries):
            count = len(entries)
            count_seed[request, shard, 0] = count
            token_seed[request, shard, :count] = torch.tensor([token for token, _ in entries], dtype=torch.int32)
            slot_seed[request, shard, :count] = torch.tensor([slot for _, slot in entries], dtype=torch.int16)
    state.tokens.copy_(token_seed.to(state.tokens.device))
    state.slots.copy_(slot_seed.to(state.slots.device))
    state.counts.copy_(count_seed.to(state.counts.device))
    state.generations.copy_(generation_seed.to(state.generations.device))
    return hit_count, miss_count


def _validate_resident_sharded_union(
    source: torch.Tensor,
    boundary: int,
    shard_packed: torch.Tensor,
    shard_mapping: torch.Tensor,
    shard_counts: torch.Tensor,
    *,
    mtp: int,
) -> None:
    source_cpu = source.reshape(shard_packed.shape[0], mtp, -1).cpu()
    packed_cpu = shard_packed.cpu()
    mapping_cpu = shard_mapping.cpu()
    counts_cpu = shard_counts[:, :, 0].cpu()
    shard_count = shard_packed.shape[1]
    for request in range(shard_packed.shape[0]):
        for shard in range(shard_count):
            expected = sorted(
                {
                    int(token)
                    for token in source_cpu[request].reshape(-1).tolist()
                    if 0 <= token < boundary and token % shard_count == shard
                }
            )
            count = int(counts_cpu[request, shard])
            actual = packed_cpu[request, shard, :count].tolist()
            if actual != expected:
                raise AssertionError(f"resident shard {shard} payload differs: {actual[:8]} != {expected[:8]}")
        for position, token in enumerate(source_cpu[request].reshape(-1).tolist()):
            if not 0 <= token < boundary:
                continue
            shard = token % shard_count
            rank = int(mapping_cpu[request, shard, position])
            if int(packed_cpu[request, shard, rank]) != token:
                raise AssertionError("resident shard mapping does not point at its token")


def _validate_resident_remap(
    source: torch.Tensor,
    boundary: int,
    remapped: torch.Tensor,
    state,
    *,
    mtp: int,
) -> None:
    request_batch = int(source.shape[0]) // mtp
    source_cpu = source.reshape(request_batch, mtp, -1).cpu()
    remapped_cpu = remapped.reshape(request_batch, mtp, -1).cpu()
    tokens_cpu = state.tokens[:request_batch].cpu()
    slots_cpu = state.slots[:request_batch].cpu()
    counts_cpu = state.counts[:request_batch, :, 0].cpu()
    for request in range(request_batch):
        token_to_slot = {}
        for shard in range(tokens_cpu.shape[1]):
            count = int(counts_cpu[request, shard])
            token_to_slot.update(
                zip(
                    tokens_cpu[request, shard, :count].tolist(),
                    slots_cpu[request, shard, :count].tolist(),
                    strict=True,
                )
            )
        expected = source_cpu[request].clone()
        for row in range(mtp):
            for position, token in enumerate(source_cpu[request, row].tolist()):
                if 0 <= token < boundary:
                    expected[row, position] = token_to_slot[token]
        if not torch.equal(remapped_cpu[request], expected):
            raise AssertionError(f"resident fused remap differs for request {request}")


def _summary(name: str, samples: list[float]) -> None:
    ordered = sorted(samples)
    p50 = ordered[len(ordered) // 2]
    p90 = ordered[int((len(ordered) - 1) * 0.9)]
    print(f"{name:>12}: mean={statistics.fmean(samples):.6f} ms p50={p50:.6f} ms p90={p90:.6f} ms")


def _staged_runner(
    *,
    legacy_op,
    union_op,
    remap_op,
    values,
    boundaries,
    valid_rows,
    local_scratch_base,
    row_requests,
    request_block_table,
    selected_packed,
    local_to_union,
    selected_count,
    target_slots,
    block_size,
    max_tokens,
    use_sort,
):
    row_packed = legacy_op(
        values,
        boundaries,
        valid_rows,
        local_scratch_base,
        True,
        row_requests,
    )
    union_op(
        row_packed,
        selected_packed,
        local_to_union,
        selected_count,
        request_block_table,
        target_slots,
        block_size,
        max_tokens,
        use_sort,
    )
    remap_op(values, local_to_union)
    return values, selected_packed, selected_count, target_slots


def _staged_no_union_runner(
    *,
    legacy_op,
    copy_rows_op,
    values,
    local_indices,
    boundaries,
    valid_rows,
    local_scratch_base,
    row_requests,
):
    legacy_op(
        local_indices,
        boundaries,
        valid_rows,
        local_scratch_base,
        True,
        row_requests,
    )
    copy_rows_op(values, local_indices)
    return values


def _staged_native_unique_runner(
    *,
    legacy_op,
    finalize_op,
    remap_op,
    values,
    boundaries,
    valid_rows,
    local_scratch_base,
    row_requests,
    request_block_table,
    selected_packed,
    local_to_union,
    selected_count,
    target_slots,
    block_size,
    packed_key_stride,
):
    packed_keys = legacy_op(
        values,
        boundaries,
        valid_rows,
        local_scratch_base,
        True,
        row_requests,
        packed_key_stride,
    )
    unique_keys, inverse = torch.unique(
        packed_keys.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    finalize_op(
        unique_keys,
        inverse,
        row_requests,
        selected_packed,
        local_to_union,
        selected_count,
        request_block_table,
        target_slots,
        block_size,
        packed_key_stride,
    )
    remap_op(values, local_to_union)
    return values, selected_packed, selected_count, target_slots


def _staged_sharded_sort_runner(
    *,
    sharded_union_op,
    values,
    boundaries,
    request_block_table,
    selected_packed,
    local_to_union,
    selected_count,
    target_slots,
    shard_packed,
    shard_mapping,
    shard_counts,
    block_size,
):
    sharded_union_op(
        values,
        boundaries,
        selected_packed,
        local_to_union,
        selected_count,
        request_block_table,
        target_slots,
        shard_packed,
        shard_mapping,
        shard_counts,
        block_size,
    )
    return values, selected_packed, selected_count, target_slots


def _staged_sharded_vector_runner(
    *,
    sharded_union_op,
    values,
    boundaries,
    request_block_table,
    selected_packed,
    local_to_union,
    selected_count,
    target_slots,
    shard_packed,
    shard_mapping,
    shard_counts,
    shard_pairs,
    block_size,
):
    sharded_union_op(
        values,
        boundaries,
        selected_packed,
        local_to_union,
        selected_count,
        request_block_table,
        target_slots,
        shard_packed,
        shard_mapping,
        shard_counts,
        shard_pairs,
        block_size,
    )
    return values, selected_packed, selected_count, target_slots


def _validate_sharded_result(
    *,
    label,
    result,
    source,
    boundary,
    request_batch,
    mtp,
    topk,
    max_tokens,
):
    if result[2][:, 0].cpu().tolist() != [boundary] * request_batch:
        raise AssertionError(f"{label} staged union count is incorrect")
    row_count = request_batch * mtp
    selected_mask = source.reshape(row_count, topk) < boundary
    remapped = result[0].reshape(row_count, topk)
    safe_indices = torch.where(selected_mask, remapped, torch.zeros_like(remapped))
    selected_reconstructed = torch.gather(
        result[1].repeat_interleave(mtp, dim=0),
        1,
        safe_indices.to(torch.long),
    )
    reconstructed = torch.where(
        selected_mask,
        selected_reconstructed,
        remapped,
    )
    if not torch.equal(
        reconstructed.cpu(),
        source.reshape(row_count, topk).cpu(),
    ):
        raise AssertionError(f"{label} remapped rows do not reconstruct source tokens")
    selected_cpu = result[1].cpu()
    targets_cpu = result[3].cpu()
    expected_token_set = set(range(boundary))
    for request in range(request_batch):
        selected_tokens = selected_cpu[request, :boundary].tolist()
        if set(selected_tokens) != expected_token_set:
            raise AssertionError(f"{label} output is not the expected deduplicated union")
        expected_targets = torch.arange(boundary, dtype=torch.long) + request * max_tokens
        if not torch.equal(
            targets_cpu[request, :boundary],
            expected_targets,
        ):
            raise AssertionError(f"{label} target slots are incorrect")


def production_only_main(
    topk: int = 2048,
    mtp: int = 2,
    iterations: int = 200,
    warmups: int = 20,
) -> None:
    """Benchmark only the selected production sparse-index operator."""
    if topk != 2048:
        raise ValueError("the production staged operator requires --topk 2048")
    if mtp not in (1, 2):
        raise ValueError("the production staged operator supports only --mtp 1 or 2")
    if not enable_custom_op():
        raise RuntimeError("vllm-ascend custom operators could not be loaded")
    request_batch = 4
    row_count = request_batch * mtp
    source = _mtp_rows_with_half_overlap(
        topk,
        request_batch,
        mtp,
        "npu",
    )
    values = source.clone()
    source_max = int(source.max().item())
    split_boundary = source_max - 100
    boundaries = torch.full(
        (row_count,),
        split_boundary,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.arange(
        request_batch,
        dtype=torch.int32,
        device="npu",
    ).repeat_interleave(mtp)

    block_size = 128
    max_tokens = 131072
    capacity = mtp * topk
    blocks_per_request = max_tokens // block_size
    block_table = torch.arange(
        request_batch * blocks_per_request,
        dtype=torch.int32,
        device="npu",
    ).reshape(request_batch, blocks_per_request)
    selected = torch.empty(
        (request_batch, capacity),
        dtype=torch.int32,
        device="npu",
    )
    counts = torch.empty(
        (request_batch, 16),
        dtype=torch.int32,
        device="npu",
    )
    targets = torch.empty(
        (request_batch, capacity),
        dtype=torch.long,
        device="npu",
    )
    local_to_union_workspace = torch.empty_like(selected)
    shard_packed_workspace = torch.empty(
        (request_batch, 2, capacity),
        dtype=torch.int32,
        device="npu",
    )
    shard_mapping_workspace = torch.empty_like(shard_packed_workspace)
    shard_counts_workspace = torch.empty(
        (request_batch, 2, 16),
        dtype=torch.int32,
        device="npu",
    )

    def run() -> None:
        prepare_sparse_indices(
            values,
            boundaries,
            row_req_indices=row_requests,
            request_block_table=block_table,
            selected_packed=selected,
            selected_counts=counts,
            target_slot_mapping=targets,
            block_size=block_size,
            need_packed=True,
            clear_invalid_rows=True,
            local_to_union_workspace=local_to_union_workspace,
            shard_packed_workspace=shard_packed_workspace,
            shard_mapping_workspace=shard_mapping_workspace,
            shard_counts_workspace=shard_counts_workspace,
            staged_mtp=mtp,
        )

    # Correctness is checked once outside the timed section.
    run()
    torch.npu.synchronize()
    expected_count = split_boundary
    expected_counts = [expected_count] * request_batch
    if counts[:, 0].cpu().tolist() != expected_counts:
        raise AssertionError(
            "production staged selected counts are incorrect: "
            f"actual={counts[:, 0].cpu().tolist()}, "
            f"expected={expected_counts}"
        )
    expected_selected = torch.arange(
        expected_count,
        dtype=torch.int32,
    )
    selected_cpu = selected.cpu()
    targets_cpu = targets.cpu()
    for request in range(request_batch):
        if not torch.equal(
            torch.sort(selected_cpu[request, :expected_count]).values,
            expected_selected,
        ):
            raise AssertionError(f"request {request} production staged union is incorrect")
        expected_targets = torch.arange(expected_count, dtype=torch.long) + request * max_tokens
        if not torch.equal(
            targets_cpu[request, :expected_count],
            expected_targets,
        ):
            raise AssertionError(f"request {request} production staged targets are incorrect")

    source_2d = source.reshape(row_count, topk)
    remapped = values.reshape(row_count, topk)
    selected_mask = (source_2d >= 0) & (source_2d < split_boundary)
    safe_ranks = torch.where(
        selected_mask,
        remapped,
        torch.zeros_like(remapped),
    )
    reconstructed_selected = torch.gather(
        selected.repeat_interleave(mtp, dim=0),
        1,
        safe_ranks.to(torch.long),
    )
    reconstructed = torch.where(
        selected_mask,
        reconstructed_selected,
        remapped,
    )
    if not torch.equal(reconstructed.cpu(), source_2d.cpu()):
        raise AssertionError("production staged remapped rows do not reconstruct the input")

    samples = _measure_npu_ms(
        run,
        lambda: values.copy_(source),
        warmups,
        iterations,
    )
    print(
        "production-only benchmark: "
        f"topk={topk}, MTP={mtp}, requests={request_batch}, "
        f"split_boundary={split_boundary} (source max={source_max})"
    )
    uses_sharded_operator = _sparse_index_op_name(mtp) == "npu_dsa_prepare_sparse_indices_sharded_"
    production_label = (
        "production-sharded-single-row"
        if uses_sharded_operator and mtp == 1
        else ("production-sharded-sort" if uses_sharded_operator else "production-staged-sort")
    )
    _summary(production_label, samples)


def resident_only_main(
    topk: int = 2048,
    mtp: int = 2,
    request_batch: int = 4,
    hit_rate: float = 0.9,
    parallel_map: bool = False,
    iterations: int = 200,
    warmups: int = 20,
) -> None:
    """Compatibility shim for the standalone resident benchmark."""
    if __package__:
        from .benchmark_resident_sparse_cache import (
            main as benchmark_resident_sparse_cache,
        )
    else:
        from benchmark_resident_sparse_cache import (
            main as benchmark_resident_sparse_cache,
        )

    _ = parallel_map
    benchmark_resident_sparse_cache(
        topk,
        mtp,
        request_batch,
        hit_rate,
        iterations,
        warmups,
    )


def main(
    topk: int = 2048,
    mtp: int = 2,
    iterations: int = 200,
    warmups: int = 20,
    resident_hit_rate: float = 0.9,
    shards_per_row: int | None = None,
) -> None:
    if topk != 2048:
        raise ValueError("the experimental staged kernels currently require --topk 2048")
    if mtp < 1 or mtp > 8:
        raise ValueError("the sharded benchmark supports --mtp from 1 to 8")
    if not enable_custom_op():
        raise RuntimeError("vllm-ascend custom operators could not be loaded")
    default_shard_count = 1 << (mtp - 1).bit_length()
    shard_count = (
        default_shard_count
        if shards_per_row is None
        else mtp * shards_per_row
    )
    if (
        shard_count < default_shard_count
        or shard_count > 8
        or shard_count & (shard_count - 1)
        or topk % shard_count
    ):
        raise ValueError(
            "--shards-per-row must produce a power-of-two total shard count "
            "between the default and 8"
        )
    print(
        f"MTP depth={mtp}, "
        f"legacy value shards={default_shard_count}, "
        f"benchmark shards/row={shards_per_row or 'default'}, "
        f"benchmark total shards={shard_count}, "
        f"approx elements/shard={mtp * topk // shard_count}, "
        f"resident shards={shard_count if mtp <= 2 else 'n/a'}"
    )
    legacy_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_legacy_
    union_op = torch.ops._C_ascend.npu_dsa_staged_union_
    remap_op = torch.ops._C_ascend.npu_dsa_staged_remap_rows_
    copy_rows_op = torch.ops._C_ascend.npu_dsa_staged_copy_rows_
    unique_finalize_op = torch.ops._C_ascend.npu_dsa_staged_unique_finalize_
    sharded_union_op = torch.ops._C_ascend.npu_dsa_staged_sharded_union_
    sharded_vector_union_op = torch.ops._C_ascend.npu_dsa_staged_sharded_vector_union_
    sharded_vector_dedup_op = torch.ops._C_ascend.npu_dsa_staged_sharded_vector_dedup_

    request_batch = 4
    row_count = mtp * request_batch
    source = _mtp_rows_with_half_overlap(topk, request_batch, mtp, "npu")
    legacy_values = source.clone()
    union_values = source.clone()
    no_union_values = source.clone()
    no_union_local_indices = source.clone()
    hash_values = source.clone()
    sort_values = source.clone()
    sort_union_only_values = source.clone()
    native_unique_values = source.clone()
    sharded_sort_values = source.clone()
    sharded_vector_values = source.clone()
    sharded_vector_dedup_values = source.clone()
    production_staged_values = source.clone()
    resident_union_values = source.clone()
    max_tokens = 131072
    boundaries = torch.full((row_count,), max_tokens, dtype=torch.int32, device="npu")
    source_max = (mtp + 1) * topk // 2 - 1
    sharded_boundary = source_max - 100
    sharded_boundaries = torch.full(
        (row_count,),
        sharded_boundary,
        dtype=torch.int32,
        device="npu",
    )
    print(f"sharded split_boundary={sharded_boundary} (source max={source_max})")
    row_requests = torch.arange(request_batch, dtype=torch.int32, device="npu").repeat_interleave(mtp)

    # The pre-union operator assigns each row its own compact scratch range.
    valid_rows = torch.arange(row_count, dtype=torch.int32, device="npu")
    scratch_base = torch.arange(row_count, dtype=torch.int32, device="npu") * topk
    local_scratch_base = torch.zeros(row_count, dtype=torch.int32, device="npu")

    block_size = 128
    capacity = mtp * topk
    blocks_per_request = max_tokens // block_size
    block_table = torch.arange(
        request_batch * blocks_per_request,
        dtype=torch.int32,
        device="npu",
    ).reshape(request_batch, blocks_per_request)
    selected = torch.empty((request_batch, capacity), dtype=torch.int32, device="npu")
    counts = torch.empty((request_batch, 16), dtype=torch.int32, device="npu")
    targets = torch.empty((request_batch, capacity), dtype=torch.long, device="npu")
    hash_buffers = (
        torch.empty((request_batch, capacity), dtype=torch.int32, device="npu"),
        torch.empty((row_count, topk), dtype=torch.int32, device="npu"),
        torch.empty((request_batch, 16), dtype=torch.int32, device="npu"),
        torch.empty((request_batch, capacity), dtype=torch.long, device="npu"),
    )
    sort_buffers = tuple(torch.empty_like(item) for item in hash_buffers)
    native_unique_buffers = tuple(torch.empty_like(item) for item in hash_buffers)
    sharded_sort_buffers = tuple(torch.empty_like(item) for item in hash_buffers)
    sharded_vector_buffers = tuple(torch.empty_like(item) for item in hash_buffers)
    sharded_vector_dedup_buffers = tuple(torch.empty_like(item) for item in hash_buffers)
    production_staged_buffers = (
        torch.empty_like(hash_buffers[0]),
        torch.empty_like(hash_buffers[0]),
        torch.empty_like(hash_buffers[2]),
        torch.empty_like(hash_buffers[3]),
    )
    production_sharded_scratch = (
        torch.empty(
            (request_batch, 2, capacity),
            dtype=torch.int32,
            device="npu",
        ),
        torch.empty(
            (request_batch, 2, capacity),
            dtype=torch.int32,
            device="npu",
        ),
        torch.empty(
            (request_batch, 2, 16),
            dtype=torch.int32,
            device="npu",
        ),
    )
    sharded_sort_scratch = (
        torch.empty(
            (request_batch, shard_count, topk),
            dtype=torch.int32,
            device="npu",
        ),
        torch.empty(
            (request_batch, shard_count, mtp * topk),
            dtype=torch.int32,
            device="npu",
        ),
        torch.empty(
            (request_batch, shard_count, 16),
            dtype=torch.int32,
            device="npu",
        ),
    )
    sharded_vector_scratch = (
        torch.empty_like(sharded_sort_scratch[0]),
        torch.empty_like(sharded_sort_scratch[1]),
        torch.empty_like(sharded_sort_scratch[2]),
        torch.empty(
            (request_batch, shard_count, 2 * topk),
            dtype=torch.int32,
            device="npu",
        ),
    )
    sharded_vector_dedup_scratch = tuple(torch.empty_like(item) for item in sharded_sort_scratch)
    resident_union_workspace = (
        allocate_sorted_resident_workspace(
            request_batch,
            mtp,
            device=torch.device("npu"),
            shard_count=shard_count,
        )
        if mtp <= 2
        else None
    )
    resident_union_state = (
        allocate_sorted_resident_state(
            request_batch,
            request_batch,
            mtp,
            device=torch.device("npu"),
            shard_count=shard_count,
        )
        if mtp <= 2
        else None
    )
    resident_request_states = torch.arange(request_batch, dtype=torch.int32, device="npu")
    resident_request_generations = torch.ones(request_batch, dtype=torch.int64, device="npu")
    resident_cold_generations = torch.full((request_batch,), 2, dtype=torch.int64, device="npu")

    def staged(values, buffers, use_sort):
        return _staged_runner(
            legacy_op=legacy_op,
            union_op=union_op,
            remap_op=remap_op,
            values=values,
            boundaries=boundaries,
            valid_rows=valid_rows,
            local_scratch_base=local_scratch_base,
            row_requests=row_requests,
            request_block_table=block_table,
            selected_packed=buffers[0],
            local_to_union=buffers[1],
            selected_count=buffers[2],
            target_slots=buffers[3],
            block_size=block_size,
            max_tokens=max_tokens,
            use_sort=use_sort,
        )

    def staged_no_union():
        return _staged_no_union_runner(
            legacy_op=legacy_op,
            copy_rows_op=copy_rows_op,
            values=no_union_values,
            local_indices=no_union_local_indices,
            boundaries=boundaries,
            valid_rows=valid_rows,
            local_scratch_base=local_scratch_base,
            row_requests=row_requests,
        )

    def staged_native_unique():
        return _staged_native_unique_runner(
            legacy_op=legacy_op,
            finalize_op=unique_finalize_op,
            remap_op=remap_op,
            values=native_unique_values,
            boundaries=boundaries,
            valid_rows=valid_rows,
            local_scratch_base=local_scratch_base,
            row_requests=row_requests,
            request_block_table=block_table,
            selected_packed=native_unique_buffers[0],
            local_to_union=native_unique_buffers[1],
            selected_count=native_unique_buffers[2],
            target_slots=native_unique_buffers[3],
            block_size=block_size,
            packed_key_stride=max_tokens,
        )

    def staged_sharded_sort():
        return _staged_sharded_sort_runner(
            sharded_union_op=sharded_union_op,
            values=sharded_sort_values,
            boundaries=sharded_boundaries,
            request_block_table=block_table,
            selected_packed=sharded_sort_buffers[0],
            local_to_union=sharded_sort_buffers[1],
            selected_count=sharded_sort_buffers[2],
            target_slots=sharded_sort_buffers[3],
            shard_packed=sharded_sort_scratch[0],
            shard_mapping=sharded_sort_scratch[1],
            shard_counts=sharded_sort_scratch[2],
            block_size=block_size,
        )

    def staged_sharded_vector():
        return _staged_sharded_vector_runner(
            sharded_union_op=sharded_vector_union_op,
            values=sharded_vector_values,
            boundaries=sharded_boundaries,
            request_block_table=block_table,
            selected_packed=sharded_vector_buffers[0],
            local_to_union=sharded_vector_buffers[1],
            selected_count=sharded_vector_buffers[2],
            target_slots=sharded_vector_buffers[3],
            shard_packed=sharded_vector_scratch[0],
            shard_mapping=sharded_vector_scratch[1],
            shard_counts=sharded_vector_scratch[2],
            shard_pairs=sharded_vector_scratch[3],
            block_size=block_size,
        )

    def staged_sharded_vector_dedup():
        return _staged_sharded_sort_runner(
            sharded_union_op=sharded_vector_dedup_op,
            values=sharded_vector_dedup_values,
            boundaries=sharded_boundaries,
            request_block_table=block_table,
            selected_packed=sharded_vector_dedup_buffers[0],
            local_to_union=sharded_vector_dedup_buffers[1],
            selected_count=sharded_vector_dedup_buffers[2],
            target_slots=sharded_vector_dedup_buffers[3],
            shard_packed=sharded_vector_dedup_scratch[0],
            shard_mapping=sharded_vector_dedup_scratch[1],
            shard_counts=sharded_vector_dedup_scratch[2],
            block_size=block_size,
        )

    def resident_sharded_union(
        request_generations=resident_request_generations,
    ):
        assert resident_union_workspace is not None
        assert resident_union_state is not None
        prepare_resident_sharded_union_(
            resident_union_values,
            sharded_boundaries,
            row_requests,
            resident_request_states,
            request_generations,
            resident_union_state,
            resident_union_workspace,
            mtp=mtp,
        )

    def resident_fused_finalize():
        assert resident_union_workspace is not None
        assert resident_union_state is not None
        prepare_sorted_resident_cache_fused_(
            resident_union_values,
            block_table,
            resident_request_states,
            resident_request_generations,
            resident_union_state,
            resident_union_workspace,
            block_size=block_size,
        )

    def resident_end_to_end():
        resident_sharded_union()
        resident_fused_finalize()

    def resident_finalize_coordinator():
        assert resident_union_workspace is not None
        coordinate_sorted_resident_finalize_(resident_union_workspace)

    def resident_sharded_finalize_worker():
        assert resident_union_workspace is not None
        run_sharded_resident_finalize_(
            block_table,
            resident_union_workspace,
            block_size=block_size,
        )

    def resident_coordinated_plan():
        assert resident_union_workspace is not None
        assert resident_union_state is not None
        prepare_sorted_resident_cache_coordinated_(
            resident_union_values,
            block_table,
            resident_request_states,
            resident_request_generations,
            resident_union_state,
            resident_union_workspace,
            block_size=block_size,
        )

    def resident_coordinated_end_to_end():
        resident_sharded_union()
        resident_coordinated_plan()

    pair_baselines = mtp == 2
    native_unique_baseline = mtp in (2, 3)
    if not pair_baselines:
        print("staged-hash/staged-sort skipped: those legacy experiment kernels are fixed to two rows per request")
    if not native_unique_baseline:
        print("native-unique skipped: its finalize UB layout is only benchmarked through MTP=3")
    no_union_result = staged_no_union()
    hash_result = staged(hash_values, hash_buffers, False) if pair_baselines else None
    sort_result = staged(sort_values, sort_buffers, True) if pair_baselines else None
    native_unique_result = staged_native_unique() if native_unique_baseline else None
    sharded_sort_result = staged_sharded_sort()
    sharded_vector_result = staged_sharded_vector()
    sharded_vector_dedup_result = staged_sharded_vector_dedup()
    if resident_union_workspace is not None:
        resident_sharded_union()
    production_staged_result = None
    if mtp <= 2:
        production_result = prepare_sparse_indices(
            production_staged_values,
            sharded_boundaries,
            row_req_indices=row_requests,
            request_block_table=block_table,
            selected_packed=production_staged_buffers[0],
            selected_counts=production_staged_buffers[2],
            target_slot_mapping=production_staged_buffers[3],
            block_size=block_size,
            local_to_union_workspace=production_staged_buffers[1],
            shard_packed_workspace=production_sharded_scratch[0],
            shard_mapping_workspace=production_sharded_scratch[1],
            shard_counts_workspace=production_sharded_scratch[2],
            staged_mtp=mtp,
        )
        production_staged_result = (
            production_result[0],
            production_result[1],
            production_staged_buffers[2],
            production_result[3],
        )
    sort_union_only_packed = None
    if pair_baselines:
        sort_union_only_packed = legacy_op(
            sort_union_only_values,
            boundaries,
            valid_rows,
            local_scratch_base,
            True,
            row_requests,
        )
    remap_only_seed = None
    remap_only_values = None
    if pair_baselines:
        union_op(
            sort_union_only_packed,
            sort_buffers[0],
            sort_buffers[1],
            sort_buffers[2],
            block_table,
            sort_buffers[3],
            block_size,
            max_tokens,
            True,
        )
        remap_only_seed = sort_union_only_values.clone()
        remap_only_values = remap_only_seed.clone()
    torch.npu.synchronize()
    expected_local_indices = torch.arange(topk, dtype=torch.int32, device="npu").expand(row_count, 1, -1)
    if not torch.equal(no_union_result.cpu(), expected_local_indices.cpu()):
        raise AssertionError("staged no-union remapped rows are incorrect")
    full_expected_count = (mtp + 1) * topk // 2
    expected_counts = [full_expected_count] * request_batch
    if pair_baselines:
        if hash_result[2][:, 0].cpu().tolist() != expected_counts:
            raise AssertionError("hash staged union count is incorrect")
        if sort_result[2][:, 0].cpu().tolist() != expected_counts:
            raise AssertionError("sort staged union count is incorrect")
    if native_unique_baseline:
        if native_unique_result[2][:, 0].cpu().tolist() != expected_counts:
            raise AssertionError("native unique staged union count is incorrect")
    if pair_baselines:
        if not torch.equal(hash_result[0].cpu(), sort_result[0].cpu()):
            raise AssertionError("hash and sort remapped rows differ")
        for index in (1, 3):
            if not torch.equal(
                hash_result[index][:, :full_expected_count].cpu(),
                sort_result[index][:, :full_expected_count].cpu(),
            ):
                raise AssertionError("hash and sort staged payloads differ")
            if not torch.equal(
                hash_result[index][:, :full_expected_count].cpu(),
                native_unique_result[index][:, :full_expected_count].cpu(),
            ):
                raise AssertionError("hash and native unique staged payloads differ")
        if not torch.equal(hash_result[0].cpu(), native_unique_result[0].cpu()):
            raise AssertionError("hash and native unique remapped rows differ")
    _validate_sharded_result(
        label="sharded sort",
        result=sharded_sort_result,
        source=source,
        boundary=sharded_boundary,
        request_batch=request_batch,
        mtp=mtp,
        topk=topk,
        max_tokens=max_tokens,
    )
    _validate_sharded_result(
        label="sharded vector map",
        result=sharded_vector_result,
        source=source,
        boundary=sharded_boundary,
        request_batch=request_batch,
        mtp=mtp,
        topk=topk,
        max_tokens=max_tokens,
    )
    _validate_sharded_result(
        label="sharded vector dedup",
        result=sharded_vector_dedup_result,
        source=source,
        boundary=sharded_boundary,
        request_batch=request_batch,
        mtp=mtp,
        topk=topk,
        max_tokens=max_tokens,
    )
    if production_staged_result is not None:
        _validate_sharded_result(
            label="production staged",
            result=production_staged_result,
            source=source,
            boundary=sharded_boundary,
            request_batch=request_batch,
            mtp=mtp,
            topk=topk,
            max_tokens=max_tokens,
        )
    if resident_union_workspace is not None:
        _validate_resident_sharded_union(
            source,
            sharded_boundary,
            resident_union_workspace.shard_packed,
            resident_union_workspace.shard_mapping,
            resident_union_workspace.shard_counts,
            mtp=mtp,
        )
        # Validate the exact production pair once cold and once resident. The
        # first finalize must emit the complete union; the next identical step
        # must emit no LMCache payload while still remapping every top-k row.
        resident_fused_finalize()
        torch.npu.synchronize()
        if resident_union_workspace.miss_counts[:, 0].cpu().tolist() != [sharded_boundary] * request_batch:
            raise AssertionError("cold resident fused path emitted an incorrect miss count")
        resident_union_values.copy_(source)
        resident_end_to_end()
        torch.npu.synchronize()
        if resident_union_workspace.miss_counts[:, 0].cpu().tolist() != [0] * request_batch:
            raise AssertionError("steady resident fused path did not eliminate all hits")
        _validate_resident_remap(
            source,
            sharded_boundary,
            resident_union_values,
            resident_union_state,
            mtp=mtp,
        )

        # Build a deterministic 90%-hit state for stage timing. All scratch
        # slots are occupied: 90% of the current union is resident and the
        # rest are stale tokens, so every measured miss requires eviction.
        resident_hit_count, resident_expected_misses = _seed_sorted_resident_state_for_hit_rate(
            resident_union_state,
            request_batch=request_batch,
            valid_token_count=sharded_boundary,
            capacity=capacity,
            shard_count=shard_count,
            hit_rate=resident_hit_rate,
        )
        resident_state_seed = (
            resident_union_state.tokens.clone(),
            resident_union_state.slots.clone(),
            resident_union_state.counts.clone(),
            resident_union_state.generations.clone(),
        )

        def restore_resident_state():
            resident_union_state.tokens.copy_(resident_state_seed[0])
            resident_union_state.slots.copy_(resident_state_seed[1])
            resident_union_state.counts.copy_(resident_state_seed[2])
            resident_union_state.generations.copy_(resident_state_seed[3])

        resident_union_values.copy_(source)
        resident_sharded_union()
        torch.npu.synchronize()
        count_cpu = resident_union_workspace.shard_counts[:, :, 0].cpu()
        prior_cpu = resident_union_workspace.prior_slots.cpu()
        actual_misses = []
        for request in range(request_batch):
            request_misses = 0
            for shard in range(shard_count):
                count = int(count_cpu[request, shard])
                request_misses += int((prior_cpu[request, shard, :count] < 0).sum())
            actual_misses.append(request_misses)
        if actual_misses != [resident_expected_misses] * request_batch:
            raise AssertionError(
                "resident 90%-hit union emitted incorrect miss markers: "
                f"actual={actual_misses}, "
                f"expected={resident_expected_misses}"
            )
        resident_union_seed = (
            resident_union_workspace.shard_packed.clone(),
            resident_union_workspace.shard_mapping.clone(),
            resident_union_workspace.shard_counts.clone(),
            resident_union_workspace.prior_slots.clone(),
            resident_union_workspace.shard_miss_tokens.clone(),
            resident_union_workspace.shard_miss_positions.clone(),
            resident_union_workspace.shard_evictable_slots.clone(),
        )

        resident_finalize_debug = torch.empty((request_batch, 16), dtype=torch.int32, device="npu")

        def resident_finalize_only():
            debug_sorted_resident_finalize_only_(
                block_table,
                resident_union_workspace,
                block_size=block_size,
                debug_info=resident_finalize_debug,
            )

        resident_finalize_only()
        torch.npu.synchronize()
        if resident_union_workspace.miss_counts[:, 0].cpu().tolist() != [resident_expected_misses] * request_batch:
            raise AssertionError("resident 90%-hit finalize emitted an incorrect miss count")
        resident_selected_evicts = min(
            resident_expected_misses,
            capacity - resident_hit_count,
        )
        selected_evicts = resident_union_workspace.shard_counts[:, :, 4].sum(dim=1).cpu().tolist()
        if selected_evicts != [resident_selected_evicts] * request_batch:
            raise AssertionError(
                "resident finalize selected an incorrect evict prefix: "
                f"actual={selected_evicts}, expected={resident_selected_evicts}"
            )
        resident_finalized_seed = resident_union_workspace.prior_slots.clone()

        def resident_update_only():
            debug_sorted_resident_update_only_(
                resident_union_values,
                resident_request_states,
                resident_request_generations,
                resident_union_state,
                resident_union_workspace,
            )

        resident_update_only()
        torch.npu.synchronize()
        _validate_resident_remap(
            source,
            sharded_boundary,
            resident_union_values,
            resident_union_state,
            mtp=mtp,
        )

        # Validate the self-coordinating sharded finalize from the same
        # union/state seed. The coordinator remains an isolated comparison.
        resident_union_values.copy_(source)
        restore_resident_state()
        resident_union_workspace.shard_packed.copy_(resident_union_seed[0])
        resident_union_workspace.shard_mapping.copy_(resident_union_seed[1])
        resident_union_workspace.shard_counts.copy_(resident_union_seed[2])
        resident_union_workspace.prior_slots.copy_(resident_union_seed[3])
        resident_union_workspace.shard_miss_tokens.copy_(resident_union_seed[4])
        resident_union_workspace.shard_miss_positions.copy_(resident_union_seed[5])
        resident_union_workspace.shard_evictable_slots.copy_(resident_union_seed[6])
        resident_sharded_finalize_worker()
        torch.npu.synchronize()
        if resident_union_workspace.miss_counts[:, 0].cpu().tolist() != [resident_expected_misses] * request_batch:
            raise AssertionError("sharded resident finalize emitted an incorrect miss count")
        coordinated_selected_evicts = resident_union_workspace.shard_counts[:, :, 4].sum(dim=1).cpu().tolist()
        if coordinated_selected_evicts != [resident_selected_evicts] * request_batch:
            raise AssertionError(
                "resident sharded finalize selected an incorrect evict prefix: "
                f"actual={coordinated_selected_evicts}, "
                f"expected={resident_selected_evicts}"
            )
        resident_update_only()
        torch.npu.synchronize()
        _validate_resident_remap(
            source,
            sharded_boundary,
            resident_union_values,
            resident_union_state,
            mtp=mtp,
        )

        def restore_resident_union_seed():
            resident_union_workspace.shard_packed.copy_(resident_union_seed[0])
            resident_union_workspace.shard_mapping.copy_(resident_union_seed[1])
            resident_union_workspace.shard_counts.copy_(resident_union_seed[2])
            resident_union_workspace.prior_slots.copy_(resident_union_seed[3])
            resident_union_workspace.shard_miss_tokens.copy_(resident_union_seed[4])
            resident_union_workspace.shard_miss_positions.copy_(resident_union_seed[5])
            resident_union_workspace.shard_evictable_slots.copy_(resident_union_seed[6])

        def reset_resident_union():
            resident_union_values.copy_(source)
            restore_resident_state()

        def reset_resident_finalize():
            resident_union_workspace.prior_slots.copy_(resident_union_seed[3])

        def reset_resident_coordinator():
            resident_union_workspace.shard_counts.copy_(resident_union_seed[2])

        def reset_resident_sharded_finalize():
            resident_union_workspace.shard_counts.copy_(resident_union_seed[2])
            resident_union_workspace.prior_slots.copy_(resident_union_seed[3])

        def reset_resident_update():
            resident_union_values.copy_(source)
            restore_resident_state()
            resident_union_workspace.prior_slots.copy_(resident_finalized_seed)

        def reset_resident_plan():
            resident_union_values.copy_(source)
            restore_resident_state()
            restore_resident_union_seed()

        def reset_resident_end_to_end():
            resident_union_values.copy_(source)
            restore_resident_state()

    legacy_samples = _measure_npu_ms(
        lambda: legacy_op(
            legacy_values,
            boundaries,
            valid_rows,
            scratch_base,
            True,
            row_requests,
        ),
        lambda: legacy_values.copy_(source),
        warmups,
        iterations,
    )
    union_samples = _measure_npu_ms(
        lambda: prepare_sparse_indices(
            union_values,
            sharded_boundaries,
            row_req_indices=row_requests,
            request_block_table=block_table,
            selected_packed=selected,
            selected_counts=counts,
            target_slot_mapping=targets,
            block_size=block_size,
        ),
        lambda: union_values.copy_(source),
        warmups,
        iterations,
    )
    no_union_samples = _measure_npu_ms(
        staged_no_union,
        lambda: no_union_local_indices.copy_(source),
        warmups,
        iterations,
    )
    hash_samples = []
    sort_samples = []
    sort_union_only_samples = []
    remap_only_samples = []
    if pair_baselines:
        hash_samples = _measure_npu_ms(
            lambda: staged(hash_values, hash_buffers, False),
            lambda: hash_values.copy_(source),
            warmups,
            iterations,
        )
        sort_samples = _measure_npu_ms(
            lambda: staged(sort_values, sort_buffers, True),
            lambda: sort_values.copy_(source),
            warmups,
            iterations,
        )
        sort_union_only_samples = _measure_npu_ms(
            lambda: union_op(
                sort_union_only_packed,
                sort_buffers[0],
                sort_buffers[1],
                sort_buffers[2],
                block_table,
                sort_buffers[3],
                block_size,
                max_tokens,
                True,
            ),
            lambda: None,
            warmups,
            iterations,
        )
        remap_only_samples = _measure_npu_ms(
            lambda: remap_op(remap_only_values, sort_buffers[1]),
            lambda: remap_only_values.copy_(remap_only_seed),
            warmups,
            iterations,
        )
    native_unique_samples = []
    if native_unique_baseline:
        native_unique_samples = _measure_npu_ms(
            staged_native_unique,
            lambda: native_unique_values.copy_(source),
            warmups,
            iterations,
        )
    sharded_sort_samples = _measure_npu_ms(
        staged_sharded_sort,
        lambda: sharded_sort_values.copy_(source),
        warmups,
        iterations,
    )
    sharded_vector_samples = _measure_npu_ms(
        staged_sharded_vector,
        lambda: sharded_vector_values.copy_(source),
        warmups,
        iterations,
    )
    sharded_vector_dedup_samples = _measure_npu_ms(
        staged_sharded_vector_dedup,
        lambda: sharded_vector_dedup_values.copy_(source),
        warmups,
        iterations,
    )
    production_staged_samples = []
    if mtp <= 2:
        production_staged_samples = _measure_npu_ms(
            lambda: prepare_sparse_indices(
                production_staged_values,
                sharded_boundaries,
                row_req_indices=row_requests,
                request_block_table=block_table,
                selected_packed=production_staged_buffers[0],
                selected_counts=production_staged_buffers[2],
                target_slot_mapping=production_staged_buffers[3],
                block_size=block_size,
                local_to_union_workspace=production_staged_buffers[1],
                shard_packed_workspace=production_sharded_scratch[0],
                shard_mapping_workspace=production_sharded_scratch[1],
                shard_counts_workspace=production_sharded_scratch[2],
                staged_mtp=mtp,
            ),
            lambda: production_staged_values.copy_(source),
            warmups,
            iterations,
        )
    resident_union_samples = []
    resident_no_intersection_samples = []
    resident_finalize_samples = []
    resident_coordinator_samples = []
    resident_sharded_finalize_samples = []
    resident_update_samples = []
    resident_plan_samples = []
    resident_coordinated_plan_samples = []
    resident_end_to_end_samples = []
    resident_coordinated_end_to_end_samples = []
    if resident_union_workspace is not None:
        resident_union_samples = _measure_npu_ms(
            resident_sharded_union,
            reset_resident_union,
            warmups,
            iterations,
        )
        resident_no_intersection_samples = _measure_npu_ms(
            lambda: resident_sharded_union(resident_cold_generations),
            reset_resident_union,
            warmups,
            iterations,
        )
        resident_finalize_samples = _measure_npu_ms(
            resident_finalize_only,
            reset_resident_finalize,
            warmups,
            iterations,
        )
        resident_coordinator_samples = _measure_npu_ms(
            resident_finalize_coordinator,
            reset_resident_coordinator,
            warmups,
            iterations,
        )
        resident_sharded_finalize_samples = _measure_npu_ms(
            resident_sharded_finalize_worker,
            reset_resident_sharded_finalize,
            warmups,
            iterations,
        )
        resident_update_samples = _measure_npu_ms(
            resident_update_only,
            reset_resident_update,
            warmups,
            iterations,
        )
        resident_plan_samples = _measure_npu_ms(
            resident_fused_finalize,
            reset_resident_plan,
            warmups,
            iterations,
        )
        resident_coordinated_plan_samples = _measure_npu_ms(
            resident_coordinated_plan,
            reset_resident_plan,
            warmups,
            iterations,
        )
        resident_end_to_end_samples = _measure_npu_ms(
            resident_end_to_end,
            reset_resident_end_to_end,
            warmups,
            iterations,
        )
        resident_coordinated_end_to_end_samples = _measure_npu_ms(
            resident_coordinated_end_to_end,
            reset_resident_end_to_end,
            warmups,
            iterations,
        )

    legacy_mean = statistics.fmean(legacy_samples)
    union_mean = statistics.fmean(union_samples)
    no_union_mean = statistics.fmean(no_union_samples)
    sharded_sort_mean = statistics.fmean(sharded_sort_samples)
    sharded_vector_mean = statistics.fmean(sharded_vector_samples)
    sharded_vector_dedup_mean = statistics.fmean(sharded_vector_dedup_samples)
    _summary("pre-union", legacy_samples)
    _summary("fused-union", union_samples)
    _summary("staged-no-union", no_union_samples)
    _summary("sharded-sort", sharded_sort_samples)
    _summary("sharded-vector-map", sharded_vector_samples)
    _summary("sharded-vector-dedup", sharded_vector_dedup_samples)
    production_staged_mean = None
    uses_sharded_operator = _sparse_index_op_name(mtp) == "npu_dsa_prepare_sparse_indices_sharded_"
    production_label = (
        "production-sharded-single-row"
        if uses_sharded_operator and mtp == 1
        else ("production-sharded-sort" if uses_sharded_operator else "production-staged-sort")
    )
    if production_staged_samples:
        production_staged_mean = statistics.fmean(production_staged_samples)
        _summary(production_label, production_staged_samples)
    resident_union_mean = None
    resident_no_intersection_mean = None
    resident_end_to_end_mean = None
    if resident_union_samples:
        resident_union_mean = statistics.fmean(resident_union_samples)
        resident_no_intersection_mean = statistics.fmean(resident_no_intersection_samples)
        _summary(
            "resident-union-no-intersect",
            resident_no_intersection_samples,
        )
        _summary(
            "resident-sort+intersect" if mtp == 1 else "resident-union+intersect",
            resident_union_samples,
        )
        resident_end_to_end_mean = statistics.fmean(resident_end_to_end_samples)
        print(
            "resident stage workload: "
            f"hit_rate={resident_hit_count / sharded_boundary:.2%}, "
            f"hits={resident_hit_count}, "
            f"misses={resident_expected_misses}, "
            f"evict_candidates={capacity - resident_hit_count}, "
            f"evict_slots_copied={resident_selected_evicts}, "
            "scratch_full=True"
        )
        _summary("resident-finalize-kernel", resident_finalize_samples)
        _summary("resident-coordinator-kernel", resident_coordinator_samples)
        _summary(
            "resident-sharded-finalize-kernel",
            resident_sharded_finalize_samples,
        )
        _summary("resident-update+remap-kernel", resident_update_samples)
        _summary("resident-plan-two-kernel", resident_plan_samples)
        _summary(
            "resident-sharded-plan-two-kernel",
            resident_coordinated_plan_samples,
        )
        _summary("resident-end-to-end", resident_end_to_end_samples)
        _summary(
            "resident-sharded-end-to-end",
            resident_coordinated_end_to_end_samples,
        )
        resident_finalize_mean = statistics.fmean(resident_finalize_samples)
        resident_update_mean = statistics.fmean(resident_update_samples)
        resident_plan_mean = statistics.fmean(resident_plan_samples)
        resident_coordinator_mean = statistics.fmean(resident_coordinator_samples)
        resident_sharded_finalize_mean = statistics.fmean(resident_sharded_finalize_samples)
        resident_coordinated_plan_mean = statistics.fmean(resident_coordinated_plan_samples)
        resident_isolated_sum = resident_finalize_mean + resident_update_mean
        slower_stage = "finalize" if resident_finalize_mean >= resident_update_mean else "update+remap"
        print(
            "resident two-kernel breakdown: "
            f"finalize={resident_finalize_mean / resident_isolated_sum:.2%}, "
            f"update+remap={resident_update_mean / resident_isolated_sum:.2%}, "
            f"slower={slower_stage}, "
            f"isolated_sum-plan={resident_isolated_sum - resident_plan_mean:+.6f} ms"
        )
        coordinated_isolated_sum = resident_sharded_finalize_mean + resident_update_mean
        print(
            "resident sharded-plan breakdown: "
            f"coordinator-isolated={resident_coordinator_mean:.6f} ms, "
            f"sharded_finalize={resident_sharded_finalize_mean:.6f} ms, "
            f"update+remap={resident_update_mean:.6f} ms, "
            f"isolated_sum-plan="
            f"{coordinated_isolated_sum - resident_coordinated_plan_mean:+.6f} ms, "
            f"plan_delta_vs_original="
            f"{resident_coordinated_plan_mean - resident_plan_mean:+.6f} ms"
        )
        print(
            "resident intersection-only overhead: "
            f"{resident_union_mean - resident_no_intersection_mean:+.6f} ms "
            f"({(resident_union_mean / resident_no_intersection_mean - 1) * 100:+.2f}%)"
        )
    native_unique_mean = None
    if native_unique_baseline:
        native_unique_mean = statistics.fmean(native_unique_samples)
        _summary("native-unique", native_unique_samples)
    hash_mean = None
    sort_mean = None
    if pair_baselines:
        hash_mean = statistics.fmean(hash_samples)
        sort_mean = statistics.fmean(sort_samples)
        _summary("staged-hash", hash_samples)
        _summary("staged-sort", sort_samples)
        _summary("sort-union", sort_union_only_samples)
        _summary("remap-only", remap_only_samples)
    print(f"union overhead: {union_mean - legacy_mean:+.6f} ms ({(union_mean / legacy_mean - 1) * 100:+.2f}%)")
    if pair_baselines:
        print(
            "staged hash union cost: "
            f"{hash_mean - no_union_mean:+.6f} ms "
            f"({(hash_mean / no_union_mean - 1) * 100:+.2f}%)"
        )
        print(
            "staged sort union cost: "
            f"{sort_mean - no_union_mean:+.6f} ms "
            f"({(sort_mean / no_union_mean - 1) * 100:+.2f}%)"
        )
    if native_unique_baseline:
        print(
            "native unique cost: "
            f"{native_unique_mean - no_union_mean:+.6f} ms "
            f"({(native_unique_mean / no_union_mean - 1) * 100:+.2f}%)"
        )
    print(
        "sharded sort union cost: "
        f"{sharded_sort_mean - no_union_mean:+.6f} ms "
        f"({(sharded_sort_mean / no_union_mean - 1) * 100:+.2f}%)"
    )
    print(
        "sharded vector-map union cost: "
        f"{sharded_vector_mean - no_union_mean:+.6f} ms "
        f"({(sharded_vector_mean / no_union_mean - 1) * 100:+.2f}%)"
    )
    print(
        "sharded vector-dedup union cost: "
        f"{sharded_vector_dedup_mean - no_union_mean:+.6f} ms "
        f"({(sharded_vector_dedup_mean / no_union_mean - 1) * 100:+.2f}%)"
    )
    if resident_end_to_end_mean is not None:
        print(
            "resident intersection/state overhead vs sharded-sort: "
            f"{resident_end_to_end_mean - sharded_sort_mean:+.6f} ms "
            f"({(resident_end_to_end_mean / sharded_sort_mean - 1) * 100:+.2f}%)"
        )
    candidates = [
        ("pre-union", legacy_mean),
        ("fused-union", union_mean),
        ("staged-no-union", no_union_mean),
        ("sharded-sort", sharded_sort_mean),
        ("sharded-vector-map", sharded_vector_mean),
        ("sharded-vector-dedup", sharded_vector_dedup_mean),
    ]
    if production_staged_mean is not None:
        candidates.append((production_label, production_staged_mean))
    if resident_union_mean is not None:
        candidates.append(
            (
                "resident-sort+intersect" if mtp == 1 else "resident-union+intersect",
                resident_union_mean,
            )
        )
        candidates.append(("resident-end-to-end", resident_end_to_end_mean))
    if native_unique_baseline:
        candidates.append(("native-unique", native_unique_mean))
    if pair_baselines:
        candidates.extend(
            (
                ("staged-hash", hash_mean),
                ("staged-sort", sort_mean),
            )
        )
    fastest = min(candidates, key=lambda item: item[1])
    print(f"fastest: {fastest[0]} ({fastest[1]:.6f} ms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--mtp", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--hit-rate", type=float, default=0.9)
    parser.add_argument(
        "--shards-per-row",
        type=int,
        default=None,
        help=(
            "benchmark-only row shard multiplier; total launch shards are "
            "MTP * shards_per_row, while the default remains unchanged"
        ),
    )
    parser.add_argument(
        "--parallel-map",
        action="store_true",
        help=("deprecated compatibility flag for the standalone resident benchmark"),
    )
    parser.add_argument(
        "--production-only",
        action="store_true",
        help=(
            "benchmark only the selected production staged operator; "
            "do not initialize or run experimental sharded paths"
        ),
    )
    parser.add_argument(
        "--resident-only",
        action="store_true",
        help=("deprecated compatibility redirect to benchmark_resident_sparse_cache.py"),
    )
    args = parser.parse_args()
    if args.production_only and args.resident_only:
        parser.error("--production-only and --resident-only are exclusive")
    if args.resident_only:
        resident_only_main(
            args.topk,
            args.mtp,
            args.requests,
            args.hit_rate,
            args.parallel_map,
            args.iterations,
            args.warmups,
        )
    elif args.production_only:
        production_only_main(
            args.topk,
            args.mtp,
            args.iterations,
            args.warmups,
        )
    else:
        main(
            args.topk,
            args.mtp,
            args.iterations,
            args.warmups,
            args.hit_rate,
            args.shards_per_row,
        )
