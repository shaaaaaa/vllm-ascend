import argparse
import statistics

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    prepare_sparse_indices,
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
    """Benchmark only the production staged sparse-index operator."""
    if topk != 2048:
        raise ValueError("the production staged operator requires --topk 2048")
    if mtp not in (1, 2):
        raise ValueError("the production staged operator supports only --mtp 1 or 2")
    if not enable_custom_op():
        raise RuntimeError("vllm-ascend custom operators could not be loaded")
    try:
        production_op = (
            torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_staged_
        )
    except AttributeError as exc:
        raise RuntimeError(
            "vllm_ascend_C does not expose the production staged operator; "
            "rebuild the custom-op extension"
        ) from exc

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

    def run() -> None:
        production_op(
            values,
            boundaries,
            row_requests,
            block_table,
            selected,
            counts,
            targets,
            local_to_union_workspace,
            block_size,
            mtp,
            True,
            True,
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
            selected_cpu[request, :expected_count],
            expected_selected,
        ):
            raise AssertionError(
                f"request {request} production staged union is incorrect"
            )
        expected_targets = (
            torch.arange(expected_count, dtype=torch.long)
            + request * max_tokens
        )
        if not torch.equal(
            targets_cpu[request, :expected_count],
            expected_targets,
        ):
            raise AssertionError(
                f"request {request} production staged targets are incorrect"
            )

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
        raise AssertionError(
            "production staged remapped rows do not reconstruct the input"
        )

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
    _summary("production-staged", samples)


def main(
    topk: int = 2048,
    mtp: int = 2,
    iterations: int = 200,
    warmups: int = 20,
) -> None:
    if topk != 2048:
        raise ValueError("the experimental staged kernels currently require --topk 2048")
    if mtp < 1 or mtp > 8:
        raise ValueError("the sharded benchmark supports --mtp from 1 to 8")
    if not enable_custom_op():
        raise RuntimeError("vllm-ascend custom operators could not be loaded")
    print(f"MTP depth={mtp}, value shards={1 << (mtp - 1).bit_length()}")
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
    shard_count = 1 << (mtp - 1).bit_length()
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
                staged_mtp=mtp,
            ),
            lambda: production_staged_values.copy_(source),
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
    if production_staged_samples:
        production_staged_mean = statistics.fmean(
            production_staged_samples
        )
        _summary("production-staged", production_staged_samples)
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
    candidates = [
        ("pre-union", legacy_mean),
        ("fused-union", union_mean),
        ("staged-no-union", no_union_mean),
        ("sharded-sort", sharded_sort_mean),
        ("sharded-vector-map", sharded_vector_mean),
        ("sharded-vector-dedup", sharded_vector_dedup_mean),
    ]
    if production_staged_mean is not None:
        candidates.append(("production-staged", production_staged_mean))
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
    parser.add_argument(
        "--production-only",
        action="store_true",
        help=(
            "benchmark only npu_dsa_prepare_sparse_indices_staged_; "
            "do not initialize or run experimental sharded paths"
        ),
    )
    args = parser.parse_args()
    if args.production_only:
        production_only_main(
            args.topk,
            args.mtp,
            args.iterations,
            args.warmups,
        )
    else:
        main(args.topk, args.mtp, args.iterations, args.warmups)
