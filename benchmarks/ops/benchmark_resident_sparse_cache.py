"""Benchmark persistent sparse-cache planning independently of MTP union.

This benchmark starts from a synthetic, already-deduplicated request union.
It compares only the resident-cache implementations and separately measures
the two torch-npu ACLNN primitives used by the hybrid path.
"""

import argparse
import statistics

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sparse_cache import (
    allocate_resident_workspace,
    prepare_resident_sparse_cache_,
)
from vllm_ascend.utils import enable_custom_op


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
    print(
        f"{name:>14}: mean={statistics.fmean(samples):.6f} ms "
        f"p50={p50:.6f} ms p90={p90:.6f} ms"
    )


def _synthetic_union(
    topk: int,
    mtp: int,
    requests: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Create top-k rows plus a matching sorted union without running union."""
    shared = torch.arange(topk // 2, dtype=torch.int32, device=device)
    rows = torch.stack(
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
    local_positions = rows.reshape(1, mtp * topk).expand(requests, -1)
    # Match the existing benchmark convention: ignore the highest 100 token
    # ids. Since the synthetic union is [0, max], its valid count equals the
    # exclusive split boundary.
    split_boundary = int(rows.max().item()) - 100
    union_count = split_boundary
    selected = torch.zeros(
        (requests, mtp * topk), dtype=torch.int32, device=device
    )
    selected[:, :union_count] = torch.arange(
        union_count, dtype=torch.int32, device=device
    )
    mapping = torch.where(
        local_positions < split_boundary,
        local_positions,
        torch.full_like(local_positions, -1),
    )
    topk_indices = rows.repeat(requests, 1).reshape(
        requests * mtp, 1, topk
    )
    return topk_indices, selected, mapping, union_count


def main(
    topk: int = 2048,
    mtp: int = 2,
    requests: int = 4,
    hit_rate: float = 0.9,
    iterations: int = 500,
    warmups: int = 50,
) -> None:
    if topk != 2048:
        raise ValueError("resident benchmark currently requires --topk 2048")
    if mtp not in (1, 2):
        raise ValueError("resident benchmark supports only --mtp 1 or 2")
    if requests <= 0:
        raise ValueError("--requests must be positive")
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError("--hit-rate must be between 0 and 1")
    if not enable_custom_op():
        raise RuntimeError("vllm-ascend custom operators could not be loaded")

    device = torch.device("npu")
    capacity = mtp * topk
    max_tokens = 131072
    block_size = 128
    topk_seed, union_seed, mapping_seed, union_count = _synthetic_union(
        topk,
        mtp,
        requests,
        device=device,
    )
    count_seed = torch.zeros(
        (requests, 16), dtype=torch.int32, device=device
    )
    count_seed[:, 0] = union_count
    target_seed = torch.zeros(
        (requests, capacity), dtype=torch.int64, device=device
    )
    blocks_per_request = max_tokens // block_size
    block_table = torch.arange(
        requests * blocks_per_request,
        dtype=torch.int32,
        device=device,
    ).reshape(requests, blocks_per_request)

    values = topk_seed.clone()
    selected = union_seed.clone()
    counts = count_seed.clone()
    mapping = mapping_seed.clone()
    targets = target_seed.clone()

    token_stride = ((max_tokens + 1 + 31) // 32) * 32
    slot_stride = ((capacity + 1 + 15) // 16) * 16
    token_to_slot = torch.full(
        (2 * requests, token_stride),
        -1,
        dtype=torch.int16,
        device=device,
    )
    slot_to_token = torch.full(
        (2 * requests, slot_stride),
        -1,
        dtype=torch.int32,
        device=device,
    )
    state_generations = torch.ones(
        (2 * requests, 8), dtype=torch.int64, device=device
    )
    request_states = torch.arange(
        requests, dtype=torch.int32, device=device
    )
    request_generations = torch.ones(
        requests, dtype=torch.int64, device=device
    )
    workspace = allocate_resident_workspace(
        requests, capacity, device=device
    )

    hit_count = int(union_count * hit_rate)
    seed_token_to_slot = token_to_slot.clone()
    seed_slot_to_token = slot_to_token.clone()
    if hit_count:
        hit_tokens = union_seed[:, :hit_count].to(torch.int64)
        hit_slots = torch.arange(
            capacity - hit_count,
            capacity,
            dtype=torch.int16,
            device=device,
        ).expand(requests, -1)
        seed_token_to_slot[
            request_states.to(torch.int64).reshape(-1, 1),
            hit_tokens,
        ] = hit_slots
        seed_slot_to_token[:, capacity - hit_count : capacity].copy_(
            union_seed[:, :hit_count]
        )

    def reset_inputs() -> None:
        values.copy_(topk_seed)
        selected.copy_(union_seed)
        counts.copy_(count_seed)
        mapping.copy_(mapping_seed)
        targets.copy_(target_seed)
        token_to_slot.copy_(seed_token_to_slot)
        slot_to_token.copy_(seed_slot_to_token)
        state_generations.fill_(1)

    def resident_plan(use_hybrid_aiv: bool) -> None:
        prepare_resident_sparse_cache_(
            values,
            mapping,
            selected,
            counts,
            targets,
            block_table,
            request_states,
            request_generations,
            token_to_slot,
            slot_to_token,
            state_generations,
            workspace,
            block_size=block_size,
            scratch_capacity=capacity,
            parallel_map=use_hybrid_aiv,
        )

    expected_misses = [union_count - hit_count] * requests
    reset_inputs()
    resident_plan(False)
    torch.npu.synchronize()
    if counts[:, 0].cpu().tolist() != expected_misses:
        raise AssertionError("all-Torch resident miss counts are incorrect")
    reference = (
        values.clone(),
        counts[:, 0].clone(),
        token_to_slot.clone(),
        slot_to_token.clone(),
        state_generations.clone(),
    )
    reference_selected = selected.clone()
    reference_targets = targets.clone()

    reset_inputs()
    resident_plan(True)
    torch.npu.synchronize()
    actual = (
        values,
        counts[:, 0],
        token_to_slot,
        slot_to_token,
        state_generations,
    )
    mismatch_names = [
        name
        for name, expected, result in zip(
            (
                "topk",
                "miss counts",
                "token_to_slot",
                "slot_to_token",
                "state generations",
            ),
            reference,
            actual,
        )
        if not torch.equal(expected.cpu(), result.cpu())
    ]
    for request, miss_count in enumerate(expected_misses):
        if not torch.equal(
            reference_selected[request, :miss_count].cpu(),
            selected[request, :miss_count].cpu(),
        ):
            mismatch_names.append(f"miss payload request {request}")
        if not torch.equal(
            reference_targets[request, :miss_count].cpu(),
            targets[request, :miss_count].cpu(),
        ):
            mismatch_names.append(f"target slots request {request}")
    if mismatch_names:
        raise AssertionError(
            "hybrid resident planner differs from all-Torch reference: "
            + ", ".join(mismatch_names)
        )

    # Save the exact scatter payload emitted by the hybrid planner.
    scatter_indices_seed = (
        workspace.state_token_indices[:requests].clone()
    )
    scatter_values_seed = workspace.short_sources[:requests].clone()

    # Recreate the exact gather indices without invoking resident finalize.
    reset_inputs()
    torch.ops._C_ascend.npu_dsa_resident_lookup_rows_(
        selected,
        counts,
        request_states,
        workspace.state_token_indices[:requests],
        token_stride,
        requests,
    )
    gather_indices_seed = (
        workspace.state_token_indices[:requests].clone()
    )
    gather_output = workspace.old_slots_i16[:requests]
    scatter_target = seed_token_to_slot.clone()

    def aclnn_gather_only() -> None:
        torch.gather(
            token_to_slot.reshape(-1),
            0,
            gather_indices_seed.reshape(-1),
            out=gather_output.reshape(-1),
        )

    def aclnn_scatter_only() -> None:
        scatter_target.reshape(-1).scatter_(
            0,
            scatter_indices_seed.reshape(-1),
            scatter_values_seed.reshape(-1),
        )

    gather_samples = _measure_npu_ms(
        aclnn_gather_only,
        lambda: None,
        warmups,
        iterations,
    )
    scatter_samples = _measure_npu_ms(
        aclnn_scatter_only,
        lambda: None,
        warmups,
        iterations,
    )
    torch_samples = _measure_npu_ms(
        lambda: resident_plan(False),
        reset_inputs,
        warmups,
        iterations,
    )
    hybrid_samples = _measure_npu_ms(
        lambda: resident_plan(True),
        reset_inputs,
        warmups,
        iterations,
    )

    print(
        "resident sparse-cache benchmark: "
        f"topk={topk}, MTP={mtp}, requests={requests}, "
        f"hit_rate={hit_rate:.2%}, union={union_count}, "
        f"misses={union_count - hit_count}"
    )
    _summary("aclnn-gather", gather_samples)
    _summary("aclnn-scatter", scatter_samples)
    _summary("all-torch", torch_samples)
    _summary("hybrid-aiv", hybrid_samples)
    torch_mean = statistics.fmean(torch_samples)
    hybrid_mean = statistics.fmean(hybrid_samples)
    gather_scatter_mean = (
        statistics.fmean(gather_samples)
        + statistics.fmean(scatter_samples)
    )
    print(
        "ACLNN gather+scatter: "
        f"{gather_scatter_mean:.6f} ms "
        f"({gather_scatter_mean / hybrid_mean * 100:.2f}% of hybrid)"
    )
    print(
        "hybrid delta: "
        f"{hybrid_mean - torch_mean:+.6f} ms "
        f"({(hybrid_mean / torch_mean - 1) * 100:+.2f}%), "
        f"speedup={torch_mean / hybrid_mean:.3f}x"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--mtp", type=int, default=2)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--hit-rate", type=float, default=0.9)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmups", type=int, default=50)
    args = parser.parse_args()
    main(
        args.topk,
        args.mtp,
        args.requests,
        args.hit_rate,
        args.iterations,
        args.warmups,
    )
