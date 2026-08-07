import statistics
from dataclasses import fields

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sorted_cache import (
    INDEX_TOPK,
    SortedResidentState,
    SortedResidentWorkspace,
    allocate_sorted_resident_state,
    allocate_sorted_resident_workspace,
    debug_sorted_resident_finalize_only_,
    debug_sorted_resident_finalize_optimized_,
    debug_sorted_resident_update_only_,
    debug_sorted_resident_update_optimized_,
    prepare_resident_sharded_union_,
    prepare_resident_sharded_union_optimized_,
    resident_shard_count,
)
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def _load_optimized_sorted_resident_ops():
    if not enable_custom_op():
        pytest.fail("vllm-ascend custom operators could not be loaded")
    for name in (
        "npu_dsa_resident_sharded_union_optimized_",
        "npu_dsa_resident_sorted_finalize_optimized_",
        "npu_dsa_resident_sorted_update_optimized_",
    ):
        if not hasattr(torch.ops._C_ascend, name):
            pytest.fail(f"vllm_ascend_C does not contain {name}")


def _fill_resident_buffers(
    state: SortedResidentState,
    workspace: SortedResidentWorkspace,
) -> None:
    state.tokens.fill_(-1)
    state.slots.fill_(-1)
    state.counts.zero_()
    state.generations.fill_(-1)
    for field in fields(workspace):
        tensor = getattr(workspace, field.name)
        if "counts" in field.name:
            tensor.zero_()
        else:
            tensor.fill_(-1)


def _seed_resident_state(
    state: SortedResidentState,
    *,
    request_count: int,
    shard_count: int,
) -> None:
    # A dense slot prefix containing both hits and stale entries exercises
    # hit preservation, eviction reuse, miss allocation, and state merging.
    resident_tokens = list(range(512)) + list(range(5000, 5256))
    token_to_slot = {token: slot for slot, token in enumerate(resident_tokens)}
    for request in range(request_count):
        for shard in range(shard_count):
            tokens = [token for token in resident_tokens if token % shard_count == shard]
            slots = [token_to_slot[token] for token in tokens]
            count = len(tokens)
            state.tokens[request, shard, :count].copy_(torch.tensor(tokens, dtype=torch.int32, device="npu"))
            state.slots[request, shard, :count].copy_(torch.tensor(slots, dtype=torch.int16, device="npu"))
            state.counts[request, shard, 0] = count
        state.generations[request, 0] = request + 1


def _source(request_count: int, mtp: int) -> torch.Tensor:
    rows = []
    for request in range(request_count):
        for mtp_row in range(mtp):
            row = torch.arange(
                mtp_row * 1024,
                mtp_row * 1024 + INDEX_TOPK,
                dtype=torch.int32,
            )
            row = torch.roll(row, 97 * (request + 1) * (mtp_row + 1))
            # Exercise the signed-input and split-boundary masks as well as
            # the optimized power-of-two shard selection.
            row[-8:] = -1
            row[-16:-8] = 4096 + request
            rows.append(row)
    return torch.stack(rows).unsqueeze(1)


def _make_case(
    request_count: int,
    mtp: int,
    shard_count: int,
):
    device = torch.device("npu")
    capacity = mtp * INDEX_TOPK
    block_size = 128
    source_cpu = _source(request_count, mtp)
    values = source_cpu.to(device=device)
    boundaries = torch.full(
        (request_count * mtp,),
        3000,
        dtype=torch.int32,
        device=device,
    )
    row_requests = torch.arange(request_count, dtype=torch.int32, device=device).repeat_interleave(mtp)
    request_states = torch.arange(request_count, dtype=torch.int32, device=device)
    request_generations = torch.arange(1, request_count + 1, dtype=torch.int64, device=device)
    blocks_per_request = capacity // block_size
    block_rows = []
    for request in range(request_count):
        block_rows.append(torch.arange(blocks_per_request - 1, -1, -1, dtype=torch.int32) + request * 1000 + 17)
    block_table = torch.stack(block_rows).to(device=device)
    state = allocate_sorted_resident_state(
        request_count,
        request_count,
        mtp,
        device=device,
        shard_count=shard_count,
    )
    workspace = allocate_sorted_resident_workspace(
        request_count,
        mtp,
        device=device,
        shard_count=shard_count,
    )
    _fill_resident_buffers(state, workspace)
    _seed_resident_state(
        state,
        request_count=request_count,
        shard_count=shard_count,
    )
    return (
        source_cpu,
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        block_table,
        state,
        workspace,
        block_size,
    )


def _measure_pair_us(
    old_run,
    optimized_run,
    *,
    warmups: int,
    iterations: int,
    old_reset=None,
    optimized_reset=None,
):
    for _ in range(warmups):
        if old_reset is not None:
            old_reset()
        old_run()
        if optimized_reset is not None:
            optimized_reset()
        optimized_run()
    torch.npu.synchronize()

    old_events = []
    optimized_events = []
    for _ in range(iterations):
        old_start = torch.npu.Event(enable_timing=True)
        old_end = torch.npu.Event(enable_timing=True)
        optimized_start = torch.npu.Event(enable_timing=True)
        optimized_end = torch.npu.Event(enable_timing=True)
        if old_reset is not None:
            old_reset()
        old_start.record()
        old_run()
        old_end.record()
        if optimized_reset is not None:
            optimized_reset()
        optimized_start.record()
        optimized_run()
        optimized_end.record()
        old_events.append((old_start, old_end))
        optimized_events.append((optimized_start, optimized_end))
    torch.npu.synchronize()
    old_us = [start.elapsed_time(end) * 1000 for start, end in old_events]
    optimized_us = [start.elapsed_time(end) * 1000 for start, end in optimized_events]
    return old_us, optimized_us


def _assert_union_equal(
    old_workspace: SortedResidentWorkspace,
    optimized_workspace: SortedResidentWorkspace,
) -> None:
    assert torch.equal(
        old_workspace.shard_mapping.cpu(),
        optimized_workspace.shard_mapping.cpu(),
    )
    assert torch.equal(
        old_workspace.shard_counts[:, :, :4].cpu(),
        optimized_workspace.shard_counts[:, :, :4].cpu(),
    )
    counts = old_workspace.shard_counts.cpu()
    assert int(counts[:, :, 1].sum()) > 0
    assert int(counts[:, :, 2].sum()) > 0
    observed_hit = False
    for request in range(counts.shape[0]):
        for shard in range(counts.shape[1]):
            current_count = int(counts[request, shard, 0])
            miss_count = int(counts[request, shard, 1])
            evictable_count = int(counts[request, shard, 2])
            for name, count in (
                ("shard_packed", current_count),
                ("prior_slots", current_count),
                ("shard_miss_tokens", miss_count),
                ("shard_miss_positions", miss_count),
                ("shard_evictable_slots", evictable_count),
            ):
                old = getattr(old_workspace, name)[request, shard, :count]
                optimized = getattr(optimized_workspace, name)[request, shard, :count]
                assert torch.equal(old.cpu(), optimized.cpu()), (
                    name,
                    request,
                    shard,
                )
            if current_count:
                observed_hit |= bool(torch.any(old_workspace.prior_slots[request, shard, :current_count].cpu() >= 0))
    assert observed_hit


def _assert_finalize_equal(
    old_workspace: SortedResidentWorkspace,
    optimized_workspace: SortedResidentWorkspace,
) -> None:
    assert torch.equal(
        old_workspace.shard_counts[:, :, :5].cpu(),
        optimized_workspace.shard_counts[:, :, :5].cpu(),
    )
    old_counts = old_workspace.miss_counts[:, 0].cpu()
    optimized_counts = optimized_workspace.miss_counts[:, 0].cpu()
    assert torch.equal(old_counts, optimized_counts)
    current_counts = old_workspace.shard_counts[:, :, 0].cpu()
    for request in range(old_counts.numel()):
        miss_count = int(old_counts[request])
        assert torch.equal(
            old_workspace.miss_tokens[request, :miss_count].cpu(),
            optimized_workspace.miss_tokens[request, :miss_count].cpu(),
        )
        assert torch.equal(
            old_workspace.target_slots[request, :miss_count].cpu(),
            optimized_workspace.target_slots[request, :miss_count].cpu(),
        )
        for shard in range(current_counts.shape[1]):
            count = int(current_counts[request, shard])
            assert torch.equal(
                old_workspace.prior_slots[request, shard, :count].cpu(),
                optimized_workspace.prior_slots[request, shard, :count].cpu(),
            )


def _assert_update_equal(
    old_values: torch.Tensor,
    optimized_values: torch.Tensor,
    old_state: SortedResidentState,
    optimized_state: SortedResidentState,
) -> None:
    assert torch.equal(old_values.cpu(), optimized_values.cpu())
    for name in ("tokens", "slots", "counts", "generations"):
        assert torch.equal(
            getattr(old_state, name).cpu(),
            getattr(optimized_state, name).cpu(),
        ), name


def _print_latency(label: str, old_us: list[float], optimized_us: list[float]):
    old_mean = statistics.fmean(old_us)
    optimized_mean = statistics.fmean(optimized_us)
    old_p50 = statistics.median(old_us)
    optimized_p50 = statistics.median(optimized_us)
    print(
        f"[resident-optimized-benchmark] kernel={label} "
        f"old_mean_us={old_mean:.3f} optimized_mean_us={optimized_mean:.3f} "
        f"old_p50_us={old_p50:.3f} optimized_p50_us={optimized_p50:.3f} "
        f"speedup={old_mean / optimized_mean:.3f}x"
    )


@pytest.mark.parametrize(
    "shards_per_row,request_count",
    [(2, 6), (4, 6), (4, 41)],
    ids=[
        "two-shards-per-row-six-requests",
        "four-shards-per-row-six-requests",
        "four-shards-per-row-forty-one-requests",
    ],
)
def test_optimized_resident_kernels_match_and_report_latency(
    shards_per_row,
    request_count,
):
    mtp = 2
    shard_count = resident_shard_count(mtp, shards_per_row)
    old = _make_case(request_count, mtp, shard_count)
    optimized = _make_case(request_count, mtp, shard_count)
    (
        _,
        old_values,
        old_boundaries,
        old_row_requests,
        old_request_states,
        old_request_generations,
        old_block_table,
        old_state,
        old_workspace,
        block_size,
    ) = old
    (
        _,
        optimized_values,
        optimized_boundaries,
        optimized_row_requests,
        optimized_request_states,
        optimized_request_generations,
        optimized_block_table,
        optimized_state,
        optimized_workspace,
        _,
    ) = optimized

    def old_union():
        prepare_resident_sharded_union_(
            old_values,
            old_boundaries,
            old_row_requests,
            old_request_states,
            old_request_generations,
            old_state,
            old_workspace,
            mtp=mtp,
        )

    def optimized_union():
        prepare_resident_sharded_union_optimized_(
            optimized_values,
            optimized_boundaries,
            optimized_row_requests,
            optimized_request_states,
            optimized_request_generations,
            optimized_state,
            optimized_workspace,
            mtp=mtp,
        )

    def old_finalize():
        debug_sorted_resident_finalize_only_(
            old_block_table,
            old_workspace,
            block_size=block_size,
        )

    def optimized_finalize():
        debug_sorted_resident_finalize_optimized_(
            optimized_block_table,
            optimized_workspace,
            block_size=block_size,
        )

    def old_update():
        debug_sorted_resident_update_only_(
            old_values,
            old_request_states,
            old_request_generations,
            old_state,
            old_workspace,
        )

    def optimized_update():
        debug_sorted_resident_update_optimized_(
            optimized_values,
            optimized_request_states,
            optimized_request_generations,
            optimized_state,
            optimized_workspace,
        )

    warmups = 10
    iterations = 50
    union_old_us, union_optimized_us = _measure_pair_us(
        old_union,
        optimized_union,
        warmups=warmups,
        iterations=iterations,
    )
    _assert_union_equal(old_workspace, optimized_workspace)

    finalize_old_us, finalize_optimized_us = _measure_pair_us(
        old_finalize,
        optimized_finalize,
        warmups=warmups,
        iterations=iterations,
    )
    _assert_finalize_equal(old_workspace, optimized_workspace)

    old_values_seed = old_values.clone()
    optimized_values_seed = optimized_values.clone()
    old_state_seed = tuple(
        getattr(old_state, name)[:request_count].clone() for name in ("tokens", "slots", "counts", "generations")
    )
    optimized_state_seed = tuple(
        getattr(optimized_state, name)[:request_count].clone() for name in ("tokens", "slots", "counts", "generations")
    )

    def reset_old_update():
        old_values.copy_(old_values_seed)
        for name, seed in zip(
            ("tokens", "slots", "counts", "generations"),
            old_state_seed,
        ):
            getattr(old_state, name)[:request_count].copy_(seed)

    def reset_optimized_update():
        optimized_values.copy_(optimized_values_seed)
        for name, seed in zip(
            ("tokens", "slots", "counts", "generations"),
            optimized_state_seed,
        ):
            getattr(optimized_state, name)[:request_count].copy_(seed)

    update_old_us, update_optimized_us = _measure_pair_us(
        old_update,
        optimized_update,
        warmups=warmups,
        iterations=iterations,
        old_reset=reset_old_update,
        optimized_reset=reset_optimized_update,
    )
    _assert_update_equal(
        old_values,
        optimized_values,
        old_state,
        optimized_state,
    )

    print(
        f"\n[resident-optimized-benchmark] requests={request_count} "
        f"mtp={mtp} shards_per_row={shards_per_row} "
        f"total_shards={shard_count}"
    )
    _print_latency("sharded_union", union_old_us, union_optimized_us)
    _print_latency("sorted_finalize", finalize_old_us, finalize_optimized_us)
    _print_latency("sorted_update", update_old_us, update_optimized_us)
