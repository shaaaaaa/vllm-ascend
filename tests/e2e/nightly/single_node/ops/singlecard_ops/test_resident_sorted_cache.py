import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sorted_cache import (
    RESIDENT_FINALIZE_DEBUG_INTS,
    RESIDENT_READ_PROBE_DEBUG_INTS,
    allocate_sorted_resident_state,
    allocate_sorted_resident_workspace,
    debug_sorted_resident_finalize_only_,
    prepare_resident_sharded_union_,
    prepare_sorted_resident_cache_,
    prepare_sorted_resident_cache_fused_,
    prepare_sorted_resident_cache_no_remap_,
    probe_sorted_resident_reads_,
    remap_sorted_resident_cache_,
    resident_shard_count,
)
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def _load_sorted_resident_ops():
    if not enable_custom_op():
        pytest.fail("vllm-ascend custom operators could not be loaded")
    for name in (
        "npu_dsa_resident_sharded_union_",
        "npu_dsa_resident_sorted_plan_",
        "npu_dsa_resident_sorted_plan_no_remap_",
        "npu_dsa_resident_sorted_remap_",
        "npu_dsa_resident_sorted_read_probe_",
        "npu_dsa_resident_sorted_finalize_debug_",
    ):
        if not hasattr(torch.ops._C_ascend, name):
            pytest.fail(f"vllm_ascend_C does not contain {name}")


def _source(mtp: int, offset: int, requests: int = 1) -> torch.Tensor:
    rows = torch.stack(
        tuple(
            torch.roll(
                torch.arange(
                    offset + row * 1024,
                    offset + row * 1024 + 2048,
                    dtype=torch.int32,
                ),
                137 * (row + 1),
            )
            for row in range(mtp)
        )
    )
    return rows.repeat(requests, 1).unsqueeze(1)


def _expected_shards(
    source: torch.Tensor,
    boundaries: torch.Tensor,
    requests: int,
    mtp: int,
    shard_count: int,
) -> list[list[list[int]]]:
    rows = source.reshape(requests, mtp, 2048)
    result = []
    for request in range(requests):
        union = set()
        for row in range(mtp):
            boundary = int(boundaries[request * mtp + row])
            union.update(int(token) for token in rows[request, row].tolist() if 0 <= token < boundary)
        result.append(
            [sorted(token for token in union if token % shard_count == shard) for shard in range(shard_count)]
        )
    return result


def _assert_union_outputs(
    source: torch.Tensor,
    boundaries: torch.Tensor,
    workspace,
    *,
    mtp: int,
) -> list[list[list[int]]]:
    requests = workspace.shard_packed.shape[0]
    shard_count = workspace.shard_packed.shape[1]
    expected = _expected_shards(source, boundaries, requests, mtp, shard_count)
    packed = workspace.shard_packed.cpu()
    mapping = workspace.shard_mapping.cpu()
    counts = workspace.shard_counts[:, :, 0].cpu()
    source_flat = source.reshape(requests, -1)
    for request in range(requests):
        for shard in range(shard_count):
            count = int(counts[request, shard])
            assert packed[request, shard, :count].tolist() == expected[request][shard]
        for position, token in enumerate(source_flat[request].tolist()):
            row = position // 2048
            boundary = int(boundaries[request * mtp + row])
            shard = token % shard_count
            rank = int(mapping[request, shard, position])
            if 0 <= token < boundary:
                assert rank >= 0
                assert int(packed[request, shard, rank]) == token
            else:
                assert rank < 0
    return expected


def _seed_resident_state(
    state,
    *,
    state_index: int,
    generation: int,
    shards: list[list[int]],
    resident: dict[int, int],
) -> None:
    state.counts[state_index].zero_()
    for shard, tokens in enumerate(shards):
        resident_tokens = [token for token in tokens if token in resident]
        count = len(resident_tokens)
        if count == 0:
            continue
        state.tokens[state_index, shard, :count].copy_(
            torch.tensor(resident_tokens, dtype=torch.int32, device="npu")
        )
        state.slots[state_index, shard, :count].copy_(
            torch.tensor(
                [resident[token] for token in resident_tokens],
                dtype=torch.int16,
                device="npu",
            )
        )
        state.counts[state_index, shard, 0] = count
    state.generations[state_index, 0] = generation


def _print_empty_state_finalize_debug(
    *,
    mtp: int,
    capacity: int,
    counts: torch.Tensor,
    prior_before: list[torch.Tensor],
    workspace,
) -> None:
    prior_after = workspace.prior_slots[0].cpu()
    overwritten = workspace.overwritten_slots[0].cpu()
    miss_count = int(workspace.miss_counts[0, 0].cpu())
    active_after = torch.cat(
        [
            prior_after[shard, : int(counts[shard])]
            for shard in range(counts.numel())
            if int(counts[shard]) > 0
        ]
    )
    active_before = torch.cat(prior_before)
    valid_after = (active_after >= 0) & (active_after < capacity)
    invalid_after = active_after >= capacity
    still_negative = active_after < 0
    unique_valid = torch.unique(active_after[valid_after]).numel()
    treated_as_hit = torch.zeros_like(valid_after)
    treated_as_hit[valid_after] = (
        overwritten[active_after[valid_after].to(torch.int64)] == 0
    )
    sample_indices = torch.nonzero(
        treated_as_hit | invalid_after | still_negative,
        as_tuple=False,
    ).flatten()[:32]
    sample = [
        (int(index), int(active_after[index]))
        for index in sample_indices
    ]
    print(
        "\n[resident-finalize-debug]"
        f" mtp={mtp}"
        f" active={active_before.numel()}"
        f" pre_negative={int((active_before < 0).sum())}"
        f" pre_nonnegative={int((active_before >= 0).sum())}"
        f" miss_count={miss_count}"
        f" post_valid={int(valid_after.sum())}"
        f" post_invalid={int(invalid_after.sum())}"
        f" post_negative={int(still_negative.sum())}"
        f" post_unique_valid={unique_valid}"
        f" overwritten={int(overwritten.sum())}"
        f" treated_as_hit={int(treated_as_hit.sum())}"
        f" suspicious_sample={sample}"
    )


@pytest.mark.parametrize("mtp", [1, 2])
def test_resident_union_is_sorted_per_fixed_token_shard(mtp):
    requests = 2
    source = _source(mtp, 0, requests)
    boundary = 1948 if mtp == 1 else 2972
    boundaries = boundary - 73 * torch.arange(
        requests * mtp, dtype=torch.int32
    )
    row_requests = torch.arange(requests, dtype=torch.int32).repeat_interleave(mtp)
    workspace = allocate_sorted_resident_workspace(
        requests,
        mtp,
        device=torch.device("npu"),
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    request_states = torch.arange(requests, dtype=torch.int32, device="npu")
    request_generations = torch.ones(requests, dtype=torch.int64, device="npu")

    prepare_resident_sharded_union_(
        source.npu(),
        boundaries.npu(),
        row_requests.npu(),
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    torch.npu.synchronize()
    _assert_union_outputs(source, boundaries, workspace, mtp=mtp)


@pytest.mark.parametrize("mtp", [1, 2])
def test_resident_union_zero_boundary_is_empty(mtp):
    requests = 1
    source = _source(mtp, 0)
    boundaries = torch.zeros(requests * mtp, dtype=torch.int32)
    row_requests = torch.zeros(requests * mtp, dtype=torch.int32)
    workspace = allocate_sorted_resident_workspace(
        requests,
        mtp,
        device=torch.device("npu"),
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    request_states = torch.zeros(requests, dtype=torch.int32, device="npu")
    request_generations = torch.ones(requests, dtype=torch.int64, device="npu")

    prepare_resident_sharded_union_(
        source.npu(),
        boundaries.npu(),
        row_requests.npu(),
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    torch.npu.synchronize()
    assert torch.all(workspace.shard_counts[:, :, 0].cpu() == 0)
    assert torch.all(workspace.shard_mapping.cpu() < 0)


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_zero_boundary_full_path_is_noop(mtp):
    """Exercise every dynamic zero-count transfer in the production path."""
    requests = 1
    capacity = mtp * 2048
    block_size = 128
    source = _source(mtp, 0)
    values = source.npu()
    boundaries = torch.zeros(requests * mtp, dtype=torch.int32, device="npu")
    row_requests = torch.zeros(requests * mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(requests, dtype=torch.int32, device="npu")
    request_generations = torch.ones(requests, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    block_table = torch.arange(
        capacity // block_size, dtype=torch.int32, device="npu"
    ).reshape(requests, -1)

    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    prepare_sorted_resident_cache_(
        values,
        block_table,
        request_states,
        request_generations,
        state,
        workspace,
        block_size=block_size,
    )
    torch.npu.synchronize()

    assert torch.all(workspace.shard_counts[:, :, 0].cpu() == 0)
    assert int(workspace.miss_counts[0, 0].cpu()) == 0
    assert torch.all(workspace.overwritten_slots[0].cpu() == 0)
    assert torch.all(state.counts[0, :, 0].cpu() == 0)
    assert torch.equal(values.cpu(), source)


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_zero_boundary_finalize_only(mtp):
    """Isolate zero-count finalize from the following update/remap kernel."""
    requests = 1
    capacity = mtp * 2048
    block_size = 128
    source = _source(mtp, 0)
    values = source.npu()
    boundaries = torch.zeros(requests * mtp, dtype=torch.int32, device="npu")
    row_requests = torch.zeros(requests * mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(requests, dtype=torch.int32, device="npu")
    request_generations = torch.ones(requests, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    block_table = torch.arange(
        capacity // block_size, dtype=torch.int32, device="npu"
    ).reshape(requests, -1)

    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    torch.npu.synchronize()
    assert torch.all(workspace.shard_counts[:, :, 0].cpu() == 0)

    debug_sorted_resident_finalize_only_(
        block_table,
        workspace,
        block_size=block_size,
    )
    torch.npu.synchronize()

    assert int(workspace.miss_counts[0, 0].cpu()) == 0
    assert torch.all(workspace.overwritten_slots[0].cpu() == 0)


@pytest.mark.parametrize("mtp", [1, 2])
def test_fused_union_intersection_and_generation_invalidation(mtp):
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = _source(mtp, 0)
    boundaries = torch.full((mtp,), 1600, dtype=torch.int32)
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.full((1,), 7, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    expected_shards = _expected_shards(
        source, boundaries, requests, mtp, shard_count
    )[0]
    union_tokens = [token for shard in expected_shards for token in shard]
    resident = {
        token: (index * 37) % capacity
        for index, token in enumerate(union_tokens[::5])
    }
    _seed_resident_state(
        state,
        state_index=0,
        generation=7,
        shards=expected_shards,
        resident=resident,
    )

    prepare_resident_sharded_union_(
        source.npu(),
        boundaries.npu(),
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    torch.npu.synchronize()

    counts = workspace.shard_counts[0, :, 0].cpu()
    packed = workspace.shard_packed[0].cpu()
    prior = workspace.prior_slots[0].cpu()
    for shard in range(shard_count):
        count = int(counts[shard])
        tokens = packed[shard, :count].tolist()
        assert tokens == expected_shards[shard]
        assert prior[shard, :count].tolist() == [
            resident.get(token, -1) for token in tokens
        ]

    request_generations.fill_(8)
    prepare_resident_sharded_union_(
        source.npu(),
        boundaries.npu(),
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    torch.npu.synchronize()
    for shard in range(shard_count):
        count = int(workspace.shard_counts[0, shard, 0].cpu())
        assert torch.all(workspace.prior_slots[0, shard, :count].cpu() == -1)
    assert torch.all(state.counts[0, :, 0].cpu() == 0)


@pytest.mark.parametrize("mtp", [1, 2])
def test_kernel_read_probe_matches_full_capacity_union_output(mtp):
    """Capture the exact GM values observed by an AIV before finalization."""
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = (
        torch.arange(capacity, dtype=torch.int32)
        .mul(shard_count)
        .reshape(mtp, 1, 2048)
    )
    values = source.npu()
    boundaries = torch.full(
        (mtp,),
        int(source.max()) + 1,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.ones(1, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )

    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    debug_info = torch.full(
        (requests, RESIDENT_READ_PROBE_DEBUG_INTS),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    prior_readback = torch.full_like(workspace.prior_slots, -12345)
    probe_sorted_resident_reads_(
        workspace,
        debug_info,
        prior_readback,
    )
    torch.npu.synchronize()

    host_counts = workspace.shard_counts[0, :, 0].cpu()
    host_prior = workspace.prior_slots[0].cpu()
    probe_debug = debug_info[0].cpu()
    probe_prior = prior_readback[0].cpu()
    assert int(probe_debug[0]) == 0x52535031
    assert int(probe_debug[1]) == shard_count
    assert int(probe_debug[2]) == capacity
    for shard in range(shard_count):
        base = 4 + shard * 7
        raw_count = int(probe_debug[base])
        fresh_count = int(probe_debug[base + 1])
        bulk_count = int(probe_debug[base + 2])
        first_slot = int(probe_debug[base + 3])
        last_slot = int(probe_debug[base + 4])
        negative = int(probe_debug[base + 5])
        nonnegative = int(probe_debug[base + 6])
        expected_count = int(host_counts[shard])
        print(
            "\n[resident-kernel-read-probe]"
            f" mtp={mtp}"
            f" shard={shard}"
            f" host_count={expected_count}"
            f" raw_count={raw_count}"
            f" fresh_count={fresh_count}"
            f" bulk_count={bulk_count}"
            f" first_slot={first_slot}"
            f" last_slot={last_slot}"
            f" negative={negative}"
            f" nonnegative={nonnegative}"
        )
        assert fresh_count == expected_count
        assert bulk_count == expected_count
        assert torch.equal(
            probe_prior[shard, :expected_count],
            host_prior[shard, :expected_count],
        )
        assert torch.all(
            probe_prior[shard, expected_count:] == -12345
        )
        if expected_count:
            assert first_slot == int(host_prior[shard, 0])
            assert last_slot == int(
                host_prior[shard, expected_count - 1]
            )
            assert negative == int(
                (host_prior[shard, :expected_count] < 0).sum()
            )
            assert nonnegative == int(
                (host_prior[shard, :expected_count] >= 0).sum()
            )


@pytest.mark.parametrize(
    "mtp,debug_stage",
    [
        pytest.param(mtp, stage, id=f"mtp{mtp}-stage{stage}")
        for mtp in (1, 2)
        for stage in range(1, 12)
    ],
)
def test_finalize_kernel_internal_stage_at_full_capacity(
    mtp,
    debug_stage,
):
    """Stop inside finalize and validate the first observable bad stage."""
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = (
        torch.arange(capacity, dtype=torch.int32)
        .mul(shard_count)
        .reshape(mtp, 1, 2048)
    )
    values = source.npu()
    boundaries = torch.full(
        (mtp,),
        int(source.max()) + 1,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.ones(1, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    block_table = torch.arange(
        capacity // 128, dtype=torch.int32, device="npu"
    ).reshape(1, -1)
    debug_info = torch.full(
        (requests, RESIDENT_FINALIZE_DEBUG_INTS),
        -1,
        dtype=torch.int32,
        device="npu",
    )

    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    debug_sorted_resident_finalize_only_(
        block_table,
        workspace,
        block_size=128,
        debug_info=debug_info,
        debug_stage=debug_stage,
    )
    torch.npu.synchronize()

    debug = debug_info[0].cpu()
    packed = workspace.shard_packed[0, 0, :capacity].cpu()
    fields = {
        "stage": int(debug[1]),
        "packed_end": int(debug[4]),
        "protected_count": int(debug[5]),
        "free_count": int(debug[6]),
        "miss_count": int(debug[7]),
        "first_slot": int(debug[8]),
        "last_slot": int(debug[9]),
        "first_token": int(debug[10]),
        "last_token": int(debug[11]),
        "first_target": int(debug[12]),
        "last_target": int(debug[13]),
    }
    print(
        "\n[resident-finalize-internal-stage]"
        f" mtp={mtp}"
        f" requested_stage={debug_stage}"
        f" fields={fields}"
    )
    assert int(debug[0]) == 0x52534631
    assert fields["stage"] == debug_stage
    assert int(debug[2]) == shard_count
    assert int(debug[3]) == capacity
    assert fields["packed_end"] == capacity

    if debug_stage >= 2:
        assert fields["protected_count"] == 0
    if debug_stage == 3:
        assert torch.all(
            workspace.prior_slots[0, 0, :capacity].cpu() == 0
        )
    if debug_stage >= 4:
        assert fields["free_count"] == capacity
    if debug_stage == 5:
        assert (
            workspace.prior_slots[0, 0, :capacity].cpu().tolist()
            == list(range(capacity))
        )
    if debug_stage >= 6:
        assert fields["miss_count"] == capacity
        assert fields["first_slot"] == 0
        assert fields["last_slot"] == capacity - 1
        assert fields["first_token"] == int(packed[0])
        assert fields["last_token"] == int(packed[-1])
        assert fields["first_target"] == 0
        assert fields["last_target"] == capacity - 1
    if debug_stage >= 7:
        assert (
            workspace.prior_slots[0, 0, :capacity].cpu().tolist()
            == list(range(capacity))
        )
    if debug_stage >= 8:
        assert torch.all(workspace.overwritten_slots[0].cpu() == 1)
    if debug_stage >= 9:
        assert torch.equal(
            workspace.miss_tokens[0, :capacity].cpu(),
            packed,
        )
    if debug_stage >= 10:
        assert (
            workspace.target_slots[0, :capacity].cpu().tolist()
            == list(range(capacity))
        )
    if debug_stage >= 11:
        assert int(workspace.miss_counts[0, 0].cpu()) == capacity


@pytest.mark.parametrize("mtp", [1, 2])
def test_finalize_kernel_boundary_at_full_shard_capacity(mtp):
    """Validate finalize outputs before state update/remap is launched."""
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = (
        torch.arange(capacity, dtype=torch.int32)
        .mul(shard_count)
        .reshape(mtp, 1, 2048)
    )
    values = source.npu()
    boundaries = torch.full(
        (mtp,),
        int(source.max()) + 1,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.ones(1, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    block_table = torch.arange(
        capacity // 128, dtype=torch.int32, device="npu"
    ).reshape(1, -1)

    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    # Deliberately do not synchronize between the two kernels. This matches
    # the production stream-ordered launch and the established staged-union
    # implementation.
    debug_sorted_resident_finalize_only_(
        block_table,
        workspace,
        block_size=128,
    )
    torch.npu.synchronize()

    miss_count = int(workspace.miss_counts[0, 0].cpu())
    prior = workspace.prior_slots[0, 0, :capacity].cpu()
    overwritten = workspace.overwritten_slots[0].cpu()
    misses = workspace.miss_tokens[0, :miss_count].cpu()
    targets = workspace.target_slots[0, :miss_count].cpu()
    packed = workspace.shard_packed[0, 0, :capacity].cpu()
    print(
        "\n[resident-finalize-kernel-boundary]"
        f" mtp={mtp}"
        f" miss_count={miss_count}"
        f" prior_negative={int((prior < 0).sum())}"
        f" prior_unique={torch.unique(prior).numel()}"
        f" overwritten={int(overwritten.sum())}"
        f" miss_first={int(misses[0]) if miss_count else -1}"
        f" miss_last={int(misses[-1]) if miss_count else -1}"
        f" target_first={int(targets[0]) if miss_count else -1}"
        f" target_last={int(targets[-1]) if miss_count else -1}"
    )
    assert miss_count == capacity
    assert prior.tolist() == list(range(capacity))
    assert torch.all(overwritten == 1)
    assert torch.equal(misses, packed)
    assert targets.tolist() == list(range(capacity))
    # The isolated boundary op must not launch update/remap.
    assert values.cpu().reshape(-1).tolist() == source.reshape(-1).tolist()
    assert torch.all(state.counts[:, :, 0].cpu() == 0)
    assert torch.all(state.generations[:, 0].cpu() == -1)


@pytest.mark.parametrize("mtp", [1, 2])
def test_int16_mapping_and_uint8_overwrite_at_full_shard_capacity(mtp):
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = (
        torch.arange(capacity, dtype=torch.int32)
        .mul(shard_count)
        .reshape(mtp, 1, 2048)
    )
    values = source.npu()
    boundaries = torch.full(
        (mtp,),
        int(source.max()) + 1,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.ones(1, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    block_table = torch.arange(
        capacity // 128, dtype=torch.int32, device="npu"
    ).reshape(1, -1)

    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    torch.npu.synchronize()
    assert workspace.shard_mapping.dtype == torch.int16
    assert int(workspace.shard_counts[0, 0, 0].cpu()) == capacity
    assert torch.all(workspace.shard_counts[0, 1:, 0].cpu() == 0)
    assert workspace.shard_mapping[0, 0].cpu().tolist() == list(range(capacity))
    counts = workspace.shard_counts[0, :, 0].cpu()
    prior_before = [
        workspace.prior_slots[0, shard, : int(counts[shard])].cpu().clone()
        for shard in range(shard_count)
        if int(counts[shard]) > 0
    ]
    print(
        "\n[resident-finalize-debug-pre]"
        f" mtp={mtp}"
        f" shard_counts={counts.tolist()}"
        f" active={sum(prior.numel() for prior in prior_before)}"
        f" negative={sum(int((prior < 0).sum()) for prior in prior_before)}"
        f" nonnegative={sum(int((prior >= 0).sum()) for prior in prior_before)}"
    )
    assert sum(prior.numel() for prior in prior_before) == capacity
    assert all(torch.all(prior == -1) for prior in prior_before)

    prepare_sorted_resident_cache_(
        values,
        block_table,
        request_states,
        request_generations,
        state,
        workspace,
        block_size=128,
    )
    torch.npu.synchronize()
    _print_empty_state_finalize_debug(
        mtp=mtp,
        capacity=capacity,
        counts=counts,
        prior_before=prior_before,
        workspace=workspace,
    )
    assert int(workspace.miss_counts[0, 0].cpu()) == capacity
    assert workspace.overwritten_slots.dtype == torch.uint8
    assert torch.all(workspace.overwritten_slots[0].cpu() == 1)
    assert values.reshape(-1).cpu().tolist() == list(range(capacity))
    assert int(state.counts[0, 0, 0].cpu()) == capacity


def _reference_step(
    shards: list[list[int]],
    resident: dict[int, int],
    capacity: int,
) -> tuple[dict[int, int], list[int], list[int]]:
    protected = {resident[token] for shard in shards for token in shard if token in resident}
    free_slots = [slot for slot in range(capacity) if slot not in protected]
    assigned = {}
    misses = []
    miss_slots = []
    free_index = 0
    for shard in shards:
        for token in shard:
            if token in resident:
                assigned[token] = resident[token]
            else:
                slot = free_slots[free_index]
                free_index += 1
                assigned[token] = slot
                misses.append(token)
                miss_slots.append(slot)
    overwritten = set(miss_slots)
    updated = {token: slot for token, slot in resident.items() if slot not in overwritten}
    updated.update({token: assigned[token] for token in misses})
    return updated, misses, miss_slots


def _state_dict(state, state_index: int, shard_count: int) -> dict[int, int]:
    counts = state.counts[state_index, :, 0].cpu()
    tokens = state.tokens[state_index].cpu()
    slots = state.slots[state_index].cpu()
    result = {}
    for shard in range(shard_count):
        count = int(counts[shard])
        shard_tokens = tokens[shard, :count].tolist()
        assert shard_tokens == sorted(shard_tokens)
        assert all(token % shard_count == shard for token in shard_tokens)
        result.update(
            zip(
                shard_tokens,
                slots[shard, :count].tolist(),
                strict=True,
            )
        )
    return result


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_matches_three_step_reference(mtp):
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    block_size = 128
    boundary = 100_000
    workspace = allocate_sorted_resident_workspace(
        requests,
        mtp,
        device=torch.device("npu"),
    )
    state = allocate_sorted_resident_state(
        requests,
        requests,
        mtp,
        device=torch.device("npu"),
    )
    row_requests = torch.zeros(requests * mtp, dtype=torch.int32, device="npu")
    boundaries = torch.full(
        (requests * mtp,),
        boundary,
        dtype=torch.int32,
        device="npu",
    )
    request_states = torch.zeros(requests, dtype=torch.int32, device="npu")
    request_generations = torch.ones(requests, dtype=torch.int64, device="npu")
    blocks_per_request = capacity // block_size
    block_table = torch.arange(
        requests * blocks_per_request,
        dtype=torch.int32,
        device="npu",
    ).reshape(requests, blocks_per_request)

    reference: dict[int, int] = {}
    for offset in (0, 100, 0):
        source = _source(mtp, offset)
        values = source.npu()
        prepare_resident_sharded_union_(
            values,
            boundaries,
            row_requests,
            request_states,
            request_generations,
            state,
            workspace,
            mtp=mtp,
        )
        expected_shards = _expected_shards(
            source,
            boundaries.cpu(),
            requests,
            mtp,
            shard_count,
        )[0]
        reference, expected_misses, expected_slots = _reference_step(expected_shards, reference, capacity)
        prepare_sorted_resident_cache_(
            values,
            block_table,
            request_states,
            request_generations,
            state,
            workspace,
            block_size=block_size,
        )
        torch.npu.synchronize()

        miss_count = int(workspace.miss_counts[0, 0].cpu())
        assert workspace.miss_tokens[0, :miss_count].cpu().tolist() == expected_misses
        assert workspace.target_slots[0, :miss_count].cpu().tolist() == expected_slots
        overwritten = workspace.overwritten_slots[0].cpu()
        assert overwritten.dtype == torch.uint8
        assert int(overwritten.max()) <= 1
        assert int(overwritten.sum()) == miss_count
        assert _state_dict(state, 0, shard_count) == reference
        remapped = values.reshape(-1).cpu().tolist()
        original = source.reshape(-1).tolist()
        for position, token in enumerate(original):
            expected = reference[token] if token < boundary else token
            assert remapped[position] == expected


@pytest.mark.parametrize("mtp", [1, 2])
def test_split_plan_preserves_topk_until_standalone_remap(mtp):
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = _source(mtp, 0)
    values = source.npu()
    boundaries = torch.full(
        (mtp,),
        100_000,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.ones(1, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    block_table = torch.arange(
        capacity // 128,
        dtype=torch.int32,
        device="npu",
    ).reshape(1, -1)

    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    expected_shards = _expected_shards(
        source,
        boundaries.cpu(),
        requests,
        mtp,
        shard_count,
    )[0]
    reference, expected_misses, expected_slots = _reference_step(
        expected_shards, {}, capacity
    )

    prepare_sorted_resident_cache_no_remap_(
        values,
        block_table,
        request_states,
        request_generations,
        state,
        workspace,
        block_size=128,
    )
    torch.npu.synchronize()
    assert values.reshape(-1).cpu().tolist() == source.reshape(-1).tolist()
    miss_count = int(workspace.miss_counts[0, 0].cpu())
    assert workspace.miss_tokens[0, :miss_count].cpu().tolist() == expected_misses
    assert workspace.target_slots[0, :miss_count].cpu().tolist() == expected_slots
    assert _state_dict(state, 0, shard_count) == reference

    remap_sorted_resident_cache_(values, workspace)
    torch.npu.synchronize()
    assert values.reshape(-1).cpu().tolist() == [
        reference[token] for token in source.reshape(-1).tolist()
    ]


@pytest.mark.parametrize("mtp", [1, 2])
def test_fused_remap_bisect_loop_load_quarter_does_not_crash(mtp):
    """Exercise count lookup and GM-to-UB loads in the fused remap loop."""
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = _source(mtp, 0)
    values = source.npu()
    boundaries = torch.full(
        (mtp,),
        100_000,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.ones(1, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    block_table = torch.arange(
        capacity // 128,
        dtype=torch.int32,
        device="npu",
    ).reshape(1, -1)

    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    expected_shards = _expected_shards(
        source,
        boundaries.cpu(),
        requests,
        mtp,
        shard_count,
    )[0]
    reference, expected_misses, expected_slots = _reference_step(
        expected_shards, {}, capacity
    )

    prepare_sorted_resident_cache_fused_(
        values,
        block_table,
        request_states,
        request_generations,
        state,
        workspace,
        block_size=128,
    )
    torch.npu.synchronize()

    # The compile-time bisect keeps only count lookup and the two GM-to-UB
    # loads in each iteration. It removes all vector work and final writeback,
    # so every position retains its original token.
    assert values.reshape(-1).cpu().tolist() == source.reshape(-1).tolist()
    miss_count = int(workspace.miss_counts[0, 0].cpu())
    assert workspace.miss_tokens[0, :miss_count].cpu().tolist() == expected_misses
    assert workspace.target_slots[0, :miss_count].cpu().tolist() == expected_slots
    assert _state_dict(state, 0, shard_count) == reference


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_all_hit_emits_zero_misses(mtp):
    """A non-empty resident step must skip zero-length miss writebacks."""
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    block_size = 128
    source = _source(mtp, 0)
    boundaries = torch.full(
        (requests * mtp,), 100_000, dtype=torch.int32, device="npu"
    )
    row_requests = torch.zeros(requests * mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(requests, dtype=torch.int32, device="npu")
    request_generations = torch.ones(requests, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    block_table = torch.arange(
        capacity // block_size, dtype=torch.int32, device="npu"
    ).reshape(requests, -1)

    for step in range(2):
        values = source.npu()
        prepare_resident_sharded_union_(
            values,
            boundaries,
            row_requests,
            request_states,
            request_generations,
            state,
            workspace,
            mtp=mtp,
        )
        prepare_sorted_resident_cache_(
            values,
            block_table,
            request_states,
            request_generations,
            state,
            workspace,
            block_size=block_size,
        )
        torch.npu.synchronize()
        miss_count = int(workspace.miss_counts[0, 0].cpu())
        if step == 0:
            assert miss_count > 0
        else:
            assert miss_count == 0
            assert torch.all(workspace.overwritten_slots[0].cpu() == 0)

    resident = _state_dict(state, 0, shard_count)
    assert values.reshape(-1).cpu().tolist() == [
        resident[token] for token in source.reshape(-1).tolist()
    ]


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_keeps_multiple_request_states_isolated(mtp):
    requests = 2
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = torch.cat((_source(mtp, 0), _source(mtp, 5000)), dim=0)
    values = source.npu()
    boundaries = torch.tensor(
        [1500] * mtp + [6500] * mtp,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.arange(
        requests, dtype=torch.int32, device="npu"
    ).repeat_interleave(mtp)
    request_states = torch.arange(requests, dtype=torch.int32, device="npu")
    request_generations = torch.tensor(
        [11, 22], dtype=torch.int64, device="npu"
    )
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    blocks_per_request = capacity // 128
    block_table = torch.arange(
        requests * blocks_per_request,
        dtype=torch.int32,
        device="npu",
    ).reshape(requests, blocks_per_request)
    expected_shards = _expected_shards(
        source,
        boundaries.cpu(),
        requests,
        mtp,
        shard_count,
    )

    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    prepare_sorted_resident_cache_(
        values,
        block_table,
        request_states,
        request_generations,
        state,
        workspace,
        block_size=128,
    )
    torch.npu.synchronize()

    references = []
    for request in range(requests):
        reference, expected_misses, expected_slots = _reference_step(
            expected_shards[request], {}, capacity
        )
        references.append(reference)
        miss_count = int(workspace.miss_counts[request, 0].cpu())
        assert miss_count == len(expected_misses)
        assert (
            workspace.miss_tokens[request, :miss_count].cpu().tolist()
            == expected_misses
        )
        expected_targets = [
            request * capacity + slot for slot in expected_slots
        ]
        assert (
            workspace.target_slots[request, :miss_count].cpu().tolist()
            == expected_targets
        )
        assert _state_dict(state, request, shard_count) == reference
        assert int(workspace.overwritten_slots[request].sum().cpu()) == miss_count

    values.copy_(source.npu())
    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    prepare_sorted_resident_cache_(
        values,
        block_table,
        request_states,
        request_generations,
        state,
        workspace,
        block_size=128,
    )
    torch.npu.synchronize()
    assert torch.all(workspace.miss_counts[:, 0].cpu() == 0)
    assert torch.all(workspace.overwritten_slots.cpu() == 0)
    for request in range(requests):
        assert _state_dict(state, request, shard_count) == references[request]


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_negative_state_uses_request_private_dummy_row(mtp):
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = _source(mtp, 0)
    values = source.npu()
    boundaries = torch.full((mtp,), 512, dtype=torch.int32, device="npu")
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.full((1,), -1, dtype=torch.int32, device="npu")
    request_generations = torch.full((1,), 9, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    block_table = torch.arange(
        capacity // 128, dtype=torch.int32, device="npu"
    ).reshape(1, -1)
    expected_shards = _expected_shards(
        source, boundaries.cpu(), requests, mtp, shard_count
    )[0]
    expected, expected_misses, _ = _reference_step(
        expected_shards, {}, capacity
    )

    prepare_resident_sharded_union_(
        values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    prepare_sorted_resident_cache_(
        values,
        block_table,
        request_states,
        request_generations,
        state,
        workspace,
        block_size=128,
    )
    torch.npu.synchronize()

    assert int(workspace.miss_counts[0, 0].cpu()) == len(expected_misses)
    assert torch.all(state.counts[0, :, 0].cpu() == 0)
    assert _state_dict(state, state.dummy_state_base, shard_count) == expected
    assert int(state.generations[state.dummy_state_base, 0].cpu()) == 9


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_full_path_supports_graph_replay(mtp):
    requests = 1
    capacity = mtp * 2048
    source = _source(mtp, 0)
    values = source.npu()
    boundaries = torch.full(
        (mtp,),
        1948 if mtp == 1 else 2972,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.ones(1, dtype=torch.int64, device="npu")
    block_table = torch.arange(capacity // 128, dtype=torch.int32, device="npu").reshape(1, -1)
    workspace = allocate_sorted_resident_workspace(requests, mtp, device=torch.device("npu"))
    state = allocate_sorted_resident_state(
        requests,
        requests,
        mtp,
        device=torch.device("npu"),
    )

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        prepare_resident_sharded_union_(
            values,
            boundaries,
            row_requests,
            request_states,
            request_generations,
            state,
            workspace,
            mtp=mtp,
        )
        prepare_sorted_resident_cache_(
            values,
            block_table,
            request_states,
            request_generations,
            state,
            workspace,
            block_size=128,
        )

    values.copy_(source.npu())
    state.counts.zero_()
    state.generations.fill_(-1)
    graph.replay()
    torch.npu.synchronize()
    expected_count = 1948 if mtp == 1 else 2972
    assert int(workspace.miss_counts[0, 0].cpu()) == expected_count
    assert int(workspace.overwritten_slots[0].sum().cpu()) == expected_count

    values.copy_(source.npu())
    graph.replay()
    torch.npu.synchronize()
    assert int(workspace.miss_counts[0, 0].cpu()) == 0
    assert int(workspace.overwritten_slots[0].sum().cpu()) == 0

    values.copy_(source.npu())
    request_generations.fill_(2)
    graph.replay()
    torch.npu.synchronize()
    assert int(workspace.miss_counts[0, 0].cpu()) == expected_count
    assert int(workspace.overwritten_slots[0].sum().cpu()) == expected_count
