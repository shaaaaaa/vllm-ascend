import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sorted_cache import (
    RESIDENT_FINALIZE_DEBUG_INTS,
    RESIDENT_READ_PROBE_DEBUG_INTS,
    allocate_sorted_resident_state,
    allocate_sorted_resident_workspace,
    debug_sorted_resident_finalize_only_,
    prepare_resident_sharded_union_,
    prepare_sorted_resident_cache_coordinated_,
    prepare_sorted_resident_cache_fused_,
    probe_sorted_resident_reads_,
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
        "npu_dsa_resident_finalize_coordinator_",
        "npu_dsa_resident_sharded_finalize_worker_",
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
            union.update(
                int(token)
                for token in rows[request, row].tolist()
                if 0 <= token < boundary
            )
        result.append(
            [
                sorted(token for token in union if token % shard_count == shard)
                for shard in range(shard_count)
            ]
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


def _assert_and_print_fused_remap_copy_inputs(
    source: torch.Tensor,
    workspace,
    expected_shards: list[list[list[int]]],
) -> None:
    """Publish the exact GM slices consumed by the fused remap copy."""
    requests = workspace.shard_mapping.shape[0]
    shard_count = workspace.shard_mapping.shape[1]
    request_width = workspace.shard_mapping.shape[2]
    part_width = request_width // shard_count
    mapping_base_ptr = workspace.shard_mapping.data_ptr()
    mapping = workspace.shard_mapping.cpu()
    counts = workspace.shard_counts[:, :, 0].cpu()
    source_flat = source.reshape(requests, request_width)

    print(
        "\n[resident-fused-remap-copy-input]"
        f" shape={tuple(workspace.shard_mapping.shape)}"
        f" dtype={workspace.shard_mapping.dtype}"
        f" contiguous={workspace.shard_mapping.is_contiguous()}"
        f" base_ptr=0x{mapping_base_ptr:x}"
        f" base_mod64={mapping_base_ptr % 64}"
        f" request_width={request_width}"
        f" part_width={part_width}"
        f" copy_elements={part_width}"
        f" copy_bytes={part_width * mapping.element_size()}"
    )

    for request in range(requests):
        expected_mapping = []
        tokens = source_flat[request].tolist()
        for source_shard in range(shard_count):
            rank_by_token = {
                token: rank
                for rank, token in enumerate(expected_shards[request][source_shard])
            }
            expected_mapping.append([rank_by_token.get(token, -1) for token in tokens])

        for part in range(shard_count):
            begin = part * part_width
            end = begin + part_width
            for source_shard in range(shard_count):
                element_offset = (
                    request * shard_count + source_shard
                ) * request_width + begin
                byte_offset = element_offset * mapping.element_size()
                source_ptr = mapping_base_ptr + byte_offset
                actual = mapping[request, source_shard, begin:end]
                expected = expected_mapping[source_shard][begin:end]
                assert actual.tolist() == expected
                nonnegative = actual[actual >= 0]
                rank_min = int(nonnegative.min()) if nonnegative.numel() else -1
                rank_max = int(nonnegative.max()) if nonnegative.numel() else -1
                print(
                    "[resident-fused-remap-copy-slice]"
                    f" request={request}"
                    f" part={part}"
                    f" source_shard={source_shard}"
                    f" shard_count_value={int(counts[request, source_shard])}"
                    f" begin={begin}"
                    f" end={end}"
                    f" element_offset={element_offset}"
                    f" byte_offset={byte_offset}"
                    f" source_ptr=0x{source_ptr:x}"
                    f" source_mod64={source_ptr % 64}"
                    f" count={part_width}"
                    f" negative={int((actual < 0).sum())}"
                    f" nonnegative={int((actual >= 0).sum())}"
                    f" rank_min={rank_min}"
                    f" rank_max={rank_max}"
                    f" first16={actual[:16].tolist()}"
                    f" last16={actual[-16:].tolist()}"
                )


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
    sample_indices = torch.nonzero(
        invalid_after | still_negative,
        as_tuple=False,
    ).flatten()[:32]
    sample = [(int(index), int(active_after[index])) for index in sample_indices]
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
        f" suspicious_sample={sample}"
    )


@pytest.mark.parametrize("mtp", [1, 2])
def test_resident_union_is_sorted_per_fixed_token_shard(mtp):
    requests = 2
    source = _source(mtp, 0, requests)
    boundary = 1948 if mtp == 1 else 2972
    boundaries = boundary - 73 * torch.arange(requests * mtp, dtype=torch.int32)
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
    prepare_sorted_resident_cache_fused_(
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
    expected_shards = _expected_shards(source, boundaries, requests, mtp, shard_count)[
        0
    ]
    union_tokens = [token for shard in expected_shards for token in shard]
    resident = {
        token: (index * 37) % capacity for index, token in enumerate(union_tokens[::5])
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
def test_shard_intersection_compacts_misses_and_evictable_slots(mtp):
    requests = 1
    shard_count = resident_shard_count(mtp)
    source = _source(mtp, 0)
    boundaries = torch.full((mtp,), 1600, dtype=torch.int32)
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.full((1,), 11, dtype=torch.int64, device="npu")
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    current_shards = _expected_shards(source, boundaries, requests, mtp, shard_count)[0]
    current_tokens = [token for shard in current_shards for token in shard]
    hit_tokens = current_tokens[::4]
    stale_tokens = list(range(20_000, 20_000 + 257))
    old_tokens = sorted(set(hit_tokens + stale_tokens))
    resident = {token: slot for slot, token in enumerate(old_tokens)}
    old_shards = [
        sorted(token for token in old_tokens if token % shard_count == shard)
        for shard in range(shard_count)
    ]
    _seed_resident_state(
        state,
        state_index=0,
        generation=11,
        shards=old_shards,
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

    counts = workspace.shard_counts[0].cpu()
    for shard in range(shard_count):
        current = current_shards[shard]
        expected_misses = [token for token in current if token not in resident]
        expected_positions = [
            position for position, token in enumerate(current) if token not in resident
        ]
        expected_evictable_slots = [
            resident[token] for token in old_shards[shard] if token not in set(current)
        ]
        assert int(counts[shard, 0]) == len(current)
        assert int(counts[shard, 1]) == len(expected_misses)
        assert int(counts[shard, 2]) == len(expected_evictable_slots)
        assert int(counts[shard, 3]) == len(old_shards[shard])
        assert (
            workspace.shard_miss_tokens[0, shard, : len(expected_misses)].cpu().tolist()
            == expected_misses
        )
        assert (
            workspace.shard_miss_positions[0, shard, : len(expected_positions)]
            .cpu()
            .tolist()
            == expected_positions
        )
        assert (
            workspace.shard_evictable_slots[0, shard, : len(expected_evictable_slots)]
            .cpu()
            .tolist()
            == expected_evictable_slots
        )


@pytest.mark.parametrize("mtp", [1, 2])
def test_kernel_read_probe_matches_full_capacity_union_output(mtp):
    """Capture the exact GM values observed by an AIV before finalization."""
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = (
        torch.arange(capacity, dtype=torch.int32).mul(shard_count).reshape(mtp, 1, 2048)
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
        assert torch.all(probe_prior[shard, expected_count:] == -12345)
        if expected_count:
            assert first_slot == int(host_prior[shard, 0])
            assert last_slot == int(host_prior[shard, expected_count - 1])
            assert negative == int((host_prior[shard, :expected_count] < 0).sum())
            assert nonnegative == int((host_prior[shard, :expected_count] >= 0).sum())


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
        torch.arange(capacity, dtype=torch.int32).mul(shard_count).reshape(mtp, 1, 2048)
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
        f"\n[resident-finalize-internal-stage] mtp={mtp} requested_stage={debug_stage} fields={fields}"
    )
    assert int(debug[0]) == 0x52534631
    assert fields["stage"] == debug_stage
    assert int(debug[2]) == shard_count
    assert int(debug[3]) == capacity
    assert fields["packed_end"] == capacity

    if debug_stage >= 2:
        assert fields["protected_count"] == 0
    if debug_stage == 3:
        assert torch.all(workspace.prior_slots[0, 0, :capacity].cpu() == 0)
    if debug_stage >= 4:
        assert fields["free_count"] == capacity
    if debug_stage == 5:
        assert workspace.prior_slots[0, 0, :capacity].cpu().tolist() == list(
            range(capacity)
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
        assert workspace.prior_slots[0, 0, :capacity].cpu().tolist() == list(
            range(capacity)
        )
    if debug_stage >= 9:
        assert torch.equal(
            workspace.miss_tokens[0, :capacity].cpu(),
            packed,
        )
    if debug_stage >= 10:
        assert workspace.target_slots[0, :capacity].cpu().tolist() == list(
            range(capacity)
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
        torch.arange(capacity, dtype=torch.int32).mul(shard_count).reshape(mtp, 1, 2048)
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
    misses = workspace.miss_tokens[0, :miss_count].cpu()
    targets = workspace.target_slots[0, :miss_count].cpu()
    packed = workspace.shard_packed[0, 0, :capacity].cpu()
    print(
        "\n[resident-finalize-kernel-boundary]"
        f" mtp={mtp}"
        f" miss_count={miss_count}"
        f" prior_negative={int((prior < 0).sum())}"
        f" prior_unique={torch.unique(prior).numel()}"
        f" miss_first={int(misses[0]) if miss_count else -1}"
        f" miss_last={int(misses[-1]) if miss_count else -1}"
        f" target_first={int(targets[0]) if miss_count else -1}"
        f" target_last={int(targets[-1]) if miss_count else -1}"
    )
    assert miss_count == capacity
    assert prior.tolist() == list(range(capacity))
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
        torch.arange(capacity, dtype=torch.int32).mul(shard_count).reshape(mtp, 1, 2048)
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
    _print_empty_state_finalize_debug(
        mtp=mtp,
        capacity=capacity,
        counts=counts,
        prior_before=prior_before,
        workspace=workspace,
    )
    assert int(workspace.miss_counts[0, 0].cpu()) == capacity
    assert values.reshape(-1).cpu().tolist() == list(range(capacity))
    assert int(state.counts[0, 0, 0].cpu()) == capacity


def _reference_step(
    shards: list[list[int]],
    resident: dict[int, int],
    capacity: int,
) -> tuple[dict[int, int], list[int], list[int]]:
    occupied = set(resident.values())
    assert occupied == set(range(len(occupied)))
    current = {token for shard in shards for token in shard}
    shard_count = len(shards)
    evictable_slots = []
    for shard in range(shard_count):
        old_tokens = sorted(
            token
            for token in resident
            if token % shard_count == shard and token not in current
        )
        evictable_slots.extend(resident[token] for token in old_tokens)
    assigned = {}
    misses = []
    miss_slots = []
    for shard in shards:
        for token in shard:
            if token in resident:
                assigned[token] = resident[token]
            else:
                miss_index = len(misses)
                if miss_index < len(evictable_slots):
                    slot = evictable_slots[miss_index]
                else:
                    slot = len(occupied) + miss_index - len(evictable_slots)
                assert slot < capacity
                assigned[token] = slot
                misses.append(token)
                miss_slots.append(slot)
    overwritten = set(miss_slots)
    updated = {
        token: slot for token, slot in resident.items() if slot not in overwritten
    }
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
        reference, expected_misses, expected_slots = _reference_step(
            expected_shards, reference, capacity
        )
        prepare_sorted_resident_cache_fused_(
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
        assert _state_dict(state, 0, shard_count) == reference
        remapped = values.reshape(-1).cpu().tolist()
        original = source.reshape(-1).tolist()
        for position, token in enumerate(original):
            expected = reference[token] if token < boundary else token
            assert remapped[position] == expected


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_grows_contiguous_slot_tail_before_reuse(mtp):
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    block_size = 128
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.ones(1, dtype=torch.int64, device="npu")
    block_table = torch.arange(
        capacity // block_size,
        dtype=torch.int32,
        device="npu",
    ).reshape(1, -1)

    reference: dict[int, int] = {}
    for offset, boundary in ((0, 1000), (0, 1200), (100, 1300)):
        source = _source(mtp, offset)
        values = source.npu()
        boundaries = torch.full((mtp,), boundary, dtype=torch.int32, device="npu")
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
            expected_shards, reference, capacity
        )
        prepare_sorted_resident_cache_fused_(
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
        assert miss_count == len(expected_misses)
        assert workspace.miss_tokens[0, :miss_count].cpu().tolist() == expected_misses
        assert workspace.target_slots[0, :miss_count].cpu().tolist() == expected_slots
        actual = _state_dict(state, 0, shard_count)
        assert actual == reference
        assert set(actual.values()) == set(range(len(actual)))


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_pools_evict_slots_across_shards(mtp):
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    block_size = 128
    boundary = 100
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
    state = allocate_sorted_resident_state(
        requests, requests, mtp, device=torch.device("npu")
    )
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    boundaries = torch.full((mtp,), boundary, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.ones(1, dtype=torch.int64, device="npu")
    block_table = torch.arange(
        capacity // block_size,
        dtype=torch.int32,
        device="npu",
    ).reshape(1, -1)

    def make_source(selected):
        source = torch.arange(
            mtp * 2048,
            dtype=torch.int32,
        ).reshape(mtp, 1, 2048)
        source.add_(10_000)
        source[0, 0, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
        return source

    first = make_source([0, 1, 2, 3, 4])
    first_values = first.npu()
    prepare_resident_sharded_union_(
        first_values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    prepare_sorted_resident_cache_fused_(
        first_values,
        block_table,
        request_states,
        request_generations,
        state,
        workspace,
        block_size=block_size,
    )
    torch.npu.synchronize()
    first_resident = _state_dict(state, 0, shard_count)
    evicted_slot = first_resident[0]

    second = make_source([1, 2, 3, 5])
    second_values = second.npu()
    prepare_resident_sharded_union_(
        second_values,
        boundaries,
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    prepare_sorted_resident_cache_fused_(
        second_values,
        block_table,
        request_states,
        request_generations,
        state,
        workspace,
        block_size=block_size,
    )
    torch.npu.synchronize()

    assert 0 % shard_count != 5 % shard_count
    assert int(workspace.miss_counts[0, 0].cpu()) == 1
    assert int(workspace.miss_tokens[0, 0].cpu()) == 5
    assert int(workspace.target_slots[0, 0].cpu()) == evicted_slot
    selected_evict_counts = workspace.shard_counts[0, :, 4].cpu().tolist()
    assert sum(selected_evict_counts) == 1
    assert selected_evict_counts[0 % shard_count] == 1
    assert selected_evict_counts[5 % shard_count] == 0
    second_resident = _state_dict(state, 0, shard_count)
    assert 0 not in second_resident
    assert second_resident[4] == first_resident[4]
    assert second_resident[5] == evicted_slot


@pytest.mark.parametrize("mtp", [1, 2])
def test_fused_sorted_resident_matches_three_step_reference(mtp):
    """Validate fused update+remap across misses, hits, and replacements."""
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
    row_requests = torch.zeros(
        requests * mtp,
        dtype=torch.int32,
        device="npu",
    )
    boundaries = torch.full(
        (requests * mtp,),
        boundary,
        dtype=torch.int32,
        device="npu",
    )
    request_states = torch.zeros(
        requests,
        dtype=torch.int32,
        device="npu",
    )
    request_generations = torch.ones(
        requests,
        dtype=torch.int64,
        device="npu",
    )
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
        reference, expected_misses, expected_slots = _reference_step(
            expected_shards,
            reference,
            capacity,
        )
        prepare_sorted_resident_cache_fused_(
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
        assert _state_dict(state, 0, shard_count) == reference

        remapped = values.reshape(-1).cpu().tolist()
        original = source.reshape(-1).tolist()
        for position, token in enumerate(original):
            expected = reference[token] if token < boundary else token
            assert remapped[position] == expected


@pytest.mark.parametrize("mtp", [1, 2])
def test_fused_remap_bisect_synced_exact_copy_loop_does_not_crash(mtp):
    """Exercise the complete fused remap with per-shard MTE2 scalar sync."""
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

    assert values.reshape(-1).cpu().tolist() == [
        reference[token] for token in source.reshape(-1).tolist()
    ]
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
    boundaries = torch.full((requests * mtp,), 100_000, dtype=torch.int32, device="npu")
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
        prepare_sorted_resident_cache_fused_(
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
    request_generations = torch.tensor([11, 22], dtype=torch.int64, device="npu")
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
        expected_targets = [request * capacity + slot for slot in expected_slots]
        assert (
            workspace.target_slots[request, :miss_count].cpu().tolist()
            == expected_targets
        )
        assert _state_dict(state, request, shard_count) == reference

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
    assert torch.all(workspace.miss_counts[:, 0].cpu() == 0)
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
    expected, expected_misses, _ = _reference_step(expected_shards, {}, capacity)

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
    block_table = torch.arange(
        capacity // 128, dtype=torch.int32, device="npu"
    ).reshape(1, -1)
    workspace = allocate_sorted_resident_workspace(
        requests, mtp, device=torch.device("npu")
    )
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
        prepare_sorted_resident_cache_fused_(
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

    values.copy_(source.npu())
    graph.replay()
    torch.npu.synchronize()
    assert int(workspace.miss_counts[0, 0].cpu()) == 0

    values.copy_(source.npu())
    request_generations.fill_(2)
    graph.replay()
    torch.npu.synchronize()
    assert int(workspace.miss_counts[0, 0].cpu()) == expected_count


def _coordinated_seed(current_shards, scenario):
    shard_count = len(current_shards)
    if scenario == "empty":
        return [[] for _ in range(shard_count)], {}
    if scenario == "resident90":
        old_shards = [
            [token for index, token in enumerate(tokens) if index % 10 != 0]
            for tokens in current_shards
        ]
    elif scenario == "cross_shard":
        old_shards = [list(tokens) for tokens in current_shards]
        missing = old_shards[-1][:17]
        old_shards[-1] = old_shards[-1][17:]
        used = {token for tokens in current_shards for token in tokens}
        fillers = []
        candidate = 100_000
        while len(fillers) < len(missing):
            if candidate % shard_count == 0 and candidate not in used:
                fillers.append(candidate)
            candidate += 1
        old_shards[0] = sorted(old_shards[0] + fillers)
    else:
        raise AssertionError(f"unknown coordinated scenario: {scenario}")
    resident_tokens = [token for tokens in old_shards for token in tokens]
    return old_shards, {
        token: slot for slot, token in enumerate(resident_tokens)
    }


@pytest.mark.parametrize(
    "mtp,shard_count_override",
    [(1, None), (1, 4), (2, None), (2, 8)],
)
@pytest.mark.parametrize("scenario", ["empty", "resident90", "cross_shard"])
def test_coordinated_finalize_matches_production(
    mtp, shard_count_override, scenario
):
    requests = 2
    capacity = mtp * 2048
    block_size = 128
    source = _source(mtp, 0, requests)
    boundary = 1948 if mtp == 1 else 2972
    boundaries_cpu = torch.full(
        (requests * mtp,), boundary, dtype=torch.int32
    )
    boundaries = boundaries_cpu.npu()
    row_requests = torch.arange(
        requests, dtype=torch.int32, device="npu"
    ).repeat_interleave(mtp)
    request_states = torch.arange(requests, dtype=torch.int32, device="npu")
    request_generations = torch.arange(
        11, 11 + requests, dtype=torch.int64, device="npu"
    )
    shard_count = shard_count_override or resident_shard_count(mtp)
    expected_shards = _expected_shards(
        source, boundaries_cpu, requests, mtp, shard_count
    )
    block_table = (
        torch.arange(
            requests * (capacity // block_size),
            dtype=torch.int32,
            device="npu",
        ).reshape(requests, -1)
        + 100
    )

    states = [
        allocate_sorted_resident_state(
            requests,
            requests,
            mtp,
            device=torch.device("npu"),
            shard_count=shard_count_override,
        )
        for _ in range(2)
    ]
    workspaces = [
        allocate_sorted_resident_workspace(
            requests,
            mtp,
            device=torch.device("npu"),
            shard_count=shard_count_override,
        )
        for _ in range(2)
    ]
    for request in range(requests):
        old_shards, resident = _coordinated_seed(
            expected_shards[request], scenario
        )
        for state in states:
            if resident:
                _seed_resident_state(
                    state,
                    state_index=request,
                    generation=11 + request,
                    shards=old_shards,
                    resident=resident,
                )

    values = [source.npu(), source.npu()]
    for value, state, workspace in zip(values, states, workspaces):
        prepare_resident_sharded_union_(
            value,
            boundaries,
            row_requests,
            request_states,
            request_generations,
            state,
            workspace,
            mtp=mtp,
        )
    prepare_sorted_resident_cache_fused_(
        values[0],
        block_table,
        request_states,
        request_generations,
        states[0],
        workspaces[0],
        block_size=block_size,
    )
    prepare_sorted_resident_cache_coordinated_(
        values[1],
        block_table,
        request_states,
        request_generations,
        states[1],
        workspaces[1],
        block_size=block_size,
    )
    torch.npu.synchronize()

    assert torch.equal(values[0].cpu(), values[1].cpu())
    assert torch.equal(
        workspaces[0].shard_counts[:, :, :5].cpu(),
        workspaces[1].shard_counts[:, :, :5].cpu(),
    )
    for request in range(requests):
        miss_count = int(workspaces[0].miss_counts[request, 0].cpu())
        assert int(workspaces[1].miss_counts[request, 0].cpu()) == miss_count
        metadata = workspaces[1].shard_counts[request].cpu()
        miss_prefix = 0
        evict_prefix = 0
        total_old = int(metadata[:, 3].sum())
        total_selected = int(metadata[:, 4].sum())
        for shard in range(shard_count):
            assert metadata[shard, 5:10].tolist() == [
                miss_prefix,
                evict_prefix,
                total_selected,
                total_old,
                miss_count,
            ]
            miss_prefix += int(metadata[shard, 1])
            evict_prefix += int(metadata[shard, 4])
        assert torch.equal(
            workspaces[0].miss_tokens[request, :miss_count].cpu(),
            workspaces[1].miss_tokens[request, :miss_count].cpu(),
        )
        assert torch.equal(
            workspaces[0].target_slots[request, :miss_count].cpu(),
            workspaces[1].target_slots[request, :miss_count].cpu(),
        )
        for shard in range(shard_count):
            count = int(workspaces[0].shard_counts[request, shard, 0].cpu())
            assert torch.equal(
                workspaces[0].prior_slots[request, shard, :count].cpu(),
                workspaces[1].prior_slots[request, shard, :count].cpu(),
            )
        assert _state_dict(states[0], request, shard_count) == _state_dict(
            states[1], request, shard_count
        )


@pytest.mark.parametrize(
    "mtp,shard_count_override",
    [(1, None), (1, 4), (2, None), (2, 8)],
)
def test_coordinated_finalize_supports_graph_replay(
    mtp, shard_count_override
):
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
    block_table = torch.arange(
        capacity // 128, dtype=torch.int32, device="npu"
    ).reshape(1, -1)
    workspace = allocate_sorted_resident_workspace(
        requests,
        mtp,
        device=torch.device("npu"),
        shard_count=shard_count_override,
    )
    state = allocate_sorted_resident_state(
        requests,
        requests,
        mtp,
        device=torch.device("npu"),
        shard_count=shard_count_override,
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
        prepare_sorted_resident_cache_coordinated_(
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

    values.copy_(source.npu())
    graph.replay()
    torch.npu.synchronize()
    assert int(workspace.miss_counts[0, 0].cpu()) == 0
