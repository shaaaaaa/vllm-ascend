import os
from pathlib import Path

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sorted_cache import (
    INDEX_TOPK,
    RESIDENT_FINALIZE_DEBUG_INTS,
    RESIDENT_READ_PROBE_DEBUG_INTS,
    allocate_sorted_resident_state,
    allocate_sorted_resident_workspace,
    debug_sorted_resident_finalize_only_,
    prepare_resident_sharded_union_,
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


def _load_target_sfa_planner_snapshot() -> tuple[dict, dict, Path]:
    raw_path = os.getenv("VLLM_ASCEND_TARGET_SFA_REPLAY_SNAPSHOT")
    if not raw_path:
        pytest.skip(
            "set VLLM_ASCEND_TARGET_SFA_REPLAY_SNAPSHOT to a target slot "
            "directory or layer_NN_pre.pt"
        )
    path = Path(raw_path)
    if path.is_dir():
        pre_paths = sorted(path.glob("layer_*_pre.pt"))
        if not pre_paths:
            pytest.fail(f"target snapshot directory has no layer pre file: {path}")
        pre_path = pre_paths[0]
    else:
        pre_path = path
    if not pre_path.is_file() or not pre_path.name.endswith("_pre.pt"):
        pytest.fail(f"target replay snapshot is not a layer pre file: {pre_path}")
    input_path = pre_path.with_name(pre_path.name.replace("_pre.pt", "_input.pt"))
    if not input_path.is_file():
        pytest.fail(f"paired target input snapshot is missing: {input_path}")

    pre = torch.load(pre_path, map_location="cpu", weights_only=False)
    failure_input_path = os.getenv(
        "VLLM_ASCEND_TARGET_SFA_REPLAY_FAILURE_INPUT"
    )
    if not failure_input_path:
        pytest.fail(
            "set VLLM_ASCEND_TARGET_SFA_REPLAY_FAILURE_INPUT to the paired "
            "layer input snapshot from a rank that reported the device fault"
        )
    input_path = Path(failure_input_path)
    if not input_path.is_file() or not input_path.name.endswith("_input.pt"):
        pytest.fail(
            "failure-rank replay input is not a layer input file: "
            f"{input_path}"
        )
    layer_input = torch.load(input_path, map_location="cpu", weights_only=False)
    # The failing rank faults before its pre snapshot can be written. Allow its
    # input/state to be paired with the successful rank's planner intermediates,
    # but require the same logical forward and layer. Rank is deliberately not
    # part of this identity check and is reported separately by the test.
    identity = ("schema_version", "step_id", "layer")
    mismatched = {
        name: (layer_input.get(name), pre.get(name))
        for name in identity
        if layer_input.get(name) != pre.get(name)
    }
    if mismatched:
        pytest.fail(f"target input/pre snapshot identity mismatch: {mismatched}")
    return layer_input, pre, pre_path


def _workspace_snapshot_fields(pre: dict) -> dict[str, torch.Tensor]:
    snapshot = pre.get("resident_workspace")
    if not isinstance(snapshot, dict):
        pytest.fail("target pre snapshot has no resident_workspace")
    fields = snapshot.get("fields")
    if not isinstance(fields, dict):
        pytest.fail("resident_workspace is not a dataclass snapshot")
    return fields


def _reconstruct_resident_source_topk(pre: dict) -> torch.Tensor:
    """Invert the saved shard mapping to recover planner input token IDs."""
    fields = _workspace_snapshot_fields(pre)
    packed = fields["shard_packed"]
    mapping = fields["shard_mapping"]
    remapped = pre["topk_indices"].clone()
    request_count, shard_count, request_width = mapping.shape
    source = remapped.reshape(request_count, request_width).clone()
    assigned = torch.zeros_like(source, dtype=torch.bool)

    for shard in range(shard_count):
        ranks = mapping[:, shard].to(torch.long)
        valid = ranks >= 0
        if torch.any(valid & assigned):
            pytest.fail("saved shard mapping assigns one source position twice")
        gathered = packed[:, shard].gather(1, ranks.clamp_min(0))
        source[valid] = gathered[valid]
        assigned |= valid
    return source.reshape_as(remapped)


def _copy_resident_snapshot_(state, snapshot: dict) -> None:
    rows = torch.tensor(snapshot["row_indices"], dtype=torch.long, device="npu")
    for name in ("tokens", "slots", "counts", "generations"):
        getattr(state, name).index_copy_(
            0,
            rows,
            snapshot[name].to(device="npu"),
        )


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
def test_fused_update_trusts_union_old_count_on_generation_rollover(mtp):
    """A stale persistent count must not resurrect invalidated state."""
    requests = 1
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    source = _source(mtp, 0)
    boundaries = torch.full((mtp,), 1600, dtype=torch.int32)
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.full((1,), 8, dtype=torch.int64, device="npu")
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

    stale_shards = [
        [100_000 + shard + index * shard_count for index in range(8)]
        for shard in range(shard_count)
    ]
    stale_resident = {
        token: slot
        for slot, token in enumerate(
            token for shard in stale_shards for token in shard
        )
    }
    _seed_resident_state(
        state,
        state_index=0,
        generation=7,
        shards=stale_shards,
        resident=stale_resident,
    )

    values = source.npu()
    prepare_resident_sharded_union_(
        values,
        boundaries.npu(),
        row_requests,
        request_states,
        request_generations,
        state,
        workspace,
        mtp=mtp,
    )
    torch.npu.synchronize()
    assert torch.all(workspace.shard_counts[0, :, 3].cpu() == 0)
    assert torch.all(state.counts[0, :, 0].cpu() == 0)

    # Model the stale scalar cacheline observed in the production graph: GM
    # contains an obsolete nonzero count even though union's generation-aware
    # workspace correctly records an empty old prefix for this invocation.
    for shard in range(shard_count):
        state.counts[0, shard, 0] = len(stale_shards[shard])

    expected_shards = _expected_shards(
        source,
        boundaries,
        requests,
        mtp,
        shard_count,
    )[0]
    expected_state, expected_misses, expected_slots = _reference_step(
        expected_shards,
        {},
        capacity,
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

    assert _state_dict(state, 0, shard_count) == expected_state
    miss_count = int(workspace.miss_counts[0, 0].cpu())
    assert workspace.miss_tokens[0, :miss_count].cpu().tolist() == expected_misses
    assert workspace.target_slots[0, :miss_count].cpu().tolist() == expected_slots
    assert int(state.generations[0, 0].cpu()) == 8


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
    """Validate concurrent fused update+remap across fixed-address reuse."""
    requests = 2
    shard_count = resident_shard_count(mtp)
    capacity = mtp * 2048
    block_size = 128
    boundary = 100_000
    request_token_stride = 10_000
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
    row_requests = torch.arange(
        requests, dtype=torch.int32, device="npu"
    ).repeat_interleave(mtp)
    boundaries = torch.full(
        (requests * mtp,),
        boundary,
        dtype=torch.int32,
        device="npu",
    )
    request_states = torch.arange(
        requests, dtype=torch.int32, device="npu"
    )
    request_generations = torch.arange(
        1, requests + 1, dtype=torch.int64, device="npu"
    )
    blocks_per_request = capacity // block_size
    block_table_cpu = torch.arange(
        requests * blocks_per_request, dtype=torch.int32
    ).reshape(requests, blocks_per_request)
    block_table = block_table_cpu.npu()

    references: list[dict[int, int]] = [{} for _ in range(requests)]
    for offset in (0, 100, 0):
        source = torch.cat(
            tuple(
                _source(mtp, offset + request * request_token_stride)
                for request in range(requests)
            )
        )
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
        )
        expected_steps = []
        for request in range(requests):
            reference, expected_misses, expected_slots = _reference_step(
                expected_shards[request],
                references[request],
                capacity,
            )
            references[request] = reference
            expected_steps.append((expected_misses, expected_slots))
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

        remapped = values.reshape(requests, -1).cpu()
        original = source.reshape(requests, -1)
        for request in range(requests):
            expected_misses, expected_slots = expected_steps[request]
            expected_targets = [
                int(block_table_cpu[request, slot // block_size]) * block_size
                + slot % block_size
                for slot in expected_slots
            ]
            miss_count = int(workspace.miss_counts[request, 0].cpu())
            assert (
                workspace.miss_tokens[request, :miss_count].cpu().tolist()
                == expected_misses
            )
            assert (
                workspace.target_slots[request, :miss_count].cpu().tolist()
                == expected_targets
            )
            reference = references[request]
            assert _state_dict(state, request, shard_count) == reference
            for position, token in enumerate(original[request].tolist()):
                expected = reference[token] if token < boundary else token
                assert int(remapped[request, position]) == expected


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
    shard_count = resident_shard_count(mtp)
    expected_shards = _expected_shards(
        source,
        boundaries.cpu(),
        requests,
        mtp,
        shard_count,
    )[0]
    expected_state, expected_misses, expected_slots = _reference_step(
        expected_shards,
        {},
        capacity,
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
    _assert_union_outputs(source, boundaries.cpu(), workspace, mtp=mtp)
    assert workspace.miss_tokens[0, :expected_count].cpu().tolist() == (
        expected_misses
    )
    # This test uses an identity physical block table, so physical target
    # slots equal the reference logical resident slots.
    assert workspace.target_slots[0, :expected_count].cpu().tolist() == (
        expected_slots
    )
    assert _state_dict(state, 0, shard_count) == expected_state

    values.copy_(source.npu())
    graph.replay()
    torch.npu.synchronize()
    assert int(workspace.miss_counts[0, 0].cpu()) == 0
    assert _state_dict(state, 0, shard_count) == expected_state

    values.copy_(source.npu())
    request_generations.fill_(2)
    graph.replay()
    torch.npu.synchronize()
    assert int(workspace.miss_counts[0, 0].cpu()) == expected_count
    assert _state_dict(state, 0, shard_count) == expected_state


@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("remap", [False, True])
def test_sorted_resident_split_graph_replay_completes_writeback(mtp, remap):
    requests = 1
    capacity = mtp * INDEX_TOPK
    source = _source(mtp, 0)
    values = source.npu()
    boundaries_cpu = torch.full(
        (mtp,),
        1948 if mtp == 1 else 2972,
        dtype=torch.int32,
    )
    boundaries = boundaries_cpu.npu()
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(1, dtype=torch.int32, device="npu")
    request_generations = torch.ones(1, dtype=torch.int64, device="npu")
    block_table = torch.arange(
        capacity // 128,
        dtype=torch.int32,
        device="npu",
    ).reshape(1, -1)
    shard_count = resident_shard_count(mtp)
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
    expected_shards = _expected_shards(
        source,
        boundaries_cpu,
        requests,
        mtp,
        shard_count,
    )[0]
    expected_state, _, _ = _reference_step(
        expected_shards,
        {},
        capacity,
    )
    expected_values = source.clone().reshape(requests, -1)
    if remap:
        for position, token in enumerate(expected_values[0].tolist()):
            row = position // INDEX_TOPK
            if 0 <= token < int(boundaries_cpu[row]):
                expected_values[0, position] = expected_state[token]
    expected_values = expected_values.reshape_as(source)

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
        prepare_sorted_resident_cache_no_remap_(
            values,
            block_table,
            request_states,
            request_generations,
            state,
            workspace,
            block_size=128,
        )
        if remap:
            remap_sorted_resident_cache_(values, workspace)

    state.counts.zero_()
    state.generations.fill_(-1)
    graph.replay()
    torch.npu.synchronize()

    assert torch.equal(values.cpu(), expected_values)
    assert _state_dict(state, 0, shard_count) == expected_state


def test_target_sfa_snapshot_replays_production_resident_planner():
    """Bisect the saved layer-0 failure through graph-replayed planner stages."""
    layer_input, pre, pre_path = _load_target_sfa_planner_snapshot()
    metadata = layer_input.get("metadata")
    state_before = layer_input.get("resident_state_before")
    if not isinstance(metadata, dict) or not isinstance(state_before, dict):
        pytest.fail("target input snapshot lacks metadata or resident state-before")

    fields = _workspace_snapshot_fields(pre)
    raw_topk = pre.get("raw_topk")
    source_cpu = (
        raw_topk
        if isinstance(raw_topk, torch.Tensor)
        else _reconstruct_resident_source_topk(pre)
    ).contiguous()
    request_count = int(pre["selected_counts"].shape[0])
    request_width = int(source_cpu.numel()) // request_count
    if request_width % INDEX_TOPK:
        pytest.fail(f"saved request width is not a multiple of top-k: {request_width}")
    mtp = request_width // INDEX_TOPK
    if mtp not in (1, 2):
        pytest.fail(f"saved resident planner has unsupported MTP={mtp}")
    shard_count = int(fields["shard_packed"].shape[1])
    dummy_state_base = int(state_before["dummy_state_base"])

    def metadata_tensor(*names: str) -> torch.Tensor:
        for name in names:
            value = metadata.get(name)
            if isinstance(value, torch.Tensor):
                return value
        pytest.fail(f"target snapshot lacks metadata tensor from {names}")

    rows = request_count * mtp
    raw_boundary = pre.get("remap_boundary")
    boundaries_cpu = (
        raw_boundary
        if isinstance(raw_boundary, torch.Tensor)
        else metadata_tensor(
            "decode_remap_boundary",
            "decode_split_boundary",
            "split_boundary",
        )
    )[:rows].to(torch.int32).contiguous()
    row_requests_cpu = (
        metadata_tensor("decode_req_indices")[:rows]
        .to(torch.int32)
        .contiguous()
    )
    request_states_cpu = (
        metadata_tensor("resident_state_indices")[:request_count]
        .to(torch.int32)
        .contiguous()
    )
    request_generations_cpu = (
        metadata_tensor("resident_state_generations")[:request_count]
        .to(torch.int64)
        .contiguous()
    )
    block_table_cpu = (
        metadata_tensor("block_table")[:request_count]
        .to(torch.int32)
        .contiguous()
    )
    block_size = int(
        pre.get(
            "block_size",
            os.getenv("VLLM_ASCEND_TARGET_SFA_REPLAY_BLOCK_SIZE", "128"),
        )
    )
    replay_count = int(os.getenv("VLLM_ASCEND_TARGET_SFA_REPLAY_COUNT", "8"))
    if replay_count <= 0:
        pytest.fail("VLLM_ASCEND_TARGET_SFA_REPLAY_COUNT must be positive")

    replay_device = os.getenv("VLLM_ASCEND_TARGET_SFA_REPLAY_DEVICE")
    if not replay_device:
        pytest.fail(
            "set VLLM_ASCEND_TARGET_SFA_REPLAY_DEVICE to a failing physical "
            "device such as npu:5; running this replay implicitly on rank-0's "
            "device is only a reference and cannot reproduce the reported fault"
        )
    device = torch.device(replay_device)
    torch.npu.set_device(device)
    values = source_cpu.to(device=device)
    source = values.clone()
    boundaries = boundaries_cpu.to(device=device)
    row_requests = row_requests_cpu.to(device=device)
    request_states = request_states_cpu.to(device=device)
    request_generations = request_generations_cpu.to(device=device)
    block_table = block_table_cpu.to(device=device)
    state = allocate_sorted_resident_state(
        dummy_state_base,
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

    def restore_failure_input() -> None:
        values.copy_(source)
        state.counts.zero_()
        state.generations.fill_(-1)
        _copy_resident_snapshot_(state, state_before)

    restore_failure_input()
    torch.npu.synchronize()
    resident_before: list[dict[int, int]] = []
    for request in range(request_count):
        state_index = int(request_states_cpu[request])
        generation = int(request_generations_cpu[request])
        if (
            0 <= state_index < dummy_state_base
            and int(state.generations[state_index, 0].cpu()) == generation
        ):
            resident_before.append(
                _state_dict(state, state_index, shard_count)
            )
        else:
            resident_before.append({})

    expected_shards = _expected_shards(
        source_cpu,
        boundaries_cpu,
        request_count,
        mtp,
        shard_count,
    )
    expected_states: list[dict[int, int]] = []
    expected_misses: list[list[int]] = []
    expected_miss_slots: list[list[int]] = []
    for request in range(request_count):
        expected_state, misses, miss_slots = _reference_step(
            expected_shards[request],
            resident_before[request],
            request_width,
        )
        expected_states.append(expected_state)
        expected_misses.append(misses)
        expected_miss_slots.append(miss_slots)

    expected_remap = source_cpu.reshape(request_count, request_width).clone()
    for request in range(request_count):
        for position, token in enumerate(expected_remap[request].tolist()):
            row = position // INDEX_TOPK
            boundary = int(boundaries_cpu[request * mtp + row])
            if 0 <= token < boundary:
                expected_remap[request, position] = expected_states[request][token]
    expected_remap = expected_remap.reshape_as(source_cpu)

    def union_body() -> None:
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

    def finalize_body() -> None:
        union_body()
        debug_sorted_resident_finalize_only_(
            block_table,
            workspace,
            block_size=block_size,
        )

    def no_remap_body() -> None:
        union_body()
        prepare_sorted_resident_cache_no_remap_(
            values,
            block_table,
            request_states,
            request_generations,
            state,
            workspace,
            block_size=block_size,
        )

    def split_remap_body() -> None:
        no_remap_body()
        remap_sorted_resident_cache_(values, workspace)

    def fused_body() -> None:
        union_body()
        prepare_sorted_resident_cache_fused_(
            values,
            block_table,
            request_states,
            request_generations,
            state,
            workspace,
            block_size=block_size,
        )

    def assert_union() -> None:
        _assert_union_outputs(
            source_cpu,
            boundaries_cpu,
            workspace,
            mtp=mtp,
        )

    def assert_finalize() -> None:
        assert_union()
        for request in range(request_count):
            miss_count = int(workspace.miss_counts[request, 0].cpu())
            assert miss_count == len(expected_misses[request])
            assert (
                workspace.miss_tokens[request, :miss_count].cpu().tolist()
                == expected_misses[request]
            )
            slots = expected_miss_slots[request]
            expected_targets = [
                int(block_table_cpu[request, slot // block_size]) * block_size
                + slot % block_size
                for slot in slots
            ]
            assert (
                workspace.target_slots[request, :miss_count].cpu().tolist()
                == expected_targets
            )

    def assert_state_updated() -> None:
        assert_finalize()
        assert torch.equal(values.cpu(), source_cpu)
        for request in range(request_count):
            state_index = int(request_states_cpu[request])
            if state_index < 0:
                state_index = state.dummy_state_base + request
            assert _state_dict(state, state_index, shard_count) == (
                expected_states[request]
            )

    def assert_remapped() -> None:
        assert_finalize()
        assert torch.equal(values.cpu(), expected_remap)
        for request in range(request_count):
            state_index = int(request_states_cpu[request])
            if state_index < 0:
                state_index = state.dummy_state_base + request
            assert _state_dict(state, state_index, shard_count) == (
                expected_states[request]
            )

    def run_graph_stage(label: str, body, assertion) -> None:
        restore_failure_input()
        print(f"[target-sfa-resident-replay] stage={label} capture started")
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            body()
        torch.npu.synchronize()
        # Ascend graph capture records the body but does not execute it. Validate
        # only after replay; checking here would inspect the zero-initialized
        # workspace and incorrectly report the union stage as broken.
        print(f"[target-sfa-resident-replay] stage={label} capture completed")
        for replay in range(replay_count):
            restore_failure_input()
            print(
                f"[target-sfa-resident-replay] stage={label} "
                f"replay={replay + 1} started"
            )
            graph.replay()
            torch.npu.synchronize()
            assertion()
            print(
                f"[target-sfa-resident-replay] stage={label} "
                f"replay={replay + 1} passed"
            )

    saved_topk = pre["topk_indices"].reshape_as(expected_remap)
    saved_mismatches = int((saved_topk != expected_remap).sum())
    print(
        "[target-sfa-resident-replay]"
        f" planner_snapshot={pre_path}"
        f" planner_snapshot_rank={pre.get('rank')}"
        f" failure_input_rank={layer_input.get('rank')}"
        f" execution_device={device}"
        f" requests={request_count} mtp={mtp}"
        f" shards={shard_count} block_size={block_size}"
        f" boundaries={boundaries_cpu.tolist()}"
        f" replays_per_stage={replay_count}"
        f" saved_oracle_mismatches={saved_mismatches}"
    )
    run_graph_stage("union", union_body, assert_union)
    run_graph_stage("finalize", finalize_body, assert_finalize)
    run_graph_stage("state_update_no_remap", no_remap_body, assert_state_updated)
    run_graph_stage("standalone_remap", split_remap_body, assert_remapped)
    run_graph_stage("production_fused", fused_body, assert_remapped)


def test_mtp1_decode_6400_boundary_graph_replay_does_not_poison_stream():
    """Stress the TP-local resident shape from the 6400-token crash."""
    num_speculative_tokens = 1
    mtp = 1 + num_speculative_tokens
    requests = 1
    block_size = 128
    activation_boundary = mtp * INDEX_TOPK
    cached_tokens = 6400
    stress_replay_pairs = 32
    capacity = mtp * INDEX_TOPK
    shard_count = resident_shard_count(mtp)

    # One speculative token produces two target rows. With the production
    # defaults this is the 4096-slot, eight-shard operator shape from the log.
    assert capacity == 4096
    assert shard_count == 8

    def make_source(offset: int, boundary: int) -> torch.Tensor:
        cached_selection_count = capacity - mtp
        assert offset + cached_selection_count <= boundary
        cached_positions = torch.arange(
            offset,
            offset + cached_selection_count,
            dtype=torch.int32,
        )
        live_positions = torch.arange(
            boundary,
            boundary + mtp,
            dtype=torch.int32,
        )
        return torch.cat((cached_positions, live_positions)).reshape(mtp, 1, INDEX_TOPK)

    warmup_steps = (
        (0, activation_boundary),
        (997, 5120),
        (2048, 6144),
        (2303, cached_tokens),
        (0, cached_tokens),
    )
    churn_steps = ((2303, cached_tokens), (0, cached_tokens))
    replay_steps = warmup_steps + churn_steps * stress_replay_pairs
    unique_steps = tuple(dict.fromkeys(replay_steps))
    sources = {step: make_source(*step) for step in unique_steps}
    npu_sources = {step: source.npu() for step, source in sources.items()}

    capture_source = sources[warmup_steps[0]]
    values = capture_source.npu()
    boundaries = torch.full(
        (mtp,),
        activation_boundary,
        dtype=torch.int32,
        device="npu",
    )
    row_requests = torch.zeros(mtp, dtype=torch.int32, device="npu")
    request_states = torch.zeros(requests, dtype=torch.int32, device="npu")
    request_generations = torch.ones(requests, dtype=torch.int64, device="npu")
    block_table_cpu = torch.arange(capacity // block_size, dtype=torch.int32).reshape(requests, -1)
    block_table = block_table_cpu.npu()
    workspace = allocate_sorted_resident_workspace(requests, mtp, device=torch.device("npu"))
    state = allocate_sorted_resident_state(requests, requests, mtp, device=torch.device("npu"))

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
            block_size=block_size,
        )

    # Reuse captured tensor addresses and queue every replay before the first
    # sync, retaining the asynchronous failure pattern from the server log.
    state.counts.zero_()
    state.generations.fill_(-1)
    reference: dict[int, int] = {}
    expected_misses: list[int] = []
    expected_slots: list[int] = []
    final_source = capture_source
    for step in replay_steps:
        source = sources[step]
        boundary = step[1]
        expected_shards = _expected_shards(
            source,
            torch.full((mtp,), boundary, dtype=torch.int32),
            requests,
            mtp,
            shard_count,
        )[0]
        reference, expected_misses, expected_slots = _reference_step(
            expected_shards,
            reference,
            capacity,
        )
        values.copy_(npu_sources[step])
        boundaries.fill_(boundary)
        graph.replay()
        final_source = source

    torch.npu.synchronize()

    assert _state_dict(state, 0, shard_count) == reference
    miss_count = int(workspace.miss_counts[0, 0].cpu())
    assert miss_count == len(expected_misses)
    assert workspace.miss_tokens[0, :miss_count].cpu().tolist() == expected_misses
    expected_targets = [
        int(block_table_cpu[0, slot // block_size]) * block_size + slot % block_size for slot in expected_slots
    ]
    assert workspace.target_slots[0, :miss_count].cpu().tolist() == expected_targets
    remapped = values.reshape(-1).cpu().tolist()
    assert remapped == [
        reference[token] if 0 <= token < cached_tokens else token for token in final_source.reshape(-1).tolist()
    ]
