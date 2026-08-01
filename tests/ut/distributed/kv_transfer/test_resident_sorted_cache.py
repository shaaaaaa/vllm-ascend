import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sorted_cache import (
    allocate_sorted_resident_state,
    allocate_sorted_resident_workspace,
    resident_shard_count,
    sorted_resident_workspace_prefix,
)


@pytest.mark.parametrize(
    "mtp,capacity,shard_count,shard_count_override",
    [
        (1, 2048, 2, None),
        (1, 2048, 1, 1),
        (1, 2048, 4, 4),
        (2, 4096, 4, None),
        (2, 4096, 2, 2),
        (2, 4096, 8, 8),
    ],
)
def test_sorted_resident_allocations_are_cacheline_partitioned(
    mtp, capacity, shard_count, shard_count_override
):
    requests = 3
    state = allocate_sorted_resident_state(
        requests,
        requests,
        mtp,
        device=torch.device("cpu"),
        shard_count=shard_count_override,
    )
    workspace = allocate_sorted_resident_workspace(
        requests,
        mtp,
        device=torch.device("cpu"),
        shard_count=shard_count_override,
    )

    assert state.tokens.shape == (
        2 * requests,
        shard_count,
        capacity,
    )
    assert state.slots.shape == state.tokens.shape
    assert state.counts.shape == (
        2 * requests,
        shard_count,
        16,
    )
    assert state.generations.shape == (2 * requests, 8)
    assert state.dummy_state_base == requests
    assert workspace.shard_packed.shape == (
        requests,
        shard_count,
        capacity,
    )
    assert workspace.shard_mapping.shape == workspace.shard_packed.shape
    assert workspace.shard_mapping.dtype == torch.int16
    assert workspace.prior_slots.shape == workspace.shard_packed.shape
    assert workspace.shard_miss_tokens.shape == workspace.shard_packed.shape
    assert workspace.shard_miss_tokens.dtype == torch.int32
    assert workspace.shard_miss_positions.shape == workspace.shard_packed.shape
    assert workspace.shard_miss_positions.dtype == torch.int16
    assert workspace.shard_evictable_slots.shape == workspace.shard_packed.shape
    assert workspace.shard_evictable_slots.dtype == torch.int16
    assert not hasattr(workspace, "assigned_slots")
    assert not hasattr(workspace, "overwritten_slots")
    assert workspace.shard_counts.shape == (requests, shard_count, 16)
    assert workspace.miss_counts.shape == (requests, 16)
    assert state.tokens.data_ptr() % 64 == 0
    assert state.slots.data_ptr() % 64 == 0
    assert state.counts.data_ptr() % 64 == 0
    assert workspace.shard_packed.data_ptr() % 64 == 0
    assert workspace.shard_mapping.data_ptr() % 64 == 0
    assert workspace.shard_counts.data_ptr() % 64 == 0
    assert state.tokens.stride(1) * state.tokens.element_size() % 64 == 0
    assert state.slots.stride(1) * state.slots.element_size() % 64 == 0
    assert workspace.shard_mapping.stride(1) * workspace.shard_mapping.element_size() % 64 == 0
    assert workspace.prior_slots.stride(1) * workspace.prior_slots.element_size() % 64 == 0
    assert workspace.shard_miss_tokens.stride(1) * workspace.shard_miss_tokens.element_size() % 64 == 0
    assert workspace.shard_miss_positions.stride(1) * workspace.shard_miss_positions.element_size() % 64 == 0
    assert workspace.shard_evictable_slots.stride(1) * workspace.shard_evictable_slots.element_size() % 64 == 0


def test_sorted_resident_uses_strict_next_power_of_two():
    assert resident_shard_count(1) == 2
    assert resident_shard_count(2) == 4


def test_sorted_resident_rejects_unsupported_mtp_and_state_size():
    with pytest.raises(ValueError, match="MTP=1 or MTP=2"):
        allocate_sorted_resident_workspace(1, 3, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="cover every active request"):
        allocate_sorted_resident_state(
            1,
            2,
            1,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="MTP=1 or MTP=2"):
        resident_shard_count(3)
    with pytest.raises(ValueError, match="power of two from 1 to 8"):
        allocate_sorted_resident_workspace(
            1,
            2,
            device=torch.device("cpu"),
            shard_count=16,
        )


@pytest.mark.parametrize("mtp", [1, 2])
def test_sorted_resident_workspace_prefix_keeps_fixed_storage(mtp):
    workspace = allocate_sorted_resident_workspace(
        4,
        mtp,
        device=torch.device("cpu"),
    )
    active = sorted_resident_workspace_prefix(workspace, 2)

    assert active.shard_packed.shape[0] == 2
    assert active.miss_tokens.shape[0] == 2
    assert active.target_slots.shape[0] == 2
    assert active.shard_packed.data_ptr() == workspace.shard_packed.data_ptr()
    assert active.miss_tokens.data_ptr() == workspace.miss_tokens.data_ptr()
    assert active.target_slots.data_ptr() == workspace.target_slots.data_ptr()
    assert active.shard_packed.is_contiguous()
    assert active.prior_slots.data_ptr() % 64 == 0
    assert active.shard_miss_tokens.data_ptr() == workspace.shard_miss_tokens.data_ptr()
    assert active.shard_miss_positions.data_ptr() == workspace.shard_miss_positions.data_ptr()
    assert active.shard_evictable_slots.data_ptr() == workspace.shard_evictable_slots.data_ptr()

    with pytest.raises(ValueError, match="out of range"):
        sorted_resident_workspace_prefix(workspace, 0)
    with pytest.raises(ValueError, match="out of range"):
        sorted_resident_workspace_prefix(workspace, 5)
