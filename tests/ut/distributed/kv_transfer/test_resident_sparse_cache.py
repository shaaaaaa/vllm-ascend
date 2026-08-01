import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sparse_cache import (
    ResidentRequestStateRegistry,
    _resident_sparse_cache_reference,
    allocate_resident_workspace,
    prepare_resident_sparse_cache_,
    remap_union_positions_,
    validate_resident_shapes,
)


def _state(
    *,
    requests: int = 1,
    max_model_len: int = 64,
    capacity: int = 8,
):
    return (
        torch.full(
            (requests, max_model_len + 1), -1, dtype=torch.int16
        ),
        torch.full((requests, capacity + 1), -1, dtype=torch.int32),
    )


def _plan(
    values,
    counts,
    token_to_slot,
    slot_to_token,
    *,
    states=None,
    capacity=8,
):
    unions = torch.zeros(
        (len(values), capacity), dtype=torch.int32
    )
    for row, tokens in enumerate(values):
        unions[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    if states is None:
        states = torch.arange(len(values), dtype=torch.int32)
    return _resident_sparse_cache_reference(
        unions,
        torch.tensor(counts, dtype=torch.int32),
        states,
        token_to_slot,
        slot_to_token,
        scratch_capacity=capacity,
    )


def test_cold_plan_uses_lowest_slots_and_builds_only_misses():
    token_to_slot, slot_to_token = _state()
    union_to_slot, misses, miss_counts = _plan(
        [[7, 11, 19]],
        [3],
        token_to_slot,
        slot_to_token,
    )

    assert union_to_slot[0, :3].tolist() == [0, 1, 2]
    assert misses[0, :3].tolist() == [7, 11, 19]
    assert miss_counts.tolist() == [3]
    assert slot_to_token[0, :4].tolist() == [7, 11, 19, -1]
    assert [int(token_to_slot[0, token]) for token in (7, 11, 19)] == [
        0,
        1,
        2,
    ]


def test_all_hits_emit_empty_lmcache_payload():
    token_to_slot, slot_to_token = _state()
    _plan([[7, 11, 19]], [3], token_to_slot, slot_to_token)
    union_to_slot, _, miss_counts = _plan(
        [[19, 7, 11]],
        [3],
        token_to_slot,
        slot_to_token,
    )

    assert union_to_slot[0, :3].tolist() == [2, 0, 1]
    assert miss_counts.tolist() == [0]


def test_unselected_unoverwritten_token_can_hit_later():
    token_to_slot, slot_to_token = _state(capacity=4)
    _plan(
        [[20, 30, 40, 10]],
        [4],
        token_to_slot,
        slot_to_token,
        capacity=4,
    )
    # 10 remains in slot 3 because miss 50 replaces the lower unprotected
    # slot 2.
    _plan(
        [[20, 30, 50]],
        [3],
        token_to_slot,
        slot_to_token,
        capacity=4,
    )
    union_to_slot, _, counts = _plan(
        [[10, 20, 30]],
        [3],
        token_to_slot,
        slot_to_token,
        capacity=4,
    )

    assert counts.tolist() == [0]
    assert union_to_slot[0, :3].tolist() == [3, 0, 1]


def test_miss_replaces_lowest_unprotected_slot_and_invalidates_reverse_map():
    token_to_slot, slot_to_token = _state(capacity=4)
    _plan(
        [[10, 20, 30, 40]],
        [4],
        token_to_slot,
        slot_to_token,
        capacity=4,
    )
    union_to_slot, misses, counts = _plan(
        [[20, 30, 40, 50]],
        [4],
        token_to_slot,
        slot_to_token,
        capacity=4,
    )

    assert union_to_slot[0, :4].tolist() == [1, 2, 3, 0]
    assert misses[0, 0].item() == 50
    assert counts.tolist() == [1]
    assert token_to_slot[0, 10].item() == -1
    assert token_to_slot[0, 50].item() == 0
    assert slot_to_token[0, :4].tolist() == [50, 20, 30, 40]


def test_stale_reverse_entry_is_rejected_by_forward_map():
    token_to_slot, slot_to_token = _state(capacity=4)
    token_to_slot[0, 10] = 1
    slot_to_token[0, 1] = 99

    union_to_slot, _, counts = _plan(
        [[10]],
        [1],
        token_to_slot,
        slot_to_token,
        capacity=4,
    )

    assert counts.tolist() == [1]
    assert union_to_slot[0, 0].item() == 0
    assert slot_to_token[0, 0].item() == 10


def test_dummy_state_is_cold_and_does_not_mutate_persistent_maps():
    token_to_slot, slot_to_token = _state(capacity=4)
    before_reverse = token_to_slot.clone()
    before_forward = slot_to_token.clone()
    union_to_slot, misses, counts = _plan(
        [[3, 5]],
        [2],
        token_to_slot,
        slot_to_token,
        states=torch.tensor([-1], dtype=torch.int32),
        capacity=4,
    )

    assert union_to_slot[0, :2].tolist() == [0, 1]
    assert misses[0, :2].tolist() == [3, 5]
    assert counts.tolist() == [2]
    assert torch.equal(token_to_slot, before_reverse)
    assert torch.equal(slot_to_token, before_forward)


def test_requests_have_independent_state_rows():
    token_to_slot, slot_to_token = _state(requests=2, capacity=4)
    union_to_slot, _, counts = _plan(
        [[10, 20], [20, 30]],
        [2, 2],
        token_to_slot,
        slot_to_token,
        capacity=4,
    )

    assert union_to_slot[:, :2].tolist() == [[0, 1], [0, 1]]
    assert counts.tolist() == [2, 2]
    assert slot_to_token[:, :2].tolist() == [[10, 20], [20, 30]]


def test_position_mapping_preserves_non_union_absolute_indices():
    topk = torch.tensor([[100, 200, 300, 400]], dtype=torch.int32)
    position_to_union = torch.tensor([[2, -1, 0, -1]], dtype=torch.int32)
    union_to_slot = torch.tensor([[7, 3, 5, -1]], dtype=torch.int32)

    remap_union_positions_(topk, position_to_union, union_to_slot)
    assert topk.tolist() == [[5, 200, 7, 400]]


def test_shape_validation_requires_int16_and_one_sentinel():
    token_to_slot, slot_to_token = _state(capacity=4)
    validate_resident_shapes(token_to_slot, slot_to_token, 4)

    with pytest.raises(TypeError, match="int16"):
        validate_resident_shapes(token_to_slot.int(), slot_to_token, 4)
    with pytest.raises(ValueError, match="sentinel"):
        validate_resident_shapes(
            token_to_slot,
            torch.empty((1, 4), dtype=torch.int32),
            4,
        )
    with pytest.raises(ValueError, match="32768"):
        validate_resident_shapes(token_to_slot, slot_to_token, 32768)


def test_request_registry_keeps_unscheduled_state_and_tracks_block_changes():
    registry = ResidentRequestStateRegistry(2)
    states, generations = registry.bind(["a", "b"], [("x",), ("y",)])
    a_state, b_state = states.tolist()
    a_generation, b_generation = generations.tolist()

    states, generations = registry.bind(["b"], [("y",)])
    assert states.tolist() == [b_state]
    assert generations.tolist() == [b_generation]

    states, generations = registry.bind(["a"], [("z",)])
    assert states.tolist() == [a_state]
    assert generations[0] == a_generation + 1


def test_request_registry_releases_and_generates_new_owner():
    registry = ResidentRequestStateRegistry(1)
    first_state, first_generation = registry.bind(["a"], [("x",)])
    registry.release(["a"])
    second_state, second_generation = registry.bind(["b"], [("x",)])

    assert second_state.tolist() == first_state.tolist()
    assert second_generation[0] > first_generation[0]


def test_request_registry_invalidate_keeps_row_and_makes_state_cold():
    registry = ResidentRequestStateRegistry(1)
    first_state, first_generation = registry.bind(["a"], [("x",)])
    registry.invalidate(["a", "not-bound"])
    second_state, second_generation = registry.bind(["a"], [("x",)])

    assert second_state.tolist() == first_state.tolist()
    assert second_generation[0] == first_generation[0] + 1


def _native_state(*, requests=1, capacity=8, token_stride=64):
    # The second half contains cacheline-private graph-dummy sinks.
    token_to_slot = torch.full(
        (2 * requests, token_stride), -1, dtype=torch.int16
    )
    slot_to_token = torch.full(
        (2 * requests, capacity + 1), -1, dtype=torch.int32
    )
    generations = torch.full(
        (2 * requests, 8), -1, dtype=torch.int64
    )
    workspace = allocate_resident_workspace(
        requests, capacity, device=torch.device("cpu")
    )
    return token_to_slot, slot_to_token, generations, workspace


def _native_plan(
    values,
    *,
    token_to_slot,
    slot_to_token,
    generations,
    workspace,
    request_generations,
    states=None,
    capacity=8,
):
    requests = len(values)
    selected = torch.zeros((requests, capacity), dtype=torch.int32)
    counts = torch.zeros((requests, 16), dtype=torch.int32)
    mapping = torch.full((requests, capacity), -1, dtype=torch.int32)
    topk = torch.full((requests, capacity), 60, dtype=torch.int32)
    for row, tokens in enumerate(values):
        count = len(tokens)
        selected[row, :count] = torch.tensor(tokens, dtype=torch.int32)
        counts[row, 0] = count
        mapping[row, :count] = torch.arange(count, dtype=torch.int32)
        topk[row, :count] = torch.arange(count, dtype=torch.int32)
    if states is None:
        states = torch.arange(requests, dtype=torch.int32)
    targets = torch.empty((requests, capacity), dtype=torch.long)
    block_table = (
        torch.arange(requests * capacity, dtype=torch.int32)
        .reshape(requests, capacity)
        + 100
    )
    prepare_resident_sparse_cache_(
        topk,
        mapping,
        selected,
        counts,
        targets,
        block_table,
        states,
        torch.tensor(request_generations, dtype=torch.int64),
        token_to_slot,
        slot_to_token,
        generations,
        workspace,
        block_size=1,
        scratch_capacity=capacity,
    )
    return topk, selected, counts[:, 0], targets


def test_native_planner_cold_hits_and_lowest_unprotected_eviction():
    state = _native_state(capacity=4)
    topk, selected, counts, targets = _native_plan(
        [[10, 20, 30, 40]],
        token_to_slot=state[0],
        slot_to_token=state[1],
        generations=state[2],
        workspace=state[3],
        request_generations=[1],
        capacity=4,
    )
    assert topk.tolist() == [[0, 1, 2, 3]]
    assert selected.tolist() == [[10, 20, 30, 40]]
    assert counts.tolist() == [4]
    assert targets[0].tolist() == [100, 101, 102, 103]

    topk, selected, counts, targets = _native_plan(
        [[20, 30, 40, 50]],
        token_to_slot=state[0],
        slot_to_token=state[1],
        generations=state[2],
        workspace=state[3],
        request_generations=[1],
        capacity=4,
    )
    assert topk.tolist() == [[1, 2, 3, 0]]
    assert selected[0, :1].tolist() == [50]
    assert counts.tolist() == [1]
    assert targets[0, 0].item() == 100
    assert state[0][0, 10].item() == -1
    assert state[1][0, :4].tolist() == [50, 20, 30, 40]


def test_native_planner_uses_request_major_block_table_rows():
    state = _native_state(requests=2, capacity=4)
    assert torch.count_nonzero(state[3].miss_slot_payload).item() == 0
    _, _, counts, targets = _native_plan(
        [[10, 20], [30, 40]],
        token_to_slot=state[0],
        slot_to_token=state[1],
        generations=state[2],
        workspace=state[3],
        request_generations=[1, 1],
        capacity=4,
    )

    assert counts.tolist() == [2, 2]
    assert targets[0, :2].tolist() == [100, 101]
    assert targets[1, :2].tolist() == [104, 105]
    assert torch.all(state[3].miss_slot_payload[:, :4] >= 0)
    assert torch.all(state[3].miss_slot_payload[:, :4] < 4)


def test_native_planner_generation_change_rejects_all_old_slots():
    state = _native_state(capacity=4)
    _native_plan(
        [[10, 20]],
        token_to_slot=state[0],
        slot_to_token=state[1],
        generations=state[2],
        workspace=state[3],
        request_generations=[1],
        capacity=4,
    )
    _, selected, counts, _ = _native_plan(
        [[10]],
        token_to_slot=state[0],
        slot_to_token=state[1],
        generations=state[2],
        workspace=state[3],
        request_generations=[2],
        capacity=4,
    )

    assert counts.tolist() == [1]
    assert selected[0, 0].item() == 10
    assert state[1][0, :4].tolist() == [10, -1, -1, -1]


def test_native_dummy_plan_does_not_touch_real_state_rows():
    state = _native_state(capacity=4)
    real_reverse = state[0][0].clone()
    real_forward = state[1][0].clone()
    _, selected, counts, _ = _native_plan(
        [[10, 20]],
        token_to_slot=state[0],
        slot_to_token=state[1],
        generations=state[2],
        workspace=state[3],
        request_generations=[-1],
        states=torch.tensor([-1], dtype=torch.int32),
        capacity=4,
    )

    assert selected[0, :2].tolist() == [10, 20]
    assert counts.tolist() == [2]
    assert torch.equal(state[0][0], real_reverse)
    assert torch.equal(state[1][0], real_forward)
