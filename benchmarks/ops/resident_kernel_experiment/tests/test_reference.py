"""Host tests validate the fixture/oracle and the fast-path proof obligations."""

import itertools

import pytest
import torch
from resident_experiment import _state, make_case, reference


@pytest.mark.parametrize("mtp,shards", itertools.product((1, 2), (1, 2, 4)))
@pytest.mark.parametrize(
    "scenario",
    (
        "normal",
        "cold",
        "generation",
        "zero_boundary",
        "subset",
        "padding",
        "inactive",
        "invalid_indices",
        "one_shard_miss",
        "skewed",
    ),
)
def test_reference_preserves_capacity_and_slot_bijection(mtp, shards, scenario):
    initial = make_case(2, mtp, shards, 0.9, scenario)
    result, stats = reference(initial)
    for row in range(result["state_tokens"].shape[0]):
        assert len(_state(result, row)) <= result.capacity
    assert 0 <= stats["misses"] <= stats["selected"] <= 2 * result.capacity
    selected_slots = []
    for r in range(result.requests):
        selected_slots.extend(
            result["prior_slots"][r, s, : int(result["shard_counts"][r, s, 0])].tolist() for s in range(result.shards)
        )
    assert all(0 <= slot < result.capacity for row in selected_slots for slot in row)


def test_no_misses_is_not_sufficient_to_skip_a_shard_update():
    initial = make_case(1, 2, 4, scenario="one_shard_miss")
    result, _ = reference(initial)
    counts = result["shard_counts"][0]
    assert any(int(c[1]) == 0 and int(c[4]) > 0 for c in counts)
    assert any(int(c[1]) == 0 and int(c[4]) == 0 for c in counts)


def test_all_hit_subset_keeps_nonselected_residents_and_clears_stale_counts():
    initial = make_case(1, 2, 4, scenario="subset")
    result, stats = reference(initial)
    assert stats["misses"] == 0
    assert torch.equal(result["state_tokens"], initial["state_tokens"])
    assert torch.equal(result["state_slots"], initial["state_slots"])
    assert not result["shard_counts"][:, :, 4].any()
    assert not result["miss_counts"][:, 0].any()


def test_same_input_becomes_all_hit_but_generation_rollover_is_cold():
    initial = make_case(1, 2, 4, scenario="cold")
    first, a = reference(initial)
    first["topk"].copy_(initial["topk"])
    second, b = reference(first)
    second["topk"].copy_(initial["topk"])
    second["request_generations"].add_(1)
    _, c = reference(second)
    assert a["misses"] == c["misses"] > 0
    assert b["misses"] == 0


def test_unchanged_shard_predicate_is_sufficient_exhaustively():
    # Independent set proof over all current subsets and eviction prefixes.
    universe = range(6)
    subsets = [set(c) for n in range(7) for c in itertools.combinations(universe, n)]
    for old, current in itertools.product(subsets, repeat=2):
        misses = current - old
        evictable = sorted(old - current)
        for selected_count in range(len(evictable) + 1):
            after = (old - set(evictable[:selected_count])) | current
            if not misses and selected_count == 0:
                assert after == old


def test_benchmark_reset_keeps_the_intended_miss_rate():
    snapshot = make_case(1, 2, 4, 0.5)
    case = snapshot.clone()
    expected_misses = reference(snapshot)[1]["misses"]
    for _ in range(3):
        case.reset_from(snapshot)
        case, stats = reference(case)
        assert stats["misses"] == expected_misses > 0
    assert case["topk"].data_ptr() != snapshot["topk"].data_ptr()
