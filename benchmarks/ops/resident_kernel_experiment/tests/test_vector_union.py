"""Host formulation and native output checks for vector union/deduplication."""
import bisect
import random

import pytest
import torch
from resident_experiment import assert_result, make_case, reference


def vector_model(rows, boundaries, shard, shards):
    selected = sorted(t for row, boundary in zip(rows, boundaries)
                      for t in row if 0 <= t < boundary and t % shards == shard)
    keys = [t for i, t in enumerate(selected) if i == 0 or t != selected[i - 1]]
    maps = []
    for row, boundary in zip(rows, boundaries):
        low = torch.zeros(len(row), dtype=torch.int32)
        high = torch.full_like(low, len(keys))
        source = torch.tensor(row, dtype=torch.int32)
        if keys:
            union = torch.tensor(keys, dtype=torch.int32)
            remaining = len(keys)
            while remaining:
                mid = (low + high) >> 1
                candidate = union[mid.clamp_max(len(keys) - 1).long()]
                less = ~(torch.minimum(candidate, source) == source)
                low = torch.where(less, (mid + 1).clamp_max(len(keys)), low)
                high = torch.where(less, high, mid)
                remaining >>= 1
            valid = (union[low.clamp_max(len(keys) - 1).long()] == source) & (source < boundary)
            maps.append(torch.where(valid, low, -1).tolist())
        else:
            maps.append([-1] * len(row))
    return keys, maps


@pytest.mark.parametrize("count", [0, 1, 2, 31, 32, 33, 511, 512, 513, 2048, 4096])
@pytest.mark.parametrize("shards", [1, 2, 8])
def test_vector_lower_bound_matches_independent_bisect(count, shards):
    rng = random.Random(31)
    first = list(range(count))
    rng.shuffle(first)
    second = first[::-1] + [-3, -1, count, count + 100]
    # Different boundaries ensure a key selected in one row cannot authorize
    # a hit beyond another row's boundary.
    rows, boundaries = (first, second), (count, max(0, count - 7))
    for shard in range(shards):
        keys, maps = vector_model(rows, boundaries, shard, shards)
        expected_keys = sorted({t for row, bound in zip(rows, boundaries)
                                for t in row if 0 <= t < bound and t % shards == shard})
        assert keys == expected_keys
        for row, bound, ranks in zip(rows, boundaries, maps):
            expected = [bisect.bisect_left(keys, t) if 0 <= t < bound and t % shards == shard else -1 for t in row]
            assert ranks == expected


@pytest.mark.parametrize("variant", ["baseline", "vector_union"])
@pytest.mark.parametrize("overlap", [0, 1024, 2048])
@pytest.mark.parametrize("scenario", ["normal", "zero_boundary", "invalid_indices", "padding", "skewed"])
def test_union_dedup_probe_matches_union_and_inverse_map(native, variant, overlap, scenario):
    initial = make_case(2, 2, 4, scenario=scenario, overlap=overlap)
    case = initial.clone(native)
    case.run(variant, "union_dedup")
    torch.npu.synchronize()
    result = case.clone("cpu")
    for request in range(initial.requests):
        rows = initial["topk"][request * 2:request * 2 + 2, 0].tolist()
        bounds = initial["boundary"][request * 2:request * 2 + 2].tolist()
        for q in range(2):
            if int(initial["row_requests"][request * 2 + q]) != request:
                bounds[q] = 0
        for shard in range(initial.shards):
            keys, maps = vector_model(rows, bounds, shard, initial.shards)
            count = int(result["shard_counts"][request, shard, 0])
            assert count == len(keys)
            assert result["packed"][request, shard, :count].tolist() == keys
            assert result["mapping"][request, shard].tolist() == maps[0] + maps[1]
    # Prefix probes deliberately stop before resident-state intersection.
    for name in ("state_tokens", "state_slots", "state_counts", "state_generations"):
        assert torch.equal(initial[name], result[name])


@pytest.mark.parametrize("variant", ["baseline", "vector_union"])
def test_sort_probe_runs_without_publishing_resident_state(native, variant):
    initial = make_case(1, 2, 4, scenario="generation", overlap=0)
    case = initial.clone(native)
    case.run(variant, "union_sort")
    torch.npu.synchronize()
    result = case.clone("cpu")
    raw = initial["topk"].flatten().tolist()
    for shard in range(initial.shards):
        selected = sorted(t for t in raw if t % initial.shards == shard)
        assert int(result["shard_counts"][0, shard, 0]) == len(selected)
        # The prefix probe writes the beginning of sorted (negative key,
        # original position) pairs; these are diagnostics, not a valid union.
        keys = result["packed"][0, shard, :16].view(torch.float32)[::2]
        assert keys.tolist() == [-float(t) for t in selected[:8]]
    for name in ("state_tokens", "state_slots", "state_counts", "state_generations"):
        assert torch.equal(initial[name], result[name])


@pytest.mark.parametrize('mtp', (1, 2))
@pytest.mark.parametrize('selected', (1, 31, 32, 33, 63, 64, 65, 95, 96, 97))
def test_small_shards_cover_compare_repeat_boundaries(native, mtp, selected):
    initial = make_case(1, mtp, 1, .9)
    initial['topk'].fill_(-1)
    initial['topk'].view(-1)[:selected] = torch.arange(selected, dtype=torch.int32)*initial.shards + 10000
    expected, _ = reference(initial)
    for variant in ('baseline', 'vector_union', 'vector_intersection'):
        device_case = initial.clone(native)
        device_case.run(variant)
        torch.npu.synchronize()
        assert_result(device_case, expected)
