"""Validate the packed-offset remap formulation and its UB capacity bound."""
import random

import pytest
import torch


@pytest.mark.parametrize("capacity", [2048, 4096])
@pytest.mark.parametrize("shards", [1, 2, 4, 8])
@pytest.mark.parametrize("distribution", ["balanced", "skewed", "empty", "gaps"])
def test_compact_remap_matches_per_shard_gather(capacity, shards, distribution):
    rng = random.Random(7)
    counts = [capacity // shards] * shards
    if distribution == "skewed":
        counts = [0] * (shards - 1) + [capacity]
    elif distribution == "empty":
        counts = [0] * shards
    elif distribution == "gaps":
        counts = [max(0, count - rng.randrange(17)) for count in counts]
    slots = torch.full((2 * capacity,), -1, dtype=torch.int16)
    offsets = torch.full((capacity,), -1, dtype=torch.int32)
    expected = torch.arange(capacity, dtype=torch.int32) + 10000
    end = 0
    positions = list(range(capacity))
    rng.shuffle(positions)
    selected = 0
    for count in counts:
        if count == 0:
            continue
        base = (end + 15) & ~15
        values = torch.tensor(rng.sample(range(capacity), count), dtype=torch.int16)
        slots[base:base + count] = values
        for position in positions[selected:selected + count]:
            rank = rng.randrange(count)  # duplicate selections share slots
            offsets[position] = base + rank
            expected[position] = int(values[rank])
        selected += count
        end = base + count
    assert end <= capacity + 15 * shards <= 2 * capacity
    actual = torch.arange(capacity, dtype=torch.int32) + 10000
    if end:
        actual = torch.where(offsets >= 0, slots[offsets.clamp_min(0).long()].int(), actual)
    assert torch.equal(actual, expected)
