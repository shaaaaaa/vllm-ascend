# SPDX-License-Identifier: Apache-2.0
"""Stateful DCP inputs and independent policy expectations; CPU-only import."""

import hashlib
import random

import torch
from resident_experiment import Case, make_case, membership_reference, reference

TOPK = 2048
CAPACITY = 4096
BLOCK = 128


def make_trace(
    requests=1,
    steps=128,
    overlap=0.9,
    query_overlap=0.5,
    working_set=32768,
    history=65536,
    seed=17,
    workload="reuse",
    distribution="uniform",
    world=8,
    rank=0,
):
    """Exact per-query adjacent-step overlap; rank-local overlap is measured.

    Disjoint common/target-only/draft-only pools keep within-step overlap fixed.
    'reuse' can revisit older selections; 'fresh' never reintroduces a dropped
    token. Neither specifies or manipulates the resulting resident hit rate.
    """
    if requests < 1 or steps < 2 or not 0 <= overlap <= 1 or not 0 <= query_overlap <= 1:
        raise ValueError("positive requests, steps >= 2 and overlaps in [0,1] required")
    if world not in (2, 4, 8) or not 0 <= rank < world or not 1 <= history <= 1048576:
        raise ValueError("DCP size 2/4/8, a valid rank, and history <= 1M required")
    if workload not in ("reuse", "fresh") or distribution not in ("uniform", "owner-skew", "shard-skew"):
        raise ValueError("unknown workload/distribution")
    common = int(TOPK * query_overlap)
    kept = int(TOPK * overlap)
    common_kept = int(common * overlap)
    sizes = [common, TOPK - common, TOPK - common]
    keep = [common_kept, kept - common_kept, kept - common_kept]
    available = [
        t
        for t in range(history)
        if (distribution != "owner-skew" or t // BLOCK % world == rank) and (distribution != "shard-skew" or t % 8 == 0)
    ]
    universe_size = working_set if workload == "reuse" else len(available)
    if not sum(sizes) <= universe_size <= len(available):
        raise ValueError("working set/history is too small for the selected distribution")
    pool_sizes = [universe_size * n // sum(sizes) for n in sizes]
    pool_sizes[sizes.index(max(sizes))] += universe_size - sum(pool_sizes)
    needed = [n + (n - k) * (1 if workload == "reuse" else steps - 1) for n, k in zip(sizes, keep)]
    if any(n > p for n, p in zip(needed, pool_sizes)):
        raise ValueError("increase working-set/history or reduce steps: insufficient distinct replacement tokens")
    trace = torch.empty((steps, requests * 2, 1, TOPK), dtype=torch.int32)
    for request in range(requests):
        rng = random.Random(seed + request)
        universe = rng.sample(available, universe_size)
        pools, offset = [], 0
        for size in pool_sizes:
            pools.append(universe[offset : offset + size])
            offset += size
        current = [rng.sample(pool, n) for pool, n in zip(pools, sizes)]
        used = [set(row) for row in current]
        for step in range(steps):
            if step:
                for category, pool in enumerate(pools):
                    excluded = used[category] if workload == "fresh" else set(current[category])
                    replacements = rng.sample([t for t in pool if t not in excluded], sizes[category] - keep[category])
                    current[category] = rng.sample(current[category], keep[category]) + replacements
                    used[category].update(replacements)
            for query in range(2):
                row = current[0] + current[query + 1]
                rng.shuffle(row)
                trace[step, request * 2 + query, 0] = torch.tensor(row, dtype=torch.int32)
    return trace


def trace_metadata(trace, world, rank):
    overlaps, local_overlaps, query_overlaps, owned_counts = [], [], [], []
    union_overlaps, local_union_overlaps = [], []
    previous, previous_owned = [], []
    previous_unions, previous_local_unions = [], []
    for step in range(len(trace)):
        rows = [set(row) for row in trace[step, :, 0].tolist()]
        owned = [{t for t in row if t // BLOCK % world == rank} for row in rows]
        unions = [rows[q] | rows[q + 1] for q in range(0, len(rows), 2)]
        local_unions = [owned[q] | owned[q + 1] for q in range(0, len(rows), 2)]
        owned_counts.extend(map(len, owned))
        query_overlaps.extend(len(rows[q] & rows[q + 1]) / TOPK for q in range(0, len(rows), 2))
        if step:
            overlaps.extend(len(a & b) / TOPK for a, b in zip(previous, rows))
            local_overlaps.extend(len(a & b) / len(b) for a, b in zip(previous_owned, owned) if b)
            union_overlaps.extend(len(a & b) / len(b) for a, b in zip(previous_unions, unions))
            local_union_overlaps.extend(len(a & b) / len(b) for a, b in zip(previous_local_unions, local_unions) if b)
        previous, previous_owned = rows, owned
        previous_unions, previous_local_unions = unions, local_unions

    def limits(values):
        return {
            "min": min(values, default=0),
            "max": max(values, default=0),
            "mean": sum(values) / len(values) if values else 0,
        }

    return {
        "sha256": hashlib.sha256(trace.numpy().tobytes()).hexdigest(),
        "temporal_overlap": limits(overlaps),
        "query_overlap": limits(query_overlaps),
        "rank_temporal_overlap": limits(local_overlaps),
        "rank_selected_per_query": limits(owned_counts),
        "union_temporal_overlap": limits(union_overlaps),
        "rank_union_temporal_overlap": limits(local_union_overlaps),
    }


def cold_case(requests, history):
    case = make_case(requests, mtp=2, scenario="cold")
    case["boundary"].fill_(history)
    # A real scratch reservation needs only these non-null physical pages.
    for r in range(requests):
        case["block_table"][r] = torch.arange(r * 32 + 1, r * 32 + 33, dtype=torch.int32).flip(0)
    case.tensors.extend(
        [torch.full((requests,), CAPACITY, dtype=torch.int32), torch.zeros((requests * 2, 16), dtype=torch.int32)]
    )
    return case


def dcp_reference(case: Case, world, rank, free_first):
    filtered = case.clone("cpu")
    counts = []
    for row in range(case.requests * 2):
        source = case["topk"][row, 0].tolist()
        if any(t >= int(case["boundary"][row]) for t in source if t >= 0):
            raise ValueError("this benchmark qualifies historical scratch, not live-tail mapping")
        tokens = [t for t in source if t >= 0 and t // BLOCK % world == rank]
        if len(tokens) != len(set(tokens)):
            raise ValueError("Top-K rows must have unique token IDs")
        filtered["topk"][row, 0] = torch.tensor(tokens + [-1] * (TOPK - len(tokens)), dtype=torch.int32)
        counts.append(len(tokens) if int(case["row_requests"][row]) == row // 2 else 0)
    result, _ = reference(filtered, free_first=free_first)
    result = membership_reference(filtered, result)
    result.tensors[21][:, 0] = torch.tensor(counts, dtype=torch.int32)
    return result


def step_stats(case):
    states = case["request_states"].long()
    return {
        "misses": case["miss_counts"][:, 0].tolist(),
        "resident": case["state_counts"][states, :, 0].sum(-1).tolist(),
        "union": case["shard_counts"][:, :, 0].sum(-1).tolist(),
        "max_resident_shard": case["state_counts"][states, :, 0].max(-1).values.tolist(),
    }


def summarize(records):
    misses = sum(sum(row["misses"]) for row in records)
    selected = sum(sum(row["union"]) for row in records)
    return {
        "total_misses": misses,
        "total_selected_union": selected,
        "resident_hit_rate": 1 - misses / selected if selected else 1,
        "final_resident_per_request": records[-1]["resident"],
        "estimated_h2d_bytes_per_latent_layer": misses * 576 * 2,
        "h2d_measured": False,
        "steps": records,
    }
