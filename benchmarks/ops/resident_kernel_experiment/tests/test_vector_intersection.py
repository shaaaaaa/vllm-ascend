"""Exact union outputs and scalar fallback for the intersection-only variant."""
import random

import pytest
import torch

from generate_sources import HERE, KERNELS, SOURCE, generate
from resident_experiment import _put_state, assert_result, make_case, reference


def search_masks(query, target):
    if not len(query):
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)
    low = torch.zeros_like(query)
    high = torch.full_like(query, len(target))
    if not len(target):
        return low.long(), torch.zeros(len(query), dtype=torch.bool)
    remaining = len(target)
    while remaining:
        mid = (low + high) >> 1
        candidate = target[mid.clamp_max(len(target)-1).long()]
        less = ~(torch.minimum(candidate, query) == query)
        low = torch.where(less, (mid+1).clamp_max(len(target)), low)
        high = torch.where(less, high, mid)
        remaining >>= 1
    position = low.clamp_max(len(target)-1).long()
    return position, target[position] == query


@pytest.mark.parametrize('old_n', (0, 1, 31, 32, 33, 511, 512, 513, 2047, 2048, 2049, 4096))
@pytest.mark.parametrize('current_n', (0, 1, 33, 512, 2048, 4096))
def test_two_searches_preserve_exact_order(old_n, current_n):
    rng = random.Random(old_n*7001+current_n)
    old = sorted(rng.sample(range(15000), old_n))
    current = sorted(rng.sample(range(15000), current_n))
    slots = rng.sample(range(old_n), old_n)
    resident = dict(zip(old, slots, strict=True))
    current_set = set(current)
    expected = ([resident.get(t, -1) for t in current],
                [t for t in current if t not in resident],
                [i for i, t in enumerate(current) if t not in resident],
                [resident[t] for t in old if t not in current_set])
    a, b, s = (torch.tensor(x, dtype=torch.int32) for x in (current, old, slots))
    indices, hits = search_masks(a, b)
    _, selected = search_masks(b, a)
    prior = torch.where(hits, s[indices], -1) if len(b) else torch.full_like(a, -1)
    assert (prior.tolist(), a[~hits].tolist(), torch.arange(len(a))[~hits].tolist(),
            s[~selected].tolist()) == expected


def test_codegen_enables_only_intersection_and_reuses_original_arguments():
    code = generate(SOURCE.read_text(encoding='utf-8'), HERE/'launch.h', variants=(('intersection', 0),),
                    kernels={'union': KERNELS['union']}, vector_intersection=True)['intersection_union.cpp']
    assert code.count('extern "C" __global__ __aicore__ void') == 1
    assert '#define RESIDENT_EXPERIMENT_VECTOR_INTERSECTION 1' in code
    assert '#define RESIDENT_EXPERIMENT_VECTOR_UNION 0' in code
    assert '#define RESIDENT_EXPERIMENT_SKIP_UNCHANGED 0' in code


def test_supported_integer_less_comparison_is_exact_at_int32_extremes():
    values = torch.tensor([-2**31, -16777217, -1, 0, 1, 16777216, 16777217, 2**31-1], dtype=torch.int32)
    a, b = torch.broadcast_tensors(values[:, None], values[None, :])
    assert torch.equal(~(torch.minimum(a, b) == b), a < b)


def test_experimental_searches_use_only_supported_a2_integer_compare_modes():
    source = SOURCE.read_text(encoding='utf-8')
    intersection = source.split('inline void IntersectionMasks(', 1)[1].split('inline void VectorIntersection(', 1)[0]
    inverse = source.split('// 910B has no local vector Scatter.', 1)[1].split('inline uint32_t SortElementCount(', 1)[0]
    for body in (intersection, inverse):
        assert 'AscendC::Min(' in body
        assert 'AscendC::Not(' in body
        assert 'CMPMODE::LT' not in body
        assert 'CMPMODE::NE' not in body
    assert '(count + 63U) & ~63U' in source


def assert_union_equal(actual, expected):
    actual, expected = actual.clone('cpu'), expected.clone('cpu')
    for name in ('topk', 'mapping', 'shard_counts', 'state_counts', 'state_generations'):
        assert torch.equal(actual[name], expected[name]), name
    for r in range(actual.requests):
        for shard in range(actual.shards):
            current, misses, evictable = expected['shard_counts'][r, shard, :3].tolist()
            for name, count in (('packed', current), ('prior_slots', current), ('shard_miss_tokens', misses),
                                ('shard_miss_positions', misses), ('evictable_slots', evictable)):
                assert torch.equal(actual[name][r, shard, :count], expected[name][r, shard, :count]), name


@pytest.mark.parametrize('scenario', ('normal', 'cold', 'generation', 'padding', 'inactive', 'zero_boundary',
                                     'subset', 'invalid_indices', 'one_shard_miss', 'skewed'))
@pytest.mark.parametrize('mtp', (1, 2))
def test_union_contract_before_finalize(native, scenario, mtp):
    initial = make_case(2, mtp, 4, .9, scenario)
    baseline, candidate = initial.clone(native), initial.clone(native)
    baseline.run('baseline', 'union')
    candidate.run('vector_intersection', 'union')
    torch.npu.synchronize()
    assert_union_equal(candidate, baseline)


@pytest.mark.parametrize('current_n,old_n', ((0, 0), (0, 513), (1, 0), (31, 32), (32, 31), (33, 33),
                                         (511, 513), (513, 511), (2047, 2048), (2048, 2047),
                                         (2049, 512), (512, 2049), (4096, 4096)))
def test_exact_counts_and_large_shard_fallback(native, current_n, old_n):
    initial = make_case(1, 2, 1, 0)
    initial['topk'].fill_(-1)
    initial['topk'].view(-1)[:current_n] = torch.arange(current_n, dtype=torch.int32)*2+2
    initial['state_counts'].zero_()
    row = int(initial['request_states'][0])
    slots = torch.randperm(old_n, generator=torch.Generator().manual_seed(31)).tolist()
    _put_state(initial, row, dict(zip(range(0, old_n*4, 4), slots, strict=True)))
    baseline, candidate = initial.clone(native), initial.clone(native)
    baseline.run('baseline', 'union')
    candidate.run('vector_intersection', 'union')
    torch.npu.synchronize()
    assert_union_equal(candidate, baseline)
    candidate = initial.clone(native)
    candidate.run('vector_intersection')
    torch.npu.synchronize()
    expected, _ = reference(initial)
    assert_result(candidate, expected)
