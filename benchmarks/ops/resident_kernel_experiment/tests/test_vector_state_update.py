"""Exact state retention and merge-path formulation; native tests added below."""
import random

import pytest
import torch
from resident_experiment import assert_result, make_case, reference
from generate_sources import HERE, KERNELS, SOURCE, generate


def merge_path(a, b):
    if not len(a):
        return [(1, j) for j in range(len(b))]
    if not len(b):
        return [(0, i) for i in range(len(a))]
    k = torch.arange(len(a)+len(b))
    low, high = (k-len(b)).clamp_min(0), k.clamp_max(len(a))
    x, y = torch.tensor(a), torch.tensor(b)
    remaining = len(a)
    while remaining:
        i = (low+high) >> 1
        j = k-i
        av, bv = x[i.clamp_max(len(a)-1)], y[(j-1).clamp(0, len(b)-1)]
        small = (i < len(a)) & (j > 0) & ~(torch.minimum(av, bv) == bv)
        low = torch.where(small, i+1, low)
        high = torch.where(small, high, i)
        remaining >>= 1
    i, j = low, k-low
    av, bv = x[i.clamp_max(len(a)-1)], y[j.clamp_max(len(b)-1)]
    choose_a = (i < len(a)) & ((j >= len(b)) | (torch.minimum(av, bv) == av))
    return [(0, int(i[t])) if choose_a[t] else (1, int(j[t])) for t in range(len(k))]


def vector_state_model(old, old_slots, current, current_slots, evictions):
    current_set = set(current)
    absent = [(t, s) for t, s in zip(old, old_slots, strict=True) if t not in current_set][evictions:]
    survivor = [t for t, _ in absent]
    return [(current[i], current_slots[i]) if side == 0 else absent[i] for side, i in merge_path(current, survivor)]


@pytest.mark.parametrize('old_n,current_n', ((0, 0), (0, 1), (1, 0), (31, 33), (64, 63),
                                           (511, 513), (1024, 1024), (2048, 0), (0, 2048)))
@pytest.mark.parametrize('evict_fraction', (0, .5, 1))
def test_vector_state_matches_exact_retention(old_n, current_n, evict_fraction):
    rng = random.Random(old_n*9871+current_n)
    old = sorted(rng.sample(range(8192), old_n))
    current = sorted(rng.sample(range(8192), current_n))
    slots = list(range(old_n))
    rng.shuffle(slots)
    prior = dict(zip(old, slots, strict=True))
    assigned = [prior.get(t, old_n+i) for i, t in enumerate(current)]
    absent = [t for t in old if t not in set(current)]
    evictions = int(len(absent)*evict_fraction)
    expected = {t: s for t, s in zip(old, slots, strict=True) if t not in set(absent[:evictions])}
    expected.update(zip(current, assigned, strict=True))
    assert vector_state_model(old, slots, current, assigned, evictions) == sorted(expected.items())


def test_merge_path_keeps_exact_large_integer_keys():
    a = [-2147483648, -1, 16777217, 2147483647]
    b = [-2147483647, 0, 16777216, 2147483646]
    assert [(a if side == 0 else b)[index] for side, index in merge_path(a, b)] == sorted(a+b)


def test_codegen_keeps_state_update_and_probes_separate():
    source = SOURCE.read_text(encoding='utf-8')
    for stage, entry in (('update', KERNELS['update']), ('state_update', 'dsa_resident_sorted_state_update_kernel')):
        code = generate(source, HERE/'launch.h', variants=(('state', 0),),
                        kernels={stage: entry}, vector_state=True)[f'state_{stage}.cpp']
        assert '#define RESIDENT_EXPERIMENT_VECTOR_STATE 1' in code
        assert '#define RESIDENT_EXPERIMENT_SKIP_UNCHANGED 0' in code
        assert '#define RESIDENT_EXPERIMENT_VECTOR_INTERSECTION 0' in code
        assert code.count('extern "C" __global__ __aicore__ void') == 1


@pytest.mark.parametrize('scenario', ('normal', 'cold', 'generation', 'subset', 'padding', 'inactive', 'one_shard_miss', 'skewed'))
@pytest.mark.parametrize('mtp,shards', ((1, 1), (1, 4), (2, 1), (2, 4)))
def test_state_update_and_split_probes_preserve_full_result(native, scenario, mtp, shards):
    initial = make_case(2, mtp, shards, .9, scenario)
    expected, _ = reference(initial)
    prepared = initial.clone(native)
    prepared.run('baseline', 'union')
    prepared.run('baseline', 'finalize')
    for variant in ('baseline', 'vector_state_update', 'exact_combined'):
        split = prepared.clone()
        before = split['topk'].clone()
        split.run(variant, 'state_update')
        torch.npu.synchronize()
        assert torch.equal(split['topk'], before), 'state-only probe remapped inputs'
        split.run(variant, 'remap')
        torch.npu.synchronize()
        assert_result(split, expected)
        full = initial.clone(native)
        full.run(variant)
        torch.npu.synchronize()
        assert_result(full, expected)


@pytest.mark.parametrize('hit_rate', (0., 1.))
def test_vector_state_keeps_retrieval_descriptors(native, hit_rate):
    initial = make_case(2, 2, 4, hit_rate)
    prepared = initial.clone(native)
    prepared.run('baseline', 'union')
    prepared.run('baseline', 'finalize')
    before = prepared.clone('cpu')
    prepared.run('vector_state_update', 'update')
    torch.npu.synchronize()
    for name in ('packed', 'mapping', 'shard_counts', 'prior_slots', 'miss_tokens', 'miss_counts', 'target_slots'):
        assert torch.equal(prepared[name].cpu(), before[name]), name
