# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from candidates import (VARIANTS, Plan, Query, Snapshot, Transaction,
                        assert_safe, attention, gather_dense, lookup, materialize,
                        shared_plan, signed_bits, split_attention)

torch.set_num_threads(1)


def case(seed=0, b=2, q=2, k=32, universe=257, width=16):
    rng = torch.Generator().manual_seed(seed)
    dense = torch.randn(b, universe, width, generator=rng, dtype=torch.float64)
    tokens = torch.randint(0, universe, (b, q, k), generator=rng, dtype=torch.int32)
    versions = torch.zeros_like(tokens)
    query = Query(tokens, versions, torch.ones(b, q, dtype=torch.bool),
                  torch.full((b, q), universe - 8, dtype=torch.int32),
                  torch.full((b, q), universe, dtype=torch.int32),
                  torch.arange(b, dtype=torch.int64) + 100, universe)
    previous = tokens.flatten(1).roll(2, dims=1).clone()
    previous[:, ::5] = torch.randint(0, universe, previous[:, ::5].shape, generator=rng)
    ready = torch.rand(b, q * k, generator=rng) > .1
    old_kv = dense.gather(1, previous.long()[..., None].expand(b, q*k, width)).clone()
    old_kv[~ready] = torch.nan
    snap = Snapshot(previous, torch.zeros_like(previous), ready, query.epochs.clone(), old_kv)
    return query, snap, dense


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('scenario', ('normal', 'cold', 'epoch', 'version', 'padding', 'duplicates', 'invalid_metadata'))
def test_final_attention(variant, scenario):
    query, snap, dense = case()
    if scenario == 'cold':
        snap = replace(snap, ready=torch.zeros_like(snap.ready))
    elif scenario == 'epoch':
        snap = replace(snap, epochs=snap.epochs + 1, kv=torch.full_like(snap.kv, torch.nan))
    elif scenario == 'version':
        query.versions[..., ::3] = 17
        # Changed version means current source data can differ from old KV.
        dense = dense.clone()
        for r in range(dense.shape[0]):
            changed = query.tokens[r, :, ::3].long()
            dense[r, changed] += 10
            # Mark all occurrences of these changed tokens, not just one row.
            for t in changed.flatten().unique():
                query.versions[r][query.tokens[r] == t] = 17
    elif scenario == 'padding':
        query.active[0, 1] = False
        query.tokens[:, :, :4] = torch.tensor([-1, -2, 257, 2**31-1], dtype=torch.int32)
    elif scenario == 'duplicates':
        query.tokens[:, 1] = query.tokens[:, 0]
        query.tokens[:, :, 1] = query.tokens[:, :, 0]
    elif scenario == 'invalid_metadata':
        query.boundary[0, 0] = -1
        query.lengths[1, 0] = 258
        query.boundary[0, 1] = query.lengths[0, 1] + 1
    before = [x.clone() for x in (snap.tokens, snap.versions, snap.ready, snap.kv)]
    plan = lookup(query, snap, variant, buckets=8, tile=13)
    assert_safe(query, snap, plan)
    actual = materialize(query, snap, plan, dense)
    expected = gather_dense(query, dense)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    b, q, k = query.tokens.shape
    d = dense.shape[-1] // 2
    vectors = torch.randn(b, q, 3, d, dtype=dense.dtype)
    valid = query.masks()[0].reshape(b, q, k)
    result = attention(vectors, actual.reshape(b, q, k, -1), valid)
    expected_result = attention(vectors, expected.reshape(b, q, k, -1), valid)
    torch.testing.assert_close(result, expected_result, rtol=0, atol=0)
    for old, current in zip(before, (snap.tokens, snap.versions, snap.ready, snap.kv), strict=True):
        torch.testing.assert_close(old, current, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('variant', VARIANTS)
def test_randomized_safe_hits(variant):
    for seed in range(12):
        query, snap, dense = case(seed, q=1 + seed % 2, k=7 + seed, universe=37)
        plan = lookup(query, snap, variant, buckets=3, tile=5)
        assert_safe(query, snap, plan)
        torch.testing.assert_close(materialize(query, snap, plan, dense), gather_dense(query, dense), rtol=0, atol=0)


def test_false_hit_oracle_rejects_bad_plans():
    query, snap, _ = case()
    plan = lookup(query, snap, 'reload')
    source = plan.source.clone()
    source[0, 0] = 0
    with pytest.raises(AssertionError):
        assert_safe(query, snap, Plan(source, 'bad'))
    source[0, 0] = 999999
    with pytest.raises(AssertionError):
        assert_safe(query, snap, Plan(source, 'bad'))


@pytest.mark.parametrize('q', [1, 2])
def test_real_topk_2048(q):
    query, snap, dense = case(b=1, q=q, k=2048, universe=8193)
    for variant in ('fixed_position', 'bounded_position', 'hash_snapshot', 'direct_directory', 'sorted_snapshot'):
        plan = lookup(query, snap, variant)
        assert_safe(query, snap, plan)
        torch.testing.assert_close(materialize(query, snap, plan, dense), gather_dense(query, dense), rtol=0, atol=0)


def test_matrix_encoding_is_exact_at_int32_extremes():
    tokens = torch.tensor([0, 1, 2048, 65535, 2**24, 2**24+1, 2**31-1, -1], dtype=torch.int32)
    versions = torch.tensor([0, 1, -1, 17, 0, 0, 2**31-1, -2**31], dtype=torch.int32)
    encoding = signed_bits(tokens, versions)
    equality = encoding @ encoding.T == 64
    torch.testing.assert_close(equality, torch.eye(len(tokens), dtype=torch.bool))


def test_failed_transfer_cannot_publish_and_inputs_are_snapshotted():
    query, snap, dense = case()
    payload = materialize(query, snap, lookup(query, snap, 'reload'), dense)
    pending = Transaction(query, payload)
    original = query.tokens.clone()
    query.tokens.fill_(-1)
    with pytest.raises(RuntimeError):
        pending.publish()
    pending.complete(False)
    with pytest.raises(RuntimeError):
        pending.publish()
    with pytest.raises(RuntimeError):
        pending.complete(True)
    torch.testing.assert_close(pending.pending.tokens, original.flatten(1))


def test_multistep_publish_and_rollback():
    query, snap, dense = case()
    for step in range(8):
        query = replace(query, tokens=query.tokens.roll(1, dims=-1))
        if step == 3:
            query = replace(query, epochs=query.epochs + 1)
        plan = lookup(query, snap, 'hash_snapshot', buckets=127)
        assert_safe(query, snap, plan)
        payload = materialize(query, snap, plan, dense)
        torch.testing.assert_close(payload, gather_dense(query, dense), rtol=0, atol=0)
        transaction = Transaction(query, payload)
        transaction.complete(step != 4)
        if step != 4:
            snap = transaction.publish()
            with pytest.raises(RuntimeError):
                transaction.publish()


def test_shared_group_checks_every_layers_readiness_and_identity():
    query, first, dense = case()
    # Different layer values at identical logical slots are expected.
    second = replace(first, kv=first.kv + 10, ready=first.ready.clone(), tokens=first.tokens.clone())
    second.ready[:, ::3] = False
    second.tokens[:, 1::5] = -1
    second = replace(second, kv=second.kv.clone())
    second.kv[~second.ready | (second.tokens < 0)] = torch.nan
    plan = shared_plan(query, [first, second], 'bounded_position', radius=2)
    for snap, source in ((first, dense), (second, dense + 10)):
        assert_safe(query, snap, plan)
        torch.testing.assert_close(materialize(query, snap, plan, source), gather_dense(query, source), rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['mixed', 'empty', 'all_hit', 'all_miss'])
def test_split_softmax_empty_subsets(mode):
    query, snap, dense = case()
    values = gather_dense(query, dense).reshape(2, 2, 32, 16)
    valid = query.masks()[0].reshape(2, 2, 32)
    hits = torch.rand(2, 2, 32) > .5
    if mode == 'empty':
        valid.zero_()
    elif mode == 'all_hit':
        hits.fill_(True)
    elif mode == 'all_miss':
        hits.zero_()
    vectors = torch.randn(2, 2, 4, 8, dtype=dense.dtype) * 100
    torch.testing.assert_close(split_attention(vectors, values, valid, hits), attention(vectors, values, valid),
                               rtol=1e-12, atol=1e-12)


def test_selection_multiplicity_is_preserved():
    query, snap, dense = case()
    query.tokens.fill_(3)
    plan = lookup(query, snap, 'cube_join', tile=5)
    values = materialize(query, snap, plan, dense)
    assert values.shape[1] == 64
    torch.testing.assert_close(values, dense[:, 3:4].expand_as(values), rtol=0, atol=0)
