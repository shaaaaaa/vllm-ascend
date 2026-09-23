# SPDX-License-Identifier: Apache-2.0
"""Hardware qualification. Skip ONLY when NPU support/hardware is absent.

A build/registration/native mismatch is a failure, never a fallback to Torch.
"""
from dataclasses import replace
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from candidates import assert_safe, attention, gather_dense
from native import MODES, NativeCase, load_library
from test_candidates import case

if importlib.util.find_spec('torch_npu') is not None:
    import torch_npu  # noqa: F401
AVAILABLE = hasattr(torch, 'npu') and torch.npu.is_available()
pytestmark = pytest.mark.skipif(not AVAILABLE, reason='requires a real Ascend NPU and torch_npu')


@pytest.fixture(scope='module', autouse=True)
def library(request):
    if AVAILABLE:
        load_library(request.config.getoption('--redesign-build-dir'))


@pytest.mark.parametrize('variant', tuple(MODES))
@pytest.mark.parametrize('scenario', ('normal', 'cold', 'epoch', 'version', 'padding', 'duplicate', 'collisions'))
@pytest.mark.parametrize('q', (1, 2))
@pytest.mark.parametrize('copy_mode', ('row', 'batched', 'pipelined'))
def test_native_plan_and_fused_payload(variant, scenario, q, copy_mode):
    query, snap, dense = case(seed=7, b=3, q=q, k=256, universe=1025, width=16)
    if scenario == 'cold':
        snap = replace(snap, ready=torch.zeros_like(snap.ready))
    elif scenario == 'epoch':
        snap = replace(snap, epochs=snap.epochs+1, kv=torch.full_like(snap.kv, float('nan')))
    elif scenario == 'version':
        query.versions.add_(1)
        dense = dense + 100
    elif scenario == 'padding':
        query.active[1].zero_()
        query.tokens[:, :, :4] = torch.tensor([-1, -2, 1025, 2**31-1])
    elif scenario == 'duplicate':
        query.tokens[:, :, 1] = query.tokens[:, :, 0]
    elif scenario == 'collisions':
        query.tokens.remainder_(4).mul_(256)
    snap = replace(snap, kv=snap.kv.float())
    dense = dense.float()
    native = NativeCase(query.to('npu'), snap.to('npu'), dense.to('npu'), variant, buckets=256, copy_mode=copy_mode)
    for fused in (False, True):
        native.run(fused)
        torch.npu.synchronize()
        assert_safe(query, snap, native.plan)
        if fused:
            actual = native.payload.cpu()
            expected = gather_dense(query, dense)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            b, rows, k = query.tokens.shape
            valid = query.masks()[0].reshape(b, rows, k)
            vectors = torch.randn(b, rows, 2, 8)
            torch.testing.assert_close(
                attention(vectors, actual.reshape(b, rows, k, -1), valid),
                attention(vectors, expected.reshape(b, rows, k, -1), valid), atol=0, rtol=0)


@pytest.mark.parametrize('variant', tuple(MODES))
@pytest.mark.parametrize('copy_mode', ('row', 'batched', 'pipelined'))
def test_native_2048_graph_replays_change_epochs_and_values(variant, copy_mode):
    query, snap, dense = case(b=1, q=2, k=2048, universe=8193)
    snap = replace(snap, kv=snap.kv.float())
    dense = dense.float()
    native = NativeCase(query.to('npu'), snap.to('npu'), dense.to('npu'), variant, copy_mode=copy_mode)
    native.run(True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        native.run(True)
    torch.npu.synchronize()
    saved = []
    for step in range(6):
        updated = replace(query, tokens=query.tokens.roll(step, dims=-1), epochs=query.epochs + step % 2)
        native.refresh(updated.to('npu'), snap.to('npu'), dense.to('npu'))
        graph.replay()
        # Snapshot each replay before the next replay overwrites its output.
        saved.append((updated, native.plan.source.clone(), native.payload.clone()))
    torch.npu.synchronize()
    from candidates import Plan
    for updated, source, payload in saved:
        assert_safe(updated, snap, Plan(source, variant))
        torch.testing.assert_close(payload.cpu(), gather_dense(updated, dense), rtol=0, atol=0)


@pytest.mark.parametrize('index,dtype,message', (
    (0, torch.int64, 'metadata dtype mismatch'),
    (6, torch.int32, 'metadata dtype mismatch'),
    (11, torch.float16, 'KV dtypes differ'),
))
def test_native_rejects_mismatched_dtypes(index, dtype, message):
    query, snap, dense = case(b=1, q=2, k=256)
    snap = replace(snap, kv=snap.kv.float())
    native = NativeCase(query.to('npu'), snap.to('npu'), dense.float().to('npu'), 'fixed_position')
    native.tensors[index] = native.tensors[index].to(dtype)
    with pytest.raises(RuntimeError, match=message):
        native.run(True)


def test_native_rejects_aliases():
    query, snap, dense = case(b=1, q=2, k=256)
    snap = replace(snap, kv=snap.kv.float())
    native = NativeCase(query.to('npu'), snap.to('npu'), dense.float().to('npu'), 'fixed_position')
    native.tensors[11] = native.tensors[9]
    with pytest.raises(RuntimeError, match='overlap'):
        native.run(True)


@pytest.mark.parametrize('variant', ('bitmap_directory', 'sorted_snapshot', 'cube_join', 'full_hbm'))
def test_functional_candidates_on_npu(variant):
    from candidates import lookup, materialize
    query, snap, dense = case(b=1, q=2, k=32, universe=257)
    snap = replace(snap, kv=snap.kv.float())
    dense = dense.float()
    device_query, device_snap = query.to('npu'), snap.to('npu')
    plan = lookup(device_query, device_snap, variant, tile=16)
    actual = materialize(device_query, device_snap, plan, dense.to('npu'))
    torch.npu.synchronize()
    assert_safe(query, snap, plan)
    torch.testing.assert_close(actual.cpu(), gather_dense(query, dense), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16, torch.uint8))
@pytest.mark.parametrize('variant', ('fixed_position', 'hash_snapshot'))
@pytest.mark.parametrize('copy_mode', ('row', 'batched', 'pipelined'))
def test_fused_copy_preserves_payload_dtype_bits(dtype, variant, copy_mode):
    query, snap, dense = case(b=1, q=2, k=256, universe=1025, width=32)
    if dtype == torch.uint8:
        dense = torch.randint(0, 256, dense.shape, dtype=dtype)
    else:
        dense = dense.to(dtype)
    previous = dense.gather(1, snap.tokens.long()[..., None].expand(1, 512, 32)).clone()
    snap = replace(snap, kv=previous)
    query.tokens[..., :2] = -1
    native = NativeCase(query.to('npu'), snap.to('npu'), dense.to('npu'), variant, buckets=256, copy_mode=copy_mode)
    native.run(True)
    torch.npu.synchronize()
    expected = gather_dense(query, dense)
    assert torch.equal(native.payload.cpu().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize('width', (16, 560, 576, 4096))
@pytest.mark.parametrize('copy_rows', (8, 16, 32, 64))
def test_batched_copy_row_sizes_and_partial_groups(width, copy_rows):
    query, snap, dense = case(b=1, q=2, k=256, universe=1025, width=width)
    dense = dense.to(torch.bfloat16)
    previous = dense.gather(1, snap.tokens.long()[..., None].expand(1, 512, width)).clone()
    snap = replace(snap, kv=previous)
    # Interleave invalid, hit/miss and live-tail sources within copy groups.
    query.tokens[..., ::7] = -1
    query.boundary.fill_(512)
    expected = gather_dense(query, dense)
    plans = []
    for mode in ('row', 'batched', 'pipelined'):
        native = NativeCase(query.to('npu'), snap.to('npu'), dense.to('npu'),
                            'bounded_position', copy_mode=mode, copy_rows=16 if mode == 'row' else copy_rows)
        native.run(True)
        torch.npu.synchronize()
        assert_safe(query, snap, native.plan)
        assert torch.equal(native.payload.cpu().view(torch.uint8), expected.contiguous().view(torch.uint8))
        plans.append(native.plan.source.cpu())
    assert all(torch.equal(plans[0], plan) for plan in plans[1:])


@pytest.mark.parametrize('variant', tuple(MODES))
@pytest.mark.parametrize('radius', (0, 2, 32))
def test_tuned_lookup_matches_source_slots(variant, radius):
    query, snap, dense = case(b=1, q=2, k=2048, universe=8193, width=16)
    query.tokens[..., ::127] = -1
    snap = replace(snap, kv=snap.kv.float())
    baseline = NativeCase(query.to('npu'), snap.to('npu'), dense.float().to('npu'), variant,
                          radius=radius, copy_mode='batched')
    tuned = NativeCase(query.to('npu'), snap.to('npu'), dense.float().to('npu'), variant,
                       radius=radius, copy_mode='pipelined', copy_rows=64,
                       lookup_mode='interior', table_mode='wide')
    for fused in (False, True):
        baseline.run(fused)
        tuned.run(fused)
        torch.npu.synchronize()
        assert torch.equal(baseline.plan.source.cpu(), tuned.plan.source.cpu())
        assert_safe(query, snap, tuned.plan)
        if fused:
            assert torch.equal(tuned.payload.cpu(), gather_dense(query, dense.float()))


@pytest.mark.parametrize('variant,size', (('hash_snapshot', 256), ('hash_snapshot', 8192),
                                         ('direct_directory', 256), ('direct_directory', 2304)))
def test_wide_table_preserves_collision_winners(variant, size):
    universe = 2301 if variant == 'direct_directory' and size == 2304 else 257
    if variant == 'direct_directory' and size == 256:
        universe = 255
    query, snap, dense = case(b=3, q=2, k=256, universe=universe, width=16)
    snap = replace(snap, kv=snap.kv.float())
    tables, plans = [], []
    for table_mode in ('baseline', 'wide'):
        native = NativeCase(query.to('npu'), snap.to('npu'), dense.float().to('npu'), variant,
                            buckets=size, table_mode=table_mode)
        native.run(False)
        torch.npu.synchronize()
        tables.append(native.tensors[7].cpu())
        plans.append(native.plan.source.cpu())
    assert torch.equal(*tables)
    assert torch.equal(*plans)


def test_pipelined_grid_stride_replay():
    query, snap, dense = case(b=16, q=2, k=2048, universe=4097, width=16)
    snap = replace(snap, kv=snap.kv.float())
    native = NativeCase(query.to('npu'), snap.to('npu'), dense.float().to('npu'), 'bounded_position',
                        copy_mode='pipelined', copy_rows=64, lookup_mode='interior')
    native.run(True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        native.run(True)
    for step in range(4):
        updated = replace(query, tokens=query.tokens.roll(step, dims=-1), epochs=query.epochs + step % 2)
        native.refresh(updated.to('npu'), snap.to('npu'), dense.float().to('npu'))
        graph.replay()
        torch.npu.synchronize()
        assert_safe(updated, snap, native.plan)
        assert torch.equal(native.payload.cpu(), gather_dense(updated, dense.float()))
