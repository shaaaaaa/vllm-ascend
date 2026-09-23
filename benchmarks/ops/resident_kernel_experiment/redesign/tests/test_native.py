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
def test_native_plan_and_fused_payload(variant, scenario, q):
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
    native = NativeCase(query.to('npu'), snap.to('npu'), dense.to('npu'), variant, buckets=256)
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
def test_native_2048_graph_replays_change_epochs_and_values(variant):
    query, snap, dense = case(b=1, q=2, k=2048, universe=8193)
    snap = replace(snap, kv=snap.kv.float())
    dense = dense.float()
    native = NativeCase(query.to('npu'), snap.to('npu'), dense.to('npu'), variant)
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
def test_fused_copy_preserves_payload_dtype_bits(dtype, variant):
    query, snap, dense = case(b=1, q=2, k=256, universe=1025, width=32)
    if dtype == torch.uint8:
        dense = torch.randint(0, 256, dense.shape, dtype=dtype)
    else:
        dense = dense.to(dtype)
    previous = dense.gather(1, snap.tokens.long()[..., None].expand(1, 512, 32)).clone()
    snap = replace(snap, kv=previous)
    query.tokens[..., :2] = -1
    native = NativeCase(query.to('npu'), snap.to('npu'), dense.to('npu'), variant, buckets=256)
    native.run(True)
    torch.npu.synchronize()
    expected = gather_dense(query, dense)
    assert torch.equal(native.payload.cpu().view(torch.uint8), expected.contiguous().view(torch.uint8))
