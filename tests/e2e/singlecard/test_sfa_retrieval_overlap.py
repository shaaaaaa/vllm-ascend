# SPDX-License-Identifier: Apache-2.0
"""Native fork/join and real registered-CPU transfer qualification."""
import ctypes
import gc
import os
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


@pytest.mark.parametrize("main_gate", [False, True], ids=["early_fork", "cube_windows"])
def test_native_capture_contract(main_gate):
    from vllm_ascend.compilation.sfa_retrieval_overlap import CaptureProbe
    torch.npu.set_device(0)
    stream = torch.npu.Stream()
    probes = [CaptureProbe(stream, torch.device("npu:0"), main_gate=main_gate) for _ in range(2)]
    try:
        for probe in probes:
            probe.verify()
        for iteration in range(32):
            probe = probes[iteration % 2]
            probe.source.fill_(iteration)
            if main_gate:
                probe.dest.fill_(-iteration - 2)
            probe.graph.replay()
            done = torch.npu.Event()
            done.record()
            done.synchronize()  # Main completion must cover side-stream work.
            assert torch.equal(probe.observed.cpu(), torch.full(probe.source.shape, float(iteration)))
            if main_gate:
                assert torch.equal(probe.before_release.cpu(), torch.full(probe.source.shape, float(-iteration - 2)))
                assert torch.equal(probe.after_close.cpu(), probe.before_reopen.cpu())
    finally:
        torch.npu.synchronize()


def test_dynamic_copy_windows_same_graph():
    """Change readiness on replay. Require multiple real copy kernels under a single opening."""
    from vllm_ascend.compilation.sfa_retrieval_overlap import CaptureProbe

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    probe = CaptureProbe(torch.npu.Stream(), device, main_gate=True)
    ready = torch.zeros(1, dtype=torch.int64, device=device)
    router_ready = torch.zeros_like(ready)
    after_vup = torch.empty_like(probe.dest)
    after_router = torch.empty_like(probe.dest)
    rows = probe.dest.view(torch.int64).reshape(64, 4)
    marker = 0x3C003C003C003C00  # Four FP16 ones in a K row.
    acl = ctypes.CDLL("libacl_rt.so")
    for fn in (acl.aclrtValueWait, acl.aclrtValueWrite):
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p]
        fn.restype = ctypes.c_int

    def wait_value(word, value):
        rc = acl.aclrtValueWait(word.data_ptr(), value, 1, torch.npu.current_stream().npu_stream)
        assert rc == 0

    edge = probe.edge
    transfer = edge.transfer

    class DelayedTransfer:
        def load(self, *args, **kwargs):
            wait_value(ready, 1)
            transfer.load(*args, **kwargs)

    edge.transfer = DelayedTransfer()
    probe.source.fill_(1)
    torch.npu.synchronize()
    with torch.npu.graph(probe.graph):
        edge.launch(*probe.payload, edge.destinations)
        with edge.cube_window("vup"):
            wait_value(rows[31, 0], marker)
        after_vup.copy_(probe.dest)
        ready.copy_(router_ready)
        with edge.cube_window("router"):
            wait_value(rows[63, 0], marker)
        after_router.copy_(probe.dest)
        rc = acl.aclrtValueWrite(ready.data_ptr(), 1, 0, torch.npu.current_stream().npu_stream)
        assert rc == 0
        edge.join()
    for label, begin, reopen in (("router", 0, 1), ("vup", 1, 1), ("tail", 0, 0)) * 2:
        ready.fill_(begin)
        router_ready.fill_(reopen)
        for plane in edge.destinations:
            plane.zero_()
        if not begin:
            rows[31, 0].fill_(marker)
        if not reopen:
            rows[63, 0].fill_(marker)
        probe.graph.replay()
        # Use the last column to count actual copy writes, independent of the test markers.
        first, second = (int(snapshot.reshape(64, 16)[:, -1].cpu().sum()) for snapshot in (after_vup, after_router))
        assert first % 8 == second % 8 == 0
        if label == "vup":
            assert 32 <= first <= second == 64
        else:
            assert first == 0 and second == (64 if label == "router" else 0)
        assert all(torch.all(plane.cpu() == 1) for plane in edge.destinations)
        print(f"[SFA_COPY_WINDOWS] {label} vup={first // 8} router={(second - first) // 8} tail={(64 - second) // 8}")


@pytest.mark.parametrize("gated", [False, True], ids=["ordinary", "main_gate"])
def test_shared_indexer_prefetch_joins_only_its_layer(gated):
    """A later transfer waits for an earlier consumer to run. A whole-FIFO join would deadlock."""
    from lmcache_ascend.v1.npu_connector.sparse_graph import SparseGraphTransfer

    from vllm_ascend.compilation.sfa_retrieval_overlap import CopyGate, RetrievalEdge, SharedPrefetchGroup

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    side = torch.npu.Stream(device=device)
    acl = ctypes.CDLL("libacl_rt.so")
    for fn in (acl.aclrtValueWait, acl.aclrtValueWrite):
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p]
        fn.restype = ctypes.c_int
    ready = torch.ones(3, dtype=torch.int64, device=device)
    all_done = torch.ones(1, dtype=torch.int64, device=device)
    qkv_started = torch.ones_like(all_done)
    x = torch.ones((4, 512), device=device, dtype=torch.float16)
    weight = torch.ones((512, 128), device=device, dtype=torch.float16)
    qkv = x.new_empty((4, 128))
    selected = torch.arange(64, dtype=torch.int32, device=device).view(1, 64)
    slots = selected.long()
    counts = torch.full((1,), 64, dtype=torch.int32, device=device)
    source, edges, snapshots = [], [], []

    def signal(word):
        assert acl.aclrtValueWrite(word.data_ptr(), 1, 0, torch.npu.current_stream().npu_stream) == 0

    def wait(word):
        assert acl.aclrtValueWait(word.data_ptr(), 1, 1, torch.npu.current_stream().npu_stream) == 0

    class DelayedTransfer:
        request_capacity = 1

        def __init__(self, layer, transfer):
            self.layer, self.transfer = layer, transfer

        def load(self, *args, **kwargs):
            if self.layer:
                wait(qkv_started)
            wait(ready[self.layer])
            self.transfer.load(*args, **kwargs)
            if self.layer == 2:
                signal(all_done)

    gate = CopyGate(device) if gated else None
    for i in range(3):
        values = torch.arange(64, dtype=torch.float16, device=device).view(1, 64, 1, 1)
        planes = tuple((values + 256 * i + 64 * p).expand(-1, -1, -1, w).contiguous()
                       for p, w in enumerate((32, 16)))
        source.append(planes)
        dest = tuple(torch.empty_like(plane) for plane in planes)
        real = SparseGraphTransfer(dest, slots, 64, 64, request_capacity=1)
        for p, plane in enumerate(planes):
            real.ptrs[p].fill_(plane.data_ptr())
        real.valid_tokens.fill_(64)
        edges.append(RetrievalEdge(side, DelayedTransfer(i, real), dest, pieces=4, gate=gate))
        snapshots.append(tuple(torch.empty_like(plane) for plane in dest))
    group = SharedPrefetchGroup(tuple(edges))
    first_boundary = torch.empty_like(edges[2].destinations[0])
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()

    def project(layer):
        # The two later transfers cannot begin before the first QKV callback.
        # In the fast case, hold this test window until both have finished.
        if layer == 0:
            signal(qkv_started)
        torch.mm(x, weight, out=qkv)
        for plane, observed in zip(edges[layer].destinations, snapshots[layer]):
            observed.copy_(plane)
        if layer == 0:
            wait(all_done)
            first_boundary.copy_(edges[2].destinations[0])

    with torch.npu.graph(graph):
        group.launch(selected, counts, slots)
        for i, edge in enumerate(edges):
            if i:
                signal(ready[i])
            edge.join(project, i)

    # The fast replay must finish layers 2/3 within layer 1's QKV window.
    # The slow replay withholds each later transfer until the preceding QKV returns.
    for early, shift, count in ((True, 5, 64), (False, 11, 29), (True, 7, 1), (False, 3, 0)) * 2:
        ready.fill_(int(early))
        ready[0].fill_(1)
        all_done.fill_(int(not early))
        qkv_started.zero_()
        ids = (torch.arange(64, dtype=torch.int32) + shift) % 64
        selected.copy_(ids.view_as(selected))
        counts.fill_(count)
        for edge in edges:
            for plane in edge.destinations:
                plane.fill_(-1)
        graph.replay()
        for i, pair in enumerate(snapshots):
            for p, observed in enumerate(pair):
                expected = torch.full(observed.shape, -1, dtype=torch.float16)
                expected[:, :count] = source[i][p].cpu()[:, ids[:count].long()]
                torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
                if i == 2 and p == 0:
                    boundary = expected if early else torch.full_like(expected, -1)
                    torch.testing.assert_close(first_boundary.cpu(), boundary, rtol=0, atol=0)


@pytest.mark.parametrize("requests,main_gate", [(1, False), (8, False), (16, False), (8, True)])
@pytest.mark.parametrize("query_rows", [1, 2])
def test_registered_sources_consumer_boundary_and_replay(requests, query_rows, main_gate):
    root = Path(os.environ.get("LMCACHE_ASCEND_SOURCE_DIR", str(ROOT.parent / "LMCache-Ascend")))
    if not (root / "benchmark/v1/kv_transfer/load_benchmark_utils.py").exists():
        pytest.fail("set LMCACHE_ASCEND_SOURCE_DIR to the matching LMCache-Ascend source checkout")
    sys.path.insert(0, str(ROOT / "tools"))
    from sfa_retrieval_overlap_benchmark import Workload
    workload = Workload(root, requests, query_rows, main_gate=main_gate, multi_layer=True)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        graphs = [workload.capture(mode, observe=True) for mode in (False, True)]
        graphs.append(workload.capture(True, observe=True, multi_layer=True))
        for active, shift in ((0, 0), (1, 4096), (410, 3), (4096, 17), (37, 299), (0, 0)):
            for graph in graphs:
                tokens = workload.set_payload(active, shift, idle=requests > 1)
                graph.replay()
                workload.check(tokens, active, idle=requests > 1)
        # A different immutable lane-to-source binding models request replacement.
        workload.sources.reverse()
        for layer, transfer in enumerate(workload.transfers):
            transfer.bind_batch(workload.sources, layer)
        tokens = workload.set_payload(37, 21)
        for graph in graphs:
            graph.replay()
            for layer in range(workload.layers):
                for plane in range(2):
                    observed = workload.observed[layer][plane].cpu()
                    for lane in range(requests):
                        expected = (workload.values(tokens[:37], requests - lane - 1, layer) + 32 * plane).bfloat16()
                        torch.testing.assert_close(observed[lane, :37], expected, rtol=0, atol=0)
    finally:
        workload.close()
        if was_enabled:
            gc.enable()
