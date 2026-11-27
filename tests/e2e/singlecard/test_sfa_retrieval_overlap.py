# SPDX-License-Identifier: Apache-2.0
"""Native fork/join and real registered-CPU transfer qualification."""
import ctypes
import gc
import json
import os
import statistics
import sys
from contextlib import contextmanager
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


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
                probe.pe_source.fill_(-iteration - 1)
                probe.edge.destinations[1].zero_()
            probe.graph.replay()
            done = torch.npu.Event()
            done.record()
            done.synchronize()  # Main completion must cover side-stream work.
            assert torch.equal(probe.observed.cpu(), torch.full(probe.source.shape, float(iteration)))
            if main_gate:
                assert torch.all(probe.edge.destinations[1].cpu() == -iteration - 1)
                assert torch.equal(probe.before_release.cpu(), torch.full(probe.source.shape, float(-iteration - 2)))
                assert torch.equal(probe.after_close.cpu(), probe.before_reopen.cpu())
    finally:
        torch.npu.synchronize()


@pytest.mark.parametrize("rows", [16, 32])
def test_20_aiv_decoder_projection_capture(rows, monkeypatch, request):
    """Replay real 20-AIV copy, BF16/INT8 projection and SwiGlu with changing inputs."""
    import vllm_ascend.ops.register_custom_ops  # noqa: F401
    from vllm_ascend.compilation import sfa_copy_windows as windows
    from vllm_ascend.compilation.sfa_retrieval_overlap import CaptureProbe

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    probe = CaptureProbe(torch.npu.Stream(), device, main_gate=True)
    edge = probe.edge
    gate = edge.gate
    if not gate.supports_decoder_windows:
        pytest.skip(f"qualified tiling requires 910B3 / CANN 8.5.1, found {gate.soc} / {gate.cann}")
    monkeypatch.setattr(windows, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(windows, "get_forward_context", lambda: SimpleNamespace(sfa_copy_gate=gate))
    edge.max_aiv_cores = 20
    previous_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    request.addfinalizer(lambda: setattr(torch.npu.config, "allow_internal_format", previous_format == b"enable"))
    torch.npu.config.allow_internal_format = True
    x = torch.randn(rows, 1536, device=device, dtype=torch.bfloat16)
    weight = torch_npu.npu_format_cast(torch.randint(-4, 5, (1536, 1024), device=device, dtype=torch.int8), 29)
    scale = torch.full((1024,), 0.01, device=device, dtype=torch.bfloat16)

    def reference():
        q, s = torch_npu.npu_dynamic_quant(x)
        out = torch_npu.npu_quant_matmul(q, weight, scale, pertoken_scale=s, bias=None, output_dtype=x.dtype)
        return torch_npu.npu_swiglu(out)

    reference()  # Materialize the real operators before capture.
    torch.npu.synchronize()
    with torch.npu.graph(probe.graph):
        edge.launch(*probe.payload, edge.destinations)
        projected = torch.ops.vllm.sfa_decode_w8a8(x, weight, scale, None, "q_b", False)
        activated = torch.ops.vllm.sfa_decode_swiglu(projected)
        edge.join()
        probe.observed.copy_(probe.dest)
    assert gate.counts["q_b"] == gate.counts["swiglu"] == 1
    assert gate.counts["dynamic_quant"] == int(rows <= 20)
    for value in (1, 3, 7):
        x.normal_(mean=value / 8)
        expected = reference()
        probe.source.fill_(value)
        probe.pe_source.fill_(-value)
        probe.dest.zero_()
        probe.graph.replay()
        torch.testing.assert_close(activated.cpu(), expected.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(probe.observed.cpu(), torch.full_like(probe.source.cpu(), value))
        assert torch.all(edge.destinations[1].cpu() == -value)


def test_dynamic_copy_windows_same_graph():
    """Change readiness on replay. Require multiple real copy kernels under a single opening."""
    from vllm_ascend.compilation.sfa_retrieval_overlap import CaptureProbe

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    probe = CaptureProbe(torch.npu.Stream(), device, main_gate=True)
    ready = torch.ones(1, dtype=torch.int64, device=device)
    router_ready = torch.ones_like(ready)
    after_vup = torch.empty_like(probe.dest)
    after_router = torch.empty_like(probe.dest)
    marker_words = probe.dest.view(torch.int64)[0, :, 0, 0]
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
    probe.pe_source.fill_(2)
    probe.dest.zero_()
    torch.npu.synchronize()
    with torch.npu.graph(probe.graph):
        edge.launch(*probe.payload, edge.destinations)
        with edge.cube_window("vup"):
            wait_value(marker_words[31], marker)
        after_vup.copy_(probe.dest)
        ready.copy_(router_ready)
        with edge.cube_window("router"):
            wait_value(marker_words[63], marker)
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
            marker_words[31].fill_(marker)
        if not reopen:
            marker_words[63].fill_(marker)
        probe.graph.replay()
        # Use the last column to count actual copy writes, independent of the test markers.
        first, second = (int(snapshot[0, :, 0, -1].cpu().sum()) for snapshot in (after_vup, after_router))
        assert first % 8 == second % 8 == 0
        if label == "vup":
            assert 32 <= first <= second == 64
        else:
            assert first == 0 and second == (64 if label == "router" else 0)
        assert torch.all(edge.destinations[0].cpu() == 1)
        assert torch.all(edge.destinations[1].cpu() == 2)
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
    print(f"[SHARED_PREFETCH_ORDER] gated={gated} first_QKV_started_with_BC_pending=pass "
          "BC_finished_inside_one_first_QKV_window=pass replays=8")


@contextmanager
def glm_group_workload(request):
    root = Path(os.environ.get("LMCACHE_ASCEND_SOURCE_DIR", str(ROOT.parent / "LMCache-Ascend")))
    from sfa_retrieval_overlap_benchmark import Workload

    path = Path("/workspace/models/GLM-5.3-w4a8c8/config.json")
    if path.is_file():
        config = json.loads(path.read_text(encoding="utf-8"))
        config = config.get("text_config") or config
        shape_source = str(path)
    else:
        config = dict(hidden_size=6144, q_lora_rank=2048, kv_lora_rank=512, qk_rope_head_dim=64,
                      num_attention_heads=64, v_head_dim=256, n_routed_experts=256)
        shape_source = "GLM reference shape"
    print(f"[SHARED_PREFETCH_SETUP] NPU=0 TP=4 requests=16 query_rows=2 width=4096 "
          f"QKV=W8A8_dynamic hidden={config['hidden_size']} shape_source={shape_source}", flush=True)
    # Keep CPU fixture fills and small reference comparisons off the host thread pool.
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    torch.set_num_threads(1)
    # The 8.5 torch_npu property only has a setter. Save the configured value directly.
    previous_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    if previous_format in (b"enable", b"disable"):
        request.addfinalizer(lambda: setattr(torch.npu.config, "allow_internal_format", previous_format == b"enable"))
    torch.npu.config.allow_internal_format = True
    w = Workload(root, requests=16, query_rows=2, history=8192, main_gate=True, multi_layer=True)
    try:
        w.use_glm_projections(config, tp=4)
        yield w
    finally:
        w.close()


def test_shared_indexer_prefetch_effect(request):
    """Measure the 1+3 group on registered host KV. Observe the first consumer in a separate graph."""
    from sfa_retrieval_overlap_benchmark import paired_graph_times, ready_rows

    with glm_group_workload(request) as w:
        # A is the first consumer (layer 1); B/C have their own destinations (layers 2/3).
        snapshots = [tuple(torch.empty_like(t) for t in targets) for targets in w.observed[2:]]

        def observe_after_a():
            for layer, targets in enumerate(snapshots, 2):
                w.snapshot_layer(layer, targets)

        w.set_payload(1)
        checks = [w.capture(True, multi_layer=grouped, after_first_qkv=observe_after_a) for grouped in (False, True)]
        timed = [w.capture(True, multi_layer=grouped) for grouped in (False, True)]
        for misses in (1, 410, 4096):
            progress = []
            for grouped, graph in zip((False, True), checks):
                for shift in (5, 43, 199):
                    tokens = w.set_payload(misses, shift)  # Clear all destinations before every observation.
                    assert len(tokens[:misses].unique()) == misses
                    graph.replay()
                    for layer, targets in enumerate(w.observed):
                        w.snapshot_layer(layer, targets)
                    w.check(tokens, misses)
                    b_rows, c_rows = (ready_rows(early, final) for early, final in zip(snapshots, w.observed[2:]))
                    if grouped:
                        assert c_rows == 0 or b_rows == w.requests * misses  # B precedes C on one FIFO.
                        progress.append((b_rows, c_rows))
                    else:
                        assert (b_rows, c_rows) == (0, 0)  # Neither transfer has been submitted at A's QKV.
            w.set_payload(misses, 43)
            timing = paired_graph_times(*timed)
            b, c = (int(statistics.median(column)) for column in zip(*progress))
            print(f"[SHARED_PREFETCH_EFFECT] misses_per_request={misses} "
                  f"B_ready_at_A_close={b}/{w.requests * misses} C_ready_at_A_close={c}/{w.requests * misses} "
                  f"adjacent_us={timing['adjacent_us']:.1f} grouped_us={timing['grouped_us']:.1f} "
                  f"paired_saved_us={timing['paired_saved_us']:+.1f} "
                  f"grouped_faster_pairs={timing['grouped_faster_pairs']}/{timing['pairs']}", flush=True)


def test_sfa_copy_core_limit(request, monkeypatch):
    """Compare the same scatter, gate and captures at 12, 24 and the device's automatic grid."""
    from sfa_retrieval_overlap_benchmark import paired_graph_times

    from vllm_ascend import envs

    torch.npu.set_device(0)
    try:
        query = ctypes.CDLL("libascendcl.so").aclrtGetDeviceInfo
    except AttributeError:
        query = ctypes.CDLL("libacl_rt.so").aclrtGetDeviceInfo
    query.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)]
    query.restype = ctypes.c_int32
    cores = ctypes.c_int64()
    assert query(0, 201, ctypes.byref(cores)) == 0  # ACL_DEV_ATTR_VECTOR_CORE_NUM
    if cores.value <= 12:
        pytest.skip(f"hardware has {cores.value} AIV cores, comparison requires more than 12")
    limits = (12, 24, 0) if cores.value > 24 else (12, 0)
    variable = "VLLM_ASCEND_SFA_COPY_MAX_AIV_CORES"
    monkeypatch.delenv(variable, raising=False)
    with glm_group_workload(request) as w:
        graphs = {"copy_only": {}, "group_1plus3": {}}
        w.set_payload(1)
        for cap in limits:
            monkeypatch.setenv(variable, str(cap))
            w.set_copy_core_limit(envs.VLLM_ASCEND_SFA_COPY_MAX_AIV_CORES)
            assert w.max_aiv_cores == cap
            graphs["copy_only"][cap] = w.capture_copy()
            graphs["group_1plus3"][cap] = w.capture(True, multi_layer=True)
        side_pieces = w.edges[0].pieces
        capacity_per_piece = (w.selected.numel() + side_pieces - 1) // side_pieces
        assert capacity_per_piece >= cores.value
        print(f"[SFA_COPY_CORES] device_AIV={cores.value} pieces_per_prefetch={side_pieces} "
              f"auto_launch_grid={cores.value} requests={w.requests}", flush=True)
        for misses in (1, 410, 4096):
            for scope, variants in graphs.items():
                for graph in variants.values():
                    tokens = w.set_payload(misses, 43)
                    graph.replay()
                    for layer in ((1,) if scope == "copy_only" else range(w.layers)):
                        w.check_copied_layer(layer, tokens, misses)
                for cap in limits[1:]:
                    # Both widths see the same rows in each pair. Change the selection between pairs.
                    timing = paired_graph_times(
                        variants[12], variants[cap], names=("cap12", "candidate"), pairs=8, replays=2,
                        prepare_pair=lambda i, n=misses: w.set_payload(n, 43 + 613 * i),
                    )
                    base, candidate = timing["cap12_us"], timing["candidate_us"]
                    rate = ""
                    if scope == "copy_only":
                        bytes_per_row = sum(p.shape[-1] * p.element_size() for p in w.caches[1])
                        nbytes = w.requests * misses * bytes_per_row
                        rate = (f" rows_per_piece={w.requests * misses // side_pieces} payload_bytes={nbytes}"
                                f" effective_GB_s_12={nbytes / base / 1000:.2f}"
                                f" effective_GB_s_candidate={nbytes / candidate / 1000:.2f}")
                    print(f"[SFA_COPY_CORES] {scope} misses_per_request={misses} "
                          f"candidate={cap or 'auto'} candidate_grid={cap or cores.value} "
                          f"cap12_us={base:.1f} candidate_us={candidate:.1f} "
                          f"paired_saved_us={timing['paired_saved_us']:+.1f} "
                          f"candidate_faster_pairs={timing['candidate_faster_pairs']}/{timing['pairs']}{rate}",
                          flush=True)


def test_grouped_layer_copy_times_410(request, tmp_path):
    """Measure each layer's actual device kernels in the 1+3 graph at 410 misses per request."""
    from analyze_decode_layer_gaps import grouped_copy_layer_times

    misses = 410
    limits = (12, 24, 0)
    orders = list(permutations(limits))
    trace = tmp_path / "grouped-410.json"
    with glm_group_workload(request) as w:
        graphs = {}
        w.set_payload(misses, 43)
        for limit in limits:
            w.set_copy_core_limit(limit)
            graphs[limit] = w.capture(True, multi_layer=True)
            tokens = w.set_payload(misses, 43)
            graphs[limit].replay()
            for layer in range(w.layers):
                w.check_copied_layer(layer, tokens, misses)
        torch.npu.synchronize()
        profiler = torch_npu.profiler
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]) as prof:
            for i, order in enumerate(orders):
                w.set_payload(misses, 43 + 613 * i)
                for limit in order:
                    graphs[limit].replay()
            torch.npu.synchronize()
        prof.export_chrome_trace(str(trace))
        pieces = w.group.edges[0].pieces
        layers = w.layers

    schedule = [limit for order in orders for limit in order]
    times = grouped_copy_layer_times(trace, schedule, layers, pieces)
    print(f"[SFA_LAYER_COPY] misses_per_request={misses} requests=16 replays_per_limit={len(orders)} trace={trace}")
    print("[SFA_LAYER_COPY] layer  kernels  12-core kernel/span_us  24-core kernel/span_us  auto kernel/span_us")
    for layer, name in enumerate(("0(indexer)", "1(A)", "2(B)", "3(C)")):
        cells = []
        for limit in limits:
            active, span = times[limit, layer]
            cells.append(f"{active:.2f}/{span:.2f}")
        print(f"[SFA_LAYER_COPY] {name:<10} {1 if layer == 0 else pieces:>2}  " + "  ".join(cells))


def _row_signature(w, tokens, request_ids, layer, plane):
    # Three exactly-representable BF16 integers distinguish the plane, layer,
    # request and all 8192 source tokens, including rows with the same fill value.
    ids = ((request_ids * w.layers + layer) * w.history + tokens) * 2 + plane
    return torch.stack([ids & 255, (ids >> 8) & 255, ids >> 16], dim=-1).bfloat16()


def _tag_registered_rows(w):
    for request_id, source in enumerate(w.sources):
        for layer, storage in enumerate(source.layers):
            start = 0
            for chunk, count in zip(storage.tensors, source.chunk_token_counts, strict=True):
                tokens = torch.arange(start, start + count)
                for plane, data in enumerate(chunk.split((count * 512, count * 64))):
                    data.view(count, -1)[:, -3:].copy_(_row_signature(w, tokens, request_id, layer, plane))
                start += count


def test_shared_slice_plan_real_io_and_latency(request, tmp_path):
    """Compare the two serving schedules with identical registered pages, gates, and 20 AIVs."""
    from analyze_decode_layer_gaps import compare_shared_plan_kernels
    from sfa_retrieval_overlap_benchmark import paired_graph_times

    from vllm_ascend.compilation.sfa_retrieval_overlap import SharedPrefetchGroup

    cases = {"one_each": [1] * 16, "410_each": [410] * 16, "skew_6560": [4096, 2000, 464] + [0] * 13}
    groups_per_round, reference_ms = 19, 4.207
    trace = tmp_path / "shared-slice-plan.json"
    schedule = []
    with glm_group_workload(request) as w:
        _tag_registered_rows(w)
        w.set_copy_core_limit(20)
        dense = torch.empty_like(w.counts, memory_format=torch.contiguous_format)
        assert dense.is_contiguous() and w.counts.stride() == (16,)
        w.set_payload(410)
        legacy = w.capture(True, multi_layer=True, bridge_counts=dense)
        w.group = SharedPrefetchGroup(w.group.edges, packed_width=w.width)
        assert w.group.copy_plan is not None and w.group.pieces == 16
        planned = w.capture(True, multi_layer=True, bridge_counts=dense)
        observed = w.capture(True, observe=True, multi_layer=True, bridge_counts=dense)
        graphs = {"legacy": legacy, "planned": planned}
        lanes, columns = torch.arange(w.requests)[:, None], torch.arange(w.width)[None, :]

        def set_inputs(counts, shift):
            tokens = ((w.token_order[None, :] + lanes * 97 + shift) % w.history).int()
            # Different request pages and scatter destinations. 73 is coprime to 4096.
            local_slots = (columns * 73 + lanes * 11 + shift) % w.width
            w.selected.copy_(tokens)
            w.slots.copy_(local_slots + lanes * w.width)
            w.counts.copy_(torch.tensor(counts, dtype=torch.int32))
            return tokens, local_slots

        scenarios = [cases["410_each"], cases["skew_6560"], [0] * 16, [4096] * 16, cases["one_each"]]
        for iteration, counts in enumerate(scenarios):
            # One positive-count unbound lane also tests the source-validity mask.
            unbound = 1 if iteration == 1 else None
            sources = [None if i == unbound else s for i, s in enumerate(w.sources)]
            for layer, transfer in enumerate(w.transfers):
                transfer.bind_batch(sources, layer)
            tokens, local_slots = set_inputs(counts, 37 * iteration)
            live = (columns < torch.tensor(counts)[:, None]) & (lanes != (-1 if unbound is None else unbound))
            for graph in (legacy, planned, observed):
                dense.fill_(-123)
                for cache in w.caches:
                    for plane in cache:
                        plane.fill_(-7)
                graph.replay()
                assert dense.cpu().tolist() == counts
                for layer, cache in enumerate(w.caches):
                    for p, plane in enumerate(cache):
                        expected = torch.full((w.requests, w.width), -7, dtype=plane.dtype)
                        values = (w.values(tokens, lanes, layer) + 32 * p).to(plane.dtype)
                        expected.scatter_(1, local_slots, torch.where(live, values, -7))
                        rows = plane.view(w.requests, w.width, -1).cpu()
                        fill = rows[..., :-3]
                        assert torch.equal(fill, expected.unsqueeze(-1).expand_as(fill)), (iteration, layer, p)
                        tags = _row_signature(w, tokens, lanes, layer, p)
                        expected_tags = torch.full_like(tags, -7)
                        expected_tags.scatter_(1, local_slots.unsqueeze(-1).expand_as(tags),
                                               torch.where(live.unsqueeze(-1), tags, -7))
                        assert torch.equal(rows[..., -3:], expected_tags), (iteration, layer, p)
                        if graph is observed:
                            assert torch.equal(w.observed[layer][p].cpu(), expected), (iteration, layer, p, "join")

        # Use the identical, observation-free graphs both for timing and for the
        # trace. All inputs, synchronization and host oracle work are outside the timers.
        print(f"[SFA_PLAN] correctness=pass cases={len(scenarios)} registered_K_PE=512/64", flush=True)
        print(f"[SFA_PLAN] old_profile_extra_side_copy_ms={reference_ms} groups={groups_per_round} "
              "side_layers=57 graph_batch=19_replays_of_one_indexer_plus_ABC foreground=QKV,Vup,router single_rank")
        for case, counts in cases.items():
            set_inputs(counts, 43)
            t = paired_graph_times(legacy, planned, names=("legacy", "planned"), pairs=8,
                                   replays=groups_per_round, include_means=True,
                                   prepare_pair=lambda i, n=counts: set_inputs(n, 43 + 613 * i))
            scale = groups_per_round / 1000
            print(f"[SFA_PLAN] {case} graph_batch_mean_ms={t['legacy_mean_us'] * scale:.3f}->"
                  f"{t['planned_mean_us'] * scale:.3f} paired_graph_saved_ms={t['paired_mean_saved_us'] * scale:+.3f} "
                  f"planned_faster_pairs={t['planned_faster_pairs']}/{t['pairs']}", flush=True)
        profiler = torch_npu.profiler
        torch.npu.synchronize()
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]) as prof:
            for case, counts in cases.items():
                for pair in range(groups_per_round):
                    set_inputs(counts, 43 + 613 * pair)
                    for mode in (("planned", "legacy") if pair % 2 else ("legacy", "planned")):
                        schedule.append((case, pair, mode))
                        graphs[mode].replay()
            torch.npu.synchronize()
        prof.export_chrome_trace(str(trace))
    print(f"[SFA_PLAN] trace={trace}", flush=True)
    kernels = compare_shared_plan_kernels(trace, schedule)
    for case, k in kernels.items():
        assert k["groups"] == groups_per_round
        print(f"[SFA_PLAN] {case} side_copy_ms_57_layers={k['legacy_ms']:.3f}->{k['planned_ms']:.3f} "
              f"publisher_us={k['publisher_us']:.2f} side_saved_ms_57_layers={k['saved_ms']:+.3f} "
              f"remaining_to_4.207ms={reference_ms - k['saved_ms']:+.3f}", flush=True)


@pytest.mark.parametrize("requests,main_gate", [(1, False), (8, False), (16, False), (8, True)])
@pytest.mark.parametrize("query_rows", [1, 2])
def test_registered_sources_consumer_boundary_and_replay(requests, query_rows, main_gate):
    root = Path(os.environ.get("LMCACHE_ASCEND_SOURCE_DIR", str(ROOT.parent / "LMCache-Ascend")))
    if not (root / "benchmark/v1/kv_transfer/load_benchmark_utils.py").exists():
        pytest.fail("set LMCACHE_ASCEND_SOURCE_DIR to the matching LMCache-Ascend source checkout")
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
