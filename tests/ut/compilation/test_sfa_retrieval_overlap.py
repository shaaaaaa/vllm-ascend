# SPDX-License-Identifier: Apache-2.0
"""CPU ordering/layout contracts; native capture has separate NPU coverage."""
import ast
import gc
import symtable
import sys
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import extract, load_module
from test_sfa_full_graph import graph_module as graph_fixture

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/attention/sfa_v1.py"


def test_runner_constructor_reads_module_environment_without_local_shadowing():
    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    table = symtable.symtable(source.read_text(encoding="utf-8"), str(source), "exec")
    runner = next(t for t in table.get_children() if t.get_name() == "NPUModelRunner")
    constructor = next(t for t in runner.get_children() if t.get_name() == "__init__")
    environment = constructor.lookup("envs_ascend")
    # Inspect the ENTIRE function: a later import makes even earlier reads local.
    assert environment.is_global() and not environment.is_local()
    assert table.lookup("envs_ascend").is_imported()


@pytest.fixture
def graph_module(monkeypatch):
    return graph_fixture.__wrapped__(monkeypatch)


@pytest.fixture
def runtime(monkeypatch):
    log = []

    class Event:
        def __init__(self, enable_timing):
            assert not enable_timing
            self.recorded = False

        def record(self, stream):
            self.recorded = True
            log.append(("record", stream.name))

    class Stream:
        def __init__(self, device=None, name="transfer"):
            self.name = name

        def wait_event(self, event):
            assert event.recorded, "waiting on an unrecorded event"
            log.append(("wait", self.name))

    main = Stream(name="main")
    current = [main]

    @contextmanager
    def switch(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch, "npu", NS(Event=Event, Stream=Stream, stream=switch,
                                        current_stream=lambda: current[0]), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", NS(npu_transpose_batchmatmul=Mock()))
    monkeypatch.setitem(sys.modules, "lmcache_ascend.c_ops", NS(
        SparsePrefetchGate=lambda word: NS(set_open=lambda value: log.append(("gate", value)))
    ))
    module = load_module(ROOT / "vllm_ascend/compilation/sfa_retrieval_overlap.py", "overlap_test", monkeypatch)
    monkeypatch.setattr(module, "CaptureProbe", Mock(return_value=NS(verify=Mock())))
    return module, log


@pytest.fixture
def bench(runtime, monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "vllm_ascend.compilation.sfa_retrieval_overlap", runtime[0])
    return load_module(ROOT / "tools/sfa_retrieval_overlap_benchmark.py", "prefetch_bench", monkeypatch)


def test_shared_prefetch_benchmark_compares_the_same_distinct_host_rows(bench):
    for history in (4097, 8192):
        order = bench.scattered_token_order(4096, history)
        assert order.dtype == torch.int32 and order.unique().numel() == 4096
        for shift in (0, 199):
            tokens = (order + shift) % history
            assert tokens.min() >= 0 and tokens.max() < history and tokens.unique().numel() == 4096
        torch.testing.assert_close(order, bench.scattered_token_order(4096, history))


def test_copy_only_benchmark_captures_the_selected_grid_and_joins(bench, monkeypatch):
    w = bench.Workload.__new__(bench.Workload)
    w.selected = torch.zeros((2, 8), dtype=torch.int32)
    w.slots, w.counts = w.selected.long(), torch.ones(2, dtype=torch.int32)
    stream, gate = torch.npu.Stream(), bench.CopyGate("cpu")
    transfer = NS(load=Mock(), request_capacity=2)
    w.edges = [bench.RetrievalEdge(stream, transfer, (torch.zeros(2),), gate=gate, pieces=16)]
    w.group = None
    w.graphs = []
    monkeypatch.setattr(torch.npu, "NPUGraph", object, raising=False)
    monkeypatch.setattr(torch.npu, "graph", lambda _: nullcontext(), raising=False)
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None, raising=False)
    for limit in (12, 24, 0):
        w.set_copy_core_limit(limit)
        graph = w.capture_copy()
        assert w.graphs[-1] is graph and not w.edges[0].pending_join and gate.pending == 0
        assert transfer.load.call_count == 2  # Warmup and capture.
        for call in transfer.load.call_args_list:
            assert call.kwargs == {"max_aiv_cores": limit, "pieces": 16, "admission": gate.native}
        transfer.load.reset_mock()


@pytest.mark.parametrize("grouped", [False, True])
def test_shared_prefetch_benchmark_observes_at_first_join(bench, grouped):
    w = bench.Workload.__new__(bench.Workload)
    w.requests, w.layers, w.width, w.history = 2, 4, 8, 19
    w.selected = torch.empty((2, 8), dtype=torch.int32)
    w.slots = torch.arange(16).view(2, 8)
    w.counts = torch.empty(2, dtype=torch.int32)
    w.token_order = bench.scattered_token_order(8, 19)
    w.caches = [tuple(torch.empty((2, 8, 1, d), dtype=torch.bfloat16) for d in (32, 16)) for _ in range(4)]
    w.observed = [tuple(torch.empty((2, 8), dtype=torch.bfloat16) for _ in range(2)) for _ in range(4)]
    w.x = torch.ones(1)
    w.vup_weights = [w.x] * 4
    w.products = [None] * 4
    w.router_x = torch.ones(1, 1)
    w.router_weights = [w.router_x] * 4
    qkv_calls = []
    w.qkv_projections = [lambda i=i: qkv_calls.append(i) for i in range(4)]

    def transfer(layer):
        def load(selected, counts, slots, **kwargs):
            assert kwargs["max_aiv_cores"] == 24
            for lane, n in enumerate(counts.tolist()):
                values = w.values(selected[lane, :n], lane, layer).bfloat16()[:, None]
                for p, plane in enumerate(w.caches[layer]):
                    plane.view(16, -1)[slots[lane, :n]] = values + 32 * p
        return NS(load=load, request_capacity=2)

    w.transfers = [transfer(i) for i in range(4)]
    stream, gate = torch.npu.Stream(), bench.CopyGate("cpu")
    w.edges = [bench.RetrievalEdge(stream, w.transfers[i], w.caches[i], pieces=2, gate=gate) for i in range(1, 4)]
    w.group = bench.SharedPrefetchGroup(tuple(w.edges)) if grouped else None
    w.set_copy_core_limit(24)
    early = [tuple(torch.empty_like(t) for t in pair) for pair in w.observed[2:]]

    def sample():
        assert qkv_calls == [0, 1]
        for i, targets in enumerate(early, 2):
            w.snapshot_layer(i, targets)

    for n, shift in ((0, 1), (1, 3), (5, 7), (8, 11)):
        qkv_calls.clear()
        tokens = w.set_payload(n, shift)
        w.chain(True, observe=True, multi_layer=grouped, after_first_qkv=sample)
        assert qkv_calls == [0, 1, 2, 3]
        w.check(tokens, n)
        for layer in range(w.layers):
            w.check_copied_layer(layer, tokens, n)
        ready = [bench.ready_rows(pair, ref) for pair, ref in zip(early, w.observed[2:])]
        assert ready == [2 * n if grouped else 0] * 2
    early[0][1][0, 0] += 2
    with pytest.raises(AssertionError):
        bench.ready_rows(early[0], w.observed[2])
    w.caches[2][1][0, 0, 0, -1] += 1
    with pytest.raises(AssertionError, match="layer 2, plane 1"):
        w.check_copied_layer(2, tokens, n)


@pytest.mark.parametrize("durations", [(10, 7), (7, 10), (7, 7)])
def test_shared_prefetch_benchmark_pairs_device_times(bench, monkeypatch, durations):
    now = 0
    calls = []

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            self.timestamp = now

        def elapsed_time(self, end):
            return (end.timestamp - self.timestamp) / 1000

    def replay(mode):
        nonlocal now
        calls.append(mode)
        now += durations[mode]

    monkeypatch.setattr(torch.npu, "Event", Event)
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None, raising=False)
    graphs = [NS(replay=lambda i=i: replay(i)) for i in (0, 1)]
    prepared = []
    result = bench.paired_graph_times(*graphs, names=("cap12", "auto"), pairs=4, replays=2,
                                      prepare_pair=lambda i: prepared.append((i, len(calls))))
    assert calls == [0, 0, 1, 1] + [0, 0, 1, 1, 1, 1, 0, 0] * 2
    assert prepared == [(0, 4), (1, 8), (2, 12), (3, 16)]
    assert result == dict(cap12_us=durations[0], auto_us=durations[1],
                          paired_saved_us=durations[0] - durations[1],
                          auto_faster_pairs=4 * int(durations[1] < durations[0]), pairs=4)


def test_shared_prefetch_benchmark_uses_the_serving_quantized_qkv_shapes(bench, monkeypatch):
    native = sys.modules["torch_npu"]
    native.npu_dynamic_quant = lambda x: (x.to(torch.int8), torch.ones(x.shape[0]))
    native.npu_quant_matmul = lambda x, w, scale, *, pertoken_scale, bias, output_dtype: (
        (x.float() @ w.float()) * scale * pertoken_scale[:, None]
    ).to(output_dtype)
    scheme = type("CpuLinear", (), {"apply": extract(
        ROOT / "vllm_ascend/quantization/methods/w8a8_dynamic.py", "apply", {"torch_npu": native, "torch": torch}
    )})
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.methods.w8a8_dynamic",
                        NS(AscendW8A8DynamicLinearMethod=scheme))
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", NS(maybe_trans_nz=lambda w: w))
    w = bench.Workload.__new__(bench.Workload)
    w.device, w.layers, w.x = torch.device("cpu"), 4, torch.ones((2, 1), dtype=torch.bfloat16)
    config = dict(hidden_size=16, q_lora_rank=32, kv_lora_rank=512, qk_rope_head_dim=64,
                  num_attention_heads=8, v_head_dim=16, n_routed_experts=4)
    w.use_glm_projections(config, tp=4)
    assert all(weight.shape == (2, 512, 16) for weight in w.vup_weights)
    assert all(weight.shape == (4, 16) for weight in w.router_weights)
    assert all(len({tensor.data_ptr() for tensor in weights}) == 4 for weights in (w.vup_weights, w.router_weights))
    assert len({projection.args[0].weight.data_ptr() for projection in w.qkv_projections}) == 4
    for projection in w.qkv_projections:
        torch.testing.assert_close(projection(), torch.full((2, 32 + 512 + 64), 16, dtype=torch.bfloat16))


@dataclass(frozen=True)
class Key:
    request_capacity: int = 2
    token_capacity: int = 4
    query_profile: str = "bounded"


def topology(lengths=(4,), dtype=torch.bfloat16, *, max_aiv_cores=0):
    groups, impls, caches = [], {}, {}
    for g, length in enumerate(lengths):
        group = NS(members=tuple(f"g{g}l{i}" for i in range(length)), active=True)
        groups.append(group)
        for i, name in enumerate(group.members):
            caches[name] = (torch.empty((2, 8, 1, 4), dtype=dtype), torch.empty((2, 8, 1, 2), dtype=dtype))
            impls[name] = NS(skip_topk=i > 0, shared_resident_plan=group, sfa_copy_max_aiv_cores=max_aiv_cores,
                             _full_graph_transfers={2: NS(load=Mock(), request_capacity=2)},
                             _staged_sfa_bridge_buffers=(torch.empty(1, dtype=dtype),),
                             _staged_sfa_capture_state=NS(runtime=(None, caches[name]), initialized_cache_capacity=2))
    return groups, impls, caches


@pytest.mark.parametrize("limit", [0, 12, 24, 40])
def test_prefetch_pieces_only_split_the_side_transfer(runtime, limit):
    module, _ = runtime
    groups, impls, caches = topology((3,), max_aiv_cores=limit)
    owner = module.SharedRetrievalOverlap(pieces=16)
    owner.configure(tuple(impls.items()), groups, caches)
    key = Key()
    owner.prepare(key, tuple(impls.items()), groups, 4)
    first = owner.outgoing[key]["g0l0"]
    selected = torch.zeros((2, 4), dtype=torch.int32)
    counts = torch.ones(2, dtype=torch.int32)
    slots = torch.zeros((2, 4), dtype=torch.int64)
    first.launch(selected, counts, slots, first.destinations)
    first.join()
    first.transfer.load.assert_called_once_with(selected, counts, slots, max_aiv_cores=limit, pieces=16)
    impls["g0l0"]._full_graph_transfers[2].load.assert_not_called()


@pytest.mark.parametrize("pieces", [1, 8])
@pytest.mark.parametrize("windows", [(), ("vup",), ("router",), ("vup", "router")])
def test_main_gate_captures_one_train_shared_across_windows(runtime, pieces, windows):
    module, log = runtime
    groups, impls, caches = topology((3,))
    owner = module.SharedRetrievalOverlap(pieces=pieces, main_gate=True)
    owner.configure(tuple(impls.items()), groups, caches)
    key, other, large = Key(), Key(query_profile="fixed"), Key(token_capacity=9)
    for shape in (key, other, large):
        owner.prepare(shape, tuple(impls.items()), groups, 4)
    assert not owner.outgoing[large]
    data = torch.zeros((1, 4), dtype=torch.int32)
    gates = []
    for name, edge in owner.outgoing[key].items():
        assert edge.gate is not owner.outgoing[other][name].gate
        gates.append(edge.gate)
        edge.transfer.load.side_effect = lambda *a, **kw: log.append(("whole train", torch.npu.current_stream().name))
        log.clear()
        edge.launch(data, data, data, edge.destinations)
        assert log == [("record", "main"), ("wait", "transfer"), ("whole train", "transfer"), ("record", "transfer")]
        with pytest.raises(RuntimeError, match="missing a prefetch launch or consumer wait"):
            owner.finish_capture(key)
        log.clear()
        for window in windows:
            with edge.cube_window(window):
                log.append(("compute", window))
        edge.join()
        assert log == [entry for w in windows for entry in (("gate", True), ("compute", w), ("gate", False))] + [
            ("gate", True), ("wait", "main"), ("gate", False),
        ]
        edge.transfer.load.assert_called_once_with(data, data, data, pieces=min(pieces, data.numel()),
                                                   max_aiv_cores=0, admission=edge.gate.native)
        assert edge.windows_used == set(windows) and not edge.gate.pending
    assert gates[0] is gates[1]
    owner.finish_capture(key)
    owner.finish_capture(large)


def test_gate_rejects_overlapping_edges_duplicate_and_nested_windows(runtime):
    module, _ = runtime
    data = torch.zeros(8)
    gate = module.CopyGate(data.device)
    edges = [module.RetrievalEdge(torch.npu.Stream(), NS(load=Mock()), (data,), pieces=4, gate=gate) for _ in range(2)]
    first, second = edges
    first.launch(data, data, data, (data,))
    with pytest.raises(RuntimeError, match="twice"):
        second.launch(data, data, data, (data,))
    with first.cube_window("router"):
        with pytest.raises(RuntimeError, match="unexpected"), first.cube_window("vup"):
            pytest.fail("nested window")
        with pytest.raises(RuntimeError, match="no producer"):
            first.join()
    for invalid in ("router", "vup", "moe"):
        with pytest.raises(RuntimeError, match="unexpected"), first.cube_window(invalid):
            pytest.fail("duplicate or out-of-order window")
    first.join()
    second.launch(data, data, data, (data,))
    second.join()


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("key", [Key(), Key(query_profile="fixed")])
@pytest.mark.parametrize("limit", [0, 24])
def test_indexer_group_forks_once_and_waits_for_each_consumers_own_kv(runtime, gated, key, limit):
    module, log = runtime
    groups, impls, caches = topology((1, 4, 2, 3), max_aiv_cores=limit)
    owner = module.SharedRetrievalOverlap(main_gate=gated, multi_layer=True, pieces=4)
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(key, tuple(impls.items()), groups, 4)
    incoming, outgoing = owner.edges(key)
    batches = owner.indexer_groups[key]
    assert set(batches) == {"g1l0", "g2l0", "g3l0"}
    selected = torch.tensor([[0, 3, 5, 7], [2, 4, 6, 1], [-1] * 4], dtype=torch.int32)
    slots = torch.tensor([[0, 2, 4, 5], [6, 7, 8, 9], [-1] * 4], dtype=torch.int64)
    counts = torch.tensor([3, 0, 0], dtype=torch.int32)
    main = torch.npu.current_stream()
    main.wait_event = Mock(wraps=main.wait_event)
    for group in groups[1:]:
        producer, *consumers = group.members
        batch = batches[producer]
        assert batch.edges == tuple(incoming[name] for name in consumers)
        assert producer not in incoming and consumers[-1] not in outgoing
        payloads = []
        for index, (name, edge) in enumerate(zip(consumers, batch.edges)):
            # Each destination must use its own layer's source. Verify shifts, idle lanes and unchanged slots.
            for plane in caches[name]:
                plane.fill_(-1)

            def load(tokens, numbers, targets, *, cache=caches[name], layer=index, seen=payloads, **kw):
                assert kw["max_aiv_cores"] == limit
                seen.append((tokens, numbers, targets))
                for lane, n in enumerate(numbers.tolist()):
                    for p, plane in enumerate(cache):
                        flat = plane.view(-1, plane.shape[-1])
                        flat[targets[lane, :n]] = (100 * layer + 10 * p + tokens[lane, :n].float())[:, None].to(flat)
            edge.transfer.load.side_effect = load

        log.clear()
        batch.launch(selected, counts, slots)
        assert log == [("record", "main"), ("wait", "transfer")] + [("record", "transfer")] * len(consumers)
        assert all(e.pending_join for e in batch.edges)
        assert len(payloads) == len(consumers)
        assert all(t.data_ptr() == source.data_ptr() and t.shape[0] == key.request_capacity
                   for args in payloads for t, source in zip(args, (selected, counts, slots)))
        for index, edge in enumerate(batch.edges):
            if gated:
                transitions = []
                edge.gate.native.set_open = transitions.append
                with edge.cube_window("vup"):
                    assert transitions == [True]
                assert transitions == [True, False]
                transitions.clear()
            main.wait_event.reset_mock()
            def project(x, completed=edge.done, gated=gated, native_log=transitions if gated else None):
                main.wait_event.assert_called_once_with(completed)
                if gated:
                    assert native_log == [True]
                return x
            assert edge.join(project, counts) is counts
            main.wait_event.assert_called_once_with(edge.done)
            if gated:
                assert transitions == [True, False] and edge.gate.pending == len(consumers) - index - 1
            assert all(e.pending_join for e in batch.edges[index + 1:])
            for p, plane in enumerate(caches[consumers[index]]):
                actual = plane.view(-1, plane.shape[-1])[:, 0]
                expected = torch.full_like(actual, -1)
                expected[slots[0, :3]] = (100 * index + 10 * p + selected[0, :3]).to(expected)
                torch.testing.assert_close(actual, expected)
        impls[producer]._full_graph_transfers[key.request_capacity].load.assert_not_called()
    owner.finish_capture(key)
    if gated:
        assert len({e.gate for e in incoming.values()}) == 1
    owner.clear()
    assert not owner.indexer_groups


def test_indexer_lookahead_keeps_partially_eligible_groups_on_adjacent_schedule(runtime):
    module, _ = runtime
    groups, impls, caches = topology((4, 4))
    impls["g0l1"]._staged_sfa_bridge_buffers = (torch.empty(1, dtype=torch.float32),)
    owner = module.SharedRetrievalOverlap(main_gate=True, multi_layer=True)
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(Key(), tuple(impls.items()), groups, 4)
    incoming, _ = owner.edges(Key())
    assert set(owner.indexer_groups[Key()]) == {"g1l0"}
    assert "g0l2" not in incoming and not incoming["g0l3"].from_indexer
    assert all(incoming[n].from_indexer for n in groups[1].members[1:])


def test_prepare_initializes_only_unwarmed_kv_before_the_indexer_can_write(runtime):
    module, _ = runtime
    groups, impls, caches = topology((4,))
    state = impls["g0l3"]._staged_sfa_capture_state
    state.initialized_cache_capacity = 1
    for plane in caches["g0l3"]:
        plane.fill_(7)
    owner = module.SharedRetrievalOverlap(main_gate=True, multi_layer=True)
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(Key(), tuple(impls.items()), groups, 4)
    assert state.initialized_cache_capacity == 2
    for plane in caches["g0l3"]:
        assert torch.all(plane[0] == 7) and torch.all(plane[1] == 0)


def test_three_layer_device_commands_allow_futures_to_finish_during_first_qkv(runtime):
    module, _ = runtime
    groups, impls, caches = topology((4,))
    owner = module.SharedRetrievalOverlap(main_gate=True, multi_layer=True, pieces=2)
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(Key(), tuple(impls.items()), groups, 4)
    batch = owner.indexer_groups[Key()]["g0l0"]
    root, side = [], []
    for i, edge in enumerate(batch.edges):
        # Expand the native CANN value/Notify commands in their production order.
        edge.transfer.load.side_effect = lambda *a, pieces, i=i, **kw: side.extend(
            [("check", i), ("take", i), ("copy", i), ("give", i)] * pieces
        )
        edge.done.record = lambda stream, i=i: side.append(("done", i))
    event_layers = {e.done: i for i, e in enumerate(batch.edges)}
    torch.npu.current_stream().wait_event = lambda e: root.append(("join", event_layers[e]))
    batch.edges[0].gate.native.set_open = lambda value: root.extend(
        [("open", 0), ("give", 0)] if value else [("close", 0), ("take", 0)]
    )
    data = torch.ones(2, 2)
    batch.launch(data, data, data)
    for i, edge in enumerate(batch.edges):
        root.append(("protected", i))
        with edge.cube_window("vup"):
            root.append(("gemm", i))
        root.append(("protected", i))
        with edge.cube_window("router"):
            root.append(("gemm", i))
        root.append(("protected", i))
        edge.join(lambda i=i: root.extend([("qkv begin", i), ("qkv end", i)]))
    root.append(("protected", 3))
    first_qkv_counts = set()
    done_pc = {i: pc for pc, (command, i) in enumerate(side) if command == "done"}
    copy_count = [sum(op == "copy" for op, _ in side[:pc]) for pc in range(len(side) + 1)]

    @lru_cache(None)
    def explore(front, back, words, free_tokens, side_owner, at_qkv_start):
        if front == len(root) and back == len(side):
            assert words == free_tokens == 0 and side_owner == at_qkv_start == -1
            return
        advanced = False
        for foreground, queue, pc in ((True, root, front), (False, side, back)):
            if pc == len(queue):
                continue
            op, layer = queue[pc]
            w, t, holder, qkv_count = words, free_tokens, side_owner, at_qkv_start
            if op == "open":
                w = 1
            elif op == "close":
                w = 0
            elif op == "check" and not w:
                continue
            elif op == "take":
                if not t:
                    continue
                t = 0
                if not foreground:
                    assert holder == -1
                    holder = layer
            elif op == "give":
                assert not t and holder == (-1 if foreground else layer)
                t = 1
                if not foreground:
                    holder = -1
            elif op == "copy":
                assert holder == layer
            elif op == "join" and back <= done_pc[layer]:
                continue
            elif op == "protected":
                assert w == t == 0 and holder == -1
            elif op == "qkv begin":
                assert w and back > done_pc[layer]
                if layer == 0:
                    qkv_count = copy_count[back]
            elif op == "qkv end" and layer == 0:
                first_qkv_counts.add((qkv_count, copy_count[back]))
                qkv_count = -1
            advanced = True
            explore(front + int(foreground), back + int(not foreground), w, t, holder, qkv_count)
        assert advanced, "device queues deadlocked"

    explore(0, 0, 0, 0, -1, -1)
    assert first_qkv_counts == {(start, end) for start in range(2, 7) for end in range(start, 7)}


def test_failed_copy_never_submits_a_consumer_wait_on_an_unrecorded_done(runtime):
    module, log = runtime
    dst = torch.zeros(4)
    transfer = NS(load=Mock(side_effect=RuntimeError("copy capture failed")))
    edge = module.RetrievalEdge(torch.npu.Stream(), transfer, (dst,), gate=module.CopyGate(dst.device))
    log.clear()
    with pytest.raises(RuntimeError, match="copy capture failed"):
        edge.launch(dst, dst, dst, (dst,))
    with pytest.raises(RuntimeError, match="no producer"):
        edge.join()
    assert log == [("record", "main"), ("wait", "transfer")]


@pytest.mark.parametrize("fail", [False, True])
def test_target_validates_main_gate_only_after_successful_capture(graph_module, fail):
    module, context, captures, _ = graph_module
    wrapper = module.SFAFullGraph()
    finish = Mock()
    outbound = {"layer": object()}
    previous = {"unrelated": object()}
    context.sfa_router_copy_edges = previous
    wrapper.retrieval_overlap = NS(main_gate=True, finish_capture=finish, edges=lambda key: ({}, outbound))

    def run_target(**kwargs):
        assert context.sfa_router_copy_edges is outbound
        if fail:
            raise ValueError("capture aborted")
        return kwargs["input_ids"]

    target = Mock(side_effect=run_target)
    data = torch.ones(1)
    if fail:
        with pytest.raises(ValueError, match="capture aborted"):
            wrapper.run(target, input_ids=data)
        finish.assert_not_called()
    else:
        wrapper.run(target, input_ids=data)
        wrapper.seal(("q2",))
        context.staged_sfa_graph_dummy_run = False
        wrapper.run(target, input_ids=data)
        finish.assert_called_once_with("q2")
        assert target.call_count == 1 and captures[0][0].replay.call_count == 1
    assert context.sfa_router_copy_edges is previous


def test_main_gate_configuration(runtime, monkeypatch):
    module, _ = runtime
    monkeypatch.setitem(sys.modules, "vllm.v1.core.dsa_shared_pool", NS(layerwise_prefill_bank_count=lambda: 4))
    env = load_module(ROOT / "vllm_ascend/envs.py", "sfa_gate_envs", monkeypatch)
    name = "VLLM_ASCEND_SFA_PREFETCH_MULTI_LAYER"
    monkeypatch.delenv(name, raising=False)
    assert not env.VLLM_ASCEND_SFA_PREFETCH_MULTI_LAYER
    monkeypatch.setenv(name, "1")
    assert env.VLLM_ASCEND_SFA_PREFETCH_MULTI_LAYER
    monkeypatch.setenv(name, "2")
    with pytest.raises(ValueError, match="0 or 1"):
        _ = env.VLLM_ASCEND_SFA_PREFETCH_MULTI_LAYER
    for name in ("PIECES", "MAIN_GATE", "MOE_GATE"):
        monkeypatch.delenv("VLLM_ASCEND_SFA_PREFETCH_" + name, raising=False)
    assert not env.VLLM_ASCEND_SFA_PREFETCH_MAIN_GATE
    assert env.VLLM_ASCEND_SFA_PREFETCH_PIECES == 1
    monkeypatch.setenv("VLLM_ASCEND_SFA_PREFETCH_MOE_GATE", "1")
    assert env.VLLM_ASCEND_SFA_PREFETCH_MAIN_GATE and env.VLLM_ASCEND_SFA_PREFETCH_PIECES == 16
    monkeypatch.setenv("VLLM_ASCEND_SFA_PREFETCH_MAIN_GATE", "0")
    assert not env.VLLM_ASCEND_SFA_PREFETCH_MAIN_GATE
    monkeypatch.setenv("VLLM_ASCEND_SFA_PREFETCH_MAIN_GATE", "1")
    assert env.VLLM_ASCEND_SFA_PREFETCH_MOE_GATE
    monkeypatch.setenv("VLLM_ASCEND_SFA_PREFETCH_PIECES", "4")
    assert env.VLLM_ASCEND_SFA_PREFETCH_PIECES == 4
    with pytest.raises(ValueError, match="LOW_PRIORITY=0"):
        module.SharedRetrievalOverlap(4, True, main_gate=True)
    monkeypatch.setenv("VLLM_ASCEND_SFA_PREFETCH_MAIN_GATE", "2")
    with pytest.raises(ValueError, match="0 or 1"):
        _ = env.VLLM_ASCEND_SFA_PREFETCH_MAIN_GATE


@pytest.mark.parametrize(("raw", "expected"), [
    (None, 0), ("", 0), ("auto", 0), ("default", 0), ("0", 0), ("12", 12),
    (" 24 ", 24), ("40", 40), ("4294967295", 4294967295), ("-1", None), ("4294967296", None),
])
def test_decode_copy_core_configuration(monkeypatch, raw, expected):
    monkeypatch.setitem(sys.modules, "vllm.v1.core.dsa_shared_pool", NS(layerwise_prefill_bank_count=lambda: 4))
    env = load_module(ROOT / "vllm_ascend/envs.py", "copy_core_envs", monkeypatch)
    name = "VLLM_ASCEND_SFA_COPY_MAX_AIV_CORES"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    if expected is None:
        with pytest.raises(ValueError, match=name):
            _ = env.VLLM_ASCEND_SFA_COPY_MAX_AIV_CORES
    else:
        actual = env.VLLM_ASCEND_SFA_COPY_MAX_AIV_CORES
        assert actual == expected


def test_priority_identifies_only_one_new_queue_and_verifies_piece_count(monkeypatch):
    path = ROOT / "vllm_ascend/compilation/sfa_copy_priority.py"
    module = load_module(path, "copy_priority_test", monkeypatch)
    tracker = module.CapturedCopyPriority.__new__(module.CapturedCopyPriority)
    tracker.main, tracker.side, tracker.graph = 60, None, 900
    queues, levels = [60, 77], {60: 0, 77: 0}
    tracker.handles = lambda api, obj: queues if api == "aclmdlRIGetStreams" else [2, 4, 3, 5]
    tracker.capture = lambda stream: 900
    tracker.stream_id = lambda stream: stream
    priority_calls = []

    def call(api, *args, operation=None):
        if api == "aclmdlRITaskGetType":
            args[1]._obj.value = 1 if args[0] in (3, 5) else 3
        elif api == "aclrtDeviceGetStreamPriorityRange":
            args[0]._obj.value, args[1]._obj.value = 7, 0
        elif api == "aclrtSetStreamAttribute":
            priority_calls.append((operation, args[0]))
            levels[args[0]] = args[2]._obj.priority
        elif api == "aclrtGetStreamAttribute":
            priority_calls.append((operation, args[0]))
            args[2]._obj.priority = levels[args[0]]
        else:
            raise AssertionError(api)

    tracker.call = call
    queues.append(78)
    with pytest.raises(RuntimeError, match="uniquely identify"):
        tracker.after_first_fork(object(), {60})
    queues.pop()
    tracker.after_first_fork(object(), {60})
    with pytest.raises(RuntimeError, match="expected 4"):
        tracker.finish(2, 2)
    assert levels == {60: 0, 77: 0}
    assert not priority_calls
    tracker.finish(1, 2)
    assert levels == {60: 0, 77: 7}
    assert priority_calls == [("set copy", 77), ("read main after copy set", 60), ("read copy after set", 77)]


@pytest.mark.parametrize("failure", ["set copy", "read main after copy set", "read copy after set"])
def test_graph_worker_names_the_driver_operation_that_failed(monkeypatch, failure):
    module = load_module(ROOT / "vllm_ascend/compilation/sfa_copy_priority.py", "priority_error_test", monkeypatch)
    tracker = module.CapturedCopyPriority.__new__(module.CapturedCopyPriority)
    tracker.main, tracker.side, tracker.graph = 60, 77, 900
    tracker.handles = lambda api, obj: [60, 77] if api == "aclmdlRIGetStreams" else [1]

    def task_type(task, result):
        result._obj.value = module.ACL_MODEL_RI_TASK_KERNEL
        return 0

    def priority_range(least, greatest):
        least._obj.value, greatest._obj.value = 7, 0
        return 0

    def read(stream, attribute, result):
        result._obj.priority = 0 if stream == 60 else 7
        role = "read main after copy set" if stream == 60 else "read copy after set"
        return 507899 if role == failure else 0

    tracker.functions = {
        "aclmdlRITaskGetType": task_type,
        "aclrtDeviceGetStreamPriorityRange": priority_range,
        "aclrtSetStreamAttribute": lambda *args: 507899 if failure == "set copy" else 0,
        "aclrtGetStreamAttribute": read,
    }
    with pytest.raises(RuntimeError) as exc:
        tracker.finish(1, 1)
    assert f"({failure}) rc=507899" in str(exc.value)


@pytest.mark.parametrize("lengths", [(1,), (2,), (4,), (2, 3)])
def test_edges_never_cross_groups_and_keep_one_stream(runtime, lengths):
    module, log = runtime
    groups, impls, caches = topology(lengths)
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    assert owner.stream is None
    for key in (Key(), Key(query_profile="fixed")):
        owner.prepare(key, tuple(impls.items()), groups, 1024)
        incoming, outgoing = owner.edges(key)
        assert len(incoming) == len(outgoing) == sum(n - 1 for n in lengths)
        for group in groups:
            assert group.members[0] not in incoming
            assert group.members[-1] not in outgoing
            for before, after in zip(group.members, group.members[1:]):
                assert outgoing[before] is incoming[after]
                assert incoming[after].stream is owner.stream
    module.CaptureProbe.assert_called_once() if owner.pairs else module.CaptureProbe.assert_not_called()
    previous = list(log)
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    assert log == previous  # No per-key reinitialization.


@pytest.mark.parametrize("dtype,tokens", [(torch.float32, 4), (torch.bfloat16, 2048)])
def test_ineligible_paths_have_no_stream_or_probe(runtime, dtype, tokens):
    module, log = runtime
    groups, impls, caches = topology(dtype=dtype)
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    key = Key(token_capacity=tokens)
    owner.prepare(key, tuple(impls.items()), groups, 1024)
    assert owner.edges(key) == ({}, {})
    assert owner.stream is None and not log
    module.CaptureProbe.assert_not_called()


def test_layout_accepts_disjoint_views_and_rejects_real_alias(runtime):
    module, _ = runtime
    backing = torch.zeros(64)
    caches = {"producer": (backing[:16], backing[16:32]), "consumer": (backing[32:48], backing[48:])}
    module.validate_destinations(caches, {"consumer"})
    caches["consumer"] = (backing[30:46], backing[48:])
    with pytest.raises(ValueError, match="ranges overlap"):
        module.validate_destinations(caches, {"consumer"})
    caches["consumer"] = (backing[32:48:2], backing[48:])
    with pytest.raises(ValueError, match="contiguous"):
        module.validate_destinations(caches, {"consumer"})


@pytest.mark.parametrize("fault", ["missing", "destination", "probe"])
def test_setup_failure_cannot_be_reused(runtime, fault):
    module, _ = runtime
    groups, impls, caches = topology()
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    if fault == "missing":
        impls["g0l1"]._full_graph_transfers.clear()
    elif fault == "destination":
        impls["g0l1"]._staged_sfa_capture_state.runtime = (None, tuple(t.clone() for t in caches["g0l1"]))
    else:
        module.CaptureProbe.return_value.verify.side_effect = RuntimeError("native capture failed")
    with pytest.raises(RuntimeError):
        owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    assert owner.failed
    if fault == "probe":
        assert owner.probe is module.CaptureProbe.return_value
    with pytest.raises(RuntimeError, match="restart"):
        owner.prepare(Key(), tuple(impls.items()), groups, 1024)


def test_actual_query_dtype_controls_cube_eligibility(runtime):
    module, log = runtime
    groups, impls, caches = topology()
    for impl in impls.values():
        impl._staged_sfa_bridge_buffers = (torch.empty(1, dtype=torch.float32),)
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    assert owner.edges(Key()) == ({}, {})
    assert owner.stream is None and not log


def test_edge_uses_explicit_payload_and_refuses_substituted_destination(runtime):
    module, log = runtime
    destination = torch.zeros(4)
    transfer = NS(load=Mock())
    edge = module.RetrievalEdge(torch.npu.Stream(), transfer, (destination,))
    selected, counts, slots = torch.arange(4), torch.tensor([4]), torch.arange(4)
    log.clear()
    edge.launch(selected, counts, slots, (destination.view(4),))
    edge.join()
    transfer.load.assert_called_once_with(selected, counts, slots)
    assert log == [("record", "main"), ("wait", "transfer"), ("record", "transfer"), ("wait", "main")]
    with pytest.raises(RuntimeError, match="destination changed"):
        edge.launch(selected, counts, slots, (destination.clone(),))
    with pytest.raises(RuntimeError, match="payload storage changed"):
        edge.launch(selected.clone(), counts, slots, (destination,))
    assert transfer.load.call_count == 1


def test_edge_retains_actual_payload_until_cleanup(runtime):
    module, _ = runtime
    edge = module.RetrievalEdge(torch.npu.Stream(), NS(load=lambda *args: None), (torch.zeros(4),))
    selected, counts, slots = torch.arange(4), torch.tensor([4]), torch.arange(4)
    refs = [weakref.ref(t) for t in (selected, counts, slots)]
    edge.launch(selected, counts, slots, edge.destinations)
    del selected, counts, slots
    assert all(ref() is not None for ref in refs)
    del edge  # The root owner synchronizes before dropping edges in production.
    assert all(ref() is None for ref in refs)


def test_capture_join_requires_a_launch_not_just_a_primed_event(runtime):
    module, log = runtime
    edge = module.RetrievalEdge(torch.npu.Stream(), NS(load=Mock()), (torch.zeros(4),))
    log.clear()
    with pytest.raises(RuntimeError, match="no producer launch"):
        edge.join()
    assert not log  # Reject before submitting an invalid native event wait.
    selected, counts, slots = torch.arange(4), torch.tensor([4]), torch.arange(4)
    for _ in range(2):
        edge.launch(selected, counts, slots, edge.destinations)
        with pytest.raises(RuntimeError, match="twice"):
            edge.launch(selected, counts, slots, edge.destinations)
        edge.join()
        with pytest.raises(RuntimeError, match="no producer launch"):
            edge.join()


def test_no_owner_cycle_with_gc_disabled(runtime):
    module, _ = runtime
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        owner = module.SharedRetrievalOverlap()
        groups, impls, caches = topology()
        owner.configure(tuple(impls.items()), groups, caches)
        owner.prepare(Key(), tuple(impls.items()), groups, 1024)
        ref = weakref.ref(owner)
        del owner, impls
        assert ref() is None
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("already_queued", [False, True])
def test_post_forks_only_adjacent_transfers_and_preserves_vup_window(gated, already_queued):
    log = []
    value = torch.ones(2)
    ns = dict(torch=torch, get_weight_prefetch_method=lambda: NS(
        maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kw: None), MAX_O_PROJ_PREFETCH_SIZE=0)
    post = extract(SOURCE, "_cross_layer_post_compute", ns)
    proj = Mock(side_effect=lambda x: (log.append("output") or x,))
    impl = NS(_execute_sparse_flash_attention_process=lambda *a, **kw: log.append("sfa") or value,
              _v_up_proj=lambda x, **kw: log.append(("cube", kw)) or x, o_proj=proj)
    edge = NS(gate=object() if gated else None, from_indexer=already_queued, pending_join=already_queued,
              launch=lambda *args: log.append("prefetch"))
    post(impl, value, value, value, value, value, value, value, value, torch.empty_like(value),
         trace_label="test", prefetch=(edge, value, value, value, (value,)))
    assert log == ["sfa", *([] if already_queued else ["prefetch"]), ("cube", {"copy_edge": edge} if gated else {}),
                   "output"]


def test_compiled_fp32_router_resolves_current_layer_at_capture():
    from torch.fx.experimental.proxy_tensor import make_fx

    source = ROOT / "vllm_ascend/patch/worker/patch_unquantized_gemm.py"
    context = NS()
    visits = []

    @contextmanager
    def window(name):
        visits.append(("open", name))
        yield
        visits.append(("close", name))

    def linear(x, weight):
        assert x.dtype == weight.dtype == torch.float32
        visits.append("fp32 linear")
        return torch.nn.functional.linear(x, weight)

    routed = extract(source, "sfa_router_gemm", dict(torch=torch, is_forward_context_available=lambda: True,
                     get_forward_context=lambda: context, unquantized_gemm=linear))
    tagged = torch.library.custom_op("test_sfa_windows::router", mutates_args=())(routed)
    tagged.register_fake(extract(source, "sfa_router_gemm_fake", dict(torch=torch,
                         unquantized_gemm_fake=lambda x, weight: x.new_empty((x.shape[0], weight.shape[0])))))
    x = torch.ones((3, 4), dtype=torch.bfloat16)
    weight = torch.arange(24, dtype=torch.float32).view(6, 4)
    model = make_fx(lambda a, b: tagged(a.float(), b, "layer0").bfloat16(), tracing_mode="fake")(x, weight)
    for key in ("layer1", "layer0"):
        context.sfa_router_copy_edges = {key: NS(cube_window=window)}
        visits.clear()
        torch.testing.assert_close(model(x, weight), torch.nn.functional.linear(x.float(), weight).bfloat16())
        expected = [("open", "router"), "fp32 linear", ("close", "router")] if key == "layer0" else ["fp32 linear"]
        assert visits == expected
    visits.clear()
    routed(torch.ones((3, 8))[:, ::2], weight, "layer0")
    assert visits == ["fp32 linear"]


def test_pre_joins_before_any_consumer_cache_access():
    log = []
    context = NS(staged_sfa_graph_key=Key())
    pre = extract(SOURCE, "cross_layer_graph_pre", {"get_forward_context": lambda: context})
    def cache(*args):
        log.append("cache")
        raise RuntimeError("stop after first access")
    edge = NS(from_indexer=False, join=lambda: log.append("join"))
    impl = NS(_retrieval_overlap=object(), _shared_retrieval_edges=lambda: ({"consumer": edge}, {}),
              _cross_layer_kv_cache=cache)
    with pytest.raises(RuntimeError, match="first access"):
        pre(impl, "consumer", torch.ones(1), (), NS(), False, torch.empty(1))
    assert log == ["join", "cache"]


@pytest.mark.parametrize("limit", [0, 12, 24, 40])
def test_serial_retrieve_suppressed_only_for_incoming_full_graph_edge(limit):
    context = NS(staged_sfa_graph_key=Key(), sfa_full_graph_active=True)
    retrieve = extract(SOURCE, "cross_layer_lmcache_retrieve",
                       {"_staged_sfa_profile_scope": lambda _: nullcontext()})
    load = Mock()
    impl = NS(_retrieval_overlap=object(), sfa_copy_max_aiv_cores=limit,
              _shared_retrieval_edges=lambda: ({"consumer": object()}, {}),
              _full_graph_transfer=NS(load=load))
    data = torch.ones(2, 4)
    retrieve(impl, "producer", "consumer", data, data, data, NS(), context)
    retrieve(impl, "consumer", "", data, data, data, NS(), context)
    assert load.call_count == 1
    impl._retrieval_overlap = None
    retrieve(impl, "consumer", "", data, data, data, NS(), context)
    assert load.call_count == 2
    assert all(call.kwargs == {"max_aiv_cores": limit} for call in load.call_args_list)


@pytest.mark.parametrize("active", [False, True])
def test_edges_are_only_used_in_root_graph(active):
    context = NS(sfa_full_graph_active=active, staged_sfa_graph_key=Key())
    get_edges = extract(SOURCE, "_shared_retrieval_edges", {"get_forward_context": lambda: context})
    owner = NS(edges=Mock(return_value=({"in": 1}, {"out": 2})))
    impl = NS(_retrieval_overlap=owner, shared_resident_plan=NS(active=True))
    assert get_edges(impl) == (({"in": 1}, {"out": 2}) if active else ({}, {}))
    assert owner.edges.call_count == int(active)


def test_private_post_operator_passes_call_payload_not_hidden_plan():
    selected, counts, slots = torch.ones(4, 8), torch.ones(4), torch.ones(4, 8)
    dest = [torch.empty(4)]
    edge = NS(transfer=NS(request_capacity=2), destinations=dest)
    impl = NS(_shared_retrieval_edges=lambda: ({}, {"producer": edge}), cross_layer_graph_post=Mock())
    op = extract(ROOT / "vllm_ascend/ops/mla.py", "sfa_forward_post_prefetch",
                 {"_mla_runtime_state": lambda name: (impl, "producer", (), NS())})
    op(selected, selected, selected, selected, counts, slots, selected, "wrapper")
    actual = impl.cross_layer_graph_post.call_args.kwargs["prefetch"]
    assert actual[0] is edge and actual[-1] is dest
    for sliced, source in zip(actual[1:4], (selected, counts, slots)):
        assert sliced.data_ptr() == source.data_ptr() and sliced.shape[0] == 2


def test_prefetch_operator_preserves_output_effect_ordering():
    tree = ast.parse((ROOT / "vllm_ascend/ops/mla.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "direct_register_custom_op"]
    matches = [n for n in calls if any(k.arg == "op_name" and isinstance(k.value, ast.Constant)
                                      and k.value.value == "sfa_forward_post_prefetch" for k in n.keywords)]
    assert len(matches) == 1
    writes = next(k.value for k in matches[0].keywords if k.arg == "mutates_args")
    assert ast.literal_eval(writes) == ["output"]


def test_partial_capture_failure_keeps_owner_until_synchronized_clear(graph_module, monkeypatch):
    module, context, _, _ = graph_module
    root = module.SFAFullGraph()
    events = []
    owner = NS(stream=object(), clear=lambda: events.append("clear"))
    root.retrieval_overlap = owner
    monkeypatch.setattr(torch.npu, "synchronize", lambda: events.append("sync"))
    with pytest.raises(RuntimeError, match="partial submission"):
        root.run(Mock(side_effect=RuntimeError("partial submission")))
    assert root._submission_failed and root.retrieval_overlap is owner
    assert not context.sfa_full_graph_active and not context.capturing
    with pytest.raises(RuntimeError):
        root.run(Mock())
    root.clear()
    assert events == ["sync", "clear"]


def test_live_replay_never_reenters_overlap_python(graph_module):
    module, context, _, _ = graph_module
    root = module.SFAFullGraph()
    root.retrieval_overlap = NS(stream=object(), clear=Mock())
    launch = Mock()
    root.run(launch)
    root.seal((context.staged_sfa_graph_key,))
    context.staged_sfa_graph_dummy_run = False
    for _ in range(8):
        root.run(launch)
    assert launch.call_count == 1
    assert root.replay_count == 8


def test_lease_retirement_waits_for_root_completion(graph_module):
    module, _, _, _ = graph_module
    root = module.SFAFullGraph()
    lease, completion = NS(close=Mock()), NS(query=Mock(return_value=False))
    root.retired_sources.append(NS(lease=lease, completion=completion))
    root.collect_retired_sources()
    lease.close.assert_not_called()
    completion.query.return_value = True
    root.collect_retired_sources()
    lease.close.assert_called_once()


@pytest.mark.parametrize("active", [0, 1, 8])
def test_each_layer_copies_once_with_layer_specific_values(runtime, active):
    module, log = runtime
    groups, impls, caches = topology()
    selected, counts, slots = torch.arange(8), torch.tensor([active]), torch.arange(8)
    copies = []
    for layer, name in enumerate(groups[0].members):
        def load(indices, lengths, destinations, *, layer=layer, name=name, max_aiv_cores=0):
            assert max_aiv_cores == 0
            copies.append(name)
            for p, tensor in enumerate(caches[name]):
                tensor.view(-1, tensor.shape[-1])[:active].fill_(layer + p + 1)
        impls[name]._full_graph_transfers[2].load = load
        for tensor in caches[name]:
            tensor.fill_(-7)
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    incoming, outgoing = owner.edges(Key())
    for layer, name in enumerate(groups[0].members):
        if name in incoming:
            incoming[name].join()
        else:
            impls[name]._full_graph_transfers[2].load(selected, counts, slots)
        for p, tensor in enumerate(caches[name]):
            flat = tensor.view(-1, tensor.shape[-1])
            assert torch.all(flat[:active] == layer + p + 1)
            assert torch.all(flat[active:] == -7)
        if name in outgoing:
            outgoing[name].launch(selected, counts, slots, outgoing[name].destinations)
    assert copies == list(groups[0].members)


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("shared_storage", [False, True])
def test_aot_resolves_late_destinations_inside_prefetch_operator(v2, shared_storage):
    import functools

    from torch._dynamo.backends.common import aot_autograd
    from torch._inductor import config
    from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
    from torch._inductor.decomposition import select_decomp_table

    data = torch.ones(2, 4)
    backing = torch.zeros(12)
    destinations = ([backing[:8].view(2, 4), backing[8:].view(2, 2)] if shared_storage
                    else [torch.zeros(2, 4), torch.zeros(2, 2)])
    seen = []
    edge = NS(transfer=NS(request_capacity=2), destinations=destinations)
    active = NS(value=False)
    def post(*args, prefetch):
        args[-1].copy_(args[1])
        if prefetch is None:
            return
        buffers = prefetch[-1]
        seen.append([t.data_ptr() for t in buffers])
        for buf in buffers:
            buf.add_(1)
        args[-1].copy_(args[1])
    impl = NS(_shared_retrieval_edges=lambda: ({}, {"producer": edge} if active.value else {}),
              cross_layer_graph_post=post)
    ns = {"torch": torch, "_mla_runtime_state": lambda name: (impl, "producer", (), NS())}
    op_fn = extract(ROOT / "vllm_ascend/ops/mla.py", "sfa_forward_post_prefetch", ns)
    fake = extract(ROOT / "vllm_ascend/ops/mla.py", "sfa_forward_post_prefetch_fake", {})
    namespace = f"prefetch_island_{int(v2)}_{int(shared_storage)}"
    lib = torch.library.Library(namespace, "DEF")
    lib.define("post" + torch.library.infer_schema(op_fn, mutates_args=["output"]))
    lib.impl("post", op_fn, "CPU")
    lib._register_fake("post", fake)
    op = getattr(torch.ops, namespace).post
    compiler = dict(functools=functools, aot_autograd=aot_autograd,
                    graph_returns_tuple=graph_returns_tuple, make_graph_return_tuple=make_graph_return_tuple,
                    select_decomp_table=select_decomp_table, COMPILATION_PASS_KEY="passes",
                    torch=NS(ops=NS(vllm=NS(sfa_forward_post_prefetch=op), higher_order=torch.ops.higher_order)))
    source = ROOT / "vllm_ascend/compilation/compiler_interface.py"
    for name in ("compile_fx", "_reuse_shared_resident_buffers", "fusion_pass_compile"):
        compiler[name] = extract(source, name, compiler)
    def backend(graph, inputs):
        return compiler["fusion_pass_compile"](graph, inputs, {"passes": lambda g: g}, None)[0]
    def forward(x):
        output = torch.empty_like(x)
        op(x, x, x, x, x, x, output, "producer")
        return output
    with config.patch(enable_auto_functionalized_v2=v2):
        compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
        torch.testing.assert_close(compiled(data), data)  # Initial model profiling: no edges.
        assert not seen
        active.value = True  # KV setup after compilation, before capture.
        for _ in range(2):
            torch.testing.assert_close(compiled(data), data)
    assert seen == [[t.data_ptr() for t in destinations]] * 2
    assert all(t.eq(2).all() for t in destinations)


def test_compiler_ignores_unrelated_list_arguments():
    fn = extract(ROOT / "vllm_ascend/compilation/compiler_interface.py", "_reuse_shared_resident_buffers",
                 {"torch": torch})
    graph = torch.fx.symbolic_trace(lambda x: torch.cat([x, x]))
    before = str(graph.graph)
    assert not fn(graph)
    assert str(graph.graph) == before


def test_disabled_mla_keeps_original_operator_sequence():
    events = []
    x = torch.ones(2, 4)
    bridge = (x, x, x, x, x, x)
    ops = NS(
        sfa_forward_pre=Mock(side_effect=lambda *args: events.append("pre") or bridge),
        sfa_forward_pre_shared=Mock(side_effect=AssertionError("unexpected shared operator")),
        sfa_lmcache_retrieve=Mock(side_effect=lambda *args: events.append("retrieve")),
        sfa_forward_post=Mock(side_effect=lambda *args: events.append("post")),
        sfa_forward_post_prefetch=Mock(side_effect=AssertionError("unexpected prefetch")),
    )
    source = ROOT / "vllm_ascend/ops/mla.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMultiHeadLatentAttention")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    module = ast.parse("from __future__ import annotations")
    module.body.append(fn)
    ns = {"torch": NS(empty=torch.empty, ops=NS(vllm=ops)), "_EXTRA_CTX": NS(flash_comm_v1_enabled=False)}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
    impl = NS(shared_resident_plan=None, local_num_heads=1, kv_lora_rank=4, qk_rope_head_dim=2,
              index_topk=2048, _staged_sfa_graph_capture_sizes=(4,), decode_threshold=2)
    layer = NS(use_cross_layer_sfa=True, use_retrieval_overlap=False, target_sfa_debug=False, mla_attn=NS(impl=impl),
               prefix="layer", next_layer_name="next")
    ns["forward"](layer, x, x)
    assert events == ["pre", "retrieve", "post"]


@pytest.mark.parametrize("full,shared", [(False, False), (False, True), (True, False), (True, True)])
def test_configuration_requires_both_prerequisites(full, shared):
    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    block = next(n for n in init.body if isinstance(n, ast.If)
                 and "VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP" in ast.unparse(n.test))
    # Execute validation only; allocation/import is covered by the owner tests.
    validate = block.body[0]
    ns = {"sfa_full_graph_enabled": lambda cfg: full, "vllm_config": None,
          "envs_ascend": NS(VLLM_ASCEND_SFA_SHARED_RESIDENT_PLAN=shared)}
    code = compile(ast.fix_missing_locations(ast.Module(body=[validate], type_ignores=[])), str(source), "exec")
    if full and shared:
        exec(code, ns)
    else:
        with pytest.raises(ValueError, match="requires full graphs"):
            exec(code, ns)


def test_profiling_cleanup_releases_old_cache_before_final_allocation(runtime):
    module, _ = runtime
    owner = module.SharedRetrievalOverlap()
    groups, impls, caches = topology()
    owner.configure(tuple(impls.items()), groups, caches)
    references = [weakref.ref(t) for planes in caches.values() for t in planes]

    class Parent:
        def _cleanup_profiling_kv_cache(self):
            for impl in impls.values():
                impl._staged_sfa_capture_state.runtime = None
            caches.clear()

    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_cleanup_profiling_kv_cache")
    cls = ast.ClassDef(name="Runner", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                       keywords=[], body=[method], decorator_list=[])
    ns = {"Parent": Parent, "envs_ascend": NS(VLLM_ASCEND_SFA_FULL_GRAPH=True)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), ns)
    runner = ns["Runner"]()
    runner._sfa_full_graph = NS(retrieval_overlap=owner)
    runner._reset_staged_sfa_startup_capture = owner.clear
    runner._collect_staged_sfa_impls = lambda: tuple(impls.items())
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        runner._cleanup_profiling_kv_cache()
        assert all(ref() is None for ref in references), "overlap retained discarded profiling KV"
    finally:
        if was_enabled:
            gc.enable()


def test_graph_reset_retains_real_layout_but_profile_release_requires_clear(runtime):
    module, _ = runtime
    owner = module.SharedRetrievalOverlap()
    groups, impls, caches = topology()
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    with pytest.raises(RuntimeError, match="clear overlap graphs"):
        owner.release_layout(tuple(impls.items()))
    destinations = owner.pairs["g0l0"][1]
    owner.clear()
    assert owner.pairs["g0l0"][1] is destinations
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    assert owner.edges(Key())[1]["g0l0"].destinations == tuple(destinations)
    owner.clear()
    owner.release_layout(tuple(impls.items()))
    assert not owner.pairs
    assert impls["g0l0"]._retrieval_overlap is None
    # Final cache allocation uses the same layer objects but different storage.
    fresh = {name: tuple(t.clone() for t in values) for name, values in caches.items()}
    owner.configure(tuple(impls.items()), groups, fresh)
    assert owner.pairs["g0l0"][1][0] is fresh["g0l1"][0]


def test_compile_before_kv_initialization_keeps_prefetch_dispatch():
    """Model profiling compiles without KV; later execution bypasses guards."""
    from torch.fx.experimental.proxy_tensor import make_fx

    def pre(*args):
        return (args[0],) * 6

    def serial_post(*args):
        args[6].copy_(args[0] + 1)

    def overlap_post(*args):
        args[6].copy_(args[0] + 2)

    ops = NS(sfa_forward_pre=pre, sfa_forward_pre_shared=pre,
             sfa_lmcache_retrieve=lambda *args: None,
             sfa_forward_post=serial_post, sfa_forward_post_prefetch=overlap_post)
    source = ROOT / "vllm_ascend/ops/mla.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMultiHeadLatentAttention")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    mod = ast.parse("from __future__ import annotations")
    mod.body.append(fn)
    ns = {"torch": NS(empty=torch.empty, ops=NS(vllm=ops)), "_EXTRA_CTX": NS(flash_comm_v1_enabled=False)}
    exec(compile(ast.fix_missing_locations(mod), str(source), "exec"), ns)
    impl = NS(shared_resident_plan=None, local_num_heads=1, kv_lora_rank=4, qk_rope_head_dim=2,
              index_topk=2048, _staged_sfa_graph_capture_sizes=(4,), decode_threshold=2)
    layer = NS(use_cross_layer_sfa=True, use_retrieval_overlap=True, target_sfa_debug=False,
               mla_attn=NS(impl=impl), prefix="layer", next_layer_name="next")
    x = torch.ones(2, 4)
    compiled = make_fx(lambda value: ns["forward"](layer, value, value))(x)
    impl._retrieval_overlap = NS()
    # Execute the already-compiled callable, as the no-guards runner does.
    torch.testing.assert_close(compiled(x), x + 2)


def test_native_projection_capability_controls_overlap(runtime, monkeypatch):
    module, log = runtime
    monkeypatch.delattr(module.torch_npu, "npu_transpose_batchmatmul")
    groups, impls, caches = topology()
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    assert owner.edges(Key()) == ({}, {})
    assert owner.stream is None and not log
