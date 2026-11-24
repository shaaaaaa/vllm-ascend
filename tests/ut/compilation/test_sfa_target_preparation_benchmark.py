# SPDX-License-Identifier: Apache-2.0
"""Benchmark input fairness, chain boundaries and graph lifetime on the host."""

import sys
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import ROOT, extract, load_module
from test_sfa_dcp_prepare import ops as preparation_ops


@pytest.fixture
def benchmark(monkeypatch):
    return load_module(ROOT / "benchmarks/ops/benchmark_sfa_target_preparation.py", "target_prep_bench", monkeypatch)


def test_transfer_check_detects_reused_host_layer(benchmark):
    p = benchmark.Pipeline.__new__(benchmark.Pipeline)
    p.args = NS(history=1025, dcp_size=8, rank=0)
    selected, slots = torch.tensor([[0, 1024]]), torch.tensor([[2, 1]])
    p.bridge = (None, None, None, selected, torch.tensor([2]), slots)
    p.caches = [torch.zeros(3, width, dtype=torch.bfloat16) for width in (512, 64)]
    for plane, cache in enumerate(p.caches):
        values = benchmark.source_values(selected[0], 0, layer=1) + 32 * plane
        cache[slots[0]] = values.to(cache.dtype)[:, None].expand(-1, cache.shape[-1])
    assert p.check_transfer(layer=1) == [2]
    with pytest.raises(AssertionError):
        p.check_transfer(layer=0)


@pytest.mark.parametrize("free_first", [False, True])
def test_benchmark_selects_serving_policy_before_replay(benchmark, monkeypatch, free_first):
    resident = "vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sorted_cache"
    load_module(ROOT / (resident.replace(".", "/") + ".py"), resident, monkeypatch)
    for module in ("vllm_ascend.attention.sfa_v1", "lmcache_ascend.v1.npu_connector.sparse_graph"):
        monkeypatch.setitem(sys.modules, module, Mock())
    monkeypatch.setenv("VLLM_ASCEND_DSA_DCP_FREE_SLOT_FIRST", str(int(free_first)))
    monkeypatch.setenv("VLLM_ASCEND_DSA_RESIDENT_QUERY_MEMBERSHIP", "0")
    configure = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "configure_dsa_resident_dcp_free_slots", configure, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels", Mock(), raising=False)
    args = NS(
        layout="uniform",
        requests=1,
        history=4096,
        live_tokens=2,
        dcp_size=8,
        rank=0,
        heads=1,
        resident=False,
        shared_plan=False,
    )
    inputs = benchmark.make_inputs(1, 1, args.history, args.live_tokens, 0.9, args.layout)
    for optimized in (False, True):
        benchmark.Pipeline(args, inputs, NS(chunk=1024, sources=[]), optimized)
    configure.assert_called_once_with(free_first)


@pytest.mark.parametrize("overlap", [0, 0.5, 0.9, 1])
@pytest.mark.parametrize("layout,padding", [("uniform", 0), ("bounded", 1), ("mixed", 2)])
def test_benchmark_sequences_have_exact_overlap_and_causal_ids(benchmark, overlap, layout, padding):
    frames, owners, ends = benchmark.make_inputs(2, 4, 4096, 128, overlap, layout, padding, seed=7)
    again = benchmark.make_inputs(2, 4, 4096, 128, overlap, layout, padding, seed=7)[0]
    for step, (raw, positions) in enumerate(frames):
        assert torch.equal(raw, again[step][0])
        assert raw.shape == (2 * (2 + padding), 1, 2048)
        assert raw[owners < 0].eq(-1).all()
        for row in range(int(ends[-1])):
            values = set(raw[row, 0].tolist())
            assert len(values) == 2048 and min(values) >= 0 and max(values) <= positions[row]
            if step:
                assert len(values & set(frames[step - 1][0][row, 0].tolist())) == round(2048 * overlap)


@pytest.mark.parametrize("layout", ["uniform", "bounded", "mixed"])
@pytest.mark.parametrize("shared", [False, True])
def test_benchmark_includes_planner_bridge_and_transfer(benchmark, monkeypatch, layout, shared):
    prepare, copy_queries, calls = preparation_ops.__wrapped__()
    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    module.prepare_sfa_dcp_topk = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    query_module = ModuleType("vllm_ascend.ops.triton.sfa_query")
    query_module.copy_sfa_query_bridge = copy_queries
    monkeypatch.setitem(sys.modules, query_module.__name__, query_module)
    packing = load_module(ROOT / "vllm_ascend/attention/sfa_graph_layout.py", "host_prep_layout", monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.sfa_graph_layout", packing)
    bridge = extract(
        ROOT / "vllm_ascend/attention/sfa_v1.py", "_copy_to_staged_sfa_bridge", {"torch": torch, "HAS_TRITON": True}
    )
    frames, rows, ends = benchmark.make_inputs(1, 1, 4096, 128, 0.9, layout)
    raw_before = frames[0][0].clone()
    outputs = []
    for optimized in (False, True):
        p = benchmark.Pipeline.__new__(benchmark.Pipeline)
        p.optimized, p.args, p.tokens = optimized, NS(layout=layout, shared_plan=shared, dcp_size=8, rank=0), 2
        p.rows, p.ends, p.boundaries = rows, ends, torch.tensor([4096, 4096], dtype=torch.int32)
        p.published, p.table = torch.empty_like(raw_before), torch.zeros((1, 64), dtype=torch.int32)
        p.counts = p.planner_counts = torch.zeros((2, 16), dtype=torch.int32)
        p.dcp = NS(topk=torch.empty_like(raw_before), layout=object())
        p.query, p.rope = torch.ones(2, 8, 512).transpose(0, 1).contiguous().transpose(0, 1), torch.ones(2, 8, 64)
        p.states, p.generations = torch.tensor([0]), torch.tensor([1])
        selected = torch.zeros(1, 4096, dtype=torch.int32)
        count_storage, slots = torch.zeros(1, 16, dtype=torch.int32), torch.zeros(1, 4096, dtype=torch.int64)
        p.workspace = NS(miss_tokens=selected, miss_counts=count_storage, target_slots=slots)
        p.union_mapping = selected.clone()
        p.bridge = tuple(
            torch.empty_like(t) for t in (p.query, p.rope, raw_before, selected, count_storage[:, 0], slots)
        )
        p.impl = NS(
            _ensure_staged_sfa_bridge_buffers=lambda _, p=p: p.bridge,
            shared_resident_plan=NS(attention_topk=torch.empty_like(raw_before)),
        )
        events = []

        def plan(
            impl,
            topk,
            boundary,
            requests,
            table,
            *args,
            p=p,
            events=events,
            selected=selected,
            count_storage=count_storage,
            slots=slots,
            **kwargs,
        ):
            events.append("resident_union_finalize_update")
            assert torch.equal(p.published, raw_before)
            if shared:
                topk = topk.clone()  # Shared serving planner owns its copy.
            assert topk.data_ptr() != frames[0][0].data_ptr()
            topk.add_(1)
            return topk, selected, count_storage[:, 0], slots

        def copy_bridge(*args, events=events):
            events.append("bridge")
            return bridge(*args)

        def transfer(*args, events=events, p=p):
            events.append("sfa_copy")
            assert all(a is b for a, b in zip(args, p.bridge[3:]))

        p.plan, p.copy_bridge, p.transfer = plan, copy_bridge, NS(load=transfer)
        p.step(frames[0])
        assert events == ["resident_union_finalize_update", "bridge", "sfa_copy"]
        assert torch.equal(frames[0][0], raw_before)
        outputs.append([t.clone() for t in p.bridge])
    benchmark.assert_snapshots(*outputs)
    assert [name for name, _ in calls].count("_mask_topk") == 1


def test_capture_failure_retains_graph_for_cleanup(benchmark, monkeypatch):
    events = []

    class Capture:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *args):
            pass

    graph = NS(reset=lambda: events.append("reset"))
    monkeypatch.setattr(
        torch,
        "npu",
        NS(synchronize=lambda: events.append("sync"), NPUGraph=lambda: graph, graph=lambda g: Capture()),
        raising=False,
    )
    pipeline = benchmark.Pipeline.__new__(benchmark.Pipeline)
    pipeline.graphs = []
    pipeline.step = Mock(side_effect=[None, RuntimeError("capture failed")])
    with pytest.raises(RuntimeError, match="capture failed"):
        pipeline.capture([object()])
    assert pipeline.graphs == [graph]
    pipeline.close()
    assert events == ["sync", "enter", "sync", "reset"] and not pipeline.graphs
