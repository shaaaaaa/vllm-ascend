# SPDX-License-Identifier: Apache-2.0
"""Full-operator ownership, fallback and complete benchmark-chain checks."""

import sys
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import ROOT, extract, load_module
from test_sfa_dcp_prepare import ops


@pytest.mark.parametrize(
    "active,plan,copy_bridge",
    [
        (True, NS(active=True), False),
        (False, NS(active=True), True),
        (True, NS(active=False), True),
        (True, None, True),
    ],
)
@pytest.mark.parametrize("failure", [None, "pre", "retrieve"])
def test_full_operator_preserves_aliases_and_stage_order(active, plan, copy_bridge, failure):
    context = NS(sfa_full_graph_active=active)
    query, rope = torch.ones(2, 8, 512), torch.ones(2, 8, 64)
    topk, selected, slots = torch.zeros(2, 1, 2048), torch.zeros(1, 4096), torch.zeros(1, 4096)
    counts = torch.zeros(1, 16, dtype=torch.int32)[:, 0]
    writes = [topk, selected, slots] if plan is not None else []
    reads, events = [], []
    edge = NS(transfer=NS(request_capacity=1), destinations=object())

    def pre(*args, **kwargs):
        events.append("pre")
        assert kwargs["resident_writes"] is (writes or None) and kwargs["resident_reads"] is None
        assert kwargs["copy_bridge"] is copy_bridge
        if failure == "pre":
            raise RuntimeError("pre")
        return query, rope, topk, selected, counts, slots

    def retrieve(layer, following, tokens, misses, destinations, metadata, ctx):
        events.append("retrieve")
        assert tokens is selected and misses is counts and destinations is slots
        assert misses.stride() == (16,)
        if failure == "retrieve":
            raise RuntimeError("retrieve")

    def post(layer, q, r, indices, caches, metadata, output, *, prefetch):
        events.append("post")
        assert q is query and r is rope and indices is topk
        assert prefetch[0] is edge
        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(prefetch[1:4], (selected, counts, slots)))

    impl = NS(
        shared_resident_plan=plan,
        cross_layer_graph_pre=pre,
        cross_layer_lmcache_retrieve=retrieve,
        cross_layer_graph_post=post,
        _shared_retrieval_edges=lambda: ({}, {"layer": edge}),
    )
    run = extract(
        ROOT / "vllm_ascend/ops/mla.py",
        "sfa_forward_full",
        {
            "get_forward_context": lambda: context,
            "_mla_runtime_state": lambda _: (impl, "layer", (), None),
        },
    )
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run(query, False, query, "layer", "next", reads, writes)
    else:
        assert run(query, False, query, "layer", "next", reads, writes) is None
    assert events == (["pre"] if failure == "pre" else ["pre", "retrieve"] if failure else ["pre", "retrieve", "post"])


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/ops"))
    return load_module(ROOT / "benchmarks/ops/benchmark_sfa_copy_reduction.py", "copy_reduction_bench", monkeypatch)


@pytest.mark.parametrize("world", [1, 8])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("serial_cap", [0, 12])
def test_complete_benchmark_uses_real_layouts_and_preserves_overlap(
    benchmark, monkeypatch, world, direct, overlap, serial_cap
):
    _, copy_queries, calls = ops.__wrapped__()
    query_module = ModuleType("vllm_ascend.ops.triton.sfa_query")
    query_module.copy_sfa_query_bridge = copy_queries
    monkeypatch.setitem(sys.modules, query_module.__name__, query_module)
    p = benchmark.AttentionPipeline.__new__(benchmark.AttentionPipeline)
    p.args, p.direct, p.group = NS(dcp_size=world, serial_aiv_cores=serial_cap), direct, object()
    q = torch.ones(8, 2, 512, dtype=torch.bfloat16).transpose(0, 1)
    r = torch.ones(2, 8, 64, dtype=q.dtype)
    topk = torch.zeros(2, 1, 2048, dtype=torch.int32)
    selected, counts, slots = (
        torch.zeros(1, 4096, dtype=torch.int32),
        torch.zeros(1, 16, dtype=torch.int32)[:, 0],
        torch.zeros(1, 4096, dtype=torch.long),
    )
    original = q, r, topk, selected, counts, slots
    p.prepare = Mock(return_value=original)
    p.queries = [(q, r)] * 2
    p.layer_caches = [(torch.zeros(1), torch.zeros(1))] * 2
    p.bridges = [tuple(torch.empty(t.shape, dtype=t.dtype) for t in original) for _ in range(2)]
    p.copy_bridge = extract(
        ROOT / "vllm_ascend/attention/sfa_v1.py", "_copy_to_staged_sfa_bridge", {"torch": torch, "HAS_TRITON": True}
    )
    events = []

    caps = []

    def transfer(tokens, misses, destinations, *, max_aiv_cores=0):
        caps.append(max_aiv_cores)
        events.append("transfer")
        assert all(torch.equal(a, b) for a, b in zip((tokens, misses, destinations), original[3:]))
        assert (misses.stride() == (16,)) is direct

    p.transfers = [NS(load=transfer)] * 2

    def launch(*args):
        events.append("prefetch")
        transfer(*args[:3], max_aiv_cores=12)

    p.edges = [NS(join=lambda: events.append("join"), launch=launch, destinations=object())] if overlap else []
    p.counts = torch.zeros(2, 16, dtype=torch.int32)
    p.attention_table = p.attention_ends = p.local_lens = object()

    def attend(query, rope, cache, indices, selected_counts, *args, prefetch=None):
        events.append("attention")
        assert selected_counts is p.counts  # Not the request's H2D miss counts.
        assert torch.equal(query, q) and torch.equal(rope, r) and torch.equal(indices, topk)
        if prefetch is not None:
            prefetch[0].launch(*prefetch[1:])
        events.append("merge")
        return query

    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    module.execute_sfa_dcp = attend
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def single(query, key, value, indices, *args):
        events.append("attention")
        return query

    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_attention", single, raising=False)
    p.step(object())
    assert p.prepare.call_count == 1  # The consumer must reuse the producer plan.
    assert events.count("attention") == events.count("transfer") == 2
    assert caps == [serial_cap, 12 if overlap else serial_cap]
    if overlap:
        assert events.index("prefetch") < events.index("join")
        assert events.index("join") < len(events) - 1
    assert len(p.attention_outputs) == 2
    assert all(torch.equal(a, b) for a, b in zip(p.current_outputs, original))
    assert len(calls) == (0 if direct and world > 1 else 2)
