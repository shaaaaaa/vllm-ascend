# SPDX-License-Identifier: Apache-2.0
"""Deferred-query lifetime, ordering and complete benchmark scope on CPU."""

import gc
import sys
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from sfa_test_support import ROOT, definitions, extract, load_module
from test_sfa_dcp_forward import gather_queries as make_gather_queries

OPS = ROOT / "vllm_ascend/ops/triton/sfa_dcp.py"


@pytest.fixture
def gather_queries(monkeypatch):
    return make_gather_queries.__wrapped__(monkeypatch)


def async_ops(gather_queries, events):
    namespace = dict(gather_queries.__globals__)
    namespace.update(dataclass=dataclass)
    definitions(OPS, ["SfaDcpQueryGather"], namespace)

    def collective(output, packet, *, group, async_op):
        assert group == "dcp" and async_op
        events.append("submit")
        source, destination = weakref.ref(packet), weakref.ref(output)

        def wait():
            events.append("wait")
            # Only SfaDcpQueryGather holds the DMA operands at this point.
            assert source() is not None and destination() is not None
            destination().copy_(torch.cat((source(), source()), dim=0))

        return NS(wait=wait)

    namespace["dist"] = NS(all_gather_into_tensor=collective)
    for name in ("start_sfa_dcp_query_gather", "execute_sfa_dcp"):
        extract(OPS, name, namespace)
    return namespace


def test_early_gather_retains_dma_operands_and_waits_once(gather_queries):
    events = []
    namespace = async_ops(gather_queries, events)
    group = NS(device_group="dcp", world_size=2)
    query, rope = torch.ones(2, 8, 512), torch.ones(2, 8, 64)
    context = namespace["start_sfa_dcp_query_gather"](query, rope, group)
    del query, rope
    gc.collect()
    assert events == ["submit"]
    context.wait()
    context.wait()  # cleanup after successful attention must not add a wait
    assert events == ["submit", "wait"]
    assert torch.equal(context.gathered, context.packet.repeat(2, 1, 1))


@pytest.mark.parametrize("failure", [None, "gather", "unpack"])
def test_early_unpack_orders_streams_and_joins_once(gather_queries, monkeypatch, failure):
    events = []
    ns = async_ops(gather_queries, events)
    done = object()
    main = NS(wait_event=lambda event: events.append(event), wait_stream=lambda stream: events.append("drain"))
    side = NS(wait_stream=lambda stream: events.append("fork"), record_event=lambda: events.append("record") or done)
    current = main

    @contextmanager
    def use(stream):
        nonlocal current
        current = stream
        try:
            yield
        finally:
            current = main

    monkeypatch.setattr(torch, "npu", NS(current_stream=lambda: current, stream=use), raising=False)
    bits = torch.arange(2 * 3 * 576, dtype=torch.int16).reshape(2, 3, 576)
    bits[0, 0, :4] = torch.tensor([-32768, 32767, -1, 0], dtype=torch.int16)
    q, r = bits.view(torch.bfloat16).split((512, 64), dim=-1)
    ctx = ns["start_sfa_dcp_query_gather"](q, r, NS(device_group="dcp", world_size=2))
    wait = ctx.work.wait

    def wait_gather():
        if current is side and failure == "gather":
            events.append("gather_error")
            raise RuntimeError("gather failed")
        assert current is (main if failure == "gather" else side)
        wait()

    ctx.work.wait = wait_gather
    unpack = ns["_unpack_sfa_dcp_queries"]

    def launch(*args, **kwargs):
        events.append("unpack")
        assert kwargs["outputs"] is ctx.planes and kwargs["max_cores"] == 1
        assert current is side
        if failure == "unpack":
            raise RuntimeError("launch failed")
        return unpack(*args, **kwargs)

    ns["_unpack_sfa_dcp_queries"] = launch
    with pytest.raises(RuntimeError, match="failed") if failure else nullcontext():
        ctx.start_unpack(q.shape[-1], side, 1)
    ctx.wait()
    ctx.wait()
    with pytest.raises(RuntimeError, match="already joined"):
        ctx.start_unpack(q.shape[-1], side, 1)
    if failure == "gather":
        assert events == ["submit", "fork", "gather_error", "drain", "wait"]
        return
    assert events == ["submit", "fork", "wait", "unpack"] + (["drain"] if failure else ["record", done])
    if not failure:
        for plane, source in zip(ctx.planes, (q, r)):
            assert torch.equal(plane.view(torch.int16), source.view(torch.int16).repeat(1, 2, 1))


@pytest.mark.parametrize("early_unpack", [False, True])
def test_early_attention_uses_one_gather_and_keeps_prefetch_order(gather_queries, monkeypatch, early_unpack):
    events = []
    namespace = async_ops(gather_queries, events)
    group = NS(device_group="dcp", world_size=2)
    query, rope = torch.ones(2, 8, 512, dtype=torch.bfloat16), torch.ones(2, 8, 64, dtype=torch.bfloat16)
    context = namespace["start_sfa_dcp_query_gather"](query, rope, group)
    if early_unpack:
        stream = NS(wait_stream=lambda main: None, record_event=lambda: "unpacked")
        main = NS(wait_event=lambda event: events.append(event))
        monkeypatch.setattr(
            torch, "npu", NS(current_stream=lambda: main, stream=lambda s: nullcontext()), raising=False
        )
        context.start_unpack(512, stream, 1)
        namespace["_unpack_sfa_dcp_queries"] = lambda *args, **kwargs: pytest.fail("unpacked twice")
    events.append("independent_prepare")

    def native(q, r, *args):
        assert q.is_contiguous() and r.is_contiguous() and q.shape == (2, 16, 512)
        assert q.eq(1).all() and r.eq(1).all()
        events.append("attention")
        return q, r, None

    def merge(*args, **kwargs):
        events.append("merge")
        assert kwargs == {}
        return "output"

    namespace["merge_sfa_dcp"] = merge
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_attention_dcp", native, raising=False)
    metadata = torch.zeros(1, dtype=torch.int32)
    edge = NS(launch=lambda *args: events.append("prefetch"))
    assert (
        namespace["execute_sfa_dcp"](
            query,
            rope,
            (None, None),
            None,
            None,
            metadata,
            metadata,
            metadata,
            1.0,
            group,
            query_gather=context,
            prefetch=(edge, None, None, None, None),
        )
        == "output"
    )
    context.wait()
    order = ["wait", "independent_prepare", "unpacked"] if early_unpack else ["independent_prepare", "wait"]
    assert events == ["submit", *order, "attention", "prefetch", "merge"]


@pytest.mark.parametrize("failure", [None, "prepare", "retrieve", "post"])
@pytest.mark.parametrize("eligible", [False, True])
@pytest.mark.parametrize("consumer,prefetched", [(False, False), (True, False), (True, True)])
def test_full_operator_owns_early_gather_through_failure_cleanup(monkeypatch, failure, eligible, consumer, prefetched):
    events, pending = [], []
    overlap = eligible and not prefetched
    output = torch.zeros(2, 8, 512)
    counts = torch.ones(1, dtype=torch.int32)

    def start(query, rope, group):
        assert group == "dcp"
        events.append("gather")
        pending.append(NS(wait=lambda: events.append("join")))
        return pending[-1]

    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    module.start_sfa_dcp_query_gather = start
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def pre(*args, query_ready=None, **kwargs):
        events.append("project")
        assert (query_ready is not None) == overlap
        if query_ready:
            query_ready(output, output)
        events.append("prepare")
        if failure == "prepare":
            raise RuntimeError(failure)
        return output, output, output, output, counts, output

    def retrieve(*args):
        events.append("retrieve")
        if failure == "retrieve":
            raise RuntimeError(failure)

    def post(*args, **kwargs):
        events.append("post")
        assert kwargs.get("query_gather") is (pending[0] if overlap else None)
        if failure == "post":
            raise RuntimeError(failure)

    impl = NS(
        shared_resident_plan=NS(active=True),
        skip_topk=consumer,
        _dcp_early_gather=True,
        cross_layer_graph_pre=pre,
        cross_layer_lmcache_retrieve=retrieve,
        cross_layer_graph_post=post,
        _shared_retrieval_edges=lambda: ({"layer": object()} if prefetched else {}, {}),
    )
    run = extract(
        ROOT / "vllm_ascend/ops/mla.py",
        "sfa_forward_full",
        {
            "get_forward_context": lambda: NS(sfa_full_graph_active=eligible),
            "_mla_runtime_state": lambda _: (impl, "layer", (), NS(sfa_dcp=object())),
            "get_dcp_group": lambda: "dcp",
        },
    )
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run(output, False, output, "layer", "next", [], [])
    else:
        run(output, False, output, "layer", "next", [], [])
    assert events[:3] == ["project", "gather", "prepare"] if overlap else events[:2] == ["project", "prepare"]
    assert events[-1] == "join" if overlap else "join" not in events


@pytest.fixture
def tuning_bench(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/ops"))
    return load_module(ROOT / "benchmarks/ops/benchmark_sfa_dcp_tuning.py", "dcp_tuning_bench", monkeypatch)


def test_benchmark_only_changes_gather_timing(tuning_bench, monkeypatch):
    monkeypatch.setattr(tuning_bench, "AttentionPipeline", lambda *args, **kwargs: kwargs)
    baseline = tuning_bench.tuning_pipeline(NS(variant="early"), None, None, None, False)
    candidate = tuning_bench.tuning_pipeline(NS(variant="early"), None, None, None, True)
    assert baseline == {"direct": True, "early_gather": False}
    assert candidate == {"direct": True, "early_gather": True}


@pytest.mark.parametrize("fail_prepare", [False, True])
@pytest.mark.parametrize("prefetched", [False, True])
def test_whole_pipeline_overlaps_only_independent_work_and_joins(monkeypatch, tuning_bench, fail_prepare, prefetched):
    pipeline = tuning_bench.AttentionPipeline.__new__(tuning_bench.AttentionPipeline)
    events = []
    tensor = torch.zeros(2, 8, 512)
    pending = []
    namespace = {"dataclass": dataclass}
    definitions(OPS, ["SfaDcpQueryGather"], namespace)

    def start(*args):
        events.append("gather")
        context = namespace["SfaDcpQueryGather"](tensor, tensor, NS(wait=lambda: events.append("wait")), "group")
        pending.append(context)
        return context

    def prepare(frame):
        events.append("plan")
        if fail_prepare:
            raise RuntimeError("prepare failed")
        return (tensor,) * 6

    def attend(*args, query_gather=None, prefetch=None):
        if query_gather is not None:
            query_gather.wait()
        else:
            events.append("late_gather")
        events.append("attention")
        if prefetch is not None:
            prefetch[0].launch(*prefetch[1:])
        events.append("merge")
        return tensor

    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    module.start_sfa_dcp_query_gather, module.execute_sfa_dcp = start, attend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    pipeline.args, pipeline.group = NS(dcp_size=8), "group"
    pipeline.direct = pipeline.early_gather = True
    pipeline.queries = [(tensor, tensor)] * 2
    pipeline.layer_caches = [(tensor, tensor)] * 2
    pipeline.bridges = [()] * 2
    pipeline.transfers = [NS(load=lambda *args: events.append("load"))] * 2
    pipeline.edges = (
        [NS(join=lambda: events.append("join"), launch=lambda *args: events.append("prefetch"), destinations=())]
        if prefetched
        else []
    )
    pipeline.counts = pipeline.attention_table = pipeline.attention_ends = pipeline.local_lens = tensor
    pipeline.prepare = prepare
    if fail_prepare:
        with pytest.raises(RuntimeError, match="prepare failed"):
            pipeline.step(None)
        assert events == ["gather", "plan", "wait"]
    else:
        pipeline.step(None)
        if prefetched:
            assert events == [
                "gather",
                "plan",
                "load",
                "wait",
                "attention",
                "prefetch",
                "merge",
                "join",
                "late_gather",
                "attention",
                "merge",
            ]
        else:
            assert events == [
                "gather",
                "plan",
                "load",
                "wait",
                "attention",
                "merge",
                "gather",
                "load",
                "wait",
                "attention",
                "merge",
            ]
    assert all(context.work is None for context in pending)
