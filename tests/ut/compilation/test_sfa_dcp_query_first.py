# SPDX-License-Identifier: Apache-2.0
"""Q-first dependency, cache ownership and failure cleanup tests on CPU."""

import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from sfa_test_support import ROOT, extract, load_module
from test_sfa_event_handoff import handoff  # noqa: F401


@pytest.mark.parametrize("query_first", [False, True])
def test_real_precompute_moves_query_and_rope_before_cache_write(query_first):
    events = []
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    pre = extract(
        path,
        "_cross_layer_pre_compute",
        {
            "torch": torch,
            "get_weight_prefetch_method": lambda: NS(
                maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kw: None
            ),
            "get_forward_context": lambda: NS(staged_sfa_graph_key=NS(request_capacity=1, query_profile="uniform")),
            "StagedSFAQueryProfile": NS(DECODE_BOUNDED="bounded"),
        },
    )
    q = torch.arange(2 * 8 * 512).view(2, 8, 512).float()
    r = q[..., :64]
    plan = (torch.zeros(2, 1, 2048), torch.zeros(1, 4096), torch.zeros(1), torch.zeros(1, 4096))

    class Projection:
        weight = torch.zeros(1)

        def __call__(self, x):
            events.append("qkv")
            return (torch.zeros(2, 577),)

    def kv(*args):
        events.append("kv_write")
        if query_first:
            assert events[-2] == "h2d_join"
        return r, q

    impl = NS(
        fused_qkv_a_proj=Projection(),
        q_lora_rank=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        q_a_layernorm=lambda x: events.append("q_norm") or x,
        dsa_cp_decoder=False,
        has_indexer=False,
        skip_topk=True,
        shared_resident_plan=NS(active=True, reads=plan),
        exec_kv=kv,
        _q_proj_and_k_up_proj=lambda x: events.append("q_projection") or (q, r),
        rope_single=lambda *args: events.append("q_rope") or r,
    )

    def ready(*args):
        events.append("gather")
        if query_first:
            events.append("h2d_join")

    args = [torch.zeros(2, 1), None, None, None, None, None, torch.arange(2), None] + [None] * 15
    result = pre(
        impl,
        *args,
        dcp=NS(row_requests=torch.zeros(2), split_boundary=torch.zeros(2)),
        query_ready=ready,
        query_first=query_first,
    )
    assert all(torch.equal(a, b) for a, b in zip(result, (q, r, *plan)))
    assert events == (
        ["qkv", "q_norm", "q_projection", "q_rope", "gather", "h2d_join", "kv_write"]
        if query_first
        else ["qkv", "q_norm", "kv_write", "q_projection", "q_rope", "gather"]
    )


@pytest.mark.parametrize(
    "failure,prefetched",
    [(failure, prefetched) for failure in (None, "before_query", "after_query", "post") for prefetched in (False, True)]
    + [("join", True), ("no pending", True)],
)
@pytest.mark.parametrize("early_unpack", [False, True])
def test_full_operator_defers_prefetch_and_cleans_up(monkeypatch, failure, prefetched, early_unpack):
    events = []
    edge = NS(pending_capture_launch=prefetched and failure != "no pending")

    def join():
        assert edge.pending_capture_launch
        events.append("h2d_join")
        if failure == "join":
            raise RuntimeError("join")
        edge.pending_capture_launch = False

    edge.join = join
    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    module.start_sfa_dcp_query_gather = lambda *args: events.append("gather") or NS(
        start_unpack=lambda *args: events.append("unpack"), wait=lambda: events.append("gather_join")
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    tensor = torch.zeros(1)

    def pre(*args, query_first=False, query_ready=None, **kwargs):
        assert query_first and query_ready is not None
        if failure == "before_query":
            raise RuntimeError(failure)
        events.append("q_projection_rope")
        query_ready(tensor, tensor)
        assert not edge.pending_capture_launch
        if failure == "after_query":
            raise RuntimeError(failure)
        events.append("kv_write")
        return (tensor,) * 6

    def post(*args, **kwargs):
        events.append("attention")
        if failure == "post":
            raise RuntimeError(failure)

    impl = NS(
        shared_resident_plan=NS(active=True),
        _dcp_early_gather=True,
        _dcp_query_first=True,
        _dcp_early_unpack=(object(), 12) if early_unpack else None,
        skip_topk=True,
        has_indexer=False,
        dsa_cp_decoder=False,
        _shared_retrieval_edges=lambda: ({"layer": edge} if prefetched else {}, {}),
        cross_layer_graph_pre=pre,
        cross_layer_lmcache_retrieve=lambda *args: None,
        cross_layer_graph_post=post,
    )
    run = extract(
        ROOT / "vllm_ascend/ops/mla.py",
        "sfa_forward_full",
        {
            "get_forward_context": lambda: NS(sfa_full_graph_active=True, staged_sfa_graph_key=object()),
            "_mla_runtime_state": lambda _: (impl, "layer", (), NS(sfa_dcp=object())),
            "get_dcp_group": lambda: "dcp",
        },
    )
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run(tensor, False, tensor, "layer", "", [], [])
    else:
        run(tensor, False, tensor, "layer", "", [], [])
    if failure == "no pending":
        assert events == []
        return
    if failure == "join":
        assert events[-1] == "gather_join"  # a broken prefetch wait cannot leak the query DMA operands
        assert "unpack" not in events
        return
    assert not edge.pending_capture_launch
    assert events.count("h2d_join") == int(prefetched)
    assert events.count("gather_join") == int(failure != "before_query")
    assert events.count("unpack") == int(early_unpack and failure != "before_query")
    if "unpack" in events:
        assert events.index("h2d_join" if prefetched else "gather") < events.index("unpack")
        if "kv_write" in events:
            assert events.index("unpack") < events.index("kv_write") < events.index("attention")
    if prefetched and failure != "before_query":
        assert events.index("gather") < events.index("h2d_join")


@pytest.mark.parametrize(
    "disabled", ["knob", "early", "dcp", "full", "shared", "producer", "indexer", "cp", "key"]
)
def test_query_first_remains_scoped_to_qualified_consumers(monkeypatch, disabled):
    # Fail before any tensor execution; inspect the production dispatch decision.
    seen = []
    impl = NS(
        shared_resident_plan=NS(active=disabled != "shared"),
        _dcp_early_gather=disabled != "early",
        _dcp_query_first=disabled != "knob",
        skip_topk=disabled != "producer",
        has_indexer=disabled == "indexer",
        dsa_cp_decoder=disabled == "cp",
        _shared_retrieval_edges=lambda: ({}, {}),
    )

    def pre(*args, **kwargs):
        seen.append(kwargs.get("query_first", False))
        raise RuntimeError("stop")

    impl.cross_layer_graph_pre = pre
    run = extract(
        ROOT / "vllm_ascend/ops/mla.py",
        "sfa_forward_full",
        {
            "get_forward_context": lambda: NS(
                sfa_full_graph_active=disabled != "full", staged_sfa_graph_key=None if disabled == "key" else object()
            ),
            "_mla_runtime_state": lambda _: (impl, "layer", (), NS(sfa_dcp=None if disabled == "dcp" else object())),
        },
    )
    with pytest.raises(RuntimeError, match="stop"):
        run(None, False, None, "layer", "", [], [])
    assert seen == [False]


@pytest.mark.parametrize("initialize", [False, True])
def test_graph_pre_preserves_warmup_cache_zero_dependency(request, initialize):
    h, events = request.getfixturevalue("handoff"), []
    edge = NS(pending_capture_launch=True)

    def join():
        assert edge.pending_capture_launch
        edge.pending_capture_launch = False
        events.append("join")

    edge.join = join

    class Cache:
        def __getitem__(self, key):
            return self

        def zero_(self):
            assert not edge.pending_capture_launch
            events.append("zero")

    h.impl._retrieval_overlap = object()
    h.impl._shared_retrieval_edges = lambda: ({"layer": edge}, {})
    h.context.staged_sfa_graph_dummy_run = initialize
    h.state.initialized_cache_capacity = 0 if initialize else 2
    h.impl.has_indexer = False
    h.impl._cross_layer_kv_cache.return_value = ((Cache(), Cache()), None, False)

    def pre(*args, query_ready=None, query_first=False, **kwargs):
        assert query_first
        events.append("q")
        query_ready(None, None)
        assert not edge.pending_capture_launch
        events.append("kv")
        return h.bridge

    h.impl._cross_layer_pre_compute = pre

    def ready(*args):
        events.append("gather")
        if edge.pending_capture_launch:
            edge.join()

    h.namespace["cross_layer_graph_pre"](
        h.impl,
        "layer",
        h.bridge[0],
        (),
        h.metadata,
        False,
        h.bridge[0],
        query_ready=ready,
        query_first=True,
    )
    assert events == (["join", "zero", "zero", "q", "gather", "kv"] if initialize else ["q", "gather", "join", "kv"])


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/ops"))
    return load_module(ROOT / "benchmarks/ops/benchmark_sfa_dcp_query_first.py", "query_first_bench", monkeypatch)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("origin", [0, 4096])
def test_benchmark_current_slots_have_exactly_one_owner(bench, world, origin):
    positions = torch.tensor([0, 127, 128, 255, 256, 511, 1023, 1024, 16512, 16513, 0])
    rows = torch.tensor([0] * 5 + [1] * 5 + [-1])
    table = torch.arange(400).reshape(2, 200).int()
    owned = []
    for rank in range(world):
        slots = bench.current_slots(positions, rows, table, origin, world, rank)
        for token, row, slot in zip(positions, rows, slots):
            if row >= 0 and token // 128 % world == rank:
                local = origin + token // (128 * world) * 128 + token % 128
                assert slot == table[row, local // 128] * 128 + local % 128
            else:
                assert slot == -1
        owned.append(slots >= 0)
    assert torch.equal(torch.stack(owned).sum(0), (rows >= 0).long())


@pytest.mark.parametrize(
    "candidate,placement", [(False, "off"), (True, "off"), (True, "producer"), (True, "consumer"), (True, "both")]
)
@pytest.mark.parametrize("prefetched", [False, True])
def test_benchmark_measures_projections_current_kv_and_completed_attention(
    bench, monkeypatch, candidate, placement, prefetched
):
    events = []
    pipeline = bench.QueryFirstPipeline.__new__(bench.QueryFirstPipeline)
    tensor = torch.zeros(2, 8, 512)
    frame = (torch.zeros(2, 1, 2048), None)
    pipeline.frame_inputs = {frame[0].data_ptr(): (torch.ones(2, 1), None, None, torch.zeros(2))}
    pipeline.query_first = candidate
    pipeline.args = NS(q_rank=1, dcp_size=8, unpack_cores=12)
    pipeline.unpack_placement, pipeline.unpack_stream = placement, object()
    pipeline.tokens, pipeline.group, pipeline.direct = 2, object(), True
    pipeline.norm_weight = None
    pipeline.rms_norm = lambda *args, **kwargs: events.append("q_norm") or (args[0],)
    pipeline.layer_caches = [()] * 2
    pipeline.bridges = [()] * 2
    pipeline.transfers = [NS(load=lambda *args: events.append("h2d"))] * 2
    pipeline.counts = pipeline.attention_table = pipeline.attention_ends = pipeline.local_lens = None
    pipeline.projectors = [
        NS(
            qkv=lambda x: events.append("qkv") or (torch.zeros(2, 577),),
            _q_proj_and_k_up_proj=lambda x: events.append("q") or (tensor, tensor),
            rope_single=lambda *args: events.append("rope") or tensor,
            exec_kv=lambda *args: events.append("kv") or (tensor, tensor),
            W_UV=torch.zeros(8, 512, 1),
            o_proj=lambda x: events.append("o") or (torch.zeros(2, 1),),
        )
    ] * 2
    edge = NS(pending_capture_launch=False, destinations=())

    def join():
        assert edge.pending_capture_launch
        events.append("h2d_join")
        edge.pending_capture_launch = False

    def launch(*args):
        assert not edge.pending_capture_launch
        events.append("prefetch")
        edge.pending_capture_launch = True

    edge.join, edge.launch = join, launch
    pipeline.edges = [edge] if prefetched else []
    pipeline.prepare = lambda _: events.append("plan") or (tensor,) * 6
    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    module.start_sfa_dcp_query_gather = lambda *args: events.append("gather") or NS(
        start_unpack=lambda *args: events.append("unpack"), wait=lambda: None
    )

    def execute(*args, query_gather=None, prefetch=None, **kwargs):
        events.append("gather_join" if query_gather else "late_gather")
        events.append("sfa")
        if prefetch:
            prefetch[0].launch(*prefetch[1:])
        events.append("merge")
        return tensor

    module.execute_sfa_dcp = execute
    monkeypatch.setitem(sys.modules, module.__name__, module)
    pipeline.step(frame)
    producer = ["unpack"] if placement in ("producer", "both") else []
    consumer = ["unpack"] if placement in ("consumer", "both") else []
    assert events[: 7 + len(producer)] == ["qkv", "q_norm", "kv", "q", "rope", "gather", *producer, "plan"]
    tail = events[events.index("o") + 1 :]
    if candidate:
        assert tail == ["qkv", "q_norm", "q", "rope", "gather"] + (
            ["h2d_join", *consumer, "kv"] if prefetched else [*consumer, "kv", "h2d"]
        ) + ["gather_join", "sfa", "merge", "o"]
    else:
        assert tail == (["h2d_join"] if prefetched else []) + ["qkv", "q_norm", "kv", "q", "rope"] + (
            ["late_gather"] if prefetched else ["gather", "h2d", "gather_join"]
        ) + ["sfa", "merge", "o"]
    assert not edge.pending_capture_launch
    assert torch.equal(pipeline.projected_output, torch.ones(2, 1))


@pytest.mark.parametrize("layout,padding", [("uniform", 0), ("mixed", 1)])
def test_unpack_benchmark_feeds_real_indexer_results_to_planner(bench, monkeypatch, layout, padding):
    from benchmark_sfa_dcp_early_unpack import UnpackPipeline
    from benchmark_sfa_dcp_query_first import QueryFirstPipeline

    def make():
        p = UnpackPipeline.__new__(UnpackPipeline)
        p.args = NS(indexer="c8", requests=2, history=4096, live_tokens=128, seed=4)
        p.frames, p.rows, p.ends = bench.make_inputs(2, 3, 4096, 128, 0.9, layout, padding)
        p.device, p.tokens = p.rows.device, len(p.rows)
        p._setup_indexer()
        return p

    pipeline, other = make(), make()
    fake_topk = torch.empty_like(pipeline.frames[0][0])
    monkeypatch.setattr(QueryFirstPipeline, "prepare", lambda self, frame: frame)
    for (raw, pos), (raw2, _) in zip(pipeline.frames, other.frames):
        query, lengths = pipeline.index_frames[raw.data_ptr()]
        assert torch.equal(query, other.index_frames[raw2.data_ptr()][0])
        ends = pipeline.index_inputs["actual_seq_lengths_query"]
        assert int(ends[-1]) == pipeline.tokens
        assert lengths[:2].tolist() == [int(pos[pipeline.rows == lane].max()) + 1 for lane in (0, 1)]
        assert len(lengths) == len(ends) == pipeline.index_inputs["block_table"].shape[0]
        assert lengths[2:].eq(0).all()
        assert lengths.max() <= pipeline.index_inputs["block_table"].shape[1] * 128

        def indexer(expected_q=query, expected_lens=lengths, **kwargs):
            assert kwargs["query"] is expected_q and kwargs["actual_seq_lengths_key"] is expected_lens
            assert kwargs["sparse_mode"] == 3 and kwargs["key"].dtype == expected_q.dtype == torch.int8
            return fake_topk

        monkeypatch.setattr(torch.ops._C_ascend, "npu_lightning_indexer_quant", indexer, raising=False)
        actual, positions = pipeline.prepare((raw, pos))
        assert actual is fake_topk and positions is pos


@pytest.mark.parametrize("corrupt", [False, True])
def test_benchmark_detects_missing_current_kv_writes(bench, monkeypatch, corrupt):
    pipeline = bench.QueryFirstPipeline.__new__(bench.QueryFirstPipeline)
    monkeypatch.setattr(bench.AttentionPipeline, "check_transfer", lambda *args, **kwargs: [3])
    pipeline.tokens = 2
    caches = [torch.full((2, 128, 1, width), -7, dtype=torch.bfloat16) for width in (512, 64)]
    values = [torch.full((2, 1, 1, width), value, dtype=torch.bfloat16) for width, value in ((512, 3), (64, 5))]
    slots = torch.tensor([-1, 129])
    for cache, value in zip(caches, values):
        cache.view(256, -1)[129].copy_(value[1].flatten())
    pipeline.layer_caches = [caches]
    pipeline.current_writes = [(slots, (values[1], values[0]))]
    if corrupt:
        caches[0].fill_(-7)
        with pytest.raises(AssertionError):
            pipeline.check_transfer()
    else:
        assert pipeline.check_transfer() == [3]


def test_indexer_benchmark_replicated_padding_preserves_null_page(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/ops"))
    from benchmark_sfa_dcp_indexer_sharding import IndexerPipeline

    def scatter(cache, slots, values):
        cache.index_copy_(0, slots.flatten(), values)

    monkeypatch.setitem(sys.modules, "torch_npu", NS(npu_scatter_nd_update_=scatter))
    mask = extract(ROOT / "vllm_ascend/attention/sfa_v1.py", "_mask_staged_index_scatter_padding", {"torch": torch})
    rows = torch.tensor([0, -1])
    key_cache, scale_cache = torch.zeros(2, 4), torch.ones(2, 1)
    keys, scales = torch.tensor([[1.0] * 4, [9.0] * 4]), torch.tensor([[3.0], [7.0]])
    pipeline = NS(
        rows=rows,
        tokens=2,
        group=None,
        owner_rows=None,
        attention_ends=None,
        idx_table=None,
        idx_cache=(None, None, key_cache, scale_cache),
        idx_frames={rows.data_ptr(): (rows, None, torch.tensor([1, 0]))},
        prepare=lambda result: result,
        indexer=NS(
            _mask_staged_index_scatter_padding=mask,
            indexer_select_pre_process=lambda *a: (keys, scales),
            indexer_select_post_process=lambda *a, **kw: rows.view(2, 1, 1),
        ),
    )
    IndexerPipeline.prepare_producer(pipeline, (rows, rows), None, None, None, None)
    assert torch.equal(key_cache[1], keys[0]) and torch.equal(scale_cache[1], scales[0])
    assert key_cache[0].eq(0).all() and scale_cache[0].eq(1).all()
