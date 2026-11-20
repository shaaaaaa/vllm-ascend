# SPDX-License-Identifier: Apache-2.0
"""Post-pack prefetch fences and exchange lifetime, without an NPU runtime."""

import sys
import weakref
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from sfa_test_support import ROOT, extract, load_module
from test_sfa_retrieval_overlap import runtime as overlap_runtime

OPS = ROOT / "vllm_ascend/ops/triton/sfa_dcp.py"


@pytest.fixture
def runtime(monkeypatch):
    return overlap_runtime.__wrapped__(monkeypatch)


@pytest.mark.parametrize("failure", [None, "pack", "submit", "transfer", "wait", "combine"])
def test_merge_releases_copy_after_pack_before_exchange_wait(runtime, failure):
    module, log = runtime
    state = NS(submitted=False, joined=False)
    caches = (torch.zeros(2),)

    def transfer(*args, max_aiv_cores):
        assert max_aiv_cores == 12
        assert state.submitted and not state.joined
        log.append("transfer")
        if failure == "transfer":
            raise RuntimeError(failure)

    edge = module.RetrievalEdge(torch.npu.Stream(), NS(load=transfer), caches, max_aiv_cores=12)
    log.clear()  # Exclude the existing event initialization outside capture.

    class Pack:
        def __getitem__(self, grid):
            def run(partial, maximum, total, send, *args):
                log.append("pack")
                if failure == "pack":
                    raise RuntimeError(failure)
                send.fill_(7)

            return run

    def collective(recv, send, *, group, async_op):
        assert group == "dcp" and async_op and log == ["pack"]
        log.append("submit")
        if failure == "submit":
            raise RuntimeError(failure)
        state.submitted = True
        source, dest = weakref.ref(send), weakref.ref(recv)

        def wait():
            assert source() is not None and dest() is not None, "DMA operands released before join"
            assert "transfer" in log and not state.joined
            log.append("wait")
            if failure == "wait":
                raise RuntimeError(failure)
            dest().copy_(source())
            state.joined = True

        return NS(wait=wait)

    class Combine:
        def __getitem__(self, grid):
            def run(recv, out, *args):
                assert state.joined and edge.pending_capture_launch
                assert recv.eq(7).all()
                log.append("combine")
                if failure == "combine":
                    raise RuntimeError(failure)
                out.fill_(7)

            return run

    merge = extract(
        OPS,
        "merge_sfa_dcp",
        {
            "torch": torch,
            "dist": NS(get_world_size=lambda _: 8, all_to_all_single=collective),
            "init_device_properties_triton": lambda: None,
            "get_vectorcore_num": lambda: 32,
            "_pack": Pack(),
            "_combine": Combine(),
        },
    )
    partial = torch.zeros(2, 64, 512, dtype=torch.bfloat16)
    extract(OPS, "_exchange_sfa_dcp_packet", merge.__globals__)
    stats = torch.ones(2, 64)
    payload = (
        edge,
        torch.zeros(1, 4096, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, 4096, dtype=torch.int64),
        caches,
    )
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            merge(partial, stats, stats, "dcp", prefetch=payload)
    else:
        assert merge(partial, stats, stats, "dcp", prefetch=payload).eq(7).all()
        assert log == [
            "pack",
            "submit",
            ("record", "main"),
            ("wait", "transfer"),
            "transfer",
            ("record", "transfer"),
            "wait",
            "combine",
        ]
        assert edge.pending_capture_launch  # Only the NEXT layer joins H2D.
        edge.join()
        assert log[-1] == ("wait", "main") and not edge.pending_capture_launch
    assert log.count("wait") == int(failure not in ("pack", "submit"))
    assert ("combine" in log) == (failure in (None, "combine"))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("has_edge", [False, True])
def test_execute_routes_one_prefetch_without_changing_no_edge_path(monkeypatch, enabled, has_edge):
    log = []
    value = torch.ones(2, 8, 512, dtype=torch.bfloat16)
    edge = NS(launch=lambda *args: log.append("prefetch"))
    payload = (edge, None, None, None, None) if has_edge else None

    def merge(*args, **kwargs):
        assert kwargs == ({"prefetch": payload} if has_edge and enabled else {})
        log.append("merge")
        return value

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_sparse_flash_attention_dcp",
        lambda *args: log.append("sfa") or (value, value, value),
        raising=False,
    )
    run = extract(
        OPS,
        "execute_sfa_dcp",
        {
            "torch": torch,
            "_gather_sfa_dcp_queries": lambda q, r, g: (q, r),
            "merge_sfa_dcp": merge,
        },
    )
    metadata = torch.zeros(1, dtype=torch.int32)
    assert (
        run(
            value,
            value,
            (None, None),
            metadata,
            metadata,
            metadata,
            metadata,
            metadata,
            1.0,
            NS(device_group="dcp"),
            prefetch=payload,
            prefetch_after_pack=enabled,
        )
        is value
    )
    assert log == (["sfa", "prefetch", "merge"] if has_edge and not enabled else ["sfa", "merge"])


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("has_edge", [False, True])
@pytest.mark.parametrize("dcp", [False, True])
def test_serving_only_opts_in_existing_dcp_edges(monkeypatch, enabled, has_edge, dcp):
    log = []
    value = torch.ones(2, 8, 512)
    edge = NS(launch=lambda *args: log.append("ordinary_prefetch"))
    payload = (edge, None, None, None, None) if has_edge else None
    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")

    def execute(*args, **kwargs):
        assert kwargs.get("prefetch_after_pack", False) == (enabled and has_edge)
        assert kwargs["prefetch"] is payload
        log.append("dcp")
        return value

    module.execute_sfa_dcp = execute
    monkeypatch.setitem(sys.modules, module.__name__, module)
    post = extract(
        ROOT / "vllm_ascend/attention/sfa_v1.py",
        "_cross_layer_post_compute",
        {
            "get_dcp_group": lambda: None,
            "MAX_O_PROJ_PREFETCH_SIZE": 0,
            "get_weight_prefetch_method": lambda: NS(
                maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kw: None
            ),
        },
    )
    impl = NS(
        _dcp_prefetch_after_pack=enabled,
        scale=1.0,
        dsa_cp_decoder=False,
        _v_up_proj=lambda x: x,
        o_proj=NS(),
        _execute_sparse_flash_attention_process=lambda *a, **kw: log.append("ordinary") or value,
    )

    class Projection:
        weight = value

        def __call__(self, x):
            return (x,)

    impl.o_proj = Projection()
    post(
        impl,
        value,
        value,
        value,
        value,
        value,
        value,
        value,
        value,
        torch.empty_like(value),
        trace_label="test",
        prefetch=payload,
        dcp=NS(counts=value, local_seq_lens=value) if dcp else None,
    )
    assert log == (["dcp"] if dcp else ["ordinary"] + (["ordinary_prefetch"] if has_edge else []))


@pytest.mark.parametrize("query_first", [False, True])
def test_benchmark_only_changes_prefetch_point(monkeypatch, query_first):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/ops"))
    bench = load_module(ROOT / "benchmarks/ops/benchmark_sfa_dcp_prefetch.py", "prefetch_bench", monkeypatch)
    monkeypatch.setattr(bench, "QueryFirstPipeline", lambda *args, **kw: NS(**kw))
    args = NS(query_first=query_first)
    baseline = bench.prefetch_pipeline(args, None, None, None, False)
    candidate = bench.prefetch_pipeline(args, None, None, None, True)
    assert baseline.candidate == candidate.candidate == query_first
    assert not baseline.prefetch_after_pack and candidate.prefetch_after_pack


@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("layer", [0, 1, 2, 3])
def test_whole_group_benchmark_opts_in_only_outgoing_edges(monkeypatch, candidate, layer):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/ops"))
    bench = load_module(ROOT / "benchmarks/ops/benchmark_sfa_copy_reduction.py", "prefetch_scope_bench", monkeypatch)
    value = torch.ones(2, 8, 512)
    pipeline = NS(
        direct=True,
        args=NS(dcp_size=8),
        prefetch_after_pack=candidate,
        counts=value,
        attention_table=value,
        attention_ends=value,
        local_lens=value,
        group=None,
        attention_outputs=[],
        edges=[NS(destinations=()) for _ in range(3)],
    )

    def execute(*args, **kwargs):
        assert kwargs.get("prefetch_after_pack", False) == (candidate and layer < 3)
        assert (kwargs["prefetch"] is not None) == (layer < 3)
        return value

    bench.AttentionPipeline._attend_layer(
        pipeline,
        layer,
        (value, value),
        (),
        NS(load=lambda *args: None),
        (),
        (value,) * 6,
        None,
        execute,
    )
    assert pipeline.attention_outputs == [value]
