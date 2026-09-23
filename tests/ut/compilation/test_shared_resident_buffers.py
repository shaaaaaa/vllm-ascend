# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual Ascend AOT compiler on separate producer/consumer islands."""

import ast
import functools
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch._inductor import config
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
from torch._inductor.decomposition import select_decomp_table

from tests.ut.distributed.kv_transfer.test_shared_resident_plan import ROOT, grouped, methods, plan
from tests.ut.distributed.kv_transfer.test_shared_resident_plan import api as resident_api


@pytest.fixture
def planner_api():
    return resident_api.__wrapped__()


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("bounded", [False, True])
def test_ascend_aot_keeps_group_storage_without_clones(planner_api, mtp, v2, bounded):
    _, (producer, consumer), group, _ = grouped(planner_api, mtp, bounded)
    ns = dict(
        torch=torch,
        StagedSFABridge=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    )
    methods(
        "vllm_ascend/ops/mla.py",
        {
            "sfa_forward_pre_fake",
            "sfa_forward_pre_shared",
            "sfa_forward_pre_shared_fake",
        },
        ns,
    )
    observed_writes = []

    def pre(layer, hidden, cache, metadata, gather, output, *, resident_reads, resident_writes):
        current = producer if layer == "producer" else consumer
        if resident_writes:
            observed_writes.append(tuple(t.data_ptr() for t in resident_writes))
        raw = hidden[:, :1].to(torch.int32).expand(-1, 2048).unsqueeze(1)
        values = plan(current, raw, writes=resident_writes, reads=resident_reads)
        return (hidden[:, None, :].clone(), hidden[:, None, :2].clone(), *(t.clone() for t in values))

    producer.cross_layer_graph_pre = consumer.cross_layer_graph_pre = pre
    ns["_mla_runtime_state"] = lambda name: (producer if name == "producer" else consumer, name, (), None)
    namespace = f"resident_islands_{mtp}_{int(v2)}_{int(bounded)}"
    lib = torch.library.Library(namespace, "DEF")
    lib.define(
        "pre" + torch.library.infer_schema(ns["sfa_forward_pre_shared"], mutates_args=["output", "resident_writes"])
    )
    lib.impl("pre", ns["sfa_forward_pre_shared"], "CPU")
    lib._register_fake("pre", ns["sfa_forward_pre_shared_fake"])
    op = getattr(torch.ops, namespace).pre
    compiler = dict(
        functools=functools,
        aot_autograd=aot_autograd,
        graph_returns_tuple=graph_returns_tuple,
        make_graph_return_tuple=make_graph_return_tuple,
        select_decomp_table=select_decomp_table,
        COMPILATION_PASS_KEY="passes",
        torch=NS(ops=NS(vllm=NS(sfa_forward_pre_shared=op), higher_order=torch.ops.higher_order)),
    )
    methods(
        "vllm_ascend/compilation/compiler_interface.py",
        {
            "compile_fx",
            "fusion_pass_compile",
            "_reuse_shared_resident_buffers",
        },
        compiler,
    )

    def backend(graph, inputs):
        # The real fusion passes do not match these SFA-only islands.
        return compiler["fusion_pass_compile"](graph, inputs, {"passes": lambda g: g}, None)[0]

    def write(x, buffers):
        return op(x, False, torch.empty_like(x), "producer", 1, 4, 2, 2048, mtp, 1, mtp * 2048, [], buffers)

    def read(x, buffers):
        return op(x, False, torch.empty_like(x), "consumer", 1, 4, 2, 2048, mtp, 1, mtp * 2048, buffers, [])

    with config.patch(enable_auto_functionalized_v2=v2):
        write_compiled = torch.compile(write, backend=backend, fullgraph=True, dynamic=False)
        read_compiled = torch.compile(read, backend=backend, fullgraph=True, dynamic=False)
        for value in (1, 7):
            first = write_compiled(torch.full((mtp, 4), float(value)), group.writes)
            second = read_compiled(first[0][:, 0], group.reads)
            assert torch.equal(first[2], second[2]) and second[2].eq(value + 100).all()
            assert torch.equal(first[4], second[4])
    assert observed_writes == [tuple(t.data_ptr() for t in group.writes)] * 2


def test_regular_graph_is_not_rewritten():
    ns = dict(torch=torch)
    methods("vllm_ascend/compilation/compiler_interface.py", {"_reuse_shared_resident_buffers"}, ns)
    graph = torch.fx.symbolic_trace(lambda x: x.sin() + x)
    before = str(graph.graph)
    assert not ns["_reuse_shared_resident_buffers"](graph)
    assert str(graph.graph) == before


def test_tp4_dp4_piecewise_launcher_selects_ascend_fusion_backend():
    tree = ast.parse((ROOT / "vllm_ascend/platform.py").read_text(encoding="utf-8"))
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE"
    )
    compilation = NS(mode=3, splitting_ops=[], set_splitting_ops_for_v1=Mock())
    ascend = NS(ascend_compilation_config=NS(enable_npugraph_ex=True))
    ns = dict(
        logger=Mock(),
        CompilationMode=NS(VLLM_COMPILE=3),
        compilation_config=compilation,
        ascend_config=ascend,
        staged_sfa_graph_configured=lambda cfg: True,
        update_aclgraph_sizes=Mock(),
        envs_ascend=NS(VLLM_ASCEND_MTP_DRAFT_DEBUG=False),
        vllm_config=NS(parallel_config=NS(tensor_parallel_size=4, data_parallel_size=4, all2all_backend="naive")),
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=branch.body, type_ignores=[])), "piecewise_setup", "exec"), ns
    )
    assert not ascend.ascend_compilation_config.enable_npugraph_ex
    assert not compilation.use_inductor_graph_partition
    assert "vllm::sfa_lmcache_retrieve" in compilation.splitting_ops
