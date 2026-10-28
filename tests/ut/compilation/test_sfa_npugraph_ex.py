# SPDX-License-Identifier: Apache-2.0
"""Actual compiler functions with CPU AOT; TorchAir/device entry points are mocked."""

import ast
import copy
import gc
import hashlib
import sys
from contextlib import contextmanager
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import ROOT, definitions, load_module
from test_sfa_capture_budget import CUDAGraphMode, config
from test_sfa_capture_budget import sizing as _sizing
from torch._dynamo.backends.common import aot_autograd
from torch._inductor import config as inductor_config
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized


@pytest.fixture
def sizing(monkeypatch):
    yield from _sizing.__wrapped__(monkeypatch)


@pytest.fixture
def compiler(monkeypatch, sizing):
    state = NS(
        enabled=True,
        active_range=None,
        configs=[],
        calls=[],
        hook_available=True,
        keep_mutations=False,
        identity_copy_counts=[],
    )

    @contextmanager
    def pass_context(compile_range):
        previous = state.active_range
        state.active_range = compile_range
        try:
            yield
        finally:
            state.active_range = previous

    modules = {
        "vllm.compilation.compiler_interface": {"CompilerInterface": object},
        "vllm.config.utils": {"Range": object},
        "vllm.compilation.passes.inductor_pass": {
            "pass_context": pass_context,
            "get_pass_context": lambda: NS(compile_range=state.active_range),
        },
    }
    for name, attrs in modules.items():
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys.modules["vllm.config"], "VllmConfig", object, raising=False)
    ascend = sys.modules["vllm_ascend.ascend_config"]
    monkeypatch.setattr(ascend, "AscendCompilationConfig", object, raising=False)
    monkeypatch.setattr(
        ascend, "get_ascend_config", lambda: NS(ascend_compilation_config=NS(enable_npugraph_ex=state.enabled))
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", sizing)
    monkeypatch.setattr(torch, "npu", NS(set_compile_mode=Mock()), raising=False)

    def compiler_config():
        cfg = NS(debug=NS(aclgraph=NS()), experimental_config=NS(aclgraph=NS()))
        if state.hook_available:
            cfg.post_grad_custom_pre_pass = None
        state.configs.append(cfg)
        return cfg

    def get_backend(*, compiler_config):
        def backend(gm, inputs):
            state.calls.append("backend")

            def inner(graph, example_inputs):
                hook = getattr(compiler_config, "post_grad_custom_pre_pass", None)
                if hook is not None:
                    graph = hook(graph, example_inputs, compiler_config)
                state.identity_copy_counts.append(
                    sum(
                        node.target == torch.ops.aten.copy_.default and node.args[0] is node.args[1]
                        for node in graph.graph.nodes
                    )
                )
                state.calls.append("generic_lowering")
                decompose_auto_functionalized(graph.graph)
                graph.recompile()
                return graph

            return aot_autograd(fw_compiler=inner, keep_inference_input_mutations=state.keep_mutations)(gm, inputs)

        return backend

    torchair = ModuleType("torchair")
    torchair.CompilerConfig = compiler_config
    torchair.get_npu_backend = get_backend
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    module = load_module(ROOT / "vllm_ascend/compilation/compiler_interface.py", "tested_sfa_npugraph", monkeypatch)
    return module, state


def normalize(cfg, sizing):
    """Execute the real request read, defaults expansion and mode decision."""
    tree = ast.parse((ROOT / "vllm_ascend/platform.py").read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "check_and_update_config")
    ns = dict(
        sizing.__dict__,
        vllm_config=cfg,
        logger=Mock(),
        CompilationMode=NS(NONE=0, VLLM_COMPILE=3),
        CUDAGraphMode=CUDAGraphMode,
    )
    read_request = [
        n
        for n in method.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "sfa_npugraph_ex_requested" for t in n.targets)
    ]
    if read_request:
        exec(compile(ast.fix_missing_locations(ast.Module(read_request, [])), "request_read", "exec"), ns)
    definitions(ROOT / "vllm_ascend/ascend_config.py", {"AscendCompilationConfig"}, ns)
    ascend = ns["AscendCompilationConfig"](**cfg.additional_config.get("ascend_compilation_config", {}))
    ns.update(
        ascend_compilation_config=ascend,
        ascend_config=NS(ascend_compilation_config=ascend),
        compilation_config=cfg.compilation_config,
    )
    defaults = next(
        n for n in method.body if isinstance(n, ast.If) and ast.unparse(n.test) == "ascend_compilation_config"
    )
    decision = next(
        n
        for n in method.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "compilation_config.cudagraph_mode == CUDAGraphMode.NONE"
    )
    exec(compile(ast.fix_missing_locations(ast.Module([defaults, decision], [])), "mode_decision", "exec"), ns)
    if read_request:
        assert read_request[0].lineno < defaults.lineno < decision.lineno
    return ascend


def platform_config(requested):
    cfg = config()
    cfg.compilation_config.mode = 3
    cfg.compilation_config.splitting_ops = []
    cfg.compilation_config.set_splitting_ops_for_v1 = Mock()
    cfg.parallel_config.all2all_backend = "naive"
    if requested is not None:
        cfg.additional_config = {"ascend_compilation_config": {"enable_npugraph_ex": requested}}
    return cfg


@pytest.mark.parametrize("requested", [None, False, True])
def test_full_sfa_opt_in_survives_worker_normalization(sizing, requested):
    cfg = platform_config(requested)
    for _ in range(2):
        assert normalize(cfg, sizing).enable_npugraph_ex is (requested is True)
        assert cfg.additional_config["ascend_compilation_config"]["enable_npugraph_ex"] is (requested is True)
        assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
        assert "vllm::sfa_lmcache_retrieve" in cfg.compilation_config.splitting_ops
        cfg = copy.deepcopy(cfg)  # The next process receives the normalized config.


@pytest.mark.parametrize("mode", ["staged", "eager", "unrelated", "disabled"])
def test_no_new_backend_on_other_paths(sizing, monkeypatch, mode):
    cfg = platform_config(True)
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "1,2,4,8")
    if mode == "staged":
        monkeypatch.setenv("VLLM_ASCEND_SFA_FULL_GRAPH", "0")
    elif mode == "eager":
        cfg.model_config.enforce_eager = True
    elif mode == "unrelated":
        cfg.model_config.use_mla = False
    else:
        monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH", "0")
    assert not normalize(cfg, sizing).enable_npugraph_ex


@pytest.mark.parametrize("requested", [None, False, True])
def test_native_full_defaults_unchanged(sizing, requested):
    cfg = platform_config(requested)
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    assert normalize(cfg, sizing).enable_npugraph_ex is (requested is not False)


def test_hash_separates_only_new_route(compiler, monkeypatch):
    module, state = compiler
    cfg = config()
    cfg.compute_hash = lambda: "unchanged-base-hash"
    instance = module.AscendCompiler()
    state.enabled = False
    assert instance.compute_hash(cfg) == cfg.compute_hash()
    state.enabled = True
    expected = hashlib.sha256(f"{cfg.compute_hash()}:{module._SFA_NPUGRAPH_EX_REVISION}".encode()).hexdigest()
    assert instance.compute_hash(cfg) == instance.compute_hash(cfg) == expected
    monkeypatch.setenv("VLLM_ASCEND_SFA_FULL_GRAPH", "0")
    assert instance.compute_hash(cfg) == cfg.compute_hash()


def test_missing_hook_fails_before_backend(compiler):
    module, state = compiler
    state.hook_available = False
    graph = torch.fx.symbolic_trace(lambda x: (x + 1,))
    with pytest.raises(RuntimeError, match="post_grad_custom_pre_pass"):
        module.npugraph_ex_compile(graph, [torch.ones(2)], {}, config(), NS(enable_static_kernel=False), None)
    assert state.calls == []


def test_native_backend_does_not_require_sfa_hook(compiler):
    module, state = compiler
    state.hook_available = False
    cfg = config()
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    graph = torch.fx.symbolic_trace(lambda x: x + 1)
    fn, _ = module.npugraph_ex_compile(graph, [torch.ones(2)], {}, cfg, NS(enable_static_kernel=False), None)
    torch.testing.assert_close(fn(torch.ones(2)), torch.full((2,), 2.0))


def test_disabled_backend_does_not_construct_torchair(compiler, monkeypatch):
    module, state = compiler
    state.enabled = False
    factory = Mock(side_effect=AssertionError("disabled backend must stay direct"))
    monkeypatch.setattr(sys.modules["torchair"], "CompilerConfig", factory)
    graph = torch.fx.symbolic_trace(lambda x: (x + 1,))
    fn, _ = module.AscendCompiler().compile(graph, [torch.ones(2)], {module.COMPILATION_PASS_KEY: lambda g: g}, None)
    torch.testing.assert_close(fn(torch.ones(2))[0], torch.full((2,), 2.0))
    factory.assert_not_called()


def test_pass_failure_restores_context_without_fallback(compiler):
    module, state = compiler

    def fail(gm):
        assert state.active_range == "requested-range"
        raise RuntimeError("injected pass failure")

    graph = torch.fx.symbolic_trace(lambda x: (x + 1,))
    with pytest.raises(RuntimeError, match="injected pass failure"):
        module.npugraph_ex_compile(
            graph,
            [torch.ones(2)],
            {module.COMPILATION_PASS_KEY: fail},
            config(),
            NS(enable_static_kernel=False),
            "requested-range",
        )
    assert state.active_range is None
    assert state.calls == ["backend"]


def test_existing_pass_manager_keeps_range_filtering(compiler, monkeypatch):
    module, state = compiler
    stub = ModuleType("vllm.compilation.passes.vllm_inductor_pass")
    stub.VllmInductorPass = object
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    manager_module = load_module(
        ROOT / "vllm_ascend/compilation/graph_fusion_pass_manager.py", "tested_sfa_pass_manager", monkeypatch
    )
    manager = manager_module.GraphFusionPassManager()

    class Pass:
        def __init__(self, selected):
            self.selected = selected

        def is_applicable_for_range(self, interval):
            assert interval == "selected-range"
            return self.selected

        def __call__(self, gm):
            state.calls.append("eligible_pass")

    manager.add(Pass(False))
    manager.add(Pass(True))
    graph = torch.fx.symbolic_trace(lambda x: (x + 1,))
    fn, _ = module.npugraph_ex_compile(
        graph,
        [torch.ones(2)],
        {module.COMPILATION_PASS_KEY: manager},
        config(),
        NS(enable_static_kernel=False),
        "selected-range",
    )
    torch.testing.assert_close(fn(torch.ones(2))[0], torch.full((2,), 2.0))
    assert state.calls == ["backend", "eligible_pass", "generic_lowering"]


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("alias_output", [False, True])
def test_real_aot_pipeline_preserves_sources_and_unrelated_mutation(compiler, monkeypatch, v2, strided, alias_output):
    module, state = compiler
    lib = torch.library.Library(f"sfa_ex_{int(v2)}_{int(strided)}_{int(alias_output)}", "DEF")
    lib.define("pre(Tensor x, Tensor(a!) output, Tensor(b!)[] resident_writes) -> Tensor")
    lib.define("other(Tensor(a!) x) -> ()")
    seen = []

    def pre(x, output, resident_writes):
        seen.append((output.data_ptr(), resident_writes[0].data_ptr()))
        output.copy_(x * 2)
        resident_writes[0].add_(x)
        return x.clone()

    def other(x):
        x.add_(3)

    lib.impl("pre", pre, "CPU")
    lib._register_fake("pre", lambda x, output, resident_writes: torch.empty_like(x))
    lib.impl("other", other, "CPU")
    lib._register_fake("other", lambda x: None)
    ops = getattr(torch.ops, f"sfa_ex_{int(v2)}_{int(strided)}_{int(alias_output)}")
    monkeypatch.setattr(torch.ops.vllm, "sfa_forward_pre_shared", ops.pre, raising=False)
    compile_range = object()

    def passes(gm):
        assert state.active_range is compile_range
        state.calls.append("configured_passes")
        return gm

    def backend(gm, inputs):
        return module.npugraph_ex_compile(
            gm, inputs, {module.COMPILATION_PASS_KEY: passes}, config(), NS(enable_static_kernel=False), compile_range
        )[0]

    def forward(x, output, resident, unrelated):
        y = ops.pre(x, output, [resident])
        ops.other(unrelated)
        return y + unrelated

    backing = torch.zeros(12)
    resident = backing[4::2] if strided else backing[4:8]
    output = backing[:4] if alias_output else torch.zeros(4)
    unrelated, x = torch.zeros(4), torch.ones(4)
    with inductor_config.patch(enable_auto_functionalized_v2=v2):
        fn = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
        for i in (1, 2):
            torch.testing.assert_close(fn(x, output, resident, unrelated), x + 3 * i)
            torch.testing.assert_close(resident, x * i)
            torch.testing.assert_close(output, x * 2)
    assert all(out != output.data_ptr() for out, ptr in seen)
    if not alias_output:
        assert all(ptr == resident.data_ptr() for out, ptr in seen)
    assert state.calls == ["backend", "configured_passes", "generic_lowering"]
    assert state.active_range is None
    assert state.configs[0].debug.run_eagerly is True
    assert state.configs[0].mode == "reduce-overhead"


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("bounded", [False, True])
@torch.inference_mode()
def test_existing_sfa_planner_contract_through_new_backend(compiler, monkeypatch, v2, mtp, bounded):
    from tests.ut.compilation import test_shared_resident_buffers as existing

    module, state = compiler
    # TorchAir's default keeps mutation epilogues inside the inference graph.
    state.keep_mutations = True
    original = existing.methods

    def load_methods(path, names, ns):
        original(path, names, ns)
        if "fusion_pass_compile" in names:
            monkeypatch.setattr(
                torch.ops.vllm, "sfa_forward_pre_shared", ns["torch"].ops.vllm.sfa_forward_pre_shared, raising=False
            )
            ns["fusion_pass_compile"] = lambda gm, inputs, passes, interval: module.npugraph_ex_compile(
                gm,
                inputs,
                {module.COMPILATION_PASS_KEY: lambda g: g},
                config(),
                NS(enable_static_kernel=False),
                interval,
            )

    monkeypatch.setattr(existing, "methods", load_methods)
    existing.test_ascend_aot_keeps_group_storage_without_clones(
        existing.resident_api.__wrapped__(), mtp, v2, bounded, namespace_suffix="_npugraph"
    )
    assert state.identity_copy_counts and not any(state.identity_copy_counts)


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("shared_storage", [False, True])
def test_late_prefetch_binding_through_new_backend(compiler, monkeypatch, v2, shared_storage):
    from tests.ut.compilation import test_sfa_retrieval_overlap as existing

    module, _ = compiler
    original = existing.extract

    def load(path, name, ns):
        value = original(path, name, ns)
        if name == "fusion_pass_compile":
            return lambda gm, inputs, passes, interval: module.npugraph_ex_compile(
                gm,
                inputs,
                {module.COMPILATION_PASS_KEY: lambda g: g},
                config(),
                NS(enable_static_kernel=False),
                interval,
            )
        return value

    monkeypatch.setattr(existing, "extract", load)
    existing.test_aot_resolves_late_destinations_inside_prefetch_operator(
        v2, shared_storage, namespace_suffix="_npugraph"
    )


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("keep_mutations", [False, True])
def test_inference_writers_and_readers_with_gc_disabled(compiler, monkeypatch, v2, keep_mutations):
    module, state = compiler
    state.keep_mutations = keep_mutations
    name = f"sfa_ex_inference_{int(v2)}_{int(keep_mutations)}"
    lib = torch.library.Library(name, "DEF")
    lib.define("pre(Tensor(a!) output, Tensor[] resident_reads, Tensor(b!)[] resident_writes) -> ()")
    addresses = []

    def pre(output, resident_reads, resident_writes):
        addresses.append(tuple(t.data_ptr() for t in resident_writes))
        for tensor in resident_writes:
            tensor.add_(1)
        planes = resident_reads or resident_writes
        output.copy_(planes[0] + planes[1])

    lib.impl("pre", pre, "CPU")
    lib._register_fake("pre", lambda output, resident_reads, resident_writes: None)
    op = getattr(torch.ops, name).pre
    monkeypatch.setattr(torch.ops.vllm, "sfa_forward_pre_shared", op, raising=False)

    def forward(output, first, second):
        # Match MLA.forward: a producer has writes only; consumers have reads only.
        op(output, [], [first, second])
        op(output, [first, second], [])
        op(output, [first, second], [])
        return output

    def backend(gm, inputs):
        return module.npugraph_ex_compile(
            gm, inputs, {module.COMPILATION_PASS_KEY: lambda g: g}, config(), NS(enable_static_kernel=False), None
        )[0]

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.inference_mode(), inductor_config.patch(enable_auto_functionalized_v2=v2):
            backing, output = torch.zeros(12), torch.zeros(3)
            first, second = backing[::4], backing[2::4]
            fn = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
            for step in (1, 2, 3):
                torch.testing.assert_close(fn(output, first, second), torch.full((3,), float(step * 2)))
                torch.testing.assert_close(backing[1::2], torch.zeros(6))
            expected = (first.data_ptr(), second.data_ptr())
            assert [ptrs for ptrs in addresses if ptrs] == [expected] * 3
            assert state.calls == ["backend", "generic_lowering"]
    finally:
        if was_enabled:
            gc.enable()
