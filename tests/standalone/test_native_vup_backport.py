# SPDX-License-Identifier: Apache-2.0
"""Exercise native serving V-up with real CPU tensors."""

import ast
import sys
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def serving_projection(native_available=True, native_error=False):
    tree = ast.parse((ROOT / "vllm_ascend/attention/sfa_v1.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFAImpl")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_v_up_proj")
    limit = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "TRANSPOSE_BMM_MAX_SUPPORTED_DIM" for target in node.targets
        )
    )
    calls = []

    def native(x, weight, *, perm_x1, perm_y):
        calls.append((perm_x1, perm_y, x.is_contiguous()))
        if native_error:
            raise RuntimeError("native failure")
        return torch.bmm(x.permute(perm_x1), weight).permute(perm_y).contiguous()

    def bmm(x, weight):
        calls.append("bmm")
        return torch.bmm(x, weight)

    # Deliberately expose only the allowed math operations. Accessing the
    # retired custom op or a synchronization call fails the test immediately.
    namespace = {
        "nullcontext": nullcontext,
        "torch": NS(float16=torch.float16, bfloat16=torch.bfloat16, bmm=bmm, compiler=torch.compiler),
        "torch_npu": NS(npu_transpose_batchmatmul=native) if native_available else NS(),
        "logger": NS(info_once=Mock()),
    }
    exec(compile(ast.Module(body=[limit, method], type_ignores=[]), "<serving-vup>", "exec"), namespace)
    return namespace["_v_up_proj"], calls


@pytest.mark.parametrize("heads,rank,permutation", [(2, 32, (1, 0, 2)), (128, 512, (0, 1, 2))])
def test_only_native_vup_is_inside_copy_window(heads, rank, permutation):
    projection, calls = serving_projection()
    x = torch.ones((3, heads, rank), dtype=torch.bfloat16)
    impl = NS(local_num_heads=heads, kv_lora_rank=rank, v_head_dim=2,
              W_UV=torch.ones((heads, rank, 2), dtype=x.dtype))

    @contextmanager
    def window(name):
        calls.append(("open", name))
        yield
        calls.append(("close", name))

    result = projection(impl, x, copy_edge=NS(cube_window=window))
    assert calls == [("open", "vup"), (permutation, (1, 0, 2), True), ("close", "vup")]
    assert result.shape == (3, heads * 2) and torch.all(result == rank)


def test_strided_native_vup_stays_in_the_protected_path():
    projection, calls = serving_projection()
    x = torch.ones((3, 2, 8), dtype=torch.float16)[..., ::2]
    state = NS(local_num_heads=2, kv_lora_rank=4, v_head_dim=4,
               W_UV=torch.eye(4, dtype=x.dtype).repeat(2, 1, 1))
    edge = NS(cube_window=Mock(side_effect=AssertionError("strided input opened a Cube window")))
    torch.testing.assert_close(projection(state, x, copy_edge=edge), x.reshape(3, 8))
    assert calls == [((1, 0, 2), (1, 0, 2), False)]


@pytest.mark.parametrize(
    "heads,rank,expected_perm", [(64, 512, (1, 0, 2)), (128, 512, (0, 1, 2)), (129, 512, (0, 1, 2))]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_old_cann_dimension_boundary_and_unchanged_weight_layout(heads, rank, expected_perm, dtype):
    projection, calls = serving_projection()
    generator = torch.Generator().manual_seed(7)
    x = torch.randn((3, heads, rank), generator=generator).to(dtype)
    weight = torch.zeros((heads, rank, 8), dtype=dtype)
    columns = torch.arange(8) * 7
    signs = (1 - 2 * (torch.arange(heads) % 2)).to(dtype)
    weight[:, columns, torch.arange(8)] = signs[:, None]
    original = weight.clone()
    state = NS(local_num_heads=heads, kv_lora_rank=rank, v_head_dim=8, W_UV=weight)
    result = projection(state, x)
    expected = (x[:, :, columns] * signs[None, :, None]).reshape(3, -1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(weight, original, rtol=0, atol=0)
    assert calls == [(expected_perm, (1, 0, 2), True)]
    assert state.W_UV is weight


@pytest.mark.parametrize("native_available,dtype", [(False, torch.bfloat16), (True, torch.float32)])
def test_fallback_keeps_bmm_and_propagates_native_errors(native_available, dtype):
    projection, calls = serving_projection(native_available=native_available)
    x = torch.arange(24).reshape(3, 2, 4).to(dtype)
    state = NS(local_num_heads=2, kv_lora_rank=4, v_head_dim=4, W_UV=torch.eye(4, dtype=dtype).expand(2, -1, -1))
    torch.testing.assert_close(projection(state, x), x.reshape(3, 8))
    assert calls == ["bmm"]
    projection, calls = serving_projection(native_error=True)
    with pytest.raises(RuntimeError, match="native failure"):
        projection(state, x.bfloat16())
    assert len(calls) == 1














@pytest.mark.parametrize("tokens", [384, 1536])
@pytest.mark.parametrize("overlap_enabled", [False, True])
def test_actual_runner_startup_imports_sfa_constants_before_capture(monkeypatch, tokens, overlap_enabled):
    # Export the actual module's literal constants, without importing its NPU
    # dependencies. Do not inject a fake value for a deleted public constant.
    sfa_tree = ast.parse((ROOT / "vllm_ascend/attention/sfa_v1.py").read_text(encoding="utf-8"))
    constants = [
        node for node in sfa_tree.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
    ]
    sfa_module = ModuleType("vllm_ascend.attention.sfa_v1")
    exec(compile(ast.Module(body=constants, type_ignores=[]), "<actual-sfa-constants>", "exec"), sfa_module.__dict__)
    monkeypatch.setitem(sys.modules, sfa_module.__name__, sfa_module)

    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_model_forward"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    key = NS(request_capacity=2, token_capacity=tokens)
    context = NS(staged_sfa_graph_key=key, staged_sfa_graph_dummy_run=True, cudagraph_runtime_mode="FULL")
    namespace = {
        "get_forward_context": lambda: context,
        "envs_ascend": NS(VLLM_ASCEND_SFA_FULL_GRAPH=True),
        "CUDAGraphMode": NS(NONE="NONE"),
    }
    exec(compile(module, str(source), "exec"), namespace)

    overlap = NS(prepare=Mock()) if overlap_enabled else None
    graph = NS(
        sealed=False,
        retrieval_overlap=overlap,
        register_transfers=Mock(),
        bind_sources=Mock(),
        prepare_run=Mock(return_value="prepared"),
        run=Mock(return_value="captured"),
    )
    impl = NS(prepare_full_graph_layer=Mock(return_value=("metadata",)), _full_graph_transfer=object())

    def coordinate(error, dummy_run):
        assert dummy_run is True
        if error is not None:
            raise RuntimeError("startup preparation failed") from error

    runner = NS(
        model=object(),
        _sfa_full_graph=graph,
        _staged_sfa_impls=(("layer0", impl),),
        model_config=NS(max_model_len=8192),
        _coordinate_sfa_full_graph_preparation=Mock(side_effect=coordinate),
        _run_sfa_full_graph_target=object(),
    )
    assert namespace["_model_forward"](runner, tokens) == "captured"
    runner._coordinate_sfa_full_graph_preparation.assert_called_once_with(None, True)
    if overlap_enabled:
        overlap.prepare.assert_called_once_with(key, runner._staged_sfa_impls, (), 1024)
    graph.run.assert_called_once_with(runner._run_sfa_full_graph_target, prepared="prepared")


@pytest.mark.parametrize(
    "available,heads,rank,dtype,expected",
    [
        (True, 2, 4, torch.bfloat16, "branch=npu_transpose_batchmatmul input_transpose=internal"),
        (True, 128, 512, torch.bfloat16, "branch=npu_transpose_batchmatmul input_transpose=explicit"),
        (False, 2, 4, torch.bfloat16, "branch=torch.bmm reason=missing_native_api"),
        (True, 2, 4, torch.float32, "branch=torch.bmm reason=unsupported_dtype"),
    ],
)
def test_logs_the_executed_branch_once_across_layers_and_token_counts(available, heads, rank, dtype, expected):
    projection, _ = serving_projection(native_available=available)
    messages = []

    # Match vLLM's process-local info_once cache, including its message/args key.
    @lru_cache
    def info_once(message, *args, scope):
        assert scope == "process"
        messages.append(message % args if args else message)

    projection.__globals__["logger"] = NS(info_once=info_once)
    for _layer in range(2):
        weight = torch.eye(rank, 4, dtype=dtype).expand(heads, -1, -1).contiguous()
        state = NS(local_num_heads=heads, kv_lora_rank=rank, v_head_dim=4, W_UV=weight)
        for tokens in (1, 3):
            x = torch.ones((tokens, heads, rank), dtype=dtype)
            actual = projection(state, x)
            torch.testing.assert_close(actual, torch.ones((tokens, heads * 4), dtype=dtype))
    assert messages == ["[SFA_VUP] " + expected]


@pytest.mark.parametrize("available,heads,rank", [(True, 2, 4), (True, 128, 512), (False, 2, 4)])
def test_branch_logging_keeps_torch_compile_fullgraph(available, heads, rank):
    projection, _ = serving_projection(native_available=available)
    projection.__globals__["torch"] = torch

    def unexpected_log(*args, **kwargs):
        raise AssertionError("logging entered the compiled graph")

    projection.__globals__["logger"] = NS(info_once=unexpected_log)
    weight = torch.eye(rank, 4, dtype=torch.bfloat16).expand(heads, -1, -1).contiguous()
    state = NS(local_num_heads=heads, kv_lora_rank=rank, v_head_dim=4, W_UV=weight)
    compiled = torch.compile(lambda x: projection(state, x), backend="eager", fullgraph=True)
    for value in (1, 2):
        x = torch.full((3, heads, rank), value, dtype=torch.bfloat16)
        actual = compiled(x)
        torch.testing.assert_close(actual, torch.full((3, heads * 4), value, dtype=torch.bfloat16))
