# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the P-only projection boundary and upstream arithmetic."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    unit = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def projection():
    events = []
    ns = dict(
        torch=torch,
        split_tensor_along_last_dim=lambda x, num_partitions: x.chunk(num_partitions, dim=-1),
        tensor_model_parallel_all_reduce=lambda x: events.append("all_reduce") or x * 2,
    )
    fn = load_function(ROOT / "vllm_ascend/ops/linear_op.py", "row_parallel_with_prefill_transfer", ns)
    return fn, events, ns


@pytest.mark.parametrize("rank,tp", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "skip_bias,return_bias,reduce", [(False, True, True), (True, True, True), (True, False, False)]
)
def test_projection_matches_upstream_and_submits_copy_before_collective(
    projection, rank, tp, parallel, skip_bias, return_bias, reduce
):
    fn, events, ns = projection
    path = ROOT.parent / "vllm/vllm/model_executor/layers/linear.py"
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "RowParallelLinear"
    )
    oracle = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), oracle],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        ns,
    )
    weight = torch.arange(12, dtype=torch.float32).view(3, 4)

    def gemm(layer, x, bias):
        events.append("gemm")
        result = x @ weight.T
        return result if bias is None else result + bias

    layer = NS(
        input_is_parallel=parallel,
        tp_rank=rank,
        tp_size=tp,
        bias=torch.ones(3),
        skip_bias_add=skip_bias,
        return_bias=return_bias,
        reduce_results=reduce,
        quant_method=NS(apply=gemm),
        custom_op=None,
    )
    x = torch.arange(8 if parallel else 8 * tp, dtype=torch.float32).view(2, -1)
    expected = ns["forward"](layer, x)
    events.clear()
    actual = fn(layer, x, lambda: events.append("copy_submit"))
    assert events == ["gemm", "copy_submit"] + (["all_reduce"] if reduce and tp > 1 else [])
    if return_bias:
        assert torch.equal(actual[0], expected[0])
        assert actual[1] is expected[1]
    else:
        assert torch.equal(actual, expected)


def test_custom_projection_is_not_reimplemented(projection):
    fn, events, _ = projection

    class Fused:
        custom_op = object()

        def __call__(self, x):
            events.append("fused_gemm_collective")
            return x + 2, None

    output = fn(Fused(), torch.ones(1), lambda: events.append("copy_submit"))
    assert events == ["copy_submit", "fused_gemm_collective"]
    assert torch.equal(output[0], torch.tensor([3.0]))


def test_sfa_queues_transfer_after_attention_and_before_projection():
    tree = ast.parse((ROOT / "vllm_ascend/attention/sfa_v1.py").read_text(encoding="utf-8"))
    forward = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "forward"
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_submit_sfa_layerwise_transfer_window"
            for call in ast.walk(n)
        )
    )
    calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]

    def lines(name):
        return [call.lineno for call in calls if call.func.attr == name]

    assert (
        max(lines("_execute_sparse_flash_attention_process"))
        < min(lines("_submit_sfa_layerwise_transfer_window"))
        < min(lines("_v_up_proj"))
        < min(lines("o_proj"))
        < min(lines("_finish_sfa_layerwise_transfer_window"))
    )


def test_non_p_projection_keeps_ordinary_forward():
    tree = ast.parse((ROOT / "vllm_ascend/attention/sfa_v1.py").read_text(encoding="utf-8"))
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "use_layerwise_transfer_window"
        and "self._finish_sfa_layerwise_transfer_window" in ast.unparse(n)
    )
    events = []
    sfa = NS(
        o_proj=lambda x: (events.append("ordinary_projection") or x, None),
        _submit_sfa_save_operations=lambda ops: events.append("ordinary_save"),
    )
    output = torch.zeros(1)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])), "<projection branch>", "exec"),
        dict(
            self=sfa, use_layerwise_transfer_window=False, attn_output=torch.ones(1), output=output,
            save_operations=[], layerwise_submitted_names=[]
        ),
    )
    assert events == ["ordinary_projection", "ordinary_save"]
    assert output.item() == 1
