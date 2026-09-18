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


def test_sfa_finishes_only_after_projection_collective(projection):
    fn, events, _ = projection
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    project = load_function(
        path, "_project_with_layerwise_prefill_transfer", dict(row_parallel_with_prefill_transfer=fn)
    )
    layer = NS(
        custom_op=None,
        input_is_parallel=True,
        tp_rank=0,
        tp_size=2,
        bias=None,
        skip_bias_add=False,
        return_bias=True,
        reduce_results=True,
        quant_method=NS(apply=lambda *args: events.append("gemm") or torch.ones(1)),
    )
    operations = [("latent", []), ("index", [])]

    def submit(ops):
        assert ops is operations
        events.extend(["save_latent", "save_index", "load_latent", "load_index"])
        return ["latent", "index"]

    def finish(ops, names):
        assert ops is operations and names == ["latent", "index"]
        events.append("finish")

    sfa = NS(o_proj=layer, _submit_sfa_layerwise_transfer_window=submit, _finish_sfa_layerwise_transfer_window=finish)
    assert torch.equal(project(sfa, torch.ones(1), operations), torch.tensor([2.0]))
    assert events == ["gemm", "save_latent", "save_index", "load_latent", "load_index", "all_reduce", "finish"]


def test_non_p_projection_keeps_ordinary_forward():
    tree = ast.parse((ROOT / "vllm_ascend/attention/sfa_v1.py").read_text(encoding="utf-8"))
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "use_layerwise_transfer_window"
        and "self._project_with_layerwise_prefill_transfer" in ast.unparse(n)
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
            self=sfa, use_layerwise_transfer_window=False, attn_output=torch.ones(1), output=output, save_operations=[]
        ),
    )
    assert events == ["ordinary_projection", "ordinary_save"]
    assert output.item() == 1


@pytest.mark.parametrize("fifo", [False, True])
@pytest.mark.parametrize("p_node", [False, True])
def test_fifo_submits_once_after_sfa_before_all_projection_work(fifo, p_node):
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forward = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any(isinstance(c, ast.Name) and c.id == "early_transfer_layer_names" for c in ast.walk(n))
    )
    start = next(
        i
        for i, n in enumerate(forward.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "early_transfer_layer_names"
    )
    end = next(
        i
        for i in range(start, len(forward.body))
        if isinstance(forward.body[i], ast.If) and ast.unparse(forward.body[i].test) == "use_layerwise_transfer_window"
    )
    events = ["sfa"]
    ops = [("latent", []), ("index", [])]

    def submit(actual):
        assert actual is ops
        events.extend(["store_latent", "store_index", "load_latent", "load_index"])
        return ["latent", "index"]

    def finish(actual, names):
        assert actual is ops and names == ["latent", "index"]
        events.append("finish")

    class Projection:
        weight = None

        def __call__(self, x):
            events.extend(["gemm", "all_reduce"])
            return x, None

    def legacy(x, actual):
        events.append("gemm")
        names = submit(actual)
        events.append("all_reduce")
        finish(actual, names)
        return x

    owner = NS(
        _prefill_fifo_transfers=fifo,
        _v_up_proj=lambda x: events.append("v_up") or x,
        o_proj=Projection(),
        enable_dsa_cp_with_o_proj_tp=False,
        enable_dsa_cp_strict_accuracy=False,
        _submit_sfa_layerwise_transfer_window=submit,
        _finish_sfa_layerwise_transfer_window=finish,
        _project_with_layerwise_prefill_transfer=legacy,
        _submit_sfa_save_operations=lambda actual: events.append("legacy_save"),
    )
    wrapper = ast.parse("def run_tail(attn_output): pass").body[0]
    wrapper.body = forward.body[start : end + 1]
    scope = dict(
        self=owner,
        use_layerwise_transfer_window=p_node,
        save_operations=ops,
        attn_output=torch.ones(1),
        output=torch.zeros(1),
        MAX_O_PROJ_PREFETCH_SIZE=1,
        get_weight_prefetch_method=lambda: NS(maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kw: None),
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), "exec"),
        scope,
    )
    scope["run_tail"](torch.ones(1))
    transfers = ["store_latent", "store_index", "load_latent", "load_index"]
    if not p_node:
        assert events == ["sfa", "v_up", "gemm", "all_reduce", "legacy_save"]
    elif fifo:
        assert events == ["sfa", *transfers, "v_up", "gemm", "all_reduce", "finish"]
    else:
        assert events == ["sfa", "v_up", "gemm", *transfers, "all_reduce", "finish"]
