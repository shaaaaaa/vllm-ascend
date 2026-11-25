# SPDX-License-Identifier: Apache-2.0
"""Check the executable paths left after retiring corruption diagnostics."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative_path",
    [
        "vllm_ascend/worker/sfa_async_mtp.py",
        "vllm_ascend/worker/sfa_decode_timing.py",
    ],
)
def test_retired_diagnostics_leave_no_empty_conditionals(relative_path):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            assert not all(isinstance(statement, ast.Pass) for statement in node.body)
        if isinstance(node, ast.BoolOp):
            assert not any(isinstance(value, ast.Constant) and value.value is None for value in node.values)


@pytest.mark.parametrize(
    "relative_path",
    [
        "vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py",
        "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py",
    ],
)
def test_retired_log_hook_is_removed(relative_path):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.FunctionDef) and node.name == "_cold_live_log" for node in ast.walk(tree))


def test_sfa_initialization_has_no_discarded_layer_count():
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            assert "get_num_layers" not in ast.unparse(node.value)
