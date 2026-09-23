# SPDX-License-Identifier: Apache-2.0
"""Packaging metadata fallback without importing the NPU build toolchain."""
import ast
from pathlib import Path
from unittest.mock import Mock

import pytest
from packaging.version import InvalidVersion


@pytest.mark.parametrize("result,error,expected", [
    ("0.13.0rc1", None, "0.13.0rc1"),
    (None, LookupError("no tags"), "0.0.0"),
    (None, InvalidVersion("96.8ms-baseline-20260910"), "0.0.0"),
])
def test_setup_accepts_non_release_checkouts(result, error, expected):
    path = Path(__file__).resolve().parents[2] / "setup.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    block = next(node for node in tree.body if isinstance(node, ast.Try)
                 and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                         and call.func.id == "get_version" for call in ast.walk(node)))
    get_version = Mock(return_value=result, side_effect=error)
    ns = {"get_version": get_version, "InvalidVersion": InvalidVersion}
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec"), ns)
    assert ns["VERSION"] == expected
    get_version.assert_called_once_with(write_to="vllm_ascend/_version.py")
