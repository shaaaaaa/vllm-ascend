# SPDX-License-Identifier: Apache-2.0
"""Execute the real DSA role-init block on CPU, without worker/NPU imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def initialize_roles(*, p_node=False, d_node=False, shared=True, shrink=2):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    start = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.dsa_shared_pool"
    )
    end = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.use_sparse_c8_indexer"
    )
    runner = SimpleNamespace(dsa_two_groups=True)
    env = SimpleNamespace(
        VLLM_ASCEND_DSA_SHARED_POOL=shared,
        VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=p_node,
        VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE=d_node,
        VLLM_ASCEND_DSA_SHRINK_LATENT=shrink,
    )
    module = ast.Module(body=init.body[start:end], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"), dict(self=runner, envs_ascend=env, logger=Mock())
    )
    return runner


def test_explicit_d_role_initializes_shrink_before_validation():
    assert initialize_roles(d_node=True, shrink=2).dsa_shrink_latent == 2


def test_d_role_without_shrink_raises_configuration_error_not_attribute_error():
    with pytest.raises(ValueError, match="SHRINK_LATENT=2"):
        initialize_roles(d_node=True, shrink=0)


@pytest.mark.parametrize("p_node", [False, True])
def test_non_d_role_keeps_zero_shrink_supported(p_node):
    assert initialize_roles(p_node=p_node, shrink=0).dsa_shrink_latent == 0


def test_d_role_still_requires_shared_pool():
    with pytest.raises(ValueError, match="shared pool"):
        initialize_roles(d_node=True, shared=False)


def test_roles_still_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        initialize_roles(p_node=True, d_node=True)
