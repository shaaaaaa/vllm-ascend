# SPDX-License-Identifier: Apache-2.0
"""Execute the real DSA role-init block on CPU, without worker/NPU imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def initialize_roles(*, p_node=False, shared=True, shrink=2):
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
        VLLM_ASCEND_DSA_SHRINK_LATENT=shrink,
    )
    module = ast.Module(body=init.body[start:end], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"), dict(self=runner, envs_ascend=env, logger=Mock())
    )
    return runner


@pytest.mark.parametrize("p_node", [False, True])
def test_prefill_role_keeps_zero_shrink_supported(p_node):
    assert initialize_roles(p_node=p_node, shrink=0).dsa_shrink_latent == 0


def test_prefill_role_requires_shared_pool():
    with pytest.raises(ValueError, match="shared pool"):
        initialize_roles(p_node=True, shared=False)


def test_disabled_prefill_role_preserves_compact_decode():
    runner = initialize_roles(p_node=False, shrink=2)
    assert runner.dsa_shrink_latent == 2
    assert not runner.layerwise_prefill_p_node


def validate_platform_prefill_role(*, p_node, speculative_tokens):
    """Run the production P-role config guard without importing NPU modules."""
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/platform.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUPlatform")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "check_and_update_config")
    guard = next(
        n
        for n in method.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "layerwise_prefill_p_node"
    )
    speculative_config = (
        None if speculative_tokens is None else SimpleNamespace(num_speculative_tokens=speculative_tokens, method="mtp")
    )
    namespace = dict(
        layerwise_prefill_p_node=p_node,
        vllm_config=SimpleNamespace(speculative_config=speculative_config),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), str(path), "exec"), namespace)


@pytest.mark.parametrize("speculative_tokens", [None, 1])
def test_prefill_role_allows_no_mtp_or_one_draft_token(speculative_tokens):
    validate_platform_prefill_role(p_node=True, speculative_tokens=speculative_tokens)


@pytest.mark.parametrize("speculative_tokens", [2, 4])
def test_prefill_role_rejects_repeated_mtp_before_worker_start(speculative_tokens):
    with pytest.raises(ValueError, match="num_speculative_tokens <= 1"):
        validate_platform_prefill_role(p_node=True, speculative_tokens=speculative_tokens)


@pytest.mark.parametrize("speculative_tokens", [None, 1, 2, 4])
def test_disabled_prefill_role_does_not_restrict_mtp(speculative_tokens):
    validate_platform_prefill_role(p_node=False, speculative_tokens=speculative_tokens)
