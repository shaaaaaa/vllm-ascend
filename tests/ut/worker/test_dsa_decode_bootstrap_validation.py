# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise the real D bootstrap validator without worker/NPU imports."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def bootstrap_validation(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    method = next(n for n in cls.body if getattr(n, "name", None) == "_validate_dsa_sparse_decode_d_node")
    # Execute the production method, postponing accelerator type annotations.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    has_connector = Mock(return_value=False)
    get_connector = Mock(side_effect=AssertionError("profiling must not access the connector"))
    staged_graph = Mock(return_value=True)
    namespace = dict(
        os=os,
        has_kv_transfer_group=has_connector,
        get_kv_transfer_group=get_connector,
        staged_sfa_graph_configured=staged_graph,
    )
    exec(compile(module, str(path), "exec"), namespace)
    monkeypatch.setenv("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE", "4096")
    runner = SimpleNamespace(
        dsa_sparse_decode_d_node=True,
        _profiling_cudagraph_memory=True,
        vllm_config=SimpleNamespace(),
    )
    config = SimpleNamespace(kv_cache_groups=[object(), object()])
    return SimpleNamespace(
        validate=lambda: namespace[method.name](runner, config),
        runner=runner,
        config=config,
        has_connector=has_connector,
        get_connector=get_connector,
        staged_graph=staged_graph,
    )


def test_profiling_defers_connector_checks_but_final_initialization_enforces_them(bootstrap_validation):
    case = bootstrap_validation
    # Graph profiling precedes Worker.initialize_from_config and has no connector.
    case.validate()
    case.has_connector.assert_not_called()
    case.get_connector.assert_not_called()

    # The same runner must still reject a missing connector for the final cache.
    case.runner._profiling_cudagraph_memory = False
    with pytest.raises(ValueError, match="requires an active LMCache connector"):
        case.validate()
    case.get_connector.assert_not_called()

    # Once initialized, its actual compact-load capability must be checked.
    case.has_connector.return_value = True
    case.get_connector.side_effect = None
    connector = SimpleNamespace(supports_dsa_compact_external_load=False)
    case.get_connector.return_value = connector
    with pytest.raises(ValueError, match="supports DSA cold compact external load"):
        case.validate()
    connector.supports_dsa_compact_external_load = True
    case.validate()


@pytest.mark.parametrize("profiling", [False, True])
@pytest.mark.parametrize("groups", [1, 3])
def test_invalid_group_count_rejected_in_both_phases(bootstrap_validation, profiling, groups):
    case = bootstrap_validation
    case.runner._profiling_cudagraph_memory = profiling
    case.config.kv_cache_groups = [object()] * groups
    with pytest.raises(ValueError, match="exactly two registered KV cache groups"):
        case.validate()


@pytest.mark.parametrize("profiling", [False, True])
def test_staged_graph_required_in_both_phases(bootstrap_validation, profiling):
    case = bootstrap_validation
    case.runner._profiling_cudagraph_memory = profiling
    case.staged_graph.return_value = False
    with pytest.raises(ValueError, match="requires the staged SFA sparse-load path"):
        case.validate()


@pytest.mark.parametrize("profiling", [False, True])
@pytest.mark.parametrize("window", [None, "0", "-1"])
def test_bounded_decode_window_required_in_both_phases(bootstrap_validation, monkeypatch, profiling, window):
    case = bootstrap_validation
    case.runner._profiling_cudagraph_memory = profiling
    if window is None:
        monkeypatch.delenv("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE")
    else:
        monkeypatch.setenv("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE", window)
    with pytest.raises(ValueError, match="LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE > 0"):
        case.validate()


@pytest.mark.parametrize("profiling", [False, True])
def test_other_roles_do_not_require_d_bootstrap(bootstrap_validation, profiling):
    case = bootstrap_validation
    case.runner._profiling_cudagraph_memory = profiling
    case.runner.dsa_sparse_decode_d_node = False
    case.validate()
    case.has_connector.assert_not_called()
    case.get_connector.assert_not_called()
    case.staged_graph.assert_not_called()
