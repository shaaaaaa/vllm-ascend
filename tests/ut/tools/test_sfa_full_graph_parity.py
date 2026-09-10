# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of isolated process configuration; no inference is mocked as passing."""

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[3] / "tools/sfa_full_graph_parity.py"
    spec = importlib.util.spec_from_file_location("tested_sfa_parity_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_child_environment_does_not_inherit_server_state(driver, monkeypatch):
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "server.yaml")
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "server.json")
    monkeypatch.setenv("LMCACHE_ENABLE_SHARED_CPU_CACHE", "true")
    monkeypatch.setenv("VLLM_ASCEND_MTP_DRAFT_DEBUG", "1")
    monkeypatch.setenv("VLLM_ASCEND_SFA_FULL_GRAPH", "1")
    eager = driver.child_environment("eager", "3")
    graph = driver.child_environment("graph", "3")
    assert "LMCACHE_CONFIG_FILE" not in eager
    assert "MOONCAKE_CONFIG_PATH" not in eager
    assert eager["LMCACHE_ENABLE_SHARED_CPU_CACHE"] == "false"
    assert eager["VLLM_ASCEND_MTP_DRAFT_DEBUG"] == "0"
    assert eager["ASCEND_RT_VISIBLE_DEVICES"] == "3"
    assert eager["VLLM_ASCEND_SFA_FULL_GRAPH"] == eager["VLLM_ASCEND_SFA_STAGED_GRAPH"] == "0"
    assert graph["VLLM_ASCEND_SFA_FULL_GRAPH"] == graph["VLLM_ASCEND_SFA_STAGED_GRAPH"] == "1"
    assert driver.os.environ["LMCACHE_CONFIG_FILE"] == "server.yaml"


@pytest.mark.parametrize("overrides", [{"device": "0,1"}, {"atol": -1}, {"rtol": float("nan")}, {"atol": float("inf")}])
def test_invalid_configuration_never_starts_engine(driver, tmp_path, monkeypatch, overrides):
    (tmp_path / "config.json").write_text("{}")
    launch = Mock()
    monkeypatch.setattr(driver.subprocess, "run", launch)
    with pytest.raises(ValueError):
        driver.run_pair(str(tmp_path), **overrides)
    launch.assert_not_called()


def test_eager_failure_stops_before_graph_and_propagates_exit_status(driver, tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    launch = Mock(side_effect=subprocess.CalledProcessError(1, "eager"))
    monkeypatch.setattr(driver.subprocess, "run", launch)
    with pytest.raises(subprocess.CalledProcessError):
        driver.run_pair(str(tmp_path))
    assert launch.call_count == 1
    assert launch.call_args.args[0][3] == "eager"


def test_pair_checks_coverage_not_identical_cache_allocation_counts(driver, tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    launches = []

    def launch(argv, *, env, check):
        launches.append(argv[3])
        directory = Path(argv[argv.index("--reference") + 1])
        mode = argv[argv.index("--child") + 1]
        summary = [
            {
                "steps": 12,
                "decode_steps": 3,
                "q2_steps": 3,
                "draft_calls": 3,
                "loaded_tokens_per_layer": [100 if mode == "graph" else 101] * 8,
            }
        ]
        (directory / f"{mode}-summary.json").write_text(json.dumps(summary))

    monkeypatch.setattr(driver.subprocess, "run", launch)
    driver.run_pair(str(tmp_path))
    assert launches == ["eager", "graph"]
