# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of isolated process configuration; no inference is mocked as passing."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
    assert json.loads(eager["LMCACHE_EXTRA_CONFIG"])["save_only_first_rank"] is False
    assert eager["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "2"
    assert eager["VLLM_ASCEND_ENABLE_FLASHCOMM1"] == "0"
    assert eager["VLLM_ASCEND_MTP_DRAFT_DEBUG"] == "0"
    assert eager["ASCEND_RT_VISIBLE_DEVICES"] == "3"
    assert eager["VLLM_ASCEND_SFA_FULL_GRAPH"] == eager["VLLM_ASCEND_SFA_STAGED_GRAPH"] == "0"
    assert graph["VLLM_ASCEND_SFA_FULL_GRAPH"] == graph["VLLM_ASCEND_SFA_STAGED_GRAPH"] == "1"
    assert driver.os.environ["LMCACHE_CONFIG_FILE"] == "server.yaml"


@pytest.mark.parametrize(
    "overrides", [{"devices": "0,0"}, {"atol": -1}, {"rtol": float("nan")}, {"atol": float("inf")}]
)
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
        assert argv[argv.index("--devices") + 1] == "0,1,2,3,4,5,6,7"
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
        summary = [
            {
                "rank": rank,
                "tp_size": 8,
                "steps": 12,
                "decode_steps": 3,
                "q2_steps": 3,
                "draft_calls": 3,
                "loaded_tokens_per_layer": [100 if mode == "graph" else 101] * 8,
            }
            for rank in range(8)
        ]
        (directory / f"{mode}-summary.json").write_text(json.dumps(summary))

    monkeypatch.setattr(driver.subprocess, "run", launch)
    driver.run_pair(str(tmp_path))
    assert launches == ["eager", "graph"]


@pytest.mark.parametrize("devices", ["", "0,", "-1", "0,0", "0,1,2", "0, 1", "０", "0,a"])
def test_invalid_device_lists(driver, devices):
    with pytest.raises(ValueError):
        driver.parse_devices(devices)


@pytest.mark.parametrize(
    "devices,expected", [("0", (0,)), ("4,5", (4, 5)), ("7,6,5,4,3,2,1,0", tuple(reversed(range(8))))]
)
def test_explicit_devices_define_tp_size(driver, devices, expected):
    assert driver.parse_devices(devices) == expected


def reports():
    return [
        {
            "rank": rank,
            "tp_size": 8,
            "steps": 12,
            "decode_steps": 3,
            "q2_steps": 3,
            "draft_calls": 3,
            "loaded_tokens_per_layer": [10] * 8,
        }
        for rank in range(8)
    ]


def test_rpc_order_does_not_define_rank_identity(driver):
    driver.validate_summaries(reports(), list(reversed(reports())), 8)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "world", "step", "no_q2", "no_transfer"])
def test_rank_zero_success_cannot_hide_peer_failure(driver, kind):
    actual = reports()
    if kind == "missing":
        actual.pop()
    elif kind == "duplicate":
        actual[7]["rank"] = 0
    elif kind == "world":
        actual[7]["tp_size"] = 1
    elif kind == "step":
        actual[7]["steps"] -= 1
    elif kind == "no_q2":
        actual[7]["q2_steps"] = 0
    else:
        actual[7]["loaded_tokens_per_layer"][5] = 0
    with pytest.raises(AssertionError):
        driver.validate_summaries(reports(), actual, 8)


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_child_constructs_tp8_not_eight_dp_replicas(driver, monkeypatch, tmp_path, mode):
    # Configuration wiring only: this stub is not an inference parity test.
    llm = Mock()
    llm.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[driver.FIXED_TOKEN] * 16)])]
    llm.collective_rpc.return_value = reports()
    constructor = Mock(return_value=llm)
    stub = ModuleType("vllm")
    stub.LLM = constructor
    stub.SamplingParams = Mock()
    monkeypatch.setitem(sys.modules, "vllm", stub)
    driver.run_child(
        SimpleNamespace(
            child=mode,
            devices=driver.DEFAULT_DEVICES,
            model="local-model",
            reference=str(tmp_path),
            atol=1e-7,
            rtol=1e-2,
        )
    )
    config = constructor.call_args.kwargs
    assert config["tensor_parallel_size"] == 8
    assert config["data_parallel_size"] == 1
    assert config["distributed_executor_backend"] == "mp"
    assert not config["enable_expert_parallel"]
    assert config["hf_overrides"] == {"num_hidden_layers": 8}
    assert config["speculative_config"]["num_speculative_tokens"] == 1
    assert not config["compilation_config"]["pass_config"]["enable_sp"]
    assert config["enforce_eager"] == (mode == "eager")
    assert config["worker_cls"].endswith(".SFAParityWorker")
