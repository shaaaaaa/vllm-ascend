# SPDX-License-Identifier: Apache-2.0
"""No NPU required: isolation, reload proof and sequential-process contracts."""

import ast
import copy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional

import pytest


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    return importlib.import_module("layerwise_prefill_mooncake_check")


def options(tool):
    args = tool.parser().parse_args([])
    args.local_hostname = "7.150.7.133"
    return args


def test_no_required_config_and_deployment_defaults(tool):
    args = options(tool)
    assert not hasattr(args, "config")
    assert args.master == "7.150.4.174:58888"
    base = tool.deployment_config(args.master, args.local_hostname)
    assert base["chunk_size"] == 1024
    assert base["pin_timeout_sec"] == 1800
    assert base["shared_cpu_cache_numa_policy"] == "interleave"
    assert base["extra_config"]["local_hostname"] == "7.150.7.133"
    assert base["extra_config"]["transfer_timeout"] == 120
    assert base["extra_config"]["mooncake_dsa_raw_token_dims"] == {0: 576, 1: 128}


def test_local_hostname_comes_from_route_not_remote_master(tool, monkeypatch):
    calls = []

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect(self, destination):
            calls.append(destination)

        def getsockname(self):
            return ("7.150.7.133", 45000)

    monkeypatch.setattr(tool.socket, "socket", lambda *args: Probe())
    assert tool.detect_local_hostname("7.150.4.174:58888") == "7.150.7.133"
    assert calls == [("7.150.4.174", 58888)]


def base_config():
    return {
        "remote_url": "mooncakestore://localhost:58888/",
        "chunk_size": 1024,
        "shared_cpu_cache_name": "old-cache",
        "extra_config": {
            "global_segment_size": 100000000000,
            "local_hostname": "localhost",
            "master_server_address": "localhost:58888",
            "shared_cpu_cache_size_gb": 190,
        },
    }


@pytest.mark.parametrize("stage", ["baseline", "prefill", "decode"])
def test_models_cannot_own_persistent_storage_or_reuse_old_cache(tool, stage):
    source = base_config()
    config = tool.stage_config(source, options(tool), stage)
    assert source == base_config()
    assert config["extra_config"]["global_segment_size"] == 0
    assert config["extra_config"]["mooncake_prefer_local_alloc"] is False
    assert config["shared_cpu_cache_name"] is None
    assert config["extra_config"]["shared_cpu_cache_name"] is None
    assert config["extra_config"]["shared_cpu_cache_size_gb"] == 8
    assert config["enable_remote_lmcache_store"] == (stage != "baseline")
    assert bool(config["remote_url"]) == (stage != "baseline")


@pytest.mark.parametrize("stage", ["baseline", "prefill", "decode"])
def test_generated_config_passes_real_lmcache_validation(tool, stage, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "LMCache"))
    import lmcache.v1.config as config_module

    # Execute the actual Ascend config extension, without importing its NPU
    # plugin initializer. Restore global config/class state after each test.
    monkeypatch.setattr(config_module, "_CONFIG_DEFINITIONS", copy.deepcopy(config_module._CONFIG_DEFINITIONS))
    monkeypatch.setattr(config_module, "LMCacheEngineConfig", config_module.LMCacheEngineConfig)
    patch_path = Path(__file__).resolve().parents[4] / "LMCache-Ascend/lmcache_ascend/__init__.py"
    tree = ast.parse(patch_path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if getattr(node, "name", None) == "_patch_config")
    namespace = {"Optional": Optional, "sys": NS(modules={})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(patch_path), "exec"), namespace)
    namespace["_patch_config"]()

    for key in list(os.environ):
        if key.startswith(("LMCACHE_", "MOONCAKE_")):
            monkeypatch.delenv(key)
    env = tool.child_environment(options(tool), tmp_path, stage)
    for key, value in env.items():
        if key.startswith("LMCACHE_"):
            monkeypatch.setenv(key, value)
    obj = config_module.LMCacheEngineConfig.from_env()
    obj.validate()
    assert obj.chunk_size == 1024
    assert obj.extra_config["transfer_timeout"] == 120
    assert obj.remote_url == (None if stage == "baseline" else "mooncakestore://7.150.4.174:58888/")
    assert obj.pd_role == ("receiver" if stage == "decode" else "sender")


def test_no_archive_or_probe_environment_leaks(tool, tmp_path, monkeypatch):
    monkeypatch.setenv("LMCACHE_EXTRA_CONFIG", '{"validation_stage_dir":"stale"}')
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "other.json")
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "stale.yaml")
    env = tool.child_environment(options(tool), tmp_path, "prefill")
    assert "validation_stage_dir" not in json.loads(env["LMCACHE_EXTRA_CONFIG"])
    assert "MOONCAKE_CONFIG_PATH" not in env
    assert "LMCACHE_CONFIG_FILE" not in env
    assert env["LMCACHE_CHUNK_SIZE"] == "1024"
    assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == "true"


@pytest.mark.parametrize("stage", ["holder", "baseline", "prefill", "decode"])
def test_only_holder_environment_has_a_storage_segment(tool, tmp_path, stage):
    env = tool.child_environment(options(tool), tmp_path, stage)
    extra = json.loads(env["LMCACHE_EXTRA_CONFIG"])
    assert extra["global_segment_size"] == (8 * 1024**3 if stage == "holder" else 0)
    assert "LMCACHE_CONFIG_FILE" not in env


@pytest.mark.parametrize("p_cached,d_cached,valid", [(0, 8192, True), (1, 8192, False), (0, 0, False)])
def test_summary_requires_real_fresh_p_and_d_reload(tool, tmp_path, p_cached, d_cached, valid):
    tool.write_json(tmp_path / "prompt.json", {"length": 9000})
    for stage, cached, tokens in [("baseline", 0, [1, 2]), ("prefill", p_cached, [1]), ("decode", d_cached, [1, 3])]:
        (tmp_path / stage).mkdir()
        tool.write_json(tmp_path / stage / "output.json", {"num_cached_tokens": cached, "token_ids": tokens})
    if valid:
        tool.analyse(tmp_path, 1024)
    else:
        with pytest.raises(RuntimeError, match="Did not exercise"):
            tool.analyse(tmp_path, 1024)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["p_computed_and_d_reloaded"] is valid
    assert summary["decode_first_difference"] == 1  # differences are reported, not raised


def test_prefill_group_is_fully_stopped_before_decode_launch(tool, tmp_path, monkeypatch):
    actions = []

    def start(args, root, stage):
        actions.append(("start", stage))
        return NS(stage=stage, poll=lambda: 0, returncode=0)

    monkeypatch.setattr(tool, "start_child", start)
    monkeypatch.setattr(tool, "finish_child", lambda proc: actions.append(("stop", proc.stage)))
    tool.run_models(options(tool), tmp_path, NS(poll=lambda: None))
    assert actions == [(action, stage) for stage in ("baseline", "prefill", "decode") for action in ("start", "stop")]
