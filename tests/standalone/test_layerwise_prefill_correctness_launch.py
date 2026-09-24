# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the full-model correctness launch contract."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("correctness_launch", TOOLS / "layerwise_prefill_correctness.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_off_on_preserve_model_and_storage_layout(monkeypatch):
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "/unrelated/deployment.yaml")
    monkeypatch.setenv("LMCACHE_REMOTE_URL", "mooncakestore://unrelated:1234/")
    monkeypatch.setenv("LMCACHE_ENABLE_REMOTE_LMCACHE_STORE", "true")
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH", "1")
    args = runner.parser().parse_args([])
    assert args.model == "/workspace/models/GLM-5.2-w4a8c8-0723"
    off = runner.correctness_environment(args, "off")
    on = runner.correctness_environment(args, "on")
    differences = {key for key in off.keys() | on.keys() if off.get(key) != on.get(key)}
    assert differences == {"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "LMCACHE_STORE_ASYNC"}
    for env in (off, on):
        assert "LMCACHE_CONFIG_FILE" not in env
        assert "LMCACHE_REMOTE_URL" not in env
        assert "LMCACHE_ENABLE_REMOTE_LMCACHE_STORE" not in env
        assert "VLLM_ASCEND_SFA_STAGED_GRAPH" not in env
        assert env["VLLM_ASCEND_ENABLE_FLASHCOMM1"] == "1"
        extra = json.loads(env["LMCACHE_EXTRA_CONFIG"])
        assert extra["mooncake_layer_merged_page_objects"] is True
        assert extra["mooncake_page_first_multi_buffer"] is True
        assert extra["save_only_first_rank"] is True
    options = runner.correctness_options(args, 10000)
    assert options["enforce_eager"] is True
    assert options["speculative_config"]["num_speculative_tokens"] == 1
    assert options["tensor_parallel_size"] == 8
    assert options["max_num_batched_tokens"] == 4096
    assert "profiler_config" not in options
    assert "num_hidden_layers" not in options


def test_failed_off_never_launches_on_or_clears_shared_memory(tmp_path, monkeypatch):
    args = runner.parser().parse_args(["--run-dir", str(tmp_path)])
    launches = []
    finished = []
    monkeypatch.setattr(runner, "check_shm_capacity", lambda *_: None)

    def start(command, env, log_path, case, **kwargs):
        launches.append(case)
        return SimpleNamespace(wait=lambda: 17)

    monkeypatch.setattr(runner, "start_logged_process", start)
    monkeypatch.setattr(runner, "finish_child", lambda proc: finished.append(proc))
    with pytest.raises(RuntimeError, match="off failed"):
        runner.run_cases(args, tmp_path)
    assert launches == ["off"]
    assert len(finished) == 1
    assert not (tmp_path / "on").exists()


def test_successful_exit_without_complete_result_is_not_accepted(tmp_path, monkeypatch):
    args = runner.parser().parse_args(["--run-dir", str(tmp_path)])
    monkeypatch.setattr(runner, "check_shm_capacity", lambda *_: None)
    monkeypatch.setattr(runner, "start_logged_process", lambda *a, **kw: SimpleNamespace(wait=lambda: 0))
    monkeypatch.setattr(runner, "finish_child", lambda *_: None)
    with pytest.raises(RuntimeError, match="did not complete"):
        runner.run_cases(args, tmp_path)
    assert not (tmp_path / "on").exists()


def test_existing_artifacts_are_never_overwritten(tmp_path, monkeypatch):
    sentinel = tmp_path / "off_tensor.pt"
    sentinel.write_bytes(b"baseline")
    monkeypatch.setattr(runner.sys, "platform", "linux")
    with pytest.raises(SystemExit):
        runner.main(["--run-dir", str(tmp_path)])
    assert sentinel.read_bytes() == b"baseline"


def test_analysis_can_run_on_cpu_without_initializing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "win32")
    analysed = []
    monkeypatch.setattr(runner, "compare", lambda root: analysed.append(root) or 0)
    assert runner.main(["--compare-only", str(tmp_path)]) == 0
    assert analysed == [tmp_path.resolve()]


@pytest.mark.parametrize("length", [4097, 4098, 8193, 8194])
def test_decode_remapped_tail_is_rejected_before_model_loading(length):
    with pytest.raises(ValueError, match="decode KV remapping"):
        runner.validate_prompt_length(length)


@pytest.mark.parametrize("length", [4099, 8192, 10000, 80000])
def test_prefill_tail_with_or_without_tp_padding_is_supported(length):
    runner.validate_prompt_length(length)
