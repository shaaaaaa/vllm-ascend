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
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")
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
        assert env["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == "1800"
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


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_rpc_timeout_rejects_nonpositive_or_fractional_values(value):
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["--rpc-timeout-seconds", value])


def test_rpc_timeout_reaches_both_child_arguments_and_recorded_environment(tmp_path, monkeypatch):
    args = runner.parser().parse_args(["--rpc-timeout-seconds", "3600"])
    monkeypatch.setattr(runner, "check_shm_capacity", lambda *_: None)
    monkeypatch.setattr(runner, "finish_child", lambda *_: None)
    launched = []

    def launch(command, env, log_path, case, **kwargs):
        assert command[command.index("--rpc-timeout-seconds") + 1] == "3600"
        assert env["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == "3600"
        saved = json.loads((tmp_path / case / "environment.json").read_text())
        assert saved["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == "3600"
        runner.write_json(tmp_path / case / "result.json", {"completed": True})
        launched.append(case)
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(runner, "start_logged_process", launch)
    runner.run_cases(args, tmp_path)
    assert launched == ["off", "on"]


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


@pytest.fixture
def saved_off(tmp_path, monkeypatch):
    import layerwise_prefill_correctness_compare as comparator

    previous = tmp_path / "previous"
    off = previous / "off"
    off.mkdir(parents=True)
    (off / "tensors").mkdir()
    (off / "tensors" / "baseline.pt").write_bytes(b"untouched baseline")
    args = runner.parser().parse_args(["--model", "/models/baseline", "--devices", "4,5", "--prompt-tokens", "80000"])
    ids = [7] * 79992
    model_info = {"model": args.model, "num_hidden_layers": 2, "indexer_types": ["full", "shared"]}
    data = {
        "result": {"case": "off", "completed": True, "prompt_token_ids": ids, "prompt_length": len(ids)},
        "engine_options": runner.correctness_options(args, len(ids)),
        "environment": runner.recorded_environment(runner.correctness_environment(args, "off")),
    }
    assert data["environment"]["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "24"
    for name, value in data.items():
        runner.write_json(off / f"{name}.json", value)
    runner.write_json(previous / "model_info.json", model_info)
    runner.write_json(previous / "prompt.json", {"length": len(ids), "token_ids": ids, "target_tokens": 80000})
    # A failed old ON does not invalidate its successful OFF baseline.
    runner.write_json(previous / "failure.json", {"error": "previous ON failed"})
    (previous / "prompt.txt").write_text("saved prompt", encoding="utf-8")
    calls = []

    def validate(path, info):
        assert path == off.resolve() and info == model_info
        calls.append("validate")
        return data

    def identity(model, root):
        assert calls == ["validate"]
        assert model == args.model
        calls.append("config")
        runner.write_json(root / "model_info.json", model_info)
        return model_info

    monkeypatch.setattr(comparator, "validate_off_baseline", validate)
    monkeypatch.setattr(runner, "record_model_identity", identity)
    return SimpleNamespace(root=previous, off=off, data=data, ids=ids, calls=calls)


@pytest.mark.parametrize("direct_case", [False, True])
def test_reuse_inherits_long_prompt_and_launch_settings_without_copying_tensors(saved_off, tmp_path, direct_case):
    root = tmp_path / "new"
    root.mkdir()
    source = saved_off.off if direct_case else saved_off.root
    args = runner.parser().parse_args(["--off-dir", str(source)])
    old_files = {path: path.read_bytes() for path in saved_off.root.rglob("*") if path.is_file()}
    assert runner.prepare_reused_off(args, root) == len(saved_off.ids)
    assert args.model == "/models/baseline" and args.devices == "4,5"
    assert args.prompt_tokens == 80000 and args.cpu_cache_gb == 24
    assert json.loads((root / "prompt.json").read_text())["token_ids"] == saved_off.ids
    assert json.loads((root / "off_reference.json").read_text())["off_dir"] == str(saved_off.off.resolve())
    assert not (root / "off").exists()
    assert {path: path.read_bytes() for path in saved_off.root.rglob("*") if path.is_file()} == old_files


@pytest.mark.parametrize(
    "override,reason",
    [
        (["--model", "/models/different"], "engine_options differs: model"),
        (["--devices", "0"], "engine_options differs: tensor_parallel_size"),
        (["--cpu-cache-gb", "32"], "environment differs: LMCACHE_MAX_LOCAL_CPU_SIZE"),
        (["--prompt-tokens", "10000"], "--prompt-tokens differs"),
    ],
)
def test_reuse_rejects_incompatible_explicit_overrides_before_loading_config(saved_off, tmp_path, override, reason):
    root = tmp_path / "new"
    root.mkdir()
    args = runner.parser().parse_args(["--off-dir", str(saved_off.root), *override])
    with pytest.raises(ValueError, match=reason):
        runner.prepare_reused_off(args, root)
    assert saved_off.calls == ["validate"]
    assert not (root / "off_reference.json").exists()


def test_reuse_launches_only_on_with_saved_reference(saved_off, tmp_path, monkeypatch):
    root = tmp_path / "new"
    args = ["--off-dir", str(saved_off.root), "--run-dir", str(root), "--cpu-cache-gb", "24.0"]
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner, "check_shm_capacity", lambda *_: None)
    monkeypatch.setattr(runner, "prepare_prompt", lambda *_: pytest.fail("Reuse must not tokenize or rebuild a prompt"))
    launches = []

    def launch(command, env, log_path, case, **kwargs):
        assert case == "on"
        assert (root / "off_reference.json").is_file()
        assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == "true"
        assert command[command.index("--model") + 1] == "/models/baseline"
        assert command[command.index("--prompt-tokens") + 1] == "80000"
        runner.write_json(root / case / "result.json", {"completed": True})
        launches.append(case)
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(runner, "start_logged_process", launch)
    monkeypatch.setattr(runner, "finish_child", lambda _: None)
    monkeypatch.setattr(runner, "compare", lambda path: 0)
    assert runner.main(args) == 0
    assert launches == ["on"] and not (root / "off").exists()
    metadata = json.loads((root / "run.json").read_text())
    assert metadata["cases"] == ["on"] and metadata["devices"] == ["4", "5"]
    assert metadata["rpc_timeout_seconds"] == 1800


@pytest.mark.parametrize("previous_timeout", [None, "300"])
def test_completed_off_reuse_accepts_new_diagnostic_deadline(saved_off, tmp_path, previous_timeout):
    environment = saved_off.data["environment"]
    field = "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
    if previous_timeout is None:
        environment.pop(field)
    else:
        environment[field] = previous_timeout
    root = tmp_path / "new"
    root.mkdir()
    args = runner.parser().parse_args(["--off-dir", str(saved_off.root), "--rpc-timeout-seconds", "3600"])
    assert runner.prepare_reused_off(args, root) == len(saved_off.ids)
    assert runner.correctness_environment(args, "on")[field] == "3600"


def test_reuse_fails_before_launch_when_baseline_is_incomplete(saved_off, tmp_path, monkeypatch):
    import layerwise_prefill_correctness_compare as comparator

    def invalid(*args):
        raise ValueError("OFF tensor file missing")

    monkeypatch.setattr(comparator, "validate_off_baseline", invalid)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner, "run_cases", lambda *_: pytest.fail("Invalid baseline must not launch a model"))
    root = tmp_path / "new"
    assert runner.main(["--off-dir", str(saved_off.root), "--run-dir", str(root)]) == 1
    assert "OFF tensor file missing" in json.loads((root / "failure.json").read_text())["error"]
    assert saved_off.calls == []


def test_reuse_does_not_allow_a_different_prompt_file(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(["--off-dir", str(tmp_path), "--prompt-file", "different.txt"])


def test_reuse_rejects_changed_checkpoint_configuration(saved_off, tmp_path, monkeypatch):
    root = tmp_path / "new"
    root.mkdir()
    args = runner.parser().parse_args(["--off-dir", str(saved_off.root)])
    changed = json.loads((saved_off.root / "model_info.json").read_text())
    changed["indexer_types"] = ["full", "full"]
    monkeypatch.setattr(runner, "record_model_identity", lambda *args: changed)
    with pytest.raises(ValueError, match="model configuration differs: indexer_types"):
        runner.prepare_reused_off(args, root)
    assert not (root / "off_reference.json").exists()
