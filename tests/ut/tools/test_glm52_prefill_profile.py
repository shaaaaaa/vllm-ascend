# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the single-node profiler entry point."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PATH = Path(__file__).resolve().parents[3] / "tools/glm52_prefill_profile.py"
SPEC = importlib.util.spec_from_file_location("glm52_prefill_profile", PATH)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


@pytest.fixture
def model(tmp_path):
    types = ["full" if i < 3 or (i >= 6 and (i - 6) % 4 == 0) else "shared" for i in range(79)]
    config = {
        "num_hidden_layers": 79,
        "indexer_types": types,
        "vocab_size": 154880,
        "index_topk_pattern": ["F" if t == "full" else "S" for t in types],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_eight_layer_config_preserves_physical_indexer_topology(model):
    overrides = profile.model_overrides(model)
    assert overrides["num_hidden_layers"] == 8
    assert [i for i, t in enumerate(overrides["indexer_types"]) if t == "full"] == [0, 1, 2, 6]
    assert len(overrides["index_topk_pattern"]) == 8
    # Never alter the actual model config on disk.
    assert json.loads((model / "config.json").read_text())["num_hidden_layers"] == 79


@pytest.mark.parametrize(
    "change",
    [
        {"num_hidden_layers": 7},
        {"indexer_types": ["full"] * 79},
        {"indexer_types": ["shared"] * 79},
        {"indexer_types": None},
        {"index_topk_pattern": ["F"] * 79},
        {"vocab_size": 128},
    ],
)
def test_bad_or_unrepresentative_model_config_rejected(model, change):
    path = model / "config.json"
    config = json.loads(path.read_text())
    config.update(change)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        profile.model_overrides(model)


def test_environment_isolated_from_deployed_pd_settings(monkeypatch):
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "external.yaml")
    monkeypatch.setenv("LMCACHE_REMOTE_URL", "mooncakestore://deployment")
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "external.json")
    monkeypatch.setenv("LMCACHE_STORE_ASYNC", "true")
    monkeypatch.setenv("VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE", "true")
    monkeypatch.setenv("VLLM_DP_SIZE", "8")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/cann/lib64")
    env = profile.environment("0,1,2,3,4,5,6,7")
    assert env["LMCACHE_STORE_ASYNC"] == "false"
    assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == "true"
    assert env["VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE"] == "false"
    assert env["LD_LIBRARY_PATH"] == "/cann/lib64"
    assert "LMCACHE_CONFIG_FILE" not in env
    assert "MOONCAKE_CONFIG_PATH" not in env
    assert "LMCACHE_REMOTE_URL" not in env
    assert "VLLM_DP_SIZE" not in env


@pytest.mark.parametrize("no_profile", [False, True])
def test_single_node_options_with_multiple_prefill_chunks(model, no_profile):
    args = profile.parse_args(["--model", str(model)] + (["--no-profile"] if no_profile else []))
    options = profile.engine_options(args, profile.model_overrides(model))
    assert options["tensor_parallel_size"] == 8
    assert options["data_parallel_size"] == 1
    assert options["max_num_seqs"] == 1
    assert args.prompt_tokens == 30000 > options["max_num_batched_tokens"] == 4096
    assert options["load_format"] == "dummy"
    assert options["enforce_eager"] and options["enable_chunked_prefill"]
    assert options["additional_config"]["recompute_scheduler_enable"] is False
    assert options["kv_transfer_config"]["kv_role"] == "kv_producer"
    assert ("profiler_config" in options) is not no_profile
    assert "speculative_config" not in options


@pytest.mark.parametrize(
    "extra",
    [["--devices", "0"], ["--devices", "0,0,1,2,3,4,5,6"], ["--prompt-tokens", "4096"], ["--chunk-tokens", "4095"]],
)
def test_bad_run_arguments_rejected(extra):
    with pytest.raises(SystemExit):
        profile.parse_args(["--model", "model", *extra])


@pytest.mark.parametrize("profiling", [False, True])
@pytest.mark.parametrize("failure", [None, "generate", "stop_profile"])
def test_profiles_exactly_one_complete_prefill_and_shuts_down(profiling, failure):
    events = []
    llm = Mock()
    llm.start_profile.side_effect = lambda **_: events.append("start")

    def generate(prompt, params, **kwargs):
        events.append("generate")
        assert len(prompt["prompt_token_ids"]) == 30000
        assert max(prompt["prompt_token_ids"]) < 512
        if failure == "generate":
            raise ValueError("model failed")
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[0])])]

    def stop():
        events.append("stop")
        if failure in ("generate", "stop_profile"):
            raise RuntimeError("stop failed")

    llm.generate.side_effect = generate
    llm.stop_profile.side_effect = stop
    llm.llm_engine.engine_core.shutdown.side_effect = lambda: events.append("shutdown")
    if failure == "generate":
        with pytest.raises(ValueError, match="model failed"):
            profile.run_request(llm, object(), 30000, profiling)
    elif failure == "stop_profile" and profiling:
        with pytest.raises(RuntimeError, match="stop failed"):
            profile.run_request(llm, object(), 30000, profiling)
    else:
        assert profile.run_request(llm, object(), 30000, profiling) >= 0
    assert events == (["start", "generate", "stop", "shutdown"] if profiling else ["generate", "shutdown"])
    assert llm.generate.call_count == 1


def test_offline_analysis_uses_existing_raw_directory_and_checks_all_ranks(tmp_path, monkeypatch):
    execute = Mock()
    monkeypatch.setattr(profile.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="Expected 8 worker traces"):
        profile.analyse(tmp_path)
    for rank in range(8):
        output = tmp_path / f"worker{rank}_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
        output.mkdir(parents=True)
        (output / "trace_view.json").write_text("{}")
    profile.analyse(tmp_path)
    command = execute.call_args.args[0]
    assert "max_process_number=2" in command[2]
    assert command[-1] == str(tmp_path.resolve())


def test_dry_run_requires_no_accelerator_imports(model, capsys):
    profile.main(["--model", str(model), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["hf_overrides"]["num_hidden_layers"] == 8
