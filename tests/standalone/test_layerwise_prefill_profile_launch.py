# SPDX-License-Identifier: Apache-2.0
"""Profile launch isolation and request diagnostics without importing an NPU runtime."""

import ast
import builtins
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def profile_tool(monkeypatch):
    tools_dir = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools_dir))
    original_import = builtins.__import__
    blocked_roots = {"torch", "torch_npu", "vllm", "vllm_ascend", "lmcache", "lmcache_ascend"}

    def import_without_runtime(name, *args, **kwargs):
        if name.split(".")[0] in blocked_roots:
            raise AssertionError(f"Importing the launch tool loaded a device runtime: {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as import_patch:
        import_patch.setattr(builtins, "__import__", import_without_runtime)
        spec = importlib.util.spec_from_file_location(
            "standalone_layerwise_prefill_profile_launch", tools_dir / "layerwise_prefill_profile.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def test_checkpoint_default_and_recorded_indexer_schedule(profile_tool, tmp_path, monkeypatch, capsys):
    assert profile_tool.parser().parse_args([]).model == "/workspace/models/GLM-5.2-w4a8c8-0723"
    config = SimpleNamespace(
        model_type="glm_moe_dsa", num_hidden_layers=4, indexer_types=["full", "shared", "full", "shared"]
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config))
    )
    info = profile_tool.record_model_identity("/models/explicit-checkpoint", tmp_path)
    assert json.loads((tmp_path / "model_info.json").read_text()) == info
    assert info["model"] == "/models/explicit-checkpoint"
    assert info["num_hidden_layers"] == 4
    assert "producer layers=[0, 2]; shared layers=2" in capsys.readouterr().out


@pytest.mark.parametrize("prompt_size,prompt_tokens,max_len", [("10k", 10000, 16384), ("80k", 80000, 84096)])
def test_case_launch_uses_isolated_inline_configuration(
    profile_tool, monkeypatch, tmp_path, prompt_size, prompt_tokens, max_len
):
    inherited = {
        "LMCACHE_CONFIG_FILE": "/missing/deployment-config.yaml",
        "LMCACHE_REMOTE_URL": "mooncake://old-deployment",
        "LMCACHE_EXTERNAL_LOOKUP_CLIENT": "old-client",
        "LMCACHE_ENABLE_REMOTE_LMCACHE_STORE": "true",
        "LMCACHE_STORE_ASYNC": "false",
        "LMCACHE_STORE_ASYNC_MAX_QUEUE_SIZE": "99",
        "LMCACHE_ENABLE_ASYNC_LOADING": "true",
        "MOONCAKE_CONFIG_PATH": "/missing/mooncake.json",
        "MOONCAKE_MASTER_SERVER": "old-master:50051",
        "VLLM_STALE_PROFILE_SETTING": "old-value",
        "HCCL_IF_IP": "192.0.2.1",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    args = SimpleNamespace(devices="0,1,2,3,4,5,6,7", cpu_cache_gb=24, model="/models/test")

    off = profile_tool.case_environment(args, f"{prompt_size}_off")
    on = profile_tool.case_environment(args, f"{prompt_size}_on")

    for env in (off, on):
        for key in (
            "LMCACHE_CONFIG_FILE",
            "LMCACHE_REMOTE_URL",
            "LMCACHE_EXTERNAL_LOOKUP_CLIENT",
            "LMCACHE_ENABLE_REMOTE_LMCACHE_STORE",
            "VLLM_STALE_PROFILE_SETTING",
        ):
            assert key not in env
        assert not any(key.startswith("MOONCAKE_") for key in env)
        assert env["LMCACHE_LOCAL_CPU"] == "true"
        assert env["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "24"
        assert env["LMCACHE_ENABLE_SHARED_CPU_CACHE"] == "true"
        assert env["LMCACHE_STORE_ASYNC_MAX_QUEUE_SIZE"] == "2"
        assert env["LMCACHE_ENABLE_ASYNC_LOADING"] == "false"
        assert env["LMCACHE_PREFILL_START_TIMING"] == "0"
        assert env["PD_SERVING_PERF"] == "0"
        assert env["LMCACHE_PREFILL_REUSE_DEBUG_RANK"] == "1"
        assert json.loads(env["LMCACHE_EXTRA_CONFIG"])["save_only_first_rank"] is True
        assert env["HCCL_IF_IP"] == inherited["HCCL_IF_IP"]

    switch = "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"
    assert off.pop(switch) == "false"
    assert on.pop(switch) == "true"
    assert off.pop("LMCACHE_STORE_ASYNC") == "false"
    assert on.pop("LMCACHE_STORE_ASYNC") == "true"
    assert off == on
    assert profile_tool.os.environ["LMCACHE_CONFIG_FILE"] == inherited["LMCACHE_CONFIG_FILE"]

    options = profile_tool.engine_options(args, tmp_path, prompt_tokens)
    assert options["tensor_parallel_size"] == 8
    assert options["data_parallel_size"] == 1
    assert options["max_model_len"] == max_len
    assert options["max_num_batched_tokens"] == 4096
    assert options["async_scheduling"] is None
    assert options["kv_transfer_config"]["kv_role"] == "kv_both"


def test_reuse_log_is_available_before_worker_launch(profile_tool, monkeypatch, tmp_path):
    args = SimpleNamespace(devices="0,1,2,3,4,5,6,7", cpu_cache_gb=24, model="/models/test")
    launched = []

    def launch(command, env, server_log, case, **kwargs):
        target = Path(env["LMCACHE_PREFILL_REUSE_DEBUG_FILE"])
        assert target == (tmp_path / case / "reuse.log").resolve()
        assert target.is_file()
        assert env["LMCACHE_PREFILL_REUSE_DEBUG_RANK"] == "1"
        assert env["PD_SERVING_PERF"] == env["LMCACHE_PREFILL_START_TIMING"] == "0"
        launched.append(target)
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(profile_tool, "start_logged_process", launch)
    monkeypatch.setattr(profile_tool, "finish_child", lambda _: None)
    monkeypatch.setattr(profile_tool, "analyse_case", lambda _: None)
    profile_tool.run_cases(args, tmp_path, ("80k_on",))
    assert len(launched) == 1


@pytest.fixture
def validate_adapter_store_configuration():
    adapter_path = (
        Path(__file__).resolve().parents[3] / "LMCache-Ascend/lmcache_ascend/integration/vllm/vllm_v1_adapter.py"
    )
    if not adapter_path.is_file():
        pytest.skip("The sibling LMCache-Ascend checkout is required for the real adapter guard")
    tree = ast.parse(adapter_path.read_text(encoding="utf-8"))
    adapter = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LMCacheAscendConnectorV1Impl"
    )
    constructor = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    fields = {
        "use_layerwise",
        "store_async",
        "_force_layerwise_prefill_store",
        "_remote_store_requested",
        "_direct_store_requested",
    }
    nodes = []
    for node in constructor.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                (isinstance(target, ast.Attribute) and target.attr in fields)
                or (isinstance(target, ast.Name) and target.id == "get_extra")
                for target in node.targets
            )
            or isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Constant) and child.value == "Layerwise storing is not supported with async store"
                for child in ast.walk(node)
            )
        ):
            nodes.append(node)
    assert len(nodes) == len(fields) + 2
    code = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(adapter_path), "exec")

    def validate(env, kv_role):
        extra = json.loads(env["LMCACHE_EXTRA_CONFIG"])
        config = SimpleNamespace(
            use_layerwise=env["LMCACHE_USE_LAYERWISE"] == "true",
            store_async=env["LMCACHE_STORE_ASYNC"] == "true",
            enable_remote_lmcache_store=env.get("LMCACHE_ENABLE_REMOTE_LMCACHE_STORE", "false") == "true",
            pd_role=env.get("LMCACHE_PD_ROLE"),
            get_extra_config_value=lambda key, default: extra.get(key, default),
        )
        exec(
            code,
            {
                "self": SimpleNamespace(config=config, kv_role=kv_role),
                "role": "worker",
                "KVConnectorRole": SimpleNamespace(SCHEDULER="scheduler"),
                "os": SimpleNamespace(getenv=env.get),
            },
        )

    return validate


@pytest.mark.parametrize("case", ["10k_off", "10k_on", "80k_off", "80k_on"])
def test_profile_cases_pass_real_adapter_store_guard(
    profile_tool, validate_adapter_store_configuration, tmp_path, case
):
    args = SimpleNamespace(devices="0,1,2,3,4,5,6,7", cpu_cache_gb=24, model="/models/test")
    env = profile_tool.case_environment(args, case)
    options = profile_tool.engine_options(args, tmp_path, 80000 if case.startswith("80k") else 10000)
    kv_role = options["kv_transfer_config"]["kv_role"]

    validate_adapter_store_configuration(env, kv_role)

    # Replay the former OFF combination to prove the actual guard detects it.
    if case.endswith("_off"):
        env["LMCACHE_STORE_ASYNC"] = "true"
        with pytest.raises(ValueError, match="Layerwise storing is not supported with async store"):
            validate_adapter_store_configuration(env, kv_role)


@pytest.fixture
def submission_watchdog(profile_tool, monkeypatch):
    watchdog = SimpleNamespace(dump_traceback_later=Mock(), cancel_dump_traceback_later=Mock())
    monkeypatch.setattr(profile_tool, "faulthandler", watchdog)
    return watchdog


def test_submission_success_cancels_before_execution_and_restores_method(profile_tool, submission_watchdog, tmp_path):
    original = Mock(return_value="request-id")
    engine = SimpleNamespace(add_request=original)
    llm = SimpleNamespace(llm_engine=engine)

    with profile_tool.trace_request_submission(llm, "80k_on", tmp_path):
        submission_watchdog.dump_traceback_later.assert_called_once()
        call = submission_watchdog.dump_traceback_later.call_args
        assert call.args == (120,)
        assert call.kwargs["repeat"] is False
        stack_file = call.kwargs["file"]
        assert not stack_file.closed
        submission_watchdog.cancel_dump_traceback_later.assert_not_called()
        assert engine.add_request("r0", prompt="tokens") == "request-id"
        original.assert_called_once_with("r0", prompt="tokens")
        # Inference may still be running here; the frontend watchdog must be off.
        submission_watchdog.cancel_dump_traceback_later.assert_called_once_with()
        assert not stack_file.closed

    assert engine.add_request is original
    assert stack_file.closed
    assert submission_watchdog.cancel_dump_traceback_later.call_count >= 1
    logs = list((tmp_path / "startup-stacks").glob("frontend-*.log"))
    assert len(logs) == 1
    assert "80k_on: frontend request submission" in logs[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("failure_point", ["preprocessing", "enqueue", "execution"])
def test_submission_error_restores_method_cancels_watchdog_and_closes_log(
    profile_tool, submission_watchdog, tmp_path, failure_point
):
    error = RuntimeError(f"failed during {failure_point}")
    original = Mock(side_effect=error if failure_point == "enqueue" else None)
    engine = SimpleNamespace(add_request=original)
    llm = SimpleNamespace(llm_engine=engine)

    with pytest.raises(RuntimeError) as raised, profile_tool.trace_request_submission(llm, "80k_on", tmp_path):
        stack_file = submission_watchdog.dump_traceback_later.call_args.kwargs["file"]
        assert not stack_file.closed
        if failure_point != "preprocessing":
            engine.add_request("r0")
        raise error

    assert raised.value is error
    assert engine.add_request is original
    assert stack_file.closed
    submission_watchdog.cancel_dump_traceback_later.assert_called_with()
    assert original.call_count == (0 if failure_point == "preprocessing" else 1)
