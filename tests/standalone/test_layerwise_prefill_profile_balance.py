# SPDX-License-Identifier: Apache-2.0
"""Reproduce balance-scheduler admission starvation without an NPU runtime."""

import ast
import builtins
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def waiting_admission():
    source = ROOT / "vllm_ascend" / "patch" / "platform" / "patch_balance_schedule.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    scheduler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BalanceScheduler")
    schedule = next(node for node in scheduler.body if isinstance(node, ast.FunctionDef) and node.name == "schedule")
    waiting_loop = next(
        node
        for node in ast.walk(schedule)
        if isinstance(node, ast.While) and ast.unparse(node.test) == "self.waiting and token_budget > 0"
    )
    peek_index = next(
        index
        for index, node in enumerate(waiting_loop.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.value) == "self.waiting.peek_request()"
    )
    lookup = next(
        node
        for node in ast.walk(waiting_loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_num_new_matched_tokens"
    )
    assert waiting_loop.body[peek_index].lineno < lookup.lineno

    # Execute the original capacity and balance guards through the first peek.
    # Return at that boundary rather than requiring the later KV/model runtime.
    function = copy.deepcopy(schedule)
    function.name = "waiting_admission"
    function.returns = None
    loop = copy.deepcopy(waiting_loop)
    loop.body = loop.body[: peek_index + 1] + [ast.Return(value=ast.Name(id="request", ctx=ast.Load()))]
    budget = next(
        node
        for node in schedule.body
        if isinstance(node, ast.Assign) and any(ast.unparse(target) == "token_budget" for target in node.targets)
    )
    function.body = [copy.deepcopy(budget), loop, ast.Return(value=ast.Constant(value=None))]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["waiting_admission"]


def idle_scheduler(max_num_seqs):
    request = object()
    waiting = SimpleNamespace(peek_request=Mock(return_value=request))
    return SimpleNamespace(
        waiting=waiting,
        running=[],
        balance_queue=[SimpleNamespace(item=lambda: 0)],
        max_num_running_reqs=max_num_seqs,
        max_num_scheduled_tokens=4096,
    ), request


def test_balance_scheduler_single_request_capacity_never_reaches_lookup(waiting_admission):
    scheduler, _ = idle_scheduler(max_num_seqs=1)
    for _ in range(3):
        assert waiting_admission(scheduler) is None
    scheduler.waiting.peek_request.assert_not_called()
    assert scheduler.running == []


def test_balance_scheduler_admission_resumes_above_reserved_slot(waiting_admission):
    scheduler, request = idle_scheduler(max_num_seqs=2)
    assert waiting_admission(scheduler) is request
    scheduler.waiting.peek_request.assert_called_once_with()


@pytest.fixture
def profile_tool(monkeypatch):
    tools_dir = ROOT / "tools"
    monkeypatch.syspath_prepend(str(tools_dir))
    original_import = builtins.__import__

    def import_without_runtime(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_npu", "vllm", "vllm_ascend", "lmcache", "lmcache_ascend"}:
            raise AssertionError(f"The profile tool imported a device runtime: {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as import_patch:
        import_patch.setattr(builtins, "__import__", import_without_runtime)
        spec = importlib.util.spec_from_file_location(
            "standalone_layerwise_prefill_profile_balance", tools_dir / "layerwise_prefill_profile.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def balance_patch_imports(enabled):
    source = ROOT / "vllm_ascend" / "patch" / "platform" / "__init__.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    gate = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "envs.VLLM_ASCEND_BALANCE_SCHEDULING"
    )
    imported = []

    def record_import(name, *args, **kwargs):
        imported.append(name)
        return SimpleNamespace()

    namespace = {
        "envs": SimpleNamespace(VLLM_ASCEND_BALANCE_SCHEDULING=enabled),
        "__builtins__": {"__import__": record_import},
    }
    exec(compile(ast.Module(body=[gate], type_ignores=[]), str(source), "exec"), namespace)
    return imported


@pytest.mark.parametrize("case,prompt_tokens", [("80k_on", 80000), ("80k_off", 80000), ("10k_on", 10000)])
def test_single_dp_profile_does_not_install_balance_patch(profile_tool, monkeypatch, tmp_path, case, prompt_tokens):
    monkeypatch.setenv("VLLM_ASCEND_BALANCE_SCHEDULING", "1")
    args = SimpleNamespace(devices="0,1,2,3,4,5,6,7", cpu_cache_gb=24, model="/models/test")
    options = profile_tool.engine_options(args, tmp_path, prompt_tokens)
    env = profile_tool.case_environment(args, case)
    assert options["tensor_parallel_size"] == 8
    assert options["data_parallel_size"] == 1
    assert options["max_num_seqs"] == 1
    assert options["async_scheduling"] is None
    assert env["VLLM_ASCEND_BALANCE_SCHEDULING"] == "0"
    assert balance_patch_imports(bool(int(env["VLLM_ASCEND_BALANCE_SCHEDULING"]))) == []
    assert balance_patch_imports(True) == ["vllm_ascend.patch.platform.patch_balance_schedule"]
