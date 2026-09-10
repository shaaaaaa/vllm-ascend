# SPDX-License-Identifier: Apache-2.0
"""Preemption must not bind side-effecting next-step metadata on other children."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS, MethodType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def compile_function(path, name, ns, *, prefix_only=False):
    source = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    if prefix_only:
        # Run the actual execute_model prefix up through preemption handling,
        # then stop before input preparation needs the accelerator runtime.
        stop = next(
            i
            for i, n in enumerate(function.body)
            if isinstance(n, ast.If) and "preempted_req_ids" in ast.unparse(n.test)
        )
        function.body = function.body[: stop + 1]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), str(path), "exec"), ns
    )
    return ns[name]


def test_plain_preemption_does_not_bind_or_clear_new_metadata():
    calls = []
    connector = NS(
        handle_preemptions=lambda ids: calls.append(ids),
        bind_connector_metadata=lambda meta: pytest.fail("bound next-step metadata early"),
        clear_connector_metadata=lambda: pytest.fail("cleared unrelated metadata"),
    )
    execute = compile_function(
        ROOT / "vllm_ascend/worker/model_runner_v1.py",
        "execute_model",
        {"has_kv_transfer_group": lambda: True, "get_kv_transfer_group": lambda: connector},
        prefix_only=True,
    )
    runner = NS(vllm_config=NS(model_config=NS(enable_return_routed_experts=False)), execute_model_state=None)
    execute(runner, NS(preempted_req_ids={"r"}, kv_connector_metadata="new-step"))
    assert calls == [{"r"}]


def test_ordinary_execution_does_not_touch_checkpoint_connector():
    execute = compile_function(
        ROOT / "vllm_ascend/worker/model_runner_v1.py",
        "execute_model",
        {"has_kv_transfer_group": lambda: pytest.fail("connector checked during ordinary decode")},
        prefix_only=True,
    )
    runner = NS(vllm_config=NS(model_config=NS(enable_return_routed_experts=False)), execute_model_state=None)
    execute(runner, NS(preempted_req_ids=set()))


def test_multi_connector_scopes_checkpoint_metadata_to_the_checkpoint_child():
    class Metadata:
        metadata = ("checkpoint-meta", "other-meta")

    calls = []
    checkpoint = NS(
        supports_preemption_checkpoint=True,
        handle_preemptions_with_metadata=lambda ids, meta: calls.append(("checkpoint", meta, ids)),
    )
    other = NS(
        handle_preemptions=lambda ids: calls.append(("other", ids)),
        bind_connector_metadata=lambda meta: pytest.fail("bound unrelated child metadata"),
    )
    dispatch = compile_function(
        ROOT / "vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py",
        "handle_preemptions_with_metadata",
        {"MultiKVConnectorMetadata": Metadata},
    )
    dispatch(NS(_connectors=[checkpoint, other]), {"r"}, Metadata())
    assert calls == [("checkpoint", "checkpoint-meta", {"r"}), ("other", {"r"})]


def test_queued_model_call_cannot_continue_after_preemption_failure():
    def fail(ids):
        raise RuntimeError("uncertain capture fence")

    connector = NS(handle_preemptions=fail)
    execute = compile_function(
        ROOT / "vllm_ascend/worker/model_runner_v1.py",
        "execute_model",
        {"has_kv_transfer_group": lambda: True, "get_kv_transfer_group": lambda: connector},
        prefix_only=True,
    )
    runner = NS(vllm_config=NS(model_config=NS(enable_return_routed_experts=False)), execute_model_state=None)
    reject = compile_function(
        ROOT / "vllm_ascend/worker/model_runner_v1.py", "_reject_work_after_preemption_failure", {}
    )
    runner._reject_work_after_preemption_failure = MethodType(reject, runner)
    runner.execute_model = MethodType(execute, runner)
    with pytest.raises(RuntimeError, match="uncertain capture"):
        execute(runner, NS(preempted_req_ids={"r"}, kv_connector_metadata=None))
    with pytest.raises(RuntimeError, match="prior preemption"):
        runner.execute_model(NS(preempted_req_ids=set(), kv_connector_metadata=None))
