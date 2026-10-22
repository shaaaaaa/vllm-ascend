# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-order checks; native worker execution requires NPU qualification."""

import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


def test_storage_registration_precedes_the_initial_memory_snapshot():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    init = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "_init_device")
    calls = [node for node in ast.walk(init) if isinstance(node, ast.Call)]
    bootstrap = next(
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "early_bootstrap_decoder_storage"
    )
    snapshot = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "MemorySnapshot")
    assert bootstrap.lineno < snapshot.lineno


@pytest.mark.parametrize("storage", ["attached", "local", "absent"])
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("enabled", [False, True])
def test_storage_slab_keeps_numa_policy_without_losing_cpu_or_irq_affinity(storage, rank, enabled):
    """Execute binding and worker routing with only OS/native boundaries mocked."""
    root = Path(__file__).resolve().parents[3]
    source = ast.parse((root / "vllm_ascend/cpu_binding.py").read_text(encoding="utf-8"))
    definitions = [node for node in source.body if getattr(node, "name", None) in {"CpuAlloc", "bind_cpus"}]
    command = Mock(return_value=("threads", 0))
    namespace = dict(
        defaultdict=defaultdict,
        DeviceInfo=lambda: NS(running_npu_list=[4, 5, 6, 7]),
        logger=Mock(),
        is_arm_cpu=lambda: True,
        execute_command=command,
        psutil=NS(Process=lambda: NS(pid=100)),
        shutil=NS(which=lambda name: name),
    )
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[prefix, *definitions], type_ignores=[])), "cpu_binding", "exec"
        ),
        namespace,
    )
    allocator = namespace["CpuAlloc"]
    instances = []

    def create(*args, **kwargs):
        instance = allocator(*args, **kwargs)
        instances.append(instance)
        npu = 4 + rank
        instance.npu_cpu_pool = {npu: [10, 11, 12, 13, 14]}
        instance.cpu_node = {10: rank}
        instance.numa_to_cpu_map = {i: [] for i in range(4)}
        instance.assign_main = {npu: [12]}
        instance.assign_acl = {npu: [13]}
        instance.assign_rel = {npu: [14]}
        for method in ("build_cpu_pools", "allocate", "print_plan", "bind", "bind_npu_irq"):
            setattr(instance, method, Mock())
        instance.get_threads_map = lambda text: {"100": {"acl_thread": ["101", "102"], "release_thread": ["103"]}}
        return instance

    namespace["CpuAlloc"] = create
    tree = ast.parse((root / "vllm_ascend/worker/worker.py").read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    init = next(node for node in worker.body if getattr(node, "name", None) == "_init_device")
    binding = next(
        node
        for node in init.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and node.test.attr == "enable_cpu_binding"
    )
    owner = NS(local_rank=rank)
    if storage != "absent":
        owner._decoder_storage_bootstrap = object() if storage == "attached" else None
    namespace.update(self=owner, get_ascend_config=lambda: NS(enable_cpu_binding=enabled))
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[binding], type_ignores=[])), "worker_binding", "exec"),
        namespace,
    )
    namespace["logger"].warning.assert_not_called()
    if not enabled:
        assert not instances
        command.assert_not_called()
        return
    (instance,) = instances
    assert instance.bind.call_args_list == [
        (("100", [12], True),),
        (("101", [13], False),),
        (("102", [13], False),),
        (("103", [14], False),),
    ]
    instance.bind_npu_irq.assert_called_once_with()
    expected = [["ps", "-Te"]]
    if storage != "attached":
        expected += [["migratepages", tid, "0,1,2,3", str(rank)] for tid in ("101", "102")]
    assert [call.args[0] for call in command.call_args_list] == expected
