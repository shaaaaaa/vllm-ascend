# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-order checks; native worker execution requires NPU qualification."""

import ast
from pathlib import Path


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
