# SPDX-License-Identifier: Apache-2.0
"""Ascend schedulers must share vLLM's P-only request metadata builder."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "source",
    [
        "core/recompute_scheduler.py",
        "core/scheduler_dynamic_batch.py",
        "patch/platform/patch_balance_schedule.py",
    ],
)
def test_scheduler_uses_shared_new_request_builder(source):
    path = Path(__file__).resolve().parents[2] / "vllm_ascend" / source
    tree = ast.parse(path.read_text(encoding="utf-8"))
    schedules = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "schedule"]
    assert schedules
    for schedule in schedules:
        calls = [n.func for n in ast.walk(schedule) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        assert any(n.attr == "_make_new_request_data" for n in calls)
        assert not any(
            n.attr in {"get_allocation_mode", "get_block_ids_by_bank"}
            or (n.attr == "from_request" and isinstance(n.value, ast.Name) and n.value.id == "NewRequestData")
            for n in calls
        )
