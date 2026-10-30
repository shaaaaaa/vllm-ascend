# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for real-batch membership diagnostics."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("batch_debug_test", ROOT / "vllm_ascend/prefill_batch_debug.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def observe(ids, starts, prompts, counts, ubatches=None):
    lines = []
    probe = M.PrefillBatchLogger(lines.append, dp_rank=2, pp_rank=0)
    probe.identity = "host:99"
    batch = NS(req_ids=ids, num_reqs=len(ids), num_computed_tokens_cpu=starts, num_prompt_tokens=prompts)
    probe.log(batch, NS(num_scheduled_tokens=counts), ubatches)
    return lines, probe, batch


def test_mixed_batch_tracks_request_local_ranges_and_flat_offsets():
    lines, _, _ = observe(["A", "B"], [20000, 10000], [30000, 20000], {"A": 2000, "B": 2096})
    assert len(lines) == 3
    assert "q=4096" in lines[0] and "ubatch=0:4096" in lines[0]
    assert all("b=host:99:1" in line and "n=2 mixed=1" in line for line in lines)
    assert "req='A' span=[20000,22000) prompt=30000 flat=[0,2000) mode=prefill" in lines[1]
    assert "req='B' span=[10000,12096) prompt=20000 flat=[2000,4096) mode=prefill" in lines[2]


def test_uses_actual_batch_order_not_scheduler_dictionary_order():
    lines, _, _ = observe(["B", "A"], [5, 99], [10, 100], {"A": 1, "B": 4})
    assert "i=0 req='B' span=[5,9)" in lines[1]
    assert "i=1 req='A' span=[99,100)" in lines[2]
    assert "last_prefill=1" in lines[2]


@pytest.mark.parametrize(
    "start,prompt,count,mode,last",
    [
        (499712, 500088, 376, "prefill", 1),
        (499999, 500000, 1, "prefill", 1),
        (500000, 500000, 2, "decode", 0),
        (499999, 500000, 2, "mixed", 1),
    ],
)
def test_short_suffix_and_single_token_prefill(start, prompt, count, mode, last):
    lines, _, _ = observe(["full-server-id"], [start], [prompt], {"full-server-id": count})
    assert all("n=1 mixed=0" in line for line in lines)
    assert f"mode={mode} last_prefill={last}" in lines[1]
    assert f"span=[{start},{start + count})" in lines[1]


def test_each_submission_has_new_id_and_escaped_request_id():
    request = "req\nnewline"
    lines, probe, batch = observe([request], [0], [10], {request: 5})
    batch.num_computed_tokens_cpu[0] = 5
    probe.log(batch, NS(num_scheduled_tokens={request: 5}))
    assert "b=host:99:2" in lines[-1]
    assert "span=[5,10)" in lines[-1]
    assert "\\n" in lines[-1] and "\n" not in lines[-1]


def test_microbatch_boundaries_are_visible():
    ubatches = [NS(token_slice=slice(0, 2048)), NS(token_slice=slice(2048, 4096))]
    lines, _, _ = observe(["A", "B"], [0, 0], [5000, 5000], {"A": 2000, "B": 2096}, ubatches)
    assert "ubatch=0:2048,2048:4096" in lines[0]


def test_malformed_metadata_logs_error_without_aborting_inference():
    lines, _, _ = observe(["missing"], [0], [10], {})
    assert len(lines) == 1 and "error=" in lines[0]


def runner_ast():
    tree = ast.parse((ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")


@pytest.mark.parametrize("enabled,rank,created", [(False, 0, False), (True, 3, False), (True, 0, True)])
def test_real_initialization_is_opt_in_and_tp0_only(monkeypatch, enabled, rank, created):
    init = next(n for n in runner_ast().body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    index = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Attribute)
        and n.targets[0].attr == "_prefill_batch_logger"
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.prefill_batch_debug", M)
    calls = []
    instance = NS(dp_rank=2)

    def tp():
        calls.append("tp")
        return NS(rank_in_group=rank)

    scope = dict(
        self=instance,
        envs_ascend=NS(VLLM_ASCEND_PREFILL_BATCH_DEBUG=enabled),
        get_tp_group=tp,
        get_pp_group=lambda: NS(rank_in_group=0),
        logger=NS(info=lambda _: None),
    )
    exec(compile(ast.Module(body=init.body[index : index + 2], type_ignores=[]), "init", "exec"), scope)
    assert (instance._prefill_batch_logger is not None) == created
    if not enabled:
        assert not calls


def test_real_submission_hook_follows_input_preparation_and_covers_both_forward_routes():
    execute = next(n for n in runner_ast().body if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    calls = [n for n in ast.walk(execute) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    hooks = [
        n
        for n in calls
        if n.func.attr == "log"
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "_prefill_batch_logger"
    ]
    assert len(hooks) == 1
    hook = hooks[0]
    assert hook.lineno > max(n.lineno for n in calls if n.func.attr == "_prepare_inputs")
    assert ast.unparse(hook.args[0]) == "self.input_batch"
    assert ast.unparse(hook.args[1]) == "scheduler_output"
    assert hook.lineno < min(n.lineno for n in calls if n.func.attr in ("_model_forward", "_run_cold_perf_npu_stage"))
