# SPDX-License-Identifier: Apache-2.0
"""Execute the Ascend scheduling loops with a recording KV allocator."""

import ast
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "boundaries,valid", [([0] * 8, True), ([4096] * 2, True), ([4096] * 3, False), ([2048], False)]
)
def test_native_scratch_bounds_count_only_external_rows(boundaries, valid):
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_dsa_scratch_capacity")
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    ns = {"np": np}
    exec(compile(module, str(path), "exec"), ns)
    with nullcontext() if valid else pytest.raises(RuntimeError, match="scratch"):
        ns[method.name](boundaries, [0] * len(boundaries), None, 2048, 4096)


class Queue(list):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.pop(0)

    def prepend_request(self, request):
        self.insert(0, request)

    def prepend_requests(self, requests):
        self[:0] = requests


class Request(NS):
    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)


def request(name, computed=131614, end=132325, compact=True, status="RUNNING"):
    return Request(
        request_id=name, num_computed_tokens=computed, num_tokens=end,
        num_prompt_tokens=131614, num_output_tokens=end - 131614,
        num_output_placeholders=0, max_tokens=3000, spec_token_ids=[],
        has_encoder_inputs=False, dsa_compact_allocated=compact, status=status,
        num_preemptions=1, num_cached_tokens=-1, lora_request=None, use_structured_output=False,
    )


def scheduler(running=(), waiting=(), width=2, budget=32):
    path = ROOT / "vllm_ascend/core/recompute_scheduler.py"
    cls = next(n for n in ast.parse(path.read_text(encoding="utf8")).body
               if isinstance(n, ast.ClassDef) and n.name == "RecomputeScheduler")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    def output(**kwargs):
        return NS(bootstrap_sample_req_ids=None, has_structured_output_requests=False,
                  pending_structured_output_tokens=False, **kwargs)

    ns = dict(time=time, cold_perf_enabled=lambda: False, RecomputeSchedulerOutput=output,
              record_function_or_nullcontext=lambda _: nullcontext(),
              PauseState=NS(PAUSED_ALL="paused", UNPAUSED="unpaused"),
              RequestStatus=NS(WAITING="WAITING", PREEMPTED="PREEMPTED", RUNNING="RUNNING",
                               WAITING_FOR_REMOTE_KVS="remote"),
              create_request_queue=lambda _: Queue(),
              NewRequestData=NS(from_request=lambda req, blocks: req), PLACEHOLDER_TOKEN_ID=-1)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(path), "exec"), ns)
    allocations = []
    blocks = NS(get_block_ids=lambda: ([], []))
    manager = NS(
        new_step_starts=lambda: None, empty_kv_cache_blocks=blocks,
        get_num_common_prefix_blocks=lambda _: [0, 0], get_blocks=lambda _: blocks,
        take_new_block_ids=lambda: [],
    )

    def allocate(req, count, **kwargs):
        allocations.append((req.request_id, req.num_computed_tokens, count))
        if kwargs.get("dsa_compact_external_load"):
            req.dsa_compact_allocated = True
        return blocks

    manager.allocate_slots = allocate
    obj = NS(
        running=list(running), waiting=Queue(waiting), skipped_waiting=Queue(),
        max_num_scheduled_tokens=budget, max_num_encoder_input_tokens=0,
        _pause_state="unpaused", _dsa_query_rows=width, max_model_len=140000,
        scheduler_config=NS(long_prefill_token_threshold=0, enable_chunked_prefill=True),
        need_mamba_block_aligned_split=False, num_lookahead_tokens=width - 1,
        num_spec_tokens=width - 1, kv_cache_manager=manager, max_num_running_reqs=16,
        lora_config=None, policy=None, is_mtp_kv_consumer=True, log_stats=False,
        _checkpoint_capture_enabled=False, connector=None, ec_connector=None,
        use_v2_model_runner=False, kv_cache_config=NS(kv_cache_groups=[0, 1]),
        is_encoder_decoder=False, prev_step_scheduled_req_ids=set(), needs_kv_cache_zeroing=False,
        encoder_cache_manager=NS(get_freed_mm_hashes=lambda: []), finished_req_ids=set(),
        _make_cached_request_data=lambda *args: NS(),
        _is_blocked_waiting_status=lambda _: False,
    )
    obj._select_waiting_queue_for_scheduling = lambda: obj.waiting

    def advance(output):
        for req in obj.running:
            req.num_computed_tokens += output.num_scheduled_tokens.get(req.request_id, 0)

    obj._update_after_schedule = advance
    obj.schedule = lambda: ns["schedule"](obj)
    return obj, allocations


@pytest.mark.parametrize("queue", ["running", "waiting"])
def test_reported_22_row_resume_is_capped_before_allocating(queue):
    live = [request(str(i), end=131615) for i in range(5)]
    for req in live:
        req.spec_token_ids = [-1]
    resumed = request("resumed", status="PREEMPTED" if queue == "waiting" else "RUNNING")
    running = live + ([resumed] if queue == "running" else [])
    obj, allocated = scheduler(running, [resumed] if queue == "waiting" else [])
    output = obj.schedule()
    assert output.num_scheduled_tokens == {**dict.fromkeys(map(str, range(5)), 2), "resumed": 2}
    assert output.total_num_scheduled_tokens == 12
    assert output.scheduled_spec_decode_tokens == dict.fromkeys(map(str, range(5)), [-1])
    assert allocated[-1] == ("resumed", 131614, 2)
    assert resumed.num_computed_tokens == 131616


@pytest.mark.parametrize("width", [1, 2])
def test_entire_missing_history_progresses_without_skips_or_overshoot(width):
    req = request("resumed")
    obj, allocations = scheduler(waiting=[req], width=width)
    req.status = "PREEMPTED"
    while req.num_computed_tokens < req.num_tokens:
        output = obj.schedule()
        assert 0 < output.num_scheduled_tokens[req.request_id] <= width
    assert sum(n for _, _, n in allocations) == 711
    assert req.num_computed_tokens == 132325
    assert all(a[1] + a[2] == b[1] for a, b in zip(allocations, allocations[1:]))
    req.num_tokens += 1
    req.spec_token_ids = [-1] if width == 2 else []
    assert obj.schedule().num_scheduled_tokens == {"resumed": width}


@pytest.mark.parametrize("queue", ["running", "waiting"])
@pytest.mark.parametrize("compact,remaining,budget,expected", [
    (False, 711, 32, 32), (True, 711, 1, 1), (True, 1, 32, 1),
])
def test_dense_recovery_and_smaller_existing_limits_are_preserved(queue, compact, remaining, budget, expected):
    req = request("r", end=131614 + remaining, compact=compact,
                  status="PREEMPTED" if queue == "waiting" else "RUNNING")
    obj, _ = scheduler([req] if queue == "running" else [], [req] if queue == "waiting" else [], budget=budget)
    assert obj.schedule().num_scheduled_tokens == {"r": expected}


def test_checkpoint_full_hit_preserves_target_and_draft_admission():
    req = request("r", computed=132324, status="PREEMPTED")
    req.kv_resume_checkpoint = (req.num_preemptions, req.num_tokens, req.num_computed_tokens)
    obj, allocated = scheduler(waiting=[req])
    obj._checkpoint_capture_enabled = True
    output = obj.schedule()
    assert output.num_scheduled_tokens == {"r": 2}
    assert output.scheduled_spec_decode_tokens == {"r": [-1]}
    assert allocated == [("r", 132324, 2)]


def test_async_restore_reserves_no_query_work_then_applies_sparse_budget():
    req = request("r", computed=0, compact=False, status="PREEMPTED")
    obj, allocated = scheduler(waiting=[req])
    obj.connector_prefix_cache_stats = None
    obj.kv_cache_manager.get_computed_blocks = lambda _: (
        obj.kv_cache_manager.empty_kv_cache_blocks, 0,
    )
    obj.connector = NS(
        supports_dsa_compact_external_load=True,
        get_num_new_matched_tokens=lambda *_: (131614, True),
        update_state_after_alloc=lambda *args: None,
        build_connector_meta=lambda _: NS(),
    )
    assert obj.schedule().num_scheduled_tokens == {}
    assert req.status == "remote" and req.dsa_compact_allocated
    assert allocated == [("r", 0, 0)]
    # The connector has completed; scheduler promotion re-admits the request.
    obj.skipped_waiting.clear()
    req.status = "PREEMPTED"
    obj.waiting.append(req)
    assert obj.schedule().num_scheduled_tokens == {"r": 2}
    assert allocated[-1] == ("r", 131614, 2)


def test_preemption_drops_old_compact_allocation_before_dense_cache_miss():
    req = request("r")
    req.kv_resume_checkpoint = (1, 132325, 131614)
    obj, allocations = scheduler()
    freed = []
    obj.kv_cache_manager.free = lambda r: freed.append(r.request_id)
    obj.encoder_cache_manager.free = lambda _: None
    obj.kv_cache_manager.get_computed_blocks = lambda _: (
        obj.kv_cache_manager.empty_kv_cache_blocks, 0,
    )
    namespace = dict(RequestStatus=NS(RUNNING="RUNNING", PREEMPTED="PREEMPTED"))
    classes = []
    for path, name, bases in (
        (ROOT.parent / "vllm/vllm/v1/core/sched/scheduler.py", "Base", []),
        (ROOT / "vllm_ascend/core/recompute_scheduler.py", "Derived", [ast.Name(id="Base", ctx=ast.Load())]),
    ):
        tree = ast.parse(path.read_text(encoding="utf8"))
        method = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "_preempt_request")
        classes.append(ast.ClassDef(name=name, bases=bases, keywords=[], body=[method], decorator_list=[]))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *classes], type_ignores=[])),
                 "preemption_chain", "exec"), namespace)
    derived = namespace["Derived"]()
    derived.__dict__.update(vars(obj))
    derived._preempt_request(req, 0.0)
    assert freed == ["r"] and req.num_computed_tokens == 0
    assert req.kv_resume_checkpoint is None
    # No external hit: reallocate densely, without the old two-row restriction.
    assert obj.schedule().num_scheduled_tokens == {"r": 32}
    assert not req.dsa_compact_allocated
    assert allocations == [("r", 0, 32)]


def test_actual_async_counters_do_not_generate_tokens_during_history_replay():
    req = request("r", status="PREEMPTED")
    obj, _ = scheduler(waiting=[req])
    classes = []
    for filename, name, bases in (
        ("scheduler.py", "Base", []),
        ("async_scheduler.py", "Async", [ast.Name(id="Base", ctx=ast.Load())]),
    ):
        path = ROOT.parent / "vllm/vllm/v1/core/sched" / filename
        tree = ast.parse(path.read_text(encoding="utf8"))
        method = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "_update_after_schedule")
        classes.append(ast.ClassDef(name=name, bases=bases, keywords=[], body=[method], decorator_list=[]))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *classes], type_ignores=[])),
                 "async_counter_chain", "exec"), namespace)
    updater = namespace["Async"]()
    updater.requests = {"r": req}
    updater._spec_token_placeholders = [-1]
    obj._update_after_schedule = updater._update_after_schedule
    while req.num_computed_tokens < req.num_tokens:
        output = obj.schedule()
        assert not output.scheduled_spec_decode_tokens
        if req.num_computed_tokens < req.num_tokens:
            assert req.is_prefill_chunk
            assert req.num_output_placeholders == 0 and req.spec_token_ids == []
    assert not req.is_prefill_chunk
    assert req.num_output_placeholders == 1 and req.spec_token_ids == [-1]
    output = obj.schedule()
    assert output.num_scheduled_tokens == {"r": 2}
    assert output.scheduled_spec_decode_tokens == {"r": [-1]}
    assert req.num_output_placeholders == 3


@pytest.mark.parametrize("drafts", [0, 1, 3])
def test_startup_uses_allocator_scratch_row_contract(drafts):
    path = ROOT / "vllm_ascend/core/recompute_scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Attribute) and t.attr == "_dsa_query_rows" for t in n.targets))
    obj = NS(kv_cache_config=NS(dsa_num_speculative_tokens=drafts))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), {"self": obj})
    assert obj._dsa_query_rows == drafts + 1
