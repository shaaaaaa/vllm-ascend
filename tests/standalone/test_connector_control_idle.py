# SPDX-License-Identifier: Apache-2.0
"""Idle connector cleanup must not add a check to active scheduling."""

import ast
from abc import abstractmethod
from pathlib import Path
import time
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]


def method(file, name):
    tree = ast.parse(file.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ns = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[prefix, node], type_ignores=[])),
            str(file),
            "exec",
        ),
        ns,
    )
    return ns[name]


def test_active_scheduling_never_queries_connector_control():
    has_requests = method(ROOT / "vllm_ascend/core/recompute_scheduler.py", "has_requests")
    obj = NS(
        has_unfinished_requests=lambda: True,
        has_finished_requests=lambda: pytest.fail("active path consulted cleanup"),
        connector=NS(has_pending_control=lambda: pytest.fail("active path queried connector")),
    )
    for _ in range(100):
        assert has_requests(obj)


def test_idle_work_follows_the_actual_multi_connector_release_queue():
    root = ROOT.parent
    impl_pending = method(
        root / "LMCache-NPU/lmcache/integration/vllm/vllm_v1_adapter.py",
        "has_pending_control",
    )
    dynamic_pending = method(
        root / "LMCache-NPU/lmcache/integration/vllm/lmcache_connector_v1.py",
        "has_pending_control",
    )
    multi_pending = method(
        root / "vllm-ascend/vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py",
        "has_pending_control",
    )
    has_requests = method(ROOT / "vllm_ascend/core/recompute_scheduler.py", "has_requests")
    impl = NS(_checkpoint_restore_releases=[("r", 1, 7)])
    impl.has_pending_control = lambda: impl_pending(impl)
    child = NS(_lmcache_engine=impl)
    child.has_pending_control = lambda: dynamic_pending(child)
    multi = NS(_connectors=[object(), child])
    multi.has_pending_control = lambda: multi_pending(multi)
    scheduler = NS(
        connector=multi,
        has_unfinished_requests=lambda: False,
        has_finished_requests=lambda: False,
    )
    assert has_requests(scheduler)
    impl._checkpoint_restore_releases.clear()
    assert not has_requests(scheduler)


def test_idle_scheduler_without_connector_remains_idle():
    has_requests = method(ROOT / "vllm_ascend/core/recompute_scheduler.py", "has_requests")
    assert not has_requests(
        NS(
            connector=None,
            has_unfinished_requests=lambda: False,
            has_finished_requests=lambda: False,
        )
    )


@pytest.fixture(params=["RecomputeScheduler", "AsyncRecomputeScheduler"])
def scheduler_type(request):
    """Retain actual class bases and the migrated methods across the async MRO."""
    vllm = ROOT.parent / "vllm/vllm/v1/core/sched"
    sources = [
        (vllm / "interface.py", {"SchedulerInterface"}),
        (vllm / "scheduler.py", {"Scheduler"}),
        (vllm / "async_scheduler.py", {"AsyncScheduler"}),
        (ROOT / "vllm_ascend/core/recompute_scheduler.py", {"RecomputeScheduler", "AsyncRecomputeScheduler"}),
    ]
    nodes = []
    for path, names in sources:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and node.name in names:
                node.body = [
                    n
                    for n in node.body
                    if getattr(n, "name", None)
                    in {"has_requests", "_preempt_request", "reset_prefix_cache", "_update_waiting_for_remote_kv"}
                    or (isinstance(n, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "supports_checkpoint_restore_retry"
                        for t in n.targets
                    ))
                ] or [ast.Pass()]
                nodes.append(node)
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ns = dict(
        ABC=object, abstractmethod=abstractmethod, RequestStatus=NS(RUNNING="running", PREEMPTED="preempted"), time=time
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[prefix, *nodes], type_ignores=[])), "scheduler_mro", "exec"),
        ns,
    )
    cls = ns[request.param]
    assert cls.has_requests is ns["RecomputeScheduler"].has_requests
    assert cls._preempt_request is ns["RecomputeScheduler"]._preempt_request
    assert cls._update_waiting_for_remote_kv is ns["RecomputeScheduler"]._update_waiting_for_remote_kv
    assert cls.supports_checkpoint_restore_retry is True
    return cls


@pytest.mark.parametrize("unfinished,finished", [(True, False), (False, True)])
def test_both_ascend_schedulers_short_circuit_before_connector_access(scheduler_type, unfinished, finished):
    scheduler = scheduler_type()
    scheduler.has_unfinished_requests = lambda: unfinished
    scheduler.has_finished_requests = lambda: finished
    # No connector attribute: even fetching it on this path would fail.
    assert scheduler.has_requests()


def test_both_ascend_schedulers_dispatch_and_drain_idle_control(scheduler_type):
    scheduler = scheduler_type()
    scheduler.has_unfinished_requests = lambda: False
    scheduler.has_finished_requests = lambda: False
    pending = [("r", 1, 7)]
    scheduler.connector = NS(has_pending_control=lambda: bool(pending))
    assert scheduler.has_requests()
    pending.clear()
    assert not scheduler.has_requests()
    scheduler.connector = object()
    assert not scheduler.has_requests()


@pytest.mark.parametrize("forced_reset", [False, True])
@pytest.mark.parametrize("old_proof", [None, (1, 15, 14)])
def test_proof_is_lazy_and_cleared_before_base_preemption(scheduler_type, forced_reset, old_proof):
    scheduler = scheduler_type()
    request = NS(
        status="running", num_computed_tokens=15, num_preemptions=1, spec_token_ids=[-1], num_output_placeholders=2
    )
    if old_proof is not None:
        request.kv_resume_checkpoint = old_proof
    else:
        assert not hasattr(request, "kv_resume_checkpoint")
    calls, waiting = [], []

    def free(req):
        assert req.kv_resume_checkpoint is None
        assert req.num_computed_tokens == 15 and req.num_preemptions == 1
        calls.append("free")

    scheduler.kv_cache_manager = NS(free=free, reset_prefix_cache=lambda: True)
    scheduler.encoder_cache_manager = NS(free=lambda req: calls.append("encoder"))
    scheduler.waiting = NS(prepend_request=waiting.append)
    scheduler.log_stats = False
    if forced_reset:
        scheduler.running = [request]
        scheduler.prev_step_scheduled_req_ids = {"r"}
        assert scheduler.reset_prefix_cache(reset_running_requests=True)
        assert not scheduler.prev_step_scheduled_req_ids
        assert request.num_output_placeholders == 0
        assert request.discard_latest_async_tokens is True
    else:
        scheduler._preempt_request(request, 1.0)
    assert calls == ["free", "encoder"] and waiting == [request]
    assert request.status == "preempted" and request.num_computed_tokens == 0
    assert request.num_preemptions == 2 and request.spec_token_ids == []


def test_invalid_preemption_does_not_modify_request(scheduler_type):
    scheduler = scheduler_type()
    request = NS(status="waiting", kv_resume_checkpoint=(1, 15, 14))
    with pytest.raises(AssertionError, match="Only running"):
        scheduler._preempt_request(request, 1.0)
    assert request.kv_resume_checkpoint == (1, 15, 14)


def test_base_scheduler_no_longer_handles_checkpoint_controls():
    path = ROOT.parent / "vllm/vllm/v1/core/sched/interface.py"
    has_requests = method(path, "has_requests")
    obj = NS(
        has_unfinished_requests=lambda: False,
        has_finished_requests=lambda: False,
        connector=NS(has_pending_control=lambda: pytest.fail("base scheduler queried checkpoint control")),
    )
    assert not has_requests(obj)
