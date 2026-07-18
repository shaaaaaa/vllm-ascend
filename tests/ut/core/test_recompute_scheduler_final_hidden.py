from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler


def test_bootstrap_remote_full_hit_keeps_all_prompt_tokens():
    scheduler = object.__new__(RecomputeScheduler)
    scheduler.connector = object()
    scheduler.block_size = 16
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = {"request"}
    cached = []
    scheduler.kv_cache_manager = SimpleNamespace(
        get_block_ids=lambda req_id: ([1, 2], [11, 12]),
        cache_blocks=lambda request, num_tokens: cached.append(num_tokens),
    )
    request = SimpleNamespace(
        request_id="request",
        num_tokens=32,
        num_prompt_tokens=32,
        num_computed_tokens=0,
        num_cached_tokens=-1,
        bootstrap_sample_pending=True,
        dsa_compact_allocated=True,
    )

    scheduler._update_waiting_for_remote_kv(request)

    assert request.num_computed_tokens == 32
    assert request.num_cached_tokens == 32
    assert cached == [32]
    assert not scheduler.finished_recving_kv_req_ids


def test_normal_remote_full_hit_still_recomputes_last_prompt_token():
    scheduler = object.__new__(RecomputeScheduler)
    scheduler.connector = object()
    scheduler.block_size = 16
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = {"request"}
    scheduler.kv_cache_manager = SimpleNamespace(
        get_block_ids=lambda req_id: ([1, 2], [11, 12]),
        cache_blocks=lambda request, num_tokens: None,
    )
    request = SimpleNamespace(
        request_id="request",
        num_tokens=32,
        num_prompt_tokens=32,
        num_computed_tokens=0,
        num_cached_tokens=-1,
        bootstrap_sample_pending=False,
        dsa_compact_allocated=False,
    )

    scheduler._update_waiting_for_remote_kv(request)

    assert request.num_computed_tokens == 31


def test_bootstrap_load_failure_rolls_back_async_placeholders(monkeypatch):
    scheduler = object.__new__(RecomputeScheduler)
    request = SimpleNamespace(
        request_id="bootstrap",
        bootstrap_sample_pending=True,
        num_output_placeholders=2,
        spec_token_ids=[-1],
        num_computed_tokens=0,
        num_cached_tokens=32,
        num_external_computed_tokens=32,
        dsa_compact_allocated=False,
        status=RequestStatus.RUNNING,
        num_preemptions=0,
    )
    scheduler.running = [request]
    scheduler.skipped_waiting = []
    scheduler.requests = {request.request_id: request}
    scheduler.kv_cache_manager = SimpleNamespace(
        get_block_ids=lambda req_id: ([1, 2], [11, 12])
    )
    waiting = []
    scheduler.waiting = SimpleNamespace(prepend_request=waiting.append)
    scheduler.prev_step_scheduled_req_ids = {request.request_id}
    scheduler.log_stats = False
    monkeypatch.setattr(
        Scheduler,
        "_handle_invalid_blocks",
        lambda self, invalid_block_ids: {"bootstrap"},
    )

    affected = scheduler._handle_invalid_blocks({123})

    assert affected == {"bootstrap"}
    assert request.num_output_placeholders == 0
    assert request.spec_token_ids == []
    assert request.status == RequestStatus.PREEMPTED
    assert scheduler.running == []
    assert waiting == [request]
    assert not scheduler.prev_step_scheduled_req_ids


def test_normal_load_failure_does_not_touch_async_placeholders(monkeypatch):
    scheduler = object.__new__(RecomputeScheduler)
    request = SimpleNamespace(
        request_id="normal",
        bootstrap_sample_pending=False,
        num_output_placeholders=2,
        spec_token_ids=[-1],
    )
    scheduler.running = [request]
    scheduler.skipped_waiting = []
    scheduler.requests = {request.request_id: request}
    scheduler.waiting = SimpleNamespace(
        prepend_request=lambda request: (_ for _ in ()).throw(
            AssertionError("normal request was requeued")
        )
    )
    scheduler.prev_step_scheduled_req_ids = {request.request_id}
    scheduler.log_stats = False
    monkeypatch.setattr(
        Scheduler,
        "_handle_invalid_blocks",
        lambda self, invalid_block_ids: {"normal"},
    )

    scheduler._handle_invalid_blocks({123})

    assert request.num_output_placeholders == 2
    assert request.spec_token_ids == [-1]


def test_bootstrap_async_barrier_only_suppresses_inflight_schedule(monkeypatch):
    scheduler = object.__new__(RecomputeScheduler)
    monkeypatch.setattr(Scheduler, "has_requests", lambda self: True)

    assert scheduler.has_requests()

    scheduler._bootstrap_async_barrier_req_ids = frozenset({"bootstrap"})
    assert not scheduler.has_requests()

    del scheduler._bootstrap_async_barrier_req_ids
    assert scheduler.has_requests()
