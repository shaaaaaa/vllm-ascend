# SPDX-License-Identifier: Apache-2.0
"""CPU checks of test-worker teardown ordering, not native HCCL deinitialization."""

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
from test_sfa_parity import checkpoint as checkpoint
from test_sfa_parity import parity as parity
from test_sfa_parity import worker as worker


@pytest.fixture
def cleanup(worker, monkeypatch):
    events = []
    npu = SimpleNamespace(
        is_initialized=lambda: True,
        synchronize=lambda: events.append("sync"),
    )
    monkeypatch.setattr(worker.torch, "npu", npu, raising=False)
    connector = ModuleType("vllm.distributed.kv_transfer")
    connector.ensure_kv_transfer_shutdown = lambda: events.append("cache")
    ascend = ModuleType("vllm_ascend.distributed.parallel_state")
    ascend.destroy_ascend_model_parallel = lambda: events.append("ascend_groups")
    monkeypatch.setitem(sys.modules, connector.__name__, connector)
    monkeypatch.setitem(sys.modules, ascend.__name__, ascend)
    monkeypatch.setattr(worker.NPUWorker, "shutdown", lambda self: events.append("base"), raising=False)
    instance = object.__new__(worker.SFAParityWorker)
    instance.rank = 3
    return instance, npu, connector, events


def test_worker_drains_and_releases_once_before_base_shutdown(cleanup):
    instance, _, _, events = cleanup
    identity = {"rank": 3, "pid": os.getpid()}
    assert instance.parity_process_info() == identity
    assert events == []
    assert instance.parity_release_resources() == identity
    assert instance.parity_release_resources() == identity
    instance.shutdown()
    assert events == ["sync", "cache", "ascend_groups", "base"]


def test_worker_shutdown_also_cleans_up_without_prior_success_rpc(cleanup):
    instance, _, _, events = cleanup
    instance.shutdown()
    assert events == ["sync", "cache", "ascend_groups", "base"]


def test_uninitialized_npu_is_not_synchronized(cleanup):
    instance, npu, _, events = cleanup
    npu.is_initialized = lambda: False
    instance.shutdown()
    assert events == ["cache", "ascend_groups", "base"]


def test_cache_failure_still_destroys_ascend_groups_and_calls_base(cleanup):
    instance, _, connector, events = cleanup

    def fail():
        events.append("cache")
        raise RuntimeError("cache shutdown failed")

    connector.ensure_kv_transfer_shutdown = fail
    with pytest.raises(RuntimeError, match="cache shutdown failed"):
        instance.shutdown()
    assert events == ["sync", "cache", "ascend_groups", "base"]
    assert not getattr(instance, "_parity_resources_released", False)


def test_sync_failure_is_reported_and_base_shutdown_still_runs(cleanup):
    instance, npu, _, events = cleanup

    def fail():
        raise RuntimeError("NPU synchronization failed")

    npu.synchronize = fail
    with pytest.raises(RuntimeError, match="NPU synchronization failed"):
        instance.shutdown()
    assert events == ["base"]
    assert not getattr(instance, "_parity_resources_released", False)
