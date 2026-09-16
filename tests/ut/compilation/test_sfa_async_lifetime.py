# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ownership tests: real graph/lease code and LMCache reference counting.

NPU events/streams and the allocator are controlled fakes, not an NPU test.
"""

import ast
import importlib.util
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_sfa_full_graph import graph_module as graph_module


@dataclass(frozen=True)
class Key:
    request_capacity: int = 1


@pytest.fixture
def asynchronous(graph_module, monkeypatch):
    module, context, _, stream = graph_module
    root = Path(__file__).resolve().parents[3]
    memory_path = root.parent / "LMCache/lmcache/v1/memory_management.py"
    if not memory_path.exists():
        pytest.skip("Ownership contract needs the sibling LMCache checkout")
    # Use LMCache's actual locked increment/decrement/free logic, avoiding its
    # optional native imports and global monitor initialization on this host.
    tree = ast.parse(memory_path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TensorMemoryObj")
    methods = [node for node in cls.body if getattr(node, "name", "") in ("ref_count_up", "ref_count_down")]
    namespace = {"logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(memory_path), "exec"), namespace)

    class Owner:
        ref_count_up = namespace["ref_count_up"]
        ref_count_down = namespace["ref_count_down"]

        def __init__(self):
            self.lock = threading.Lock()
            self.meta = SimpleNamespace(ref_count=1, pin_count=0, address=0)
            self.freed = False
            self.parent_allocator = SimpleNamespace(free=self.free)

        def free(self, obj):
            assert obj is self and not self.freed
            self.freed = True

        def is_valid(self):
            return not self.freed

    memory = ModuleType("lmcache.v1.memory_management")
    memory.TensorMemoryObj = Owner
    monkeypatch.setitem(sys.modules, memory.__name__, memory)
    spec = importlib.util.spec_from_file_location(
        "tested_sfa_source_lifetime", root / "vllm_ascend/compilation/sfa_source_lifetime.py"
    )
    lifetime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifetime)
    monkeypatch.setattr(module, "SFASourceLease", lifetime.SFASourceLease)

    events = []

    class Event:
        def __init__(self):
            self.ready = False
            self.records = 0
            self.fail_record = self.fail_query = False
            events.append(self)

        def record(self, target_stream):
            assert target_stream is stream
            if self.fail_record:
                raise RuntimeError("record failed")
            self.records += 1
            self.ready = False

        def query(self):
            if self.fail_query:
                raise RuntimeError("query failed")
            return self.ready

    monkeypatch.setattr(torch.npu, "Event", Event)
    context.staged_sfa_graph_key = Key()
    graph = module.SFAFullGraph()
    graph.run(lambda: "output")
    context.staged_sfa_graph_dummy_run = False
    transfer = SimpleNamespace(bind_batch=Mock())

    def source(owner, *, layers=8):
        return SimpleNamespace(
            layers=tuple(
                SimpleNamespace(memory_objs=(owner,), chunk_ptrs_npu=torch.ones(1, dtype=torch.int64))
                for _ in range(layers)
            )
        )

    def bind(value, request="a"):
        return graph.bind_sources((value,), (request,), lambda: (transfer,))

    return SimpleNamespace(
        module=module,
        context=context,
        graph=graph,
        owner=Owner,
        source=source,
        bind=bind,
        transfer=transfer,
        stream=stream,
        events=events,
        lease=lifetime.SFASourceLease,
    )


def test_warm_replays_record_without_wait_query_or_retain_again(asynchronous):
    a = asynchronous
    owner = a.owner()
    value = a.source(owner)
    a.bind(value)
    assert owner.meta.ref_count == 2  # One owner shared by all eight layers.
    event = a.events[0]
    event.fail_query = True  # Active events must never be queried per replay.
    for _ in range(300):
        assert not a.bind(value)
        assert a.graph.run(Mock()) == "output"
    assert owner.meta.ref_count == 2
    assert event.records == 301 and len(a.events) == 1
    a.stream.synchronize.assert_not_called()
    torch.npu.synchronize.assert_not_called()


@pytest.mark.parametrize("retire", ["replace", "finish", "reuse_request_id"])
def test_cleanup_cannot_free_an_inflight_source(asynchronous, retire):
    a = asynchronous
    old = a.owner()
    a.bind(a.source(old))
    a.graph.run(Mock())
    old.ref_count_down()  # Simulate engine/request dropping its original ref.
    assert not old.freed
    event = a.events[0]
    if retire == "replace":
        a.bind(a.source(a.owner()))
    else:
        a.graph.release_requests({"a"})
        if retire == "reuse_request_id":
            a.bind(a.source(a.owner()), "a")
    a.graph.collect_retired_sources()
    assert not old.freed
    event.ready = True
    a.graph.collect_retired_sources()
    assert old.freed and not a.graph.retired_sources
    a.stream.synchronize.assert_not_called()
    torch.npu.synchronize.assert_not_called()


def test_latest_replay_event_not_previous_completion_controls_release(asynchronous):
    a = asynchronous
    owner = a.owner()
    a.bind(a.source(owner))
    a.graph.run(Mock())
    a.events[0].ready = True
    a.graph.run(Mock())
    assert not a.events[0].ready
    owner.ref_count_down()
    a.graph.release_requests({"a"})
    assert not owner.freed
    a.events[0].ready = True
    a.graph.collect_retired_sources()
    assert owner.freed


@pytest.mark.parametrize("failure", ["query", "record", "replay"])
def test_event_or_submission_failure_never_releases_inflight_memory(asynchronous, failure):
    a = asynchronous
    owner = a.owner()
    a.bind(a.source(owner))
    owner.ref_count_down()
    if failure == "query":
        a.events[0].fail_query = True
        with pytest.raises(RuntimeError, match="query failed"):
            a.graph.release_requests({"a"})
    else:
        if failure == "record":
            a.events[0].fail_record = True
        else:
            a.graph.entries[Key()].graph.replay.side_effect = RuntimeError("replay failed")
        with pytest.raises(RuntimeError, match="failed"):
            a.graph.run(Mock())
        with pytest.raises(RuntimeError, match="submission failed"):
            a.graph.prepare_run()
    assert not owner.freed
    a.graph.clear()  # Only lifecycle teardown may block to establish safety.
    torch.npu.synchronize.assert_called_once()
    assert owner.freed


def test_failed_binding_keeps_pointer_copy_sources_until_event(asynchronous):
    a = asynchronous
    owner = a.owner()
    a.transfer.bind_batch.side_effect = RuntimeError("partly copied pointers")
    with pytest.raises(RuntimeError, match="partly copied"):
        a.bind(a.source(owner))
    owner.ref_count_down()
    assert not owner.freed
    a.graph.collect_retired_sources()
    assert not owner.freed
    a.events[0].ready = True
    a.graph.collect_retired_sources()
    assert owner.freed


def test_completed_retirements_are_reaped_during_unchanged_binding(asynchronous):
    a = asynchronous
    old, new = a.owner(), a.owner()
    a.bind(a.source(old))
    old.ref_count_down()
    value = a.source(new)
    a.bind(value)
    assert not old.freed
    a.events[0].ready = True
    assert not a.bind(value)
    assert old.freed and new.meta.ref_count == 2


def test_backlog_is_bounded_without_cpu_wait_or_unsafe_free(asynchronous, monkeypatch):
    a = asynchronous
    monkeypatch.setattr(a.module, "MAX_PENDING_SOURCE_RETIREMENTS", 2)
    owners = [a.owner() for _ in range(4)]
    for owner in owners[:3]:
        a.bind(a.source(owner))
        owner.ref_count_down()
    with pytest.raises(RuntimeError, match="unbounded retention"):
        a.bind(a.source(owners[3]))
    assert len(a.graph.retired_sources) == 2 and not any(o.freed for o in owners)
    torch.npu.synchronize.assert_not_called()


@pytest.mark.parametrize("invalid", ["tensor_only", "proxy", "freed"])
def test_unowned_invalid_or_noop_proxy_source_fails_before_pointer_upload(asynchronous, invalid):
    a = asynchronous
    owner = a.owner()
    value = a.source(owner)
    if invalid == "tensor_only":
        value.layers[0].memory_objs = ()
    elif invalid == "proxy":
        value.layers[0].memory_objs = (SimpleNamespace(ref_count_up=Mock()),)
    else:
        owner.ref_count_down()
    with pytest.raises(RuntimeError):
        a.bind(value)
    a.transfer.bind_batch.assert_not_called()


def test_cross_stream_replay_is_rejected_before_submission(asynchronous, monkeypatch):
    a = asynchronous
    a.bind(a.source(a.owner()))
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(other=True))
    with pytest.raises(RuntimeError, match="stream"):
        a.graph.run(Mock())
    a.graph.entries[Key()].graph.replay.assert_not_called()


def test_seal_allows_startup_capture_to_handoff_to_live_stream(asynchronous, monkeypatch):
    a = asynchronous
    a.bind(None)
    a.graph.seal((Key(),))
    torch.npu.synchronize.assert_called_once()
    torch.npu.synchronize.reset_mock()
    live_stream = SimpleNamespace(synchronize=Mock())
    monkeypatch.setattr(torch.npu, "current_stream", lambda: live_stream)
    monkeypatch.setattr(torch.npu, "Event", lambda: SimpleNamespace(record=Mock(), query=lambda: True))
    a.bind(a.source(a.owner()))
    a.graph.run(Mock())
    live_stream.synchronize.assert_not_called()
    torch.npu.synchronize.assert_not_called()


def test_partial_retain_failure_releases_only_acquired_references(asynchronous):
    a = asynchronous
    first, second = a.owner(), a.owner()
    second.ref_count_up = Mock(side_effect=RuntimeError("retain failed"))
    with pytest.raises(RuntimeError, match="retain failed"):
        a.lease((a.source(first), a.source(second)))
    assert first.meta.ref_count == second.meta.ref_count == 1
    assert not first.freed and not second.freed


def test_shutdown_sync_failure_keeps_all_owners(asynchronous, monkeypatch):
    a = asynchronous
    owner = a.owner()
    a.bind(a.source(owner))
    owner.ref_count_down()
    monkeypatch.setattr(torch.npu, "synchronize", Mock(side_effect=RuntimeError("device failure")))
    with pytest.raises(RuntimeError, match="device failure"):
        a.graph.clear()
    assert not owner.freed and a.graph.source_bindings


def test_worker_cleanup_executes_real_graph_release_before_base(asynchronous):
    a = asynchronous
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    cls.body = [node for node in cls.body if getattr(node, "name", "") in ("release_sfa_graph_resources", "shutdown")]
    calls = []
    namespace = {"WorkerBase": type("Base", (), {"shutdown": lambda self: calls.append("base")})}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    instance = object.__new__(namespace["NPUWorker"])
    instance.model_runner = SimpleNamespace(_sfa_full_graph=a.graph)
    owner = a.owner()
    a.bind(a.source(owner))
    owner.ref_count_down()
    instance.shutdown()
    assert owner.freed and calls == ["base"]
    torch.npu.synchronize.assert_called_once()


def test_finished_binding_cannot_be_replayed_without_rebinding(asynchronous):
    a = asynchronous
    a.bind(a.source(a.owner()))
    a.events[0].ready = True
    a.graph.release_requests({"a"})
    with pytest.raises(RuntimeError, match="bound source batch"):
        a.graph.run(Mock())
    a.graph.entries[Key()].graph.replay.assert_not_called()
