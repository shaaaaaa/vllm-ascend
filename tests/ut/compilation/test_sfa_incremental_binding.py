# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Managed source tables with real transfer math and allocator ownership."""

import ast
import weakref
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import ROOT
from test_sfa_async_lifetime import Key, asynchronous  # noqa: F401
from test_sfa_full_graph import graph_module  # noqa: F401


@pytest.fixture
def managed(request):
    a = request.getfixturevalue("asynchronous")
    path = ROOT.parent / "LMCache-Ascend/lmcache_ascend/v1/npu_connector/sparse_graph.py"
    node = next(n for n in ast.parse(path.read_text(encoding="utf8")).body if isinstance(n, ast.ClassDef))
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(node)
    ns = dict(
        torch=torch,
        KVCacheFormat=NS(MLA_LATENT=NS(value=0)),
        prepare_sparse_direct_destination_state=lambda *args: None,
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    a.graph.clear()
    a.context.staged_sfa_graph_key = Key(4)
    a.context.staged_sfa_graph_dummy_run = True
    a.transfers = tuple(
        ns["SparseGraphTransfer"](
            (torch.zeros(2, 16, 1, 8), torch.zeros(2, 16, 1, 2)),
            torch.zeros(4, 4, dtype=torch.int32),
            256,
            1024,
            request_capacity=4,
        )
        for _ in range(2)
    )
    a.graph.register_transfers(4, a.transfers)
    a.graph.run(lambda: "output")
    a.graph.seal((Key(4),))
    a.context.staged_sfa_graph_dummy_run = False
    original = a.source

    def source(owner, *, ptr=1000, marked=True):
        value = original(owner, layers=2)
        value.chunk_token_counts, value.total_tokens = (256,), 256
        if marked:
            value.binding_token = object()
        for layer in value.layers:
            layer.chunk_ptrs_npu.fill_(ptr)
        return value

    a.source = source
    a.bind = lambda sources, ids=None: a.graph.bind_sources(tuple(sources), tuple(ids or map(str, range(len(sources)))))
    torch.npu.synchronize.reset_mock()
    return a


def test_only_changed_lane_is_written_and_old_owners_wait(managed, monkeypatch):
    a = managed
    owners = [a.owner() for _ in range(4)]
    sources = [a.source(o, ptr=1000 + i * 1000) for i, o in enumerate(owners)]
    a.bind(sources)
    a.graph.run(Mock())
    first_event = a.events[-1]
    before = [t.ptrs.clone() for t in a.transfers]
    observed = []
    for t in a.transfers:
        original = t.bind_batch

        def record(*args, _original=original, **kwargs):
            observed.append(kwargs.get("lanes"))
            return _original(*args, **kwargs)

        monkeypatch.setattr(t, "bind_batch", record)
    old = owners[1]
    old.ref_count_down()
    sources[1] = a.source(a.owner(), ptr=9000)
    a.bind(sources)
    assert observed == [(1,), (1,)] and not old.freed
    for t, previous in zip(a.transfers, before):
        torch.testing.assert_close(t.ptrs[:, :4], previous[:, :4])
        torch.testing.assert_close(t.ptrs[:, 8:], previous[:, 8:])
        assert t.ptrs[0, 4] == 9000 and t.ptrs[:, 5:8].eq(0).all()
    first_event.ready = True
    a.graph.collect_retired_sources()
    assert old.freed
    observed.clear()
    a.bind(sources)
    assert observed == []
    torch.npu.synchronize.assert_not_called()


def test_finish_keeps_only_identity_history_not_sources_or_allocations(managed):
    a = managed
    owner = a.owner()
    source = a.source(owner)
    pointer = weakref.ref(source.layers[0].chunk_ptrs_npu)
    token = source.binding_token
    a.bind([source], ["old"])
    a.graph.run(Mock())
    event = a.events[-1]
    owner.ref_count_down()
    a.graph.release_requests({"old"})
    assert not owner.freed and 4 not in a.graph.source_bindings
    with pytest.raises(RuntimeError, match="requires a bound"):
        a.graph.run(Mock())
    del source
    event.ready = True
    a.graph.collect_retired_sources()
    assert owner.freed and pointer() is None
    assert a.graph._table_history[4] == (token,)
    new = a.source(a.owner(), ptr=3000)
    a.bind([new], ["old"])
    assert a.transfers[0].ptrs[0, 0] == 3000


def test_idle_binding_clears_sources_but_preserves_inflight_owners(managed):
    a = managed
    owner = a.owner()
    a.bind([a.source(owner)])
    a.graph.run(Mock())
    active_event = a.events[-1]
    owner.ref_count_down()
    a.context.staged_sfa_graph_dummy_run = True
    a.bind([])
    assert not owner.freed
    for transfer in a.transfers:
        assert transfer.ptrs.eq(0).all() and transfer.valid_tokens.eq(0).all()
    a.graph.run(Mock())
    assert a.graph.source_bindings[4].sources == ()
    active_event.ready = True
    a.graph.collect_retired_sources()
    assert owner.freed
    a.context.staged_sfa_graph_dummy_run = False
    a.bind([a.source(a.owner(), ptr=9000)])
    a.graph.run(Mock())
    assert a.transfers[0].ptrs[0, 0] == 9000
    torch.npu.synchronize.assert_not_called()


def test_partial_update_failure_forces_full_retry_and_preserves_leases(managed, monkeypatch):
    a = managed
    sources = [a.source(a.owner(), ptr=1000 + i * 1000) for i in range(4)]
    a.bind(sources)
    sources[0] = a.source(a.owner(), ptr=9000)
    original = a.transfers[1].bind_batch

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("partial write")

    monkeypatch.setattr(a.transfers[1], "bind_batch", fail)
    with pytest.raises(RuntimeError, match="partial write"):
        a.bind(sources)
    assert 4 not in a.graph.source_bindings and 4 not in a.graph._table_history
    assert len(a.graph.retired_sources) == 2
    with pytest.raises(RuntimeError, match="requires a bound"):
        a.graph.run(Mock())
    seen = []
    for t in a.transfers:
        method = original if t is a.transfers[1] else t.bind_batch

        def record(*args, _method=method, **kwargs):
            seen.append(kwargs.get("lanes"))
            return _method(*args, **kwargs)

        monkeypatch.setattr(t, "bind_batch", record)
    a.bind(sources)
    assert seen == [None, None]
    assert 4 in a.graph._table_history


def test_record_failure_blocks_replay_and_does_not_publish_history(managed, monkeypatch):
    a = managed

    def event():
        return NS(record=Mock(side_effect=RuntimeError("record failed")))

    monkeypatch.setattr(torch.npu, "Event", event)
    with pytest.raises(RuntimeError, match="record failed"):
        a.bind([a.source(a.owner())])
    assert 4 not in a.graph._table_history and a.graph._submission_failed
    with pytest.raises(RuntimeError, match="submission failed"):
        a.graph.run(Mock())


def test_unmarked_sources_and_callback_compatibility_use_full_binding(managed, monkeypatch):
    a = managed
    source = a.source(a.owner(), marked=False)
    a.bind([source])
    assert 4 not in a.graph._table_history
    source = a.source(a.owner())
    planner = Mock(wraps=a.transfers[0].plan_bind_update)
    a.graph._transfer_bundles[4] = (a.transfers, planner)
    a.bind([source])
    planner.assert_not_called()
    with pytest.raises(RuntimeError, match="callback"):
        a.graph.bind_sources((source,), ("new",), lambda: a.transfers)


def test_identical_physical_sources_reacquire_lease_without_table_writes(managed, monkeypatch):
    a = managed
    value = a.source(a.owner())
    a.bind([value], ["a"])
    for t in a.transfers:
        monkeypatch.setattr(t, "bind_batch", Mock(side_effect=AssertionError("unchanged table")))
    a.graph.release_requests({"a"})
    a.bind([value], ["b"])
    assert a.graph.source_bindings[4].request_ids == ("b",)
    a.graph.run(Mock())


def test_clear_discards_history_and_startup_transfer_bundles(managed):
    a = managed
    a.bind([a.source(a.owner())])
    a.graph.clear()
    assert not a.graph._table_history and not a.graph._transfer_bundles
    with pytest.raises(RuntimeError, match="not registered"):
        a.graph.get_transfers(4)


def test_capacity_switching_uses_independent_registered_tables(managed):
    a = managed
    a.graph.clear()
    a.context.staged_sfa_graph_dummy_run = True
    bundles = {4: a.transfers}
    cls = type(a.transfers[0])
    bundles[8] = tuple(
        cls(
            (torch.zeros(2, 16, 1, 8), torch.zeros(2, 16, 1, 2)),
            torch.zeros(8, 4, dtype=torch.int32),
            256,
            1024,
            request_capacity=8,
        )
        for _ in range(2)
    )
    for cap, bundle in bundles.items():
        a.context.staged_sfa_graph_key = Key(cap)
        a.graph.register_transfers(cap, bundle)
        a.graph.run(lambda: "output")
    a.graph.seal((Key(4), Key(8)))
    a.context.staged_sfa_graph_dummy_run = False
    value = a.source(a.owner())
    for cap in (4, 8):
        a.context.staged_sfa_graph_key = Key(cap)
        assert a.bind([value])
    before = [t.ptrs.clone() for t in bundles[4]]
    a.bind([a.source(a.owner(), ptr=9000)])
    a.context.staged_sfa_graph_key = Key(4)
    assert not a.bind([value])
    for actual, expected in zip(bundles[4], before):
        torch.testing.assert_close(actual.ptrs, expected)
    assert a.graph.get_transfers(4) is bundles[4]


def test_registration_rejects_replacement_and_seal_requires_all_capacities(managed):
    a = managed
    a.graph.clear()
    a.context.staged_sfa_graph_dummy_run = True
    a.graph.register_transfers(4, a.transfers)
    a.graph.register_transfers(4, a.transfers)
    with pytest.raises(RuntimeError, match="changed"):
        a.graph.register_transfers(4, tuple(reversed(a.transfers)))
    a.graph.run(lambda: "output")
    a.context.staged_sfa_graph_key = Key(8)
    a.graph.run(lambda: "output")
    with pytest.raises(RuntimeError, match="missing a transfer"):
        a.graph.seal((Key(4), Key(8)))
    a.context.staged_sfa_graph_dummy_run = False
    with pytest.raises(RuntimeError, match="startup"):
        a.graph.register_transfers(4, a.transfers)


def test_binding_without_incremental_backend_keeps_full_api(managed):
    a = managed
    a.graph.clear()
    a.context.staged_sfa_graph_dummy_run = True
    legacy = NS(bind_batch=Mock())
    a.graph.register_transfers(4, (legacy,))
    a.graph.run(lambda: "output")
    a.graph.seal((Key(4),))
    a.context.staged_sfa_graph_dummy_run = False
    first = a.source(a.owner())
    a.bind([first])
    a.bind([a.source(a.owner())])
    assert legacy.bind_batch.call_count == 2
    assert all(not call.kwargs for call in legacy.bind_batch.call_args_list)
    assert not a.graph._table_history
