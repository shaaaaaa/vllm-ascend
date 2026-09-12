# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic trace-schema tests, not hardware capture validation."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def trace():
    path = Path(__file__).resolve().parents[3] / "tools/sfa_graph_trace.py"
    spec = importlib.util.spec_from_file_location("tested_sfa_graph_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event(name, ts=0, dur=10, pid=1, tid=2):
    return dict(name=name, ts=ts, dur=dur, pid=pid, tid=tid, ph="X")


def test_one_execute_per_root_and_draft_outside_root(trace):
    events = [
        event(trace.ROOT),
        event("aclmdlRIExecuteAsync", 1, 5),
        event("aclmdlRIExecuteAsync", 20, 5),
        event(trace.RETRIEVE, 30, 3),
    ]
    result = trace.audit_events(events, "full")
    assert result["status"] == "ONE_EXECUTE_PER_ROOT"
    assert result["executes_per_root"] == [1]
    assert result["acl_execute_events"] == 2  # outside calls are NOT target splits


@pytest.mark.parametrize("extra", [event("aclmdlExecuteAsync", 7, 1), event("sfa_cross_layer::lmcache_retrieve", 7, 1)])
def test_extra_submission_or_python_retrieval_in_root_is_a_split(trace, extra):
    result = trace.audit_events([event(trace.ROOT), event("aclmdlRIExecuteAsync", 1, 5), extra], "full")
    assert result["status"] == "SPLIT_DETECTED"


def test_nested_api_aliases_are_one_submission_not_two_disjoint_replays(trace):
    result = trace.audit_events(
        [event(trace.ROOT), event("aclmdlRIExecuteAsync", 1, 8), event("aclmdlExecuteAsync", 2, 5)], "full"
    )
    assert result["status"] == "ONE_EXECUTE_PER_ROOT"


@pytest.mark.parametrize(
    "api",
    [
        event("aclmdlRIExecuteAsync", 1, 5, pid=999),
        event("aclmdlRIExecuteAsync", 1, 5, tid=999),
        event("unknownRuntimeAPI", 1, 5),
    ],
)
def test_overlapping_other_thread_and_unknown_runtime_cannot_fake_proof(trace, api):
    assert trace.audit_events([event(trace.ROOT), api], "full")["status"] == "UNVERIFIED"


def test_empty_or_missing_scope_does_not_pass(trace):
    assert trace.audit_events([], "full")["status"] == "UNVERIFIED"
    assert trace.audit_events([event("aclmdlExecuteAsync")], "full")["status"] == "UNVERIFIED"
    assert trace.audit_events([event(trace.ROOT)], "full")["status"] == "UNVERIFIED"
    assert trace.audit_events([event(trace.ROOT, pid=None)], "full")["status"] == "UNVERIFIED"


def test_b_e_events_and_metadata(trace):
    events = [
        dict(ph="M", name="process_name", pid=1, args={"name": "worker"}),
        dict(ph="B", name=trace.ROOT, ts=0, pid=1, tid=2),
        event("aclmdlExecuteAsync", 1, 2),
        dict(ph="E", ts=5, pid=1, tid=2),
    ]
    assert trace.audit_events(events, "full")["status"] == "ONE_EXECUTE_PER_ROOT"


def test_staged_requires_both_retrieval_and_execution_without_root(trace):
    events = [event(trace.RETRIEVE), event("aclmdlExecuteAsync", 15, 2)]
    assert trace.audit_events(events, "staged")["status"] == "STAGED_TRACE_PRESENT"
    assert trace.audit_events(events[:1], "staged")["status"] == "UNVERIFIED"
    assert trace.audit_events(events + [event(trace.ROOT)], "staged")["status"] == "UNVERIFIED"


def write_trace(root, rank, events, suffix="capture"):
    path = root / f"sfa_full_dp0_pp0_tp{rank}_rank{rank}_{suffix}_ascend_pt" / "ASCEND_PROFILER_OUTPUT/trace_view.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"traceEvents": events}))
    return path


def test_all_eight_rank_files_checked_individually(trace, tmp_path):
    for rank in range(8):
        write_trace(tmp_path, rank, [event(trace.ROOT), event("aclmdlExecuteAsync", 1, 3)])
    result = trace.analyse_traces(tmp_path, "full", 8)
    assert result["status"] == "ONE_EXECUTE_PER_ROOT" and len(result["workers"]) == 8


def test_one_missing_rank_is_not_replaced_by_another_ranks_duplicate(trace, tmp_path):
    write_trace(tmp_path, 0, [])
    with pytest.raises(ValueError, match="Incomplete"):
        trace.analyse_traces(tmp_path, "full", 2)
    write_trace(tmp_path, 0, [], suffix="duplicate")
    with pytest.raises(ValueError, match="Duplicate"):
        trace.analyse_traces(tmp_path, "full", 2)


def test_one_rank_split_or_unknown_is_not_hidden_by_other_ranks(trace, tmp_path):
    write_trace(tmp_path, 0, [event(trace.ROOT), event("aclmdlExecuteAsync", 1, 3)])
    bad = write_trace(tmp_path, 1, [event(trace.ROOT)])
    assert trace.analyse_traces(tmp_path, "full", 2)["status"] == "UNVERIFIED"
    bad.write_text(
        json.dumps([event(trace.ROOT), event("aclmdlExecuteAsync", 1, 3), event("aclmdlExecuteAsync", 7, 2)])
    )
    assert trace.analyse_traces(tmp_path, "full", 2)["status"] == "SPLIT_DETECTED"
