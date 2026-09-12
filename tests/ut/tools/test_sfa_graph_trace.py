# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic trace-schema tests, not hardware capture validation."""

import importlib.util
import json
import os
import subprocess
import sys
from decimal import Decimal
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


@pytest.mark.parametrize("ts_type", [int, float, str])
@pytest.mark.parametrize("dur_type", [int, float, str])
def test_numeric_and_string_trace_times_are_equivalent(trace, ts_type, dur_type):
    events = [
        event(trace.ROOT, ts_type(100), dur_type(10)),
        event("aclmdlRIExecuteAsync", ts_type(101), dur_type(5)),
        event(trace.RETRIEVE, ts_type(120), dur_type(3)),
    ]
    assert trace.audit_events(events, "full")["status"] == "ONE_EXECUTE_PER_ROOT"
    # Caller-owned trace data must not be rewritten during normalization.
    assert type(events[0]["ts"]) is ts_type
    assert type(events[0]["dur"]) is dur_type


def test_reported_string_timestamp_plus_float_duration_crash(trace):
    events = [
        event(trace.ROOT, "100.0", 10.0),
        event("aclmdlRIExecuteAsync", "101.0", 5.0, tid=999),
        event(trace.RETRIEVE, "102.0", 1.0),
    ]
    # A known Python retrieval inside the root is still a split, even if
    # runtime API lanes cannot be correlated. It must not crash or be skipped.
    assert trace.audit_events(events, "full")["status"] == "SPLIT_DETECTED"


def test_mixed_numeric_string_begin_end_and_scientific_notation(trace):
    events = [
        dict(ph="B", name=trace.ROOT, ts="1e2", pid=1, tid=2),
        event("aclmdlExecuteAsync", 101, "2.5"),
        dict(ph="E", ts=110.0, pid=1, tid=2),
    ]
    assert trace.audit_events(events, "full")["status"] == "ONE_EXECUTE_PER_ROOT"


def test_large_timestamp_decimal_boundaries_do_not_round_into_the_root(trace, tmp_path):
    events = [
        event(trace.ROOT, "1789000000000000.000", "0.100"),
        event("aclmdlExecuteAsync", "1789000000000000.101", "0.001"),
    ]
    assert trace.audit_events(events, "full")["status"] == "UNVERIFIED"
    # Also preserve precision when JSON contains a number, not a string.
    path = write_trace(tmp_path, 0, [])
    path.write_text("""{"traceEvents":[
        {"name":"sfa_full_graph::target_replay","ph":"X","pid":1,"tid":2,
         "ts":1789000000000000.000,"dur":0.100},
        {"name":"aclmdlExecuteAsync","ph":"X","pid":1,"tid":2,
         "ts":1789000000000000.101,"dur":0.001}]}""")
    assert trace.analyse_traces(tmp_path, "full", 1)["status"] == "UNVERIFIED"


@pytest.mark.parametrize("bad", ["bad", "NaN", "Infinity", float("nan"), None, True])
@pytest.mark.parametrize("field", ["ts", "dur"])
def test_invalid_relevant_times_are_unverified_not_silently_dropped(trace, bad, field):
    events = [event(trace.ROOT), event("aclmdlExecuteAsync", 1, 2), event(trace.RETRIEVE, 3, 2)]
    events[-1][field] = bad
    assert trace.audit_events(events, "full")["status"] == "UNVERIFIED"


def test_intervals_normalize_x_and_be_without_mutation(trace):
    raw = [
        event(trace.ROOT, "100", "10.5"),
        dict(ph="B", name="other", ts="101", pid=1, tid=2),
        dict(ph="E", ts="103.25", pid=1, tid=2),
    ]
    normalized = trace.intervals(raw)
    assert normalized[0]["ts"] == Decimal("100")
    assert normalized[1]["dur"] == Decimal("2.25")
    assert raw[0]["ts"] == "100" and "dur" not in raw[1]


@pytest.mark.parametrize(
    "bad_event",
    [
        event("aclmdlExecuteAsync", 1, "-1"),
        {"name": "aclmdlExecuteAsync", "ph": "X", "ts": 1, "pid": 1, "tid": 2},
        {"name": "aclmdlExecuteAsync", "ph": "X", "dur": 1, "pid": 1, "tid": 2},
        {"name": "aclmdlExecuteAsync", "ph": "B", "ts": "1", "pid": 1, "tid": 2},
    ],
)
def test_missing_negative_and_unclosed_intervals_cannot_be_ignored(trace, bad_event):
    events = [event(trace.ROOT), event("aclmdlExecuteAsync", 1, 2), bad_event]
    assert trace.audit_events(events, "full")["status"] == "UNVERIFIED"


def test_reversed_be_times_are_unverified(trace):
    events = [
        dict(ph="B", name=trace.ROOT, ts="100", pid=1, tid=2),
        event("aclmdlExecuteAsync", 1, 2),
        dict(ph="E", ts="99", pid=1, tid=2),
    ]
    assert trace.audit_events(events, "full")["status"] == "UNVERIFIED"


def profiled_run(trace, root, *, split=False):
    staged = [event(trace.RETRIEVE, "0", "1"), event("aclmdlExecuteAsync", "2", 3)]
    full = [event(trace.ROOT, "0.0", 10), event("aclmdlExecuteAsync", 1, "2")]
    if split:
        full.append(event("aclmdlExecuteAsync", "5", 2))
    write_trace(root / "staged", 0, staged)
    write_trace(root / "full", 0, full)
    (root / "comparison.json").write_text('{"saved_performance":"keep exactly"}')
    return {path: path.read_bytes() for path in root.rglob("*.json")}


@pytest.mark.parametrize("split", [False, True])
def test_real_cpu_cli_rechecks_existing_data_without_rewriting_measurements_or_traces(trace, tmp_path, split):
    before = profiled_run(trace, tmp_path, split=split)
    process = subprocess.run(
        [sys.executable, trace.__file__, "--run-dir", str(tmp_path), "--ranks", "1"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == int(split), process.stderr
    assert "STAGED_TRACE_PRESENT" in process.stdout
    assert ("SPLIT_DETECTED" if split else "ONE_EXECUTE_PER_ROOT") in process.stdout
    assert "Existing measurements unchanged" in process.stdout
    for path, original in before.items():
        assert path.read_bytes() == original
    assert (tmp_path / "staged-trace-check.json").is_file()
    assert (tmp_path / "full-trace-check.json").is_file()


def test_latest_uses_measurement_time_not_recheck_time_and_ignores_nonprofile_runs(trace, tmp_path):
    old, new = tmp_path / "sfa-old", tmp_path / "sfa-new"
    profiled_run(trace, old)
    profiled_run(trace, new)
    os.utime(old / "comparison.json", (100, 100))
    os.utime(new / "comparison.json", (200, 200))
    (old / "full-trace-check.json").write_text("{}")  # touches old run directory today
    perf = tmp_path / "sfa-perf-only"
    perf.mkdir()
    (perf / "comparison.json").write_text("{}")
    assert trace.latest_run(tmp_path) == new
    assert trace.main(["--latest", str(tmp_path), "--ranks", "1"]) == 0
    assert (new / "full-trace-check.json").is_file()


def test_recheck_missing_traces_never_starts_model_or_profiler(trace, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(SystemExit) as error:
        trace.main(["--run-dir", str(tmp_path)])
    assert error.value.code == 1
    with pytest.raises(ValueError, match="No completed benchmark"):
        trace.latest_run(tmp_path)
