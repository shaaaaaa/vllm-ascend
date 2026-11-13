# SPDX-License-Identifier: Apache-2.0
"""Standalone policy, trace and device regressions; no serving imports."""

import re
from contextlib import contextmanager

import benchmark_dcp as benchmark
import pytest
import torch
from dcp_workload import CAPACITY, TOPK, cold_case, dcp_reference, make_trace, trace_metadata
from generate_sources import SOURCE, build_sources
from resident_experiment import Case, _put_state, _state, assert_result


@pytest.fixture(autouse=True, scope="module")
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_dcp_build_changes_only_finalize_and_keeps_safety_guards():
    sources = build_sources("dcp")
    assert set(sources) == {"dcp_union.cpp", "dcp_update.cpp", "dcp_old_finalize.cpp", "dcp_new_finalize.cpp"}
    original = SOURCE.read_text(encoding="utf-8")
    classes = original.split('extern "C" __global__ __aicore__ void\n', 1)[0]
    for name, source in sources.items():
        assert source.count('extern "C" __global__ __aicore__ void\n') == 1
        assert "#define RESIDENT_DCP_SHARDING 1" in source
        assert "#define RESIDENT_EXPERIMENT_SKIP_UNCHANGED 0" in source
        for feature in ("VECTOR_INTERSECTION", "VECTOR_STATE", "QUERY_MEMBERSHIP"):
            assert f"#define RESIDENT_EXPERIMENT_{feature} 1" in source
        assert classes in source
        free_first = int(name == "dcp_new_finalize.cpp")
        assert f"#define RESIDENT_FREE_SLOT_FIRST {free_first}" in source
        assert "oldCount + currentCount <= kStateWidth" in source
    launch = sources["dcp_update.cpp"].split("void resident_experiment_dcp_update", 1)[1]
    assert list(map(int, re.findall(r"a\.tensors\[(\d+)\]", launch))) == [0, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 20, 21]
    assert "a.dcpBits,\n        a.dcpRank" in launch


@pytest.mark.parametrize("overlap", [0, 0.333, 0.9, 1])
@pytest.mark.parametrize("query_overlap", [0, 0.5, 1])
def test_trace_has_configured_consecutive_overlap_and_unique_queries(overlap, query_overlap):
    trace = make_trace(requests=2, steps=5, overlap=overlap, query_overlap=query_overlap)
    metadata = trace_metadata(trace, 8, 0)
    for row in trace.flatten(0, 1)[:, 0].tolist():
        assert len(set(row)) == TOPK
    assert metadata["temporal_overlap"]["min"] == metadata["temporal_overlap"]["max"] == int(TOPK * overlap) / TOPK
    assert metadata["query_overlap"]["min"] == metadata["query_overlap"]["max"] == int(TOPK * query_overlap) / TOPK
    assert torch.equal(trace, make_trace(requests=2, steps=5, overlap=overlap, query_overlap=query_overlap, rank=7))


def test_fresh_trace_never_reintroduces_a_retired_token():
    trace = make_trace(steps=8, workload="fresh")
    for query in (0, 1):
        previous, seen = set(), set()
        for row in trace[:, query, 0].tolist():
            current = set(row)
            assert not (current - previous) & seen
            seen |= current
            previous = current


@pytest.mark.parametrize(
    "kwargs",
    [
        {"overlap": -0.1},
        {"overlap": 1.1},
        {"overlap": float("nan")},
        {"query_overlap": 2},
        {"steps": 1},
        {"world": 3},
        {"rank": 8},
        {"history": 2047},
        {"working_set": 100},
        {"steps": 300, "overlap": 0, "workload": "fresh"},
    ],
)
def test_invalid_trace_inputs_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        make_trace(**kwargs)


def policy_case(occupancy, world=8, rank=0, skew=False):
    case = cold_case(1, 1 << 20)
    tokens = [t for t in range(1 << 20) if t // 128 % world == rank and (not skew or t % 8 == 0)]
    resident = tokens[:occupancy]
    _put_state(case, 0, dict(zip(resident, reversed(range(occupancy)), strict=True)))
    # Hits must keep their deliberately permuted slots. Missing tokens include
    # the next owner's token, which must never enter this rank's resident map.
    selected = resident[: min(32, occupancy)] + tokens[occupancy : occupancy + 65]
    foreign = ((rank + 1) % world) * 128
    case["topk"].fill_(-1)
    for query, row in enumerate((selected, list(reversed(selected[::2])))):
        case["topk"][query, 0, : len(row) + 1] = torch.tensor([*row, foreign])
    return case


@pytest.mark.parametrize("occupancy", [0, 1, 2047, 2048, 4095, 4096])
@pytest.mark.parametrize("world,rank", [(2, 1), (4, 2), (8, 7)])
def test_free_slots_precede_eviction_and_keep_owned_payload_identity(occupancy, world, rank):
    case = policy_case(occupancy, world, rank)
    old = _state(case, 0)
    selected = {t for t in case["topk"].flatten().tolist() if t >= 0 and t // 128 % world == rank}
    result = dcp_reference(case, world, rank, True)
    after = _state(result, 0)
    missing = sorted(selected - old.keys(), key=lambda t: (t % case.shards, t))
    fresh = min(len(missing), CAPACITY - occupancy)
    assert len(after) == min(CAPACITY, occupancy + len(missing))
    assert [after[t] for t in missing[:fresh]] == list(range(occupancy, occupancy + fresh))
    assert all(after[t] == old[t] for t in selected & old.keys())
    assert len(old.keys() - after.keys()) == len(missing) - fresh
    payload = {slot: t for t, slot in old.items()}
    for token, physical in zip(
        result["miss_tokens"][0, : len(missing)].tolist(), result["target_slots"][0, : len(missing)].tolist()
    ):
        pages = case["block_table"][0].tolist()
        logical = pages.index(physical // 128) * 128 + physical % 128
        assert 0 <= logical < CAPACITY
        payload[logical] = token
    for query in range(2):
        count = int(result.tensors[21][query, 0])
        recovered = {payload[slot] for slot in result["topk"][query, 0, :count].tolist()}
        assert recovered == {t for t in case["topk"][query, 0].tolist() if t >= 0 and t // 128 % world == rank}


def test_retention_saves_returning_tokens_but_not_fresh_tokens():
    snapshot = cold_case(1, 65536)
    trace = make_trace(steps=32, working_set=8192)
    old = benchmark.validate(snapshot, trace, 8, 0, 0)
    new = benchmark.validate(snapshot, trace, 8, 0, 1)
    assert new["total_misses"] < old["total_misses"]
    assert new["final_resident_per_request"][0] > old["final_resident_per_request"][0]
    fresh = make_trace(steps=8, workload="fresh")
    assert (
        benchmark.validate(snapshot, fresh, 8, 0, 0)["total_misses"]
        == (benchmark.validate(snapshot, fresh, 8, 0, 1)["total_misses"])
    )


def test_generation_rollover_all_hits_and_padding():
    initial = policy_case(2048)
    inputs = initial["topk"].clone()
    first = dcp_reference(initial, 8, 0, True)
    first["topk"].copy_(inputs)
    hits = dcp_reference(first, 8, 0, True)
    assert int(hits["miss_counts"][0, 0]) == 0
    assert _state(hits, 0) == _state(first, 0)
    hits["topk"].copy_(inputs)
    hits["request_generations"].add_(1)
    hits["row_requests"][1] = -1
    cold = dcp_reference(hits, 8, 0, True)
    assert len(_state(cold, 0)) == int(cold["miss_counts"][0, 0])
    assert int(cold.tensors[21][1, 0]) == 0
    assert not cold["topk"][1].any()


def test_sequence_keeps_state_between_steps(monkeypatch):
    case = cold_case(1, 65536)
    inputs = make_trace(steps=4, overlap=1)
    counts = []

    def emulate(case, world, rank, policy):
        result = dcp_reference(case, world, rank, bool(policy))
        case.reset_from(result)
        counts.append(int(case["miss_counts"][0, 0]))

    monkeypatch.setattr(benchmark, "run", emulate)
    benchmark.sequence(case, inputs, 8, 0, 1)
    assert counts[0] > 0 and counts[1:] == [0, 0, 0]


def test_timing_resets_before_events_and_alternates_policies(monkeypatch):
    calls, graphs = [], []

    class Graph:
        def __init__(self):
            self.policy = len(graphs)
            graphs.append(self)

        def replay(self):
            calls.append(("replay", self.policy))

        def reset(self):
            calls.append(("graph_clear", self.policy))

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            calls.append(("event",))

        def elapsed_time(self, end):
            return 1.0

    @contextmanager
    def capture(graph):
        yield

    npu = type("NPU", (), {"NPUGraph": Graph, "Event": Event, "graph": capture, "synchronize": lambda: None})
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(benchmark, "run", lambda *args: None)
    original = Case.reset_from

    def reset(case, snapshot):
        calls.append(("reset",))
        original(case, snapshot)

    monkeypatch.setattr(Case, "reset_from", reset)
    benchmark.measure(cold_case(1, 65536), make_trace(steps=2), 8, 0, "cpu", repeats=2, warmup=0)
    start = calls.index(("event",)) - 1
    assert calls[start:-2] == [
        item for policy in (0, 1, 1, 0) for item in (("reset",), ("event",), ("replay", policy), ("event",))
    ]


@pytest.mark.parametrize("world,rank", [(2, 1), (4, 2), (8, 0), (8, 7)])
@pytest.mark.parametrize("policy", [0, 1])
def test_native_dcp_stateful_replay(native, world, rank, policy):
    trace = make_trace(requests=2, steps=12, overlap=0.9, world=world, rank=rank)
    benchmark.validate(cold_case(2, 65536), trace, world, rank, policy, native)


@pytest.mark.parametrize("occupancy", [0, 2047, 2048, 2049, 4095, 4096])
@pytest.mark.parametrize("policy", [0, 1])
def test_native_dcp_skew_fresh_mixed_full_and_scalar_fallback(native, occupancy, policy):
    snapshot = policy_case(occupancy, skew=True)
    trace = torch.stack([snapshot["topk"].clone()] * 3)
    benchmark.validate(snapshot, trace, 8, 0, policy, native)


@pytest.mark.parametrize("policy", [0, 1])
def test_native_dcp_generation_padding_and_full_union(native, policy):
    snapshot = cold_case(1, 1 << 20)
    tokens = [t for t in range(1 << 20) if t // 128 % 8 == 0 and t % 8 == 0][:CAPACITY]
    snapshot["topk"][0, 0] = torch.tensor(tokens[:TOPK])
    snapshot["topk"][1, 0] = torch.tensor(tokens[TOPK:])
    result = dcp_reference(snapshot, 8, 0, bool(policy))
    assert len(_state(result, 0)) == CAPACITY
    actual = snapshot.clone(native)
    benchmark.run(actual, 8, 0, policy)
    torch.npu.synchronize()
    assert_result(actual, result, membership=True)
    result["topk"].copy_(snapshot["topk"])
    result["request_generations"].add_(1)
    result["row_requests"][1] = -1
    benchmark.validate(result, torch.stack([result["topk"].clone()] * 2), 8, 0, policy, native)


@pytest.mark.parametrize("policy", [0, 1])
def test_native_dcp_zero_owned_selections_preserve_residents(native, policy):
    snapshot = cold_case(1, 65536)
    trace = make_trace(steps=3)
    foreign = [t for t in range(65536) if t // 128 % 8 == 1][:TOPK]
    trace[1, :, 0] = torch.tensor(foreign)
    trace[2].copy_(trace[0])
    stats = benchmark.validate(snapshot, trace, 8, 0, policy, native)
    assert stats["steps"][0]["misses"][0] > 0
    assert [row["misses"][0] for row in stats["steps"][1:]] == [0, 0]
