# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for client timing and graph-prefetch trace pairing."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def clients(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    import analyze_decode_layer_gaps
    import bench_decode_copy_scheduling

    return bench_decode_copy_scheduling, analyze_decode_layer_gaps


def data(ids, request="req"):
    message = {"id": request, "choices": [{"index": 0, "token_ids": ids}]}
    return b"data: " + json.dumps(message).encode() + b"\n"


def test_tpot_uses_tokens_after_the_first_chunk(clients, monkeypatch):
    bench, _ = clients
    clock = iter((2.0, 2.05, 2.08, 2.1))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))
    result = bench.receive([data([10, 11]), data([12]), data([13, 14]), b"data: [DONE]"], 1.0, 5)
    assert result["tpot_ms"] == pytest.approx(80.0 / 3)
    assert result["first_chunk_tokens"] == 2
    assert result["e2e_ms"] == pytest.approx(1100)
    assert result["output_sha256"] == bench.digest([10, 11, 12, 13, 14])


@pytest.mark.parametrize(
    "chunks", [[data([1]), data([2], "retry"), b"data: [DONE]"], [data([1]), b"data: [DONE]"], [data([1]), data([2])]]
)
def test_failed_streams_are_excluded(clients, chunks):
    bench, _ = clients
    with pytest.raises(ValueError):
        bench.receive(chunks, 0, 2)


@pytest.mark.parametrize("opens_with_gate", [False, True])
def test_copy_pieces_match_the_graph_fork_and_consumer_join(clients, tmp_path, opens_with_gate):
    _, gaps = clients
    events = [{"ph": "M", "name": "process_name", "pid": 8, "args": {"name": "Ascend Hardware"}}]

    def task(start, duration, name, kind, task_id, stream=3):
        events.append(
            {
                "ph": "X",
                "pid": 8,
                "tid": stream,
                "ts": str(start),
                "dur": duration,
                "name": name,
                "args": {"Physic Stream Id": stream, "Task Id": task_id, "Model Id": 9, "Task Type": kind},
            }
        )

    if opens_with_gate:
        task(0, 1, "MEM_WRITE_VALUE", "MEM_WRITE_VALUE", 0)
    else:
        task(0, 1, "Cast", "AI_VECTOR_CORE", 0)
    task(1, 1, "RmsNorm_input", "AI_VECTOR_CORE", 1)
    task(2, 1, "KvRmsNormRopeCache", "AI_VECTOR_CORE", 2)
    task(3, 3, "SparseFlashAttention", "MIX_AIC", 3)
    task(6, 1, "CAPTURE_RECORD", "CAPTURE_RECORD", 4)
    task(8, 1, "NOTIFY_RECORD", "NOTIFY_RECORD", 5)
    task(10, 6, "MoeDistributeDispatch", "MIX_AIV", 6)
    task(17, 2, "MoeDistributeCombine", "MIX_AIV", 7)
    task(20, 1, "AddRmsNormBias_input", "AI_VECTOR_CORE", 8)
    task(22, 1, "CAPTURE_WAIT", "CAPTURE_WAIT", 9)
    task(24, 1, "KvRmsNormRopeCache", "AI_VECTOR_CORE", 10)
    task(25, 2, "SparseFlashAttention", "MIX_AIC", 11)
    task(28, 1, "SwiGlu", "AI_VECTOR_CORE", 12)
    task(30, 1, "AddRmsNormBias_final", "AI_VECTOR_CORE", 13)
    for index, (start, duration) in enumerate(((7.2, 2.2), (9.6, 2), (13, 1), (21.4, 1.1))):
        task(start, duration, "sfa_copy_piece_i32_s64", "AI_VECTOR_CORE", 30 + index, stream=7)
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(events))
    summary, layers, samples = gaps.analyze(trace, expected=2)
    assert summary["copy_pieces_per_edge"] == {4: 1}
    assert samples[0]["copy_device"] == 6300
    assert samples[0]["copy_span"] == 15300
    assert samples[0]["join_wait_for_copy"] == 500
    assert samples[0]["pieces_started_after_notify_before_dispatch"] == 1
    assert samples[0]["pieces_started_during_dispatch"] == 1
    assert summary["pieces_started_dispatch_to_combine_histogram"] == {1: 1}
    assert samples[0]["dispatch_to_combine_copy_overlap"] == 2600
    assert samples[0]["fork_to_dispatch"] == 3000
    assert samples[0]["combine_to_join_arrival"] == 3000
    assert samples[0]["copy_excess_outside_dispatch_to_combine"] == 300
    assert summary["copy_fits_outside_dispatch_to_combine"] == 0
    assert layers[1]["async_to_next"] == 0
