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
@pytest.mark.parametrize("at_consumer", [False, True])
def test_copy_pieces_match_the_graph_fork_and_consumer_join(clients, tmp_path, opens_with_gate, at_consumer):
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
    if at_consumer:
        task(6, 1, "batch_matmul_transpose", "AI_CORE", 4)
    else:
        task(6, 1, "CAPTURE_RECORD", "CAPTURE_RECORD", 4)
    task(8, 1, "AivKernel", "COMMUNICATION", 81, stream=5)
    task(8, 1, "NOTIFY_RECORD", "NOTIFY_RECORD", 5)
    task(10, 6, "MoeDistributeDispatch", "MIX_AIV", 6)
    task(17, 2, "MoeDistributeCombine", "MIX_AIV", 7)
    task(20, 1, "AddRmsNormBias_input", "AI_VECTOR_CORE", 8)
    if at_consumer:
        task(21.02, 0.08, "CAPTURE_RECORD", "CAPTURE_RECORD", 9)
        task(21.14, 2.56, "CAPTURE_WAIT", "CAPTURE_WAIT", 10)
        task(21.8, 0.3, "IndependentWork", "AI_VECTOR_CORE", 91, stream=6)
        intervals = ((21.1, 0.7), (21.8, 0.5), (22.3, 0.5), (22.8, 0.6))
    else:
        task(22, 1, "CAPTURE_WAIT", "CAPTURE_WAIT", 9)
        intervals = ((7.2, 2.2), (9.6, 2), (13, 1), (21.4, 1.1))
    task(24, 1, "KvRmsNormRopeCache", "AI_VECTOR_CORE", 11)
    task(25, 2, "SparseFlashAttention", "MIX_AIC", 12)
    task(28, 1, "SwiGlu", "AI_VECTOR_CORE", 13)
    task(30, 1, "AddRmsNormBias_final", "AI_VECTOR_CORE", 14)
    for index, (start, duration) in enumerate(intervals):
        task(start, duration, "sfa_copy_piece_i32_s64", "AI_VECTOR_CORE", 30 + index, stream=7)
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(events))
    summary, layers, samples = gaps.analyze(trace, expected=2)
    assert summary["copy_pieces_per_edge"] == {4: 1}
    assert summary["consumer_release_edges"] == int(at_consumer)
    assert summary["edges_overlapping_main_compute"] == int(not at_consumer)
    assert summary["edges_overlapping_comm_aiv"] == int(not at_consumer)
    assert layers[0]["released_at_consumer"] == int(at_consumer)
    assert layers[1]["async_to_next"] == 0
    if at_consumer:
        assert samples[0]["copy_device"] == 2300
        assert samples[0]["copy_before_consumer_norm_end"] == 0
        assert samples[0]["copy_overlap_main_compute"] == 0
        assert samples[0]["copy_overlap_comm_aiv"] == 0
        assert samples[0]["copy_overlap_other_device_compute"] == 300
        assert samples[0]["pieces_started_during_dispatch"] == 0
        assert samples[0]["dispatch_to_combine_copy_overlap"] == 0
        assert samples[0]["join_wait_for_copy"] == 2260
        return
    assert samples[0]["copy_before_consumer_norm_end"] == 5200
    assert samples[0]["copy_overlap_main_compute"] == 2600
    assert samples[0]["copy_overlap_comm_aiv"] == 1000
    assert samples[0]["copy_overlap_other_device_compute"] == 0
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


@pytest.mark.parametrize("vup,router,stray", [(0, 0, 0), (0, 4, 0), (3, 2, 0), (5, 0, 0), (1, 1, 1)])
def test_dynamic_windows_report_actual_piece_counts_and_protected_overlap(clients, tmp_path, vup, router, stray):
    _, gaps = clients
    events = []

    def emit(name, kind, tick, duration, ordinal, stream=3):
        events.append(dict(ph="X", name=name, pid=8, tid=stream, ts=tick, dur=duration,
                           args={"Physic Stream Id": stream, "Model Id": 9, "Task Id": ordinal, "Task Type": kind}))

    tasks = [
        ("RmsNorm_input", "AI_VECTOR_CORE", 1, 1), ("KvRmsNormRopeCache", "AI_VECTOR_CORE", 3, 1),
        ("SparseFlashAttention", "MIX_AIC", 5, 2), ("CAPTURE_RECORD", "CAPTURE_RECORD", 8, 1),
        ("open", "MEM_WRITE_VALUE", 10, 1), ("release", "NOTIFY_RECORD", 11, 1),
        ("batch_matmul_transpose", "AI_CORE", 13, 20), ("close", "MEM_WRITE_VALUE", 33, 1),
        ("take", "NOTIFY_WAIT", 34, 3), ("DynamicQuant", "AI_VECTOR_CORE", 38, 5),
        ("Cast", "AI_VECTOR_CORE", 45, 2), ("open", "MEM_WRITE_VALUE", 48, 1),
        ("release", "NOTIFY_RECORD", 49, 1), ("MatMulV2_router", "AI_CORE", 51, 20),
        ("close", "MEM_WRITE_VALUE", 71, 1), ("take", "NOTIFY_WAIT", 72, 1),
        ("Cast", "AI_VECTOR_CORE", 74, 1), ("MoeDistributeDispatch", "MIX_AIV", 76, 4),
        ("MoeDistributeCombine", "MIX_AIV", 81, 4), ("AddRmsNormBias_input", "AI_VECTOR_CORE", 86, 2),
        ("open", "MEM_WRITE_VALUE", 88, 1), ("release", "NOTIFY_RECORD", 89, 1),
        ("join", "CAPTURE_WAIT", 91, 16), ("close", "MEM_WRITE_VALUE", 108, 1),
        ("take", "NOTIFY_WAIT", 109, 1), ("KvRmsNormRopeCache", "AI_VECTOR_CORE", 111, 1),
        ("SparseFlashAttention", "MIX_AIC", 113, 2), ("SwiGlu", "AI_VECTOR_CORE", 116, 1),
        ("AddRmsNormBias_final", "AI_VECTOR_CORE", 118, 1),
    ]
    for i, row in enumerate(tasks):
        emit(*row, i)
    for i, tick in enumerate([14 + 3*j for j in range(vup)] + [52 + 3*j for j in range(router)] +
                             [40]*stray + [92 + 2*j for j in range(6-vup-router-stray)]):
        emit("sfa_copy_piece_i32_s64", "AI_VECTOR_CORE", tick, 1, 101 + i, 7)
    trace = tmp_path / "dynamic.json"
    trace.write_text(json.dumps(events))
    summary, layers, samples = gaps.analyze(trace, expected=2)
    assert summary["cube_windows"] == {"vup": 1, "router": 1}
    assert summary["pieces_per_window_histogram"] == {
        "vup": {vup: 1}, "router": {router: 1}, "tail": {6 - vup - router - stray: 1},
    }
    assert samples[0]["vup_close_to_acquire"] == 4000 and samples[0]["router_close_to_acquire"] == 2000
    assert layers[0]["router_pieces_mean"] == router
    assert samples[0]["copy_overlap_protected_main_compute"] == 1000 * stray
    assert summary["pieces_outside_or_ambiguous_window"] == stray


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("placement", ["producer_vup", "first_qkv", "per_consumer"])
def test_three_prefetch_completions_match_three_consumers_even_in_one_window(clients, tmp_path, gated, placement):
    _, gaps = clients
    events, cubes, joins, qkvs = [], [], [], []
    tick = 0
    root_task = 0

    def emit(name, kind, start, duration, ordinal, stream):
        events.append(dict(ph="X", name=name, pid=8, tid=stream, ts=start, dur=duration,
                           args={"Physic Stream Id": stream, "Model Id": 9, "Task Id": ordinal, "Task Type": kind}))

    def main(name, kind, duration=2):
        nonlocal tick, root_task
        start = tick
        emit(name, kind, start, duration, root_task, 3)
        tick += duration
        root_task += 1
        return start

    for layer in range(4):
        main("RmsNorm_input" if layer == 0 else "AddRmsNormBias_input", "AI_VECTOR_CORE")
        if layer:
            if gated:
                main("open", "MEM_WRITE_VALUE")
                main("release", "NOTIFY_RECORD")
            joins.append(main("join", "CAPTURE_WAIT", 20))
            qkvs.append(main("FusedQKVGemm", "MIX_AIC", 48))
            if gated:
                main("close", "MEM_WRITE_VALUE")
                main("take", "NOTIFY_WAIT")
        main("KvRmsNormRopeCache", "AI_VECTOR_CORE")
        if layer == 0:
            main("indexer plan", "AI_VECTOR_CORE")
            main("fork", "CAPTURE_RECORD")
            main("sfa_copy_i32_s64", "AI_VECTOR_CORE")
        main("SparseFlashAttention", "MIX_AIC")
        if gated and layer < 3:
            main("open", "MEM_WRITE_VALUE")
            main("release", "NOTIFY_RECORD")
        cubes.append(main("batch_matmul_transpose", "AI_CORE", 48))
        if gated and layer < 3:
            main("close", "MEM_WRITE_VALUE")
            main("take", "NOTIFY_WAIT")
        main("SwiGlu", "AI_VECTOR_CORE")
    main("AddRmsNormBias_final", "AI_VECTOR_CORE")
    ordinal = 1
    for i in range(3):
        if placement == "producer_vup":
            begin = cubes[0] + 2 + 12 * i
        elif placement == "first_qkv" and i:
            begin = qkvs[0] + 2 + 12 * i
        else:
            begin = joins[i] + 2
        for piece in range(2):
            emit("sfa_copy_piece", "AI_VECTOR_CORE", begin + 4 * piece, 2, ordinal, 7)
            ordinal += 1
            if gated:
                emit("give", "NOTIFY_RECORD", begin + 4 * piece + 2, 1, ordinal, 7)
                ordinal += 3 if piece == 0 else 1  # Next piece waits on the word and the token.
        emit("done", "CAPTURE_RECORD", begin + 7, 1, ordinal, 7)
        ordinal += 3 if gated else 1
    trace = tmp_path / "group.json"
    trace.write_text(json.dumps(events))
    summary, layers, samples = gaps.analyze(trace, expected=4)
    assert summary["indexer_fork_layers_zero_based"] == [0]
    assert summary["indexer_prefetch_edges"] == 3 and summary["copy_pieces_per_edge"] == {2: 3}
    assert [row["indexer_layer_for_next"] for row in layers] == [0, 0, 0, None]
    assert [s["consumer_layer"] for s in samples[:-1]] == [1, 2, 3]
    assert all(s["copy_overlap_vup"] == (4000 if placement == "producer_vup" else 0) for s in samples[:-1])
    if gated:
        assert summary["pieces_outside_or_ambiguous_window"] == 0
        assert summary["pieces_per_window_histogram"]["vup"] == {2 if placement == "producer_vup" else 0: 3}
        assert summary["pieces_per_window_histogram"]["qkv"] == {0 if placement == "producer_vup" else 2: 3}
        expected = [0, 4000, 4000] if placement == "first_qkv" else [0, 0, 0]
        assert [s["copy_overlap_qkv"] for s in samples[:-1]] == expected
        assert summary["edges_overlapping_protected_main_compute"] == 0
    # Dropping a completion must never attribute the two merged trains to one consumer.
    events.remove(next(e for e in events if e["name"] == "done"))
    trace.write_text(json.dumps(events))
    with pytest.raises(ValueError, match="layer trains"):
        gaps.analyze(trace, expected=4)
