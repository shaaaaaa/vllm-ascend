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


@pytest.mark.parametrize("split_at", [6, 13, 15])
def test_offline_trace_events_header_across_read_boundary(clients, tmp_path, split_at):
    import analyze_decode_moe_copy

    prefix = '{"label":"traceEvents","padding":"'
    closing = '",'
    padding = "x" * ((1 << 20) - len(prefix) - len(closing) - split_at)
    trace = tmp_path / "wrapped.json"
    header = '"traceEvents"  :  '
    trace.write_text(prefix + padding + closing + header + '[{"name":"kernel"}]}', encoding="utf-8-sig")
    assert list(analyze_decode_moe_copy.events(trace)) == [{"name": "kernel"}]


def test_grouped_copy_layer_times_count_device_work_and_inter_piece_gaps(clients, tmp_path):
    _, gaps = clients
    schedule = (12, 24, 12)
    events = []

    def task(start, duration, name, stream):
        return {
            "ph": "X", "pid": 8, "tid": stream, "ts": str(start), "dur": duration, "name": name,
            "args": {"Task Type": "AI_VECTOR_CORE", "Physic Stream Id": stream},
        }

    for replay, (limit, unit) in enumerate(zip(schedule, (2, 10, 4))):
        start = 1000 * replay
        events.append(task(start, unit, "sfa_copy_i32_s64_155", limit))
        for layer in (1, 2, 3):
            start += 100
            duration = layer * unit
            for ts in (start, start + duration + 3 * layer):
                events.append(task(ts, duration, "sfa_copy_piece_i32_s64_156", limit + 1))
    trace = tmp_path / "grouped.json"
    cpu_op = {"ph": "X", "pid": 3, "tid": 4, "ts": "10", "dur": 90, "name": "sfa_copy_i32_s64"}
    trace.write_text(json.dumps({"traceEvents": [cpu_op, *reversed(events)]}))
    timings = gaps.grouped_copy_layer_times(trace, schedule, layers=4, pieces=2)
    for limit, median_unit in ((12, 3), (24, 10)):
        assert timings[limit, 0] == (median_unit, median_unit)
        for layer in (1, 2, 3):
            active = 2 * layer * median_unit
            assert timings[limit, layer] == (active, active + 3 * layer)

    trace.write_text(json.dumps(events[:-1]))
    with pytest.raises(ValueError, match="expected 21 device copy tasks"):
        gaps.grouped_copy_layer_times(trace, schedule, layers=4, pieces=2)
    events[2] = task(110, 1, "sfa_copy_piece_i32_s64", 99)
    trace.write_text(json.dumps(events))
    with pytest.raises(ValueError, match="main/side copy schedule"):
        gaps.grouped_copy_layer_times(trace, schedule, layers=4, pieces=2)


@pytest.mark.parametrize("wait_kind", ["CAPTURE_WAIT", "EVENT_WAIT"])
@pytest.mark.parametrize("fragmented", [False, True])
def test_grouped_copy_waits_pair_with_consumer_arrival(clients, tmp_path, wait_kind, fragmented):
    _, gaps = clients
    events = []
    schedule = (12, 24, 12)

    def task(start, duration, name, kind, stream, task_id=None):
        events.append({
            "ph": "X", "pid": 8, "tid": stream, "ts": str(start), "dur": duration, "name": name,
            "args": {
                "Task Type": kind, "Physic Stream Id": stream, "Task Id": len(events) if task_id is None else task_id,
            },
        })

    for i, limit in enumerate(schedule):
        ts = 100 * i
        task(ts, 2, "sfa_copy_i32_s64", "AI_VECTOR_CORE", limit)
        task(ts + 3, 0.2, "other wait", wait_kind, limit)
        for tick in (10, 14, 20, 24, 39, 43):
            task(ts + tick, 1, "sfa_copy_piece_i32_s64", "AI_VECTOR_CORE", limit + 1)
        for index, (begin, duration) in enumerate(((8, 12), (35, 0.4), (40, 5))):
            task(ts + begin - 2, 0.2, "gate", "MEM_WRITE_VALUE", limit)
            task(ts + begin - 1, 0.1, "give", "NOTIFY_RECORD", limit)
            if fragmented and duration > 1:
                task(ts + begin, 0.25, "consumer", wait_kind, limit, 100 + index)
                task(ts + begin + 1, duration - 1, "consumer", wait_kind, limit, 100 + index)
            else:
                task(ts + begin, duration, "consumer", wait_kind, limit, 100 + index)
    trace = tmp_path / "grouped.json"
    trace.write_text(json.dumps(events[::-1]))
    waits = gaps.grouped_copy_waits(trace, schedule, layers=4, pieces=2)
    for limit in (12, 24):
        for layer, sample in enumerate(((7000, 12000), (0, 400), (4000, 5000)), 1):
            assert waits[limit, layer] == [sample] * schedule.count(limit)

    events[:] = [e for e in events if not (e["args"]["Task Id"] == 100 and float(e["ts"]) < 100)]
    trace.write_text(json.dumps(events))
    with pytest.raises(ValueError, match="consumer waits"):
        gaps.grouped_copy_waits(trace, schedule, layers=4, pieces=2)


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
    import analyze_decode_moe_copy

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
    offline, _ = analyze_decode_moe_copy.analyze(trace, threshold=5_000, layers=2, top=1)
    device = offline["devices"][0]
    assert device["dispatch"]["target_first"]["long_n"] == 1
    assert device["dispatch"]["target_regular"]["n"] == 0
    assert device["slowest_dispatch"]["target_first"][0]["copy_overlap_us"] == (0 if at_consumer else 2.6)
    assert device["moe_stage_wall"]["dispatch_to_combine_wall_first"]["max_us"] == 9
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


@pytest.mark.parametrize("vup,router,stray,quant_open",
                         [(0, 0, 0, False), (0, 4, 0, False), (3, 2, 0, False), (5, 0, 0, False),
                          (1, 1, 1, False), (1, 1, 1, True)])
def test_dynamic_windows_report_actual_piece_counts_and_protected_overlap(
    clients, tmp_path, vup, router, stray, quant_open
):
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
    if quant_open:
        index = next(i for i, row in enumerate(tasks) if row[0] == "DynamicQuant")
        tasks[index:index + 1] = [
            ("open quant", "MEM_WRITE_VALUE", 37, 0.4), ("give quant", "NOTIFY_RECORD", 37.5, 0.4),
            tasks[index], ("close quant", "MEM_WRITE_VALUE", 43, 0.4), ("take quant", "NOTIFY_WAIT", 43.5, 0.4),
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
        **({"additional": {stray: 1}} if quant_open else {}),
    }
    assert samples[0]["vup_close_to_acquire"] == 4000 and samples[0]["router_close_to_acquire"] == 2000
    assert layers[0]["router_pieces_mean"] == router
    assert samples[0]["copy_overlap_additional_windows"] == 1000 * (stray if quant_open else 0)
    assert samples[0]["copy_overlap_protected_main_compute"] == 1000 * (0 if quant_open else stray)
    assert summary["pieces_outside_or_ambiguous_window"] == (0 if quant_open else stray)


@pytest.mark.parametrize(("gated", "extra"), [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("placement", ["producer_vup", "first_qkv", "per_consumer"])
def test_three_prefetch_completions_match_three_consumers_even_in_one_window(
    clients, tmp_path, gated, placement, extra
):
    import analyze_decode_moe_copy

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
        if extra and layer:
            main("open norm", "MEM_WRITE_VALUE")
            main("give norm", "NOTIFY_RECORD")
        main("RmsNorm_input" if layer == 0 else "AddRmsNormBias_input", "AI_VECTOR_CORE")
        if layer:
            if extra:
                main("close norm", "MEM_WRITE_VALUE")
                main("take norm", "NOTIFY_WAIT")
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
    offline, per_layer = analyze_decode_moe_copy.analyze(trace, threshold=200_000, layers=4, top=1)
    device = offline["devices"][0]
    assert device["consumer_wait"]["joined_layers"] == 3
    stalls = {"producer_vup": [0, 0, 0], "first_qkv": [8, 0, 0], "per_consumer": [8, 8, 8]}
    assert [row["wait_p50_us"] for row in per_layer if row["matched_joins"]] == stalls[placement]
    assert device["overlap"]["overlap_us"] == {"producer_vup": 12, "first_qkv": 8, "per_consumer": 0}[placement]
    summary, layers, samples = gaps.analyze(trace, expected=4)
    assert summary["indexer_fork_layers_zero_based"] == [0]
    assert summary["indexer_prefetch_edges"] == 3 and summary["copy_pieces_per_edge"] == {2: 3}
    assert [row["indexer_layer_for_next"] for row in layers] == [0, 0, 0, None]
    assert [s["consumer_layer"] for s in samples[:-1]] == [1, 2, 3]
    assert all(s["copy_overlap_vup"] == (4000 if placement == "producer_vup" else 0) for s in samples[:-1])
    if gated:
        assert summary["pieces_outside_or_ambiguous_window"] == 0
        assert ("additional" in summary["pieces_per_window_histogram"]) == extra
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
    incomplete, _ = analyze_decode_moe_copy.analyze(trace, threshold=200_000, layers=4, top=1)
    waits = incomplete["devices"][0]["consumer_wait"]
    assert waits["status"] == "unavailable"
    assert waits["copy_stall_sum_per_replay"]["sum_us"] is None
