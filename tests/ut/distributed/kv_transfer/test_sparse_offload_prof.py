# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.distributed.kv_transfer.sparse_offload import _prof


def test_forward_detail_partitions_sfa_and_non_sfa(monkeypatch):
    timestamps = iter((0, 10, 20, 30, 50, 70, 75, 85, 100, 120))
    monkeypatch.setattr(_prof, "_TRACE_ENABLED", True)
    monkeypatch.setattr(_prof, "_sync", lambda trace: None)
    monkeypatch.setattr(
        _prof.time,
        "perf_counter_ns",
        lambda: next(timestamps) * 1_000_000,
    )

    forward = _prof.start_forward(7, "request-1", sync_npu=False)

    layer0 = _prof.begin("sfa_fwd", layer_name="model.layers.0.self_attn")
    with _prof.section("indexer"):
        pass
    _prof.end(layer0)

    layer1 = _prof.begin("sfa_fwd", layer_name="model.layers.1.self_attn")
    with _prof.section("fa"):
        pass
    _prof.end(layer1)

    result = _prof.finish_forward(forward)

    assert result is not None
    assert result["total_ms"] == 120
    assert result["sfa_total_ms"] == 70
    assert result["non_sfa_total_ms"] == 50
    assert result["sfa_child_total_ms"] == 20
    assert result["sfa_unattributed_ms"] == 50
    assert result["sfa_calls"] == 2
    assert result["non_sfa_segments"] == 3
    assert result["non_sfa_max_ms"] == 20
    assert result["non_sfa_slowest_segment"] == "model.layers.0.self_attn->model.layers.1.self_attn"
    assert "indexer:10.000/1/10.000@model.layers.0.self_attn" in result["phases"]
    assert "fa:10.000/1/10.000@model.layers.1.self_attn" in result["phases"]


def test_forward_detail_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(_prof, "_TRACE_ENABLED", False)

    assert _prof.start_forward(0, "request-1", sync_npu=False) is None
    assert _prof.begin("sfa_fwd", layer_name="model.layers.0.self_attn") is None
    assert _prof.finish_forward(None) is None


def test_forward_detail_reports_prepare_stages_without_double_counting(monkeypatch):
    timestamps = iter((0, 10, 20, 25, 30, 40, 45, 48, 50, 60))
    monkeypatch.setattr(_prof, "_TRACE_ENABLED", True)
    monkeypatch.setattr(_prof, "_sync", lambda trace: None)
    monkeypatch.setattr(
        _prof.time,
        "perf_counter_ns",
        lambda: next(timestamps) * 1_000_000,
    )

    forward = _prof.start_forward(8, "request-2", sync_npu=False)
    sfa = _prof.begin("sfa_fwd", layer_name="model.layers.0.self_attn")

    boundary = _prof.begin("prepare_boundary")
    _prof.end(boundary)
    fused_op = _prof.begin("prepare_sparse_indices")
    _prof.end(fused_op)
    payload = _prof.begin("prepare_payload")
    _prof.end(payload)

    _prof.end(sfa)
    result = _prof.finish_forward(forward)

    assert result is not None
    assert result["sfa_total_ms"] == 40
    assert result["sfa_child_total_ms"] == 18
    assert result["sfa_unattributed_ms"] == 22
    assert "prepare_sparse_indices:10.000/1/10.000" in result["phases"]
    assert "prepare_boundary:5.000/1/5.000" in result["phases"]
    assert "prepare_payload:3.000/1/3.000" in result["phases"]
