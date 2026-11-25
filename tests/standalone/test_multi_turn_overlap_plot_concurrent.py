# SPDX-License-Identifier: Apache-2.0
"""Keep cross-round batch timelines separate from client TTFT round statistics."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def plot(monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "multi_turn_overlap_concurrent_test", tools / "plot_multi_turn_overlap_concurrent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rounds():
    return [
        dict(
            round=i,
            phase="initial" if i == 0 else "followup",
            request_id=f"multi-run-{i:03d}",
            response_id=f"cmpl-response-{i}",
            prompt_tokens=500000 + i * 128,
            ttft_s=(271.099606, 26.227285, 23.801610)[i],
        )
        for i in range(3)
    ]


def record(turn=0, tp=0, step=1, coverage=0.5, samples=2, duration=10, first=1):
    return dict(
        schema_version=2,
        coverage_definition="same_bank_reuse_v1",
        status="complete",
        host="host",
        pid=10 + tp,
        dp_rank=0,
        tp_rank=tp,
        step=step,
        batch_start_time_ns=1_700_000_000_000_000_000 + (100 * turn + step + tp) * 100_000_000,
        requests=[
            dict(req_id=f"cmpl-multi-run-{turn:03d}-0-internal", token_start=(step - 1) * 4096, token_end=step * 4096)
        ],
        coverage=coverage,
        layer_samples=samples,
        first_layer_wait_ms=first,
        all_wait_ms=duration + first,
        initial_bank_wait_ms=first,
        **{name: duration for name in ("dma_ms", "save_ms", "load_ms", "wait_ms", "interval_ms")},
    )


def test_maps_both_request_and_response_ids_without_round_substring_matches(plot):
    for request_id in (
        "multi-run-001",
        "cmpl-multi-run-001-0-internal",
        "chatcmpl-multi-run-001-0",
        "cmpl-response-1-0-internal",
    ):
        assert plot.match_round(request_id, rounds()) == 1
    assert plot.match_round("cmpl-multi-run-0010-0", rounds()) is None
    assert plot.match_round("cmpl-other-run-001", rounds()) is None
    bad = rounds()
    bad[2]["response_id"] = bad[1]["response_id"]
    with pytest.raises(ValueError, match="Ambiguous"):
        plot.match_round("cmpl-response-1-0", bad)


def test_timeline_keeps_every_batch_and_sorts_submission_not_emission_time(plot):
    records = [
        record(coverage=0.9, samples=1),
        record(step=2, coverage=0.3, samples=3, duration=30),
        record(turn=1, duration=7),
    ]
    chunks, diagnostic = plot.timeline_records(list(reversed(records)), rounds())
    assert [r["time_s"] for r in chunks] == [0, 0.1, 10]
    assert [r["coverage"] for r in chunks] == [0.9, 0.3, 0.5]
    assert [r["save_ms"] for r in chunks] == [10, 30, 7]
    assert [r["first_layer_wait_ms"] for r in chunks] == [1, 1, 1]
    assert diagnostic["missing_rounds"] == [2]
    assert all(r["time_source"] == "host_batch_submission" for r in chunks)
    assert not any("round" in r for r in chunks)


def test_timeline_rejects_old_coverage_definition(plot):
    old = dict(record(), schema_version=1)
    old.pop("coverage_definition")
    with pytest.raises(ValueError, match="old adjacent-layer"):
        plot.timeline_records([old], rounds())


def test_time_bins_average_recorded_layer_ratios_without_round_grouping(plot):
    records = [record(coverage=0.9, samples=1), record(step=2, coverage=0.3, samples=3, duration=30)]
    records.append(
        dict(record(turn=2, tp=1, duration=100), batch_start_time_ns=records[0]["batch_start_time_ns"] + 200_000_000)
    )
    chunks, _ = plot.timeline_records(records, rounds())
    rows = plot.aggregate_timeline(chunks, 1.0)
    assert len(rows) == 1
    assert rows[0]["coverage"] == pytest.approx((0.9 + 0.3 * 3 + 0.5 * 2) / 6)
    assert rows[0]["time_s"] == 0.5
    assert rows[0]["save_ms"] == pytest.approx(140 / 3)
    assert rows[0]["worker_count"] == 2 and rows[0]["batch_records"] == 3
    assert rows[0]["first_layer_wait_ms"] == 1
    assert rows[0]["all_wait_ms"] == pytest.approx(143 / 3)
    chunks, _ = plot.timeline_records(records, rounds(), tp_rank=1)
    assert len(chunks) == 1 and chunks[0]["save_ms"] == 100


def test_time_bins_leave_gaps_and_missing_waits_unknown(plot):
    first = record(turn=1)
    later = dict(record(turn=2), batch_start_time_ns=first["batch_start_time_ns"] + 2_500_000_000)
    del later["first_layer_wait_ms"], later["all_wait_ms"]
    chunks, diagnostic = plot.timeline_records([first, later], rounds())
    rows = plot.aggregate_timeline(chunks, 1.0)
    assert [r["time_start_s"] for r in rows] == [0, 2]
    assert rows[1]["first_layer_wait_ms"] is None and rows[1]["all_wait_ms"] is None
    assert rows[1]["wait_ms"] == 10 and rows[1]["coverage"] == 0.5
    assert diagnostic["missing_wait_detail_records"] == 1
    rows = plot.aggregate_timeline(chunks, 10.0)
    assert rows[0]["first_layer_wait_ms"] is None


def test_mixed_batch_with_unknown_corequest_is_kept_once(plot):
    mixed = record(turn=1)
    mixed["requests"].append(dict(req_id="another", token_start=8000, token_end=8512))
    unknown = record()
    unknown["requests"][0]["req_id"] = "unrelated"
    chunks, diagnostic = plot.timeline_records([mixed, unknown], rounds())
    assert len(chunks) == 1 and chunks[0]["request_count"] == 2
    assert chunks[0]["dma_ms"] == 10 and chunks[0]["coverage"] == 0.5
    assert diagnostic["unmatched_request_count"] == 2
    assert diagnostic["batches_with_other_requests"] == 1


def test_missing_timestamp_is_not_replaced_with_batch_order_or_round(plot):
    item = record(turn=1)
    del item["batch_start_time_ns"]
    chunks, diagnostic = plot.timeline_records([item], rounds())
    assert not chunks and diagnostic["untimed_records"] == 1
    item["log_time_ns"] = 1_700_000_000_000_000_000
    chunks, diagnostic = plot.timeline_records([item], rounds())
    assert chunks[0]["time_s"] == 0
    assert chunks[0]["time_source"] == "log_emission_approximate"
    assert diagnostic["approximate_timestamp_records"] == 1
    item["batch_start_time_ns"] = item["log_time_ns"] - 9_000_000_000
    chunks, _ = plot.timeline_records([item], rounds())
    assert chunks[0]["timestamp_ns"] == item["batch_start_time_ns"]
    assert chunks[0]["time_source"] == "host_batch_submission"


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_invalid_time_bin_width_is_rejected(plot, width):
    with pytest.raises(ValueError, match="finite and positive"):
        plot.aggregate_timeline([], width)


def test_read_saved_summary_and_csv_fallback(plot, tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(dict(rounds=rounds())), encoding="utf-8")
    assert plot.read_rounds(tmp_path) == rounds()
    path.unlink()
    with (tmp_path / "metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rounds()[0]))
        writer.writeheader()
        writer.writerows(rounds())
    assert [r["round"] for r in plot.read_rounds(tmp_path)] == [0, 1, 2]


def test_real_cli_deduplicates_batches_and_uses_distinct_time_and_round_axes(plot, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    saved = []
    original = Figure.savefig

    def save(figure, *args, **kwargs):
        saved.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    (tmp_path / "summary.json").write_text(json.dumps(dict(rounds=rounds())), encoding="utf-8")
    records = [record(), record(step=2), record(turn=2, duration=7)]
    log = tmp_path / "server.log"
    line = "\x1b[32m(Worker) [PREFILL_OVERLAP] "
    log.write_text("\n".join(line + json.dumps(r) + "\x1b[0m" for r in records + records), encoding="utf-8")
    output = tmp_path / "timeline.png"
    assert plot.main([str(log), "--run-dir", str(tmp_path), "--output", str(output)]) == 0
    assert output.stat().st_size > 0
    figure = saved[0]
    assert len(figure.axes) == 8
    assert list(figure.axes[0].lines[0].get_xdata()) == [0]
    assert list(figure.axes[2].lines[0].get_ydata()) == [7]
    assert list(figure.axes[4].lines[0].get_ydata()) == [8]
    assert list(figure.axes[5].lines[0].get_ydata()) == [1]
    assert "Elapsed time" in figure.axes[0].get_xlabel()
    assert "round" in figure.axes[7].get_xlabel()
    assert not figure.axes[0].get_shared_x_axes().joined(figure.axes[0], figure.axes[7])
    assert not any("interval" in axis.get_title().lower() for axis in figure.axes)
    assert list(figure.axes[7].lines[0].get_xdata()) == [1, 2]
    assert [text.get_text() for text in figure.axes[7].texts] == ["26.23", "23.80"]
    assert not any(axis.texts for axis in figure.axes[:7])  # No per-request labels.
    diagnostic = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert diagnostic["missing_rounds"] == [1]
    with output.with_suffix(".chunks.csv").open(encoding="utf-8") as source:
        chunk_rows = list(csv.DictReader(source))
    assert len(chunk_rows) == 1 and json.loads(chunk_rows[0]["requests"])[0]["token_start"] == 0
    with output.with_suffix(".ttft.csv").open(encoding="utf-8") as source:
        assert [r["round"] for r in csv.DictReader(source)] == ["1", "2"]


def test_ttft_is_one_unchanged_client_curve_with_multiple_worker_timelines(plot, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved.append(figure))
    chunks, _ = plot.timeline_records([record(), record(step=2), record(tp=1)], rounds())
    plot.plot_timeline(chunks, rounds(), tmp_path / "ranks.png")
    axes = saved[0].axes
    assert len(axes[0].lines) == 2
    assert len(axes[7].lines) == 1
    assert list(axes[7].lines[0].get_ydata()) == pytest.approx([271.099606, 26.227285, 23.801610])
    assert len(axes[7].texts) == 3


def test_csv_ttft_strings_blanks_and_zero_keep_their_meaning_in_the_plot(plot, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved.append(figure))
    metadata = rounds()
    metadata[0]["ttft_s"] = 0
    metadata[1]["ttft_s"] = ""
    with (tmp_path / "metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    metadata = plot.read_rounds(tmp_path)
    chunks, _ = plot.timeline_records([record(turn=i) for i in range(3)], metadata)
    plot.plot_timeline(chunks, metadata, tmp_path / "blanks.png")
    axis = saved[0].axes[7]
    values = list(axis.lines[0].get_ydata())
    assert values[0] == 0 and str(values[1]) == "nan" and values[2] == pytest.approx(23.801610)
    assert [text.get_text() for text in axis.texts] == ["0.00", "23.80"]


def test_unmatched_logs_fail_clearly(plot, tmp_path, capsys):
    (tmp_path / "summary.json").write_text(json.dumps(dict(rounds=rounds())), encoding="utf-8")
    log = tmp_path / "server.log"
    log.write_text("ordinary log", encoding="utf-8")
    assert plot.main([str(log), "--run-dir", str(tmp_path)]) == 2
    assert "No timestamped overlap records matched" in capsys.readouterr().err


@pytest.mark.parametrize("source", ["summary.json", "metrics.csv"])
def test_ttft_only_reads_completed_client_rounds_without_server_access(plot, tmp_path, monkeypatch, source):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    metadata = rounds()
    metadata[1]["ttft_s"] = ""
    if source == "summary.json":
        (tmp_path / source).write_text(json.dumps(dict(status="interrupted", rounds=metadata)), encoding="utf-8")
    else:
        with (tmp_path / source).open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(metadata)

    def unexpected_server_access(*args, **kwargs):
        pytest.fail("Client-only TTFT must not access server logs or SSH")

    monkeypatch.setattr(plot, "collect_logs", unexpected_server_access)
    monkeypatch.setattr(plot, "read_records", unexpected_server_access)
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved.append(figure))
    output = tmp_path / "plots" / "ttft.png"
    assert (
        plot.main(
            [
                "--run-dir",
                str(tmp_path),
                "--ttft-only",
                "--start-round",
                "1",
                "--end-round",
                "10",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(saved[0].axes) == 1
    axis = saved[0].axes[0]
    assert list(axis.lines[0].get_xdata()) == [1, 2]
    values = list(axis.lines[0].get_ydata())
    assert str(values[0]) == "nan" and values[1] == pytest.approx(23.801610)
    assert [text.get_text() for text in axis.texts] == ["23.80"]
    with output.with_suffix(".csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["round"] for row in rows] == ["1", "2"]
    assert rows[0]["ttft_s"] == ""
    assert float(rows[1]["ttft_s"]) == pytest.approx(23.801610)


def test_ttft_only_rejects_absent_client_measurements(plot, tmp_path, capsys):
    metadata = rounds()
    for row in metadata:
        row.pop("ttft_s")
    (tmp_path / "summary.json").write_text(json.dumps(dict(rounds=metadata)), encoding="utf-8")
    assert plot.main(["--run-dir", str(tmp_path), "--ttft-only"]) == 2
    assert "No client TTFT values" in capsys.readouterr().err


def concurrent_rounds():
    return [
        dict(
            round=turn,
            session_id=session,
            concurrency=4,
            phase="followup",
            request_id=f"multi-run-s{session:03d}-{turn:03d}",
            ttft_s=10 * (session + 1) + turn,
        )
        for turn in (1, 2)
        for session in range(4)
    ]


def test_sixteen_request_cross_round_batch_remains_one_unlabelled_time_point(plot, tmp_path, monkeypatch):
    metadata = [
        dict(round=2 if i % 2 else 5, session_id=i, concurrency=16, request_id=f"run-s{i:03}", ttft_s=i + 1)
        for i in range(16)
    ]
    item = record()
    item["requests"] = [
        dict(req_id=row["request_id"], token_start=1000 * i, token_end=1000 * i + 512) for i, row in enumerate(metadata)
    ]
    chunks, diagnostic = plot.timeline_records([item], metadata)
    assert len(chunks) == 1 and chunks[0]["dma_ms"] == 10
    assert chunks[0]["request_count"] == 16 and chunks[0]["coverage"] == 0.5
    assert len(chunks[0]["requests"]) == 16
    assert diagnostic["cross_round_batch_records"] == 1 and diagnostic["missing_rounds"] == []
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved.append(figure))
    plot.plot_timeline(chunks, metadata, tmp_path / "mixed.png")
    assert list(saved[0].axes[0].lines[0].get_xdata()) == [0]
    assert not saved[0].axes[0].texts
    assert len(saved[0].axes[7].patches) == 2


def test_legacy_log_timestamps_are_explicitly_approximate_and_new_timestamp_wins(plot, tmp_path, capsys):
    (tmp_path / "summary.json").write_text(json.dumps(dict(rounds=rounds())), encoding="utf-8")
    item = record(turn=1)
    del item["batch_start_time_ns"]
    log = tmp_path / "legacy.log"
    log.write_text("(Worker) [2026-10-22 12:00:00,123] INFO [PREFILL_OVERLAP] " + json.dumps(item), encoding="utf-8")
    records, _ = plot.read_records([log], include_log_time=True)
    chunks, diagnostic = plot.timeline_records(records, rounds())
    assert chunks[0]["time_source"] == "log_emission_approximate"
    assert diagnostic["approximate_timestamp_records"] == 1
    pytest.importorskip("matplotlib")
    assert (
        plot.main([str(log), "--run-dir", str(tmp_path), "--aggregate-ranks", "--output", str(tmp_path / "old.png")])
        == 0
    )
    assert "LOG EMISSION" in capsys.readouterr().err


def test_ttft_boxes_use_session_samples_and_report_partial_rounds(plot, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    metadata = concurrent_rounds()[:-1]  # Interrupted after three of four responses in round 2.
    (tmp_path / "summary.json").write_text(json.dumps(dict(concurrency=4, rounds=metadata)), encoding="utf-8")
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved.append(figure))
    output = tmp_path / "ttft.png"
    assert plot.main(["--run-dir", str(tmp_path), "--ttft-only", "--output", str(output)]) == 0
    axis = saved[0].axes[0]
    assert len(axis.patches) == 2  # Boxes, not a mean-only point curve.
    assert any("p50 26.00" in text.get_text() and "p99 40.70" in text.get_text() for text in axis.texts)
    assert any("n=3/4" in text.get_text() for text in axis.texts)
    with output.with_suffix(".csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["ttft_samples"] for row in rows] == ["4", "3"]
    assert [row["round"] for row in rows] == ["1", "2"]


def test_independent_session_progress_keeps_round_distributions_without_a_barrier(plot, tmp_path):
    metadata = [
        dict(round=turn, session_id=session, concurrency=2, request_id=f"run-s{session}-{turn}", ttft_s=turn + session)
        for session, turn in ((1, 1), (1, 2), (1, 3), (0, 1))
    ]
    (tmp_path / "summary.json").write_text(
        json.dumps(dict(status="running", concurrency=2, rounds=metadata)), encoding="utf-8"
    )
    rows = plot.ttft_summary(plot.read_rounds(tmp_path))
    assert [r["round"] for r in rows] == [1, 2, 3]
    assert [r["ttft_samples"] for r in rows] == [2, 1, 1]
    assert [r["concurrency"] for r in rows] == [2, 2, 2]
