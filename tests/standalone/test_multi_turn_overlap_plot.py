# SPDX-License-Identifier: Apache-2.0
"""Match actual dialogue IDs and preserve per-round measurement semantics."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def plot(monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("multi_turn_overlap_test", tools / "plot_multi_turn_overlap.py")
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
        )
        for i in range(3)
    ]


def record(turn=0, tp=0, step=1, coverage=0.5, samples=2, duration=10):
    return dict(
        schema_version=1,
        status="complete",
        host="host",
        pid=10 + tp,
        dp_rank=0,
        tp_rank=tp,
        step=step,
        requests=[
            dict(req_id=f"cmpl-multi-run-{turn:03d}-0-internal", token_start=(step - 1) * 4096, token_end=step * 4096)
        ],
        coverage=coverage,
        layer_samples=samples,
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


def test_chunk_times_sum_per_round_and_coverage_keeps_mean_of_layer_ratios(plot):
    records = [
        record(coverage=0.9, samples=1),
        record(step=2, coverage=0.3, samples=3, duration=30),
        record(turn=1, duration=7),
    ]
    rows, chunks, diagnostics = plot.aggregate_rounds(list(reversed(records)), rounds())
    assert [r["round"] for r in rows] == [0, 1]
    assert rows[0]["coverage"] == pytest.approx(0.45)
    assert rows[0]["save_ms"] == 40  # Sum, not mean of the two chunks.
    assert rows[1]["save_ms"] == 7
    assert rows[0]["worker_count"] == 1 and rows[0]["chunk_records"] == 2
    assert [r["round"] for r in chunks] == [0, 0, 1]
    assert diagnostics["missing_rounds"] == [2]


def test_aggregate_ranks_averages_complete_worker_round_totals(plot):
    records = [
        record(tp=0),
        record(tp=0, step=2, duration=30),
        record(tp=1, duration=100),
        record(tp=1, step=2, duration=200),
    ]
    rows, _, _ = plot.aggregate_rounds(records, rounds(), aggregate_ranks=True)
    assert rows[0]["worker_count"] == 2 and rows[0]["chunk_records"] == 4
    assert rows[0]["save_ms"] == 170  # ((10 + 30) + (100 + 200)) / 2.
    assert rows[0]["coverage"] == 0.5
    rows, _, _ = plot.aggregate_rounds(records, rounds(), tp_rank=1)
    assert rows[0]["save_ms"] == 300 and rows[0]["tp_rank"] == 1


def test_mixed_batches_and_other_requests_are_excluded(plot):
    mixed = record()
    mixed["requests"].append(dict(req_id="another", token_start=0, token_end=4096))
    unknown = record()
    unknown["requests"][0]["req_id"] = "unrelated"
    rows, _, diagnostic = plot.aggregate_rounds([mixed, unknown, record(turn=1)], rounds())
    assert [r["round"] for r in rows] == [1]
    assert diagnostic["mixed_request_records"] == 1
    assert diagnostic["unmatched_request_examples"] == ["unrelated"]


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


def test_real_cli_deduplicates_logs_and_exports_rounds_and_chunks(plot, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    saved = []
    original = Figure.savefig

    def save(figure, *args, **kwargs):
        saved.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    (tmp_path / "summary.json").write_text(json.dumps(dict(rounds=rounds())), encoding="utf-8")
    records = [
        record(coverage=0.9, samples=1),
        record(step=2, coverage=0.3, samples=3, duration=30),
        record(turn=2, duration=7),
    ]
    log = tmp_path / "server.log"
    line = "\x1b[32m(Worker) [PREFILL_OVERLAP] "
    log.write_text("\n".join(line + json.dumps(r) + "\x1b[0m" for r in records + records), encoding="utf-8")
    output = tmp_path / "rounds.png"
    assert plot.main([str(log), "--run-dir", str(tmp_path), "--output", str(output)]) == 0
    assert output.stat().st_size > 0
    figure = saved[0]
    assert len(figure.axes) == 6
    assert list(figure.axes[0].lines[0].get_xdata()) == [0, 1, 2]
    y = figure.axes[2].lines[0].get_ydata()
    assert y[0] == 40 and y[2] == 7
    assert str(y[1]) == "nan"  # Missing round breaks the curve; it is not zero.
    diagnostic = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert diagnostic["missing_rounds"] == [1]
    with output.with_suffix(".chunks.csv").open(encoding="utf-8") as source:
        assert len(list(csv.DictReader(source))) == 3
    assert (
        plot.main(
            [str(log), "--run-dir", str(tmp_path), "--start-round", "1", "--output", str(tmp_path / "followups.png")]
        )
        == 0
    )
    assert list(saved[-1].axes[0].lines[0].get_xdata()) == [1, 2]


def test_unmatched_logs_fail_clearly(plot, tmp_path, capsys):
    (tmp_path / "summary.json").write_text(json.dumps(dict(rounds=rounds())), encoding="utf-8")
    log = tmp_path / "server.log"
    log.write_text("ordinary log", encoding="utf-8")
    assert plot.main([str(log), "--run-dir", str(tmp_path)]) == 2
    assert "No overlap records matched" in capsys.readouterr().err
