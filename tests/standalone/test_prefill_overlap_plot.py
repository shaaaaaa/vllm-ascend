# SPDX-License-Identifier: Apache-2.0
"""Offline log parsing/aggregation does not need an NPU or matplotlib."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "tools/plot_prefill_overlap.py"
_SPEC = importlib.util.spec_from_file_location("prefill_overlap_plot_test", _PATH)
plot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plot)


def record(coverage=0.5, samples=2, step=1, tp=0, start=0, end=4096):
    return dict(
        schema_version=1,
        status="complete",
        step=step,
        host="host",
        pid=1,
        tp_rank=tp,
        dp_rank=0,
        coverage=coverage,
        layer_samples=samples,
        requests=[dict(req_id="r", token_start=start, token_end=end)],
        dma_ms=20,
        load_ms=10,
        save_ms=10,
        wait_ms=5,
        interval_ms=40,
        first_layer_wait_ms=3,
        all_wait_ms=8,
    )


def test_prefixed_logs_deduplicate_and_exclude_incomplete(tmp_path):
    log = tmp_path / "server.log"
    line = "\x1b[32m(Worker_TP0 pid=1) INFO [PREFILL_OVERLAP] " + json.dumps(record())
    log.write_text(
        "other\n" + line + "\x1b[0m\n" + line + "\n"
        "[PREFILL_OVERLAP] broken\n"
        "[PREFILL_OVERLAP] " + json.dumps(dict(record(), status="incomplete")),
        encoding="utf-8",
    )
    records, ignored = plot.read_records([log, log])
    assert len(records) == 1
    assert ignored == 4


def test_rank_filter_and_weighted_mean_of_layer_ratios():
    records = [record(0.9, 1), record(0.3, 3, step=2), record(0.1, 2, tp=1)]
    rows, mixed = plot.aggregate_records(records, tp_rank=0)
    assert not mixed
    assert len(rows) == 1
    assert rows[0]["coverage"] == pytest.approx(0.45)
    assert rows[0]["worker_steps"] == 2
    rows, _ = plot.aggregate_records(records)
    assert len(rows) == 2


def test_any_multi_request_batch_is_excluded_even_identical_ranges():
    item = record()
    item["requests"].append(dict(req_id="b", token_start=0, token_end=4096))
    rows, skipped = plot.aggregate_records([item])
    assert rows == []
    assert skipped == 1


def test_partial_chunks_and_exact_cache_hit_start_are_not_rebucketed(tmp_path):
    records = [record(start=256, end=4096), record(step=2, start=4096, end=5043)]
    rows, _ = plot.aggregate_records(records, aggregate_ranks=True)
    assert [(row["token_start"], row["token_end"]) for row in rows] == [
        (256, 4096),
        (4096, 5043),
    ]
    path = tmp_path / "result.csv"
    plot.write_csv(rows, path)
    assert "5043" in path.read_text()


def test_no_matching_logs_is_a_clear_error(tmp_path, capsys):
    path = tmp_path / "empty.log"
    path.write_text("ordinary service output")
    assert plot.main([str(path)]) == 2
    assert "No complete" in capsys.readouterr().err


def test_first_layer_and_all_wait_details_are_preserved_without_inventing_missing_values():
    rows, _ = plot.aggregate_records([record(), record(step=2)])
    assert rows[0]["first_layer_wait_ms"] == 6 and rows[0]["all_wait_ms"] == 16
    legacy = record(step=2)
    del legacy["first_layer_wait_ms"], legacy["all_wait_ms"]
    rows, _ = plot.aggregate_records([record(), legacy])
    assert rows[0]["first_layer_wait_ms"] is None and rows[0]["all_wait_ms"] is None
    assert rows[0]["coverage"] == 0.5


def test_figure_shows_timings_per_worker_chunk_without_changing_csv_totals(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    second = dict(
        record(0.3, 3, step=2),
        dma_ms=40,
        load_ms=24,
        save_ms=16,
        wait_ms=15,
        interval_ms=80,
        first_layer_wait_ms=7,
        all_wait_ms=22,
    )
    rows, _ = plot.aggregate_records([record(0.9, 1), second])
    saved = []
    original_save = Figure.savefig

    def save(figure, *args, **kwargs):
        saved.append(figure)
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    output = tmp_path / "overlap.png"
    plot.plot_rows(rows, output)
    assert output.stat().st_size > 0
    axes = saved[0].axes
    assert len(axes) == 8
    expected = (0.45, 30, 13, 17, 15, 5, 10, 60)
    for axis, value in zip(axes, expected):
        assert list(axis.lines[0].get_ydata()) == pytest.approx([value])
        assert axis.get_xticklabels()[0].get_text() == "0-4,095"
    assert "save_ms" in axes[2].get_title()
    assert "load_ms" in axes[3].get_title()
    assert rows[0]["save_ms"] == 26
    assert rows[0]["load_ms"] == 34
    assert rows[0]["worker_steps"] == 2
