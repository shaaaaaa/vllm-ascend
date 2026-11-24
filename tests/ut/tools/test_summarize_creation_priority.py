# SPDX-License-Identifier: Apache-2.0
import csv
import json
from pathlib import Path

import pytest


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    import summarize_creation_priority

    return summarize_creation_priority


def fixture_rows(mod):
    # Reuse one model ID for all six builds, as on the 910B3.
    for case in range(6):
        for replay in range(mod.REPLAYS + 1):
            t = 1794888718310773000 + (case * (mod.REPLAYS + 1) + replay) * 100000
            # Give the warm-up a different order so including it is detectable.
            delay = 7000 if not replay else (0, 4200, -4200)[case % 3]
            for lane in range(2):
                stream = ("76", "75")[lane if case < 3 else 1 - lane]
                shift = max(delay, 0) if lane == 1 else max(-delay, 0)
                for index in range(mod.STEPS):
                    start_ns = t + index * 1000 + shift
                    yield stream, f"{start_ns // 1000}.{start_ns % 1000:03d}", "1.000", index


def make_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.writer(out)
        writer.writerow(
            ["Model ID", "Stream ID", "OP Type", "Op Name", "Task Start Time(us)", "Task Duration(us)", "Block Dim"]
        )
        writer.writerow([0, 87, "Abs", "abs_warm", "1.001", 5, 8])
        for stream, start, length, index in rows:
            writer.writerow([49, stream, "Abs", f"aclnnAbs_AbsAiCore/{index}", start + "\t", length, 24])


def check_cases(report):
    cases = report["cases"]
    assert [c["priority"] for c in cases] == ["equal", "A_high", "B_high"] * 2
    assert [c["b_minus_a_half_us"] for c in cases] == [0, 4.2, -4.2] * 2
    assert [c["b_minus_a_last_us"] for c in cases] == [0, 4.2, -4.2] * 2
    assert [c["a_lead_at_half_kernels"] for c in cases] == [0, 5, -5] * 2
    assert [c["interval_overlap_pct"] for c in cases] == [100, 91.25, 91.25] * 2
    assert cases[0]["ready_start_counts"] == {"a": 0, "b": 0, "tie": 47 * 5}
    assert report["block_dim"] == ["24"]


def test_start_counts_follow_device_timestamps_after_both_predecessors_end(analyzer):
    a = [(10, 15), (30, 35), (60, 65), (80, 85)]
    b = [(11, 20), (40, 45), (50, 55), (70, 75)]
    counts = {"a": 0, "b": 0, "tie": 0}
    analyzer.count_ready_starts(a, b, counts)
    assert counts == {"a": 2, "b": 3, "tie": 0}


def test_split_csv_slices_reused_model_streams_and_warmup(analyzer, tmp_path):
    export = tmp_path / "PROF_000001" / "mindstudio_profiler_output"
    export.mkdir(parents=True)
    rows = list(fixture_rows(analyzer))
    make_csv(export / "op_summary_1.csv", rows[:1700])
    make_csv(export / "op_summary_2.csv", rows[1699:])
    check_cases(analyzer.analyze(tmp_path, ["76", "75"]))


def test_incomplete_summary_falls_back_to_device_timeline(analyzer, tmp_path):
    rows = list(fixture_rows(analyzer))
    make_csv(tmp_path / "op_summary_1.csv", rows[:6])
    timeline = [{"name": "abs", "ph": "X", "pid": 12, "tid": 76, "ts": 1, "dur": 5, "cat": "cpu_op"}]
    for stream, start, length, _ in rows:
        timeline.append(
            {
                "name": "Abs_f00d_0",
                "ph": "X",
                "pid": 34,
                "tid": 999,
                "ts": start,
                "dur": length,
                "args": {"Physic Stream Id": stream, "Block Dim": 24},
            }
        )
    (tmp_path / "msprof_1.json").write_text(json.dumps({"traceEvents": timeline}))
    result = analyzer.analyze(tmp_path, ["76", "75"])
    assert result["sources"] == ["msprof_1.json"]
    check_cases(result)


def test_missing_kernels_and_two_capture_directories_report_the_ambiguity(analyzer, tmp_path):
    rows = list(fixture_rows(analyzer))
    p = tmp_path / "op_summary_1.csv"
    make_csv(p, rows[:-1])
    with pytest.raises(ValueError, match="1727"):
        analyzer.analyze(p, ["76", "75"])
    for capture in ("PROF_1", "PROF_2"):
        (tmp_path / capture).mkdir()
    with pytest.raises(ValueError, match="2 captures"):
        analyzer.analyze(tmp_path, ["76", "75"])


def test_catches_rows_from_the_following_replay_with_matching_total_count(analyzer, tmp_path):
    rows = list(fixture_rows(analyzer))
    stream, _, length, index = rows[0]
    rows[0] = stream, "1794888718310888.350", length, index
    p = tmp_path / "op_summary_1.csv"
    make_csv(p, rows)
    with pytest.raises(ValueError, match="boundary"):
        analyzer.analyze(p, ["76", "75"])
