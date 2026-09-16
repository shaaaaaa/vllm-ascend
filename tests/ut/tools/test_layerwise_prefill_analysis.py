# SPDX-License-Identifier: Apache-2.0
"""CPU-only concurrency and bounded-I/O regressions for saved KV analysis."""

import hashlib
import importlib.util
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[3] / "tools"
SPEC = importlib.util.spec_from_file_location("prefill_analysis_under_test", TOOLS / "layerwise_prefill_check.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def save_rows(path, positions):
    torch.save({"positions": torch.tensor(positions), "values": torch.tensor(positions).double().reshape(-1, 1)}, path)
    return [path]


@pytest.mark.parametrize("cutoff", [3, 4])
def test_one_load_per_trace_and_cached_results_match_independent_comparisons(tmp_path, monkeypatch, cutoff):
    files = {
        "baseline": save_rows(tmp_path / "b.pt", [0, 1, 2, 3]),
        "prefill": save_rows(tmp_path / "p.pt", [2, 0, 1]),
        "decode": save_rows(tmp_path / "d.pt", [2, 3]),
        "loaded": save_rows(tmp_path / "l.pt", [0, 1]),
        "prefill_loaded": save_rows(tmp_path / "pl.pt", [0, 1]),
    }
    key = ("rank0", "layer0", "nope", "current")
    loader = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", loader)
    actual, chunks, errors = CHECK.compare_trace_job((key, files, 3, cutoff, 2))
    assert errors == []
    assert len(chunks) == 4
    counts = Counter(call.args[0] for call in loader.call_args_list)
    assert counts == {paths[0]: 1 for paths in files.values()}
    for row, (left, right, start, stop) in zip(
        actual,
        [
            ("baseline", "prefill", 0, 3),
            ("prefill", "loaded", 0, 3),
            ("baseline", "decode", 3, cutoff),
            ("baseline", "decode", 0, 3),
            ("prefill", "prefill_loaded", 0, 3),
        ],
        strict=True,
    ):
        reference = CHECK.compare_rows(files[left], files[right], start, stop)
        assert {name: row[name] for name in reference} == reference


def test_empty_comparison_does_not_open_any_files(monkeypatch):
    loader = Mock(side_effect=AssertionError("Empty interval must not read files"))
    monkeypatch.setattr(torch, "load", loader)
    result = CHECK.compare_rows([Path("absent-baseline.pt")], [Path("absent-decode.pt")], 9565, 9565)
    assert result["matched_rows"] == result["baseline_rows"] == result["candidate_rows"] == 0
    loader.assert_not_called()


def test_cached_intervals_keep_duplicate_write_validation(tmp_path):
    files = save_rows(tmp_path / "dup.pt", [0, 1, 1, 2])
    cache = {}
    assert CHECK.load_rows(files, 2, 3, cache)[0].tolist() == [2]
    with pytest.raises(RuntimeError, match="Duplicate"):
        CHECK.load_rows(files, 0, 3, cache)


def test_analysis_submission_is_concurrent_and_bounded():
    # Run the same scheduler with a thread executor to inspect submission without
    # IPC. A barrier proves overlap; an oversized executor exposes over-submit.
    barrier = threading.Barrier(2, timeout=10)
    lock = threading.Lock()
    active, maximum = 0, 0

    def task(index):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait()
        with lock:
            active -= 1
        return index

    with ThreadPoolExecutor(max_workers=8) as executor:
        submit = Mock(wraps=executor.submit)
        executor.submit = submit
        results = CHECK.analysis_results(task, list(range(8)), executor, 2, "test")
        first = next(results)
        assert submit.call_count == 2
        completed = [first, *results]
    assert sorted(result for _, result, _ in completed) == list(range(8))
    assert maximum == 2


def test_parallel_moment_merge_preserves_nonfinite_and_variance():
    data = torch.tensor([float("nan"), -1000.0, 3.0, 3.125, float("inf"), 2000.0], dtype=torch.float64)
    total, reference = CHECK.Moments(), CHECK.Moments()
    reference.add(data)
    for chunk in data.split(2):
        partial = CHECK.Moments()
        partial.add(chunk)
        total.merge(partial)
    assert total.result() == pytest.approx(reference.result())


def test_archive_batches_produce_same_statistics_as_unsplit(tmp_path, monkeypatch):
    for stage in ("baseline", "prefill"):
        directory = tmp_path / stage / "archive"
        directory.mkdir(parents=True)
        for index in range(5):
            values = torch.tensor([index, index + 0.5], dtype=torch.bfloat16)
            if stage == "prefill":
                values += 0.125
            torch.save(
                {
                    "key": str(index),
                    "worker_id": 0,
                    "kv_group": 0,
                    "layer_id": index % 2,
                    "shapes": [[2]],
                    "dtypes": ["torch.bfloat16"],
                    "fmt": 1,
                    "valid_tokens": 2,
                    "raw": values.view(torch.uint8),
                },
                directory / f"{index}.pt",
            )
    reference = CHECK.compare_archives(tmp_path)
    monkeypatch.setattr(CHECK, "ARCHIVE_BATCH_FILES", 2)
    batches = []
    result = CHECK.compare_archives(tmp_path, on_progress=batches.append)
    assert len(batches) == 3
    assert result["common_keys"] == reference["common_keys"] == 5
    for actual, expected in zip(result["per_layer"], reference["per_layer"], strict=True):
        for metric in expected["stats"]:
            assert actual["stats"][metric] == pytest.approx(expected["stats"][metric])


def test_summary_is_available_before_work_and_preserved_on_failure(tmp_path, monkeypatch):
    CHECK.write_json(tmp_path / "prompt.json", {"length": 4, "sha256": "same"})
    for stage in CHECK.STAGES:
        (tmp_path / stage).mkdir()
        CHECK.write_json(
            tmp_path / stage / "output.json",
            {
                "prompt_sha256": "same",
                "token_ids": [7 if stage == "baseline" else 9],
                "num_hidden_layers": 1,
            },
        )
    monkeypatch.setattr(CHECK, "read_index", lambda path: {("rank0", "layer0", "nope", "current"): []})
    monkeypatch.setattr(CHECK, "validate_reload", lambda *args: {})
    monkeypatch.setattr(CHECK, "verify_archive_reads", lambda *args: None)

    def fail(job):
        partial = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert partial["analysis_status"] == "running"
        assert partial["first_different_output_token_index"] == 0
        raise RuntimeError("corrupt tensor file")

    monkeypatch.setattr(CHECK, "compare_trace_job", fail)
    with pytest.raises(RuntimeError, match="corrupt tensor file"):
        CHECK.analyse(tmp_path, 1, workers=1)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["analysis_status"] == "failed"
    assert summary["analysis_error"] == "corrupt tensor file"
    assert summary["prefill_first_token_equal"] is False


def test_analysis_worker_limit_is_positive(tmp_path):
    assert CHECK.parser().parse_args([]).analysis_workers == 64
    assert CHECK.parser().parse_args(["--analysis-workers", "4"]).analysis_workers == 4
    with pytest.raises(ValueError, match="positive"):
        CHECK.analyse(tmp_path, 1, workers=0)


def test_real_9565_token_three_chunk_drift_and_prefill_reload_corruption(tmp_path):
    length = 9565
    base = torch.arange(length, dtype=torch.float64).reshape(-1, 1)
    prefill = base.clone()
    prefill[4096:8192] += 1
    prefill[8192:] += 2
    reloaded = prefill[:8192].clone()
    reloaded[4100] += 10  # Only the reload is damaged, not P's own original KV.
    files = {}
    for name, pos, values in (
        ("baseline", torch.arange(length), base),
        ("prefill", torch.arange(length), prefill),
        ("prefill_loaded", torch.arange(8192), reloaded),
        ("loaded", torch.arange(length), prefill),
        ("decode", torch.tensor([9564]), prefill[-1:]),
    ):
        path = tmp_path / f"{name}.pt"
        torch.save({"positions": pos, "values": values}, path)
        files[name] = [path]
    hashes = {name: hashlib.sha256(paths[0].read_bytes()).hexdigest() for name, paths in files.items()}
    rows, chunks, errors = CHECK.compare_trace_job(
        (("rank0", "model.layers.2.self_attn.attn", "nope", "current"), files, length, length, 4096)
    )
    assert errors == []
    written = [row for row in chunks if row["comparison"] == "prefill_written"]
    assert [(r["position_start"], r["position_end_exclusive"]) for r in written] == [
        (0, 4096),
        (4096, 8192),
        (8192, 9565),
    ]
    assert [r["matched_rows"] for r in written] == [4096, 4096, 1373]
    assert [r["stats"]["abs_diff"]["mean"] for r in written] == [0, 1, 2]
    reloads = [row for row in chunks if row["comparison"] == "prefill_reloaded"]
    assert reloads[0]["stats"]["abs_diff"]["max"] == 0
    assert reloads[1]["stats"]["abs_diff"]["max"] == 10
    assert reloads[1]["stats"]["abs_diff"]["mean"] == pytest.approx(10 / 4096)
    assert reloads[2]["status"] == "not_observed" and "stats" not in reloads[2]
    total_reload = next(row for row in rows if row["comparison"] == "prefill_reloaded")
    assert total_reload["matched_rows"] == 8192
    assert total_reload["baseline_only_rows"] == 1373
    assert total_reload["stats"]["abs_diff"]["mean"] == pytest.approx(10 / 8192)
    total_written = next(row for row in rows if row["comparison"] == "prefill_written")
    reference = CHECK.tensor_statistics(base, prefill)
    for metric in reference:
        assert total_written["stats"][metric] == pytest.approx(reference[metric])
    assert {name: hashlib.sha256(paths[0].read_bytes()).hexdigest() for name, paths in files.items()} == hashes


def test_nonfinite_statistics_survive_position_band_merge(tmp_path):
    positions = torch.arange(6)
    base = torch.tensor([0, 1, float("nan"), 3, 4, 5], dtype=torch.float64).reshape(-1, 1)
    other = torch.tensor([0, float("inf"), 2, 2.5, 4, 6], dtype=torch.float64).reshape(-1, 1)
    for path, values in ((tmp_path / "a.pt", base), (tmp_path / "b.pt", other)):
        torch.save({"positions": positions, "values": values}, path)
    cache = {}
    reports = [CHECK.compare_rows([tmp_path / "a.pt"], [tmp_path / "b.pt"], i, i + 2, cache) for i in range(0, 6, 2)]
    combined = CHECK.merge_comparisons(reports)
    reference = CHECK.tensor_statistics(base, other)
    for metric in reference:
        assert combined["stats"][metric] == pytest.approx(reference[metric])
    assert combined["stats"]["abs_diff"]["nonfinite"] == 2
    json.dumps(combined, allow_nan=False)


@pytest.mark.parametrize(
    "source,field",
    [
        ("run.json", "prefill_chunk_tokens"),
        ("prefill/engine_options.json", "max_num_batched_tokens"),
        ("baseline/engine_options.json", "max_num_batched_tokens"),
    ],
)
def test_analysis_uses_original_compute_chunk_size(tmp_path, source, field):
    path = tmp_path / source
    path.parent.mkdir(parents=True, exist_ok=True)
    CHECK.write_json(path, {field: 3072})
    assert CHECK.saved_prefill_chunk_size(tmp_path) == (3072, source)


def test_prefill_engine_options_win_and_legacy_fallback_is_explicit(tmp_path):
    assert CHECK.saved_prefill_chunk_size(tmp_path) == (4096, "fallback_default_4096")
    CHECK.write_json(tmp_path / "run.json", {"prefill_chunk_tokens": 4096})
    (tmp_path / "prefill").mkdir()
    CHECK.write_json(tmp_path / "prefill/engine_options.json", {"max_num_batched_tokens": 2048})
    assert CHECK.saved_prefill_chunk_size(tmp_path) == (2048, "prefill/engine_options.json")


@pytest.mark.parametrize("bad", [0, -1, 3.5, "4096", True])
def test_invalid_saved_chunk_size_is_not_silently_replaced(tmp_path, bad):
    CHECK.write_json(tmp_path / "run.json", {"prefill_chunk_tokens": bad})
    with pytest.raises(ValueError, match="Invalid saved"):
        CHECK.saved_prefill_chunk_size(tmp_path)


def test_layers_are_sorted_numerically_and_missing_reload_is_not_printed_as_zero(capsys):
    rows = [
        {
            "rank": "rank0",
            "layer": f"model.layers.{index}.self_attn.attn",
            "part": "nope",
            "comparison": "prefill_reloaded",
            "status": "not_observed",
            "matched_rows": 0,
            "candidate_rows": 0,
        }
        for index in (10, 2, 1)
    ]
    rows.sort(key=CHECK.kv_row_sort_key)
    assert [row["layer"] for row in rows] == [f"model.layers.{index}.self_attn.attn" for index in (1, 2, 10)]
    CHECK.print_prefill_diagnostics(rows, [])
    output = capsys.readouterr().out
    assert output.count("status=not_observed") == 3
    assert "mean=None,max=None" in output
    assert "mean=0" not in output


@pytest.mark.parametrize(
    "base,candidate,matched,status",
    [
        (10, 0, 0, "not_observed"),
        (0, 10, 0, "missing_reference"),
        (10, 10, 0, "no_overlap"),
        (10, 10, 5, "partial_overlap"),
        (10, 5, 5, "compared"),
    ],
)
def test_reload_coverage_status(base, candidate, matched, status):
    assert (
        CHECK.comparison_status(
            {
                "baseline_rows": base,
                "candidate_rows": candidate,
                "matched_rows": matched,
                "candidate_only_rows": candidate - matched,
            }
        )
        == status
    )
