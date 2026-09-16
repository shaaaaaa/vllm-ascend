# SPDX-License-Identifier: Apache-2.0
"""CPU-only concurrency and bounded-I/O regressions for saved KV analysis."""

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
    }
    key = ("rank0", "layer0", "nope", "current")
    loader = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", loader)
    actual, errors = CHECK.compare_trace_job((key, files, 3, cutoff))
    assert errors == []
    counts = Counter(call.args[0] for call in loader.call_args_list)
    assert counts == {paths[0]: 1 for paths in files.values()}
    for row, (left, right, start, stop) in zip(
        actual,
        [
            ("baseline", "prefill", 0, 3),
            ("prefill", "loaded", 0, 3),
            ("baseline", "decode", 3, cutoff),
            ("baseline", "decode", 0, 3),
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
