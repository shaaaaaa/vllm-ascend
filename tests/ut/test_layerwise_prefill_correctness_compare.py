# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests; run with pytest --noconftest to avoid NPU worker setup."""

import copy
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "layerwise_prefill_correctness_compare.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("correctness_compare_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _fixture(root):
    _write(
        root / "model_info.json",
        {"num_hidden_layers": 2, "indexer_types": ["full", "shared"], "index_topk_pattern": ["F", "S"]},
    )
    roles = [
        {"kind": kind, "name": name, "layers": layers, "steps": selector}
        for kind, names, layers, selector in (
            ("decoder", ("input", "output", "positions"), [0, 1], "all"),
            ("sfa", ("input", "output"), [0, 1], "all"),
            ("attention", ("query_nope", "query_rope", "topk", "output"), [0, 1], "all"),
            ("indexer", ("query", "weights", "topk"), [0], "all"),
            ("indexer_input", ("x", "q_c"), [0], "all"),
            ("kv_current", ("nope", "rope"), [0, 1], "all"),
            ("kv_loaded", ("nope", "rope"), [0, 1], "history"),
            ("kv_current", ("index", "scale"), [0], "all"),
            ("kv_loaded", ("index", "scale"), [0], "history"),
        )
        for name in names
    ]
    result = {
        "completed": True,
        "prompt_token_ids": list(range(8)),
        "prompt_length": 8,
        "token_ids": [91],
        "output_length": 1,
        "text": "word",
        "num_cached_tokens": 0,
    }
    for case in ("off", "on"):
        directory = root / case
        _write(directory / "result.json", result)
        _write(directory / "engine_options.json", {"tensor_parallel_size": 2})
        _write(
            directory / "environment.json",
            {
                "PYTHONHASHSEED": "0",
                "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": "true" if case == "on" else "false",
                "LMCACHE_STORE_ASYNC": "true" if case == "on" else "false",
            },
        )
        coverage = []
        for rank in range(2):
            records = []
            rank_dir = directory / "tensors" / f"rank{rank}"
            rank_dir.mkdir(parents=True)
            for step in range(2):
                span = [4 * step, 4 * (step + 1)]
                for role in roles:
                    if role["steps"] == "history" and step == 0:
                        continue
                    for layer in role["layers"]:
                        shape = [4, 2] if role["kind"].startswith("kv_") else [2]
                        tensor = torch.ones(shape, dtype=torch.int64 if role["name"] == "topk" else torch.float32)
                        statistics = compare.compare_tensor_values(tensor, tensor)
                        record = {
                            "rank": rank,
                            "step": step,
                            "layer": layer,
                            "kind": role["kind"],
                            "name": role["name"],
                            "span": span,
                            "shape": shape,
                            "dtype": str(tensor.dtype),
                            "numel": tensor.numel(),
                            "nonfinite": 0,
                            "path": None,
                        }
                        if role["kind"].startswith("kv_"):
                            record.update(
                                positions=span if role["kind"] == "kv_current" else [0, span[0]], token_axis=0
                            )
                        if case == "off":
                            path = rank_dir / f"{len(records)}.pt"
                            torch.save(tensor, path)
                            record["path"] = path.relative_to(directory).as_posix()
                        else:
                            record["comparison"] = statistics
                        records.append(record)
            _manifest(rank_dir / "index.jsonl", records)
            coverage.append(
                {
                    "rank": rank,
                    "complete": True,
                    "records": len(records),
                    "model_num_hidden_layers": 2,
                    "prompt_length": 8,
                    "decoder_layers": [0, 1],
                    "sfa_layers": [0, 1],
                    "indexer_layers": [0],
                    "indexer_cache_layers": [0],
                    "scale_layers": [0],
                    "layout": {
                        "layout": "merged",
                        "remote_url": None,
                        "native_mooncake": False,
                        "production_transfer_unchanged": True,
                    },
                    "merged_load_sources": 2,
                    "legacy_load_sources": 0,
                    "steps": [
                        {"step": step, "phase": "prefill", "span": [4 * step, 4 * (step + 1)]} for step in range(2)
                    ],
                    "required_roles": roles,
                }
            )
        _write(directory / "coverage.json", coverage)
    return root


def _records(root, case="on", rank=0):
    path = root / case / "tensors" / f"rank{rank}" / "index.jsonl"
    return path, [json.loads(line) for line in path.read_text().splitlines()]


def test_full_coverage_passes_without_on_tensors(tmp_path, capsys):
    report = compare.compare_runs(_fixture(tmp_path))
    assert report["passed"] and report["complete"] and report["output_tokens_equal"]
    assert report["status"] == "equal"
    assert report["numeric_tolerance_applied"] is False
    assert report["counts"]["compared"] > 0
    assert json.loads((tmp_path / "report.json").read_text())["passed"]
    assert (tmp_path / "diff.jsonl").read_text() == ""
    compare.print_report(report)
    assert len(capsys.readouterr().out.splitlines()) <= 20


def test_float_difference_reports_distribution_without_tolerance_verdict(tmp_path):
    _fixture(tmp_path)
    path, records = _records(tmp_path)
    records[0]["comparison"] = compare.compare_tensor_values(torch.ones(2), torch.tensor([1.0, 1.5]))
    _manifest(path, records)
    report = compare.compare_runs(tmp_path)
    assert report["passed"] and report["status"] == "complete_with_differences"
    assert report["counts"]["floating_differences"] == 1
    assert report["first_divergence"]["step"] == 0
    assert report["worst"]["relative_l2"]["value"] == pytest.approx(math.sqrt(0.125))
    assert json.loads((tmp_path / "diff.jsonl").read_text())["comparison"]["abs_diff"]["p95"] > 0


def test_topk_mismatch_is_separate_from_float_difference(tmp_path):
    _fixture(tmp_path)
    path, records = _records(tmp_path)
    record = next(record for record in records if record["name"] == "topk")
    record["comparison"] = compare.compare_tensor_values(torch.tensor([1, 1]), torch.tensor([1, 2]))
    _manifest(path, records)
    report = compare.compare_runs(tmp_path)
    assert report["passed"] and report["counts"]["integer_differences"] == 1
    assert report["first_divergence"]["comparison"]["mismatch_rate"] == 0.5


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "duplicate",
        "bad_json",
        "bad_shape",
        "bad_dtype",
        "missing_comparison",
        "missing_file",
        "traversal",
        "missing_rank",
    ],
)
def test_structural_failures_fail_closed(tmp_path, mutation):
    _fixture(tmp_path)
    path, records = _records(tmp_path)
    if mutation == "missing":
        records.pop()
    elif mutation == "extra":
        record = copy.deepcopy(records[0])
        record["name"] = "unexpected_extra"
        records.append(record)
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "bad_shape":
        records[0]["shape"] = [3]
    elif mutation == "bad_dtype":
        records[0]["dtype"] = "torch.float16"
    elif mutation == "missing_comparison":
        del records[0]["comparison"]
    elif mutation in ("missing_file", "traversal"):
        path, records = _records(tmp_path, "off")
        records[0]["path"] = "missing.pt" if mutation == "missing_file" else "../escape.pt"
    elif mutation == "missing_rank":
        path = tmp_path / "on" / "tensors" / "rank1" / "index.jsonl"
        path.unlink()
    if mutation != "missing_rank":
        _manifest(path, records)
    if mutation == "bad_json":
        with path.open("a") as stream:
            stream.write('{"partial":')
    report = compare.compare_runs(tmp_path)
    assert not report["complete"] and not report["passed"]
    assert report["output_tokens_equal"]


@pytest.mark.parametrize(
    "mutation",
    ["last_layer", "last_step", "loaded_history", "role", "incomplete", "wrong_kv_positions", "cached_prompt"],
)
def test_common_omissions_in_both_runs_cannot_pass(tmp_path, mutation):
    _fixture(tmp_path)
    for case in ("off", "on"):
        coverage_path = tmp_path / case / "coverage.json"
        coverage = json.loads(coverage_path.read_text())
        for rank in range(2):
            path, records = _records(tmp_path, case, rank)
            if mutation == "last_layer":
                records = [record for record in records if record["layer"] != 1]
                coverage[rank]["decoder_layers"] = [0]
            elif mutation == "last_step":
                records = [record for record in records if record["step"] == 0]
                coverage[rank]["steps"].pop()
            elif mutation == "loaded_history":
                records = [record for record in records if record["kind"] != "kv_loaded"]
            elif mutation == "role":
                coverage[rank]["required_roles"] = [
                    role for role in coverage[rank]["required_roles"] if role["kind"] != "attention"
                ]
                records = [record for record in records if record["kind"] != "attention"]
            elif mutation == "incomplete":
                coverage[rank]["complete"] = False
            elif mutation == "wrong_kv_positions":
                next(record for record in records if record["kind"] == "kv_loaded")["positions"] = [1, 4]
            else:
                result_path = tmp_path / case / "result.json"
                result = json.loads(result_path.read_text())
                result["num_cached_tokens"] = 4
                _write(result_path, result)
            coverage[rank]["records"] = len(records)
            _manifest(path, records)
        _write(coverage_path, coverage)
    report = compare.compare_runs(tmp_path)
    assert not report["complete"] and not report["passed"]


def test_new_nonfinite_and_changed_tokens_have_distinct_status(tmp_path):
    _fixture(tmp_path)
    path, records = _records(tmp_path)
    records[0]["comparison"] = compare.compare_tensor_values(torch.ones(2), torch.tensor([1.0, float("nan")]))
    records[0]["nonfinite"] = 1
    _manifest(path, records)
    report = compare.compare_runs(tmp_path)
    assert report["complete"] and not report["passed"] and report["status"] == "new_nonfinite"
    assert report["counts"]["new_nonfinite"] == 1
    records[0]["comparison"] = compare.compare_tensor_values(torch.ones(2), torch.ones(2))
    records[0]["nonfinite"] = 0
    _manifest(path, records)
    result_path = tmp_path / "on" / "result.json"
    result = json.loads(result_path.read_text())
    result["token_ids"] = [92]
    _write(result_path, result)
    report = compare.compare_runs(tmp_path)
    assert report["complete"] and not report["passed"] and report["status"] == "output_tokens_changed"


def test_tensor_statistics_use_full_values_and_population_std():
    result = compare.compare_tensor_values(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 2.0, 5.0]))
    assert result["baseline"]["mean"] == 2
    assert result["baseline"]["std"] == pytest.approx(math.sqrt(2 / 3))
    assert result["candidate"]["mean"] == 3
    assert result["abs_diff"]["mean"] == 1
    assert result["abs_diff"]["max"] == 2
    assert result["abs_diff"]["p50"] == 1
    assert result["rmse"] == pytest.approx(math.sqrt(5 / 3))
    assert result["relative_l2"] == pytest.approx(math.sqrt(5 / 14))
    assert result["rmse_over_std"] == pytest.approx(math.sqrt(2.5))


def test_bfloat16_widened_before_subtraction_and_zero_denominator():
    left, right = torch.tensor([256.0], dtype=torch.bfloat16), torch.tensor([1.0], dtype=torch.bfloat16)
    assert compare.compare_tensor_values(left, right)["abs_diff"]["max"] == 255
    result = compare.compare_tensor_values(torch.zeros(2), torch.ones(2))
    assert result["relative_l2"] is None and result["rmse_over_std"] is None
    assert compare.compare_tensor_values(torch.zeros(2), torch.zeros(2))["relative_l2"] == 0


def test_nonfinite_existing_vs_new_and_json_finite():
    result = compare.compare_tensor_values(
        torch.tensor([float("nan"), 1.0]), torch.tensor([float("nan"), float("inf")])
    )
    assert result["baseline"]["nonfinite"] == 1
    assert result["candidate"]["nonfinite"] == 2
    assert result["new_nonfinite"] == 1
    json.dumps(result, allow_nan=False)


def test_nonfinite_type_and_position_changes_are_explicit():
    result = compare.compare_tensor_values(
        torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0]),
        torch.tensor([float("nan"), float("inf"), float("nan"), 0.0]),
    )
    assert result["new_nonfinite"] == 0
    assert result["nonfinite_pattern_changed"] == 2
    assert result["baseline"]["nan"] == 1 and result["candidate"]["nan"] == 2
    assert result["baseline"]["neginf"] == 1 and result["candidate"]["neginf"] == 0
    assert result["mismatched"] == 2


def test_float8_cpu_uses_widened_comparison():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch has no float8")
    result = compare.compare_tensor_values(torch.tensor([1.0, 2.0]).to(dtype), torch.tensor([1.0, 3.0]).to(dtype))
    assert result["mismatched"] == 1 and result["abs_diff"]["max"] == 1


def test_large_integer_identity_and_tensor_metadata_errors():
    result = compare.compare_tensor_values(torch.tensor([2**60, 2]), torch.tensor([2**60 + 1, 2]))
    assert result["category"] == "integer" and result["mismatched"] == 1 and result["mismatch_rate"] == 0.5
    assert not compare.compare_tensor_values(torch.ones(2), torch.ones(3))["comparable"]
    assert not compare.compare_tensor_values(torch.ones(2), torch.ones(2).half())["comparable"]


def test_histogram_percentiles_include_every_element(monkeypatch):
    monkeypatch.setattr(compare, "EXACT_PERCENTILE_LIMIT", 3)
    monkeypatch.setattr(compare, "VALUE_BLOCK_SIZE", 2)
    result = compare.compare_tensor_values(torch.zeros(8), torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 100.0]))
    error = result["abs_diff"]
    assert result["candidate"]["count"] == 8
    assert error["mean"] == 106 / 8
    assert error["percentile_method"] == "all_element_log_histogram_upper_bound"
    assert error["p50"] == 0 and error["p95"] == pytest.approx(100)
    assert error["percentile_bounds"]["p99"][0] <= 100 <= error["percentile_bounds"]["p99"][1]


def test_empty_directory_is_failed_report_not_exception(tmp_path):
    report = compare.compare_runs(tmp_path)
    assert not report["passed"] and not report["complete"]


@pytest.mark.parametrize("field,value", [("environment", {"PYTHONHASHSEED": "99"}), ("coverage", None), ("result", [])])
def test_malformed_or_mismatched_run_metadata(tmp_path, field, value):
    _fixture(tmp_path)
    _write(tmp_path / "on" / f"{field}.json", value)
    report = compare.compare_runs(tmp_path)
    assert not report["passed"] and not report["complete"]


@pytest.mark.parametrize(
    "mutation",
    [
        "sfa",
        "producer",
        "index_input",
        "kv_index",
        "scale_role",
        "layout",
        "remote",
        "native",
        "transfer",
        "no_source",
        "legacy",
        "same_path",
    ],
)
def test_independent_inventory_and_real_layout_evidence(tmp_path, mutation):
    _fixture(tmp_path)
    for case in ("off", "on"):
        path = tmp_path / case / "coverage.json"
        coverage = json.loads(path.read_text())
        for entry in coverage:
            if mutation == "sfa":
                entry["sfa_layers"] = [0]
            elif mutation == "producer":
                entry["indexer_layers"] = []
                entry["indexer_cache_layers"] = []
            elif mutation in ("index_input", "kv_index", "scale_role"):
                entry["required_roles"] = [
                    role
                    for role in entry["required_roles"]
                    if not (
                        role["kind"] == "indexer_input"
                        if mutation == "index_input"
                        else role["kind"].startswith("kv_")
                        and role["name"] == ("index" if mutation == "kv_index" else "scale")
                    )
                ]
            elif mutation == "layout":
                del entry["layout"]
            elif mutation == "remote":
                entry["layout"]["remote_url"] = "test://remote"
            elif mutation == "native":
                entry["layout"]["native_mooncake"] = True
            elif mutation == "transfer":
                entry["layout"]["production_transfer_unchanged"] = False
            elif mutation == "no_source":
                entry["merged_load_sources"] = 0
            elif mutation == "legacy":
                entry["legacy_load_sources"] = 1
            else:
                environment_path = tmp_path / case / "environment.json"
                env = json.loads(environment_path.read_text())
                env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] = "false"
                _write(environment_path, env)
        _write(path, coverage)
    assert not compare.compare_runs(tmp_path)["passed"]


def test_model_config_missing_or_conflicting_cannot_pass(tmp_path):
    _fixture(tmp_path)
    model_path = tmp_path / "model_info.json"
    model = json.loads(model_path.read_text())
    model["index_topk_pattern"] = ["F", "F"]
    _write(model_path, model)
    assert not compare.compare_runs(tmp_path)["passed"]
    model_path.unlink()
    assert not compare.compare_runs(tmp_path)["passed"]


def test_padding_archived_but_excluded_from_numeric_statistics(tmp_path):
    _fixture(tmp_path)
    for case in ("off", "on"):
        path, records = _records(tmp_path, case)
        record = records[0]
        record.update(
            comparison_slice={"axis": 0, "start": 0, "end": 1},
            comparison_shape=[1],
            comparison_numel=1,
            comparison_nonfinite=0,
        )
        if case == "on":
            record["nonfinite"] = 1  # Padding only; whole raw archive metadata remains visible.
            record["comparison"] = compare.compare_tensor_values(torch.ones(1), torch.ones(1))
        _manifest(path, records)
    report = compare.compare_runs(tmp_path)
    assert report["passed"]
    assert report["counts"]["archived_elements"] - report["counts"]["compared_elements"] == 1
    path, records = _records(tmp_path)
    records[0]["comparison_slice"]["end"] = 0
    _manifest(path, records)
    assert not compare.compare_runs(tmp_path)["passed"]


def test_logical_kv_cannot_be_treated_as_padding(tmp_path):
    _fixture(tmp_path)
    for case in ("off", "on"):
        path, records = _records(tmp_path, case)
        record = next(item for item in records if item["kind"] == "kv_loaded")
        record.update(
            comparison_slice={"axis": 0, "start": 0, "end": 1},
            comparison_shape=[1, 2],
            comparison_numel=2,
            comparison_nonfinite=0,
        )
        if case == "on":
            record["comparison"] = compare.compare_tensor_values(torch.ones(1, 2), torch.ones(1, 2))
        _manifest(path, records)
    assert not compare.compare_runs(tmp_path)["passed"]


def test_numeric_overflow_fails_as_uncomparable_without_invalid_json():
    stats = compare.compare_tensor_values(
        torch.tensor([1e300], dtype=torch.float64), torch.tensor([-1e300], dtype=torch.float64)
    )
    assert not stats["comparable"]
    json.dumps(stats, allow_nan=False)


def test_real_worker_archive_online_comparison_padding_contract(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_PATH.parent))
    monkeypatch.setitem(
        sys.modules,
        "layerwise_prefill_correctness_layout",
        SimpleNamespace(install_local_merged_layout=lambda: None, validate_local_merged_engine=lambda _: {}),
    )
    spec = importlib.util.spec_from_file_location(
        "correctness_archive_integration_test", MODULE_PATH.with_name("layerwise_prefill_correctness_worker.py")
    )
    assert spec and spec.loader
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    identity = dict(step=0, layer=0, kind="decoder", name="input", span=(0, 1), valid_rows=1)
    off = worker.TensorArchive(tmp_path / "off", 0)
    off.record(torch.tensor([[1.0], [999.0]]), **identity)
    off.close()
    on = worker.TensorArchive(tmp_path / "on", 0, compare=compare.compare_tensor_values)
    on.record(torch.tensor([[1.25], [float("nan")]]), **identity)
    on.close()
    left = json.loads(off.index_path.read_text())
    right = json.loads(on.index_path.read_text())
    compare._validate_record(left, 0, tmp_path / "off", True)
    compare._validate_record(right, 0, tmp_path / "on", False)
    assert torch.load(tmp_path / "off" / left["path"], weights_only=True).shape == (2, 1)
    assert right["nonfinite"] == 1 and right["comparison_nonfinite"] == 0
    assert right["comparison"]["numel"] == 1 and right["comparison"]["new_nonfinite"] == 0
    assert right["comparison"]["abs_diff"]["max"] == 0.25
    assert right["path"] is None and not on.errors


def test_external_off_archive_compares_without_copying_or_writing_old_run(tmp_path):
    old = _fixture(tmp_path / "old")
    renamed = old / "saved_reference"
    (old / "off").rename(renamed)
    current = tmp_path / "new"
    current.mkdir()
    shutil.copyfile(old / "model_info.json", current / "model_info.json")
    shutil.copytree(old / "on", current / "on")
    _write(current / "off_reference.json", {"schema": 1, "off_dir": str(renamed.resolve())})
    before = {str(path.relative_to(old)): path.read_bytes() for path in old.rglob("*") if path.is_file()}
    model_info = json.loads((old / "model_info.json").read_text())
    baseline = compare.validate_off_baseline(renamed, model_info)
    assert baseline["result"]["completed"] and len(baseline["coverage"]) == 2
    report = compare.compare_runs(current)
    assert report["passed"] and not (current / "off").exists()
    assert (current / "report.json").exists()
    assert before == {str(path.relative_to(old)): path.read_bytes() for path in old.rglob("*") if path.is_file()}


@pytest.mark.parametrize("mutation", ["missing_file", "missing_rank", "incomplete", "on_case", "capacity"])
def test_validate_off_baseline_fails_before_model_start(tmp_path, mutation):
    _fixture(tmp_path)
    off = tmp_path / "off"
    if mutation == "missing_file":
        next((off / "tensors" / "rank1").glob("*.pt")).unlink()
    elif mutation == "missing_rank":
        (off / "tensors" / "rank1" / "index.jsonl").unlink()
    elif mutation == "incomplete":
        coverage = json.loads((off / "coverage.json").read_text())
        coverage[1]["complete"] = False
        _write(off / "coverage.json", coverage)
    elif mutation == "on_case":
        result = json.loads((off / "result.json").read_text())
        result["case"] = "on"
        _write(off / "result.json", result)
    else:
        environment = json.loads((off / "environment.json").read_text())
        environment["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "NaN"
        _write(off / "environment.json", environment)
    with pytest.raises(ValueError):
        compare.validate_off_baseline(off, json.loads((tmp_path / "model_info.json").read_text()))


@pytest.mark.parametrize("capacity,passes", [("24.0", True), ("32", False), ("0", False), ("NaN", False)])
def test_cpu_capacity_uses_numeric_equality_without_ignoring_size(tmp_path, capacity, passes):
    _fixture(tmp_path)
    for case, value in (("off", "24"), ("on", capacity)):
        path = tmp_path / case / "environment.json"
        environment = json.loads(path.read_text())
        environment["LMCACHE_MAX_LOCAL_CPU_SIZE"] = value
        _write(path, environment)
    assert compare.compare_runs(tmp_path)["passed"] is passes


@pytest.mark.parametrize("value", ["-1", "0", "Infinity", "NaN", "garbage", None, True])
def test_comparable_environment_rejects_invalid_capacity(value):
    with pytest.raises(ValueError, match="positive finite"):
        compare.comparable_environment({"LMCACHE_MAX_LOCAL_CPU_SIZE": value})


def test_compare_invalid_reference_reports_failure_without_falling_back(tmp_path):
    _fixture(tmp_path)
    _write(tmp_path / "off_reference.json", {"schema": 2, "off_dir": str((tmp_path / "off").resolve())})
    report = compare.compare_runs(tmp_path)
    assert not report["passed"] and report["counts"]["off"] == 0
    assert "off_reference.json schema" in report["errors"][0]["detail"]
