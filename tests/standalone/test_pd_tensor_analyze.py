# SPDX-License-Identifier: Apache-2.0
"""CPU tests for real PD raw-tensor analysis; run with --noconftest."""

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from urllib.parse import quote

import pytest
import torch

MODULE = Path(__file__).resolve().parents[2] / "tools" / "pd_tensor_analyze.py"
sys.path.insert(0, str(MODULE.parent))
SPEC = importlib.util.spec_from_file_location("pd_analyze_under_test", MODULE)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_worker(
    root,
    *,
    role="P",
    request="req/one",
    tp=1,
    rank=0,
    dp=0,
    name=None,
    batches=None,
    context=None,
    kind="kv_current",
    tensor_name="nope",
    complete=True,
    finished=True,
    layerwise=False,
    delta=0.0,
    phase=None,
    row_axis=0,
    context_complete=True,
    model="main",
):
    context = list(range(10, 15)) if context is None else context
    batches = [list(range(len(context)))] if batches is None else batches
    directory = root / role / quote(request, safe="") / (name or f"host-rank{rank}")
    records, calls = [], []
    for number, positions in enumerate(batches):
        end = max(positions, default=-1) + 1
        calls.append(number)
        write(
            directory / "calls" / f"{number}.json",
            {
                "schema": 1,
                "call": number,
                "model": model,
                "phase": phase or ("prefill" if role == "P" else "decode"),
                "positions": positions,
                "token_ids": [context[p] for p in positions],
                "context_token_ids": context[:end],
                "context_complete": context_complete,
                "complete": True,
                "expected": [{"layer": 0, "kind": kind, "name": tensor_name}],
            },
        )
        value = torch.tensor([[float(p) + delta, float(p) + 1 + delta] for p in positions]).reshape(len(positions), 2)
        if row_axis is None:
            value = value.sum(dim=0)
        tensor_path = f"tensors/{number:06}.pt"
        (directory / "tensors").mkdir(exist_ok=True)
        torch.save(value, directory / tensor_path)
        records.append(
            {
                "schema": 1,
                "request_id": request,
                "model": model,
                "call": number,
                "layer": 0,
                "kind": kind,
                "name": tensor_name,
                "path": tensor_path,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "positions": positions if row_axis == 0 else None,
                "token_ids": [context[p] for p in positions] if row_axis == 0 else None,
                "row_axis": row_axis,
                "tensor_layout": "rank_local",
                "mapping_only": False,
            }
        )
    write(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "tool": "pd_tensor_dump",
            "request_id": request,
            "internal_request_id": f"internal-{role}-{rank}",
            "role": role,
            "host": "host",
            "pid": 1,
            "tp_rank": rank,
            "tp_size": tp,
            "dp_rank": dp,
            "dp_size": 2,
            "model_id": "glm",
            "num_layers": 1,
            "layerwise_prefill": layerwise,
            "prompt_token_ids": context,
            "complete": complete,
            "request_finished": finished,
            "errors": [],
            "records": len(records),
            "calls": calls,
            "scope": "main backbone only; MTP excluded",
        },
    )
    write_records(directory, records)
    write_sampled(
        directory,
        [
            {"after_call": number, "token_ids": [99] if number == len(calls) - 1 else []}
            for number in calls
            if model == "main"
        ],
    )
    return directory


def read_records(directory):
    return [json.loads(line) for line in (directory / "index.jsonl").read_text().splitlines()]


def write_records(directory, records):
    (directory / "index.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def write_sampled(directory, records):
    (directory / "sampled.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def sample_worker(directory, limit=64):
    spec = importlib.util.spec_from_file_location(
        "_sampling_writer", MODULE.parent.parent / "vllm_ascend/pd_tensor_dump.py"
    )
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    records = read_records(directory)
    for record in records:
        call = read(directory / f"calls/{record['call']}.json")
        value = torch.load(directory / record["path"], weights_only=True)
        positions, capture = writer.sample_rows(
            record["positions"], limit, len(call["context_token_ids"]), aligned=True
        )
        value = writer.sampled_tensor(value, capture)
        torch.save(value, directory / record["path"])
        record.update(
            positions=positions,
            token_ids=[call["context_token_ids"][p] for p in positions],
            shape=list(value.shape),
            row_capture=capture,
        )
    write_records(directory, records)


def feature_worker(directory, source_width=16, saved_width=None, delta_index=None):
    records = read_records(directory)
    for record in records:
        value = torch.tensor(record["positions"], dtype=torch.float32)[:, None] + torch.arange(source_width)[None, :]
        if delta_index is not None:
            value[0, delta_index] += 1
        if saved_width is not None:
            value = value[:, :saved_width].contiguous()
            record["feature_capture"] = dict(axis=1, start=0, source_size=source_width, saved_size=saved_width)
        record["shape"] = list(value.shape)
        torch.save(value, directory / record["path"])
    write_records(directory, records)


@pytest.mark.parametrize("reference_width,candidate_width", [(None, 8), (8, None), (4, 8), (8, 4)])
def test_feature_samples_align_with_full_archives_without_claiming_full_vectors(
    tmp_path, reference_width, candidate_width
):
    for side, width in (("reference", reference_width), ("candidate", candidate_width)):
        directory = make_worker(tmp_path / side, context=list(range(200)))
        feature_worker(directory, saved_width=width)
    report = run(tmp_path)
    assert report["status"] == "analysis_complete", report
    assert report["compared_rows"] == 200
    assert report["counts"]["equal"] == 1
    assert report["capture_sampling"]["candidate"]["feature_sampled_tensors"] == int(candidate_width is not None)
    assert "full vectors" in report["capture_sampling"]["candidate"]["scope"]


@pytest.mark.parametrize("delta_index,expected", [(7, "different"), (15, "equal")])
def test_only_archived_feature_components_are_assessed(tmp_path, delta_index, expected):
    p = make_worker(tmp_path / "reference", role="P")
    d = make_worker(tmp_path / "candidate", role="D", kind="kv_consumed")
    feature_worker(p)
    feature_worker(d, saved_width=8, delta_index=delta_index)
    report = run(tmp_path, mode="pd-kv")
    assert report["status"] == "analysis_complete", report
    assert report["counts"][expected] == 1
    assert report["accuracy_verdict"] == "not_assessed"


def test_identical_eight_saved_features_cannot_hide_different_original_shapes(tmp_path):
    p = make_worker(tmp_path / "reference")
    d = make_worker(tmp_path / "candidate")
    feature_worker(p, source_width=16, saved_width=8)
    feature_worker(d, source_width=32, saved_width=8)
    report = run(tmp_path)
    assert report["status"] == "incomplete_or_incomparable"
    assert report["issues"] > 0


def test_reference_snapshots_with_mixed_feature_limits_align_to_common_prefix(tmp_path):
    p = make_worker(tmp_path / "reference", batches=[[0, 1], [2, 3, 4]])
    d = make_worker(tmp_path / "candidate")
    feature_worker(p)
    feature_worker(d)
    records = read_records(p)
    first = records[0]
    value = torch.load(p / first["path"], weights_only=True)[:, :8].contiguous()
    torch.save(value, p / first["path"])
    first["feature_capture"] = dict(axis=1, start=0, source_size=16, saved_size=8)
    first["shape"][-1] = 8
    write_records(p, records)
    report = run(tmp_path)
    assert report["status"] == "analysis_complete", report
    assert report["compared_rows"] == 5


@pytest.mark.parametrize(
    "sampled_side,status", [("reference", "outside_reference_sample"), ("candidate", "outside_candidate_sample")]
)
def test_sampled_scope_does_not_claim_unobserved_rows_are_missing_or_checked(tmp_path, sampled_side, status):
    for side in ("reference", "candidate"):
        directory = make_worker(tmp_path / side, context=list(range(8200)))
        if side == sampled_side:
            sample_worker(directory)
    report = run(tmp_path)
    assert report["status"] == "analysis_complete", report
    assert report["compared_rows"] == 64 + 64 + 8
    assert report["counts"][status] == 1
    capture = report["capture_sampling"][sampled_side]
    assert capture["sampled_tensors"] == 1
    assert capture["omitted_token_rows"] == 8200 - 136
    assert report["accuracy_verdict"] == "not_assessed"


def test_pd_comparison_uses_identical_absolute_positions_from_two_chunks(tmp_path):
    p = make_worker(
        tmp_path / "reference",
        role="P",
        context=list(range(8200)),
        batches=[list(range(4096)), list(range(4096, 8200))],
    )
    d = make_worker(
        tmp_path / "candidate",
        role="D",
        context=list(range(8200)),
        kind="kv_consumed",
        batches=[[0, 10, 63, 100, 4096, 4159, 5000, 8195, 8199]],
        delta=0.25,
    )
    sample_worker(p)
    sample_worker(d)
    report = run(tmp_path, mode="pd-kv")
    assert report["status"] == "analysis_complete", report
    assert report["compared_rows"] == 7
    assert report["first_difference"]["position"] == 0
    assert report["issues"] == 0


def test_sampling_does_not_hide_missing_tensor_files(tmp_path):
    p = make_worker(tmp_path / "reference", context=list(range(200)))
    make_worker(tmp_path / "candidate", context=list(range(200)))
    sample_worker(p)
    (p / read_records(p)[0]["path"]).unlink()
    report = run(tmp_path)
    assert report["status"] == "incomplete_or_incomparable"
    assert report["issues"] > 0


@pytest.mark.parametrize("field,value", [("saved_rows", 65), ("source_indices", [199] * 64), ("limit", 32)])
def test_inconsistent_sampling_metadata_is_not_silently_accepted(tmp_path, field, value):
    for side in ("reference", "candidate"):
        directory = make_worker(tmp_path / side, context=list(range(200)))
        sample_worker(directory)
    records = read_records(directory)
    records[0]["row_capture"][field] = value
    write_records(directory, records)
    report = run(tmp_path)
    assert report["status"] == "incomplete_or_incomparable"
    assert report["issues"] > 0


def test_pd_compares_mtp_prompt_kv_with_shifted_context_and_excludes_generated_boundary(tmp_path):
    make_worker(tmp_path / "reference", role="P", model="mtp", context=[11, 12, 13, 99], kind="kv_current")
    make_worker(tmp_path / "candidate", role="D", model="mtp", context=[11, 12, 13, 88], kind="kv_consumed", delta=0.5)
    report = run(tmp_path, mode="pd-kv")
    assert report["status"] == "analysis_complete", report
    assert report["compared_rows"] == 3
    assert report["counts"]["not_prompt_kv"] == 1
    assert report["models_observed"] == {"reference": ["mtp"], "candidate": ["mtp"]}
    assert report["first_difference_by_model"]["mtp"]["kind"] == "kv_consumed"
    assert report["counts_by_model"]["mtp"]["different"] == 1


def test_off_on_reports_mtp_logits_differences_even_without_output_difference(tmp_path):
    for side, delta in (("reference", 0), ("candidate", 0.5)):
        make_worker(tmp_path / side, role="D", model="mtp", kind="logits", tensor_name="output", delta=delta)
    report = run(tmp_path)
    assert report["status"] == "analysis_complete", report
    assert report["first_difference_by_model"]["mtp"]["kind"] == "logits"


def test_mtp_tensor_cannot_claim_a_main_call(tmp_path):
    directory = make_worker(tmp_path / "reference", model="mtp")
    call = read(directory / "calls/0.json")
    call["model"] = "main"
    write(directory / "calls/0.json", call)
    archive = analyzer.load_archive([tmp_path / "reference"])
    assert any("tensor/call model differs" in issue.get("reason", "") for issue in archive.issues)


def run(tmp_path, *, mode="off-on", request_map=None):
    return analyzer.analyze(
        [tmp_path / "reference"],
        [tmp_path / "candidate"],
        mode=mode,
        output=tmp_path / "report",
        request_map=request_map,
    )


def details(tmp_path):
    return [json.loads(line) for line in (tmp_path / "report" / "comparisons.jsonl").read_text().splitlines()]


def add_rejection_fixture(directory):
    records = read_records(directory)
    call = read(directory / "calls/0.json")
    call["rejection"] = {"mode": "greedy", "num_draft_tokens": 1, "placeholder_token_id": -1, "draft_positions": [4]}
    for name, value in {
        "draft_token_ids": torch.tensor([1]),
        "target_logits": torch.tensor([[0.0, 1.0, -1.0]]),
        "bonus_token_ids": torch.tensor([2]),
        "sampler_output": torch.tensor([1, 2]),
        "kernel_output": torch.tensor([1, 2]),
    }.items():
        record = {
            **records[0],
            "layer": -1,
            "kind": "rejection",
            "name": name,
            "path": f"tensors/rejection-{name}.pt",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "row_axis": None,
            "positions": None,
            "token_ids": None,
        }
        record.pop("physical_slots", None)
        records.append(record)
        call["expected"].append({"layer": -1, "kind": "rejection", "name": name})
        torch.save(value, directory / record["path"])
    write(directory / "calls/0.json", call)
    write_records(directory, records)
    manifest = read(directory / "manifest.json")
    manifest["records"] = len(records)
    write(directory / "manifest.json", manifest)


@pytest.mark.parametrize("mode", ["off-on", "pd-kv"])
@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_reports_match_serial_bytes_including_aliases_mtp_and_rejection(tmp_path, mode, workers):
    root = tmp_path / "dumps"
    for request, model in (("first", "main"), ("second", "mtp")):
        for rank in range(2):
            for side in ("reference", "candidate"):
                directory = make_worker(
                    root / side,
                    role="P" if side == "reference" and mode == "pd-kv" else "D",
                    request=request,
                    tp=2,
                    rank=rank,
                    model=model,
                    kind="kv_current" if side == "reference" and mode == "pd-kv" else "kv_consumed",
                    phase="prefill" if side == "reference" else "decode",
                    delta=0.5 if side == "candidate" else 0.0,
                )
                if side == "candidate":
                    add_physical_aliases(directory)
                if model == "main":
                    add_rejection_fixture(directory)
    reports = []
    for parallelism in (1, workers):
        destination = tmp_path / f"report-{parallelism}"
        report = analyzer.analyze(
            [root / "reference"], [root / "candidate"], mode=mode, output=destination, workers=parallelism
        )
        assert report["rejection_sampling"]["workers"]
        assert report["counts"]["different"] > 0
        assert report["kv_aliases"]["different_tensors"] > 0
        assert report["first_difference_by_model"].keys() == {"main", "mtp"}
        assert not list(destination.glob(".analysis-*"))
        reports.append(destination)
    for name in ("report.json", "comparisons.jsonl", "rejection.jsonl"):
        assert (reports[0] / name).read_bytes() == (reports[1] / name).read_bytes()


def test_parallel_comparisons_actually_overlap_and_preserve_input_order(tmp_path):
    barrier = threading.Barrier(2, timeout=10)
    first_written = threading.Event()
    thread_ids = set()
    emitted = []

    def compare(value, emit):
        thread_ids.add(threading.get_ident())
        assert torch.get_num_threads() == 1
        barrier.wait()
        if value == 0:
            assert first_written.wait(10)
        emit({"value": value})
        if value == 1:
            first_written.set()

    original_threads = torch.get_num_threads()
    with analyzer._analysis_pool(2) as pool:
        analyzer._run_comparison_jobs([(compare, (0,)), (compare, (1,))], pool, tmp_path, 2, emitted.append)
    assert emitted == [{"value": 0}, {"value": 1}]
    assert len(thread_ids) == 2
    assert torch.get_num_threads() == original_threads
    assert not list(tmp_path.glob(".analysis-*"))


def add_layers_fixture(directory, layers):
    original = read_records(directory)
    records = []
    for record in original:
        value = torch.load(directory / record["path"], weights_only=True)
        for layer in range(layers):
            path = f"tensors/call-{record['call']}-layer-{layer}.pt"
            torch.save(value + layer, directory / path)
            records.append({**record, "layer": layer, "path": path})
    write_records(directory, records)
    manifest = read(directory / "manifest.json")
    manifest.update(records=len(records), num_layers=layers)
    write(directory / "manifest.json", manifest)
    for number in manifest["calls"]:
        call = read(directory / "calls" / f"{number}.json")
        call["expected"] = [
            {key: record[key] for key in ("layer", "kind", "name")} for record in records if record["call"] == number
        ]
        write(directory / "calls" / f"{number}.json", call)


@pytest.mark.parametrize("mode", ["off-on", "pd-kv"])
@pytest.mark.parametrize("invalid_tensor", [False, True])
def test_multiple_layers_and_calls_keep_serial_detail_order(tmp_path, mode, invalid_tensor):
    source = make_worker(tmp_path / "reference", batches=[[0, 1, 2], [0, 1, 2, 3, 4]])
    destination = make_worker(
        tmp_path / "candidate",
        role="D" if mode == "pd-kv" else "P",
        kind="kv_consumed" if mode == "pd-kv" else "kv_current",
        batches=[[0, 1], [0, 1, 2, 3]],
        delta=0.5,
    )
    add_layers_fixture(source, 4)
    add_layers_fixture(destination, 4)
    if invalid_tensor:
        (destination / "tensors/call-0-layer-2.pt").write_bytes(b"invalid tensor")
    for workers in (1, 4):
        analyzer.analyze(
            [tmp_path / "reference"],
            [tmp_path / "candidate"],
            mode=mode,
            output=tmp_path / f"report-{workers}",
            workers=workers,
        )
    for name in ("report.json", "comparisons.jsonl", "rejection.jsonl"):
        assert (tmp_path / "report-1" / name).read_bytes() == (tmp_path / "report-4" / name).read_bytes()


def test_one_rank_runs_multiple_layers_concurrently(tmp_path, monkeypatch):
    for name in ("reference", "candidate"):
        directory = make_worker(tmp_path / name)
        add_layers_fixture(directory, 4)
    barrier = threading.Barrier(4, timeout=10)
    thread_ids = set()
    layers = set()
    original = analyzer._compare_worker

    def compare(reference, candidate, mode, emit):
        layer = {record["layer"] for record in candidate.records}
        assert len(layer) == 1
        thread_ids.add(threading.get_ident())
        layers.update(layer)
        barrier.wait()
        original(reference, candidate, mode, emit)

    monkeypatch.setattr(analyzer, "_compare_worker", compare)
    monkeypatch.setattr(analyzer.os, "cpu_count", lambda: 4)
    report = analyzer.analyze(
        [tmp_path / "reference"],
        [tmp_path / "candidate"],
        mode="off-on",
        output=tmp_path / "report",
        workers=4,
    )
    assert report["status"] == "analysis_complete"
    assert report["counts"]["equal"] == 4
    assert len(thread_ids) == 4 and layers == {0, 1, 2, 3}


def test_parallel_failure_joins_writers_cleans_files_and_restores_torch_threads(tmp_path):
    barrier = threading.Barrier(2, timeout=10)
    finished = threading.Event()

    def compare(value, emit):
        barrier.wait()
        if value == 0:
            raise ValueError("broken comparison")
        emit({"value": value})
        finished.set()

    original_threads = torch.get_num_threads()
    with pytest.raises(ValueError, match="broken comparison"), analyzer._analysis_pool(2) as pool:
        analyzer._run_comparison_jobs([(compare, (0,)), (compare, (1,))], pool, tmp_path, 2, lambda _: None)
    assert finished.is_set()
    assert torch.get_num_threads() == original_threads
    assert not list(tmp_path.glob(".analysis-*"))


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_analysis_worker_count_is_rejected(tmp_path, workers):
    with pytest.raises(ValueError, match="positive integer"):
        analyzer.analyze([], [], mode="pd-kv", output=tmp_path / "report", workers=workers)
    assert not (tmp_path / "report").exists()


def test_pd_same_archive_metadata_is_loaded_once(tmp_path, monkeypatch):
    root = tmp_path / "collect"
    make_worker(root)
    make_worker(root, role="D", kind="kv_consumed")
    original = analyzer.load_archive
    calls = []

    def load(roots):
        calls.append(roots)
        return original(roots)

    monkeypatch.setattr(analyzer, "load_archive", load)
    report = analyzer.analyze([root], [root], mode="pd-kv", output=tmp_path / "report", workers=2)
    assert report["status"] == "analysis_complete"
    assert len(calls) == 1


def test_long_prompt_alignment_compares_call_context_once_across_layers(tmp_path):
    source = make_worker(tmp_path / "reference", context=list(range(4096)))
    destination = make_worker(tmp_path / "candidate", context=list(range(4096)))
    source_worker = next(iter(analyzer.load_archive([source]).workers.values()))
    candidate_worker = next(iter(analyzer.load_archive([destination]).workers.values()))
    visits = []

    class ContextTokens(list):
        def __iter__(self):
            visits.append(len(self))
            return super().__iter__()

        def __getitem__(self, key):
            assert not isinstance(key, slice), "per-token prefix slicing regressed"
            return super().__getitem__(key)

    for worker in (source_worker, candidate_worker):
        call = worker.calls[0]
        call["context_token_ids"] = ContextTokens(call["context_token_ids"])
        worker.records = [{**worker.records[0], "layer": layer} for layer in range(3)]
    rows = []
    analyzer._compare_worker(source_worker, candidate_worker, "off-on", rows.append)
    assert [row["status"] for row in rows] == ["equal"] * 3
    assert visits == [4096, 4096]


def test_off_on_aligns_logical_rows_across_chunks_with_full_statistics(tmp_path):
    make_worker(tmp_path / "reference", batches=[[0, 1, 2], [3, 4]])
    make_worker(tmp_path / "candidate", batches=[[0], [1, 2, 3, 4]], layerwise=True, delta=0.01)
    report = run(tmp_path)
    assert report["status"] == "analysis_complete"
    assert report["accuracy_verdict"] == "not_assessed"
    assert "passed" not in report
    assert report["counts"]["different"] == 2
    assert report["compared_rows"] == 5
    values = next(item["comparison"] for item in reversed(details(tmp_path)) if "comparison" in item)
    assert values["baseline"]["std"] > 0
    assert values["candidate"]["rms"] > 0
    assert values["abs_diff"]["max"] == pytest.approx(0.01, abs=1e-6)
    assert values["relative_l2"] > 0


@pytest.mark.parametrize("length", [5, 4099])
def test_pd_only_consumed_prompt_rows_short_tail_and_long_chunk(tmp_path, length):
    context = list(range(length + 1))
    make_worker(
        tmp_path / "reference",
        context=context[:-1],
        batches=[list(range(min(length, 4096))), list(range(4096, length))] if length > 4096 else None,
    )
    make_worker(
        tmp_path / "candidate", role="D", dp=1, kind="kv_consumed", context=context, batches=[[0, length - 1, length]]
    )
    report = run(tmp_path, mode="pd-kv")
    assert report["status"] == "analysis_complete"
    assert report["compared_rows"] == 2
    assert report["counts"]["not_prompt_kv"] == 1
    assert report["counts"]["not_consumed"] >= 1
    assert report["counts"].get("missing_reference_tensor", 0) == 0


def test_pd_same_collect_root_supports_indexer_and_repeated_full_prefix(tmp_path):
    make_worker(tmp_path / "collect", kind="kv_indexer", tensor_name="key", batches=[[0, 1, 2], [0, 1, 2, 3, 4]])
    make_worker(tmp_path / "collect", role="D", kind="kv_indexer", tensor_name="key", batches=[[0, 1, 2, 3, 4]])
    report = analyzer.analyze([tmp_path / "collect"], [tmp_path / "collect"], mode="pd-kv", output=tmp_path / "report")
    assert report["status"] == "analysis_complete"
    assert report["compared_rows"] == 5
    assert details(tmp_path)[0]["repeated_reference_choices"] == 3
    assert "not_consumed" not in report["counts"]


def add_physical_aliases(directory, *, different=True, nonfinite=False):
    record = json.loads((directory / "index.jsonl").read_text())
    value = torch.load(directory / record["path"], weights_only=True)
    # Two extra copies of token zero: first is equal, second can conflict.
    value = torch.cat((value, value[:1], value[:1]), dim=0)
    if different:
        value[-1] += 7
    if nonfinite:
        value[0] = value[-2] = value[-1] = float("nan")
    torch.save(value, directory / record["path"])
    record["positions"] += [0, 0]
    record["token_ids"] += record["token_ids"][:1] * 2
    record["shape"] = list(value.shape)
    record["physical_slots"] = list(range(100, 100 + len(value)))
    write_records(directory, [record])


@pytest.mark.parametrize("different,nonfinite", [(True, False), (False, False), (False, True)])
def test_pd_analysis_compares_all_physical_copies_and_reports_aliases_once(tmp_path, different, nonfinite):
    root = tmp_path / "collect"
    make_worker(root)
    destination = make_worker(root, role="D", kind="kv_consumed")
    add_physical_aliases(destination, different=different, nonfinite=nonfinite)
    report = analyzer.analyze([root], [root], mode="pd-kv", output=tmp_path / "report")
    assert report["status"] == "analysis_complete"
    assert report["accuracy_verdict"] == "not_assessed"
    assert report["compared_rows"] == 7  # No conflicting copy was silently dropped.
    aliases = report["kv_aliases"]
    assert aliases["different_tensors"] == int(different)
    assert aliases["equal_tensors"] == int(not different)
    detail = next(item for item in details(tmp_path) if "alias_comparison" in item)
    assert detail["alias_rows"] == [5, 6]
    assert detail["baseline_physical_slots"] == [100, 100]
    assert detail["alias_physical_slots"] == [105, 106]
    if different:
        assert aliases["first_conflict"]["alias_physical_slot"] == 106
        assert detail["alias_comparison"]["mismatched"] == 2
        assert report["counts"]["different"] == 1
    else:
        assert aliases["first_conflict"] is None
    if nonfinite:
        assert detail["alias_comparison"]["baseline"]["nonfinite"] == 4
        assert report["new_nonfinite"] == 6


def test_conflicting_off_alias_is_reported_even_when_first_copy_matches_on(tmp_path):
    source = make_worker(tmp_path / "reference", role="D", kind="kv_consumed")
    make_worker(tmp_path / "candidate", role="D", kind="kv_consumed")
    add_physical_aliases(source)
    report = run(tmp_path)
    assert report["counts"]["equal"] == 1
    assert report["kv_aliases"]["different_tensors"] == 1
    assert report["kv_aliases"]["first_conflict"]["archive"] == "reference"


@pytest.mark.parametrize("slots", [[-1, 2, 3, 4, 5], [1], "invalid"])
def test_invalid_physical_slot_evidence_is_not_accepted(tmp_path, slots):
    make_worker(tmp_path / "reference")
    destination = make_worker(tmp_path / "candidate", role="D", kind="kv_consumed")
    record = json.loads((destination / "index.jsonl").read_text())
    record["physical_slots"] = slots
    write_records(destination, [record])
    report = run(tmp_path, mode="pd-kv")
    assert report["status"] == "incomplete_or_incomparable"
    assert report["counts"]["invalid_record"] >= 1


@pytest.mark.parametrize("missing", [None, "D", "rank"])
def test_four_host_collection_auto_matches_request_and_tp_with_idle_hosts(tmp_path, monkeypatch, missing):
    import pd_tensor_collect as collector

    payloads = {}
    # Distinct DP and host identities: P/D must match by external request and TP.
    for host, role, dp in (("p-used", "P", 0), ("d-used", "D", 1)):
        root = tmp_path / "remote" / host
        if role == "D" and missing == "D":
            continue
        for rank in range(2):
            if role == "D" and rank == 1 and missing == "rank":
                continue
            for request in ("req/one", "req/two"):
                make_worker(
                    root,
                    role=role,
                    dp=dp,
                    tp=2,
                    rank=rank,
                    request=request,
                    name=f"{host}-rank{rank}",
                    kind="kv_current" if role == "P" else "kv_consumed",
                    delta=float(rank),
                    batches=[[0, 1], [2, 3, 4]] if role == "P" else [[0, 1, 2, 3, 4]],
                )
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            archive.add(root, arcname=".")
        payloads[host] = stream.getvalue()

    def run_ssh(command, *, stdout, stderr, check):
        host = command[-2]
        if host not in payloads:
            stderr.write(collector.MISSING_RUN_MARKER.encode())
            return subprocess.CompletedProcess(command, collector.MISSING_RUN_EXIT)
        stdout.write(payloads[host])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(collector.subprocess, "run", run_ssh)
    monkeypatch.setitem(sys.modules, "pd_tensor_analyze", analyzer)
    output = tmp_path / "collected"
    status = collector.main(
        [
            "--hosts",
            "p-used",
            "p-idle",
            "d-idle",
            "d-used",
            "--repo-path",
            "/workspace/repo",
            "--run-id",
            "case-on",
            "--output",
            str(output),
            "--analyze-pd",
        ]
    )
    report = read(output / "report-pd/report.json")
    assert read(output / "collection.json")["complete"]
    if missing:
        assert status == 2
        assert report["status"] == "incomplete_or_incomparable"
        assert report["counts"]["missing_candidate_worker"] > 0
    else:
        assert status == 0
        assert report["status"] == "analysis_complete"
        assert report["compared_rows"] == 20
        assert report["counts"].get("different", 0) == 0


def test_request_ids_require_explicit_mapping(tmp_path):
    make_worker(tmp_path / "reference", request="old-id")
    make_worker(tmp_path / "candidate", request="new-id")
    assert run(tmp_path)["counts"]["missing_reference_worker"] == 1
    report = run(tmp_path, request_map={"old-id": "new-id"})
    assert report["status"] == "analysis_complete"
    assert report["compared_rows"] == 5


@pytest.mark.parametrize("mapping", [{"a": "z", "b": "z"}, {"a": 1}, []])
def test_invalid_request_mapping_rejected(tmp_path, mapping):
    with pytest.raises(ValueError, match="request map"):
        run(tmp_path, request_map=mapping)


def test_pd_disallows_remapping(tmp_path):
    with pytest.raises(ValueError, match="identical request IDs"):
        run(tmp_path, mode="pd-kv", request_map={"a": "b"})


def test_context_divergence_excludes_following_rows_without_numeric_verdict(tmp_path):
    make_worker(tmp_path / "reference")
    make_worker(tmp_path / "candidate", context=[10, 11, 999, 13, 14])
    report = run(tmp_path)
    assert report["status"] == "incomplete_or_incomparable"
    assert report["compared_rows"] == 2
    mismatch = next(item for item in details(tmp_path) if item["status"] == "incomparable_context")
    assert mismatch["positions"] == [2, 3, 4]


def test_incomplete_context_never_compared(tmp_path):
    make_worker(tmp_path / "reference")
    make_worker(tmp_path / "candidate", context_complete=False)
    report = run(tmp_path)
    assert report["compared_tensors"] == 0
    assert report["counts"]["incomparable_context"] == 1


@pytest.mark.parametrize("complete,finished", [(False, True), (True, False)])
def test_unfinished_worker_keeps_values_but_marks_report_incomplete(tmp_path, complete, finished):
    make_worker(tmp_path / "reference")
    make_worker(tmp_path / "candidate", complete=complete, finished=finished)
    report = run(tmp_path)
    assert report["counts"]["incomplete_worker"] == 1
    assert report["compared_tensors"] == 1
    assert report["status"] == "incomplete_or_incomparable"


def test_missing_tp_rank_is_visible_even_when_present_rank_equal(tmp_path):
    for rank in range(2):
        make_worker(tmp_path / "reference", tp=2, rank=rank)
    make_worker(tmp_path / "candidate", tp=2, rank=0)
    report = run(tmp_path)
    assert report["counts"]["missing_tp_ranks"] == 1
    assert report["counts"]["missing_candidate_worker"] == 1
    assert report["status"] == "incomplete_or_incomparable"


def test_ranks_are_not_cross_matched(tmp_path):
    for rank in range(2):
        make_worker(tmp_path / "reference", tp=2, rank=rank, delta=float(rank))
        make_worker(tmp_path / "candidate", tp=2, rank=rank, delta=float(1 - rank))
    report = run(tmp_path)
    assert report["counts"]["different"] == 2


def test_ambiguous_workers_not_silently_chosen(tmp_path):
    make_worker(tmp_path / "reference")
    make_worker(tmp_path / "reference", name="another-host", dp=1)
    make_worker(tmp_path / "candidate")
    report = run(tmp_path)
    assert report["counts"]["ambiguous_worker"] == 1
    assert report["compared_tensors"] == 0


def test_missing_expected_tensor_and_count_mismatch(tmp_path):
    make_worker(tmp_path / "reference")
    directory = make_worker(tmp_path / "candidate")
    write_records(directory, [])
    report = run(tmp_path)
    assert report["counts"]["missing_expected_tensor"] == 1
    assert report["counts"]["record_count_mismatch"] == 1


@pytest.mark.parametrize("mutation", ["missing", "shape", "dtype", "payload", "escape"])
def test_raw_tensor_integrity_and_path_checks(tmp_path, mutation):
    make_worker(tmp_path / "reference")
    directory = make_worker(tmp_path / "candidate")
    records = read_records(directory)
    path = directory / records[0]["path"]
    if mutation == "missing":
        path.unlink()
    elif mutation == "shape":
        torch.save(torch.zeros(8, 2), path)
    elif mutation == "dtype":
        torch.save(torch.zeros(5, 2, dtype=torch.int64), path)
    elif mutation == "payload":
        torch.save({"not": "a tensor"}, path)
    else:
        records[0]["path"] = "../outside.pt"
        write_records(directory, records)
    report = run(tmp_path)
    assert report["status"] == "incomplete_or_incomparable"
    assert report["counts"].get("invalid_tensor", 0) + report["counts"].get("invalid_record", 0) == 1


def test_partial_folders_ignored_and_no_data_is_not_success(tmp_path):
    make_worker(tmp_path / "reference" / "copy.partial")
    make_worker(tmp_path / "candidate")
    report = run(tmp_path)
    assert report["compared_tensors"] == 0
    assert report["counts"]["no_worker_manifests"] == 1
    assert report["status"] == "incomplete_or_incomparable"


def test_nonrow_requires_identical_query_context(tmp_path):
    make_worker(tmp_path / "reference", row_axis=None)
    make_worker(tmp_path / "candidate", row_axis=None, batches=[[3, 4]])
    report = run(tmp_path)
    assert report["counts"]["incomparable_context"] == 1


def test_prefill_outputs_never_compared_to_decode_outputs(tmp_path):
    make_worker(tmp_path / "reference", kind="decoder", tensor_name="output")
    make_worker(tmp_path / "candidate", role="D", kind="decoder", tensor_name="output", batches=[[4]])
    report = run(tmp_path, mode="pd-kv")
    assert report["compared_tensors"] == 0
    assert report["status"] == "incomplete_or_incomparable"


def test_nonfinite_and_distribution_reported_without_pass_claim(tmp_path):
    make_worker(tmp_path / "reference")
    directory = make_worker(tmp_path / "candidate")
    record = read_records(directory)[0]
    value = torch.load(directory / record["path"], weights_only=True)
    value[2, 0] = float("nan")
    torch.save(value, directory / record["path"])
    report = run(tmp_path)
    assert report["new_nonfinite"] == 1
    assert report["accuracy_verdict"] == "not_assessed"
    assert report["first_difference"]["position"] == 2
    assert next(item for item in details(tmp_path) if "comparison" in item)["comparison"]["candidate"]["nan"] == 1


def test_pd_replication_and_sequence_layout_can_match_same_rank_logical_rows(tmp_path):
    source = make_worker(tmp_path / "reference")
    destination = make_worker(tmp_path / "candidate", role="D", kind="kv_consumed", batches=[[1, 4]])
    for directory, layout in ((source, "replicated"), (destination, "sequence_sharded")):
        records = read_records(directory)
        records[0]["tensor_layout"] = layout
        write_records(directory, records)
    report = run(tmp_path, mode="pd-kv")
    assert report["status"] == "analysis_complete"
    assert report["compared_rows"] == 2


def test_missing_p_source_within_prompt_is_failure_not_not_consumed(tmp_path):
    directory = make_worker(tmp_path / "reference")
    records = read_records(directory)
    records[0]["positions"] = records[0]["positions"][:3]
    records[0]["token_ids"] = records[0]["token_ids"][:3]
    write_records(directory, records)
    make_worker(tmp_path / "candidate", role="D", kind="kv_consumed", batches=[[4]])
    report = run(tmp_path, mode="pd-kv")
    assert report["counts"]["missing_reference_tensor"] == 1
    assert report["status"] == "incomplete_or_incomparable"


def test_nonrow_equal_and_mapping_only_evidence(tmp_path):
    make_worker(tmp_path / "reference", row_axis=None)
    make_worker(tmp_path / "candidate", row_axis=None)
    assert run(tmp_path)["counts"]["equal"] == 1
    for root in (tmp_path / "reference", tmp_path / "candidate"):
        directory = next(root.rglob("manifest.json")).parent
        records = read_records(directory)
        records[0].update(mapping_only=True, tensor_layout="mapping")
        write_records(directory, records)
    report = run(tmp_path)
    assert report["compared_tensors"] == 0
    assert report["counts"]["mapping_evidence"] == 1


def test_cli_writes_report_and_zero_does_not_claim_accuracy(tmp_path):
    make_worker(tmp_path / "reference")
    make_worker(tmp_path / "candidate", delta=0.5)
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--mode",
            "off-on",
            "--reference",
            str(tmp_path / "reference"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "report"),
            "--workers",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["accuracy_verdict"] == "not_assessed"
    assert read(tmp_path / "report" / "report.json")["counts"]["different"] == 1


def test_unknown_context_tokens_are_incomparable_not_invalid(tmp_path):
    make_worker(tmp_path / "reference")
    directory = make_worker(tmp_path / "candidate", context_complete=False)
    call = read(directory / "calls" / "0.json")
    call["context_token_ids"][1] = None
    write(directory / "calls" / "0.json", call)
    records = read_records(directory)
    records[0]["token_ids"][1] = None
    write_records(directory, records)
    report = run(tmp_path)
    assert report["compared_tensors"] == 0
    assert report["counts"]["incomparable_context"] == 1
    assert "invalid_call" not in report["counts"]
    assert "invalid_record" not in report["counts"]


def test_additional_optional_tensors_are_compared(tmp_path):
    for name in ("reference", "candidate"):
        directory = make_worker(tmp_path / name)
        call = read(directory / "calls" / "0.json")
        call["expected"] = []
        write(directory / "calls" / "0.json", call)
    report = run(tmp_path)
    assert report["counts"]["additional_tensor"] == 2
    assert report["counts"]["equal"] == 1
    assert report["status"] == "analysis_complete"


def test_real_request_archive_schema_roundtrip(tmp_path):
    module_path = MODULE.parents[1] / "vllm_ascend" / "pd_tensor_dump.py"
    spec = importlib.util.spec_from_file_location("pd_dump_for_analyzer_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, role, kind in (("reference", "P", "kv_current"), ("candidate", "D", "kv_consumed")):
        archive = module.RequestArchive(
            tmp_path / name,
            dict(
                host="host",
                pid=1,
                tp_rank=0,
                tp_size=1,
                dp_rank=0,
                dp_size=1,
                role=role,
                request_id="真实/request",
                internal_request_id="internal",
                model_id="glm",
                num_layers=1,
                layerwise_prefill=role == "P",
                prompt_token_ids=[10, 11],
            ),
        )
        call = archive.begin(
            dict(
                phase="prefill" if role == "P" else "decode",
                positions=[0, 1],
                token_ids=[10, 11],
                context_token_ids=[10, 11],
                context_complete=True,
            ),
            [dict(layer=0, kind=kind, name="nope")],
        )
        archive.record(torch.arange(4).reshape(2, 2), call, 0, kind, "nope", [0, 1])
        write_sampled(archive.root, [{"after_call": 0, "token_ids": [99]}])
        archive.end(call, {(0, kind, "nope")})
        archive.finish()
    report = run(tmp_path, mode="pd-kv")
    assert report["status"] == "analysis_complete"
    assert report["counts"]["equal"] == 1


def test_accepted_outputs_align_by_token_order_not_sampling_batch(tmp_path):
    source = make_worker(tmp_path / "reference", batches=[[0, 1], [2, 3, 4]])
    actual = make_worker(tmp_path / "candidate", batches=[[0], [1], [2, 3, 4]])
    write_sampled(source, [{"after_call": 0, "token_ids": []}, {"after_call": 1, "token_ids": [90, 91, 92]}])
    write_sampled(
        actual,
        [
            {"after_call": 0, "token_ids": []},
            {"after_call": 1, "token_ids": [90]},
            {"after_call": 2, "token_ids": [91, 92]},
        ],
    )
    report = run(tmp_path)
    assert report["sampled_outputs"]["equal_workers"] == 1
    assert report["sampled_outputs"]["first_difference"] is None


@pytest.mark.parametrize("actual,first,left,right", [([90, 999, 92], 1, 91, 999), ([90], 1, 91, None)])
def test_accepted_output_first_difference_and_length_difference(tmp_path, actual, first, left, right):
    source = make_worker(tmp_path / "reference")
    destination = make_worker(tmp_path / "candidate")
    write_sampled(source, [{"after_call": 0, "token_ids": [90, 91, 92]}])
    write_sampled(destination, [{"after_call": 0, "token_ids": actual}])
    report = run(tmp_path)
    assert report["status"] == "analysis_complete"
    assert report["sampled_outputs"]["different_workers"] == 1
    marker = report["sampled_outputs"]["first_difference"]
    assert marker["first_unequal_output_index"] == first
    assert marker["first_unequal_context_position"] == 5 + first
    assert marker["reference_first_unequal_token"] == left
    assert marker["candidate_first_unequal_token"] == right
    assert report["accuracy_verdict"] == "not_assessed"


@pytest.mark.parametrize(
    "mutation,status",
    [
        ("missing", "missing_sampled_outputs"),
        ("empty", "missing_sampled_calls"),
        ("invalid_json", "invalid_sampled_outputs"),
        ("invalid_call", "invalid_sampled_outputs"),
        ("invalid_token", "invalid_sampled_outputs"),
        ("duplicate_empty", "invalid_sampled_outputs"),
    ],
)
def test_sampled_stream_integrity_is_required(tmp_path, mutation, status):
    make_worker(tmp_path / "reference")
    directory = make_worker(tmp_path / "candidate")
    if mutation == "missing":
        (directory / "sampled.jsonl").unlink()
    elif mutation == "empty":
        write_sampled(directory, [])
    elif mutation == "invalid_json":
        (directory / "sampled.jsonl").write_text("{truncated", encoding="utf-8")
    elif mutation == "invalid_call":
        write_sampled(directory, [{"after_call": 5, "token_ids": [1]}])
    elif mutation == "invalid_token":
        write_sampled(directory, [{"after_call": 0, "token_ids": [-1]}])
    else:
        write_sampled(directory, [{"after_call": 0, "token_ids": []}, {"after_call": 0, "token_ids": []}])
    report = run(tmp_path)
    assert report["counts"][status] == 1
    assert report["counts"]["incomplete_sampled_outputs"] == 1
    assert report["status"] == "incomplete_or_incomparable"


def test_sampled_output_comparison_requires_identical_known_prompts(tmp_path):
    make_worker(tmp_path / "reference")
    directory = make_worker(tmp_path / "candidate")
    manifest = read(directory / "manifest.json")
    manifest["prompt_token_ids"] = [99, 11, 12, 13, 14]
    write(directory / "manifest.json", manifest)
    report = run(tmp_path)
    assert report["counts"]["incomparable_sampled_context"] == 1
    assert report["sampled_outputs"]["compared_workers"] == 0


def test_pd_ignores_output_phase_convention_but_requires_sampled_records(tmp_path):
    source = make_worker(tmp_path / "reference")
    destination = make_worker(tmp_path / "candidate", role="D", kind="kv_consumed")
    write_sampled(source, [{"after_call": 0, "token_ids": [80]}])
    write_sampled(destination, [{"after_call": 0, "token_ids": [91, 92, 93]}])
    report = run(tmp_path, mode="pd-kv")
    assert report["status"] == "analysis_complete"
    assert report["sampled_outputs"]["compared_workers"] == 0
    assert report["sampled_outputs"]["first_difference"] is None
    (destination / "sampled.jsonl").unlink()
    assert run(tmp_path, mode="pd-kv")["counts"]["missing_sampled_outputs"] == 1


@pytest.mark.parametrize("include_input,expected", [(False, "decoder"), (True, "model_input")])
def test_first_observation_uses_computational_order_not_minus_one_layer(tmp_path, include_input, expected):
    for root_name, delta in (("reference", 0.0), ("candidate", 0.5)):
        directory = make_worker(tmp_path / root_name, kind="model_output", tensor_name="hidden_states", delta=delta)
        records = read_records(directory)
        records[0]["layer"] = -1
        roles = [
            (-1, "logits", "output"),
            (0, "decoder", "output"),
            (0, "attention", "output"),
            (0, "attention", "query_nope"),
            (0, "sfa", "input"),
            (0, "decoder", "input"),
        ]
        if include_input:
            roles.append((-1, "model_input", "input_ids"))
        for layer, kind, name in roles:
            record = {**records[0], "layer": layer, "kind": kind, "name": name}
            records.append(record)
        write_records(directory, records)
        manifest = read(directory / "manifest.json")
        manifest["records"] = len(records)
        write(directory / "manifest.json", manifest)
        call = read(directory / "calls" / "0.json")
        call["expected"] = [{key: item[key] for key in ("layer", "kind", "name")} for item in records]
        write(directory / "calls" / "0.json", call)
    report = run(tmp_path)
    assert report["first_observed_difference"]["kind"] == expected
    assert "not a proven root cause" in report["first_difference_interpretation"]
    if not include_input:
        assert report["first_observed_difference"]["name"] == "input"
