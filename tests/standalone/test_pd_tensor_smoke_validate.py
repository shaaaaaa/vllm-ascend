# SPDX-License-Identifier: Apache-2.0
"""CPU smoke validation against archives written by the actual recorder."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

validator = importlib.import_module("pd_tensor_smoke_validate")


@pytest.fixture
def recorder():
    spec = importlib.util.spec_from_file_location(
        "pd_tensor_dump_smoke_fixture", ROOT / "vllm_ascend/pd_tensor_dump.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def mutate_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    write_json(path, data)


def make_case(recorder, root, *, layerwise=True, tp_size=2, p_chunks=None, d_chunks=None, extra=False, output_tokens=3):
    prompt = [10, 11, 12, 13]
    outputs = list(range(100, 100 + output_tokens))
    p_chunks = [[0, 1], [2, 3]] if p_chunks is None else p_chunks
    d_chunks = [[position] for position in range(3, 3 + output_tokens)] if d_chunks is None else d_chunks
    archives = {}
    for role, stage, chunks in (("P", "prefill", p_chunks), ("D", "decode", d_chunks)):
        output_tokens = outputs[:1] if role == "P" else outputs
        write_json(
            root / stage / "output.json",
            dict(
                completed=True,
                stage=stage,
                case="on" if layerwise else "off",
                request_id="0",
                prompt_token_ids=prompt,
                prompt_length=len(prompt),
                token_ids=output_tokens,
                output_token_limit=len(output_tokens),
                num_cached_tokens=0 if role == "P" else len(prompt) - 1,
                finish_reason="length",
            ),
        )
        for rank in range(tp_size):
            archive = recorder.RequestArchive(
                root / "capture",
                dict(
                    role=role,
                    request_id="0",
                    internal_request_id=f"0-{role}-random",
                    model_id="test-model",
                    num_layers=1,
                    layerwise_prefill=layerwise if role == "P" else False,
                    prompt_token_ids=prompt,
                    host="single-host",
                    pid=(100 if role == "P" else 200) + rank,
                    tp_rank=rank,
                    tp_size=tp_size,
                    dp_rank=0,
                    dp_size=1,
                ),
            )
            # Use the recorder's real required inventory and writer rather
            # than a lookalike JSON fixture with weaker invariants.
            expected = recorder.PDTensorDump.expected(SimpleNamespace(layers={0: None}))
            samples = []
            for index, positions in enumerate(chunks):
                context = (prompt + outputs)[: max(positions) + 1]
                tokens = [context[position] for position in positions]
                call = archive.begin(
                    dict(
                        phase="prefill" if min(positions) < len(prompt) else "decode",
                        positions=positions,
                        token_ids=tokens,
                        context_token_ids=context,
                        context_complete=True,
                    ),
                    expected,
                )
                observed = set()
                for item in expected:
                    if item["kind"] == "model_input":
                        values = positions if item["name"] == "positions" else tokens
                        tensor = torch.tensor(values)
                    else:
                        tensor = torch.arange(len(positions), dtype=torch.float32).reshape(-1, 1)
                    archive.record(tensor, call, item["layer"], item["kind"], item["name"], positions)
                    observed.add((item["layer"], item["kind"], item["name"]))
                if extra:
                    archive.record(torch.zeros(len(positions), 1), call, 0, "indexer", "query", positions)
                    observed.add((0, "indexer", "query"))
                archive.end(call, observed)
                samples.append(
                    dict(
                        after_call=index,
                        token_ids=([] if index < len(chunks) - 1 else output_tokens)
                        if role == "P"
                        else [outputs[index]],
                    )
                )
            (archive.root / "sampled.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8"
            )
            archive.finish()
            archives[(role, rank)] = archive
    return SimpleNamespace(
        root=root, archives=archives, prompt=prompt, outputs=outputs, layerwise=layerwise, tp_size=tp_size
    )


def validate(case, **kwargs):
    options = dict(output_tokens=3, expect_layerwise=case.layerwise)
    options.update(kwargs)
    return validator.validate_case(case.root, case.tp_size, case.prompt, **options)


def records(archive):
    return [json.loads(line) for line in (archive.root / "index.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("layerwise", [False, True])
def test_actual_recorder_archives_cover_both_p_and_d(recorder, tmp_path, layerwise):
    case = make_case(recorder, tmp_path, layerwise=layerwise, extra=True)
    report = validate(case, model_id="test-model")
    assert report["complete"], report["errors"]
    assert report["details"]["request_id"] == "0"
    assert report["details"]["roles"]["P"][0]["prefill_calls"] == 2
    assert report["details"]["roles"]["D"][1]["decode_calls"] == 2
    assert "passed" not in report
    json.dumps(report)


@pytest.mark.parametrize(
    "role,change,expected",
    [
        ("D", {"layerwise_prefill": True}, "layerwise_prefill"),
        ("P", {"complete": False}, "incomplete_worker"),
        ("D", {"request_finished": False}, "incomplete_worker"),
        ("P", {"errors": ["disk full"]}, "incomplete_worker"),
        ("D", {"model_id": "wrong-model"}, "inventories differ"),
        ("D", {"num_layers": 2}, "required main-backbone"),
        ("P", {"prompt_token_ids": [1, 2]}, "worker prompt"),
        ("D", {"tp_size": 3}, "TP size mismatch"),
    ],
)
def test_invalid_worker_metadata_is_not_complete(recorder, tmp_path, role, change, expected):
    case = make_case(recorder, tmp_path)
    mutate_json(case.archives[(role, 0)].root / "manifest.json", lambda data: data.update(change))
    report = validate(case)
    assert not report["complete"]
    assert any(expected in error for error in report["errors"]), report["errors"]


@pytest.mark.parametrize(
    "stage,change,expected",
    [
        ("prefill", {"completed": False}, "did not complete"),
        ("decode", {"num_cached_tokens": 0}, "cached token count"),
        ("decode", {"token_ids": [100]}, "token count"),
        ("decode", {"token_ids": []}, "token count"),
        ("decode", {"output_token_limit": 64}, "token limit"),
        ("decode", {"request_id": "other"}, "request IDs differ"),
        ("decode", {"token_ids": [99, 101, 102]}, "first output token differs"),
        ("prefill", {"case": "off"}, "label mismatch"),
        ("decode", {"finish_reason": "stop"}, "requested output limit"),
        ("decode", {"prompt_token_ids": [1, 2, 3, 4]}, "prompt mismatch"),
    ],
)
def test_output_evidence_must_match_the_smoke_request(recorder, tmp_path, stage, change, expected):
    case = make_case(recorder, tmp_path)
    mutate_json(tmp_path / stage / "output.json", lambda data: data.update(change))
    report = validate(case)
    assert not report["complete"]
    assert any(expected in error for error in report["errors"]), report["errors"]


def test_missing_tp_rank_cannot_pass(recorder, tmp_path):
    case = make_case(recorder, tmp_path)
    (case.archives[("D", 1)].root / "manifest.json").rename(case.archives[("D", 1)].root / "missing.json")
    report = validate(case)
    assert not report["complete"]
    assert any("D TP rank coverage" in error for error in report["errors"])


@pytest.mark.parametrize("chunks", [[[0, 1], [3]], [[0, 1], [1, 2, 3]], [[2, 3], [0, 1]]])
def test_p_must_cover_full_prompt_once_in_order(recorder, tmp_path, chunks):
    case = make_case(recorder, tmp_path, p_chunks=chunks)
    report = validate(case)
    assert not report["complete"]
    assert any("prompt exactly once" in error for error in report["errors"])


def test_single_prefill_chunk_only_allowed_by_explicit_minimum(recorder, tmp_path):
    case = make_case(recorder, tmp_path, p_chunks=[[0, 1, 2, 3]])
    assert not validate(case)["complete"]
    assert validate(case, min_prefill_calls=1)["complete"]


def test_d_full_prompt_recomputation_is_rejected_even_if_api_claims_cache_hit(recorder, tmp_path):
    case = make_case(recorder, tmp_path, d_chunks=[[0, 1, 2, 3], [4], [5]])
    report = validate(case)
    assert not report["complete"]
    assert any("prefix may have recomputed" in error for error in report["errors"])


def test_missing_raw_tensor_is_not_complete(recorder, tmp_path):
    case = make_case(recorder, tmp_path)
    archive = case.archives[("P", 1)]
    (archive.root / records(archive)[0]["path"]).unlink()
    report = validate(case)
    assert not report["complete"]
    assert any("missing tensor file" in error for error in report["errors"])


@pytest.mark.parametrize("change", ["wrong_value", "wrong_shape", "wrong_dtype", "not_tensor", "corrupt"])
def test_actual_input_payload_must_match_metadata(recorder, tmp_path, change):
    case = make_case(recorder, tmp_path)
    archive = case.archives[("P", 0)]
    row = next(row for row in records(archive) if (row["kind"], row["name"]) == ("model_input", "input_ids"))
    path = archive.root / row["path"]
    value = {
        "wrong_value": torch.tensor([10, 99]),
        "wrong_shape": torch.tensor([10]),
        "wrong_dtype": torch.tensor([10.0, 11.0]),
        "not_tensor": [10, 11],
    }.get(change)
    if change == "corrupt":
        path.write_bytes(b"not a tensor")
    else:
        torch.save(value, path)
    report = validate(case)
    assert not report["complete"]
    assert any("P TP0:" in error for error in report["errors"])


def test_declaring_empty_expected_inventory_does_not_hide_missing_capture(recorder, tmp_path):
    case = make_case(recorder, tmp_path)
    archive = case.archives[("D", 0)]
    remaining = [row for row in records(archive) if not (row["call"] == 1 and row["kind"] == "decoder")]
    (archive.root / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in remaining), encoding="utf-8")
    mutate_json(archive.root / "manifest.json", lambda data: data.update(records=len(remaining)))
    mutate_json(archive.root / "calls/1.json", lambda data: data.update(expected=[]))
    report = validate(case)
    assert not report["complete"]
    assert any("required main-backbone" in error for error in report["errors"])


def test_worker_sampled_tokens_must_match_api_output(recorder, tmp_path):
    case = make_case(recorder, tmp_path)
    archive = case.archives[("D", 1)]
    path = archive.root / "sampled.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["token_ids"] = [999]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = validate(case)
    assert not report["complete"]
    assert any("accepted tokens differ" in error for error in report["errors"])


@pytest.mark.parametrize("bonus,mtp,passes", [(1, 1, True), (2, 1, False), (1, 0, False)])
def test_final_mtp_bonus_may_be_truncated_by_scheduler(recorder, tmp_path, bonus, mtp, passes):
    case = make_case(recorder, tmp_path, tp_size=1, output_tokens=16)
    mutate_json(tmp_path / "decode/output.json", lambda data: data.update(mtp=dict(configured_tokens=mtp)))
    for rank in range(case.tp_size):
        path = case.archives[("D", rank)].root / "sampled.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["token_ids"].extend([999] * bonus)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = validate(case, output_tokens=16)
    assert report["complete"] is passes, report["errors"]
    if passes:
        assert report["details"]["roles"]["D"][0]["clipped_final_tokens"] == 1
        assert report["details"]["roles"]["D"][0]["sampled_tokens"] == 17
        assert report["details"]["roles"]["D"][0]["api_output_tokens"] == 16


def test_mtp_does_not_permit_sampling_an_extra_call_after_output_limit(recorder, tmp_path):
    case = make_case(recorder, tmp_path)
    mutate_json(tmp_path / "decode/output.json", lambda data: data.update(mtp=dict(configured_tokens=1)))
    path = case.archives[("D", 0)].root / "sampled.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-2]["token_ids"] = [101, 102]
    rows[-1]["token_ids"] = [999]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = validate(case)
    assert not report["complete"]
    assert any("continued sampling after" in error for error in report["errors"])


def test_requested_model_identity_must_match(recorder, tmp_path):
    case = make_case(recorder, tmp_path)
    report = validate(case, model_id="different-model")
    assert not report["complete"]
    assert "captured model ID differs from requested model" in report["errors"]


def test_no_archives_or_outputs_is_incomplete(tmp_path):
    report = validator.validate_case(tmp_path, 2, [1, 2, 3, 4])
    assert not report["complete"]
    assert any("no_worker_manifests" in error for error in report["errors"])


def test_real_recorder_validation_analyzer_and_smoke_report_integrate(recorder, tmp_path, monkeypatch):
    smoke = importlib.import_module("pd_tensor_smoke")
    file_store = importlib.import_module("layerwise_prefill_file_store")
    # Byte transport has its own real CPU roundtrip tests. Everything after
    # capture here (inventory, raw tensor compare and final gate) runs normally.
    monkeypatch.setattr(file_store, "validate_store", lambda root: dict(passed=True, errors=[], reads=2, groups=[0, 1]))
    case = make_case(recorder, tmp_path / "on", layerwise=True, tp_size=2)
    write_json(
        tmp_path / "run_config.json",
        dict(
            schema_version=1,
            tool="pd_tensor_smoke",
            cases=["on"],
            tp_size=2,
            min_prefill_calls=2,
            options=dict(output_tokens=3, model="test-model"),
        ),
    )
    write_json(tmp_path / "prompt.json", dict(token_ids=case.prompt))
    assert smoke.analyze_run(tmp_path) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["smoke_passed"], report["errors"]
    assert report["accuracy_verdict"] == "not_assessed"
    assert report["cases"]["on"]["coverage"]["complete"]
    assert report["cases"]["on"]["pd_analysis"]["status"] == "analysis_complete"


@pytest.mark.parametrize(
    "kwargs",
    [{"tp_size": 0}, {"prompt_ids": []}, {"output_tokens": 1}, {"expect_layerwise": 1}, {"min_prefill_calls": 0}],
)
def test_invalid_validation_options_fail_closed(tmp_path, kwargs):
    options = dict(tp_size=2, prompt_ids=[1, 2, 3, 4])
    options.update(kwargs)
    report = validator.validate_case(tmp_path, **options)
    assert not report["complete"]
    assert len(report["errors"]) == 1
