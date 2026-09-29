# SPDX-License-Identifier: Apache-2.0
"""CPU archive regression tests; run with pytest --noconftest."""

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "layerwise_prefill_file_compare.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("file_compare_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def manifest(stage, records, rank=0):
    path = stage / "tensors" / f"rank{rank}" / "index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def read_records(stage, rank=0):
    return [json.loads(line) for line in (stage / "tensors" / f"rank{rank}" / "index.jsonl").read_text().splitlines()]


def inventory(layers):
    roles = []
    for kind, names, selected in (
        ("decoder", ("input", "output", "positions"), layers),
        ("sfa", ("input", "output"), layers),
        ("attention", ("query_nope", "query_rope", "topk", "logical_topk", "output"), layers),
        ("kv_consumed", ("nope", "rope"), layers),
        ("kv_current", ("nope", "rope"), layers),
        ("indexer", ("query", "weights", "topk"), []),
        ("indexer_input", ("x", "q_c"), []),
        ("kv_indexer", ("key",), []),
    ):
        roles.extend(dict(kind=kind, name=name, layers=selected) for name in names)
    return dict(layers=layers, sfa_layers=layers, indexer_layers=[], required_roles=roles)


def fixture(root, *, tp=1, mtp=0):
    write(root / "run_config.json", dict(tp_size=tp, expected_output_tokens=3, main_num_layers=1, mtp_tokens=mtp))
    write(root / "model_info.json", dict(num_hidden_layers=1))
    context = [1, 2, 3, 10, 11, 12]
    for stage, batches in (("baseline", [[0, 1, 2], [3], [4]]), ("prefill", [[0], [1, 2]]), ("decode", [[2, 3], [4]])):
        directory = root / stage
        write(
            directory / "output.json",
            dict(
                completed=True,
                stage=stage,
                prompt_token_ids=context[:3],
                prompt_length=3,
                token_ids=[10] if stage == "prefill" else [10, 11, 12],
                output_token_limit=1 if stage == "prefill" else 3,
                num_cached_tokens=2 if stage == "decode" else 0,
                mtp={"configured_tokens": mtp, "metrics": {"drafts": 0 if stage == "prefill" else 2}},
                enforce_eager=True,
            ),
        )
        summaries = []
        for rank in range(tp):
            records, calls = [], []
            models = {"main": inventory([0])}
            if mtp:
                models["mtp"] = inventory([1])
            for model, info in models.items():
                model_context = context if model == "main" else context[1:]
                model_batches = batches if model == "main" else ([] if stage == "prefill" else [[3], [4]])
                for number, positions in enumerate(model_batches):
                    # Deliberately unrelated call IDs across the three processes.
                    number += {"baseline": 0, "prefill": 10, "decode": 20}[stage]
                    call = dict(
                        model=model,
                        call=number,
                        phase="prefill" if min(positions) < 3 else "decode",
                        positions=positions,
                        token_ids=[model_context[p] for p in positions],
                        context_token_ids=model_context[: max(positions) + 1],
                        expected=[],
                    )
                    calls.append(call)
                    roles = list(info["required_roles"])
                    if max(positions) >= 2:
                        roles.append(dict(kind="logits", name="output", layers=[-1]))
                    for role in roles:
                        for layer in role["layers"]:
                            rows = list(range(max(positions) + 1)) if role["kind"] == "kv_consumed" else positions
                            if role["kind"] == "logits":
                                rows = [p for p in positions if p >= 2]
                            # Raw archives include deliberately different padding.
                            value = torch.tensor(
                                [[float(p + layer), float(p + layer + 1)] for p in rows]
                                + [[float("nan"), float("nan")]]
                            )
                            if role["name"] in ("positions", "topk", "logical_topk"):
                                value = torch.tensor([[p, p + 1] for p in rows] + [[999, 999]], dtype=torch.int64)
                            path = f"tensors/rank{rank}/{len(records)}.pt"
                            (directory / path).parent.mkdir(parents=True, exist_ok=True)
                            torch.save(value, directory / path)
                            record = dict(
                                schema=1,
                                rank=rank,
                                model=model,
                                call=number,
                                layer=layer,
                                kind=role["kind"],
                                name=role["name"],
                                path=path,
                                shape=list(value.shape),
                                dtype=str(value.dtype),
                                positions=rows,
                                token_ids=[model_context[p] for p in rows],
                                row_axis=0,
                                valid_rows=len(rows),
                                phase=call["phase"],
                                call_positions=positions,
                                call_token_ids=call["token_ids"],
                                mapping_only=(role["kind"], role["name"]) == ("attention", "topk"),
                            )
                            record.update(
                                cross_rank=False, tensor_layout="mapping" if record["mapping_only"] else "rank_local"
                            )
                            records.append(record)
                            call["expected"].append(
                                {
                                    key: record[key]
                                    for key in (
                                        "layer",
                                        "kind",
                                        "name",
                                        "row_axis",
                                        "positions",
                                        "token_ids",
                                        "mapping_only",
                                        "cross_rank",
                                        "tensor_layout",
                                    )
                                }
                            )
            manifest(directory, records, rank)
            summaries.append(
                dict(schema=1, rank=rank, complete=True, errors=[], records=len(records), models=models, calls=calls)
            )
        write(directory / "coverage.json", summaries)
    return root


def details(root):
    return [json.loads(line) for line in (root / "comparisons.jsonl").read_text().splitlines()]


def change_tensor(root, *, stage="decode", name="query_nope", position=4, value=0.25):
    directory = root / stage
    record = next(row for row in read_records(directory) if row["name"] == name and position in row["positions"])
    tensor = torch.load(directory / record["path"], weights_only=True)
    tensor[record["positions"].index(position), 0] += value
    torch.save(tensor, directory / record["path"])
    return record


def test_aligns_split_batches_recomputed_prompt_tail_padding_and_mtp(tmp_path):
    fixture(tmp_path, tp=2, mtp=1)
    report = compare.compare_run(tmp_path)
    assert report["passed"], report
    assert report["complete"] and report["output_tokens_equal"]
    assert not report["numeric_tolerance_applied"]
    assert report["stages"]["decode"]["counts"]["different"] == 0
    assert report["stages"]["decode"]["counts"]["mapping_evidence"] > 0
    rows = [row for row in details(tmp_path) if row["stage"] == "decode" and row["kind"] == "decoder"]
    assert any(2 in row["positions"] and 3 in row["positions"] for row in rows)
    assert any(row["model"] == "mtp" for row in rows)


def test_float_difference_is_statistics_not_failure_with_exact_first_position(tmp_path):
    fixture(tmp_path)
    change_tensor(tmp_path, position=3)
    report = compare.compare_run(tmp_path)
    assert report["passed"]
    first = report["stages"]["decode"]["first_difference"]
    assert first["position"] == 3 and first["name"] == "query_nope"
    changed = next(row for row in details(tmp_path) if row.get("status") == "different")
    values = changed["comparison"]
    assert values["abs_diff"]["max"] == 0.25
    assert values["relative_l2"] > 0 and values["rmse_over_std"] > 0
    assert set(values["baseline"]) >= {"mean", "std", "min", "max"}


def test_integer_topk_difference_reported_separately(tmp_path):
    fixture(tmp_path)
    change_tensor(tmp_path, name="logical_topk", value=1)
    report = compare.compare_run(tmp_path)
    assert report["passed"]
    assert report["stages"]["decode"]["counts"]["integer_mismatched"] == 1


def test_new_nonfinite_fails_gate_but_padding_nan_does_not(tmp_path):
    fixture(tmp_path)
    change_tensor(tmp_path, value=float("nan"))
    report = compare.compare_run(tmp_path)
    assert report["complete"] and not report["passed"] and report["new_nonfinite"] == 1


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_output_difference_keeps_numerical_comparisons(tmp_path, stage):
    fixture(tmp_path)
    path = tmp_path / stage / "output.json"
    value = json.loads(path.read_text())
    value["token_ids"][0] = 999
    write(path, value)
    report = compare.compare_run(tmp_path)
    assert report["complete"] and not report["passed"]
    assert report["stages"][stage]["counts"]["compared"] > 0
    assert not report["prefill_output_tokens_equal" if stage == "prefill" else "output_tokens_equal"]


@pytest.mark.parametrize(
    "damage",
    [
        "missing_rank",
        "missing_record",
        "extra_record",
        "missing_file",
        "bad_file",
        "shape",
        "dtype",
        "incomplete",
        "layers",
        "sfa",
        "roles",
        "rows",
        "cached",
    ],
)
def test_coverage_and_archive_damage_fails_closed(tmp_path, damage):
    fixture(tmp_path, tp=2)
    stage = tmp_path / "decode"
    summaries = json.loads((stage / "coverage.json").read_text())
    records = read_records(stage)
    if damage == "missing_rank":
        summaries.pop()
    elif damage == "missing_record":
        records.pop()
        summaries[0]["records"] -= 1
    elif damage == "extra_record":
        records.append(copy.deepcopy(records[0]))
        records[-1]["call"] = 999
        summaries[0]["records"] += 1
    elif damage == "missing_file":
        (stage / records[0]["path"]).unlink()
    elif damage == "bad_file":
        (stage / records[0]["path"]).write_bytes(b"not a tensor")
    elif damage == "shape":
        records[0]["shape"][-1] += 1
    elif damage == "dtype":
        records[0]["dtype"] = "torch.float16"
    elif damage == "incomplete":
        summaries[0]["complete"] = False
    elif damage == "layers":
        for summary in summaries:
            summary["models"]["main"]["layers"] = [1]
    elif damage == "sfa":
        for summary in summaries:
            summary["models"]["main"]["sfa_layers"] = []
    elif damage == "roles":
        for summary in summaries:
            summary["models"]["main"]["required_roles"] = [
                role for role in summary["models"]["main"]["required_roles"] if role["kind"] != "attention"
            ]
    elif damage == "rows":
        # Self-consistent empty records on every rank must not hide real query rows.
        for rank in (0, 1):
            current = read_records(stage, rank)
            for record in current:
                if record["kind"] == "decoder":
                    record.update(positions=[], token_ids=[], valid_rows=0)
            for call in summaries[rank]["calls"]:
                for item in call["expected"]:
                    if item["kind"] == "decoder":
                        item.update(positions=[], token_ids=[])
            manifest(stage, current, rank)
        records = read_records(stage)
    elif damage == "cached":
        output = json.loads((stage / "output.json").read_text())
        output["num_cached_tokens"] = 0
        write(stage / "output.json", output)
    manifest(stage, records)
    write(stage / "coverage.json", summaries)
    report = compare.compare_run(tmp_path)
    assert not report["passed"] and not report["complete"], (damage, report)


def test_context_divergence_is_unmatched_and_does_not_compare_wrong_token(tmp_path):
    fixture(tmp_path)
    stage = tmp_path / "decode"
    coverage = json.loads((stage / "coverage.json").read_text())
    output = json.loads((stage / "output.json").read_text())
    output["token_ids"][0] = 99
    write(stage / "output.json", output)
    records = read_records(stage)
    for call in coverage[0]["calls"]:
        call["context_token_ids"][3] = 99
        call["token_ids"] = [call["context_token_ids"][p] for p in call["positions"]]
        for item in call["expected"]:
            item["token_ids"] = [call["context_token_ids"][p] for p in item["positions"]]
        for record in records:
            if record["call"] == call["call"]:
                record["call_token_ids"] = call["token_ids"]
                record["token_ids"] = [call["context_token_ids"][p] for p in record["positions"]]
    manifest(stage, records)
    write(stage / "coverage.json", coverage)
    report = compare.compare_run(tmp_path)
    assert not report["passed"] and not report["complete"]
    assert any(row["status"] == "after_output_divergence" for row in details(tmp_path))


def test_validate_baseline_requires_full_decode_and_independent_inventory(tmp_path):
    fixture(tmp_path, mtp=1)
    assert compare.validate_baseline(tmp_path / "baseline", 1, 3)["valid"]
    assert not compare.validate_baseline(tmp_path / "baseline", 1, 4)["valid"]
    assert not compare.validate_baseline(tmp_path / "baseline", 2, 3)["valid"]
    write(tmp_path / "model_info.json", {"num_hidden_layers": 2})
    assert not compare.validate_baseline(tmp_path / "baseline", 1, 3)["valid"]


def test_reuses_external_baseline_without_modifying_it(tmp_path):
    first, second = tmp_path / "original", tmp_path / "new"
    fixture(first)
    shutil.copytree(first, second)
    shutil.rmtree(second / "baseline")
    write(second / "off_reference.json", {"baseline_dir": str(first / "baseline")})
    report = compare.compare_run(second)
    assert report["passed"]
    assert not (first / "report.json").exists()
    assert not (second / "baseline").exists()


def test_failed_rerun_truncates_previous_details(tmp_path):
    fixture(tmp_path)
    assert compare.compare_run(tmp_path)["passed"]
    (tmp_path / "run_config.json").unlink()
    assert not compare.compare_run(tmp_path)["passed"]
    assert len(details(tmp_path)) == 1 and details(tmp_path)[0]["status"] == "validation_error"


@pytest.mark.parametrize("cross_rank", [True, False])
def test_sequence_shard_position_moves_ranks_only_with_explicit_full_feature_layout(tmp_path, cross_rank):
    fixture(tmp_path, tp=2)
    for name in ("baseline", "prefill", "decode"):
        stage = tmp_path / name
        coverage = json.loads((stage / "coverage.json").read_text())
        for rank in (0, 1):
            records = read_records(stage, rank)
            for record in records:
                if record["kind"] in ("kv_consumed", "logits"):
                    continue
                previous = record["positions"]
                # Baseline tail is on rank1, while D tail is on rank0.
                selected = [i for i, p in enumerate(previous) if (p + (name == "decode") + 1) % 2 == rank]
                value = torch.load(stage / record["path"], weights_only=True)
                value = value[selected + [len(previous)]]
                torch.save(value, stage / record["path"])
                record.update(
                    shape=list(value.shape),
                    positions=[previous[i] for i in selected],
                    token_ids=[record["token_ids"][i] for i in selected],
                    valid_rows=len(selected),
                )
                if not record["mapping_only"] and cross_rank:
                    record.update(cross_rank=True, tensor_layout="sequence_sharded")
                call = next(call for call in coverage[rank]["calls"] if call["call"] == record["call"])
                expected = next(
                    item
                    for item in call["expected"]
                    if (item["layer"], item["kind"], item["name"]) == (record["layer"], record["kind"], record["name"])
                )
                expected.update({key: record[key] for key in expected})
            manifest(stage, records, rank)
        write(stage / "coverage.json", coverage)
    report = compare.compare_run(tmp_path)
    assert report["passed"] is cross_rank, report
    if cross_rank:
        tail = next(
            row
            for row in details(tmp_path)
            if row["stage"] == "decode"
            and row["rank"] == 0
            and row["kind"] == "decoder"
            and row.get("positions") == [2]
        )
        assert tail["baseline_sources"][0]["rank"] == 1
    else:
        assert report["stages"]["decode"]["counts"]["unmatched"] > 0


def test_duplicate_logical_kv_rows_are_compared_not_dropped(tmp_path):
    fixture(tmp_path)
    stage = tmp_path / "decode"
    records = read_records(stage)
    record = next(row for row in records if row["kind"] == "kv_consumed")
    value = torch.load(stage / record["path"], weights_only=True)
    value = torch.cat((value[:1], value), dim=0)
    value[0, 0] += 0.5
    torch.save(value, stage / record["path"])
    record["positions"].insert(0, record["positions"][0])
    record["token_ids"].insert(0, record["token_ids"][0])
    record.update(shape=list(value.shape), valid_rows=len(record["positions"]))
    coverage = json.loads((stage / "coverage.json").read_text())
    item = next(
        item
        for call in coverage[0]["calls"]
        if call["call"] == record["call"]
        for item in call["expected"]
        if item["kind"] == record["kind"] and item["name"] == record["name"]
    )
    item.update(positions=record["positions"], token_ids=record["token_ids"])
    manifest(stage, records)
    write(stage / "coverage.json", coverage)
    report = compare.compare_run(tmp_path)
    assert report["passed"]
    delta = next(row for row in details(tmp_path) if row["status"] == "different")
    assert delta["comparison"]["mismatched"] == 1
    assert delta["positions"].count(0) == 2


def test_common_missing_indexer_inventory_cannot_override_checkpoint(tmp_path):
    fixture(tmp_path)
    write(
        tmp_path / "model_info.json", {"num_hidden_layers": 1, "indexer_types": ["full"], "index_topk_pattern": ["F"]}
    )
    report = compare.compare_run(tmp_path)
    assert not report["complete"] and not report["passed"]
    assert any("indexer inventory differs" in error for error in report["errors"])


def test_consumed_kv_can_use_raw_baseline_when_topk_selection_differs(tmp_path):
    fixture(tmp_path)
    stage = tmp_path / "baseline"
    coverage = json.loads((stage / "coverage.json").read_text())
    records = read_records(stage)
    for record in records:
        if record["kind"] != "kv_consumed" or 2 not in record["positions"]:
            continue
        removed = record["positions"].index(2)
        value = torch.load(stage / record["path"], weights_only=True)
        value = value[[i for i in range(value.shape[0]) if i != removed]]
        torch.save(value, stage / record["path"])
        record["positions"].pop(removed)
        record["token_ids"].pop(removed)
        record.update(shape=list(value.shape), valid_rows=len(record["positions"]))
        item = next(
            item
            for call in coverage[0]["calls"]
            if call["call"] == record["call"]
            for item in call["expected"]
            if item["kind"] == record["kind"] and item["name"] == record["name"]
        )
        item.update(positions=record["positions"], token_ids=record["token_ids"])
    manifest(stage, records)
    write(stage / "coverage.json", coverage)
    report = compare.compare_run(tmp_path)
    assert report["passed"], report
    consumed = [row for row in details(tmp_path) if row["stage"] == "decode" and row["kind"] == "kv_consumed"]
    assert any(source["kind"] == "kv_current" for row in consumed for source in row["baseline_sources"])


def test_common_missing_logits_cannot_be_hidden_in_dynamic_expectations(tmp_path):
    fixture(tmp_path)
    for name in ("baseline", "prefill", "decode"):
        stage = tmp_path / name
        records = [record for record in read_records(stage) if record["kind"] != "logits"]
        coverage = json.loads((stage / "coverage.json").read_text())
        coverage[0]["records"] = len(records)
        for call in coverage[0]["calls"]:
            call["expected"] = [item for item in call["expected"] if item["kind"] != "logits"]
        manifest(stage, records)
        write(stage / "coverage.json", coverage)
    report = compare.compare_run(tmp_path)
    assert not report["complete"]
    assert any("logits were not captured" in error for error in report["errors"])
