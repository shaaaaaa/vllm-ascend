#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare complete tool-only tensor archives by logical input context.

``passed`` is a coverage/structure/output/nonfinite gate, not a floating-point
accuracy claim. Calls, batch boundaries and physical cache slots may differ.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layerwise_prefill_correctness_compare import compare_tensor_values


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ints(value: Any) -> bool:
    return isinstance(value, list) and all(type(item) is int and item >= 0 for item in value)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _identity(record: dict) -> tuple:
    return tuple(record[field] for field in ("model", "call", "layer", "kind", "name"))


def _group(record: dict) -> tuple:
    rank = None if record["cross_rank"] else record["rank"]
    return (rank, *(record[field] for field in ("model", "layer", "kind", "name")))


def _tensor_path(stage: Path, relative: Any) -> Path:
    _require(isinstance(relative, str) and bool(relative), "tensor path is missing")
    _require(not Path(relative).is_absolute(), "tensor path must be relative")
    path = (stage / relative).resolve()
    _require(path.is_relative_to(stage.resolve()) and path.is_file(), f"invalid tensor path: {relative}")
    return path


@dataclass
class Archive:
    directory: Path
    output: dict
    records: list[dict]
    calls: dict[tuple, dict]
    coverage: list[dict]
    errors: list[str]


def _coverage_rows(value: Any) -> list[dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "rank" in value:
        return [value]
    if isinstance(value, dict) and isinstance(value.get("ranks"), list):
        return value["ranks"]
    raise ValueError("coverage must contain rank summaries")


def _check_record(record: dict, rank: int, directory: Path) -> None:
    _require(record.get("schema") == 1, "unsupported tensor schema")
    _require(record.get("rank") == rank, "record rank mismatch")
    _require(record.get("model") in ("main", "mtp"), "invalid record model")
    _require(type(record.get("call")) is int and record["call"] >= 0, "invalid call id")
    _require(type(record.get("layer")) is int and record["layer"] >= -1, "invalid layer id")
    _require(
        all(isinstance(record.get(key), str) and record[key] for key in ("kind", "name", "dtype")),
        "record kind/name/dtype missing",
    )
    _require(_ints(record.get("shape")), "invalid tensor shape")
    _require(record.get("phase") in ("prefill", "decode"), "invalid record phase")
    _require(type(record.get("mapping_only")) is bool, "missing mapping classification")
    _require(type(record.get("cross_rank")) is bool, "missing rank layout classification")
    layout = record.get("tensor_layout")
    _require(layout in ("sequence_sharded", "replicated", "rank_local", "mapping"), "invalid tensor layout")
    _require(record["cross_rank"] == (layout in ("sequence_sharded", "replicated")), "rank layout contradiction")
    _require(record["mapping_only"] == (layout == "mapping"), "mapping layout contradiction")
    if record["mapping_only"]:
        _require(
            (
                record["kind"] == "mapping"
                and record["name"]
                in (
                    "indexer_key_slots",
                    "indexer_scale_slots",
                    "attention_slots",
                    "attention_block_table",
                    "current_slots",
                )
            )
            or (record["kind"], record["name"]) == ("attention", "topk"),
            "numeric tensor cannot be marked mapping-only",
        )
    _tensor_path(directory, record.get("path"))
    axis = record.get("row_axis")
    _require(axis is None or type(axis) is int and axis == 0, "unsupported row axis")
    if axis == 0:
        positions, tokens = record.get("positions"), record.get("token_ids")
        _require(_ints(positions) and _ints(tokens) and len(positions) == len(tokens), "invalid row positions/tokens")
        _require(
            type(record.get("valid_rows")) is int and record["valid_rows"] == len(positions),
            "valid_rows disagrees with positions",
        )
        _require(bool(record["shape"]) and len(positions) <= record["shape"][0], "valid rows exceed raw tensor")
        _require(all(size > 0 for size in record["shape"][1:]), "row tensor has an empty feature dimension")
    else:
        _require(
            record.get("positions") is None and record.get("token_ids") is None, "non-row tensor must use call context"
        )


def _mtp_tokens(output: dict) -> int:
    value = output.get("mtp")
    _require(
        isinstance(value, dict)
        and type(value.get("configured_tokens")) is int
        and value["configured_tokens"] in (0, 1),
        "invalid MTP configuration",
    )
    return value["configured_tokens"]


def _check_inventory(models: dict, main_layers: int) -> None:
    for model, inventory in models.items():
        _require(model in ("main", "mtp"), "unsupported model inventory")
        layers = inventory.get("layers")
        _require(_ints(layers) and bool(layers) and layers == sorted(set(layers)), "invalid layer inventory")
        if model == "main":
            _require(layers == list(range(main_layers)), "main layer inventory differs from model configuration")
        else:
            _require(len(layers) == 1, "MTP1 requires exactly one inventoried layer")
        _require(inventory.get("sfa_layers") == layers, "SFA inventory differs from decoder layers")
        indexers = inventory.get("indexer_layers")
        _require(_ints(indexers) and set(indexers) <= set(layers), "invalid indexer inventory")
        roles = inventory.get("required_roles")
        _require(isinstance(roles, list) and bool(roles), "empty required-role inventory")
        listed = {(role["kind"], role["name"]): role["layers"] for role in roles}
        _require(len(listed) == len(roles), "duplicate role inventory")
        for kind, names, selected in (
            ("decoder", ("input", "output", "positions"), layers),
            ("sfa", ("input", "output"), layers),
            ("attention", ("query_nope", "query_rope", "topk", "logical_topk", "output"), layers),
            ("kv_consumed", ("nope", "rope"), layers),
            ("kv_current", ("nope", "rope"), layers),
            ("indexer", ("query", "weights", "topk"), indexers),
            ("indexer_input", ("x", "q_c"), indexers),
            ("kv_indexer", ("key",), indexers),
        ):
            for name in names:
                _require(listed.get((kind, name)) == selected, f"required inventory omitted: {model}/{kind}/{name}")


def _check_rank(summary: dict, records: list[dict], output: dict, calls: dict, stage: str, main_layers: int) -> None:
    rank = summary["rank"]
    _require(summary.get("schema") == 1, "unsupported coverage schema")
    _require(
        summary.get("complete") is True and summary.get("errors") == [],
        f"rank {rank} coverage incomplete: {summary.get('errors')}",
    )
    _require(summary.get("records") == len(records), "coverage record count differs")
    models = summary.get("models")
    _require(isinstance(models, dict) and "main" in models, "missing model inventory")
    _require(not _mtp_tokens(output) or "mtp" in models, "MTP enabled but inventory missing")
    _check_inventory(models, main_layers)
    indexed = {_identity(record): record for record in records}
    _require(len(indexed) == len(records), "duplicate record identity")
    expected_ids: set[tuple] = set()
    main_positions: set[int] = set()
    model_calls: set[str] = set()
    for call in summary.get("calls", []):
        model, number = call.get("model"), call.get("call")
        _require(model in models and type(number) is int and number >= 0, "call outside inventory")
        call_key = (rank, model, number)
        _require(call_key not in calls, "duplicate call identity")
        positions, tokens, context = call.get("positions"), call.get("token_ids"), call.get("context_token_ids")
        _require(_ints(positions) and _ints(tokens) and len(positions) == len(tokens), "invalid call rows")
        _require(
            _ints(context) and all(p < len(context) and context[p] == t for p, t in zip(positions, tokens)),
            "call input context does not describe its positions",
        )
        prompt = output["prompt_token_ids"]
        if model == "main":
            _require(context[: min(len(context), len(prompt))] == prompt[: len(context)], "call prompt context differs")
        _require(call.get("phase") in ("prefill", "decode"), "invalid call phase")
        calls[call_key] = call
        model_calls.add(model)
        if model == "main":
            main_positions.update(positions)
        inventory = models[model]
        layers = inventory.get("layers")
        _require(_ints(layers) and bool(layers) and len(set(layers)) == len(layers), "invalid layer inventory")
        _require(layers == list(range(min(layers), max(layers) + 1)), "noncontiguous layer inventory")
        roles = inventory.get("required_roles")
        _require(isinstance(roles, list) and bool(roles), "empty required-role inventory")
        expected = call.get("expected")
        _require(isinstance(expected, list) and bool(expected), "empty call expectations")
        present = {(item["layer"], item["kind"], item["name"]) for item in expected}
        _require(len(present) == len(expected), "duplicate call expectation")
        for role in roles:
            _require(_ints(role.get("layers")) and set(role["layers"]) <= set(layers), "role outside model layers")
            for layer in role["layers"]:
                _require((layer, role["kind"], role["name"]) in present, "required role omitted from call")
        for layer in layers:
            for name in ("input", "output"):
                _require((layer, "decoder", name) in present, "decoder layer omitted from call")
        for item in expected:
            key = (model, number, item["layer"], item["kind"], item["name"])
            _require(key in indexed, f"missing record {key}")
            expected_ids.add(key)
            record = indexed[key]
            _require(record["phase"] == call["phase"], "record/call phase mismatch")
            _require(
                record.get("call_positions") == positions and record.get("call_token_ids") == tokens,
                "record/call input mismatch",
            )
            for field in ("row_axis", "positions", "token_ids", "mapping_only", "cross_rank", "tensor_layout"):
                _require(record.get(field) == item.get(field), f"record expectation mismatch: {field}")
            if record["row_axis"] == 0:
                _require(
                    all(
                        p < len(context) and context[p] == token
                        for p, token in zip(record["positions"], record["token_ids"])
                    ),
                    "tensor rows do not match call input context",
                )
    _require(expected_ids == set(indexed), "unexpected records outside call coverage")
    required_models = set(models) - ({"mtp"} if stage == "prefill" else set())
    _require(required_models <= model_calls, "model inventory has no captured calls")
    for model in model_calls:
        _require(
            any(
                record["model"] == model
                and record["layer"] == -1
                and (record["kind"], record["name"]) == ("logits", "output")
                for record in records
            ),
            f"{model} pre-sampling logits were not captured",
        )
    prompt_length = len(output["prompt_token_ids"])
    if stage == "prefill":
        required = set(range(prompt_length))
    elif stage == "decode":
        required = set(range(prompt_length - 1, prompt_length + len(output["token_ids"]) - 1))
    else:
        required = set(range(prompt_length + len(output["token_ids"]) - 1))
    _require(required <= main_positions, "main prompt/decode logical input coverage incomplete")


def _configured_layers(directory: Path) -> int:
    config = _read(directory.parent / "model_info.json")
    layers = config.get("num_hidden_layers")
    _require(type(layers) is int and layers > 0, "missing independent model layer count")
    return layers


def _check_indexer_configuration(archive: Archive) -> None:
    config = _read(archive.directory.parent / "model_info.json")
    # The runner records this checkpoint field independently of the probe.
    if "indexer_types" not in config:
        return
    layers = config["num_hidden_layers"]
    kinds = config["indexer_types"]
    pattern = config.get("index_topk_pattern")
    if kinds is None:
        _require(not pattern or "S" not in pattern, "legacy shared-indexer configuration is unsupported")
        producers = list(range(layers))
    else:
        _require(
            isinstance(kinds, list) and len(kinds) == layers and all(kind in ("full", "shared") for kind in kinds),
            "invalid checkpoint indexer inventory",
        )
        if pattern is not None:
            _require(
                isinstance(pattern, list)
                and len(pattern) == layers
                and all((entry == "S") == (kind == "shared") for entry, kind in zip(pattern, kinds)),
                "checkpoint indexer schedules disagree",
            )
        producers = [i for i, kind in enumerate(kinds) if kind == "full"]
    for rank in archive.coverage:
        _require(
            rank["models"]["main"]["indexer_layers"] == producers, "main indexer inventory differs from checkpoint"
        )


def _check_query_coverage(archive: Archive, tp_size: int) -> None:
    """Check TP row union independently of each record's self-described shape."""
    ranks = {row["rank"]: row for row in archive.coverage}
    expectations = {
        key: {(item["layer"], item["kind"], item["name"]): item for item in call["expected"]}
        for key, call in archive.calls.items()
    }
    reference_calls = ranks[0]["calls"]
    for rank in range(1, tp_size):
        left = [{k: v for k, v in call.items() if k != "expected"} for call in reference_calls]
        right = [{k: v for k, v in call.items() if k != "expected"} for call in ranks[rank]["calls"]]
        _require(left == right, "TP ranks observed different model call contexts")
    for call in reference_calls:
        inventory = ranks[0]["models"][call["model"]]
        for role in inventory["required_roles"]:
            if role["kind"] not in ("decoder", "sfa", "attention", "indexer", "indexer_input", "kv_current"):
                continue
            for layer in role["layers"]:
                rows = set()
                for rank in range(tp_size):
                    current = expectations[(rank, call["model"], call["call"])]
                    item = current[(layer, role["kind"], role["name"])]
                    _require(item.get("row_axis") == 0, "query tensor lacks logical row mapping")
                    rows.update(item["positions"])
                _require(rows == set(call["positions"]), "TP query tensor row coverage incomplete")


def _load_archive(
    directory: Path, tp_size: int, stage: str, expected_output_tokens: int | None = None, main_layers: int | None = None
) -> Archive:
    archive = Archive(directory, {}, [], {}, [], [])
    try:
        output = _read(directory / "output.json")
        archive.output = output
        if main_layers is None:
            main_layers = _configured_layers(directory)
        _require(output.get("stage") == stage, "output stage mismatch")
        _require(output.get("completed") is True, "output run is incomplete")
        _require(_ints(output.get("prompt_token_ids")) and len(output["prompt_token_ids"]) >= 2, "invalid prompt ids")
        _require(output.get("prompt_length") == len(output["prompt_token_ids"]), "prompt length mismatch")
        _require(_ints(output.get("token_ids")) and bool(output["token_ids"]), "missing output tokens")
        _require(output.get("enforce_eager") is True, "archive was not produced in eager mode")
        if expected_output_tokens is not None:
            _require(len(output["token_ids"]) == expected_output_tokens, "output token count incomplete")
        expected_cached = len(output["prompt_token_ids"]) - 1 if stage == "decode" else 0
        _require(output.get("num_cached_tokens") == expected_cached, "unexpected cached token count")
        archive.coverage = _coverage_rows(_read(directory / "coverage.json"))
        ranks = [row.get("rank") for row in archive.coverage]
        _require(all(type(rank) is int for rank in ranks), "invalid coverage rank type")
        _require(sorted(ranks) == list(range(tp_size)), "rank coverage missing/extra/duplicate")
        disk_ranks = {path.parent.name for path in (directory / "tensors").glob("rank*/index.jsonl")}
        _require(disk_ranks == {f"rank{rank}" for rank in range(tp_size)}, "manifest rank inventory differs")
        for summary in archive.coverage:
            rank = summary["rank"]
            records = []
            for line in (
                (directory / "tensors" / f"rank{rank}" / "index.jsonl").read_text(encoding="utf-8").splitlines()
            ):
                _require(bool(line.strip()), "blank manifest record")
                record = json.loads(line)
                _check_record(record, rank, directory)
                records.append(record)
            _check_rank(summary, records, output, archive.calls, stage, main_layers)
            archive.records.extend(records)
        inventories = [row["models"] for row in archive.coverage]
        _require(all(value == inventories[0] for value in inventories), "rank model inventories differ")
        _check_indexer_configuration(archive)
        _check_query_coverage(archive, tp_size)
    except (ValueError, TypeError, KeyError, OSError, AttributeError, IndexError, StopIteration) as exc:
        archive.errors.append(f"{stage}: {exc}")
    return archive


def _load_tensor(archive: Archive, record: dict) -> Any:
    torch = importlib.import_module("torch")
    value = torch.load(_tensor_path(archive.directory, record["path"]), map_location="cpu", weights_only=True)
    _require(isinstance(value, torch.Tensor), "archive file is not a Tensor")
    _require(
        list(value.shape) == record["shape"] and str(value.dtype) == record["dtype"],
        "tensor file shape/dtype differs from manifest",
    )
    return value


def validate_baseline(stage_dir: str | Path, tp_size: int, expected_output_tokens: int) -> dict:
    """Fail closed before starting P/D, including malformed/missing tensor files."""
    archive = _load_archive(Path(stage_dir).resolve(), tp_size, "baseline", expected_output_tokens)
    if not archive.errors:
        for record in archive.records:
            try:
                _load_tensor(archive, record)
            except Exception as exc:
                archive.errors.append(f"baseline: {record['path']}: {type(exc).__name__}: {exc}")
                break
    return {
        "valid": not archive.errors,
        "complete": not archive.errors,
        "errors": archive.errors,
        "records": len(archive.records),
        "output_tokens": len(archive.output.get("token_ids", [])),
    }


def _context(archive: Archive, record: dict) -> list[int]:
    return archive.calls[(record["rank"], record["model"], record["call"])]["context_token_ids"]


def _context_prefix(left: list[int], right: list[int]) -> int:
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def _first_output_difference(left: list[int], right: list[int]) -> int | None:
    return next(
        (i for i, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)) if len(left) != len(right) else None,
    )


def _compare_stage(baseline: Archive, candidate: Archive, stage: str, write) -> dict:
    torch = importlib.import_module("torch")
    reference_groups: dict[tuple, list[dict]] = defaultdict(list)
    candidate_groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in baseline.records:
        reference_groups[_group(record)].append(record)
    for record in candidate.records:
        candidate_groups[_group(record)].append(record)
    counts = {
        "records": len(candidate.records),
        "compared": 0,
        "rows": 0,
        "unmatched": 0,
        "mapping_evidence": 0,
        "structural_errors": 0,
        "different": 0,
        "new_nonfinite": 0,
        "integer_mismatched": 0,
    }
    first_difference = worst_difference = None
    context_prefixes: dict[tuple, int] = {}
    output_difference = _first_output_difference(baseline.output["token_ids"], candidate.output["token_ids"])
    prompt_length = len(baseline.output["prompt_token_ids"])
    for key, records in candidate_groups.items():
        refs = list(reference_groups.get(key, []))
        if records[0]["kind"] == "kv_consumed":
            current_key = (*key[:3], "kv_current", key[-1])
            refs.extend(reference_groups.get(current_key, []))
        positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for ref_index, ref in enumerate(refs):
            for row, position in enumerate(ref.get("positions") or []):
                positions[position].append((ref_index, row))
        for record in records:
            detail = {
                "stage": stage,
                **{field: record[field] for field in ("rank", "model", "call", "layer", "kind", "name", "path")},
            }
            try:
                actual = _load_tensor(candidate, record)
                if record["mapping_only"]:
                    counts["mapping_evidence"] += 1
                    write({**detail, "status": "mapping_evidence", "shape": record["shape"]})
                    continue
                context = _context(candidate, record)

                def context_matches(ref: dict, position: int, record=record, context=context) -> bool:
                    pair = (record["rank"], record["model"], record["call"], ref["rank"], ref["call"])
                    if pair not in context_prefixes:
                        context_prefixes[pair] = _context_prefix(_context(baseline, ref), context)
                    return position < context_prefixes[pair]

                if record["row_axis"] == 0:
                    matched: list[tuple[int, int, int]] = []
                    missing = []
                    for row, (position, token) in enumerate(zip(record["positions"], record["token_ids"])):
                        choices = [
                            (i, j)
                            for i, j in positions.get(position, [])
                            if refs[i]["token_ids"][j] == token and context_matches(refs[i], position)
                        ]
                        if choices:
                            i, j = min(
                                choices,
                                key=lambda choice: (
                                    refs[choice[0]]["kind"] != record["kind"],
                                    refs[choice[0]]["call_positions"] != record["call_positions"],
                                    refs[choice[0]]["call_token_ids"] != record["call_token_ids"],
                                    refs[choice[0]]["phase"] != record["phase"],
                                    refs[choice[0]]["rank"] != record["rank"],
                                    choice,
                                ),
                            )
                            matched.append((row, i, j))
                        else:
                            missing.append(position)
                    if missing:
                        counts["unmatched"] += len(missing)
                        boundary = prompt_length - int(record["model"] == "mtp")
                        after = output_difference is not None and min(missing) >= boundary + output_difference
                        write(
                            {
                                **detail,
                                "status": "after_output_divergence" if after else "unmatched_context_or_position",
                                "positions": missing,
                            }
                        )
                    if not matched:
                        if not record["positions"]:
                            write({**detail, "status": "empty_valid_rows", "rows": 0})
                        continue
                    # Group by archive file, then restore candidate row order.
                    # Only one baseline file is loaded at a time.
                    parts = []
                    by_ref: dict[int, list[tuple[int, int]]] = defaultdict(list)
                    for row, i, j in matched:
                        by_ref[i].append((row, j))
                    ordering = []
                    for i, pairs in by_ref.items():
                        reference = _load_tensor(baseline, refs[i])
                        _require(
                            reference.dtype == actual.dtype and reference.shape[1:] == actual.shape[1:],
                            "aligned tensor trailing shape/dtype differs",
                        )
                        parts.append(reference[[j for _, j in pairs]])
                        ordering.extend(row for row, _ in pairs)
                    reference = torch.cat(parts, dim=0)
                    actual = actual[ordering]
                    detail["positions"] = [record["positions"][row] for row in ordering]
                    detail["baseline_sources"] = [
                        dict(
                            rank=refs[i]["rank"],
                            call=refs[i]["call"],
                            kind=refs[i]["kind"],
                            path=refs[i]["path"],
                            rows=len(pairs),
                        )
                        for i, pairs in by_ref.items()
                    ]
                    counts["rows"] += len(ordering)
                else:
                    refs_matching = [
                        ref
                        for ref in refs
                        if ref["row_axis"] is None
                        and ref["call_positions"] == record["call_positions"]
                        and ref["call_token_ids"] == record["call_token_ids"]
                        and _context(baseline, ref) == context
                    ]
                    if not refs_matching:
                        counts["unmatched"] += 1
                        write({**detail, "status": "unmatched_nonrow_context"})
                        continue
                    reference = _load_tensor(baseline, refs_matching[0])
                values = compare_tensor_values(reference, actual)
                if not values.get("comparable"):
                    raise ValueError(values.get("reason", "values cannot be compared"))
                counts["compared"] += 1
                counts["new_nonfinite"] += values["new_nonfinite"]
                if values["category"] == "integer":
                    counts["integer_mismatched"] += values["mismatched"]
                different = values["mismatched"] > 0
                counts["different"] += int(different)
                detail.update(status="different" if different else "equal", comparison=values)
                if different:
                    marker = {
                        field: detail[field] for field in ("stage", "rank", "model", "call", "layer", "kind", "name")
                    }
                    difference_positions = detail.get("positions") or record["call_positions"] or [-1]
                    if record["row_axis"] == 0:
                        left, right = reference, actual
                        if left.is_floating_point():
                            left, right = left.to(torch.float64), right.to(torch.float64)
                            unequal = (left != right) & ~(torch.isnan(left) & torch.isnan(right))
                        else:
                            unequal = left != right
                        rows_differ = unequal.reshape(len(difference_positions), -1).any(dim=1).tolist()
                        difference_positions = [p for p, differs in zip(difference_positions, rows_differ) if differs]
                    marker.update(
                        position=min(difference_positions),
                        abs_max=values["abs_diff"]["max"],
                        relative_l2=values["relative_l2"],
                        rmse_over_std=values["rmse_over_std"],
                    )
                    if first_difference is None or (marker["position"], marker["layer"]) < (
                        first_difference["position"],
                        first_difference["layer"],
                    ):
                        first_difference = marker
                    if worst_difference is None or (marker["abs_max"] or 0) > (worst_difference["abs_max"] or 0):
                        worst_difference = marker
                write(detail)
            except Exception as exc:
                counts["structural_errors"] += 1
                write({**detail, "status": "structural_error", "error": f"{type(exc).__name__}: {exc}"})
    return {"counts": counts, "first_difference": first_difference, "worst_abs_difference": worst_difference}


def compare_run(root: str | Path, baseline_dir: str | Path | None = None) -> dict:
    """Validate all stages and write a context-aligned numerical diagnostic."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    stages: dict[str, dict] = {}
    tokens_equal = False
    baseline_path = Path(baseline_dir).resolve() if baseline_dir is not None else root / "baseline"
    (root / "comparisons.jsonl").write_text("", encoding="utf-8")
    prefill_tokens_equal = False
    try:
        config = _read(root / "run_config.json")
        tp_size = config.get("tp_size", config.get("tensor_parallel_size"))
        expected = config.get("expected_output_tokens", config.get("max_tokens", config.get("output_tokens")))
        _require(type(tp_size) is int and tp_size > 0, "run_config lacks positive tp_size")
        _require(type(expected) is int and expected > 0, "run_config lacks positive expected_output_tokens")
        main_layers = config.get("main_num_layers")
        _require(type(main_layers) is int and main_layers > 0, "run_config lacks positive main_num_layers")
        _require(_configured_layers(root / "baseline") == main_layers, "run_config/model layer count differs")
        if baseline_dir is None and (root / "off_reference.json").is_file():
            supplied = Path(_read(root / "off_reference.json")["baseline_dir"])
            baseline_path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
        baseline = _load_archive(baseline_path, tp_size, "baseline", expected, main_layers)
        prefill = _load_archive(root / "prefill", tp_size, "prefill", 1, main_layers)
        decode = _load_archive(root / "decode", tp_size, "decode", expected, main_layers)
        for archive in (baseline, prefill, decode):
            errors.extend(archive.errors)
        if not errors:
            for archive in (prefill, decode):
                _require(archive.output["prompt_token_ids"] == baseline.output["prompt_token_ids"], "prompt ids differ")
                _require(
                    _mtp_tokens(archive.output) == _mtp_tokens(baseline.output) == config.get("mtp_tokens"),
                    "MTP configuration differs",
                )
                for actual, reference in zip(
                    sorted(archive.coverage, key=lambda row: row["rank"]),
                    sorted(baseline.coverage, key=lambda row: row["rank"]),
                ):
                    _require(actual["models"] == reference["models"], "model/role inventories differ")
            tokens_equal = decode.output["token_ids"] == baseline.output["token_ids"]
            p_tokens = prefill.output["token_ids"]
            prefill_tokens_equal = p_tokens == baseline.output["token_ids"][: len(p_tokens)]
        with (root / "comparisons.jsonl").open("w", encoding="utf-8") as output:

            def write(value: dict) -> None:
                output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

            if not errors:
                for stage, archive in (("prefill", prefill), ("decode", decode)):
                    stages[stage] = _compare_stage(baseline, archive, stage, write)
            for error in errors:
                write({"status": "validation_error", "error": error})
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        with (root / "comparisons.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps({"status": "validation_error", "error": errors[-1]}) + "\n")
    complete = (
        not errors
        and len(stages) == 2
        and all(
            not value["counts"]["unmatched"]
            and not value["counts"]["structural_errors"]
            and value["counts"]["compared"] > 0
            for value in stages.values()
        )
    )
    new_nonfinite = sum(value["counts"]["new_nonfinite"] for value in stages.values())
    passed = complete and tokens_equal and prefill_tokens_equal and new_nonfinite == 0
    report = {
        "schema": 1,
        "passed": passed,
        "complete": complete,
        "output_tokens_equal": tokens_equal,
        "prefill_output_tokens_equal": prefill_tokens_equal,
        "new_nonfinite": new_nonfinite,
        "numeric_tolerance_applied": False,
        "status": "gates_passed_numeric_report_only" if passed else "gates_failed",
        "meaning": "Coverage, alignment, output tokens and new nonfinite gates; floats are statistics only.",
        "baseline_dir": str(baseline_path),
        "errors": errors,
        "stages": stages,
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    report = compare_run(args.root, args.baseline_dir)
    print(f"{report['status']}: complete={report['complete']} output_tokens_equal={report['output_tokens_equal']}")
    for stage, result in report["stages"].items():
        print(f"{stage}: {result['counts']}")
        print(f"{stage} first difference: {result['first_difference']}")
    print(f"Details: {args.root / 'report.json'}; {args.root / 'comparisons.jsonl'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
