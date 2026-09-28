#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze raw tensor dumps from real P/D services, without tensor hashes.

Examples::

    python tools/pd_tensor_analyze.py --mode off-on --reference OFF --candidate ON --output report
    python tools/pd_tensor_analyze.py --mode pd-kv --reference COLLECT --candidate COLLECT --output report

Directories may contain dumps collected from multiple machines. Request IDs are
matched literally unless --request-map supplies an explicit old-ID/new-ID map.
TP ranks remain separate; DP rank numbers need not agree between P and D.
Numerical results describe differences, never an automatic accuracy verdict.
OFF/ON also compares accepted sampled outputs for the same request prompt and
TP rank. PD KV mode deliberately does not compare outputs/logits across phases.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from layerwise_prefill_correctness_compare import compare_tensor_values

IDENTITY_FIELDS = ("model", "layer", "kind", "name")
PD_KINDS = {"kv_consumed": "kv_current", "kv_indexer": "kv_indexer"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _integers(value: Any) -> bool:
    return isinstance(value, list) and all(_integer(item) for item in value)


def _tokens(value: Any, *, incomplete: bool) -> bool:
    return isinstance(value, list) and all(_integer(item) or incomplete and item is None for item in value)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(record: dict) -> tuple:
    return tuple(record[key] for key in IDENTITY_FIELDS)


def _local_path(directory: Path, value: Any) -> Path:
    _require(isinstance(value, str) and bool(value), "missing tensor path")
    path = Path(value)
    _require(not path.is_absolute() and path.suffix == ".pt", "tensor path must be relative .pt")
    resolved = (directory / path).resolve()
    _require(resolved.is_relative_to(directory.resolve()), "tensor path escapes worker directory")
    _require(resolved.is_file(), f"missing tensor file: {value}")
    return resolved


@dataclass
class Worker:
    directory: Path
    manifest: dict
    calls: dict[int, dict] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    sampled: list[dict] = field(default_factory=list)
    sampled_complete: bool = False

    @property
    def key(self) -> tuple:
        return tuple(self.manifest[key] for key in ("role", "request_id", "tp_rank"))

    def evidence(self) -> dict:
        return {
            **{key: self.manifest.get(key) for key in ("role", "request_id", "tp_rank", "dp_rank", "host", "pid")},
            "worker_directory": str(self.directory),
        }


@dataclass
class Archive:
    workers: dict[tuple, Worker] = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)


def _load_sampled(worker: Worker, archive: Archive) -> None:
    """Validate the CPU accepted-token stream, including empty prefill samples."""
    path = worker.directory / "sampled.jsonl"
    if not path.is_file():
        archive.issues.append({**worker.evidence(), "status": "missing_sampled_outputs"})
        return
    seen = set()
    previous_call = -1
    valid = True
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
            _require(isinstance(item, dict), "sampled record must be an object")
            number, tokens = item.get("after_call"), item.get("token_ids")
            _require(_integer(number) and number in worker.calls, "sampled record has no valid call")
            _require(worker.calls[number].get("model", "main") == "main", "accepted samples belong to main calls")
            _require(number >= previous_call, "sampled calls are not ordered")
            _require(_integers(tokens), "sampled outputs must contain accepted nonnegative token IDs")
            _require(number not in seen, "duplicate sampled output for one call")
            previous_call = number
            seen.add(number)
            worker.sampled.append(item)
        except Exception as exc:
            valid = False
            archive.issues.append(
                {**worker.evidence(), "status": "invalid_sampled_outputs", "line": line_number, "reason": str(exc)}
            )
    main_calls = {number for number, call in worker.calls.items() if call.get("model", "main") == "main"}
    missing = sorted(main_calls - seen)
    if missing:
        valid = False
        archive.issues.append({**worker.evidence(), "status": "missing_sampled_calls", "calls": missing})
    worker.sampled_complete = valid and len(worker.calls) == len(worker.manifest["calls"])


def _check_call(call: dict, number: int) -> None:
    _require(call.get("schema") == 1 and call.get("call") == number, "invalid call schema/id")
    _require(call.get("model", "main") in ("main", "mtp"), "invalid call model")
    _require(call.get("phase") in ("prefill", "decode"), "invalid call phase")
    positions, tokens, context = (call.get(key) for key in ("positions", "token_ids", "context_token_ids"))
    _require(_integers(positions) and _integers(tokens) and len(positions) == len(tokens), "invalid call rows")
    _require(type(call.get("context_complete")) is bool, "missing context completeness")
    _require(_tokens(context, incomplete=not call["context_complete"]), "invalid causal token context")
    _require(isinstance(call.get("expected"), list), "missing expected tensor inventory")
    expected = []
    for item in call["expected"]:
        _require(isinstance(item, dict), "invalid expected tensor")
        _require(_integer(item.get("layer"), -1), "invalid expected layer")
        _require(
            all(isinstance(item.get(key), str) and item[key] for key in ("kind", "name")), "invalid expected identity"
        )
        expected.append((item["layer"], item["kind"], item["name"]))
    _require(len(set(expected)) == len(expected), "duplicate expected tensor")
    if call["context_complete"]:
        _require(
            all(p < len(context) and context[p] == token for p, token in zip(positions, tokens)),
            "call tokens disagree with causal context",
        )


def _check_row_capture(capture, context_length, saved_rows=None):
    _require(isinstance(capture, dict), "invalid row capture metadata")
    source, saved, limit = (capture.get(key) for key in ("source_rows", "saved_rows", "limit"))
    _require(all(_integer(value) for value in (source, saved, limit)), "invalid row capture counts")
    _require(capture.get("chunk_size") == 4096 and limit <= 4096, "invalid capture segment size/limit")
    _require(capture.get("strategy") in ("absolute_chunk_prefix", "visible_chunk_prefix"), "invalid capture strategy")
    indices = capture.get("source_indices")
    _require(
        _integers(indices)
        and len(indices) == saved
        and indices == sorted(set(indices))
        and all(index < source for index in indices),
        "invalid captured source row indices",
    )
    if saved_rows is not None:
        _require(saved == saved_rows, "row capture count differs from stored shape")
    segments = capture.get("segments")
    _require(isinstance(segments, list), "missing capture segment metadata")
    total_source = total_saved = 0
    previous = -1
    for segment in segments:
        _require(isinstance(segment, dict), "invalid capture segment")
        base, length = segment.get("start"), segment.get("length")
        _require(
            _integer(base)
            and base % 4096 == 0
            and base > previous
            and _integer(length, 1)
            and length <= min(4096, context_length - base),
            "invalid capture segment range",
        )
        previous = base
        count, kept = segment.get("source_rows"), segment.get("saved_rows")
        _require(_integer(count, 1) and _integer(kept) and kept <= count, "invalid capture segment counts")
        _require(kept <= limit if limit else kept == count, "capture exceeds segment limit")
        if capture["strategy"] == "visible_chunk_prefix":
            _require(kept == min(count, limit) if limit else kept == count, "missing visible segment rows")
        bounds = segment.get("omitted_position_bounds")
        if count == kept:
            _require(bounds is None, "complete segment claims omitted rows")
        else:
            _require(
                _integers(bounds) and len(bounds) == 2 and base <= bounds[0] < bounds[1] <= base + length,
                "invalid omitted logical position bounds",
            )
        total_source += count
        total_saved += kept
    _require((source, saved) == (total_source, total_saved), "capture totals differ from segment counts")


def _check_record(record: dict, worker: Worker) -> None:
    _require(record.get("schema") == 1, "invalid tensor record schema")
    _require(record.get("request_id") == worker.manifest["request_id"], "record request ID differs from manifest")
    _require(record.get("model") in ("main", "mtp"), "unsupported model scope")
    _require(_integer(record.get("call")) and record["call"] in worker.calls, "record has no valid call")
    _require(record["model"] == worker.calls[record["call"]].get("model", "main"), "tensor/call model differs")
    _require(_integer(record.get("layer"), -1), "invalid layer")
    _require(
        all(isinstance(record.get(key), str) and record[key] for key in ("kind", "name", "dtype")),
        "missing tensor identity/dtype",
    )
    _require(_integers(record.get("shape")), "invalid tensor shape")
    if "feature_capture" in record:
        capture = record["feature_capture"]
        shape = record["shape"]
        _require(
            isinstance(capture, dict)
            and len(shape) >= 2
            and capture.get("axis") == len(shape) - 1
            and capture.get("start") == 0
            and _integer(capture.get("source_size"), 1)
            and _integer(capture.get("saved_size"), 1)
            and capture["source_size"] > capture["saved_size"] == shape[-1],
            "invalid feature capture metadata",
        )
        _require(
            record["kind"] not in ("mapping", "rejection", "logits", "draft")
            and record["name"] not in ("topk", "logical_topk", "input_ids", "positions"),
            "routing/sampling evidence cannot truncate its feature axis",
        )
    _local_path(worker.directory, record.get("path"))
    layout = record.get("tensor_layout")
    _require(layout in ("replicated", "sequence_sharded", "rank_local", "mapping"), "invalid tensor layout")
    _require(
        type(record.get("mapping_only")) is bool and record["mapping_only"] == (layout == "mapping"),
        "invalid mapping classification",
    )
    axis = record.get("row_axis")
    _require(axis is None or type(axis) is int and axis == 0, "unsupported row axis")
    if "row_capture" in record:
        _require(axis == 0 and bool(record["shape"]), "row capture requires a token axis")
        capture = record["row_capture"]
        context_length = len(worker.calls[record["call"]]["context_token_ids"])
        _check_row_capture(capture, context_length, record["shape"][0])
        if "query_rows" in capture:
            _require(record["kind"] == "kv_consumed", "query sampling only describes consumed KV")
            _check_row_capture(capture["query_rows"], context_length)
        if "logical_chunk_filter" in capture:
            _require(
                record["kind"] == "kv_consumed" and type(capture["logical_chunk_filter"]) is bool,
                "invalid consumed KV sampling filter",
            )
    if "physical_slots" in record:
        slots = record["physical_slots"]
        _require(
            record["kind"] == "kv_consumed"
            and axis == 0
            and _integers(slots)
            and bool(record["shape"])
            and len(slots) == record["shape"][0]
            and isinstance(record.get("positions"), list)
            and len(slots) == len(record["positions"]),
            "invalid consumed KV physical slots",
        )
    if axis is None:
        _require(
            record.get("positions") is None and record.get("token_ids") is None,
            "nonrow tensor cannot claim logical rows",
        )
        return
    positions, tokens = record.get("positions"), record.get("token_ids")
    call = worker.calls[record["call"]]
    _require(
        _integers(positions)
        and _tokens(tokens, incomplete=not call["context_complete"])
        and len(positions) == len(tokens),
        "invalid tensor logical rows",
    )
    _require(bool(record["shape"]) and len(positions) <= record["shape"][0], "logical rows exceed tensor shape")
    _require(all(size > 0 for size in record["shape"][1:]), "empty tensor feature dimension")
    if call["context_complete"]:
        context = call["context_token_ids"]
        _require(
            all(p < len(context) and context[p] == token for p, token in zip(positions, tokens)),
            "tensor tokens disagree with causal context",
        )


def load_archive(roots: list[str | Path]) -> Archive:
    """Read collected worker directories; retain incomplete/error evidence."""
    archive = Archive()
    paths = set()
    for value in roots:
        root = Path(value)
        if not root.exists():
            archive.issues.append({"status": "missing_archive", "path": str(root)})
            continue
        found = [root] if root.is_file() else root.rglob("manifest.json")
        paths.update(path.resolve() for path in found if not any(part.endswith(".partial") for part in path.parts))
    grouped: dict[tuple, list[Worker]] = defaultdict(list)
    for path in sorted(paths):
        try:
            manifest = _read(path)
            if manifest.get("tool") != "pd_tensor_dump":
                continue
            _require(manifest.get("schema_version") == 1, "unsupported manifest schema")
            _require(manifest.get("role") in ("P", "D"), "invalid P/D role")
            _require(isinstance(manifest.get("request_id"), str) and bool(manifest["request_id"]), "missing request ID")
            _require(_integer(manifest.get("tp_size"), 1) and _integer(manifest.get("tp_rank")), "invalid TP topology")
            _require(manifest["tp_rank"] < manifest["tp_size"], "TP rank out of range")
            _require(_integer(manifest.get("dp_size"), 1) and _integer(manifest.get("dp_rank")), "invalid DP topology")
            _require(manifest["dp_rank"] < manifest["dp_size"], "DP rank out of range")
            _require(_integer(manifest.get("num_layers"), 1), "invalid model layer count")
            _require(isinstance(manifest.get("model_id"), str) and bool(manifest["model_id"]), "missing model ID")
            _require(type(manifest.get("layerwise_prefill")) is bool, "missing layerwise state")
            if "prompt_token_ids" in manifest:
                _require(_integers(manifest["prompt_token_ids"]), "invalid request prompt tokens")
            _require(isinstance(manifest.get("errors"), list), "missing recorder error list")
            _require(
                _integer(manifest.get("records")) and _integers(manifest.get("calls")), "invalid manifest inventory"
            )
            _require(len(set(manifest["calls"])) == len(manifest["calls"]), "duplicate manifest call")
            worker = Worker(path.parent, manifest)
            grouped[worker.key].append(worker)
            if (
                manifest.get("complete") is not True
                or manifest.get("request_finished") is not True
                or manifest["errors"]
            ):
                archive.issues.append(
                    {**worker.evidence(), "status": "incomplete_worker", "recorder_errors": manifest["errors"]}
                )
            for number in manifest["calls"]:
                try:
                    call = _read(path.parent / "calls" / f"{number}.json")
                    _check_call(call, number)
                    worker.calls[number] = call
                    if call.get("complete") is not True:
                        archive.issues.append({**worker.evidence(), "status": "incomplete_call", "call": number})
                except Exception as exc:
                    archive.issues.append(
                        {**worker.evidence(), "status": "invalid_call", "call": number, "reason": str(exc)}
                    )
            lines = (path.parent / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if len(lines) != manifest["records"]:
                archive.issues.append(
                    {
                        **worker.evidence(),
                        "status": "record_count_mismatch",
                        "declared": manifest["records"],
                        "observed": len(lines),
                    }
                )
            seen = set()
            for line_number, line in enumerate(lines, 1):
                try:
                    record = json.loads(line)
                    _check_record(record, worker)
                    identity = (record["call"], *_identity(record))
                    _require(identity not in seen, "duplicate tensor identity in call")
                    seen.add(identity)
                    worker.records.append(record)
                except Exception as exc:
                    archive.issues.append(
                        {**worker.evidence(), "status": "invalid_record", "line": line_number, "reason": str(exc)}
                    )
            for number, call in worker.calls.items():
                expected = {(item["layer"], item["kind"], item["name"]) for item in call["expected"]}
                observed = {
                    (item["layer"], item["kind"], item["name"]) for item in worker.records if item["call"] == number
                }
                for status, entries in (
                    ("missing_expected_tensor", expected - observed),
                    ("additional_tensor", observed - expected),
                ):
                    for layer, kind, name in sorted(entries):
                        archive.issues.append(
                            {
                                **worker.evidence(),
                                "status": status,
                                "call": number,
                                "layer": layer,
                                "kind": kind,
                                "name": name,
                            }
                        )
            _load_sampled(worker, archive)
            if manifest.get("mtp_layers") and not any(c.get("model") == "mtp" for c in worker.calls.values()):
                archive.issues.append({**worker.evidence(), "status": "missing_mtp_calls"})
        except Exception as exc:
            archive.issues.append({"status": "invalid_manifest", "path": str(path), "reason": str(exc)})
    for key, workers in grouped.items():
        if len(workers) != 1:
            archive.issues.append(
                {
                    "status": "ambiguous_worker",
                    "role": key[0],
                    "request_id": key[1],
                    "tp_rank": key[2],
                    "paths": [str(item.directory) for item in workers],
                }
            )
        else:
            archive.workers[key] = workers[0]
    ranks: dict[tuple, list[Worker]] = defaultdict(list)
    for worker in archive.workers.values():
        ranks[worker.key[:2]].append(worker)
    for key, workers in ranks.items():
        sizes = {item.manifest["tp_size"] for item in workers}
        if len(sizes) != 1:
            archive.issues.append({"status": "inconsistent_tp_size", "role": key[0], "request_id": key[1]})
            continue
        missing = sorted(set(range(next(iter(sizes)))) - {item.manifest["tp_rank"] for item in workers})
        if missing:
            archive.issues.append(
                {"status": "missing_tp_ranks", "role": key[0], "request_id": key[1], "ranks": missing}
            )
    if not grouped:
        archive.issues.append({"status": "no_worker_manifests"})
    return archive


def _load_tensor(worker: Worker, record: dict):
    torch = importlib.import_module("torch")
    value = torch.load(_local_path(worker.directory, record["path"]), map_location="cpu", weights_only=True)
    _require(isinstance(value, torch.Tensor), "archive payload is not a tensor")
    _require(
        list(value.shape) == record["shape"] and str(value.dtype) == record["dtype"],
        "raw tensor shape/dtype differs from index",
    )
    return value


def _record_evidence(worker: Worker, record: dict) -> dict:
    return {
        **worker.evidence(),
        **{key: record[key] for key in ("model", "call", "layer", "kind", "name", "path", "tensor_layout")},
        **({"feature_capture": record["feature_capture"]} if "feature_capture" in record else {}),
    }


def _align_features(left, right, left_record, right_record):
    """Compare shared feature coordinates; old full-width archives remain usable."""
    left_capture = left_record.get("feature_capture")
    right_capture = right_record.get("feature_capture")
    if left.ndim < 2 or right.ndim < 2:
        return left, right
    _require(left.ndim == right.ndim and left.dtype == right.dtype, "feature tensor rank/dtype differs")
    left_size = left_capture["source_size"] if left_capture else left_record["shape"][-1]
    right_size = right_capture["source_size"] if right_capture else right_record["shape"][-1]
    _require(left_size == right_size, "original feature sizes differ")
    width = min(left.shape[-1], right.shape[-1])
    return left[..., :width], right[..., :width]


def _compare_kv_aliases(worker: Worker, side: str, emit) -> None:
    """Compare every physical copy with the first copy in the SAME call.

    This is independent of P/D or OFF/ON reference selection. In particular,
    a conflicting OFF alias must not be hidden by selecting its first row.
    """
    torch = importlib.import_module("torch")
    for record in worker.records:
        if record["kind"] != "kv_consumed" or record["row_axis"] != 0:
            continue
        first, baseline_rows, alias_rows = {}, [], []
        for row, position in enumerate(record["positions"]):
            if position in first:
                baseline_rows.append(first[position])
                alias_rows.append(row)
            else:
                first[position] = row
        if not alias_rows:
            continue
        detail = {**_record_evidence(worker, record), "archive": side}
        try:
            value = _load_tensor(worker, record)
            baseline, actual = value[baseline_rows], value[alias_rows]
            comparison = compare_tensor_values(baseline, actual)
            _require(comparison.get("comparable", False), "KV aliases cannot be compared")
            unequal = baseline != actual
            if baseline.is_floating_point():
                unequal &= ~(torch.isnan(baseline) & torch.isnan(actual))
            different = unequal.reshape(len(alias_rows), -1).any(dim=1).tolist()
            positions = [record["positions"][row] for row in alias_rows]
            detail.update(
                status="kv_aliases_different" if comparison["mismatched"] else "kv_aliases_equal",
                positions=positions,
                baseline_rows=baseline_rows,
                alias_rows=alias_rows,
                alias_comparison=comparison,
                different_positions=sorted({p for p, d in zip(positions, different) if d}),
                different_alias_rows=[row for row, d in zip(alias_rows, different) if d],
            )
            slots = record.get("physical_slots")
            if slots is not None:
                detail.update(
                    baseline_physical_slots=[slots[row] for row in baseline_rows],
                    alias_physical_slots=[slots[row] for row in alias_rows],
                )
            emit(detail)
        except Exception as exc:
            emit({**detail, "status": "invalid_tensor", "reason": str(exc)})


def _compatible_context(left: dict, right: dict, position: int | None) -> bool:
    if not left["context_complete"] or not right["context_complete"]:
        return False
    if position is None:
        return all(left[key] == right[key] for key in ("positions", "token_ids", "context_token_ids"))
    size = position + 1
    return (
        min(len(left["context_token_ids"]), len(right["context_token_ids"])) >= size
        and left["context_token_ids"][:size] == right["context_token_ids"][:size]
    )


def _compare_sampled(reference: Worker, candidate: Worker, emit) -> None:
    detail = {**candidate.evidence(), "reference_request_id": reference.manifest["request_id"]}
    if not reference.sampled_complete or not candidate.sampled_complete:
        emit({**detail, "status": "incomplete_sampled_outputs"})
        return
    left_prompt, right_prompt = (worker.manifest.get("prompt_token_ids") for worker in (reference, candidate))
    if left_prompt is None or right_prompt is None or left_prompt != right_prompt:
        emit(
            {
                **detail,
                "status": "incomparable_sampled_context",
                "reason": "recorded complete request prompts are missing or differ",
            }
        )
        return
    for key in ("model_id", "num_layers", "tp_size"):
        if reference.manifest[key] != candidate.manifest[key]:
            emit({**detail, "status": "incomparable_sampled_context", "reason": f"{key} differs"})
            return
    left = [token for item in reference.sampled for token in item["token_ids"]]
    right = [token for item in candidate.sampled for token in item["token_ids"]]
    first = next((i for i, (lhs, rhs) in enumerate(zip(left, right)) if lhs != rhs), None)
    if first is None and len(left) != len(right):
        first = min(len(left), len(right))
    emit(
        {
            **detail,
            "status": "sampled_outputs_equal" if first is None else "sampled_outputs_different",
            "reference_token_ids": left,
            "candidate_token_ids": right,
            "reference_tokens": len(left),
            "candidate_tokens": len(right),
            "first_unequal_output_index": first,
            "first_unequal_context_position": None if first is None else len(left_prompt) + first,
            "reference_first_unequal_token": None if first is None or first >= len(left) else left[first],
            "candidate_first_unequal_token": None if first is None or first >= len(right) else right[first],
            "comparison_scope": (
                "Accepted token order under the same prompt and preceding accepted-token prefix; "
                "call batching may differ."
            ),
        }
    )


def _rejection_decisions(worker: Worker):
    """Interpret archived sampler results offline, never infer from HTTP tokens."""
    rows = {(record["call"], record["name"]): record for record in worker.records if record["kind"] == "rejection"}
    sampled = {item["after_call"]: item["token_ids"] for item in worker.sampled}
    for number, call in worker.calls.items():
        metadata = call.get("rejection")
        if metadata is None:
            continue
        detail = {**worker.evidence(), "call": number, "phase": call["phase"]}
        try:

            def tensor(name, number=number):
                return _load_tensor(worker, rows[number, name])

            drafts = tensor("draft_token_ids").reshape(-1).tolist()
            target = tensor("target_logits")
            bonus = tensor("bonus_token_ids").reshape(-1).tolist()
            raw_output = tensor("sampler_output").reshape(-1).tolist()
            kernel_output = tensor("kernel_output").reshape(-1).tolist()
            count = metadata["num_draft_tokens"]
            _require(len(drafts) == count == target.shape[0], "rejection draft/target count differs")
            _require(len(bonus) == 1, "rejection bonus row differs")
            placeholder = metadata["placeholder_token_id"]
            output = [t for t in raw_output if t != placeholder]
            _require(1 <= len(output) <= count + 1, "invalid rejection output length")
            _require(
                raw_output == output + [placeholder] * (len(raw_output) - len(output))
                and all(0 <= token < target.shape[-1] for token in output),
                "invalid rejection output tokens/padding",
            )
            accepted = len(output) - 1
            argmax = target.argmax(dim=-1).tolist()
            expected = None
            if metadata["mode"] == "greedy":
                first = next((i for i, (d, t) in enumerate(zip(drafts, argmax)) if d != t), count)
                expected = drafts[:first] + ([argmax[first]] if first < count else bonus)
            valid_used = bool(sampled.get(number))
            yield {
                **detail,
                "status": "recorded",
                "mode": metadata["mode"],
                "draft_positions": metadata["draft_positions"],
                "draft_token_ids": drafts,
                "target_argmax_ids": argmax,
                "bonus_token_id": bonus[0],
                "sampler_output": raw_output,
                "accepted_drafts": accepted,
                "proposed_drafts": count,
                "first_rejected_index": accepted if accepted < count else None,
                "draft_status": ["accepted"] * accepted
                + (["rejected"] + ["not_reached"] * (count - accepted - 1) if accepted < count else []),
                "terminal_token_kind": "recovered" if accepted < count else "bonus",
                "terminal_token_id": output[-1],
                "greedy_expected_output": expected,
                "greedy_decision_consistent": None if expected is None else output == expected,
                "kernel_output_matches_sampler": kernel_output == raw_output,
                "counted_for_acceptance": valid_used and count > 0,
                "random_evidence": [
                    name
                    for name in ("draft_probs", "target_probs", "uniform_probs", "recovered_token_ids")
                    if (number, name) in rows
                ],
            }
        except Exception as exc:
            yield {**detail, "status": "incomplete", "reason": str(exc)}


def _write_rejection_report(reference: Archive, candidate: Archive, directory: Path) -> dict:
    summaries, seen = [], set()
    with (directory / "rejection.jsonl").open("w", encoding="utf-8") as stream:
        for side, archive in (("reference", reference), ("candidate", candidate)):
            for worker in archive.workers.values():
                if worker.directory in seen:
                    continue
                seen.add(worker.directory)
                decisions = list(_rejection_decisions(worker))
                if not decisions:
                    continue
                for item in decisions:
                    stream.write(json.dumps({"archive": side, **item}, ensure_ascii=False, allow_nan=False) + "\n")
                counted = [d for d in decisions if d.get("counted_for_acceptance")]
                proposed = sum(d["proposed_drafts"] for d in counted)
                accepted = sum(d["accepted_drafts"] for d in counted)
                summaries.append(
                    {
                        "archive": side,
                        **worker.evidence(),
                        "calls": len(decisions),
                        "incomplete_calls": sum(d["status"] != "recorded" for d in decisions),
                        "proposed_drafts": proposed,
                        "accepted_drafts": accepted,
                        "acceptance_rate": accepted / proposed if proposed else None,
                        "greedy_inconsistent_calls": sum(
                            d.get("greedy_decision_consistent") is False for d in decisions
                        ),
                        "kernel_output_mismatch_calls": sum(
                            d.get("kernel_output_matches_sampler") is False for d in decisions
                        ),
                    }
                )
    return {
        "scope": (
            "Per worker/request; TP replicas are not summed. Discarded prefill samples are excluded; "
            "counts are before HTTP stop/length trimming."
        ),
        "workers": summaries,
        "details": "rejection.jsonl",
    }


def _observation_order(marker: dict) -> tuple:
    """Order observations in the main model, not by sentinel layer -1."""
    kind, name, layer = (marker[key] for key in ("kind", "name", "layer"))
    if kind == "model_input":
        computation = (0, 0, 0)
    elif kind == "model_output":
        computation = (2, 0, 0)
    elif kind == "logits":
        computation = (3, 0, 0)
    elif layer >= 0:
        if kind == "decoder":
            stage = 0 if name.startswith("input") else 90
        elif kind == "sfa":
            stage = 10 if name.startswith("input") else 80
        elif kind == "attention":
            stage = 20 if name.startswith("query") else 70 if name == "output" else 50
        else:
            stage = {"indexer_input": 30, "indexer": 40, "kv_current": 45, "kv_indexer": 46, "kv_consumed": 60}.get(
                kind, 65
            )
        computation = (1, layer, stage)
    else:
        computation = (4, 0, 0)
    position = marker["position"]
    return (
        position if position >= 0 else float("inf"),
        *computation,
        marker["request_id"],
        marker["tp_rank"],
        kind,
        name,
    )


def _outside_sample(worker, refs, call, position):
    """A matching capture may have omitted this row; never claim it was checked."""
    for ref in refs:
        if not _compatible_context(worker.calls[ref["call"]], call, position):
            continue
        capture = ref.get("row_capture", {})
        for segment in capture.get("segments", []):
            bounds = segment.get("omitted_position_bounds")
            if bounds and bounds[0] <= position < bounds[1]:
                return True
        if capture.get("logical_chunk_filter") and position % capture["chunk_size"] >= capture["limit"]:
            return True
        # Unobserved attention queries can select any preceding KV position.
        # We did not inspect those indices, so cannot infer consumed coverage.
        for segment in capture.get("query_rows", {}).get("segments", []):
            query_bounds = segment.get("omitted_position_bounds")
            if query_bounds and 0 <= position < query_bounds[1]:
                return True
    return False


def _capture_summary(archive):
    result = dict(
        sampled_tensors=0, feature_sampled_tensors=0, saved_token_rows=0, omitted_token_rows=0, query_limited_tensors=0
    )
    for worker in archive.workers.values():
        for record in worker.records:
            capture = record.get("row_capture", {})
            omitted = capture.get("source_rows", 0) - capture.get("saved_rows", 0)
            query_limited = any(
                s.get("omitted_position_bounds") for s in capture.get("query_rows", {}).get("segments", [])
            )
            result["sampled_tensors"] += int(
                bool(omitted) or query_limited or capture.get("logical_chunk_filter", False)
            )
            result["query_limited_tensors"] += int(query_limited)
            result["feature_sampled_tensors"] += int("feature_capture" in record)
            result["saved_token_rows"] += len(record.get("positions") or [])
            result["omitted_token_rows"] += omitted
    result["scope"] = (
        "Only archived rows are checked; omitted_token_rows excludes KV filtered by logical chunk offset "
        "or reachable only from unsampled queries. "
        "Feature statistics cover only recorded/common feature components, not full vectors. "
        "Counts sum observations across calls/layers, not unique request tokens."
    )
    return result


def _compare_worker(reference: Worker, candidate: Worker, mode: str, emit) -> None:
    torch = importlib.import_module("torch")
    for key in ("model_id", "num_layers", "tp_size"):
        if reference.manifest[key] != candidate.manifest[key]:
            emit(
                {
                    **candidate.evidence(),
                    "status": "incompatible_model_metadata",
                    "field": key,
                    "reference": reference.manifest[key],
                    "candidate": candidate.manifest[key],
                }
            )
            return
    groups: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    used: set[tuple[int, int | None]] = set()
    source_records = []
    for index, record in enumerate(reference.records):
        if mode == "pd-kv" and record["kind"] not in PD_KINDS.values():
            continue
        groups[_identity(record)].append((index, record))
        source_records.append((index, record))
    prompt_end = max(
        (max(call["positions"], default=-1) + 1 for call in reference.calls.values() if call["phase"] == "prefill"),
        default=0,
    )
    if mode == "pd-kv" and not prompt_end:
        emit({**reference.evidence(), "status": "missing_prefill_context"})
        return
    for record in candidate.records:
        detail = _record_evidence(candidate, record)
        if mode == "pd-kv" and record["kind"] not in PD_KINDS:
            continue
        if record["mapping_only"]:
            emit({**detail, "status": "mapping_evidence"})
            continue
        source_kind = PD_KINDS[record["kind"]] if mode == "pd-kv" else record["kind"]
        refs = groups.get((record["model"], record["layer"], source_kind, record["name"]), [])
        call = candidate.calls[record["call"]]
        try:
            actual = _load_tensor(candidate, record)
            pairs: dict[int, list[tuple[int, int]]] = defaultdict(list)
            references: dict[int, dict] = {}
            choices_count = 0
            if record["row_axis"] == 0:
                positions: dict[int, list[tuple[int, dict, int]]] = defaultdict(list)
                for index, ref in refs:
                    if ref["row_axis"] == 0 and not ref["mapping_only"]:
                        for row, position in enumerate(ref["positions"]):
                            positions[position].append((index, ref, row))
                unmatched: dict[str, list[int]] = defaultdict(list)
                for row, (position, token) in enumerate(zip(record["positions"], record["token_ids"])):
                    # MTP position p depends on target token p+1. The last
                    # prompt-position MTP KV already depends on generated output.
                    boundary = prompt_end - int(record["model"] == "mtp")
                    if mode == "pd-kv" and position >= boundary:
                        unmatched["not_prompt_kv"].append(position)
                        continue
                    options = positions.get(position, [])
                    choices = [
                        (index, ref, ref_row)
                        for index, ref, ref_row in options
                        if ref["token_ids"][ref_row] == token
                        and _compatible_context(reference.calls[ref["call"]], call, position)
                    ]
                    if not choices:
                        status = "incomparable_context" if options else "missing_reference_tensor"
                        if _outside_sample(reference, [ref for _, ref in refs], call, position):
                            status = "outside_reference_sample"
                        unmatched[status].append(position)
                        continue
                    choices.sort(key=lambda item: (item[1]["call"], item[0], item[2]))
                    index, ref, ref_row = choices[0]
                    choices_count += len(choices) - 1
                    pairs[index].append((row, ref_row))
                    references[index] = ref
                    # Repeated historical snapshots are equivalent coverage
                    # candidates, but actual values use the earliest observation.
                    used.update((i, j) for i, _, j in choices)
                for status, selected in unmatched.items():
                    emit({**detail, "status": status, "positions": selected})
                if not record["positions"]:
                    emit({**detail, "status": "empty_valid_rows"})
                if not pairs:
                    continue
                parts, ordering, sources = [], [], []
                for index, aligned in pairs.items():
                    ref = references[index]
                    value = _load_tensor(reference, ref)
                    value, actual = _align_features(value, actual, ref, record)
                    _require(
                        value.dtype == actual.dtype and value.shape[1:] == actual.shape[1:],
                        "aligned tensor trailing shape/dtype differs",
                    )
                    parts.append(value[[source_row for _, source_row in aligned]])
                    ordering.extend(row for row, _ in aligned)
                    sources.append(
                        {
                            "call": ref["call"],
                            "path": ref["path"],
                            "kind": ref["kind"],
                            "tensor_layout": ref["tensor_layout"],
                            "rows": len(aligned),
                            **({"feature_capture": ref["feature_capture"]} if "feature_capture" in ref else {}),
                        }
                    )
                # Different reference snapshots may retain different feature
                # widths; align earlier parts after the narrowest is known.
                if any("feature_capture" in ref for ref in references.values()) or "feature_capture" in record:
                    parts = [part[..., : actual.shape[-1]] for part in parts]
                    detail["compared_feature_size"] = int(actual.shape[-1])
                baseline = torch.cat(parts, dim=0)
                actual = actual[ordering]
                detail.update(
                    positions=[record["positions"][row] for row in ordering],
                    reference_sources=sources,
                    repeated_reference_choices=choices_count,
                )
                if "physical_slots" in record:
                    detail["physical_slots"] = [record["physical_slots"][row] for row in ordering]
            else:
                if mode == "pd-kv":
                    emit({**detail, "status": "incomparable_nonrow_kv"})
                    continue
                choices = [
                    (index, ref)
                    for index, ref in refs
                    if ref["row_axis"] is None and _compatible_context(reference.calls[ref["call"]], call, None)
                ]
                if not choices:
                    emit({**detail, "status": "incomparable_context" if refs else "missing_reference_tensor"})
                    continue
                choices.sort(key=lambda item: (item[1]["call"], item[0]))
                index, ref = choices[0]
                _require(ref["tensor_layout"] == record["tensor_layout"], "tensor layout differs")
                baseline = _load_tensor(reference, ref)
                baseline, actual = _align_features(baseline, actual, ref, record)
                used.update((i, None) for i, _ in choices)
                detail.update(
                    reference_sources=[{"call": ref["call"], "path": ref["path"]}],
                    repeated_reference_choices=len(choices) - 1,
                )
            comparison = compare_tensor_values(baseline, actual)
            _require(comparison.get("comparable", False), comparison.get("reason", "tensor values cannot be compared"))
            if comparison["mismatched"] and record["row_axis"] == 0:
                unequal = baseline != actual
                if baseline.is_floating_point():
                    unequal &= ~(torch.isnan(baseline) & torch.isnan(actual))
                differing = unequal.reshape(len(detail["positions"]), -1).any(dim=1).tolist()
                detail["first_different_position"] = min(
                    position for position, different in zip(detail["positions"], differing) if different
                )
            emit({**detail, "status": "different" if comparison["mismatched"] else "equal", "comparison": comparison})
        except Exception as exc:
            emit({**detail, "status": "invalid_tensor", "reason": str(exc)})
    for index, record in source_records:
        if record["mapping_only"]:
            continue
        if record["row_axis"] == 0:
            missing = [position for row, position in enumerate(record["positions"]) if (index, row) not in used]
            if missing:
                statuses = defaultdict(list)
                refs = [r for r in candidate.records if _identity(r) == _identity(record)]
                for position in missing:
                    status = "not_consumed" if mode == "pd-kv" else "missing_candidate_tensor"
                    if mode != "pd-kv" and _outside_sample(candidate, refs, reference.calls[record["call"]], position):
                        status = "outside_candidate_sample"
                    statuses[status].append(position)
                for status, positions in statuses.items():
                    emit({**_record_evidence(reference, record), "status": status, "positions": positions})
        elif (index, None) not in used:
            emit(
                {
                    **_record_evidence(reference, record),
                    "status": "not_consumed" if mode == "pd-kv" else "missing_candidate_tensor",
                }
            )


def analyze(
    reference_roots: list[str | Path],
    candidate_roots: list[str | Path],
    *,
    mode: str,
    output: str | Path,
    request_map: dict[str, str] | None = None,
) -> dict:
    """Write report.json + comparisons.jsonl; return diagnostic summary."""
    _require(mode in ("off-on", "pd-kv"), "unsupported comparison mode")
    request_map = {} if request_map is None else request_map
    _require(
        isinstance(request_map, dict)
        and all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in request_map.items()),
        "request map must map nonempty old IDs to new IDs",
    )
    _require(len(set(request_map.values())) == len(request_map), "request map must be one-to-one")
    _require(mode != "pd-kv" or not request_map, "PD KV comparison requires identical request IDs")
    reference, candidate = load_archive(reference_roots), load_archive(candidate_roots)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    model_counts: dict[str, Counter] = defaultdict(Counter)
    first_by_model: dict[str, dict] = {}
    compared_rows = new_nonfinite = 0
    first_difference = worst_difference = first_output_difference = None
    first_alias_conflict = None
    inverse = {value: key for key, value in request_map.items()}
    matched_workers = set()
    with (directory / "comparisons.jsonl").open("w", encoding="utf-8") as stream:

        def emit(item: dict) -> None:
            nonlocal compared_rows, new_nonfinite, first_difference, worst_difference, first_output_difference
            nonlocal first_alias_conflict
            counts[item["status"]] += 1
            if item.get("model") in ("main", "mtp"):
                model_counts[item["model"]][item["status"]] += 1
            if item["status"] == "kv_aliases_different" and first_alias_conflict is None:
                index = item["alias_rows"].index(item["different_alias_rows"][0])
                first_alias_conflict = {
                    key: item[key]
                    for key in (
                        "archive",
                        "request_id",
                        "role",
                        "tp_rank",
                        "call",
                        "layer",
                        "name",
                        "path",
                        "worker_directory",
                    )
                }
                first_alias_conflict.update(
                    position=item["positions"][index],
                    baseline_row=item["baseline_rows"][index],
                    alias_row=item["alias_rows"][index],
                    abs_max=item["alias_comparison"]["abs_diff"]["max"],
                    relative_l2=item["alias_comparison"]["relative_l2"],
                )
                if "baseline_physical_slots" in item:
                    first_alias_conflict.update(
                        baseline_physical_slot=item["baseline_physical_slots"][index],
                        alias_physical_slot=item["alias_physical_slots"][index],
                    )
            if item["status"] == "sampled_outputs_different":
                marker = {
                    key: item[key]
                    for key in (
                        "request_id",
                        "role",
                        "tp_rank",
                        "first_unequal_output_index",
                        "first_unequal_context_position",
                        "reference_first_unequal_token",
                        "candidate_first_unequal_token",
                    )
                }
                if (
                    first_output_difference is None
                    or marker["first_unequal_output_index"] < first_output_difference["first_unequal_output_index"]
                ):
                    first_output_difference = marker
            values = item.get("comparison")
            if values:
                compared_rows += len(item.get("positions", []))
                new_nonfinite += values["new_nonfinite"]
                if values["mismatched"]:
                    marker = {
                        key: item[key]
                        for key in ("request_id", "role", "tp_rank", "model", "call", "layer", "kind", "name")
                    }
                    marker.update(
                        position=item.get("first_different_position", -1),
                        abs_max=values["abs_diff"]["max"],
                        relative_l2=values["relative_l2"],
                    )
                    if first_difference is None or _observation_order(marker) < _observation_order(first_difference):
                        first_difference = marker
                    previous = first_by_model.get(marker["model"])
                    if previous is None or _observation_order(marker) < _observation_order(previous):
                        first_by_model[marker["model"]] = marker
                    if worst_difference is None or (marker["abs_max"] or 0) > (worst_difference["abs_max"] or 0):
                        worst_difference = marker
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")

        checked_alias_workers = set()
        for side, archive in (("reference", reference), ("candidate", candidate)):
            for issue in archive.issues:
                emit({"archive": side, **issue})
            for worker in archive.workers.values():
                if worker.directory not in checked_alias_workers:
                    _compare_kv_aliases(worker, side, emit)
                    checked_alias_workers.add(worker.directory)
        for key, worker in candidate.workers.items():
            role, request_id, rank = key
            if mode == "pd-kv" and role != "D":
                continue
            source_key = ("P" if mode == "pd-kv" else role, inverse.get(request_id, request_id), rank)
            source = reference.workers.get(source_key)
            if source is None:
                emit({**worker.evidence(), "status": "missing_reference_worker", "reference_request_id": source_key[1]})
                continue
            matched_workers.add(source_key)
            if mode == "off-on":
                _compare_sampled(source, worker, emit)
            _compare_worker(source, worker, mode, emit)
        for key, worker in reference.workers.items():
            if key in matched_workers or mode == "pd-kv" and key[0] != "P":
                continue
            emit({**worker.evidence(), "status": "missing_candidate_worker"})
        if not counts["equal"] and not counts["different"]:
            emit({"status": "no_comparable_tensors"})
    benign = {
        "equal",
        "different",
        "mapping_evidence",
        "empty_valid_rows",
        "not_consumed",
        "not_prompt_kv",
        "additional_tensor",
        "sampled_outputs_equal",
        "sampled_outputs_different",
        "kv_aliases_equal",
        "kv_aliases_different",
        "outside_reference_sample",
        "outside_candidate_sample",
    }
    issues = sum(value for key, value in counts.items() if key not in benign)
    comparisons = counts["equal"] + counts["different"]
    report = {
        "schema_version": 1,
        "mode": mode,
        "status": "incomplete_or_incomparable" if issues or not comparisons else "analysis_complete",
        "accuracy_verdict": "not_assessed",
        "scope": (
            "Archived main and MTP tensor rows/features only; unsaved values are not assessed. "
            "PD mode compares consumed prompt KV, excluding the MTP next-token boundary."
        ),
        "capture_sampling": {"reference": _capture_summary(reference), "candidate": _capture_summary(candidate)},
        "models_observed": {
            side: sorted({r["model"] for w in archive.workers.values() for r in w.records})
            for side, archive in (("reference", reference), ("candidate", candidate))
        },
        "reference_selection": "Earliest observation with the same TP rank, logical token and complete causal context.",
        "numeric_policy": (
            "All CPU elements are compared; differences and distributions are reported "
            "without tolerance-based pass/fail."
        ),
        "reference_workers": len(reference.workers),
        "candidate_workers": len(candidate.workers),
        "compared_tensors": comparisons,
        "compared_rows": compared_rows,
        "new_nonfinite": new_nonfinite,
        "issues": issues,
        "counts": dict(sorted(counts.items())),
        "counts_by_model": {model: dict(value) for model, value in sorted(model_counts.items())},
        "first_difference_by_model": first_by_model,
        "first_difference": first_difference,
        "first_observed_difference": first_difference,
        "first_difference_interpretation": (
            "First observed by logical token, then main-model computational order: model input, "
            "each decoder layer, final model output, logits. This is not a proven root cause; "
            "parallel operations within a layer have only an approximate order."
        ),
        "sampled_outputs": {
            "scope": "Accepted OFF/ON outputs at the same TP rank and request prompt."
            if mode == "off-on"
            else "Not compared: P and D output/logit phases have different conventions.",
            "compared_workers": counts["sampled_outputs_equal"] + counts["sampled_outputs_different"],
            "equal_workers": counts["sampled_outputs_equal"],
            "different_workers": counts["sampled_outputs_different"],
            "first_difference": first_output_difference,
        },
        "rejection_sampling": _write_rejection_report(reference, candidate, directory),
        "worst_difference": worst_difference,
        "kv_aliases": {
            "scope": "Different physical copies of one logical token within the same worker/call/layer.",
            "equal_tensors": counts["kv_aliases_equal"],
            "different_tensors": counts["kv_aliases_different"],
            "first_conflict": first_alias_conflict,
        },
        "details": "comparisons.jsonl",
    }
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reference", nargs="+", required=True, help="Collected reference worker trees, including all TP ranks"
    )
    parser.add_argument(
        "--candidate", nargs="+", required=True, help="Collected candidate worker trees, including all TP ranks"
    )
    parser.add_argument("--mode", choices=("off-on", "pd-kv"), required=True)
    parser.add_argument("--output", required=True, help="Directory for report.json and comparisons.jsonl")
    parser.add_argument("--request-map", type=Path, help="JSON object mapping OFF request IDs to ON request IDs")
    args = parser.parse_args()
    try:
        report = analyze(
            args.reference,
            args.candidate,
            mode=args.mode,
            output=args.output,
            request_map=_read(args.request_map) if args.request_map else None,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "analysis_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
