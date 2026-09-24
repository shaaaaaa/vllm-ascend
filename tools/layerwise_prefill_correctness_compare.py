#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Full-value OFF/ON diagnostics. Floating-point differences are reported, not thresholded.

OFF saves CPU tensors; ON loads the corresponding OFF tensor and calls
``compare_tensor_values`` before discarding the pair. No tensor fingerprint is
used. ``compare_runs`` validates both manifests and coverage, and writes the
compact report plus a JSONL record of every difference or structural failure.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from layerwise_prefill_correctness_baseline import normalize_off_directory, resolve_off_directory

VALUE_BLOCK_SIZE = 262144
EXACT_PERCENTILE_LIMIT = 1048576
PERCENTILE_BINS = 2048
ALLOWED_ENV_DIFFERENCES = frozenset({"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "LMCACHE_STORE_ASYNC"})
RECORD_FIELDS = frozenset(
    {"rank", "step", "layer", "kind", "name", "span", "shape", "dtype", "numel", "nonfinite", "path"}
)


def comparable_environment(environment: dict[str, Any]) -> dict[str, Any]:
    """Normalize equivalent CPU capacity spellings without ignoring capacity."""
    if not isinstance(environment, dict):
        raise ValueError("environment must be an object")
    result = dict(environment)
    field = "LMCACHE_MAX_LOCAL_CPU_SIZE"
    if field in result:
        try:
            capacity = Decimal(str(result[field]))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{field} must be a positive finite number") from exc
        if not capacity.is_finite() or capacity <= 0:
            raise ValueError(f"{field} must be a positive finite number")
        result[field] = str(capacity.normalize())
    return result


@dataclass
class _Moments:
    count: int = 0
    finite: int = 0
    nan: int = 0
    posinf: int = 0
    neginf: int = 0
    mean: float = 0.0
    m2: float = 0.0
    squares: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, values: Any) -> None:
        torch = importlib.import_module("torch")
        self.count += values.numel()
        self.nan += int(torch.isnan(values).sum().item())
        self.posinf += int(torch.isposinf(values).sum().item())
        self.neginf += int(torch.isneginf(values).sum().item())
        values = values[torch.isfinite(values)]
        size = values.numel()
        if not size:
            return
        mean = values.mean().item()
        delta = mean - self.mean
        total = self.finite + size
        self.m2 += ((values - mean) ** 2).sum().item() + delta * delta * self.finite * size / total
        self.mean += delta * size / total
        self.finite = total
        self.squares += (values * values).sum().item()
        minimum, maximum = values.min().item(), values.max().item()
        self.minimum = minimum if self.minimum is None else min(self.minimum, minimum)
        self.maximum = maximum if self.maximum is None else max(self.maximum, maximum)

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "finite": self.finite,
            "nonfinite": self.count - self.finite,
            "nan": self.nan,
            "posinf": self.posinf,
            "neginf": self.neginf,
            "mean": self.mean if self.finite else None,
            "std": math.sqrt(max(0.0, self.m2 / self.finite)) if self.finite else None,
            "min": self.minimum,
            "max": self.maximum,
            "rms": math.sqrt(self.squares / self.finite) if self.finite else None,
        }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def compare_tensor_values(reference: Any, actual: Any) -> dict[str, Any]:
    """Compare every CPU element, with bounded float64 scratch memory.

    Population standard deviations and error moments use every finite element.
    Error moments use finite pairs. Small tensors have exact error quantiles;
    large tensors use an all-element logarithmic histogram, reporting each
    quantile's upper bound and its bin interval. Nothing is sampled. Undefined
    normalized errors (nonzero error against zero baseline) are JSON null.
    """
    torch = importlib.import_module("torch")  # Keep manifest-only reporting independent of torch/NPU.
    if not isinstance(reference, torch.Tensor) or not isinstance(actual, torch.Tensor):
        return {"comparable": False, "reason": "reference and actual must be tensors"}
    if reference.device.type != "cpu" or actual.device.type != "cpu":
        return {"comparable": False, "reason": "online comparison requires CPU tensor copies"}
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        return {"comparable": False, "reason": "shape or dtype differs"}
    if reference.is_complex() or reference.is_quantized:
        return {"comparable": False, "reason": "complex/quantized tensors are not supported"}
    left, right = reference.reshape(-1), actual.reshape(-1)
    baseline, candidate, errors, paired_baseline = _Moments(), _Moments(), _Moments(), _Moments()
    mismatched = new_nonfinite = nonfinite_pattern_changed = zeros = 0
    min_positive: float | None = None
    exact_parts = []
    for start in range(0, left.numel(), VALUE_BLOCK_SIZE):
        original_left, original_right = left[start : start + VALUE_BLOCK_SIZE], right[start : start + VALUE_BLOCK_SIZE]
        lhs, rhs = original_left.to(torch.float64), original_right.to(torch.float64)
        if reference.is_floating_point():
            mismatched += int(((lhs != rhs) & ~(torch.isnan(lhs) & torch.isnan(rhs))).sum().item())
        else:
            mismatched += int((original_left != original_right).sum().item())
        baseline.update(lhs)
        candidate.update(rhs)
        lhs_finite, rhs_finite = torch.isfinite(lhs), torch.isfinite(rhs)
        new_nonfinite += int((lhs_finite & ~rhs_finite).sum().item())
        nonfinite_pattern_changed += int(
            (
                (lhs_finite != rhs_finite)
                | (torch.isnan(lhs) != torch.isnan(rhs))
                | (torch.isposinf(lhs) != torch.isposinf(rhs))
                | (torch.isneginf(lhs) != torch.isneginf(rhs))
            )
            .sum()
            .item()
        )
        paired = lhs_finite & rhs_finite
        paired_baseline.update(lhs[paired])
        absolute = (lhs[paired] - rhs[paired]).abs()
        errors.update(absolute)
        zeros += int((absolute == 0).sum().item())
        positive = absolute[absolute > 0]
        if positive.numel():
            value = positive.min().item()
            min_positive = value if min_positive is None else min(min_positive, value)
        if left.numel() <= EXACT_PERCENTILE_LIMIT:
            exact_parts.append(absolute)
    if any(
        not math.isfinite(moment.squares) or not math.isfinite(moment.m2) for moment in (baseline, candidate, errors)
    ):
        return {"comparable": False, "reason": "numeric moments overflow float64"}
    if errors.count != errors.finite:
        return {"comparable": False, "reason": "finite-pair subtraction overflowed float64"}
    error_summary = errors.summary()
    percentile: dict[str, Any] = {"p50": None, "p95": None, "p99": None, "percentile_method": "no_finite_pairs"}
    if errors.finite and left.numel() <= EXACT_PERCENTILE_LIMIT:
        values = torch.cat(exact_parts)
        quantiles = torch.quantile(values, torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64)).tolist()
        percentile.update(zip(("p50", "p95", "p99"), quantiles))
        percentile["percentile_method"] = "exact_linear"
    elif errors.finite:
        maximum = errors.maximum or 0.0
        lower = min_positive or maximum
        edges = (
            torch.logspace(math.log10(lower), math.log10(maximum), PERCENTILE_BINS, dtype=torch.float64)
            if lower
            else None
        )
        histogram = torch.zeros(PERCENTILE_BINS, dtype=torch.int64)
        if edges is not None:
            for start in range(0, left.numel(), VALUE_BLOCK_SIZE):
                lhs = left[start : start + VALUE_BLOCK_SIZE].to(torch.float64)
                rhs = right[start : start + VALUE_BLOCK_SIZE].to(torch.float64)
                paired = torch.isfinite(lhs) & torch.isfinite(rhs)
                values = (lhs[paired] - rhs[paired]).abs()
                bins = torch.bucketize(values[values > 0], edges).clamp_max(PERCENTILE_BINS - 1)
                histogram += torch.bincount(bins, minlength=PERCENTILE_BINS)
        cumulative = histogram.cumsum(0)
        bounds = {}
        for name, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
            target = math.ceil(quantile * errors.finite) - zeros
            if target <= 0 or edges is None:
                value, interval = 0.0, [0.0, 0.0]
            else:
                index = int(torch.searchsorted(cumulative, target).item())
                index = min(index, PERCENTILE_BINS - 1)
                value = min(maximum, edges[index].item())
                interval = [lower if index == 0 else edges[index - 1].item(), value]
            percentile[name], bounds[name] = value, interval
        percentile.update(percentile_method="all_element_log_histogram_upper_bound", percentile_bounds=bounds)
    base_summary, actual_summary = baseline.summary(), candidate.summary()
    return {
        "comparable": True,
        "category": "floating" if reference.is_floating_point() else "integer",
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "numel": reference.numel(),
        "baseline": base_summary,
        "candidate": actual_summary,
        "new_nonfinite": new_nonfinite,
        "nonfinite_pattern_changed": nonfinite_pattern_changed,
        "mismatched": mismatched,
        "mismatch_rate": mismatched / reference.numel() if reference.numel() else 0.0,
        "abs_diff": {"mean": error_summary["mean"], "max": error_summary["max"], **percentile},
        "rmse": error_summary["rms"],
        "relative_l2": _ratio(error_summary["rms"], paired_baseline.summary()["rms"]),
        "rmse_over_std": _ratio(error_summary["rms"], base_summary["std"]),
    }


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _no_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant {value}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_no_constant)


def _key(record: dict[str, Any]) -> tuple[int, int, int, str, str]:
    return record["rank"], record["step"], record["layer"], record["kind"], record["name"]


def _identity(key: tuple[int, int, int, str, str]) -> dict[str, Any]:
    return dict(zip(("rank", "step", "layer", "kind", "name"), key))


def _order(key: tuple[int, int, int, str, str]) -> tuple[Any, ...]:
    return key[1], key[2], key[3], key[4], key[0]


def _span(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_integer(x) for x in value) and value[0] < value[1]


def _tensor_path(case: Path, value: Any) -> Path:
    _require(isinstance(value, str) and bool(value), "tensor path is missing")
    relative = Path(value)
    _require(not relative.is_absolute() and relative.suffix == ".pt", "tensor path must be a relative .pt file")
    resolved = (case / relative).resolve()
    _require(resolved.is_relative_to(case.resolve()), "tensor path escapes case directory")
    _require(resolved.is_file(), f"tensor file missing: {value}")
    return resolved


def _comparison_metadata(record: dict[str, Any]) -> tuple[list[int], int, int]:
    fields = {"comparison_slice", "comparison_shape", "comparison_numel", "comparison_nonfinite"}
    if not fields.intersection(record):
        return record["shape"], record["numel"], record["nonfinite"]
    _require(record.keys() >= fields, "partial comparison slice metadata")
    shape = list(record["shape"])
    selection = record["comparison_slice"]
    if selection is not None:
        _require(isinstance(selection, dict) and bool(shape), "invalid comparison slice")
        _require(selection.get("axis") == 0 and selection.get("start") == 0, "comparison slice must be a row prefix")
        _require(
            _integer(selection.get("end")) and selection["end"] <= shape[0], "comparison slice exceeds archived rows"
        )
        if record["kind"].startswith("kv_"):
            _require(selection["end"] == shape[0], "comparison cannot omit logical KV rows")
        shape[0] = selection["end"]
    _require(record["comparison_shape"] == shape, "comparison shape differs from slice")
    size = math.prod(shape)
    _require(record["comparison_numel"] == size, "comparison numel differs from slice")
    _require(
        _integer(record["comparison_nonfinite"]) and record["comparison_nonfinite"] <= size,
        "invalid comparison nonfinite",
    )
    return shape, size, record["comparison_nonfinite"]


def _validate_record(record: Any, rank: int, case: Path, off: bool) -> None:
    _require(isinstance(record, dict) and record.keys() >= RECORD_FIELDS, "record has missing fields")
    _require(record["rank"] == rank and _integer(record["rank"]), "record rank differs from rank directory")
    _require(_integer(record["step"]) and _integer(record["layer"]), "invalid step/layer")
    _require(
        all(isinstance(record[x], str) and record[x] for x in ("kind", "name", "dtype")), "invalid kind/name/dtype"
    )
    _require(_span(record["span"]), "invalid chunk span")
    shape = record["shape"]
    _require(isinstance(shape, list) and all(_integer(x) for x in shape), "invalid shape")
    _require(_integer(record["numel"]) and math.prod(shape) == record["numel"], "shape/numel differs")
    _require(_integer(record["nonfinite"]) and record["nonfinite"] <= record["numel"], "invalid nonfinite count")
    comparison_shape, comparison_size, comparison_nonfinite = _comparison_metadata(record)
    if off or record["path"] is not None:
        _tensor_path(case, record["path"])
    if not off:
        comparison = record.get("comparison")
        _require(isinstance(comparison, dict), "ON record lacks online comparison")
        _require(comparison.get("comparable") is True, f"tensor not comparable: {comparison.get('reason', 'unknown')}")
        for field, expected in (("shape", comparison_shape), ("dtype", record["dtype"]), ("numel", comparison_size)):
            _require(comparison.get(field) == expected, f"comparison {field} differs from tensor metadata")
        for field in ("new_nonfinite", "mismatched", "nonfinite_pattern_changed"):
            _require(_integer(comparison.get(field)) and comparison[field] <= comparison_size, f"invalid {field} count")
        _require(comparison.get("category") in ("floating", "integer"), "invalid numeric category")
        for field in ("baseline", "candidate"):
            summary = comparison.get(field)
            _require(isinstance(summary, dict) and summary.get("count") == comparison_size, f"invalid {field} summary")
            for count in ("finite", "nonfinite", "nan", "posinf", "neginf"):
                _require(_integer(summary.get(count)) and summary[count] <= comparison_size, f"invalid {field}.{count}")
            _require(summary["finite"] + summary["nonfinite"] == comparison_size, f"invalid {field} finite count")
            _require(
                summary["nan"] + summary["posinf"] + summary["neginf"] == summary["nonfinite"],
                f"invalid {field} nonfinite types",
            )
            for metric in ("mean", "std", "min", "max", "rms"):
                _require(metric in summary, f"missing {field}.{metric}")
        _require(comparison["candidate"].get("nonfinite") == comparison_nonfinite, "candidate nonfinite count differs")
        _require(isinstance(comparison.get("abs_diff"), dict), "comparison lacks absolute error statistics")
        _require(
            comparison["abs_diff"].keys() >= {"mean", "max", "p50", "p95", "p99", "percentile_method"},
            "comparison error distribution incomplete",
        )
        for field in ("rmse", "relative_l2", "rmse_over_std", "mismatch_rate"):
            _require(field in comparison, f"comparison lacks {field}")


def _layer_list(value: Any, label: str, allow_empty: bool = False) -> list[int]:
    _require(isinstance(value, list) and all(_integer(x) for x in value), f"invalid {label}")
    _require(len(value) == len(set(value)) and (allow_empty or bool(value)), f"empty/duplicate {label}")
    return sorted(value)


def _validate_coverage(
    summary: dict[str, Any], records: dict[tuple[Any, ...], dict[str, Any]], prompt: int, model_info: dict[str, Any]
) -> None:
    rank = summary["rank"]
    _require(summary.get("complete") is True, f"rank {rank} coverage incomplete")
    _require(not summary.get("errors"), f"rank {rank} worker coverage errors: {summary.get('errors')}")
    _require(summary.get("records") == len(records), f"rank {rank} record count differs")
    _require(summary.get("prompt_length") == prompt, f"rank {rank} prompt length differs")
    layers = _layer_list(summary.get("decoder_layers"), "decoder_layers")
    hidden = summary.get("model_num_hidden_layers", summary.get("num_hidden_layers"))
    _require(_integer(hidden, 1) and layers == list(range(hidden)), "decoder layers do not cover full model")
    _require(hidden == model_info.get("num_hidden_layers"), "layer count differs from independent model config")
    sfa = _layer_list(summary.get("sfa_layers"), "sfa_layers")
    indexer = _layer_list(summary.get("indexer_layers"), "indexer_layers", allow_empty=True)
    _require(sfa == layers, "SFA layers do not cover every decoder layer")
    indexer_types = model_info.get("indexer_types")
    pattern = model_info.get("index_topk_pattern")
    if indexer_types is None:
        _require(
            pattern is None or (isinstance(pattern, list) and "S" not in pattern),
            "legacy index_topk_pattern sharing is unsupported",
        )
        expected_indexer = layers
    else:
        _require(isinstance(indexer_types, list) and len(indexer_types) == hidden, "invalid model indexer_types")
        _require(all(value in ("full", "shared") for value in indexer_types), "unsupported model indexer_types")
        _require(indexer_types[0] == "full", "first indexer layer must be a full producer")
        if isinstance(pattern, list) and len(pattern) == hidden:
            _require(
                all((entry == "S") == (kind == "shared") for entry, kind in zip(pattern, indexer_types)),
                "index_topk_pattern disagrees with indexer_types",
            )
        expected_indexer = [layer for layer, value in enumerate(indexer_types) if value == "full"]
    _require(indexer == expected_indexer, "indexer layers differ from independent model config")
    index_cache = _layer_list(summary.get("indexer_cache_layers"), "indexer_cache_layers", allow_empty=True)
    _require(index_cache == expected_indexer, "indexer cache layers differ from independent model config")
    scale = _layer_list(summary.get("scale_layers"), "scale_layers", allow_empty=True)
    _require(set(scale) <= set(index_cache), "scale layers outside indexer cache inventory")
    steps = summary.get("steps")
    _require(isinstance(steps, list) and bool(steps), "missing prefill steps")
    frontier = 0
    for index, step in enumerate(steps):
        _require(isinstance(step, dict) and step.get("step") == index, "prefill step IDs are not contiguous")
        _require(step.get("phase") == "prefill" and _span(step.get("span")), "invalid prefill step")
        _require(step["span"][0] == frontier, "prefill steps have a gap or overlap")
        frontier = step["span"][1]
    _require(frontier == prompt, "prefill steps do not cover the complete prompt")
    layout = summary.get("layout")
    _require(isinstance(layout, dict) and layout.get("layout") == "merged", "merged layout evidence missing")
    _require("remote_url" in layout and layout["remote_url"] is None, "remote layout backend is not disabled")
    _require(layout.get("native_mooncake") is False, "native Mooncake must be disabled")
    _require(layout.get("production_transfer_unchanged") is True, "production DMA implementation was changed")
    _require(_integer(summary.get("merged_load_sources")), "merged load source evidence missing")
    _require(summary.get("legacy_load_sources") == 0, "legacy H2D sources observed or evidence missing")
    if len(steps) > 1:
        _require(summary["merged_load_sources"] > 0, "no merged H2D sources observed for historical prefix")
    declared: set[tuple[str, str, int, str]] = set()
    roles = summary.get("required_roles")
    _require(isinstance(roles, list) and bool(roles), "missing required_roles")
    for role in roles:
        _require(isinstance(role, dict) and role.get("steps") in ("all", "history"), "invalid required role")
        role_layers = _layer_list(role.get("layers"), "role layers", allow_empty=True)
        _require(set(role_layers) <= set(layers), "required role has an unknown layer")
        _require(all(isinstance(role.get(x), str) and role[x] for x in ("kind", "name")), "invalid required role name")
        for layer in role_layers:
            declared.add((role["kind"], role["name"], layer, role["steps"]))
    minimum_roles = (
        ("decoder", ("input", "output", "positions"), layers, "all"),
        ("sfa", ("input", "output"), sfa, "all"),
        ("attention", ("query_nope", "query_rope", "topk", "output"), sfa, "all"),
        ("indexer", ("query", "weights", "topk"), indexer, "all"),
        ("indexer_input", ("x", "q_c"), indexer, "all"),
        ("kv_current", ("nope", "rope"), sfa, "all"),
        ("kv_loaded", ("nope", "rope"), sfa, "history"),
        ("kv_current", ("index",), index_cache, "all"),
        ("kv_loaded", ("index",), index_cache, "history"),
        ("kv_current", ("scale",), scale, "all"),
        ("kv_loaded", ("scale",), scale, "history"),
    )
    for kind, names, role_layers, selector in minimum_roles:
        for name in names:
            for layer in role_layers:
                _require(
                    (kind, name, layer, selector) in declared, f"required role omitted: {kind}/{name}/layer{layer}"
                )
    for kind, name, layer, selector in sorted(declared):
        for step in steps:
            if selector == "history" and step["span"][0] == 0:
                continue
            key = (rank, step["step"], layer, kind, name)
            _require(key in records, f"required tensor missing: {key}")
    for key, record in records.items():
        _require(record["layer"] in layers and record["step"] < len(steps), f"unexpected layer/step: {key}")
        span = steps[record["step"]]["span"]
        _require(record["span"] == span, f"record span differs from step: {key}")
        if record["kind"] in ("kv_current", "kv_loaded"):
            expected = span if record["kind"] == "kv_current" else [0, span[0]]
            _require(
                expected[1] > expected[0] and record.get("positions") == expected, f"KV positions incomplete: {key}"
            )
            _require(record.get("token_axis") == 0 and bool(record["shape"]), f"KV token axis missing: {key}")
            _require(
                record["shape"][0] == expected[1] - expected[0], f"KV tensor does not contain all positions: {key}"
            )


def _load_case(
    case: Path, errors: list[dict[str, Any]], model_info: dict[str, Any], *, case_name: str | None = None
) -> dict[str, Any]:
    name = case_name or case.name
    data: dict[str, Any] = {"records": {}, "coverage": []}

    def error(detail: str) -> None:
        errors.append({"type": "structure", "case": name, "detail": detail})

    for filename in ("result", "environment", "engine_options", "coverage"):
        try:
            data[filename] = _read_json(case / f"{filename}.json")
        except (OSError, ValueError) as exc:
            error(f"{filename}.json: {exc}")
    result, options = data.get("result", {}), data.get("engine_options", {})
    try:
        _require(isinstance(result, dict) and result.get("completed") is True, "run did not complete")
        _require("case" not in result or result["case"] == name, f"result.case differs from {name}")
        prompt = result.get("prompt_token_ids")
        _require(
            isinstance(prompt, list) and bool(prompt) and all(_integer(x) for x in prompt), "prompt tokens missing"
        )
        _require(result.get("prompt_length") == len(prompt), "prompt length differs from token IDs")
        output = result.get("token_ids")
        _require(
            isinstance(output, list) and bool(output) and all(_integer(x) for x in output), "output tokens missing"
        )
        _require(result.get("output_length") == len(output), "output length differs from token IDs")
        _require(result.get("num_cached_tokens") == 0, "run used a pre-existing prefix cache")
        _require(isinstance(options, dict) and _integer(options.get("tensor_parallel_size"), 1), "TP size missing")
    except (TypeError, ValueError) as exc:
        error(str(exc))
    if not isinstance(result, dict):
        result = {}
    environment = data.get("environment")
    try:
        comparable_environment(environment)
    except ValueError as exc:
        error(str(exc))
    for field in ALLOWED_ENV_DIFFERENCES:
        if not isinstance(environment, dict) or environment.get(field) != ("false" if name == "off" else "true"):
            error(f"{field} must be explicitly {'false' if name == 'off' else 'true'} for {name}")
    expected_ranks = (
        set(range(options["tensor_parallel_size"]))
        if isinstance(options, dict) and _integer(options.get("tensor_parallel_size"), 1)
        else set()
    )
    actual_ranks: set[int] = set()
    for directory in sorted((case / "tensors").glob("rank*")):
        try:
            rank = int(directory.name.removeprefix("rank"))
            _require(rank >= 0 and directory.is_dir() and rank not in actual_ranks, "invalid/duplicate rank directory")
            actual_ranks.add(rank)
            with (directory / "index.jsonl").open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line, parse_constant=_no_constant)
                        _validate_record(record, rank, case, name == "off")
                        key = _key(record)
                        _require(key not in data["records"], f"duplicate tensor record: {key}")
                        data["records"][key] = record
                    except (TypeError, ValueError, KeyError) as exc:
                        error(f"{directory.name}/index.jsonl:{line_number}: {exc}")
        except (OSError, ValueError) as exc:
            error(str(exc))
    if actual_ranks != expected_ranks:
        error(f"rank directories differ: expected {sorted(expected_ranks)}, got {sorted(actual_ranks)}")
    coverage = data.get("coverage")
    try:
        _require(isinstance(coverage, list) and bool(coverage), "coverage must be a nonempty rank summary list")
        ranks = [entry.get("rank") for entry in coverage if isinstance(entry, dict)]
        _require(all(_integer(rank) for rank in ranks) and len(ranks) == len(coverage), "invalid coverage rank")
        _require(set(ranks) == expected_ranks and len(ranks) == len(set(ranks)), "coverage ranks missing/duplicated")
        for summary in coverage:
            records = {key: value for key, value in data["records"].items() if key[0] == summary["rank"]}
            _validate_coverage(summary, records, result.get("prompt_length", 0), model_info)
        first = coverage[0]
        for summary in coverage[1:]:
            for field in ("decoder_layers", "sfa_layers", "indexer_layers", "scale_layers", "steps", "required_roles"):
                _require(summary.get(field) == first.get(field), f"rank coverage differs for {field}")
    except (TypeError, ValueError, KeyError) as exc:
        error(str(exc))
    if not isinstance(data.get("coverage"), list):
        data["coverage"] = []
    return data


def validate_off_baseline(off_dir: str | Path, model_info: dict[str, Any]) -> dict[str, Any]:
    """Validate every OFF manifest/file and full coverage, without writing files."""
    case = normalize_off_directory(off_dir)
    if not isinstance(model_info, dict) or not _integer(model_info.get("num_hidden_layers"), 1):
        raise ValueError("OFF baseline independent model layer count is missing")
    errors: list[dict[str, Any]] = []
    data = _load_case(case, errors, model_info, case_name="off")
    if errors:
        raise ValueError(f"Invalid OFF baseline: {errors[0]['detail']}")
    return data


def compare_runs(root: str | Path) -> dict[str, Any]:
    """Validate OFF/ON artifacts and write report.json/diff.jsonl; return report.

    ``passed`` means complete comparable coverage, unchanged output token IDs,
    and no newly nonfinite values. It does NOT assert a numerical tolerance.
    Float/integer internal differences remain explicit diagnostic results.
    """
    root = Path(root)
    differences: list[dict[str, Any]] = []
    try:
        model_info = _read_json(root / "model_info.json")
        _require(
            isinstance(model_info, dict) and _integer(model_info.get("num_hidden_layers"), 1),
            "model layer count missing",
        )
    except (OSError, ValueError) as exc:
        differences.append({"type": "structure", "detail": f"independent model_info.json: {exc}"})
        model_info = {}
    try:
        off_dir = resolve_off_directory(root)
        off = _load_case(off_dir, differences, model_info, case_name="off")
    except ValueError as exc:
        differences.append({"type": "structure", "case": "off", "detail": str(exc)})
        off = {"records": {}, "coverage": []}
    on = _load_case(root / "on", differences, model_info, case_name="on")
    for field in ("engine_options", "environment"):
        left, right = off.get(field), on.get(field)
        if field == "environment" and isinstance(left, dict) and isinstance(right, dict):
            try:
                left, right = comparable_environment(left), comparable_environment(right)
            except ValueError as exc:
                differences.append({"type": "structure", "detail": str(exc)})
            left = {key: value for key, value in left.items() if key not in ALLOWED_ENV_DIFFERENCES}
            right = {key: value for key, value in right.items() if key not in ALLOWED_ENV_DIFFERENCES}
        if left != right or not isinstance(left, dict):
            differences.append({"type": "structure", "detail": f"OFF/ON {field} differs or is missing"})
    off_result, on_result = off.get("result", {}), on.get("result", {})
    if not isinstance(off_result, dict):
        off_result = {}
    if not isinstance(on_result, dict):
        on_result = {}
    if off_result.get("prompt_token_ids") != on_result.get("prompt_token_ids"):
        differences.append({"type": "structure", "detail": "OFF/ON prompt token IDs differ"})
    for field in ("steps", "decoder_layers", "sfa_layers", "indexer_layers", "scale_layers", "required_roles"):
        left = [entry.get(field) for entry in off.get("coverage", []) if isinstance(entry, dict)]
        right = [entry.get(field) for entry in on.get("coverage", []) if isinstance(entry, dict)]
        if left != right:
            differences.append({"type": "structure", "detail": f"OFF/ON coverage {field} differs"})
    left_records, right_records = off["records"], on["records"]
    all_keys = sorted(left_records.keys() | right_records.keys(), key=_order)
    counts = {
        "off": len(left_records),
        "on": len(right_records),
        "compared": 0,
        "different": 0,
        "missing": 0,
        "extra": 0,
    }
    new_nonfinite = nonfinite_pattern_changed = integer_differences = floating_differences = 0
    topk_elements = topk_mismatched = archived_elements = compared_elements = 0
    worst: dict[str, Any] = {}
    for key in all_keys:
        left, right = left_records.get(key), right_records.get(key)
        if left is None or right is None:
            name = "extra" if left is None else "missing"
            counts[name] += 1
            differences.append({"type": "structure", **_identity(key), "detail": f"ON {name} tensor"})
            continue
        fields = (
            "span",
            "shape",
            "dtype",
            "numel",
            "positions",
            "token_axis",
            "comparison_slice",
            "comparison_shape",
            "comparison_numel",
        )
        changed = [field for field in fields if left.get(field) != right.get(field)]
        comparison = right["comparison"]
        if comparison["baseline"].get("nonfinite") != _comparison_metadata(left)[2]:
            changed.append("baseline.nonfinite")
        if changed:
            differences.append({"type": "structure", **_identity(key), "detail": f"tensor metadata differs: {changed}"})
            continue
        counts["compared"] += 1
        archived_elements += right["numel"]
        compared_elements += comparison["numel"]
        new_nonfinite += comparison["new_nonfinite"]
        nonfinite_pattern_changed += comparison["nonfinite_pattern_changed"]
        if right["name"] == "topk" and comparison["category"] == "integer":
            topk_elements += comparison["numel"]
            topk_mismatched += comparison["mismatched"]
        if comparison["mismatched"] or comparison["new_nonfinite"]:
            counts["different"] += 1
            if comparison.get("category") == "integer":
                integer_differences += 1
            else:
                floating_differences += 1
            differences.append({"type": "numeric", **_identity(key), "comparison": comparison})
        for metric in ("relative_l2", "rmse_over_std", "rmse"):
            value = comparison.get(metric)
            if comparison["category"] == "floating" and isinstance(value, int | float) and math.isfinite(value):
                if metric not in worst or value > worst[metric]["value"]:
                    worst[metric] = {**_identity(key), "value": value}
    output_equal = bool(off_result.get("token_ids")) and off_result.get("token_ids") == on_result.get("token_ids")
    if not output_equal:
        left, right = off_result.get("token_ids", []), on_result.get("token_ids", [])
        left = left if isinstance(left, list) else []
        right = right if isinstance(right, list) else []
        index = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
        differences.append({"type": "output_tokens", "first_index": index, "off": left, "on": right})
    structural_errors = sum(item["type"] == "structure" for item in differences)
    complete = structural_errors == 0
    passed = complete and output_equal and new_nonfinite == 0
    status = (
        "incomplete"
        if not complete
        else "new_nonfinite"
        if new_nonfinite
        else "output_tokens_changed"
        if not output_equal
        else "complete_with_differences"
        if counts["different"]
        else "equal"
    )
    tensor_differences = [item for item in differences if "step" in item]
    first = min(tensor_differences, key=lambda item: _order(_key(item))) if tensor_differences else None
    report = {
        "schema_version": 1,
        "complete": complete,
        "passed": passed,
        "status": status,
        "output_tokens_equal": output_equal,
        "numeric_tolerance_applied": False,
        "scope": "all recorded main-backbone prefill tensors; MTP/decode tensor internals excluded",
        "counts": {
            **counts,
            "structural_errors": structural_errors,
            "floating_differences": floating_differences,
            "integer_differences": integer_differences,
            "new_nonfinite": new_nonfinite,
            "nonfinite_pattern_changed": nonfinite_pattern_changed,
            "archived_elements": archived_elements,
            "compared_elements": compared_elements,
        },
        "topk": {
            "elements": topk_elements,
            "mismatched": topk_mismatched,
            "mismatch_rate": topk_mismatched / topk_elements if topk_elements else None,
        },
        "first_divergence": first,
        "worst": worst,
        "errors": [item for item in differences if item["type"] == "structure"],
        "interpretation": (
            "passed checks coverage, comparability, new nonfinite values and output tokens; "
            "it is not a float-error tolerance verdict. Valid rows exclude declared TP padding; "
            "OFF archives remain full."
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (root / "diff.jsonl").open("w", encoding="utf-8") as stream:
        for difference in differences:
            stream.write(json.dumps(difference, allow_nan=False) + "\n")
    return report


def print_report(report: dict[str, Any]) -> None:
    """Print at most ten compact lines, without a numerical pass claim."""
    print(
        f"Status: {report['status']}  complete={report['complete']} output_tokens_equal={report['output_tokens_equal']}"
    )
    counts = report["counts"]
    print(
        f"Tensors: OFF={counts['off']} ON={counts['on']} compared={counts['compared']} different={counts['different']}"
    )
    print(
        f"Coverage/errors={counts['structural_errors']} new_nonfinite={counts['new_nonfinite']} "
        f"nonfinite_pattern_changed={counts['nonfinite_pattern_changed']}"
    )
    print(
        f"Internal differences: float={counts['floating_differences']} integer={counts['integer_differences']}; "
        "no tolerance applied"
    )
    topk = report["topk"]
    print(f"Integer top-k: mismatched={topk['mismatched']}/{topk['elements']} rate={topk['mismatch_rate']}")
    first = report["first_divergence"]
    if first:
        print(
            f"First: rank={first['rank']} step={first['step']} layer={first['layer']} {first['kind']}/{first['name']}"
        )
    for metric, entry in report["worst"].items():
        print(
            f"Max float {metric}={entry['value']:.6g}: r{entry['rank']} step{entry['step']} "
            f"layer{entry['layer']} {entry['kind']}/{entry['name']}"
        )
    if report["errors"]:
        print("First structural error: " + str(report["errors"][0]["detail"]).replace("\n", " ")[:200])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing off/ and on/")
    args = parser.parse_args()
    report = compare_runs(args.root)
    print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
