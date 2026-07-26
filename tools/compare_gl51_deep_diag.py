#!/usr/bin/env python3
"""Compare two GLM-5.1/DeepSeek flight-recorder runs exactly."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import torch


def _discover(path: Path) -> dict[tuple[int, int], Path]:
    files = [path] if path.is_file() else sorted(path.glob("gl51-deep-*.pt"))
    result: dict[tuple[int, int], Path] = {}
    for file_path in files:
        bundle = torch.load(file_path, map_location="cpu", weights_only=False)
        key = (int(bundle["dp_rank"]), int(bundle["tp_rank"]))
        if key in result:
            raise RuntimeError(
                f"{path}: multiple bundles found for dp/tp rank {key}: "
                f"{result[key]} and {file_path}"
            )
        result[key] = file_path
    if not result:
        raise RuntimeError(f"No gl51-deep-*.pt bundles found under {path}")
    return result


def _load(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != 1:
        raise RuntimeError(
            f"{path}: unsupported schema {bundle.get('schema_version')!r}"
        )
    return bundle


def _raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1)


def _sha256(tensor: torch.Tensor) -> str:
    raw = _raw_bytes(tensor).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _first_different_element(
    baseline: torch.Tensor, candidate: torch.Tensor
) -> tuple[int, ...] | None:
    element_size = baseline.element_size()
    baseline_bytes = _raw_bytes(baseline).reshape(-1, element_size)
    candidate_bytes = _raw_bytes(candidate).reshape(-1, element_size)
    different = (baseline_bytes != candidate_bytes).any(dim=1)
    indices = different.nonzero()
    if indices.numel() == 0:
        return None
    flat_index = int(indices[0].item())
    if baseline.ndim == 0:
        return ()
    coordinates: list[int] = []
    remaining = flat_index
    for size in reversed(baseline.shape):
        coordinates.append(remaining % int(size))
        remaining //= int(size)
    return tuple(reversed(coordinates))


def _float_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float()
    finite = torch.isfinite(values)
    return {
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(values).sum().item()),
        "posinf": int(torch.isposinf(values).sum().item()),
        "neginf": int(torch.isneginf(values).sum().item()),
    }


def _difference_summary(
    baseline: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "baseline_sha256": _sha256(baseline),
        "candidate_sha256": _sha256(candidate),
        "first_different_element": _first_different_element(
            baseline, candidate
        ),
    }
    if baseline.is_floating_point() or baseline.is_complex():
        baseline_float = baseline.float()
        candidate_float = candidate.float()
        both_finite = torch.isfinite(baseline_float) & torch.isfinite(
            candidate_float
        )
        if bool(both_finite.any()):
            abs_diff = (
                baseline_float[both_finite] - candidate_float[both_finite]
            ).abs()
            summary["finite_max_abs_diff"] = float(abs_diff.max().item())
            summary["finite_mean_abs_diff"] = float(abs_diff.mean().item())
        else:
            summary["finite_max_abs_diff"] = math.nan
            summary["finite_mean_abs_diff"] = math.nan
        summary["baseline_nonfinite"] = _float_stats(baseline)
        summary["candidate_nonfinite"] = _float_stats(candidate)
    return summary


_COMPARABLE_METADATA = (
    "model",
    "dtype",
    "tp_size",
    "dp_size",
    "num_reqs",
    "num_tokens_unpadded",
    "num_tokens_padded",
    "num_scheduled_tokens",
    "num_computed_tokens_before",
    "num_prompt_tokens",
    "compilation_mode",
    "configured_cudagraph_mode",
    "runtime_cudagraph_mode",
    "staged_sfa_graph_key",
    "attn_state",
    "is_kv_producer",
    "is_kv_consumer",
)


def _compare_rank(
    rank: tuple[int, int],
    baseline_path: Path,
    candidate_path: Path,
) -> tuple[int | None, str | None]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)

    for key in _COMPARABLE_METADATA:
        baseline_value = baseline["metadata"].get(key)
        candidate_value = candidate["metadata"].get(key)
        if baseline_value != candidate_value:
            raise RuntimeError(
                f"rank={rank}: invalid comparison: metadata {key!r} differs: "
                f"baseline={baseline_value!r}, candidate={candidate_value!r}"
            )

    baseline_values = baseline["values"]
    candidate_values = candidate["values"]
    if baseline_values != candidate_values:
        raise RuntimeError(
            f"rank={rank}: routing/configuration values differ:\n"
            f"baseline={baseline_values!r}\n"
            f"candidate={candidate_values!r}"
        )

    baseline_tensors = baseline["tensors"]
    candidate_tensors = candidate["tensors"]
    baseline_labels = [record["label"] for record in baseline_tensors]
    candidate_labels = [record["label"] for record in candidate_tensors]
    if baseline_labels != candidate_labels:
        for index, (left, right) in enumerate(
            zip(baseline_labels, candidate_labels, strict=False)
        ):
            if left != right:
                raise RuntimeError(
                    f"rank={rank}: checkpoint sequence first differs at "
                    f"index={index}: baseline={left!r}, candidate={right!r}"
                )
        raise RuntimeError(
            f"rank={rank}: checkpoint counts differ: "
            f"baseline={len(baseline_labels)}, "
            f"candidate={len(candidate_labels)}"
        )

    for index, (baseline_record, candidate_record) in enumerate(
        zip(baseline_tensors, candidate_tensors, strict=True)
    ):
        label = baseline_record["label"]
        baseline_tensor = baseline_record["tensor"]
        candidate_tensor = candidate_record["tensor"]
        if (
            baseline_record["dtype"] != candidate_record["dtype"]
            or baseline_record["shape"] != candidate_record["shape"]
            or baseline_record["stride"] != candidate_record["stride"]
        ):
            raise RuntimeError(
                f"rank={rank} checkpoint={label}: tensor contract differs: "
                f"baseline={(baseline_record['dtype'], baseline_record['shape'], baseline_record['stride'])}, "
                f"candidate={(candidate_record['dtype'], candidate_record['shape'], candidate_record['stride'])}"
            )
        if not torch.equal(
            _raw_bytes(baseline_tensor), _raw_bytes(candidate_tensor)
        ):
            summary = _difference_summary(
                baseline_tensor, candidate_tensor
            )
            print(
                f"FIRST_DIVERGENCE rank=dp{rank[0]}/tp{rank[1]} "
                f"checkpoint_index={index} label={label} summary={summary}"
            )
            return index, label

    print(
        f"EXACT_MATCH rank=dp{rank[0]}/tp{rank[1]} "
        f"checkpoints={len(baseline_tensors)}"
    )
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    baseline_files = _discover(args.baseline)
    candidate_files = _discover(args.candidate)
    if baseline_files.keys() != candidate_files.keys():
        raise RuntimeError(
            "Rank sets differ: "
            f"baseline={sorted(baseline_files)}, "
            f"candidate={sorted(candidate_files)}"
        )

    divergences: list[tuple[int, tuple[int, int], str]] = []
    for rank in sorted(baseline_files):
        index, label = _compare_rank(
            rank, baseline_files[rank], candidate_files[rank]
        )
        if index is not None and label is not None:
            divergences.append((index, rank, label))

    if divergences:
        first = min(divergences)
        print(
            "GLOBAL_FIRST_DIVERGENCE "
            f"checkpoint_index={first[0]} "
            f"rank=dp{first[1][0]}/tp{first[1][1]} label={first[2]}"
        )
        return 1
    print("ALL_RANKS_EXACT_MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
