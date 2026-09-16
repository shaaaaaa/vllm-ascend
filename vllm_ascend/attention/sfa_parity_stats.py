# SPDX-License-Identifier: Apache-2.0
"""CPU-only descriptive statistics; no tolerances, device hooks or parity verdict."""

import math

import torch


def moments(value: torch.Tensor) -> dict:
    """Population moments of all selected elements, accumulated in float64."""
    if value.device.type != "cpu":
        raise ValueError("Statistics must use CPU snapshots after forward")
    value = value.double().reshape(-1)
    count = value.numel()
    if not count:
        return {"count": 0, "mean": 0.0, "m2": 0.0, "max": 0.0, "nonzero": 0}
    if not torch.isfinite(value).all():
        raise ValueError("Statistics contain NaN/Inf")
    variance, mean = torch.var_mean(value, unbiased=False)
    return {
        "count": count,
        "mean": mean.item(),
        "m2": variance.item() * count,
        "max": value.max().item(),
        "nonzero": int(torch.count_nonzero(value)),
    }


def merge_moments(left: dict, right: dict) -> dict:
    """Merge element-weighted moments, not averages of per-step variances."""
    if not left["count"]:
        return right.copy()
    if not right["count"]:
        return left.copy()
    n, m = left["count"], right["count"]
    delta = right["mean"] - left["mean"]
    return {
        "count": n + m,
        "mean": left["mean"] + delta * m / (n + m),
        "m2": left["m2"] + right["m2"] + delta * delta * n * m / (n + m),
        "max": max(left["max"], right["max"]),
        "nonzero": left["nonzero"] + right["nonzero"],
    }


def alignment_reason(reference: dict, actual: dict) -> str | None:
    # A differing speculative row also makes the forward incomparable. Do not
    # silently treat equal latest tokens as an equal complete decode history.
    for key in ("rank", "tp_size", "decode", "layers", "trace_residual"):
        if reference[key] != actual[key]:
            raise ValueError(f"Different statistics configuration: {key}")
    for key in ("step", "rows"):
        if reference[key] != actual[key]:
            return key
    for key in ("input_ids", "positions", "seq_lens", "query_ends"):
        ref, val = reference[key], actual[key]
        if ref.dtype != val.dtype or not torch.equal(ref, val):
            return key
    return None


def add_step(stages: dict, reference: dict, actual: dict) -> None:
    """Update every observed stage; padding/different logical KV are excluded."""
    refs, values = reference["tensors"], actual["tensors"]
    if not refs or list(refs) != list(values):
        raise ValueError("Different statistics probe coverage")
    for name, ref in refs.items():
        val = values[name]
        if ref.device.type != "cpu" or val.device.type != "cpu":
            raise ValueError("Statistics must use CPU snapshots after forward")
        if ref.shape != val.shape or ref.dtype != val.dtype or not ref.numel():
            raise ValueError(f"{name}: invalid statistics shape/dtype/empty observation")
        if not torch.isfinite(ref).all() or not torch.isfinite(val).all():
            raise ValueError(f"{name}: statistics contain NaN/Inf")
        total = ref.numel()
        if name.endswith((" kv_nope", " kv_pe")):
            prefix = name.split(" ")[0]
            # Compare KV for the SAME logical selected token only, never padded
            # zeros or two unrelated tokens occupying the same top-k column.
            valid = refs[f"{prefix} valid"] & values[f"{prefix} valid"]
            valid &= refs[f"{prefix} topk"] == values[f"{prefix} topk"]
            ref, val = ref[valid], val[valid]
        ref, val = ref.double(), val.double()
        observation = {
            "diff_abs": moments((val - ref).abs()),
            "eager_abs": moments(ref.abs()),
            "graph_abs": moments(val.abs()),
        }
        stage = stages.setdefault(name, {"observations": 0, "excluded": 0})
        stage["observations"] += 1
        stage["excluded"] += total - ref.numel()
        for metric, data in observation.items():
            stage[metric] = merge_moments(stage[metric], data) if metric in stage else data


def merge_stages(reports: list[dict]) -> dict:
    result = {}
    for report in reports:
        for name, stage in report.items():
            merged = result.setdefault(name, {"observations": 0, "excluded": 0})
            merged["observations"] += stage["observations"]
            merged["excluded"] += stage["excluded"]
            for metric in ("diff_abs", "eager_abs", "graph_abs"):
                merged[metric] = merge_moments(merged[metric], stage[metric]) if metric in merged else stage[metric]
    return result


def print_statistics(eager: list[dict], graph: list[dict]) -> None:
    """Compact pooled stage rows, retaining max errors instead of hiding bad ranks."""
    eager_by_rank = {report["rank"]: report for report in eager}
    reports = []
    for rank in sorted(graph, key=lambda report: report["rank"]):
        stats = rank.get("absolute_statistics")
        if not stats or (
            stats["eager_steps"] != eager_by_rank[rank["rank"]]["decode_observations"]
            or stats["graph_steps"] != rank["decode_observations"]
            or not 0 < stats["compared_steps"] <= min(stats["eager_steps"], stats["graph_steps"])
            or not stats["stages"]
        ):
            raise ValueError(f"rank={rank['rank']}: missing/incomplete absolute statistics")
        reports.append(stats["stages"])
        complete = stats["compared_steps"] == stats["eager_steps"] == stats["graph_steps"]
        unaligned = stats["first_unaligned"]
        if not complete and unaligned is None:
            unaligned = "graph generation ended before the eager forwards"
        print(
            f"[SFA_STATS] rank={rank['rank']} {'COMPLETE' if complete else 'PARTIAL'} "
            f"compared_steps={stats['compared_steps']} eager_steps={stats['eager_steps']} "
            f"graph_steps={stats['graph_steps']} first_unaligned={unaligned}",
            flush=True,
        )
    print(
        "[SFA_STATS] pooled aligned decode elements across TP ranks/steps; "
        "diff_abs=abs(graph-eager), eager_abs=abs(eager), graph_abs=abs(graph); "
        "population variance; no tolerance gate; excluded KV=padding/different logical selection",
        flush=True,
    )
    for name, stage in merge_stages(reports).items():
        fields = []
        for metric in ("diff_abs", "eager_abs", "graph_abs"):
            data = stage[metric]
            if not data["count"]:
                fields.append(f"{metric}(no comparable elements)")
                continue
            variance = max(0.0, data["m2"] / data["count"])
            fields.append(
                f"{metric}(mean={data['mean']:.6g} std={math.sqrt(variance):.6g} "
                f"var={variance:.6g} max={data['max']:.6g} nonzero={data['nonzero']})"
            )
        print(
            f"[SFA_STATS] {name} n={stage['diff_abs']['count']} excluded={stage['excluded']} " + " ".join(fields),
            flush=True,
        )
