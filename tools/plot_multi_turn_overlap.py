#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot per-round prefill overlap using multi_turn_kv_reuse.py artifacts.

python3 tools/plot_multi_turn_overlap.py server.log --run-dir multi-turn-on \
    --aggregate-ranks --output multi-turn-overlap.png
python3 tools/plot_multi_turn_overlap.py --run-dir multi-turn-on \
    --ttft-only --output multi-turn-ttft.png
Only offline plotting needs matplotlib; no vLLM or NPU imports are required.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

from plot_prefill_overlap import (
    COVERAGE_DEFINITION,
    WAIT_DETAILS,
    read_records,
    require_bank_reuse_coverage,
    wait_details,
    write_csv,
)
from prefill_overlap_collect import add_collection_arguments, collect_logs

CORE_TIMINGS = ("dma_ms", "save_ms", "load_ms", "wait_ms", "interval_ms")
TIMINGS = (*CORE_TIMINGS, *WAIT_DETAILS)


def read_rounds(directory: Path) -> list[dict[str, Any]]:
    """Read saved request/response IDs from completed client round metrics."""
    summary = directory / "summary.json"
    if summary.exists():
        saved = json.loads(summary.read_text(encoding="utf-8"))
        rounds = saved["rounds"]
        concurrency = int(saved.get("concurrency", 1))
    else:
        with (directory / "metrics.csv").open(encoding="utf-8", newline="") as source:
            rounds = list(csv.DictReader(source))
        concurrency = 1
    if not rounds:
        raise ValueError("The client run contains no completed rounds")
    seen = set()
    result = []
    for item in rounds:
        row = dict(item)
        row["round"] = int(row["round"])
        if "session_id" in row:
            row["session_id"] = int(row["session_id"])
        if "concurrency" in row or concurrency > 1:
            row["concurrency"] = int(row.get("concurrency", concurrency))
        key = (row.get("session_id", 0), row["round"])
        if row["round"] < 0 or key in seen:
            raise ValueError(f"Invalid or duplicate session/round: {key}")
        if not row.get("request_id"):
            raise ValueError(f"Round {row['round']} has no request_id")
        seen.add(key)
        result.append(row)
    return sorted(result, key=lambda row: (row["round"], row.get("session_id", 0)))


def ttft_seconds(row: dict[str, Any]) -> float:
    """Use client TTFT as measured; CSV blanks are missing, never zero."""
    value = row.get("ttft_s")
    if value is None or value == "":
        return math.nan
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid client TTFT for round {row['round']}")
    return value


def ttft_summary(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute empirical, linearly interpolated quantiles across sessions, never ranks."""
    concurrency = max(
        max(int(row.get("concurrency", 1)) for row in rounds),
        len({row.get("session_id", 0) for row in rounds}),
    )
    result = []
    for turn in sorted({row["round"] for row in rounds}):
        clients = [row for row in rounds if row["round"] == turn]
        values = [ttft_seconds(row) for row in clients]
        values = sorted(value for value in values if math.isfinite(value))
        reported = {}
        for metric in ("prompt_tokens", "cached_tokens", "latency_s"):
            items = [row.get(metric) for row in clients]
            reported[metric] = (
                sum(float(value) for value in items) / len(items)
                if all(value is not None and value != "" for value in items)
                else None
            )

        def percentile(p: float, values: list[float] = values) -> float | None:
            if not values:
                return None
            index = (len(values) - 1) * p
            low, high = math.floor(index), math.ceil(index)
            return values[low] + (values[high] - values[low]) * (index - low)

        result.append(
            dict(
                round=turn,
                concurrency=concurrency,
                ttft_samples=len(values),
                ttft_s=sum(values) / len(values) if values else None,
                ttft_p50_s=percentile(0.5),
                ttft_p99_s=percentile(0.99),
                ttft_p01_s=percentile(0.01),
                ttft_p25_s=percentile(0.25),
                ttft_p75_s=percentile(0.75),
                **reported,
            )
        )
    return result


def draw_ttft(axis: Any, client_rounds: list[dict[str, Any]], turns: list[int]) -> None:
    """Use boxes for concurrent client sessions; keep the serial point curve."""
    summaries = ttft_summary(client_rounds)
    values = {row["round"]: row for row in summaries}
    if summaries[0]["concurrency"] <= 1:
        points = [values[turn]["ttft_s"] if values[turn]["ttft_s"] is not None else math.nan for turn in turns]
        (line,) = axis.plot(turns, points, marker="o", markersize=3.5, color="tab:purple", label="Client TTFT")
        label_points(axis, "ttft_s", turns, points, line.get_color())
        axis.legend(loc="upper right", fontsize="small")
        return
    boxes, positions = [], []
    for index, turn in enumerate(turns):
        row = values[turn]
        if not row["ttft_samples"]:
            continue
        positions.append(turn)
        boxes.append(
            dict(
                label=str(turn),
                mean=row["ttft_s"],
                med=row["ttft_p50_s"],
                q1=row["ttft_p25_s"],
                q3=row["ttft_p75_s"],
                whislo=row["ttft_p01_s"],
                whishi=row["ttft_p99_s"],
                fliers=[],
            )
        )
        axis.annotate(
            f"mean {row['ttft_s']:.2f}\np50 {row['ttft_p50_s']:.2f}\np99 {row['ttft_p99_s']:.2f}"
            f"\nn={row['ttft_samples']}/{row['concurrency']}",
            (turn, row["ttft_p99_s"]),
            xytext=(0, 8 + 20 * (index % 2)),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            annotation_clip=False,
        )
    if boxes:
        axis.bxp(
            boxes,
            positions=positions,
            widths=0.55,
            showmeans=True,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            boxprops=dict(facecolor="lavender", edgecolor="tab:purple"),
            medianprops=dict(color="tab:red"),
            meanprops=dict(marker="D", markerfacecolor="tab:blue", markeredgecolor="tab:blue", markersize=4),
        )
    axis.set_title("Client TTFT across sessions\nBox P25–P75; median P50; whiskers P1–P99; diamond mean", fontsize=10)


def label_points(axis: Any, metric: str, turns: list[int], values: list[float], color: Any) -> None:
    """Label finite points, staggering adjacent labels to keep small changes visible."""
    for index, (turn, value) in enumerate(zip(turns, values)):
        if not math.isfinite(value):
            continue
        if metric == "coverage":
            label = f"{100 * value:.4f}%"
        elif value != 0 and abs(value) < 1:
            label = f"{value:.3g}"
        else:
            label = f"{value:.2f}"
        axis.annotate(
            label,
            (turn, value),
            xytext=(0, 6 if index % 2 == 0 else 17),
            textcoords="offset points",
            ha="left" if index == 0 else "right" if index == len(turns) - 1 else "center",
            va="bottom",
            fontsize=7,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3),
            annotation_clip=False,
        )


def match_round(request_id: str, rounds: list[dict[str, Any]]) -> int | None:
    """Match exact client/response IDs with vLLM prefixes and internal suffixes.

    Never infer rounds from timestamps: timing summaries arrive asynchronously.
    A hyphen boundary prevents round 001 from matching round 0010.
    """
    matched = set()
    for row in rounds:
        for identifier in (row["request_id"], row.get("response_id")):
            if not identifier:
                continue
            for alias in (identifier, "cmpl-" + identifier, "chatcmpl-" + identifier):
                if request_id == alias or request_id.startswith(alias + "-"):
                    matched.add(row["round"])
    if len(matched) > 1:
        raise ValueError(f"Ambiguous round mapping for server request {request_id!r}")
    return next(iter(matched), None)


def aggregate_rounds(
    records: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    *,
    tp_rank: int | None = None,
    dp_rank: int | None = None,
    aggregate_ranks: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Sum batch times per worker/round and average the recorded layer ratios.

    With rank aggregation, timing totals are divided by distinct workers,
    never by chunk/request count. Every request in one batch must map to the
    same client round. Missing measurements remain absent, not zero.
    """
    metadata = {row["round"]: row for row in ttft_summary(rounds)}
    buckets: dict[tuple, dict[str, Any]] = {}
    workers: dict[tuple, set[tuple]] = {}
    chunks = []
    unmatched, mixed, dropped = set(), 0, 0
    for record in records:
        require_bank_reuse_coverage(record)
        if tp_rank is not None and record.get("tp_rank") != tp_rank:
            continue
        if dp_rank is not None and record.get("dp_rank") != dp_rank:
            continue
        requests = record.get("requests", [])
        if not requests:
            mixed += 1
            continue
        matched = {match_round(str(request["req_id"]), rounds) for request in requests}
        if None in matched:
            unmatched.update(
                str(request["req_id"]) for request in requests if match_round(str(request["req_id"]), rounds) is None
            )
            continue
        if len(matched) != 1:
            mixed += 1
            continue
        turn = matched.pop()
        request_id = ",".join(str(request["req_id"]) for request in requests)
        if any(request["token_end"] <= request["token_start"] for request in requests):
            continue
        samples = int(record["layer_samples"])
        coverage = float(record["coverage"])
        times = {name: float(record[name]) for name in CORE_TIMINGS}
        times.update(wait_details(record))
        if samples <= 0 or not all(math.isfinite(v) for v in (coverage, *times.values()) if v is not None):
            raise ValueError(f"Non-finite or invalid measurements for {request_id}")
        dp, tp = (None, None) if aggregate_ranks else (record.get("dp_rank"), record.get("tp_rank"))
        key = (turn, dp, tp)
        worker = (record.get("host"), record.get("pid"), record.get("dp_rank"), record.get("tp_rank"))
        workers.setdefault(key, set()).add(worker)
        row = buckets.setdefault(
            key,
            dict(
                round=turn,
                phase="followup" if turn else "initial",
                request_id=",".join(row["request_id"] for row in rounds if row["round"] == turn),
                response_id=",".join(row.get("response_id") or "" for row in rounds if row["round"] == turn),
                dp_rank=dp,
                tp_rank=tp,
                prompt_tokens=metadata[turn]["prompt_tokens"],
                cached_tokens=metadata[turn]["cached_tokens"],
                ttft_s=metadata[turn].get("ttft_s"),
                ttft_p50_s=metadata[turn]["ttft_p50_s"],
                ttft_p99_s=metadata[turn]["ttft_p99_s"],
                ttft_samples=metadata[turn]["ttft_samples"],
                concurrency=metadata[turn]["concurrency"],
                latency_s=metadata[turn]["latency_s"],
                worker_count=0,
                chunk_records=0,
                layer_samples=0,
                coverage_definition=COVERAGE_DEFINITION,
                coverage=0.0,
                **dict.fromkeys(TIMINGS, 0.0),
            ),
        )
        row["layer_samples"] += samples
        row["coverage"] += coverage * samples
        row["chunk_records"] += 1
        for name in TIMINGS:
            row[name] = None if row[name] is None or times[name] is None else row[name] + times[name]
        chunks.append(
            dict(
                round=turn,
                req_id=request_id,
                host=record.get("host"),
                pid=record.get("pid"),
                dp_rank=record.get("dp_rank"),
                tp_rank=record.get("tp_rank"),
                step=record.get("step"),
                token_start=min(request["token_start"] for request in requests),
                token_end=max(request["token_end"] for request in requests),
                requests=requests,
                layer_samples=samples,
                coverage_definition=COVERAGE_DEFINITION,
                coverage=coverage,
                **times,
            )
        )
        dropped = max(dropped, int(record.get("dropped_steps", 0)))
    rows = []
    for key, row in sorted(buckets.items()):
        row["worker_count"] = len(workers[key])
        row["coverage"] /= row["layer_samples"]
        for name in TIMINGS:
            if row[name] is not None:
                row[name] /= row["worker_count"]
        rows.append(row)
    chunks.sort(key=lambda row: (row["round"], str(row["dp_rank"]), str(row["tp_rank"]), row["token_start"]))
    present = {row["round"] for row in rows}
    diagnostics = dict(
        missing_rounds=sorted(set(metadata) - present),
        unmatched_request_count=len(unmatched),
        unmatched_request_examples=sorted(unmatched)[:5],
        mixed_request_records=mixed,
        dropped_steps=dropped,
        coverage_policy=(
            "Mean of recorded same-bank reuse layer ratios, weighted by layer_samples across chunks/workers."
        ),
        coverage_definition=COVERAGE_DEFINITION,
        timing_policy=(
            "Count each batch once; sum batch times per worker/round, then mean over observed workers; not wall time."
        ),
        first_layer_policy=(
            "First use of each group/bank has no reuse ratio; retain first-layer, initial-bank and all-layer waits."
        ),
        missing_wait_detail_records=sum(any(chunk[name] is None for name in WAIT_DETAILS) for chunk in chunks),
    )
    return rows, chunks, diagnostics


def plot_rounds(
    rows: list[dict[str, Any]],
    rounds: list[int],
    output: Path,
    *,
    client_rounds: list[dict[str, Any]] | None = None,
) -> None:
    """Draw annotated bank metrics and one end-to-end client TTFT curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    panels = (
        ("coverage", "Mean same-bank reuse coverage", "Coverage"),
        ("dma_ms", "Save + load: dma_ms", "ms / worker / round"),
        ("save_ms", "NPU to CPU: save_ms", "ms / worker / round"),
        ("load_ms", "CPU to NPU: load_ms", "ms / worker / round"),
        ("all_wait_ms", "Bank wait: ALL layers, including first", "ms / worker / round"),
        ("first_layer_wait_ms", "First-layer bank wait (sum over chunks)", "ms / worker / round"),
        ("wait_ms", "Wait at same-bank reuse", "ms / worker / round"),
        ("ttft_s", "Client TTFT by round", "seconds / request"),
    )
    width = min(36, max(16, len(rounds) * 0.7))
    figure, axes = plt.subplots(4, 2, figsize=(width, 15), sharex=True, facecolor="white")
    series: dict[tuple, dict[int, dict[str, Any]]] = {}
    for row in rows:
        series.setdefault((row["dp_rank"], row["tp_rank"]), {})[row["round"]] = row
    for (dp, tp), values in series.items():
        label = "Observed workers (mean)" if tp is None else f"DP{dp} TP{tp}"
        for axis, (metric, _, _) in zip(axes.flat, panels):
            if metric == "ttft_s":
                continue
            points = [
                values[turn][metric] if turn in values and values[turn].get(metric) is not None else math.nan
                for turn in rounds
            ]
            (line,) = axis.plot(
                rounds,
                points,
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                label=label,
            )
            label_points(axis, metric, rounds, points, line.get_color())
    # TTFT belongs to the client request, not a rank/chunk. Plot it once and
    # retain completed client rounds even when a server measurement is missing.
    ttft_axis = axes[-1, -1]
    draw_ttft(ttft_axis, client_rounds if client_rounds is not None else rows, rounds)
    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        axis.set_facecolor("white")
        if metric != "ttft_s" or not ttft_axis.get_title():
            axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Follow-up dialogue round")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
        axis.tick_params(axis="x", labelbottom=True)
        axis.grid(alpha=0.25)
        if metric != "coverage":
            upper = axis.dataLim.ymax
            axis.set_ylim(bottom=0, top=upper * (1.6 if metric == "ttft_s" else 1.35) if upper > 0 else 1)
    values = [row["coverage"] for row in rows]
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].set_ylim(min(0.0, min(values, default=0.0) - 0.03), max(1.12, max(values, default=1.0) + 0.12))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=min(8, len(series)))
    figure.suptitle(
        "Multi-turn prefill overlap and timing\n"
        "Times include first-layer wait; absent wait details are left blank.\n"
        "Bank times: ms / worker / round (not wall time). Client TTFT: seconds / request.",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)


def plot_ttft(rounds: list[dict[str, Any]], output: Path) -> None:
    """Plot client TTFT alone, without server records or rank aggregation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    turns = sorted({row["round"] for row in rounds})
    figure, axis = plt.subplots(figsize=(min(36, max(10, len(turns) * 0.7)), 5), facecolor="white")
    axis.set_facecolor("white")
    draw_ttft(axis, rounds, turns)
    if not axis.get_title():
        axis.set_title("Client TTFT by round")
    axis.set_xlabel("Follow-up dialogue round")
    axis.set_ylabel("TTFT (seconds / request)")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    axis.grid(alpha=0.25)
    upper = axis.dataLim.ymax
    axis.set_ylim(bottom=0, top=upper * 1.6 if upper > 0 else 1)
    figure.tight_layout()
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", help="Local P-node service logs or quoted glob patterns")
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Client output directory containing summary.json/metrics.csv"
    )
    parser.add_argument("--output", type=Path, default=Path("multi-turn-overlap.png"))
    parser.add_argument("--ttft-only", action="store_true", help="Plot only client TTFT; no server logs or SSH")
    parser.add_argument("--tp-rank", type=int)
    parser.add_argument("--dp-rank", type=int)
    parser.add_argument("--aggregate-ranks", action="store_true", help="One curve averaging observed workers")
    parser.add_argument(
        "--start-round", type=int, default=1, help="First measured follow-up round; warmup is always excluded"
    )
    parser.add_argument("--end-round", type=int)
    add_collection_arguments(parser)
    args = parser.parse_args(argv)
    try:
        rounds = [
            row
            for row in read_rounds(args.run_dir)
            if row["round"] > 0
            and row["round"] >= args.start_round
            and (args.end_round is None or row["round"] <= args.end_round)
        ]
        if not rounds:
            raise ValueError("No client rounds matched the selected range")
        if args.ttft_only:
            rows = ttft_summary(rounds)
            if not any(row["ttft_s"] is not None for row in rows):
                raise ValueError("No client TTFT values in the selected rounds")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            plot_ttft(rounds, args.output)
            write_csv(rows, args.output.with_suffix(".csv"))
            print(f"[MULTI_TURN_TTFT] rounds={len(rows)} plot={args.output} csv={args.output.with_suffix('.csv')}")
            return 0
        if not args.logs and not args.hosts:
            raise ValueError("Provide local log paths or --hosts for automatic SSH collection, or use --ttft-only")
        paths = sorted({Path(p) for pattern in args.logs for p in (glob.glob(pattern) or [pattern])})
        collected = None
        if args.hosts:
            remote_paths, collected = collect_logs(args)
            paths.extend(remote_paths)
        records, ignored = read_records(paths)
        rows, chunks, diagnostics = aggregate_rounds(
            records,
            rounds,
            tp_rank=args.tp_rank,
            dp_rank=args.dp_rank,
            aggregate_ranks=args.aggregate_ranks,
        )
        if not rows and max(row.get("concurrency", 1) for row in rounds) <= 1:
            raise ValueError(
                "No overlap records matched client request/response IDs and rank filters; "
                "check P-node logs and the stats switch"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, args.output.with_suffix(".csv"))
        write_csv(chunks, args.output.with_suffix(".chunks.csv"))
        diagnostics.update(ignored_records=ignored, matched_rounds=len({row["round"] for row in rows}))
        if collected is not None:
            diagnostics["collection_directory"] = str(collected)
        args.output.with_suffix(".json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        plot_rounds(rows, sorted({row["round"] for row in rounds}), args.output, client_rounds=rounds)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"[MULTI_TURN_OVERLAP] {error}", file=sys.stderr)
        return 2
    print(
        f"[MULTI_TURN_OVERLAP] rounds={diagnostics['matched_rounds']} "
        f"plot={args.output} csv={args.output.with_suffix('.csv')}"
    )
    if diagnostics["missing_rounds"]:
        print(f"[MULTI_TURN_OVERLAP] missing rounds (not zero): {diagnostics['missing_rounds']}", file=sys.stderr)
    if ignored or diagnostics["mixed_request_records"] or diagnostics["dropped_steps"]:
        print(
            f"[MULTI_TURN_OVERLAP] incomplete/skipped measurements; see {args.output.with_suffix('.json')}",
            file=sys.stderr,
        )
    if diagnostics["missing_wait_detail_records"]:
        print(
            "[MULTI_TURN_OVERLAP] missing startup/all-layer wait data; left blank, never treated as zero",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
