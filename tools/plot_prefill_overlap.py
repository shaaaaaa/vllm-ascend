#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot P-node [PREFILL_OVERLAP] summaries; no vLLM/NPU imports required.

Example: python3 tools/plot_prefill_overlap.py server.log --output overlap.png
Install matplotlib only on the machine doing offline plotting.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any

MARKER = "[PREFILL_OVERLAP]"


def read_records(paths: list[Path]) -> tuple[list[dict[str, Any]], int]:
    """Read JSON inside prefixed/colored service logs, deduplicating copies."""
    records, seen, ignored = [], set(), 0
    decoder = json.JSONDecoder()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                if MARKER not in line:
                    continue
                try:
                    record, _ = decoder.raw_decode(line.split(MARKER, 1)[1].lstrip())
                except (ValueError, TypeError):
                    ignored += 1
                    continue
                if (
                    record.get("schema_version") != 1
                    or record.get("status") != "complete"
                    or record.get("coverage") is None
                    or not record.get("layer_samples", 0)
                ):
                    ignored += 1
                    continue
                key = json.dumps(record, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    records.append(record)
    return records, ignored


def aggregate_records(
    records: list[dict[str, Any]],
    *,
    request_id: str | None = None,
    tp_rank: int | None = None,
    dp_rank: int | None = None,
    aggregate_ranks: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Average the layer ratios, preserving real token spans and rank identity.

    Only single-request batches are eligible, including when several requests
    happen to have exactly the same scheduled token span.
    """
    buckets: dict[tuple, dict[str, Any]] = {}
    mixed = 0
    for record in records:
        if tp_rank is not None and record.get("tp_rank") != tp_rank:
            continue
        if dp_rank is not None and record.get("dp_rank") != dp_rank:
            continue
        requests = record.get("requests", [])
        if len(requests) != 1:
            mixed += 1
            continue
        if request_id and not any(r["req_id"] == request_id for r in requests):
            continue
        spans = {(r["token_start"], r["token_end"]) for r in requests}
        if len(spans) != 1:
            mixed += 1
            continue
        start, end = spans.pop()
        if end <= start:
            continue
        dp, tp = (
            (None, None)
            if aggregate_ranks
            else (
                record.get("dp_rank"),
                record.get("tp_rank"),
            )
        )
        key = (dp, tp, start, end)
        row = buckets.setdefault(
            key,
            dict(
                dp_rank=dp,
                tp_rank=tp,
                token_start=start,
                token_end=end,
                layer_samples=0,
                worker_steps=0,
                coverage=0.0,
                dma_ms=0.0,
                load_ms=0.0,
                save_ms=0.0,
                wait_ms=0.0,
                interval_ms=0.0,
            ),
        )
        samples = record["layer_samples"]
        row["coverage"] += record["coverage"] * samples
        row["layer_samples"] += samples
        row["worker_steps"] += 1
        for metric in ("dma_ms", "load_ms", "save_ms", "wait_ms", "interval_ms"):
            row[metric] += record[metric]
    rows = list(buckets.values())
    for row in rows:
        row["coverage"] /= row["layer_samples"]
        # Times are totals across worker steps; coverage remains a mean of
        # per-layer ratios and is not recomputed from these aggregate totals.
    rows.sort(
        key=lambda r: (
            r["token_start"],
            r["token_end"],
            r["dp_rank"] if r["dp_rank"] is not None else -1,
            r["tp_rank"] if r["tp_rank"] is not None else -1,
        )
    )
    return rows, mixed


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    """Save the plotted measurements, including all supporting timing totals."""
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, Any]], output: Path) -> None:
    """Render a white-background plot; x labels are inclusive token ranges."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    spans = sorted({(r["token_start"], r["token_end"]) for r in rows})
    span_indices = {span: index for index, span in enumerate(spans)}
    series: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault((row["dp_rank"], row["tp_rank"]), []).append(row)
    figure, axis = plt.subplots(figsize=(min(22, max(10, len(spans) * 0.8)), 6))
    figure.set_facecolor("white")
    axis.set_facecolor("white")
    for (dp, tp), measurements in sorted(series.items()):
        label = "All ranks" if tp is None else f"DP{dp} TP{tp}"
        axis.plot(
            [span_indices[(r["token_start"], r["token_end"])] for r in measurements],
            [r["coverage"] for r in measurements],
            marker="o",
            linewidth=1.6,
            label=label,
        )
    axis.set_xticks(range(len(spans)))
    axis.set_xticklabels([f"{start:,}-{end - 1:,}" for start, end in spans], rotation=35, ha="right")
    axis.set_xlabel("Prefill token range (request-local, inclusive)")
    axis.set_ylabel("Mean layer coverage")
    axis.set_title("Two-bank prefill overlap")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    values = [row["coverage"] for row in rows]
    axis.set_ylim(min(0.0, min(values) - 0.03), max(1.02, max(values) + 0.03))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=min(4, len(series)), fontsize="small")
    figure.tight_layout()
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Service logs or glob patterns")
    parser.add_argument("--output", type=Path, default=Path("prefill-overlap.png"))
    parser.add_argument("--request-id")
    parser.add_argument("--tp-rank", type=int)
    parser.add_argument("--dp-rank", type=int)
    parser.add_argument("--aggregate-ranks", action="store_true", help="One curve averaging all recorded layer ratios")
    args = parser.parse_args(argv)
    try:
        paths = sorted({Path(path) for pattern in args.logs for path in (glob.glob(pattern) or [pattern])})
        records, ignored = read_records(paths)
        rows, mixed = aggregate_records(
            records,
            request_id=args.request_id,
            tp_rank=args.tp_rank,
            dp_rank=args.dp_rank,
            aggregate_ranks=args.aggregate_ranks,
        )
        if not rows:
            raise ValueError("No complete, plottable prefill steps matched the filters")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        csv_path = args.output.with_suffix(".csv")
        write_csv(rows, csv_path)
        plot_rows(rows, args.output)
    except (OSError, ValueError, ImportError) as error:
        print(f"[PREFILL_OVERLAP_PLOT] {error}", file=sys.stderr)
        return 2
    print(
        f"[PREFILL_OVERLAP_PLOT] points={len(rows)} ignored={ignored} "
        f"mixed_span_steps={mixed} plot={args.output} csv={csv_path}"
    )
    if mixed:
        print(
            "Mixed-span batches share waits and are excluded; use an isolated request for its own token-range curve.",
            file=sys.stderr,
        )
    if any(r.get("dropped_steps", 0) for r in records):
        print("WARNING: collector queue overflowed; some steps are missing.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
