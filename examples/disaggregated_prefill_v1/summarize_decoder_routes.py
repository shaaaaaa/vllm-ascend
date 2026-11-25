# SPDX-License-Identifier: Apache-2.0
"""Offline, allow-listed A/B report. Never print request IDs, session keys, endpoints or prompts."""

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime

EVENT = re.compile(r"\bROUTE_(RUN|WINDOW|COMPLETE) (\{.*\})\s*$")
COUNTER = re.compile(
    r"(?:decisions|warm_available|warm_chosen|(?:valid|skew|missing)_s|dp:\d+|pair:\d+,\d+|"
    r"reason:(?:legacy|stale|missing_key|warm|load|queue|kv|reserved|escape)|"
    r"key:(?:session|anchor|first_user|missing))$"
)
CLIENT = re.compile(r"✓[^\n]*?TTFT=\s*([\d.]+)ms\s+TPOT=\s*([\d.]+|-)\s*ms")


def timestamp(text):
    try:
        return float(text)
    except ValueError:
        return datetime.fromisoformat(text).timestamp()


def percent(n, d):
    return round(100 * n / d, 2) if d else None


def latency_summary(values):
    values = sorted(v for v in values if math.isfinite(v) and v >= 0)
    return {
        "n": len(values),
        **{
            f"p{p}_ms": round(values[math.ceil(len(values) * p / 100) - 1], 2) if values else None for p in (50, 95, 99)
        },
    }


def benchmark(path):
    """Successful client measurements, including drain. Only consume explicit units."""
    ttft, tpot = [], []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        if path.lower().endswith(".json"):
            obj = json.load(f)
            rows = obj if isinstance(obj, list) else obj.get("results", obj.get("request_results", []))
            for row in rows if isinstance(rows, list) else ():
                if not isinstance(row, dict) or row.get("success") is False or row.get("error"):
                    continue
                for field, values in (("ttft_ms", ttft), ("tpot_ms", tpot)):
                    if isinstance(row.get(field), (int, float)):
                        values.append(row[field])
        else:
            for line in f:
                for first, per_token in CLIENT.findall(line):
                    ttft.append(float(first))
                    if per_token != "-":
                        tpot.append(float(per_token))
    return {"ttft": latency_summary(ttft), "tpot": latency_summary(tpot)}


def summarize(lines, start, end):
    counts, successes = Counter(), Counter()
    policy = None
    restarts = windows = skipped_edges = 0
    for line in lines:
        m = EVENT.search(line)
        if not m:
            continue
        kind, obj = m[1], json.loads(m[2])
        if kind == "RUN" and obj["ts"] < end:
            if policy is not None and obj["ts"] >= start:
                restarts += 1
            if obj["policy"] in {"dual-home", "legacy", "global"}:
                policy = {
                    k: obj.get(k) for k in ("policy", "hosts", "capacity", "kv_limit", "slack", "interval", "max_age")
                }
        elif kind == "WINDOW" and obj["end"] > start and obj["start"] < end:
            if obj["start"] < start or obj["end"] > end:
                skipped_edges += 1
                continue  # no prorating of discrete decisions
            windows += 1
            counts.update({k: v for k, v in obj["counts"].items() if COUNTER.fullmatch(k)})
        elif kind == "COMPLETE" and start <= obj["arrived"] < end:
            successes["in_window" if obj["ts"] < end else "drain"] += 1
    total = counts["decisions"]
    return dict(
        config=policy,
        restarts_in_window=restarts,
        complete_windows=windows,
        excluded_edge_windows=skipped_edges,
        wall_seconds=end - start,
        proxy_success_in_window=successes["in_window"] if policy or windows or successes else None,
        proxy_success_during_drain=successes["drain"] if policy or windows or successes else None,
        window_counts={k: round(v, 3) for k, v in counts.items()},
        key_available_pct=percent(total - counts["key:missing"], total),
        stats_coverage_pct=percent(counts["valid_s"], counts["valid_s"] + counts["missing_s"]),
        host_skew_pct_of_valid=percent(counts["skew_s"], counts["valid_s"]),
        congestion_redirect_pct=percent(
            sum(counts["reason:" + k] for k in ("queue", "kv", "reserved", "escape")), total
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--start", type=timestamp, required=True, help="Epoch or ISO time; include timezone")
    parser.add_argument("--end", type=timestamp, required=True, help="When dispatch stopped, excluding drain")
    parser.add_argument("--bench", help="Single replay's stdout log or JSON results rows with ttft_ms/tpot_ms")
    args = parser.parse_args()
    if not args.start < args.end:
        parser.error("--end must be after --start")
    with open(args.proxy, encoding="utf-8", errors="replace") as f:
        report = summarize(f, args.start, args.end)
    if args.bench:
        report["client_latency_including_drain"] = benchmark(args.bench)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
