#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize PREFILL_TIMING server logs against benchmark request timestamps."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKER_RANK_RE = re.compile(r"Worker_TP(?P<rank>\d+)")
WORKER_CHUNK_RE = re.compile(
    r"worker_chunk rank=(?P<rank>\d+) mode=(?P<mode>\w+) "
    r"request=(?P<request>\S+) chunk=(?P<chunk>\d+)/(?P<chunks>\d+) .* "
    r"gap_ms=(?P<gap>\S+) execute_ms=(?P<execute>[\d.]+) "
    r"start_unix_ns=(?P<start>\d+) end_unix_ns=(?P<end>\d+)"
)
WORKER_SAMPLE_RE = re.compile(
    r"worker_sample rank=(?P<rank>\d+) mode=(?P<mode>\w+) "
    r"request=(?P<request>\S+) .* "
    r"forward_to_sample_ms=(?P<forward_to_sample>[\d.]+) "
    r"sample_ms=(?P<sample>[\d.]+) .* end_unix_ns=(?P<end>\d+)"
)
LMCACHE_FENCE_RE = re.compile(
    r"lmcache_save_fence mode=(?P<mode>\w+) request=(?P<request>\S+) .* "
    r"load_wait_count=(?P<load_count>\d+) load_wait_ms=(?P<load>[\d.]+) "
    r"callback_count=(?P<callback_count>\d+) callback_ms=(?P<callback>[\d.]+) "
    r"active_storers_before=(?P<storers>\d+) "
    r"pending_sync_before_finish=(?P<pending_before>-?\d+) "
    r"pending_sync_after_finish=(?P<pending_after>-?\d+) "
    r"wait_impl_ms=(?P<wait>[\d.]+) finish_batch_ms=(?P<finish>[\d.]+) "
    r"total_ms=(?P<total>[\d.]+)"
)
LMCACHE_LOAD_RE = re.compile(
    r"lmcache_start_load mode=(?P<mode>\w+) request=(?P<request>\S+) .* "
    r"load_tokens=(?P<tokens>\d+) elapsed_ms=(?P<elapsed>[\d.]+)"
)
SERVER_REQUEST_SUFFIX = "-0"


def _engine_request_id(request_id: str) -> str:
    return f"cmpl-{request_id}{SERVER_REQUEST_SUFFIX}"


def load_requests(path: Path) -> dict[str, dict[str, Any]]:
    """Load one case JSON and index measured requests by engine request ID."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "requests" not in payload:
        raise ValueError(f"{path} is an aggregate file; pass a case file such as on-1x.json")
    requests = {}
    for request in payload["requests"]:
        request_id = request.get("request_id")
        if request_id:
            requests[_engine_request_id(request_id)] = request
    if not requests:
        raise ValueError(f"{path} has no request IDs; rerun the updated benchmark first")
    return requests


def parse_log(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Parse per-request, per-rank worker and connector timing fields."""
    parsed: dict[str, dict[int, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "chunks": [],
                "sample": None,
                "fences": [],
                "loads": [],
            }
        )
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[PREFILL_TIMING]" not in line:
            continue
        if match := WORKER_CHUNK_RE.search(line):
            values = match.groupdict()
            parsed[values["request"]][int(values["rank"])]["chunks"].append(values)
            continue
        if match := WORKER_SAMPLE_RE.search(line):
            values = match.groupdict()
            parsed[values["request"]][int(values["rank"])]["sample"] = values
            continue
        if match := LMCACHE_FENCE_RE.search(line):
            values = match.groupdict()
            rank_match = WORKER_RANK_RE.search(line)
            if rank_match is not None:
                parsed[values["request"]][int(rank_match["rank"])]["fences"].append(values)
            continue
        if match := LMCACHE_LOAD_RE.search(line):
            values = match.groupdict()
            rank_match = WORKER_RANK_RE.search(line)
            if rank_match is not None:
                parsed[values["request"]][int(rank_match["rank"])]["loads"].append(values)
    return parsed


def _sum(items: list[dict[str, str]], field: str) -> float:
    return sum(float(item[field]) for item in items)


def summarize(
    requests: dict[str, dict[str, Any]],
    parsed: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build one critical-rank timing summary per measured request."""
    rows = []
    for request_id, request in requests.items():
        ranks = parsed.get(request_id, {})
        if not ranks:
            rows.append({"request": request_id, "error": "no server timing logs"})
            continue
        critical_rank, fields = max(
            ranks.items(),
            key=lambda item: max(
                (int(chunk["end"]) for chunk in item[1]["chunks"]),
                default=0,
            ),
        )
        chunks = fields["chunks"]
        if not chunks:
            rows.append({"request": request_id, "error": "no worker chunks"})
            continue
        chunks.sort(key=lambda item: int(item["chunk"]))
        first_start = int(chunks[0]["start"])
        client_start = request.get("client_start_unix_ns")
        first_token = request.get("first_token_unix_ns")
        sample = fields["sample"]
        fences = fields["fences"]
        loads = fields["loads"]
        sample_end = int(sample["end"]) if sample is not None else None
        sample_end_to_first_token_ms = (
            (int(first_token) - sample_end) / 1_000_000
            if first_token is not None and sample_end is not None
            else None
        )
        client_to_first_chunk_ms = (
            (first_start - int(client_start)) / 1_000_000
            if client_start is not None
            else None
        )
        worker_execute_ms = _sum(chunks, "execute")
        chunk_gap_ms = sum(
            float(chunk["gap"])
            for chunk in chunks
            if chunk["gap"] != "first"
        )
        last_chunk_to_sample_ms = (
            float(sample["forward_to_sample"])
            if sample is not None
            else None
        )
        sample_ms = float(sample["sample"]) if sample is not None else None
        reconstructed_parts = (
            client_to_first_chunk_ms,
            worker_execute_ms,
            chunk_gap_ms,
            last_chunk_to_sample_ms,
            sample_ms,
            sample_end_to_first_token_ms,
        )
        reconstructed_ttft_ms = (
            sum(reconstructed_parts)
            if all(part is not None for part in reconstructed_parts)
            else None
        )
        ttft_ms = float(request["ttft_seconds"]) * 1000
        rows.append(
            {
                "request": request_id,
                "rank": critical_rank,
                "chunks": len(chunks),
                "ttft_ms": ttft_ms,
                "client_to_first_chunk_ms": client_to_first_chunk_ms,
                # execute_model includes model compute and connector work that
                # runs inside the worker call.  Do not label it as pure kernel
                # time; that distinction comes from the device profile.
                "worker_execute_ms": worker_execute_ms,
                "chunk_gap_ms": chunk_gap_ms,
                "last_chunk_to_sample_ms": last_chunk_to_sample_ms,
                "sample_ms": sample_ms,
                "sample_end_to_first_token_ms": sample_end_to_first_token_ms,
                "reconstructed_ttft_ms": reconstructed_ttft_ms,
                "timeline_residual_ms": (
                    ttft_ms - reconstructed_ttft_ms
                    if reconstructed_ttft_ms is not None
                    else None
                ),
                "lmcache_layer_load_wait_count": sum(
                    int(fence["load_count"]) for fence in fences
                ),
                "lmcache_layer_load_wait_ms": _sum(fences, "load"),
                "lmcache_callback_count": sum(
                    int(fence["callback_count"]) for fence in fences
                ),
                "lmcache_callback_ms": _sum(fences, "callback"),
                "lmcache_max_active_storers": max(
                    (int(fence["storers"]) for fence in fences),
                    default=0,
                ),
                "lmcache_max_pending_sync_before_finish": max(
                    (int(fence["pending_before"]) for fence in fences),
                    default=0,
                ),
                "lmcache_max_pending_sync_after_finish": max(
                    (int(fence["pending_after"]) for fence in fences),
                    default=0,
                ),
                "lmcache_start_load_ms": _sum(loads, "elapsed"),
                "lmcache_max_load_tokens": max(
                    (int(load["tokens"]) for load in loads),
                    default=0,
                ),
                "lmcache_wait_impl_ms": _sum(fences, "wait"),
                "lmcache_finish_batch_ms": _sum(fences, "finish"),
                "lmcache_fence_total_ms": _sum(fences, "total"),
            }
        )
    return rows


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_rows(rows: list[dict[str, Any]]) -> None:
    """Print the compact critical-path table."""
    critical_path_columns = (
        "request",
        "rank",
        "ttft_ms",
        "client_to_first_chunk_ms",
        "worker_execute_ms",
        "chunk_gap_ms",
        "last_chunk_to_sample_ms",
        "sample_ms",
        "sample_end_to_first_token_ms",
        "reconstructed_ttft_ms",
        "timeline_residual_ms",
    )
    lmcache_columns = (
        "request",
        "rank",
        "lmcache_start_load_ms",
        "lmcache_max_load_tokens",
        "lmcache_layer_load_wait_count",
        "lmcache_layer_load_wait_ms",
        "lmcache_callback_count",
        "lmcache_callback_ms",
        "lmcache_max_active_storers",
        "lmcache_max_pending_sync_before_finish",
        "lmcache_max_pending_sync_after_finish",
        "lmcache_wait_impl_ms",
        "lmcache_finish_batch_ms",
        "lmcache_fence_total_ms",
    )
    for title, columns in (
        ("Critical path", critical_path_columns),
        ("LMCache phases (subsets of worker_execute_ms)", lmcache_columns),
    ):
        print(title)
        print("| " + " | ".join(columns) + " |")
        print("|" + "|".join("---" for _ in columns) + "|")
        for row in rows:
            if "error" in row:
                print(f"{row['request']}: {row['error']}")
                continue
            print(
                "| "
                + " | ".join(_display(row.get(key)) for key in columns)
                + " |"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_rows(summarize(load_requests(args.benchmark_json), parse_log(args.server_log)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
