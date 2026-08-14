#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct prefill TTFT from host-side point events.

The report subtracts adjacent timestamps. It does not time Python callbacks,
wait functions, or accelerator events and therefore does not assign a device
synchronization to whichever Python function happened to observe it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG_PREFIX = "[PREFILL_TRACE] "


@dataclass(frozen=True)
class TraceEvent:
    event: str
    unix_ns: int
    request: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkRequest:
    request_id: str
    ttft_ms: float
    send_ns: int
    response_headers_ns: int
    first_token_ns: int
    done_ns: int


def _read_case(path: Path) -> list[BenchmarkRequest]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "requests" not in data:
        requests = []
        for case_path in data.get("case_outputs", ()):
            resolved = Path(case_path)
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            requests.extend(_read_case(resolved))
        return requests

    requests = []
    for item in data["requests"]:
        requests.append(
            BenchmarkRequest(
                request_id=item["request_id"],
                ttft_ms=float(item["ttft_seconds"]) * 1000,
                send_ns=int(item["client_send_unix_ns"]),
                response_headers_ns=int(item["client_response_headers_unix_ns"]),
                first_token_ns=int(item["client_first_token_unix_ns"]),
                done_ns=int(item["client_done_unix_ns"]),
            )
        )
    return requests


def parse_log(path: Path) -> list[TraceEvent]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = line.find(LOG_PREFIX)
        if marker < 0:
            continue
        payload_text = line[marker + len(LOG_PREFIX) :].strip()
        try:
            decoded = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        payloads = decoded if isinstance(decoded, list) else [decoded]
        for raw_payload in payloads:
            if not isinstance(raw_payload, dict):
                continue
            payload = dict(raw_payload)
            event = payload.pop("event", None)
            unix_ns = payload.pop("unix_ns", None)
            request = payload.pop("request", None)
            if not isinstance(event, str) or not isinstance(request, str):
                continue
            if not isinstance(unix_ns, int):
                continue
            events.append(TraceEvent(event, unix_ns, request, payload))
    return sorted(events, key=lambda item: item.unix_ns)


def _belongs_to(event: TraceEvent, request_id: str) -> bool:
    base = f"cmpl-{request_id}"
    candidates = (
        event.request,
        event.fields.get("external_request"),
        event.fields.get("engine_request"),
    )
    return any(
        isinstance(candidate, str) and (candidate == base or candidate.startswith(f"{base}-"))
        for candidate in candidates
    )


def request_events(
    events: Iterable[TraceEvent],
    request_id: str,
    rank: int,
) -> list[TraceEvent]:
    selected = []
    for event in events:
        if not _belongs_to(event, request_id):
            continue
        event_rank = event.fields.get("rank")
        if event_rank is not None and int(event_rank) != rank:
            continue
        selected.append(event)
    return selected


def _first(events: Iterable[TraceEvent], name: str) -> int | None:
    return next((event.unix_ns for event in events if event.event == name), None)


def _last(events: Iterable[TraceEvent], name: str) -> int | None:
    return next(
        (event.unix_ns for event in reversed(tuple(events)) if event.event == name),
        None,
    )


def _ms(end_ns: int | None, start_ns: int | None) -> float | None:
    if end_ns is None or start_ns is None:
        return None
    return (end_ns - start_ns) / 1_000_000


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _event_map(events: Iterable[TraceEvent], name: str) -> dict[int, TraceEvent]:
    return {
        int(event.fields["computed_tokens"]): event
        for event in events
        if event.event == name and "computed_tokens" in event.fields
    }


def _print_request_summary(
    request: BenchmarkRequest,
    events: list[TraceEvent],
) -> None:
    api_received = _first(events, "api_request_received")
    api_ready = _first(events, "api_handler_ready")
    render_start = _first(events, "serving_render_start")
    render_done = _first(events, "serving_render_done")
    assigned = _first(events, "frontend_request_id_assigned")
    submit_start = _first(events, "frontend_core_submit_start")
    submit_return = _first(events, "frontend_core_submit_return")
    core_received = _first(events, "core_request_received")
    core_queued = _first(events, "core_request_queued")
    first_schedule = _first(events, "core_schedule_start")
    last_update = _last(events, "core_scheduler_update_done")
    core_output = _first(events, "core_output_enqueued")
    frontend_output = _first(events, "frontend_engine_output_received")
    request_output = _first(events, "frontend_request_output_ready")
    serving_output = _first(events, "serving_engine_output_received")
    sse_ready = _first(events, "serving_first_sse_ready")

    wall_ttft = _ms(request.first_token_ns, request.send_ns)
    print(f"\nRequest {request.request_id}")
    print(f"Client TTFT: {request.ttft_ms:.3f} ms; wall-clock reconstruction: {_fmt(wall_ttft)} ms")
    phases = (
        ("client send -> API body parsed", request.send_ns, api_received),
        ("API handler", api_received, api_ready),
        ("prompt render (inside API handler)", render_start, render_done),
        ("handler ready -> internal ID assigned", api_ready, assigned),
        ("internal ID assigned -> submit start", assigned, submit_start),
        ("submit call", submit_start, submit_return),
        ("submit start -> EngineCore receive", submit_start, core_received),
        ("EngineCore receive -> scheduler queued", core_received, core_queued),
        ("scheduler queue", core_queued, first_schedule),
        ("final scheduler update -> core output queue", last_update, core_output),
        ("core output queue -> frontend receive", core_output, frontend_output),
        ("frontend output processing", frontend_output, request_output),
        ("frontend generator -> serving", request_output, serving_output),
        ("serving output -> first SSE ready", serving_output, sse_ready),
        ("first SSE ready -> client first token", sse_ready, request.first_token_ns),
        ("API body parsed -> first SSE ready", api_received, sse_ready),
        (
            "client send -> response headers",
            request.send_ns,
            request.response_headers_ns,
        ),
        ("client first token -> DONE", request.first_token_ns, request.done_ns),
    )
    print("| phase | ms |")
    print("|---|---:|")
    for phase, start_ns, end_ns in phases:
        print(f"| {phase} | {_fmt(_ms(end_ns, start_ns))} |")


def _print_chunks(request: BenchmarkRequest, events: list[TraceEvent]) -> None:
    event_names = (
        "core_schedule_start",
        "core_chunk_scheduled",
        "core_model_dispatch_start",
        "core_model_dispatched",
        "worker_execute_start",
        "worker_execute_return",
        "worker_sampling_start",
        "worker_sampling_return",
        "core_model_result_ready",
        "core_scheduler_update_start",
        "core_scheduler_update_done",
    )
    maps = {name: _event_map(events, name) for name in event_names}
    schedule_starts = maps["core_schedule_start"]
    scheduled = maps["core_chunk_scheduled"]
    starts = sorted(scheduled)
    print("\nChunk host-side boundaries")
    print(
        "| chunk | tokens | schedule start from client ms | scheduler call ms | "
        "schedule-to-worker ms | worker execute call ms | worker sample call ms | "
        "dispatch call ms | dispatch-to-result ms | scheduler update ms | "
        "next chunk gap ms |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, computed in enumerate(starts):
        schedule = scheduled[computed]
        schedule_start = schedule_starts.get(computed)
        dispatch_start = maps["core_model_dispatch_start"].get(computed)
        dispatched = maps["core_model_dispatched"].get(computed)
        worker_start = maps["worker_execute_start"].get(computed)
        worker_return = maps["worker_execute_return"].get(computed)
        sample_start = maps["worker_sampling_start"].get(computed)
        sample_return = maps["worker_sampling_return"].get(computed)
        result = maps["core_model_result_ready"].get(computed)
        update_start = maps["core_scheduler_update_start"].get(computed)
        update_done = maps["core_scheduler_update_done"].get(computed)
        next_schedule_start = schedule_starts.get(starts[index + 1]) if index + 1 < len(starts) else None
        scheduled_tokens = int(schedule.fields["scheduled_tokens"])
        worker_start_ns = worker_start.unix_ns if worker_start else None
        worker_return_ns = worker_return.unix_ns if worker_return else None
        sample_start_ns = sample_start.unix_ns if sample_start else None
        sample_return_ns = sample_return.unix_ns if sample_return else None
        dispatch_start_ns = dispatch_start.unix_ns if dispatch_start else None
        dispatched_ns = dispatched.unix_ns if dispatched else None
        result_ns = result.unix_ns if result else None
        update_start_ns = update_start.unix_ns if update_start else None
        update_done_ns = update_done.unix_ns if update_done else None
        schedule_start_ns = schedule_start.unix_ns if schedule_start else None
        next_schedule_start_ns = next_schedule_start.unix_ns if next_schedule_start else None
        print(
            f"| {schedule.fields['chunk']} | "
            f"[{computed},{computed + scheduled_tokens}) | "
            f"{_fmt(_ms(schedule_start_ns, request.send_ns))} | "
            f"{_fmt(_ms(schedule.unix_ns, schedule_start_ns))} | "
            f"{_fmt(_ms(worker_start_ns, schedule.unix_ns))} | "
            f"{_fmt(_ms(worker_return_ns, worker_start_ns))} | "
            f"{_fmt(_ms(sample_return_ns, sample_start_ns))} | "
            f"{_fmt(_ms(dispatched_ns, dispatch_start_ns))} | "
            f"{_fmt(_ms(result_ns, dispatch_start_ns))} | "
            f"{_fmt(_ms(update_done_ns, update_start_ns))} | "
            f"{_fmt(_ms(next_schedule_start_ns, update_done_ns))} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_requests = _read_case(args.benchmark_json)
    log_events = parse_log(args.server_log)
    if not benchmark_requests:
        raise RuntimeError("benchmark JSON contains no measured requests")
    if not log_events:
        raise RuntimeError("server log contains no PREFILL_TRACE points; restart with PREFILL_TRACE=true")

    for request in benchmark_requests:
        events = request_events(log_events, request.request_id, args.rank)
        if not events:
            print(f"\nRequest {request.request_id}: no matching server points")
            continue
        _print_request_summary(request, events)
        _print_chunks(request, events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
