# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_prefill_trace as analyzer


def _event(event: str, unix_ns: int, request: str, **fields) -> str:
    payload = {
        "event": event,
        "unix_ns": unix_ns,
        "request": request,
        "pid": 1,
        **fields,
    }
    return f"INFO [PREFILL_TRACE] {json.dumps(payload)}"


def test_parse_and_match_internal_request_suffix(tmp_path: Path) -> None:
    external = "prefill-off-1x-measure-0"
    internal = f"cmpl-{external}-0-deadbeef"
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            (
                _event("api_request_received", 10, f"cmpl-{external}"),
                _event(
                    "frontend_request_id_assigned",
                    20,
                    internal,
                    external_request=f"cmpl-{external}-0",
                ),
                _event(
                    "worker_execute_start",
                    30,
                    internal,
                    rank=0,
                    computed_tokens=0,
                ),
                _event(
                    "worker_execute_start",
                    31,
                    internal,
                    rank=1,
                    computed_tokens=0,
                ),
            )
        ),
        encoding="utf-8",
    )

    events = analyzer.parse_log(log)
    selected = analyzer.request_events(events, external, rank=0)
    assert [event.event for event in selected] == [
        "api_request_received",
        "frontend_request_id_assigned",
        "worker_execute_start",
    ]


def test_load_benchmark_client_points(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "request_id": "prefill-on-1x-measure-0",
                        "ttft_seconds": 1.25,
                        "client_send_unix_ns": 1,
                        "client_response_headers_unix_ns": 2,
                        "client_first_token_unix_ns": 3,
                        "client_done_unix_ns": 4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    requests = analyzer._read_case(path)
    assert requests == [
        analyzer.BenchmarkRequest(
            request_id="prefill-on-1x-measure-0",
            ttft_ms=1250.0,
            send_ns=1,
            response_headers_ns=2,
            first_token_ns=3,
            done_ns=4,
        )
    ]


def test_parse_batched_trace_line(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    payload = [
        {
            "event": "core_schedule_start",
            "unix_ns": 10,
            "request": "cmpl-prefill-test",
        },
        {
            "event": "core_chunk_scheduled",
            "unix_ns": 20,
            "request": "cmpl-prefill-test",
        },
    ]
    log.write_text(
        f"INFO [PREFILL_TRACE] {json.dumps(payload)}\n",
        encoding="utf-8",
    )
    assert [event.event for event in analyzer.parse_log(log)] == [
        "core_schedule_start",
        "core_chunk_scheduled",
    ]
