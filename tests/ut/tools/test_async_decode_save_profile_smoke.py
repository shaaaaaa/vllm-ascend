# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import async_decode_save_profile_smoke as smoke


def _args(server_log: Path) -> SimpleNamespace:
    return SimpleNamespace(
        base_url="http://127.0.0.1:9000",
        model="model",
        server_log=server_log,
        chunk_size=256,
        window_size=256,
        prompt_tokens=504,
        prompt_token_id=1000,
        profile_after_tokens=2,
        max_tokens=10,
        request_id="profile-test",
        request_timeout=5,
        completion_timeout=5,
        profile_control_timeout=5,
    )


def _commit_line(request_id: str, end: int) -> str:
    payload = {
        "schema": 1,
        "event": "commit_advanced",
        "request_id": request_id,
        "generation": 0,
        "trigger_job_id": 1,
        "committed_job_ids": [1],
        "start": end - 256,
        "end": end,
        "tokens": 256,
        "is_final": False,
        "ordered_committed_end": end,
        "published_committed_end": end,
        "pending_jobs": 0,
    }
    return f"INFO {smoke.ASYNC_DECODE_SAVE_LOG_MARKER} {json.dumps(payload)} (adapter.py:1)\n"


class _StreamingResponse:
    def __init__(
        self,
        server_log: Path,
        commit_line: str,
        *,
        chunks: int = 10,
        commit_after_chunk: int = 8,
    ):
        self.server_log = server_log
        self.commit_line = commit_line
        self.chunks = chunks
        self.commit_after_chunk = commit_after_chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        for chunk in range(1, self.chunks + 1):
            if chunk == self.commit_after_chunk:
                with self.server_log.open("a", encoding="utf-8") as log_file:
                    log_file.write(self.commit_line)
                    log_file.flush()
            yield (b'data: {"choices": [{"text": "x", "token_ids": [' + str(40 + chunk).encode() + b"]}]}\n")
        yield b"data: [DONE]\n"


def test_default_prompt_exceeds_6000_and_leaves_profile_start_lead():
    prompt_tokens = smoke.default_prompt_tokens(256, 256)

    assert prompt_tokens == 6080
    assert prompt_tokens > 6000
    assert smoke.calculate_decode_save_boundary(prompt_tokens, 256, 256).generated_tokens_to_trigger == 64


def test_boundary_is_64_decode_tokens_after_6080_token_prompt():
    assert smoke.calculate_decode_save_boundary(6080, 256, 256) == (
        smoke.DecodeSaveBoundary(
            start=5888,
            end=6144,
            generated_tokens_to_trigger=64,
        )
    )


def test_boundary_respects_chunk_anchor_and_larger_window():
    assert smoke.calculate_decode_save_boundary(511, 256, 512) == (
        smoke.DecodeSaveBoundary(
            start=256,
            end=768,
            generated_tokens_to_trigger=257,
        )
    )


def test_log_tail_handles_fragmented_json_and_logger_suffix(tmp_path: Path):
    server_log = tmp_path / "server.log"
    server_log.write_text("startup\n", encoding="utf-8")
    tail = smoke.AsyncDecodeSaveLogTail(server_log)
    line = _commit_line("cmpl-request", 512)
    split = len(line) // 2

    with server_log.open("a", encoding="utf-8") as log_file:
        log_file.write(line[:split])
        log_file.flush()
    assert tail.read_events() == []

    with server_log.open("a", encoding="utf-8") as log_file:
        log_file.write(line[split:])
        log_file.flush()
    events = tail.read_events()

    assert len(events) == 1
    assert events[0]["event"] == "commit_advanced"
    assert events[0]["request_id"] == "cmpl-request"
    assert events[0]["ordered_committed_end"] == 512


def test_matching_commit_requires_request_and_target_frontier():
    events = [
        {
            "event": "commit_advanced",
            "request_id": "cmpl-other",
            "ordered_committed_end": 1024,
        },
        {
            "event": "commit_advanced",
            "request_id": "cmpl-target-0-12345678",
            "ordered_committed_end": 256,
        },
        {
            "event": "persist_complete",
            "request_id": "cmpl-target-0-12345678",
            "ordered_committed_end": 512,
        },
    ]

    assert smoke.matching_commit_event(events, "cmpl-target-0", 512) is None
    events.append(
        {
            "event": "commit_advanced",
            "request_id": "cmpl-target-0-12345678",
            "ordered_committed_end": 512,
        }
    )
    assert smoke.matching_commit_event(events, "cmpl-target-0", 512) == events[-1]


def test_profile_skips_prefill_and_stops_after_matching_internal_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    server_log = tmp_path / "server.log"
    server_log.write_text("startup\n", encoding="utf-8")
    args = _args(server_log)
    timeline = []
    requests = []

    def profile_control(base_url, action, timeout):
        timeline.append(action)

    def urlopen(request, **kwargs):
        timeline.append("request")
        requests.append(json.loads(request.data))
        return _StreamingResponse(
            server_log,
            _commit_line("cmpl-profile-test-0-12345678", 512),
        )

    monkeypatch.setattr(smoke, "profile_control", profile_control)
    monkeypatch.setattr(smoke.urllib.request, "urlopen", urlopen)

    result = smoke.run_profiled_decode(args)

    assert timeline == ["request", "start", "stop"]
    assert requests == [
        {
            "model": "model",
            "prompt": [1000] * 504,
            "max_tokens": 10,
            "temperature": 0,
            "stream": True,
            "ignore_eos": True,
            "return_token_ids": True,
            "request_id": "profile-test",
        }
    ]
    assert result.request_id == "cmpl-profile-test-0-12345678"
    assert result.generated_tokens == 10
    assert result.commit_event["ordered_committed_end"] == 512
    assert result.trigger_observed_seconds is not None


def test_start_failure_gets_one_cleanup_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    server_log = tmp_path / "server.log"
    server_log.write_text("startup\n", encoding="utf-8")
    actions = []

    def profile_control(base_url, action, timeout):
        actions.append(action)
        if action == "start":
            raise urllib.error.URLError("start response lost")

    monkeypatch.setattr(smoke, "profile_control", profile_control)
    monkeypatch.setattr(
        smoke.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _StreamingResponse(
            server_log,
            _commit_line("cmpl-profile-test-0-12345678", 512),
        ),
    )

    with pytest.raises(urllib.error.URLError, match="start response lost"):
        smoke.run_profiled_decode(_args(server_log))

    assert actions == ["start", "stop"]


def test_stop_failure_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    server_log = tmp_path / "server.log"
    server_log.write_text("startup\n", encoding="utf-8")
    actions = []

    def profile_control(base_url, action, timeout):
        actions.append(action)
        if action == "stop":
            raise urllib.error.URLError("stop response lost")

    monkeypatch.setattr(smoke, "profile_control", profile_control)
    monkeypatch.setattr(
        smoke.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _StreamingResponse(
            server_log,
            _commit_line("cmpl-profile-test-0-12345678", 512),
        ),
    )

    with pytest.raises(urllib.error.URLError, match="stop response lost"):
        smoke.run_profiled_decode(_args(server_log))

    assert actions == ["start", "stop"]


def test_trace_check_accepts_transfer_on_only_one_worker(tmp_path: Path):
    trace_without_save = tmp_path / "rank0" / "trace_view.json"
    trace_with_save = tmp_path / "rank1" / "trace_view.json"
    trace_without_save.parent.mkdir()
    trace_with_save.parent.mkdir()
    trace_without_save.write_text("unrelated op", encoding="utf-8")
    trace_with_save.write_text(
        smoke.DECODE_SAVE_TRACE_MARKERS[0],
        encoding="utf-8",
    )

    totals = smoke.check_traces([trace_without_save, trace_with_save])

    assert totals[smoke.DECODE_SAVE_TRACE_MARKERS[0]] == 1


def test_trace_check_rejects_profile_without_transfer(tmp_path: Path):
    trace = tmp_path / "trace_view.json"
    trace.write_text("unrelated op", encoding="utf-8")

    with pytest.raises(smoke.SmokeFailure, match="KV-transfer"):
        smoke.check_traces([trace])
