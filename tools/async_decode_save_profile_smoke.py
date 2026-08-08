#!/usr/bin/env python3
"""Capture an NPU profile from async decode-save trigger through commit.

The vLLM server must already be running with async decode save and completion
logging enabled, for example::

    LMCACHE_CHUNK_SIZE=256 \\
    LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE=256 \\
    LMCACHE_ASYNC_DECODE_SAVE_LOG_COMPLETIONS=1 \\
    vllm serve ... --profiler-config \\
      '{"profiler":"torch","torch_profiler_dir":"/path/to/profile", \\
        "ignore_frontend":true}'

The profiler starts before the request, so it cannot miss the save trigger.
With the default chunk/window size, the 6143-token raw prompt is one token short
of the first decode-save boundary. Profiling stops only after the matching
``commit_advanced`` event proves that the save completed and advanced the
ordered committed frontier. The script then parses and validates the worker
traces. It intentionally drives one unique request so an unrelated save cannot
satisfy the completion gate.

When ``LMCACHE_ASYNC_DECODE_SAVE`` is unset, LMCache enables it automatically
when decode-window saving is configured and layerwise mode is enabled. The
completion-log switch above is still required because it defaults to disabled.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ASYNC_DECODE_SAVE_LOG_MARKER = "[ASYNC_DECODE_SAVE]"
COMMIT_ADVANCED_EVENT = "commit_advanced"
FRONTEND_PROFILER_ENABLED = "Torch profiler enabled. AsyncLLM CPU traces will be collected under"
DECODE_SAVE_TRACE_MARKERS = (
    "dense_mla_dsa_group_direct_kv_transfer",
    "dense_mla_dsa_batched_direct_kv_transfer",
    "single_layer_kv_transfer_kernel_v2",
    "batched_fused_single_layer_kv_transfer",
)
TRACE_FILE_NAME = "trace_view.json"
TRACE_STABLE_POLLS = 2
TRACE_POLL_INTERVAL_SECONDS = 1.0
LOG_POLL_INTERVAL_SECONDS = 0.05
TRACE_SCAN_CHUNK_BYTES = 8 * 1024 * 1024
MINIMUM_DEFAULT_PROMPT_TOKENS = 6000


class SmokeFailure(RuntimeError):
    """A deterministic async decode-save smoke gate did not pass."""


@dataclass(frozen=True, slots=True)
class DecodeSaveBoundary:
    """The first decode-save range reached after the supplied prompt."""

    start: int
    end: int
    generated_tokens_to_trigger: int


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Host-side observations for one profiled decode-save request."""

    request_id: str
    generated_tokens: int
    boundary: DecodeSaveBoundary
    commit_event: dict[str, Any]
    profile_seconds: float
    trigger_observed_seconds: float | None


class AsyncDecodeSaveLogTail:
    """Read only newly appended structured completion events from a log."""

    def __init__(self, path: Path, *, start_at_end: bool = True) -> None:
        if not path.is_file():
            raise SmokeFailure(f"server log does not exist: {path}")
        self.path = path
        self.offset = path.stat().st_size if start_at_end else 0
        self.partial_line = b""

    def read_events(self) -> list[dict[str, Any]]:
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.partial_line = b""
        with self.path.open("rb") as log_file:
            log_file.seek(self.offset)
            appended = log_file.read()
        self.offset += len(appended)
        if not appended:
            return []

        lines = (self.partial_line + appended).split(b"\n")
        self.partial_line = lines.pop()
        events = []
        for raw_line in lines:
            event = parse_async_decode_save_event(raw_line.decode("utf-8", errors="replace"))
            if event is not None:
                events.append(event)
        return events


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float,
):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    return urllib.request.urlopen(request, timeout=timeout)


def wait_until_ready(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with _request(
                _url(base_url, "/health"),
                timeout=min(5.0, timeout),
            ) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise SmokeFailure(f"server did not become healthy within {timeout:g}s: {last_error}")


def require_worker_only_profiling(server_log: Path) -> None:
    """Reject frontend profiling before it can block the stop endpoint."""
    if not server_log.is_file():
        raise SmokeFailure(f"server log does not exist: {server_log}")
    with server_log.open("r", encoding="utf-8", errors="replace") as log_file:
        if any(FRONTEND_PROFILER_ENABLED in line for line in log_file):
            raise SmokeFailure(
                "AsyncLLM frontend profiling is enabled; restart the server "
                "with profiler config ignore_frontend=true so /stop_profile "
                "only finalizes the TP worker traces"
            )


def profile_control(base_url: str, action: str, timeout: float) -> None:
    with _request(
        _url(base_url, f"/{action}_profile"),
        method="POST",
        timeout=timeout,
    ) as response:
        body = response.read().decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise SmokeFailure(f"{action}_profile returned HTTP {response.status}: {body}")


def calculate_decode_save_boundary(
    prompt_tokens: int,
    chunk_size: int,
    window_size: int,
) -> DecodeSaveBoundary:
    """Mirror LMCache's initial decode-window frontier calculation."""
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if window_size < chunk_size:
        raise ValueError("window_size must be at least chunk_size")
    start = (prompt_tokens // chunk_size) * chunk_size
    end = start + window_size
    generated_tokens_to_trigger = end - prompt_tokens
    if generated_tokens_to_trigger <= 0:
        raise ValueError("the configured prompt is already beyond the calculated decode-save boundary")
    return DecodeSaveBoundary(start, end, generated_tokens_to_trigger)


def default_prompt_tokens(chunk_size: int) -> int:
    """Return the first chunk-boundary-minus-one prompt above 6000 tokens."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    next_boundary = (MINIMUM_DEFAULT_PROMPT_TOKENS // chunk_size + 1) * chunk_size
    return next_boundary - 1


def parse_async_decode_save_event(line: str) -> dict[str, Any] | None:
    """Parse the leading JSON object while tolerating logger source suffixes."""
    marker_index = line.find(ASYNC_DECODE_SAVE_LOG_MARKER)
    if marker_index < 0:
        return None
    json_start = line.find("{", marker_index + len(ASYNC_DECODE_SAVE_LOG_MARKER))
    if json_start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(line[json_start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def matching_commit_event(
    events: Iterable[dict[str, Any]],
    request_id: str,
    target_end: int,
) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") != COMMIT_ADVANCED_EVENT:
            continue
        if event.get("request_id") != request_id:
            continue
        ordered_end = event.get("ordered_committed_end")
        if isinstance(ordered_end, int) and ordered_end >= target_end:
            return event
    return None


def wait_for_commit_event(
    log_tail: AsyncDecodeSaveLogTail,
    request_id: str,
    target_end: int,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = matching_commit_event(
            log_tail.read_events(),
            request_id,
            target_end,
        )
        if event is not None:
            return event
        time.sleep(LOG_POLL_INTERVAL_SECONDS)
    raise SmokeFailure(
        "no matching commit_advanced event within "
        f"{timeout:g}s for request_id={request_id}, target_end={target_end}; "
        "ensure LMCACHE_ASYNC_DECODE_SAVE_LOG_COMPLETIONS=1 and async "
        "decode save are enabled"
    )


def _parse_stream_event(raw_line: bytes) -> dict[str, Any] | None:
    line = raw_line.decode("utf-8", errors="replace").strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        event = json.loads(data)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"invalid completion stream event: {data[:200]!r}") from exc
    if not isinstance(event, dict):
        raise SmokeFailure(f"completion stream event is not an object: {data[:200]!r}")
    return event


def _count_delta_token_ids(event: dict[str, Any]) -> int:
    count = 0
    for choice in event.get("choices", []):
        token_ids = choice.get("token_ids")
        if token_ids is None:
            continue
        if not isinstance(token_ids, list):
            raise SmokeFailure(f"stream token_ids is not a list: {token_ids!r}")
        count += len(token_ids)
    return count


def run_profiled_decode(args: argparse.Namespace) -> ProfileResult:
    boundary = calculate_decode_save_boundary(
        args.prompt_tokens,
        args.chunk_size,
        args.window_size,
    )
    if args.max_tokens < boundary.generated_tokens_to_trigger:
        raise SmokeFailure(
            f"max_tokens={args.max_tokens} cannot reach the first decode-save "
            f"boundary; need at least {boundary.generated_tokens_to_trigger}"
        )

    public_request_id = args.request_id
    engine_request_id = f"cmpl-{public_request_id}"
    payload = {
        "model": args.model,
        "prompt": [args.prompt_token_id] * args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
        "request_id": public_request_id,
    }
    request = urllib.request.Request(
        _url(args.base_url, "/v1/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    log_tail = AsyncDecodeSaveLogTail(args.server_log)
    generated_tokens = 0
    commit_event: dict[str, Any] | None = None
    trigger_observed_at: float | None = None
    profile_started_at: float | None = None
    profile_stopped_at: float | None = None
    profile_start_attempted = False
    profile_stop_attempted = False
    try:
        # Start before dispatch: reacting to a completion or trigger log would
        # necessarily miss the beginning of the device-side store operation.
        profile_start_attempted = True
        profile_control(
            args.base_url,
            "start",
            args.profile_control_timeout,
        )
        profile_started_at = time.monotonic()
        print(
            "profiler started before request "
            f"{engine_request_id}; decode-save target=[{boundary.start}, "
            f"{boundary.end}), trigger after "
            f"{boundary.generated_tokens_to_trigger} generated token(s)",
            flush=True,
        )

        with urllib.request.urlopen(
            request,
            timeout=args.request_timeout,
        ) as response:
            for raw_line in response:
                event = _parse_stream_event(raw_line)
                if event is None:
                    continue
                generated_tokens += _count_delta_token_ids(event)
                if trigger_observed_at is None and generated_tokens >= boundary.generated_tokens_to_trigger:
                    trigger_observed_at = time.monotonic()
                if commit_event is None:
                    commit_event = matching_commit_event(
                        log_tail.read_events(),
                        engine_request_id,
                        boundary.end,
                    )
                if commit_event is not None and not profile_stop_attempted:
                    profile_stop_attempted = True
                    profile_control(
                        args.base_url,
                        "stop",
                        args.profile_control_timeout,
                    )
                    profile_stopped_at = time.monotonic()
                    print(
                        "profiler stopped after matching commit_advanced: "
                        + json.dumps(commit_event, separators=(",", ":")),
                        flush=True,
                    )

        if generated_tokens < boundary.generated_tokens_to_trigger:
            raise SmokeFailure(
                f"completion produced {generated_tokens} token(s), but "
                f"{boundary.generated_tokens_to_trigger} are required to "
                "trigger decode save"
            )
        if commit_event is None:
            commit_event = wait_for_commit_event(
                log_tail,
                engine_request_id,
                boundary.end,
                args.completion_timeout,
            )
        if not profile_stop_attempted:
            profile_stop_attempted = True
            profile_control(
                args.base_url,
                "stop",
                args.profile_control_timeout,
            )
            profile_stopped_at = time.monotonic()
            print(
                "profiler stopped after matching commit_advanced: " + json.dumps(commit_event, separators=(",", ":")),
                flush=True,
            )
    finally:
        if profile_start_attempted and not profile_stop_attempted:
            # torch_npu profiler stop is not safely idempotent. Record the
            # attempt before the request so a lost response is never retried.
            profile_stop_attempted = True
            try:
                profile_control(
                    args.base_url,
                    "stop",
                    args.profile_control_timeout,
                )
            except Exception as exc:  # best-effort cleanup after primary error
                print(f"warning: failed to stop profiler: {exc}", file=sys.stderr)

    if profile_started_at is None or profile_stopped_at is None:
        raise SmokeFailure("profile interval did not complete")
    assert commit_event is not None
    trigger_seconds = None if trigger_observed_at is None else trigger_observed_at - profile_started_at
    return ProfileResult(
        request_id=engine_request_id,
        generated_tokens=generated_tokens,
        boundary=boundary,
        commit_event=commit_event,
        profile_seconds=profile_stopped_at - profile_started_at,
        trigger_observed_seconds=trigger_seconds,
    )


def analyse_profile_data(
    profile_dir: Path,
    expected_ranks: int,
    timeout: float,
) -> None:
    print(
        f"offline parsing profiler data under {profile_dir} (timeout {timeout:g}s)...",
        flush=True,
    )
    command = [
        sys.executable,
        "-c",
        (
            "from torch_npu.profiler.profiler import analyse; "
            "import sys; analyse(sys.argv[1], "
            "max_process_number=int(sys.argv[2]))"
        ),
        str(profile_dir),
        str(expected_ranks),
    ]
    try:
        subprocess.run(command, check=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SmokeFailure(f"offline profiler analysis did not finish within {timeout:g}s: {profile_dir}") from exc
    except subprocess.CalledProcessError as exc:
        raise SmokeFailure(
            f"offline profiler analysis failed with exit status {exc.returncode}: {profile_dir}"
        ) from exc
    print("offline profiler analysis completed", flush=True)


def trace_snapshot(root: Path) -> dict[Path, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        path.resolve(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob(TRACE_FILE_NAME)
        if path.is_file()
    }


def wait_for_new_traces(
    root: Path,
    before: dict[Path, tuple[int, int]],
    *,
    expected_ranks: int,
    timeout: float,
) -> list[Path]:
    deadline = time.monotonic() + timeout
    last_states: dict[Path, tuple[int, int]] = {}
    stable_polls = 0
    newest: list[Path] = []
    while time.monotonic() < deadline:
        current = trace_snapshot(root)
        newest = sorted(path for path, state in current.items() if path not in before or before[path] != state)
        states = {path: current[path] for path in newest}
        if len(newest) >= expected_ranks and states == last_states:
            stable_polls += 1
            if stable_polls >= TRACE_STABLE_POLLS:
                return newest
        else:
            stable_polls = 0
        last_states = states
        time.sleep(TRACE_POLL_INTERVAL_SECONDS)
    raise SmokeFailure(
        f"found {len(newest)} new {TRACE_FILE_NAME} files under {root}; "
        f"expected at least {expected_ranks} stable worker traces within "
        f"{timeout:g}s"
    )


def scan_binary(path: Path, needles: Iterable[str]) -> dict[str, int]:
    encoded = {needle: needle.encode("utf-8") for needle in needles}
    overlap = max(len(value) for value in encoded.values()) - 1
    counts = {needle: 0 for needle in encoded}
    carry = b""
    with path.open("rb") as trace_file:
        while chunk := trace_file.read(TRACE_SCAN_CHUNK_BYTES):
            data = carry + chunk
            for needle, value in encoded.items():
                counts[needle] += data.count(value)
            carry = data[-overlap:] if overlap else b""
    return counts


def check_traces(paths: list[Path]) -> dict[str, int]:
    totals = {marker: 0 for marker in DECODE_SAVE_TRACE_MARKERS}
    for path in paths:
        counts = scan_binary(path, DECODE_SAVE_TRACE_MARKERS)
        for marker, count in counts.items():
            totals[marker] += count
        print(f"trace {path}: " + ", ".join(f"{marker}={counts[marker]}" for marker in DECODE_SAVE_TRACE_MARKERS))
    if not any(totals.values()):
        raise SmokeFailure("none of the new worker traces contains a known decode-save KV-transfer operation")
    print(
        "decode-save transfer inventory: "
        + ", ".join(f"{marker}={totals[marker]}" for marker in DECODE_SAVE_TRACE_MARKERS)
    )
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument(
        "--model",
        default="/workspace/models/GLM-5.1-w4a8",
        help="served model name/path used in the OpenAI request",
    )
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        help=(
            "raw prompt length; default is the first chunk boundary minus "
            "one above 6000 tokens (6143 for chunk_size=256)"
        ),
    )
    parser.add_argument("--prompt-token-id", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--request-id",
        default=f"decode-save-profile-{time.time_ns()}",
        help="unique public completion request id (vLLM adds the cmpl- prefix)",
    )
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--completion-timeout", type=float, default=300)
    parser.add_argument("--profile-control-timeout", type=float, default=300)
    parser.add_argument("--profile-analysis-timeout", type=float, default=900)
    parser.add_argument("--trace-timeout", type=float, default=600)
    args = parser.parse_args()

    if args.expected_ranks <= 0:
        parser.error("--expected-ranks must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.prompt_tokens is None:
        args.prompt_tokens = default_prompt_tokens(args.chunk_size)
    if args.window_size < args.chunk_size:
        parser.error("--window-size must be at least --chunk-size")
    if args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be positive")
    if args.prompt_token_id < 0:
        parser.error("--prompt-token-id must be non-negative")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if not args.request_id:
        parser.error("--request-id must not be empty")
    for name in (
        "ready_timeout",
        "request_timeout",
        "completion_timeout",
        "profile_control_timeout",
        "profile_analysis_timeout",
        "trace_timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        boundary = calculate_decode_save_boundary(
            args.prompt_tokens,
            args.chunk_size,
            args.window_size,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_tokens < boundary.generated_tokens_to_trigger:
        parser.error(f"--max-tokens must be at least {boundary.generated_tokens_to_trigger} for this prompt/config")
    return args


def main() -> int:
    args = parse_args()
    before_traces = trace_snapshot(args.profile_dir)
    try:
        wait_until_ready(args.base_url, args.ready_timeout)
        require_worker_only_profiling(args.server_log)
        result = run_profiled_decode(args)
        analyse_profile_data(
            args.profile_dir,
            expected_ranks=args.expected_ranks,
            timeout=args.profile_analysis_timeout,
        )
        traces = wait_for_new_traces(
            args.profile_dir,
            before_traces,
            expected_ranks=args.expected_ranks,
            timeout=args.trace_timeout,
        )
        check_traces(traces)
    except (SmokeFailure, urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"ASYNC DECODE-SAVE PROFILE SMOKE FAILED: {exc}", file=sys.stderr)
        return 1

    trigger_text = (
        "not observed in the HTTP stream"
        if result.trigger_observed_seconds is None
        else f"{result.trigger_observed_seconds:.6f}s after profile start"
    )
    print(
        "\nASYNC DECODE-SAVE PROFILE CAPTURE PASSED\n"
        f"  request_id: {result.request_id}\n"
        f"  save range: [{result.boundary.start}, {result.boundary.end})\n"
        f"  generated tokens: {result.generated_tokens}\n"
        f"  trigger observed: {trigger_text}\n"
        f"  profiled wall time: {result.profile_seconds:.6f}s\n"
        f"  committed_end: {result.commit_event['ordered_committed_end']}\n"
        f"  profile directory: {args.profile_dir}\n"
        "Open the new trace_view.json files in MindStudio and inspect the "
        "decode-save KV-transfer operation immediately before the matching "
        "completion frontier."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
