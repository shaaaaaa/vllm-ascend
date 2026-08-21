#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture all chunks of one P-node prefill request batch.

Run this once against an ``off`` server and once against an ``on`` server.
Both captures use the same deterministic measured prompt. A distinct prompt is
first run with profiling disabled to warm model/runtime paths without creating
an LMCache prefix hit for the measured request.

The script follows ``staged_sfa_graph_smoke.py``: it controls the worker-only
profiler through ``/start_profile`` and ``/stop_profile``. The worker records
all chunks into one ``all-N`` directory. This script invokes the bounded
``torch_npu.profiler.analyse`` subprocess and waits for one stable
``trace_view.json`` per TP rank.
It deliberately does not aggregate operator times; open the resulting profile
directories in MindStudio Insight.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from array import array
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from urllib.parse import urlparse

PROMPT_TOKENS = 65_536
PROFILE_CHUNK_TOKENS = 2_048
DEFAULT_SEED = 20_260_813
DEFAULT_WARMUP_SEED = 10_001
DEFAULT_CACHE_CHUNK_TOKENS = 256
TRACE_FILE_NAME = "trace_view.json"
TRACE_STABLE_POLLS = 2
TRACE_POLL_INTERVAL_SECONDS = 1.0
FRONTEND_PROFILER_ENABLED = (
    "Torch profiler enabled. AsyncLLM CPU traces will be collected under"
)
DEFERRED_PROFILE_ENABLED = "Deferred full prefill profiler enabled:"
TOPOLOGY_PATTERN = re.compile(r"\bTP=(\d+) DP=(\d+)\b")
DP_WORKER_PATTERN = re.compile(r"\bWorker_DP(\d+)_TP(\d+)_")
TP_WORKER_PATTERN = re.compile(r"\bWorker_TP(\d+)\b")
DEFAULT_EXPECTED_RANKS = 8


class SmokeFailure(RuntimeError):
    """A deterministic prefill profile capture gate failed."""


@dataclass(frozen=True, slots=True)
class Prompt:
    token_ids: list[int]
    digest: str
    first_chunk_digest: str


@dataclass(frozen=True, slots=True)
class RequestTiming:
    ttft_ms: float
    e2e_ms: float
    prompt_tokens_reported: int | None


def _digest_token_ids(token_ids: list[int]) -> str:
    return hashlib.sha256(array("I", token_ids).tobytes()).hexdigest()


def make_prompt(
    seed: int,
    *,
    prompt_tokens: int = PROMPT_TOKENS,
    cache_chunk_tokens: int = DEFAULT_CACHE_CHUNK_TOKENS,
) -> Prompt:
    rng = random.Random(seed)
    token_ids = [rng.randrange(1_000, 100_000) for _ in range(prompt_tokens)]
    return Prompt(
        token_ids=token_ids,
        digest=_digest_token_ids(token_ids),
        first_chunk_digest=_digest_token_ids(token_ids[:cache_chunk_tokens]),
    )


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _request(url: str, *, method: str = "GET", timeout: float):
    request = urllib.request.Request(url, method=method)
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
    raise SmokeFailure(
        f"server did not become healthy within {timeout:g}s: {last_error}"
    )


def require_worker_only_profiling(
    server_log: Path,
) -> None:
    if not server_log.is_file():
        raise SmokeFailure(f"server log does not exist: {server_log}")
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    if FRONTEND_PROFILER_ENABLED in log_text:
        raise SmokeFailure(
            "frontend profiling is enabled; restart with profiler config "
            "ignore_frontend=true so only TP worker traces are captured"
        )
    expected = f"{DEFERRED_PROFILE_ENABLED} chunk_size={PROFILE_CHUNK_TOKENS}"
    if expected not in log_text:
        raise SmokeFailure(
            "server is not configured to capture all prefill chunks; "
            "start it with VLLM_ASCEND_PREFILL_PROFILE_ALL_CHUNKS=true"
        )


def resolve_expected_ranks(
    server_log: Path,
    *,
    concurrency: int,
    configured: int | None,
) -> int:
    if configured is not None:
        return configured
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    matches = TOPOLOGY_PATTERN.findall(log_text)
    if matches:
        tp_size, dp_size = map(int, matches[-1])
        return tp_size * min(dp_size, concurrency)

    dp_workers = {
        (int(dp_rank), int(tp_rank))
        for dp_rank, tp_rank in DP_WORKER_PATTERN.findall(log_text)
    }
    if dp_workers:
        dp_size = max(dp_rank for dp_rank, _ in dp_workers) + 1
        tp_size = max(tp_rank for _, tp_rank in dp_workers) + 1
        return tp_size * min(dp_size, concurrency)

    tp_workers = {int(rank) for rank in TP_WORKER_PATTERN.findall(log_text)}
    if tp_workers:
        return max(tp_workers) + 1

    print(
        "warning: could not infer TP/DP topology from server log; "
        f"using {DEFAULT_EXPECTED_RANKS} expected worker traces",
        file=sys.stderr,
    )
    return DEFAULT_EXPECTED_RANKS


def expected_chunk_markers(
    prompt_tokens: int = PROMPT_TOKENS,
) -> tuple[str, ...]:
    total_chunks = math.ceil(prompt_tokens / PROFILE_CHUNK_TOKENS)
    return tuple(
        "prefill_profile::all::"
        f"chunk_{chunk}_of_{total_chunks}::"
        f"tokens_{(chunk - 1) * PROFILE_CHUNK_TOKENS}_"
        f"{min(chunk * PROFILE_CHUNK_TOKENS, prompt_tokens)}"
        for chunk in range(1, total_chunks + 1)
    )


def scan_binary(path: Path, needles: Iterable[str]) -> dict[str, int]:
    encoded = {needle: needle.encode("utf-8") for needle in needles}
    overlap = max(len(value) for value in encoded.values()) - 1
    counts = {needle: 0 for needle in encoded}
    carry = b""
    with path.open("rb") as trace_file:
        while chunk := trace_file.read(8 * 1024 * 1024):
            data = carry + chunk
            for needle, value in encoded.items():
                counts[needle] += data.count(value)
            carry = data[-overlap:] if overlap else b""
    return counts


def validate_chunk_markers(
    traces: list[Path],
    *,
    expected_ranks: int,
    prompt_tokens: int = PROMPT_TOKENS,
) -> None:
    markers = expected_chunk_markers(prompt_tokens)
    valid_traces = 0
    failures = []
    for trace in traces:
        counts = scan_binary(trace, markers)
        missing = [marker for marker, count in counts.items() if count == 0]
        if missing:
            failures.append(f"{trace}: missing chunk markers {missing}")
            continue
        valid_traces += 1

    if valid_traces < expected_ranks:
        raise SmokeFailure(
            f"only {valid_traces} worker traces contained all "
            f"{len(markers)} chunk markers; expected "
            f"{expected_ranks}\n  " + "\n  ".join(failures)
        )
    print(
        f"validated all {len(markers)} chunk markers in "
        f"{valid_traces} worker traces",
        flush=True,
    )


def profile_control(base_url: str, action: str, timeout: float) -> None:
    with _request(
        _url(base_url, f"/{action}_profile"),
        method="POST",
        timeout=timeout,
    ) as response:
        body = response.read().decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise SmokeFailure(
                f"{action}_profile returned HTTP {response.status}: {body}"
            )


class CompletionClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("--base-url must be an HTTP(S) URL")
        connection_type = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        self.connection = connection_type(
            parsed.hostname,
            parsed.port,
            timeout=timeout,
        )
        self.endpoint = f"{parsed.path.rstrip('/')}/v1/completions"

    def close(self) -> None:
        self.connection.close()

    def run(self, *, model: str, prompt: Prompt, request_id: str) -> RequestTiming:
        payload = {
            "model": model,
            "prompt": prompt.token_ids,
            "max_tokens": 1,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        self.connection.request(
            "POST",
            self.endpoint,
            body=json.dumps(payload, separators=(",", ":")),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "X-Request-Id": request_id,
            },
        )
        response = self.connection.getresponse()
        if response.status != 200:
            body = response.read().decode("utf-8", errors="replace")
            raise SmokeFailure(
                f"completion returned HTTP {response.status}: {body}"
            )

        first_token_at: float | None = None
        prompt_tokens_reported: int | None = None
        saw_done = False
        while raw_line := response.readline():
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            if not data:
                continue
            event = json.loads(data)
            choices = event.get("choices") or []
            if first_token_at is None and any("text" in choice for choice in choices):
                first_token_at = time.perf_counter()
            usage = event.get("usage")
            if usage:
                prompt_tokens_reported = usage.get("prompt_tokens")
        response.read()
        completed = time.perf_counter()
        if not saw_done or first_token_at is None:
            raise SmokeFailure(
                "completion stream did not produce one token followed by [DONE]"
            )
        return RequestTiming(
            ttft_ms=(first_token_at - started) * 1000.0,
            e2e_ms=(completed - started) * 1000.0,
            prompt_tokens_reported=prompt_tokens_reported,
        )


def run_prompt_batch(
    args: argparse.Namespace,
    prompts: list[Prompt],
    *,
    phase: str,
) -> list[RequestTiming]:
    barrier = Barrier(len(prompts) + 1)

    def run_one(index: int, prompt: Prompt) -> RequestTiming:
        client = CompletionClient(args.base_url, args.request_timeout)
        try:
            barrier.wait(timeout=30)
            return client.run(
                model=args.model,
                prompt=prompt,
                request_id=(
                    f"prefill-profile-{args.label}-{phase}-{index}"
                ),
            )
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            executor.submit(run_one, index, prompt)
            for index, prompt in enumerate(prompts)
        ]
        barrier.wait(timeout=30)
        return [future.result() for future in futures]


def analyse_profile_data(
    profile_dir: Path,
    expected_ranks: int,
    timeout: float,
) -> None:
    print(
        f"offline parsing profiler data under {profile_dir} "
        f"(timeout {timeout:g}s)...",
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
        raise SmokeFailure(
            "offline profiler analysis did not finish within "
            f"{timeout:g}s: {profile_dir}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SmokeFailure(
            "offline profiler analysis failed with exit status "
            f"{exc.returncode}: {profile_dir}"
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
        newest = sorted(
            path
            for path, state in current.items()
            if path not in before or before[path] != state
        )
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


def run_capture(args: argparse.Namespace) -> list[Path]:
    warmups = [
        make_prompt(
            args.warmup_seed + index,
            prompt_tokens=args.prompt_tokens,
            cache_chunk_tokens=args.cache_chunk_tokens,
        )
        for index in range(args.concurrency)
    ]
    measured = [
        make_prompt(
            args.seed + index,
            prompt_tokens=args.prompt_tokens,
            cache_chunk_tokens=args.cache_chunk_tokens,
        )
        for index in range(args.concurrency)
    ]
    first_chunk_digests = {
        prompt.first_chunk_digest for prompt in [*warmups, *measured]
    }
    if len(first_chunk_digests) != 2 * args.concurrency:
        raise SmokeFailure(
            "warmup and measured prompts contain a duplicate first LMCache chunk"
        )

    profile_start_attempted = False
    profile_stop_attempted = False
    try:
        print(
            f"warmup: prompt_tokens={args.prompt_tokens}, "
            f"concurrency={args.concurrency}, "
            f"hashes={[prompt.digest[:16] for prompt in warmups]}, "
            "profiler=off",
            flush=True,
        )
        warmup_timings = run_prompt_batch(args, warmups, phase="warmup")
        print(
            "warmup complete: ttft_ms="
            f"{[round(timing.ttft_ms, 3) for timing in warmup_timings]}, "
            "e2e_ms="
            f"{[round(timing.e2e_ms, 3) for timing in warmup_timings]}",
            flush=True,
        )

        total_chunks = math.ceil(args.prompt_tokens / PROFILE_CHUNK_TOKENS)
        profile_dir = args.profile_dir / f"all-{total_chunks}"
        before_traces = trace_snapshot(profile_dir)
        profile_start_attempted = True
        profile_control(
            args.base_url,
            "start",
            args.profile_control_timeout,
        )
        print(
            f"capture: label={args.label}, prompt_tokens={args.prompt_tokens}, "
            f"concurrency={args.concurrency}, "
            f"hashes={[prompt.digest[:16] for prompt in measured]}, "
            f"profiler=all_{total_chunks}_chunks",
            flush=True,
        )
        measured_timings = run_prompt_batch(
            args,
            measured,
            phase="measure",
        )
        profile_stop_attempted = True
        profile_control(
            args.base_url,
            "stop",
            args.profile_control_timeout,
        )
    finally:
        if profile_start_attempted and not profile_stop_attempted:
            profile_stop_attempted = True
            try:
                profile_control(
                    args.base_url,
                    "stop",
                    args.profile_control_timeout,
                )
            except Exception as exc:
                print(f"warning: failed to stop profiler: {exc}", file=sys.stderr)

    for timing in measured_timings:
        if timing.prompt_tokens_reported not in (None, args.prompt_tokens):
            raise SmokeFailure(
                "server reported unexpected prompt length: expected="
                f"{args.prompt_tokens}, actual={timing.prompt_tokens_reported}"
            )
    print(
        "capture requests complete: ttft_ms="
        f"{[round(timing.ttft_ms, 3) for timing in measured_timings]}, "
        "e2e_ms="
        f"{[round(timing.e2e_ms, 3) for timing in measured_timings]}",
        flush=True,
    )
    analyse_profile_data(
        profile_dir,
        expected_ranks=args.expected_ranks,
        timeout=args.profile_analysis_timeout,
    )
    traces = wait_for_new_traces(
        profile_dir,
        before_traces,
        expected_ranks=args.expected_ranks,
        timeout=args.trace_timeout,
    )
    validate_chunk_markers(
        traces,
        expected_ranks=args.expected_ranks,
        prompt_tokens=args.prompt_tokens,
    )
    return traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--label", required=True, choices=("off", "on"))
    parser.add_argument("--base-url", default="http://127.0.0.1:9960")
    parser.add_argument("--model", default="glm51-prefill")
    parser.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-seed", type=int, default=DEFAULT_WARMUP_SEED)
    parser.add_argument(
        "--cache-chunk-tokens",
        type=int,
        default=DEFAULT_CACHE_CHUNK_TOKENS,
    )
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--profile-control-timeout", type=float, default=900)
    parser.add_argument("--profile-analysis-timeout", type=float, default=1800)
    parser.add_argument("--trace-timeout", type=float, default=900)
    args = parser.parse_args()

    if args.expected_ranks is not None and args.expected_ranks <= 0:
        parser.error("--expected-ranks must be positive")
    if args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.seed == args.warmup_seed:
        parser.error("--seed and --warmup-seed must differ")
    if not 0 < args.cache_chunk_tokens <= args.prompt_tokens:
        parser.error("--cache-chunk-tokens must be within the prompt")
    for name in (
        "ready_timeout",
        "request_timeout",
        "profile_control_timeout",
        "profile_analysis_timeout",
        "trace_timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        wait_until_ready(args.base_url, args.ready_timeout)
        require_worker_only_profiling(
            args.server_log,
        )
        args.expected_ranks = resolve_expected_ranks(
            args.server_log,
            concurrency=args.concurrency,
            configured=args.expected_ranks,
        )
        print(
            f"expecting {args.expected_ranks} worker traces for "
            f"concurrency={args.concurrency}",
            flush=True,
        )
        traces = run_capture(args)
    except (
        SmokeFailure,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        print(f"PREFILL PROFILE SMOKE FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nPREFILL PROFILE CAPTURE PASSED")
    total_chunks = math.ceil(args.prompt_tokens / PROFILE_CHUNK_TOKENS)
    print(
        f"label={args.label} prompt_tokens={args.prompt_tokens} "
        f"profile_chunks={total_chunks}"
    )
    profile_dir = args.profile_dir / f"all-{total_chunks}"
    print(f"MindStudio all-chunk profile root: {profile_dir}")
    for trace in traces:
        print(f"trace: {trace}")
    print(
        "Run this script again against the other mode with the default seeds; "
        "then compare both profile roots in MindStudio Insight."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
