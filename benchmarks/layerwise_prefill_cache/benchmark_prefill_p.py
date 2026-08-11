#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure prefill-only latency against a standalone vLLM P node.

The benchmark sends token IDs directly to the OpenAI completions endpoint, so
prompt construction and tokenization are outside the timed region. Each request
asks for exactly one output token. Producing that token uses the logits from the
prefill forward and therefore does not require a separate decode forward.

Warmup and measured prompts come from different deterministic random streams.
The script also verifies that their first cache-chunk hashes are distinct, which
prevents a warmup request from creating an LMCache prefix hit in a measured
request.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import random
import statistics
import time
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class Prompt:
    token_ids: list[int]
    digest: str
    first_chunk_digest: str


@dataclass(frozen=True)
class RequestResult:
    request_index: int
    prompt_digest: str
    first_chunk_digest: str
    ttft_seconds: float
    e2e_seconds: float
    prompt_tokens_reported: int | None
    completion_tokens_reported: int | None


def _digest_token_ids(token_ids: list[int]) -> str:
    return hashlib.sha256(array("I", token_ids).tobytes()).hexdigest()


def make_prompts(
    *,
    seed: int,
    count: int,
    prompt_tokens: int,
    token_id_min: int,
    token_id_max: int,
    cache_chunk_tokens: int,
) -> list[Prompt]:
    rng = random.Random(seed)
    prompts: list[Prompt] = []
    for _ in range(count):
        token_ids = [rng.randrange(token_id_min, token_id_max) for _ in range(prompt_tokens)]
        prompts.append(
            Prompt(
                token_ids=token_ids,
                digest=_digest_token_ids(token_ids),
                first_chunk_digest=_digest_token_ids(token_ids[:cache_chunk_tokens]),
            )
        )
    return prompts


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentage / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


class CompletionClient:
    def __init__(self, base_url: str, endpoint: str, timeout: float) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("--base-url must be an http(s) URL such as http://127.0.0.1:9960")
        connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        self.connection = connection_type(
            parsed.hostname,
            parsed.port,
            timeout=timeout,
        )
        base_path = parsed.path.rstrip("/")
        self.endpoint = f"{base_path}/{endpoint.lstrip('/')}"

    def close(self) -> None:
        self.connection.close()

    def run(
        self,
        *,
        model: str,
        prompt: Prompt,
        request_id: str,
    ) -> tuple[float, float, int | None, int | None]:
        payload = {
            "model": model,
            "prompt": prompt.token_ids,
            "max_tokens": 1,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body = json.dumps(payload, separators=(",", ":"))
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Request-Id": request_id,
        }

        started = time.perf_counter()
        self.connection.request(
            "POST",
            self.endpoint,
            body=body,
            headers=headers,
        )
        response = self.connection.getresponse()
        if response.status != 200:
            error_body = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"request failed with HTTP {response.status}: {error_body}")

        first_token_at: float | None = None
        prompt_tokens_reported: int | None = None
        completion_tokens_reported: int | None = None
        saw_done = False

        while True:
            raw_line = response.readline()
            if not raw_line:
                break
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
                completion_tokens_reported = usage.get("completion_tokens")

        # The SSE [DONE] marker precedes the HTTP chunked-transfer terminator.
        # Consume the response tail before reusing this persistent connection.
        response.read()
        completed = time.perf_counter()
        if not saw_done:
            raise RuntimeError("stream ended without a [DONE] event")
        if first_token_at is None:
            raise RuntimeError("stream completed without a completion-token event")
        return (
            first_token_at - started,
            completed - started,
            prompt_tokens_reported,
            completion_tokens_reported,
        )


def summarize(results: list[RequestResult], prompt_tokens: int) -> dict[str, Any]:
    ttft_ms = [result.ttft_seconds * 1000 for result in results]
    e2e_ms = [result.e2e_seconds * 1000 for result in results]
    total_e2e_seconds = sum(result.e2e_seconds for result in results)

    def latency_summary(values: list[float]) -> dict[str, float]:
        return {
            "mean_ms": statistics.fmean(values),
            "p50_ms": percentile(values, 50),
            "p90_ms": percentile(values, 90),
            "min_ms": min(values),
            "max_ms": max(values),
        }

    return {
        "ttft": latency_summary(ttft_ms),
        "e2e": latency_summary(e2e_ms),
        "aggregate_input_tokens_per_second": (len(results) * prompt_tokens / total_e2e_seconds),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9960")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--model", default="glm51-prefill")
    parser.add_argument("--label", required=True, help="For example: off or on")
    parser.add_argument("--prompt-tokens", type=int, default=65536)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--warmup-seed", type=int, default=10001)
    parser.add_argument("--token-id-min", type=int, default=1000)
    parser.add_argument("--token-id-max", type=int, default=100000)
    parser.add_argument("--cache-chunk-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("--warmups must be >= 0 and --repeats must be > 0")
    if not 0 <= args.token_id_min < args.token_id_max <= 2**32:
        parser.error("invalid token-id range")
    if not 0 < args.cache_chunk_tokens <= args.prompt_tokens:
        parser.error("--cache-chunk-tokens must be in [1, prompt-tokens]")
    if args.warmups and args.seed == args.warmup_seed:
        parser.error("--seed and --warmup-seed must differ")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite")
    return args


def main() -> int:
    args = parse_args()
    print(
        f"Generating {args.warmups} warmup and {args.repeats} measured prompts of {args.prompt_tokens} tokens...",
        flush=True,
    )
    warmup_prompts = make_prompts(
        seed=args.warmup_seed,
        count=args.warmups,
        prompt_tokens=args.prompt_tokens,
        token_id_min=args.token_id_min,
        token_id_max=args.token_id_max,
        cache_chunk_tokens=args.cache_chunk_tokens,
    )
    measured_prompts = make_prompts(
        seed=args.seed,
        count=args.repeats,
        prompt_tokens=args.prompt_tokens,
        token_id_min=args.token_id_min,
        token_id_max=args.token_id_max,
        cache_chunk_tokens=args.cache_chunk_tokens,
    )
    all_prompts = [*warmup_prompts, *measured_prompts]
    first_chunk_digests = [prompt.first_chunk_digest for prompt in all_prompts]
    if len(first_chunk_digests) != len(set(first_chunk_digests)):
        raise RuntimeError("warmup/measured prompts contain a duplicate first cache chunk")

    client = CompletionClient(args.base_url, args.endpoint, args.timeout)
    results: list[RequestResult] = []
    try:
        for index, prompt in enumerate(warmup_prompts):
            ttft, e2e, _, _ = client.run(
                model=args.model,
                prompt=prompt,
                request_id=f"prefill-{args.label}-warmup-{index}",
            )
            print(
                f"warmup {index + 1}/{args.warmups} "
                f"hash={prompt.digest[:16]} ttft={ttft * 1000:.3f} ms "
                f"e2e={e2e * 1000:.3f} ms",
                flush=True,
            )

        for index, prompt in enumerate(measured_prompts):
            ttft, e2e, prompt_count, completion_count = client.run(
                model=args.model,
                prompt=prompt,
                request_id=f"prefill-{args.label}-measure-{index}",
            )
            if prompt_count is not None and prompt_count != args.prompt_tokens:
                raise RuntimeError(
                    f"server reported an unexpected prompt length: expected={args.prompt_tokens}, actual={prompt_count}"
                )
            result = RequestResult(
                request_index=index,
                prompt_digest=prompt.digest,
                first_chunk_digest=prompt.first_chunk_digest,
                ttft_seconds=ttft,
                e2e_seconds=e2e,
                prompt_tokens_reported=prompt_count,
                completion_tokens_reported=completion_count,
            )
            results.append(result)
            print(
                f"measure {index + 1}/{args.repeats} "
                f"hash={prompt.digest[:16]} ttft={ttft * 1000:.3f} ms "
                f"e2e={e2e * 1000:.3f} ms",
                flush=True,
            )
    finally:
        client.close()

    summary = summarize(results, args.prompt_tokens)
    output = {
        "label": args.label,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "warmup_seed": args.warmup_seed,
        "token_id_min": args.token_id_min,
        "token_id_max": args.token_id_max,
        "cache_chunk_tokens": args.cache_chunk_tokens,
        "warmup_prompt_digests": [prompt.digest for prompt in warmup_prompts],
        "measured_prompt_digests": [prompt.digest for prompt in measured_prompts],
        "requests": [asdict(result) for result in results],
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("\nSummary")
    print(
        f"TTFT: mean={summary['ttft']['mean_ms']:.3f} ms "
        f"p50={summary['ttft']['p50_ms']:.3f} ms "
        f"p90={summary['ttft']['p90_ms']:.3f} ms"
    )
    print(
        f"E2E:  mean={summary['e2e']['mean_ms']:.3f} ms "
        f"p50={summary['e2e']['p50_ms']:.3f} ms "
        f"p90={summary['e2e']['p90_ms']:.3f} ms"
    )
    print(f"Input throughput: {summary['aggregate_input_tokens_per_second']:.3f} token/s")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
