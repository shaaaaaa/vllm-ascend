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
import math
import random
import statistics
import time
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

BASE_PROMPT_TOKENS = 65536
ON_PROMPT_MULTIPLIERS = (1, 2, 4)
DEFAULT_CHUNK_SIZE = 8192
OFF_TIMEOUT_SECONDS = 300.0


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


def summarize(
    results: list[RequestResult],
    prompt_tokens: int,
    chunk_size: int,
) -> dict[str, Any]:
    ttft_ms = [result.ttft_seconds * 1000 for result in results]
    e2e_ms = [result.e2e_seconds * 1000 for result in results]
    total_e2e_seconds = sum(result.e2e_seconds for result in results)
    num_chunks = math.ceil(prompt_tokens / chunk_size)
    average_chunk_ttft_ms = [value / num_chunks for value in ttft_ms]

    def latency_summary(values: list[float]) -> dict[str, float]:
        return {
            "mean_ms": statistics.fmean(values),
            "p50_ms": percentile(values, 50),
            "p90_ms": percentile(values, 90),
            "min_ms": min(values),
            "max_ms": max(values),
        }

    return {
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "average_chunk_ttft": latency_summary(average_chunk_ttft_ms),
        "ttft": latency_summary(ttft_ms),
        "e2e": latency_summary(e2e_ms),
        "aggregate_input_tokens_per_second": (len(results) * prompt_tokens / total_e2e_seconds),
    }


def prompt_multipliers(label: str) -> tuple[int, ...]:
    return ON_PROMPT_MULTIPLIERS if label == "on" else (1,)


def case_output_path(output: Path, multiplier: int) -> Path:
    suffix = output.suffix or ".json"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}-{multiplier}x{suffix}")


def case_seed(seed: int, label: str, multiplier: int) -> int:
    label_offset = 10_000_019 if label == "on" else 0
    return seed + label_offset + multiplier * 1_000_003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9960")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--model", default="glm51-prefill")
    parser.add_argument("--label", required=True, choices=("off", "on"))
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=BASE_PROMPT_TOKENS,
        help=(
            f"Base prompt length (default: {BASE_PROMPT_TOKENS}); on runs "
            f"multipliers {','.join(map(str, ON_PROMPT_MULTIPLIERS))}"
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--warmup-seed", type=int, default=10001)
    parser.add_argument("--token-id-min", type=int, default=1000)
    parser.add_argument("--token-id-max", type=int, default=100000)
    parser.add_argument("--cache-chunk-tokens", type=int, default=256)
    parser.add_argument(
        "--timeout",
        type=float,
        help=(
            f"Defaults to {OFF_TIMEOUT_SECONDS:g}s for off and "
            f"{OFF_TIMEOUT_SECONDS * max(ON_PROMPT_MULTIPLIERS):g}s for on"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.timeout is None:
        timeout_multiplier = max(prompt_multipliers(args.label))
        args.timeout = OFF_TIMEOUT_SECONDS * timeout_multiplier

    if args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("--warmups must be >= 0 and --repeats must be > 0")
    if not 0 <= args.token_id_min < args.token_id_max <= 2**32:
        parser.error("invalid token-id range")
    if not 0 < args.cache_chunk_tokens <= args.prompt_tokens:
        parser.error("--cache-chunk-tokens must be in [1, prompt-tokens]")
    if args.warmups and args.seed == args.warmup_seed:
        parser.error("--seed and --warmup-seed must differ")
    return args


def run_case(
    *,
    args: argparse.Namespace,
    client: CompletionClient,
    multiplier: int,
    seen_first_chunk_digests: set[str],
) -> dict[str, Any]:
    prompt_tokens = args.prompt_tokens * multiplier
    measured_seed = case_seed(args.seed, args.label, multiplier)
    warmup_seed = case_seed(args.warmup_seed, args.label, multiplier)
    print(
        f"\n[{multiplier}x] Generating {args.warmups} warmup and "
        f"{args.repeats} measured prompts of {prompt_tokens} tokens...",
        flush=True,
    )
    warmup_prompts = make_prompts(
        seed=warmup_seed,
        count=args.warmups,
        prompt_tokens=prompt_tokens,
        token_id_min=args.token_id_min,
        token_id_max=args.token_id_max,
        cache_chunk_tokens=args.cache_chunk_tokens,
    )
    measured_prompts = make_prompts(
        seed=measured_seed,
        count=args.repeats,
        prompt_tokens=prompt_tokens,
        token_id_min=args.token_id_min,
        token_id_max=args.token_id_max,
        cache_chunk_tokens=args.cache_chunk_tokens,
    )
    all_prompts = [*warmup_prompts, *measured_prompts]
    first_chunk_digests = [prompt.first_chunk_digest for prompt in all_prompts]
    if len(first_chunk_digests) != len(set(first_chunk_digests)):
        raise RuntimeError("warmup/measured prompts contain a duplicate first cache chunk")
    duplicate_digests = seen_first_chunk_digests.intersection(first_chunk_digests)
    if duplicate_digests:
        raise RuntimeError("prompt-length cases contain a duplicate first cache chunk")
    seen_first_chunk_digests.update(first_chunk_digests)

    results: list[RequestResult] = []
    for index, prompt in enumerate(warmup_prompts):
        ttft, e2e, _, _ = client.run(
            model=args.model,
            prompt=prompt,
            request_id=f"prefill-{args.label}-{multiplier}x-warmup-{index}",
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
            request_id=f"prefill-{args.label}-{multiplier}x-measure-{index}",
        )
        if prompt_count is not None and prompt_count != prompt_tokens:
            raise RuntimeError(
                f"server reported an unexpected prompt length: expected={prompt_tokens}, actual={prompt_count}"
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

    summary = summarize(results, prompt_tokens, args.chunk_size)
    return {
        "schema_version": 2,
        "label": args.label,
        "multiplier": multiplier,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "model": args.model,
        "base_prompt_tokens": args.prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "chunk_size": args.chunk_size,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": measured_seed,
        "warmup_seed": warmup_seed,
        "token_id_min": args.token_id_min,
        "token_id_max": args.token_id_max,
        "cache_chunk_tokens": args.cache_chunk_tokens,
        "warmup_prompt_digests": [prompt.digest for prompt in warmup_prompts],
        "measured_prompt_digests": [prompt.digest for prompt in measured_prompts],
        "requests": [asdict(result) for result in results],
        "summary": summary,
    }


def print_case_summary(output: dict[str, Any], output_path: Path) -> None:
    summary = output["summary"]
    print(f"\nSummary {output['multiplier']}x: prompt={output['prompt_tokens']} chunks={summary['num_chunks']}")
    print(
        "Average chunk TTFT: "
        f"mean={summary['average_chunk_ttft']['mean_ms']:.3f} ms "
        f"p50={summary['average_chunk_ttft']['p50_ms']:.3f} ms "
        f"p90={summary['average_chunk_ttft']['p90_ms']:.3f} ms"
    )
    print(
        f"Total TTFT: mean={summary['ttft']['mean_ms']:.3f} ms "
        f"p50={summary['ttft']['p50_ms']:.3f} ms "
        f"p90={summary['ttft']['p90_ms']:.3f} ms"
    )
    print(
        f"E2E:  mean={summary['e2e']['mean_ms']:.3f} ms "
        f"p50={summary['e2e']['p50_ms']:.3f} ms "
        f"p90={summary['e2e']['p90_ms']:.3f} ms"
    )
    print(f"Input throughput: {summary['aggregate_input_tokens_per_second']:.3f} token/s")
    print(f"Wrote {output_path}")


def main() -> int:
    args = parse_args()
    multipliers = prompt_multipliers(args.label)
    multiple_cases = len(multipliers) > 1
    case_paths = [
        case_output_path(args.output, multiplier) if multiple_cases else args.output for multiplier in multipliers
    ]
    all_output_paths = [*case_paths, *([args.output] if multiple_cases else [])]
    existing = [path for path in all_output_paths if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists: {paths}; pass --overwrite")

    client = CompletionClient(args.base_url, args.endpoint, args.timeout)
    case_outputs: list[dict[str, Any]] = []
    seen_first_chunk_digests: set[str] = set()
    try:
        for multiplier, output_path in zip(multipliers, case_paths, strict=True):
            output = run_case(
                args=args,
                client=client,
                multiplier=multiplier,
                seen_first_chunk_digests=seen_first_chunk_digests,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(output, indent=2) + "\n",
                encoding="utf-8",
            )
            print_case_summary(output, output_path)
            case_outputs.append(output)
    finally:
        client.close()

    if multiple_cases:
        aggregate = {
            "schema_version": 2,
            "label": args.label,
            "base_prompt_tokens": args.prompt_tokens,
            "chunk_size": args.chunk_size,
            "multipliers": list(multipliers),
            "case_outputs": [str(path) for path in case_paths],
            "cases": [
                {
                    "multiplier": output["multiplier"],
                    "prompt_tokens": output["prompt_tokens"],
                    "summary": output["summary"],
                }
                for output in case_outputs
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(aggregate, indent=2) + "\n",
            encoding="utf-8",
        )
        print("\n| multiplier | prompt tokens | chunks | avg chunk mean ms | total TTFT mean ms |")
        print("|---:|---:|---:|---:|---:|")
        for output in case_outputs:
            summary = output["summary"]
            print(
                f"| {output['multiplier']}x | {output['prompt_tokens']} | "
                f"{summary['num_chunks']} | "
                f"{summary['average_chunk_ttft']['mean_ms']:.3f} | "
                f"{summary['ttft']['mean_ms']:.3f} |"
            )
        print(f"Wrote aggregate summary {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
