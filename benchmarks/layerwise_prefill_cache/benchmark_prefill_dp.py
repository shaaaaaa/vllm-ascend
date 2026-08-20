#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Client companion for ``examples/run_glm51_tp_dp.sh``.

Compare prefill throughput and TTFT at concurrency 1 and 2. Prompts are sent
as token IDs to keep tokenization outside the timed region. Every request asks
for one streamed output token, so TTFT covers request submission, scheduling,
chunked prefill, sampling, and delivery of that token, without running a
separate decode forward.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmark_prefill_p import (
    ClientTiming,
    CompletionClient,
    Prompt,
    make_prompts,
    percentile,
)

DEFAULT_PROMPT_TOKENS = 65536


@dataclass(frozen=True)
class RequestResult:
    batch_index: int
    request_index: int
    request_id: str
    prompt_digest: str
    first_chunk_digest: str
    ttft_seconds: float
    e2e_seconds: float
    prompt_tokens_reported: int | None
    completion_tokens_reported: int | None
    client_send_unix_ns: int
    client_first_token_unix_ns: int
    client_done_unix_ns: int


@dataclass(frozen=True)
class BatchResult:
    batch_index: int
    concurrency: int
    prefill_makespan_seconds: float
    e2e_makespan_seconds: float
    aggregate_prefill_input_tokens_per_second: float
    aggregate_e2e_input_tokens_per_second: float
    requests: list[RequestResult]


def parse_concurrencies(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("concurrencies must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("concurrencies must contain positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("concurrencies must not contain duplicates")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9960")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--model", default="glm51-prefill")
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument(
        "--concurrencies",
        type=parse_concurrencies,
        default=parse_concurrencies("1,2"),
        help="Comma-separated concurrency cases (default: 1,2)",
    )
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5, help="Measured batches per concurrency")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--warmup-seed", type=int, default=10001)
    parser.add_argument("--token-id-min", type=int, default=1000)
    parser.add_argument("--token-id-max", type=int, default=100000)
    parser.add_argument("--cache-chunk-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/lmy/layerwise-prefill-results/prefill-dp.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be positive")
    if args.warmup_batches < 0 or args.repeats <= 0:
        parser.error("--warmup-batches must be >= 0 and --repeats must be > 0")
    if not 0 <= args.token_id_min < args.token_id_max <= 2**32:
        parser.error("invalid token-id range")
    if not 0 < args.cache_chunk_tokens <= args.prompt_tokens:
        parser.error("--cache-chunk-tokens must be in [1, prompt-tokens]")
    if args.seed == args.warmup_seed:
        parser.error("--seed and --warmup-seed must differ")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite")
    return args


def run_one_request(
    *,
    args: argparse.Namespace,
    barrier: threading.Barrier,
    prompt: Prompt,
    batch_index: int,
    request_index: int,
    request_id: str,
) -> RequestResult:
    client = CompletionClient(args.base_url, args.endpoint, args.timeout)
    try:
        barrier.wait(timeout=30)
        timing: ClientTiming = client.run(
            model=args.model,
            prompt=prompt,
            request_id=request_id,
        )
    finally:
        client.close()

    if timing.prompt_tokens_reported not in (None, args.prompt_tokens):
        raise RuntimeError(
            "server reported an unexpected prompt length: "
            f"expected={args.prompt_tokens}, actual={timing.prompt_tokens_reported}"
        )
    return RequestResult(
        batch_index=batch_index,
        request_index=request_index,
        request_id=request_id,
        prompt_digest=prompt.digest,
        first_chunk_digest=prompt.first_chunk_digest,
        ttft_seconds=timing.ttft_seconds,
        e2e_seconds=timing.e2e_seconds,
        prompt_tokens_reported=timing.prompt_tokens_reported,
        completion_tokens_reported=timing.completion_tokens_reported,
        client_send_unix_ns=timing.send_unix_ns,
        client_first_token_unix_ns=timing.first_token_unix_ns,
        client_done_unix_ns=timing.done_unix_ns,
    )


def run_batch(
    *,
    args: argparse.Namespace,
    prompts: list[Prompt],
    concurrency: int,
    batch_index: int,
    phase: str,
) -> BatchResult:
    barrier = threading.Barrier(concurrency + 1)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_one_request,
                args=args,
                barrier=barrier,
                prompt=prompt,
                batch_index=batch_index,
                request_index=request_index,
                request_id=(
                    f"prefill-dp-c{concurrency}-{phase}-b{batch_index}-r{request_index}"
                ),
            )
            for request_index, prompt in enumerate(prompts)
        ]
        barrier.wait(timeout=30)
        requests = [future.result() for future in futures]

    first_send_ns = min(request.client_send_unix_ns for request in requests)
    last_first_token_ns = max(request.client_first_token_unix_ns for request in requests)
    last_done_ns = max(request.client_done_unix_ns for request in requests)
    prefill_makespan_seconds = (last_first_token_ns - first_send_ns) / 1e9
    e2e_makespan_seconds = (last_done_ns - first_send_ns) / 1e9
    total_input_tokens = concurrency * args.prompt_tokens
    return BatchResult(
        batch_index=batch_index,
        concurrency=concurrency,
        prefill_makespan_seconds=prefill_makespan_seconds,
        e2e_makespan_seconds=e2e_makespan_seconds,
        aggregate_prefill_input_tokens_per_second=(
            total_input_tokens / prefill_makespan_seconds
        ),
        aggregate_e2e_input_tokens_per_second=(total_input_tokens / e2e_makespan_seconds),
        requests=requests,
    )


def latency_summary_ms(values_seconds: list[float]) -> dict[str, float]:
    values_ms = [value * 1000 for value in values_seconds]
    return {
        "mean": statistics.fmean(values_ms),
        "p50": percentile(values_ms, 50),
        "p90": percentile(values_ms, 90),
        "min": min(values_ms),
        "max": max(values_ms),
    }


def throughput_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "min": min(values),
        "max": max(values),
    }


def summarize_case(batches: list[BatchResult], prompt_tokens: int) -> dict[str, Any]:
    requests = [request for batch in batches for request in batch.requests]
    total_tokens = len(requests) * prompt_tokens
    return {
        "request_ttft_ms": latency_summary_ms([request.ttft_seconds for request in requests]),
        "request_e2e_ms": latency_summary_ms([request.e2e_seconds for request in requests]),
        "batch_prefill_makespan_ms": latency_summary_ms(
            [batch.prefill_makespan_seconds for batch in batches]
        ),
        "aggregate_prefill_input_tokens_per_second": throughput_summary(
            [batch.aggregate_prefill_input_tokens_per_second for batch in batches]
        ),
        "aggregate_e2e_input_tokens_per_second": (
            total_tokens / sum(batch.e2e_makespan_seconds for batch in batches)
        ),
    }


def print_batch(batch: BatchResult, phase: str) -> None:
    ttft_ms = [request.ttft_seconds * 1000 for request in batch.requests]
    print(
        f"{phase} batch={batch.batch_index + 1} concurrency={batch.concurrency} "
        f"request_ttft_ms=[{', '.join(f'{value:.3f}' for value in ttft_ms)}] "
        f"batch_prefill_ms={batch.prefill_makespan_seconds * 1000:.3f} "
        f"aggregate_input_tps={batch.aggregate_prefill_input_tokens_per_second:.3f}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    seen_first_chunk_digests: set[str] = set()
    cases: list[dict[str, Any]] = []

    for concurrency in args.concurrencies:
        warmup_count = args.warmup_batches * concurrency
        measured_count = args.repeats * concurrency
        seed_offset = concurrency * 1_000_003
        warmup_prompts = make_prompts(
            seed=args.warmup_seed + seed_offset,
            count=warmup_count,
            prompt_tokens=args.prompt_tokens,
            token_id_min=args.token_id_min,
            token_id_max=args.token_id_max,
            cache_chunk_tokens=args.cache_chunk_tokens,
        )
        measured_prompts = make_prompts(
            seed=args.seed + seed_offset,
            count=measured_count,
            prompt_tokens=args.prompt_tokens,
            token_id_min=args.token_id_min,
            token_id_max=args.token_id_max,
            cache_chunk_tokens=args.cache_chunk_tokens,
        )
        all_prompts = [*warmup_prompts, *measured_prompts]
        digests = [prompt.first_chunk_digest for prompt in all_prompts]
        if len(digests) != len(set(digests)) or seen_first_chunk_digests.intersection(digests):
            raise RuntimeError("benchmark prompts contain a duplicate first LMCache chunk")
        seen_first_chunk_digests.update(digests)

        print(
            f"\nConcurrency {concurrency}: prompt_tokens={args.prompt_tokens}, "
            f"warmup_batches={args.warmup_batches}, measured_batches={args.repeats}",
            flush=True,
        )
        for batch_index in range(args.warmup_batches):
            start = batch_index * concurrency
            batch = run_batch(
                args=args,
                prompts=warmup_prompts[start : start + concurrency],
                concurrency=concurrency,
                batch_index=batch_index,
                phase="warmup",
            )
            print_batch(batch, "warmup")

        batches: list[BatchResult] = []
        for batch_index in range(args.repeats):
            start = batch_index * concurrency
            batch = run_batch(
                args=args,
                prompts=measured_prompts[start : start + concurrency],
                concurrency=concurrency,
                batch_index=batch_index,
                phase="measure",
            )
            batches.append(batch)
            print_batch(batch, "measure")

        summary = summarize_case(batches, args.prompt_tokens)
        cases.append(
            {
                "concurrency": concurrency,
                "summary": summary,
                "batches": [asdict(batch) for batch in batches],
            }
        )

    baseline_tps = cases[0]["summary"]["aggregate_prefill_input_tokens_per_second"]["mean"]
    print("\n| concurrency | request TTFT mean ms | request TTFT p50 ms | aggregate input token/s | scaling |")
    print("|---:|---:|---:|---:|---:|")
    for case in cases:
        summary = case["summary"]
        input_tps = summary["aggregate_prefill_input_tokens_per_second"]["mean"]
        print(
            f"| {case['concurrency']} | {summary['request_ttft_ms']['mean']:.3f} | "
            f"{summary['request_ttft_ms']['p50']:.3f} | {input_tps:.3f} | "
            f"{input_tps / baseline_tps:.3f}x |"
        )

    output = {
        "schema_version": 1,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "concurrencies": list(args.concurrencies),
        "warmup_batches": args.warmup_batches,
        "repeats": args.repeats,
        "seed": args.seed,
        "warmup_seed": args.warmup_seed,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
