#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline benchmark for graph-mode chunked prefill at large prompt lengths.

The public ``run`` command starts one local offline vLLM engine subprocess per
chunk size. It does not start an HTTP server or use disaggregated serving.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROFILE_DIR_ENV = "VLLM_ASCEND_CHUNKED_PREFILL_GRAPH_PROFILE_DIR"


def _prompt_token_ids(
    count: int,
    repetition: int,
    *,
    token_id_min: int,
    token_id_max: int,
) -> list[int]:
    if count <= 0:
        raise ValueError("--prompt-tokens must be positive")
    if token_id_min < 0 or token_id_max <= token_id_min:
        raise ValueError("invalid token-id range")
    span = token_id_max - token_id_min
    offset = (repetition * 104729) % span
    return [token_id_min + ((offset + index * 8191) % span) for index in range(count)]


def _run_offline_case(args: argparse.Namespace) -> None:
    """Run one chunk size directly through the local offline vLLM API."""
    os.environ[PROFILE_DIR_ENV] = str(Path(args.profile_dir).resolve())

    # Import only in the hardware child process. The summarizer and its unit
    # tests remain usable on machines without vLLM/torch_npu installed.
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    additional_config = json.loads(args.additional_config_json)
    if not isinstance(additional_config, dict):
        raise ValueError("--additional-config-json must decode to an object")

    llm = LLM(
        model=args.model,
        skip_tokenizer_init=True,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=1,
        enable_expert_parallel=args.enable_expert_parallel,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.prompt_tokens + 1,
        max_num_batched_tokens=args.chunk_size,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=False,
        compilation_config={
            "cudagraph_mode": "PIECEWISE",
            "cudagraph_capture_sizes": [1, args.chunk_size],
        },
        additional_config=additional_config,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        ignore_eos=True,
        detokenize=False,
    )
    total = args.warmups + args.repeats
    print(
        f"Running {total} sequential offline requests ({args.warmups} warmup, "
        f"{args.repeats} measured), prompt_tokens={args.prompt_tokens}, "
        f"chunk_size={args.chunk_size}, model={args.model}"
    )
    measured: list[float] = []
    for repetition in range(total):
        prompt = TokensPrompt(
            prompt_token_ids=_prompt_token_ids(
                args.prompt_tokens,
                repetition,
                token_id_min=args.token_id_min,
                token_id_max=args.token_id_max,
            )
        )
        start = time.perf_counter()
        outputs = llm.generate(
            prompt,
            sampling_params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one output, got {len(outputs)}")
        phase = "warmup" if repetition < args.warmups else "measure"
        print(f"{phase:>7} {repetition + 1:>2}/{total}: request_id={outputs[0].request_id} wall={elapsed:.3f}s")
        if repetition >= args.warmups:
            measured.append(elapsed)
    case_result = {
        "prompt_tokens": args.prompt_tokens,
        "chunk_size": args.chunk_size,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "wall_seconds": measured,
        "wall_mean_seconds": statistics.fmean(measured),
        "wall_p50_seconds": statistics.median(measured),
    }
    result_path = Path(args.profile_dir) / "offline_case.json"
    result_path.write_text(json.dumps(case_result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Offline wall time: mean={case_result['wall_mean_seconds']:.3f}s, p50={case_result['wall_p50_seconds']:.3f}s"
    )


def _append_case_options(command: list[str], args: argparse.Namespace, chunk_size: int, profile_dir: Path) -> None:
    command.extend(
        [
            "--model",
            args.model,
            "--chunk-size",
            str(chunk_size),
            "--profile-dir",
            str(profile_dir),
            "--prompt-tokens",
            str(args.prompt_tokens),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--seed",
            str(args.seed),
            "--token-id-min",
            str(args.token_id_min),
            "--token-id-max",
            str(args.token_id_max),
            "--additional-config-json",
            args.additional_config_json,
        ]
    )
    if args.quantization is not None:
        command.extend(["--quantization", args.quantization])
    if args.enable_expert_parallel:
        command.append("--enable-expert-parallel")
    command.append("--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code")


def run_offline(args: argparse.Namespace) -> None:
    """Run every chunk size in a fresh local process, then draw the plots."""
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(f"chunked_prefill_graph_{datetime.now():%Y%m%d_%H%M%S}")
    ).resolve()
    if output_dir.exists() and any(output_dir.rglob("*.jsonl")):
        raise FileExistsError(f"{output_dir} already contains profile data; use a new --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve()
    for chunk_size in args.chunk_sizes:
        case_dir = output_dir / f"c{chunk_size}"
        case_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(script), "_run_case"]
        _append_case_options(command, args, chunk_size, case_dir)
        print(f"\n=== chunk_size={chunk_size} ===", flush=True)
        subprocess.run(command, check=True)

    summary = build_summary(
        _read_records([str(output_dir)]),
        prompt_tokens=args.prompt_tokens,
        warmup_requests=args.warmups,
    )
    _write_summary(
        summary,
        output_json=output_dir / "results.json",
        plot_dir=output_dir / "plots",
    )


def _input_files(inputs: list[str]) -> list[Path]:
    files: set[Path] = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.update(path.rglob("*.jsonl"))
        elif path.is_file():
            files.add(path)
        else:
            files.update(Path(match) for match in glob.glob(value, recursive=True))
    result = sorted(path for path in files if path.is_file())
    if not result:
        raise FileNotFoundError(f"No JSONL profile files found in {inputs!r}")
    return result


def _read_records(inputs: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _input_files(inputs):
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                if record.get("schema_version") != 1:
                    raise ValueError(f"Unsupported profile schema in {path}:{line_number}")
                records.append(record)
    return records


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _rank_reduced_steps(records: list[dict[str, Any]], prompt_tokens: int) -> list[dict[str, Any]]:
    rank_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if len(record.get("request_ids", [])) != 1:
            continue
        if int(record["prompt_tokens"][0]) != prompt_tokens:
            continue
        key = (
            int(record["configured_chunk_size"]),
            int(record["dp_rank"]),
            str(record["request_ids"][0]),
            int(record["query_tokens"][0]),
            int(record["context_tokens_before"][0]),
        )
        rank_groups[key].append(record)

    steps: list[dict[str, Any]] = []
    for key, rank_records in rank_groups.items():
        modes = {str(record["cudagraph_mode"]) for record in rank_records}
        if "NONE" in modes:
            raise RuntimeError(f"Eager prefill record found for {key}: {modes}")
        layer_counts = {int(record["num_hidden_layers"]) for record in rank_records}
        if len(layer_counts) != 1:
            raise RuntimeError(f"Ranks disagree on model layer count for {key}")
        configured_chunk, dp_rank, request_id, query, context = key
        steps.append(
            {
                "configured_chunk_size": configured_chunk,
                "dp_rank": dp_rank,
                "request_id": request_id,
                "query_tokens": query,
                "context_tokens": context,
                "prompt_tokens": prompt_tokens,
                "num_hidden_layers": next(iter(layer_counts)),
                "cudagraph_modes": sorted(modes),
                "rank_count": len({int(r["rank"]) for r in rank_records}),
                "capture_delta": max(int(r["graph_capture_count_delta"]) for r in rank_records),
                # The slowest rank determines completion of a TP/EP layer.
                "model_forward_npu_ms": max(float(r["model_forward_npu_ms"]) for r in rank_records),
                "timestamp_ns": min(int(r["timestamp_ns"]) for r in rank_records),
            }
        )
    return steps


def _measured_requests(steps: list[dict[str, Any]], warmup_requests: int) -> dict[int, list[list[dict[str, Any]]]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        grouped[
            (
                int(step["configured_chunk_size"]),
                int(step["dp_rank"]),
                str(step["request_id"]),
            )
        ].append(step)

    by_chunk: dict[int, list[list[dict[str, Any]]]] = defaultdict(list)
    for (chunk_size, _dp_rank, _request_id), request_steps in grouped.items():
        request_steps.sort(key=lambda step: int(step["context_tokens"]))
        expected_context = 0
        for step in request_steps:
            if int(step["context_tokens"]) != expected_context:
                raise RuntimeError(
                    "Request does not contain a contiguous uncached prefill: "
                    f"chunk={chunk_size}, expected_context={expected_context}, "
                    f"actual={step['context_tokens']}"
                )
            expected_context += int(step["query_tokens"])
        if expected_context != int(request_steps[0]["prompt_tokens"]):
            raise RuntimeError(f"Incomplete prefill request for chunk size {chunk_size}: covered={expected_context}")
        by_chunk[chunk_size].append(request_steps)

    measured: dict[int, list[list[dict[str, Any]]]] = {}
    for chunk_size, requests in by_chunk.items():
        requests.sort(key=lambda request: int(request[0]["timestamp_ns"]))
        selected = requests[warmup_requests:]
        if not selected:
            raise RuntimeError(f"No measured requests remain for chunk size {chunk_size}")
        captured = [request for request in selected if any(int(step["capture_delta"]) > 0 for step in request)]
        if captured:
            raise RuntimeError(
                "ACL graph capture occurred in a measured request for chunk "
                f"size {chunk_size}; add another warmup request"
            )
        measured[chunk_size] = selected
    return measured


def build_summary(
    records: list[dict[str, Any]],
    *,
    prompt_tokens: int,
    warmup_requests: int,
) -> dict[str, Any]:
    steps = _rank_reduced_steps(records, prompt_tokens)
    requests_by_chunk = _measured_requests(steps, warmup_requests)
    cases = []
    for chunk_size, requests in sorted(requests_by_chunk.items()):
        num_chunks_set = {len(request) for request in requests}
        if len(num_chunks_set) != 1:
            raise RuntimeError(f"Requests disagree on chunk count for chunk size {chunk_size}")
        num_chunks = next(iter(num_chunks_set))
        num_layers = int(requests[0][0]["num_hidden_layers"])
        curves = []
        all_chunk_times: list[float] = []
        for ordinal in range(num_chunks):
            ordinal_steps = [request[ordinal] for request in requests]
            contexts = {int(step["context_tokens"]) for step in ordinal_steps}
            queries = {int(step["query_tokens"]) for step in ordinal_steps}
            if len(contexts) != 1 or len(queries) != 1:
                raise RuntimeError(f"Chunk shape mismatch at ordinal {ordinal + 1}")
            values = [float(step["model_forward_npu_ms"]) for step in ordinal_steps]
            all_chunk_times.extend(values)
            p50 = statistics.median(values)
            curves.append(
                {
                    "chunk_ordinal": ordinal + 1,
                    "context_tokens_before": next(iter(contexts)),
                    "query_tokens": next(iter(queries)),
                    "model_forward_p50_ms": p50,
                    "model_forward_p90_ms": _percentile(values, 0.9),
                    "average_layer_p50_ms": p50 / num_layers,
                }
            )
        totals = [sum(float(step["model_forward_npu_ms"]) for step in request) for request in requests]
        median_chunk = statistics.median(all_chunk_times)
        cases.append(
            {
                "prompt_tokens": prompt_tokens,
                "configured_chunk_size": chunk_size,
                "num_chunks": num_chunks,
                "num_hidden_layers": num_layers,
                "measured_requests": len(requests),
                "min_rank_count": min(int(step["rank_count"]) for request in requests for step in request),
                "cudagraph_modes": sorted(
                    {mode for request in requests for step in request for mode in step["cudagraph_modes"]}
                ),
                "single_chunk_median_ms": median_chunk,
                "single_chunk_average_layer_median_ms": (median_chunk / num_layers),
                "total_model_forward_p50_ms": statistics.median(totals),
                "total_model_forward_p90_ms": _percentile(totals, 0.9),
                "total_average_layer_p50_ms": (statistics.median(totals) / num_layers),
                "curve": curves,
            }
        )
    if not cases:
        raise RuntimeError(f"No records matched prompt_tokens={prompt_tokens}")
    return {"schema_version": 1, "prompt_tokens": prompt_tokens, "cases": cases}


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        "| chunk | chunks | graph | requests | ranks | median chunk ms | avg layer ms | total p50 ms | total p90 ms |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for case in summary["cases"]:
        print(
            f"| {case['configured_chunk_size']} | {case['num_chunks']} | "
            f"{','.join(case['cudagraph_modes'])} | "
            f"{case['measured_requests']} | {case['min_rank_count']} | "
            f"{case['single_chunk_median_ms']:.3f} | "
            f"{case['single_chunk_average_layer_median_ms']:.4f} | "
            f"{case['total_model_forward_p50_ms']:.3f} | "
            f"{case['total_model_forward_p90_ms']:.3f} |"
        )


def _plot(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install benchmark requirements") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_k = summary["prompt_tokens"] / 1024

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for case in summary["cases"]:
        context_k = [point["context_tokens_before"] / 1024 for point in case["curve"]]
        label = f"chunk={case['configured_chunk_size'] // 1024}K"
        axes[0].plot(
            context_k,
            [point["model_forward_p50_ms"] for point in case["curve"]],
            marker="o",
            label=label,
        )
        axes[1].plot(
            context_k,
            [point["average_layer_p50_ms"] for point in case["curve"]],
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("One chunk model-forward P50 (ms)")
    axes[1].set_ylabel("Average layer P50 (ms)")
    axes[1].set_xlabel("Historical context before chunk (K tokens)")
    axes[0].set_title(f"Graph-mode chunked prefill, prompt={prompt_k:g}K")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    timeline = output_dir / "chunk_timeline.png"
    fig.savefig(timeline, dpi=180)
    plt.close(fig)

    chunks_k = [case["configured_chunk_size"] / 1024 for case in summary["cases"]]
    fig, left = plt.subplots(figsize=(10, 5.5))
    right = left.twinx()
    left.plot(
        chunks_k,
        [case["single_chunk_median_ms"] for case in summary["cases"]],
        color="tab:blue",
        marker="o",
        label="Median one-chunk time",
    )
    right.plot(
        chunks_k,
        [case["total_model_forward_p50_ms"] for case in summary["cases"]],
        color="tab:red",
        marker="s",
        label="Total prefill model-forward P50",
    )
    left.set_xlabel("Configured chunk size (K tokens)")
    left.set_ylabel("Median one-chunk time (ms)", color="tab:blue")
    right.set_ylabel("Total model-forward time (ms)", color="tab:red")
    left.grid(True, alpha=0.3)
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    left.set_title(f"Chunk-size comparison, prompt={prompt_k:g}K")
    fig.tight_layout()
    comparison = output_dir / "chunk_size_comparison.png"
    fig.savefig(comparison, dpi=180)
    plt.close(fig)
    return [timeline, comparison]


def _write_summary(summary: dict[str, Any], *, output_json: Path, plot_dir: Path) -> None:
    _print_summary(summary)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    paths = _plot(summary, plot_dir)
    print(f"Wrote {output_json}", file=sys.stderr)
    for path in paths:
        print(f"Wrote {path}", file=sys.stderr)


def summarize(args: argparse.Namespace) -> None:
    summary = build_summary(
        _read_records(args.input),
        prompt_tokens=args.prompt_tokens,
        warmup_requests=args.warmup_requests,
    )
    _write_summary(
        summary,
        output_json=Path(args.output_json),
        plot_dir=Path(args.plot_dir),
    )


def _add_offline_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=65536)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.93)
    parser.add_argument("--quantization")
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--token-id-min", type=int, default=1000)
    parser.add_argument("--token-id-max", type=int, default=100000)
    parser.add_argument("--additional-config-json", default="{}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="run local offline engines for all chunk sizes and draw plots",
    )
    _add_offline_case_arguments(run)
    run.add_argument("--chunk-sizes", type=int, nargs="+", default=[4096, 8192])
    run.add_argument("--output-dir")
    run.set_defaults(func=run_offline)

    # Internal entry point used so every graph/chunk configuration gets a
    # completely fresh engine and device context.
    case = subparsers.add_parser("_run_case", help=argparse.SUPPRESS)
    _add_offline_case_arguments(case)
    case.add_argument("--chunk-size", type=int, required=True)
    case.add_argument("--profile-dir", required=True)
    case.set_defaults(func=_run_offline_case)

    summary = subparsers.add_parser("summarize", help="merge TP ranks, summarize chunks, and draw plots")
    summary.add_argument("--input", nargs="+", required=True)
    summary.add_argument("--prompt-tokens", type=int, default=65536)
    summary.add_argument("--warmup-requests", type=int, default=1)
    summary.add_argument("--output-json", default="chunked_prefill_graph_results.json")
    summary.add_argument("--plot-dir", default="chunked_prefill_graph_plots")
    summary.set_defaults(func=summarize)

    args = parser.parse_args()
    if getattr(args, "warmups", 0) < 0 or getattr(args, "repeats", 1) <= 0:
        parser.error("warmups must be >= 0 and repeats must be > 0")
    chunk_sizes = getattr(args, "chunk_sizes", [getattr(args, "chunk_size", 1)])
    if any(chunk_size <= 0 for chunk_size in chunk_sizes):
        parser.error("chunk sizes must be positive")
    if getattr(args, "prompt_tokens", 1) <= 0:
        parser.error("--prompt-tokens must be positive")
    if getattr(args, "tensor_parallel_size", 1) <= 0:
        parser.error("--tensor-parallel-size must be positive")
    if getattr(args, "warmup_requests", 0) < 0:
        parser.error("--warmup-requests must be >= 0")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
