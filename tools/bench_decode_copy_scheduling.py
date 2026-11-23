#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time real P->D streaming completions with the profiler off, then capture one batch.

The proxy's target and prefill barrier headers keep each batch on the selected D.
Run on that D host with --capture --profile-dir auto to export the new MindStudio traces.
"""

import argparse
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def receive(lines, started, expected):
    ids = []
    first = last = None
    first_count = 0
    ended = False
    response_id = None
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            ended = True
            break
        message = json.loads(data)
        if message.get("error"):
            raise RuntimeError(str(message["error"]))
        if response_id is not None and message.get("id", response_id) != response_id:
            raise ValueError("request restarted during streaming; discard this batch")
        response_id = message.get("id", response_id)
        for choice in message.get("choices", []):
            tokens = choice.get("token_ids")
            if not tokens:
                continue
            if choice.get("index") != 0 or any(type(t) is not int for t in tokens):
                raise ValueError("unexpected completion token IDs")
            last = time.perf_counter()
            if first is None:
                first, first_count = last, len(tokens)
            ids.extend(tokens)
    if not ended or len(ids) != expected or first_count >= expected:
        raise ValueError(f"need a complete {expected}-token stream, received {len(ids)}, first chunk {first_count}")
    return {
        "first_chunk_tokens": first_count,
        "generated_tokens": len(ids),
        "output_sha256": digest(ids),
        "ttft_ms": 1000 * (first - started),
        "tpot_ms": 1000 * (last - first) / (len(ids) - first_count),
        "e2e_ms": 1000 * (time.perf_counter() - started),
        "response_id": response_id,
    }


def batch(client, prompts, args, target):
    batch_id = uuid.uuid4().hex

    def issue(member):
        payload = {
            "model": args.model,
            "prompt": prompts[member],
            "add_special_tokens": False,
            "max_tokens": args.output_tokens,
            "min_tokens": args.output_tokens,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 1024,
            "stream": True,
            "return_token_ids": True,
        }
        headers = {
            "Content-Type": "application/json",
            "X-VLLM-Profile-Target": target,
            "X-VLLM-Profile-Batch": batch_id,
            "X-VLLM-Profile-Batch-Size": str(len(prompts)),
            "X-VLLM-Profile-Member": str(member),
            "X-VLLM-Profile-Batch-Timeout": str(args.batch_timeout),
        }
        if client.api_key:
            headers["Authorization"] = "Bearer " + client.api_key
        req = Request(client.base_url + "/v1/completions", json.dumps(payload).encode(), headers)
        start = time.perf_counter()
        with urlopen(req, timeout=client.timeout) as response:
            result = receive(response, start, args.output_tokens)
        return {"member": member, **result}

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        sessions = list(pool.map(issue, range(len(prompts))))
    return {
        "batch_ms": 1000 * (time.perf_counter() - start),
        "median_tpot_ms": statistics.median(r["tpot_ms"] for r in sessions),
        "median_ttft_ms": statistics.median(r["ttft_ms"] for r in sessions),
        "median_e2e_ms": statistics.median(r["e2e_ms"] for r in sessions),
        "sessions": sessions,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--tools",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the existing P/D profiling clients",
    )
    cli.add_argument("--base-url", required=True, help="Existing P/D proxy URL")
    cli.add_argument("--target-d-url", required=True, help="D instance to receive the entire batch")
    cli.add_argument("--model", required=True, help="Served model ID")
    cli.add_argument("--concurrency", type=int, default=16)
    cli.add_argument("--prefix-tokens", type=int, default=8192)
    cli.add_argument("--output-tokens", type=int, default=128)
    cli.add_argument("--rounds", type=int, default=3)
    cli.add_argument("--timeout", type=float, default=7200)
    cli.add_argument("--batch-timeout", type=int, default=600)
    cli.add_argument("--output", type=Path, required=True, help="New directory for this server configuration")
    cli.add_argument("--reference", type=Path, help="Unprofiled baseline metrics.json")
    cli.add_argument("--capture", action="store_true", help="Run one additional real batch with D profiling enabled")
    cli.add_argument(
        "--profile-dir", type=Path, help="D worker profiler root or 'auto'; export and analyze only the new ranks"
    )
    args = cli.parse_args()
    if min(args.rounds, args.concurrency, args.prefix_tokens, args.batch_timeout) < 1 or args.output_tokens < 2:
        cli.error("positive rounds/concurrency/prefix/timeout and at least two output tokens required")
    if args.profile_dir and not args.capture:
        cli.error("--profile-dir needs --capture")
    sys.path.insert(0, str(args.tools.resolve()))
    from profile_decode_client import Client, check_proxy, make_prompts, preflight

    key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    proxy, target = Client(args.base_url, args.timeout, key), Client(args.target_d_url, args.timeout, key)
    check_proxy(proxy, target.base_url, args.concurrency)
    if args.capture:
        _, instances = preflight([target], target, args.model, args.prefix_tokens + args.output_tokens)
        if args.profile_dir is not None:
            if not callable(getattr(importlib.import_module("torch_npu.profiler.profiler"), "analyse", None)):
                raise RuntimeError("Run the exporter with the D worker's torch_npu Python environment")

            if str(args.profile_dir) == "auto":
                configured = instances[0].get("profile_dir")
                if not configured or not Path(configured).is_absolute():
                    raise ValueError(
                        f"D reported profiler root {configured!r}. Pass its locally visible absolute --profile-dir."
                    )
                args.profile_dir = Path(configured)
    prompts = make_prompts(target, args.model, args.prefix_tokens, args.concurrency)
    config = {
        "model": args.model,
        "target": target.base_url,
        "proxy": proxy.base_url,
        "output_tokens": args.output_tokens,
        "prompt_hashes": [digest(p) for p in prompts],
    }
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    if reference and config != reference["config"]:
        raise ValueError("baseline route, model, prompts or generation length differ")
    args.output.mkdir(parents=True, exist_ok=False)
    print("[DECODE_BENCH] unprofiled warmup", flush=True)
    batch(proxy, prompts, args, target.base_url)
    rounds = []
    for i in range(args.rounds):
        result = batch(proxy, prompts, args, target.base_url)
        rounds.append(result)
        print(
            "[DECODE_BENCH]",
            json.dumps({"round": i + 1, **{k: v for k, v in result.items() if k != "sessions"}}),
            flush=True,
        )
    metric_names = ("median_tpot_ms", "median_e2e_ms", "median_ttft_ms", "batch_ms")
    medians = {k: statistics.median(r[k] for r in rounds) for k in metric_names}
    result = {"config": config, "rounds": rounds, "medians": medians}
    if reference:
        if len(reference["rounds"]) != len(rounds):
            raise ValueError("use the same number of measurement rounds as the baseline")
        mismatches = [
            (i + 1, a["member"])
            for i, (x, y) in enumerate(zip(rounds, reference["rounds"]))
            for a, b in zip(x["sessions"], y["sessions"])
            if a["output_sha256"] != b["output_sha256"]
        ]
        result["comparison"] = {
            "delta_pct": {k: 100 * (medians[k] / reference["medians"][k] - 1) for k in metric_names},
            "mismatching_outputs_round_and_member": mismatches,
        }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("[DECODE_BENCH]", json.dumps(result.get("comparison", medians)), flush=True)
    if args.capture:
        profile = args.output / "profile"
        command = [
            sys.executable,
            str(args.tools / "profile_decode_client.py"),
            "--base-url",
            proxy.base_url,
            "--target-d-url",
            target.base_url,
            "--model",
            args.model,
            "--concurrency",
            str(args.concurrency),
            "--prefix-tokens",
            str(args.prefix_tokens),
            "--output-tokens",
            str(args.output_tokens),
            "--batch-timeout",
            str(args.batch_timeout),
            "--timeout",
            str(args.timeout),
            "--output",
            str(profile),
        ]
        if args.profile_dir:
            command += ["--profile-dir", str(args.profile_dir.resolve())]
        subprocess.run(command, check=True)
        if args.profile_dir:
            for trace in json.loads((profile / "traces.json").read_text(encoding="utf-8")):
                capture = next(p for p in Path(trace).parents if p.name.endswith("_ascend_pt"))
                basename = profile / capture.name
                with basename.with_suffix(".json").open("w", encoding="utf-8") as out:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("analyze_decode_layer_gaps.py")),
                            trace,
                            "--csv",
                            str(basename.with_suffix(".csv")),
                        ],
                        stdout=out,
                        check=True,
                    )
                print(f"[DECODE_BENCH] layer timing: {basename.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
