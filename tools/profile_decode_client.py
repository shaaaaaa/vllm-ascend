#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture the entire decode request on existing vLLM Ascend D instances.

Send requests through the existing P/D proxy with --base-url and pass each
possible decoder's direct API address with --d-url. The proxy starts profiling
on all listed decoders after every P request in the batch is ready, then submits
the batch to D. Profiling stops after the full response. This includes D-side
KV loading and every generation step. The default request has
exactly 50,000 input tokens and generates 128 output tokens.

Run inside the D container and pass --profile-dir to export newly created
captures into MindStudio trace_view.json files. Without a locally accessible
profile directory, captures remain in each server's configured profiler path.
The D servers need --profiler-config with profiler="torch" enabled.
The updated P/D proxy returns actual per-request routes for this client. Each
selected decoder's address and configured server-side profile directory are
printed and saved in result.json, including any recompute route changes.

For 16 concurrent sessions on one D instance, pass --concurrency 16 and
--target-d-url. The original proxy chooses P and performs the existing KV
handoff, then releases the batch to D after all P requests are ready. The proxy
must use one API process for this optional in-memory batch barrier. --warmup
runs the same batch once before profiling. Sessions have distinct fixed prompts.

Example command
  python tools/profile_decode_client.py --base-url http://proxy:9000 \
      --target-d-url http://decoder:7920 --concurrency 16 --warmup \
      --profile-dir /path/to/D_profiles

The HTTP client and trace exporter use the neighboring profile_prefill_client
module. Trace export runs after all profilers have stopped.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

from profile_prefill_client import Client, analyse_runs, profile_runs, repeat_tokens, write_json

PREFIX = "[DECODE_PROFILE]"


def preflight(decoders, tokenizer, model, required_tokens):
    paths = tokenizer.request("/openapi.json").get("paths", {})
    if "/tokenize" not in paths:
        raise RuntimeError("The tokenizer API needs /tokenize; pass a vLLM instance with --tokenizer-url")
    models = tokenizer.request("/v1/models").get("data", [])
    if model is None:
        if len(models) != 1:
            raise ValueError("Use --model to select a served model")
        model = models[0]["id"]
    instances = []
    for client in decoders:
        instance = {
            "url": client.base_url,
            "profile_dir": None,
            "dp_rank": None,
        }
        paths = client.request("/openapi.json").get("paths", {})
        missing = {"/start_profile", "/stop_profile"} - paths.keys()
        if missing:
            raise RuntimeError(
                f"{client.base_url} is missing {sorted(missing)}; enable --profiler-config on this D instance"
            )
        models = client.request("/v1/models").get("data", [])
        selected = next((item for item in models if item["id"] == model), None)
        if selected is None:
            raise ValueError(f"Model {model!r} is absent from {client.base_url}")
        maximum = selected.get("max_model_len")
        if maximum is not None and maximum < required_tokens:
            raise ValueError(f"{client.base_url} max_model_len={maximum}, required at least {required_tokens}")
        if "/server_info" in paths:
            config = client.request("/server_info?config_format=json").get("vllm_config", {})
            profiler = config.get("profiler_config") or {}
            if profiler.get("profiler") != "torch":
                raise ValueError(f"{client.base_url} requires profiler='torch' for MindStudio traces")
            for field in ("delay_iterations", "max_iterations", "warmup_iterations", "wait_iterations"):
                if profiler.get(field, 0):
                    raise ValueError(f"Set {field}=0 on {client.base_url} to capture the whole decode request")
            role = (config.get("kv_transfer_config") or {}).get("kv_role")
            if role == "kv_producer":
                raise ValueError(f"{client.base_url} is a P instance; --d-url needs a decoder")
            instance["profile_dir"] = profiler.get("torch_profiler_dir")
            parallel = config.get("parallel_config") or {}
            instance["dp_rank"] = parallel.get("data_parallel_rank")
            instance["dp_size"] = parallel.get("data_parallel_size", 1)
            instance["external_lb"] = parallel.get("data_parallel_external_lb", False)
            instance["max_num_seqs"] = (config.get("scheduler_config") or {}).get("max_num_seqs")
        instances.append(instance)
    return model, instances


def make_prompt(client, model, count):
    # A fixed seed text makes token IDs reproducible with the same tokenizer.
    result = client.request(
        "/tokenize",
        {
            "model": model,
            "prompt": (
                "The archive records measurements of rivers, mountains, cities and forests. "
                "Read the records carefully and summarize their patterns.\n"
            ),
            "add_special_tokens": False,
        },
    )
    return repeat_tokens(result["tokens"], count)


def measured_request(client, model, prompt, output_tokens, root, headers=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "add_special_tokens": False,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "seed": 1024,
        "stream": False,
    }
    write_json(root / "request.json", payload)
    started = time.perf_counter()
    response = client.request("/v1/completions", payload, headers={"X-VLLM-Profile-Routes": "1", **(headers or {})})
    elapsed = time.perf_counter() - started
    write_json(root / "response.json", response)
    choices = response.get("choices", [])
    usage = response.get("usage") or {}
    if response.get("error"):
        raise RuntimeError(f"Completion failed: {response['error']}")
    if len(choices) != 1 or choices[0].get("finish_reason") not in {"length", "stop"}:
        raise RuntimeError("Expected one completed response; see response.json")
    if usage.get("prompt_tokens") != len(prompt) or usage.get("completion_tokens") != output_tokens:
        raise RuntimeError(f"Expected {len(prompt)} prompt and {output_tokens} output tokens, received {usage}")
    root.joinpath("output.txt").write_text(choices[0].get("text") or "", encoding="utf-8")
    return {
        "response_id": response.get("id"),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "request_seconds": elapsed,
        "finish_reason": choices[0]["finish_reason"],
        "routes": response.get("vllm_profile_routes", []),
    }


def report_routes(measurement, instances, report):
    routes = measurement["routes"]
    if not routes:
        report["routing_status"] = "proxy_metadata_unavailable"
        print(
            f"{PREFIX} proxy route metadata unavailable; update the P/D proxy to report the executed instance",
            flush=True,
        )
        return
    instance_by_url = {instance["url"]: instance for instance in instances}
    resolved = []
    printed_destinations = set()
    for route in routes:
        decoder_url = str(route["decoder_url"]).rstrip("/").removesuffix("/v1")
        instance = instance_by_url.get(decoder_url)
        profile_dir = instance["profile_dir"] if instance else None
        dp_rank = route.get("decoder_dp_rank")
        if dp_rank is None and instance:
            dp_rank = instance["dp_rank"]
        trace_pattern = (
            profile_dir.rstrip("/") + "/*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json" if profile_dir else None
        )
        resolved.append(
            {
                **route,
                "decoder_dp_rank": dp_rank,
                "profile_dir": profile_dir,
                "trace_path_pattern": trace_pattern,
                "listed_for_profiling": instance is not None,
            }
        )
        destination = decoder_url, dp_rank
        if destination not in printed_destinations:
            printed_destinations.add(destination)
            print(
                f"{PREFIX} executed decoder={decoder_url}, dp_rank={dp_rank}, "
                f"server_profile_dir={profile_dir or 'unavailable'}",
                flush=True,
            )
            if trace_pattern:
                print(f"{PREFIX} MindStudio path on {urlsplit(decoder_url).netloc}: {trace_pattern}", flush=True)
    report["routing_status"] = "reported_by_proxy"
    report["executed_routes"] = resolved


def check_proxy(client, target, concurrency):
    capabilities = client.request("/profile_routes")
    if capabilities.get("version", 0) < 3 or not capabilities.get("deferred_profile_start"):
        raise ValueError("Update and restart the P/D proxy to start profiling after prefill completes")
    if not capabilities.get("prefill_batch_barrier"):
        raise ValueError("The P/D proxy must support the profiling prefill barrier")
    if target is not None and not capabilities.get("target_decoder"):
        raise ValueError("The P/D proxy must support a target decoder")
    if target is not None and target not in capabilities.get("decoder_urls", []):
        raise ValueError(f"Target {target!r} is absent from the proxy decoder list")


def make_prompts(client, model, count, concurrency):
    base = make_prompt(client, model, count)
    if concurrency == 1:
        return [base]
    prompts = []
    for index in range(concurrency):
        marker = client.request(
            "/tokenize",
            {"model": model, "prompt": f"Decode profiling session {index}.\n", "add_special_tokens": False},
        )["tokens"]
        repeat_tokens(marker, 1)
        if len(marker) >= count:
            raise ValueError("Concurrent prompts must be longer than their session markers")
        prompts.append(marker + base[len(marker) :])
    if len({tuple(prompt) for prompt in prompts}) != concurrency:
        raise ValueError("Tokenizer produced identical prompts for distinct session markers")
    return prompts


def run_requests(client, model, prompts, output_tokens, root, target, gate_timeout, profile_urls=None):
    concurrency = len(prompts)
    batch_id = uuid.uuid4().hex
    directories = []
    for index in range(concurrency):
        directory = root if concurrency == 1 else root / f"session-{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        directories.append(directory)

    def request(index):
        headers = {}
        if target:
            headers["X-VLLM-Profile-Target"] = target
        if concurrency > 1 or profile_urls:
            headers.update(
                {
                    "X-VLLM-Profile-Batch": batch_id,
                    "X-VLLM-Profile-Batch-Size": str(concurrency),
                    "X-VLLM-Profile-Member": str(index),
                    "X-VLLM-Profile-Batch-Timeout": str(gate_timeout),
                }
            )
        if profile_urls:
            headers["X-VLLM-Profile-Start"] = json.dumps(profile_urls)
        result = measured_request(client, model, prompts[index], output_tokens, directories[index], headers)
        result["session"] = index
        if target and (
            not result["routes"]
            or any(str(route["decoder_url"]).rstrip("/").removesuffix("/v1") != target for route in result["routes"])
        ):
            raise RuntimeError(f"Session {index} did not report the required target {target}")
        write_json(directories[index] / "measurement.json", result)
        return result

    if concurrency == 1:
        return [request(0)]
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(request, index): index for index in range(concurrency)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                index = futures[future]
                errors.append({"session": index, "error": str(error)})
                write_json(directories[index] / "error.json", errors[-1])
    if errors:
        write_json(root / "failures.json", errors)
        raise RuntimeError(f"{len(errors)} of {concurrency} requests failed; see failures.json")
    return sorted(results, key=lambda item: item["session"])


def profile_request(decoders, client, model, prompts, output_tokens, root, report, target=None, gate_timeout=600):
    request_error = None
    try:
        print(f"{PREFIX} proxy will start D profiling after every P request is ready", flush=True)
        measurements = run_requests(
            client, model, prompts, output_tokens, root, target, gate_timeout,
            profile_urls=[decoder.base_url for decoder in decoders],
        )
        starts = [
            route["profile_start"]
            for measurement in measurements
            for route in measurement["routes"]
            if "profile_start" in route
        ]
        if not starts:
            raise RuntimeError("The P/D proxy did not confirm starting profiling after prefill")
        report["profile_start"] = starts[0]
        report["profile_started_at_ns"] = starts[0]["ready_at_ns"]
        return measurements
    except BaseException as error:
        request_error = error
        raise
    finally:
        stop_errors = []
        # A lost request response can hide a successful proxy-side start.
        # Stop each requested D even when one request reports an error.
        for decoder in reversed(decoders):
            try:
                decoder.request("/stop_profile", {})
                print(f"{PREFIX} stopped {decoder.base_url}", flush=True)
            except Exception as error:
                stop_errors.append(f"{decoder.base_url}: {error}")
        report["profile_stopped_at_ns"] = time.time_ns()
        if stop_errors:
            report["profiler_stop_errors"] = stop_errors
            print(f"{PREFIX} profiler stop errors: {' | '.join(stop_errors)}", file=sys.stderr, flush=True)
            if request_error is None:
                raise RuntimeError("Profiler stop failed; see result.json for the affected instances")


def capture(args):
    if min(args.prefix_tokens, args.output_tokens, args.concurrency, args.timeout, args.batch_timeout) <= 0:
        raise ValueError("Token counts, concurrency and timeouts must be positive")
    if not math.isfinite(args.timeout) or not math.isfinite(args.batch_timeout):
        raise ValueError("Timeouts must be finite")
    key = os.environ.get("OPENAI_API_KEY")
    client = Client(args.base_url, args.timeout, key)
    target = Client(args.target_d_url, args.timeout, key).base_url if args.target_d_url else None
    decoder_urls = args.d_url or ([target] if target else [])
    decoders = [Client(url, args.timeout, key) for url in decoder_urls]
    if len({decoder.base_url for decoder in decoders}) != len(decoders):
        raise ValueError("Pass each --d-url once")
    if target is not None and target not in {decoder.base_url for decoder in decoders}:
        raise ValueError("Include --target-d-url in --d-url so the target is profiled")
    tokenizer = Client(args.tokenizer_url or decoders[0].base_url, args.timeout, key)
    run_id = uuid.uuid4().hex
    root = (args.output or Path(f"decode-profile-{time.strftime('%Y%m%d-%H%M%S')}-{run_id[:6]}")).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "preflight",
        "run_id": run_id,
        "base_url": client.base_url,
        "d_urls": [decoder.base_url for decoder in decoders],
        "target_d_url": target,
        "concurrency": args.concurrency,
        "tokenizer_url": tokenizer.base_url,
        "prefix_tokens": args.prefix_tokens,
        "output_tokens": args.output_tokens,
        "profile_dir": str(args.profile_dir) if args.profile_dir else None,
        "capture_scope": "After every P request is ready, D-side KV loading and all generation steps",
        "prefill_batch_barrier": True,
        "profile_start_trigger": "proxy_after_prefill",
    }
    try:
        check_proxy(client, target, args.concurrency)
        model, instances = preflight(decoders, tokenizer, args.model, args.prefix_tokens + args.output_tokens)
        report["model"] = model
        report["decoder_instances"] = instances
        for instance in instances:
            if instance["url"] == target:
                if instance.get("dp_size", 1) > 1 and not instance.get("external_lb"):
                    raise ValueError("Use a single-DP D API endpoint for targeted instance profiling")
                capacity = instance.get("max_num_seqs")
                if capacity is not None and args.concurrency > capacity:
                    raise ValueError(f"D max_num_seqs={capacity}, requested concurrency={args.concurrency}")
            print(
                f"{PREFIX} decoder={instance['url']}, server_profile_dir={instance['profile_dir'] or 'unavailable'}",
                flush=True,
            )
        prompts = make_prompts(tokenizer, model, args.prefix_tokens, args.concurrency)
        prompt_hashes = [hashlib.sha256(json.dumps(prompt).encode()).hexdigest() for prompt in prompts]
        report["prompt_sha256"] = prompt_hashes[0] if args.concurrency == 1 else prompt_hashes
        if args.warmup:
            print(f"{PREFIX} warming {args.concurrency} session(s) through the P/D proxy", flush=True)
            warmup_dir = root / "warmup"
            warmup_dir.mkdir()
            report["warmup"] = run_requests(
                client,
                model,
                prompts,
                args.output_tokens,
                warmup_dir,
                target,
                args.batch_timeout,
            )
        before = profile_runs(args.profile_dir) if args.profile_dir else set()
        report["status"] = "capturing"
        write_json(root / "result.json", report)
        print(
            f"{PREFIX} concurrency={args.concurrency}, input={args.prefix_tokens:,}, "
            f"output={args.output_tokens:,}, target={target or 'proxy load balancing'}",
            flush=True,
        )
        measurements = profile_request(
            decoders,
            client,
            model,
            prompts,
            args.output_tokens,
            root,
            report,
            target,
            args.batch_timeout,
        )
        report["measurements"] = measurements
        if args.concurrency == 1:
            report["measurement"] = measurements[0]
        report_routes({"routes": [route for item in measurements for route in item["routes"]]}, instances, report)
        report["status"] = "captured"
        if args.profile_dir:
            runs = profile_runs(args.profile_dir) - before
            report["raw_profiles"] = sorted(map(str, runs))
            write_json(root / "raw_profiles.json", report["raw_profiles"])
            write_json(root / "result.json", report)
            report["traces"] = analyse_runs(runs, root, prefix=PREFIX)
            report["status"] = "exported"
        else:
            print(f"{PREFIX} raw captures saved under each D server's torch_profiler_dir", flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        write_json(root / "result.json", report)
        print(f"{PREFIX} result directory: {root}", flush=True)


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--base-url", default="http://127.0.0.1:9000", help="Request URL, normally the existing P/D proxy")
    cli.add_argument("--d-url", action="append", help="Direct D API URL; repeat for every decoder the proxy may select")
    cli.add_argument(
        "--target-d-url", help="Pin all requests to this configured D instance; also profiles it by default"
    )
    cli.add_argument(
        "--concurrency", type=int, default=1, help="Sessions released together after all P requests are ready"
    )
    cli.add_argument("--warmup", action="store_true", help="Run the same batch once before profiling")
    cli.add_argument(
        "--batch-timeout", type=float, default=600, help="Maximum seconds for all P requests to become ready"
    )
    cli.add_argument("--tokenizer-url", help="vLLM tokenization URL; defaults to the first --d-url")
    cli.add_argument("--model", help="Served model ID; auto-selected when exactly one is listed")
    cli.add_argument("--prefix-tokens", type=int, default=50000, help="Exact input token count")
    cli.add_argument("--output-tokens", type=int, default=128, help="Exact generated token count")
    cli.add_argument("--timeout", type=float, default=7200, help="HTTP timeout in seconds")
    cli.add_argument("--output", type=Path, help="New client result directory")
    cli.add_argument("--profile-dir", type=Path, help="Locally visible D profiler root; exports only new captures")
    cli.add_argument("--analyse-only", type=Path, help="Export existing captures inside the Ascend environment")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.analyse_only:
        directory = args.analyse_only.resolve(strict=True)
        runs = {directory} if directory.name.endswith("_ascend_pt") else profile_runs(directory)
        analyse_runs(runs, directory, prefix=PREFIX)
        return
    if not args.d_url and not args.target_d_url:
        cli.error("--d-url or --target-d-url is required for capture")
    if args.profile_dir:
        args.profile_dir = args.profile_dir.resolve(strict=True)
    capture(args)


if __name__ == "__main__":
    main()
