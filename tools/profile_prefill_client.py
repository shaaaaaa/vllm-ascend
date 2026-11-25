#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
r"""Capture one cached-prefix P-node request through its existing HTTP API.

Server prerequisites: four prefill banks, vLLM prefix caching disabled,
--enable-prompt-tokens-details, and a torch profiler configured with an absolute
output directory. Connect directly to one P instance. Default workload is a
512,000-token cached prefix followed by 128 new input tokens and one output token.

Examples:
  python tools/profile_prefill_client.py --base-url http://127.0.0.1:8000 \
      --profile-dir /workspace/sqh/vllm-ascend/vllm_profile/P_dp0
  python tools/profile_prefill_client.py --analyse-only /path/to/capture_ascend_pt

Capture uses only Python's standard library. Local trace export uses the
server's torch_npu installation in a separate process, after /stop_profile.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

PREFIX = "[PREFILL_PROFILE]"


class Client:
    def __init__(self, base_url, timeout, api_key=None):
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("--base-url must be an HTTP(S) P-instance address")
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        self.base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        self.timeout = timeout
        self.api_key = api_key

    def request(self, path, payload=None, *, headers=None):
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        if self.api_key:
            request_headers["Authorization"] = "Bearer " + self.api_key
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(self.base_url + path, data=data, headers=request_headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"HTTP {error.code} from {path}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"Request to {path} failed: {error}") from error
        return json.loads(body) if body else {}


def repeat_tokens(tokens, count):
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Tokenizer must return a nonempty list of nonnegative token IDs")
    repeats, tail = divmod(count, len(tokens))
    return tokens * repeats + tokens[:tail]


def workload(client, model, prefix_tokens, new_tokens, run_id):
    def tokenize(text):
        result = client.request("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})
        return result["tokens"]

    header = tokenize(f"Prefill overlap profiling run {run_id}.\n")
    body = tokenize(
        "The archive records measurements of rivers, mountains, cities and forests. "
        "Read the records carefully and summarize their patterns.\n"
    )
    if len(header) >= prefix_tokens:
        raise ValueError("Prefix length must exceed the unique request header")
    prefix = header + repeat_tokens(body, prefix_tokens - len(header))
    distinct = list(dict.fromkeys(body))
    if len(distinct) < 2:
        raise ValueError("At least two distinct body tokens are required")
    # Different first suffix tokens prevent the short probe's partial page from
    # becoming a cache hit for the measured request.
    probe_suffix = repeat_tokens(distinct, new_tokens)
    capture_suffix = repeat_tokens(distinct[1:] + distinct[:1], new_tokens)
    return prefix, probe_suffix, capture_suffix


def complete(client, model, tokens, phase, root):
    print(f"{PREFIX} {phase}: sending {len(tokens):,} prompt tokens", flush=True)
    started = time.perf_counter()
    response = client.request(
        "/v1/completions",
        {
            "model": model,
            "prompt": tokens,
            "add_special_tokens": False,
            "max_tokens": 1,
            "min_tokens": 1,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 1024,
            "stream": False,
            "kv_transfer_params": {"do_remote_decode": True, "do_remote_prefill": False},
        },
    )
    elapsed = time.perf_counter() - started
    usage = response.get("usage") or {}
    if usage.get("prompt_tokens") != len(tokens):
        raise RuntimeError(f"{phase}: prompt length differs from submitted IDs: {usage}")
    if usage.get("completion_tokens") != 1 or len(response.get("choices", [])) != 1:
        raise RuntimeError(f"{phase}: expected one completion with one output token: {usage}")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    report = {
        "phase": phase,
        "response_id": response.get("id"),
        "prompt_tokens": len(tokens),
        "cached_tokens": cached,
        "new_prompt_tokens": None if cached is None else len(tokens) - cached,
        "completion_tokens": 1,
        "seconds": elapsed,
    }
    write_json(root / f"{phase}.json", report)
    print(f"{PREFIX} {phase}: cached={cached}, seconds={elapsed:.3f}", flush=True)
    return report


def require_hit(report, expected):
    actual = report["cached_tokens"]
    if actual is None:
        raise RuntimeError(
            "Cache accounting is absent. Start P with --enable-prompt-tokens-details "
            "and check LMCache reuse before collecting the long trace."
        )
    if actual != expected:
        raise RuntimeError(
            f"{report['phase']}: expected {expected:,} cached tokens, observed {actual:,}; "
            f"actual new prompt tokens={report['new_prompt_tokens']}. "
            "Inspect P-instance affinity, chunk alignment and cache capacity."
        )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def preflight(client, model, required_tokens):
    paths = client.request("/openapi.json").get("paths", {})
    needed = {"/start_profile", "/stop_profile", "/tokenize", "/v1/completions"}
    missing = sorted(needed - paths.keys())
    if missing:
        raise RuntimeError(
            f"Missing P-instance endpoints {missing}. Enable --profiler-config and connect directly to the P instance."
        )
    models = client.request("/v1/models").get("data", [])
    if model is None:
        if len(models) != 1:
            raise ValueError("Use --model to choose one of the server's model IDs")
        model = models[0]["id"]
    selected = next((item for item in models if item["id"] == model), None)
    if selected is None:
        raise ValueError(f"Model {model!r} is absent from /v1/models")
    maximum = selected.get("max_model_len")
    if maximum is not None and maximum < required_tokens:
        raise ValueError(f"Server max_model_len={maximum}, required at least {required_tokens}")
    if "/server_info" in paths:
        info = client.request("/server_info?config_format=json")
        config = info.get("vllm_config", {})
        if config.get("cache_config", {}).get("enable_prefix_caching"):
            raise ValueError("Use --no-enable-prefix-caching to measure LMCache bank reloads")
        profiler = config.get("profiler_config") or {}
        for field in ("delay_iterations", "max_iterations", "warmup_iterations", "wait_iterations"):
            if profiler.get(field, 0):
                raise ValueError(f"Set profiler {field}=0 to capture this single request")
    return model


def profile_runs(directory):
    return {path.resolve() for path in directory.rglob("*_ascend_pt") if path.is_dir()}


def analyse_runs(runs, root, prefix=PREFIX):
    runs = sorted(runs)
    if not runs:
        raise RuntimeError("No new *_ascend_pt captures found in --profile-dir")
    print(f"{prefix} exporting {len(runs)} capture(s) after profiling stopped", flush=True)
    code = (
        "import sys; from torch_npu.profiler.profiler import analyse; "
        "[analyse(path, max_process_number=2) for path in sys.argv[1:]]"
    )
    subprocess.run([sys.executable, "-c", code, *map(str, runs)], check=True)
    traces = []
    for run in runs:
        worker_traces = sorted(str(path) for path in run.rglob("trace_view.json"))
        if not worker_traces:
            raise RuntimeError(f"Trace export produced no trace_view.json for {run}")
        traces.extend(worker_traces)
    write_json(root / "traces.json", traces)
    for trace in traces:
        print(f"{prefix} MindStudio trace: {trace}", flush=True)
    return traces


def capture(args):
    if args.prefix_tokens <= 0 or args.new_tokens <= 0 or args.chunk_tokens <= 0:
        raise ValueError("Token counts must be positive")
    if args.prefix_tokens % args.chunk_tokens:
        raise ValueError("--prefix-tokens must be divisible by --chunk-tokens")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    run_id = uuid.uuid4().hex
    root = (args.output or Path(f"prefill-profile-{time.strftime('%Y%m%d-%H%M%S')}-{run_id[:6]}")).resolve()
    root.mkdir(parents=True, exist_ok=False)
    client = Client(args.base_url, args.timeout, os.environ.get("OPENAI_API_KEY"))
    report = {
        "status": "preflight",
        "run_id": run_id,
        "base_url": client.base_url,
        "prefix_tokens": args.prefix_tokens,
        "new_prompt_tokens": args.new_tokens,
        "chunk_tokens": args.chunk_tokens,
        "expected_server_banks": 4,
        "bank_count_source": "server startup environment; set VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS=4",
        "profile_dir": str(args.profile_dir) if args.profile_dir else None,
    }
    try:
        model = preflight(client, args.model, args.prefix_tokens + args.new_tokens + 1)
        report["model"] = model
        prefix, probe_suffix, capture_suffix = workload(client, model, args.prefix_tokens, args.new_tokens, run_id)
        report["prefix_sha256"] = hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
        # Validate the reuse path with a small request before the long warmup.
        short_prefix = prefix[: min(args.chunk_tokens, args.prefix_tokens)]
        complete(client, model, short_prefix, "short_warmup", root)
        require_hit(complete(client, model, short_prefix + probe_suffix, "short_reuse", root), len(short_prefix))
        complete(client, model, prefix, "prefix_warmup", root)
        before = profile_runs(args.profile_dir) if args.profile_dir else set()
        report["status"] = "capturing"
        write_json(root / "result.json", report)
        print(f"{PREFIX} starting profiler for the final request", flush=True)
        client.request("/start_profile", {})
        try:
            measured = complete(client, model, prefix + capture_suffix, "profiled", root)
        finally:
            client.request("/stop_profile", {})
            print(f"{PREFIX} profiler stopped", flush=True)
        report["measurement"] = measured
        require_hit(measured, args.prefix_tokens)
        report["status"] = "captured"
        if args.profile_dir:
            runs = profile_runs(args.profile_dir) - before
            report["raw_profiles"] = sorted(map(str, runs))
            write_json(root / "raw_profiles.json", report["raw_profiles"])
            report["traces"] = analyse_runs(runs, root)
            report["status"] = "exported"
        else:
            print(
                f"{PREFIX} raw profiles are in the server's torch_profiler_dir. "
                "Export on the server with --analyse-only PATH.",
                flush=True,
            )
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        write_json(root / "result.json", report)
        print(f"{PREFIX} result directory: {root}", flush=True)


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--base-url", default="http://127.0.0.1:8000", help="Direct P-instance URL")
    cli.add_argument("--model", help="Served model ID; auto-selected when exactly one is listed")
    cli.add_argument("--prefix-tokens", type=int, default=512000)
    cli.add_argument("--new-tokens", type=int, default=128, help="New input tokens")
    cli.add_argument("--chunk-tokens", type=int, default=1024, help="Actual LMCACHE_CHUNK_SIZE")
    cli.add_argument("--timeout", type=float, default=7200, help="Seconds per HTTP request; no POST retries")
    cli.add_argument("--output", type=Path, help="New client result directory")
    cli.add_argument("--profile-dir", type=Path, help="Locally visible server profiler directory; exports new captures")
    cli.add_argument("--analyse-only", type=Path, help="Export existing raw profiles on the Ascend server")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if args.analyse_only:
        directory = args.analyse_only.resolve(strict=True)
        runs = {directory} if directory.name.endswith("_ascend_pt") else profile_runs(directory)
        analyse_runs(runs, directory)
    else:
        if args.profile_dir:
            args.profile_dir = args.profile_dir.resolve(strict=True)
        capture(args)


if __name__ == "__main__":
    main()
