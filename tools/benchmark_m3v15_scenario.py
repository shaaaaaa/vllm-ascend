#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay the public M3v15 cached-prefill workload on an installed vLLM tree.

Reference: ZD-euler/vllm-ascend commit add12600783b7818d217757cd93fa11bf7d2526d,
benchmarks/selected_kv/recovered/trace-m3v15/harness/run_cases.py.
Default geometry is H=524288, Q=4096, four independent requests. Run each arm
in its own environment with the same prepared input file and model weights.

  prepare --corpus /data/corpus/manifest.json --output /data/inputs.json
  run --inputs /data/inputs.json --engine-args /data/ours.json --output /data/ours
  compare /data/native/result.json /data/ours/result.json

The corpus is produced by the reference repository's regenerate_corpus.py.
--tokens-dir overrides the token-file directory recorded in that manifest.
Run accepts an AsyncEngineArgs JSON, preserving its connector configuration.
Required reference settings are TP4/DP4/EP16, eager, chunk6144, max_num_seqs=1,
block_size=128 and prefix caching. --allow-topology-change permits smaller runs
and records their topology. Model config, runtime versions, inputs, actual H/Q,
one-token logprobs, forward events and rank-zero profile are recorded.
Worker events measure model.forward only. Backend-specific completion/route
assertions from the reference harness require that harness's SDK integration.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import statistics
import sys
import time
from functools import wraps
from pathlib import Path

REFERENCE_COMMIT = "add12600783b7818d217757cd93fa11bf7d2526d"
SCHEMA = "cached-prefill-m3v15-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def token_digest(tokens):
    values = array.array("I", tokens)
    if values.itemsize != 4:
        raise RuntimeError("This platform requires a 32-bit unsigned array type")
    if sys.byteorder != "little":
        values.byteswap()
    return hashlib.sha256(values.tobytes()).hexdigest()


def checked_tokens(values):
    if not isinstance(values, list) or not values:
        raise ValueError("Token arrays must be nonempty JSON lists")
    if any(type(value) is not int or not 0 <= value < 2**32 for value in values):
        raise ValueError("Token IDs must be unsigned 32-bit integers")
    return values


def prepare_inputs(args):
    manifest = read_json(args.corpus)
    if manifest.get("synthetic_padding_tokens") is not False:
        raise ValueError("Use the reconstructed real-coding corpus manifest")
    candidates = [
        case for case in manifest["cases"] if case["history"] == args.history and case["total_query_tokens"] == 2048
    ]
    if not candidates:
        raise ValueError("The corpus has no reference windows for this history length")
    base = max(candidates, key=lambda case: case["requests"])
    windows = base["windows"][: args.requests]
    if len(windows) != args.requests:
        raise ValueError("The corpus has too few independent reference windows")
    sources = {source["task_name"]: source for source in manifest["sources"]}
    arrays = {}
    requests = []
    for window in windows:
        task = window["task"]
        if task not in arrays:
            source = sources[task]
            path = Path(source["token_file"])
            if args.tokens_dir:
                path = args.tokens_dir / path.name
            elif not path.is_absolute():
                path = args.corpus.parent / path
            if path.suffix == ".npy":
                import numpy as np

                values = np.load(path, allow_pickle=False).tolist()
            else:
                values = read_json(path)
            values = checked_tokens(values)
            if len(values) != source["glm52_tokens"] or token_digest(values) != source["tokens_sha256_uint32_le"]:
                raise ValueError(f"Corpus token hash/length mismatch for {task}")
            arrays[task] = values
        values = arrays[task]
        start = window["start_token"]
        original = values[start : start + args.history + base["query"]]
        if token_digest(original) != window["token_sha256"]:
            raise ValueError(f"Reference window hash mismatch for {task}")
        history = original[: args.history]
        seen = set()
        followups = []
        offsets = list(range(start + args.history, len(values) - args.query + 1, 128))
        offsets += range(0, start + args.history - args.query + 1, 128)
        for offset in offsets:
            fragment = values[offset : offset + args.query]
            key = token_digest(fragment[:128])
            if key in seen:
                continue
            seen.add(key)
            followups.append(fragment)
            if len(followups) == args.rounds:
                break
        if len(followups) != args.rounds:
            raise ValueError(f"Insufficient distinct real followups for {task}")
        requests.append({"task": task, "start_token": start, "history": history, "followups": followups})
    data = {
        "schema": SCHEMA,
        "reference_commit": REFERENCE_COMMIT,
        "history_tokens": args.history,
        "query_tokens": args.query,
        "requests": requests,
    }
    validate_inputs(data)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, data)
    print(json.dumps({"inputs": str(args.output), **input_summary(data)}, indent=2))


def input_summary(data):
    histories = [token_digest(row["history"]) for row in data["requests"]]
    followups = [[token_digest(tokens) for tokens in row["followups"]] for row in data["requests"]]
    proof = {
        "history_tokens": data["history_tokens"],
        "query_tokens": data["query_tokens"],
        "histories": histories,
        "followups": followups,
    }
    proof["sha256"] = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()
    return proof


def validate_inputs(data):
    if data.get("schema") != SCHEMA:
        raise ValueError("Unrecognized input schema")
    history, query = data["history_tokens"], data["query_tokens"]
    if history <= 0 or query < 128 or history % 128:
        raise ValueError("History must be 128-aligned and query must be at least 128")
    if not data["requests"]:
        raise ValueError("At least one request is required")
    blocks = set()
    rounds = set()
    for row in data["requests"]:
        prefix = checked_tokens(row["history"])
        if len(prefix) != history:
            raise ValueError("History length mismatch")
        block = token_digest(prefix[:128])
        if block in blocks:
            raise ValueError("Independent histories must have different first cache blocks")
        blocks.add(block)
        suffixes = row["followups"]
        rounds.add(len(suffixes))
        if len(suffixes) < 2:
            raise ValueError("Provide at least one warmup and one measured followup")
        if any(len(checked_tokens(tokens)) != query for tokens in suffixes):
            raise ValueError("Followup length mismatch")
        if len({token_digest(tokens[:128]) for tokens in suffixes}) != len(suffixes):
            raise ValueError("Followups must differ in their first cache block")
    if len(rounds) != 1:
        raise ValueError("Requests must have the same number of followups")


async def maybe_await(value):
    return await value if inspect.isawaitable(value) else value


class CachedPrefillObserver:
    """Injected through vLLM's worker_extension_cls, after normal worker setup."""

    def cpb_install(self):
        import torch
        from vllm.distributed import get_tp_group

        runner = self.model_runner
        self._cpb_rank = (int(self.parallel_config.data_parallel_rank), int(get_tp_group().rank_in_group))
        self._cpb_sample = None
        self._cpb_dummy = False
        self._cpb_original_forward = runner.model.forward
        self._cpb_original_dummy = runner._dummy_run
        original = self._cpb_original_forward
        dummy_original = self._cpb_original_dummy

        @wraps(dummy_original)
        def dummy(*args, **kwargs):
            self._cpb_dummy = True
            try:
                return dummy_original(*args, **kwargs)
            finally:
                self._cpb_dummy = False

        @wraps(original)
        def forward(*args, **kwargs):
            sample = self._cpb_sample
            if sample is None or self._cpb_dummy:
                return original(*args, **kwargs)
            count = runner.input_batch.num_reqs
            histories = runner.input_batch.num_computed_tokens_cpu[:count].tolist()
            starts = runner.query_start_loc
            if hasattr(starts, "np"):
                starts = starts.np
            elif getattr(starts, "device", None) is not None and starts.device.type == "cpu":
                starts = starts.detach().numpy()
            else:
                raise RuntimeError("The model runner must expose CPU query offsets")
            queries = [int(starts[i + 1] - starts[i]) for i in range(count)]
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            step = {"H": histories, "Q": queries, "begin": begin, "end": end}
            sample["steps"].append(step)
            entered = time.perf_counter()
            begin.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                step["host_ms"] = (time.perf_counter() - entered) * 1000

        runner._dummy_run = dummy
        runner.model.forward = forward
        return {"dp_rank": self._cpb_rank[0], "tp_rank": self._cpb_rank[1]}

    def cpb_begin(self, trial, profile_dir):
        import torch

        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        self._cpb_sample = {"trial": trial, "steps": []}
        self._cpb_profiler = None
        if profile_dir and self._cpb_rank == (0, 0):
            import torch_npu
            from torch_npu.profiler import ProfilerActivity
            from torch_npu.profiler.experimental_config import ProfilerLevel, _ExperimentalConfig

            destination = Path(profile_dir) / trial
            destination.mkdir(parents=True, exist_ok=False)
            handler = torch_npu.profiler.tensorboard_trace_handler(str(destination), analyse_flag=False)
            self._cpb_profiler = torch_npu.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                on_trace_ready=handler,
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
                experimental_config=_ExperimentalConfig(profiler_level=ProfilerLevel.Level2),
            )
            self._cpb_profiler.start()
        return {"dp_rank": self._cpb_rank[0], "tp_rank": self._cpb_rank[1]}

    def cpb_collect(self):
        import torch

        sample = self._cpb_sample
        self._cpb_sample = None
        torch.npu.synchronize()
        if self._cpb_profiler is not None:
            self._cpb_profiler.stop()
            self._cpb_profiler = None
        steps = [
            {
                "H": step["H"],
                "Q": step["Q"],
                "host_ms": step["host_ms"],
                "device_ms": float(step["begin"].elapsed_time(step["end"])),
            }
            for step in sample["steps"]
        ]
        return {
            "dp_rank": self._cpb_rank[0],
            "tp_rank": self._cpb_rank[1],
            "trial": sample["trial"],
            "steps": steps,
            "model_forward_ms": sum(step["device_ms"] for step in steps),
            "allocated_peak_bytes": int(torch.npu.max_memory_allocated()),
        }


async def first_token(collector, started):
    while True:
        output = await maybe_await(collector.get())
        if isinstance(output, BaseException):
            raise output
        if output is None:
            continue
        if output.outputs and output.outputs[0].token_ids:
            value = output.outputs[0]
            if not output.finished or len(value.token_ids) != 1:
                raise ValueError("Expected one finished single-token completion")
            logprobs = value.logprobs[0] if value.logprobs else {}
            scores = {str(token): float(score.logprob) for token, score in logprobs.items()}
            if not scores or any(not math.isfinite(score) for score in scores.values()):
                raise ValueError("Expected finite first-token logprobs")
            return {
                "ttft_ms": (time.perf_counter() - started) * 1000,
                "token_id": int(value.token_ids[0]),
                "logprobs": scores,
            }
        if output.finished:
            raise ValueError("Request finished without an output token")


def validate_ranks(rows, tp, dp, history=None, query=None, local_requests=1):
    expected = {(d, t) for d in range(dp) for t in range(tp)}
    if len(rows) != len(expected) or {(row["dp_rank"], row["tp_rank"]) for row in rows} != expected:
        raise ValueError("Worker RPC did not return every distinct DP/TP rank")
    if history is None:
        return []
    issues = []
    for row in rows:
        steps = row["steps"]
        if len(steps) != 1 or steps[0]["H"] != [history] * local_requests or steps[0]["Q"] != [query] * local_requests:
            issues.append(
                {
                    "dp_rank": row["dp_rank"],
                    "tp_rank": row["tp_rank"],
                    "actual_steps": steps,
                    "expected_H": history,
                    "expected_Q": query,
                }
            )
    return issues


async def run_burst(engine, data, params, ordinal, trial, profile_dir, timeout, tp, dp):
    await maybe_await(engine.pause_generation(mode="keep", wait_for_inflight_requests=True, clear_cache=False))
    acknowledgements = await engine.collective_rpc("cpb_begin", timeout=timeout, args=(trial, profile_dir))
    validate_ranks(acknowledgements, tp, dp)
    try:
        collectors = []
        for index, row in enumerate(data["requests"]):
            prompt = row["history"] + row["followups"][ordinal]
            collector = await maybe_await(
                engine.add_request(
                    f"cpb-{trial}-{index}", {"prompt_token_ids": prompt}, params, data_parallel_rank=index % dp
                )
            )
            collectors.append(collector)
        started = time.perf_counter()
        tasks = [asyncio.create_task(first_token(collector, started)) for collector in collectors]
        try:
            await maybe_await(engine.resume_generation())
            completions = await asyncio.wait_for(asyncio.gather(*tasks), timeout)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = (time.perf_counter() - started) * 1000
    finally:
        # Metrics and profiler export run after the first-token endpoint.
        await maybe_await(engine.pause_generation(mode="keep", wait_for_inflight_requests=True, clear_cache=False))
        ranks = await engine.collective_rpc("cpb_collect", timeout=timeout)
    issues = validate_ranks(ranks, tp, dp, data["history_tokens"], data["query_tokens"], len(data["requests"]) // dp)
    return {
        "trial": trial,
        "completions": completions,
        "all_first_tokens_ms": elapsed,
        "max_rank_forward_ms": max(row["model_forward_ms"] for row in ranks),
        "max_rank_allocated_peak_bytes": max(row["allocated_peak_bytes"] for row in ranks),
        "ranks": ranks,
        "geometry_issues": issues,
        "profiled": bool(profile_dir),
    }


def validate_engine(config, data, allow_topology_change):
    tp, dp = config.get("tensor_parallel_size", 1), config.get("data_parallel_size", 1)
    required = {"enforce_eager": True, "enable_prefix_caching": True, "enable_chunked_prefill": True, "block_size": 128}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Reference scenario requires engine argument {key}={value}")
    if config.get("pipeline_parallel_size", 1) != 1 or config.get("speculative_config"):
        raise ValueError("Reference scenario requires PP1 and speculation disabled")
    additional = config.get("additional_config") or {}
    if additional.get("enable_dsa_cp", False):
        raise ValueError("Reference scenario uses enable_dsa_cp=false")
    if not allow_topology_change:
        reference = {
            "tensor_parallel_size": 4,
            "data_parallel_size": 4,
            "enable_expert_parallel": True,
            "max_num_batched_tokens": 6144,
            "max_num_seqs": 1,
            "async_scheduling": False,
        }
        for key, value in reference.items():
            if config.get(key) != value:
                raise ValueError(
                    f"Reference scenario requires {key}={value}; use --allow-topology-change for a variant"
                )
        if len(data["requests"]) != 4:
            raise ValueError("Reference topology requires four requests")
    if tp < 1 or dp < 1 or len(data["requests"]) % dp:
        raise ValueError("Requests must divide evenly across a positive DP size")
    local_requests = len(data["requests"]) // dp
    if config.get("max_num_batched_tokens", 0) < local_requests * data["query_tokens"]:
        raise ValueError("Token budget must fit the entire measured local batch")
    if config.get("max_num_seqs", 0) < local_requests:
        raise ValueError("max_num_seqs must fit the measured local batch")
    if config.get("max_model_len", 0) < data["history_tokens"] + data["query_tokens"] + 1:
        raise ValueError("max_model_len must fit H + Q + one output token")
    if config.get("worker_extension_cls"):
        raise ValueError("This observer requires an unused worker_extension_cls")
    if not config.get("kv_transfer_config"):
        raise ValueError("Configure the layerwise KV connector used by the tested code tree")
    return tp, dp


def load_dtype_schemas(config):
    """Apply an installed tree's C8 frontend patches before EngineArgs imports."""
    attention = config.get("attention_config") or {}
    leaves = []
    if config.get("kv_cache_dtype") == "int8":
        leaves.append("patch_kv_cache_dtype.py")
    if attention.get("indexer_kv_dtype") == "int8":
        leaves.append("patch_indexer_kv_dtype.py")
    if not leaves:
        return []
    package = importlib.util.find_spec("vllm_ascend")
    if package is None or not package.submodule_search_locations:
        raise ImportError("Install the tested vllm_ascend package before enabling C8")
    root = Path(next(iter(package.submodule_search_locations)))
    receipts = []
    for leaf in leaves:
        path = root / "patch" / "platform" / leaf
        if not path.is_file():
            raise ValueError(f"The installed tree has no frontend dtype patch for this configuration: {path}")
        name = "_cached_prefill_" + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        receipts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return receipts


async def run(args):
    data = read_json(args.inputs)
    validate_inputs(data)
    config = read_json(args.engine_args)
    if args.model:
        config["model"] = args.model
    tp, dp = validate_engine(config, data, args.allow_topology_change)
    rounds = len(data["requests"][0]["followups"])
    if not 1 <= args.warmups < rounds:
        raise ValueError("Warmups must leave at least one measured round")
    model_config = read_json(Path(config["model"]) / "config.json")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA,
        "label": args.label,
        "status": "running",
        "inputs": input_summary(data),
        "engine_args": config.copy(),
        "model_config": model_config,
        "model_config_sha256": hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest(),
        "samples": [],
        "warmups": args.warmups,
        "profile": args.profile,
        "topology_variant": args.allow_topology_change,
    }
    write_json(args.output / "result.json", result)
    module_dir = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = module_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    config["worker_extension_cls"] = Path(__file__).stem + ".CachedPrefillObserver"
    versions = {}
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "lmcache", "lmcache-ascend"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result["versions"] = versions
    engine = None
    try:
        result["dtype_schema_patches"] = load_dtype_schemas(config)
        from vllm import SamplingParams
        from vllm.config import KVTransferConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        if isinstance(config.get("kv_transfer_config"), dict):
            config["kv_transfer_config"] = KVTransferConfig(**config["kv_transfer_config"])
        params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False, seed=0, logprobs=20)
        seed_params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False, seed=0)
        engine = await maybe_await(AsyncLLM.from_engine_args(AsyncEngineArgs(**config)))
        ranks = await engine.collective_rpc("cpb_install", timeout=args.timeout)
        validate_ranks(ranks, tp, dp)

        async def seed_one(index, row):
            prompt = row["history"] + row["followups"][0][:1]
            async for output in engine.generate(
                {"prompt_token_ids": prompt}, seed_params, f"cpb-seed-{index}", data_parallel_rank=index % dp
            ):
                if output.finished:
                    return
            raise ValueError("Seed request ended without completion")

        print("[CACHED_PREFILL] seeding histories", flush=True)
        await asyncio.wait_for(
            asyncio.gather(*(seed_one(i, row) for i, row in enumerate(data["requests"]))), args.timeout
        )
        await asyncio.sleep(args.settle_seconds)
        for ordinal in range(rounds):
            phase = "warmup" if ordinal < args.warmups else "measure"
            trial = f"{phase}-{ordinal}"
            profile_dir = str((args.output / "profile").resolve()) if args.profile and ordinal == rounds - 1 else None
            sample = await run_burst(engine, data, params, ordinal, trial, profile_dir, args.timeout, tp, dp)
            sample["phase"] = phase
            result["samples"].append(sample)
            write_json(args.output / "result.json", result)
            print(
                json.dumps(
                    {
                        key: sample[key]
                        for key in ("trial", "all_first_tokens_ms", "max_rank_forward_ms", "geometry_issues")
                    }
                ),
                flush=True,
            )
            if sample["geometry_issues"]:
                raise ValueError("Actual worker H/Q differs from the prepared scenario; see result.json")
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.output / "result.json", result)
        if engine is not None:
            await maybe_await(engine.shutdown())


def compare(args):
    left, right = (read_json(path) for path in args.results)
    for key in ("schema", "inputs", "model_config_sha256", "warmups", "profile"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Comparison input/configuration mismatch: {key}")
    for key in (
        "tensor_parallel_size",
        "data_parallel_size",
        "enable_expert_parallel",
        "dtype",
        "kv_cache_dtype",
        "quantization",
        "enforce_eager",
        "max_num_batched_tokens",
        "max_num_seqs",
        "block_size",
        "async_scheduling",
    ):
        if left["engine_args"].get(key) != right["engine_args"].get(key):
            raise ValueError(f"Comparison engine geometry mismatch: {key}")
    if any(value["status"] != "complete" for value in (left, right)):
        raise ValueError("Both runs must complete successfully")
    if len(left["samples"]) != len(right["samples"]):
        raise ValueError("Different sample counts")
    token_mismatches, support_mismatches, differences = [], [], []
    for a, b in zip(left["samples"], right["samples"], strict=True):
        if a["trial"] != b["trial"] or len(a["completions"]) != len(b["completions"]):
            raise ValueError("Different trials or completion counts")
        for index, (x, y) in enumerate(zip(a["completions"], b["completions"], strict=True)):
            if x["token_id"] != y["token_id"]:
                token_mismatches.append([a["trial"], index])
            if x["logprobs"].keys() != y["logprobs"].keys():
                support_mismatches.append([a["trial"], index])
            differences.extend(
                abs(x["logprobs"][token] - y["logprobs"][token])
                for token in x["logprobs"].keys() & y["logprobs"].keys()
            )
    maximum = max(differences, default=math.inf)
    summary = {
        "max_abs_logprob_difference": maximum,
        "logprobs_compared": len(differences),
        "token_mismatches": token_mismatches,
        "logprob_support_mismatches": support_mismatches,
        "tolerance": args.tolerance,
        "numerically_matched": not token_mismatches and not support_mismatches and maximum <= args.tolerance,
    }
    for label, value in zip(("baseline", "candidate"), (left, right), strict=True):
        measured = [sample for sample in value["samples"] if sample["phase"] == "measure"]
        summary[label] = {
            "label": value["label"],
            "samples": len(measured),
            "median_all_first_tokens_ms": statistics.median(sample["all_first_tokens_ms"] for sample in measured),
            "median_max_rank_forward_ms": statistics.median(sample["max_rank_forward_ms"] for sample in measured),
            "peak_allocated_GiB": max(sample["max_rank_allocated_peak_bytes"] for sample in measured) / 2**30,
        }
    for metric in ("median_all_first_tokens_ms", "median_max_rank_forward_ms"):
        baseline = summary["baseline"][metric]
        summary[metric + "_reduction_pct"] = 100 * (1 - summary["candidate"][metric] / baseline) if baseline else None
    print(json.dumps(summary, indent=2))
    return 0 if summary["numerically_matched"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Save deterministic reference input token arrays")
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--tokens-dir", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--history", type=int, default=524288)
    prepare.add_argument("--query", type=int, default=4096)
    prepare.add_argument("--requests", type=int, default=4)
    prepare.add_argument("--rounds", type=int, default=2)
    execute = commands.add_parser("run", help="Run one installed-code arm using AsyncEngineArgs JSON")
    execute.add_argument("--inputs", type=Path, required=True)
    execute.add_argument("--engine-args", type=Path, required=True)
    execute.add_argument("--model")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--label", default="candidate")
    execute.add_argument("--warmups", type=int, default=1)
    execute.add_argument("--settle-seconds", type=float, default=31)
    execute.add_argument("--timeout", type=float, default=7200)
    execute.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    execute.add_argument("--allow-topology-change", action="store_true")
    comparison = commands.add_parser("compare", help="Compare first-token logprobs and measured timings")
    comparison.add_argument("results", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    comparison.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.history <= 0 or args.query < 128 or args.requests <= 0 or args.rounds < 2:
            parser.error("Use positive history/requests, query >= 128 and rounds >= 2")
        prepare_inputs(args)
    elif args.command == "run":
        if args.timeout <= 0 or args.settle_seconds < 0:
            parser.error("timeout must be positive and settle-seconds nonnegative")
        asyncio.run(run(args))
    else:
        if args.tolerance < 0 or not math.isfinite(args.tolerance):
            parser.error("tolerance must be finite and nonnegative")
        return compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
