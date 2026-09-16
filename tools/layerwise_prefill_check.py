#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Three sequential full-model runs: baseline -> offload P -> archive-backed D.

Usage: python tools/layerwise_prefill_check.py 2>&1 | tee log.log
No profiler, no dummy weights, no hidden-layer override, no numeric pass/fail
threshold. Baseline/D use PIECEWISE graphs; only P is eager. MTP remains off.
The KV probes copy/synchronize data: do not use these timings as performance data.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext, suppress
from numbers import Integral
from pathlib import Path

CHUNK_SIZE = 256
MAX_MODEL_LEN = 16384
DEFAULT_ANALYSIS_WORKERS = 64
ARCHIVE_BATCH_FILES = 64
ANALYSIS_PROGRESS_SECONDS = 5
STAGES = ("baseline", "prefill", "decode")
DEFAULT_PROMPT_FILE = Path(__file__).resolve().parents[1] / "examples/layerwise_prefill/article_summary.txt"


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class Moments:
    """Population moments, merged in float64; non-finite values are counted."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.maximum = None
        self.nonfinite = 0

    def add(self, tensor):
        import torch

        values = tensor.detach().double().reshape(-1)
        finite = torch.isfinite(values)
        self.nonfinite += int((~finite).sum())
        values = values[finite]
        count = values.numel()
        if not count:
            return
        mean = float(values.mean())
        m2 = float(((values - mean) ** 2).sum())
        delta = mean - self.mean
        total = self.n + count
        self.m2 += m2 + delta * delta * self.n * count / total
        self.mean += delta * count / total
        self.n = total
        maximum = float(values.max())
        self.maximum = maximum if self.maximum is None else max(self.maximum, maximum)

    def result(self):
        return {
            "count": self.n,
            "mean": self.mean if self.n else None,
            "variance": self.m2 / self.n if self.n else None,
            "max": self.maximum,
            "nonfinite": self.nonfinite,
        }

    def merge(self, other):
        self.nonfinite += other.nonfinite
        if not other.n:
            return
        delta = other.mean - self.mean
        total = self.n + other.n
        self.m2 += other.m2 + delta * delta * self.n * other.n / total
        self.mean += delta * other.n / total
        self.n = total
        self.maximum = other.maximum if self.maximum is None else max(self.maximum, other.maximum)


def tensor_statistics(base, candidate):
    if base.shape != candidate.shape:
        raise ValueError(f"Compared KV shapes differ: {base.shape} vs {candidate.shape}")
    accumulators = {
        name: Moments() for name in ("baseline", "candidate", "baseline_abs", "candidate_abs", "diff", "abs_diff")
    }
    # Widen BEFORE subtracting: BF16/FP16 arithmetic would hide small errors.
    for left, right in zip(base.reshape(-1).split(262144), candidate.reshape(-1).split(262144), strict=True):
        left, right = left.double(), right.double()
        values = (left, right, left.abs(), right.abs(), right - left, (right - left).abs())
        for metric, value in zip(accumulators.values(), values, strict=True):
            metric.add(value)
    return {name: metric.result() for name, metric in accumulators.items()}


def first_difference(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return None if len(left) == len(right) else min(len(left), len(right))


def normalize_prompt_token_ids(encoded) -> list[int]:
    """Extract one sequence, not the number of BatchEncoding fields or rows."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("Tokenizer result has no input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)) or not encoded:
        raise ValueError("Tokenizer must return a non-empty token ID sequence")
    if any(not isinstance(token, Integral) or isinstance(token, bool) or token < 0 for token in encoded):
        raise ValueError("Tokenizer must return one sequence of non-negative integer token IDs")
    return [int(token) for token in encoded]


def prepare_prompt(args, root):
    # Read a committed, reviewable input; never generate or pad it at runtime.
    source = args.prompt_file.resolve()
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty prompt file: {source}")
    (root / "prompt.txt").write_text(text, encoding="utf-8")
    print(f"[PREFILL_CHECK] prompt: {source}; importing tokenizer dependencies", flush=True)
    from transformers import AutoTokenizer

    print(f"[PREFILL_CHECK] loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("[PREFILL_CHECK] tokenizing fixed article once", flush=True)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    # Some custom tokenizers still return a mapping/tensor; normalize before
    # measuring length, hashing, persisting, or passing IDs to any engine.
    ids = normalize_prompt_token_ids(encoded)
    print(f"[PREFILL_CHECK] prompt_tokens={len(ids)}, output_tokens={args.output_tokens}", flush=True)
    if len(ids) <= max(args.prefill_chunk_tokens, CHUNK_SIZE):
        raise ValueError("Prompt must span multiple compute-prefill chunks AND LMCache chunks")
    if args.prompt_tokens is not None and len(ids) < args.prompt_tokens:
        raise ValueError(f"Fixed prompt has {len(ids)} tokens, below --prompt-tokens={args.prompt_tokens}")
    validate_sequence_length(len(ids), args.output_tokens)
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    write_json(root / "prompt.json", {"token_ids": ids, "sha256": digest, "length": len(ids), "source": str(source)})
    return len(ids)


def stage_environment(args, root, stage):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LMCACHE_", "VLLM_", "MOONCAKE_"))}
    stage_dir = root / stage
    archive = root / ("prefill" if stage == "decode" else stage) / "archive"
    extra = {
        "save_only_first_rank": True,
        "save_chunk_meta": True,
        "enable_cache_usage_details_in_response": True,
        "validation_archive": str(archive),
        "validation_stage_dir": str(stage_dir),
        "validation_read_only": stage == "decode",
    }
    env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", ""),
            "ASCEND_RT_VISIBLE_DEVICES": args.devices,
            "PYTHONHASHSEED": "0",
            "HCCL_DETERMINISTIC": "strict",
            "HCCL_BUFFSIZE": "200",
            "MSMONITOR_USE_DAEMON": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": str(stage == "prefill").lower(),
            "VLLM_ASCEND_DSA_UNBUNDLE": "1",
            "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
            "VLLM_ASCEND_DSA_SHARED_POOL": "1",
            "VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE": "0",
            "VLLM_ASCEND_DSA_SHRINK_LATENT": "0" if stage == "prefill" else "2",
            "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "0",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
            "VLLM_ASCEND_SFA_STAGED_GRAPH": "0",
            "VLLM_ASCEND_SFA_FULL_GRAPH": "0",
            "LMCACHE_CHUNK_SIZE": str(CHUNK_SIZE),
            "LMCACHE_LOCAL_CPU": "true",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(args.cpu_cache_gb),
            "LMCACHE_USE_LAYERWISE": "true",
            "LMCACHE_ENABLE_SPARSE_ATTENTION": "true",
            "LMCACHE_DSA_TWO_GROUPS": "true",
            "LMCACHE_STORE_ASYNC": "false",
            "LMCACHE_SAVE_DECODE_CACHE": "false",
            "LMCACHE_SAVE_UNFULL_CHUNK": "true",
            "LMCACHE_SAVE_FULL_CHUNK_IN_DECODE": "false",
            "LMCACHE_ENABLE_SHARED_CPU_CACHE": "true",
            "LMCACHE_SHARED_CPU_CACHE_STRICT": "true",
            "LMCACHE_LOOKUP_TIMEOUT_MS": "300000",
            "LMCACHE_BLOCKING_TIMEOUT_SECS": "300",
            "LMCACHE_REMOTE_URL": "external://localhost:0/layerwise_prefill_store/?connector_name=ValidationFileConnector",
            "LMCACHE_REMOTE_SERDE": "naive",
            "LMCACHE_EXTRA_CONFIG": json.dumps(extra),
        }
    )
    return env


def validate_sequence_length(prompt_len, output_tokens):
    if prompt_len + output_tokens > MAX_MODEL_LEN:
        raise ValueError(
            f"Prompt ({prompt_len}) + output ({output_tokens}) exceeds max_model_len={MAX_MODEL_LEN}; "
            "use a shorter --prompt-file or reduce --output-tokens"
        )


def engine_options(args, prompt_len, stage):
    validate_sequence_length(prompt_len, args.output_tokens)
    options = {
        "model": args.model,
        "trust_remote_code": True,
        "load_format": "safetensors",
        "quantization": "ascend",
        "tensor_parallel_size": len(args.devices.split(",")),
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "enable_expert_parallel": True,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.prefill_chunk_tokens,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "gpu_memory_utilization": 0.96,
        "seed": 1024,
        "worker_extension_cls": "layerwise_prefill_probe.PrefillValidationWorker",
        "additional_config": {
            "recompute_scheduler_enable": False,
            "multistream_overlap_shared_expert": False,
            "fuse_muls_add": True,
            "fuse_qknorm_rope": False,
            "enable_npugraph_ex": True,
            "layer_sharding": ["q_b_proj"],
        },
        "kv_transfer_config": {
            "kv_connector": "LMCacheAscendConnectorV1Dynamic",
            "kv_role": {"baseline": "kv_both", "prefill": "kv_producer", "decode": "kv_consumer"}[stage],
            "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
        },
    }
    if stage == "prefill":
        options["enforce_eager"] = True
    else:
        # Ordinary PIECEWISE keeps mla_forward outside capture, so the KV
        # hooks still execute on each real request, not only during warmup.
        options["compilation_config"] = {"cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1]}
    return options


def run_child(args):
    print(f"[PREFILL_CHECK] {args.stage}: importing vLLM", flush=True)
    from vllm import LLM, SamplingParams

    root = Path(args.run_dir)
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    options = engine_options(args, prompt["length"], args.stage)
    write_json(root / args.stage / "engine_options.json", options)
    llm = LLM(**options)
    try:
        workers = llm.collective_rpc(
            "install_prefill_validation", timeout=300, args=(str(root / args.stage), prompt["length"])
        )
        write_json(root / args.stage / "workers.json", workers)
        output = llm.generate(
            {"prompt_token_ids": prompt["token_ids"]},
            SamplingParams(
                temperature=0,
                seed=1024,
                max_tokens=1 if args.stage == "prefill" else args.output_tokens,
                ignore_eos=False,
            ),
            use_tqdm=False,
        )
        traces = llm.collective_rpc("finish_prefill_validation", timeout=300)
        write_json(root / args.stage / "trace_coverage.json", traces)
        if len(output) != 1 or len(output[0].outputs) != 1:
            raise RuntimeError("Expected one completed request")
        result = output[0]
        completion = result.outputs[0]
        record = {
            "stage": args.stage,
            "text": completion.text,
            "token_ids": list(completion.token_ids),
            "prompt_sha256": prompt["sha256"],
            "num_cached_tokens": result.num_cached_tokens,
            "kv_transfer_params": result.kv_transfer_params,
            "finish_reason": completion.finish_reason,
            "output_tokens": len(completion.token_ids),
            "output_token_limit": 1 if args.stage == "prefill" else args.output_tokens,
            "enforce_eager": llm.llm_engine.model_config.enforce_eager,
            "num_hidden_layers": llm.llm_engine.model_config.hf_config.num_hidden_layers,
        }
        write_json(root / args.stage / "output.json", record)
        (root / args.stage / "output.txt").write_text(completion.text, encoding="utf-8")
        print(
            f"[PREFILL_CHECK] {args.stage} output_tokens={len(completion.token_ids)}, "
            f"finish_reason={completion.finish_reason}; output: {completion.text!r}",
            flush=True,
        )
    finally:
        # Same shutdown used by the offline engine client; parent also fences
        # the process group it created before starting the next model copy.
        llm.llm_engine.engine_core.shutdown()


def stop_process_group(pgid):
    """Only terminate descendants in this launcher's own new session."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    with suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def run_stage(args, root, stage):
    stage_dir = root / stage
    stage_dir.mkdir()
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--run-dir",
        str(root),
        "--model",
        args.model,
        "--devices",
        args.devices,
        "--output-tokens",
        str(args.output_tokens),
        "--prefill-chunk-tokens",
        str(args.prefill_chunk_tokens),
        "--cpu-cache-gb",
        str(args.cpu_cache_gb),
    ]
    print(f"[PREFILL_CHECK] starting {stage}; log: {stage_dir / 'server.log'}", flush=True)
    with (stage_dir / "server.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            env=stage_environment(args, root, stage),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
            bufsize=1,
        )

        def copy_log():
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

        reader = threading.Thread(target=copy_log, daemon=True)
        reader.start()
        try:
            code = proc.wait()
        finally:
            stop_process_group(proc.pid)
            proc.wait()
            reader.join(timeout=15)
            if reader.is_alive():
                raise RuntimeError("A test child still holds the output pipe; refusing to start another model")
    if code:
        raise RuntimeError(f"{stage} failed ({code}); see {stage_dir / 'server.log'}")
    if not (stage_dir / "output.json").is_file():
        raise RuntimeError(f"{stage} produced no output artifact")


def read_index(stage_dir):
    index = defaultdict(list)
    for path in sorted(stage_dir.glob("kv/rank*/index.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            index[(path.parent.name, item["layer"], item["part"], item["kind"])].append(path.parent / item["path"])
    return index


def load_rows(files, start, stop, cache=None):
    import torch

    if stop <= start or not files:
        return torch.empty(0, dtype=torch.long), None
    key = tuple(files)
    cached = cache.get(key) if cache is not None else None
    if cached is None:
        chunks = [torch.load(path, map_location="cpu", weights_only=True) for path in files]
        pos = torch.cat([item["positions"] for item in chunks])
        values = torch.cat([item["values"] for item in chunks])
        del chunks
        if pos.numel() > 1 and bool((pos[1:] < pos[:-1]).any()):
            order = torch.argsort(pos)
            pos, values = pos[order], values[order]
        if cache is not None:
            cache[key] = (pos, values)
    else:
        pos, values = cached
    # Sorted positions allow slicing without copying a complete KV plane again.
    begin, end = torch.searchsorted(pos, torch.tensor([start, stop], dtype=pos.dtype)).tolist()
    pos, values = pos[begin:end], values[begin:end]
    if pos.numel() > 1 and bool((pos[1:] == pos[:-1]).any()):
        raise RuntimeError("Duplicate logical KV writes: trace cannot be aligned unambiguously")
    return pos, values


def compare_rows(base_files, candidate_files, start, stop, cache=None):
    import torch

    bp, bv = load_rows(base_files, start, stop, cache)
    cp, cv = load_rows(candidate_files, start, stop, cache)
    report = {
        "baseline_rows": bp.numel(),
        "candidate_rows": cp.numel(),
        "matched_rows": 0,
        "baseline_only_rows": bp.numel(),
        "candidate_only_rows": cp.numel(),
    }
    if not bp.numel() or not cp.numel():
        return report
    indices = torch.searchsorted(bp, cp)
    valid = indices < bp.numel()
    valid &= bp[indices.clamp(max=bp.numel() - 1)] == cp
    common = int(valid.sum())
    report.update(matched_rows=common, baseline_only_rows=bp.numel() - common, candidate_only_rows=cp.numel() - common)
    if common:
        report["stats"] = tensor_statistics(bv[indices[valid]], cv[valid])
    return report


def init_analysis_worker():
    # Spawned CPU-only workers do not inherit the model/NPU runtime. Avoid each
    # of the processes starting another full-size OpenMP thread pool.
    import torch

    torch.set_num_threads(1)


def run_analysis_job(function, job):
    started = time.monotonic()
    return function(job), {"seconds": time.monotonic() - started, "pid": os.getpid()}


def analysis_results(function, jobs, executor, workers, phase):
    """Bound queued work; only paths enter workers and small statistics return."""
    started = time.monotonic()
    last_progress = started

    def report(completed, timing=None, inflight=0):
        nonlocal last_progress
        now = time.monotonic()
        if completed not in (1, len(jobs)) and now - last_progress < ANALYSIS_PROGRESS_SECONDS:
            return
        last_progress = now
        detail = f"job={timing['seconds']:.2f}s" if timing is not None else f"inflight={inflight}"
        print(
            f"[PREFILL_CHECK] analyse {phase}: {completed}/{len(jobs)}, {detail}, elapsed={now - started:.1f}s",
            flush=True,
        )

    if executor is None:
        for index, job in enumerate(jobs):
            result, timing = run_analysis_job(function, job)
            report(index + 1, timing)
            yield index, result, timing
        return
    pending = {}
    remaining = iter(enumerate(jobs))

    def submit_next():
        item = next(remaining, None)
        if item is not None:
            index, job = item
            pending[executor.submit(run_analysis_job, function, job)] = index

    for _ in range(min(workers, len(jobs))):
        submit_next()
    completed = 0
    try:
        while pending:
            done, _ = wait(pending, timeout=ANALYSIS_PROGRESS_SECONDS, return_when=FIRST_COMPLETED)
            if not done:
                report(completed, inflight=len(pending))
            for future in done:
                index = pending.pop(future)
                result, timing = future.result()
                completed += 1
                report(completed, timing)
                yield index, result, timing
                submit_next()
    finally:
        for future in pending:
            future.cancel()


def compare_trace_job(job):
    key, files, length, cutoff = job
    rank, layer, part, _ = key
    cache = {}  # One rank/layer/part only; released before this worker's next job.
    rows, errors = [], []
    for label, left, right, start, stop in (
        ("prefill_written", "baseline", "prefill", 0, length),
        ("decode_reloaded", "prefill", "loaded", 0, length),
        ("decode_written_same_prefix", "baseline", "decode", length, cutoff),
        ("decode_recomputed_prompt_tail", "baseline", "decode", 0, length),
    ):
        row = {"comparison": label, "rank": rank, "layer": layer, "part": part}
        row.update(compare_rows(files[left], files[right], start, stop, cache))
        if label == "prefill_written" and row["matched_rows"] != length:
            errors.append(f"Incomplete prefill trace: {rank}/{layer}/{part}")
        if label == "decode_reloaded":
            if not row["candidate_rows"]:
                errors.append(f"No observed NPU reload: {rank}/{layer}/{part}")
            elif not row["baseline_rows"]:
                errors.append(f"Missing P reference trace for NPU reload comparison: {rank}/{layer}/{part}")
            elif not row["matched_rows"]:
                errors.append(f"No common P/D reload positions: {rank}/{layer}/{part}")
        if label == "decode_recomputed_prompt_tail" and row["candidate_rows"] > CHUNK_SIZE:
            errors.append(f"D recomputed more than the uncached prompt tail: {rank}/{layer}/{part}")
        rows.append(row)
    return rows, errors


def compare_archive_job(job):
    import torch

    base_dir, candidate_dir, names = job
    reports = []
    # Compare saved values, not just pre-D2H device snapshots; this catches
    # corruption in bank reuse / D2H / deferred publication itself.
    aggregates = {}
    for name in names:
        left = torch.load(base_dir / name, weights_only=True, map_location="cpu")
        right = torch.load(candidate_dir / name, weights_only=True, map_location="cpu")
        group = (left["worker_id"], left["kv_group"], left["layer_id"])
        if any(left[k] != right[k] for k in ("key", "shapes", "dtypes", "fmt", "valid_tokens")):
            reports.append({"key": left["key"], "status": "layout_mismatch"})
            continue
        metrics = aggregates.setdefault(
            group, {name: Moments() for name in ("baseline", "candidate", "diff", "abs_diff")}
        )
        offset = 0
        for shape, dtype_str in zip(left["shapes"], left["dtypes"], strict=True):
            dtype = getattr(torch, dtype_str.removeprefix("torch."))
            count = torch.Size(shape).numel() * dtype.itemsize
            a = left["raw"][offset : offset + count].view(dtype).double()
            b = right["raw"][offset : offset + count].view(dtype).double()
            for metric, data in zip(metrics.values(), (a, b, b - a, (b - a).abs()), strict=True):
                metric.add(data)
            offset += count
    return reports, aggregates


def archive_layer_reports(aggregates):
    return [
        {
            "worker": group[0],
            "kv_group": group[1],
            "layer_ordinal": group[2],
            "stats": {name: m.result() for name, m in metrics.items()},
        }
        for group, metrics in sorted(aggregates.items())
    ]


def compare_archives(root, executor=None, workers=1, on_progress=None):
    base_dir, candidate_dir = root / "baseline/archive", root / "prefill/archive"
    left_names = {p.name for p in base_dir.glob("*.pt")}
    right_names = {p.name for p in candidate_dir.glob("*.pt")}
    names = sorted(left_names & right_names)
    jobs = [
        (base_dir, candidate_dir, names[i : i + ARCHIVE_BATCH_FILES]) for i in range(0, len(names), ARCHIVE_BATCH_FILES)
    ]
    partials = {}
    for index, result, timing in analysis_results(compare_archive_job, jobs, executor, workers, "archive"):
        partials[index] = result
        if on_progress is not None:
            reports, aggregates = result
            on_progress(
                {
                    "phase": "archive",
                    "batch": index,
                    "partial": True,
                    "layers": reports + archive_layer_reports(aggregates),
                    **timing,
                }
            )
    # Merge in file order, not completion order, for reproducible reductions.
    reports, aggregates = [], {}
    for index in sorted(partials):
        batch_reports, batch_aggregates = partials[index]
        reports.extend(batch_reports)
        for group, metrics in batch_aggregates.items():
            merged = aggregates.setdefault(group, {name: Moments() for name in metrics})
            for name, metric in metrics.items():
                merged[name].merge(metric)
    reports.extend(archive_layer_reports(aggregates))
    return {
        "common_keys": len(left_names & right_names),
        "baseline_only_keys": len(left_names - right_names),
        "prefill_only_keys": len(right_names - left_names),
        "per_layer": reports,
    }


def validate_reload(root, prompt_len):
    records = []
    for path in (root / "decode").glob("archive_reads_*.jsonl"):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    reads = {item["key"]: item for item in records}
    if not reads or {item["kv_group"] for item in reads.values()} != {0, 1}:
        raise RuntimeError("D did not reload BOTH KV groups from the P archive")
    output = json.loads((root / "decode/output.json").read_text(encoding="utf-8"))
    details = output.get("kv_transfer_params") or {}
    cached = max(int(output.get("num_cached_tokens") or 0), int(details.get("num_lmcache_cached_tokens") or 0))
    required = (prompt_len - 1) // CHUNK_SIZE * CHUNK_SIZE
    if cached < required:
        raise RuntimeError(f"D cache hit too short ({cached} < {required}); cannot claim a PD reload test")
    return {
        "distinct_keys_read": len(reads),
        "groups": [0, 1],
        "cached_tokens": cached,
        "required_cached_tokens": required,
        "transport": "test-only durable files, NOT Mooncake",
    }


def seal_archive(root):
    import torch

    manifest = {}
    groups = set()
    for path in (root / "prefill/archive").glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        digest = hashlib.sha256(payload["raw"].numpy().tobytes()).hexdigest()
        if digest != payload["sha256"]:
            raise RuntimeError(f"P archive corrupted before D startup: {path}")
        manifest[payload["key"]] = digest
        groups.add(payload["kv_group"])
    if groups != {0, 1}:
        raise RuntimeError("P has not persisted both latent and indexer groups; D will not be started")
    write_json(root / "prefill/archive_manifest.json", manifest)


def verify_archive_reads(root):
    sealed = json.loads((root / "prefill/archive_manifest.json").read_text(encoding="utf-8"))
    for path in (root / "decode").glob("archive_reads_*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            read = json.loads(line)
            if sealed.get(read["key"]) != read["sha256"]:
                raise RuntimeError("D read bytes that were not in the sealed P output")


def analyse(root, ranks, workers=DEFAULT_ANALYSIS_WORKERS):
    if workers < 1:
        raise ValueError("Analysis workers must be positive")
    started = time.monotonic()
    print(f"[PREFILL_CHECK] analysing saved KV with {workers} CPU workers; no model will be started", flush=True)
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    length = prompt["length"]
    outputs = {stage: json.loads((root / stage / "output.json").read_text(encoding="utf-8")) for stage in STAGES}
    if any(out["prompt_sha256"] != prompt["sha256"] for out in outputs.values()):
        raise RuntimeError("Three runs used different prompts")
    mismatch = first_difference(outputs["baseline"]["token_ids"], outputs["decode"]["token_ids"])
    cutoff = length + (mismatch if mismatch is not None else len(outputs["baseline"]["token_ids"]))
    rows, errors = [], []
    summary = {
        "analysis_status": "running",
        "analysis_workers": workers,
        "prompt_tokens": length,
        "mtp": False,
        "enforce_eager": {stage: outputs[stage].get("enforce_eager") for stage in STAGES},
        "full_model": True,
        "tokens_equal": mismatch is None,
        "first_different_output_token_index": mismatch,
        "prefill_first_token_equal": outputs["baseline"]["token_ids"][:1] == outputs["prefill"]["token_ids"][:1],
        "decode_compare_position_exclusive": cutoff,
        "reload": None,
        "structural_errors": errors,
        "kv": rows,
        "persisted_kv": None,
    }
    last_snapshot = started

    def snapshot():
        nonlocal last_snapshot
        last_snapshot = time.monotonic()
        summary["analysis_elapsed_seconds"] = last_snapshot - started
        temp = root / "summary.json.tmp"
        write_json(temp, summary)
        temp.replace(root / "summary.json")

    snapshot()
    print(
        f"[PREFILL_CHECK] tokens_equal={mismatch is None}, first_difference={mismatch}; "
        f"partial summary: {root / 'summary.json'}",
        flush=True,
    )
    try:
        indices = {stage: read_index(root / stage) for stage in STAGES}
        base_keys = {key for key in indices["baseline"] if key[-1] == "current"}
        if {key[0] for key in base_keys} != {f"rank{i}" for i in range(ranks)}:
            raise RuntimeError("Missing baseline worker KV traces")
        expected_layers = outputs["baseline"]["num_hidden_layers"]
        for rank in range(ranks):
            layers = {key[1] for key in base_keys if key[0] == f"rank{rank}"}
            if len(layers) != expected_layers:
                errors.append(f"rank{rank} traced {len(layers)} layers, expected {expected_layers}")
        jobs = []
        for key in sorted(base_keys):
            files = {stage: indices[stage].get(key, []) for stage in STAGES}
            files["loaded"] = indices["decode"].get((*key[:3], "loaded"), [])
            jobs.append((key, files, length, cutoff))
        try:
            reload = validate_reload(root, length)
            verify_archive_reads(root)
        except RuntimeError as error:
            errors.append(str(error))
            reload = {"error": str(error)}
        summary["reload"] = reload
        pool = (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=init_analysis_worker,
            )
            if workers > 1
            else nullcontext(None)
        )
        with pool as executor, (root / "analysis_progress.jsonl").open("w", encoding="utf-8") as progress:

            def record(data):
                progress.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
                progress.flush()
                summary["analysis_phase"] = data["phase"]
                if time.monotonic() - last_snapshot >= ANALYSIS_PROGRESS_SECONDS:
                    snapshot()

            for index, (new_rows, new_errors), timing in analysis_results(
                compare_trace_job, jobs, executor, workers, "kv"
            ):
                rows.extend(new_rows)
                errors.extend(new_errors)
                record({"phase": "kv", "key": jobs[index][0], "rows": new_rows, "errors": new_errors, **timing})
            rows.sort(key=lambda row: (row["rank"], row["layer"], row["part"], row["comparison"]))
            snapshot()
            archive = compare_archives(root, executor, workers, record)
            if not archive["common_keys"]:
                errors.append("No common persisted baseline/P KV keys")
            summary["persisted_kv"] = archive
    except BaseException as error:
        summary["analysis_status"] = "failed"
        summary["analysis_error"] = str(error)
        snapshot()
        raise
    errors.sort()
    summary["analysis_status"] = "complete"
    summary["analysis_phase"] = "complete"
    snapshot()
    fields = [
        "comparison",
        "rank",
        "layer",
        "part",
        "baseline_rows",
        "candidate_rows",
        "matched_rows",
        "baseline_only_rows",
        "candidate_only_rows",
    ]
    for metric in ("baseline", "candidate", "baseline_abs", "candidate_abs", "diff", "abs_diff"):
        fields.extend(f"{metric}_{stat}" for stat in ("mean", "variance", "max", "nonfinite"))
    with (root / "kv_statistics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {k: v for k, v in row.items() if k != "stats"}
            for metric, stats in row.get("stats", {}).items():
                flat.update({f"{metric}_{name}": value for name, value in stats.items() if name != "count"})
            writer.writerow(flat)
    print(f"[PREFILL_CHECK] tokens_equal={mismatch is None}, first_difference={mismatch}; {root / 'kv_statistics.csv'}")
    # Print rank0 per-layer summaries too; CSV/JSON retain every TP rank.
    for row in rows:
        if row["rank"] == "rank0" and row["comparison"] in ("prefill_written", "decode_written_same_prefix"):
            stats = row.get("stats")
            if stats:
                print(
                    f"[PREFILL_KV] {row['comparison']} {row['layer']} {row['part']} "
                    f"base_mean={stats['baseline']['mean']} base_var={stats['baseline']['variance']} "
                    f"new_mean={stats['candidate']['mean']} new_var={stats['candidate']['variance']} "
                    f"abs_diff_mean={stats['abs_diff']['mean']} abs_diff_var={stats['abs_diff']['variance']}"
                )
    if errors:
        raise RuntimeError(f"Trace coverage incomplete; report saved: {errors[:3]}")
    return summary


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", default="/workspace/models/GLM-5.2-w4a8c8-0723")
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE, help="Fixed UTF-8 summarization prompt")
    cli.add_argument(
        "--prompt-tokens", type=int, help="Optional minimum length check; never pads or truncates the fixed prompt"
    )
    cli.add_argument("--output-tokens", type=int, default=4000, help="Maximum generated tokens (not words); allow EOS")
    cli.add_argument("--prefill-chunk-tokens", type=int, default=4096)
    cli.add_argument("--cpu-cache-gb", type=float, default=8)
    cli.add_argument("--run-dir", type=Path)
    cli.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)
    cli.add_argument("--analyse-only", action="store_true")
    cli.add_argument(
        "--analysis-workers",
        type=int,
        default=DEFAULT_ANALYSIS_WORKERS,
        help="CPU processes for offline KV statistics (default: %(default)s; 1 for serial analysis)",
    )
    return cli


def main():
    print(f"[PREFILL_CHECK] starting; max_model_len={MAX_MODEL_LEN}, gpu_memory_utilization=0.96", flush=True)
    args = parser().parse_args()
    if args.analysis_workers < 1:
        raise ValueError("--analysis-workers must be positive")
    if args.stage:
        run_child(args)
        return
    if args.analyse_only:
        if not args.run_dir:
            raise ValueError("--analyse-only needs --run-dir")
        analyse(args.run_dir.resolve(), len(args.devices.split(",")), args.analysis_workers)
        return
    if sys.platform != "linux":
        raise RuntimeError("Run the model test on the Linux Ascend server; CPU unit tests run separately")
    if args.output_tokens < 2:
        raise ValueError("Need at least two output tokens")
    if args.cpu_cache_gb <= 0:
        raise ValueError("CPU cache size must be positive")
    if shutil.disk_usage("/dev/shm").free < args.cpu_cache_gb * 1024**3:
        raise RuntimeError(f"Need {args.cpu_cache_gb} GiB free /dev/shm; nothing has been launched or deleted")
    if args.run_dir:
        root = args.run_dir.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(tempfile.mkdtemp(prefix="layerwise-prefill-", dir=Path.cwd()))
    print(f"[PREFILL_CHECK] results: {root}", flush=True)
    prompt_len = prepare_prompt(args, root)
    validate_sequence_length(prompt_len, args.output_tokens)
    write_json(
        root / "run.json",
        {
            **vars(args),
            "prompt_file": str(args.prompt_file.resolve()),
            "run_dir": str(root),
            "actual_prompt_tokens": prompt_len,
        },
    )
    for stage in STAGES:
        run_stage(args, root, stage)
        if stage == "prefill":
            seal_archive(root)
    analyse(root, len(args.devices.split(",")), args.analysis_workers)


if __name__ == "__main__":
    main()
