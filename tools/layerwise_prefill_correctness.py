#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Save OFF prefill tensors, then compare ON tensors to those files while running.

Usage: python tools/layerwise_prefill_correctness.py 2>&1 | tee log.log

Both fresh processes use TP8, FlashComm1, eager execution, MTP1 and one identical
tokenized prompt. The default 10k prompt spans multiple 4096-token compute chunks
and a partial LMCache chunk. Use --prompt-tokens 80000 for the long-prefix case.
There are no stage/dummy shortcuts or external configuration files.

The tools-only layout selector enables the real merged CPU-page allocator in a
local-only test: no Mooncake service/SDK emulation and no transfer replacement.
OFF uses synchronous store because its original local path rejects async store;
ON exercises async store. All computation settings and the page layout match.

Full tensor readback perturbs execution; this is not a performance test or proof
of race freedom. OFF saves complete tensors, not samples or fingerprints. ON
loads each matching file and reports value distributions and numerical error.
--save-on-tensors additionally keeps ON tensors. OFF archives can be very large.
Use --off-dir OLD_RUN (or OLD_RUN/off) to reuse a completed OFF archive and run
only ON. The saved prompt token IDs and unspecified launch settings are reused.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from layerwise_prefill_check import DEFAULT_PROMPT_FILE, write_json
from layerwise_prefill_mooncake_check import finish_child, start_logged_process
from layerwise_prefill_profile import (
    COMPUTE_CHUNK_TOKENS,
    DEFAULT_LONG_PROMPT_FILE,
    DEFAULT_MODEL,
    build_prompt,
    case_environment,
    check_shm_capacity,
    engine_options,
    record_model_identity,
)

PREFIX = "[PREFILL_CORRECTNESS]"
CASES = ("off", "on")
DEFAULT_PROMPT_TOKENS = 10000
MAX_PROMPT_TOKENS = 80000
DEFAULT_RPC_TIMEOUT_SECONDS = 1800
SEED = 1024
DECODE_QUERY_THRESHOLD = 2  # Main model plus the configured single MTP token.
ENVIRONMENT_PREFIXES = ("LMCACHE_", "VLLM_", "HCCL_", "ASCEND_", "OMP_", "PYTORCH_NPU_")
ENVIRONMENT_KEYS = frozenset(
    {
        "PYTHONHASHSEED",
        "TASK_QUEUE_ENABLE",
        "CPU_AFFINITY_CONF",
        "ACL_OP_INIT_MODE",
        "PD_SERVING_PERF",
        "MSMONITOR_USE_DAEMON",
    }
)


class ExplicitOption(argparse.Action):
    """Keep ordinary defaults while tracking user overrides for OFF reuse."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.specified_options = getattr(namespace, "specified_options", frozenset()) | {self.dest}


def positive_seconds(value):
    seconds = int(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("RPC timeout must be a positive number of seconds")
    return seconds


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        action=ExplicitOption,
        help="Actual checkpoint directory; defaults to GLM-5.2-w4a8c8-0723",
    )
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7", action=ExplicitOption)
    cli.add_argument("--prompt-file", type=Path, help="Fixed source article; tokenized once for both runs")
    cli.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS, action=ExplicitOption)
    cli.add_argument("--cpu-cache-gb", type=float, default=24, action=ExplicitOption)
    cli.add_argument(
        "--rpc-timeout-seconds",
        type=positive_seconds,
        default=DEFAULT_RPC_TIMEOUT_SECONDS,
        help="Per-RPC deadline including queued forward and full tensor I/O (default: 1800)",
    )
    cli.add_argument("--run-dir", type=Path, help="New, empty results directory")
    cli.add_argument("--off-dir", type=Path, help="Previous correctness run or its off/ directory; run ON only")
    cli.add_argument(
        "--save-on-tensors",
        action="store_true",
        help="Also save ON tensors; OFF always saves complete tensors",
    )
    cli.add_argument("--compare-only", type=Path, help="Compare an existing run without loading the model")
    cli.add_argument("--child", choices=CASES, help=argparse.SUPPRESS)
    return cli


def correctness_environment(args, case):
    """Keep the established P compute settings; select real local merged pages."""
    if case not in CASES:
        raise ValueError(f"Unknown correctness case: {case}")
    env = case_environment(args, f"10k_{case}")
    env.pop("LMCACHE_PREFILL_REUSE_DEBUG_RANK", None)
    env.pop("LMCACHE_PREFILL_REUSE_DEBUG_FILE", None)
    # The tools-only selector keeps the actual page allocator while omitting
    # the remote URL. No fake SDK or file-store connector is installed.
    env["LMCACHE_EXTRA_CONFIG"] = json.dumps(
        {
            "save_only_first_rank": True,
            "save_chunk_meta": False,
            "mooncake_page_first_multi_buffer": True,
            "mooncake_layer_merged_page_objects": True,
        },
        sort_keys=True,
    )
    env["HCCL_DETERMINISTIC"] = "strict"
    env["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(args.rpc_timeout_seconds)
    return env


def correctness_options(args, prompt_length):
    """Exactly the same model configuration in both fresh model processes."""
    options = engine_options(args, Path("."), prompt_length)
    options.pop("profiler_config")
    options["worker_extension_cls"] = "layerwise_prefill_correctness_worker.PrefillCorrectnessWorker"
    if prompt_length + 1 > options["max_model_len"]:
        raise ValueError("Prompt and first output token exceed max_model_len")
    return options


def recorded_environment(env):
    return {key: value for key, value in env.items() if key.startswith(ENVIRONMENT_PREFIXES) or key in ENVIRONMENT_KEYS}


def require_matching_settings(label, expected, actual):
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise ValueError(f"OFF {label} must be a configuration object")
    changed = sorted(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
    if changed:
        raise ValueError(
            f"OFF baseline {label} differs: {', '.join(changed)}; use matching settings or record a new OFF"
        )


def prepare_reused_off(args, root):
    """Validate a read-only OFF archive before loading a model or launching ON."""
    from layerwise_prefill_correctness_baseline import normalize_off_directory
    from layerwise_prefill_correctness_compare import comparable_environment, validate_off_baseline

    off_dir = normalize_off_directory(args.off_dir)
    if root.resolve().is_relative_to(off_dir):
        raise ValueError("The new --run-dir must be outside the reused OFF archive")
    previous_root = off_dir.parent
    previous_model = json.loads((previous_root / "model_info.json").read_text(encoding="utf-8"))
    baseline = validate_off_baseline(off_dir, previous_model)
    saved_prompt_path = previous_root / "prompt.json"
    saved_prompt = json.loads(saved_prompt_path.read_text(encoding="utf-8")) if saved_prompt_path.is_file() else {}
    result = baseline["result"]
    length = result["prompt_length"]
    validate_prompt_length(length)
    if length > MAX_PROMPT_TOKENS:
        raise ValueError(f"OFF prompt exceeds the supported {MAX_PROMPT_TOKENS} tokens")
    if saved_prompt and saved_prompt.get("token_ids") != result["prompt_token_ids"]:
        raise ValueError("OFF prompt.json token IDs differ from its completed result")
    defaults = {
        "model": baseline["engine_options"]["model"],
        "devices": baseline["environment"]["ASCEND_RT_VISIBLE_DEVICES"],
        "cpu_cache_gb": float(baseline["environment"]["LMCACHE_MAX_LOCAL_CPU_SIZE"]),
        "prompt_tokens": saved_prompt.get("target_tokens", length),
    }
    specified = getattr(args, "specified_options", frozenset())
    for name, value in defaults.items():
        if name not in specified:
            setattr(args, name, value)
    if args.prompt_tokens != defaults["prompt_tokens"]:
        raise ValueError("--prompt-tokens differs from OFF; reuse uses its exact saved token IDs")
    require_matching_settings("engine_options", baseline["engine_options"], correctness_options(args, length))
    require_matching_settings(
        "environment",
        comparable_environment(baseline["environment"]),
        comparable_environment(recorded_environment(correctness_environment(args, "off"))),
    )
    current_model = record_model_identity(args.model, root)
    require_matching_settings("model configuration", previous_model, current_model)
    args.off_dir = off_dir
    write_json(
        root / "prompt.json",
        {
            **saved_prompt,
            "length": length,
            "token_ids": result["prompt_token_ids"],
            "target_tokens": args.prompt_tokens,
        },
    )
    if (previous_root / "prompt.txt").is_file():
        shutil.copyfile(previous_root / "prompt.txt", root / "prompt.txt")
    write_json(root / "off_reference.json", {"schema": 1, "off_dir": str(off_dir)})
    print(f"{PREFIX} reuse OFF: {off_dir}; archive validated; OFF launch skipped", flush=True)
    return length


def validate_prompt_length(length):
    if length <= COMPUTE_CHUNK_TOKENS:
        raise ValueError("Correctness validation requires multiple compute-prefill chunks")
    remainder = length % COMPUTE_CHUNK_TOKENS
    if 0 < remainder <= DECODE_QUERY_THRESHOLD:
        raise ValueError(
            f"Actual prompt has a {remainder}-token final chunk, which uses the decode KV remapping path. "
            "This probe covers multi-token prefill; choose a different --prompt-tokens target."
        )


def prepare_prompt(args, root):
    from transformers import AutoTokenizer

    source = args.prompt_file or (
        DEFAULT_LONG_PROMPT_FILE if args.prompt_tokens > DEFAULT_PROMPT_TOKENS else DEFAULT_PROMPT_FILE
    )
    article = source.read_text(encoding="utf-8")
    if not article.strip():
        raise ValueError(f"Empty prompt file: {source}")
    print(f"{PREFIX} tokenize once: {source}", flush=True)
    record_model_identity(args.model, root)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text, ids = build_prompt(tokenizer, article, args.prompt_tokens)
    validate_prompt_length(len(ids))
    (root / "prompt.txt").write_text(text, encoding="utf-8")
    write_json(
        root / "prompt.json",
        {
            "length": len(ids),
            "token_ids": ids,
            "source": str(source.resolve()),
            "target_tokens": args.prompt_tokens,
        },
    )
    return len(ids)


def compare(root):
    from layerwise_prefill_correctness_compare import compare_runs, print_report

    report = compare_runs(root)
    print_report(report)
    return 0 if report["passed"] else 1


def run_child(args):
    """Install only explicitly selected tool hooks, after normal model warmup."""
    from layerwise_prefill_correctness_compare import require_worker_completion
    from layerwise_prefill_correctness_layout import install_local_merged_layout

    os.environ.pop("LMCACHE_CONFIG_FILE", None)
    os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(args.rpc_timeout_seconds)
    install_local_merged_layout()
    from vllm import LLM, SamplingParams

    root = args.run_dir.resolve()
    case_dir = root / args.child
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    options = correctness_options(args, prompt["length"])
    write_json(case_dir / "engine_options.json", options)
    print(f"{PREFIX} {args.child}: model={args.model}; RPC timeout={args.rpc_timeout_seconds}s", flush=True)
    print(f"{PREFIX} {args.child}: loading full model (no capture during warmup)", flush=True)
    llm = LLM(**options)
    report = {
        "case": args.child,
        "completed": False,
        "prompt_length": prompt["length"],
        "prompt_token_ids": prompt["token_ids"],
        "output_length": 0,
        "token_ids": [],
        "text": "",
        "scope": "all main-backbone prefill layers and first output token; local merged CPU pages",
        "timing_is_performance_data": False,
    }
    try:
        installed = llm.collective_rpc(
            "install_correctness_probe",
            args=(str(case_dir), prompt["length"], args.save_on_tensors),
        )
        write_json(case_dir / "installation.json", installed)
        print(f"{PREFIX} {args.child}: capture {prompt['length']} tokens on every TP rank", flush=True)
        started = time.perf_counter()
        results = llm.generate(
            {"prompt_token_ids": prompt["token_ids"]},
            SamplingParams(temperature=0, seed=SEED, max_tokens=1),
            use_tqdm=False,
        )
        if len(results) != 1 or len(results[0].outputs) != 1:
            raise RuntimeError("Expected exactly one completed request and one output")
        result = results[0]
        completion = result.outputs[0]
        if not result.finished or len(completion.token_ids) != 1:
            raise RuntimeError("Expected a completed prefill with exactly one output token")
        if result.num_cached_tokens:
            raise RuntimeError("Fresh OFF/ON validation unexpectedly reused a prompt cache hit")
        coverage = llm.collective_rpc("finish_correctness_probe")
        write_json(case_dir / "coverage.json", coverage)
        report.update(
            {
                "output_length": len(completion.token_ids),
                "token_ids": list(completion.token_ids),
                "text": completion.text,
                "num_cached_tokens": result.num_cached_tokens,
                "diagnostic_seconds": time.perf_counter() - started,
            }
        )
        valid_ranks = isinstance(coverage, list) and all(
            isinstance(summary, dict) and type(summary.get("rank")) is int for summary in coverage
        )
        if not valid_ranks or sorted(summary["rank"] for summary in coverage) != list(
            range(options["tensor_parallel_size"])
        ):
            raise RuntimeError(f"Worker coverage ranks missing/invalid/duplicated; see {case_dir / 'coverage.json'}")
        for summary in coverage:
            require_worker_completion(summary)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        write_json(case_dir / "result.json", report)
        raise
    finally:
        llm.llm_engine.engine_core.shutdown()
    report["completed"] = True
    write_json(case_dir / "result.json", report)
    print(f"{PREFIX} {args.child}: complete; worker coverage saved", flush=True)


def run_cases(args, root):
    for case in ("on",) if args.off_dir else CASES:
        case_dir = root / case
        case_dir.mkdir()
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--child",
            case,
            "--run-dir",
            str(root),
            "--model",
            args.model,
            "--devices",
            args.devices,
            "--cpu-cache-gb",
            str(args.cpu_cache_gb),
            "--prompt-tokens",
            str(args.prompt_tokens),
            "--rpc-timeout-seconds",
            str(args.rpc_timeout_seconds),
        ]
        if args.save_on_tensors:
            command.append("--save-on-tensors")
        env = correctness_environment(args, case)
        write_json(
            case_dir / "environment.json",
            recorded_environment(env),
        )
        # Never clear all of /dev/shm: only this process group's normal engine
        # cleanup owns its resources. A failed run stops before the next case.
        check_shm_capacity(Path("/dev/shm"), args.cpu_cache_gb)
        proc = start_logged_process(command, env, case_dir / "server.log", case, prefix=PREFIX)
        try:
            code = proc.wait()
        finally:
            finish_child(proc)
        if code:
            raise RuntimeError(f"{case} failed ({code}); see {case_dir / 'server.log'}")
        result_path = case_dir / "result.json"
        if not result_path.is_file() or not json.loads(result_path.read_text())["completed"]:
            raise RuntimeError(f"{case} did not complete; refusing to report parity")


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.compare_only:
        if args.off_dir:
            cli.error("--compare-only reads the saved OFF reference; do not combine it with --off-dir")
        return compare(args.compare_only.resolve())
    if args.off_dir and args.prompt_file:
        cli.error("--off-dir uses saved prompt token IDs; do not combine it with --prompt-file")
    if not COMPUTE_CHUNK_TOKENS < args.prompt_tokens <= MAX_PROMPT_TOKENS:
        cli.error(f"--prompt-tokens must be in ({COMPUTE_CHUNK_TOKENS}, {MAX_PROMPT_TOKENS}]")
    if args.cpu_cache_gb <= 0:
        cli.error("--cpu-cache-gb must be positive")
    devices = args.devices.split(",")
    if not devices or any(not item.isdigit() for item in devices) or len(set(devices)) != len(devices):
        cli.error("--devices must contain distinct comma-separated NPU indices")
    if args.child:
        if args.run_dir is None:
            cli.error("Internal --child requires --run-dir")
        run_child(args)
        return 0
    if not sys.platform.startswith("linux"):
        cli.error("Run model validation on the Linux Ascend server; --compare-only also works on CPU")
    root = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else Path(tempfile.mkdtemp(prefix="layerwise-correctness-", dir=".")).resolve()
    )
    if root.exists() and any(root.iterdir()):
        cli.error(f"Results directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    args.run_dir = root
    try:
        count = prepare_reused_off(args, root) if args.off_dir else prepare_prompt(args, root)
        write_json(
            root / "run.json",
            {
                "schema": 1,
                "model": args.model,
                "devices": args.devices.split(","),
                "compute_chunk_tokens": COMPUTE_CHUNK_TOKENS,
                "rpc_timeout_seconds": args.rpc_timeout_seconds,
                "save_on_tensors": args.save_on_tensors,
                "off_source": str(args.off_dir) if args.off_dir else None,
                "cases": ["on"] if args.off_dir else list(CASES),
                "comparison": "OFF full tensors; ON online numerical errors and distributions; output tokens exact",
                "layout": "tools-only local merged-page selection; real allocator and transfers",
                "case_differences": ["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "LMCACHE_STORE_ASYNC"],
                "excluded": ["Mooncake transport", "MTP/decode intermediate tensors"],
            },
        )
        sequence = "ON only against saved OFF" if args.off_dir else "OFF then ON"
        print(f"{PREFIX} {count} tokens; {sequence}; results: {root}", flush=True)
        print(f"{PREFIX} full tensor probes perturb timing; no performance claims", flush=True)
        run_cases(args, root)
        return compare(root)
    except (Exception, KeyboardInterrupt) as error:
        write_json(root / "failure.json", {"error": f"{type(error).__name__}: {error}", "passed": False})
        print(f"{PREFIX} FAILED: {error}; artifacts: {root}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
