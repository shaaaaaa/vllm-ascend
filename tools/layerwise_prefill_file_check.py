#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Archive OFF prefill+decode, then test a file-backed P -> process exit -> D.

Adapted from lmy_merge_prefill_layerwise_cache's file check. Only the Mooncake
SDK is replaced; LMCache keys, merged pages, allocation and NPU transfers remain
production code. All stages use eager execution to capture real request tensors
after warmup. This is a correctness diagnostic, not a performance benchmark or a
test of native Mooncake networking, concurrent RemoteFill, or graph replay.

All settings are inline. Example:
  python3 tools/layerwise_prefill_file_check.py --model /path/to/model 2>&1 | tee log.log
  python3 tools/layerwise_prefill_file_check.py --off-dir OLD_RUN 2>&1 | tee log.log
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from layerwise_prefill_check import DEFAULT_PROMPT_FILE, normalize_prompt_token_ids, write_json
from layerwise_prefill_mooncake_check import (
    config_environment,
    deployment_config,
    finish_child,
    stage_config,
    start_logged_process,
)
from layerwise_prefill_profile import (
    DEFAULT_MODEL,
    build_prompt,
    case_environment,
    check_shm_capacity,
    record_model_identity,
)
from layerwise_prefill_profile import (
    engine_options as profile_options,
)

PREFIX = "[PREFILL_FILE]"
STAGES = ("baseline", "prefill", "decode")
SEED = 1024
SCHEMA_VERSION = 1
RUN_OPTIONS = (
    "model",
    "devices",
    "output_tokens",
    "prefill_chunk_tokens",
    "cpu_cache_gb",
    "store_gb",
    "max_model_len",
    "gpu_memory_utilization",
    "mtp_tokens",
    "flashcomm1",
)


class ExplicitOption(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.specified_options = getattr(namespace, "specified_options", frozenset()) | {self.dest}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument(
        "--model", default=DEFAULT_MODEL, action=ExplicitOption, help=f"Actual checkpoint; default: {DEFAULT_MODEL}"
    )
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7", action=ExplicitOption)
    prompt = cli.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="Literal prompt, e.g. the failing /v1/completions request")
    prompt.add_argument("--prompt-file", type=Path, help="Read a literal prompt from this file")
    cli.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    cli.add_argument(
        "--prompt-tokens",
        type=positive_int,
        help="Optional truncation for explicit prompts; default example targets 10000 tokens",
    )
    cli.add_argument("--output-tokens", default=256, type=positive_int, action=ExplicitOption)
    cli.add_argument("--prefill-chunk-tokens", default=4096, type=positive_int, action=ExplicitOption)
    cli.add_argument("--cpu-cache-gb", default=24, type=float, action=ExplicitOption)
    cli.add_argument(
        "--store-gb",
        default=8,
        type=float,
        action=ExplicitOption,
        help="SDK setup metadata only; file payloads use disk, not a native Mooncake segment",
    )
    cli.add_argument("--max-model-len", default=16384, type=positive_int, action=ExplicitOption)
    cli.add_argument("--gpu-memory-utilization", default=0.97, type=float, action=ExplicitOption)
    cli.add_argument("--mtp-tokens", default=1, type=int, choices=(0, 1), action=ExplicitOption)
    cli.add_argument(
        "--flashcomm1",
        default=1,
        type=int,
        choices=(0, 1),
        action=ExplicitOption,
        help="Same compute setting in all three stages (default: 1)",
    )
    cli.add_argument("--rpc-timeout-seconds", default=1800, type=positive_int)
    cli.add_argument(
        "--stage-timeout-seconds",
        default=21600,
        type=positive_int,
        help="Includes full tensor I/O; a timeout stops only this script's process group",
    )
    cli.add_argument("--run-dir", type=Path, help="New empty output directory")
    cli.add_argument(
        "--off-dir", type=Path, help="Completed file-check run or its baseline/ directory; skip OFF entirely"
    )
    cli.add_argument("--compare-only", type=Path, help="Reanalyse saved tensors without loading any model")
    cli.add_argument("--child", choices=STAGES, help=argparse.SUPPRESS)
    return cli


def validate_args(args):
    devices = args.devices.split(",")
    if not devices or len(devices) != len(set(devices)) or any(not x.isdecimal() for x in devices):
        raise ValueError("--devices must contain distinct numeric device IDs")
    if args.output_tokens < 2:
        raise ValueError("--output-tokens must be at least 2 to exercise decode")
    if args.cpu_cache_gb <= 0 or args.store_gb <= 0:
        raise ValueError("Cache capacities must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between 0 and 1")
    if args.off_dir and (args.prompt is not None or args.prompt_file or args.prompt_tokens):
        raise ValueError("--off-dir reuses exact saved token IDs; do not supply a new prompt")


def engine_options(args, stage, prompt_len):
    options = profile_options(args, Path("."), prompt_len)
    options.pop("profiler_config")
    options.update(
        worker_extension_cls="layerwise_prefill_file_worker.FileStoreWorker",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.prefill_chunk_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Python module hooks must execute on every actual forward, not only
        # during graph capture. Record this restriction in every report.
        enforce_eager=True,
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
        async_scheduling=False,
        disable_log_stats=False,
    )
    options["additional_config"]["enable_npugraph_ex"] = False
    options["kv_transfer_config"]["kv_role"] = {
        "baseline": "kv_both",
        "prefill": "kv_producer",
        "decode": "kv_consumer",
    }[stage]
    if args.mtp_tokens:
        options["speculative_config"] = {"method": "deepseek_mtp", "num_speculative_tokens": args.mtp_tokens}
    else:
        options.pop("speculative_config")
    return options


def child_environment(args, root, stage):
    env = case_environment(args, "10k_on" if stage == "prefill" else "10k_off")
    env = {key: value for key, value in env.items() if not key.startswith(("LMCACHE_", "MOONCAKE_"))}
    for key in ("MC_FORCE_TCP", "MC_FORCE_SHM", "MC_STORE_MEMCPY"):
        env.pop(key, None)
    config = stage_config(deployment_config("127.0.0.1:1", "127.0.0.1"), args, stage)
    # File-backed SDK setup never creates a real Mooncake/ADXL connection.
    if stage != "baseline":
        config["extra_config"]["prefill_check_file_sdk"] = {"root": str(root.resolve()), "stage": stage}
    env.update(config_environment(config))
    env.update(
        {
            "PYTHONPATH": str(root / "bootstrap") + os.pathsep + env["PYTHONPATH"],
            "HCCL_DETERMINISTIC": "strict",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": str(args.flashcomm1),
            "VLLM_ASCEND_DSA_SHRINK_LATENT": "0" if stage == "prefill" else "2",
            "VLLM_ASCEND_SFA_STAGED_GRAPH": "0",
            "VLLM_ASCEND_SFA_STAGED_MTP_DRAFT_GRAPH": "0",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": str(args.rpc_timeout_seconds),
            "VLLM_ENGINE_READY_TIMEOUT_S": str(args.rpc_timeout_seconds),
        }
    )
    return env


def recorded_environment(env):
    return {
        key: value
        for key, value in env.items()
        if key.startswith(("VLLM_", "LMCACHE_", "HCCL_", "ASCEND_"))
        or key in ("PYTHONPATH", "PYTHONHASHSEED", "PD_SERVING_PERF")
    }


def prepare_bootstrap(root):
    bootstrap = root / "bootstrap"
    bootstrap.mkdir()
    # Spawned EngineCore, lookup and TP processes install before importing the
    # native SDK. A broken shim must stop execution, never fall back to ADXL.
    (bootstrap / "sitecustomize.py").write_text(
        "import os, traceback\n"
        "try:\n"
        "    from layerwise_prefill_file_store import install\n"
        "    install()\n"
        "except BaseException:\n"
        "    traceback.print_exc()\n"
        "    os._exit(1)\n",
        encoding="utf-8",
    )


def prepare_prompt(args, root):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.prompt is None and args.prompt_file is None:
        article = DEFAULT_PROMPT_FILE.read_text(encoding="utf-8")
        text, ids = build_prompt(tokenizer, article, args.prompt_tokens or 10000)
        prompt_format = "chat"
    else:
        text = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
        prompt_format = args.prompt_format
        if prompt_format == "chat":
            ids = normalize_prompt_token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            )
        else:
            ids = normalize_prompt_token_ids(tokenizer.encode(text, add_special_tokens=True))
        if args.prompt_tokens is not None:
            ids = ids[: args.prompt_tokens]
    if len(ids) < 2 or len(ids) + args.output_tokens > args.max_model_len:
        raise ValueError(f"Prompt has {len(ids)} tokens; require >=2 and prompt + outputs <= max-model-len")
    (root / "prompt.txt").write_text(text, encoding="utf-8")
    write_json(root / "prompt.json", {"token_ids": ids, "length": len(ids), "format": prompt_format})
    return len(ids)


def prepare_reused_off(args, root):
    from layerwise_prefill_file_compare import validate_baseline

    baseline = args.off_dir.expanduser().resolve(strict=True)
    if not (baseline / "output.json").is_file():
        baseline = baseline / "baseline"
    previous_root = baseline.parent
    if root.resolve().is_relative_to(baseline):
        raise ValueError("New run directory must be outside the reused OFF archive")
    config_path = previous_root / "run_config.json"
    if not config_path.is_file():
        raise ValueError("--off-dir requires this file-check tool's full prefill+decode archive, not prefill-only OFF")
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    if saved.get("schema_version") != SCHEMA_VERSION or saved.get("tool") != "layerwise_prefill_file_check":
        raise ValueError("Unsupported OFF archive schema")
    for key in RUN_OPTIONS:
        if key not in getattr(args, "specified_options", frozenset()):
            setattr(args, key, saved["options"][key])
        elif getattr(args, key) != saved["options"][key]:
            raise ValueError(f"--{key.replace('_', '-')} differs from OFF; record a new OFF or use matching settings")
    validate_args(args)
    status = validate_baseline(baseline, len(args.devices.split(",")), args.output_tokens)
    if not status.get("valid", status.get("complete", False)):
        raise ValueError(f"Invalid OFF baseline: {status.get('errors', [])[:8]}")
    prompt = json.loads((previous_root / "prompt.json").read_text(encoding="utf-8"))
    output = json.loads((baseline / "output.json").read_text(encoding="utf-8"))
    if prompt["token_ids"] != output["prompt_token_ids"] or prompt["length"] != len(prompt["token_ids"]):
        raise ValueError("OFF prompt token IDs do not match its completed output")
    if prompt["length"] + args.output_tokens > args.max_model_len:
        raise ValueError("Saved prompt and requested decode exceed max-model-len")
    expected_options = engine_options(args, "baseline", prompt["length"])
    if json.loads((baseline / "engine_options.json").read_text(encoding="utf-8")) != expected_options:
        raise ValueError("OFF engine configuration differs from this version; use matching code or record a new OFF")
    model_info = record_model_identity(args.model, root)
    if json.loads((previous_root / "model_info.json").read_text(encoding="utf-8")) != model_info:
        raise ValueError("OFF checkpoint configuration differs from the current model")
    write_json(root / "prompt.json", prompt)
    if (previous_root / "prompt.txt").is_file():
        shutil.copyfile(previous_root / "prompt.txt", root / "prompt.txt")
    write_json(root / "off_reference.json", {"schema_version": SCHEMA_VERSION, "baseline_dir": str(baseline)})
    args.off_dir = baseline
    print(f"{PREFIX} reusing OFF: {baseline}; OFF launch skipped", flush=True)
    return prompt["length"]


def mtp_snapshot(llm):
    names = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens")
    metrics = llm.get_metrics()
    values = {name: [float(metric.value) for metric in metrics if metric.name == name] for name in names}
    return {name: sum(items) if items else None for name, items in values.items()}


def load_worker_coverage(stage_dir, replies, tp_size):
    """RPC returns small manifests; large per-call inventories stay on disk."""
    if not isinstance(replies, list) or sorted(row.get("rank", -1) for row in replies) != list(range(tp_size)):
        raise RuntimeError("Worker coverage ranks are missing, duplicated or invalid")
    coverage = []
    for reply in replies:
        expected_path = f"tensors/rank{reply['rank']}/coverage.json"
        if reply.get("coverage_path") != expected_path:
            raise RuntimeError("Worker returned an invalid coverage path")
        path = stage_dir / expected_path
        if not path.resolve().is_relative_to(stage_dir.resolve()):
            raise RuntimeError("Worker coverage path escaped the stage directory")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if any(summary.get(key) != reply.get(key) for key in ("rank", "complete", "errors", "records")):
            raise RuntimeError("Worker coverage file differs from its RPC manifest")
        coverage.append(summary)
    return coverage


def run_child(args):
    from layerwise_prefill_file_store import install

    install()
    from vllm import LLM, SamplingParams

    root = args.run_dir.resolve()
    stage = args.child
    stage_dir = root / stage
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    options = engine_options(args, stage, prompt["length"])
    write_json(stage_dir / "engine_options.json", options)
    print(f"{PREFIX} {stage}: model={args.model}; capture starts after warmup; eager=True", flush=True)
    llm = LLM(**options)
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "completed": False,
        "prompt_token_ids": prompt["token_ids"],
        "prompt_length": prompt["length"],
        "token_ids": [],
        "output_token_limit": 1 if stage == "prefill" else args.output_tokens,
        "enforce_eager": True,
        "timing_is_performance_data": False,
    }
    try:
        installed = llm.collective_rpc(
            "install_file_probe", timeout=args.rpc_timeout_seconds, args=(str(stage_dir), prompt["token_ids"])
        )
        write_json(stage_dir / "installation.json", installed)
        before = mtp_snapshot(llm) if args.mtp_tokens else {}
        started = time.perf_counter()
        results = llm.generate(
            {"prompt_token_ids": prompt["token_ids"]},
            SamplingParams(temperature=0, seed=SEED, max_tokens=report["output_token_limit"], ignore_eos=True),
            use_tqdm=False,
        )
        if len(results) != 1 or len(results[0].outputs) != 1 or not results[0].finished:
            raise RuntimeError("Expected one completed request")
        result = results[0]
        completion = result.outputs[0]
        after = mtp_snapshot(llm) if args.mtp_tokens else {}
        report.update(
            token_ids=list(completion.token_ids),
            text=completion.text,
            num_cached_tokens=result.num_cached_tokens,
            finish_reason=completion.finish_reason,
            diagnostic_seconds=time.perf_counter() - started,
            mtp={
                "configured_tokens": args.mtp_tokens,
                "metrics": {
                    key: after[key] - before[key] if after[key] is not None and before[key] is not None else None
                    for key in before
                },
            },
        )
        # This is a test teardown wait outside inference. The actual production
        # async store path still owns DMA event waits/publication in its worker.
        flushed = llm.collective_rpc("flush_file_store", timeout=args.rpc_timeout_seconds)
        write_json(stage_dir / "store_flush.json", flushed)
        replies = llm.collective_rpc("finish_file_probe", timeout=args.rpc_timeout_seconds)
        coverage = load_worker_coverage(stage_dir, replies, len(args.devices.split(",")))
        write_json(stage_dir / "coverage.json", coverage)
        if len(report["token_ids"]) != report["output_token_limit"]:
            raise RuntimeError("Generation did not reach the requested token count")
        if args.mtp_tokens and stage != "prefill":
            verified = report["mtp"]["metrics"]["vllm:spec_decode_num_draft_tokens"]
            if verified is None or verified <= 0:
                raise RuntimeError("MTP was configured but no target verification was observed")
        expected_cached = prompt["length"] - 1 if stage == "decode" else 0
        if report["num_cached_tokens"] != expected_cached:
            raise RuntimeError(
                f"{stage} cached_tokens={report['num_cached_tokens']}, expected {expected_cached}; "
                "D must restore P's complete prefix, not recompute it"
            )
        ranks = sorted(row.get("rank", -1) for row in coverage)
        if ranks != list(range(len(args.devices.split(",")))) or any(not row.get("complete") for row in coverage):
            raise RuntimeError(f"Incomplete tensor coverage; inspect {stage_dir / 'coverage.json'}")
        report["completed"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        error_in_flight = sys.exc_info()[0] is not None
        try:
            # NPUWorker currently inherits WorkerBase's empty shutdown method.
            # Close this instance's connector/allocator explicitly; otherwise
            # a named shared slab can survive into the next model stage.
            llm.collective_rpc("close_file_store", timeout=args.rpc_timeout_seconds)
        except BaseException as exc:
            report["completed"] = False
            report["close_error"] = f"{type(exc).__name__}: {exc}"
            if not error_in_flight:
                raise
        finally:
            write_json(stage_dir / "output.json", report)
            (stage_dir / "output.txt").write_text(report.get("text", ""), encoding="utf-8")
            llm.llm_engine.engine_core.shutdown()
    print(
        f"{PREFIX} {stage}: output_tokens={len(report['token_ids'])}; cache_hit={report['num_cached_tokens']}",
        flush=True,
    )


def child_command(args, root, stage):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", stage, "--run-dir", str(root)]
    for key in (*RUN_OPTIONS, "rpc_timeout_seconds", "stage_timeout_seconds"):
        command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    return command


def run_stages(args, root):
    from layerwise_prefill_file_store import seal_store

    for stage in ("prefill", "decode") if args.off_dir else STAGES:
        stage_dir = root / stage
        stage_dir.mkdir()
        check_shm_capacity(Path("/dev/shm"), args.cpu_cache_gb)
        env = child_environment(args, root, stage)
        write_json(stage_dir / "environment.json", recorded_environment(env))
        proc = start_logged_process(
            child_command(args, root, stage), env, stage_dir / "server.log", stage, prefix=PREFIX
        )
        try:
            try:
                code = proc.wait(timeout=args.stage_timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{stage} exceeded --stage-timeout-seconds; see {stage_dir / 'server.log'}") from exc
            if code:
                raise RuntimeError(f"{stage} exited with code {code}; see {stage_dir / 'server.log'}")
        finally:
            finish_child(proc)
        # The child and its engine/TP process group have stopped BEFORE sealing
        # or starting D. No shared CPU object from P can satisfy a D cache hit.
        write_json(stage_dir / "process-exited.json", {"pid": proc.pid, "returncode": proc.returncode})
        output = json.loads((stage_dir / "output.json").read_text(encoding="utf-8"))
        if not output.get("completed"):
            raise RuntimeError(f"{stage} did not complete; refusing to start the next instance")
        if stage == "prefill":
            manifest = seal_store(root)
            if not manifest.get("passed"):
                raise RuntimeError(f"Cannot seal P files: {manifest.get('errors')}")
            print(f"{PREFIX} P process exited; store sealed; starting independent D", flush=True)


def analyse(root):
    from layerwise_prefill_file_compare import compare_run
    from layerwise_prefill_file_store import validate_store

    store = validate_store(root)
    write_json(root / "store-report.json", store)
    report = compare_run(root)
    report["store"] = store
    report["passed"] = bool(report.get("passed") and store.get("passed"))
    report["scope"] = (
        "Single-request TP eager OFF vs file-backed sequential P/D; full recorded tensors and output tokens. "
        "No native Mooncake/network, concurrent RemoteFill, multi-host DP/EP or graph replay validation."
    )
    if not store.get("passed"):
        report.setdefault("errors", []).extend(store.get("errors", []))
        report["status"] = "failed"
    write_json(root / "report.json", report)
    print(
        f"{PREFIX} passed={report['passed']} (structural/output/nonfinite gates, not float tolerance); "
        f"report: {root / 'report.json'}",
        flush=True,
    )
    return 0 if report["passed"] else 1


def main(argv=None):
    args = parser().parse_args(argv)
    if args.compare_only:
        return analyse(args.compare_only.expanduser().resolve(strict=True))
    if args.child:
        run_child(args)
        return 0
    if sys.platform != "linux":
        raise RuntimeError("Model execution requires the Linux Ascend environment; --help and CPU tests work locally")
    root = (args.run_dir or Path(tempfile.mkdtemp(prefix="layerwise-file-check-", dir=Path.cwd()))).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("--run-dir must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.off_dir:
            length = prepare_reused_off(args, root)
        else:
            validate_args(args)
            record_model_identity(args.model, root)
            length = prepare_prompt(args, root)
        validate_args(args)
        write_json(
            root / "run_config.json",
            {
                "schema_version": SCHEMA_VERSION,
                "tool": "layerwise_prefill_file_check",
                "tp_size": len(args.devices.split(",")),
                "expected_output_tokens": args.output_tokens,
                "mtp_tokens": args.mtp_tokens,
                "main_num_layers": json.loads((root / "model_info.json").read_text(encoding="utf-8"))[
                    "num_hidden_layers"
                ],
                "options": {key: getattr(args, key) for key in RUN_OPTIONS},
                "sampling": {"temperature": 0, "seed": SEED, "ignore_eos": True},
            },
        )
        print(
            f"{PREFIX} model={args.model}; prompt={length}; decode={args.output_tokens}; "
            f"TP={len(args.devices.split(','))}; "
            f"MTP={args.mtp_tokens}; artifacts={root}",
            flush=True,
        )
        prepare_bootstrap(root)
        run_stages(args, root)
        return analyse(root)
    except BaseException as exc:
        write_json(root / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        print(f"{PREFIX} FAILED: {exc}; artifacts: {root}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
