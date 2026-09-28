#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the actual PD tensor recorder on one Ascend host, with real model P/D.

Default: OFF P -> exit -> D, then ON P -> exit -> D, each using all eight NPUs.
Only the Mooncake SDK is replaced by the existing file transport. No old tensor
probe is installed: observations come from VLLM_ASCEND_PD_TENSOR_DUMP_DIR in the
production runner. This checks recording and sequential reload, not networking,
multi-host DP, graph replay or unobserved asynchronous races.

    python3 tools/pd_tensor_smoke.py 2>&1 | tee log.log
    python3 tools/pd_tensor_smoke.py --case on
    python3 tools/pd_tensor_smoke.py --dry-run
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from layerwise_prefill_check import write_json
from layerwise_prefill_file_check import (
    child_environment as file_environment,
)
from layerwise_prefill_file_check import (
    engine_options as file_options,
)
from layerwise_prefill_file_check import (
    mtp_snapshot,
    positive_int,
    prepare_bootstrap,
    prepare_prompt,
    recorded_environment,
)
from layerwise_prefill_mooncake_check import finish_child, start_logged_process
from layerwise_prefill_profile import DEFAULT_MODEL, check_shm_capacity, record_model_identity

PREFIX = "[PD_TENSOR_SMOKE]"
STAGES = ("prefill", "decode")
SEED = 1024
CHILD_OPTIONS = (
    "model",
    "devices",
    "output_tokens",
    "prefill_chunk_tokens",
    "cpu_cache_gb",
    "store_gb",
    "max_model_len",
    "gpu_memory_utilization",
    "mtp_tokens",
    "prefill_flashcomm1",
    "decode_flashcomm1",
    "rpc_timeout_seconds",
    "stage_timeout_seconds",
)


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--model", default=DEFAULT_MODEL, help=f"Default: {DEFAULT_MODEL}")
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--case", choices=("all", "off", "on"), default="all")
    prompts = cli.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", help="Optional literal prompt, also useful for a short-tail regression")
    prompts.add_argument("--prompt-file", type=Path)
    cli.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    cli.add_argument(
        "--prompt-tokens",
        type=positive_int,
        default=4608,
        help="Default article target; spans two 4096-token chunks (default: 4608)",
    )
    cli.add_argument("--output-tokens", type=positive_int, default=16)
    cli.add_argument("--prefill-chunk-tokens", type=positive_int, default=4096)
    cli.add_argument("--cpu-cache-gb", type=float, default=24)
    cli.add_argument("--store-gb", type=float, default=8, help="File SDK metadata only, no native segment allocation")
    cli.add_argument("--max-model-len", type=positive_int, default=16384)
    cli.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    cli.add_argument("--mtp-tokens", type=int, choices=(0, 1), default=1)
    cli.add_argument("--prefill-flashcomm1", type=int, choices=(0, 1), default=1)
    cli.add_argument("--decode-flashcomm1", type=int, choices=(0, 1), default=0)
    cli.add_argument("--rpc-timeout-seconds", type=positive_int, default=3600)
    cli.add_argument("--stage-timeout-seconds", type=positive_int, default=21600)
    cli.add_argument("--run-dir", type=Path, help="New empty output directory")
    cli.add_argument("--dry-run", action="store_true", help="Print configuration without importing NPU/model packages")
    cli.add_argument("--analyze-only", type=Path, help="Validate and reanalyse an existing run, using CPU PyTorch")
    cli.add_argument("--child", choices=STAGES, help=argparse.SUPPRESS)
    return cli


def validate_args(args):
    devices = args.devices.split(",")
    if not devices or any(not item.isdecimal() for item in devices) or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct numeric device IDs")
    if args.output_tokens < 2:
        raise ValueError("At least two output tokens are required to exercise decode")
    if not 0 < args.gpu_memory_utilization < 1 or not (args.cpu_cache_gb > 0 and args.store_gb > 0):
        raise ValueError("Invalid memory capacities/utilization")
    if args.output_tokens >= args.max_model_len:
        raise ValueError("Output token limit leaves no room for a prompt")
    if args.child and (args.case == "all" or args.run_dir is None):
        raise ValueError("A child needs exactly one --case and an explicit --run-dir")


def selected_cases(args):
    return ("off", "on") if args.case == "all" else (args.case,)


def engine_options(args, stage, prompt_len):
    # FileStoreWorker is used ONLY for transport flush/close, never its old
    # install_file_probe/finish_file_probe hooks. The production runner installs
    # the new diagnostic automatically, before the request reaches the model.
    return file_options(args, stage, prompt_len)


def child_environment(args, root, case, stage):
    role_args = argparse.Namespace(**vars(args))
    role_args.flashcomm1 = args.prefill_flashcomm1 if stage == "prefill" else args.decode_flashcomm1
    env = file_environment(role_args, root, stage)
    env["VLLM_ASCEND_PD_TENSOR_DUMP_DIR"] = str((root / "capture").resolve())
    env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] = str(stage == "prefill" and case == "on").lower()
    env.pop("LMCACHE_PREFILL_REUSE_DEBUG_RANK", None)
    env.pop("LMCACHE_PREFILL_REUSE_DEBUG_FILE", None)
    # Keep the file-PD transport's async store in both P cases. OFF uses its
    # existing direct-store implementation; D keeps the original consumer path.
    return env


def child_command(args, root, case, stage):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        stage,
        "--case",
        case,
        "--run-dir",
        str(root),
    ]
    for key in CHILD_OPTIONS:
        command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    return command


def stage_manifests(root, role):
    manifests = []
    for path in sorted((root / "capture" / role).glob("*/*/manifest.json")):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    return manifests


def wait_capture_finished(root, role, tp_size, request_id, timeout=120):
    """Observe real scheduler completion; never mark an archive finished by hand."""
    deadline = time.monotonic() + timeout
    while True:
        rows = stage_manifests(root, role)
        errors = [error for row in rows for error in row.get("errors", [])]
        if errors:
            raise RuntimeError(f"{role} tensor capture failed: {errors[:3]}")
        if len(rows) == tp_size:
            if {row.get("request_id") for row in rows} != {request_id}:
                raise RuntimeError(f"{role} captured ID differs from generated request {request_id!r}")
            if sorted(row.get("tp_rank", -1) for row in rows) != list(range(tp_size)):
                raise RuntimeError(f"{role} capture has missing/duplicate TP ranks")
            if all(row.get("complete") and row.get("request_finished") for row in rows):
                return [
                    {key: row[key] for key in ("request_id", "tp_rank", "records", "complete", "request_finished")}
                    for row in rows
                ]
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"{role} capture did not finish on all {tp_size} ranks; inspect {root / 'capture' / role}"
            )
        time.sleep(0.2)


def run_child(args):
    from layerwise_prefill_file_store import install

    install()
    from vllm import LLM, SamplingParams

    root, stage = args.run_dir.resolve(), args.child
    stage_dir = root / stage
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    options = engine_options(args, stage, prompt["length"])
    write_json(stage_dir / "engine_options.json", options)
    report = dict(
        schema_version=1,
        completed=False,
        case=args.case,
        stage=stage,
        prompt_token_ids=prompt["token_ids"],
        prompt_length=prompt["length"],
        token_ids=[],
        output_token_limit=1 if stage == "prefill" else args.output_tokens,
    )
    llm = None
    try:
        llm = LLM(**options)
        before = mtp_snapshot(llm) if args.mtp_tokens else {}
        results = llm.generate(
            {"prompt_token_ids": prompt["token_ids"]},
            SamplingParams(temperature=0, seed=SEED, max_tokens=report["output_token_limit"], ignore_eos=True),
            use_tqdm=False,
        )
        if len(results) != 1 or len(results[0].outputs) != 1 or not results[0].finished:
            raise RuntimeError("Expected one completed request")
        result, completion = results[0], results[0].outputs[0]
        after = mtp_snapshot(llm) if args.mtp_tokens else {}
        report.update(
            request_id=result.request_id,
            token_ids=list(completion.token_ids),
            text=completion.text,
            num_cached_tokens=result.num_cached_tokens,
            finish_reason=completion.finish_reason,
            mtp=dict(
                configured_tokens=args.mtp_tokens,
                metrics={
                    key: after[key] - before[key] if after[key] is not None and before[key] is not None else None
                    for key in before
                },
            ),
        )
        if len(report["token_ids"]) != report["output_token_limit"]:
            raise RuntimeError("Generation did not reach the requested output token count")
        expected_cached = prompt["length"] - 1 if stage == "decode" else 0
        if report["num_cached_tokens"] != expected_cached:
            raise RuntimeError(f"{stage}: cached_tokens={report['num_cached_tokens']}, expected {expected_cached}")
        if args.mtp_tokens and stage == "decode":
            count = report["mtp"]["metrics"].get("vllm:spec_decode_num_draft_tokens")
            if count is None or count <= 0:
                raise RuntimeError("MTP enabled but no target verification was observed")
        # Teardown-only flush of the actual background stores, after generation.
        flushed = llm.collective_rpc("flush_file_store", timeout=args.rpc_timeout_seconds)
        write_json(stage_dir / "store_flush.json", flushed)
        capture = wait_capture_finished(
            root, "P" if stage == "prefill" else "D", len(args.devices.split(",")), result.request_id
        )
        write_json(stage_dir / "capture-completion.json", capture)
        report["completed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        error_in_flight = sys.exc_info()[0] is not None
        teardown_error = None
        try:
            if llm is not None:
                llm.collective_rpc("close_file_store", timeout=args.rpc_timeout_seconds)
        except BaseException as error:
            report.update(completed=False, close_error=f"{type(error).__name__}: {error}")
            teardown_error = error
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
        except BaseException as error:
            report.update(completed=False, shutdown_error=f"{type(error).__name__}: {error}")
            teardown_error = teardown_error or error
        write_json(stage_dir / "output.json", report)
        (stage_dir / "output.txt").write_text(report.get("text", ""), encoding="utf-8")
        if teardown_error is not None and not error_in_flight:
            raise teardown_error
    print(
        f"{PREFIX} {args.case}/{stage}: capture complete; output={len(report['token_ids'])} "
        f"cached={report['num_cached_tokens']}",
        flush=True,
    )


def run_stages(args, root, case):
    from layerwise_prefill_file_store import seal_store

    for stage in STAGES:
        directory = root / stage
        directory.mkdir()
        check_shm_capacity(Path("/dev/shm"), args.cpu_cache_gb)
        env = child_environment(args, root, case, stage)
        write_json(directory / "environment.json", recorded_environment(env))
        proc = start_logged_process(
            child_command(args, root, case, stage), env, directory / "server.log", f"{case}/{stage}", prefix=PREFIX
        )
        try:
            try:
                code = proc.wait(timeout=args.stage_timeout_seconds)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"{case}/{stage} timed out; inspect {directory / 'server.log'}") from error
            if code:
                raise RuntimeError(f"{case}/{stage} exited {code}; inspect {directory / 'server.log'}")
        finally:
            finish_child(proc)
        write_json(directory / "process-exited.json", dict(pid=proc.pid, returncode=proc.returncode))
        if not json.loads((directory / "output.json").read_text(encoding="utf-8")).get("completed"):
            raise RuntimeError(f"{case}/{stage} output is incomplete")
        if stage == "prefill":
            seal = seal_store(root)
            write_json(root / "store-seal-report.json", seal)
            if not seal.get("passed"):
                raise RuntimeError(f"P file store could not be sealed: {seal.get('errors')}")
            print(f"{PREFIX} {case}: P and its workers exited; D will start with a new CPU cache", flush=True)


def analyze_run(root):
    from layerwise_prefill_file_store import validate_store
    from pd_tensor_analyze import analyze
    from pd_tensor_smoke_validate import validate_case

    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    if config.get("tool") != "pd_tensor_smoke" or config.get("schema_version") != 1:
        raise ValueError("Expected this smoke test's run_config.json")
    if config.get("cases") not in (["off"], ["on"], ["off", "on"]):
        raise ValueError("Invalid smoke test case inventory")
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    report = dict(
        schema_version=1,
        smoke_passed=False,
        accuracy_verdict="not_assessed",
        cases={},
        errors=[],
        scope="Single-host eager P/D with file SDK; new main-backbone recorder, MTP draft internals excluded",
    )
    for case in config["cases"]:
        case_root = root / case
        coverage = validate_case(
            case_root,
            config["tp_size"],
            prompt["token_ids"],
            config["options"]["output_tokens"],
            expect_layerwise=case == "on",
            min_prefill_calls=config["min_prefill_calls"],
            model_id=config["options"]["model"],
        )
        store = validate_store(case_root)
        comparison = analyze(
            [case_root / "capture"], [case_root / "capture"], mode="pd-kv", output=case_root / "report-pd"
        )
        report["cases"][case] = dict(coverage=coverage, store=store, pd_analysis=comparison)
        if not coverage["complete"] or not store["passed"] or comparison["status"] != "analysis_complete":
            report["errors"].append(f"{case}: capture, file transfer or analysis is incomplete")
        if comparison["new_nonfinite"]:
            report["errors"].append(f"{case}: newly nonfinite KV values")
    if config["cases"] == ["off", "on"]:
        comparison = analyze(
            [root / "off" / "capture"], [root / "on" / "capture"], mode="off-on", output=root / "report-off-on"
        )
        report["off_on_analysis"] = comparison
        if comparison["status"] != "analysis_complete" or comparison["new_nonfinite"]:
            report["errors"].append("OFF/ON comparison has incomplete contexts/coverage or new nonfinite values")
        try:
            outputs = [
                json.loads((root / case / "decode" / "output.json").read_text(encoding="utf-8"))["token_ids"]
                for case in ("off", "on")
            ]
            report["output_tokens_equal"] = outputs[0] == outputs[1]
            if not report["output_tokens_equal"]:
                report["errors"].append("OFF/ON final output token sequences differ")
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["output_tokens_equal"] = None
            report["errors"].append(f"OFF/ON output unavailable: {error}")
    report["smoke_passed"] = not report["errors"]
    report["interpretation"] = (
        "smoke_passed checks execution/coverage/transport/output gates, not floating-point tolerance"
    )
    write_json(root / "report.json", report)
    print(f"{PREFIX} smoke_passed={report['smoke_passed']}; report: {root / 'report.json'}", flush=True)
    return 0 if report["smoke_passed"] else 1


def main(argv=None):
    args = parser().parse_args(argv)
    if args.analyze_only:
        return analyze_run(args.analyze_only.expanduser().resolve(strict=True))
    validate_args(args)
    if args.dry_run:
        root = (args.run_dir or Path.cwd() / "pd-tensor-smoke-preview").resolve()
        plan = []
        for case in selected_cases(args):
            for stage in STAGES:
                env = child_environment(args, root / case, case, stage)
                plan.append(
                    dict(
                        case=case,
                        stage=stage,
                        engine_options=engine_options(args, stage, args.prompt_tokens),
                        environment=recorded_environment(env),
                        output_token_limit=1 if stage == "prefill" else args.output_tokens,
                    )
                )
        print(json.dumps(plan, indent=2))
        return 0
    if sys.platform != "linux":
        raise RuntimeError("Actual execution requires Linux Ascend; --dry-run and CPU tests run locally")
    if args.child:
        run_child(args)
        return 0
    root = (args.run_dir or Path(tempfile.mkdtemp(prefix="pd-tensor-smoke-", dir=Path.cwd()))).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("--run-dir must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    try:
        check_shm_capacity(Path("/dev/shm"), args.cpu_cache_gb)
        record_model_identity(args.model, root)
        length = prepare_prompt(args, root)
        config = dict(
            schema_version=1,
            tool="pd_tensor_smoke",
            cases=list(selected_cases(args)),
            tp_size=len(args.devices.split(",")),
            min_prefill_calls=2 if length > args.prefill_chunk_tokens else 1,
            options={key: getattr(args, key) for key in CHILD_OPTIONS},
        )
        write_json(root / "run_config.json", config)
        print(
            f"{PREFIX} model={args.model}; prompt={length}; output={args.output_tokens}; "
            f"TP={config['tp_size']}; cases={config['cases']}; artifacts={root}",
            flush=True,
        )
        for case in selected_cases(args):
            case_root = root / case
            case_root.mkdir()
            shutil.copyfile(root / "prompt.json", case_root / "prompt.json")
            prepare_bootstrap(case_root)
            run_stages(args, case_root, case)
        return analyze_run(root)
    except BaseException as error:
        write_json(root / "failure.json", dict(error=f"{type(error).__name__}: {error}"))
        print(f"{PREFIX} FAILED: {error}; artifacts: {root}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
