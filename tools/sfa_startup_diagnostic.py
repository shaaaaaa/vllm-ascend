#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Trace TP8 startup without loading weights, KV caches, or running inference.

Use --load-model to include the original eight-layer parity model constructor.
Both modes stop after worker startup, before profiling/prefill/graph capture.
Run this same script with --repo-root pointing at four older checkouts for A/B;
the diagnostic tools stay fixed while imports come from the selected checkouts.
"""

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import regex as re
from sfa_full_graph_parity import DEFAULT_DEVICES, DEFAULT_MODEL, child_environment, engine_options, parse_devices
from sfa_startup_trace import TRACE_PREFIX, StartupTrace, mapping_errors

REPOSITORIES = {"vllm": "vllm", "vllm_ascend": "vllm-ascend", "lmcache": "LMCache", "lmcache_ascend": "LMCache-Ascend"}
STARTUP_ENVIRONMENT = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_DEVICE_ID",
    "RANK",
    "LOCAL_RANK",
    "RANK_ID",
    "RANK_TABLE_FILE",
    "HCCL_IF_IP",
    "HCCL_IF_BASE_PORT",
    "HCCL_SOCKET_IFNAME",
    "HCCL_SOCKET_FAMILY",
    "HCCL_NPU_SOCKET_PORT_RANGE",
    "HCCL_HOST_SOCKET_PORT_RANGE",
    "HCCL_DETERMINISTIC",
    "DYNAMIC_EPLB",
    "EXPERT_MAP_RECORD",
    "PD_SERVING_PERF",
)
NATIVE_BIND = re.compile(r"socket.*bind|listen on ip|specific port|RanktableDetect|EJ0003", re.IGNORECASE)
NATIVE_LINES_PER_PID = 60


def diagnostic_options(args):
    parity = SimpleNamespace(
        child="eager",
        devices=args.devices,
        model=args.model,
        reference=args.trace_dir,
        atol=1e-7,
        rtol=1e-2,
        compare_output=True,
        trace_residual=False,
    )
    options = engine_options(parity)
    options["worker_cls"] = "sfa_startup_worker.SFAStartupWorker"
    options["additional_config"]["sfa_startup"] = {
        "trace_dir": args.trace_dir,
        "load_model": args.load_model,
    }
    return options


def repository_paths(root):
    paths = [Path(root).resolve() / name for name in REPOSITORIES.values()]
    for module, path in zip(REPOSITORIES, paths):
        if not (path / module / "__init__.py").is_file():
            raise ValueError(f"Missing checkout: {path / module / '__init__.py'}")
    return paths


def report_versions(trace, root=None):
    for name, directory in REPOSITORIES.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ImportError(f"Cannot resolve {name}")
        source = Path(spec.origin).resolve()
        if root is not None and not source.is_relative_to(Path(root).resolve() / directory):
            raise RuntimeError(f"Requested checkout is not being used: {name}: {source}")
        revision = subprocess.run(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        trace.emit(
            "version",
            module=name,
            path=str(source),
            loaded=name in sys.modules,
            revision=revision.stdout.strip() if revision.returncode == 0 else "not-a-git-checkout",
        )


def run_executor(args, trace):
    # Import only after the child environment/tracing is in place. These are
    # the actual vLLM executor/worker startup APIs, not a stand-in engine.
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.executor.abstract import Executor

    with trace.span("engine_config"):
        config = EngineArgs(**diagnostic_options(args)).create_engine_config()
    executor = None
    try:
        with set_current_vllm_config(config), trace.span("executor_startup"):
            executor = Executor.get_class(config)(config)
        reports = executor.collective_rpc("startup_report", timeout=args.timeout)
        ranks = [report["rank"] for report in reports if report.get("complete")]
        if sorted(ranks) != list(range(len(parse_devices(args.devices)))):
            raise RuntimeError(f"Incomplete worker startup: {reports}")
        trace.emit("executor_complete", reports=reports, load_model=args.load_model)
    finally:
        if executor is not None:
            executor.shutdown()


def run_child(args):
    if args.repo_root:
        # Tools live outside the selected packages, including for spawned
        # workers. Do not copy new package code into an old checkout for A/B.
        sys.path[0:0] = [str(path) for path in repository_paths(args.repo_root)]
    trace = StartupTrace(args.trace_dir, stage=args.child)
    trace.emit("child_start", environment={key: os.environ[key] for key in STARTUP_ENVIRONMENT if key in os.environ})
    try:
        trace.observe_native_calls()
        # Fail BEFORE creating workers if an A/B checkout was not selected.
        report_versions(trace, args.repo_root)
        if args.child == "preflight":
            from sfa_full_graph_parity import preflight_dependencies

            with trace.span("dependency_preflight"):
                preflight_dependencies()
            trace.emit("preflight_state", device=trace.device(), comm_name_calls=trace.comm_calls)
        else:
            run_executor(args, trace)
    finally:
        already_failed = sys.exc_info()[0] is not None
        trace.close()
        # Metadata collection must not replace the original native exception.
        try:
            report_versions(trace, args.repo_root)
        except Exception as exc:
            trace.emit("version_error", error=str(exc))
            if not already_failed:
                raise


def stop_owned_process_group(process):
    """Only this diagnostic's new POSIX session; never kill other NPU jobs."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_stage(argv, environment, timeout):
    # Inherit stdout/stderr: live logs go directly to the caller's tee log.log.
    with subprocess.Popen(argv, env=environment, start_new_session=True) as process:
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            stop_owned_process_group(process)
            raise
    if code:
        raise subprocess.CalledProcessError(code, argv)


def read_records(directory):
    records = []
    for path in sorted(Path(directory).glob("trace-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A worker may have been killed during the final write.
                print(TRACE_PREFIX + f"incomplete trace line: {path.name}", flush=True)
    return records


def collect_native_bind_logs(pids, started, roots=None):
    """Read only this run's PID-named host plogs, on success AND failure."""
    if roots is None:
        roots = [Path.home() / "ascend/log"]
        if os.getenv("ASCEND_WORK_PATH"):
            roots.append(Path(os.environ["ASCEND_WORK_PATH"]) / "log")
        if os.getenv("ASCEND_PROCESS_LOG_PATH"):
            roots.append(Path(os.environ["ASCEND_PROCESS_LOG_PATH"]))
    seen = set()
    matches = {pid: [] for pid in pids}
    for root in roots:
        for path in Path(root).rglob("plog-*.log"):
            match = re.match(r"plog-(\d+)_", path.name)
            if match is None or int(match[1]) not in matches or path in seen:
                continue
            seen.add(path)
            if path.stat().st_mtime < started:
                continue
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if NATIVE_BIND.search(line):
                        matches[int(match[1])].append(f"{path.name}: {line.rstrip()}")
    for pid, lines in matches.items():
        if len(lines) > NATIVE_LINES_PER_PID:
            lines = lines[:20] + [f"... {len(lines) - NATIVE_LINES_PER_PID} lines omitted ..."] + lines[-40:]
        for line in lines:
            print(f"[SFA_NATIVE_BIND] pid={pid} {line}", flush=True)
    if not any(matches.values()):
        print(
            "[SFA_NATIVE_BIND] no matching current-run host plog lines; native first binder is NOT established",
            flush=True,
        )


def run_diagnostic(args):
    if sys.platform != "linux":
        raise RuntimeError("This startup diagnostic requires a Linux Ascend host")
    tp_size = len(parse_devices(args.devices))
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if not Path(args.model, "config.json").is_file():
        raise FileNotFoundError(f"Local model config not found: {args.model}/config.json")
    if args.repo_root:
        repository_paths(args.repo_root)
    started = time.time()
    with TemporaryDirectory(prefix="sfa-startup-") as directory:
        completed = False
        try:
            stages = ("startup",) if args.skip_preflight else ("preflight", "startup")
            for stage in stages:
                environment = child_environment("graph" if stage == "preflight" else "eager", args.devices)
                # Diagnostic verbosity only; never change HCCL port selection.
                environment.update(ASCEND_GLOBAL_LOG_LEVEL="1", PYTHONUNBUFFERED="1")
                argv = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    stage,
                    "--trace-dir",
                    directory,
                    "--model",
                    args.model,
                    "--devices",
                    args.devices,
                    "--timeout",
                    str(args.timeout),
                ]
                if args.load_model:
                    argv.append("--load-model")
                if args.repo_root:
                    argv += ["--repo-root", str(Path(args.repo_root).resolve())]
                print(TRACE_PREFIX + f"starting {stage}; TP={tp_size}; load_model={args.load_model}", flush=True)
                run_stage(argv, environment, args.timeout)
            completed = True
        finally:
            records = read_records(directory)
            errors = mapping_errors(records, tp_size, require_complete=completed)
            print(TRACE_PREFIX + json.dumps({"event": "mapping_check", "errors": errors}), flush=True)
            try:
                collect_native_bind_logs({record["pid"] for record in records}, started)
            except Exception as exc:
                print(f"[SFA_NATIVE_BIND] collection failed: {exc}", flush=True)
            # Do not replace an executor/timeout exception with an incomplete
            # mapping error from workers that never got through init_device.
            if completed and errors:
                raise RuntimeError("; ".join(errors))
    print(TRACE_PREFIX + "STARTUP PASS; no prefill/decode/graph-correctness claim", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--devices", default=DEFAULT_DEVICES)
    parser.add_argument("--load-model", action="store_true", help="Include original eight-layer dummy model loading")
    parser.add_argument(
        "--skip-preflight", action="store_true", help="Isolate startup from the parity dependency preflight"
    )
    parser.add_argument("--repo-root", help="Parent of vllm, vllm-ascend, LMCache, LMCache-Ascend checkouts for A/B")
    parser.add_argument("--timeout", type=int, default=300, help="Maximum seconds per child stage")
    parser.add_argument("--child", choices=("preflight", "startup"), help=argparse.SUPPRESS)
    parser.add_argument("--trace-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if not args.trace_dir:
            parser.error("Internal child requires --trace-dir")
        run_child(args)
    else:
        run_diagnostic(args)


if __name__ == "__main__":
    main()
