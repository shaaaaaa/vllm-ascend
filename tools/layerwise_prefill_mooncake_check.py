#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sequential full-model P/D against a surviving Mooncake segment.

LMCache settings are supplied through child environment variables, no YAML.
Starts a local master and two Ascend storage holders on logical devices 1/0.
P writes holder1. After P exits, holder0 copies those objects through Ascend.
D0 reads holder1; D1..7 read holder0. No profiler or KV tensor probes.
This tests persistent reload, NOT simultaneous P-to-D RemoteFill negotiation.
Baseline output comparison is opt-in with --with-baseline.
"""

import argparse
import copy
import ipaddress
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from layerwise_prefill_check import (
    DEFAULT_PROMPT_FILE,
    engine_options,
    first_difference,
    prepare_prompt,
    stage_environment,
    stop_process_group,
    write_json,
)
from layerwise_prefill_mooncake_routing import copy_objects, manifest_digest, read_manifest, require_placement

STORAGE_CHUNK_TOKENS = 1024
MASTER_STARTUP_TIMEOUT_SECONDS = 30
HOLDER_STARTUP_TIMEOUT_SECONDS = 120
HOLDER_COPY_TIMEOUT_SECONDS = 600


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--master", help="Optional existing master; default: start a private local master automatically")
    cli.add_argument("--master-bin", default="mooncake_master", help="Local master executable (name or path)")
    cli.add_argument("--local-hostname", help="Optional local IPv4 address for Ascend transport; default: auto-detect")
    cli.add_argument("--model", default="/workspace/models/GLM-5.2-w4a8c8-0723")
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    cli.add_argument("--output-tokens", type=int, default=256, help="Upper bound; EOS can end generation earlier")
    cli.add_argument(
        "--with-baseline", action="store_true", help="Also run baseline before P/D and compare output tokens"
    )
    cli.add_argument("--prefill-chunk-tokens", type=int, default=4096)
    cli.add_argument("--cpu-cache-gb", type=float, default=8)
    cli.add_argument("--store-gb", type=float, default=8, help="Storage size PER holder (two holders)")
    cli.add_argument("--run-dir", type=Path)
    cli.add_argument("--child", choices=("holder0", "holder1", "baseline", "prefill", "decode"), help=argparse.SUPPRESS)
    return cli


def detect_local_hostname(master=None):
    """Select a local NIC for Ascend, separate from loopback master RPC."""
    # UDP connect only asks the local routing table; no packet is sent. TEST-NET
    # selects the default route without depending on any real external service.
    host, port = master.rsplit(":", 1) if master else ("192.0.2.1", 9)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((host, int(port)))
            address = probe.getsockname()[0]
            if not ipaddress.ip_address(address).is_loopback:
                return address
    except OSError:
        pass
    for _, _, _, _, (address, _) in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        if not ipaddress.ip_address(address).is_loopback and address != "0.0.0.0":
            return address
    raise RuntimeError("Cannot detect a local NIC address for Ascend; pass --local-hostname <this-host-IP>")


def local_master_ports():
    """Choose distinct available RPC/admin ports without touching other jobs."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rpc,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as admin,
    ):
        rpc.bind(("127.0.0.1", 0))
        admin.bind(("127.0.0.1", 0))
        return rpc.getsockname()[1], admin.getsockname()[1]


def wait_for_master(proc, address, log_path):
    host, port = address.rsplit(":", 1)
    deadline = time.monotonic() + MASTER_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Local Mooncake master exited ({proc.returncode}); inspect {log_path}")
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                if proc.poll() is None:
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Local Mooncake master did not listen on {address}; inspect {log_path}")


@contextmanager
def managed_master(args, root):
    """Own only our new master. Never stop an explicitly supplied service."""
    if args.master:
        yield None
        return
    binary = shutil.which(args.master_bin)
    if binary is None:
        # pip may install the executable beside Python without that dir on PATH.
        candidate = (
            Path(sys.executable).with_name(args.master_bin) if Path(args.master_bin).name == args.master_bin else None
        )
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
            binary = str(candidate)
    if binary is None:
        raise RuntimeError(
            "mooncake_master executable not found; install it or pass --master-bin /path/to/mooncake_master"
        )
    rpc_port, metrics_port = local_master_ports()
    args.master = f"127.0.0.1:{rpc_port}"
    command = [
        binary,
        f"--port={rpc_port}",
        "--rpc_address=127.0.0.1",
        f"--metrics_port={metrics_port}",
        "--enable_metric_reporting=false",
        "--enable_http_metadata_server=false",
        "--logtostderr=true",
    ]
    # Do not inherit deployment YAML/HA discovery from another Mooncake cluster.
    env = {key: value for key, value in os.environ.items() if not key.startswith(("LMCACHE_", "MOONCAKE_"))}
    log_path = root / "master" / "server.log"
    proc = start_logged_process(command, env, log_path, "local master")
    try:
        wait_for_master(proc, args.master, log_path)
        write_json(root / "master" / "process.json", {"pid": proc.pid, "address": args.master, "command": command})
        print(f"[PREFILL_MOONCAKE] local master ready: {args.master}", flush=True)
        yield proc
    finally:
        finish_child(proc)


def deployment_config(master, local_hostname):
    """Settings from the supplied TP4/DP4 YAML, sized per test stage below."""
    return {
        "remote_url": f"mooncakestore://{master}/",
        "chunk_size": STORAGE_CHUNK_TOKENS,
        "shared_cpu_cache_numa_policy": "interleave",
        "shared_cpu_cache_passive_writable": True,
        "lookup_timeout_ms": 30000,
        "experimental_sampled_layerwise_lookup": True,
        "pin_timeout_sec": 1800,
        "enable_npu_content_diagnostics": False,
        "remote_fill_max_inflight_bytes": 8589934592,
        "remote_fill_max_inflight_windows_per_request": 2,
        "remote_fill_direct_worker_count": 2,
        "remote_fill_max_native_operations": 2,
        "extra_config": {
            "use_exists_async": True,
            "local_hostname": local_hostname,
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "master_server_address": master,
            "transfer_timeout": 120,
            "mooncake_dsa_raw_token_dims": {0: 576, 1: 128},
        },
    }


def config_environment(config):
    """Encode the existing LMCACHE_* schema, with JSON only for extra_config."""
    return {
        f"LMCACHE_{key.upper()}": (
            json.dumps(value)
            if isinstance(value, (dict, list))
            else str(value).lower()
            if isinstance(value, bool)
            else str(value)
        )
        for key, value in config.items()
        if value is not None
    }


def stage_config(base, args, stage):
    """Isolate model-owned storage; retain the deployment Ascend transport."""
    config = copy.deepcopy(base)
    config.update(
        local_cpu=True,
        max_local_cpu_size=args.cpu_cache_gb,
        use_layerwise=True,
        enable_sparse_attention=True,
        dsa_two_groups=True,
        save_unfull_chunk=True,
        save_decode_cache=False,
        save_full_chunk_in_decode=False,
        enable_shared_cpu_cache=True,
        shared_cpu_cache_strict=True,
        store_async=stage == "prefill",
        store_async_max_queue_size=2,
        enable_async_loading=False,
        internal_api_server_enabled=False,
        enable_remote_lmcache_store=stage != "baseline",
        pd_role="receiver" if stage == "decode" else "sender",
        enable_dsa_cold_compact_load=stage == "decode",
        dsa_group1_load_mode="p2p_preferred" if stage == "baseline" else "persistent_direct_hbm",
        enable_pd=False,
        enable_p2p=False,
        store_location=None,
        retrieve_locations=None,
        # Never attach the previous process's named cache or inherit a large
        # slab-size override from a multi-host deployment YAML.
        shared_cpu_cache_name=None,
        shared_cpu_cache_size_gb=args.cpu_cache_gb,
    )
    extra = config.setdefault("extra_config", {})
    extra.update(
        # Neither P nor D owns a Mooncake segment; exiting P cannot erase KV.
        global_segment_size=0,
        local_buffer_size=0,
        mooncake_prefer_local_alloc=False,
        save_only_first_rank=True,
        save_chunk_meta=False,
        use_ascend_direct=True,
        mooncake_page_first_multi_buffer=True,
        mooncake_layer_merged_page_objects=True,
        # Exercise the existing bank-safe layerwise P override, as in deployment.
        mooncake_direct_npu_prefill_store=stage == "prefill",
        enable_cache_usage_details_in_response=True,
        shared_cpu_cache_name=None,
        shared_cpu_cache_size_gb=args.cpu_cache_gb,
    )
    if stage == "baseline":
        config["remote_url"] = None
    return config


def child_environment(args, root, stage):
    env = stage_environment(args, root, "prefill" if stage.startswith("holder") else stage)
    env = {key: value for key, value in env.items() if not key.startswith(("LMCACHE_", "MOONCAKE_"))}
    for key in ("MC_FORCE_TCP", "MC_FORCE_SHM"):
        env.pop(key, None)
    config = stage_config(deployment_config(args.master, args.local_hostname), args, stage)
    if stage.startswith("holder"):
        config["extra_config"]["global_segment_size"] = int(args.store_gb * 1024**3)
        # Holder writes/verification of its OWN segment must be local memcpy,
        # not an ADXL self connection. Cross-holder/model reads remain Ascend.
        env["MC_STORE_MEMCPY"] = "1"
    elif stage != "baseline":
        config["extra_config"]["prefill_check_routing"] = {
            "root": str(root),
            "namespace": root.name,
            "stage": stage,
        }
    env.update(config_environment(config))
    return env


def run_holder(args):
    """Keep CPU KV alive on device 1, or clone it on device 0 before D starts."""
    extra = json.loads(os.environ["LMCACHE_EXTRA_CONFIG"])
    if extra["protocol"] != "ascend":
        raise ValueError("Two-holder validation requires the Ascend transport")
    import torch
    import torch_npu
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    device = int(args.child[-1])
    torch_npu.npu.set_device(device)
    torch_npu.npu.init()
    print(f"[PREFILL_MOONCAKE] {args.child} Ascend context ready: logical_device={device}", flush=True)

    store = MooncakeDistributedStore()
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    try:
        status = store.setup(
            extra["local_hostname"],
            extra.get("metadata_server", "P2PHANDSHAKE"),
            int(extra["global_segment_size"]),
            0,
            extra["protocol"],
            extra.get("device_name", ""),
            extra["master_server_address"],
        )
        if status not in (None, 0):
            raise RuntimeError(f"Mooncake holder setup failed: {status}")
        # Fail missing native APIs/descriptor bindings BEFORE loading a model.
        holder_preflight(store, ReplicateConfig, args.run_dir.name, torch)
        record = {"segment": store.get_hostname(), "pid": os.getpid(), "device": device, "protocol": "ascend"}
        if device == 0:
            entries = read_manifest(args.run_dir / "prefill/objects.jsonl")
            if sum(entries.values()) > int(extra["global_segment_size"]):
                raise ValueError("KV objects exceed --store-gb for holder0; refusing placement fallback")
            source = json.loads((args.run_dir / "holder1/ready.json").read_text())
            record["copy"] = copy_objects(
                store,
                ReplicateConfig,
                entries,
                args.run_dir.name,
                source["segment"],
                store.get_hostname(),
                lambda size: torch.empty(size, dtype=torch.uint8, device="cpu"),
            )
        # Parent polls for this file: never expose partially written JSON.
        ready_path = args.run_dir / args.child / "ready.json"
        pending_path = ready_path.with_suffix(".pending")
        write_json(pending_path, record)
        pending_path.replace(ready_path)
        print(f"[PREFILL_MOONCAKE] {args.child} ready: {json.dumps(record)}", flush=True)
        stopped.wait()
    finally:
        store.close()


def holder_preflight(store, config_cls, namespace, torch):
    required = ("batch_get_into_multi_buffers", "batch_put_from_multi_buffers", "get_replica_desc", "remove")
    missing = [name for name in required if not callable(getattr(store, name, None))]
    if missing:
        raise RuntimeError(f"Mooncake holder is missing required test APIs: {missing}")
    probe = torch.tensor([19, 73, 11, 241], dtype=torch.uint8)
    key = f"{namespace}/holder-probe/{store.get_hostname()}"
    if store.register_buffer(probe.data_ptr(), probe.numel()) not in (None, 0):
        raise RuntimeError("Holder probe registration failed")
    try:
        config = config_cls()
        config.replica_num = 1
        config.preferred_segment = store.get_hostname()
        if store.batch_put_from_multi_buffers([key], [[probe.data_ptr()]], [[probe.numel()]], config) != [0]:
            raise RuntimeError("Holder local store preflight failed")
        require_placement(store, key, store.get_hostname(), probe.numel())
        probe.zero_()
        if store.batch_get_into_multi_buffers([key], [[probe.data_ptr()]], [[probe.numel()]]) != [probe.numel()]:
            raise RuntimeError("Holder local load preflight failed")
        if probe.tolist() != [19, 73, 11, 241]:
            raise RuntimeError("Holder local load preflight bytes differ")
    finally:
        store.unregister_buffer(probe.data_ptr())
        store.remove(key)


def run_model(args):
    from vllm import LLM, SamplingParams

    root, stage = args.run_dir, args.child
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    options = engine_options(args, prompt["length"], stage)
    # Select a TEST-ONLY native-store key router, not KV tensor/debug probes.
    if stage == "baseline":
        options.pop("worker_extension_cls")
    else:
        options["worker_extension_cls"] = "layerwise_prefill_mooncake_worker.MooncakeRoutingWorker"
    write_json(root / stage / "engine_options.json", options)
    llm = LLM(**options)
    try:
        (result,) = llm.generate(
            {"prompt_token_ids": prompt["token_ids"]},
            SamplingParams(temperature=0, seed=1024, max_tokens=1 if stage == "prefill" else args.output_tokens),
            use_tqdm=False,
        )
        (completion,) = result.outputs
        write_json(
            root / stage / "output.json",
            {
                "stage": stage,
                "text": completion.text,
                "token_ids": list(completion.token_ids),
                "num_cached_tokens": result.num_cached_tokens,
                "finish_reason": completion.finish_reason,
            },
        )
        (root / stage / "output.txt").write_text(completion.text, encoding="utf-8")
        print(f"[PREFILL_MOONCAKE] {stage}: {completion.text!r}", flush=True)
    finally:
        # P's final completion/worker close must finish remote puts before exit.
        llm.llm_engine.engine_core.shutdown()


def start_child(args, root, stage):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        stage,
        "--master",
        args.master,
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
        "--store-gb",
        str(args.store_gb),
    ]
    return start_logged_process(command, child_environment(args, root, stage), root / stage / "server.log", stage)


def start_logged_process(command, env, log_path, label):
    print(f"[PREFILL_MOONCAKE] starting {label}: {log_path}", flush=True)
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def copy_log():
        with log_path.open("w", encoding="utf-8") as log:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

    proc.log_reader = threading.Thread(target=copy_log, daemon=True)
    proc.log_reader.start()
    return proc


def finish_child(proc):
    stop_process_group(proc.pid)
    proc.wait(timeout=30)
    proc.log_reader.join(timeout=15)
    if proc.log_reader.is_alive():
        raise RuntimeError("Child process group did not release log pipe; refusing to launch another model")


def model_stages(args):
    return ("baseline", "prefill", "decode") if args.with_baseline else ("prefill", "decode")


def check_services(holders, master):
    if master is not None and master.poll() is not None:
        raise RuntimeError("Local Mooncake master exited; inspect master/server.log")
    for name, holder in holders.items():
        if holder.poll() is not None:
            raise RuntimeError(f"Storage holder exited; inspect {name}/server.log")


def wait_for_holder(root, name, holders, master):
    timeout = HOLDER_COPY_TIMEOUT_SECONDS if name == "holder0" else HOLDER_STARTUP_TIMEOUT_SECONDS
    deadline = time.monotonic() + timeout
    while not (root / name / "ready.json").exists():
        check_services(holders, master)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Storage holder not ready; inspect {root / name / 'server.log'}")
        time.sleep(0.2)
    check_services(holders, master)
    if name == "holder0":
        copy_report(root)


def copy_report(root):
    source = json.loads((root / "holder1/ready.json").read_text())
    target = json.loads((root / "holder0/ready.json").read_text())
    entries = read_manifest(root / "prefill/objects.jsonl")
    expected = {"objects": len(entries), "bytes": sum(entries.values()), "manifest_sha256": manifest_digest(entries)}
    if (source["device"], target["device"]) != (1, 0) or source["segment"] == target["segment"]:
        raise RuntimeError("Holder identity mismatch; refusing to start D")
    if target.get("copy") != expected:
        raise RuntimeError("Holder copy does not match P's completed manifest; refusing to start D")
    return expected


def run_models(args, root, holders, master=None):
    for stage in model_stages(args):
        if stage == "decode":
            # Only start device-0 holder AFTER P and its whole process group exit.
            holders["holder0"] = start_child(args, root, "holder0")
            wait_for_holder(root, "holder0", holders, master)
        check_services(holders, master)
        proc = start_child(args, root, stage)
        try:
            while proc.poll() is None:
                check_services(holders, master)
                time.sleep(1)
            if proc.returncode:
                raise RuntimeError(f"{stage} failed ({proc.returncode}); inspect {root / stage / 'server.log'}")
        finally:
            # Fence P's entire process group BEFORE launching D.
            finish_child(proc)
        print(f"[PREFILL_MOONCAKE] {stage} exited; output: {root / stage / 'output.txt'}", flush=True)


def analyse(root, chunk_size, with_baseline=False):
    stages = ("baseline", "prefill", "decode") if with_baseline else ("prefill", "decode")
    records = {stage: json.loads((root / stage / "output.json").read_text(encoding="utf-8")) for stage in stages}
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    expected = (prompt["length"] - 1) // chunk_size * chunk_size
    cached = records["decode"]["num_cached_tokens"] or 0
    p_cached = records["prefill"]["num_cached_tokens"] or 0
    summary = {
        "transport": "ascend",
        "holder_routing": {"D0": "holder1", "D1..7": "holder0"},
        "copy": copy_report(root),
        "decode_routes": [json.loads(path.read_text()) for path in sorted((root / "decode").glob("route-*.json"))],
        "baseline_ran": with_baseline,
        "expected_cached_prefix_min": expected,
        "decode_cached_tokens": cached,
        "prefill_cached_tokens": p_cached,
        "p_computed_and_d_reloaded": p_cached == 0 and cached >= expected,
        "scope": "single-host two-copy Ascend persistent reload; no concurrent RemoteFill, MTP, DP or KV probes",
    }
    if with_baseline:
        summary.update(
            prefill_first_token_equal=records["baseline"]["token_ids"][:1] == records["prefill"]["token_ids"][:1],
            decode_first_difference=first_difference(records["baseline"]["token_ids"], records["decode"]["token_ids"]),
        )
    write_json(root / "summary.json", summary)
    print(f"[PREFILL_MOONCAKE] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    if not summary["p_computed_and_d_reloaded"]:
        raise RuntimeError("Did not exercise fresh P compute followed by D cache reload; see summary.json")


def clear_shared_memory():
    """Equivalent to rm -rf /dev/shm/*; only the parent calls this at startup."""
    root = Path("/dev/shm")
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise RuntimeError("Refusing to clean /dev/shm: expected a real directory at that exact path")
    print(
        "[PREFILL_MOONCAKE] WARNING: clearing /dev/shm/* before startup; "
        "other shared-memory users must be stopped. Deleted data cannot be recovered.",
        flush=True,
    )
    removed = 0
    for entry in root.iterdir():
        # Match shell '*' semantics, and unlink symlinks without following them.
        if entry.name.startswith("."):
            continue
        if entry.is_symlink() or not entry.is_dir():
            entry.unlink(missing_ok=True)
        else:
            if entry.resolve().parent != root:
                raise RuntimeError(f"Refusing to remove a directory outside /dev/shm: {entry}")
            shutil.rmtree(entry)
        removed += 1
    print(f"[PREFILL_MOONCAKE] /dev/shm cleanup complete: removed {removed} entries", flush=True)


def main():
    args = parser().parse_args()
    if args.child:
        run_holder(args) if args.child.startswith("holder") else run_model(args)
        return
    if os.name != "posix":
        raise RuntimeError("Run model/holder orchestration on the Linux Ascend server")
    if min(args.output_tokens, args.prefill_chunk_tokens, args.cpu_cache_gb, args.store_gb) <= 0:
        raise ValueError("Token limits and CPU/store sizes must be positive")
    devices = args.devices.split(",")
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError("Two-holder validation requires at least two distinct visible devices")
    args.local_hostname = args.local_hostname or detect_local_hostname(args.master)
    chunk_size = STORAGE_CHUNK_TOKENS
    if chunk_size <= 0 or args.prefill_chunk_tokens % chunk_size:
        raise ValueError("prefill-chunk-tokens must be a multiple of chunk_size")
    root = (
        args.run_dir.resolve()
        if args.run_dir
        else Path(tempfile.mkdtemp(prefix="layerwise-mooncake-", dir=".")).resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    for stage in ("master", "holder1", "holder0", *model_stages(args)):
        (root / stage).mkdir()  # Never reuse an old run's output/cache evidence.
    print(
        f"[PREFILL_MOONCAKE] results: {root}; full model, TP={len(args.devices.split(','))}, max_len=16384, gpu=0.96",
        flush=True,
    )
    # Never repeat in a child or between P and D: holder storage must survive.
    clear_shared_memory()
    with managed_master(args, root) as master:
        run_check(args, root, master)


def run_check(args, root, master):
    chunk_size = STORAGE_CHUNK_TOKENS
    print(
        f"[PREFILL_MOONCAKE] protocol=ascend, holders=1,0, master={args.master}, "
        f"local_hostname={args.local_hostname}, chunk_size={chunk_size}; "
        f"stages={','.join(model_stages(args))}, baseline={'enabled' if args.with_baseline else 'skipped'}",
        flush=True,
    )
    for stage in ("holder1", "holder0", *model_stages(args)):
        # Audit artifact only; children read their environment, not this file.
        env = child_environment(args, root, stage)
        write_json(root / stage / "lmcache_env.json", {k: v for k, v in env.items() if k.startswith("LMCACHE_")})
    original = args.prompt_file.read_text(encoding="utf-8")
    # A fresh first chunk prevents an old master entry bypassing this run's P.
    args.prompt_file = root / "input.txt"
    args.prompt_file.write_text(f"Validation run {uuid.uuid4().hex}.\n\n{original}", encoding="utf-8")
    args.prompt_tokens = None
    prompt_len = prepare_prompt(args, root)
    if prompt_len <= max(chunk_size, args.prefill_chunk_tokens):
        raise ValueError("Prompt must cover multiple prefill and storage chunks")
    holders = {}
    try:
        holders["holder1"] = start_child(args, root, "holder1")
        wait_for_holder(root, "holder1", holders, master)
        run_models(args, root, holders, master)
        analyse(root, chunk_size, args.with_baseline)
    finally:
        # Both copies must survive until D exits; master closes last.
        with ExitStack() as cleanup:
            for holder in holders.values():
                cleanup.callback(finish_child, holder)


if __name__ == "__main__":
    main()
