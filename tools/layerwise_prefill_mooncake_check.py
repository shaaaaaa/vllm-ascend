#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sequential full-model baseline/P/D against a surviving Mooncake segment.

LMCache settings are supplied through child environment variables, no YAML.
Uses the previously supplied master address (must already be running). No
profiler or KV probes. P exits before D starts with a fresh LocalCPU cache.
This tests persistence/reload, NOT simultaneous P-to-D RemoteFill negotiation.
"""

import argparse
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
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

DEFAULT_MASTER = "7.150.4.174:58888"
STORAGE_CHUNK_TOKENS = 1024


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--master", default=DEFAULT_MASTER, help="Existing Mooncake master; defaults to the supplied deployment"
    )
    cli.add_argument("--model", default="/workspace/models/GLM-5.2-w4a8c8-0723")
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    cli.add_argument("--output-tokens", type=int, default=256, help="Upper bound; EOS can end generation earlier")
    cli.add_argument("--prefill-chunk-tokens", type=int, default=4096)
    cli.add_argument("--cpu-cache-gb", type=float, default=8)
    cli.add_argument("--store-gb", type=float, default=8, help="Independent Mooncake storage segment size")
    cli.add_argument("--run-dir", type=Path)
    cli.add_argument("--child", choices=("holder", "baseline", "prefill", "decode"), help=argparse.SUPPRESS)
    return cli


def detect_local_hostname(master):
    """Select this host's source address for the master route, without a send."""
    host, port = master.rsplit(":", 1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((host, int(port)))
        return probe.getsockname()[0]


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
    """Isolate model-owned storage; retain deployment transport settings."""
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
        # The sequential test has no live D handoff; D reads persisted pages.
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
        # Deliberately exercise the warning + bank-safe P override.
        mooncake_direct_npu_prefill_store=stage == "prefill",
        enable_cache_usage_details_in_response=True,
        shared_cpu_cache_name=None,
        shared_cpu_cache_size_gb=args.cpu_cache_gb,
    )
    if stage == "baseline":
        config["remote_url"] = None
    return config


def child_environment(args, root, stage):
    env = stage_environment(args, root, "prefill" if stage == "holder" else stage)
    env = {key: value for key, value in env.items() if not key.startswith(("LMCACHE_", "MOONCAKE_"))}
    config = stage_config(deployment_config(args.master, args.local_hostname), args, stage)
    if stage == "holder":
        # Only this independent process mounts a Mooncake storage segment.
        config["extra_config"]["global_segment_size"] = int(args.store_gb * 1024**3)
    env.update(config_environment(config))
    return env


def run_holder(args):
    """Own CPU storage independently of both model-worker lifetimes."""
    extra = json.loads(os.environ["LMCACHE_EXTRA_CONFIG"])
    if extra.get("protocol", "ascend") == "ascend":
        # This process never constructs a vLLM worker. Ascend transport still
        # calls aclrtGetDevice even when its storage segment is in CPU RAM.
        # Initialize on the setup thread, before constructing the native store.
        import torch_npu

        # Child visibility is args.devices: logical 0 is its FIRST visible NPU,
        # not necessarily physical card 0 (e.g. --devices 4,5,6,7).
        torch_npu.npu.set_device(0)
        torch_npu.npu.init()
        print(
            "[PREFILL_MOONCAKE] holder Ascend context ready: logical_device=0, "
            f"visible_devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}; KV storage remains in CPU RAM",
            flush=True,
        )

    from mooncake.store import MooncakeDistributedStore

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
            extra.get("protocol", "ascend"),
            extra.get("device_name", ""),
            extra["master_server_address"],
        )
        if status not in (None, 0):
            raise RuntimeError(f"Mooncake holder setup failed: {status}")
        write_json(args.run_dir / "holder_ready.json", {"segment": store.get_hostname(), "pid": os.getpid()})
        print("[PREFILL_MOONCAKE] independent storage ready", flush=True)
        stopped.wait()
    finally:
        store.close()


def run_model(args):
    from vllm import LLM, SamplingParams

    root, stage = args.run_dir, args.child
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    options = engine_options(args, prompt["length"], stage)
    # No synchronization/CPU-copy probes; exercise ordinary model execution.
    options.pop("worker_extension_cls")
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
    log_path = root / stage / "server.log"
    print(f"[PREFILL_MOONCAKE] starting {stage}: {log_path}", flush=True)
    proc = subprocess.Popen(
        command,
        env=child_environment(args, root, stage),
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


def run_models(args, root, holder):
    for stage in ("baseline", "prefill", "decode"):
        proc = start_child(args, root, stage)
        try:
            while proc.poll() is None:
                if holder.poll() is not None:
                    raise RuntimeError("Storage holder exited; inspect holder/server.log")
                time.sleep(1)
            if proc.returncode:
                raise RuntimeError(f"{stage} failed ({proc.returncode}); inspect {root / stage / 'server.log'}")
        finally:
            # Fence P's entire process group BEFORE launching D.
            finish_child(proc)
        print(f"[PREFILL_MOONCAKE] {stage} exited; output: {root / stage / 'output.txt'}", flush=True)


def analyse(root, chunk_size):
    records = {
        stage: json.loads((root / stage / "output.json").read_text(encoding="utf-8"))
        for stage in ("baseline", "prefill", "decode")
    }
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    expected = (prompt["length"] - 1) // chunk_size * chunk_size
    cached = records["decode"]["num_cached_tokens"] or 0
    p_cached = records["prefill"]["num_cached_tokens"] or 0
    summary = {
        "prefill_first_token_equal": records["baseline"]["token_ids"][:1] == records["prefill"]["token_ids"][:1],
        "decode_first_difference": first_difference(records["baseline"]["token_ids"], records["decode"]["token_ids"]),
        "expected_cached_prefix_min": expected,
        "decode_cached_tokens": cached,
        "prefill_cached_tokens": p_cached,
        "p_computed_and_d_reloaded": p_cached == 0 and cached >= expected,
        "scope": "single-host sequential persistent reload; no concurrent RemoteFill, MTP, DP or KV probes",
    }
    write_json(root / "summary.json", summary)
    print(f"[PREFILL_MOONCAKE] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    if not summary["p_computed_and_d_reloaded"]:
        raise RuntimeError("Did not exercise fresh P compute followed by D cache reload; see summary.json")


def main():
    args = parser().parse_args()
    if args.child:
        run_holder(args) if args.child == "holder" else run_model(args)
        return
    if os.name != "posix":
        raise RuntimeError("Run model/holder orchestration on the Linux Ascend server")
    if min(args.output_tokens, args.prefill_chunk_tokens, args.cpu_cache_gb, args.store_gb) <= 0:
        raise ValueError("Token limits and CPU/store sizes must be positive")
    args.local_hostname = detect_local_hostname(args.master)
    base = deployment_config(args.master, args.local_hostname)
    chunk_size = base["chunk_size"]
    if chunk_size <= 0 or args.prefill_chunk_tokens % chunk_size:
        raise ValueError("prefill-chunk-tokens must be a multiple of chunk_size")
    root = (
        args.run_dir.resolve()
        if args.run_dir
        else Path(tempfile.mkdtemp(prefix="layerwise-mooncake-", dir=".")).resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    for stage in ("holder", "baseline", "prefill", "decode"):
        (root / stage).mkdir()  # Never reuse an old run's output/cache evidence.
        # Audit artifact only; children read their environment, not this file.
        env = child_environment(args, root, stage)
        write_json(root / stage / "lmcache_env.json", {k: v for k, v in env.items() if k.startswith("LMCACHE_")})
    print(
        f"[PREFILL_MOONCAKE] results: {root}; full model, TP={len(args.devices.split(','))}, max_len=16384, gpu=0.96",
        flush=True,
    )
    print(
        f"[PREFILL_MOONCAKE] master={args.master}, local_hostname={args.local_hostname}, chunk_size={chunk_size}",
        flush=True,
    )
    original = args.prompt_file.read_text(encoding="utf-8")
    # A fresh first chunk prevents an old master entry bypassing this run's P.
    args.prompt_file = root / "input.txt"
    args.prompt_file.write_text(f"Validation run {uuid.uuid4().hex}.\n\n{original}", encoding="utf-8")
    args.prompt_tokens = None
    prompt_len = prepare_prompt(args, root)
    if prompt_len <= max(chunk_size, args.prefill_chunk_tokens):
        raise ValueError("Prompt must cover multiple prefill and storage chunks")
    holder = start_child(args, root, "holder")
    try:
        deadline = time.monotonic() + 120
        while not (root / "holder_ready.json").exists():
            if holder.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f"Storage holder not ready; inspect {root / 'holder/server.log'}")
            time.sleep(0.2)
        run_models(args, root, holder)
        analyse(root, chunk_size)
    finally:
        # Existing master is NOT stopped; only this launcher's holder is stopped.
        finish_child(holder)


if __name__ == "__main__":
    main()
