#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P -> files at the Mooncake SDK boundary -> D. No Mooncake processes.

Full model, TP8, MTP1, max_model_len=16384, GPU memory=.96; no baseline by default.
Not a transport/performance test. Normal serving never imports the file shim.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from layerwise_prefill_check import DEFAULT_PROMPT_FILE, first_difference, prepare_prompt, write_json
from layerwise_prefill_check import engine_options as base_engine_options
from layerwise_prefill_mooncake_check import (
    STORAGE_CHUNK_TOKENS,
    clear_shared_memory,
    config_environment,
    deployment_config,
    finish_child,
    model_stages,
    stage_config,
    stage_environment,
    start_logged_process,
)

MTP_COUNTERS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", default="/workspace/models/GLM-5.2-w4a8c8-0723")
    cli.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    cli.add_argument("--output-tokens", type=int, default=256, help="Maximum only; EOS can stop earlier")
    cli.add_argument("--with-baseline", action="store_true")
    cli.add_argument("--prefill-chunk-tokens", type=int, default=4096)
    cli.add_argument("--cpu-cache-gb", type=float, default=8)
    cli.add_argument("--run-dir", type=Path)
    cli.add_argument("--child", choices=("baseline", "prefill", "decode"), help=argparse.SUPPRESS)
    # Same production config; the test SDK does not allocate the native segment.
    cli.set_defaults(store_gb=8, prompt_tokens=None)
    return cli


def engine_options(args, prompt_len, stage):
    options = base_engine_options(args, prompt_len, stage)
    # P and D must register the same draft KV layers. The layerwise prefill
    # protocol supports one MTP forward, not repeated multi-token drafting.
    options["speculative_config"] = {"method": "deepseek_mtp", "num_speculative_tokens": 1}
    options["disable_log_stats"] = False  # Expose actual verification/acceptance counts.
    if stage != "prefill":
        # Q1 for ordinary decode/draft, Q2 for target verification with MTP1.
        options["compilation_config"]["cudagraph_capture_sizes"] = [1, 2]
    return options


def mtp_snapshot(llm):
    counts = {}
    for metric in llm.get_metrics():
        for name in MTP_COUNTERS:
            if metric.name == f"vllm:spec_decode_{name}":
                counts[name] = counts.get(name, 0) + int(metric.value)
    return counts


def mtp_statistics(before, after):
    # vLLM counts drafts presented to target verification, not every draft
    # forward (e.g. P's last draft need never be verified). Do not claim coverage
    # when EOS/max_tokens ends the request before any verification step.
    counts = {name: after[name] - before[name] if name in before and name in after else None for name in MTP_COUNTERS}
    drafted, accepted = counts["num_draft_tokens"], counts["num_accepted_tokens"]
    return {
        "num_speculative_tokens": 1,
        **counts,
        "verification_observed": drafted is not None and drafted > 0,
        "acceptance_rate_percent": 100 * accepted / drafted if drafted and accepted is not None else None,
    }


def child_environment(args, root, stage):
    env = stage_environment(args, root, stage)
    env = {key: value for key, value in env.items() if not key.startswith(("LMCACHE_", "MOONCAKE_"))}
    # Addresses are unused placeholders: neither setup nor registration starts
    # a native transport. Keep the real connector, page layout and roles.
    config = stage_config(deployment_config("127.0.0.1:1", "127.0.0.1"), args, stage)
    if stage != "baseline":
        config["extra_config"]["prefill_check_file_sdk"] = {"root": str(root), "stage": stage}
    env.update(config_environment(config))
    env["PYTHONPATH"] = str(root / "bootstrap") + os.pathsep + env["PYTHONPATH"]
    return env


def prepare_bootstrap(root):
    # EngineCore, lookup processes AND workers must install before LMCache
    # imports Mooncake. A worker extension alone does not cover the scheduler.
    bootstrap = root / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "import os, traceback\n"
        "try:\n"
        "    from layerwise_prefill_file_store import install\n"
        "    install()\n"
        "except BaseException:\n"
        "    traceback.print_exc()\n"
        "    os._exit(1)\n",  # Do not let site.py swallow errors and use native Mooncake.
        encoding="utf-8",
    )


def run_model(args):
    from layerwise_prefill_file_store import install

    install()
    from vllm import LLM, SamplingParams

    root, stage = args.run_dir, args.child
    prompt = json.loads((root / "prompt.json").read_text(encoding="utf-8"))
    options = engine_options(args, prompt["length"], stage)
    if stage == "baseline":
        options.pop("worker_extension_cls")
    else:
        options["worker_extension_cls"] = "layerwise_prefill_file_worker.FileStoreWorker"
    write_json(root / stage / "engine_options.json", options)
    llm = LLM(**options)
    try:
        before = mtp_snapshot(llm)
        (result,) = llm.generate(
            {"prompt_token_ids": prompt["token_ids"]},
            SamplingParams(temperature=0, seed=1024, max_tokens=1 if stage == "prefill" else args.output_tokens),
            use_tqdm=False,
        )
        (completion,) = result.outputs
        mtp = mtp_statistics(before, mtp_snapshot(llm))
        write_json(
            root / stage / "output.json",
            {
                "stage": stage,
                "text": completion.text,
                "token_ids": list(completion.token_ids),
                "num_cached_tokens": result.num_cached_tokens,
                "finish_reason": completion.finish_reason,
                "mtp": mtp,
            },
        )
        (root / stage / "output.txt").write_text(completion.text, encoding="utf-8")
        print(f"[PREFILL_FILE] {stage}: {completion.text!r}", flush=True)
        print(f"[PREFILL_FILE] {stage} MTP: {json.dumps(mtp)}", flush=True)
        if stage == "prefill":
            llm.collective_rpc("prefill_check_flush_store", timeout=600)
    finally:
        llm.llm_engine.engine_core.shutdown()


def io_records(root, stage):
    return [
        json.loads(line)
        for path in sorted((root / stage).glob("store-io-*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def seal_store(root):
    puts = [record for record in io_records(root, "prefill") if record["method"].startswith("batch_put")]
    objects = {record["key"]: {name: record[name] for name in ("bytes", "sha256")} for record in puts}
    if not objects:
        raise RuntimeError("P never called Mooncake's put API; no files to reload")
    write_json(root / "store-sealed.json", objects)
    print(f"[PREFILL_FILE] P exited; stored {len(objects)} objects, {sum(x['bytes'] for x in objects.values())} bytes")


def run_stages(args, root):
    for stage in model_stages(args):
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--child",
            stage,
            "--run-dir",
            str(root),
            "--model",
            args.model,
            "--devices",
            args.devices,
            "--output-tokens",
            str(args.output_tokens),
            "--cpu-cache-gb",
            str(args.cpu_cache_gb),
            "--prefill-chunk-tokens",
            str(args.prefill_chunk_tokens),
        ]
        env = child_environment(args, root, stage)
        write_json(root / stage / "lmcache_env.json", {k: v for k, v in env.items() if k.startswith("LMCACHE_")})
        proc = start_logged_process(command, env, root / stage / "server.log", stage, prefix="[PREFILL_FILE]")
        try:
            if proc.wait():
                raise RuntimeError(f"{stage} failed; inspect {root / stage / 'server.log'}")
        finally:
            finish_child(proc)
        if stage == "prefill":
            seal_store(root)  # Only after P's complete process group has stopped.


def key_group(key):
    fields = key.split("@")
    return int(fields[7 if key.startswith("__lmcache_page_v1__@") else 5])


def analyse(root, with_baseline=False):
    records = {stage: io_records(root, stage) for stage in ("prefill", "decode")}
    objects = json.loads((root / "store-sealed.json").read_text(encoding="utf-8"))
    gets = [r for r in records["decode"] if r["method"].startswith("batch_get")]
    successful = [r for r in gets if r.get("status") == "ok"]
    errors = []
    for record in successful:
        if objects.get(record["key"]) != {name: record[name] for name in ("bytes", "sha256")}:
            errors.append(f"D read does not match P saved bytes: {record['key']}")
    if len(successful) != len(gets):
        errors.append("D had missing/undersized file reads; see decode/store-io-*.jsonl")
    groups = sorted({key_group(r["key"]) for r in successful})
    if groups != [0, 1]:
        errors.append("D did not read both real KV groups through Mooncake's get API")
    outputs = {
        stage: json.loads((root / stage / "output.json").read_text(encoding="utf-8"))
        for stage in (("baseline", "prefill", "decode") if with_baseline else ("prefill", "decode"))
    }
    prompt_len = json.loads((root / "prompt.json").read_text(encoding="utf-8"))["length"]
    required = (prompt_len - 1) // STORAGE_CHUNK_TOKENS * STORAGE_CHUNK_TOKENS
    if outputs["prefill"]["num_cached_tokens"] or (outputs["decode"]["num_cached_tokens"] or 0) < required:
        errors.append("Expected fresh P computation followed by D cache reload")
    summary = {
        "storage": "files replacing Mooncake SDK put/get; original LMCache Mooncake connector",
        "baseline_ran": with_baseline,
        "api_calls": {stage: dict(Counter(r["method"] for r in rows)) for stage, rows in records.items()},
        "stored_objects": len(objects),
        "stored_bytes": sum(item["bytes"] for item in objects.values()),
        "decode_read_objects": len({r["key"] for r in successful}),
        "decode_read_bytes": sum(r["bytes"] for r in successful),
        "decode_read_groups": groups,
        "decode_cached_tokens": outputs["decode"]["num_cached_tokens"],
        "expected_cached_prefix_min": required,
        "outputs": outputs,
        "mtp": {stage: output.get("mtp") for stage, output in outputs.items()},
        "errors": errors,
        "scope": "P/D compute with MTP1, LocalCPU offload/reload and Mooncake key/page/buffer calls; "
        "check mtp.verification_observed per stage for actual MTP verification coverage; NOT native transport, "
        "network registration, concurrent RemoteFill, DP or performance",
    }
    if with_baseline:
        summary["prefill_first_token_equal"] = (
            outputs["baseline"]["token_ids"][:1] == outputs["prefill"]["token_ids"][:1]
        )
        summary["decode_first_difference"] = first_difference(
            outputs["baseline"]["token_ids"], outputs["decode"]["token_ids"]
        )
    write_json(root / "summary.json", summary)
    print(
        f"[PREFILL_FILE] put_objects={len(objects)}, get_objects={summary['decode_read_objects']}, "
        f"get_groups={groups}, errors={len(errors)}; {root / 'summary.json'}",
        flush=True,
    )
    if errors:
        raise RuntimeError(f"File-boundary validation incomplete: {errors[:3]}")


def main():
    args = parser().parse_args()
    if args.child:
        run_model(args)
        return
    if os.name != "posix":
        raise RuntimeError("Run on the Linux Ascend server")
    if min(args.output_tokens, args.cpu_cache_gb, args.prefill_chunk_tokens) <= 0:
        raise ValueError("Token limits and CPU size must be positive")
    if args.prefill_chunk_tokens % STORAGE_CHUNK_TOKENS:
        raise ValueError("prefill-chunk-tokens must be a multiple of 1024")
    devices = args.devices.split(",")
    if any(not value.isdigit() for value in devices) or len(set(devices)) != len(devices):
        raise ValueError("devices must list distinct NPU IDs")
    root = (
        args.run_dir.resolve() if args.run_dir else Path(tempfile.mkdtemp(prefix="layerwise-file-", dir=".")).resolve()
    )
    root.mkdir(exist_ok=True)
    for stage in (*model_stages(args), "store"):
        (root / stage).mkdir()  # Never reuse old cache evidence.
    prepare_bootstrap(root)
    print(f"[PREFILL_FILE] results: {root}; no Mooncake master/holder; MTP=1, max_len=16384, gpu=0.96", flush=True)
    prepare_prompt(args, root)
    clear_shared_memory(prefix="[PREFILL_FILE]")  # Once before P; never between P and D.
    run_stages(args, root)
    analyse(root, args.with_baseline)


if __name__ == "__main__":
    main()
