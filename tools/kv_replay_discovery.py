# SPDX-License-Identifier: Apache-2.0
"""Find a complete replay snapshot using metadata-only probes on candidate hosts."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from decode_kv_compare import FORMAT, request_key
from kv_replay_files import configure_plan
from kv_replay_snapshot import snapshot_metadata
from multi_turn_kv_reuse_concurrent import save_json
from pd_tensor_collect import remote_command, run_ssh

PREFIX = "[KV_REPLAY_DISCOVERY] "


def candidate_hosts(args) -> list[str]:
    """Return explicitly supplied hosts in order, retaining the single-host CLI."""
    return list(dict.fromkeys(([args.snapshot_host] if args.snapshot_host else []) + args.snapshot_hosts))


def probe_snapshot(plan: dict) -> dict:
    """Check request identity, ranks, layout and file sizes without reading KV bytes."""
    root = Path(plan["snapshot_root"]).resolve()
    result = dict(snapshot_root=str(root), request_id=plan["previous"], manifests=0, matching_manifests=[])
    if not root.is_dir():
        return dict(result, status="missing_directory")
    try:
        for path in sorted(root.rglob("manifest.json")):
            result["manifests"] += 1
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("format") == FORMAT and request_key(meta["request_id"]) == plan["previous"]:
                result["matching_manifests"].append(str(path))
        if not result["matching_manifests"]:
            return dict(result, status="request_not_found")
        tokens, _, _ = snapshot_metadata(plan)
        return dict(result, status="ready", source_tokens=len(tokens), tp_size=plan["tp_size"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(result, status="invalid_snapshot", error=f"{type(error).__name__}: {error}")


def probe_host(args, host: str, plan: dict, output: Path, index: int) -> dict:
    """Run a bounded metadata probe; keep SSH diagnostics separate from its JSON."""
    encoded = base64.b64encode(json.dumps(plan).encode()).decode("ascii")
    command = remote_command(
        args,
        host,
        [args.remote_python, str(PurePosixPath(args.workspace) / "tools/kv_replay_discovery.py"), "--probe", encoded],
    )
    log = output / f"snapshot-probe-{index}.log"
    try:
        with log.open("wb") as errors:
            completed = run_ssh(
                args, host, command, stdout=subprocess.PIPE, stderr=errors, timeout=min(args.source_timeout, 120)
            )
        if completed.returncode:
            return dict(host=host, status="probe_failed", returncode=completed.returncode, log=str(log))
        result = json.loads(completed.stdout)
        if result.get("request_id") != plan["previous"] or result.get("status") not in (
            "ready",
            "missing_directory",
            "request_not_found",
            "invalid_snapshot",
        ):
            raise ValueError("Invalid snapshot probe response")
        return dict(result, host=host, log=str(log))
    except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError) as error:
        return dict(host=host, status="probe_failed", error=f"{type(error).__name__}: {error}", log=str(log))


def select_snapshot(args, plan: dict, output: Path) -> dict:
    """Select one verified source before starting file serving or model inference.

    Multiple complete captures are ambiguous and require an explicit single
    host. A missing/broken candidate does not hide a complete capture elsewhere.
    This never creates a new snapshot or substitutes another request's data.
    """
    hosts = candidate_hosts(args)
    if hosts:
        with ThreadPoolExecutor(max_workers=min(len(hosts), 8)) as pool:
            jobs = [pool.submit(probe_host, args, host, plan, output, i) for i, host in enumerate(hosts)]
            results = [job.result() for job in jobs]
    else:
        results = [dict(probe_snapshot(plan), host=None)]
    inventory = dict(request_id=plan["previous"], candidates=results)
    path = output / "snapshot_discovery.json"
    save_json(path, inventory)
    for result in results:
        print(
            PREFIX
            + json.dumps(
                {
                    key: result[key]
                    for key in ("host", "status", "snapshot_root", "manifests", "error", "log")
                    if key in result
                }
            ),
            flush=True,
        )
    ready = [item for item in results if item["status"] == "ready"]
    if len(ready) != 1:
        states = "; ".join(f"{item['host'] or 'local'}: {item['status']}" for item in results)
        reason = "No usable snapshot" if not ready else "Multiple complete captures; select one with --snapshot-host"
        raise ValueError(
            f"{reason} for previous request {plan['previous']} ({states}). "
            f"See {path}. Supply all D candidates with --snapshot-hosts; "
            "no file server or inference request was started."
        )
    selected = ready[0]
    args.snapshot_host = selected["host"]
    plan["snapshot_root"] = selected["snapshot_root"]
    plan["snapshot_host"] = selected["host"]
    configure_plan(args, plan)
    inventory["selected_host"] = selected["host"]
    inventory["selected_root"] = selected["snapshot_root"]
    save_json(path, inventory)
    save_json(output / "file_plan.json", plan)
    print(PREFIX + json.dumps(dict(status="selected", host=selected["host"], request_id=plan["previous"])), flush=True)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    args = parser.parse_args()
    result = probe_snapshot(json.loads(base64.b64decode(args.probe)))
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
