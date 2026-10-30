#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect full D receive archives and compare two rounds' common token prefix.

No hashes, sampling, torch or accelerator runtime required for comparison.
Run after both requests have completed. Only the D hosts need to be supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pd_tensor_collect import (
    host_directory,
    password_auth,
    remote_command,
    run_ssh,
    safe_extract,
    ssh_target,
)

FORMAT = "decode-kv-raw-v1"
READ_BYTES = 4 * 1024 * 1024

REMOTE = r"""
import gzip,io,json,pathlib,sys,tarfile
workspace=pathlib.Path(sys.argv[1]); configured=sys.argv[2]; ids=sys.argv[3:]
roots=[pathlib.Path(configured)] if configured else [
    workspace/'decode-kv-snapshots',workspace/'vllm-ascend'/'decode-kv-snapshots']
found=set()
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed:
    with tarfile.open(fileobj=compressed,mode='w|') as archive:
        for root in roots:
            for wanted in ids:
                for manifest in sorted((root/wanted).glob('*/manifest.json')):
                    directory=manifest.parent
                    if directory.resolve() in found: continue
                    found.add(directory.resolve())
                    meta=json.loads(manifest.read_text())
                    if meta.get('status')=='writing':
                        raise RuntimeError('Snapshot is still being written: '+str(directory))
                    names=['manifest.json','tokens.json']+[r['path'] for r in meta.get('files',[])]
                    if meta.get('replay_file') == 'replay.json': names.append('replay.json')
                    for name in names:
                        if pathlib.Path(name).name!=name: raise ValueError('Unsafe snapshot filename')
                        path=directory/name
                        if path.is_symlink() or not path.is_file():
                            raise ValueError('Missing/linked snapshot file: '+str(path))
                        archive.add(path,arcname=wanted+'/'+directory.name+'/'+name,recursive=False)
        payload=json.dumps(dict(requests=ids,workers=len(found),roots=[str(x) for x in roots])).encode()
        info=tarfile.TarInfo('collection-status.json'); info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
"""


def request_key(value):
    match = re.search(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value)
    result = match[0].lower() if match else value
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", result) or result in (".", ".."):
        raise ValueError("Invalid request ID")
    return result


def collect(args, host):
    """Use native OpenSSH; stream a compressed archive and publish atomically."""
    directory = args.output / "hosts" / host_directory(host)
    identity = dict(
        host=host,
        requests=[args.previous, args.current],
        workspace=args.workspace,
        snapshot_dir=args.snapshot_dir,
        container=args.container,
    )
    if directory.exists():
        marker = directory / "collection.json"
        if marker.is_file() and json.loads(marker.read_text()) == identity:
            print(f"[D_KV_COMPARE] reusing {host}: {directory}", flush=True)
            return directory
        raise ValueError(f"Existing output belongs to another collection: {directory}; use --local or a new --output")
    directory.parent.mkdir(parents=True, exist_ok=True)
    print(f"[D_KV_COMPARE] collecting {host}", flush=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix="download-") as temp:
        staging = Path(temp)
        archive = staging / "download.tar.gz"
        command = remote_command(
            args, host, ["python3", "-c", REMOTE, args.workspace, args.snapshot_dir or "", args.previous, args.current]
        )
        with archive.open("xb") as stream:
            result = run_ssh(args, host, command, stdout=stream, stderr=-1)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace")[-1800:])
        extracted = staging / "extracted"
        extracted.mkdir()
        counts = safe_extract(archive, extracted)
        (extracted / "collection.json").write_text(json.dumps(identity), encoding="utf-8")
        extracted.rename(directory)
        # Do not retain a second, potentially hundreds-of-GB compressed copy.
    print(f"[D_KV_COMPARE] {host}: {counts['files']} files, {counts['bytes']} bytes", flush=True)
    return directory


def read_workers(roots, request):
    workers = {}
    seen = set()
    for root in roots:
        for manifest in sorted(root.rglob("manifest.json")):
            if manifest.resolve() in seen:
                continue
            seen.add(manifest.resolve())
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            if meta.get("format") != FORMAT or request_key(meta["request_id"]) != request:
                continue
            rank = meta["tp_rank"]
            if rank in workers:
                raise ValueError(f"Multiple captures for TP {rank}, request {request}; select one run via --local")
            if meta["status"] != "complete":
                raise ValueError(f"Incomplete capture: {manifest}: {meta.get('errors')}")
            tokens = json.loads((manifest.parent / "tokens.json").read_text())
            if len(tokens) != meta["tokens"]:
                raise ValueError(f"Token metadata length mismatch: {manifest}")
            rows = {}
            for row in meta["files"]:
                key = (row["group"], row["layer"], row["component"])
                path = manifest.parent / row["path"]
                if Path(row["path"]).name != row["path"] or key in rows:
                    raise ValueError(f"Invalid/duplicate payload: {path}")
                if row["start"] != 0 or row["end"] != len(tokens) or row["bytes"] != len(tokens) * row["row_bytes"]:
                    raise ValueError(f"Incomplete payload range: {path}")
                if path.stat().st_size != row["bytes"]:
                    raise ValueError(f"Truncated payload: {path}")
                rows[key] = (row, path)
            expected = {(g, layer) for g, n in enumerate(meta["groups"]) for layer in range(n)}
            if {(g, layer) for g, layer, _ in rows} != expected:
                raise ValueError(f"Missing layer in {manifest}")
            workers[rank] = (meta, tokens, rows)
    if not workers:
        raise ValueError(f"No complete D receive archives for {request}")
    sizes = {m[0]["tp_size"] for m in workers.values()}
    if len(sizes) != 1 or set(workers) != set(range(next(iter(sizes)))):
        raise ValueError(f"Incomplete TP ranks for {request}: {sorted(workers)}, tp_size={sizes}")
    first = next(iter(workers.values()))[1]
    if any(tokens != first for _, tokens, _ in workers.values()):
        raise ValueError(f"TP ranks disagree on received token IDs: {request}")
    return workers


def suffix_stats(path, row, start, byteorder):
    """Describe new rows; zero/nonfinite counts are diagnostics, not a verdict."""
    count = zero = nonfinite = elements = 0
    dtype = row["dtype"]
    endian = "<" if byteorder == "little" else ">"
    with path.open("rb") as source:
        source.seek(start * row["row_bytes"])
        while data := source.read(READ_BYTES):
            raw = np.frombuffer(data, np.uint8)
            count += raw.size
            zero += int(np.count_nonzero(raw == 0))
            if dtype == "torch.bfloat16":
                bits = np.frombuffer(data, endian + "u2")
                nonfinite += int(np.count_nonzero((bits & 0x7F80) == 0x7F80))
                elements += bits.size
            elif dtype in ("torch.float16", "torch.float32", "torch.float64"):
                width = {"torch.float16": "f2", "torch.float32": "f4", "torch.float64": "f8"}[dtype]
                values = np.frombuffer(data, endian + width)
                nonfinite += int(np.count_nonzero(~np.isfinite(values)))
                elements += values.size
    return dict(
        start=start, end=row["end"], bytes=count, zero_bytes=zero, floating_elements=elements, nonfinite=nonfinite
    )


def compare_file(job):
    rank, key, left, right, common, byteorder = job
    a, ap = left
    b, bp = right
    for field in ("row_bytes", "dtype"):
        if a[field] != b[field]:
            raise ValueError(f"TP {rank} {key}: incompatible {field}")
    width = a["row_bytes"]
    rows_per_read = max(1, READ_BYTES // width)
    unequal_bytes = unequal_rows = 0
    first = []
    with ap.open("rb") as af, bp.open("rb") as bf:
        for start in range(0, common, rows_per_read):
            size = min(rows_per_read, common - start) * width
            ar, br = af.read(size), bf.read(size)
            if len(ar) != size or len(br) != size:
                raise ValueError(f"Payload changed during comparison: {ap} or {bp}")
            delta = (np.frombuffer(ar, np.uint8) != np.frombuffer(br, np.uint8)).reshape(-1, width)
            unequal_bytes += int(np.count_nonzero(delta))
            changed = np.flatnonzero(delta.any(axis=1))
            unequal_rows += changed.size
            for offset in changed[: max(0, 8 - len(first))]:
                byte = int(np.flatnonzero(delta[offset])[0])
                first.append(
                    dict(
                        token=start + int(offset),
                        byte_in_component=byte,
                        previous=ar[int(offset) * width + byte],
                        current=br[int(offset) * width + byte],
                    )
                )
    return dict(
        tp_rank=rank,
        group=key[0],
        layer=key[1],
        component=key[2],
        dtype=a["dtype"],
        compared_tokens=common,
        compared_bytes=common * width,
        different_bytes=unequal_bytes,
        different_tokens=int(unequal_rows),
        first=first,
        current_suffix=suffix_stats(bp, b, common, byteorder),
    )


def analyze(roots, previous, current, output, workers=4):
    before, after = read_workers(roots, previous), read_workers(roots, current)
    if set(before) != set(after):
        raise ValueError("TP sizes differ between rounds")
    jobs = []
    common = None
    for rank in sorted(before):
        am, at, ar = before[rank]
        bm, bt, br = after[rank]
        for field in ("groups", "byteorder", "layer_names", "model"):
            if am.get(field) != bm.get(field):
                raise ValueError(f"TP {rank}: incompatible {field}")
        if set(ar) != set(br):
            raise ValueError(f"TP {rank}: component sets differ")
        shared = next((i for i, (a, b) in enumerate(zip(at, bt)) if a != b), min(len(at), len(bt)))
        if not shared:
            raise ValueError("Requests have no common token prefix")
        common = shared
        jobs.extend((rank, key, ar[key], br[key], common, am["byteorder"]) for key in sorted(ar))
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    first_differences = []
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        (output / "layers.jsonl").open("w", encoding="utf-8") as detail,
    ):
        for result in pool.map(compare_file, jobs):
            detail.write(json.dumps(result, separators=(",", ":")) + "\n")
            for name in ("compared_bytes", "different_bytes", "different_tokens"):
                counts[name] += result[name]
            counts["components"] += 1
            counts["different_components"] += bool(result["different_bytes"])
            counts["suffix_nonfinite"] += result["current_suffix"]["nonfinite"]
            if result["first"] and len(first_differences) < 8:
                first_differences.append({k: v for k, v in result.items() if k != "current_suffix"})
    previous_end = len(next(iter(before.values()))[1])
    current_end = len(next(iter(after.values()))[1])
    report = dict(
        format=FORMAT,
        previous=previous,
        current=current,
        status="history_differs" if counts["different_bytes"] else "received_history_identical",
        common_prefix_tokens=common,
        previous_received_tokens=previous_end,
        current_received_tokens=current_end,
        current_uncompared_range=[common, current_end],
        previous_is_prefix=(common == previous_end),
        tp_ranks=sorted(before),
        **counts,
        first_differences=first_differences,
        scope=(
            "Exact received latent CPU and resident indexer bytes. New suffix has no prior reference. "
            "Equality does not validate later H2D, sparse mapping, or model arithmetic."
        ),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--previous", required=True, type=request_key, help="Earlier round's response/request UUID")
    cli.add_argument("--current", required=True, type=request_key, help="Later round's response/request UUID")
    cli.add_argument("--hosts", "--ips", nargs="+", type=ssh_target)
    cli.add_argument("--workspace", default="/workspace/fsi_lab/sqh/vllm-ascend")
    cli.add_argument("--snapshot-dir", help="Absolute path configured on D")
    cli.add_argument("--container")
    cli.add_argument("--password", dest="ssh_password")
    cli.add_argument("--password-env")
    cli.add_argument("--skip-host-key-check", action="store_true")
    cli.add_argument("--ssh-port", type=int)
    cli.add_argument("--identity-file", type=Path)
    cli.add_argument("--local", nargs="+", type=Path, help="Reuse prior collections or D snapshot roots")
    cli.add_argument("--workers", type=int, default=4, help="Parallel component readers")
    cli.add_argument("--output", type=Path, default=Path("decode-kv-comparison"))
    args = cli.parse_args(argv)
    args.password = args.password_per_host = False
    if args.previous == args.current or args.workers < 1:
        cli.error("Supply two different requests and a positive worker count")
    args.output.mkdir(parents=True, exist_ok=True)
    roots = list(args.local or [])
    try:
        if args.hosts:
            with password_auth(args), ThreadPoolExecutor(max_workers=min(4, len(args.hosts))) as pool:
                roots.extend(pool.map(lambda host: collect(args, host), args.hosts))
        report = analyze(roots, args.previous, args.current, args.output, args.workers)
    except Exception as error:
        report = dict(status="incomplete", error=str(error))
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    brief = {k: v for k, v in report.items() if k not in ("first_differences", "scope")}
    if report.get("first_differences"):
        first = report["first_differences"][0]
        brief["first_difference"] = {k: first[k] for k in ("tp_rank", "group", "layer", "component", "first")}
    print("[D_KV_COMPARE] " + json.dumps(brief, ensure_ascii=False))
    print(f"[D_KV_COMPARE] report: {args.output / 'report.json'}")
    return 1 if report["status"] == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
