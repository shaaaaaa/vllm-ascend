#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect sampled KV byte files over native SSH and compare logical positions.

Use the existing LMCACHE_PREFILL_STORE_DEBUG=1 on P and D. These are sampled
features, not hashes or complete tensors. Matching bytes cannot certify the
unsampled payload. --local reuses previously collected directories.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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

FORMAT = "token-byte0-v1"
BOUNDARIES = {
    "MAP_ACTUAL": "P attention mapping differs from the expected request bank",
    "P_LOAD": "CPU/prior-chunk history -> P bank",
    "P_COMPUTE": "P historical KV changed during computation",
    "P_PRODUCED": "P newly produced KV -> bank scatter destination",
    "P_CPU_STORE": "P completed bank -> CPU save",
    "P_CPU_LOAD": "P earlier CPU save -> CPU reload source",
    "D_CPU": "P final prefill -> D received CPU KV",
    "D_DIRECT": "P final prefill -> D direct NPU import",
    "D_SELECTED": "D CPU history -> actual sparse NPU destination",
}

REMOTE = r"""
import gzip,io,json,pathlib,sys,tarfile,time
workspace=pathlib.Path(sys.argv[1]); wanted=sys.argv[2]; wait=float(sys.argv[3])
logdir=pathlib.Path(sys.argv[4]); configured=sys.argv[5]
roots=[workspace/'kv-feature-dump',workspace/'vllm-ascend'/'kv-feature-dump']
if configured: roots.insert(0,pathlib.Path(configured))
if not logdir.is_absolute():
    logdir=next((p for p in [workspace/logdir,workspace/'vllm-ascend'/logdir] if p.is_dir()),workspace/logdir)
logs=[]; directories=set()
for path in sorted(logdir.glob('*.log')):
    with path.open(errors='replace') as source:
        for line in source:
            if '[KVTRACE] ' not in line or wanted not in line: continue
            try: record=json.loads(line.split('[KVTRACE] ',1)[1])
            except ValueError: continue
            logs.append(json.dumps(record,separators=(',',':'))+'\n')
            if record.get('stage')=='FEATURE_FILE': directories.add(pathlib.Path(record['directory']))
for root in roots:
    directories.update(p.parent for p in (root/wanted).glob('*/index.jsonl'))
deadline=time.monotonic()+wait
while directories:
    complete=True
    for directory in directories:
        try:
            with (directory/'index.jsonl').open('rb') as source:
                source.seek(max(0,source.seek(0,2)-2048)); tail=source.read()
            if b'"stage":"END"' not in tail: complete=False
        except OSError: complete=False
    if complete or time.monotonic()>=deadline: break
    time.sleep(1)
# Python 3.9's streaming tar API does not accept compresslevel. Compress the
# plain tar stream with GzipFile, whose compression-level API is stable.
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed, \
        tarfile.open(fileobj=compressed,mode='w|') as archive:
    payload=''.join(logs).encode(); info=tarfile.TarInfo('logs.jsonl'); info.size=len(payload)
    archive.addfile(info,io.BytesIO(payload))
    for directory in sorted(directories):
        if directory.is_symlink() or not directory.is_dir(): continue
        # Dedicated feature files only; never descend into arbitrary tensors.
        for name in ('index.jsonl','features.bin'):
            path=directory/name
            if path.is_file() and not path.is_symlink():
                archive.add(path,arcname=directory.name+'/'+name,recursive=False)
"""


def collect_host(args, host):
    """Collect one host atomically; retry failures and reuse completed downloads."""
    directory = args.output / "hosts" / host_directory(host)
    # Dots in an IP are part of the host identity, not a filename extension.
    archive = directory.with_name(directory.name + ".tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    identity = dict(schema=1, host=host, request_id=args.request_id)
    if directory.exists():
        marker = directory / "collection.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == identity:
            print(f"[KVFEATURE] reusing {host}: {directory}", flush=True)
            return directory
        raise ValueError(f"output is not a completed collection: {directory}; use a new --output or --local")
    command = remote_command(
        args,
        host,
        [
            "python3",
            "-c",
            REMOTE,
            args.workspace,
            args.request_id,
            str(args.wait_seconds),
            args.log_dir,
            args.feature_dir or "",
        ],
    )
    print(f"[KVFEATURE] collecting {host}", flush=True)
    # A failed SSH or truncated tar must not leave the final directory/archive
    # looking complete, or prevent rerunning the original command.
    parent = directory.parent.resolve()
    with tempfile.TemporaryDirectory(prefix=directory.name + ".", dir=parent) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(parent):
            raise ValueError("collection staging directory escapes output")
        download = staging / "download.tar.gz"
        with download.open("xb") as target:
            result = run_ssh(args, host, command, stdout=target, stderr=-1)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace")[-1200:])
        extracted = staging / "extracted"
        extracted.mkdir()
        counts = safe_extract(download, extracted)
        with (extracted / "collection.json").open("x", encoding="utf-8") as marker:
            json.dump(identity, marker, separators=(",", ":"))
        download.replace(archive)
        extracted.rename(directory)
    print(f"[KVFEATURE] {host}: {counts['files']} files, {counts['bytes']} bytes", flush=True)
    return directory


class Observation:
    """A logical range backed by immutable sampled-byte file extents."""

    def __init__(self, path, meta, row):
        self.path, self.meta, self.row = path, meta, row
        self.start, self.end = row["start"], row["end"]
        self.rank = meta.get("tp_rank", meta["rank"] % meta.get("tp_size", 1))

    def read(self, start=None, end=None):
        """Read only the requested logical feature slice, without tensor loading."""
        start = self.start if start is None else start
        end = self.end if end is None else end
        width = self.row.get("byte_width", 1)
        result = bytearray()
        with (self.path.parent / "features.bin").open("rb") as source:
            cursor = start
            for begin, stop, offset in self.row["pieces"]:
                a, b = max(start, begin), min(end, stop)
                if a >= b:
                    continue
                if a != cursor:
                    raise ValueError("feature extent gap")
                source.seek(offset + (a - begin) * width)
                data = source.read((b - a) * width)
                if len(data) != (b - a) * width:
                    raise ValueError("incomplete binary feature extent")
                result.extend(data)
                cursor = b
        if cursor != end:
            raise ValueError("feature observation does not cover requested range")
        return bytes(result)


def analyze(roots, request_id, output):
    """Compare sampled boundaries; missing stages remain explicitly incomplete."""
    output.mkdir(parents=True, exist_ok=True)
    observations, workers, errors = [], [], []
    paths = sorted({p.resolve() for root in roots for p in root.rglob("index.jsonl")})
    for path in paths:
        meta = None
        ended = False
        last_completed = None
        try:
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                if row.get("stage") == "FILE_BEGIN":
                    meta = row
                    if request_id not in row.get("req", ""):
                        break
                    if row.get("format") != FORMAT:
                        raise ValueError("feature format mismatch")
                elif meta is not None and "pieces" in row:
                    observations.append(Observation(path, meta, row))
                    last_completed = {
                        key: row.get(key) for key in ("stage", "group", "layer", "frontier", "start", "end")
                    }
                elif row.get("stage") == "END":
                    ended = True
                    if row.get("failed"):
                        errors.append(dict(worker=str(path.parent), error="recorder_failed"))
            if meta is not None and request_id in meta.get("req", ""):
                workers.append((path, meta))
                if not ended:
                    errors.append(dict(worker=str(path.parent), error="missing_END", last_completed=last_completed))
        except Exception as error:
            errors.append(dict(worker=str(path.parent), error=str(error)))
    # Include a first probe failure even if it prevented creation of a data file.
    for root in roots:
        for path in root.rglob("logs.jsonl"):
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                if request_id in row.get("req", "") and row.get("error"):
                    errors.append(row)
    index = defaultdict(list)
    for obs in observations:
        row = obs.row
        index[(obs.rank, row["group"], row["layer"], row["component"], row["stage"])].append(obs)

    def references(obs, stage, *, same_worker=False, component=None):
        row = obs.row
        refs = index[
            (obs.rank, row["group"], row["layer"], row["component"] if component is None else component, stage)
        ]
        if same_worker:
            refs = [ref for ref in refs if ref.path == obs.path]
        return refs

    counts = Counter(comparisons=0, compared_bytes=0, different_bytes=0, unmatched_ranges=0)
    missing = []
    differences = []

    def difference_key(row):
        return (
            1 if row["stage"].startswith("D_") else 0,
            row.get("frontier", 0),
            row["layer"],
            list(BOUNDARIES).index(row["stage"]),
            row["position"],
        )

    def remember(row):
        differences.append(row)
        differences.sort(key=difference_key)
        del differences[32:]

    def compare(obs, refs, start=None, end=None, *, ordered=False):
        start = obs.start if start is None else start
        end = obs.end if end is None else end
        cursor = start
        if not ordered:
            refs = sorted(refs, key=lambda ref: (ref.row["frontier"], ref.row["observation"]))
        while cursor < end:
            ref = next((ref for ref in refs if ref.start <= cursor < ref.end), None)
            if ref is None:
                counts["unmatched_ranges"] += 1
                if len(missing) < 32:
                    missing.append(
                        dict(stage=obs.row["stage"], rank=obs.rank, layer=obs.row["layer"], start=cursor, end=end)
                    )
                return
            stop = min(end, ref.end)
            a, b = obs.read(cursor, stop), ref.read(cursor, stop)
            mismatch = None if a == b else np.frombuffer(a, dtype=np.uint8) != np.frombuffer(b, dtype=np.uint8)
            neq = 0 if mismatch is None else int(np.count_nonzero(mismatch))
            counts["compared_bytes"] += len(a)
            counts["different_bytes"] += neq
            counts["comparisons"] += 1
            if neq:
                first = int(np.argmax(mismatch))
                item = dict(
                    stage=obs.row["stage"],
                    rank=obs.rank,
                    group=obs.row["group"],
                    layer=obs.row["layer"],
                    frontier=obs.row["frontier"],
                    position=cursor + first // obs.row.get("byte_width", 1),
                    different_bytes=neq,
                    compared_bytes=len(a),
                    reference=str(ref.path.parent),
                    candidate=str(obs.path.parent),
                    expected=b[first],
                    actual=a[first],
                )
                detail.write(json.dumps(item, separators=(",", ":")) + "\n")
                remember(item)
            cursor = stop

    with (output / "differences.jsonl").open("w", encoding="utf-8") as detail:
        for obs in observations:
            row, stage = obs.row, obs.row["stage"]
            try:
                if stage == "MAP_ACTUAL":
                    compare(
                        obs,
                        [
                            r
                            for r in references(obs, "MAP_EXPECTED", same_worker=True)
                            if r.row["frontier"] == row["frontier"]
                        ],
                    )
                elif stage == "P_PRODUCED":
                    compare(
                        obs,
                        [
                            r
                            for r in references(obs, "P_COMPUTE", same_worker=True)
                            if r.row["frontier"] == row["frontier"]
                        ],
                    )
                elif stage == "P_COMPUTE":
                    prefix = row.get("prefix", 0)
                    if prefix:
                        compare(
                            obs,
                            [
                                r
                                for r in references(obs, "P_LOAD", same_worker=True)
                                if r.row["frontier"] == row["frontier"]
                            ],
                            0,
                            prefix,
                        )
                elif stage == "P_CPU_STORE":
                    compare(
                        obs, [r for r in references(obs, "P_COMPUTE", same_worker=True) if r.row["frontier"] >= obs.end]
                    )
                elif stage == "P_LOAD":
                    refs = sorted(
                        [
                            r
                            for r in references(obs, "P_CPU_LOAD", same_worker=True)
                            if r.row["frontier"] <= row["frontier"]
                        ],
                        key=lambda r: r.row["frontier"],
                        reverse=True,
                    )
                    refs += sorted(
                        [
                            r
                            for r in references(obs, "P_COMPUTE", same_worker=True)
                            if r.row["frontier"] < row["frontier"]
                        ],
                        key=lambda r: r.row["frontier"],
                        reverse=True,
                    )
                    # The real CPU source has precedence over another rank's
                    # earlier independently computed (possibly rounded) KV.
                    compare(obs, refs, ordered=True)
                elif stage == "P_CPU_LOAD":
                    publisher = row.get("source_rank")
                    rank = obs.rank if publisher is None else publisher % obs.meta.get("tp_size", 1)
                    compare(obs, index[(rank, row["group"], row["layer"], row["component"], "P_CPU_STORE")])
                elif stage in ("D_CPU", "D_DIRECT"):
                    rank = row.get("source_rank")
                    key = (
                        obs.rank if rank is None else rank % obs.meta.get("tp_size", 1),
                        row["group"],
                        row["layer"],
                        row["component"],
                        "P_COMPUTE",
                    )
                    compare(obs, [r for r in index[key] if r.row.get("final")])
                elif stage == "D_SELECTED":
                    positions = references(obs, "MAP_SELECTED_POS", same_worker=True, component=0)
                    slots = references(obs, "MAP_SELECTED_SLOT", same_worker=True, component=0)
                    refs = references(obs, "D_CPU", same_worker=True)
                    if not positions or not slots or not refs:
                        compare(obs, [])
                        continue
                    logical = struct.unpack("<" + "q" * positions[0].end, positions[0].read())
                    physical = struct.unpack("<" + "q" * slots[0].end, slots[0].read())
                    actual = obs.read()
                    cached_refs = [(ref, ref.read()) for ref in refs]
                    mismatched = 0
                    first_selected = None
                    for i, position in enumerate(logical):
                        if position < 0:
                            continue
                        if physical[i] < 0 or physical[i] >= row["slot_capacity"]:
                            errors.append(dict(stage=stage, error="selected_slot_out_of_range", slot=physical[i]))
                            break
                        ref = next(((r, data) for r, data in cached_refs if r.start <= position < r.end), None)
                        if ref is None:
                            counts["unmatched_ranges"] += 1
                            continue
                        reference, data = ref
                        expected = data[position - reference.start]
                        counts["compared_bytes"] += 1
                        counts["comparisons"] += 1
                        if actual[i] != expected:
                            mismatched += 1
                            if first_selected is None:
                                first_selected = dict(
                                    stage=stage,
                                    rank=obs.rank,
                                    group=row["group"],
                                    layer=row["layer"],
                                    frontier=row["frontier"],
                                    position=position,
                                    expected=expected,
                                    actual=actual[i],
                                )
                    counts["different_bytes"] += mismatched
                    if first_selected is not None:
                        first_selected["different_bytes"] = mismatched
                        detail.write(json.dumps(first_selected) + "\n")
                        remember(first_selected)

            except Exception as error:
                errors.append(dict(stage=stage, worker=str(obs.path.parent), error=str(error)))

    # A missing entire rank/group/layer is a coverage failure, not a match.
    p_ranks = {m.get("tp_rank", m["rank"] % m.get("tp_size", 1)) for _, m in workers if m["role"] == "P"}
    d_ranks = {m.get("tp_rank", m["rank"] % m.get("tp_size", 1)) for _, m in workers if m["role"] == "D"}
    for path, meta in workers:
        role, rank = meta["role"], meta.get("tp_rank", meta["rank"] % meta.get("tp_size", 1))
        if not meta.get("groups"):
            errors.append(dict(worker=str(path.parent), error="missing_group_metadata"))
        for group, layers in enumerate(meta.get("groups", [])):
            for layer in range(layers):
                stages = ("P_COMPUTE",) if role == "P" else ("D_CPU", "D_DIRECT")
                expected_components = {
                    obs.row["component"]
                    for obs in observations
                    if obs.meta["role"] == "P"
                    and obs.row["group"] == group
                    and obs.row["layer"] == layer
                    and obs.row["stage"] == "P_COMPUTE"
                } or {0}
                for component in expected_components:
                    if not any(index[(rank, group, layer, component, stage)] for stage in stages):
                        errors.append(
                            dict(
                                worker=str(path.parent),
                                error="missing_layer_component",
                                group=group,
                                layer=layer,
                                component=component,
                            )
                        )
    expected_tp = max((m.get("tp_size", 1) for _, m in workers), default=1)
    if p_ranks != set(range(expected_tp)) or d_ranks != set(range(expected_tp)):
        errors.append(dict(error="incomplete_PD_ranks", P=sorted(p_ranks), D=sorted(d_ranks), expected=expected_tp))
    first = min(
        differences,
        key=difference_key,
        default=None,
    )
    report = dict(
        format=FORMAT,
        request_id=request_id,
        status="differences"
        if counts["different_bytes"]
        else "incomplete"
        if errors or counts["unmatched_ranges"] or not observations
        else "sampled_bytes_equal",
        observations=len(observations),
        stages=dict(Counter(obs.row["stage"] for obs in observations)),
        workers=len(workers),
        **counts,
        errors=len(errors),
        first_error=errors[0] if errors else None,
        first_unmatched=missing[0] if missing else None,
        first_difference=first,
        first_boundary=BOUNDARIES.get(first["stage"]) if first else None,
        actual_D_sparse_observed=any(o.row["stage"] == "D_SELECTED" for o in observations),
        scope=(
            "First raw byte per logical token/component. "
            "Equality does not prove unsampled bytes or model arithmetic correct."
        ),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "coverage-errors.json").write_text(json.dumps(errors + missing, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--request-id", required=True)
    cli.add_argument("--hosts", "--ips", nargs="+", type=ssh_target)
    cli.add_argument("--workspace", default="/workspace")
    cli.add_argument("--log-dir", default="tp4_dp4_logs")
    cli.add_argument("--feature-dir", help="Absolute feature root if service cwd differs from workspace")
    cli.add_argument("--container")
    cli.add_argument("--password", "--ssh-password", dest="ssh_password")
    cli.add_argument("--password-env")
    cli.add_argument("--skip-host-key-check", action="store_true")
    cli.add_argument("--ssh-port", type=int)
    cli.add_argument("--identity-file", type=Path)
    cli.add_argument("--wait-seconds", type=float, default=120)
    cli.add_argument("--local", nargs="+", type=Path)
    cli.add_argument("--output", type=Path, default=Path("kv-feature-report"))
    args = cli.parse_args(argv)
    match = re.search(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.request_id)
    if match:
        args.request_id = match[0].lower()
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", args.request_id):
        cli.error("invalid request ID")
    args.password = args.password_per_host = False
    args.output.mkdir(parents=True, exist_ok=True)
    roots = list(args.local or [])
    failures = []
    if args.hosts:
        with password_auth(args), ThreadPoolExecutor(max_workers=min(8, len(args.hosts))) as pool:
            jobs = {pool.submit(collect_host, args, host): host for host in args.hosts}
            for job in as_completed(jobs):
                try:
                    roots.append(job.result())
                except Exception as error:
                    failures.append(f"{jobs[job]}: {error}")
    if not roots:
        cli.error("no feature data; provide --hosts or --local: " + "; ".join(failures))
    report = analyze(roots, args.request_id, args.output)
    if failures:
        report["collection_errors"] = failures
        if report["status"] == "sampled_bytes_equal":
            report["status"] = "incomplete"
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[KVFEATURE] " + json.dumps(report, separators=(",", ":")), flush=True)
    print("[KVFEATURE] report:", args.output / "report.json")
    return 0 if report["status"] == "sampled_bytes_equal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
