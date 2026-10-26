#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect hash-only KV traces over native SSH and locate the first bad boundary.

No KV files are read, downloaded or deserialized. Remote logs are filtered by
request UUID and compressed before transfer. Local files can be reanalysed with
--local. Password authentication uses OpenSSH askpass, without sshpass.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from pd_tensor_collect import host_directory, password_auth, remote_command, run_ssh, ssh_target

BOUNDARIES = {
    "MAP_ACTUAL": "attention bank mapping differs from the request's expected bank",
    "P_LOAD": "P historical bank differs from its CPU/prior-tail reference",
    "P_HISTORY_TAIL": "P partial history changed during this layer",
    "P_COMPUTE": "P historical KV changed during this layer",
    "P_SCATTER": "newly produced KV differs from the model-cache scatter destination",
    "P_CPU_STORE": "P post-compute bank differs from the completed CPU save",
    "P_CPU_LOAD": "P CPU reload differs from an earlier save for this request",
    "D_CPU": "D inherited CPU KV differs from P final prefill",
    "D_MIRROR": "D independent H2D mirror differs from its CPU source",
    "D_DIRECT": "D direct-import NPU destinations differ from P final prefill",
    "D_SELECTED": "D actual sparse-load destinations differ from the selected CPU history",
}


class Findings(list):
    """Count every issue while retaining bounded earliest examples."""

    def __init__(self):
        super().__init__()
        self.total = 0

    def append(self, row):
        self.total += 1

        def key(r):
            return (
                0 if r.get("role") == "P" else 1,
                r.get("frontier") or 0,
                r.get("layer") or 0,
                r.get("group") or 0,
                r.get("start") or 0,
            )

        if len(self) == 64 and key(row) >= key(self[-1]):
            return
        super().append(row)
        self.sort(key=key)
        del self[64:]


REMOTE = r"""
import gzip,json,pathlib,re,sys,time
workspace=pathlib.Path(sys.argv[1]); logs=pathlib.Path(sys.argv[2]); wanted=sys.argv[3]
if not logs.is_absolute():
    choices=[workspace/logs,workspace/'vllm-ascend'/logs]
    logs=next((p for p in choices if p.is_dir()),choices[0])
if not logs.is_dir():
    raise RuntimeError('log directory absent: '+str(logs))
deadline=time.monotonic()+(float(sys.argv[4]) if len(sys.argv)>4 else 0)
offsets={}; started=set(); completed=set()
while True:
    for path in sorted(logs.rglob('*.log')):
        with path.open('rb') as source:
            offset=offsets.get(path,0)
            source.seek(offset if path.stat().st_size>=offset else 0)
            while True:
                position=source.tell(); line=source.readline()
                if not line or not line.endswith(b'\n'):
                    offsets[path]=position; break
                if b'[KVTRACE] ' not in line or wanted.encode() not in line: continue
                try: record=json.loads(line.split(b'[KVTRACE] ',1)[1])
                except ValueError: continue
                identity=(record.get('pid'),record.get('role'),record.get('rank'))
                if record.get('stage')=='BEGIN': started.add(identity)
                if record.get('stage')=='END': completed.add(identity)
    if not started or started<=completed or time.monotonic()>=deadline: break
    time.sleep(1)
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as out:
    for path in sorted(logs.rglob('*.log')):
        tags=set()
        with path.open(errors='replace') as source:
            for line in source:
                if wanted in line and '[PFS]' in line:
                    match=re.search(r'\[PFS\]\s+p\d+\s+q\d+\b',line)
                    if match: tags.add(match[0])
        with path.open(errors='replace') as source:
            for number,line in enumerate(source,1):
                if '[KVTRACE] ' in line:
                    try: record=json.loads(line.split('[KVTRACE] ',1)[1])
                    except ValueError: continue
                else:
                    match=re.search(r'\[PFS\]\s+p\d+\s+q\d+\b',line)
                    if match is None or match[0] not in tags: continue
                    record={'req':wanted,'stage':'STORE_DIAGNOSTIC','text':line[match.end():].strip()}
                if wanted not in str(record.get('req','')) and record.get('req')!='unknown': continue
                rank=re.search(r'Worker_(?:DP\d+_)?TP(\d+)_',line)
                if rank: record['tp_rank']=int(rank[1])
                record['source_log']=path.name; record['source_line']=number
                out.write((json.dumps(record,separators=(',',':'))+'\n').encode())
"""


def collect_host(args, host):
    """Download only matching, compressed log records from one host."""
    path = args.output / (host_directory(host) + ".jsonl.gz")
    partial = path.with_suffix(path.suffix + ".partial")
    print(f"[KVTRACE] collecting {host}", flush=True)
    command = remote_command(
        args, host, ["python3", "-c", REMOTE, args.workspace, args.log_dir, args.request_id, str(args.wait_seconds)]
    )
    with partial.open("wb") as output:
        result = run_ssh(args, host, command, stdout=output, stderr=-1)
    if result.returncode:
        raise RuntimeError(f"{host}: SSH failed: {result.stderr.decode(errors='replace')[-1200:]}")
    partial.replace(path)
    print(f"[KVTRACE] collected {host}: {path.stat().st_size} compressed bytes")
    return path


def records(paths, request_id):
    """Stream plain/gzip JSONL, accepting service prefixes and packed hash rows."""
    for path in paths:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as source:
            for number, line in enumerate(source, 1):
                if "[KVTRACE] " in line:
                    prefix, line = line.split("[KVTRACE] ", 1)
                else:
                    prefix = ""
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if request_id not in str(row.get("req", "")) and row.get("req") != "unknown":
                    continue
                row = dict(row)
                rank = re.search(r"Worker_(?:DP\d+_)?TP(\d+)_", prefix)
                row["rank"] = int(rank[1]) if rank else row.get("tp_rank", row.get("rank", -1))
                row.setdefault("source_log", str(path))
                row.setdefault("source_line", number)
                yield row


def analyze(paths, request_id, output):
    """Compare each causal boundary; incomplete observations never pass."""
    database = output / "hashes.sqlite"
    if database.exists():
        database.unlink()
    db = sqlite3.connect(database)
    db.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA cache_size=-65536;
        CREATE TABLE frames(id INTEGER PRIMARY KEY, payload TEXT);
        CREATE TABLE hashes(stage TEXT, rank INTEGER, g INTEGER, l INTEGER,
            frontier INTEGER, start INTEGER, end INTEGER, component INTEGER,
            hash TEXT, final INTEGER, prefix INTEGER, frame INTEGER);
        CREATE INDEX lookup ON hashes(stage,rank,g,l,start,end,component,frontier);
    """)
    errors, selected, plans, ends, begins, store_diagnostics, count = Findings(), [], [], [], [], [], 0
    calibrations, bad_algorithm = set(), False
    algorithms = set()
    counts = {}
    stalls = []
    observed_stages = set()
    for row in records(paths, request_id):
        if row.get("algorithm"):
            algorithms.add(row["algorithm"])
        stage = row.get("stage", "")
        observed_stages.add(stage)
        counts[stage] = counts.get(stage, 0) + 1
        if stage == "STALL":
            if len(stalls) < 16:
                stalls.append(row)
            continue
        if row.get("error"):
            errors.append(row)
            continue
        if stage == "STORE_DIAGNOSTIC":
            if len(store_diagnostics) < 64:
                store_diagnostics.append(row)
            continue
        if stage == "PLAN":
            observed_stages.add(row["phase"])
            plans.append(row)
            continue
        if stage == "SOURCE":
            for start, end in row.get("ranges") or [(row["start"], row["end"])]:
                plans.append(dict(row, start=start, end=end))
            continue
        if stage == "BEGIN":
            begins.append(row)
            continue
        if stage == "END":
            ends.append(row)
            continue
        if stage == "D_SELECTED":
            selected.append(row)
            continue
        if stage == "CALIBRATION":
            calibrations.add((row.get("host"), row.get("pid"), row.get("rank"), row.get("role")))
            if not row.get("match"):
                bad_algorithm = True
                errors.append(dict(row, error="cpu_npu_checksum_algorithm_disagrees"))
            continue
        hashes = row.pop("hashes", None)
        entries = hashes if hashes is not None else [(row.get("start"), row.get("end"), row.get("hash"))]
        frame = db.execute(
            "INSERT INTO frames(payload) VALUES (?)", (json.dumps(row, separators=(",", ":")),)
        ).lastrowid
        values = []
        for start, end, digest in entries:
            if start is None or end is None or digest is None:
                errors.append(dict(row, error="malformed_hash_record"))
                continue
            values.append(
                (
                    stage,
                    row["rank"],
                    row["group"],
                    row["layer"],
                    row["frontier"],
                    start,
                    end,
                    row["component"],
                    digest,
                    int(row.get("final", False)),
                    int(row.get("prefix", 0)),
                    frame,
                )
            )
            count += 1
        db.executemany("INSERT INTO hashes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", values)
    db.commit()
    comparable_algorithms = len(algorithms) == 1
    if not comparable_algorithms:
        bad_algorithm = True
        errors.append(
            dict(
                stage="ALGORITHM",
                algorithms=sorted(algorithms),
                error="incompatible_hash_algorithms" if algorithms else "hash_algorithm_absent",
            )
        )

    @lru_cache(maxsize=512)
    def frame_payload(frame_id):
        return json.loads(db.execute("SELECT payload FROM frames WHERE id=?", (frame_id,)).fetchone()[0])

    def observation(frame_id, start, end, digest):
        return dict(frame_payload(frame_id), start=start, end=end, hash=digest)

    mismatches, missing, comparisons = Findings(), Findings(), 0
    chain = {
        "MAP_ACTUAL": "MAP_EXPECTED",
        "P_COMPUTE": "P_LOAD",
        "P_HISTORY_TAIL": "P_LOAD",
        "P_SCATTER": "P_PRODUCED",
        "P_CPU_STORE": "P_COMPUTE",
        "P_CPU_LOAD": "P_CPU_STORE",
        "P_LOAD": "P_CPU_LOAD",
        "D_CPU": "P_FINAL",
        "D_MIRROR": "D_CPU",
        "D_DIRECT": "P_FINAL",
    }
    if not comparable_algorithms:
        chain = {}
    with (output / "comparisons.jsonl").open("w", encoding="utf-8") as detail:
        for stage, reference_stage in chain.items():
            for rank, group, layer, frontier, start, end, component, digest, prefix, frame in db.execute(
                "SELECT rank,g,l,frontier,start,end,component,hash,prefix,frame FROM hashes "
                "WHERE stage=? ORDER BY frontier,l,g,start,component",
                (stage,),
            ):
                if stage == "P_COMPUTE" and end > prefix:
                    # Fresh query KV has no pre-load reference. The optional
                    # produced/scatter pair checks it independently.
                    continue
                condition = "frontier=?" if stage == "P_SCATTER" else "frontier<=?"
                if stage in ("MAP_ACTUAL", "P_COMPUTE", "P_HISTORY_TAIL"):
                    condition = "frontier=?"
                # CPU stores can finish at a truncated full-chunk frontier;
                # select the earliest computation that covers their end.
                if stage == "P_CPU_STORE":
                    condition = "frontier>=?"
                limit = (
                    frontier
                    if stage
                    in ("MAP_ACTUAL", "P_COMPUTE", "P_HISTORY_TAIL", "P_SCATTER", "P_LOAD", "P_CPU_LOAD", "D_MIRROR")
                    else end
                )
                order = "ASC" if stage == "P_CPU_STORE" else "DESC"
                if stage in ("D_CPU", "D_DIRECT"):
                    condition, limit, order = "final=1", None, "DESC"
                query = (
                    "SELECT hash,frame FROM hashes WHERE stage=? AND rank=? AND g=? AND l=? "
                    f"AND start=? AND end=? AND component=? AND {condition} ORDER BY frontier {order} LIMIT 1"
                )
                parameters = [reference_stage, rank, group, layer, start, end, component]
                if stage in ("D_CPU", "D_DIRECT"):
                    reference_rank = frame_payload(frame).get("source_rank")
                    if reference_rank is not None:
                        parameters[1] = int(reference_rank)
                    query = query.replace("stage=?", "stage IN (?,?)")
                    parameters.insert(0, "P_D_BOUNDARY")
                elif stage == "P_CPU_STORE":
                    query = query.replace("stage=?", "stage IN (?,?)")
                    parameters.insert(0, "P_STORE_BOUNDARY")
                if limit is not None:
                    parameters.append(limit)
                reference = db.execute(query, parameters).fetchone()
                if reference is None and stage == "P_CPU_LOAD" and rank != 0:
                    # Shared-CPU passive ranks read TP0's published payload;
                    # verify those bytes explicitly rather than assuming NPU KV.
                    parameters[1] = 0
                    reference = db.execute(query, parameters).fetchone()
                if reference is None and stage == "P_LOAD":
                    # A partial tail can be restored outside the full-chunk
                    # loader. Compare it with its actual prior save/compute,
                    # rather than declaring that unaligned range unknowable.
                    for prior_stage in ("P_CPU_STORE", "P_COMPUTE"):
                        for prior_rank in dict.fromkeys((rank, 0) if prior_stage == "P_CPU_STORE" else (rank,)):
                            operator = "<" if prior_stage == "P_COMPUTE" else "<="
                            reference = db.execute(
                                "SELECT hash,frame FROM hashes WHERE stage=? AND rank=? AND g=? AND l=? "
                                "AND start=? AND end=? AND component=? "
                                f"AND frontier{operator}? ORDER BY frontier DESC LIMIT 1",
                                (prior_stage, prior_rank, group, layer, start, end, component, frontier),
                            ).fetchone()
                            if reference is not None:
                                break
                        if reference is not None:
                            break
                if reference is None:
                    row = observation(frame, start, end, digest)
                    evidence = dict(row, reference_stage=reference_stage)
                    if stage == "P_LOAD":
                        available = db.execute(
                            "SELECT MAX(end) FROM hashes WHERE stage='P_CPU_LOAD' "
                            "AND rank=? AND g=? AND l=? AND frontier<=?",
                            (rank, group, layer, frontier),
                        ).fetchone()[0]
                        evidence.update(
                            source_end=available,
                            needed_prefix=row.get("prefix"),
                            reason="cpu_source_prefix_short"
                            if available is not None and available < row.get("prefix", end)
                            else "reference_segment_absent",
                        )
                    missing.append(evidence)
                    continue
                comparisons += 1
                equal = digest == reference[0]
                if not equal:
                    row = observation(frame, start, end, digest)
                    actual_reference = frame_payload(reference[1])
                    result = dict(
                        row,
                        reference_stage=actual_reference["stage"],
                        reference_hash=reference[0],
                        match=equal,
                        reference_log=actual_reference.get("source_log"),
                        reference_line=actual_reference.get("source_line"),
                    )
                    detail.write(json.dumps(result, separators=(",", ":")) + "\n")
                    mismatches.append(result)
        for row in selected if comparable_algorithms else ():
            comparisons += 1
            if not row.get("match"):
                detail.write(json.dumps(row, separators=(",", ":")) + "\n")
                mismatches.append(row)
    # Check every declared layer/range/component, not merely existing matches.
    observed = {
        (p["rank"], p.get("metadata_frontier", p["frontier"]), p["group"], p["layer"], p["phase"]) for p in plans
    }
    for begin in begins:
        if begin["role"] == "D":
            for group, layers in enumerate(begin["groups"]):
                for layer in range(layers):
                    required_end = int(begin.get("history_end", 0))
                    if required_end:
                        for stages in (("D_CPU", "D_DIRECT"), ("D_MIRROR", "D_DIRECT")):
                            spans = db.execute(
                                "SELECT DISTINCT start,end FROM hashes WHERE stage IN (?,?) "
                                "AND rank=? AND g=? AND l=? AND component=0 ORDER BY start,end",
                                (*stages, begin["rank"], group, layer),
                            )
                            covered = 0
                            for start, end in spans:
                                if start > covered:
                                    break
                                covered = max(covered, end)
                            if covered < required_end:
                                errors.append(
                                    dict(
                                        begin,
                                        group=group,
                                        layer=layer,
                                        stage=stages[0],
                                        observed_end=covered,
                                        error="decode_history_prefix_short",
                                    )
                                )
                    if (
                        db.execute(
                            "SELECT 1 FROM hashes WHERE stage='D_DIRECT' AND rank=? AND g=? AND l=? LIMIT 1",
                            (begin["rank"], group, layer),
                        ).fetchone()
                        is not None
                    ):
                        continue
                    for phase in ("D_CPU", "D_MIRROR"):
                        if (
                            db.execute(
                                "SELECT 1 FROM hashes WHERE stage=? AND rank=? AND g=? AND l=? LIMIT 1",
                                (phase, begin["rank"], group, layer),
                            ).fetchone()
                            is None
                        ):
                            errors.append(
                                dict(begin, stage=phase, group=group, layer=layer, error="decode_history_layer_absent")
                            )
            continue
        for group, layers in enumerate(begin["groups"]):
            for layer in range(layers):
                phases = ("P_LOAD", "P_COMPUTE", "P_FINAL") if begin.get("final") else ("P_LOAD", "P_COMPUTE")
                for phase in phases:
                    if (begin["rank"], begin["frontier"], group, layer, phase) not in observed:
                        errors.append(
                            dict(begin, stage=phase, group=group, layer=layer, error="model_layer_observation_absent")
                        )
    for plan in plans:
        for component in range(plan["components"]):
            found = db.execute(
                "SELECT DISTINCT start,end FROM hashes WHERE stage=? AND rank=? AND g=? AND l=? "
                "AND frontier=? AND component=? AND start>=? AND end<=? ORDER BY start",
                (
                    plan["phase"],
                    plan["rank"],
                    plan["group"],
                    plan["layer"],
                    plan["frontier"],
                    component,
                    plan["start"],
                    plan["end"],
                ),
            ).fetchall()
            position = plan["start"]
            for start, end in found:
                if start != position:
                    break
                position = end
            if position != plan["end"]:
                errors.append(dict(plan, error="incomplete_planned_range", component=component, observed_end=position))
    required = (
        "BEGIN",
        "CALIBRATION",
        "P_LOAD",
        "P_COMPUTE",
        "P_CPU_STORE",
        "D_CPU",
        "D_MIRROR",
    )
    for stage in required:
        if stage not in observed_stages:
            errors.append(dict(stage=stage, error="required_stage_absent"))
    if not ends:
        errors.append(dict(stage="END", error="request_end_absent"))
    ended = {(row.get("host"), row.get("pid"), row.get("rank"), row.get("role")) for row in ends}
    checked = set()
    for begin in begins:
        worker = begin.get("host"), begin.get("pid"), begin.get("rank"), begin.get("role")
        if worker not in ended and worker not in checked:
            errors.append(dict(begin, error="worker_hash_queue_not_complete"))
        if worker not in calibrations and worker not in checked:
            errors.append(dict(begin, error="worker_checksum_calibration_absent"))
        checked.add(worker)
    for role in ("P", "D"):
        workers = [row for row in begins if row.get("role") == role]
        if not workers:
            errors.append(dict(role=role, stage="BEGIN", error="role_manifest_absent"))
            continue
        expected = max(int(row.get("tp_size") or 0) for row in workers)
        observed_ranks = {row["rank"] for row in workers}
        for rank in sorted(set(range(expected)) - observed_ranks):
            errors.append(dict(role=role, rank=rank, stage="BEGIN", error="tp_rank_manifest_absent"))
    mismatches.sort(
        key=lambda r: (
            0 if r.get("role") == "P" else 1,
            r.get("frontier", 0),
            r.get("layer", 0),
            r.get("group", 0),
            r.get("start", 0),
        )
    )
    report = dict(
        schema=1,
        request_id=request_id,
        status="invalid_algorithm"
        if bad_algorithm
        else "differences"
        if mismatches
        else "incomplete"
        if errors or missing
        else "consistent",
        algorithm=next(iter(algorithms)) if comparable_algorithms else None,
        algorithms=sorted(algorithms),
        hash_records=count,
        comparisons=comparisons,
        different=mismatches.total,
        unmatched=missing.total,
        errors=errors.total,
        sparse_transfer_observed=bool(selected),
        first_stall=stalls[0] if stalls else None,
        store_diagnostics=store_diagnostics,
        stages=counts,
        first_difference=mismatches[0] if mismatches else None,
        first_boundary=BOUNDARIES.get(mismatches[0]["stage"]) if mismatches else None,
        first_error=errors[0] if errors else None,
        first_unmatched=missing[0] if missing else None,
        scope="KV movement and scatter only; consistent KV does not prove model arithmetic or sampling correct",
        details="comparisons.jsonl contains differences only; counters include every compared segment",
    )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "coverage-errors.jsonl").open("w", encoding="utf-8") as target:
        for row in errors + missing:
            target.write(json.dumps(row, separators=(",", ":")) + "\n")
    db.close()
    return report


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--request-id", required=True)
    cli.add_argument("--hosts", "--ips", nargs="+", type=ssh_target)
    cli.add_argument("--workspace", default="/workspace")
    cli.add_argument("--log-dir", default="tp4_dp4_logs")
    cli.add_argument("--container")
    cli.add_argument("--password", "--ssh-password", dest="ssh_password")
    cli.add_argument("--password-env")
    cli.add_argument("--skip-host-key-check", action="store_true")
    cli.add_argument("--ssh-port", type=int)
    cli.add_argument("--identity-file", type=Path)
    cli.add_argument(
        "--wait-seconds", type=float, default=120, help="Wait for queued hash END markers before collecting"
    )
    cli.add_argument("--local", nargs="+", type=Path)
    cli.add_argument("--output", type=Path, default=Path("kv-trace-report"))
    args = cli.parse_args(argv)
    request_uuid = re.search(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.request_id)
    if request_uuid:
        args.request_id = request_uuid[0]
    args.password = args.password_per_host = False
    args.output.mkdir(parents=True, exist_ok=True)
    paths = list(args.local or ())
    collection_errors = []
    if args.hosts:
        with password_auth(args), ThreadPoolExecutor(max_workers=min(len(args.hosts), 8)) as pool:
            jobs = {pool.submit(collect_host, args, host): host for host in args.hosts}
            for job in as_completed(jobs):
                try:
                    paths.append(job.result())
                except Exception as error:
                    message = f"{jobs[job]}: {error}"
                    collection_errors.append(message)
                    print(f"[KVTRACE] collection failed: {message}", flush=True)
    if not paths:
        cli.error("provide --hosts or --local")
    report = analyze(paths, args.request_id, args.output)
    if collection_errors:
        report["collection_errors"] = collection_errors
        if report["status"] == "consistent":
            report["status"] = "incomplete"
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[KVTRACE] "
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "algorithm",
                    "hash_records",
                    "comparisons",
                    "different",
                    "unmatched",
                    "errors",
                    "first_difference",
                    "first_boundary",
                    "first_error",
                    "first_unmatched",
                    "first_stall",
                )
            },
            ensure_ascii=False,
        )
    )
    print("[KVTRACE] report:", args.output / "report.json")
    return 0 if report["status"] == "consistent" else 1


if __name__ == "__main__":
    raise SystemExit(main())
