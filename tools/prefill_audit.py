#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect prefill audit files once and compare all requested transfer paths."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from multi_turn_output_review import read_client_log
from pd_tensor_collect import (
    host_directory,
    password_auth,
    remote_command,
    run_ssh,
    safe_extract,
    ssh_target,
)

REMOTE = r"""
import gzip,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1])
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as stream:
    with tarfile.open(fileobj=stream,mode='w|') as archive:
        for path in sorted(root.glob('audit-*.jsonl')):
            if path.is_file() and not path.is_symlink():
                archive.add(path,arcname=path.name,recursive=False)
"""


def collect_host(args: argparse.Namespace, host: str) -> Path:
    """Collect a read-only snapshot using the existing native SSH helper."""
    root = args.output / "hosts" / host_directory(host)
    root.mkdir(parents=True, exist_ok=True)
    source = args.audit_dir or str(PurePosixPath(args.workspace) / "prefill-audit")
    command = remote_command(args, host, ["python3", "-c", REMOTE, source])
    with tempfile.TemporaryDirectory(dir=root) as staging:
        archive = Path(staging) / "audit.tar.gz"
        with archive.open("wb") as output:
            result = run_ssh(args, host, command, stdout=output, stderr=-1)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace")[-1000:])
        extracted = Path(staging) / "files"
        extracted.mkdir()
        safe_extract(archive, extracted)
        for path in extracted.iterdir():
            path.replace(root / path.name)
    return root


class AuditEvidence:
    """Read and index a snapshot once, shared by every request comparison."""

    def __init__(self, roots: list[Path]):
        self.rows, self.damaged = [], []
        self.by_request, self.by_key = defaultdict(set), defaultdict(set)
        self.puts, self.gets = defaultdict(set), defaultdict(set)
        self.direct_samples = defaultdict(list)
        for path in sorted({path for root in roots for path in root.rglob("audit-*.jsonl")}):
            with path.open(encoding="utf-8", errors="replace") as source:
                for line in source:
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or "kind" not in row:
                            raise ValueError("Invalid audit record")
                        if row["kind"] == "payload" and len(row.get("keys", ())) != len(row.get("samples", ())):
                            raise ValueError("Incomplete payload record")
                    except ValueError:
                        self.damaged.append(str(path))
                        continue
                    row["file"] = str(path)
                    self.rows.append(row)
        self.rows.sort(key=lambda row: row.get("ns", 0))
        for position, row in enumerate(self.rows):
            for req in [row.get("req"), *row.get("reqs", ())]:
                if req:
                    self.by_request[str(req)].add(position)
            for key in row.get("keys", ()):
                self.by_key[key].add(position)
            if row["kind"] != "payload":
                continue
            if row["stage"] == "get_npu":
                self.direct_samples[row.get("host"), row.get("pid"), row.get("req")].append(set(row["keys"]))
            for key, sample in zip(row["keys"], row["samples"]):
                if row["stage"] == "put":
                    self.puts[key].add(tuple(sample))
                elif row["stage"] in ("get", "get_npu"):
                    self.gets[key].add(tuple(sample))

    def select(self, request_id: str | None) -> tuple[list[dict], set[str]]:
        """Include historical stores for this request's keys, not only its ID."""
        if not request_id:
            return self.rows, set(self.by_key)
        positions = set()
        for req, found in self.by_request.items():
            if request_id in req:
                positions.update(found)
        keys = {key for position in positions for key in self.rows[position].get("keys", ())}
        for key in keys:
            positions.update(self.by_key[key])
        return [self.rows[position] for position in sorted(positions)], keys


def analyze(roots: list[Path], request_id: str | None = None, *, evidence: AuditEvidence | None = None) -> dict:
    """Compare sampled byte streams, keeping missing evidence inconclusive."""
    evidence = evidence if evidence is not None else AuditEvidence(roots)
    rows, keys = evidence.select(request_id)
    puts, gets = evidence.puts, evidence.gets
    # Consumed markers belong to this report; never mutate the shared index.
    direct_samples = {key: list(values) for key, values in evidence.direct_samples.items()}
    issues, gaps = [], Counter()
    compared, stages = 0, Counter()
    for row in rows:
        if row.get("error"):
            issues.append(
                {
                    key: row.get(key)
                    for key in ("kind", "stage", "host", "pid", "req", "reqs", "error", "recent", "file")
                }
            )
        if row["kind"] == "coverage":
            gaps[row.get("detail", row.get("error", "unknown"))] += 1
        if row["kind"] == "direct_npu":
            observed = direct_samples.get((row.get("host"), row.get("pid"), row.get("req")), [])
            expected_keys = set(row.get("keys", ()))
            match = next(
                (i for i, sample in enumerate(observed) if expected_keys and expected_keys.issubset(sample)), None
            )
            if match is None:
                gaps["direct_npu_payload_not_sampled"] += 1
            else:
                observed.pop(match)
        if row["kind"] != "payload":
            continue
        stages[row["stage"]] += len(row["keys"])
        for key, sample in zip(row["keys"], row["samples"]):
            if key not in keys or row["stage"] not in ("get", "get_npu", "cache_read"):
                continue
            expected = puts[key] if row["stage"] in ("get", "get_npu") else (gets[key] or puts[key])
            if not expected:
                gaps["missing_source_sample"] += 1
                continue
            compared += 1
            if tuple(sample) not in expected:
                issues.append(
                    dict(
                        kind="payload",
                        stage=row["stage"],
                        host=row["host"],
                        pid=row["pid"],
                        req=row.get("req"),
                        key=key,
                        actual=sample,
                        expected=sorted(expected),
                        error="sample_mismatch",
                        file=row["file"],
                    )
                )
    if evidence.damaged:
        gaps["incomplete_json_line"] = len(evidence.damaged)
    if not keys:
        gaps["no_matching_payload"] += 1
    return dict(
        status="difference_found" if issues else "incomplete" if gaps else "samples_consistent",
        request_id=request_id,
        stages=dict(stages),
        compared=compared,
        issues=len(issues),
        first_issue=issues[0] if issues else None,
        gaps=dict(gaps),
        details=issues[:20],
        scope=(
            "Sampled CPU/direct-NPU transfer bytes and host-side bank protocol; "
            "no model arithmetic or graph correctness verdict."
        ),
    )


def analyze_many(roots: list[Path], clients: list[dict], request_ids: list[str], output: Path) -> dict:
    """Check every answer against one collected snapshot, including normal controls."""
    evidence = AuditEvidence(roots)
    targets = {row.get("response_id") or row["request_id"]: row for row in clients}
    for request_id in request_ids:
        targets.setdefault(request_id, {})
    directory = output / "requests"
    directory.mkdir(parents=True, exist_ok=True)
    with (output / "client_outputs.jsonl").open("w", encoding="utf-8") as destination:
        for row in clients:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = []
    for request_id, client in targets.items():
        report = analyze([], request_id, evidence=evidence)
        if client and not client.get("response_id"):
            report["gaps"]["missing_response_id_in_client_log"] = 1
            if report["status"] == "samples_consistent":
                report["status"] = "incomplete"
        report["client"] = {key: value for key, value in client.items() if key != "output"}
        filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", request_id) + ".json"
        (directory / filename).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries.append(
            dict(
                request_id=request_id,
                client_request_id=client.get("request_id"),
                session_id=client.get("session_id"),
                round=client.get("round"),
                priority=client.get("priority"),
                symptoms=client.get("symptoms", []),
                output_present=client.get("output_present"),
                previous_suspect_request_id=client.get("previous_suspect_request_id"),
                audit_status=report["status"],
                compared=report["compared"],
                issues=report["issues"],
                gaps=report["gaps"],
                first_issue=report["first_issue"],
                report=str(Path("requests") / filename),
            )
        )
    statuses = Counter(row["audit_status"] for row in summaries)
    symptoms = Counter(symptom for client in clients for symptom in client.get("symptoms", []))
    with (output / "requests.csv").open("w", newline="", encoding="utf-8-sig") as destination:
        fields = (
            "session_id",
            "round",
            "request_id",
            "priority",
            "symptoms",
            "audit_status",
            "compared",
            "issues",
            "gaps",
            "previous_suspect_request_id",
            "report",
        )
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow({**row, "symptoms": ";".join(row["symptoms"]), "gaps": json.dumps(row["gaps"])})
    return dict(
        mode="multi_request",
        status="difference_found"
        if statuses["difference_found"]
        else "incomplete"
        if statuses["incomplete"]
        else "samples_consistent",
        request_count=len(summaries),
        audit_statuses=dict(statuses),
        output_symptoms=dict(symptoms),
        requests=summaries,
        interpretation=(
            "All client records are checked, not only flagged answers. Text flags are review hints, "
            "not proof of KV corruption. Generated user turns may arise from completions without stop strings. "
            "A later round includes earlier answers in its input. "
            "Sample equality does not verify all bytes or arithmetic."
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", nargs="+", action="extend", help="One or more IDs; may be repeated")
    parser.add_argument(
        "--client-log", type=Path, help="MULTI_TURN tee log; check EVERY recorded request, not only anomalies"
    )
    parser.add_argument("--local", nargs="+", type=Path)
    parser.add_argument("--hosts", nargs="+", type=ssh_target)
    parser.add_argument(
        "--workspace",
        "--repo-path",
        default="/workspace/vllm-ascend",
        help="Absolute vllm-ascend repository directory on each host",
    )
    parser.add_argument(
        "--audit-dir", help="Absolute remote audit directory if it differs from workspace/prefill-audit"
    )
    parser.add_argument("--container")
    parser.add_argument("--password", "--ssh-password", dest="ssh_password")
    parser.add_argument("--password-env")
    parser.add_argument("--skip-host-key-check", action="store_true")
    parser.add_argument("--ssh-port", type=int)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("prefill-audit-report"))
    args = parser.parse_args()
    args.password = args.password_per_host = False
    args.output.mkdir(parents=True, exist_ok=True)
    clients = read_client_log(args.client_log) if args.client_log else []
    if args.client_log and not clients:
        parser.error("--client-log contains no MULTI_TURN metrics or output records")
    roots, errors = list(args.local or []), []
    if args.hosts:
        with password_auth(args), ThreadPoolExecutor(max_workers=min(8, len(args.hosts))) as pool:
            jobs = {pool.submit(collect_host, args, host): host for host in args.hosts}
            for job in as_completed(jobs):
                try:
                    roots.append(job.result())
                except Exception as error:
                    errors.append(f"{jobs[job]}: {error}")
    if not roots:
        parser.error("provide --local or --hosts; " + "; ".join(errors))
    request_ids = args.request_id or []
    if clients or len(request_ids) > 1:
        report = analyze_many(roots, clients, request_ids, args.output)
    else:
        report = analyze(roots, request_ids[0] if request_ids else None)
    if errors:
        report["collection_errors"] = errors
        if report["status"] == "samples_consistent":
            report["status"] = "incomplete"
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report.get("mode") == "multi_request":
        compact = {key: report[key] for key in ("status", "request_count", "audit_statuses", "output_symptoms")}
        for row in report["requests"]:
            print(
                f"[PFA] session={row['session_id']} round={row['round']} id={row['request_id']} "
                f"output={row['priority']} audit={row['audit_status']} "
                f"compared={row['compared']} issues={row['issues']}"
            )
    else:
        compact = {key: report[key] for key in ("status", "stages", "compared", "issues", "gaps", "first_issue")}
    print("[PFA] " + json.dumps(compact, separators=(",", ":")))
    print("[PFA] report=" + str(args.output / "report.json"))
    return int(report["status"] != "samples_consistent")


if __name__ == "__main__":
    raise SystemExit(main())
