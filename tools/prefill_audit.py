#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect CPU-only prefill audit files and locate the first observed boundary error."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

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


def analyze(roots: list[Path], request_id: str | None = None) -> dict:
    """Compare sampled byte streams, keeping missing evidence inconclusive."""
    rows, damaged = [], []
    for path in sorted({path for root in roots for path in root.rglob("audit-*.jsonl")}):
        for line in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
                if row.get("kind") == "payload" and len(row.get("keys", ())) != len(row.get("samples", ())):
                    damaged.append(str(path))
                    continue
                row["file"] = str(path)
                rows.append(row)
            except ValueError:
                damaged.append(str(path))
    keys = set()
    if request_id:
        for row in rows:
            if request_id in str(row.get("req", "")):
                keys.update(row.get("keys", ()))
    else:
        keys = {key for row in rows for key in row.get("keys", ())}

    puts, gets = defaultdict(set), defaultdict(set)
    direct_samples = defaultdict(list)
    for row in rows:
        if row.get("kind") != "payload":
            continue
        if row["stage"] == "get_npu":
            direct_samples[row.get("host"), row.get("pid"), row.get("req")].append(set(row["keys"]))
        for key, sample in zip(row["keys"], row["samples"]):
            if row["stage"] == "put":
                puts[key].add(tuple(sample))
            elif row["stage"] in ("get", "get_npu"):
                gets[key].add(tuple(sample))
    issues, gaps = [], Counter()
    compared, stages = 0, Counter()
    for row in sorted(rows, key=lambda item: item.get("ns", 0)):
        matches = (
            not request_id
            or request_id in str(row.get("req", ""))
            or any(request_id in str(req) for req in row.get("reqs", ()))
            or bool(keys.intersection(row.get("keys", ())))
        )
        if not matches:
            continue
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
            observed = direct_samples[row.get("host"), row.get("pid"), row.get("req")]
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
    if damaged:
        gaps["incomplete_json_line"] = len(damaged)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id")
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
    report = analyze(roots, args.request_id)
    if errors:
        report["collection_errors"] = errors
        if report["status"] == "samples_consistent":
            report["status"] = "incomplete"
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {key: report[key] for key in ("status", "stages", "compared", "issues", "gaps", "first_issue")}
    print("[PFA] " + json.dumps(compact, separators=(",", ":")))
    print("[PFA] report=" + str(args.output / "report.json"))
    return int(report["status"] != "samples_consistent")


if __name__ == "__main__":
    raise SystemExit(main())
