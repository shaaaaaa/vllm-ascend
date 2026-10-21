# SPDX-License-Identifier: Apache-2.0
"""Collect filtered P-node overlap logs through native OpenSSH."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from pd_tensor_collect import (
    add_password_arguments,
    host_directory,
    password_auth,
    remote_command,
    run_ssh,
    safe_component,
    safe_extract,
    ssh_port,
    ssh_target,
)

# Pack the complete filtered snapshot before sending it. Snapshot each opened
# file's size so a live server cannot make collection chase new log lines forever.
REMOTE_COLLECT = r"""
import io, json, os, pathlib, shutil, sys, tarfile, tempfile
root = pathlib.Path(sys.argv[1])
patterns = json.loads(sys.argv[2])
if not root.is_dir():
    raise ValueError('log directory does not exist: ' + str(root))
paths = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
if not paths:
    raise ValueError('no P logs matched in ' + str(root) + ': ' + repr(patterns))
metadata = {'schema_version': 1, 'source_directory': str(root), 'files': []}
with tempfile.TemporaryDirectory(prefix='prefill-overlap-') as temporary:
    archive = pathlib.Path(temporary) / 'overlap.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as packed:
        for path in paths:
            count = scanned = 0
            with path.open('rb') as source, tempfile.TemporaryFile() as filtered:
                # fstat describes the opened inode even if a log rotates now.
                remaining = os.fstat(source.fileno()).st_size
                while remaining:
                    line = source.readline(remaining)
                    if not line:
                        break
                    remaining -= len(line)
                    scanned += len(line)
                    if line.endswith(b'\n') and b'[PREFILL_OVERLAP]' in line:
                        filtered.write(line)
                        count += 1
                size = filtered.tell()
                filtered.seek(0)
                member = tarfile.TarInfo('logs/' + path.name)
                member.size = size
                packed.addfile(member, filtered)
            metadata['files'].append({'name': path.name, 'lines': count, 'scanned_bytes': scanned})
        payload = json.dumps(metadata).encode('utf-8')
        member = tarfile.TarInfo('_collection.json')
        member.size = len(payload)
        packed.addfile(member, io.BytesIO(payload))
    with archive.open('rb') as source:
        shutil.copyfileobj(source, sys.stdout.buffer, length=1024 * 1024)
    sys.stdout.buffer.flush()
"""


def add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hosts", nargs="+", type=ssh_target, help="P-node IPs, SSH aliases or user@IP")
    parser.add_argument("--repo-path", help="Remote vllm-ascend path; logs default to <repo-path>/tp4_dp4_logs")
    parser.add_argument("--remote-log-dir", help="Absolute remote log directory, overriding --repo-path")
    parser.add_argument(
        "--log-pattern",
        action="append",
        help="Remote log filename glob; repeat for multiple patterns (default: P_dp*.log)",
    )
    parser.add_argument(
        "--collect-dir", type=Path, help="Fresh local directory for collected logs; default: beside plot"
    )
    parser.add_argument("--container", type=safe_component, help="Read logs inside this Docker container")
    parser.add_argument("--ssh-port", type=ssh_port)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument(
        "--ssh-timeout", type=int, default=600, help="Per-host collection timeout in seconds (default: 600)"
    )
    add_password_arguments(parser)


def remote_log_directory(args: argparse.Namespace) -> str:
    if not args.remote_log_dir and not args.repo_path:
        raise ValueError("--hosts requires --repo-path or --remote-log-dir")
    value = args.remote_log_dir or args.repo_path
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("remote path must be an absolute POSIX path without '..' or control characters")
    if not args.remote_log_dir:
        path /= "tp4_dp4_logs"
    return str(path)


def collect_host(args: argparse.Namespace, target: str, source: str, root: Path) -> dict[str, Any]:
    directory = root / host_directory(target)
    result: dict[str, Any] = {"target": target, "directory": directory.name, "status": "failed"}
    try:
        directory.mkdir()
        command = remote_command(args, target, ["python3", "-c", REMOTE_COLLECT, source, json.dumps(args.log_pattern)])
        # OpenSSH handles password authentication via the shared native askpass
        # transport. Neither this command nor the saved metadata contains it.
        with (directory / "overlap.tar.gz").open("xb") as archive, (directory / "ssh.stderr").open("xb") as errors:
            process = run_ssh(args, target, command, stdout=archive, stderr=errors, timeout=args.ssh_timeout)
        if process.returncode:
            with (directory / "ssh.stderr").open("rb") as errors:
                errors.seek(max(0, errors.seek(0, 2) - 4096))
                message = errors.read().decode("utf-8", errors="replace").strip()
            raise ValueError(f"SSH exited {process.returncode}: {message}")
        snapshot = directory / "snapshot"
        snapshot.mkdir()
        safe_extract(directory / "overlap.tar.gz", snapshot)
        metadata = json.loads((snapshot / "_collection.json").read_text(encoding="utf-8"))
        lines = sum(item["lines"] for item in metadata["files"])
        result.update(status="success", files=metadata["files"])
        print(f"[OVERLAP_COLLECT] {target}: logs={len(metadata['files'])} overlap_lines={lines}", flush=True)
    except (OSError, ValueError, KeyError, TypeError, EOFError, tarfile.TarError, subprocess.SubprocessError) as error:
        message = (
            f"SSH collection timed out after {args.ssh_timeout}s"
            if isinstance(error, subprocess.TimeoutExpired)
            else str(error)
        )
        result.update(status="failed", error=message)
        print(f"[OVERLAP_COLLECT] {target}: FAILED: {message}", flush=True)
    return result


def collect_logs(args: argparse.Namespace) -> tuple[list[Path], Path]:
    source = remote_log_directory(args)
    if args.ssh_timeout < 1:
        raise ValueError("--ssh-timeout must be positive")
    args.log_pattern = args.log_pattern or ["P_dp*.log"]
    if any(not p or "/" in p or "\\" in p or any(ord(c) < 32 for c in p) for p in args.log_pattern):
        raise ValueError("--log-pattern must be a filename glob without directory separators")
    hosts = list(dict.fromkeys(args.hosts))
    args.hosts = hosts
    print(f"[OVERLAP_COLLECT] remote_log_dir={source} patterns={args.log_pattern}", flush=True)
    # Always collect a fresh snapshot. A previous run may have collected the same
    # log names before the later dialogue rounds completed.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with password_auth(args):
        if args.collect_dir:
            root = args.collect_dir.expanduser().resolve()
            root.mkdir(parents=True, exist_ok=False)
        else:
            root = Path(tempfile.mkdtemp(prefix=args.output.stem + "-logs-", dir=args.output.parent)).resolve()
        print(f"[OVERLAP_COLLECT] saving snapshot to {root}", flush=True)
        with ThreadPoolExecutor(max_workers=min(4, len(hosts))) as pool:
            results = list(pool.map(lambda host: collect_host(args, host, source, root), hosts))
    manifest = dict(schema_version=1, source_directory=source, patterns=args.log_pattern, hosts=results)
    (root / "collection.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if any(host["status"] != "success" for host in results):
        raise ValueError(f"Some P-node logs could not be collected; inspect {root / 'collection.json'} and retry")
    paths = sorted(root.glob("host-*/snapshot/logs/*"))
    return paths, root
