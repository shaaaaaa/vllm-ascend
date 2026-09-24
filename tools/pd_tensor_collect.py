#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect a PD tensor run over SSH without loading or executing its tensors.

Example:
    python tools/pd_tensor_collect.py --hosts user@host1 user@host2 \
        --repo-path /workspace/vllm-ascend --run-id check-001 \
        --container inference --output ./collected-check-001

The remote command only reads ``<repo>/pd-tensor-dump/<run>``. Successful
collections live under ``<output>/hosts/<safe-host>/`` and are never replaced.
Rerunning the same command skips successful hosts and retries failed hosts.
Interrupted downloads and extraction directories remain in ``partials/``.
Add ``--analyze-pd`` to compare the collected P and D KV with local CPU PyTorch
after every host has been collected successfully; reports go to ``report-pd/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
COPY_BUFFER_BYTES = 1024 * 1024
STDERR_TAIL_BYTES = 4096
HOST_MAX_LENGTH = 255
COMPONENT_MAX_LENGTH = 128
WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_component(value: str) -> str:
    """Accept a single run/container name, never a path or command option."""
    if (
        len(value) > COMPONENT_MAX_LENGTH
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None
        or value.endswith(".")
    ):
        raise argparse.ArgumentTypeError("use a simple name containing letters, digits, '.', '_' or '-'")
    return value


def ssh_target(value: str) -> str:
    """Accept SSH aliases, hostnames and IPs, optionally prefixed by a user."""
    if (
        not value
        or len(value) > HOST_MAX_LENGTH
        or value.startswith("-")
        or value.count("@") > 1
        or re.fullmatch(r"[A-Za-z0-9_\[][A-Za-z0-9_.@:\[\]%-]*", value) is None
    ):
        raise argparse.ArgumentTypeError("host must be a hostname/IP or user@host, without options or shell characters")
    user, separator, host = value.rpartition("@")
    host = host if separator else value
    if not host or host.startswith("-") or (separator and not user):
        raise argparse.ArgumentTypeError("invalid SSH host")
    return value


def ssh_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("SSH port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("SSH port must be between 1 and 65535")
    return port


def source_path(repo_path: str, run_id: str) -> str:
    """Build the absolute Linux source directory inside the host/container."""
    if not repo_path.startswith("/") or "\\" in repo_path or any(ord(c) < 32 or ord(c) == 127 for c in repo_path):
        raise ValueError("--repo-path must be an absolute POSIX path without control characters")
    path = PurePosixPath(repo_path)
    if ".." in path.parts:
        raise ValueError("--repo-path must not contain '..'")
    safe_component(run_id)
    return str(path / "pd-tensor-dump" / run_id)


def host_directory(target: str) -> str:
    """Use a portable, collision-resistant directory name for an SSH target."""
    ssh_target(target)
    readable = re.sub(r"[^A-Za-z0-9_.-]", "_", target)[:80]
    suffix = hashlib.sha256(target.encode("ascii")).hexdigest()[:12]
    return f"host-{readable}-{suffix}"


def ssh_command(args: argparse.Namespace, target: str, remote_source: str) -> list[str]:
    """Quote the remote argv once; leave SSH host verification unchanged."""
    ssh_target(target)
    remote = ["tar", "-C", remote_source, "-cf", "-", "--", "."]
    if args.container:
        safe_component(args.container)
        remote = ["docker", "exec", "--", args.container, *remote]
    command = ["ssh", "-T", "-o", "BatchMode=yes"]
    if args.ssh_port is not None:
        command.extend(["-p", str(args.ssh_port)])
    if args.identity_file is not None:
        command.extend(["-i", str(args.identity_file.expanduser().resolve())])
    return [*command, "--", target, shlex.join(remote)]


def member_path(member: tarfile.TarInfo) -> PurePosixPath:
    """Validate tar paths for Linux and Windows before creating any entry."""
    name = member.name
    if not name or "\\" in name or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ValueError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive path escapes destination: {name!r}")
    for component in path.parts:
        if (
            any(c in '<>:"|?*' for c in component)
            or component.endswith((".", " "))
            or component.split(".", 1)[0].lower() in WINDOWS_RESERVED_NAMES
        ):
            raise ValueError(f"unsafe archive path component: {component!r}")
    if not member.isdir() and not member.isfile():
        raise ValueError(f"archive links and special files are not accepted: {name!r}")
    if path == PurePosixPath(".") and not member.isdir():
        raise ValueError("archive root must be a directory")
    return path


def safe_extract(archive_path: Path, destination: Path) -> dict[str, int]:
    """Stream regular files into a fresh directory, rejecting links and escapes.

    No archive permissions, ownership, executable bits or timestamps are applied.
    A rejected archive may leave partial files inside ``destination`` only.
    """
    if not destination.is_dir() or destination.is_symlink() or any(destination.iterdir()):
        raise ValueError("extraction requires a fresh, empty, non-symlink directory")
    root = destination.resolve()
    seen: set[str] = set()
    files = total_bytes = 0
    with tarfile.open(archive_path, mode="r|*") as archive:
        for member in archive:
            relative = member_path(member)
            if relative == PurePosixPath("."):
                continue
            identity = relative.as_posix().casefold()
            if identity in seen:
                raise ValueError(f"duplicate archive path: {member.name!r}")
            seen.add(identity)
            target = root.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"archive path escapes destination: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"archive file has no payload: {member.name!r}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=COPY_BUFFER_BYTES)
            if target.stat().st_size != member.size:
                raise ValueError(f"incomplete archive file: {member.name!r}")
            files += 1
            total_bytes += member.size
    if not files:
        raise ValueError("archive contains no regular files")
    return {"files": files, "bytes": total_bytes}


def write_json(path: Path, data: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(prefix=".collection-", suffix=".json.tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def collection_lock(root: Path) -> Iterator[None]:
    lock = root / ".collection.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError(f"collection is locked: {lock}; check whether another collector is running") from error
    try:
        with stream:
            stream.write(f"pid={os.getpid()} started={timestamp()}\n")
        yield
    finally:
        lock.unlink()


def load_manifest(root: Path, args: argparse.Namespace, remote_source: str) -> dict[str, Any]:
    path = root / "collection.json"
    if path.is_symlink():
        raise ValueError("collection.json must not be a symlink")
    if not path.exists():
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            "source_path": remote_source,
            "container": args.container,
            "created_at": timestamp(),
            "complete": False,
            "hosts": [],
        }
    else:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for key, expected in (
            ("schema_version", SCHEMA_VERSION),
            ("run_id", args.run_id),
            ("source_path", remote_source),
            ("container", args.container),
        ):
            if manifest.get(key) != expected:
                raise ValueError(f"existing collection has a different {key}; use another --output")
        if not isinstance(manifest.get("hosts"), list):
            raise ValueError("invalid existing host manifest")
    existing = {}
    for host in manifest["hosts"]:
        if not isinstance(host, dict) or not isinstance(host.get("target"), str):
            raise ValueError("invalid existing host record")
        target = host["target"]
        expected_directory = f"hosts/{host_directory(target)}"
        if target in existing or host.get("directory") != expected_directory:
            raise ValueError("invalid or duplicate host directory in existing manifest")
        if host.get("status") not in ("pending", "running", "success", "failed") or not isinstance(
            host.get("attempts"), list
        ):
            raise ValueError("invalid existing host status or attempts")
        existing[target] = host
    for target in args.hosts:
        if target not in existing:
            host = {
                "target": target,
                "directory": f"hosts/{host_directory(target)}",
                "status": "pending",
                "attempts": [],
            }
            manifest["hosts"].append(host)
            existing[target] = host
    return manifest


def stderr_tail(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - STDERR_TAIL_BYTES))
        return stream.read(STDERR_TAIL_BYTES).decode("utf-8", errors="replace").strip()


def collect_host(root: Path, args: argparse.Namespace, remote_source: str, host: dict[str, Any]) -> None:
    target = host["target"]
    destination = root / host["directory"]
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"refusing to overwrite existing host directory: {destination}")
    prefix = host_directory(target) + "-"
    fd, archive_name = tempfile.mkstemp(prefix=prefix, suffix=".tar.partial", dir=root / "partials")
    archive_path = Path(archive_name)
    stderr_path = archive_path.with_suffix(".stderr")
    attempt: dict[str, Any] = {
        "started_at": timestamp(),
        "status": "running",
        "archive_path": archive_path.relative_to(root).as_posix(),
        "stderr_path": stderr_path.relative_to(root).as_posix(),
    }
    host.setdefault("attempts", []).append(attempt)
    host["status"] = "running"
    try:
        with os.fdopen(fd, "wb") as output, stderr_path.open("xb") as errors:
            result = subprocess.run(ssh_command(args, target, remote_source), stdout=output, stderr=errors, check=False)
        attempt["ssh_returncode"] = result.returncode
        if result.returncode:
            reason = stderr_tail(stderr_path)
            raise RuntimeError(f"SSH/tar exited {result.returncode}: {reason}")
        partial = Path(tempfile.mkdtemp(prefix=prefix, suffix=".extract.partial", dir=root / "partials"))
        attempt["partial_directory"] = partial.relative_to(root).as_posix()
        counts = safe_extract(archive_path, partial)
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"refusing to overwrite existing host directory: {destination}")
        partial.rename(destination)
        attempt.pop("partial_directory")
        attempt.update(counts)
        host.update(counts)
        # Delete only this attempt's local transport copy after publication.
        # Failed transfers and partial extraction directories remain untouched.
        try:
            archive_path.unlink()
            attempt.pop("archive_path")
        except OSError as error:
            attempt["cleanup_warning"] = str(error)
        attempt["status"] = host["status"] = "success"
        host.pop("error", None)
    except (Exception, KeyboardInterrupt) as error:
        attempt["status"] = host["status"] = "failed"
        attempt["error"] = host["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempt["finished_at"] = timestamp()


def collect(args: argparse.Namespace) -> int:
    """Collect all requested hosts; retain prior successes and report failures."""
    if len(set(args.hosts)) != len(args.hosts):
        raise ValueError("--hosts must not contain duplicates")
    remote_source = source_path(args.repo_path, args.run_id)
    root = args.output.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    for name in ("hosts", "partials"):
        path = root / name
        if path.is_symlink():
            raise ValueError(f"collection directory must not be a symlink: {path}")
        path.mkdir(exist_ok=True)
    with collection_lock(root):
        manifest = load_manifest(root, args, remote_source)
        manifest["complete"] = False
        write_json(root / "collection.json", manifest)
        try:
            for host in manifest["hosts"]:
                if host["status"] == "success":
                    destination = root / host["directory"]
                    if not destination.is_dir() or destination.is_symlink():
                        host["status"] = "failed"
                        host["error"] = "previous successful host directory is missing or replaced"
                        continue
                    print(f"[PD_TENSOR_COLLECT] {host['target']}: already collected; skipped", flush=True)
                    continue
                if host["target"] not in args.hosts:
                    continue
                try:
                    collect_host(root, args, remote_source, host)
                    print(
                        f"[PD_TENSOR_COLLECT] {host['target']}: files={host['files']} bytes={host['bytes']}", flush=True
                    )
                except Exception as error:
                    host["status"] = "failed"
                    host["error"] = f"{type(error).__name__}: {error}"
                    print(f"[PD_TENSOR_COLLECT] {host['target']}: FAILED: {error}", file=sys.stderr, flush=True)
                finally:
                    write_json(root / "collection.json", manifest)
        finally:
            manifest["updated_at"] = timestamp()
            manifest["complete"] = bool(manifest["hosts"]) and all(
                host["status"] == "success" for host in manifest["hosts"]
            )
            write_json(root / "collection.json", manifest)
    return 0 if manifest["complete"] else 1


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--hosts", nargs="+", required=True, type=ssh_target, help="SSH host aliases, IPs or user@host")
    cli.add_argument("--repo-path", required=True, help="Absolute vllm-ascend path on the host or inside --container")
    cli.add_argument("--run-id", required=True, type=safe_component, help="Name below pd-tensor-dump/")
    cli.add_argument(
        "--output", required=True, type=Path, help="Local collection directory; prior successful hosts are skipped"
    )
    cli.add_argument(
        "--container", type=safe_component, help="Read with docker exec inside this container on each host"
    )
    cli.add_argument("--ssh-port", type=ssh_port, help="Override the SSH port; otherwise honor SSH configuration")
    cli.add_argument(
        "--identity-file",
        type=Path,
        help="Optional SSH private-key path; key contents are never read or stored by this script",
    )
    cli.add_argument(
        "--analyze-pd",
        action="store_true",
        help="After all hosts succeed, compare P to D KV in report-pd/; requires local CPU PyTorch",
    )
    return cli


def analyze_pd(root: Path) -> int:
    """Load the optional tensor dependencies only for an explicit analysis."""
    output = root / "report-pd"
    try:
        from pd_tensor_analyze import analyze

        report = analyze([root], [root], mode="pd-kv", output=output)
        status = report["status"]
        summary = {
            "status": status,
            "compared_tensors": report.get("compared_tensors"),
            "different": report.get("counts", {}).get("different", 0),
            "issues": report.get("issues"),
            "report": str(output / "report.json"),
        }
        print("[PD_TENSOR_ANALYZE] " + json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if status == "analysis_complete" else 2
    except Exception as error:
        print(f"[PD_TENSOR_ANALYZE] FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        status = collect(args)
        if status != 0 or not args.analyze_pd:
            return status
        return analyze_pd(args.output.expanduser().resolve())
    except KeyboardInterrupt:
        print("[PD_TENSOR_COLLECT] interrupted; partial downloads retained", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"[PD_TENSOR_COLLECT] FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
