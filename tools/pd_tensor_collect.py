#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect a PD tensor run over SSH without loading or executing its tensors.

Example:
    python tools/pd_tensor_collect.py --hosts user@host1 user@host2 \
        --repo-path /workspace/vllm-ascend --run-id check-001 \
        --container inference --output ./collected-check-001

Each host first packs ``<repo>/pd-tensor-dump/<run>`` into a temporary gzip
archive, then sends that archive over SSH. The original dump is never modified.
Temporary remote archives are cleaned up after transfer. Successful
collections live under ``<output>/hosts/<safe-host>/`` and are never replaced.
Rerunning the same command skips successful hosts and retries failed/empty hosts.
Hosts without this run's files are reported as empty; they need not handle a request.
Interrupted downloads and extraction directories remain in ``partials/``.
Add ``--analyze-pd`` to compare the collected P and D KV with local CPU PyTorch
after collection (idle hosts may be empty); reports go to ``report-pd/``.
"""

from __future__ import annotations

import argparse
import getpass
import gzip
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
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
COPY_BUFFER_BYTES = 1024 * 1024
STDERR_TAIL_BYTES = 4096
HOST_MAX_LENGTH = 255
COMPONENT_MAX_LENGTH = 128
MISSING_RUN_EXIT = 44
MISSING_RUN_MARKER = "PD_TENSOR_RUN_ABSENT"
REMOTE_ARCHIVE_MARKER = "PD_TENSOR_ARCHIVE "
COMPRESSION_THREADS = 4
# Private transport state, only in a short-lived SSH child's environment; this
# is not an engine setting and is never passed to a vLLM worker or saved on disk.
ASKPASS_PASSWORD_ENV = "_PD_TENSOR_SSH_PASSWORD"
# Shared with the cleanup tool. Resolve and validate the dedicated dump directory
# remotely, including inside docker exec, before reading or removing its children.
REMOTE_PATH_CHECK = """
import pathlib, shutil, subprocess, sys
repo = pathlib.Path(sys.argv[1]).resolve(strict=True)
if not repo.is_dir():
    raise ValueError('repository is not a directory')
root = repo / 'pd-tensor-dump'
def checked_directory(path):
    if path.is_symlink():
        raise ValueError('dump directory must not be a symlink: ' + str(path))
    try:
        path.stat()
    except FileNotFoundError:
        return False
    if not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError('invalid dump directory: ' + str(path))
    return True
"""
REMOTE_COLLECT = (
    REMOTE_PATH_CHECK
    + f"""
import gzip, json, tempfile, time
run = root / sys.argv[2]
if not checked_directory(root) or not checked_directory(run):
    print({MISSING_RUN_MARKER!r}, file=sys.stderr)
    sys.exit({MISSING_RUN_EXIT})
started = time.monotonic()
pigz = shutil.which('pigz')
# Keep the temporary archive outside the dump tree, on the repository disk
# rather than /dev/shm or a potentially small /tmp. No uncompressed copy.
with tempfile.TemporaryDirectory(prefix='.pd-tensor-collect-', dir=repo) as temporary:
    archive = pathlib.Path(temporary) / 'run.tar.gz'
    with archive.open('xb') as packed:
        with subprocess.Popen(['tar', '-C', str(run), '-cf', '-', '--', '.'],
                              stdout=subprocess.PIPE) as tar:
            try:
                if pigz:
                    subprocess.run([pigz, '-1', '-p', '{COMPRESSION_THREADS}'],
                                   stdin=tar.stdout, stdout=packed, check=True)
                else:
                    with gzip.GzipFile(fileobj=packed, mode='wb', compresslevel=1) as compressed:
                        shutil.copyfileobj(tar.stdout, compressed, length={COPY_BUFFER_BYTES})
            finally:
                tar.stdout.close()
            if tar.wait():
                raise RuntimeError('remote tar failed; archive will not be sent')
    metadata = {{'compression': 'pigz' if pigz else 'python-gzip',
                 'archive_bytes': archive.stat().st_size,
                 'pack_seconds': round(time.monotonic() - started, 3)}}
    print({REMOTE_ARCHIVE_MARKER!r} + json.dumps(metadata), file=sys.stderr, flush=True)
    # Nothing is sent until the complete compressed archive is ready.
    with archive.open('rb') as source:
        shutil.copyfileobj(source, sys.stdout.buffer, length={COPY_BUFFER_BYTES})
    sys.stdout.buffer.flush()
"""
)
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
    """Collect one run, distinguishing an absent run from transport failures."""
    path = PurePosixPath(remote_source)
    return remote_command(args, target, ["python3", "-c", REMOTE_COLLECT, str(path.parent.parent), path.name])


def remote_command(args: argparse.Namespace, target: str, remote: list[str]) -> list[str]:
    """Quote remote argv once; passwords never appear in argv or manifests."""
    ssh_target(target)
    if args.container:
        safe_component(args.container)
        remote = ["docker", "exec", "--", args.container, *remote]
    use_password = target in getattr(args, "_passwords", {})
    command = ["ssh", "-T", "-o", "BatchMode=no" if use_password else "BatchMode=yes"]
    if getattr(args, "skip_host_key_check", False):
        command.extend(
            [
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                f"UserKnownHostsFile={os.devnull}",
                "-o",
                f"GlobalKnownHostsFile={os.devnull}",
            ]
        )
    elif use_password:
        command.extend(["-o", "StrictHostKeyChecking=yes"])
    if use_password:
        command.extend(
            [
                "-o",
                "NumberOfPasswordPrompts=1",
                "-o",
                "PreferredAuthentications=password,keyboard-interactive",
            ]
        )
    if args.ssh_port is not None:
        command.extend(["-p", str(args.ssh_port)])
    if args.identity_file is not None:
        command.extend(["-i", str(args.identity_file.expanduser().resolve())])
    return [*command, "--", target, shlex.join(remote)]


def add_password_arguments(cli: argparse.ArgumentParser) -> None:
    cli.add_argument(
        "--skip-host-key-check",
        action="store_true",
        help="Skip SSH server identity verification for this command; do not read or update known_hosts",
    )
    group = cli.add_mutually_exclusive_group()
    group.add_argument("--password", action="store_true", help="Prompt once for the SSH password shared by all hosts")
    group.add_argument(
        "--password-env", metavar="NAME", help="Read a shared SSH password from this environment variable"
    )
    group.add_argument(
        "--password-per-host", action="store_true", help="Prompt separately for each host's SSH password"
    )


def write_askpass(directory: Path, *, windows: bool) -> Path:
    """Create OpenSSH's native password callback, containing no credentials.

    Linux needs only /bin/sh; Windows uses the Python already running this tool.
    Password bytes come from the child environment, never inserted into code.
    """
    helper = directory / ("askpass.cmd" if windows else "askpass")
    if windows:
        python = sys.executable.replace("%", "%%")
        callback = (
            "import os,sys;"
            "sys.exit(1) if os.environ.get('SSH_ASKPASS_PROMPT') in ('confirm','none') else None;"
            f"sys.stdout.buffer.write(os.environ[{ASKPASS_PASSWORD_ENV!r}].encode('utf-8')+bytes([10]))"
        )
        content = f'@echo off\r\n"{python}" -I -c "{callback}"\r\n'
    else:
        content = (
            '#!/bin/sh\ncase "${SSH_ASKPASS_PROMPT-}" in confirm|none) exit 1 ;; esac\n'
            f"printf '%s\\n' \"${ASKPASS_PASSWORD_ENV}\"\n"
        )
    with helper.open("x", encoding="utf-8", newline="") as stream:
        stream.write(content)
    helper.chmod(0o700)
    return helper


@contextmanager
def password_auth(args: argparse.Namespace) -> Iterator[None]:
    """Keep passwords in memory and provide a temporary native askpass helper."""
    args._passwords = {}
    args._askpass = None
    with ExitStack() as stack:
        try:
            if not (args.password or args.password_env or args.password_per_host):
                yield
                return
            if args.password_per_host:
                args._passwords = {host: getpass.getpass(f"SSH password for {host}: ") for host in args.hosts}
            else:
                password = os.environ.get(args.password_env) if args.password_env else getpass.getpass("SSH password: ")
                if password is None:
                    raise ValueError(f"password environment variable is not set: {args.password_env}")
                args._passwords = dict.fromkeys(args.hosts, password)
            if any(not password for password in args._passwords.values()):
                raise ValueError("SSH password must not be empty")
            if any(any(c in password for c in "\r\n\0") for password in args._passwords.values()):
                raise ValueError("SSH askpass passwords must not contain newline or NUL characters")
            parent = Path(tempfile.gettempdir()).resolve()
            temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix="pd-tensor-askpass-", dir=parent))
            directory = Path(temporary).resolve()
            if directory.parent != parent:
                raise ValueError("temporary askpass directory escaped its parent")
            args._askpass = str(write_askpass(directory, windows=os.name == "nt"))
            yield
        finally:
            args._passwords.clear()
            args._askpass = None


def run_ssh(args: argparse.Namespace, target: str, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    passwords = getattr(args, "_passwords", {})
    if target in passwords:
        kwargs["env"] = dict(os.environ)
        kwargs["env"].update(
            {
                ASKPASS_PASSWORD_ENV: passwords[target],
                "SSH_ASKPASS": args._askpass,
                "SSH_ASKPASS_REQUIRE": "force",
                # Older OpenSSH ignores REQUIRE. A detached session plus DISPLAY
                # activates askpass there too; the callback does not use a GUI.
                "DISPLAY": os.environ.get("DISPLAY") or "pd-tensor:0",
            }
        )
        kwargs["stdin"] = subprocess.DEVNULL
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
    return subprocess.run(command, check=False, **kwargs)


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
    with ExitStack() as stack:
        raw = stack.enter_context(archive_path.open("rb"))
        is_gzip = raw.read(2) == b"\x1f\x8b"
        raw.seek(0)
        stream = stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb")) if is_gzip else raw
        archive = stack.enter_context(tarfile.open(fileobj=stream, mode="r|"))
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
        # Tar ends before the gzip trailer. Drain it to verify CRC and length;
        # otherwise an interrupted download can be published as successful.
        if is_gzip:
            while stream.read(COPY_BUFFER_BYTES):
                pass
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
        if host.get("status") not in ("pending", "running", "success", "empty", "failed") or not isinstance(
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
    fd, archive_name = tempfile.mkstemp(prefix=prefix, suffix=".tar.gz.partial", dir=root / "partials")
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
        print(f"[PD_TENSOR_COLLECT] {target}: packing remotely, then downloading .tar.gz", flush=True)
        started = time.monotonic()
        with os.fdopen(fd, "wb") as output, stderr_path.open("xb") as errors:
            result = run_ssh(args, target, ssh_command(args, target, remote_source), stdout=output, stderr=errors)
        attempt["pack_download_seconds"] = round(time.monotonic() - started, 3)
        attempt["archive_bytes"] = archive_path.stat().st_size
        attempt["ssh_returncode"] = result.returncode
        if (
            result.returncode == MISSING_RUN_EXIT
            and stderr_tail(stderr_path) == MISSING_RUN_MARKER
            and archive_path.stat().st_size == 0
        ):
            attempt["status"] = host["status"] = "empty"
            attempt["reason"] = host["reason"] = "run directory absent"
            host.update(files=0, bytes=0)
            host.pop("error", None)
            archive_path.unlink()
            attempt.pop("archive_path")
            return
        if result.returncode:
            reason = stderr_tail(stderr_path)
            raise RuntimeError(f"SSH/tar exited {result.returncode}: {reason}")
        metadata = next(
            (
                line.removeprefix(REMOTE_ARCHIVE_MARKER)
                for line in stderr_tail(stderr_path).splitlines()
                if line.startswith(REMOTE_ARCHIVE_MARKER)
            ),
            None,
        )
        if metadata is not None:
            attempt["remote_archive"] = json.loads(metadata)
            if attempt["remote_archive"]["archive_bytes"] != attempt["archive_bytes"]:
                raise ValueError("downloaded archive size differs from remote archive")
        print(
            f"[PD_TENSOR_COLLECT] {target}: downloaded {attempt['archive_bytes']} bytes; extracting locally",
            flush=True,
        )
        partial = Path(tempfile.mkdtemp(prefix=prefix, suffix=".extract.partial", dir=root / "partials"))
        attempt["partial_directory"] = partial.relative_to(root).as_posix()
        started = time.monotonic()
        counts = safe_extract(archive_path, partial)
        attempt["extract_seconds"] = round(time.monotonic() - started, 3)
        if not counts["files"]:
            attempt.update(counts)
            host.update(counts)
            attempt["status"] = host["status"] = "empty"
            attempt["reason"] = host["reason"] = "run contains no files"
            host.pop("error", None)
            # Keep the empty extraction in partials; do not publish it as a
            # successful immutable download. A subsequent collect retries it.
            archive_path.unlink()
            attempt.pop("archive_path")
            return
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
        host.pop("reason", None)
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
                    detail = (
                        f"empty ({host['reason']}); will retry on next collect"
                        if host["status"] == "empty"
                        else f"files={host['files']} bytes={host['bytes']}"
                    )
                    attempt = host["attempts"][-1]
                    if host["status"] == "success":
                        detail += (
                            f" archive_bytes={attempt['archive_bytes']}"
                            f" pack_download_s={attempt['pack_download_seconds']:.1f}"
                            f" extract_s={attempt['extract_seconds']:.1f}"
                        )
                        if "remote_archive" in attempt:
                            remote = attempt["remote_archive"]
                            detail += f" compressor={remote['compression']} pack_s={remote['pack_seconds']:.1f}"
                    print(f"[PD_TENSOR_COLLECT] {host['target']}: {detail}", flush=True)
                except Exception as error:
                    host["status"] = "failed"
                    host["error"] = f"{type(error).__name__}: {error}"
                    print(f"[PD_TENSOR_COLLECT] {host['target']}: FAILED: {error}", file=sys.stderr, flush=True)
                finally:
                    write_json(root / "collection.json", manifest)
        finally:
            manifest["updated_at"] = timestamp()
            manifest["complete"] = (
                bool(manifest["hosts"])
                and all(host["status"] in ("success", "empty") for host in manifest["hosts"])
                and any(host["status"] == "success" for host in manifest["hosts"])
            )
            manifest["has_data"] = any(host["status"] == "success" for host in manifest["hosts"])
            write_json(root / "collection.json", manifest)
    if not manifest["has_data"]:
        print(
            "[PD_TENSOR_COLLECT] no dump data found on any host; check run-id and request completion", file=sys.stderr
        )
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
        help="Match P/D by request ID and TP rank across all hosts; idle hosts may be empty; requires CPU PyTorch",
    )
    cli.add_argument(
        "--analysis-workers",
        type=int,
        default=16,
        help="Concurrent local analysis threads across ranks and layers (default: 16); use 1 for serial analysis",
    )
    add_password_arguments(cli)
    return cli


def analyze_pd(root: Path, workers: int = 16) -> int:
    """Load the optional tensor dependencies only for an explicit analysis."""
    output = root / "report-pd"
    try:
        print("[PD_TENSOR_COLLECT] downloads complete; analyzing local tensors", flush=True)
        started = time.monotonic()
        from pd_tensor_analyze import analyze

        report = analyze([root], [root], mode="pd-kv", output=output, workers=workers)
        print(f"[PD_TENSOR_COLLECT] local analysis finished in {time.monotonic() - started:.1f}s", flush=True)
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
        if args.analysis_workers < 1:
            raise ValueError("--analysis-workers must be a positive integer")
        with password_auth(args):
            status = collect(args)
        if status != 0 or not args.analyze_pd:
            return status
        return analyze_pd(args.output.expanduser().resolve(), workers=args.analysis_workers)
    except KeyboardInterrupt:
        print("[PD_TENSOR_COLLECT] interrupted; partial downloads retained", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"[PD_TENSOR_COLLECT] FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
