#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Remove remote PD tensor dumps after requests and collection have finished.

Example (same four hosts and run ID as pd_tensor_collect.py):
    python tools/pd_tensor_cleanup.py --hosts user@p1 user@p2 user@d1 user@d2 \
        --repo-path /workspace/vllm-ascend --run-id case-on --password

Only <repo>/pd-tensor-dump/<run-id> is removed. Use --all-runs explicitly to
empty <repo>/pd-tensor-dump instead. The parent dump directory is preserved.
No local collected files, model files or Mooncake data are removed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pd_tensor_collect import (
    REMOTE_PATH_CHECK,
    add_password_arguments,
    password_auth,
    remote_command,
    run_ssh,
    safe_component,
    source_path,
    ssh_port,
    ssh_target,
)

REMOTE_CLEANUP = (
    REMOTE_PATH_CHECK
    + """
name, dry_run = sys.argv[2], sys.argv[3] == '1'
if not checked_directory(root):
    print(json.dumps({'status': 'absent', 'path': str(root), 'entries': 0}))
    sys.exit(0)
if name:
    target = root / name
    if target.parent != root or target.name in ('', '.', '..'):
        raise ValueError('run path escapes dump directory')
    targets = [target] if checked_directory(target) else []
else:
    targets = list(root.iterdir())
# Validate every target before deleting any. rmtree does not follow descendant
# symlinks; top-level symlinks are rejected so an unexpected target is visible.
for target in targets:
    if target.is_symlink() or target.resolve(strict=True).parent != root:
        raise ValueError('cleanup target is outside dump directory: ' + str(target))
if not dry_run:
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
print(json.dumps({'status': 'preview' if dry_run else ('removed' if targets else 'absent'),
                  'path': str(root / name), 'entries': len(targets)}))
"""
)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--hosts", nargs="+", required=True, type=ssh_target)
    cli.add_argument("--repo-path", required=True, help="Absolute repository path on the host or inside --container")
    scope = cli.add_mutually_exclusive_group(required=True)
    scope.add_argument("--run-id", type=safe_component, help="Remove just this run, preserving all other runs")
    scope.add_argument("--all-runs", action="store_true", help="Remove all contents below pd-tensor-dump/")
    cli.add_argument("--container", type=safe_component, help="Execute inside this Docker container on each host")
    cli.add_argument("--ssh-port", type=ssh_port)
    cli.add_argument("--identity-file", type=Path)
    cli.add_argument("--dry-run", action="store_true", help="Report the scope without deleting files")
    add_password_arguments(cli)
    return cli


def cleanup(args: argparse.Namespace) -> int:
    if len(set(args.hosts)) != len(args.hosts):
        raise ValueError("--hosts must not contain duplicates")
    # Reuse collection's path validation, even when clearing all runs.
    source_path(args.repo_path, args.run_id or "all-runs")
    script = "import json\n" + REMOTE_CLEANUP
    failures = 0
    with password_auth(args):
        for target in args.hosts:
            try:
                command = remote_command(
                    args,
                    target,
                    ["python3", "-c", script, args.repo_path, args.run_id or "", "1" if args.dry_run else "0"],
                )
                result = run_ssh(args, target, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode:
                    reason = result.stderr.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"SSH/cleanup exited {result.returncode}: {reason}")
                record = json.loads(result.stdout)
                if record.get("status") not in ("removed", "absent", "preview"):
                    raise ValueError("invalid remote cleanup result")
                print(f"[PD_TENSOR_CLEANUP] {target}: {json.dumps(record, ensure_ascii=False)}", flush=True)
            except Exception as error:
                failures += 1
                print(f"[PD_TENSOR_CLEANUP] {target}: FAILED: {error}", file=sys.stderr, flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return cleanup(args)
    except KeyboardInterrupt:
        print("[PD_TENSOR_CLEANUP] interrupted; completed deletions are not reversible", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"[PD_TENSOR_CLEANUP] FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
