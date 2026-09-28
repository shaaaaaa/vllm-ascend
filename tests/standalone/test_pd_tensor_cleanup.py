# SPDX-License-Identifier: Apache-2.0
"""Exercise remote cleanup code against temporary directories without SSH/NPU."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import pd_tensor_cleanup as cleanup


def remote(tmp_path, run="case-on", dry=False):
    return subprocess.run(
        [sys.executable, "-c", "import json\n" + cleanup.REMOTE_CLEANUP, str(tmp_path), run, str(int(dry))],
        capture_output=True,
    )


def populate(tmp_path):
    for name in ("case-on", "case-off"):
        directory = tmp_path / "pd-tensor-dump" / name / "P/request/worker"
        directory.mkdir(parents=True)
        (directory / "tensor.pt").write_bytes(b"test data")
    (tmp_path / "model.pt").write_bytes(b"preserve")
    return tmp_path / "pd-tensor-dump"


def test_remove_selected_run_preserves_siblings_and_supports_repeat(tmp_path):
    root = populate(tmp_path)
    result = remote(tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "removed"
    assert not (root / "case-on").exists()
    assert (root / "case-off/P/request/worker/tensor.pt").read_bytes() == b"test data"
    assert (tmp_path / "model.pt").read_bytes() == b"preserve"
    result = remote(tmp_path)
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "absent"


def test_all_runs_preserves_parent_and_dry_run_preserves_every_file(tmp_path):
    root = populate(tmp_path)
    result = remote(tmp_path, run="", dry=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["entries"] == 2
    assert json.loads(result.stdout)["status"] == "preview"
    assert len(list(root.rglob("tensor.pt"))) == 2
    result = remote(tmp_path, run="")
    assert result.returncode == 0, result.stderr
    assert root.is_dir() and not list(root.iterdir())
    assert (tmp_path / "model.pt").exists()


def test_missing_root_is_idempotent_but_missing_repo_is_error(tmp_path):
    assert json.loads(remote(tmp_path).stdout)["status"] == "absent"
    assert remote(tmp_path / "missing-repo").returncode != 0


@pytest.mark.parametrize("run", ["..", "../outside", "/outside"])
def test_remote_validation_prevents_run_path_escape(tmp_path, run):
    root = populate(tmp_path)
    assert remote(tmp_path, run=run).returncode != 0
    assert len(list(root.rglob("tensor.pt"))) == 2


@pytest.mark.parametrize("scope", ["root", "run", "all_runs"])
def test_symlink_target_is_rejected_without_touching_external_data(tmp_path, scope):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    link = repo / "pd-tensor-dump"
    if scope != "root":
        link.mkdir()
        link = link / "case-on"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")
    result = remote(repo, run="" if scope == "all_runs" else "case-on")
    assert result.returncode != 0
    assert (outside / "keep").read_text() == "keep"
    assert link.is_symlink()


def test_four_hosts_attempted_despite_one_failure_with_same_password(tmp_path, monkeypatch, capsys):
    import pd_tensor_collect as collector

    monkeypatch.setattr(collector.shutil, "which", lambda _: pytest.fail("unexpected dependency lookup"))
    monkeypatch.setattr(collector.getpass, "getpass", lambda _: "fake-secret")
    calls = []

    def run(command, *, stdout, stderr, check, env, stdin, **kwargs):
        calls.append(command)
        assert command[0] == "ssh"
        assert "fake-secret" not in " ".join(command)
        assert env[collector.ASKPASS_PASSWORD_ENV] == "fake-secret"
        assert env["SSH_ASKPASS_REQUIRE"] == "force"
        assert Path(env["SSH_ASKPASS"]).is_file()
        assert stdin == subprocess.DEVNULL
        remote_argv = shlex.split(command[-1])
        assert remote_argv[:5] == ["docker", "exec", "--", "inference", "python3"]
        assert remote_argv[-3:] == ["/workspace/repo", "case-on", "0"]
        return subprocess.CompletedProcess(
            command,
            255 if command[-2] == "bad" else 0,
            stdout=b'{"status":"removed","entries":1}',
            stderr=b"connection failed",
        )

    monkeypatch.setattr(collector.subprocess, "run", run)
    assert (
        cleanup.main(
            [
                "--hosts",
                "p1",
                "bad",
                "d1",
                "d2",
                "--repo-path",
                "/workspace/repo",
                "--run-id",
                "case-on",
                "--container",
                "inference",
                "--password",
            ]
        )
        == 1
    )
    assert [c[-2] for c in calls] == ["p1", "bad", "d1", "d2"]
    captured = capsys.readouterr()
    assert "fake-secret" not in captured.out + captured.err
    assert captured.out.count("removed") == 3


def test_scope_is_required_and_duplicate_hosts_rejected():
    with pytest.raises(SystemExit):
        cleanup.parser().parse_args(["--hosts", "h", "--repo-path", "/workspace/repo"])
    assert cleanup.main(["--hosts", "h", "h", "--repo-path", "/workspace/repo", "--all-runs"]) == 1
