# SPDX-License-Identifier: Apache-2.0
"""Exercise remote filtering, multi-host routing and native password transport."""

import argparse
import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def collect(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    return importlib.import_module("prefill_overlap_collect")


def arguments(collect, tmp_path, *extra):
    parser = argparse.ArgumentParser()
    collect.add_collection_arguments(parser)
    parser.add_argument("--output", type=Path, default=tmp_path / "rounds.png")
    return parser.parse_args(["--hosts", "root@192.0.2.1", "--repo-path", "/workspace/vllm-ascend", *extra])


def archive_bytes(collect, directory):
    result = subprocess.run(
        [sys.executable, "-c", collect.REMOTE_COLLECT, str(directory), json.dumps(["P_dp*.log"])],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return result.stdout


def measurement(turn):
    return dict(
        schema_version=2,
        coverage_definition="same_bank_reuse_v1",
        status="complete",
        host=f"node-{turn}",
        pid=10 + turn,
        dp_rank=turn,
        tp_rank=0,
        step=1,
        requests=[dict(req_id=f"cmpl-multi-run-{turn:03d}-0-suffix", token_start=0, token_end=4096)],
        coverage=0.8,
        layer_samples=2,
        **dict.fromkeys(("dma_ms", "save_ms", "load_ms", "wait_ms", "interval_ms"), 10.0),
    )


def test_remote_snapshot_filters_and_compresses_all_dp_logs(collect, tmp_path):
    source = tmp_path / "logs with 'quotes'; literal"
    source.mkdir()
    payload = b"[Worker] [PREFILL_OVERLAP] " + json.dumps(measurement(0)).encode() + b"\n"
    original = b"ordinary log\n" * 2000 + payload + payload.rstrip(b"\n")
    (source / "P_dp0_case.log").write_bytes(original)
    (source / "P_dp1_case.log").write_bytes(payload)
    (source / "D_dp0_case.log").write_bytes(payload)
    packed = archive_bytes(collect, source)
    assert packed.startswith(b"\x1f\x8b")
    destination = tmp_path / "unpacked"
    destination.mkdir()
    collect.safe_extract(_write(tmp_path / "snapshot.tar.gz", packed), destination)
    assert sorted(p.name for p in (destination / "logs").iterdir()) == ["P_dp0_case.log", "P_dp1_case.log"]
    assert (destination / "logs/P_dp0_case.log").read_bytes() == payload
    metadata = json.loads((destination / "_collection.json").read_text())
    assert [item["lines"] for item in metadata["files"]] == [1, 1]
    assert (source / "P_dp0_case.log").read_bytes() == original


def _write(path, content):
    path.write_bytes(content)
    return path


@pytest.mark.parametrize("mode", ["prompt", "env", "per-host", "argument"])
def test_native_password_multi_host_collection_then_round_plot(collect, tmp_path, monkeypatch, mode, capsys):
    plot = importlib.import_module("plot_multi_turn_overlap")
    transport = importlib.import_module("pd_tensor_collect")
    hosts = ["root@192.0.2.1", "root@192.0.2.2"]
    archives = {}
    for index, host in enumerate(hosts):
        source = tmp_path / str(index)
        source.mkdir()
        turns = [0] if index == 0 else [1, 2]
        (source / "P_dp0_case.log").write_text(
            "".join("[PREFILL_OVERLAP] " + json.dumps(measurement(turn)) + "\n" for turn in turns), encoding="utf-8"
        )
        (source / "P_dp1_case.log").write_text("idle\n", encoding="utf-8")
        archives[host] = archive_bytes(collect, source)
    secret = "literal-'$-test-password"
    monkeypatch.setenv("TEST_OVERLAP_PASSWORD", secret)

    def password_prompt(_):
        assert mode != "argument", "explicit SSH password must never prompt"
        return secret

    monkeypatch.setattr(transport.getpass, "getpass", password_prompt)
    helpers, commands = [], []
    source_path = "/custom service/P logs" if mode == "argument" else "/workspace/path with spaces/tp4_dp4_logs"

    def ssh(command, *, stdout, stderr, check, env, stdin, timeout, **kwargs):
        assert not check and timeout == 600
        assert command[0] == "ssh" and "BatchMode=no" in command
        assert "StrictHostKeyChecking=no" in command and "2222" in command
        assert secret not in " ".join(command)
        assert env[transport.ASKPASS_PASSWORD_ENV] == secret
        assert env["SSH_ASKPASS_REQUIRE"] == "force"
        assert stdin == subprocess.DEVNULL
        helper = Path(env["SSH_ASKPASS"])
        helpers.append(helper)
        assert helper.is_file() and secret not in helper.read_text()
        remote = shlex.split(command[-1])
        assert remote[:6] == ["docker", "exec", "--", "inference", "python3", "-c"]
        assert remote[-2:] == [source_path, '["P_dp*.log"]']
        commands.append(command)
        stdout.write(archives[command[-2]])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(transport.subprocess, "run", ssh)
    rendered = []
    monkeypatch.setattr(plot, "plot_rounds", lambda rows, rounds, path, **kwargs: rendered.append((rows, rounds)))
    run_dir = tmp_path / "client"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(dict(rounds=[dict(round=i, request_id=f"multi-run-{i:03d}") for i in range(3)])), encoding="utf-8"
    )
    flags = {
        "prompt": ["--password"],
        "env": ["--password-env", "TEST_OVERLAP_PASSWORD"],
        "per-host": ["--password-per-host"],
        "argument": ["--ssh-password", secret, "--remote-log-dir", source_path],
    }
    destination = tmp_path / "collected"
    output = tmp_path / "result.png"
    assert (
        plot.main(
            [
                "--hosts",
                *hosts,
                "--repo-path",
                "/workspace/path with spaces",
                "--run-dir",
                str(run_dir),
                "--collect-dir",
                str(destination),
                "--container",
                "inference",
                "--ssh-port",
                "2222",
                "--skip-host-key-check",
                "--aggregate-ranks",
                "--output",
                str(output),
                *flags[mode],
            ]
        )
        == 0
    )
    assert len(commands) == 2 and len(set(h.parent for h in helpers)) == 1
    assert not any(helper.exists() or helper.parent.exists() for helper in helpers)
    rows, rounds = rendered[0]
    assert rounds == [0, 1, 2] and [row["round"] for row in rows] == [0, 1, 2]
    assert [row["worker_count"] for row in rows] == [1, 1, 1]
    assert len(list(destination.glob("host-*/snapshot/logs/*.log"))) == 4
    manifest = (destination / "collection.json").read_text()
    assert secret not in manifest
    assert all(host["status"] == "success" for host in json.loads(manifest)["hosts"])
    assert json.loads(manifest)["source_directory"] == source_path
    diagnostic = json.loads(output.with_suffix(".json").read_text())
    assert diagnostic["missing_rounds"] == [] and diagnostic["collection_directory"] == str(destination.resolve())
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert f"remote_log_dir={source_path}" in captured.out


@pytest.mark.parametrize("failure", ["ssh", "timeout", "invalid-archive"])
def test_failed_collection_keeps_evidence_and_does_not_plot_partial_hosts(collect, tmp_path, monkeypatch, failure):
    args = arguments(collect, tmp_path, "--collect-dir", str(tmp_path / "snapshot"))

    def ssh(args, target, command, *, stdout, stderr, timeout):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        stdout.write(b"not an archive")
        stderr.write(b"authentication failed")
        return subprocess.CompletedProcess(command, 255 if failure == "ssh" else 0)

    monkeypatch.setattr(collect, "run_ssh", ssh)
    with pytest.raises(ValueError, match="Some P-node logs could not be collected"):
        collect.collect_logs(args)
    manifest = json.loads((args.collect_dir / "collection.json").read_text())
    assert manifest["hosts"][0]["status"] == "failed"
    assert manifest["hosts"][0]["error"]
    assert (args.collect_dir / manifest["hosts"][0]["directory"] / "overlap.tar.gz").exists()


def test_bad_remote_options_and_missing_password_fail_before_ssh(collect, tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "run_ssh", lambda *a, **k: pytest.fail("must not SSH"))
    for option in (("--repo-path", "relative"), ("--log-pattern", "../*.log"), ("--ssh-timeout", "0")):
        with pytest.raises(ValueError):
            collect.collect_logs(arguments(collect, tmp_path, *option))
    monkeypatch.delenv("MISSING_OVERLAP_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="environment variable is not set"):
        collect.collect_logs(arguments(collect, tmp_path, "--password-env", "MISSING_OVERLAP_PASSWORD"))
    args = arguments(collect, tmp_path, "--remote-log-dir", "/logs/custom")
    assert collect.remote_log_directory(args) == "/logs/custom"
    args.repo_path = args.remote_log_dir = None
    with pytest.raises(ValueError, match="requires --repo-path"):
        collect.remote_log_directory(args)


def test_idle_node_is_a_successful_empty_snapshot(collect, tmp_path, monkeypatch):
    source = tmp_path / "remote"
    source.mkdir()
    (source / "P_dp0_case.log").write_text("no requests handled\n")
    payload = archive_bytes(collect, source)

    def ssh(args, target, command, *, stdout, stderr, timeout):
        stdout.write(payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(collect, "run_ssh", ssh)
    paths, root = collect.collect_logs(arguments(collect, tmp_path))
    assert len(paths) == 1 and paths[0].stat().st_size == 0
    assert json.loads((root / "collection.json").read_text())["hosts"][0]["status"] == "success"


def test_missing_log_directory_and_pattern_report_remote_error(collect, tmp_path):
    for source in (tmp_path / "missing", tmp_path):
        with pytest.raises(subprocess.CalledProcessError) as failure:
            archive_bytes(collect, source)
        assert b"log directory does not exist" in failure.value.stderr or b"no P logs matched" in failure.value.stderr
