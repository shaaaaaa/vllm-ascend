# SPDX-License-Identifier: Apache-2.0
"""Exercise discovery without moving KV or launching a model."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import kv_replay_discovery as D
import prefill_kv_replay as R
from test_prefill_kv_replay import log_file, make_run, snapshot


def cli(tmp_path):
    return R.parser().parse_args(
        [
            "--snapshot-hosts",
            "root@3.60",
            "root@6.177",
            "--workspace",
            "/workspace/sqh/vllm-ascend",
            "--container",
            "test-container",
            "--remote-python",
            "/usr/bin/python3",
        ]
    )


def test_probe_never_reads_payload_and_supports_legacy_capture(tmp_path, monkeypatch):
    plan = snapshot(tmp_path)
    for path in tmp_path.rglob("replay.json"):
        path.unlink()
    original = Path.open

    def no_payload(self, *args, **kwargs):
        assert self.suffix != ".bin", "Discovery must not read any KV payload"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_payload)
    result = D.probe_snapshot(plan)
    assert result["status"] == "ready"
    assert result["source_tokens"] == 8
    assert result["tp_size"] == 2
    assert len(result["matching_manifests"]) == 2
    assert "tokens" not in result


@pytest.mark.parametrize(
    "condition,expected",
    [
        ("missing_directory", "missing_directory"),
        ("other_request", "request_not_found"),
        ("incomplete", "invalid_snapshot"),
        ("missing_rank", "invalid_snapshot"),
        ("truncated", "invalid_snapshot"),
        ("different_tokens", "invalid_snapshot"),
    ],
)
def test_diagnoses_missing_and_invalid_snapshots(tmp_path, condition, expected):
    plan = snapshot(tmp_path)
    if condition == "missing_directory":
        plan["snapshot_root"] = str(tmp_path / "absent")
    elif condition == "other_request":
        plan["previous"] = "a-different-round-zero-response"
    elif condition == "missing_rank":
        (tmp_path / "response-0/rank1/manifest.json").unlink()
    elif condition == "truncated":
        next(tmp_path.rglob("*.bin")).write_bytes(b"short")
    elif condition == "different_tokens":
        plan["prefix_digest"] = "wrong"
    else:
        path = next(tmp_path.rglob("manifest.json"))
        record = json.loads(path.read_text())
        record["status"] = "writing"
        path.write_text(json.dumps(record))
    assert D.probe_snapshot(plan)["status"] == expected


def test_probe_remote_entrypoint_is_json_and_only_metadata(tmp_path):
    plan = snapshot(tmp_path)
    encoded = base64.b64encode(json.dumps(plan).encode()).decode()
    result = subprocess.run(
        [sys.executable, D.__file__, "--probe", encoded],
        capture_output=True,
        check=True,
    )
    response = json.loads(result.stdout)
    assert response["status"] == "ready"
    assert len(result.stdout) < 2000


@pytest.mark.parametrize("first_status", ["request_not_found", "missing_directory", "invalid_snapshot", "probe_failed"])
def test_selects_second_host_and_updates_file_url_and_ssh_destination(tmp_path, monkeypatch, first_status):
    plan = snapshot(tmp_path / "data")
    args = cli(tmp_path)
    args.snapshot_host = args.snapshot_hosts[0]
    seen = []

    def probe(args, host, actual_plan, output, index):
        seen.append(host)
        assert actual_plan["previous"] == "response-0"
        return dict(
            host=host,
            status=first_status if host == "root@3.60" else "ready",
            snapshot_root="/workspace/sqh/vllm-ascend/decode-kv-snapshots",
        )

    monkeypatch.setattr(D, "probe_host", probe)
    D.select_snapshot(args, plan, tmp_path)
    assert set(seen) == {"root@3.60", "root@6.177"}
    assert args.snapshot_host == "root@6.177"
    assert plan["file_host"] == "6.177"
    assert plan["snapshot_host"] == "root@6.177"
    assert "root@6.177" in R.child_argv(args, "--file-child", "test")
    saved = json.loads((tmp_path / "snapshot_discovery.json").read_text())
    assert saved["selected_host"] == "root@6.177"


@pytest.mark.parametrize("statuses", [("request_not_found", "request_not_found"), ("ready", "ready")])
def test_absent_or_ambiguous_sources_fail_before_serving(tmp_path, monkeypatch, statuses):
    args = cli(tmp_path)
    plan = snapshot(tmp_path / "data")
    monkeypatch.setattr(D, "probe_host", lambda args, host, plan, output, i: dict(host=host, status=statuses[i]))
    with pytest.raises(ValueError, match="no file server or inference request was started"):
        D.select_snapshot(args, plan, tmp_path)
    assert len(json.loads((tmp_path / "snapshot_discovery.json").read_text())["candidates"]) == 2


def test_missing_source_does_not_start_file_service_or_inference(tmp_path, monkeypatch):
    make_run(tmp_path / "run")
    log_file(tmp_path / "p.log")
    monkeypatch.setattr(R, "REPO", tmp_path)
    monkeypatch.setattr(R, "check_proxy_connection", lambda *args: None)
    monkeypatch.setattr(R, "file_source", lambda *args: pytest.fail("Must not start file source"))
    monkeypatch.setattr(R, "Client", lambda *args: pytest.fail("Must not submit inference"))

    def http(url, path, body=None):
        if path == "/v1/models":
            return {"data": [{"id": "alias", "root": "/model"}]}
        return {"tokens": list(range(14 if "问题" in body["prompt"] else 11))}

    monkeypatch.setattr(R, "http_json", http)
    with pytest.raises(ValueError, match="missing_directory"):
        R.main(
            [
                "--run-dir",
                str(tmp_path / "run"),
                "--p-logs",
                str(tmp_path / "p.log"),
                "--base-url",
                "http://proxy:8080",
                "--p-url",
                "http://P:7910",
                "--local-hostname",
                "127.0.0.1",
                "--snapshot-root",
                str(tmp_path / "absent"),
            ]
        )
    (report_path,) = (tmp_path / "prefill-kv-replays").glob("*/report.json")
    assert json.loads(report_path.read_text())["status"] == "failed"


def test_ssh_failure_is_diagnostic_not_snapshot_absence(tmp_path, monkeypatch):
    args = cli(tmp_path)
    plan = snapshot(tmp_path / "data")

    def run(args, host, command, **kwargs):
        assert command[-1].startswith("docker exec -- test-container /usr/bin/python3 ")
        kwargs["stderr"].write(b"Permission denied\n")
        return subprocess.CompletedProcess(command, 255, stdout=b"")

    monkeypatch.setattr(D, "run_ssh", run)
    result = D.probe_host(args, "root@3.60", plan, tmp_path, 0)
    assert result["status"] == "probe_failed"
    assert Path(result["log"]).read_text() == "Permission denied\n"
