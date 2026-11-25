# SPDX-License-Identifier: Apache-2.0
"""Offline audit interpretation and native SSH collection contracts."""

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("audit_tool", ROOT / "tools/prefill_audit.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def write_rows(tmp_path, get_sample=(64, "aabb"), cache_sample=None, source=True):
    rows = []
    if source:
        rows.append(dict(kind="payload", stage="put", keys=["key"], samples=[[64, "aabb"]], host="P", pid=1))
    rows.append(dict(kind="payload", stage="get", keys=["key"], samples=[get_sample], host="D", pid=2))
    rows.append(
        dict(
            kind="payload",
            stage="cache_read",
            req="request-1",
            keys=["key"],
            samples=[cache_sample or get_sample],
            host="D",
            pid=2,
        )
    )
    (tmp_path / "audit-1.jsonl").write_text("\n".join(json.dumps(row) for row in rows))


def test_matching_request_is_correlated_by_keys(tmp_path):
    write_rows(tmp_path)
    report = tool.analyze([tmp_path], "request-1")
    assert report["status"] == "samples_consistent"
    assert report["compared"] == 2


def test_transport_boundary_is_reported(tmp_path):
    write_rows(tmp_path, get_sample=(64, "ccdd"))
    report = tool.analyze([tmp_path], "request-1")
    assert report["first_issue"]["stage"] == "get"


def test_cpu_read_after_receive_is_reported(tmp_path):
    write_rows(tmp_path, cache_sample=(64, "ccdd"))
    report = tool.analyze([tmp_path], "request-1")
    assert report["first_issue"]["stage"] == "cache_read"


def test_missing_and_partial_evidence_is_not_a_pass(tmp_path):
    write_rows(tmp_path, source=False)
    assert tool.analyze([tmp_path])["status"] == "incomplete"
    with (tmp_path / "audit-1.jsonl").open("a") as output:
        output.write('\n{"unfinished":')
    assert tool.analyze([tmp_path])["gaps"]["incomplete_json_line"] == 1


def test_direct_npu_gap_requires_an_actual_matching_readback(tmp_path):
    rows = [
        dict(kind="payload", stage="put", keys=["key"], samples=[[64, "aaaa"]], host="P", pid=1),
        dict(kind="direct_npu", req="request", keys=["key"], host="D", pid=2),
    ]
    path = tmp_path / "audit-1.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert tool.analyze([tmp_path], "request")["status"] == "incomplete"
    rows.append(
        dict(kind="payload", stage="get_npu", req="request", keys=["key"], samples=[[64, "aaaa"]], host="D", pid=2)
    )
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert tool.analyze([tmp_path], "request")["status"] == "samples_consistent"
    rows[-1]["samples"] = [[64, "bbbb"]]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert tool.analyze([tmp_path], "request")["first_issue"]["stage"] == "get_npu"


def test_remote_archive_and_refresh_are_limited_to_audit_files(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "audit-1.jsonl").write_text('{"v":1}\n')
    (source / "unrelated.txt").write_text("unrelated")
    packed = subprocess.check_output([sys.executable, "-c", tool.REMOTE, str(source)])
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as archive:
        assert archive.getnames() == ["audit-1.jsonl"]
    args = SimpleNamespace(
        output=tmp_path / "output",
        audit_dir="/logs/audit",
        workspace="/work",
        container=None,
        ssh_port=None,
        identity_file=None,
        skip_host_key_check=True,
    )

    def run(_args, _host, _command, **kwargs):
        kwargs["stdout"].write(packed)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool, "run_ssh", run)
    first = tool.collect_host(args, "root@10.0.0.1")
    second = tool.collect_host(args, "root@10.0.0.1")
    assert first == second and (second / "audit-1.jsonl").read_text() == '{"v":1}\n'


def test_batch_checks_normal_and_suspicious_outputs_with_one_index(tmp_path, monkeypatch):
    write_rows(tmp_path)
    clients = [
        dict(request_id="client-1", response_id="request-1", session_id=0, round=1, priority="unflagged"),
        dict(request_id="client-2", response_id="request-2", session_id=1, round=1, priority="repeated_output"),
    ]
    real_index, indexes = tool.AuditEvidence, []

    def once(roots):
        indexes.append(roots)
        return real_index(roots)

    monkeypatch.setattr(tool, "AuditEvidence", once)
    output = tmp_path / "output"
    output.mkdir()
    report = tool.analyze_many([tmp_path], clients, [], output)
    assert len(indexes) == 1
    assert report["request_count"] == 2
    assert report["requests"][0]["audit_status"] == "samples_consistent"
    assert report["requests"][1]["audit_status"] == "incomplete"
    assert len(list((output / "requests").glob("*.json"))) == 2
    assert (output / "requests.csv").exists()


def test_batch_never_silently_drops_missing_response_id(tmp_path):
    client = dict(request_id="only-client-id", session_id=0, round=1)
    report = tool.analyze_many([], [client], [], tmp_path)
    assert report["request_count"] == 1
    assert report["requests"][0]["gaps"]["missing_response_id_in_client_log"] == 1


def test_shared_direct_readback_index_is_not_consumed_by_previous_report(tmp_path):
    rows = [
        dict(kind="payload", stage="put", keys=["key"], samples=[[64, "aaaa"]], host="P", pid=1),
        dict(kind="direct_npu", req="request", keys=["key"], host="D", pid=2),
        dict(kind="payload", stage="get_npu", req="request", keys=["key"], samples=[[64, "aaaa"]], host="D", pid=2),
    ]
    (tmp_path / "audit-1.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    evidence = tool.AuditEvidence([tmp_path])
    first = tool.analyze([], "request", evidence=evidence)
    assert first == tool.analyze([], "request", evidence=evidence)
    assert first["status"] == "samples_consistent"


def test_batch_request_matching_includes_multi_request_bank_records(tmp_path):
    row = dict(kind="protocol", reqs=["client-a", "client-b"], error="wrong_load_wait")
    (tmp_path / "audit-1.jsonl").write_text(json.dumps(row))
    report = tool.analyze([tmp_path], "client-b")
    assert report["first_issue"]["error"] == "wrong_load_wait"


def test_batch_cli_keeps_every_request(tmp_path, monkeypatch):
    write_rows(tmp_path)
    log = tmp_path / "client.log"
    log.write_text(
        "\n".join(
            "[MULTI_TURN] "
            + json.dumps(dict(request_id=f"client-{i}", response_id=f"request-{i}", session_id=i, round=1))
            for i in (1, 2)
        )
    )
    output = tmp_path / "reports"
    monkeypatch.setattr(
        sys, "argv", ["prefill_audit.py", "--client-log", str(log), "--local", str(tmp_path), "--output", str(output)]
    )
    assert tool.main() == 1  # Request 2 has no captured data, rather than silently passing.
    report = json.loads((output / "report.json").read_text())
    assert report["request_count"] == 2
    assert {row["request_id"] for row in report["requests"]} == {"request-1", "request-2"}
