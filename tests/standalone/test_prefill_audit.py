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
