# SPDX-License-Identifier: Apache-2.0
"""Offline byte-feature boundary and incomplete-coverage regression tests."""

import importlib.util
import io
import json
import struct
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def tool():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tools"))
    spec = importlib.util.spec_from_file_location("feature_report_test", root / "tools/prefill_kv_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive(root, role, rows, *, ended=True, rank=0, tp_size=1):
    worker = root / role
    worker.mkdir(parents=True)
    meta = dict(
        stage="FILE_BEGIN",
        format="token-byte0-v1",
        req="cmpl-target-0",
        rank=rank,
        tp_size=tp_size,
        tp_rank=rank,
        role=role,
        groups=[1],
    )
    with (worker / "features.bin").open("wb") as binary, (worker / "index.jsonl").open("w") as index:
        index.write(json.dumps(meta) + "\n")
        for i, item in enumerate(rows):
            row = dict(item)
            data = row.pop("data")
            row.setdefault("start", 0)
            row.setdefault("end", len(data) // row.get("byte_width", 1))
            row.setdefault("group", 0)
            row.setdefault("layer", 0)
            row.setdefault("component", 0)
            row.setdefault("frontier", 4)
            row["observation"] = i + 1
            row["pieces"] = [[row["start"], row["end"], binary.tell()]]
            binary.write(data)
            index.write(json.dumps(row) + "\n")
        if ended:
            index.write(json.dumps(dict(stage="END")) + "\n")
    return worker


def chain(root, *, corrupt=None, ended=True):
    first, final = bytes([11, 22, 33, 44]), bytes([11, 22, 33, 44, 55, 66, 77, 88])
    p = [
        dict(stage="P_COMPUTE", data=first, prefix=0),
        dict(stage="P_PRODUCED", data=first),
        dict(stage="P_CPU_STORE", data=first),
        dict(stage="P_CPU_LOAD", data=first),
        dict(stage="P_LOAD", data=first, frontier=8),
        dict(stage="P_COMPUTE", data=final, frontier=8, prefix=4, final=True, metadata_frontier=7),
        dict(stage="P_PRODUCED", data=final[4:], start=4, end=8, frontier=8),
        dict(stage="P_CPU_STORE", data=final[4:7], start=4, end=7, frontier=7),
    ]
    d = [
        dict(stage="D_CPU", data=final[:6], frontier=6),
        dict(stage="MAP_SELECTED_POS", data=struct.pack("<3q", 0, 4, -1), byte_width=8, frontier=6),
        dict(stage="MAP_SELECTED_SLOT", data=struct.pack("<3q", 10, 11, -1), byte_width=8, frontier=6),
        dict(stage="D_SELECTED", data=bytes([11, 55, 99]), frontier=6, slot_capacity=12),
    ]
    if corrupt:
        target = next(r for r in p + d if r["stage"] == corrupt)
        target["data"] = bytes([target["data"][0] ^ 1]) + target["data"][1:]
    archive(root, "P", p)
    archive(root, "D", d, ended=ended)


def test_partial_chunk_and_mtp_clipping_compare_same_positions(tool, tmp_path):
    chain(tmp_path / "data")
    result = tool.analyze([tmp_path / "data"], "target", tmp_path / "report")
    assert result["status"] == "sampled_bytes_equal"
    assert result.get("different_bytes", 0) == result.get("unmatched_ranges", 0) == 0
    assert result["actual_D_sparse_observed"]


@pytest.mark.parametrize("stage", ["P_CPU_STORE", "D_CPU", "D_SELECTED"])
def test_wrong_boundary_is_reported_with_logical_token(tool, tmp_path, stage):
    chain(tmp_path / "data", corrupt=stage)
    result = tool.analyze([tmp_path / "data"], "target", tmp_path / "report")
    assert result["status"] == "differences"
    assert result["first_difference"]["stage"] == stage
    assert result["first_difference"]["position"] == 0


def test_missing_end_and_missing_D_are_never_reported_equal(tool, tmp_path):
    chain(tmp_path / "data", ended=False)
    result = tool.analyze([tmp_path / "data"], "target", tmp_path / "report")
    assert result["status"] == "incomplete"
    assert result["first_error"]["error"] == "missing_END"
    other = tmp_path / "only-p"
    archive(other, "P", [dict(stage="P_COMPUTE", data=b"abc", prefix=0, final=True)])
    result = tool.analyze([other], "target", tmp_path / "other-report")
    assert result["status"] == "incomplete"


def test_truncated_binary_is_coverage_error(tool, tmp_path):
    chain(tmp_path / "data")
    (tmp_path / "data/D/features.bin").write_bytes(b"")
    result = tool.analyze([tmp_path / "data"], "target", tmp_path / "report")
    assert result["status"] == "incomplete"
    assert result["errors"] > 0


def test_remote_archive_finds_request_directory_and_only_collects_features(tool, tmp_path, monkeypatch):
    chain(tmp_path / "repo/kv-feature-dump/target")
    logs = tmp_path / "repo/logs"
    logs.mkdir()
    records = [dict(req="cmpl-target-0", stage="BEGIN"), dict(req="cmpl-target-0", stage="END")]
    (logs / "server.log").write_text("".join("[KVTRACE] " + json.dumps(row) + "\n" for row in records))
    remote_output = io.BytesIO()
    args = ["remote", str(tmp_path / "repo"), "target", "0", "logs", ""]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(sys, "stdout", type("Stdout", (), {"buffer": remote_output})())
    original_open = tarfile.open

    def legacy_stream_open(*args, **kwargs):
        # Python 3.9 passes this kwarg through to TarFile.__init__, which rejects
        # it. Keep this regression effective on newer local Python runtimes.
        if "|" in kwargs.get("mode", "") and "compresslevel" in kwargs:
            raise TypeError("__init__() got an unexpected keyword argument 'compresslevel'")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(tarfile, "open", legacy_stream_open)
    exec(compile(tool.REMOTE, "remote", "exec"), {})
    with tarfile.open(fileobj=io.BytesIO(remote_output.getvalue()), mode="r:gz") as source:
        names = source.getnames()
        assert [json.loads(line) for line in source.extractfile("logs.jsonl").read().splitlines()] == records
    assert sorted(names) == [
        "D/features.bin",
        "D/index.jsonl",
        "P/features.bin",
        "P/index.jsonl",
        "collection-status.json",
        "logs.jsonl",
    ]


def test_passive_rank_compares_with_actual_publisher_and_CPU_source(tool, tmp_path):
    stored, passive = bytes([11, 22, 33, 44]), bytes([12, 22, 33, 44])
    final = stored + bytes([55])
    for rank in (0, 1):
        p = [
            dict(stage="P_COMPUTE", data=stored if rank == 0 else passive, prefix=0),
            dict(stage="P_CPU_LOAD", data=stored, source_rank=0),
            dict(stage="P_LOAD", data=stored, frontier=5),
            dict(stage="P_COMPUTE", data=final, frontier=5, prefix=4, final=True),
        ]
        if rank == 0:
            p.append(dict(stage="P_CPU_STORE", data=stored))
        archive(tmp_path / f"r{rank}", "P", p, rank=rank, tp_size=2)
        archive(tmp_path / f"r{rank}", "D", [dict(stage="D_CPU", data=final, source_rank=0)], rank=rank, tp_size=2)
    result = tool.analyze([tmp_path], "target", tmp_path / "report")
    assert result["status"] == "sampled_bytes_equal"
    assert result["different_bytes"] == result["unmatched_ranges"] == 0


def collection_args(tmp_path):
    return SimpleNamespace(
        output=tmp_path,
        workspace="/repo",
        request_id="target",
        wait_seconds=0,
        log_dir="logs",
        feature_dir=None,
    )


def collection_payload(host, *, features=True):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        data = json.dumps(dict(host=host)).encode()
        item = tarfile.TarInfo("logs.jsonl")
        item.size = len(data)
        archive.addfile(item, io.BytesIO(data))
        if features:
            for name, data in (("worker/index.jsonl", b'{"stage":"END"}\n'), ("worker/features.bin", b"sample")):
                item = tarfile.TarInfo(name)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
    return output.getvalue()


def test_four_concurrent_hosts_keep_distinct_complete_IP_names(tool, tmp_path, monkeypatch):
    args = collection_args(tmp_path)
    hosts = ["root@7.150.2.72", "root@7.150.3.11", "root@7.150.3.60", "root@7.150.6.177"]
    monkeypatch.setattr(tool, "remote_command", lambda *args: [])

    def run_ssh(args, host, command, *, stdout, stderr):
        stdout.write(collection_payload(host))
        return subprocess.CompletedProcess(command, 0, stderr=b"")

    monkeypatch.setattr(tool, "run_ssh", run_ssh)
    # A broken archive left by the previous version must not block the retry.
    old = tmp_path / "hosts/host-root_7.150.3.tar.gz"
    old.parent.mkdir()
    old.write_bytes(b"old failed download")
    with ThreadPoolExecutor(max_workers=4) as pool:
        directories = list(pool.map(lambda host: tool.collect_host(args, host), hosts))
    assert len(set(directories)) == 4
    for host, directory in zip(hosts, directories):
        assert json.loads((directory / "logs.jsonl").read_text())["host"] == host
        archive = directory.with_name(directory.name + ".tar.gz")
        assert archive.is_file()
        assert archive.name.startswith(tool.host_directory(host) + ".")
    assert old.read_bytes() == b"old failed download"


@pytest.mark.parametrize("failure", ["ssh", "truncated_archive"])
def test_failed_collection_can_retry_same_output_and_reuse_success(tool, tmp_path, monkeypatch, failure):
    args, host, calls = collection_args(tmp_path), "root@7.150.3.11", []
    monkeypatch.setattr(tool, "remote_command", lambda *args: [])

    def run_ssh(args, host, command, *, stdout, stderr):
        calls.append(host)
        payload = collection_payload(host)
        stdout.write(payload[:-8] if len(calls) == 1 else payload)
        code = 1 if len(calls) == 1 and failure == "ssh" else 0
        return subprocess.CompletedProcess(command, code, stderr=b"remote packing failed")

    monkeypatch.setattr(tool, "run_ssh", run_ssh)
    with pytest.raises((RuntimeError, EOFError)):
        tool.collect_host(args, host)
    assert list((tmp_path / "hosts").iterdir()) == []
    directory = tool.collect_host(args, host)
    assert directory.is_dir()
    assert tool.collect_host(args, host) == directory
    assert len(calls) == 2


def test_logs_only_cache_is_refreshed_on_retry(tool, tmp_path, monkeypatch):
    args, host, calls = collection_args(tmp_path), "root@7.150.3.11", []
    monkeypatch.setattr(tool, "remote_command", lambda *args: [])

    def run_ssh(args, host, command, *, stdout, stderr):
        calls.append(host)
        stdout.write(collection_payload(host, features=len(calls) > 1))
        return subprocess.CompletedProcess(command, 0, stderr=b"")

    monkeypatch.setattr(tool, "run_ssh", run_ssh)
    directory = tool.collect_host(args, host)
    assert not list(directory.rglob("index.jsonl"))
    assert tool.collect_host(args, host) == directory
    assert len(list(directory.rglob("index.jsonl"))) == 1
    assert len(calls) == 2


def test_prior_worker_failure_is_collected_and_reported_for_later_request(tool, tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    base = dict(host="host", pid=123, rank=0, role="P", tp_size=4)
    original = dict(base, req="first-request", stage="P_LOAD", error="original_failure")
    later = dict(base, req="target", stage="END", failed=True)
    unrelated = dict(base, pid=456, req="unrelated", stage="P_LOAD", error="unrelated_failure")
    (logs / "P.log").write_text("".join("[KVTRACE] " + json.dumps(row) + "\n" for row in (original, unrelated, later)))
    remote_output = io.BytesIO()
    monkeypatch.setattr(sys, "argv", ["remote", str(tmp_path), "target", "120", "logs", ""])
    monkeypatch.setattr(sys, "stdout", type("Stdout", (), {"buffer": remote_output})())
    exec(compile(tool.REMOTE, "remote", "exec"), {})
    compressed = tmp_path / "archive.tar.gz"
    compressed.write_bytes(remote_output.getvalue())
    extracted = tmp_path / "collected"
    extracted.mkdir()
    tool.safe_extract(compressed, extracted)
    result = tool.analyze([extracted], "target", tmp_path / "report")
    assert result["first_error"]["error"] == "original_failure"
    assert result["first_error"]["req"] == "first-request"
    assert result["first_error"]["context_for_request"] == "target"
    assert result["log_stages"] == {"END": 1}
    assert "unrelated_failure" not in (extracted / "logs.jsonl").read_text()


def test_inaccessible_advertised_feature_path_is_not_silently_ignored(tool, tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    missing = tmp_path / "container-only/target/worker"
    record = dict(req="target", stage="FEATURE_FILE", directory=str(missing))
    (logs / "P.log").write_text("[KVTRACE] " + json.dumps(record) + "\n")
    remote_output = io.BytesIO()
    monkeypatch.setattr(sys, "argv", ["remote", str(tmp_path), "target", "120", "logs", ""])
    monkeypatch.setattr(sys, "stdout", type("Stdout", (), {"buffer": remote_output})())
    exec(compile(tool.REMOTE, "remote", "exec"), {})
    with tarfile.open(fileobj=io.BytesIO(remote_output.getvalue()), mode="r:gz") as source:
        status = json.load(source.extractfile("collection-status.json"))
    assert status["feature_files"] == 0
    assert str(missing / "index.jsonl") in status["missing_paths"]


def test_failed_end_only_logs_report_stopped_recorders_and_real_tp_size(tool, tmp_path):
    root = tmp_path / "collected"
    root.mkdir()
    rows = [
        dict(
            req="cmpl-target-0",
            host=role,
            pid=100 + rank,
            role=role,
            rank=rank,
            tp_rank=rank,
            tp_size=4,
            stage="END",
            failed=True,
        )
        for role in ("P", "D")
        for rank in range(4)
    ]
    # The producer may emit END twice; do not count this as two failed workers.
    (root / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows + rows[:4]))
    result = tool.analyze([root], "target", tmp_path / "report")
    assert result["status"] == "incomplete"
    assert result["observations"] == 0
    assert result["log_stages"] == {"END": 12}
    assert result["first_error"]["error"] == "feature_recorder_stopped"
    assert result["errors"] == 9  # Eight failed recorders plus missing TP data.
    errors = json.loads((tmp_path / "report/coverage-errors.json").read_text())
    assert errors[-1]["expected"] == 4
