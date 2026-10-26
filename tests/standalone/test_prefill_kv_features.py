# SPDX-License-Identifier: Apache-2.0
"""Offline byte-feature boundary and incomplete-coverage regression tests."""

import importlib.util
import io
import json
import struct
import sys
import tarfile
from pathlib import Path

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
    remote_output = io.BytesIO()
    args = ["remote", str(tmp_path / "repo"), "target", "0", "logs", ""]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(sys, "stdout", type("Stdout", (), {"buffer": remote_output})())
    exec(compile(tool.REMOTE, "remote", "exec"), {})
    with tarfile.open(fileobj=io.BytesIO(remote_output.getvalue()), mode="r:gz") as source:
        names = source.getnames()
    assert sorted(names) == ["D/features.bin", "D/index.jsonl", "P/features.bin", "P/index.jsonl", "logs.jsonl"]


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
