# SPDX-License-Identifier: Apache-2.0
"""Exact multi-round comparison and native-SSH archive payload regressions."""

import ast
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tool():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tools"))
    spec = importlib.util.spec_from_file_location("decode_compare_test", root / "tools/decode_kv_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive(root, request, tokens, data, *, rank=0, size=1, dtype="torch.uint8", width=2):
    directory = root / request / f"host-tp{rank}"
    directory.mkdir(parents=True)
    (directory / "tokens.json").write_text(json.dumps(tokens))
    (directory / "data.bin").write_bytes(data)
    record = dict(
        group=0,
        layer=0,
        component=0,
        row_bytes=width,
        dtype=dtype,
        path="data.bin",
        start=0,
        end=len(tokens),
        bytes=len(data),
    )
    meta = dict(
        format="decode-kv-raw-v1",
        request_id=request,
        tp_rank=rank,
        tp_size=size,
        groups=[1],
        tokens=len(tokens),
        status="complete",
        byteorder="little",
        files=[record],
    )
    (directory / "manifest.json").write_text(json.dumps(meta))
    return directory


def test_identical_history_and_uncompared_new_suffix(tool, tmp_path):
    archive(tmp_path, "first", [10, 20], b"abcdef"[:4])
    archive(tmp_path, "second", [10, 20, 30], b"abcdef")
    report = tool.analyze([tmp_path], "first", "second", tmp_path / "report")
    assert report["status"] == "received_history_identical"
    assert report["compared_bytes"] == 4
    assert report["current_uncompared_range"] == [2, 3]
    assert report["previous_is_prefix"]


def test_one_bit_error_identifies_layer_token_and_byte(tool, tmp_path):
    archive(tmp_path, "first", [10, 20], b"abcd")
    archive(tmp_path, "second", [10, 20, 30], b"abCdef")
    report = tool.analyze([tmp_path], "first", "second", tmp_path / "report")
    assert report["status"] == "history_differs"
    assert report["different_bytes"] == report["different_tokens"] == 1
    assert report["first_differences"][0]["first"] == [dict(token=1, byte_in_component=0, previous=99, current=67)]


def test_actual_token_prefix_limits_comparison(tool, tmp_path):
    archive(tmp_path, "first", [10, 20], b"abcd")
    archive(tmp_path, "second", [10, 21, 30], b"abXXYY")
    report = tool.analyze([tmp_path], "first", "second", tmp_path / "report")
    assert report["status"] == "received_history_identical"
    assert report["compared_bytes"] == 2
    assert not report["previous_is_prefix"]


def test_missing_rank_and_truncated_file_are_not_matches(tool, tmp_path):
    archive(tmp_path, "first", [10, 20], b"abcd", size=2)
    with pytest.raises(ValueError, match="Incomplete TP"):
        tool.read_workers([tmp_path], "first")
    path = archive(tmp_path, "second", [10, 20], b"abcd")
    (path / "data.bin").write_bytes(b"abc")
    with pytest.raises(ValueError, match="Truncated"):
        tool.read_workers([tmp_path], "second")


def test_new_bfloat16_suffix_distribution(tool, tmp_path):
    archive(tmp_path, "first", [10], struct.pack("<H", 0x3F80), dtype="torch.bfloat16")
    archive(tmp_path, "second", [10, 20, 30], struct.pack("<HHH", 0x3F80, 0x7F80, 0), dtype="torch.bfloat16")
    report = tool.analyze([tmp_path], "first", "second", tmp_path / "report")
    assert report["status"] == "received_history_identical"
    assert report["suffix_nonfinite"] == 1


def test_components_compare_in_parallel_on_all_ranks(tool, tmp_path):
    for rank in range(4):
        archive(tmp_path, "first", [10, 20], b"abcd", rank=rank, size=4)
        archive(tmp_path, "second", [10, 20], b"abcd" if rank != 3 else b"abcD", rank=rank, size=4)
    report = tool.analyze([tmp_path], "first", "second", tmp_path / "report", workers=8)
    assert report["compared_bytes"] == 16
    assert report["different_bytes"] == 1
    assert report["first_differences"][0]["tp_rank"] == 3


def test_remote_stream_roundtrip_python39_syntax(tool, tmp_path):
    ast.parse(tool.REMOTE, feature_version=(3, 9))
    root = tmp_path / "remote"
    archive(root / "decode-kv-snapshots", "first", [10, 20], b"abcd")
    archive(root / "decode-kv-snapshots", "second", [10, 20, 30], b"abcdef")
    packed = tmp_path / "download.tar.gz"
    with packed.open("wb") as target:
        subprocess.run([sys.executable, "-c", tool.REMOTE, str(root), "", "first", "second"], stdout=target, check=True)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    counts = tool.safe_extract(packed, extracted)
    assert counts["files"] == 7
    assert tool.analyze([extracted], "first", "second", tmp_path / "report")["compared_bytes"] == 4


def test_remote_refuses_inflight_capture(tool, tmp_path):
    directory = archive(tmp_path / "decode-kv-snapshots", "first", [10], b"ab")
    path = directory / "manifest.json"
    meta = json.loads(path.read_text())
    meta["status"] = "writing"
    path.write_text(json.dumps(meta))
    result = subprocess.run([sys.executable, "-c", tool.REMOTE, str(tmp_path), "", "first"], capture_output=True)
    assert result.returncode != 0
    assert b"still being written" in result.stderr


def test_cli_records_failure_instead_of_false_success(tool, tmp_path):
    assert (
        tool.main(
            [
                "--previous",
                "first",
                "--current",
                "second",
                "--local",
                str(tmp_path),
                "--output",
                str(tmp_path / "report"),
            ]
        )
        == 1
    )
    assert json.loads((tmp_path / "report/report.json").read_text())["status"] == "incomplete"
