# SPDX-License-Identifier: Apache-2.0
"""Numeric tail inspection without a model, accelerator, or full-prefix scan."""

import builtins
import json
import math
import struct

import pytest
from test_decode_kv_segments import client, setup, worker


@pytest.mark.parametrize("byteorder", ["little", "big"])
@pytest.mark.parametrize("stdlib", [False, True])
def test_bfloat16_and_float16_values_are_decoded_not_reinterpreted(byteorder, stdlib, monkeypatch):
    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if stdlib and name == "numpy":
            raise ImportError("bare host")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    endian = "<" if byteorder == "little" else ">"
    bf = struct.pack(endian + "6H", 0x3F80, 0xC000, 0x0000, 0x8000, 0x7F80, 0x7FC0)
    values = worker._values(bf, "torch.bfloat16", byteorder)
    assert list(values[:5]) == [1, -2, 0, 0, math.inf]
    assert math.isnan(values[5])
    stats = worker.NumericStats()
    stats.add(bf, "torch.bfloat16", 2, byteorder)
    result = stats.result()
    assert result["zero_values"] == result["zero_rows"] == 2
    assert result["nan"] == result["posinf"] == 1
    assert result["finite"] == 4
    assert result["mean"] == -0.25
    assert result["rms"] == pytest.approx(math.sqrt(5 / 4))
    half = struct.pack(endian + "4e", 0.5, -1.5, 0, -math.inf)
    assert list(worker._values(half, "torch.float16", byteorder)) == [0.5, -1.5, 0, -math.inf]


def test_streamed_moments_and_repeated_rows_cross_chunk_boundary():
    stats = worker.NumericStats()
    for row in [(1, 2), (1, 2), (0, 0), (1, 2)]:
        stats.add(bytes(row), "torch.uint8", 2, "little")
    result = stats.result()
    assert result["rows"] == 4
    assert result["zero_rows"] == 1
    assert result["adjacent_equal_rows"] == 1
    assert not result["all_rows_equal"]
    assert result["mean"] == 9 / 8
    assert result["rms"] == pytest.approx(math.sqrt(15 / 8))
    assert result["std"] == pytest.approx(math.sqrt(15 / 8 - (9 / 8) ** 2))


def replace_floats(directory, rows):
    path = directory / "manifest.json"
    meta = json.loads(path.read_text())
    meta["files"][0]["dtype"] = "torch.float16"
    path.write_text(json.dumps(meta))
    flat = [value for row in rows for value in row]
    (directory / "data.bin").write_bytes(struct.pack("<" + "e" * len(flat), *flat))


@pytest.mark.parametrize("stdlib", [False, True])
def test_numeric_tail_compares_overlap_and_new_suffix_without_hash_or_raw_transfer(tmp_path, monkeypatch, stdlib):
    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if stdlib and name == "numpy":
            raise ImportError("bare host")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    args, rpc, after = setup(tmp_path, same_host=True, count=5, width=4)
    args.inspect_start = 3
    before = tmp_path / "host-a/decode-kv-snapshots/before/host-tp0"
    replace_floats(before, [(1, 2)] * 5)
    replace_floats(after / "host-tp0", [(1, 2)] * 3 + [(2, 4)] * 2 + [(0, 0)] * 3)
    monkeypatch.setattr(worker, "NUMERIC_CHUNK_BYTES", 4)
    report = client.analyze_remote(args, rpc)
    assert report["mode"] == "tail_numeric"
    assert report["current_range"] == [3, 8]
    assert report["comparable_range"] == [3, 5]
    assert report["new_range"] == [5, 8]
    assert report["counts"]["all_zero_parts"] == 1
    assert report["counts"]["zero_component_rows"] == 3
    assert report["worst_relative_l2"]["relative_l2"] == 1
    assert report["worst_relative_l2"]["rmse"] == pytest.approx(math.sqrt(2.5))
    assert all(action not in ("checksums", "blocks", "compare") for _, action in rpc.calls)
    detail = json.loads((args.output / "tail_layers.jsonl").read_text())
    assert detail["previous"]["rms"] == pytest.approx(math.sqrt(2.5))
    assert detail["overlap"]["rms"] == pytest.approx(math.sqrt(10))
    assert detail["comparison"]["cosine"] == pytest.approx(1)


def test_numeric_nonfinite_and_zero_reference_have_no_nan_json(tmp_path):
    args, rpc, after = setup(tmp_path, same_host=True, count=5, width=4)
    args.inspect_start = 3
    before = tmp_path / "host-a/decode-kv-snapshots/before/host-tp0"
    replace_floats(before, [(0, 0)] * 5)
    replace_floats(after / "host-tp0", [(0, 0)] * 3 + [(math.nan, 1), (math.inf, -math.inf)] + [(0, 0)] * 3)
    report = client.analyze_remote(args, rpc)
    assert report["counts"]["nonfinite_values"] == 3
    assert report["counts"]["new_nonfinite_in_overlap"] == 3
    detail = json.loads((args.output / "tail_layers.jsonl").read_text())
    assert detail["comparison"]["finite_pairs"] == 1
    assert detail["comparison"]["relative_l2"] is None
    json.dumps(report, allow_nan=False)


def test_cross_host_tail_is_distribution_only_without_unbounded_download(tmp_path):
    args, rpc, _ = setup(tmp_path, count=6)
    args.inspect_start = 4
    report = client.analyze_remote(args, rpc)
    assert report["counts"]["cross_host_unpaired_components"] == 1
    assert report["counts"]["compared_components"] == 0
    detail = json.loads((args.output / "tail_layers.jsonl").read_text())
    assert "previous" in detail
    assert "comparison" not in detail
    assert "different_hosts" in detail["comparison_unavailable"]
    assert not any(action == "blocks" for _, action in rpc.calls)


def test_inspect_start_may_select_only_new_rows(tmp_path):
    args, rpc, _ = setup(tmp_path, same_host=True, count=6)
    args.inspect_start = 7
    report = client.analyze_remote(args, rpc)
    assert report["comparable_range"] == [7, 7]
    assert report["new_range"] == [7, 9]
    assert report["counts"]["compared_components"] == 0


def test_tail_cli_preserves_existing_history_report(tmp_path, monkeypatch):
    from test_decode_kv_segments import cli

    args, rpc, _ = setup(tmp_path, same_host=True, count=6)
    monkeypatch.setattr(client, "RemoteClient", lambda ignored: rpc)
    history = args.output / "report.json"
    history.write_text('{"previous_analysis":true}')
    assert (
        cli.main(
            [
                "--previous",
                "before",
                "--current",
                "after",
                "--hosts",
                "p",
                "--inspect-start",
                "4",
                "--output",
                str(args.output),
            ]
        )
        == 0
    )
    assert json.loads(history.read_text()) == {"previous_analysis": True}
    assert (args.output / "tail_report.json").is_file()
