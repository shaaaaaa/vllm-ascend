# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[2] / "tools/slice_trace.py"
    spec = importlib.util.spec_from_file_location("slice_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_streaming_unicode_and_string_boundaries(tool, tmp_path, wrapped, chunk_size):
    rows = [{"name": '中文 } ] \\ " quoted', "ph": "X", "ts": "10.1", "dur": 7}, {"ts": 15}]
    value = {"traceEvents": rows, "otherData": {"test": True}} if wrapped else rows
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8-sig")
    assert list(tool.iter_events(source, chunk_size=chunk_size)) == rows


@pytest.mark.parametrize(
    "value,seconds", [("00:27:500", 27.5), ("00:28:000", 28), ("27.5", 27.5), ("01:00:001", 60.001)]
)
def test_time_units(tool, value, seconds):
    assert tool.parse_time(value) == pytest.approx(seconds)


@pytest.mark.parametrize("value", ["00:60:000", "00:20:1000", "-1", "nan", "inf", "00:27.500"])
def test_invalid_times(tool, value):
    with pytest.raises(argparse.ArgumentTypeError):
        tool.parse_time(value)


def test_window_uses_minimum_timestamp_and_preserves_metadata(tool, tmp_path):
    base = 1793774621601975
    events = [
        {"name": "later", "ph": "X", "pid": 1, "tid": 1, "ts": base + 40_000_000, "dur": 10},
        {"name": "origin", "ph": "i", "pid": 1, "tid": 1, "ts": str(base)},
        {"name": "process_name", "ph": "M", "pid": 1, "ts": 0, "args": {"name": "NPU"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "Stream 93"}},
        {"name": "left", "ph": "X", "pid": 1, "tid": 1, "ts": str(base + 27_400_000), "dur": 200000},
        {"name": "inside", "ph": "X", "pid": 1, "tid": 1, "ts": base + 27_600_000, "dur": 10},
        {"name": "right", "ph": "X", "pid": 1, "tid": 1, "ts": base + 27_900_000, "dur": "200000"},
        {"name": "end", "ph": "i", "pid": 1, "tid": 1, "ts": base + 28_000_000},
        {"name": "touch", "ph": "X", "pid": 1, "tid": 1, "ts": base + 27_400_000, "dur": 100000},
    ]
    source = tmp_path / "trace_view.json"
    source.write_text(json.dumps(events))
    tool.main([str(source)])
    output = source.with_name("trace_view_27.500-28.000.json")
    data = json.loads(output.read_text())
    assert data["otherData"]["trace_slice"]["origin_us"] == base
    rows = {item["name"]: item for item in data["traceEvents"]}
    assert set(rows) == {"process_name", "thread_name", "left", "inside", "right"}
    assert rows["left"]["ts"] == 0
    assert rows["left"]["dur"] == 100000
    assert rows["left"]["args"]["trace_slice_original"] == {"ts_us": str(base + 27_400_000), "dur_us": 200000}
    assert rows["inside"]["ts"] == 100000
    assert rows["right"]["dur"] == 100000
    assert rows["right"]["ts"] == 400000


def test_nested_begin_end_ranges_and_point_events(tool, tmp_path):
    rows = [
        {"name": "outer", "ph": "B", "pid": 1, "tid": 2, "ts": 0},
        {"name": "inner", "ph": "B", "pid": 1, "tid": 2, "ts": 100},
        {"name": "counter", "ph": "C", "pid": 1, "tid": 3, "ts": 110, "args": {"count": 7}},
        {"name": "zero", "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 0},
        {"ph": "E", "pid": 1, "tid": 2, "ts": 150},
        {"ph": "E", "pid": 1, "tid": 2, "ts": 300},
    ]
    source, output = tmp_path / "in.json", tmp_path / "out.json"
    source.write_text(json.dumps({"traceEvents": rows}))
    tool.slice_trace(source, output, .0001, .0002)
    actual = {row["name"]: row for row in json.loads(output.read_text())["traceEvents"]}
    assert actual["outer"]["ph"] == actual["inner"]["ph"] == "X"
    assert actual["outer"]["dur"] == 100
    assert actual["inner"]["dur"] == 50
    assert actual["counter"]["ts"] == 10
    assert actual["zero"]["ts"] == 0


def test_explicit_origin_and_original_times(tool, tmp_path):
    source, output = tmp_path / "in.json", tmp_path / "out.json"
    source.write_text(json.dumps([{"name": "op", "ph": "X", "ts": 1000100, "dur": 30}]))
    tool.main([str(source), "--start", ".0001", "--end", ".0002", "--origin-us", "1000000",
               "--keep-timestamps", "-o", str(output)])
    assert json.loads(output.read_text())["traceEvents"][0]["ts"] == 1000100


def test_empty_window_preserves_existing_output(tool, tmp_path):
    source, output = tmp_path / "in.json", tmp_path / "out.json"
    source.write_text(json.dumps([{"ph": "X", "ts": 10, "dur": 2}]))
    output.write_text("previous output")
    with pytest.raises(ValueError, match="No events"):
        tool.slice_trace(source, output, 1, 2)
    assert output.read_text() == "previous output"
    assert not list(tmp_path.glob("*.tmp"))


def test_incomplete_input_preserves_existing_output(tool, tmp_path):
    source, output = tmp_path / "in.json", tmp_path / "out.json"
    source.write_text('[{"ph":"X","ts":0,"dur":3000000},{"name":')
    output.write_text("previous output")
    with pytest.raises(ValueError, match="incomplete"):
        tool.slice_trace(source, output, 1, 2, origin=0)
    assert output.read_text() == "previous output"


def test_protects_input_path(tool, tmp_path):
    source = tmp_path / "in.json"
    source.write_text("[]")
    with pytest.raises(ValueError, match="different paths"):
        tool.slice_trace(source, source, 0, 1)
