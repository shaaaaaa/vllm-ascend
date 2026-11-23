# SPDX-License-Identifier: Apache-2.0
"""Exercise the graph priority verdicts with independent CANN set and query results."""

import ctypes
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    import check_decode_graph_priority

    return check_decode_graph_priority


@pytest.mark.parametrize(
    ("set_rc", "main_reads", "side_reads", "expected"),
    [
        (0, (0, 0), (7, 7), "graph_priority_verified"),
        (507899, (0, 0), (7, 7), "graph_priority_set_failed"),
        (0, (507899, 507899), (7, 7), "graph_priority_readback_failed"),
        (0, (0, 0), (507899, 7), "graph_priority_readback_failed"),
        (0, (0, 0), (7, 507899), "graph_priority_readback_failed"),
        (0, (0, 0), (7, 0), "graph_priority_mismatch"),
        (0, (7, 7), (7, 7), "graph_priority_mismatch"),
    ],
)
def test_graph_set_and_both_device_reads_are_independent(probe, set_rc, main_reads, side_reads, expected):
    rt = probe.Runtime.__new__(probe.Runtime)
    operations = []
    values = {11: iter(main_reads), 22: iter(side_reads)}

    def set_priority(stream, attribute, value):
        assert attribute == probe.PRIORITY
        operations.append(("set", stream, value._obj.priority))
        return set_rc

    def get_priority(stream, attribute, value):
        assert attribute == probe.PRIORITY
        operations.append(("get", stream))
        result = next(values[stream])
        if result == 507899:
            return result
        value._obj.priority = result
        return 0

    rt.functions = {"aclrtSetStreamAttribute": set_priority, "aclrtGetStreamAttribute": get_priority}
    rt.stream_id = lambda stream: stream
    report, replays = {}, []
    assert probe.check_graph_queues(rt, 11, 22, 7, 0, report, replays.append) == expected
    assert operations == [("set", 22, 7), ("get", 11), ("get", 22), ("get", 11), ("get", 22)]
    assert report["set_side"] == {"requested": 7, "rc": set_rc}
    assert report["verified_replays"] == 3 and replays == [1, 7, 2]
    for stage, main, side in zip(("after_set", "after_replay"), main_reads, side_reads):
        for role, result in (("main", main), ("side", side)):
            assert report[stage][role] == ({"rc": 507899} if result == 507899 else {"rc": 0, "priority": result})


def test_creation_field_is_separate_from_failing_device_query(probe):
    rt = probe.Runtime.__new__(probe.Runtime)
    rt.stream_id = lambda stream: stream
    destroyed = []

    def create(output, requested, flags):
        assert (requested, flags) == (7, 0)
        output._obj.value = 33
        return 0

    def get_creation_field(stream, output):
        assert stream.value == 33
        output._obj.value = 7
        return 0

    rt.functions = {
        "aclrtCreateStreamWithConfig": create,
        "aclrtStreamGetPriority": get_creation_field,
        "aclrtSetStreamAttribute": lambda *args: 507899,
        "aclrtGetStreamAttribute": lambda *args: 507899,
        "aclrtDestroyStream": lambda stream: destroyed.append(stream.value) or 0,
    }
    ordinary = rt.ordinary_controls(9, 7)
    assert ordinary["set_side"] == {"requested": 7, "rc": 507899}
    assert ordinary["after_set"] == {"rc": 507899}
    assert ordinary["create_low"]["creation_field"] == {"rc": 0, "priority": 7}
    assert ordinary["create_low"]["device_read"] == {"rc": 507899}
    assert destroyed == [33]
    assert ctypes.sizeof(probe.StreamAttribute) == 16
    assert ctypes.alignment(probe.StreamAttribute) == ctypes.alignment(ctypes.c_uint64)


def test_incomplete_replay_preserves_set_and_read_results(probe):
    rt = probe.Runtime.__new__(probe.Runtime)
    rt.stream_id = lambda stream: stream
    rt.set_priority = lambda stream, requested: {"rc": 0, "requested": requested}
    rt.device_priority = lambda stream: {"rc": 0, "priority": 0 if stream == 11 else 7}

    def replay(value):
        if value == 7:
            raise RuntimeError("incorrect replay output")

    report = {}
    assert probe.check_graph_queues(rt, 11, 22, 7, 0, report, replay) == "graph_replay_failed"
    assert report["set_side"]["rc"] == 0 and report["after_set"]["side"]["priority"] == 7
    assert report["verified_replays"] == 1
    assert report["replay_error"] == "incorrect replay output"


@pytest.mark.parametrize(
    ("verdict", "returncode"),
    [("graph_priority_verified", 0), ("graph_priority_set_failed", 1), ("graph_priority_readback_failed", 1)],
)
def test_cli_exit_matches_graph_verdict(probe, monkeypatch, capsys, verdict, returncode):
    monkeypatch.setattr(sys, "argv", ["probe", "--device", "0"])
    monkeypatch.setattr(probe, "Runtime", object)
    monkeypatch.setattr(probe, "run", lambda rt, device, report: verdict)
    assert probe.main() == returncode
    assert json.loads(capsys.readouterr().out.removeprefix("[GRAPH_PRIORITY] "))["status"] == verdict
