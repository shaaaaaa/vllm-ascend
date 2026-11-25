# SPDX-License-Identifier: Apache-2.0
import ctypes as C
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    import check_aiv_gate

    return check_aiv_gate


class GraphModel:
    """Play the same recorded work on two FIFO queues and one shared pool of 40 cores."""

    def __init__(self, tool):
        self.tool = tool
        self.commands = defaultdict(list)

    def initialize(self, stream):
        self.commands[stream].append(("write", self.tool.CAPACITY))

    def call(self, name, *args):
        if name == "aclrtRecordEvent":
            event, stream = args
            self.commands[stream].append(("signal", event))
        else:
            assert name == "aclrtStreamWaitEvent"
            stream, event = args
            self.commands[stream].append(("wait_event", event))

    def pulse(self, stream, slot, blocks, ticks, *, mode=0, charge=None, count=1, ready=None, signal_after=0):
        cmd = self.commands[stream]
        charge = blocks if charge is None else charge
        if mode == self.tool.FOREGROUND:
            cmd.append(("write", self.tool.CAPACITY - charge))
        if mode == self.tool.PIECE:
            assert count == 1
            cmd.append(("wait_value", charge))
        for i in range(count):
            if ready is not None and signal_after == i:
                cmd.append(("signal", ready))
            cmd.append(("kernel", slot + i, blocks, ticks))
        if ready is not None and signal_after == count:
            cmd.append(("signal", ready))
        if mode == self.tool.FOREGROUND:
            cmd.append(("write", self.tool.CAPACITY))

    def replay(self, order, start_tick, *, ignore_wait=False):
        heads = {lane: 0 for lane in order}
        events, running = set(), {}
        raw = bytearray(self.tool.STRIDE * (self.tool.SHORTS + 1))
        now, allowance = start_tick, 0
        while any(heads[lane] < len(self.commands[lane]) for lane in order) or running:
            for lane in list(running):
                if running[lane][0] <= now:
                    del running[lane]
            progress = False
            for lane in order:
                while lane not in running and heads[lane] < len(self.commands[lane]):
                    op, *args = self.commands[lane][heads[lane]]
                    if op == "write":
                        (allowance,) = args
                    elif op == "signal":
                        events.add(args[0])
                    elif op == "wait_event":
                        if args[0] not in events:
                            break
                    elif op == "wait_value":
                        if not ignore_wait and allowance < args[0]:
                            break
                    else:
                        assert op == "kernel"
                        slot, blocks, ticks = args
                        if blocks + sum(n for _, n in running.values()) > self.tool.CAPACITY:
                            break
                        running[lane] = now + ticks, blocks
                        for i in range(blocks):
                            offset = slot * self.tool.STRIDE + i * self.tool.LINE.size
                            self.tool.LINE.pack_into(raw, offset, now, now + ticks)
                    heads[lane] += 1
                    progress = True
            if not progress:
                if not running and all(heads[lane] == len(self.commands[lane]) for lane in order):
                    break
                assert running, "device graph deadlock"
                now = min(end for end, _ in running.values())
        return raw, allowance


@pytest.mark.parametrize(
    "case", ["raw_train", "open_train", "long_high", "train_high", "raw_28_12", "allow_12", "defer_13"]
)
@pytest.mark.parametrize("order", [(1, 2), (2, 1)])
def test_record_and_replay_under_opposite_resource_tie_breaks(tool, case, order):
    graph = GraphModel(tool)
    tool.record(case, graph, graph, 1, 2, 3, 4)
    for base in (10, 10_000_000, 20_000_000):
        raw, allowance = graph.replay(order, base)
        metrics, errors = tool.assess(case, raw)
        assert errors == []
        assert allowance == (0 if case in tool.CONTROLS else tool.CAPACITY)
        if case == "long_high":
            assert metrics["long_start_ms"] in (10, 12)
            assert metrics["ready_to_long_ms"] in (0, 2)
        if case == "train_high":
            assert metrics["long_start_ms"] == 120 and metrics["ready_to_long_ms"] == 110


@pytest.mark.parametrize("case,order", [("long_high", (1, 2)), ("train_high", (2, 1)), ("defer_13", (1, 2))])
def test_missing_device_wait_is_detected(tool, case, order):
    graph = GraphModel(tool)
    tool.record(case, graph, graph, 1, 2, 3, 4)
    raw, _ = graph.replay(order, 100, ignore_wait=True)
    assert tool.assess(case, raw)[1]


def test_corrupt_or_stale_kernel_stamps_and_resource_waves_are_detected(tool):
    graph = GraphModel(tool)
    tool.record("allow_12", graph, graph, 1, 2, 3, 4)
    raw, _ = graph.replay((1, 2), 100)
    raw[tool.STRIDE : tool.STRIDE + tool.LINE.size] = bytes(tool.LINE.size)
    with pytest.raises(ValueError, match="invalid stamps"):
        tool.assess("allow_12", raw)
    tool.LINE.pack_into(raw, tool.STRIDE, 100 + tool.LONG_TICKS, 100 + 2 * tool.LONG_TICKS)
    with pytest.raises(ValueError, match="common busy interval"):
        tool.assess("allow_12", raw)


def test_identical_partial_geometry_charged_12_vs_13(tool):
    recordings = []
    for case in ("allow_12", "defer_13"):
        graph = GraphModel(tool)
        tool.record(case, graph, graph, 1, 2, 3, 4)
        recordings.append(graph.commands)
    allow, defer = recordings
    assert allow[1] == defer[1]
    assert [item for item in allow[2] if item[0] != "wait_value"] == [
        item for item in defer[2] if item[0] != "wait_value"
    ]
    assert ("wait_value", 12) in allow[2] and ("wait_value", 13) in defer[2]


@pytest.mark.parametrize("has_task_query", [False, True])
def test_native_bridge_argument_layout_and_error_reporting(tool, monkeypatch, tmp_path, has_task_query):
    native = SimpleNamespace(aiv_probe_initialize=Mock(return_value=0), aiv_probe_pulses=Mock(return_value=0))
    if has_task_query:
        native.aiv_probe_kernel_width = Mock()
    monkeypatch.setattr(tool.C, "CDLL", lambda *a, **kw: native)
    probe = tool.Probe(tmp_path, tool.Ptr(77), 88, 99)
    assert (probe.read_kernel_width is not None) == has_task_query
    probe.initialize(11)
    probe.pulse(11, 3, 12, 555, mode=tool.PIECE, charge=13)
    gate, stream, mode, charge, batch = native.aiv_probe_pulses.call_args.args
    assert (gate._obj.word, gate._obj.capacity, stream, mode, charge) == (88, 40, 11, 2, 13)
    b = batch._obj
    assert (b.kernel, b.stamps, b.ticks, b.ready, b.blocks, b.count, b.signal_after) == (
        77,
        99 + 3 * tool.STRIDE,
        555,
        None,
        12,
        1,
        0,
    )
    native.aiv_probe_pulses.return_value = 123
    with pytest.raises(RuntimeError, match="rc=123"):
        probe.pulse(11, 3, 12, 555)


def test_reads_widths_only_for_captured_kernels(tool):
    per_stream = {11: [2, 41], 12: [42, 3], 13: []}
    widths = {41: 28, 42: 12}

    def call(name, first, *outputs):
        if name == "aclmdlRITaskGetType":
            outputs[0]._obj.value = 1 if first in widths else 2
        else:
            assert name == "aclmdlRIGetTasksByStream"
            tasks, count = outputs
            if tasks is None:
                count._obj.value = len(per_stream[first])
            else:
                assert count._obj.value > 0
                tasks[:] = per_stream[first]

    rt = SimpleNamespace(call=call, bind=Mock(), streams=lambda g: list(per_stream))
    probe = SimpleNamespace(read_kernel_width=lambda task, cap: widths[task])
    assert tool.kernel_widths(rt, probe, 123) == [12, 28]


def test_failed_capture_ends_before_teardown_and_preserves_error(tool, monkeypatch):
    calls = []

    def call(name, *args):
        calls.append(name)

    def end(stream, graph):
        graph._obj.value = 55
        calls.append("end")
        return 0

    rt = SimpleNamespace(call=call, functions={"aclmdlRICaptureEnd": end})
    monkeypatch.setattr(tool, "record", Mock(side_effect=RuntimeError("enqueue failed")))
    with pytest.raises(RuntimeError, match="enqueue failed"), ExitStack() as stack:
        tool.capture(rt, stack, "long_high", None, 1, 2, 3, 4)
    assert calls == ["aclmdlRICaptureBegin", "end", "aclmdlRIDestroy"]


def test_replays_one_capture_with_fresh_buffers_and_detects_missing_restore(tool, monkeypatch):
    case, captures = "allow_12", []
    graph = GraphModel(tool)
    stamps = C.create_string_buffer(2 * tool.STRIDE)
    host = C.create_string_buffer(2 * tool.STRIDE)
    word = C.c_uint64()
    graph.stamps = C.addressof(stamps)
    graph.gate = tool.Gate(C.addressof(word), tool.CAPACITY)
    count = 0
    handles = iter([11, 12, 13, 14])

    def call(name, *args):
        nonlocal count
        if name == "aclmdlRICaptureBegin":
            captures.append(args[0])
        elif name == "aclmdlRICaptureEnd":
            args[1]._obj.value = 50
        elif name in ("aclrtRecordEvent", "aclrtStreamWaitEvent"):
            graph.call(name, *args)
        elif name == "aclrtMemset":
            C.memset(args[0], args[2], args[3])
        elif name == "aclrtMemcpy":
            C.memmove(args[0], args[2], args[3])
        elif name == "aclmdlRIExecuteAsync":
            assert args == (50, 11) and word.value == 0 and not any(stamps.raw)
            count += 1
            data, allowance = graph.replay((11, 12), count * 10_000_000)
            C.memmove(graph.stamps, bytes(data), C.sizeof(stamps))
            word.value = allowance if count < 3 else 12
        else:
            assert name in (
                "aclrtSynchronizeStreamWithTimeout",
                "aclmdlRIDestroy",
                "aclrtDestroyEvent",
                "aclrtDestroyStream",
            )

    rt = SimpleNamespace(new=lambda *a: next(handles), call=call)
    monkeypatch.setattr(tool, "kernel_widths", lambda *args: [12, 28])
    rows, errors = tool.run_case(rt, graph, C.addressof(host), case, 3)
    assert captures == [11] and len(rows) == count == 3
    assert errors == ["replay 3: foreground allowance was not restored"]


@pytest.mark.parametrize("has_task_query", [False, True], ids=["acl-8.5.1", "acl-9.1"])
def test_full_command_runs_all_graphs_and_releases_allocations(tool, monkeypatch, capfd, has_task_query):
    graph = GraphModel(tool)
    buffers, streams, captured, replayed = {}, [], [], []
    event_ids = iter(range(20, 34))

    class Runtime:
        functions = {"aclrtGetSocName": lambda: b"Ascend910B3"}

        def __init__(self, *, capture_functions):
            assert not capture_functions

        def bind(self, signatures):
            metadata = {"aclmdlRIGetStreams", "aclmdlRIGetTasksByStream", "aclmdlRITaskGetType"}
            if "aclrtSynchronizeStreamWithTimeout" in signatures:
                assert metadata.isdisjoint(signatures)
            else:
                assert has_task_query and set(signatures) == metadata

        def streams(self, model):
            assert has_task_query
            return streams

        def new(self, name, *args):
            if name in ("aclrtMalloc", "aclrtMallocHost"):
                buf = C.create_string_buffer(args[0])
                buffers[C.addressof(buf)] = buf
                return C.addressof(buf)
            if name == "aclrtCreateStreamWithConfig":
                assert args == (0, 0)
                streams.append(10 + len(streams))
                return streams[-1]
            assert name == "aclrtCreateEventExWithFlag"
            return next(event_ids)

        def call(self, name, *args):
            if name == "aclrtGetDeviceInfo":
                args[-1]._obj.value = tool.CAPACITY
            elif name == "aclmdlRICaptureBegin":
                graph.commands.clear()
                captured.append(args[0])
            elif name == "aclmdlRICaptureEnd":
                args[1]._obj.value = len(captured)
            elif name == "aclmdlRIGetTasksByStream":
                stream, tasks, count = args
                if tasks is None:
                    count._obj.value = 1
                else:
                    tasks[0] = 41 if stream == streams[0] else 42
            elif name == "aclmdlRITaskGetType":
                args[1]._obj.value = 1
            elif name in ("aclrtRecordEvent", "aclrtStreamWaitEvent"):
                graph.call(name, *args)
            elif name == "aclmdlRIExecuteAsync":
                number, root = args
                assert root == streams[0] and number == len(captured)
                replayed.append(number)
                raw, allowance = graph.replay(streams, 100 + len(replayed) * 10_000_000)
                C.memmove(graph.stamps, bytes(raw), len(raw))
                C.c_uint64.from_address(graph.gate.word).value = allowance
            elif name == "aclrtMemset":
                C.memset(args[0], args[2], args[3])
            elif name == "aclrtMemcpy":
                C.memmove(args[0], args[2], args[3])
            elif name == "aclrtDestroyStream":
                streams.remove(args[0])
            elif name in ("aclrtFree", "aclrtFreeHost"):
                del buffers[args[0]]
            else:
                assert name in {
                    "aclInit",
                    "aclFinalize",
                    "aclrtSetDevice",
                    "aclrtResetDevice",
                    "aclrtDestroyEvent",
                    "aclmdlRIDestroy",
                    "aclrtSynchronizeStreamWithTimeout",
                }

    def make_probe(build, kernel, word, stamps):
        graph.gate, graph.stamps = tool.Gate(word, tool.CAPACITY), stamps
        graph.read_kernel_width = (lambda task, cap: {41: 28, 42: 12}[task]) if has_task_query else None
        return graph

    prepare = Mock(return_value=33)
    monkeypatch.setattr(tool, "Runtime", Runtime)
    monkeypatch.setattr(tool, "Probe", make_probe)
    monkeypatch.setattr(tool, "prepare_kernel", prepare)
    monkeypatch.setattr(tool.sys, "argv", ["check_aiv_gate.py", "--device", "0"])
    assert tool.main() == 0
    assert captured == [10] * 7 and replayed == [n for n in range(1, 8) for _ in range(3)]
    assert prepare.call_args.kwargs == {"gate_probe": True}
    assert not buffers and not streams
    out = capfd.readouterr().out
    expected_widths = "captured_kernel_widths=[12, 28] verified" if has_task_query else "kernel_width_source=launch"
    assert expected_widths in out
    for result in ("long_start_ms=", "all_40_overlap_ms=", "open_train_minus_raw_ms=", "defer_13_minus_allow_12_ms="):
        assert result in out
    assert out.rstrip().endswith("[AIV_GATE] gate_graph_verified")
