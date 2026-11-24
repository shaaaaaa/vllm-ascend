# SPDX-License-Identifier: Apache-2.0
"""Exercise the complete explicit-model probe with CPU buffers and a model of notify delivery."""

import ctypes as C
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    import check_decode_creation_priority

    monkeypatch.setattr(check_decode_creation_priority, "ELEMENTS", 32)
    monkeypatch.setattr(check_decode_creation_priority, "STEPS", 3)
    return check_decode_creation_priority


class CpuRuntime:
    def __init__(self, probe, *, honor=True, corrupt=False):
        self.probe, self.honor, self.corrupt = probe, honor, corrupt
        self.functions = {"aclrtGetSocName": lambda: b"CPU fixture"}
        self.next_id = 100
        self.resources = {}
        self.streams = {}
        self.bound = {}
        self.tasks = {}
        self.waits = {}
        self.events = {}
        self.stamps = {}
        self.executors = {}
        self.cleanup = []
        self.model_names = []

    def manage(self, stack, create, destroy, *args):
        self.next_id += 1
        handle = self.next_id
        if create in ("aclrtMalloc", "aclrtMallocHost"):
            buffer = C.create_string_buffer(args[0])
            handle = C.addressof(buffer)
            self.resources[handle] = buffer
        else:
            self.resources[handle] = create
        if create == "aclrtCreateStreamWithConfig":
            priority, flag = args
            assert flag == self.probe.PERSISTENT
            self.streams[handle] = priority
        stack.callback(self.resources.pop, handle)
        return handle

    def tensor(self, stack, address):
        return address

    def stream_id(self, stream):
        return stream

    def abs(self, src, dst):
        for i in range(self.probe.ELEMENTS):
            C.c_uint16.from_address(dst + 2 * i).value = C.c_uint16.from_address(src + 2 * i).value & 0x7FFF

    def execute(self, model):
        lanes = [s for s, m in self.bound.items() if m == model]
        assert len(lanes) == 2 and set(self.waits.values()) == set(self.events)
        signals = set()
        join_waits = []
        for stream in lanes:
            ops = self.tasks[stream]
            assert [op[0] for op in ops[: self.probe.STEPS]] == ["abs"] * self.probe.STEPS
            assert ops[self.probe.STEPS][0] == "signal"
            assert [op[0] for op in ops[self.probe.STEPS + 1 :]] in (["signal"], ["join", "end"])
            for name, *values in ops:
                if name == "abs" and not self.corrupt:
                    self.abs(*values)
                elif name == "signal":
                    notify = values[0]
                    signals.add(notify)
                    if notify in self.waits:
                        observer = self.waits.pop(notify)
                        priority = self.streams[stream] if self.honor else 0
                        self.stamps[self.events.pop(observer)] = 100 + 10 * priority
                elif name == "join":
                    join_waits.extend(values)
        assert len(join_waits) == 1 and join_waits[0] in signals
        assert not self.waits and not self.events

    def call(self, name, *args):
        if name == "aclrtDeviceGetStreamPriorityRange":
            args[0]._obj.value, args[1]._obj.value = 7, 0
        elif name == "aclrtMemset":
            address, capacity, byte, size = args
            assert size <= capacity
            C.memset(address, byte, size)
        elif name == "aclnnAbsGetWorkspaceSize":
            src, dst, size, executor = args
            self.next_id += 1
            size._obj.value, executor._obj.value = 64, self.next_id
            self.executors[self.next_id] = (src, dst)
        elif name == "aclnnAbs":
            workspace, size, executor, stream = args
            assert workspace in self.resources and size == 64
            src, dst = self.executors.pop(executor)
            if stream in self.bound:
                self.tasks[stream].append(("abs", src, dst))
            else:
                self.abs(src, dst)
        elif name == "aclmdlRIBindStream":
            model, stream, flag = args
            assert flag == self.probe.HEAD
            self.bound[stream] = model
            self.tasks[stream] = []
        elif name == "aclrtRecordNotify":
            notify, stream = args
            self.tasks[stream].append(("signal", notify))
        elif name == "aclrtWaitAndResetNotify":
            notify, stream, _ = args
            if stream in self.bound:
                self.tasks[stream].append(("join", notify))
            else:
                self.waits[notify] = stream
        elif name == "aclmdlRIEndTask":
            model, stream = args
            assert self.bound[stream] == model
            self.tasks[stream].append(("end",))
        elif name == "aclrtRecordEvent":
            event, stream = args
            self.events[stream] = event
        elif name == "aclmdlRIGetId":
            args[1]._obj.value = args[0]
        elif name == "aclmdlRISetName":
            self.model_names.append(args[1].decode())
        elif name == "aclmdlRIExecuteAsync":
            self.execute(args[0])
        elif name == "aclrtEventGetTimestamp":
            args[1]._obj.value = self.stamps.pop(args[0])
        elif name == "aclrtMemcpy":
            dst, capacity, src, size, kind = args
            assert kind == self.probe.D2H and size <= capacity
            C.memmove(dst, src, size)
        elif name == "aclmdlRIUnbindStream":
            model, stream = args
            assert self.bound.pop(stream) == model
        elif name in ("aclFinalize", "aclrtResetDevice"):
            self.cleanup.append(name)
        else:
            assert name in ("aclInit", "aclrtSetDevice", "aclrtSynchronizeStreamWithTimeout", "aclmdlRIBuildEnd")


@pytest.mark.parametrize("honor", [True, False])
def test_full_probe_uses_correct_replay_outputs_and_both_queue_orders(probe, honor):
    rt = CpuRuntime(probe, honor=honor)
    report = {}
    status = probe.run(rt, 0, report)
    expected = "creation_priority_effect_observed" if honor else "no_clear_creation_priority_effect"
    assert status == expected and len(report["cases"]) == 6
    assert "active_case" not in report and report["range"] == {"least": 7, "greatest": 0}
    assert len(set(rt.model_names)) == 6
    assert [case["model_name"] for case in report["cases"]] == rt.model_names
    assert [case["creation_order"] for case in report["cases"]] == [[0, 1]] * 3 + [[1, 0]] * 3
    assert all(len(case["b_minus_a_us"]) == probe.REPLAYS for case in report["cases"])
    assert not rt.bound and not rt.resources and not rt.executors
    assert rt.cleanup == ["aclrtResetDevice", "aclFinalize"]


def test_bad_device_output_fails_before_the_performance_verdict(probe):
    rt = CpuRuntime(probe, corrupt=True)
    with pytest.raises(RuntimeError, match="Abs replay output differs"):
        probe.run(rt, 0, {})
    assert not rt.bound and not rt.resources


@pytest.mark.parametrize(
    ("honor", "corrupt", "status", "rc"),
    [
        (True, False, "creation_priority_effect_observed", 0),
        (False, False, "no_clear_creation_priority_effect", 0),
        (True, True, "failed", 1),
    ],
)
def test_cli_preserves_profile_for_either_measured_outcome(probe, monkeypatch, capsys, honor, corrupt, status, rc):
    monkeypatch.setattr(sys, "argv", ["probe", "--device", "0"])
    monkeypatch.setattr(probe, "CreationRuntime", lambda: CpuRuntime(probe, honor=honor, corrupt=corrupt))
    assert probe.main() == rc
    prefix, payload = capsys.readouterr().out.split(" ", 1)
    result = json.loads(payload)
    assert prefix == "[CREATE_PRIORITY]" and result["status"] == status
    if corrupt:
        assert result["active_case"]["priorities"] == [0, 0] and "output differs" in result["error"]


@pytest.mark.parametrize(
    ("equal", "a_high", "b_high"),
    [
        ([20] * 5, [40] * 5, [15] * 5),  # Fixed A-first bias.
        ([0, 10, -10, 20, -20], [20] * 5, [-20] * 5),  # Within control noise.
        ([0] * 5, [10, 10, 10, 0, 0], [-20] * 5),  # Three wins out of five.
    ],
)
def test_fixed_order_jitter_and_partial_reversals_are_inconclusive(probe, equal, a_high, b_high):
    assert not probe.priority_effect(equal, a_high, b_high)
