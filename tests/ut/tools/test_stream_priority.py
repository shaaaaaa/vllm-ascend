# SPDX-License-Identifier: Apache-2.0
import ctypes as C
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    import check_stream_priority

    return check_stream_priority


def timeline(probe, long_position=6, unit=2_000_000):
    short = [(i * unit, (i + 1) * unit) for i in range(probe.SHORTS)]
    long_start = long_position * unit
    # One full-width lane. Already-started work finishes before the long kernel.
    short = [(a, b) if a < long_start else (a + 20 * unit, b + 20 * unit) for a, b in short]
    t0 = 123_456_789_000
    base = 5_000_000_000
    rows = [(base + a, base + b) for a, b in [*short, (long_start, long_start + 20 * unit)]]
    return probe.timing(
        rows, (t0, t0 + 20_000, t0 + 5 * unit, t0 + 5 * unit + 1000), (t0 - base - 500, t0 - base + 500), unit
    )


def test_verdict_uses_device_entry_and_both_priority_directions(probe):
    early, drained = timeline(probe), timeline(probe, 60)
    assert early["long_started_ms"] == pytest.approx(12)
    assert drained["long_started_ms"] == pytest.approx(120)
    assert early["shorts_done_at_long_start"] == 6
    pair = {"S1_high": early, "S2_high": drained}
    assert probe.verdict([pair, pair], 2) == "strict_priority_observed"
    assert probe.verdict([pair], 2) == "strict_pattern_one_pair"
    pair["S2_high"] = timeline(probe, 23)
    assert probe.verdict([pair, pair], 2) == "directional_priority_observed"
    pair["S2_high"] = early
    assert probe.verdict([pair, pair], 2) == "both_started_with_shorts_still_queued"
    pair["S1_high"] = pair["S2_high"] = drained
    assert probe.verdict([pair, pair], 2) == "both_started_after_short_train"
    pair["S1_high"] = early
    early["issues"].append("arrival too late")
    assert probe.verdict([pair, pair], 2) == "inconclusive_arrival_or_timing"


@pytest.mark.parametrize("launch_cost", [5, 70])
def test_host_submits_all_sixty_then_one_at_deadline(probe, launch_cost):
    now, calls = [100], []

    def shorts():
        calls.append((60, now[0]))
        now[0] += launch_cost

    def nap(seconds):
        now[0] += max(1, int(seconds * 1e9) // 2)

    times = probe.enqueue_train(shorts, lambda: calls.append((1, now[0])), 10, lambda: now[0], nap)
    assert times == (100, 100 + launch_cost, max(150, 100 + launch_cost), max(150, 100 + launch_cost))
    assert calls == [(60, 100), (1, times[-1])]


def test_timing_rejects_empty_backlog_late_host_and_clock_error(probe):
    tasks = [(1000 * i, 1000 * (i + 1)) for i in range(60)] + [(66000, 86000)]
    r = probe.timing(tasks, (0, 6000, 61000, 61001), (0, 600), 1000)
    assert len(r["issues"]) == 4


def test_official_arch_toolkit_layout(probe, tmp_path, monkeypatch):
    arch = tmp_path / "aarch64-linux/tikcpp/ascendc_kernel_cmake"
    arch.mkdir(parents=True)
    (arch / "ascendc.cmake").touch()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.delenv("ASCENDC_CMAKE_DIR", raising=False)
    monkeypatch.setattr(probe.platform, "machine", lambda: "aarch64")
    assert probe.toolkit() == (tmp_path.resolve(), arch.resolve())


def test_program_two_streams_entire_command_on_simulated_device(probe, monkeypatch, tmp_path, capsys):
    # Exercise the same raw timestamp buffer, calibration brackets, scheduling calls,
    # and per-trial summary as the real command.
    now = [10_000_000_000]
    origin = 100_000_000
    idle_at = [now[0]]
    memory, streams, train, launches = {}, {}, [], []
    blocks, short_ticks = 4, 100_000
    expected_priorities = []

    def stamp(ptr, block, begin, duration):
        tick = (begin - origin) // 20
        struct.pack_into("=QQ", (C.c_char * 64).from_address(ptr + block * 64), 0, tick, tick + duration)

    def enqueue(block_count, stream, ptr, count, ticks):
        if block_count == 1:
            stamp(ptr, 0, now[0] + 2000, ticks)
            idle_at[0] = now[0] + 2000 + 20 * ticks
            now[0] += 5000
            return 0
        assert block_count == blocks
        launches.append((count, ticks))
        if count == 60:
            assert not train
            train.extend((ptr, now[0] + 20_000))
        else:
            assert count == 1 and ticks == 20 * short_ticks and train
            short_ptr, first = train
            n = 6 if streams[stream] == 0 else 60
            for i in range(60):
                begin = first + (i + (20 if i >= n else 0)) * short_ticks * 20
                for core in range(blocks):
                    stamp(short_ptr + i * blocks * 64, core, begin + 80 * core, short_ticks)
            for core in range(blocks):
                stamp(ptr, core, first + n * short_ticks * 20 + 1000 + 80 * core, ticks)
            idle_at[0] = first + 80 * short_ticks * 20 + 2000
            train.clear()
        now[0] += 1000
        return 0

    class Runtime:
        fn = {"aclrtGetSocName": lambda: b"Ascend910B3"}

        def own(self, stack, create, destroy, *args):
            if create == "aclrtCreateStreamWithConfig":
                assert len(streams) < 2
                ident = len(streams) + 1
                streams[ident] = args[0]
                expected_priorities.append(args[0])
                stack.callback(streams.pop, ident)
                return ident
            buf = C.create_string_buffer(args[0])
            address = C.addressof(buf)
            memory[address] = buf
            stack.callback(memory.pop, address)
            return address

        def call(self, name, *args):
            if name == "aclrtGetDeviceInfo":
                assert args[1] == 201
                args[2]._obj.value = blocks
            elif name == "aclrtDeviceGetStreamPriorityRange":
                args[0]._obj.value, args[1]._obj.value = 7, 0
            elif name == "aclrtStreamGetPriority":
                args[1]._obj.value = streams[args[0]]
            elif name == "aclrtMemcpy":
                dst, capacity, src, size, kind = args
                assert size <= capacity and kind == 2
                C.memmove(dst, src, size)
            elif name == "aclrtMemset":
                C.memset(args[0], args[2], args[3])
            elif name == "aclrtSynchronizeStreamWithTimeout":
                assert not train, "host sync inside the measured submission interval"
                now[0] = max(now[0], idle_at[0]) + 5000
            else:
                assert name in {"aclInit", "aclFinalize", "aclrtSetDevice", "aclrtResetDevice"}

    def nap(sec):
        now[0] += int(sec * 1e9)

    schedule = probe.enqueue_train
    monkeypatch.setattr(probe, "Runtime", Runtime)
    monkeypatch.setattr(probe, "build_kernel", lambda soc, build: enqueue)
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: now[0])
    monkeypatch.setattr(probe, "enqueue_train", lambda a, b, u: schedule(a, b, u, lambda: now[0], nap))
    output = tmp_path / "result.json"
    assert probe.measure(SimpleNamespace(device=0, repeat=4, unit_ms=2, json=output)) == 0
    assert not memory and not streams
    assert launches == [(60, short_ticks), (1, 20 * short_ticks)] * 8
    assert set(zip(expected_priorities[::2], expected_priorities[1::2])) == {(0, 7), (7, 0)}
    report = json.loads(output.read_text())
    assert report["status"] == "strict_priority_observed"
    assert {p["S1_high"]["created_first"] for p in report["pairs"]} == {"S1", "S2"}
    assert "S1 started" in capsys.readouterr().out
