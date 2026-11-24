# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    import check_decode_queue_arbitration

    return check_decode_queue_arbitration


@pytest.mark.parametrize("mode", ["capture", "explicit"])
@pytest.mark.parametrize("root", ["foreground", "background"])
@pytest.mark.parametrize("short", [False, True], ids=["original", "two-sq-unit"])
def test_device_gate_replay_and_graph_stream_fields(probe, mode, root, short):
    ids, programs, closed, counts, priorities = [], defaultdict(list), [], defaultdict(int), {}
    graph, offset = 999, 100
    fg, bg = probe.ROLES
    kernels, lead, replays = probe.KERNELS, probe.LEAD_IN, probe.REPLAYS
    if short:
        import check_decode_sq_priority as unit

        kernels, lead, replays = unit.KERNELS, unit.LEAD_IN, unit.REPLAYS
    bindings = []

    class Runtime:
        @staticmethod
        def manage(stack, create, destroy, *args):
            value = graph if create == "aclmdlRIBuildBegin" else len(ids) + 1
            ids.append(value)
            stack.callback(closed.append, (destroy, value))
            if create == "aclrtCreateStreamWithConfig":
                priorities[value] = args[0]
            return value

        @staticmethod
        def call(name, *args):
            if name in ("aclrtRecordNotify", "aclrtRecordEvent"):
                programs[args[1]].append(("signal", args[0]))
            elif name == "aclrtWaitAndResetNotify":
                programs[args[1]].append(("wait", args[0]))
            elif name == "aclrtStreamWaitEvent":
                programs[args[0]].append(("wait", args[1]))
            elif name == "aclmdlRICaptureEnd":
                args[1]._obj.value = graph
            elif name == "aclmdlRICaptureGetInfo":
                args[1]._obj.value, args[2]._obj.value = 1, graph
            elif name == "aclrtStreamGetPriority":
                # The stock capture allocator creates two priority-zero graph streams.
                args[1]._obj.value = priorities[args[0]] if mode == "explicit" else 0
            elif name == "aclrtMemset":
                counts[args[0]] = 0
            elif name == "aclmdlRIExecuteAsync":
                pos, signals = defaultdict(int), set()
                while any(pos[s] < len(p) for s, p in programs.items()):
                    ready = [
                        s
                        for s, tasks in programs.items()
                        if pos[s] < len(tasks) and (tasks[pos[s]][0] != "wait" or tasks[pos[s]][1] in signals)
                    ]
                    assert ready, "graph deadlock"
                    s = min(ready, key=priorities.get)
                    op, arg = programs[s][pos[s]]
                    if op == "kernel":
                        if arg == fg:
                            assert counts[bg] >= lead
                        counts[arg] += 1
                    elif op == "signal":
                        signals.add(arg)
                    pos[s] += 1
            elif name not in {
                "aclmdlRIBindStream",
                "aclmdlRICaptureBegin",
                "aclmdlRIEndTask",
                "aclmdlRIBuildEnd",
                "aclrtSynchronizeStreamWithTimeout",
                "aclmdlRIUnbindStream",
                "aclmdlRIDestroy",
            }:
                raise AssertionError(name)

        def stream_id(self, s):
            return s

        def streams(self, g):
            assert g == graph
            return {s + offset for s in programs}

    def make_work(role):
        def check(host):
            assert counts[role] == kernels[role]

        return SimpleNamespace(
            output=role, nbytes=32, check=check, enqueue=lambda stream: programs[stream].append(("kernel", role))
        )

    case = probe.capture_case(
        Runtime(),
        mode,
        42,
        {r: make_work(r) for r in probe.ROLES},
        22,
        root,
        {fg: 0, bg: 7},
        kernels=kernels,
        lead_in=lead,
        replays=replays,
        on_bound=lambda streams: bindings.append((streams.copy(), counts.copy())),
    )
    assert case["source_ids"][root] == 1
    assert priorities == {case["source_ids"][fg]: 0, case["source_ids"][bg]: 7}
    assert case["graph_stream_creation_fields"] == {fg: 0, bg: 7 if mode == "explicit" else 0}
    assert bindings == [(case["logical_ids"], kernels)] * replays
    assert len(closed) == len(ids)


def test_counts_only_new_background_work_after_foreground_readiness(probe):
    fg = [(10, 16), (25, 32), (34, 40)]
    bg = [(0, 1), (2, 3), (4, 5), (6, 7), (12, 19), (20, 23), (25, 30), (36, 37)]
    gaps, ahead, initial = probe.replay_gaps(fg, bg)
    assert (gaps, ahead, initial) == ([9, 2], [1, 0], 3)


def test_profile_alignment_controls_and_strict_verdict(probe):
    fg, bg = probe.ROLES
    roles = {fg: 70, bg: 71}
    report = {"cases": [], "warmup_stream": 60}
    rows = [(0, 1, fg, 60), (2, 3, bg, 60)]
    clock = 100
    for root in probe.ROLES:
        for name in ("equal", "foreground_high", "background_high"):
            report["cases"].append({"root": root, "case": name, "source_ids": roles, "logical_ids": roles})
            for replay in range(probe.REPLAYS):
                observed = []
                pos = clock
                late = name == "background_high" or replay == 0
                for i in range(probe.KERNELS[bg]):
                    if probe.LEAD_IN <= i < probe.LEAD_IN + probe.KERNELS[fg]:
                        step = (pos + 2, pos + 8, fg, roles[fg])
                        observed.append(step)
                    observed.append((pos, pos + 1, bg, roles[bg]))
                    pos += 10 if late else 6
                if not late:
                    # Every FG successor starts as soon as its predecessor completes.
                    first = observed[probe.LEAD_IN][0]
                    observed = [row for row in observed if row[2] != fg] + [
                        (first + 6 * i, first + 6 * (i + 1), fg, roles[fg]) for i in range(probe.KERNELS[fg])
                    ]
                rows += sorted(observed)
                clock = pos + 100
    cases = probe.analyze(rows, report)
    assert probe.verdict(cases) == "foreground_preferred_in_both_orders"
    assert all(c["eligible_replays"] == probe.REPLAYS - 1 for c in cases)
    cases[1]["gaps_with_background_ahead"] = 1
    assert probe.verdict(cases) == "foreground_overtaken"
    cases[1]["gaps_with_background_ahead"] = 0
    cases[1]["eligible_replays"] = 0
    assert probe.verdict(cases) == "insufficient_contention"
    cases[1]["eligible_replays"] = probe.REPLAYS - 1
    for c in cases:
        if c["case"] == "background_high":
            c["overtaken_replays"] = 0
            c["eligible_replays"] = 0
            c["background_tail_started_before_foreground_replays"] = probe.REPLAYS - 1
    assert probe.verdict(cases) == "priority_reversal_delays_foreground_entry_in_both_orders"
    with pytest.raises(ValueError, match="extra"):
        probe.analyze(rows + rows[-1:], report)
