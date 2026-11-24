# SPDX-License-Identifier: Apache-2.0
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    import check_decode_sq_priority

    return check_decode_sq_priority


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    work = tmp_path_factory.mktemp("sq-fixture")
    if os.name == "nt":
        zig, node = shutil.which("zig"), shutil.which("node")
        if not zig or not node:
            pytest.skip("Zig and Node are required for the portable CPU harness on Windows")
        program = work / "test.wasm"
        command, flags = [zig, "cc", "-target", "wasm32-wasi"], []
        runner = work / "run.cjs"
        runner.write_text(
            "const {WASI} = require('node:wasi');\n"
            "const args = process.argv.slice(2);\n"
            "const wasi = new WASI({version: 'preview1', args});\n"
            "const mod = new WebAssembly.Module(require('node:fs').readFileSync(args[0]));\n"
            "wasi.start(new WebAssembly.Instance(mod, wasi.getImportObject()));\n"
        )
        invoke = [node, str(runner), str(program)]
    else:
        cc = shutil.which("cc") or shutil.which("gcc")
        if not cc:
            pytest.skip("a C compiler is required for the CPU HAL harness")
        program = work / "test"
        command, flags, invoke = [cc], ["-ldl"], [str(program)]
    subprocess.run(
        [
            *command,
            "-std=c11",
            str(TOOLS / "decode_sq_priority_fixture.c"),
            str(Path(__file__).with_name("decode_sq_priority_mock.c")),
            "-o",
            str(program),
            *flags,
        ],
        check=True,
    )
    return invoke


@pytest.mark.parametrize(("first", "fail_first"), [(0, 0), (7, 0), (7, 1)])
def test_driver_request_scope_returned_sq_and_failure(compiled, first, fail_first):
    subprocess.run([*compiled, str(first), str(fail_first)], check=True)


@pytest.mark.parametrize(
    ("version", "rc", "soc", "supported"),
    [
        (b"9.1.0", 0, b"Ascend910B3", True),
        (b"", 1, b"Ascend910B3", True),
        (b"9.2.0", 0, b"Ascend910B3", False),
        (b"9.1.0", 0, b"Ascend910_9391", False),
    ],
)
def test_check_loaded_runtime_before_changing_allocations(probe, monkeypatch, version, rc, soc, supported):
    def package(kind, result):
        assert kind == 1
        result._obj.version = version
        if version:
            result._obj.major, result._obj.minor, result._obj.release = version.split(b".")
        return rc

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: "1000-2000 r-xp 0 00:00 10 /usr/local/Ascend/cann-9.1.0/lib64/libruntime_v100.so\n",
    )
    rt = SimpleNamespace(functions={"aclrtGetSocName": lambda: soc, "aclsysGetCANNVersion": package})
    if supported:
        assert probe.require_target(rt) == "Ascend910B3"
    else:
        with pytest.raises(RuntimeError, match="requires 910B"):
            probe.require_target(rt)


def test_binding_joins_by_physical_sq_id(probe):
    role_streams = {probe.FOREGROUND: 23, probe.BACKGROUND: 27}
    bound = {23: 94, 27: 82}
    accepted = {82: 0, 94: 7}

    def get_sq(name, stream, out):
        assert name == "rtStreamGetSqid"
        out._obj.value = bound[stream]

    def lookup(sq, out):
        if sq.value not in accepted:
            return -1
        out._obj.value = accepted[sq.value]
        return 0

    rt = SimpleNamespace(call=get_sq)
    fixture = SimpleNamespace(sq_test_lookup=lookup, sq_test_attempts=lambda: 2)
    assert probe.binding(rt, fixture, role_streams) == {
        probe.FOREGROUND: {"sq_id": 94, "priority": 7},
        probe.BACKGROUND: {"sq_id": 82, "priority": 0},
    }
    bound[23] = 82
    with pytest.raises(RuntimeError, match="distinct"):
        probe.binding(rt, fixture, role_streams)
    bound[23] = 999
    with pytest.raises(RuntimeError, match="accepted driver allocation"):
        probe.binding(rt, fixture, role_streams)


def test_short_profile_and_reversal(probe, monkeypatch, tmp_path):
    def mock_msprof(cmd, **kwargs):
        root = Path(cmd[1].removeprefix("--output="))
        first = int(cmd[cmd.index("--worker") + 1])
        fg, bg = probe.ROLES
        streams = {fg: 14, bg: 15}
        case = {
            "case": "foreground_high" if first == 0 else "background_high",
            "source_ids": {fg: 2, bg: 3},
            "logical_ids": streams,
            "bound": {fg: {"sq_id": 11, "priority": first}, bg: {"sq_id": 12, "priority": 7 - first}},
        }
        (root / "queues.json").write_text(json.dumps({"cases": [case], "warmup_stream": 5}))
        out = root / "PROF_0" / "mindstudio_profiler_output"
        out.mkdir(parents=True)
        path = out / "op_summary_0.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Task Type",
                    "OP Type",
                    "Task Start Time(us)",
                    "Task Duration(us)",
                    "Input Shapes",
                    "Block Num",
                    "Stream ID",
                ]
            )

            def row(role, stream, t):
                writer.writerow(["AI_VECTOR_CORE", "Abs", t, 2, probe.ELEMENTS[role], 32, stream])

            for index, role in enumerate(probe.ROLES):
                row(role, 5, index * 3)
            for i in range(probe.REPLAYS):
                t = 50 + i * 160
                for j in range(probe.LEAD_IN):
                    row(bg, streams[bg], t + 3 * j)
                for j in range(probe.KERNELS[fg]):
                    row(fg, streams[fg], t + 9 + (7 if first == 7 else 2) * j)
                for j in range(probe.KERNELS[bg] - probe.LEAD_IN):
                    row(bg, streams[bg], t + (12 + 7 * j if first == 7 else 35 + 3 * j))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_msprof)
    cases = [probe.profile_case(0, first, tmp_path / "local.so", tmp_path / str(first)) for first in (0, 7)]
    assert probe.verdict(cases) == "ready_priority_observed"
    assert probe.verdict(cases[:1] * 2) == "priority_swap_not_verified"
    high, reverse = cases
    high["gaps_with_background_ahead"] = 1
    assert probe.verdict(cases) == "foreground_overtaken"
    high["gaps_with_background_ahead"] = 0
    reverse["overtaken_replays"] = 0
    assert probe.verdict(cases) == "no_directional_priority_effect"
    reverse["eligible_replays"] = 0
    assert probe.verdict(cases) == "insufficient_contention"
    reverse["background_tail_started_before_foreground_replays"] = probe.MEASURED
    assert probe.verdict(cases) == "priority_changed_entry_order"
