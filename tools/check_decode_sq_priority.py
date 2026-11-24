#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise physical queue priority in one two-stream graph on an idle 910B / CANN 9.1.0."""

import argparse
import ctypes as C
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

from check_decode_creation_priority import AbsWork, CreationRuntime
from check_decode_graph_priority import I32, U32, Ptr, Ref
from check_decode_queue_arbitration import BACKGROUND, ELEMENTS, FOREGROUND, ROLES, analyze, capture_case, read_rows

KERNELS = {FOREGROUND: 8, BACKGROUND: 16}
LEAD_IN = 2
MEASURED = 5
REPLAYS = MEASURED + 1  # Bind on the first replay. Check binding on every replay.


class PackageVersion(C.Structure):
    _fields_ = [
        ("version", C.c_char * 128),
        ("major", C.c_char * 64),
        ("minor", C.c_char * 64),
        ("release", C.c_char * 64),
        ("patch", C.c_char * 64),
        ("reserved", C.c_char * 128),
    ]


def require_target(rt):
    soc = (rt.functions["aclrtGetSocName"]() or b"").decode()
    pkg = PackageVersion()
    rc = rt.functions["aclsysGetCANNVersion"](1, C.byref(pkg))  # ACL_PKG_NAME_RUNTIME
    version = pkg.version.decode(errors="replace")
    match = rc == 0 and (pkg.major, pkg.minor, pkg.release) == (b"9", b"1", b"0")
    if not match:
        # Some installations relocate runtime/version.info. Check the loaded plugin itself.
        paths = [p.split(maxsplit=5)[-1] for p in Path("/proc/self/maps").read_text().splitlines()]
        plugin = next((p for p in paths if p.endswith("/libruntime_v100.so")), "")
        match = bool({"cann-9.1.0", "9.1.0"} & set(Path(plugin).parts)) and (rc != 0 or not version)
    if not soc.startswith("Ascend910B") or not match:
        raise RuntimeError(f"two-queue fixture requires 910B / CANN 9.1.0 (loaded {soc}, runtime={version!r}, rc={rc})")
    return soc


def binding(rt, fixture, logical):
    result = {}
    for role, stream in logical.items():
        sq, priority = U32(), U32()
        rt.call("rtStreamGetSqid", stream, C.byref(sq))
        if fixture.sq_test_lookup(sq, C.byref(priority)):
            raise RuntimeError(
                f"{role} bound SQ {sq.value} has no matching accepted driver allocation "
                f"(observed {fixture.sq_test_attempts()} capture-pool allocations)"
            )
        result[role] = {"sq_id": sq.value, "priority": priority.value}
    if len({r["sq_id"] for r in result.values()}) != 2 or {r["priority"] for r in result.values()} != {0, 7}:
        raise RuntimeError(f"expected two distinct bound SQs with opposite allocation priorities, got {result}")
    return result


def worker(device, first, library, report):
    fixture = C.CDLL(str(library), mode=C.RTLD_GLOBAL)
    hal = C.CDLL("libascend_hal.so", mode=C.RTLD_GLOBAL)
    fixture.sq_test_arm.argtypes = [Ptr, U32, U32]
    fixture.sq_test_lookup.argtypes = [U32, Ref(U32)]
    fixture.sq_test_attempts.argtypes, fixture.sq_test_attempts.restype = [], U32
    rt = CreationRuntime(capture_functions=True)
    rt.libs.append(C.CDLL("libruntime.so", mode=C.RTLD_GLOBAL))
    rt.bind(
        {
            "rtStreamGetSqid": [Ptr, Ref(U32)],
            "aclsysGetCANNVersion": [I32, Ref(PackageVersion)],
            "aclrtGetLogicDevIdByUserDevId": [I32, Ref(I32)],
        }
    )
    with ExitStack() as stack:
        rt.call("aclInit", None)
        stack.callback(rt.call, "aclFinalize")
        rt.call("aclrtSetDevice", device)
        stack.callback(rt.call, "aclrtResetDevice", device)
        soc = require_target(rt)
        execution = rt.manage(stack, "aclrtCreateStream", "aclrtDestroyStream")
        host = rt.manage(stack, "aclrtMallocHost", "aclrtFreeHost", 32)
        work = {r: AbsWork(rt, stack, byte, execution, ELEMENTS[r]) for r, byte in zip(ROLES, (0xBC, 0xC0))}
        logic = I32()
        rt.call("aclrtGetLogicDevIdByUserDevId", device, C.byref(logic))
        if fixture.sq_test_arm(C.cast(hal.halSqCqAllocate, Ptr), logic.value, first):
            raise RuntimeError("test fixture could not attach to the real HAL allocator")
        bindings = []
        case = capture_case(
            rt,
            "capture",
            execution,
            work,
            host,
            FOREGROUND,
            dict.fromkeys(ROLES, 0),
            kernels=KERNELS,
            lead_in=LEAD_IN,
            replays=REPLAYS,
            on_bound=lambda streams: bindings.append(binding(rt, fixture, streams)),
        )
        if any(b != bindings[0] for b in bindings[1:]):
            raise RuntimeError(f"queue binding changed between replays: {bindings}")
        actual = bindings[0]
        case.update(case="foreground_high" if actual[FOREGROUND]["priority"] == 0 else "background_high", bound=actual)
        result = {"soc": soc, "cases": [case], "warmup_stream": rt.stream_id(execution)}
    report.write_text(json.dumps(result), encoding="utf-8")


def verdict(results):
    by_case = {r["case"]: r for r in results}
    if len(results) != 2 or set(by_case) != {"foreground_high", "background_high"}:
        return "priority_swap_not_verified"
    high, reverse = by_case["foreground_high"], by_case["background_high"]
    if high["gaps_with_background_ahead"]:
        return "foreground_overtaken"
    if high["eligible_replays"] != MEASURED:
        return "insufficient_contention"
    if reverse["overtaken_replays"] > MEASURED // 2:
        return "ready_priority_observed"
    if reverse["background_tail_started_before_foreground_replays"] > MEASURED // 2:
        return "priority_changed_entry_order"
    if reverse["eligible_replays"] <= MEASURED // 2:
        return "insufficient_contention"
    return "no_directional_priority_effect"


def profile_case(device, first, library, directory):
    directory.mkdir()
    report = directory / "queues.json"
    command = [
        "msprof",
        f"--output={directory}",
        "sh",
        str(library.with_name("launch.sh")),
        str(library),
        sys.executable,
        str(Path(__file__).resolve()),
        "--device",
        str(device),
        "--worker",
        str(first),
        "--fixture",
        str(library),
        "--report",
        str(report),
    ]
    logfile = directory / "msprof.log"
    with logfile.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode or not report.is_file():
        lines = logfile.read_text(errors="replace").splitlines()
        detail = next((line for line in reversed(lines) if "[SQ_TEST] error" in line), "")
        raise RuntimeError(f"{detail or 'MindStudio worker failed'} (log: {logfile})")
    csvs = list(directory.glob("PROF_*/mindstudio_profiler_output/op_summary*.csv"))
    if len(csvs) != 1:
        raise RuntimeError(f"expected one MindStudio op_summary in {directory}, found {len(csvs)} (log: {logfile})")
    (case,) = analyze(
        read_rows(csvs[0]), json.loads(report.read_text()), kernels=KERNELS, lead_in=LEAD_IN, replays=REPLAYS
    )
    return case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", type=Path, help="new directory for both short profiles and the result")
    p.add_argument("--worker", type=int, choices=(0, 7), help=argparse.SUPPRESS)
    p.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--report", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker is not None:
        worker(args.device, args.worker, args.fixture, args.report)
        return 0
    cc = shutil.which("cc") or shutil.which("gcc")
    if sys.platform != "linux" or not cc or not shutil.which("msprof"):
        p.error("run in the Linux CANN 9.1.0 container with cc or gcc and msprof on PATH")
    if args.output:
        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(tempfile.mkdtemp(prefix="sq-priority-unit-", dir=".")).resolve()
    print(f"[SQ_TEST] output {root}", flush=True)
    cases = []
    with tempfile.TemporaryDirectory(prefix="sq-fixture-", dir=root) as tmp:
        lib = Path(tmp) / "sq_fixture.so"
        # Keep each msprof argument simple. Preload only in the child and preserve MindStudio's own preload list.
        lib.with_name("launch.sh").write_text(
            '#!/bin/sh\nexport LD_PRELOAD="$1${LD_PRELOAD:+:$LD_PRELOAD}"\nshift\nexec "$@"\n',
            encoding="utf-8",
        )
        subprocess.run(
            [
                cc,
                "-std=c11",
                "-O2",
                "-fPIC",
                "-shared",
                str(Path(__file__).with_name("decode_sq_priority_fixture.c")),
                "-o",
                str(lib),
                "-ldl",
            ],
            check=True,
        )
        for first in (0, 7):
            case = profile_case(args.device, first, lib, root / f"first-{first}")
            cases.append(case)
            tail = case["background_tail_started_before_foreground_replays"]
            print(
                f"[SQ_TEST] {case['case']} {case['bound']}  new background starts in "
                f"{case['gaps_with_background_ahead']}/{case['eligible_gaps']} ready gaps, "
                f"background tail before first foreground on {tail}/{MEASURED} replays",
                flush=True,
            )
    status = verdict(cases)
    (root / "summary.json").write_text(json.dumps({"status": status, "cases": cases}, indent=2), encoding="utf-8")
    print(f"[SQ_TEST] {status}")
    return 0 if status == "ready_priority_observed" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        sys.exit(f"[SQ_TEST] error: {exc}")
