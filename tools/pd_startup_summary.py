#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize existing PD startup logs without importing a device runtime."""

import argparse
import re
from collections import Counter
from pathlib import Path

PD = re.compile(
    r"\[PD_INIT\] h=(?P<host>[\w.:-]+) p=(?P<pid>\d+) "
    r"d=(?P<dp>\d+|\?) t=(?P<tp>\d+|\?)(?: g=(?P<global>\d+))? "
    r"(?P<stage>[\w.:-]+) (?P<state>begin|end|error)\b"
)
LM = re.compile(
    r"\[LMCACHE_INIT\] host=(?P<host>[\w.:-]+) pid=(?P<pid>\d+) "
    r"(?:rank=(?P<rank>\d+) )?stage=(?P<stage>[\w.:-]+) state=(?P<state>begin|end|error)\b"
)
ERROR = re.compile(r"\b(?:exc|error)=(\w+)")
MAX_ROWS = 390


def summarize(lines):
    """Track nested phases by host/PID; retain the first, innermost failure."""
    workers = {}
    for line in lines:
        match = PD.search(line)
        source = "PD"
        if match is None:
            match, source = LM.search(line), "LM"
        if match is None:
            continue
        values = match.groupdict()
        # Both producers retain the same hostname prefix. Do not merge distinct
        # hosts in different DNS domains merely because their short names match.
        key = (values["host"], int(values["pid"]))
        worker = workers.setdefault(key, {"open": [], "last": "?", "error": None, "ready": False})
        for name in ("dp", "tp", "global", "rank"):
            if values.get(name) not in (None, "?"):
                worker[name] = values[name]
        phase, state = f"{source}.{values['stage']}", values["state"]
        worker["last"] = phase
        if state == "begin":
            worker["open"].append(phase)
        else:
            for index in range(len(worker["open"]) - 1, -1, -1):
                if worker["open"][index] == phase:
                    del worker["open"][index]
                    break
            if state == "error" and worker["error"] is None:
                error = ERROR.search(line[match.end() :])
                worker["error"] = (phase, error[1] if error else "unknown")
            if phase == "PD.warmup" and state == "end":
                worker["ready"] = True
    for worker in workers.values():
        worker["status"] = (
            "failed"
            if worker["error"]
            else "waiting"
            if worker["open"]
            else "complete"
            if worker["ready"]
            else "partial"
        )
    return workers


def render(workers):
    """Return bounded lines containing only recognized identifiers and phases."""
    if not workers:
        return ["No valid [PD_INIT]/[LMCACHE_INIT] records found."]
    counts = Counter(worker["status"] for worker in workers.values())
    result = [
        "workers="
        + str(len(workers))
        + " "
        + " ".join(f"{name}={counts[name]}" for name in ("waiting", "complete", "failed", "partial"))
    ]
    for (host, pid), worker in sorted(workers.items())[:MAX_ROWS]:
        identity = f"h={host[:24]} p={pid}"
        if "dp" in worker or "tp" in worker:
            identity += f" d={worker.get('dp', '?')} t={worker.get('tp', '?')}"
        elif "rank" in worker:
            identity += f" r={worker['rank']}"
        if "global" in worker:
            identity += f" g={worker['global']}"
        if worker["error"]:
            phase, error = worker["error"]
            detail = f" err={phase[:40]}:{error[:32]}"
        elif worker["open"]:
            detail = f" open={worker['open'][-1][:40]}"
        else:
            detail = f" last={worker['last'][:40]}"
        result.append(f"{identity} {worker['status']}{detail}")
    if len(workers) > MAX_ROWS:
        result.append(f"{len(workers) - MAX_ROWS} additional workers omitted.")
    result.append("complete=warmup ended; partial=only subphases ended; open phases do not establish root cause.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Existing log files; read-only")
    args = parser.parse_args(argv)

    def lines():
        for path in args.logs:
            with path.open(encoding="utf-8", errors="replace") as stream:
                yield from stream

    try:
        output = render(summarize(lines()))
    except OSError as error:
        parser.error(f"Cannot read input log: {type(error).__name__}")
    print("\n".join(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
