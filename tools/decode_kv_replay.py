#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay archived receive payloads through an already-running D service."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from decode_kv_compare import read_workers, request_key


def equal_payload(left: Path, right: Path) -> bool:
    """Verify the TP-shared latent source without hashing or loading whole files."""
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            first, second = a.read(4 * 1024 * 1024), b.read(4 * 1024 * 1024)
            if first != second:
                return False
            if not first:
                return True


def build_plan(
    roots: list[Path], request: str, output: Path, *, validate_bytes: bool = True, allow_legacy: bool = False
) -> dict:
    """Reject incomplete/incompatible captures before injecting snapshot bytes."""
    workers = read_workers(roots, request_key(request))
    plan = dict(schema=1, request_id=request_key(request), output=str(output.resolve()), workers={})
    reference = None
    modes, models = set(), set()
    for rank, (meta, tokens, rows) in workers.items():
        directory = next(iter(rows.values()))[1].parent
        if meta.get("replay_file") not in (None, "replay.json"):
            raise ValueError("Unsupported replay metadata filename")
        replay_path = directory / "replay.json"
        if meta.get("replay_file") == "replay.json" or replay_path.exists():
            modes.add("replay.json")
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            req, runtime = replay.get("request"), replay.get("runtime")
            if not req or not runtime or req.get("error") or runtime.get("error"):
                raise ValueError(f"Unusable replay metadata: {replay}")
            prompt = req["prompt_token_ids"]
            if req.get("schema") != 1 or not tokens or prompt[: len(tokens)] != tokens:
                raise ValueError("Received KV does not match the archived D prompt")
            if req["sampling"].get("n", 1) != 1:
                raise ValueError("Replay requires one output sequence")
            if reference is not None and replay != reference:
                raise ValueError("TP ranks disagree on request/runtime metadata")
            reference = replay
        else:
            if not allow_legacy:
                raise ValueError("Legacy capture needs the matching client request from --run-dir")
            modes.add("legacy_raw_snapshot")
            if not meta.get("model") or not tokens:
                raise ValueError("Legacy snapshot lacks model identity or received token IDs")
            models.add(meta["model"])
        if meta["byteorder"] != sys.byteorder or len(meta["groups"]) != 2:
            raise ValueError("Replay requires native-endian, two-group DSA snapshots")
        files = []
        for key, (row, path) in sorted(rows.items()):
            if row["row_bytes"] <= 0 or path.is_symlink():
                raise ValueError(f"Invalid payload: {path}")
            files.append(dict(row, absolute_path=str(path.resolve())))
        for group, count in enumerate(meta["groups"]):
            for layer in range(count):
                components = [row["component"] for row in files if row["group"] == group and row["layer"] == layer]
                if components != list(range(len(components))):
                    raise ValueError("Missing or non-contiguous KV components")
        plan["workers"][str(rank)] = dict(groups=meta["groups"], layer_names=meta["layer_names"], files=files)
    for rank, (_, _, rows) in workers.items():
        if rank == 0 or not validate_bytes:
            continue
        print(f"[D_KV_REPLAY] validating shared latent bytes: TP {rank} versus TP 0", flush=True)
        if rows.keys() != workers[0][2].keys():
            raise ValueError("TP ranks have different component inventories")
        for key, (row, path) in rows.items():
            baseline, baseline_path = workers[0][2][key]
            if (row["row_bytes"], row["dtype"]) != (baseline["row_bytes"], baseline["dtype"]):
                raise ValueError(f"TP rank {rank} component layout mismatch: {key}")
            # Production group 0 is rank0-owned and shared. Replaying divergent
            # copies as one shared source would erase evidence of corruption.
            if key[0] == 0 and not equal_payload(baseline_path, path):
                raise ValueError(
                    f"TP rank {rank} latent bytes differ from rank 0 at {key}; "
                    "cannot faithfully replay shared CPU source"
                )
    if len(modes) != 1 or len(models) > 1:
        raise ValueError("TP ranks disagree on replay metadata availability or source model")
    if reference is None:
        reference = dict(request=None, runtime=dict(engine_options=dict(model=models.pop())))
    plan.update(reference, replay_metadata_source=modes.pop())
    plan["received_tokens"] = next(iter(workers.values()))[1]
    return plan


def main(argv=None) -> int:
    from decode_kv_live_replay import main as run

    return run(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[D_KV_REPLAY] FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
