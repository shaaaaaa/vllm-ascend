#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only snapshot worker, sent over SSH; requires only Python 3.9 stdlib."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FORMAT = "decode-kv-raw-v1"
ALGORITHM = "blake2b-128"


def digest(data):
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def request_key(value):
    match = re.search(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value)
    return match[0].lower() if match else value


def signature(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing/linked snapshot file: " + str(path))
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev]


def file_path(entry, name):
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Invalid snapshot filename: " + name)
    return Path(entry["directory"]) / name


def verify(entry, names=None):
    for name in names if names is not None else entry["stamps"]:
        if signature(file_path(entry, name)) != entry["stamps"][name]:
            raise ValueError("Snapshot changed during analysis: " + str(file_path(entry, name)))


def inventory(payload):
    workspace = Path(payload["workspace"])
    roots = (
        [Path(payload["snapshot_dir"])]
        if payload.get("snapshot_dir")
        else [workspace / "decode-kv-snapshots", workspace / "vllm-ascend" / "decode-kv-snapshots"]
    )
    entries, seen = [], set()
    for root in roots:
        for wanted in payload["requests"]:
            if not re.fullmatch(r"[a-zA-Z0-9_.-]+", wanted) or wanted in (".", ".."):
                raise ValueError("Invalid request ID")
            for manifest in sorted((root / wanted).glob("*/manifest.json")):
                directory = manifest.parent.resolve()
                if str(directory) in seen:
                    continue
                seen.add(str(directory))
                stamps = {name: signature(directory / name) for name in ("manifest.json", "tokens.json")}
                meta = json.loads(manifest.read_text(encoding="utf-8"))
                tokens = json.loads((directory / "tokens.json").read_text(encoding="utf-8"))
                if meta.get("format") != FORMAT or request_key(meta["request_id"]) != wanted:
                    raise ValueError("Invalid snapshot identity: " + str(manifest))
                if meta.get("status") != "complete":
                    raise ValueError("Snapshot is not complete: " + str(manifest))
                if len(tokens) != meta["tokens"]:
                    raise ValueError("Token metadata length mismatch: " + str(manifest))
                entry = dict(directory=str(directory), meta=meta, tokens=tokens, stamps=stamps)
                keys, names = set(), set()
                for row in meta["files"]:
                    key = (row["group"], row["layer"], row["component"])
                    if key in keys or row["path"] in names or row["row_bytes"] <= 0:
                        raise ValueError("Invalid/duplicate component: " + str(manifest))
                    keys.add(key)
                    names.add(row["path"])
                    if row["path"] in ("manifest.json", "tokens.json", "replay.json"):
                        raise ValueError("Invalid payload filename")
                    stat = signature(file_path(entry, row["path"]))
                    if (
                        row["start"] != 0
                        or row["end"] != len(tokens)
                        or row["bytes"] != len(tokens) * row["row_bytes"]
                        or stat[0] != row["bytes"]
                    ):
                        raise ValueError("Truncated/invalid payload: " + row["path"])
                    stamps[row["path"]] = stat
                expected = {(g, layer) for g, n in enumerate(meta["groups"]) for layer in range(n)}
                if not expected or {(g, layer) for g, layer, _ in keys} != expected:
                    raise ValueError("Missing layer in " + str(manifest))
                verify(entry)
                entries.append(entry)
    return entries


def read_segment(entry, name, start, end, width, expected=None):
    verify(entry, ["manifest.json", "tokens.json", name])
    if start < 0 or end <= start or end * width > entry["stamps"][name][0] or width <= 0:
        raise ValueError("Invalid segment range")
    with file_path(entry, name).open("rb") as source:
        source.seek(start * width)
        data = source.read((end - start) * width)
    if len(data) != (end - start) * width:
        raise ValueError("Truncated segment")
    verify(entry, [name])
    if expected is not None and digest(data) != expected:
        raise ValueError("Segment changed after checksum scan: " + name)
    return data


def checksums(payload):
    count, segment = payload["tokens"], payload["segment_tokens"]
    if count <= 0 or segment <= 0:
        raise ValueError("Invalid checksum range")

    def scan(job):
        entry, row = job
        name, width = row["path"], row["row_bytes"]
        if count > row["end"]:
            raise ValueError("Checksum range exceeds snapshot")
        verify(entry, ["manifest.json", "tokens.json", name])
        hashes = []
        with file_path(entry, name).open("rb") as source:
            for start in range(0, count, segment):
                size = min(segment, count - start) * width
                data = source.read(size)
                if len(data) != size:
                    raise ValueError("Truncated during checksum scan: " + name)
                hashes.append(digest(data))
        verify(entry, ["manifest.json", "tokens.json", name])
        return dict(directory=entry["directory"], path=name, hashes=hashes)

    jobs = [(entry, row) for entry in payload["entries"] for row in entry["meta"]["files"]]
    with ThreadPoolExecutor(max_workers=payload["workers"]) as pool:
        return list(pool.map(scan, jobs))


def byte_difference(left, right, start, width):
    """Exact counts, using numpy if available; stdlib fallback for SSH hosts."""
    if len(left) != len(right) or len(left) % width:
        raise ValueError("Incompatible segment lengths")
    first = []
    try:
        import numpy as np
    except ImportError:
        changed_bytes = changed_rows = 0
        for offset in range(0, len(left), width):
            a, b = left[offset : offset + width], right[offset : offset + width]
            if a == b:
                continue
            changed_rows += 1
            changed_bytes += sum(x != y for x, y in zip(a, b))
            if len(first) < 8:
                column = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
                first.append(
                    dict(token=start + offset // width, byte_in_component=column, previous=a[column], current=b[column])
                )
    else:
        delta = (np.frombuffer(left, np.uint8) != np.frombuffer(right, np.uint8)).reshape(-1, width)
        rows = np.flatnonzero(delta.any(axis=1))
        changed_bytes, changed_rows = int(np.count_nonzero(delta)), len(rows)
        for offset in rows[:8]:
            column = int(np.flatnonzero(delta[offset])[0])
            first.append(
                dict(
                    token=start + int(offset),
                    byte_in_component=column,
                    previous=left[int(offset) * width + column],
                    current=right[int(offset) * width + column],
                )
            )
    return dict(different_bytes=changed_bytes, different_tokens=changed_rows, first=first)


def exact_compare(payload):
    entries = payload["entries"]

    def compare(job):
        left = read_segment(
            entries[job["left"]], job["left_path"], job["start"], job["end"], job["width"], job["left_hash"]
        )
        right = read_segment(
            entries[job["right"]], job["right_path"], job["start"], job["end"], job["width"], job["right_hash"]
        )
        return dict(id=job["id"], **byte_difference(left, right, job["start"], job["width"]))

    with ThreadPoolExecutor(max_workers=payload["workers"]) as pool:
        return list(pool.map(compare, payload["jobs"]))


def blocks(payload):
    result = []
    for job in payload["jobs"]:
        data = read_segment(
            payload["entries"][job["entry"]], job["path"], job["start"], job["end"], job["width"], job["hash"]
        )
        result.append(dict(id=job["id"], data=base64.b64encode(data).decode("ascii")))
    return result


def dispatch(payload):
    if payload["action"] == "verify":
        for entry in payload["entries"]:
            verify(entry)
        return True
    return {"inventory": inventory, "checksums": checksums, "compare": exact_compare, "blocks": blocks}[
        payload["action"]
    ](payload)


if __name__ == "__main__":
    request = json.loads(gzip.decompress(base64.b64decode(sys.argv[1])))
    response = dispatch(request)
    # Streaming gzip avoids creating an archive or payload copy on the D host.
    with gzip.GzipFile(fileobj=sys.stdout.buffer, mode="wb", compresslevel=1) as stream:
        stream.write(json.dumps(response, separators=(",", ":")).encode("utf-8"))
