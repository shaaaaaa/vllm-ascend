#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only snapshot worker, sent over SSH; requires only Python 3.9 stdlib."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FORMAT = "decode-kv-raw-v1"
ALGORITHM = "blake2b-128"
NUMERIC_CHUNK_BYTES = 1024 * 1024


def _values(data, dtype, byteorder):
    """Decode stored numbers, including BF16, without importing torch."""
    endian = "<" if byteorder == "little" else ">"
    codes = {
        "torch.int8": "b",
        "torch.uint8": "B",
        "torch.float16": "e",
        "torch.float32": "f",
        "torch.float64": "d",
        "torch.bfloat16": "H",
    }
    if dtype not in codes or byteorder not in ("little", "big"):
        raise ValueError("Unsupported numeric layout: " + dtype + "/" + byteorder)
    try:
        import numpy as np
    except ImportError:
        values = [x[0] for x in struct.iter_unpack(endian + codes[dtype], data)]
        if dtype == "torch.bfloat16":
            values = [struct.unpack("!f", struct.pack("!I", x << 16))[0] for x in values]
        return values
    if dtype == "torch.bfloat16":
        return (np.frombuffer(data, endian + "u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    types = {"b": "i1", "B": "u1", "e": "f2", "f": "f4", "d": "f8"}
    return np.frombuffer(data, endian + types[codes[dtype]]).astype(np.float64)


def _number(value):
    return float(value) if value is not None and math.isfinite(value) else None


class NumericStats:
    """Bounded-memory moments and structural diagnostics over all selected rows."""

    def __init__(self):
        self.elements = self.finite = self.nan = self.posinf = self.neginf = self.zeros = 0
        self.rows = self.zero_rows = self.adjacent_equal_rows = 0
        self.mean = self.m2 = 0.0
        self.minimum, self.maximum = math.inf, -math.inf
        self.first_row = self.last_row = None
        self.all_rows_equal = True

    def add(self, data, dtype, width, byteorder):
        values = _values(data, dtype, byteorder)
        rows = len(data) // width
        if len(data) % width or not rows or len(values) % rows:
            raise ValueError("Invalid numeric row width")
        columns = len(values) // rows
        if isinstance(values, list):
            finite = [x for x in values if math.isfinite(x)]
            zeros = sum(x == 0 for x in values)
            nan = sum(math.isnan(x) for x in values)
            posinf, neginf = values.count(math.inf), values.count(-math.inf)
            zero_rows = sum(all(x == 0 for x in values[i : i + columns]) for i in range(0, len(values), columns))
            mean = math.fsum(finite) / len(finite) if finite else 0.0
            m2 = math.fsum((x - mean) ** 2 for x in finite)
            low, high = (min(finite), max(finite)) if finite else (math.inf, -math.inf)
        else:
            import numpy as np

            finite = values[np.isfinite(values)]
            zeros, nan = int(np.count_nonzero(values == 0)), int(np.count_nonzero(np.isnan(values)))
            posinf, neginf = int(np.count_nonzero(values == np.inf)), int(np.count_nonzero(values == -np.inf))
            zero_rows = int(np.count_nonzero(np.all(values.reshape(rows, columns) == 0, axis=1)))
            mean = float(finite.mean()) if len(finite) else 0.0
            m2 = float(np.sum((finite - mean) ** 2))
            low, high = (float(finite.min()), float(finite.max())) if len(finite) else (math.inf, -math.inf)
        count = len(finite)
        if count:
            total = self.finite + count
            delta = mean - self.mean
            self.m2 += m2 + delta * delta * self.finite * count / total
            self.mean += delta * count / total
            self.finite = total
            self.minimum = min(self.minimum, low)
            self.maximum = max(self.maximum, high)
        self.elements += len(values)
        self.zeros += zeros
        self.nan += nan
        self.posinf += posinf
        self.neginf += neginf
        self.zero_rows += zero_rows
        self.rows += rows
        for offset in range(0, len(data), width):
            row = data[offset : offset + width]
            if self.first_row is None:
                self.first_row = row
            self.all_rows_equal = self.all_rows_equal and row == self.first_row
            self.adjacent_equal_rows += row == self.last_row
            self.last_row = row

    def result(self):
        std = math.sqrt(max(0.0, self.m2 / self.finite)) if self.finite else None
        rms = math.hypot(self.mean, std) if self.finite else None
        return dict(
            rows=self.rows,
            elements=self.elements,
            finite=self.finite,
            nan=self.nan,
            posinf=self.posinf,
            neginf=self.neginf,
            zero_values=self.zeros,
            zero_fraction=self.zeros / self.elements if self.elements else None,
            zero_rows=self.zero_rows,
            all_values_zero=bool(self.elements and self.zeros == self.elements),
            all_rows_equal=bool(self.rows and self.all_rows_equal),
            adjacent_equal_rows=self.adjacent_equal_rows,
            mean=_number(self.mean) if self.finite else None,
            std=_number(std),
            rms=_number(rms),
            min=_number(self.minimum),
            max=_number(self.maximum),
        )


def _range_stats(entry, row, start, end):
    result = NumericStats()
    width = row["row_bytes"]
    step = max(1, NUMERIC_CHUNK_BYTES // width)
    for offset in range(start, end, step):
        data = read_segment(entry, row["path"], offset, min(offset + step, end), width)
        result.add(data, row["dtype"], width, entry["meta"]["byteorder"])
    return dict(start=start, end=end, **result.result())


def _numeric_pair(left, right, arow, brow, start, end):
    count = new_nonfinite = changed_nonfinite = mismatched = 0
    error_square = reference_square = candidate_square = abs_sum = dot = max_abs = 0.0
    width = arow["row_bytes"]
    step = max(1, NUMERIC_CHUNK_BYTES // width)
    for offset in range(start, end, step):
        stop = min(offset + step, end)
        a = _values(read_segment(left, arow["path"], offset, stop, width), arow["dtype"], left["meta"]["byteorder"])
        b = _values(read_segment(right, brow["path"], offset, stop, width), brow["dtype"], right["meta"]["byteorder"])
        if isinstance(a, list):
            for x, y in zip(a, b):
                xf, yf = math.isfinite(x), math.isfinite(y)
                new_nonfinite += xf and not yf
                changed_nonfinite += (math.isnan(x), x == math.inf, x == -math.inf) != (
                    math.isnan(y),
                    y == math.inf,
                    y == -math.inf,
                )
                if not (xf and yf):
                    continue
                count += 1
                mismatched += x != y
                delta = abs(x - y)
                abs_sum += delta
                max_abs = max(max_abs, delta)
                error_square += delta * delta
                reference_square += x * x
                candidate_square += y * y
                dot += x * y
        else:
            import numpy as np

            af, bf = np.isfinite(a), np.isfinite(b)
            new_nonfinite += int(np.count_nonzero(af & ~bf))
            changed_nonfinite += int(
                np.count_nonzero(
                    (np.isnan(a) != np.isnan(b)) | ((a == np.inf) != (b == np.inf)) | ((a == -np.inf) != (b == -np.inf))
                )
            )
            av, bv = a[af & bf], b[af & bf]
            count += len(av)
            mismatched += int(np.count_nonzero(av != bv))
            if len(av):
                delta = np.abs(av - bv)
                abs_sum += float(delta.sum())
                max_abs = max(max_abs, float(delta.max()))
                error_square += float(np.sum(delta * delta))
                reference_square += float(np.sum(av * av))
                candidate_square += float(np.sum(bv * bv))
                dot += float(np.sum(av * bv))
    return dict(
        start=start,
        end=end,
        finite_pairs=count,
        finite_mismatches=mismatched,
        new_nonfinite=new_nonfinite,
        changed_nonfinite_pattern=changed_nonfinite,
        rmse=_number(math.sqrt(error_square / count)) if count else None,
        mean_abs_diff=_number(abs_sum / count) if count else None,
        max_abs_diff=_number(max_abs) if count else None,
        relative_l2=_number(math.sqrt(error_square / reference_square)) if reference_square else None,
        cosine=_number(dot / (math.sqrt(reference_square) * math.sqrt(candidate_square)))
        if reference_square and candidate_square
        else None,
    )


def inspect_numeric(payload):
    """Read only requested tails; exact numeric pairs stay on their shared host."""
    entries = payload["entries"]

    def inspect(job):
        entry, row = entries[job["entry"]], job["row"]
        start, common, end = job["start"], job["common"], job["end"]
        result = dict(
            id=job["id"], overlap=_range_stats(entry, row, start, common), new=_range_stats(entry, row, common, end)
        )
        if "previous" in job:
            previous = entries[job["previous"]]
            result["previous"] = _range_stats(previous, job["previous_row"], start, common)
            result["comparison"] = _numeric_pair(previous, entry, job["previous_row"], row, start, common)
        return result

    with ThreadPoolExecutor(max_workers=payload["workers"]) as pool:
        return list(pool.map(inspect, payload["jobs"]))


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
    return {
        "inventory": inventory,
        "checksums": checksums,
        "compare": exact_compare,
        "blocks": blocks,
        "inspect": inspect_numeric,
    }[payload["action"]](payload)


if __name__ == "__main__":
    request = json.loads(gzip.decompress(base64.b64decode(sys.argv[1])))
    response = dispatch(request)
    # Streaming gzip avoids creating an archive or payload copy on the D host.
    with gzip.GzipFile(fileobj=sys.stdout.buffer, mode="wb", compresslevel=1) as stream:
        stream.write(json.dumps(response, separators=(",", ":")).encode("utf-8"))
