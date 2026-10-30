# SPDX-License-Identifier: Apache-2.0
"""Bandwidth-bounded orchestration for offline D snapshot comparison."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import decode_kv_remote as worker
from pd_tensor_collect import remote_command, run_ssh

RPC_RAW_BYTES = 8 * 1024 * 1024
BOOTSTRAP = (
    "import sys,base64,gzip;source=gzip.decompress(base64.b64decode(sys.argv[1]));"
    "sys.argv=[sys.argv[0],sys.argv[2]];exec(compile(source,'decode_kv_remote','exec'))"
)


def pack(value):
    return gzip.compress(json.dumps(value, separators=(",", ":"), sort_keys=True).encode(), compresslevel=1, mtime=0)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pack(value))
    temporary.replace(path)


class RemoteClient:
    def __init__(self, args):
        self.args = args
        source = Path(worker.__file__).read_bytes()
        self.source = base64.b64encode(gzip.compress(source)).decode("ascii")
        self.received_bytes = 0
        self.lock = threading.Lock()

    def call(self, host, payload):
        encoded = base64.b64encode(pack(payload)).decode("ascii")
        command = remote_command(self.args, host, ["python3", "-c", BOOTSTRAP, self.source, encoded])
        result = run_ssh(self.args, host, command, stdout=-1, stderr=-1)
        if result.returncode:
            raise RuntimeError(f"{host}: {result.stderr.decode(errors='replace')[-1800:]}")
        with self.lock:
            self.received_bytes += len(result.stdout)
        return json.loads(gzip.decompress(result.stdout))


def index_workers(entries, request):
    found = {}
    for entry in entries.values():
        meta = entry["meta"]
        if worker.request_key(meta["request_id"]) != request:
            continue
        rank = meta["tp_rank"]
        if rank in found:
            raise ValueError(f"Multiple captures for TP {rank}, request {request}")
        found[rank] = entry
    sizes = {e["meta"]["tp_size"] for e in found.values()}
    if len(sizes) != 1 or set(found) != set(range(next(iter(sizes)))):
        raise ValueError(f"Incomplete TP ranks for request {request}: {sorted(found)}")
    first = next(iter(found.values()))["tokens"]
    if any(e["tokens"] != first for e in found.values()):
        raise ValueError(f"TP ranks disagree on received tokens: {request}")
    return found


def slim_entries(entries, jobs):
    """Send only file identities needed by this RPC, not the 500k token lists."""
    names = defaultdict(lambda: {"manifest.json", "tokens.json"})
    for job in jobs:
        for side in ("left", "right") if "left" in job else ("entry",):
            names[job[side]].add(job[f"{side}_path"] if side != "entry" else job["path"])
    return {
        key: dict(directory=entries[key]["directory"], stamps={name: entries[key]["stamps"][name] for name in paths})
        for key, paths in names.items()
    }


def checksum_entry(client, args, entry, common):
    identity = dict(
        host=entry["host"],
        directory=entry["directory"],
        stamps=entry["stamps"],
        tokens=common,
        segment_tokens=args.segment_tokens,
        algorithm=worker.ALGORITHM,
    )
    key = hashlib.sha256(pack(identity)).hexdigest()
    path = args.output / "checksums" / f"{key}.json.gz"
    if path.is_file():
        saved = json.loads(gzip.decompress(path.read_bytes()))
        if saved["identity"] != identity:
            raise ValueError("Checksum cache identity mismatch")
        rows = saved["rows"]
        print(f"[D_KV_COMPARE] reuse checksums {entry['host']} {Path(entry['directory']).name}", flush=True)
    else:
        print(f"[D_KV_COMPARE] scanning {entry['host']} {Path(entry['directory']).name}", flush=True)
        request = {k: entry[k] for k in ("directory", "meta", "stamps")}
        rows = client.call(
            entry["host"],
            dict(
                action="checksums",
                entries=[request],
                tokens=common,
                segment_tokens=args.segment_tokens,
                workers=args.workers,
            ),
        )
        write_json(path, dict(identity=identity, rows=rows))
        print(f"[D_KV_COMPARE] scanned {entry['host']} {Path(entry['directory']).name}", flush=True)
    result = {r["path"]: r["hashes"] for r in rows}
    if len(result) != len(rows) or set(result) != {r["path"] for r in entry["meta"]["files"]}:
        raise ValueError("Checksum component coverage mismatch")
    count = (common + args.segment_tokens - 1) // args.segment_tokens
    if any(len(hashes) != count or any(len(h) != 32 for h in hashes) for hashes in result.values()):
        raise ValueError("Checksum segment coverage mismatch")
    return result


def fetch_blocks(client, args, entries, jobs):
    """Fetch only selected mismatches, reusing verified local blocks on retry."""
    data, pending = {}, defaultdict(list)
    reused = 0
    for job in jobs:
        for side in ("left", "right"):
            entry = entries[job[side]]
            name = job[f"{side}_path"]
            task = dict(
                id=f"{job['id']}-{side}",
                entry=job[side],
                path=name,
                start=job["start"],
                end=job["end"],
                width=job["width"],
                hash=job[f"{side}_hash"],
            )
            identity = [
                entry["host"],
                entry["directory"],
                name,
                entry["stamps"][name],
                task["start"],
                task["end"],
                task["hash"],
            ]
            path = args.output / "diff-blocks" / (hashlib.sha256(pack(identity)).hexdigest() + ".bin")
            size = (task["end"] - task["start"]) * task["width"]
            if path.is_file():
                block = path.read_bytes()
                if len(block) == size and worker.digest(block) == task["hash"]:
                    data[task["id"]] = block
                    reused += size
                    continue
            pending[entry["host"]].append((task, path, size))

    def download(host_tasks):
        host, tasks = host_tasks
        received = {}
        while tasks:
            batch, amount = [], 0
            while tasks and (not batch or amount + tasks[0][2] <= RPC_RAW_BYTES):
                item, tasks = tasks[0], tasks[1:]
                batch.append(item)
                amount += item[2]
            request_jobs = [item[0] for item in batch]
            result = client.call(
                host, dict(action="blocks", entries=slim_entries(entries, request_jobs), jobs=request_jobs)
            )
            expected = {t["id"]: (t, path, size) for t, path, size in batch}
            if len(result) != len(expected) or {r["id"] for r in result} != set(expected):
                raise ValueError("Incomplete raw block response")
            for row in result:
                task, path, size = expected[row["id"]]
                block = base64.b64decode(row["data"], validate=True)
                if len(block) != size or worker.digest(block) != task["hash"]:
                    raise ValueError("Downloaded block checksum mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(block)
                temporary.replace(path)
                received[row["id"]] = block
        return received

    with ThreadPoolExecutor(max_workers=max(1, min(4, len(pending)))) as pool:
        for received in pool.map(download, pending.items()):
            data.update(received)
    return data, reused


def analyze_remote(args, client=None):
    client = client or RemoteClient(args)
    entries = {}
    for host in dict.fromkeys(args.hosts):
        print(f"[D_KV_COMPARE] inventory {host}", flush=True)
        result = client.call(
            host,
            dict(
                action="inventory",
                workspace=args.workspace,
                snapshot_dir=args.snapshot_dir,
                requests=[args.previous, args.current],
            ),
        )
        for entry in result:
            entry["host"], entry["id"] = host, f"e{len(entries)}"
            entries[entry["id"]] = entry
    write_json(args.output / "inventory.json.gz", entries)
    before, after = index_workers(entries, args.previous), index_workers(entries, args.current)
    if set(before) != set(after):
        raise ValueError("TP sizes differ between rounds")
    a_tokens = next(iter(before.values()))["tokens"]
    b_tokens = next(iter(after.values()))["tokens"]
    common = next((i for i, (a, b) in enumerate(zip(a_tokens, b_tokens)) if a != b), min(len(a_tokens), len(b_tokens)))
    if not common:
        raise ValueError("Requests have no common token prefix")
    pairs = {}
    for rank in before:
        for field in ("groups", "byteorder", "layer_names", "model"):
            if before[rank]["meta"].get(field) != after[rank]["meta"].get(field):
                raise ValueError(f"TP {rank}: incompatible {field}")
        ar, br = (
            {(r["group"], r["layer"], r["component"]): r for r in e["meta"]["files"]}
            for e in (before[rank], after[rank])
        )
        if set(ar) != set(br):
            raise ValueError(f"TP {rank}: component sets differ")
        for key in ar:
            if any(ar[key][k] != br[key][k] for k in ("row_bytes", "dtype")):
                raise ValueError(f"TP {rank} {key}: incompatible layout")
        pairs[rank] = ar, br

    with ThreadPoolExecutor(max_workers=min(4, len(entries))) as pool:
        checksums = dict(zip(entries, pool.map(lambda e: checksum_entry(client, args, e, common), entries.values())))
    jobs, layers = [], []
    compared_bytes = total_segments = matching_segments = 0
    for rank in sorted(before):
        a, b = before[rank], after[rank]
        ar, br = pairs[rank]
        for key in sorted(ar):
            left, right = ar[key], br[key]
            ah, bh = checksums[a["id"]][left["path"]], checksums[b["id"]][right["path"]]
            layer = dict(
                tp_rank=rank,
                group=key[0],
                layer=key[1],
                component=key[2],
                compared_tokens=common,
                segments=len(ah),
                different_segments=0,
                exact_segments=0,
                confirmed_different_bytes=0,
                confirmed_different_tokens=0,
                first=[],
            )
            layer_id = len(layers)
            layers.append(layer)
            compared_bytes += common * left["row_bytes"]
            total_segments += len(ah)
            for segment, (ha, hb) in enumerate(zip(ah, bh)):
                if ha == hb:
                    matching_segments += 1
                    continue
                layer["different_segments"] += 1
                jobs.append(
                    dict(
                        id=str(len(jobs)),
                        layer_id=layer_id,
                        left=a["id"],
                        right=b["id"],
                        left_path=left["path"],
                        right_path=right["path"],
                        left_hash=ha,
                        right_hash=hb,
                        start=segment * args.segment_tokens,
                        end=min((segment + 1) * args.segment_tokens, common),
                        width=left["row_bytes"],
                    )
                )

    same_host, cross_host = defaultdict(list), []
    selected_bytes = 0
    for job in jobs:
        host = entries[job["left"]]["host"]
        if host == entries[job["right"]]["host"]:
            same_host[host].append(job)
        else:
            size = 2 * (job["end"] - job["start"]) * job["width"]
            if selected_bytes + size <= args.max_diff_mib * 1024 * 1024:
                selected_bytes += size
                cross_host.append(job)
    print(
        f"[D_KV_COMPARE] segments={total_segments} different={len(jobs)} "
        f"cross-host raw selected={selected_bytes / 1024**2:.2f} MiB (both copies)",
        flush=True,
    )
    exact = {}
    for host, host_jobs in same_host.items():
        print(f"[D_KV_COMPARE] compare on {host}: {len(host_jobs)} differing segments", flush=True)
        for start in range(0, len(host_jobs), 128):
            batch = host_jobs[start : start + 128]
            result = client.call(
                host, dict(action="compare", entries=slim_entries(entries, batch), jobs=batch, workers=args.workers)
            )
            if len(result) != len(batch) or {r["id"] for r in result} != {j["id"] for j in batch}:
                raise ValueError("Incomplete same-host comparison")
            exact.update((r["id"], r) for r in result)
    blocks, reused = fetch_blocks(client, args, entries, cross_host)
    for job in cross_host:
        exact[job["id"]] = worker.byte_difference(
            blocks[f"{job['id']}-left"], blocks[f"{job['id']}-right"], job["start"], job["width"]
        )
    # Even checksum-cache hits must still refer to the same immutable files.
    for host in dict.fromkeys(args.hosts):
        client.call(
            host,
            dict(
                action="verify",
                entries=[{k: e[k] for k in ("directory", "stamps")} for e in entries.values() if e["host"] == host],
            ),
        )
    with (args.output / "segments.jsonl").open("w", encoding="utf-8") as out:
        for job in jobs:
            layer = layers[job["layer_id"]]
            row = {k: layer[k] for k in ("tp_rank", "group", "layer", "component")}
            row.update(
                start=job["start"],
                end=job["end"],
                previous_checksum=job["left_hash"],
                current_checksum=job["right_hash"],
                exact_compared=job["id"] in exact,
            )
            if job["id"] in exact:
                result = exact[job["id"]]
                if result["different_bytes"] <= 0:
                    raise ValueError("Different checksums yielded identical bytes")
                row.update(result)
                layer["exact_segments"] += 1
                layer["confirmed_different_bytes"] += result["different_bytes"]
                layer["confirmed_different_tokens"] += result["different_tokens"]
                layer["first"].extend(result["first"][: max(0, 8 - len(layer["first"]))])
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (args.output / "layers.jsonl").open("w", encoding="utf-8") as out:
        for layer in layers:
            out.write(json.dumps(layer, separators=(",", ":")) + "\n")
    confirmed = sum(x["confirmed_different_bytes"] for x in layers)
    complete = len(exact) == len(jobs)
    first = next(
        (
            dict(
                **{k: layers[j["layer_id"]][k] for k in ("tp_rank", "group", "layer", "component")},
                start=j["start"],
                end=j["end"],
            )
            for j in jobs
        ),
        None,
    )
    report = dict(
        format=worker.FORMAT,
        mode="offline_segments",
        algorithm=worker.ALGORITHM,
        previous=args.previous,
        current=args.current,
        status="history_differs" if jobs else "history_checksums_match",
        common_prefix_tokens=common,
        previous_received_tokens=len(a_tokens),
        current_received_tokens=len(b_tokens),
        current_uncompared_range=[common, len(b_tokens)],
        previous_is_prefix=(common == len(a_tokens)),
        tp_ranks=sorted(before),
        components=len(layers),
        checksummed_bytes_per_snapshot=compared_bytes,
        segment_tokens=args.segment_tokens,
        segments=total_segments,
        matching_segments=matching_segments,
        different_segments=len(jobs),
        exact_compared_differing_segments=len(exact),
        unfetched_differing_segments=len(jobs) - len(exact),
        difference_counts_complete=complete,
        confirmed_different_bytes=confirmed,
        different_bytes=confirmed if complete else None,
        raw_cross_host_bytes_selected=selected_bytes,
        raw_cached_bytes_reused=reused,
        compressed_response_bytes=client.received_bytes,
        first_differing_segment=first,
        first_differences=[x for x in layers if x["first"]][:8],
        scope="All common-prefix bytes are checksummed. Matching checksums are not a direct byte comparison. "
        "Different segments are compared exactly on the same host, or fetched within the cross-host budget. "
        "Unfetched segments remain located by checksum; "
        "the new suffix and later NPU computation are not validated.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
