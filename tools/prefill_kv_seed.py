# SPDX-License-Identifier: Apache-2.0
"""Inject completed D snapshots into a live Mooncake store, without a model.

The seed process owns the injected replicas until the replay finishes. The
serving processes, their streams, and their LMCache instances are not patched.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decode_kv_compare import read_workers

PREFIX = "[PREFILL_REPLAY_SEED] "
COPY_BYTES = 4 * 1024 * 1024
PROGRESS_EVERY = 64


def token_digest(tokens: list[int]) -> str:
    # This identifies input tokens, not KV contents or their numeric accuracy.
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def live_contract(paths: list[Path]) -> dict:
    """Read the serving process's actual key ABI and hash root, never guess it."""
    layouts, roots, masters, tp_ranks = {}, {}, set(), set()
    for path in paths:
        for raw in path.open(encoding="utf-8", errors="replace"):
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
            pid = re.search(r"pid=(\d+)", line)
            owner = (str(path.resolve()), pid[1] if pid else "unlabelled")
            tp_ranks.update(int(x) for x in re.findall(r"(?:Worker_)?(?:DP\d+_)?TP(\d+)", line))
            if "Initialized NONE_HASH=" in line:
                value = line.split("Initialized NONE_HASH=", 1)[1].split(" from vLLM", 1)[0]
                root = ast.literal_eval(value)
                if type(root) not in (bytes, int):
                    raise ValueError("Unsupported serving hash root")
                roots[owner] = repr(root)
            if "Mooncake payload layout:" in line:
                match = re.search(r"hash algorithm=(\w+);.*signature=([0-9a-f]+).*schema=(\{.*\})", line)
                if not match:
                    raise ValueError(f"Cannot parse serving cache layout in {path}")
                layouts[owner] = dict(algorithm=match[1], signature=match[2], schema=ast.literal_eval(match[3]))
            masters.update(re.findall(r"mooncakestore://([^/\s'\"]+)", line))
            masters.update(re.findall(r"master_server_address:\s*([\w.\-]+:\d+)", line))
    contracts = []
    for owner, layout in layouts.items():
        if owner not in roots:
            raise ValueError(f"Missing NONE_HASH line for serving process {owner}; use its complete startup log")
        contracts.append(dict(layout, hash_root=roots[owner]))
    unique = {json.dumps(item, sort_keys=True) for item in contracts}
    if len(unique) != 1:
        raise ValueError("P logs must contain one common cache layout/hash root; use only the current startup logs")
    result = json.loads(unique.pop())
    if tp_ranks:
        result["tp_size"] = max(tp_ranks) + 1
    if result["algorithm"] not in ("sha256", "sha256_cbor", "sha256_cbor_64bit", "xxhash", "xxhash_cbor"):
        raise ValueError("Native replay requires a deterministic serving hash algorithm, not builtin hash")
    schema = result["schema"]
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.blake2b(encoded, digest_size=8).hexdigest() != result["signature"]:
        raise ValueError("Serving layout descriptor/signature disagree; startup log may be incomplete")
    if not all(schema.get(k) for k in ("use_mla", "use_layerwise", "dsa_two_groups", "page_first_multi_buffer")):
        raise ValueError("Replay requires the deployed MLA two-group merged-page format")
    if schema.get("save_chunk_meta") or schema.get("version") != 3:
        raise ValueError("Unsupported serving page ABI")
    if len(masters) > 1:
        raise ValueError("P logs contain multiple Mooncake masters; select only the current deployment logs")
    if len(masters) == 1:
        result["master"] = masters.pop()
    return result


def same_prefix(left: Path, right: Path, count: int) -> bool:
    with left.open("rb") as a, right.open("rb") as b:
        remaining = count
        while remaining:
            n = min(remaining, COPY_BYTES)
            x, y = a.read(n), b.read(n)
            if len(x) != n or x != y:
                return False
            remaining -= n
    return True


def validate_snapshot_model(meta: dict, directory: Path, model_path: str) -> None:
    """Legacy raw snapshots already record their model in manifest.json.

    replay.json adds launch/sampling metadata for standalone D replay. Injecting
    history into an existing PD server does not need those launch settings.
    When both model identities exist, neither may contradict the live server.
    """
    models = []
    if meta.get("model"):
        models.append(meta["model"])
    replay = directory / "replay.json"
    if replay.is_file():
        recorded = json.loads(replay.read_text(encoding="utf-8"))
        runtime = recorded.get("runtime") or {}
        options = runtime.get("engine_options") or {}
        if options.get("model"):
            models.append(options["model"])
    if not models:
        raise ValueError("Snapshot lacks source model identity in manifest.json or replay.json")
    if any(model != model_path for model in models):
        raise ValueError("Live P model path differs from the saved model; use the same serving weights")


def snapshot_metadata(plan: dict) -> tuple[list[int], dict, list]:
    """Validate small manifests/layouts without scanning the archived KV bytes."""
    workers = read_workers([Path(plan["snapshot_root"])], plan["previous"])
    meta, tokens, rows = workers[0]
    end = plan["cached_tokens"]
    schema = plan["contract"]["schema"]
    chunk = schema["chunk_size"]
    if not 0 < end <= len(tokens) or end % chunk:
        raise ValueError("Cached frontier must be a full cache-block boundary inside the previous snapshot")
    if token_digest(tokens[:end]) != plan["prefix_digest"]:
        raise ValueError("Previous snapshot tokens do not match the live-tokenized replay prefix")
    previous_count = plan["previous_prompt_tokens"]
    if len(tokens) >= previous_count and token_digest(tokens[:previous_count]) != plan["previous_prompt_digest"]:
        raise ValueError("Previous request token IDs changed under the live tokenizer")
    names = schema.get("runtime_kv_groups")
    if names != [[n, layers] for n, layers in zip(meta["groups"], meta["layer_names"])]:
        raise ValueError("Live P and saved snapshot have different layer inventories/order")
    if meta["byteorder"] != sys.byteorder:
        raise ValueError("Cannot inject a snapshot with different byte order")
    if len(workers) != plan["tp_size"]:
        raise ValueError("Live P and snapshot TP sizes differ")
    c8 = schema.get("indexer_execution")
    if c8 and c8.get("c8_layers") and len(c8["c8_layers"]) != meta["groups"][1]:
        raise ValueError("Indexer precision policy differs from the layer inventory")
    size = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}.get(schema["dtype"])
    if not size:
        raise ValueError("Unsupported live model dtype")
    if c8 and c8.get("head_dim") != 128:
        raise ValueError("Unsupported C8 indexer head dimension")
    for group, layers in enumerate(meta["groups"]):
        for layer in range(layers):
            parts = [row for (g, idx, _), (row, _) in sorted(rows.items()) if (g, idx) == (group, layer)]
            if [p["component"] for p in parts] != list(range(len(parts))):
                raise ValueError("Non-contiguous snapshot components")
            actual = [(p["dtype"], p["row_bytes"]) for p in parts]
            if group == 0:
                expected_bytes = schema["group0_raw_token_dim"] * size
                expected = f"dtype={schema['dtype']} total_row_bytes={expected_bytes}"
                valid = sum(p["row_bytes"] for p in parts) == expected_bytes
                valid = valid and all(p["dtype"] == schema["dtype"] for p in parts)
            else:
                if c8:
                    # Under C8/mixed storage group1_raw_token_dim is already
                    # the MAX packet width in BYTES (130 or 256), not a head
                    # dimension. Reconstruct each layer from its head/policy.
                    quantized = not c8.get("c8_layers") or c8["c8_layers"][layer]
                    expected = (
                        [("torch.int8", c8["head_dim"]), ("torch.float16", 2)]
                        if quantized
                        else [(schema["dtype"], c8["head_dim"] * size)]
                    )
                else:
                    expected = [(schema["dtype"], schema["group1_raw_token_dim"] * size)]
                valid = actual == expected
            if not valid:
                raise ValueError(
                    f"Saved wire layout disagrees with live P: group={group} layer={layer} "
                    f"actual={actual} expected={expected}"
                )
    comparisons = []
    for rank, (other, other_tokens, other_rows) in workers.items():
        validate_snapshot_model(other, next(iter(other_rows.values()))[1].parent, plan["model_path"])
        if other_tokens != tokens or other["groups"] != meta["groups"] or other["layer_names"] != meta["layer_names"]:
            raise ValueError(f"Snapshot TP {rank} token/layer metadata differs")
        if other_rows.keys() != rows.keys():
            raise ValueError(f"Snapshot TP {rank} component inventory differs")
        if rank:
            for key, (row, path) in rows.items():
                candidate, target = other_rows[key]
                if (row["dtype"], row["row_bytes"]) != (candidate["dtype"], candidate["row_bytes"]):
                    raise ValueError("Snapshot TP wire layouts differ")
                comparisons.append((path, target, end * row["row_bytes"]))
    return tokens[:end], rows, comparisons


def validate_tp_copies(comparisons: list, workers: int) -> None:
    # The production RemoteFill key namespace collapses both groups to TP0.
    # Do not silently inject TP0 when another captured rank held different KV.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, equal in enumerate(pool.map(lambda job: same_prefix(*job), comparisons), 1):
            if not equal:
                raise ValueError("Saved TP copies differ; cannot faithfully inject them into the shared TP0 namespace")
            if i % PROGRESS_EVERY == 0 or i == len(comparisons):
                print(
                    PREFIX + json.dumps(dict(status="validating_tp", files=i, total_files=len(comparisons))), flush=True
                )


def validate_snapshot(plan: dict) -> tuple[list[int], dict]:
    tokens, rows, comparisons = snapshot_metadata(plan)
    validate_tp_copies(comparisons, plan["workers"])
    return tokens, rows


def page_keys(plan: dict, tokens: list[int]):
    """Use vLLM's actual hash function and LMCache's actual key serializer."""
    import torch
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.mooncake_layout import MOONCAKE_PAYLOAD_LAYOUT_TAG, mooncake_page_key
    from lmcache.v1.token_database import DSA_INDEX_CACHE_SCHEMA, DSA_INDEX_CACHE_SCHEMA_TAG
    from vllm.utils.hashing import get_hash_fn_by_name

    contract = plan["contract"]
    algorithm = contract["algorithm"].replace("sha256_cbor_64bit", "sha256_cbor")
    hash_fn = get_hash_fn_by_name(algorithm)
    root = ast.literal_eval(contract["hash_root"])
    schema = contract["schema"]
    chunk = schema["chunk_size"]
    for start in range(0, len(tokens), chunk):
        end = start + chunk
        if end > len(tokens):
            raise ValueError("Only full historical cache blocks may be seeded")
        root = hash_fn((root, tuple(tokens[start:end]), ()))
        for group, (count, _) in enumerate(schema["runtime_kv_groups"]):
            dtype = (
                torch.uint8
                if group == 1 and schema.get("indexer_execution")
                else getattr(torch, schema["dtype"].split(".")[-1])
            )
            tags = {MOONCAKE_PAYLOAD_LAYOUT_TAG: contract["signature"]}
            if group == 1:
                tags[DSA_INDEX_CACHE_SCHEMA_TAG] = DSA_INDEX_CACHE_SCHEMA
            key = CacheEngineKey(plan["model_path"], 1, 0, root, dtype, tags, kv_group=group)
            yield mooncake_page_key(key, count), group, start, end


def fill_page(rows: dict, group: int, start: int, end: int, target: memoryview) -> int:
    """Pack layer-major planar bytes, exactly as the merged-page connector does."""
    offset = 0
    for (g, _, _), (row, path) in sorted(rows.items()):
        if g != group:
            continue
        size = (end - start) * row["row_bytes"]
        with path.open("rb") as source:
            source.seek(start * row["row_bytes"])
            if source.readinto(target[offset : offset + size]) != size:
                raise ValueError(f"Snapshot truncated while reading {path}")
        offset += size
    return offset


def seed(plan: dict) -> None:
    """Run on the snapshot host, using real native SDK transfers and replicas."""
    import threading

    print(PREFIX + json.dumps(dict(status="runtime_check", python=sys.executable)), flush=True)
    import torch
    import torch_npu
    from layerwise_prefill_mooncake_check import detect_local_hostname
    from layerwise_prefill_mooncake_routing import require_placement
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    torch_npu.npu.set_device(plan["seed_device"])
    torch_npu.npu.init()
    # Resolve the same key/hash dependencies as injection, before scanning
    # hundreds of GB of archived TP copies. Empty input creates no cache keys.
    list(page_keys(plan, []))
    print(PREFIX + json.dumps(dict(status="runtime_ready")), flush=True)
    print(PREFIX + json.dumps(dict(status="validating", previous=plan["previous"])), flush=True)
    tokens, rows, comparisons = snapshot_metadata(plan)
    keys = list(page_keys(plan, tokens))
    widths = {g: sum(r["row_bytes"] for (group, _, _), (r, _) in rows.items() if group == g) for g in (0, 1)}
    total = len(tokens) * sum(widths.values())
    capacity = plan.get("store_bytes") or int(total * 1.2) + 2 * 1024**3
    if capacity < total:
        raise ValueError("Seed Mooncake segment is smaller than the historical KV")
    release = Path(plan["release_file"])
    release.parent.mkdir(parents=True, exist_ok=True)
    if release.exists():
        raise ValueError("Seed release marker already exists")
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    store = MooncakeDistributedStore()
    registered = []
    try:
        host = plan.get("local_hostname") or detect_local_hostname(plan["master"])
        status = store.setup(host, "P2PHANDSHAKE", capacity, 0, "ascend", "", plan["master"])
        if status not in (None, 0):
            raise RuntimeError(f"Native Mooncake setup failed: {status}")
        required = (
            "batch_is_exist",
            "batch_put_from_multi_buffers",
            "batch_get_into_multi_buffers",
            "get_replica_desc",
        )
        if any(not callable(getattr(store, name, None)) for name in required):
            raise RuntimeError("Native Mooncake SDK lacks required page/placement APIs")
        if any(store.batch_is_exist([key for key, *_ in keys])):
            raise ValueError("Historical keys already exist in Mooncake; refusing to overwrite an existing run")
        size = max(widths.values()) * plan["contract"]["schema"]["chunk_size"]
        # AscendDirect's wildcard registration queries ACL pointer attributes.
        # Ordinary CPU allocations have no HOST location there. Allocate through
        # torch_npu's pinned host allocator, and retain both owners until after
        # unregister/close (including partial registration failures).
        source, check = [torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True) for _ in range(2)]
        for tensor in (source, check):
            status = store.register_buffer(tensor.data_ptr(), tensor.numel())
            if status not in (None, 0):
                raise RuntimeError(f"Cannot register pinned seed CPU staging memory: bytes={size} status={status}")
            registered.append(tensor)
        print(PREFIX + json.dumps(dict(status="staging_ready", buffers=2, bytes_each=size, pinned=True)), flush=True)
        # Catch native setup/registration failures BEFORE the expensive raw-file
        # scan. No historical page is published until every TP copy matches.
        validate_tp_copies(comparisons, plan["workers"])
        destination = ReplicateConfig()
        destination.replica_num = 1
        destination.preferred_segment = store.get_hostname()
        source_view = memoryview(source.numpy()).cast("B")
        for i, (key, group, start, end) in enumerate(keys, 1):
            if stopped.is_set() or release.exists():
                raise RuntimeError("Seed cancelled before publication completed")
            count = fill_page(rows, group, start, end, source_view)
            if store.batch_put_from_multi_buffers([key], [[source.data_ptr()]], [[count]], destination) != [0]:
                raise RuntimeError(f"Native seed put failed: group={group} range={start}:{end}")
            require_placement(store, key, store.get_hostname(), count)
            got = store.batch_get_into_multi_buffers([key], [[check.data_ptr()]], [[count]])
            if got != [count] or not torch.equal(source[:count], check[:count]):
                raise RuntimeError(f"Native seed readback differs: group={group} range={start}:{end}")
            if i % PROGRESS_EVERY == 0 or i == len(keys):
                print(PREFIX + json.dumps(dict(status="copying", pages=i, total_pages=len(keys))), flush=True)
        print(PREFIX + json.dumps(dict(status="ready", tokens=len(tokens), pages=len(keys), bytes=total)), flush=True)
        deadline = time.monotonic() + plan["hold_seconds"]
        while not stopped.wait(0.5) and not release.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Seed lease expired before the replay released it")
    finally:
        try:
            for tensor in registered:
                status = store.unregister_buffer(tensor.data_ptr())
                if status not in (None, 0):
                    raise RuntimeError(f"Cannot unregister seed CPU staging memory: status={status}")
        finally:
            # Close only this tool's segment; never remove keys/replicas owned by P/D.
            # Keep pinned tensors alive until the native transport is closed.
            store.close()
