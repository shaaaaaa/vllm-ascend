# SPDX-License-Identifier: Apache-2.0
"""Validate archived raw KV files against the running service layout."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decode_kv_compare import read_workers

PREFIX = "[KV_REPLAY_FILES] "
COPY_BYTES = 4 * 1024 * 1024
PROGRESS_EVERY = 64


def token_digest(tokens: list[int]) -> str:
    # This identifies input tokens, not KV contents or their numeric accuracy.
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def live_contract(paths: list[Path]) -> dict:
    """Read the serving process's actual key ABI and hash root, never guess it."""
    layouts, roots, tp_ranks = {}, {}, set()
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
    contracts = []
    for owner, layout in layouts.items():
        if owner not in roots:
            raise ValueError(f"Missing NONE_HASH line for serving process {owner}; use its complete startup log")
        contracts.append(dict(layout, hash_root=roots[owner]))
    unique = {json.dumps(item, sort_keys=True) for item in contracts}
    if len(unique) != 1:
        raise ValueError(
            "Service logs must contain one common cache layout/hash root; use only the current startup logs"
        )
    result = json.loads(unique.pop())
    if tp_ranks:
        result["tp_size"] = max(tp_ranks) + 1
    if result["algorithm"] not in ("sha256", "sha256_cbor", "sha256_cbor_64bit", "xxhash", "xxhash_cbor"):
        raise ValueError("File replay requires a deterministic serving hash algorithm, not builtin hash")
    schema = result["schema"]
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.blake2b(encoded, digest_size=8).hexdigest() != result["signature"]:
        raise ValueError("Serving layout descriptor/signature disagree; startup log may be incomplete")
    if not all(schema.get(k) for k in ("use_mla", "use_layerwise", "dsa_two_groups", "page_first_multi_buffer")):
        raise ValueError("Replay requires the deployed MLA two-group merged-page format")
    if schema.get("save_chunk_meta") or schema.get("version") != 3:
        raise ValueError("Unsupported serving page ABI")
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

    replay.json adds launch/sampling metadata for D replay. Reading
    history in an existing PD server does not need those launch settings.
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
        raise ValueError("Live model path differs from the saved model; use the same serving weights")


def snapshot_metadata(plan: dict) -> tuple[list[int], dict, list]:
    """Validate small manifests/layouts without scanning the archived KV bytes."""
    workers = read_workers([Path(plan["snapshot_root"])], plan["previous"])
    meta, tokens, rows = workers[0]
    end = plan["cached_tokens"]
    schema = plan["contract"]["schema"]
    chunk = schema["chunk_size"]
    complete_receive = plan.get("replay_mode") == "decode_receive" and end == len(tokens)
    if not 0 < end <= len(tokens) or (end % chunk and not complete_receive):
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
        raise ValueError("Cannot read a snapshot with different byte order")
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
                    f"Saved wire layout disagrees with live service: group={group} layer={layer} "
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
    # Do not silently use TP0 when another captured rank held different KV.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, equal in enumerate(pool.map(lambda job: same_prefix(*job), comparisons), 1):
            if not equal:
                raise ValueError("Saved TP copies differ; cannot represent them in the shared TP0 namespace")
            if i % PROGRESS_EVERY == 0 or i == len(comparisons):
                print(
                    PREFIX + json.dumps(dict(status="validating_tp", files=i, total_files=len(comparisons))), flush=True
                )


def validate_snapshot(plan: dict) -> tuple[list[int], dict]:
    tokens, rows, comparisons = snapshot_metadata(plan)
    validate_tp_copies(comparisons, plan["workers"])
    return tokens, rows
