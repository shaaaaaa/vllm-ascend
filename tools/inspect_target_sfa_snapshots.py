#!/usr/bin/env python3
"""Export target-SFA planner snapshots and an independent CPU oracle as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

INDEX_TOPK = 2048


def _load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"snapshot is not a dictionary: {path}")
    return payload


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"snapshot field {name} is not a tensor")
    return value.detach().cpu().contiguous()


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _tensor_summary(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return _json_value(value)
    tensor = value.detach().cpu().reshape(-1)
    result: dict[str, Any] = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "numel": int(value.numel()),
    }
    if tensor.numel():
        result.update(
            {
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "first16": tensor[:16].tolist(),
                "last16": tensor[-16:].tolist(),
            }
        )
    return result


def _state_rows(snapshot: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(snapshot, dict):
        raise TypeError("resident state snapshot is missing")
    rows = snapshot.get("row_indices")
    if not isinstance(rows, list):
        raise TypeError("resident state row_indices is missing")
    tokens = _tensor(snapshot.get("tokens"), "state.tokens")
    slots = _tensor(snapshot.get("slots"), "state.slots")
    counts = _tensor(snapshot.get("counts"), "state.counts")
    generations = _tensor(
        snapshot.get("generations"),
        "state.generations",
    )
    result: dict[int, dict[str, Any]] = {}
    for local_row, state_row in enumerate(rows):
        shards = []
        for shard in range(tokens.shape[1]):
            count = int(counts[local_row, shard, 0])
            shard_tokens = tokens[local_row, shard, :count].tolist()
            shard_slots = slots[local_row, shard, :count].tolist()
            inversions = [
                index
                for index in range(1, count)
                if shard_tokens[index - 1] > shard_tokens[index]
            ]
            shards.append(
                {
                    "shard": shard,
                    "count": count,
                    "sorted": not inversions,
                    "first_inversion": inversions[0] if inversions else None,
                    "tokens": shard_tokens,
                    "slots": shard_slots,
                }
            )
        result[int(state_row)] = {
            "generation": int(generations[local_row, 0]),
            "total_count": sum(shard["count"] for shard in shards),
            "shards": shards,
        }
    return result


def _resident_map(row: dict[str, Any]) -> dict[int, int]:
    result: dict[int, int] = {}
    for shard in row["shards"]:
        for token, slot in zip(
            shard["tokens"],
            shard["slots"],
            strict=True,
        ):
            result[int(token)] = int(slot)
    return result


def _difference(expected: list[int], actual: list[int]) -> dict[str, Any]:
    width = max(len(expected), len(actual))
    indices = [
        index
        for index in range(width)
        if index >= len(expected)
        or index >= len(actual)
        or expected[index] != actual[index]
    ]
    first = indices[0] if indices else None
    begin = max((first or 0) - 4, 0)
    end = min((first or 0) + 5, width)
    return {
        "matches": not indices,
        "mismatch_count": len(indices),
        "first_mismatch": first,
        "expected_length": len(expected),
        "actual_length": len(actual),
        "expected_near_first": expected[begin:end],
        "actual_near_first": actual[begin:end],
    }


def _dict_difference(
    expected: dict[int, int],
    actual: dict[int, int],
) -> dict[str, Any]:
    keys = sorted(set(expected) | set(actual))
    mismatches = [
        token for token in keys if expected.get(token) != actual.get(token)
    ]
    return {
        "matches": not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatches": [
            {
                "token": token,
                "expected_slot": expected.get(token),
                "actual_slot": actual.get(token),
            }
            for token in mismatches[:32]
        ],
        "expected_count": len(expected),
        "actual_count": len(actual),
    }


def _reference_step(
    shards: list[list[int]],
    resident: dict[int, int],
    capacity: int,
) -> tuple[dict[int, int], list[int], list[int]]:
    occupied = set(resident.values())
    if occupied != set(range(len(occupied))):
        raise ValueError(
            "CPU oracle requires dense resident slots; "
            f"observed={sorted(occupied)[:32]} count={len(occupied)}"
        )
    current = {token for shard in shards for token in shard}
    shard_count = len(shards)
    evictable_slots = []
    for shard in range(shard_count):
        old_tokens = sorted(
            token
            for token in resident
            if token % shard_count == shard and token not in current
        )
        evictable_slots.extend(resident[token] for token in old_tokens)
    assigned: dict[int, int] = {}
    misses = []
    miss_slots = []
    for shard in shards:
        for token in shard:
            if token in resident:
                assigned[token] = resident[token]
            else:
                miss_index = len(misses)
                if miss_index < len(evictable_slots):
                    slot = evictable_slots[miss_index]
                else:
                    slot = len(occupied) + miss_index - len(evictable_slots)
                if slot >= capacity:
                    raise ValueError(
                        f"CPU oracle assigned slot {slot} beyond {capacity}"
                    )
                assigned[token] = slot
                misses.append(token)
                miss_slots.append(slot)
    overwritten = set(miss_slots)
    updated = {
        token: slot for token, slot in resident.items() if slot not in overwritten
    }
    updated.update({token: assigned[token] for token in misses})
    return updated, misses, miss_slots


def _metadata_tensor(metadata: dict[str, Any], *names: str) -> torch.Tensor:
    for name in names:
        value = metadata.get(name)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
    raise KeyError(f"metadata lacks tensor from {names}")


def _workspace_fields(pre: dict[str, Any]) -> dict[str, Any]:
    workspace = pre.get("resident_workspace")
    fields = workspace.get("fields") if isinstance(workspace, dict) else None
    if not isinstance(fields, dict):
        raise TypeError("pre snapshot lacks resident_workspace.fields")
    return fields


def _build_report(
    input_path: Path,
    pre_path: Path,
    layer_input: dict[str, Any],
    pre: dict[str, Any],
) -> dict[str, Any]:
    metadata = layer_input.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("input snapshot lacks metadata")
    before_snapshot = layer_input.get("resident_state_before")
    after_snapshot = pre.get("resident_state_after")
    before_rows = _state_rows(before_snapshot)
    after_rows = _state_rows(after_snapshot)
    fields = _workspace_fields(pre)
    shard_packed = _tensor(fields.get("shard_packed"), "shard_packed")
    shard_mapping = _tensor(fields.get("shard_mapping"), "shard_mapping")
    shard_counts = _tensor(fields.get("shard_counts"), "shard_counts")
    prior_slots = _tensor(fields.get("prior_slots"), "prior_slots")
    shard_miss_tokens = _tensor(
        fields.get("shard_miss_tokens"),
        "shard_miss_tokens",
    )
    shard_miss_positions = _tensor(
        fields.get("shard_miss_positions"),
        "shard_miss_positions",
    )
    shard_evictable_slots = _tensor(
        fields.get("shard_evictable_slots"),
        "shard_evictable_slots",
    )
    miss_tokens_tensor = _tensor(fields.get("miss_tokens"), "miss_tokens")
    miss_counts_tensor = _tensor(fields.get("miss_counts"), "miss_counts")
    target_slots_tensor = _tensor(fields.get("target_slots"), "target_slots")

    raw_topk = _tensor(pre.get("raw_topk"), "raw_topk")
    remapped_topk = _tensor(pre.get("topk_indices"), "topk_indices")
    selected_counts = _tensor(pre.get("selected_counts"), "selected_counts")
    request_count = int(selected_counts.shape[0])
    request_width = int(raw_topk.numel()) // request_count
    if request_width % INDEX_TOPK:
        raise ValueError(f"request width is not divisible by {INDEX_TOPK}")
    mtp = request_width // INDEX_TOPK
    row_count = request_count * mtp
    shard_count = int(shard_packed.shape[1])
    capacity = int(shard_packed.shape[2])
    source = raw_topk.reshape(request_count, request_width).to(torch.int64)
    remapped = remapped_topk.reshape(request_count, request_width).to(torch.int64)
    boundaries = _tensor(
        pre.get("remap_boundary"),
        "remap_boundary",
    )[:row_count].to(torch.int64)
    request_states = _metadata_tensor(
        metadata,
        "resident_state_indices",
    )[:request_count].to(torch.int64)
    request_generations = _metadata_tensor(
        metadata,
        "resident_state_generations",
    )[:request_count].to(torch.int64)
    block_table = _metadata_tensor(metadata, "block_table")[:request_count].to(
        torch.int64
    )
    block_size = int(pre.get("block_size", 128))
    dummy_state_base = int(before_snapshot["dummy_state_base"])

    analyses = []
    expected_states: list[dict[int, int]] = []
    expected_remapped = source.clone()
    for request in range(request_count):
        state_index = int(request_states[request])
        safe_state = (
            state_index
            if 0 <= state_index < dummy_state_base
            else dummy_state_base + request
        )
        requested_generation = int(request_generations[request])
        before_row = before_rows[safe_state]
        stored_generation = int(before_row["generation"])
        generation_matches = (
            0 <= state_index < dummy_state_base
            and stored_generation == requested_generation
        )
        resident_before = _resident_map(before_row) if generation_matches else {}

        expected_shards = [[] for _ in range(shard_count)]
        for row in range(mtp):
            boundary = int(boundaries[request * mtp + row])
            begin = row * INDEX_TOPK
            end = begin + INDEX_TOPK
            for token in source[request, begin:end].tolist():
                token = int(token)
                if 0 <= token < boundary:
                    expected_shards[token % shard_count].append(token)
        expected_shards = [
            sorted(set(shard_tokens)) for shard_tokens in expected_shards
        ]
        expected_state, expected_misses, expected_miss_slots = _reference_step(
            expected_shards,
            resident_before,
            capacity,
        )
        expected_states.append(expected_state)
        for position, token in enumerate(source[request].tolist()):
            row = position // INDEX_TOPK
            boundary = int(boundaries[request * mtp + row])
            token = int(token)
            if 0 <= token < boundary:
                expected_remapped[request, position] = expected_state[token]

        after_row = after_rows[safe_state]
        actual_state = _resident_map(after_row)
        union_analysis = []
        workspace_dump = []
        for shard in range(shard_count):
            count_record = shard_counts[request, shard].tolist()
            current_count = int(count_record[0])
            miss_count = int(count_record[1])
            evictable_count = int(count_record[2])
            old_count = int(count_record[3])
            selected_evict_count = int(count_record[4])
            actual_packed = shard_packed[
                request, shard, :current_count
            ].tolist()
            union_analysis.append(
                {
                    "shard": shard,
                    "expected_vs_actual_packed": _difference(
                        expected_shards[shard],
                        actual_packed,
                    ),
                    "counts": {
                        "current": current_count,
                        "miss": miss_count,
                        "evictable": evictable_count,
                        "old": old_count,
                        "selected_evict": selected_evict_count,
                    },
                }
            )
            workspace_dump.append(
                {
                    "shard": shard,
                    "count_record": count_record,
                    "packed": actual_packed,
                    "mapping": shard_mapping[request, shard].tolist(),
                    "prior_slots": prior_slots[
                        request, shard, :current_count
                    ].tolist(),
                    "miss_tokens": shard_miss_tokens[
                        request, shard, :miss_count
                    ].tolist(),
                    "miss_positions": shard_miss_positions[
                        request, shard, :miss_count
                    ].tolist(),
                    "evictable_slots": shard_evictable_slots[
                        request, shard, :evictable_count
                    ].tolist(),
                }
            )

        actual_miss_count = int(miss_counts_tensor[request, 0])
        actual_misses = miss_tokens_tensor[
            request, :actual_miss_count
        ].tolist()
        actual_targets = target_slots_tensor[
            request, :actual_miss_count
        ].tolist()
        expected_targets = [
            int(block_table[request, slot // block_size]) * block_size
            + slot % block_size
            for slot in expected_miss_slots
        ]
        analyses.append(
            {
                "request": request,
                "state_index": state_index,
                "safe_state": safe_state,
                "stored_generation_before": stored_generation,
                "requested_generation": requested_generation,
                "generation_matches": generation_matches,
                "stored_generation_after": int(after_row["generation"]),
                "persistent_counts_before": [
                    shard["count"] for shard in before_row["shards"]
                ],
                "persistent_counts_after": [
                    shard["count"] for shard in after_row["shards"]
                ],
                "union": union_analysis,
                "miss_tokens": _difference(expected_misses, actual_misses),
                "target_slots": _difference(expected_targets, actual_targets),
                "resident_state_after": _dict_difference(
                    expected_state,
                    actual_state,
                ),
                "remapped_topk": _difference(
                    expected_remapped[request].tolist(),
                    remapped[request].tolist(),
                ),
                "cpu_expected": {
                    "shards": expected_shards,
                    "resident_state": [
                        {"token": token, "slot": slot}
                        for token, slot in sorted(expected_state.items())
                    ],
                    "miss_tokens": expected_misses,
                    "miss_slots": expected_miss_slots,
                    "target_slots": expected_targets,
                    "remapped_topk": expected_remapped[request].tolist(),
                },
                "workspace": workspace_dump,
            }
        )

    metadata_names = (
        "num_actual_tokens",
        "num_decode_tokens",
        "req_ids",
        "decode_request_ids_compact",
        "seq_lens",
        "seq_lens_cpu",
        "cum_query_lens",
        "block_table",
        "split_boundary",
        "decode_split_boundary",
        "decode_remap_boundary",
        "decode_req_indices",
        "decode_req_indices_cpu",
        "decode_current_positions_cpu",
        "decode_scratch_base",
        "decode_scratch_capacity",
        "resident_state_indices",
        "resident_state_generations",
    )
    return {
        "files": {"input": str(input_path), "pre": str(pre_path)},
        "identity": {
            name: {"input": layer_input.get(name), "pre": pre.get(name)}
            for name in (
                "schema_version",
                "step_id",
                "rank",
                "pid",
                "layer",
                "layer_name",
                "phase",
            )
        },
        "shape": {
            "requests": request_count,
            "mtp": mtp,
            "rows": row_count,
            "request_width": request_width,
            "shards": shard_count,
            "capacity": capacity,
            "block_size": block_size,
        },
        "analysis": analyses,
        "captured": {
            "metadata": {
                name: _json_value(metadata.get(name)) for name in metadata_names
            },
            "layer_input_tensor_summary": _tensor_summary(
                layer_input.get("tensor")
            ),
            "resident_state_before": before_rows,
            "raw_topk": _json_value(raw_topk),
            "remap_boundary": _json_value(boundaries),
            "selected_packed": _json_value(pre.get("selected_packed")),
            "selected_counts": _json_value(selected_counts),
            "target_slots": _json_value(pre.get("target_slots")),
            "remapped_topk": _json_value(remapped_topk),
            "resident_state_after": after_rows,
            "ql_nope_summary": _tensor_summary(pre.get("ql_nope")),
            "q_pe_summary": _tensor_summary(pre.get("q_pe")),
            "physical_block_ids": _json_value(pre.get("physical_block_ids")),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON here instead of stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layer_input = _load(args.input)
    pre = _load(args.pre)
    report = _build_report(args.input, args.pre, layer_input, pre)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
