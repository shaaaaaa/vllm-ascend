# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only, address-independent import of a single real prefill.

There is deliberately no model call or generator deserialization in restore.
The caller replays the recorded connector callbacks to rebuild its own local
CPU cache and request lifecycle. These helpers run at model boundaries only.
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.attention.sfa_parity import ParityError


def copy_tree(value: Any, device: Any = "cpu") -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    if isinstance(value, dict):
        return {key: copy_tree(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(copy_tree(item, device) for item in value)
    if value is None or type(value) in (int, float, str, bool):
        return value
    raise ParityError(f"Unsupported prefill checkpoint value: {type(value)}")


def assert_same_tree(expected: Any, actual: Any, label: str) -> None:
    """Exact checkpoint validation, including unused NaN payload bits in KV."""
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if expected.shape == actual.shape and expected.dtype == actual.dtype:
            a = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            b = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            if torch.equal(a, b):
                return
    elif type(expected) is type(actual):
        if isinstance(expected, dict) and expected.keys() == actual.keys():
            for key in expected:
                assert_same_tree(expected[key], actual[key], f"{label}.{key}")
            return
        if isinstance(expected, (tuple, list)) and len(expected) == len(actual):
            for index, (a, b) in enumerate(zip(expected, actual)):
                assert_same_tree(a, b, f"{label}[{index}]")
            return
        if expected is None or type(expected) in (int, float, str, bool):
            if expected == actual:
                return
    raise ParityError(f"Prefill checkpoint mismatch: {label}")


@dataclass
class CacheBinding:
    caches: tuple[torch.Tensor, ...]
    block_table: torch.Tensor


def _blocks(binding: CacheBinding) -> torch.Tensor:
    table = binding.block_table.detach().cpu()
    if table.ndim != 2 or table.dtype not in (torch.int32, torch.int64) or not binding.caches:
        raise ParityError("Invalid prefill cache/block-table layout")
    blocks = torch.unique(table[table >= 0]).long()
    if not blocks.numel():
        raise ParityError("No allocated prefill KV blocks")
    for cache in binding.caches:
        if cache.ndim != 4 or int(blocks[-1]) >= cache.shape[0]:
            raise ParityError("Prefill block table addresses invalid KV storage")
    return blocks


def capture_caches(bindings: dict[str, CacheBinding]) -> dict:
    result = {}
    for name, binding in bindings.items():
        blocks = _blocks(binding)
        result[name] = {
            "table": copy_tree(binding.block_table),
            "blocks": blocks,
            # Include partial blocks and scratch slots: MTP can populate these
            # before the first target decode. Never serialize raw host pointers.
            "values": tuple(copy_tree(cache.index_select(0, blocks.to(cache.device))) for cache in binding.caches),
        }
    return result


def plan_cache_restore(snapshot: dict, bindings: dict[str, CacheBinding]) -> dict:
    """Validate every group before writing; remap independent physical blocks."""
    if snapshot.keys() != bindings.keys():
        raise ParityError("Prefill KV group coverage differs")
    plan = {}
    for name, binding in bindings.items():
        _blocks(binding)
        saved = snapshot[name]
        old, new = saved["table"], binding.block_table.detach().cpu()
        if old.shape != new.shape or not torch.equal(old < 0, new < 0):
            raise ParityError(f"{name}: prefill block-table coverage differs")
        mapping, reverse = {}, {}
        for a, b in zip(old.reshape(-1).tolist(), new.reshape(-1).tolist()):
            if a < 0:
                continue
            if mapping.setdefault(a, b) != b or reverse.setdefault(b, a) != a:
                raise ParityError(f"{name}: prefill block alias layout differs")
        assert_same_tree(torch.unique(old[old >= 0]).long(), saved["blocks"], f"{name}.blocks")
        destinations = torch.tensor([mapping[block] for block in saved["blocks"].tolist()], dtype=torch.long)
        if len(saved["values"]) != len(binding.caches):
            raise ParityError(f"{name}: prefill cache component coverage differs")
        for cache, value in zip(binding.caches, saved["values"]):
            if value.shape != (len(destinations), *cache.shape[1:]) or value.dtype != cache.dtype:
                raise ParityError(f"{name}: prefill cache shape/dtype differs")
        plan[name] = destinations
    return plan


def restore_cache_group(saved: dict, binding: CacheBinding, blocks: torch.Tensor) -> None:
    for cache, value in zip(binding.caches, saved["values"]):
        cache.index_copy_(0, blocks.to(cache.device), value.to(cache.device))


def verify_cache_restore(snapshot: dict, bindings: dict[str, CacheBinding], plan: dict) -> None:
    for name, binding in bindings.items():
        for index, (cache, value) in enumerate(zip(binding.caches, snapshot[name]["values"])):
            restored = cache.index_select(0, plan[name].to(cache.device))
            assert_same_tree(value, restored, f"{name}.restored_KV[{index}]")


def restore_prefill(snapshot: dict, bindings: dict[str, CacheBinding], plan: dict, *, wait, save) -> None:
    """Import bytes and replay real LMCache callbacks, without running attention.

    Keep the recorded wait/save order so generator advancement, deferred stores,
    partial tails and indexer-residency bookkeeping use the original APIs.
    """
    restored = set()
    for operation, name in snapshot["callbacks"]:
        if operation == "wait":
            wait(name)
        elif operation == "save" and name not in restored:
            restore_cache_group(snapshot["caches"][name], bindings[name], plan[name])
            save(name, list(bindings[name].caches))
            restored.add(name)
        else:
            raise ParityError(f"Invalid/duplicate prefill callback: {operation} {name}")
    if restored != bindings.keys():
        raise ParityError("Prefill import did not restore/save every KV group")
    # Verification is a separate model-boundary check. A caller must not wrap
    # transfers/connector collectives in its CPU failure-agreement collective.


def validate_callbacks(callbacks: list, bindings: dict[str, CacheBinding]) -> None:
    saves = []
    for operation, name in callbacks:
        if name not in bindings or operation not in ("wait", "save"):
            raise ParityError(f"Unexpected prefill callback: {operation} {name}")
        if operation == "save":
            saves.append(name)
    if len(saves) != len(bindings) or set(saves) != bindings.keys():
        raise ParityError("Prefill must save each latent/indexer group exactly once")
