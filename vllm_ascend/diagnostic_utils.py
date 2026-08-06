# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tensor snapshot helpers shared by crash diagnostics."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch


def cpu_snapshot(value: Any) -> Any:
    """Clone tensors to CPU and turn metadata into a pickle-stable tree."""
    return _cpu_snapshot(value, set())


def _cpu_snapshot(value: Any, active_ids: set[int]) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        copied = value.copy()
        try:
            return torch.from_numpy(copied)
        except (TypeError, ValueError):
            return copied.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Enum):
        return {
            "__enum__": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.name,
            "value": _cpu_snapshot(value.value, active_ids),
        }

    value_id = id(value)
    if value_id in active_ids:
        return {"__cycle__": type(value).__qualname__}

    if is_dataclass(value) and not isinstance(value, type):
        active_ids.add(value_id)
        try:
            payload = {
                field.name: _cpu_snapshot(
                    getattr(value, field.name),
                    active_ids,
                )
                for field in fields(value)
            }
        finally:
            active_ids.remove(value_id)
        return {
            "__dataclass__": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "fields": payload,
        }

    if isinstance(value, Mapping):
        active_ids.add(value_id)
        try:
            return {
                str(key): _cpu_snapshot(item, active_ids)
                for key, item in value.items()
            }
        finally:
            active_ids.remove(value_id)

    if isinstance(value, tuple):
        active_ids.add(value_id)
        try:
            return {
                "__tuple__": [
                    _cpu_snapshot(item, active_ids) for item in value
                ]
            }
        finally:
            active_ids.remove(value_id)

    if isinstance(value, Sequence):
        active_ids.add(value_id)
        try:
            return [_cpu_snapshot(item, active_ids) for item in value]
        finally:
            active_ids.remove(value_id)

    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Atomically replace a diagnostic file with a CPU-only payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def tensor_layout(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def snapshot_cache_components(
    cache: Any,
    *,
    latent_block_ids: Sequence[int],
    indexer_block_ids: Sequence[int],
) -> Any:
    """Copy only selected physical blocks from each cache component."""
    if isinstance(cache, torch.Tensor):
        return _snapshot_cache_tensor(cache, latent_block_ids)
    if not isinstance(cache, (tuple, list)):
        return cpu_snapshot(cache)

    components = []
    for component_index, component in enumerate(cache):
        if not isinstance(component, torch.Tensor):
            components.append(cpu_snapshot(component))
            continue
        block_ids = (
            indexer_block_ids
            if component_index >= 2
            else latent_block_ids
        )
        components.append(_snapshot_cache_tensor(component, block_ids))
    return components


def _snapshot_cache_tensor(
    tensor: torch.Tensor,
    block_ids: Sequence[int],
) -> dict[str, Any]:
    layout = tensor_layout(tensor)
    if tensor.ndim == 0:
        return {
            "layout": layout,
            "whole": True,
            "physical_block_ids": [],
            "blocks": tensor.detach().cpu().clone(),
        }

    valid_ids = [
        int(block_id)
        for block_id in block_ids
        if 0 <= int(block_id) < tensor.shape[0]
    ]
    invalid_ids = [
        int(block_id)
        for block_id in block_ids
        if int(block_id) < 0 or int(block_id) >= tensor.shape[0]
    ]

    if tensor.shape[0] <= 4 and invalid_ids:
        return {
            "layout": layout,
            "whole": True,
            "physical_block_ids": list(range(tensor.shape[0])),
            "unaddressable_block_ids": invalid_ids,
            "blocks": tensor.detach().cpu().clone(),
        }

    if valid_ids:
        indices = torch.tensor(
            valid_ids,
            dtype=torch.long,
            device=tensor.device,
        )
        blocks = tensor.index_select(0, indices).detach().cpu().clone()
    else:
        blocks = torch.empty(
            (0, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device="cpu",
        )
    return {
        "layout": layout,
        "whole": False,
        "physical_block_ids": valid_ids,
        "unaddressable_block_ids": invalid_ids,
        "blocks": blocks,
    }
