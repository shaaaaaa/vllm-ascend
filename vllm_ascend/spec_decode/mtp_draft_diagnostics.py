# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Crash-localization helpers for the live MTP draft layer.

The helpers in this module deliberately produce CPU-only ``.pt`` payloads.
Once an asynchronous device failure is reported, touching another NPU tensor
can hide the original failure or prevent the reproducer input from reaching
disk.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch

from vllm_ascend.diagnostic_utils import (
    atomic_torch_save as atomic_torch_save,
)
from vllm_ascend.diagnostic_utils import (
    cpu_snapshot as cpu_snapshot,
)
from vllm_ascend.diagnostic_utils import (
    snapshot_cache_components as snapshot_cache_components,
)
from vllm_ascend.diagnostic_utils import (
    tensor_layout as tensor_layout,
)

MTP_DRAFT_DIAG_SCHEMA_VERSION = 1
MTP_DRAFT_DIAG_ROOT = Path(tempfile.gettempdir()) / "vllm_ascend_mtp_draft_diag"


def referenced_block_ids(
    block_table: torch.Tensor | None,
    slot_mappings: Sequence[torch.Tensor | None],
    block_size: int,
    seq_lens: torch.Tensor | None = None,
) -> list[int]:
    """Return physical blocks read or written by one attention invocation."""
    block_ids: set[int] = set()
    if block_table is not None:
        table = block_table.detach().cpu()
        if table.ndim >= 2 and seq_lens is not None:
            lengths = seq_lens.detach().cpu().reshape(-1).tolist()
            rows = []
            for request_index, seq_len in enumerate(lengths):
                if request_index >= table.shape[0]:
                    break
                valid_blocks = (max(int(seq_len), 0) + block_size - 1) // block_size
                rows.append(table[request_index, :valid_blocks].reshape(-1))
            table = torch.cat(rows) if rows else torch.empty(0, dtype=table.dtype)
        else:
            table = table.reshape(-1)
        block_ids.update(int(block_id) for block_id in table.tolist() if int(block_id) >= 0)
    for slot_mapping in slot_mappings:
        if slot_mapping is None:
            continue
        slots = slot_mapping.detach().cpu().reshape(-1)
        block_ids.update(int(slot) // block_size for slot in slots.tolist() if int(slot) >= 0)
    return sorted(block_ids)
