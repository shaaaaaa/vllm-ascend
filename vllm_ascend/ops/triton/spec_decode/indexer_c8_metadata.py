# SPDX-License-Identifier: Apache-2.0
"""One ordinary-metadata launch for the paired-bank physical table and slots."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _map_indexer_metadata(
    table,
    slots,
    mapping,
    out_table,
    out_slots,
    TABLE_SIZE: tl.constexpr,
    SLOT_SIZE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(table + (i // TABLE_WIDTH) * TABLE_STRIDE + i % TABLE_WIDTH, i < TABLE_SIZE, other=-1)
    physical = tl.load(mapping + b, (i < TABLE_SIZE) & (b >= 0), other=-1)
    tl.store(out_table + i, physical, i < TABLE_SIZE)
    s = tl.load(slots + i, i < SLOT_SIZE, other=-1)
    physical = tl.load(mapping + s // 128, (i < SLOT_SIZE) & (s >= 0), other=-1)
    tl.store(out_slots + i, tl.where(s >= 0, physical * 128 + s % 128, -1), i < SLOT_SIZE)


def map_indexer_metadata(
    table: torch.Tensor,
    slots: torch.Tensor,
    mapping: torch.Tensor,
    out_table: torch.Tensor,
    out_slots: torch.Tensor,
) -> None:
    """Map a padded block table and token slots into stable C8 buffers."""
    _map_indexer_metadata[(triton.cdiv(max(table.numel(), slots.numel()), 256),)](
        table,
        slots,
        mapping,
        out_table,
        out_slots,
        table.numel(),
        slots.numel(),
        table.shape[1],
        table.stride(0),
        BLOCK=256,
    )
