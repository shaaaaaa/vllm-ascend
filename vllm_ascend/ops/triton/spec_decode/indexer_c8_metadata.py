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
    start = tl.program_id(0) * BLOCK
    i = start + tl.arange(0, BLOCK)
    if start < TABLE_SIZE:
        if TABLE_WIDTH == TABLE_STRIDE:
            b = tl.load(table + i, i < TABLE_SIZE, other=0)
        else:
            # Ascend can issue discrete reads before applying a load mask.
            # Select a valid address before each gather, including tail lanes.
            j = tl.where(i < TABLE_SIZE, i, 0)
            b = tl.load(table + (j // TABLE_WIDTH) * TABLE_STRIDE + j % TABLE_WIDTH, True, other=0)
        physical = tl.load(mapping + tl.where(b >= 0, b, 0), True, other=0)
        tl.store(out_table + i, tl.where(b >= 0, physical, -1), i < TABLE_SIZE)
    if start < SLOT_SIZE:
        s = tl.load(slots + i, i < SLOT_SIZE, other=0)
        physical = tl.load(mapping + tl.where(s >= 0, s, 0) // 128, True, other=0)
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
