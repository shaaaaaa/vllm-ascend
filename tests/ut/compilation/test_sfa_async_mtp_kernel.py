# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata-kernel arithmetic over block boundaries and multiple programs."""

import pytest
import torch
from test_sfa_async_mtp import Kernel


@pytest.mark.parametrize("n,capacity", [(1, 8), (3, 8), (16, 32), (32, 64), (33, 80), (63, 128)])
@pytest.mark.parametrize("alias_lengths", [False, True])
def test_separate_group_strides_and_padding(n, capacity, alias_lengths):
    rows = capacity // 2
    bases = torch.arange(n, dtype=torch.int32) * 17 + 1023
    counts = torch.arange(n, dtype=torch.int64) % 2 + 1
    positions = torch.full((capacity,), -77, dtype=torch.int64)
    target = torch.full((rows + 1,), -77, dtype=torch.int32)
    common = target[:n] if alias_lengths else torch.full((n,), -77, dtype=torch.int32)
    tables = [
        torch.arange(rows * stride, dtype=torch.int32).view(rows, stride) + bias
        for stride, bias in ((64, 100), (96, 9000))
    ]
    slots = [torch.full((capacity,), -77, dtype=torch.int32) for _ in range(2)]
    kernel = Kernel([])
    expected_base = bases.clone()
    for _ in range(4):
        kernel[((rows + 1 + 31) // 32,)](
            bases,
            counts,
            positions,
            common,
            target,
            *tables,
            *slots,
            n,
            capacity,
            len(common),
            len(target),
            64,
            96,
            128,
            64,
            BLOCK=32,
        )
        expected_base += counts.int()
        assert torch.equal(bases, expected_base)
        assert positions[: 2 * n].tolist() == [int(b) + j for b in expected_base for j in (0, 1)]
        assert torch.all(positions[2 * n :] == 0)
        assert torch.equal(common, expected_base + 2)
        assert torch.equal(target[:n], expected_base + 2)
        assert torch.all(target[n:] == 0)
        for table, slot, block_size in zip(tables, slots, (128, 64)):
            oracle = [
                int(table[row, (int(base) + j) // block_size]) * block_size + (int(base) + j) % block_size
                for row, base in enumerate(expected_base)
                for j in (0, 1)
            ]
            assert slot.tolist() == oracle + [-1] * (capacity - 2 * n)
