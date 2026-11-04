# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata-kernel arithmetic over block boundaries and multiple programs."""

import pytest
import torch
from test_sfa_async_mtp import Kernel


@pytest.mark.parametrize("n", [0, 1, 14, 31, 32, 33, 64])
@pytest.mark.parametrize("mixed", ["none", "paired", "table"])
@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("check", ["table_reads", "output_stores"])
def test_ascend_mask_lowering_stays_within_allocations(n, mixed, pad, check, monkeypatch):
    # The reported 28-token .ttadapter reads all table lanes before selecting
    # masked values, and expands stride-2 stores into unmasked 64-element RMWs.
    # Model those two observed hazards independently, without NPU dependencies.
    capacity = 2 * (n + int(pad))
    bases = torch.arange(n, dtype=torch.int32) * 127 + 126
    counts = torch.arange(n, dtype=torch.int64) % 2 + 1
    original_bases = bases.clone()
    common = torch.full((n,), -77, dtype=torch.int32)
    target = torch.full((n + 1,), -77, dtype=torch.int32)
    positions = torch.full((capacity + 7,), -77, dtype=torch.int64)
    tables = [
        torch.arange(max(1, n) * stride, dtype=torch.int32).view(max(1, n), stride) for stride in (8192, 8208, 8224)
    ]
    slots = [torch.full((capacity + 7,), -77, dtype=torch.int32) for _ in range(2 if mixed == "none" else 3)]
    table_addresses = {t.data_ptr() for t in tables}
    output_addresses = {t.data_ptr() for t in (positions, *slots)}
    kernel = Kernel([])
    load, store = kernel.tl.load, kernel.tl.store

    def lowered_load(ptr, mask, other):
        if check == "table_reads" and ptr.data.data_ptr() in table_addresses:
            # The installed backend's scalar gather accesses inactive lanes too.
            unmasked = torch.ones_like(torch.as_tensor(ptr.offset), dtype=torch.bool)
            load(ptr, unmasked, other)
        return load(ptr, mask, other)

    def checked_store(ptr, value, mask):
        if check == "output_stores" and ptr.data.data_ptr() in output_addresses:
            assert torch.all(torch.diff(ptr.offset) == 1), "masked stride-2 store expands beyond output capacity"
        store(ptr, value, mask)

    monkeypatch.setattr(kernel.tl, "load", lowered_load)
    monkeypatch.setattr(kernel.tl, "store", checked_store)
    mixed_args = {} if mixed == "none" else {"slots_c8": slots[2], "MIXED_C8": True}
    if mixed == "table":
        mixed_args.update(table_c8=tables[2], stride_c8=8224)
    kernel[((n + 1 + 31) // 32,)](
        bases,
        counts,
        positions,
        common,
        target,
        *tables[:2],
        *slots[:2],
        n,
        capacity,
        n,
        n + 1,
        8192,
        8208,
        128,
        128,
        BLOCK=32,
        **mixed_args,
    )
    expected_base = original_bases + counts.int()
    expected_pos = (expected_base[:, None].long() + torch.arange(2)).flatten()
    torch.testing.assert_close(bases, expected_base)
    torch.testing.assert_close(common, expected_base + 2)
    torch.testing.assert_close(target[:n], common)
    assert target[n:].eq(0).all()
    torch.testing.assert_close(positions[: 2 * n], expected_pos)
    assert positions[2 * n : capacity].eq(0).all() and positions[capacity:].eq(-77).all()
    for index, slot in enumerate(slots):
        table = tables[1] if index == 2 and mixed == "paired" else tables[index]
        block_ids = table[torch.arange(n).repeat_interleave(2), expected_pos // 128]
        if index == 2 and mixed == "paired":
            block_ids = 2 * block_ids
        expected = block_ids * 128 + expected_pos % 128
        torch.testing.assert_close(slot[: 2 * n], expected.int())
        assert slot[2 * n : capacity].eq(-1).all() and slot[capacity:].eq(-77).all()


def test_padding_address_selection_does_not_clamp_active_indices():
    kernel = Kernel([])
    bases = torch.tensor([127], dtype=torch.int32)
    counts = torch.ones(1, dtype=torch.int64)
    positions = torch.empty(2, dtype=torch.int64)
    lengths = [torch.empty(n, dtype=torch.int32) for n in (1, 2)]
    tables = [torch.zeros((1, 1), dtype=torch.int32) for _ in range(2)]
    slots = [torch.empty(2, dtype=torch.int32) for _ in range(2)]
    with pytest.raises(IndexError):
        kernel[(1,)](bases, counts, positions, *lengths, *tables, *slots, 1, 2, 1, 2, 1, 1, 128, 128, BLOCK=32)


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
