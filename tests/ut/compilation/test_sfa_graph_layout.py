# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical regression tests for graph-executed ragged Q1/Q2 mapping."""

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_graph_layout.py"
_SPEC = importlib.util.spec_from_file_location("graph_layout_under_test", _PATH)
layout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(layout)


@pytest.mark.parametrize("capacity", [1, 4, 8, 12, 16])
@pytest.mark.parametrize("pattern", ["q1", "q2", "mixed"])
@pytest.mark.parametrize("padded", [False, True])
def test_topk_request_ownership_and_inverse(capacity, pattern, padded):
    count = max(1, capacity - 1) if padded else capacity
    widths = [1 if pattern == "q1" or (pattern == "mixed" and i % 2) else 2 for i in range(count)]
    total = sum(widths)
    ends = torch.tensor(widths + [0] * (capacity - count), dtype=torch.int32).cumsum(0)
    rows = torch.full((capacity * 2,), -1, dtype=torch.int32)
    rows[:total] = torch.repeat_interleave(torch.arange(count, dtype=torch.int32), torch.tensor(widths))
    topk = torch.arange(capacity * 2 * 7, dtype=torch.int32).reshape(capacity * 2, 1, 7)
    boundaries = torch.arange(capacity * 2, dtype=torch.int32) + 4096
    packed, boundary, owners, inverse = layout.pack_decode_lanes(topk, boundaries, rows, ends, 2)
    offset = 0
    for request, width in enumerate(widths):
        assert torch.equal(packed[request * 2 : request * 2 + width], topk[offset : offset + width])
        assert owners[request * 2 : request * 2 + width].eq(request).all()
        if width == 1:
            assert packed[request * 2 + 1].eq(-1).all()
        offset += width
    assert packed[owners < 0].eq(-1).all()
    assert boundary[owners < 0].eq(0).all()
    # Simulate a request-specific sparse planner changing the returned indices.
    result = layout.unpack_decode_lanes(packed + owners[:, None, None] * 1000, inverse, rows)
    assert torch.equal(result[:total], topk[:total] + rows[:total, None, None] * 1000)
    assert result[total:].eq(-1).all()


def test_attention_padding_preserves_causality_and_addresses():
    table = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    ends = torch.tensor([1, 3, 4, 4], dtype=torch.int32)
    lengths = torch.tensor([4097, 8194, 2001, 0], dtype=torch.int32)
    buffers = layout.FullGraphAttentionBuffers(16, table, table, ends, lengths)
    first = buffers.update(table, table, ends, lengths, lengths, 8)
    addresses = [tensor.data_ptr() for tensor in first]
    assert first[2].tolist() == [1, 3, 4, 4, 8]
    assert first[3].tolist() == [4097, 8194, 2001, 0, 0]
    assert first[0][-1].eq(0).all()
    # Replace requests and change Q1/Q2 lengths without changing graph inputs.
    second = buffers.update(table + 50, table + 80, ends + 1, lengths + 5, lengths + 5, 8)
    assert [tensor.data_ptr() for tensor in second] == addresses
    assert torch.equal(second[0][:-1], table + 50)
    assert second[3][-1] == 0
