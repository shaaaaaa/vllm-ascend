# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-only conversion between packed Q1/Q2 rows and fixed planner lanes."""

import torch


class FullGraphAttentionBuffers:
    """Stable attention inputs with one separate, zero-KV padding sequence."""

    def __init__(self, max_requests, block_table, indexer_table, query_ends, seq_lens):
        self.tables = tuple(
            table.new_zeros((max_requests + 1, table.shape[1])) for table in (block_table, indexer_table)
        )
        self.query_ends = query_ends.new_zeros(max_requests + 1)
        self.seq_lens = seq_lens.new_zeros(max_requests + 1)
        self.seq_lens_cpu = torch.zeros(max_requests + 1, dtype=seq_lens.dtype)

    def update(self, block_table, indexer_table, query_ends, seq_lens, seq_lens_cpu, tokens):
        """Append dummy query end without changing any real request's lengths."""
        requests = query_ends.shape[0]
        if requests >= self.query_ends.numel():
            raise ValueError("Full graph attention capacity exceeded")
        tables = []
        for storage, table in zip(self.tables, (block_table, indexer_table)):
            storage[:requests].copy_(table)
            storage[requests].zero_()
            tables.append(storage[: requests + 1])
        self.query_ends[:requests].copy_(query_ends)
        self.query_ends[requests].fill_(tokens)
        self.seq_lens[:requests].copy_(seq_lens)
        self.seq_lens[requests].zero_()
        self.seq_lens_cpu[:requests].copy_(torch.as_tensor(seq_lens_cpu))
        self.seq_lens_cpu[requests].zero_()
        return (
            *tables,
            self.query_ends[: requests + 1],
            self.seq_lens[: requests + 1],
            self.seq_lens_cpu[: requests + 1],
        )


def pack_decode_lanes(topk, boundaries, row_requests, query_ends, width):
    """Group live top-k into request-major lanes without reading device values.

    query_ends excludes the extra attention padding sequence. The planner sees
    width rows per request; missing rows are masked, never borrowed from the
    next request. Returns the planner inputs and a packed-row inverse mapping.
    """
    requests = query_ends.shape[0]
    starts = torch.cat((query_ends.new_zeros(1), query_ends[:-1]))
    indices = starts[:, None] + torch.arange(width, device=topk.device)[None, :]
    safe = indices.clamp(0, topk.shape[0] - 1).long().reshape(-1)
    lanes = torch.arange(requests, device=topk.device)[:, None].expand(-1, width).reshape(-1)
    valid = (indices < query_ends[:, None]).reshape(-1) & (row_requests[safe] == lanes)
    packed = topk[safe]
    packed = torch.where(valid.reshape(-1, *([1] * (topk.ndim - 1))), packed, -1)
    packed_boundary = torch.where(valid, boundaries[safe], 0)
    packed_requests = torch.where(valid, lanes, -1).to(row_requests.dtype)
    safe_requests = row_requests.clamp(0, requests - 1).long()
    offsets = torch.arange(topk.shape[0], device=topk.device) - starts[safe_requests]
    inverse = (safe_requests * width + offsets).clamp(0, requests * width - 1).long()
    return packed, packed_boundary, packed_requests, inverse


def unpack_decode_lanes(planned_topk, inverse, row_requests):
    """Restore packed token order after remapping request-major planner lanes."""
    selected = planned_topk[inverse]
    valid = (row_requests >= 0).reshape(-1, *([1] * (selected.ndim - 1)))
    return torch.where(valid, selected, -1)
