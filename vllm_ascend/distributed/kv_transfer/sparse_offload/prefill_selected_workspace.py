# SPDX-License-Identifier: Apache-2.0
"""Physical scratch bounds; KV payload and page tables remain native."""

PAGE_ROWS = 128
PACKED_ROW_BYTES = 656
TOPK = 2048
MIN_HISTORY_TOKENS = 16384
MAX_HISTORY_TOKENS = 524288


def selected_request_capacity(query_capacity: int, max_requests: int) -> int:
    """Each selected request consumes at least one actual query token."""
    return min(max_requests, query_capacity)


def selected_union_capacity(query_capacity: int) -> int:
    """A history union cannot exceed either H or Q times the TopK width."""
    return min(MAX_HISTORY_TOKENS, TOPK * query_capacity)


def selected_request_metadata_shape(query_capacity: int, max_requests: int) -> tuple[int, int]:
    if any(type(value) is not int or value <= 0 for value in (query_capacity, max_requests)):
        raise ValueError("Selected KV workspace dimensions must be positive integers")
    if query_capacity > 8192 or max_requests > 256:
        raise ValueError("Selected KV uses at most 8192 query tokens and 256 requests")
    # Original request row, H, original query start and Q. No payload arena,
    # private page table, compact remap or query/output packing is required.
    return selected_request_capacity(query_capacity, max_requests), 4


def selected_runtime_storage_bytes(query_capacity: int, max_requests: int) -> int:
    """Additional NPU metadata only; union and DDR addresses are charged separately."""
    rows, columns = selected_request_metadata_shape(query_capacity, max_requests)
    return rows * columns * 4
