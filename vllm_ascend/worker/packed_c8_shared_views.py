# SPDX-License-Identifier: Apache-2.0
"""Tensor views for the packed main C8 prefill shared-pool prototype."""

import torch

_BLOCK_TOKENS = 128
_PACKED_ROW_BYTES = 656
_INDEXER_DIM = 128
_MAIN_BLOCKS_PER_BUNDLE = 32
_INDEXER_LOGICAL_BLOCKS_PER_BUNDLE = 82
_C8_INDEXER_EXTENTS_PER_LOGICAL_BLOCK = 2
_C8_INDEXER_BLOCKS_PER_BUNDLE = _INDEXER_LOGICAL_BLOCKS_PER_BUNDLE * _C8_INDEXER_EXTENTS_PER_LOGICAL_BLOCK
_MAIN_ALIGNMENT = 4  # Inline FP32 scales.
_SCALE_ALIGNMENT = 2  # FP16 sidecar.
_RAW_BYTES_PER_BUNDLE = _MAIN_BLOCKS_PER_BUNDLE * _BLOCK_TOKENS * _PACKED_ROW_BYTES
_SCALE_BYTES_PER_BUNDLE = _C8_INDEXER_BLOCKS_PER_BUNDLE * _BLOCK_TOKENS * _SCALE_ALIGNMENT


def _byte_region(tensor: torch.Tensor, name: str, size: int, alignment: int) -> tuple[int, int]:
    if tensor.dtype != torch.int8 or tensor.ndim != 1 or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous one-dimensional INT8 byte tensor")
    if tensor.numel() != size:
        raise ValueError(f"{name} must expose exactly {size} bytes, got {tensor.numel()}")
    offset = tensor.storage_offset()
    if offset < 0 or offset + size > tensor.untyped_storage().nbytes():
        raise ValueError(f"{name} extends outside its backing storage")
    start = tensor.data_ptr()
    if start == 0 or offset % alignment or start % alignment:
        raise ValueError(f"{name} must have an address and offset aligned to {alignment} bytes")
    return start, start + size


def packed_c8_prefill_shared_views(
    raw: torch.Tensor, scale: torch.Tensor, capacity: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed main, LI key and LI scale views over existing byte ranges.

    Capacity counts usable bundles. Both input ranges include reserved bundle
    zero, and may be disjoint slices of the same allocation. Main and LI key
    views intentionally alias the same raw bytes. The allocator must assign
    each live bundle to one group, and block IDs already encode the bank.
    An LI logical block maps to two INT8 extents with corresponding scale
    extents. The helper checks geometry and alignment, while the caller
    provides ownership and valid packed payloads.

    This prototype checks element alignment. Production 2 MiB backing
    alignment and its associated layout budget require integration.
    """
    if type(capacity) is not int or capacity < 0:
        raise ValueError("Packed C8 capacity must be a nonnegative integer")
    if raw.device != scale.device:
        raise ValueError("Packed C8 raw and scale must be on the same device")

    slots = capacity + 1
    raw_start, raw_end = _byte_region(raw, "Packed C8 raw", slots * _RAW_BYTES_PER_BUNDLE, _MAIN_ALIGNMENT)
    scale_start, scale_end = _byte_region(scale, "LI scale", slots * _SCALE_BYTES_PER_BUNDLE, _SCALE_ALIGNMENT)
    if raw_start < scale_end and scale_start < raw_end:
        raise ValueError("Packed C8 raw and LI scale byte ranges must be disjoint")

    indexer_blocks = slots * _C8_INDEXER_BLOCKS_PER_BUNDLE
    return (
        raw.view(slots * _MAIN_BLOCKS_PER_BUNDLE, _BLOCK_TOKENS, 1, _PACKED_ROW_BYTES),
        raw.view(indexer_blocks, _BLOCK_TOKENS, 1, _INDEXER_DIM),
        scale.view(torch.float16).view(indexer_blocks, _BLOCK_TOKENS, 1, 1),
    )


def packed_c8_prefill_mixed_shared_views(
    raw: torch.Tensor, scale: torch.Tensor, capacity: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return main, C8 LI key, C8 LI scale, and BF16 LI key views.

    Main and both LI key views alias the same raw region. An owning LI
    layer uses BF16 block j or C8 extents 2*j and 2*j+1 for logical block
    j. The C8 sidecar uses the same extent IDs as the C8 keys. The caller
    chooses a view using the owning layer's cache format and allocator
    ownership. Creating a view does not assign a bundle or identify an owner.
    """
    main, c8_keys, c8_scales = packed_c8_prefill_shared_views(raw, scale, capacity)
    bf16_blocks = (capacity + 1) * _INDEXER_LOGICAL_BLOCKS_PER_BUNDLE
    bf16_keys = raw.view(torch.bfloat16).view(bf16_blocks, _BLOCK_TOKENS, 1, _INDEXER_DIM)
    return main, c8_keys, c8_scales, bf16_keys
