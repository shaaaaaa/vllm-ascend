# SPDX-License-Identifier: Apache-2.0
"""Torch NPU adapters for the M3 packed main KV."""

from __future__ import annotations

import torch

_PAGE_TOKENS = 128
_NOPE_DIM = 512
_ROPE_DIM = 64
_QUANT_TILE_SIZE = 128
_QUANT_TILES = _NOPE_DIM // _QUANT_TILE_SIZE
_PACKED_ROW_BYTES = _NOPE_DIM + 2 * _ROPE_DIM + 4 * _QUANT_TILES
_PER_TILE_QUANT_MODE = 2
_MLA_ABSORB_MODE = 2
_COMBINED_SCALE_REPO_MODE = 1
_CAUSAL_SPARSE_MODE = 3


def _validate_packed_main(packed_main: torch.Tensor) -> None:
    if (
        packed_main.ndim != 4
        or packed_main.shape[1:] != (_PAGE_TOKENS, 1, _PACKED_ROW_BYTES)
        or packed_main.dtype != torch.int8
        or not packed_main.is_contiguous()
    ):
        raise ValueError("Packed C8 main KV must be contiguous INT8 [blocks, 128, 1, 656]")


def append_packed_c8_main_kv(
    packed_main: torch.Tensor,
    projected_kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    *,
    epsilon: float,
) -> None:
    """Append live BF16 MLA tokens to their allocator-owned packed main slots.

    Before calling, slice every token input and the physical slot map to
    num_actual_tokens. The slots must be unique, in range, and owned by this
    write. This excludes both padding and reserved/shared pool slots. CPU
    inputs receive a range/duplicate check. Device inputs rely on the
    caller's allocation and padding metadata, so no device values are read
    by the host. Production attention does not call this adapter yet.
    """
    _validate_packed_main(packed_main)
    if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("Packed C8 append requires a one-dimensional integer slot map")
    tokens = slots.numel()
    if (
        projected_kv.shape != (tokens, 1, 1, _NOPE_DIM + _ROPE_DIM)
        or gamma.shape != (_NOPE_DIM,)
        or cos.shape != (tokens, 1, 1, _ROPE_DIM)
        or sin.shape != cos.shape
        or any(tensor.dtype != torch.bfloat16 for tensor in (projected_kv, gamma, cos, sin))
    ):
        raise ValueError("Packed C8 append requires BF16 projected KV, RMS weight, and per-token RoPE tables")
    if any(tensor.device != packed_main.device for tensor in (projected_kv, gamma, cos, sin, slots)):
        raise ValueError("Packed C8 append inputs and cache must be on the same device")
    if tokens == 0:
        return
    if packed_main.device.type == "cpu" and (
        torch.any((slots < 0) | (slots >= packed_main.shape[0] * _PAGE_TOKENS))
        or torch.unique(slots).numel() != tokens
    ):
        raise ValueError("Packed C8 append slots must be unique and in range, without padding")

    import torch_npu

    nope, rope = projected_kv.split((_NOPE_DIM, _ROPE_DIM), dim=-1)
    normalized, _ = torch_npu.npu_rms_norm(nope, gamma, epsilon=epsilon)
    rotated = torch_npu.npu_interleave_rope(rope, cos, sin)
    quantized, scales = torch_npu.npu_dynamic_block_quant(
        normalized.reshape(tokens, 1, _NOPE_DIM),
        dst_type=torch.int8,
        row_block_size=1,
        col_block_size=_QUANT_TILE_SIZE,
    )
    if (
        quantized.shape != (tokens, 1, _NOPE_DIM)
        or quantized.dtype != torch.int8
        or rotated.shape != rope.shape
        or rotated.dtype != torch.bfloat16
        or scales.shape != (tokens, 1, _QUANT_TILES)
        or not scales.is_floating_point()
        or any(tensor.device != packed_main.device for tensor in (quantized, rotated, scales))
    ):
        raise ValueError("Packed C8 append received incompatible RoPE or quantizer output")
    rows = torch.cat(
        (
            quantized.reshape(tokens, _NOPE_DIM),
            rotated.reshape(tokens, _ROPE_DIM).contiguous().view(torch.int8),
            scales.to(torch.float32).reshape(tokens, _QUANT_TILES).contiguous().view(torch.int8),
        ),
        dim=-1,
    )
    torch_npu.npu_scatter_nd_update_(packed_main.view(-1, _PACKED_ROW_BYTES), slots.contiguous().view(-1, 1), rows)


def packed_c8_sparse_flash_attention(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    packed_main: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    scale: float,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
) -> torch.Tensor:
    """Run TND attention over contiguous PA_BSND packed main pages.

    The caller supplies the validated main cache with INT8 NOPE, BF16 RoPE,
    and inline FP32 scales. Query lengths are cumulative TND end offsets.
    The block table, KV lengths, and topK must refer to the same request order
    and cache layout. This adapter validates tensor metadata without reading
    device values. torch_npu validates the supported topK shape.
    """
    _validate_packed_main(packed_main)
    if (
        ql_nope.ndim != 3
        or q_pe.ndim != 3
        or ql_nope.shape[:-1] != q_pe.shape[:-1]
        or ql_nope.shape[-1] != _NOPE_DIM
        or q_pe.shape[-1] != _ROPE_DIM
        or ql_nope.dtype != torch.bfloat16
        or q_pe.dtype != torch.bfloat16
    ):
        raise ValueError("Packed C8 attention requires matching BF16 TND NOPE (512) and RoPE (64) queries")
    if (
        block_table.ndim != 2
        or actual_seq_lengths_query.ndim != 1
        or actual_seq_lengths_kv.ndim != 1
        or block_table.shape[0] != actual_seq_lengths_query.shape[0]
        or actual_seq_lengths_query.shape != actual_seq_lengths_kv.shape
    ):
        raise ValueError("Packed C8 block table and query/KV lengths must describe the same batch")
    if any(
        tensor.device != packed_main.device
        for tensor in (ql_nope, q_pe, topk_indices, block_table, actual_seq_lengths_query, actual_seq_lengths_kv)
    ):
        raise ValueError("Packed C8 queries, cache, and attention metadata must be on the same device")

    # Import the device backend only when this path is called.
    import torch_npu

    quant_sfa = getattr(torch_npu, "npu_kv_quant_sparse_flash_attention", None)
    if not callable(quant_sfa):
        raise RuntimeError("torch_npu does not provide npu_kv_quant_sparse_flash_attention")

    return quant_sfa(
        query=torch.cat((ql_nope, q_pe), dim=-1),
        key=packed_main,
        value=packed_main,
        sparse_indices=topk_indices,
        scale_value=scale,
        key_quant_mode=_PER_TILE_QUANT_MODE,
        value_quant_mode=_PER_TILE_QUANT_MODE,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=_CAUSAL_SPARSE_MODE,
        attention_mode=_MLA_ABSORB_MODE,
        quant_scale_repo_mode=_COMBINED_SCALE_REPO_MODE,
        tile_size=_QUANT_TILE_SIZE,
        rope_head_dim=_ROPE_DIM,
    )
