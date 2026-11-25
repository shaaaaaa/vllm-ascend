import math

import torch


class MixedIndexerMetadata:
    """Stable C8 physical views over the mixed pool's common logical IDs."""

    def __init__(
        self,
        max_requests: int,
        max_blocks: int,
        max_tokens: int,
        device: torch.device,
        *,
        block_map: tuple[int, ...] | None = None,
    ):
        self.max_requests, self.max_blocks = max_requests, max_blocks
        self.block_map = torch.tensor(block_map, dtype=torch.int32, device=device) if block_map is not None else None
        self._map_device_metadata = None
        if self.block_map is not None and self.block_map.device.type != "cpu":
            from vllm_ascend.ops.triton.spec_decode.indexer_c8_metadata import map_indexer_metadata

            self._map_device_metadata = map_indexer_metadata
        self.tables = torch.empty(max_requests * max_blocks, dtype=torch.int32, device=device)
        self.slots = torch.empty(max_tokens, dtype=torch.int64, device=device)

    def update(self, table: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (table.ndim != 2 or slots.ndim != 1
                or table.shape[0] > self.max_requests or table.shape[1] > self.max_blocks
                or slots.numel() > self.slots.numel()):
            raise ValueError("Mixed indexer metadata exceeds its preallocated capacity")
        physical_table = self.tables[:table.numel()].view(table.shape)
        physical_slots = self.slots[:slots.numel()]
        if self.block_map is not None:
            if self._map_device_metadata is None:
                physical_table.copy_(torch.where(table >= 0, self.block_map[table.clamp(min=0).long()], table))
                blocks = torch.div(slots.clamp(min=0), 128, rounding_mode="floor").long()
                physical_slots.copy_(torch.where(slots >= 0, self.block_map[blocks].long() * 128 + slots % 128, slots))
            else:
                self._map_device_metadata(table, slots, self.block_map, physical_table, physical_slots)
            return physical_table, physical_slots
        torch.mul(table, 2, out=physical_table)
        # Truncation preserves -1 padding while mapping valid logical slots to
        # the first 128-token half of each BF16-sized shared allocation.
        torch.div(slots, 128, rounding_mode="trunc", out=physical_slots)
        physical_slots.mul_(128).add_(slots)
        return physical_table, physical_slots


def reshape_dsa_shared_pool_raw(
    raw: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    index_head_dim: int,
    *,
    is_indexer: bool,
    indexer_dtype: torch.dtype | None = None,
    indexer_scale: torch.Tensor | None = None,
    shared_indexer_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, ...]:
    """Create latent or indexer PA_BSND views from one DSA shared raw slab."""

    elt = torch.empty((), dtype=dtype).element_size()
    indexer_dtype = dtype if indexer_dtype is None else indexer_dtype
    index_elt = indexer_dtype.itemsize
    shared_elt = (shared_indexer_dtype or indexer_dtype).itemsize
    latent_page = block_size * num_kv_heads * (kv_lora_rank + qk_rope_head_dim) * elt
    indexer_page = block_size * num_kv_heads * index_head_dim * shared_elt
    bundle_page = math.lcm(latent_page, indexer_page)
    assert bundle_page == 9 * indexer_page, (
        "DSA shared pool expects one bundle to be nine indexer pages; "
        f"latent_page={latent_page}, indexer_page={indexer_page}."
    )
    assert raw.numel() % bundle_page == 0
    slot_count = raw.numel() // bundle_page
    latent_blocks = slot_count * (bundle_page // latent_page)
    indexer_blocks = slot_count * (bundle_page // indexer_page)
    nope_pages = latent_blocks * kv_lora_rank * elt // (index_head_dim * shared_elt)
    pe_pages = latent_blocks * qk_rope_head_dim * elt // (index_head_dim * shared_elt)
    assert nope_pages + pe_pages == indexer_blocks

    if is_indexer:
        physical_blocks = indexer_blocks * shared_elt // index_elt
        keys = raw.view(indexer_dtype).view(physical_blocks, block_size, num_kv_heads, index_head_dim)
        if indexer_dtype == torch.int8:
            if indexer_scale is None or indexer_scale.numel() != physical_blocks * block_size * num_kv_heads * 2:
                raise ValueError("Indexer C8 shared storage requires a complete FP16 scale sidecar")
            return keys, indexer_scale.view(torch.float16).view(physical_blocks, block_size, num_kv_heads, 1)
        return (keys,)

    nope_bytes = nope_pages * indexer_page
    pe_bytes = pe_pages * indexer_page
    k_nope = raw[:nope_bytes].view(dtype).view(
        latent_blocks,
        block_size,
        num_kv_heads,
        kv_lora_rank,
    )
    k_pe = raw[nope_bytes : nope_bytes + pe_bytes].view(dtype).view(
        latent_blocks,
        block_size,
        num_kv_heads,
        qk_rope_head_dim,
    )
    return k_nope, k_pe
