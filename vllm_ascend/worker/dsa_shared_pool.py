import math

import torch


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
) -> tuple[torch.Tensor, ...]:
    """Create latent or indexer PA_BSND views from one DSA shared raw slab."""

    elt = torch.empty((), dtype=dtype).element_size()
    indexer_dtype = dtype if indexer_dtype is None else indexer_dtype
    index_elt = indexer_dtype.itemsize
    latent_page = block_size * num_kv_heads * (kv_lora_rank + qk_rope_head_dim) * elt
    indexer_page = block_size * num_kv_heads * index_head_dim * index_elt
    bundle_page = math.lcm(latent_page, indexer_page)
    assert bundle_page == 9 * indexer_page, (
        "DSA shared pool expects one bundle to be nine indexer pages; "
        f"latent_page={latent_page}, indexer_page={indexer_page}."
    )
    assert raw.numel() % bundle_page == 0
    slot_count = raw.numel() // bundle_page
    latent_blocks = slot_count * (bundle_page // latent_page)
    indexer_blocks = slot_count * (bundle_page // indexer_page)
    nope_pages = latent_blocks * kv_lora_rank * elt // (index_head_dim * index_elt)
    pe_pages = latent_blocks * qk_rope_head_dim * elt // (index_head_dim * index_elt)
    assert nope_pages + pe_pages == indexer_blocks

    if is_indexer:
        keys = raw.view(indexer_dtype).view(indexer_blocks, block_size, num_kv_heads, index_head_dim)
        if indexer_dtype == torch.int8:
            if indexer_scale is None or indexer_scale.numel() != indexer_blocks * block_size * num_kv_heads * 2:
                raise ValueError("Indexer C8 shared storage requires a complete FP16 scale sidecar")
            return keys, indexer_scale.view(torch.float16).view(indexer_blocks, block_size, num_kv_heads, 1)
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
