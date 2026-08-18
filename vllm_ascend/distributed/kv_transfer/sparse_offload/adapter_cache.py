# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA latent hot-cache backed by the colleague's ``KVCacheAdapter``.

This wires the on-NPU latent pool (the adapter) into the DSA decode path so the
sparse-attention kernel reads the resident pool *in place* (zero-copy), with the
adapter handling residency (hit/miss) and eviction-to-LMCache.

Design (aligned in review, see DESIGN.md):
  * Two pools per layer (``k_nope`` of ``kv_lora_rank`` and ``k_pe`` of
    ``qk_rope_head_dim``) share **one** slot mapping — block ``i`` lives at the
    same physical slot in both pools. This mirrors how vLLM already allocates the
    MLA latent as two independent contiguous tensors (``kv_cache[0]``/``[1]``).
  * The pool tensor has **no request dimension**. Requests are distinguished by
    the per-request rows of the kernel ``block_table`` (native paged-attention
    mechanism); the adapter keeps requests disjoint because the logical block id
    encodes the request slot.
  * Retrieve: topk absolute positions -> logical block ids -> ``adapter.load`` ->
    physical slots -> scatter into ``block_table``; ``sparse_indices`` stays the
    original absolute positions (native contract, block_table swapped to point at
    the pool). No data copy.
  * Insert (decode token): ``adapter.load(load_missing=False)`` hands back a slot;
    the caller writes the new token's latent straight into ``pool[slot, offset]``.
    The block is resident from allocation, so a later topk hit serves it directly.
  * Eviction (pool full) is handled inside the adapter: it writes the evicted
    block back through its backend (LMCache for the real run).

Everything here is parameter-driven via :class:`AdapterCacheConfig`; nothing about
layer count / concurrency / sizes is hard-coded. The module depends only on
``torch`` and the ``kv_cache_adapter`` package, so the CPU parity test runs without
vLLM or an NPU.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch


def _import_kv_cache_adapter():
    """Import the ``kv_cache_adapter`` package, self-locating if it isn't on the path.

    Assumes ``kv_cache_adapter`` sits next to the vllm-ascend repo root (the layout
    here: ``<root>/kv_cache_adapter`` and ``<root>/vllm-ascend/...``). Falls back to
    appending ``<root>`` to ``sys.path`` (or ``KV_CACHE_ADAPTER_PARENT``) so no
    PYTHONPATH change is needed. Append-only: never disturbs other packages' paths.
    """
    try:
        import kv_cache_adapter as _mod  # noqa: PLC0415
        return _mod
    except ImportError:
        parent = os.environ.get("KV_CACHE_ADAPTER_PARENT") or os.path.abspath(
            os.path.join(os.path.dirname(__file__), *([os.pardir] * 5))
        )
        if parent not in sys.path:
            sys.path.append(parent)
        import kv_cache_adapter as _mod  # noqa: PLC0415
        return _mod


_kvca = _import_kv_cache_adapter()
BlockStoreBackend = _kvca.BlockStoreBackend
InMemoryBlockStoreBackend = _kvca.InMemoryBlockStoreBackend
LMCacheBackend = _kvca.LMCacheBackend
KVCacheAdapter = _kvca.KVCacheAdapter

# Profiling hooks are no-op in production builds. Guarded so the CPU parity test,
# which has no vLLM, falls back to the same no-op context.
try:
    from vllm_ascend.distributed.kv_transfer.sparse_offload import _prof as _aprof  # noqa: PLC0415
except Exception:  # pragma: no cover
    import contextlib

    class _aprof:  # type: ignore
        @staticmethod
        def section(_name):
            return contextlib.nullcontext()

ID_DTYPE = torch.int64
INVALID_POSITION = -1


@dataclass
class AdapterCacheConfig:
    """All sizing is derived from these fields; callers pass real values in."""

    layer_names: list[str]
    kv_lora_rank: int          # k_nope width (kv_cache[0])
    qk_rope_head_dim: int      # k_pe width (kv_cache[1])
    block_size: int            # tokens per block; align with LMCache chunk size
    topk: int                  # indexer index_topk (selected tokens per query)
    max_model_len: int         # bounds blocks-per-request (logical id space)
    max_num_seqs: int          # bounds the req-slot id space (recycled on free)
    pool_concurrency_cap: int  # concurrency the pool is sized to hold without thrash
    pool_ratio: float          # headroom over the working set (>=1; more -> more reuse)
    dtype: torch.dtype
    device: torch.device

    @property
    def latent_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def blocks_per_req(self) -> int:
        return math.ceil(self.max_model_len / self.block_size)

    @property
    def num_logical_blocks(self) -> int:
        # req-slot space x blocks-per-request; the id space, not physical memory.
        return self.max_num_seqs * self.blocks_per_req

    @property
    def num_actual_blocks(self) -> int:
        # physical slots per layer = ceil(ratio * cap * topk / block_size).
        tokens = self.pool_ratio * self.pool_concurrency_cap * self.topk
        return max(1, math.ceil(tokens / self.block_size))


@dataclass
class RetrieveResult:
    """Kernel arguments for ``npu_sparse_flash_attention`` (pool read in place)."""

    knope_pool: torch.Tensor          # (num_actual_blocks, block_size, 1, kv_lora_rank)
    kpe_pool: torch.Tensor            # (num_actual_blocks, block_size, 1, qk_rope_head_dim)
    block_table: torch.Tensor         # [b, blocks_per_req] int32, pool slot per req-block
    sparse_indices: torch.Tensor      # [b, topk] int32, absolute positions (-1 padded)
    seq_lens: torch.Tensor            # [b] int32, valid selected tokens per request
    loaded_ids: torch.Tensor          # unique logical ids pinned by this call (for release)


class _LayerState:
    __slots__ = ("knope_pool", "kpe_pool", "adapter", "backend")

    def __init__(self, knope_pool, kpe_pool, adapter, backend):
        self.knope_pool = knope_pool
        self.kpe_pool = kpe_pool
        self.adapter = adapter
        self.backend = backend


class AdapterLatentCache:
    """Per-layer adapter-backed latent hot cache (decision: one adapter per layer)."""

    def __init__(
        self,
        config: AdapterCacheConfig,
        backend_factory: Callable[[str], BlockStoreBackend] | None = None,
    ) -> None:
        self.config = config
        self._layers: dict[str, _LayerState] = {}
        self._req_slot_of: dict[str, int] = {}
        self._free_req_slots: list[int] = list(range(config.max_num_seqs))
        # remembers the slot of the decode block each request is currently filling,
        # so per-token writes don't re-allocate within a block.
        self._decode_block: dict[tuple[str, str], tuple[int, int]] = {}

        for layer_name in config.layer_names:
            knope_pool = torch.zeros(
                (config.num_actual_blocks, config.block_size, 1, config.kv_lora_rank),
                dtype=config.dtype,
                device=config.device,
            )
            kpe_pool = torch.zeros(
                (config.num_actual_blocks, config.block_size, 1, config.qk_rope_head_dim),
                dtype=config.dtype,
                device=config.device,
            )
            backend = (
                backend_factory(layer_name)
                if backend_factory is not None
                else InMemoryBlockStoreBackend(num_logical_blocks=config.num_logical_blocks)
            )
            adapter = KVCacheAdapter(
                config.num_actual_blocks,
                config.num_logical_blocks,
                [knope_pool, kpe_pool],
                backend,
            )
            self._layers[layer_name] = _LayerState(knope_pool, kpe_pool, adapter, backend)

    # ----------------------------------------------------------------- sizing
    def reserved_bytes(self) -> int:
        cfg = self.config
        elt = torch.empty((), dtype=cfg.dtype).element_size()
        per_layer = cfg.num_actual_blocks * cfg.block_size * cfg.latent_dim * elt
        return per_layer * len(cfg.layer_names)

    # ----------------------------------------------------------- req-slot map
    def req_slot(self, req_id: str) -> int:
        slot = self._req_slot_of.get(req_id)
        if slot is None:
            if not self._free_req_slots:
                raise RuntimeError("no free req-slot; exceeded max_num_seqs")
            slot = self._free_req_slots.pop()
            self._req_slot_of[req_id] = slot
        return slot

    def free_request(self, req_id: str) -> None:
        slot = self._req_slot_of.get(req_id)
        if slot is None:
            return
        # Release the still-pinned current decode block of every layer so the pool
        # can reclaim/spill it; then recycle the req-slot id.
        for layer_name, layer in self._layers.items():
            cached = self._decode_block.pop((req_id, layer_name), None)
            if cached is not None:
                logical = torch.tensor(
                    [cached[0] + slot * self.config.blocks_per_req], dtype=ID_DTYPE, device=self.config.device
                )
                layer.adapter.release(logical)
        self._req_slot_of.pop(req_id, None)
        self._free_req_slots.append(slot)

    def req_slots_tensor(self, req_ids: list[str]) -> torch.Tensor:
        """req-slot per row for the current batch; recycles slots of departed reqs.

        v1 lifecycle without a model-runner finished-hook: any req previously seen
        but absent from the current batch is treated as finished and freed. NOTE:
        this also frees preempted/swapped-out requests (bring-up limitation); a
        returning request gets a fresh slot and re-fetches from the backend.
        """
        present = set(req_ids)
        for gone in [r for r in self._req_slot_of if r not in present]:
            self.free_request(gone)
        return torch.tensor([self.req_slot(r) for r in req_ids], dtype=ID_DTYPE, device=self.config.device)

    # ----------------------------------------------------------- logical ids
    def _logical(self, req_slot: int, positions: torch.Tensor) -> torch.Tensor:
        """absolute positions -> global logical block ids (request-scoped)."""
        return positions // self.config.block_size + req_slot * self.config.blocks_per_req

    # --------------------------------------------------------------- prefill
    def store_prefill(
        self,
        layer_name: str,
        req_ids: list[str],
        query_start_loc: torch.Tensor,  # [b+1] CSR offsets into packed k_nope/k_pe
        context_lens: torch.Tensor,     # [b] tokens already computed (chunked prefill)
        k_nope: torch.Tensor,           # [num_tokens, kv_lora_rank]
        k_pe: torch.Tensor,             # [num_tokens, qk_rope_head_dim]
    ) -> None:
        """Assemble full blocks of this layer's prompt latent and save them into the
        adapter backend so decode-time ``retrieve`` can fetch prefill-selected blocks.

        Boundary block (the partial last block of a request) is stored as-is; decode
        later appends to the same block in the pool. Limitation: if that block is
        evicted before decode resumes, its prefill rows reload from here (correct) but
        a brand-new ``insert`` would re-allocate an empty slot — keep the pool large
        enough during bring-up so the boundary block stays resident.
        """
        cfg = self.config
        bs = cfg.block_size
        layer = self._layers[layer_name]
        qsl = query_start_loc.to(torch.long).tolist()
        ctx = context_lens.to(torch.long).tolist()
        kn = k_nope.reshape(-1, cfg.kv_lora_rank)
        kp = k_pe.reshape(-1, cfg.qk_rope_head_dim)

        # Vectorized per request: scatter this chunk's tokens into a block-aligned
        # buffer and reshape into whole blocks — NO per-token Python loop (that made
        # a 120k prefill launch ~prompt_len*num_layers tiny device ops and froze the
        # engine). Assumes chunk starts are block-aligned (true when
        # max_num_batched_tokens is a multiple of block_size), so blocks never split
        # across chunks; the only partial block is the prompt's last one (zero-padded).
        ids_all: list[torch.Tensor] = []
        kn_blocks_all: list[torch.Tensor] = []
        kp_blocks_all: list[torch.Tensor] = []
        for b, req_id in enumerate(req_ids):
            lo, hi = qsl[b], qsl[b + 1]
            length = hi - lo
            if length <= 0:
                continue
            req_slot = self.req_slot(req_id)
            c = ctx[b]
            blk_lo = c // bs
            blk_hi = (c + length - 1) // bs
            n_blk = blk_hi - blk_lo + 1
            start_off = c - blk_lo * bs  # first token's offset within the first block

            buf_kn = kn.new_zeros((n_blk * bs, cfg.kv_lora_rank))
            buf_kp = kp.new_zeros((n_blk * bs, cfg.qk_rope_head_dim))
            buf_kn[start_off : start_off + length] = kn[lo:hi]
            buf_kp[start_off : start_off + length] = kp[lo:hi]

            ids = torch.arange(blk_lo, blk_hi + 1, device=cfg.device, dtype=ID_DTYPE)
            ids = ids + req_slot * cfg.blocks_per_req
            ids_all.append(ids)
            kn_blocks_all.append(buf_kn.view(n_blk, bs, 1, cfg.kv_lora_rank))
            kp_blocks_all.append(buf_kp.view(n_blk, bs, 1, cfg.qk_rope_head_dim))

        if ids_all:
            # put_blocks(logical_ids, [knope_blocks, kpe_blocks]): store whole prompt
            # blocks by logical id. Both backends (InMemory / LMCache) expose this;
            # decode-time retrieve fetches these on a pool miss.
            layer.backend.put_blocks(
                torch.cat(ids_all),
                [torch.cat(kn_blocks_all), torch.cat(kp_blocks_all)],
            )

    # -------------------------------------------------------------- retrieve
    def retrieve(
        self,
        layer_name: str,
        req_slots: torch.Tensor,       # [b] int64, req-slot per batch row
        topk_positions: torch.Tensor,  # [b, topk] int64, absolute positions, -1 padded
    ) -> RetrieveResult:
        cfg = self.config
        layer = self._layers[layer_name]
        dev = cfg.device
        bs = cfg.block_size
        b, topk = topk_positions.shape

        # ACL graph pads the query/topk dim up to a captured size, so topk_positions can
        # have MORE rows (b, padded) than there are real requests (n_real). The kernel
        # wants MIXED shapes (exactly what native passes): sparse_indices at the padded
        # query length b (to match the padded query tensor), but actual_seq_lengths_kv
        # and block_table at the REAL request count. So: pad req_slots to broadcast,
        # mask the padding query tokens (they select nothing), keep sparse_indices at b,
        # and slice the per-request outputs (block_table, seq_lens) to n_real at return.
        n_real = req_slots.shape[0]
        if n_real < b:
            req_slots = torch.cat([req_slots, req_slots.new_zeros(b - n_real)])
            topk_positions = topk_positions.clone()
            topk_positions[n_real:] = INVALID_POSITION

        valid = topk_positions >= 0
        local_block = torch.where(valid, topk_positions // bs, torch.zeros_like(topk_positions))
        logical = local_block + req_slots[:, None] * cfg.blocks_per_req  # [b, topk]
        logical = torch.where(valid, logical, torch.full_like(logical, -1))

        with _aprof.section("ad_ret_dedup"):
            valid_logical = logical[valid]
            has_valid = valid_logical.numel() > 0
            unique_ids = (
                torch.unique(valid_logical).to(ID_DTYPE)
                if has_valid
                else logical.new_zeros((0,), dtype=ID_DTYPE)
            )
        if has_valid:
            with _aprof.section("ad_ret_load"):
                slots = layer.adapter.load(unique_ids, load_missing=True)
        else:
            slots = unique_ids.clone()

        # dense logical-id -> slot lookup (only loaded ids are valid).
        with _aprof.section("ad_ret_map"):
            slot_of = torch.zeros(cfg.num_logical_blocks, dtype=ID_DTYPE, device=dev)
            slot_of.index_put_((unique_ids,), slots.to(ID_DTYPE))  # empty index = no-op
            sel_slot = slot_of[logical.clamp(min=0)]                # [b, topk]

        # STATIC block_table build: scatter every (b, col) -> slot with no boolean
        # masking and no valid.any() host read (both forced device syncs before).
        # Invalid (padding) entries go to a throwaway trash column that is then sliced
        # off, so they can't corrupt real block-table slots. Duplicate (b, col) pairs
        # (multiple selected tokens in one block) write the same slot -> idempotent.
        with _aprof.section("ad_ret_bt"):
            trash_col = cfg.blocks_per_req
            col = torch.where(valid, local_block, torch.full_like(local_block, trash_col))
            bt_ext = torch.zeros(b, cfg.blocks_per_req + 1, dtype=ID_DTYPE, device=dev)
            bt_ext.scatter_(1, col, sel_slot)
            block_table = bt_ext[:, : cfg.blocks_per_req]

        return RetrieveResult(
            knope_pool=layer.knope_pool,
            kpe_pool=layer.kpe_pool,
            # per-request -> real batch (n_real); per-query-token -> padded length (b).
            block_table=block_table[:n_real].to(torch.int32),
            sparse_indices=topk_positions.to(torch.int32),
            seq_lens=valid.sum(dim=1)[:n_real].to(torch.int32),
            loaded_ids=unique_ids,
        )

    def release_after_fa(self, layer_name: str, loaded_ids: torch.Tensor) -> None:
        if loaded_ids.numel() == 0:
            return
        self._layers[layer_name].adapter.release(loaded_ids)

    # ---------------------------------------------------------------- insert
    def insert_decode_token(
        self,
        layer_name: str,
        req_id: str,
        position: int,
        k_nope: torch.Tensor,   # (kv_lora_rank,)
        k_pe: torch.Tensor,     # (qk_rope_head_dim,)
    ) -> bool:
        """Write one generated token's latent into the pool.

        Returns True iff this call ran adapter metadata kernels (it started a fresh
        block: release prev + load + mark_dirty). On a normal in-block step it only
        does an ordered pool write -- no metadata kernel -- so the caller can skip the
        insert->retrieve sync on those steps.

        ``mark_dirty`` is issued ONCE per block (at allocation), not per token: the
        block is pinned for its whole fill window so it can't be evicted/refetched
        mid-fill, the dirty bit persists, and every token write lands in the same
        resident slot -- one mark covers the whole block's spill-on-eviction.
        """
        cfg = self.config
        layer = self._layers[layer_name]
        req_slot = self.req_slot(req_id)
        block_local = position // cfg.block_size
        offset = position % cfg.block_size
        key = (req_id, layer_name)

        cached = self._decode_block.get(key)
        new_block = cached is None or cached[0] != block_local
        if new_block:
            logical = torch.tensor(
                [block_local + req_slot * cfg.blocks_per_req], dtype=ID_DTYPE, device=cfg.device
            )
            # Starting a fresh block: unpin the previous (now-complete) block so it
            # becomes evictable, then allocate a slot for the new one (no fetch).
            if cached is not None:
                prev_logical = torch.tensor(
                    [cached[0] + req_slot * cfg.blocks_per_req], dtype=ID_DTYPE, device=cfg.device
                )
                layer.adapter.release(prev_logical)
            slot = int(layer.adapter.load(logical, load_missing=False)[0])
            self._decode_block[key] = (block_local, slot)
            # Decode-written latent is NOT in the backend; mark the block dirty once at
            # allocation so eviction spills it to LMCache (see docstring).
            layer.adapter.mark_dirty(logical)
        else:
            slot = cached[1]

        layer.knope_pool[slot, offset, 0, :] = k_nope.to(cfg.dtype)
        layer.kpe_pool[slot, offset, 0, :] = k_pe.to(cfg.dtype)
        return new_block


# ----------------------------------------------------------------------------
# vLLM glue (imported lazily so the CPU parity test never needs vLLM installed).
# ----------------------------------------------------------------------------

def is_adapter_cache_enabled(vllm_config) -> bool:
    """True iff the adapter-cache flag is on AND this is a DSA model."""
    from vllm_ascend import envs as envs_ascend  # noqa: PLC0415

    if not getattr(envs_ascend, "VLLM_ASCEND_DSA_USE_ADAPTER_CACHE", 0):
        return False
    return hasattr(vllm_config.model_config.hf_text_config, "index_topk")


def config_from_vllm(vllm_config, layer_names: list[str], device) -> AdapterCacheConfig | None:
    """Build :class:`AdapterCacheConfig` from vLLM config + env knobs (or None)."""
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE  # noqa: PLC0415

    from vllm_ascend import envs as envs_ascend  # noqa: PLC0415

    if not is_adapter_cache_enabled(vllm_config):
        return None
    hf = vllm_config.model_config.hf_text_config
    cache_config = vllm_config.cache_config
    sched = vllm_config.scheduler_config

    cache_dtype = cache_config.cache_dtype
    dtype = vllm_config.model_config.dtype if cache_dtype == "auto" else STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]

    # block_size aligned to the LMCache chunk (miss-fetch = one clean chunk).
    block_size = int(os.getenv("LMCACHE_CHUNK_SIZE", str(cache_config.block_size)))
    cap = envs_ascend.VLLM_ASCEND_DSA_ADAPTER_CONCURRENCY_CAP or sched.max_num_seqs
    ratio = envs_ascend.VLLM_ASCEND_DSA_ADAPTER_POOL_RATIO

    return AdapterCacheConfig(
        layer_names=list(layer_names),
        kv_lora_rank=hf.kv_lora_rank,
        qk_rope_head_dim=hf.qk_rope_head_dim,
        block_size=block_size,
        topk=hf.index_topk,
        max_model_len=vllm_config.model_config.max_model_len,
        max_num_seqs=sched.max_num_seqs,
        pool_concurrency_cap=int(cap),
        pool_ratio=float(ratio),
        dtype=dtype,
        device=torch.device(device),
    )


def build_adapter_cache(vllm_config, layer_names: list[str], device) -> AdapterLatentCache | None:
    """Construct the per-layer adapter cache, or None if off.

    Backend selection (env ``VLLM_ASCEND_DSA_USE_LMCACHE_BACKEND``):
      * off (default): in-memory reference backend — keeps the CPU parity path and a
        no-LMCache A/B baseline.
      * on: one ``LMCacheBackend`` per layer (host KV store). Evicted pool blocks
        spill to LMCache and pool misses reload from it. Each layer gets its own
        ``lmcache_instance_id`` so their key spaces never collide.
    """
    cfg = config_from_vllm(vllm_config, layer_names, device)
    if cfg is None:
        return None

    from vllm_ascend import envs as envs_ascend  # noqa: PLC0415

    if not envs_ascend.VLLM_ASCEND_DSA_USE_LMCACHE_BACKEND:
        return AdapterLatentCache(cfg)

    # The two pools' per-block shapes (the pool tensor is [num_actual_blocks, *shape]).
    block_shapes = [
        (cfg.block_size, 1, cfg.kv_lora_rank),     # k_nope
        (cfg.block_size, 1, cfg.qk_rope_head_dim),  # k_pe
    ]

    # Per-layer host budget. The pinned pool holds one bundle per logical block; size
    # it from that need (with headroom) unless the env overrides. Mirrors the adapter's
    # 32B-aligned bundle layout so the auto value never under-sizes the allocator.
    def _align32(n: int) -> int:
        return (n + 31) // 32 * 32

    elt = torch.empty((), dtype=cfg.dtype).element_size()
    knope_nbytes = _align32(cfg.block_size * cfg.kv_lora_rank * elt)
    bundle_nbytes = _align32(knope_nbytes + cfg.block_size * cfg.qk_rope_head_dim * elt)
    pinned_need_bytes = cfg.num_logical_blocks * bundle_nbytes

    cpu_gb = float(envs_ascend.VLLM_ASCEND_DSA_LMCACHE_CPU_GB)
    if cpu_gb <= 0.0:
        cpu_gb = max(0.1, pinned_need_bytes * 1.5 / (1024 ** 3))  # 50% headroom

    def _lmcache_factory(layer_name: str) -> BlockStoreBackend:
        # NPU native path: the pool tensors live on the NPU, so save_slots / load_blocks
        # auto-dispatch to the native block_bundle kernels (no host index_select round
        # trip). Passing num_logical_blocks preallocates one pinned host bundle per
        # logical block up front and reuses them, instead of allocating pinned host on
        # every eviction. Each layer gets its own lmcache_instance_id (disjoint keys).
        # Requires the built kv_cache_adapter NPU extension and lmcache(+lmcache_ascend).
        return LMCacheBackend(
            block_shape=block_shapes,
            block_dtype=[cfg.dtype, cfg.dtype],
            num_logical_blocks=cfg.num_logical_blocks,
            max_local_cpu_size_gb=cpu_gb,
            lmcache_instance_id=f"dsa_adapter::{layer_name}",
        )

    return AdapterLatentCache(cfg, backend_factory=_lmcache_factory)


def reserved_bytes_from_vllm(vllm_config) -> int:
    """KV-budget bytes to subtract for the adapter pools (0 if disabled).

    Mirrors ``runner_integration.maybe_reserved_bytes``; layer count comes from
    ``num_hidden_layers`` here (the per-layer pools are sized identically).
    """
    cfg = config_from_vllm(
        vllm_config,
        ["l"] * int(vllm_config.model_config.hf_text_config.num_hidden_layers),
        device="cpu",
    )
    if cfg is None:
        return 0
    elt = torch.empty((), dtype=cfg.dtype).element_size()
    per_layer = cfg.num_actual_blocks * cfg.block_size * cfg.latent_dim * elt
    return per_layer * len(cfg.layer_names)
