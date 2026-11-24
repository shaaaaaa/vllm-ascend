# SPDX-License-Identifier: Apache-2.0
"""ondemand-selected main-KV reads over LMCache P-node layerwise banks."""

from __future__ import annotations

import contextlib
import math
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

import vllm_ascend.core.selected_kv as skv
from vllm_ascend.core.selected_kv import PAGE_ROWS, _check, _ga, _pages, _rcheck

QUERY_TOTAL_RATIO_NUM = 3
QUERY_TOTAL_RATIO_DEN = 200
MAX_DOMAIN = 1_048_576
MAX_DVA_CACHE_ENTRIES = 262_144
MAX_INFLIGHT_STEPS = 2
_ROW_STRIDES = np.array([skv.MLA_NOPE_ROW_BYTES, skv.MLA_PE_ROW_BYTES], dtype=np.uint64)
_BIND_FILTER_TLS = threading.local()
_PATCHED_LOCK = threading.Lock()

# Explicit LMCache + lmcache_ascend hook capability contract required at startup.
_REQUIRED_NPU_MOD_ATTRS = ("bind_incremental_copy_addresses", "BoundCopyPrefix")
_REQUIRED_HOOK_SPECS = (
    ("gpu_connector", "batched_to_gpu"),
    ("engine", "_prepare_shared_prefill_sources"),
    ("engine", "release_shared_cpu_sparse_request"),
    ("gpu_connector", "release_layerwise_prefill_dma_cache"),
    ("impl", "_submit_initial_layerwise_prefill_load"),
    ("impl", "submit_layerwise_prefill_load"),
    ("impl", "start_load_kv"),
)


@dataclass(frozen=True)
class SelectedRequest:
    request_id: str
    history_tokens: int
    query_tokens: int
    selected: bool
    omit_end_token: int = 0


def _unwrap_storer_starts(storer: Any) -> Any:
    cur = storer
    seen: set[int] = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return None
        seen.add(id(cur))
        f_locals = _ga(cur, "gi_frame.f_locals")
        if isinstance(f_locals, Mapping):
            if "starts" in f_locals:
                return f_locals["starts"]
            nxt = _ga(cur, "gi_yieldfrom")
            if nxt is None:
                nxt = f_locals.get("gen")
            if nxt is None:
                for val in f_locals.values():
                    if hasattr(val, "gi_frame"):
                        nxt = val
                        break
            cur = nxt
        else:
            cur = _ga(cur, "gi_yieldfrom")
    return None


def _requires_retrieve(load_spec: Any) -> bool:
    """A full-resident row needs no transfer, even with a stale can_load."""
    return bool(_ga(load_spec, "can_load", False)) and not _ga(load_spec, "dsa_full_resident", False)


def classify_lmcache_request(
    request_id: str,
    history_tokens: object,
    query_tokens: object,
    req_meta: Any | None,
    *,
    lmcache_chunk_size: int = 1024,
    kv_role: str = "kv_producer",
) -> SelectedRequest:
    rid = str(request_id)
    if not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in (history_tokens, query_tokens)):
        return SelectedRequest(rid, 0, 0, False)
    hist = int(cast(int, history_tokens))
    query = int(cast(int, query_tokens))
    chunk_size = int(lmcache_chunk_size or 1024)
    load_spec = getattr(req_meta, "load_spec", None)
    disallowed = ("is_sparse_decode", "is_decode_window_save", "dsa_cold_compact_load", "_invalid_block_ids")
    eligible = (
        query > 0
        and skv.MIN_HISTORY_TOKENS <= hist <= skv.MAX_HISTORY_TOKENS
        and hist + query <= skv.MAX_TOTAL_TOKENS
        and chunk_size > 0
        and query * QUERY_TOTAL_RATIO_DEN < (hist + query) * QUERY_TOTAL_RATIO_NUM
        and _requires_retrieve(load_spec)
        and int(getattr(load_spec, "vllm_cached_tokens", 0) or 0) == 0
        and getattr(req_meta, "block_allocation_mode", None) == "prefill_child"
        and not any(getattr(req_meta, flag, False) for flag in disallowed)
        and not getattr(load_spec, "dsa_cold_compact_load", False)
    )
    if not eligible or int(getattr(load_spec, "lmcache_cached_tokens", 0)) < hist:
        return SelectedRequest(rid, max(hist, 0), max(query, 0), False)

    eff_align = math.lcm(chunk_size, PAGE_ROWS)
    omit_end = (hist // eff_align) * eff_align
    save_spec = getattr(req_meta, "save_spec", None)
    if kv_role != "kv_producer" and bool(getattr(save_spec, "can_save", False)):
        save_start = (max(int(getattr(save_spec, "skip_leading_tokens", 0)), 0) // chunk_size) * chunk_size
        omit_end = min(omit_end, (save_start // PAGE_ROWS) * PAGE_ROWS)
    is_sel = omit_end >= PAGE_ROWS
    return SelectedRequest(rid, hist, query, is_sel, omit_end if is_sel else 0)


def compute_per_bank_block_partition(
    requests: Sequence[Any],
    bank_block_tables: Sequence[Any],
    *,
    domain: int,
) -> tuple[tuple[frozenset[int], ...], tuple[frozenset[int], ...]]:
    _check(domain > 0 and bool(bank_block_tables), "Selected KV requires positive domain and >= 1 bank block table")
    omitted_by_bank: list[frozenset[int]] = []
    kept_by_bank: list[frozenset[int]] = []

    for bank_idx, raw_tbl in enumerate(bank_block_tables):
        tbl = np.asarray(raw_tbl)
        _check(
            tbl.ndim == 2 and tbl.shape[0] >= len(requests), f"Bank {bank_idx} block table shape {tbl.shape} < batch"
        )
        all_blocks: list[np.ndarray] = []
        cand_blocks: list[np.ndarray] = []
        for row, req in enumerate(requests):
            bounds_ok = req.history_tokens >= 0 and req.query_tokens >= 0
            bounds_ok = bounds_ok and (not req.selected or 0 <= req.omit_end_token <= req.history_tokens)
            _check(bounds_ok, "Bad request bounds")
            total_pages = _pages(req.history_tokens + req.query_tokens)
            if total_pages <= 0:
                continue
            _check(
                tbl.shape[1] >= total_pages,
                f"Bank {bank_idx} columns {tbl.shape[1]} < {total_pages} for {req.request_id}",
            )
            blks = np.asarray(tbl[row, :total_pages], dtype=np.int64)
            _check(int(blks.min()) >= 0 and int(blks.max()) < domain, f"Bank {bank_idx} block ID out of range")
            all_blocks.append(blks)
            if req.selected:
                omit_pages = req.omit_end_token // PAGE_ROWS
                if omit_pages > 0:
                    cand_blocks.append(blks[:omit_pages])

        if not all_blocks:
            omitted_by_bank.append(frozenset())
            kept_by_bank.append(frozenset())
            continue

        counts = np.bincount(np.concatenate(all_blocks), minlength=domain)
        if cand_blocks:
            cand_mask = np.zeros(domain, dtype=bool)
            cand_mask[np.concatenate(cand_blocks)] = True
            omit_mask = cand_mask & (counts == 1)
            omitted = frozenset(np.flatnonzero(omit_mask).tolist())
            kept = frozenset(np.flatnonzero((counts > 0) & ~omit_mask).tolist())
        else:
            omitted = frozenset()
            kept = frozenset(np.flatnonzero(counts > 0).tolist())
        omitted_by_bank.append(omitted)
        kept_by_bank.append(kept)

    return tuple(omitted_by_bank), tuple(kept_by_bank)


def filter_dma_plan(plan: Any, omitted: Set[int]) -> Any:
    if not omitted or len(plan) == 0:
        return plan
    segments: list[tuple[int, int, int, int]] = []
    cols_in = [getattr(plan, col).tolist() for col in ("chunk", "slot", "chunk_token", "tokens")]
    for chunk_idx, slot0, chunk_tok0, length in zip(*cols_in):
        cur = int(slot0)
        stop = cur + int(length)
        run_start: int | None = None
        while cur < stop:
            keep = (cur // PAGE_ROWS) not in omitted
            nxt = min(stop, (cur // PAGE_ROWS + 1) * PAGE_ROWS)
            if keep:
                if run_start is None:
                    run_start = cur
            elif run_start is not None:
                segments.append((int(chunk_idx), run_start, int(chunk_tok0) + run_start - int(slot0), cur - run_start))
                run_start = None
            cur = nxt
        if run_start is not None:
            segments.append((int(chunk_idx), run_start, int(chunk_tok0) + run_start - int(slot0), stop - run_start))

    cols_out = tuple(zip(*segments)) if segments else ((), (), (), ())
    return type(plan)(*(torch.tensor(col, dtype=plan.chunk.dtype, device=plan.chunk.device) for col in cols_out))


def _expand_sources(src: Any) -> tuple[Any, ...]:
    pages = getattr(src, "pages", None)
    suffix = getattr(src, "suffix", None)
    if pages is not None and suffix is not None:
        return (*pages, *suffix)
    return tuple(src) if isinstance(src, (tuple, list)) else (src,)


def _patch_bind_incremental(npu_mod: Any) -> None:
    missing = [attr for attr in _REQUIRED_NPU_MOD_ATTRS if not callable(getattr(npu_mod, attr, None))]
    _rcheck(npu_mod is not None and not missing, f"Missing lmcache_ascend NPU connector attributes: {missing}")
    with _PATCHED_LOCK:
        orig = npu_mod.bind_incremental_copy_addresses
        if getattr(orig, "_selected_kv_patched", False):
            return

        def _wrapped(
            plan: Any,
            source_objs: Any,
            starts: Any,
            ends: Any,
            npu_ptrs: Any,
            plane_widths: Any,
            element_bytes: int,
            host_ptr: int = 0,
            host_tokens: int = 0,
            previous: Any = None,
            **kw: Any,
        ) -> Any:
            pos_args = (source_objs, starts, ends, npu_ptrs, plane_widths, element_bytes)
            ctx = getattr(_BIND_FILTER_TLS, "active_context", None)
            if ctx is None:
                return orig(plan, *pos_args, host_ptr, host_tokens, previous, **kw)

            step, req_id, bank = ctx
            source_meta = kw.get("source_metadata")
            bound_cls = npu_mod.BoundCopyPrefix
            req_bank = step._req_bank.get((req_id, bank))
            eff_starts, eff_ends = (source_meta.starts, source_meta.ends) if source_meta is not None else (starts, ends)
            omit_all = not eff_ends or (
                req_bank is not None
                and 0 < (end_tok := int(eff_ends[-1])) <= len(req_bank[1]) * PAGE_ROWS
                and bool(req_bank[1][: _pages(end_tok)].all())
            )
            if omit_all:
                objs_t = tuple(source_objs)
                owners, owner_ids = (
                    (source_meta.owners, source_meta.owner_ids)
                    if source_meta is not None
                    else (objs_t, tuple(map(id, objs_t)))
                )
                return bound_cls(
                    owners,
                    owner_ids,
                    tuple(eff_starts),
                    tuple(eff_ends),
                    tuple(npu_ptrs),
                    tuple(plane_widths),
                    int(element_bytes),
                    plan.chunk[:0],
                    [],
                    0,
                    0,
                )

            filtered_kw = {**kw, "slot_prefix_unchanged": False, "reuse_rows_in_place": False}
            filtered_plan = filter_dma_plan(plan, step.omitted_by_bank[bank])
            return orig(filtered_plan, *pos_args, host_ptr, host_tokens, None, **filtered_kw)

        _wrapped._selected_kv_patched = True  # type: ignore[attr-defined]
        npu_mod.bind_incremental_copy_addresses = _wrapped


class LMCacheSelectedLoadStep:
    def __init__(
        self, src: Any, conn_meta: Any, reqs: Sequence[Any], tbls: Any, omitted: Any, dev: Tensor, host_slot: int = 0
    ) -> None:
        self._source = src
        self.connector_metadata = conn_meta
        self.requests = tuple(reqs)
        self.request_by_id = {req.request_id: req for req in self.requests}
        self.selected_request_ids = frozenset(req.request_id for req in self.requests if req.selected)
        self.omitted_by_bank = omitted
        self._dev_bases = dev
        self._host_slot = host_slot
        self._req_bank: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._bank_plans: dict[tuple[str, int], tuple[Any, ...]] = {}

        for bank_idx, om in enumerate(omitted):
            if not om:
                continue
            mask = np.zeros(src.domain, dtype=bool)
            mask[list(om)] = True
            for row, req in enumerate(self.requests):
                if not req.selected:
                    continue
                omit_pages = req.omit_end_token // PAGE_ROWS
                if omit_pages <= 0:
                    continue
                blks = np.asarray(tbls[bank_idx][row, :omit_pages], dtype=np.int64)
                page_omitted = mask[blks]
                if np.any(page_omitted):
                    self._req_bank[(req.request_id, bank_idx)] = (blks, page_omitted)

        self._lock = threading.RLock()
        self._init_uploaded_limit = 0
        self._init_ready_event: Any | None = None
        self._shared_ready_event: Any | None = None
        self._completion: Any | None = None
        self._deferred_cbs: list[Callable[[], Any]] = []
        self._layer_ready_events: dict[int, Any] = {}
        self._shared_uploaded_lids: set[int] = set()
        self._non_shared_populated: dict[int, set[str]] = {}
        self._retained: dict[int, tuple[Any, bool]] = {}
        self._slot_events: dict[int, tuple[Any, ...]] = {}
        self._pending_main: dict[int, Callable[[], None]] = {}
        self._populated_req_layers: set[tuple[str, int]] = set()
        self._shared_reqs: set[str] = set()
        self._waited_streams: set[tuple[int, int]] = set()
        self._initializing = True
        self._finished = False
        self._failed = False
        self._closed = False

    def _mark_failed(self) -> None:
        self._failed = True
        if self._source is not None and hasattr(self._source, "_failed"):
            self._source._failed = True

    def _call(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except BaseException:
            with self._lock:
                self._mark_failed()
            raise

    def _lid(self, name: str | None = None) -> int:
        _rcheck(
            not (self._failed or self._closed or getattr(self._source, "_failed", False)),
            "Selected KV source step is failed or retired",
        )
        _check(name is None or name in self._source._layer_ids, f"Unknown selected KV layer {name!r}")
        return -1 if name is None else self._source._layer_ids[name]

    def _ensure_layer_uploaded(self, lid: int, name: str) -> None:
        populated = all((rid, lid) in self._populated_req_layers for rid in self.selected_request_ids)
        uploaded = lid in self._shared_uploaded_lids or lid in self._layer_ready_events
        has_event = (
            lid in self._layer_ready_events
            or (lid < self._init_uploaded_limit and self._init_ready_event is not None)
            or (lid in self._shared_uploaded_lids and self._shared_ready_event is not None)
        )
        _rcheck(populated and uploaded and has_event, f"Layer {name!r} DDR page_bases not uploaded")

    def page_bases(self, name: str) -> Tensor:
        with self._lock:
            lid = self._lid(name)
            self._ensure_layer_uploaded(lid, name)
            return self._dev_bases[lid]

    def ready(self, name: str) -> Any | None:
        with self._lock:
            lid = self._lid(name)
            self._ensure_layer_uploaded(lid, name)
            if lid in self._layer_ready_events:
                return self._layer_ready_events[lid]
            if lid < self._init_uploaded_limit and self._init_ready_event is not None:
                return self._init_ready_event
            return self._shared_ready_event

    def wait_main_slot_ready(self, name: str, stream: Any) -> None:
        with self._lock:
            lid = self._lid(name)
            key = (lid, int(getattr(stream, "npu_stream", id(stream))))
            events = () if key in self._waited_streams else self._slot_events.get(lid, ())
            if lid in self._slot_events:
                self._waited_streams.add(key)
        for ev in events:
            stream.wait_event(ev)

    def prefetch_main(self, name: str, submit: Callable[[], None]) -> None:
        with self._lock:
            lid = self._lid(name)
            _rcheck(not self._finished and lid not in self._pending_main, "Main prefetch finished or duplicated")
            if lid in self._slot_events:
                self._call(submit)
            else:
                self._pending_main[lid] = submit

    def finish(self, *, completion: Any | None = None, failed: bool = False) -> bool:
        with self._lock:
            if self._finished:
                _rcheck(failed and not self._closed, "Selected forward has already finished")
                self._mark_failed()
                return False
            ev_ok = all(callable(getattr(completion, method, None)) for method in ("query", "synchronize"))
            _check(failed or ev_ok, "Successful selected forward requires a real completion event", TypeError)
            if not failed and self._pending_main:
                self._finished = True
                self._mark_failed()
                raise RuntimeError("Selected follower prefetches remained unsubmitted at step end")
            self._completion = None if failed else completion
            self._finished = True
            if failed:
                self._mark_failed()
            else:
                self._failed = False
        return self.try_retire()

    def try_retire(self) -> bool:
        with self._lock:
            if self._closed or not self._finished or self._failed or getattr(self._source, "_failed", False):
                return self._closed
            if not self._call(cast(Any, self._completion).query):
                return False
            retained = list(self._retained.values())
            callbacks = list(self._deferred_cbs)
            self._retained = {}
            self._deferred_cbs = []

            def release_all() -> None:
                for obj, was_pinned in retained:
                    if was_pinned:
                        obj.unpin()
                    ref_down = getattr(obj, "ref_count_down", None)
                    if callable(ref_down):
                        ref_down()
                for cb in callbacks:
                    cb()

            self._call(release_all)
            self.connector_metadata = None
            self._closed = True
            return True

    def cancel_unused(self) -> None:
        with self._lock:
            _rcheck(not self._finished and not self._pending_main, "Cannot cancel an active selected step")
            retained = list(self._retained.values())
            callbacks = list(self._deferred_cbs)
            self._retained = {}
            self._deferred_cbs = []
            self._completion = None
            self._finished = True
            self._failed = False

            def release_all() -> None:
                for obj, was_pinned in retained:
                    if was_pinned:
                        obj.unpin()
                    ref_down = getattr(obj, "ref_count_down", None)
                    if callable(ref_down):
                        ref_down()
                for cb in callbacks:
                    cb()

            self._call(release_all)
            self.connector_metadata = None
            self._closed = True

    def wait_completion(self) -> None:
        with self._lock:
            _rcheck(self._finished and not self._failed, "Selected forward did not finish successfully")
            ev = None if self._closed else cast(Any, self._completion)
        if ev is not None:
            self._call(ev.synchronize)


class LMCacheSelectedKVSource:
    def __init__(self, runner: Any, impl: Any, names: Sequence[str], caches: Mapping[str, Any], **kw: Any) -> None:
        dev = torch.device(runner.device)
        self.device = dev
        self.impl = impl
        self.engine = getattr(impl, "lmcache_engine", None)
        self.gpu_connector = getattr(self.engine, "gpu_connector", None)
        self.layer_names = tuple(names)
        self.num_layers = len(self.layer_names)
        self._layer_ids = {name: idx for idx, name in enumerate(self.layer_names)}

        domains = {int(caches[name][0].shape[0]) for name in self.layer_names}
        _check(
            self.engine is not None and self.gpu_connector is not None and bool(names) and len(domains) == 1,
            "Bad source engine/caches",
        )
        ckpt_flag = _ga(impl, "config.decode_preemption_checkpoint")
        if ckpt_flag is None:
            get_extra_fn = _ga(impl, "config.get_extra_config_value")
            ckpt_flag = get_extra_fn("decode_preemption_checkpoint", False) if callable(get_extra_fn) else False
        ckpt_on = (
            ckpt_flag is True
            or (type(ckpt_flag) is int and ckpt_flag != 0)
            or (isinstance(ckpt_flag, str) and ckpt_flag.strip().lower() in ("1", "true", "yes", "on"))
        )
        _check(not ckpt_on, "Selected KV prefill cannot be combined with decode_preemption_checkpoint")
        domain = next(iter(domains))
        _check(0 < domain <= MAX_DOMAIN, f"Domain not in 1..{MAX_DOMAIN}")
        self.domain = domain

        default_banks = getattr(impl, "_layerwise_prefill_bank_count", 2)
        num_banks = int(getattr(self.gpu_connector, "_layerwise_prefill_bank_count", default_banks))
        _check(num_banks >= 2, f"Selected KV P-node source requires >= 2 banks, got {num_banks}")
        self.num_banks = num_banks
        self._init_lim = min(num_banks, self.num_layers)

        npu_mod = kw.get("npu_connectors_module")
        ptr_fn = kw.get("get_device_ptr_fn") or _ga(npu_mod, "lmc_ops.get_device_ptr")
        if not callable(ptr_fn):
            import lmcache_ascend.c_ops as lmc_ops  # type: ignore[import-not-found]

            ptr_fn = lmc_ops.get_device_ptr
        self._get_device_ptr_fn = ptr_fn
        self._dva_cache: dict[tuple[int, int, int, int], int] = {}
        self._dva_cache_gen: int = 0

        padded_domain = ((domain + 3) // 4) * 4
        bases_shape = (self.num_layers, padded_domain, 2)
        self._metadata_stream = torch.npu.Stream(device=dev)
        pin_host = dev.type != "cpu"
        self._host_bases_ring: tuple[Tensor, ...] | None = tuple(
            torch.zeros(bases_shape, dtype=torch.int64, pin_memory=pin_host) for _ in range(MAX_INFLIGHT_STEPS)
        )
        self._host_bases: Tensor | None = self._host_bases_ring[0]
        self._next_host_slot = 0
        self._slot_upload_events: list[Any | None] = [None] * MAX_INFLIGHT_STEPS
        self._slot_dirty: list[list[tuple[int, Any, np.ndarray]]] = [[] for _ in range(MAX_INFLIGHT_STEPS)]
        with torch.npu.stream(self._metadata_stream):
            self._dev_bases: Tensor | None = torch.zeros(bases_shape, dtype=torch.int64, device=dev)

        self._step_host: Tensor | None = None
        self._step_dev: Tensor | None = None
        self.step: LMCacheSelectedLoadStep | None = None
        self._staged_step: Any | None = None
        self._steps_lock = threading.RLock()
        self._unretired_steps: list[LMCacheSelectedLoadStep] = []
        self._restorers: list[Callable[[], None]] = []
        self._active_ranges: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        self._connector_meta_spec: tuple[tuple[int, int], ...] = ()
        self._closed = False
        self._failed = False
        self._install_hooks(npu_mod)

    def _chunk_info(self, obj: Any, lid: int, pre_dva: int = 0) -> tuple[int, int]:
        layer_size_fn = getattr(obj, "layer_size_bytes", None)
        if callable(layer_size_fn):
            byte_size = int(layer_size_fn(lid))
        else:
            layer_tensor_fn = getattr(obj, "layer_tensor", None)
            if callable(layer_tensor_fn):
                tensor = layer_tensor_fn(lid)
            elif isinstance(obj, Tensor):
                tensor = obj
            else:
                tensor = getattr(obj, "tensor", None)
            if tensor is not None:
                plane_fn = getattr(self.gpu_connector, "_lmc_plane_num_tokens", None)
                byte_size = (
                    int(plane_fn(tensor, 0, lid)) * skv.MAIN_ROW_BYTES
                    if callable(plane_fn)
                    else int(tensor.numel() * tensor.element_size())
                )
            else:
                get_size_fn = getattr(obj, "get_size", None)
                byte_size = int(get_size_fn()) if callable(get_size_fn) else -1
        if byte_size <= 0 or byte_size % skv.MAIN_ROW_BYTES != 0:
            raise ValueError(f"Invalid MLA source byte size {byte_size}")

        dva = pre_dva
        if dva <= 0:
            ptr_fn = getattr(obj, "layer_data_ptr", None)
            host_ptr = (
                int(ptr_fn(lid))
                if callable(ptr_fn)
                else int(obj.data_ptr() if isinstance(obj, Tensor) else obj.data_ptr)
            )
            parent_alloc = getattr(obj, "parent_allocator", None)
            if parent_alloc is None:
                alloc_live = False
            elif hasattr(parent_alloc, "_mapping"):
                mapping = parent_alloc._mapping
                alloc_live = mapping is not None and not bool(getattr(mapping, "_closed", True))
            else:
                alloc_live = hasattr(parent_alloc, "_unregistered") and not bool(parent_alloc._unregistered)
            gen = int(getattr(self.engine, "shared_cpu_cache_generation", 0) or 0)
            if gen != self._dva_cache_gen:
                self._dva_cache.clear()
                self._dva_cache_gen = gen
            cache_key = (gen, id(parent_alloc), host_ptr, byte_size)
            dva = self._dva_cache.get(cache_key, 0) if alloc_live else 0
            if dva <= 0:
                resolved = self._get_device_ptr_fn(host_ptr, byte_size)
                if resolved is None or int(resolved) <= 0:
                    raise RuntimeError(f"Bad DDR ptr {host_ptr}")
                dva = int(resolved)
                if alloc_live:
                    if len(self._dva_cache) >= MAX_DVA_CACHE_ENTRIES:
                        self._dva_cache.clear()
                    self._dva_cache[cache_key] = dva
        return byte_size // skv.MAIN_ROW_BYTES, dva

    def _prune_retired_steps(self) -> None:
        with self._steps_lock:
            if any(step._failed for step in self._unretired_steps):
                self._failed = True
            if self._failed:
                return
            self._unretired_steps = [step for step in self._unretired_steps if not step.try_retire()]

    def unretired_selected_request_ids(self) -> frozenset[str]:
        with self._steps_lock:
            self._prune_retired_steps()
            return frozenset(
                rid for step in self._unretired_steps if not step._closed for rid in step.selected_request_ids
            )

    def _drain_unretired(self, predicate: Callable[[LMCacheSelectedLoadStep], bool] | None = None) -> None:
        with self._steps_lock:
            try:
                for prev in list(self._unretired_steps):
                    if prev._failed or not prev._finished:
                        self._failed = True
                        raise RuntimeError("Selected forward did not finish successfully")
                    if predicate is None or predicate(prev):
                        prev.wait_completion()
                        _rcheck(prev.try_retire(), "Previous selected step did not retire")
                        self._unretired_steps.remove(prev)
            except BaseException:
                self._failed = True
                raise

    def _clear_host_slot(self, slot: int) -> Tensor:
        up_ev = self._slot_upload_events[slot]
        if up_ev is not None:
            try:
                up_ev.synchronize()
            except BaseException:
                self._failed = True
                raise
            self._slot_upload_events[slot] = None
        host_buf = cast(tuple[Tensor, ...], self._host_bases_ring)[slot]
        dirty = self._slot_dirty[slot]
        if dirty:
            flat_u64 = host_buf.view(-1).numpy().view(np.uint64)
            for prev_sd, idx0, blks in dirty:
                view = flat_u64[: self.num_layers * prev_sd * 2].reshape(self.num_layers, prev_sd, 2)
                view[idx0, blks] = 0
            dirty.clear()
        return host_buf

    def _write_host_bases(
        self, step: LMCacheSelectedLoadStep, idx0: Any, dst_blocks: np.ndarray, values: np.ndarray
    ) -> None:
        host_t = cast(Tensor, self._step_host)
        stride_domain = int(host_t.shape[1])
        slot = step._host_slot
        if self._host_bases_ring is not None and host_t.data_ptr() == self._host_bases_ring[slot].data_ptr():
            self._slot_dirty[slot].append((stride_domain, idx0, dst_blocks))
        host_u64 = host_t.numpy().view(np.uint64)
        host_u64[idx0, dst_blocks] = values

    def _retain_objs(self, step: LMCacheSelectedLoadStep, objs: Iterable[Any]) -> None:
        with step._lock:
            step._lid()
            retained = step._retained
            try:
                for obj in objs:
                    ident = id(obj)
                    if ident in retained:
                        continue
                    valid_fn = getattr(obj, "is_valid", None)
                    if callable(valid_fn) and not valid_fn():
                        raise RuntimeError("DDR MemoryObj invalid")
                    should_pin = (
                        bool(getattr(obj, "is_pinned", False))
                        and callable(getattr(obj, "pin", None))
                        and callable(getattr(obj, "unpin", None))
                    )
                    if should_pin:
                        obj.pin()
                    ref_up = getattr(obj, "ref_count_up", None)
                    if callable(ref_up):
                        ref_up()
                    retained[ident] = (obj, should_pin)
            except BaseException:
                step._mark_failed()
                raise

    def _resolve_bank_plan(
        self,
        step: LMCacheSelectedLoadStep,
        req: SelectedRequest,
        plan_key: tuple[str, int],
        req_bank: tuple[np.ndarray, np.ndarray],
        num_objs: int,
        starts: Sequence[int],
        ends: Sequence[int],
    ) -> tuple[Any, ...]:
        if len(starts) < num_objs or len(ends) < num_objs:
            raise RuntimeError(f"Ranges shorter than objs for {req.request_id}")
        s_key = starts if isinstance(starts, tuple) and len(starts) == num_objs else tuple(starts[:num_objs])
        e_key = ends if isinstance(ends, tuple) and len(ends) == num_objs else tuple(ends[:num_objs])
        with step._lock:
            plan = step._bank_plans.get(plan_key)
        if plan is not None and plan[0] == s_key and plan[1] == e_key:
            return plan

        req_blocks, page_omitted = req_bank
        s_arr = np.asarray(s_key, np.int64)
        e_arr = np.asarray(e_key, np.int64)
        ge_omit = np.flatnonzero(s_arr >= req.omit_end_token)
        k = int(ge_omit[0]) if ge_omit.size else num_objs
        _rcheck(k > 0 and s_arr[0] == 0 and e_arr[k - 1] >= req.omit_end_token, f"Incomplete DDR: {req.request_id}")

        s_arr = s_arr[:k]
        e_arr = e_arr[:k]
        clamped_end = np.minimum(e_arr, req.omit_end_token)
        contiguous = (
            bool(np.all(e_arr > s_arr)) and bool(np.all(s_arr[1:] == e_arr[:-1])) and not np.any(s_arr % PAGE_ROWS)
        )
        pages_per_chunk = (clamped_end - s_arr) // PAGE_ROWS
        _rcheck(
            contiguous
            and not np.any((clamped_end - s_arr) % PAGE_ROWS)
            and int(pages_per_chunk.sum()) == len(page_omitted),
            f"Bad DDR: {req.request_id}",
        )

        sel_pages = np.flatnonzero(page_omitted)
        sel_chunks = np.repeat(np.arange(k, dtype=np.int64), pages_per_chunk)[sel_pages]
        uniq_idx, inv_chunks = np.unique(sel_chunks, return_inverse=True)
        rel_rows = ((sel_pages - (s_arr // PAGE_ROWS)[sel_chunks]) * PAGE_ROWS)[:, None].astype(np.uint64)
        plan = (
            s_key,
            e_key,
            k,
            uniq_idx.tolist(),
            (e_arr[uniq_idx] - s_arr[uniq_idx]).tolist(),
            req_blocks[sel_pages],
            inv_chunks.reshape(-1),
            rel_rows * _ROW_STRIDES,
        )
        with step._lock:
            step._bank_plans[plan_key] = plan
        return plan

    def begin_runner_step(self, conn_meta: Any, req_ids: Sequence[str], hists: Any, q_starts: Any, tbls: Any) -> None:
        _rcheck(not self._closed, "Selected KV source is closed")
        self._prune_retired_steps()
        _rcheck(not self._failed, "Selected KV source is permanently retained after a failed step")
        self.step = None
        starts = tuple(map(int, q_starts)) if conn_meta is not None and req_ids else ()
        if len(starts) != len(req_ids) + 1:
            self._staged_step = None
            return
        queries = tuple(end - start for start, end in zip(starts, starts[1:]))
        self._staged_step = (
            conn_meta,
            tuple(map(str, req_ids)),
            tuple(map(int, hists)),
            queries,
            tuple(map(np.asarray, tbls)),
        )

    def _populate_bases(
        self,
        step: LMCacheSelectedLoadStep,
        req_key: str,
        lid: int,
        objs: Sequence[Any],
        starts: Sequence[int],
        ends: Sequence[int],
        dvas: Sequence[int] | None = None,
    ) -> None:
        req = step.request_by_id[req_key]
        if not req.selected:
            return
        with step._lock:
            if (req_key, lid) in step._populated_req_layers:
                return
        self._retain_objs(step, objs)

        bank = lid % self.num_banks
        plan_key = (req_key, bank)
        req_bank = step._req_bank.get(plan_key)
        if req_bank is not None:
            plan = self._resolve_bank_plan(step, req, plan_key, req_bank, len(objs), starts, ends)
            _, _, k, uniq_list, uniq_lens, dst_blocks, inv_chunks, row_offsets = plan
            dva_seq = dvas if dvas is not None and len(dvas) >= k else None
            b0_list: list[int] = []
            plane_list: list[int] = []
            for chunk_idx, chunk_len in zip(uniq_list, uniq_lens):
                raw_dva = dva_seq[chunk_idx] if dva_seq is not None else 0
                pre_dva = (
                    int(raw_dva) if isinstance(raw_dva, (int, np.integer)) and not isinstance(raw_dva, bool) else 0
                )
                if not (0 < pre_dva < (1 << 64)):
                    pre_dva = 0
                plane_tokens, base0 = self._chunk_info(objs[chunk_idx], lid, pre_dva)
                if plane_tokens < chunk_len:
                    raise ValueError(f"Chunk capacity P={plane_tokens} < {chunk_len}")
                b0_list.append(base0)
                plane_list.append(plane_tokens)

            b0_arr = np.fromiter(b0_list, dtype=np.uint64, count=len(b0_list))
            p_arr = np.fromiter(plane_list, dtype=np.uint64, count=len(plane_list))
            uniq_bases = np.stack((b0_arr, b0_arr + p_arr * np.uint64(skv.MLA_NOPE_ROW_BYTES)), axis=1)
            self._write_host_bases(step, lid, dst_blocks, uniq_bases[inv_chunks] + row_offsets)

        with step._lock:
            step._lid()
            step._populated_req_layers.add((req_key, lid))

    def _record_upload(self, step: LMCacheSelectedLoadStep, lid: int | None = None) -> None:
        host_t = cast(Tensor, self._step_host)
        dev_t = cast(Tensor, self._step_dev)
        init_lim = step._init_uploaded_limit or self._init_lim
        init_ev: Any | None = None
        with torch.npu.stream(self._metadata_stream):
            if lid is None:
                if self.num_layers > init_lim:
                    dev_t[:init_lim].copy_(host_t[:init_lim], non_blocking=True)
                    init_ev = torch.npu.Event()
                    init_ev.record(self._metadata_stream)
                    dev_t[init_lim:].copy_(host_t[init_lim:], non_blocking=True)
                else:
                    dev_t.copy_(host_t, non_blocking=True)
            else:
                dev_t[lid].copy_(host_t[lid], non_blocking=True)
            ready_ev = torch.npu.Event()
            ready_ev.record(self._metadata_stream)
        self._slot_upload_events[step._host_slot] = ready_ev
        with step._lock:
            if lid is None:
                step._layer_ready_events.clear()
                step._init_uploaded_limit = init_lim
                step._init_ready_event = init_ev if init_ev is not None else ready_ev
                step._shared_ready_event = ready_ev
                step._shared_uploaded_lids = {
                    l_idx
                    for l_idx in range(self.num_layers)
                    if all((rid, l_idx) in step._populated_req_layers for rid in step.selected_request_ids)
                }
            else:
                step._layer_ready_events[lid] = ready_ev

    def _mark_slots_ready(self, lids: Iterable[int]) -> None:
        step = self.step
        if step is None:
            return

        def unwrap_event(entry: Any) -> Any:
            return (entry[1] if len(entry) >= 2 else None) if isinstance(entry, tuple) else entry

        state_fn = getattr(self.gpu_connector, "_layerwise_prefill_transfer_state", None)
        state = state_fn() if callable(state_fn) else None
        generations = (
            state[1] if isinstance(state, tuple) and len(state) >= 2 and isinstance(state[1], Mapping) else None
        )
        load_done = state[3] if isinstance(state, tuple) and len(state) >= 4 and isinstance(state[3], Mapping) else {}
        bank_tails = getattr(self.gpu_connector, "_layerwise_prefill_bank_tail_events", None)

        for lid in lids:
            layer_loads = load_done.get((0, lid))
            events = [unwrap_event(entry) for entry in layer_loads.values()] if isinstance(layer_loads, Mapping) else []
            if isinstance(bank_tails, Mapping):
                bank_key = (0, lid % self.num_banks)
                tail_entry = bank_tails.get(bank_key)
                gen0 = generations.get(bank_key, generations.get(0)) if generations is not None else None
                if tail_entry is not None and not (
                    isinstance(tail_entry, tuple)
                    and (
                        len(tail_entry) < 2
                        or (type(gen0) is int and type(tail_entry[0]) is int and tail_entry[0] != gen0)
                    )
                ):
                    events.append(unwrap_event(tail_entry))
            with step._lock:
                if step._failed or step._closed or lid in step._slot_events:
                    continue
                step._slot_events[lid] = tuple({id(ev): ev for ev in events if ev is not None}.values())
                submit = step._pending_main.pop(lid, None)
                if submit is not None:
                    step._call(submit)

    def _batched_to_gpu(self, orig: Any, starts: Sequence[int], ends: Sequence[int], *args: Any, **kw: Any) -> Any:
        req_id = kw.get("req_id")
        step = self.step
        if step is None or kw.get("kv_group", 0) != 0 or req_id not in step.selected_request_ids:
            return orig(starts, ends, *args, **kw)
        req_key = str(req_id)
        starts_t = tuple(map(int, starts))
        ends_t = tuple(map(int, ends))
        self._active_ranges[req_key] = (starts_t, ends_t)
        inner = orig(starts, ends, *args, **kw)

        def _gen() -> Any:
            lid = 0
            msg = yield next(inner)
            while True:
                if msg is None:
                    with contextlib.suppress(StopIteration):
                        inner.send(None)
                        msg = yield None
                        continue
                    return
                if req_key in step.selected_request_ids and req_key not in step._shared_reqs:
                    raw = msg["memory_objs"] if isinstance(msg, Mapping) and "memory_objs" in msg else msg
                    self._populate_bases(step, req_key, lid, _expand_sources(raw), starts_t, ends_t)
                    with step._lock:
                        ns_reqs = step.selected_request_ids - step._shared_reqs
                        pop_set = step._non_shared_populated.setdefault(lid, set())
                        if req_key in ns_reqs:
                            pop_set.add(req_key)
                        should_upload = bool(ns_reqs) and not step._initializing and ns_reqs <= pop_set
                    if should_upload:
                        self._record_upload(step, lid)
                bank = lid % self.num_banks
                bound_map = getattr(self.gpu_connector, "_prefill_dma_bound_loads", None)
                req_bound = bound_map.get(req_key) if isinstance(bound_map, Mapping) else None
                bound_dict = req_bound if isinstance(req_bound, dict) else {}
                prev_ctx = getattr(_BIND_FILTER_TLS, "active_context", None)
                _BIND_FILTER_TLS.active_context = (step, req_key, bank)
                try:
                    bound_dict.pop((0, lid, bank), None)
                    out = inner.send(msg)
                finally:
                    _BIND_FILTER_TLS.active_context = prev_ctx
                    bound_dict.pop((0, lid, bank), None)
                lid += 1
                msg = yield out

        return _gen()

    def _prepare_shared(self, orig: Any, prep: Sequence[Any], grp: int, kw_in: Any, *args: Any, **kw: Any) -> Any:
        prior_ptrs = _ga(kw_in, "cached_chunk_dev_ptrs")
        req_id = _ga(kw_in, "req_id")
        prior_had_ptrs = bool(prior_ptrs and any(prior_ptrs))
        stale_unreused = bool(prior_had_ptrs and not _ga(kw_in, "_reuse_shared_dense_prefix"))
        res = orig(prep, grp, kw_in, *args, **kw)
        step = self.step
        req_key = str(req_id)
        ranges = self._active_ranges.get(req_key)
        if ranges is None:
            s_kw = _ga(kw_in, "_shared_prefill_starts")
            e_kw = _ga(kw_in, "_shared_prefill_ends")
            if isinstance(s_kw, Sequence) and isinstance(e_kw, Sequence):
                ranges = (tuple(map(int, s_kw)), tuple(map(int, e_kw)))
        if not (
            step is not None
            and grp == 0
            and len(prep) == self.num_layers
            and ranges is not None
            and req_id in step.selected_request_ids
        ):
            return res
        if any(layer_src is None for layer_src in prep):
            missing_lid = next(lid for lid, layer_src in enumerate(prep) if layer_src is None)
            raise RuntimeError(f"Shared CPU prepared_sources missing layer {missing_lid}")

        post_ptrs = None if stale_unreused else _ga(kw_in, "cached_chunk_dev_ptrs")
        ptrs = post_ptrs if isinstance(post_ptrs, Sequence) and len(post_ptrs) == self.num_layers else None
        objs0 = _expand_sources(prep[0])

        def _obj_ptr_lid(obj: Any, lid: int) -> int:
            ptr_fn = getattr(obj, "layer_data_ptr", None)
            if callable(ptr_fn):
                return int(ptr_fn(lid))
            return int(obj.data_ptr() if isinstance(obj, Tensor) else obj.data_ptr)

        def _row_dvas_valid(
            row_ptrs: Any, objs: Sequence[Any], lid: int, ref_delta: int | None
        ) -> tuple[bool, int | None]:
            if not (isinstance(row_ptrs, Sequence) and bool(objs) and len(row_ptrs) == len(objs)):
                return False, ref_delta
            with contextlib.suppress(Exception):
                d0 = int(row_ptrs[0])
                if d0 <= 0:
                    return False, ref_delta
                p0 = _obj_ptr_lid(objs[0], lid)
                if ref_delta is None or (d0 - p0) != ref_delta:
                    if d0 != self._chunk_info(objs[0], lid)[1]:
                        return False, ref_delta
                    ref_delta = d0 - p0
                ok = all(
                    (dk := int(row_ptrs[k])) > 0
                    and ((dk - _obj_ptr_lid(objs[k], lid)) == ref_delta or dk == self._chunk_info(objs[k], lid)[1])
                    for k in range(1, len(objs))
                )
                return ok, ref_delta
            return False, ref_delta

        trust_cached, slab_delta = _row_dvas_valid(ptrs[0], objs0, 0, None) if ptrs is not None else (False, None)

        pages0 = getattr(prep[0], "pages", None)
        with step._lock:
            fresh_layers = all((req_key, lid) not in step._populated_req_layers for lid in range(self.num_layers))
        can_vectorize = (
            fresh_layers
            and isinstance(pages0, tuple)
            and all(
                getattr(src, "pages", None) is pages0
                and getattr(src, "suffix", None) == ()
                and getattr(src, "layer_id", lid) == lid
                for lid, src in enumerate(prep)
            )
        )
        if not can_vectorize:
            for lid, layer_src in enumerate(prep):
                objs = objs0 if lid == 0 else _expand_sources(layer_src)
                if not (trust_cached and ptrs is not None):
                    layer_dvas = None
                elif lid == 0 or (len(objs) == len(objs0) and all(a is b for a, b in zip(objs, objs0))):
                    row_ptrs = ptrs[lid]
                    layer_dvas = row_ptrs if isinstance(row_ptrs, Sequence) and len(row_ptrs) == len(objs) else None
                else:
                    ok_lid, slab_delta = _row_dvas_valid(ptrs[lid], objs, lid, slab_delta)
                    layer_dvas = ptrs[lid] if ok_lid else None
                self._populate_bases(step, req_key, lid, objs, ranges[0], ranges[1], layer_dvas)
            with step._lock:
                step._lid()
                step._shared_reqs.add(req_key)
            return res

        req = step.request_by_id[req_key]
        self._retain_objs(step, pages0)
        num_objs = len(pages0)
        active_banks = min(self.num_banks, self.num_layers)
        ptrs_all_valid = (
            trust_cached
            and ptrs is not None
            and all(isinstance(ptrs[lid], Sequence) and len(ptrs[lid]) == num_objs for lid in range(self.num_layers))
        )

        for bank in range(active_banks):
            req_bank = step._req_bank.get((req_key, bank))
            if req_bank is None:
                continue
            plan = self._resolve_bank_plan(step, req, (req_key, bank), req_bank, num_objs, ranges[0], ranges[1])
            _, _, _, uniq_list, uniq_lens, dst_blocks, inv_chunks, row_offsets = plan
            lids = range(bank, self.num_layers, self.num_banks)
            num_bank_layers = len(lids)
            num_uniq = len(uniq_list)

            homo_planes: list[int] | None = []
            for chunk_idx, chunk_len in zip(uniq_list, uniq_lens):
                obj = pages0[chunk_idx]
                ls_fn = getattr(obj, "layer_size_bytes", None)
                raw_ls = getattr(obj, "layer_size", None)
                num_obj_layers = getattr(obj, "num_layers", None)
                if (
                    callable(ls_fn)
                    and type(raw_ls) is int
                    and type(num_obj_layers) is int
                    and num_obj_layers >= self.num_layers
                    and getattr(obj, "valid", True) is True
                    and int(ls_fn(0)) == raw_ls
                ):
                    if raw_ls <= 0 or raw_ls % skv.MAIN_ROW_BYTES != 0:
                        raise ValueError(f"Invalid MLA source byte size {raw_ls}")
                    p_tok = raw_ls // skv.MAIN_ROW_BYTES
                    if p_tok < chunk_len:
                        raise ValueError(f"Chunk capacity P={p_tok} < {chunk_len}")
                    homo_planes.append(p_tok)
                else:
                    homo_planes = None
                    break

            if homo_planes is not None:
                p_mat = np.fromiter(homo_planes, dtype=np.uint64, count=num_uniq)[None, :]
                b0_flat: list[int] = []
                for lid in lids:
                    row_dvas = ptrs[lid] if ptrs_all_valid and ptrs is not None else None
                    for chunk_idx in uniq_list:
                        raw_dva = row_dvas[chunk_idx] if row_dvas is not None else 0
                        dva_val = (
                            int(raw_dva)
                            if isinstance(raw_dva, (int, np.integer)) and not isinstance(raw_dva, bool)
                            else 0
                        )
                        if not (0 < dva_val < (1 << 64)):
                            _, dva_val = self._chunk_info(pages0[chunk_idx], lid, 0)
                        b0_flat.append(dva_val)
                b0_mat = np.fromiter(b0_flat, dtype=np.uint64, count=num_bank_layers * num_uniq).reshape(
                    num_bank_layers, num_uniq
                )
            else:
                b0_flat = []
                p_flat: list[int] = []
                for lid in lids:
                    row_dvas = ptrs[lid] if ptrs_all_valid and ptrs is not None else None
                    for chunk_idx, chunk_len in zip(uniq_list, uniq_lens):
                        raw_dva = row_dvas[chunk_idx] if row_dvas is not None else 0
                        pre_dva = (
                            int(raw_dva)
                            if isinstance(raw_dva, (int, np.integer)) and not isinstance(raw_dva, bool)
                            else 0
                        )
                        if not (0 < pre_dva < (1 << 64)):
                            pre_dva = 0
                        p_tok, base0 = self._chunk_info(pages0[chunk_idx], lid, pre_dva)
                        if p_tok < chunk_len:
                            raise ValueError(f"Chunk capacity P={p_tok} < {chunk_len}")
                        b0_flat.append(base0)
                        p_flat.append(p_tok)
                b0_mat = np.fromiter(b0_flat, dtype=np.uint64, count=num_bank_layers * num_uniq).reshape(
                    num_bank_layers, num_uniq
                )
                p_mat = np.fromiter(p_flat, dtype=np.uint64, count=num_bank_layers * num_uniq).reshape(
                    num_bank_layers, num_uniq
                )

            uniq_bases = np.stack((b0_mat, b0_mat + p_mat * np.uint64(skv.MLA_NOPE_ROW_BYTES)), axis=2)
            self._write_host_bases(
                step, slice(bank, self.num_layers, self.num_banks), dst_blocks, uniq_bases[:, inv_chunks] + row_offsets
            )

        with step._lock:
            step._lid()
            step._populated_req_layers.update((req_key, lid) for lid in range(self.num_layers))
            step._shared_reqs.add(req_key)
        return res

    def _start_load_kv(self, orig_start: Any, forward_context: Any, *args: Any, **kw: Any) -> Any:
        _rcheck(not self._failed, "Selected KV source is permanently retained after a failed step")
        self._prune_retired_steps()
        _rcheck(not self._failed, "Selected KV source is permanently retained after a failed step")
        self.step = None
        staged = self._staged_step
        self._staged_step = None
        self._active_ranges = {}
        self.impl._selected_group0_omit_end_tokens = {}
        get_meta = _ga(self.impl, "_parent._get_connector_metadata")
        meta = get_meta() if callable(get_meta) else None
        meta_reqs = getattr(meta, "requests", None) if meta is not None else None
        if self._unretired_steps and meta_reqs:
            step_req_ids: set[str] = set(staged[1]) if staged is not None and staged[0] is meta else set()
            for req_meta in meta_reqs:
                rid = getattr(req_meta, "req_id", None)
                if rid is not None:
                    step_req_ids.add(str(rid))
            self._drain_unretired(lambda prev: bool(prev.selected_request_ids & step_req_ids))
        if staged is None or meta is None or staged[0] is not meta or not meta_reqs:
            return orig_start(forward_context, *args, **kw)
        win_cap = getattr(self.impl, "supports_layerwise_prefill_transfer_window", True)
        win_ok = bool(win_cap() if callable(win_cap) else win_cap)
        loadable = [
            req_meta
            for req_meta in meta.requests
            if not getattr(req_meta, "is_decode_window_save", False)
            and _requires_retrieve(_ga(req_meta, "load_spec"))
        ]
        deferred_ok = (
            bool(getattr(self.impl, "use_layerwise", True))
            and win_ok
            and bool(loadable)
            and all(
                not getattr(req_meta, "is_sparse_decode", False)
                and getattr(req_meta, "block_allocation_mode", None) == "prefill_child"
                for req_meta in loadable
            )
        )
        if not deferred_ok:
            return orig_start(forward_context, *args, **kw)

        _, req_ids, hists, queries, tbls = staged
        req_meta_by_id: dict[str, Any] = {}
        for req_meta in meta.requests:
            if not getattr(req_meta, "is_decode_window_save", False):
                rid = str(getattr(req_meta, "req_id", ""))
                req_meta_by_id[rid] = None if rid in req_meta_by_id else req_meta

        chunk_size = int(_ga(self.engine, "config.chunk_size") or _ga(self.impl, "_lmcache_chunk_size") or 1024)
        kv_role = str(
            _ga(self.impl, "kv_role") or _ga(self.impl, "_parent._kv_transfer_config.kv_role") or "kv_producer"
        )
        classified = [
            classify_lmcache_request(
                rid, hists[row], queries[row], req_meta_by_id.get(rid), lmcache_chunk_size=chunk_size, kv_role=kv_role
            )
            for row, rid in enumerate(req_ids)
        ]
        if kv_role == "kv_both":
            sparse_attn = bool(getattr(self.impl, "enable_sparse_attention", True))
            release_gate = max(
                int(getattr(self.impl, "_dsa_scratch_capacity", 0) or 0),
                int(getattr(self.impl, "_dsa_kv_policy_threshold", 0) or 0),
            )
            classified = [
                replace(r, selected=False, omit_end_token=0)
                if r.selected
                and (
                    not sparse_attn
                    or not bool(getattr(req_meta_by_id.get(r.request_id), "is_last_prefill", True))
                    or (
                        int(
                            _ga(req_meta_by_id.get(r.request_id), "load_spec.lmcache_cached_tokens", r.history_tokens)
                            or 0
                        )
                        // chunk_size
                        * chunk_size
                    )
                    <= release_gate
                )
                else r
                for r in classified
            ]

        for row, sel_req in enumerate(classified):
            if not sel_req.selected:
                continue
            by_bank = _ga(req_meta_by_id.get(sel_req.request_id), "block_ids_by_bank")
            if by_bank is not None:
                req_pages = _pages(sel_req.history_tokens + sel_req.query_tokens)
                banks_ok = len(by_bank) == len(tbls) and all(
                    np.array_equal(
                        np.asarray(by_bank[b][0][:req_pages], np.int64), np.asarray(tbls[b][row, :req_pages], np.int64)
                    )
                    for b in range(len(by_bank))
                )
                _check(banks_ok, f"Bank mismatch: {sel_req.request_id}")

        selected_reqs = [req for req in classified if req.selected]
        if not selected_reqs:
            return orig_start(forward_context, *args, **kw)
        omitted_by_bank, kept_by_bank = compute_per_bank_block_partition(classified, tbls, domain=self.domain)
        if not any(omitted_by_bank):
            return orig_start(forward_context, *args, **kw)

        self._drain_unretired(lambda _: False)
        self._prune_retired_steps()
        _rcheck(not self._failed, "Selected KV source is permanently retained after a failed step")

        cur_sel_ids = frozenset(req.request_id for req in selected_reqs)
        self._drain_unretired(lambda prev: bool(prev.selected_request_ids & cur_sel_ids))

        with self._steps_lock:
            try:
                while len(self._unretired_steps) >= MAX_INFLIGHT_STEPS:
                    oldest = self._unretired_steps[0]
                    oldest.wait_completion()
                    _rcheck(oldest.try_retire(), "Previous selected step did not retire")
                    self._unretired_steps.pop(0)

                for prev in list(self._unretired_steps):
                    comp = prev._completion
                    if comp is not None and hasattr(comp, "record"):
                        self._metadata_stream.wait_event(comp)
                    else:
                        prev.wait_completion()
                        _rcheck(prev.try_retire(), "Previous selected step did not retire")
                        self._unretired_steps.remove(prev)
            except BaseException:
                self._failed = True
                raise

        slot = self._next_host_slot
        self._next_host_slot ^= 1
        host_buf = self._clear_host_slot(slot)
        self._host_bases = host_buf

        active_domain = max((max(om | kp, default=0) + 1 for om, kp in zip(omitted_by_bank, kept_by_bank)), default=1)
        stride_domain = ((active_domain + 3) // 4) * 4
        step_shape = (self.num_layers, stride_domain, 2)
        step_elems = self.num_layers * stride_domain * 2
        self._step_host = host_buf.view(-1)[:step_elems].view(step_shape)
        self._step_dev = cast(Tensor, self._dev_bases).view(-1)[:step_elems].view(step_shape)
        step = LMCacheSelectedLoadStep(
            self,
            meta,
            classified,
            tbls,
            omitted_by_bank,
            self._step_dev.view(torch.uint64)[:, :active_domain],
            host_slot=slot,
        )
        self.step = step
        with self._steps_lock:
            self._unretired_steps.append(step)

        skip_g0_ids: frozenset[str] = frozenset()
        shared_pred = getattr(self.engine, "_should_use_shared_layerwise_retrieve", None)
        prep_skip_fn = getattr(self.gpu_connector, "prepare_layerwise_prefill_group0_skip", None)
        impl_skip_pred = getattr(self.impl, "_should_skip_group0_prefill_dma", None)
        active_banks = min(self.num_banks, self.num_layers)
        if (
            active_banks > 0
            and bool(getattr(self.impl, "_layerwise_prefill_dma", False))
            and bool(getattr(self.impl, "_layerwise_prefill_p_node", False))
            and not bool(getattr(self.impl, "enable_blending", False))
            and callable(shared_pred)
            and bool(shared_pred(0))
            and callable(prep_skip_fn)
            and callable(impl_skip_pred)
            and getattr(self.gpu_connector, "fence_debug", None) is None
            and not hasattr(self.gpu_connector, "kv_trace")
            and not hasattr(self.gpu_connector, "prefill_audit")
        ):
            eligible_skip: set[str] = set()
            for req in selected_reqs:
                rm = req_meta_by_id.get(req.request_id)
                cached_toks = int(_ga(rm, "load_spec.lmcache_cached_tokens", 0) or 0)
                vllm_cached = int(_ga(rm, "load_spec.vllm_cached_tokens", 0) or 0)
                if not (vllm_cached == 0 and 0 < cached_toks <= req.omit_end_token):
                    continue
                cached_pages = _pages(cached_toks)
                if cached_pages > 0 and all(
                    (rb := step._req_bank.get((req.request_id, b))) is not None
                    and len(rb[1]) >= cached_pages
                    and bool(rb[1][:cached_pages].all())
                    for b in range(active_banks)
                ):
                    eligible_skip.add(req.request_id)
            skip_g0_ids = frozenset(eligible_skip)

        recompute_set = getattr(self.engine, "_recompute_fallback_req_ids", None)
        if isinstance(recompute_set, set):
            recompute_set.clear()
        else:
            self.engine._recompute_fallback_req_ids = set()
        prev_step_fb = getattr(self.impl, "_step_recompute_fallback_req_ids", None)
        if isinstance(prev_step_fb, set):
            prev_step_fb.clear()
        self.impl._selected_skip_group0_dma_req_ids = skip_g0_ids
        omit_end_map: dict[str, int] = {
            r.request_id: int(r.omit_end_token) for r in selected_reqs if r.selected and r.omit_end_token > 0
        }
        self.impl._selected_group0_omit_end_tokens = omit_end_map
        orig_completed = False
        try:
            out = orig_start(forward_context, *args, **kw)
            orig_completed = True
            fallback_ids = (
                frozenset(getattr(self.engine, "_recompute_fallback_req_ids", ()) or ()) & step.selected_request_ids
            )
            if fallback_ids:
                for rid in fallback_ids:
                    omit_end_map.pop(rid, None)
                with step._lock:
                    step.requests = tuple(
                        replace(r, selected=False, omit_end_token=0) if r.request_id in fallback_ids else r
                        for r in step.requests
                    )
                    step.request_by_id = {r.request_id: r for r in step.requests}
                    step.selected_request_ids = frozenset(r.request_id for r in step.requests if r.selected)
                    step._shared_reqs.difference_update(fallback_ids)
                    step._populated_req_layers = {
                        pair for pair in step._populated_req_layers if pair[0] not in fallback_ids
                    }
                    for pop_s in step._non_shared_populated.values():
                        pop_s.difference_update(fallback_ids)
                    for rid in fallback_ids:
                        for b_idx in range(len(step.omitted_by_bank)):
                            step._req_bank.pop((rid, b_idx), None)
                            step._bank_plans.pop((rid, b_idx), None)
                    step.omitted_by_bank = tuple(
                        frozenset(
                            int(blk)
                            for (r_id, b_idx), (blks, p_mask) in step._req_bank.items()
                            if b_idx == bank_idx and r_id in step.selected_request_ids
                            for blk in blks[p_mask]
                        )
                        for bank_idx in range(len(step.omitted_by_bank))
                    )
                selected_reqs = [r for r in selected_reqs if r.request_id not in fallback_ids]
                if not selected_reqs:
                    self.impl._selected_group0_omit_end_tokens = {}
                    step.cancel_unused()
                    with self._steps_lock:
                        if step in self._unretired_steps:
                            self._unretired_steps.remove(step)
                    self.step = None
                    return out
            _rcheck(
                getattr(self.impl, "_deferred_layerwise_prefill_load_active", True) is not False,
                "Selected KV requires deferred layerwise prefill load",
            )
            init_limit = self._init_lim if _ga(self.impl, "_initial_layerwise_prefill_load_submitted") else 1
            init_ok = all(
                (req.request_id, lid) in step._populated_req_layers
                for req in selected_reqs
                for lid in range(init_limit)
            )
            _rcheck(
                init_ok,
                f"Selected KV initial layers (limit={init_limit}) unpopulated for {sorted(step.selected_request_ids)}",
            )
            with step._lock:
                step._initializing = False
                step._init_uploaded_limit = init_limit
                all_shared = step.selected_request_ids <= step._shared_reqs
                ns_reqs = step.selected_request_ids - step._shared_reqs
                ready_ns_lids = (
                    []
                    if all_shared
                    else sorted(
                        lid
                        for lid in range(self.num_layers)
                        if len(step._non_shared_populated.get(lid, ())) >= len(ns_reqs)
                        and all((rid, lid) in step._populated_req_layers for rid in step.selected_request_ids)
                    )
                )
            if all_shared:
                shared_ok = all(
                    (req.request_id, lid) in step._populated_req_layers
                    for req in selected_reqs
                    for lid in range(self.num_layers)
                )
                _rcheck(
                    shared_ok,
                    f"Selected KV shared prefill layers incomplete for {sorted(step.selected_request_ids)}",
                )
                self._record_upload(step)
            else:
                for lid in ready_ns_lids:
                    self._record_upload(step, lid)
            self._mark_slots_ready(range(init_limit))

            storers = _ga(self.impl, "_layerwise_save_storers")
            if isinstance(storers, Mapping):
                for key, storer in storers.items():
                    req_obj = (
                        step.request_by_id.get(key[0])
                        if isinstance(key, tuple) and len(key) >= 3 and key[2] == 0
                        else None
                    )
                    save_starts = _unwrap_storer_starts(storer) if req_obj is not None and req_obj.selected else None
                    if save_starts:
                        _rcheck(
                            min(map(int, save_starts)) >= req_obj.omit_end_token, f"Save overlaps omitted: {key[0]}"
                        )
            return out
        except BaseException:
            self._failed = True
            step._failed = True
            self.step = None
            self.impl._selected_group0_omit_end_tokens = {}
            if orig_completed:
                abort_ret = getattr(self.impl, "_abort_layerwise_retrieve_step", None)
                if callable(abort_ret):
                    with contextlib.suppress(BaseException):
                        abort_ret(loadable)
                abort_save = getattr(self.impl, "_abort_save_step", None)
                if callable(abort_save):
                    with contextlib.suppress(BaseException):
                        abort_save(meta.requests)
            with contextlib.suppress(BaseException):
                step.finish(failed=True)
            raise
        finally:
            recompute_end = getattr(self.engine, "_recompute_fallback_req_ids", None)
            if isinstance(recompute_end, set):
                if recompute_end:
                    sync_fb = getattr(self.impl, "_sync_recompute_fallback_from_engine", None)
                    if callable(sync_fb):
                        sync_fb(clear_engine=True)
                    else:
                        step_fb = getattr(self.impl, "_step_recompute_fallback_req_ids", None)
                        if not isinstance(step_fb, set):
                            step_fb = set()
                            self.impl._step_recompute_fallback_req_ids = step_fb
                        step_fb.update(recompute_end)
                recompute_end.clear()
            self.impl._selected_skip_group0_dma_req_ids = frozenset()
            self._active_ranges.clear()

    def _synchronize_group0_skip(self) -> None:
        sync_meta = getattr(self._metadata_stream, "synchronize", None)
        if callable(sync_meta):
            sync_meta()
        for rt in tuple(skv._ACTIVE_RUNTIMES):
            if getattr(rt, "source", None) is self:
                sync_copy = _ga(rt, "_copy_stream.synchronize")
                if callable(sync_copy):
                    sync_copy()
        if self.device.type == "npu":
            sync_dev = _ga(torch, "npu.synchronize")
            if callable(sync_dev):
                sync_dev(self.device)

    def _install_hooks(self, npu_mod: Any) -> None:
        if npu_mod is None:
            npu_mod = sys.modules.get(type(self.gpu_connector).__module__)
        if npu_mod is None or not hasattr(npu_mod, "bind_incremental_copy_addresses"):
            with contextlib.suppress(ImportError):
                import lmcache_ascend.v1.npu_connector.npu_connectors as npu_mod  # type: ignore

        _patch_bind_incremental(npu_mod)
        missing_hooks = [
            f"{owner_name}.{attr}"
            for owner_name, attr in _REQUIRED_HOOK_SPECS
            if not callable(getattr(getattr(self, owner_name), attr, None))
        ]
        _rcheck(not missing_hooks, f"Missing required LMCache layerwise prefill hooks: {missing_hooks}")

        init_range = range(self._init_lim)

        def _detach_cb(kind: str, orig: Any, req_id: str, *args: Any, **kw: Any) -> Any:
            if args or kw or not req_id:
                return lambda: orig(req_id, *args, **kw)
            if kind == "cpu":
                detach_fn = getattr(self.engine, "detach_shared_cpu_sparse_request", None)
                if callable(detach_fn):
                    return detach_fn(req_id)
                leases = getattr(self.engine, "_shared_cpu_request_leases", None)
                if isinstance(leases, dict):
                    lease = leases.pop(req_id, None)
                    if lease is None:
                        return None
                    if hasattr(lease, "active"):
                        lease.active = False
                    seq = getattr(self, "_detached_release_seq", 0) + 1
                    self._detached_release_seq = seq
                    dkey = f"{req_id}#detached#{seq}"
                    leases[dkey] = lease
                    return lambda: orig(dkey)
                return lambda: orig(req_id)
            detach_dma = getattr(self.gpu_connector, "detach_layerwise_prefill_dma_cache", None)
            if callable(detach_dma):
                return detach_dma(req_id)
            dma_attrs = (
                "_layerwise_prefill_unfenced_loads",
                "_layerwise_prefill_unfenced_stores",
                "_layerwise_prefill_async_store_events",
                "_layerwise_prefill_load_request_events",
                "_layerwise_prefill_load_owners",
            )
            host_attrs = (
                "_prefill_dma_bound_loads",
                "_prefill_dma_plan_cache",
                "_prefill_dma_source_cache",
            )
            for a in host_attrs:
                m = getattr(self.gpu_connector, a, None)
                if isinstance(m, dict):
                    m.pop(req_id, None)
            if any(isinstance(getattr(self.gpu_connector, a, None), dict) for a in dma_attrs):
                seq = getattr(self, "_detached_release_seq", 0) + 1
                self._detached_release_seq = seq
                dkey = f"{req_id}#detached#{seq}"
                moved = False
                for a in dma_attrs:
                    m = getattr(self.gpu_connector, a, None)
                    if isinstance(m, dict) and req_id in m:
                        m[dkey] = m.pop(req_id)
                        moved = True
                return (lambda: orig(dkey)) if moved else None
            return lambda: orig(req_id)

        def defer_release(
            ret_val: Any, kind: str, orig: Any, req_id: str, *args: Any, **kw: Any
        ) -> Any:
            with self._steps_lock:
                self._prune_retired_steps()
                owner_step = next(
                    (
                        step
                        for step in reversed(self._unretired_steps)
                        if req_id in step.selected_request_ids and not step._closed
                    ),
                    None,
                )
                if owner_step is not None:
                    with owner_step._lock:
                        if not owner_step._closed or owner_step._failed:
                            cb = _detach_cb(kind, orig, req_id, *args, **kw)
                            if cb is not None:
                                owner_step._deferred_cbs.append(cb)
                            return ret_val
            return orig(req_id, *args, **kw)

        def wrap_submit_layer(orig: Any, layer_name: str | int, *args: Any, **kw: Any) -> Any:
            res = orig(layer_name, *args, **kw)
            if getattr(self.impl, "_deferred_layerwise_prefill_load_active", True) is False:
                return res
            if layer_name == -1:
                self._mark_slots_ready(init_range)
            elif isinstance(layer_name, str) and (lid := self._layer_ids.get(layer_name)) is not None:
                if 0 <= (nxt := lid + self.num_banks) < self.num_layers:
                    self._mark_slots_ready((nxt,))
            return res

        def wrap_submit_initial(orig: Any, *args: Any, **kw: Any) -> Any:
            res = orig(*args, **kw)
            if getattr(self.impl, "_deferred_layerwise_prefill_load_active", True) is not False:
                self._mark_slots_ready(init_range)
            return res

        hook_bindings = (
            (self.gpu_connector, "batched_to_gpu", self._batched_to_gpu),
            (self.engine, "_prepare_shared_prefill_sources", self._prepare_shared),
            (
                self.engine,
                "release_shared_cpu_sparse_request",
                lambda orig, *a, **k: defer_release(False, "cpu", orig, *a, **k),
            ),
            (
                self.gpu_connector,
                "release_layerwise_prefill_dma_cache",
                lambda orig, *a, **k: defer_release(None, "dma", orig, *a, **k),
            ),
            (self.impl, "_submit_initial_layerwise_prefill_load", wrap_submit_initial),
            (self.impl, "submit_layerwise_prefill_load", wrap_submit_layer),
            (self.impl, "start_load_kv", self._start_load_kv),
        )
        for target, attr, handler in hook_bindings:
            orig = getattr(target, attr)
            setattr(target, attr, lambda *a, _o=orig, _h=handler, **kw: _h(_o, *a, **kw))
            self._restorers.append(lambda t=target, a=attr, o=orig: setattr(t, a, o))
        gc_ref: Any = self.gpu_connector
        had_sync_skip = hasattr(gc_ref, "synchronize_selected_group0_skip")
        orig_sync_skip = getattr(gc_ref, "synchronize_selected_group0_skip", None)
        gc_ref.synchronize_selected_group0_skip = self._synchronize_group0_skip

        def _restore_sync_skip() -> None:
            if had_sync_skip:
                gc_ref.synchronize_selected_group0_skip = orig_sync_skip
            elif hasattr(gc_ref, "synchronize_selected_group0_skip"):
                del gc_ref.synchronize_selected_group0_skip

        self._restorers.append(_restore_sync_skip)
        impl_ref: Any = self.impl
        if callable(getattr(impl_ref, "_prepare_layerwise_store_inputs", None)) and not bool(
            getattr(type(impl_ref), "_supports_selected_group0_omit_skip", False)
        ):
            orig_prep_store = impl_ref._prepare_layerwise_store_inputs

            def _wrap_prep_store(request: Any, save_spec: Any, kv_group: int, *args: Any, **kw: Any) -> Any:
                omit_end = 0
                if (
                    kv_group == 0
                    and not bool(getattr(request, "is_sparse_decode", False))
                    and not bool(getattr(request, "is_decode_window_save", False))
                ):
                    req_id = str(getattr(request, "req_id", ""))
                    eng_fb = getattr(self.engine, "_recompute_fallback_req_ids", None) or ()
                    step_fb = getattr(impl_ref, "_step_recompute_fallback_req_ids", None) or ()
                    if req_id not in eng_fb and req_id not in step_fb:
                        omit_map = getattr(impl_ref, "_selected_group0_omit_end_tokens", None)
                        if isinstance(omit_map, Mapping):
                            omit_end = int(omit_map.get(req_id, 0) or 0)
                if omit_end > 0:
                    cs = int(_ga(impl_ref, "_lmcache_chunk_size") or _ga(self.engine, "config.chunk_size") or 256)
                    omit_skip = (omit_end // cs) * cs
                    toks_in = getattr(request, "token_ids", ())
                    if omit_skip >= len(toks_in):
                        return None
                res = orig_prep_store(request, save_spec, kv_group, *args, **kw)
                if res is None or omit_end <= 0:
                    return res
                toks_out, smap_out, smask_out, skip_out, skw_out, win_out = res
                if omit_skip > int(skip_out):
                    if omit_skip >= len(toks_out):
                        return None
                    smask_new = smask_out.clone()
                    smask_new[:omit_skip] = False
                    skw_new = dict(skw_out)
                    if "_prefill_metadata_cache" in skw_new:
                        skw_new["_prefill_skip_tokens"] = omit_skip
                    if str(_ga(impl_ref, "kv_role") or "") == "kv_producer":
                        skw_new["_unverified_prefix_skip_tokens"] = omit_skip
                    return (toks_out, smap_out, smask_new, omit_skip, skw_new, win_out)
                return res

            impl_ref._prepare_layerwise_store_inputs = _wrap_prep_store

            def _restore_prep_store() -> None:
                impl_ref._prepare_layerwise_store_inputs = orig_prep_store

            self._restorers.append(_restore_prep_store)

    def close(self) -> None:
        if self._closed:
            return
        with self._steps_lock:
            if not self._failed:
                for step in list(self._unretired_steps):
                    if step._finished and not step._failed and not step._closed:
                        step.wait_completion()
                for up_ev in self._slot_upload_events:
                    if up_ev is not None:
                        up_ev.synchronize()
            self._prune_retired_steps()
            _rcheck(not self._failed and all(step._closed for step in self._unretired_steps), "DDR leases retained")
        for restore in reversed(self._restorers):
            restore()
        self._restorers.clear()
        if hasattr(self.impl, "_selected_group0_omit_end_tokens"):
            self.impl._selected_group0_omit_end_tokens = {}
        if getattr(self.impl, "selected_kv_source", None) is self:
            self.impl.selected_kv_source = None
        self._dva_cache.clear()
        self._slot_upload_events = [None] * MAX_INFLIGHT_STEPS
        for dirty in self._slot_dirty:
            dirty.clear()
        self._step_host = None
        self._step_dev = None
        self._host_bases_ring = None
        self._host_bases = None
        self._dev_bases = None
        self._closed = True
