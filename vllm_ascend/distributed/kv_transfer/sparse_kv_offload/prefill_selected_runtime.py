# SPDX-License-Identifier: Apache-2.0
"""Device-count union planner and owner/follower ondemand-selected runtime."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Mapping, Sequence
from math import prod
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

import vllm_ascend.core.selected_kv as skv
from vllm_ascend.core.selected_kv import _check, _pages, _rcheck, _run_cleanup

_LIBS = ("sparse_prefill_fused_fixed_launch", "dsa_bf16_mla_inplace")


def _i64(val: int | None) -> int:
    return ctypes.c_int64(0 if val is None else val).value


def _plan_shapes(
    cap: int, q_cap: int, max_reqs: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    max_hist = skv.MAX_HISTORY_TOKENS
    cohorts = (q_cap + 1) // 2
    dims_ok = (
        all(type(v) is int and v > 0 for v in (cap, q_cap, max_reqs))
        and cap <= max_hist
        and cap % skv.PAGE_ROWS == 0
        and q_cap <= skv.MAX_BATCHED_TOKENS
        and max_reqs <= skv.MAX_REQUESTS
    )
    _check(dims_ok, "Plan bounds exceeded")
    scratch_shapes = ((cohorts, 4096), (cohorts, 8), (max_hist,), (32, 8), (32, 8), (max_hist // 32,))
    output_shapes = ((max_reqs, cap), (max_reqs, 8))
    return scratch_shapes, output_shapes


def selected_workspace_bytes(q_cap: int, max_reqs: int) -> int:
    scratch_shapes, output_shapes = _plan_shapes(min(skv.MAX_HISTORY_TOKENS, skv.TOPK * q_cap), q_cap, max_reqs)
    total_ints = sum(prod(shape) for shape in (*scratch_shapes, *output_shapes)) + min(max_reqs, q_cap) * 8
    return 4 * total_ints


class SelectedKVPlanner:
    def __init__(self, device: torch.device | str, capacity: int, query_capacity: int, max_requests: int) -> None:
        scratch_shapes, output_shapes = _plan_shapes(capacity, query_capacity, max_requests)
        dev = torch.device(device)
        self.device = dev
        self.capacity = capacity
        self.query_capacity = query_capacity
        self.max_requests = max_requests
        self.pending: list[Any] = []
        self._closed = False
        self.handle = ctypes.c_void_p()

        lib_dir = Path(__file__).resolve().parents[3] / "lib"
        torch.ops.load_library(str(lib_dir / f"lib{_LIBS[0]}.so"))
        fused_lib, mla_lib = (ctypes.CDLL(str(lib_dir / f"lib{name}.so")) for name in _LIBS)
        self._libs = (fused_lib, mla_lib)
        self.ops = torch.ops.dsa_prefill_queue

        create_fn = fused_lib.create_sparse_prefill_fixed_handle
        destroy_fn = fused_lib.destroy_sparse_prefill_fused_handle
        create_fn.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        create_fn.restype = ctypes.c_int32
        destroy_fn.argtypes = [ctypes.c_void_p]
        destroy_fn.restype = ctypes.c_int32
        self._destroy = destroy_fn

        self.fused_launch = _i64(ctypes.cast(fused_lib.launch_sparse_prefill_fused_plan, ctypes.c_void_p).value)
        self.mla_copy_launch = _i64(ctypes.cast(mla_lib.launch_sparse_prefill_bf16_mla_inplace, ctypes.c_void_p).value)
        self.scratch = [torch.empty(shape, dtype=torch.int32, device=dev) for shape in scratch_shapes]
        self.outputs = [torch.empty(shape, dtype=torch.int32, device=dev) for shape in output_shapes]

        with torch.npu.device(dev):
            rc = create_fn(bytes(lib_dir / "sparse_prefill_fused_fixed.o"), capacity, ctypes.byref(self.handle))
        _rcheck(
            rc == 0,
            "Selected KV kernels require >= 32 vector cores" if rc == -2 else f"Fused handle init failed: {rc}",
        )

    def build(
        self, topk: Tensor, histories: Sequence[int], starts: Sequence[int], queries: Sequence[int]
    ) -> tuple[Tensor, Tensor]:
        _rcheck(not self._closed, "Selected-KV planning requires a live workspace")
        num_reqs = len(histories)
        topk_ok = (
            0 < num_reqs <= self.max_requests
            and (topk.ndim, topk.device) == (3, self.device)
            and 0 < topk.shape[0] <= self.query_capacity
        )
        _check(topk_ok, "Invalid selected TopK")
        self.pending.append((topk, self.outputs, self.scratch))
        self.ops.batch_plan(
            topk,
            self.scratch,
            self.outputs,
            list(histories),
            list(starts),
            list(queries),
            self.capacity,
            self.fused_launch,
            _i64(self.handle.value),
        )
        return self.outputs[0][:num_reqs], self.outputs[1][:num_reqs]

    def load_mla(self, ids: Tensor, *rest: Tensor) -> None:
        _rcheck(not self._closed, "Selected copies require a live workspace")
        ids_ok = (ids.ndim, ids.shape[1:], ids.device) == (2, (self.capacity,), self.device) and 0 < ids.shape[
            0
        ] <= self.max_requests
        _check(ids_ok, "Invalid selected IDs")
        self.pending.append((ids, *rest))
        self.ops.mla_inplace_copy(ids, *rest, self.mla_copy_launch)

    def release_quiescent(self) -> None:
        self.pending.clear()

    def close_quiescent(self) -> None:
        if self._closed:
            return
        with torch.npu.device(self.device):
            _rcheck(self._destroy(self.handle) == 0, "Cannot unload selected-KV planning")
        self.handle = ctypes.c_void_p()
        self._closed = True
        self.pending.clear()


def _validate_mla_cache(cache: Sequence[Tensor], dev: torch.device) -> tuple[Tensor, Tensor]:
    _check(not isinstance(cache, Tensor) and len(cache) >= 2, "Selected KV requires (k_nope, k_pe) cache planes")
    k_nope, k_pe = cache[0], cache[1]

    def plane_ok(tensor: Tensor, dtypes: tuple[Any, ...], row_dim: int) -> bool:
        shape_ok = (tensor.ndim, *tensor.shape[1:], tensor.is_contiguous(), tensor.device) == (
            4,
            skv.PAGE_ROWS,
            1,
            row_dim,
            True,
            dev,
        )
        return shape_ok and tensor.shape[0] > 0 and tensor.dtype in dtypes

    valid = (
        plane_ok(k_nope, (torch.bfloat16, torch.float16), skv.NOPE_ROW_DIM)
        and plane_ok(k_pe, (k_nope.dtype,), skv.PE_ROW_DIM)
        and k_pe.shape[0] == k_nope.shape[0]
    )
    _check(valid, "Invalid selected MLA cache")
    return k_nope, k_pe


class SelectedKVRuntime:
    def __init__(
        self,
        runner: Any,
        owner_by_layer: Mapping[str, str],
        main_caches: Mapping[str, Any],
        source: Any,
        planner: Any = None,
        draft_layer_names: Iterable[str] = (),
    ) -> None:
        owners = dict(owner_by_layer)
        dev = torch.device(runner.device)
        self.device = dev
        self.source = source
        self.owner_by_layer = owners
        self.draft_layer_names = frozenset(map(str, draft_layer_names))
        self.target_layer_names = frozenset(owners) - self.draft_layer_names

        owners_ok = (
            owners.keys() == main_caches.keys()
            and self.draft_layer_names < owners.keys()
            and all(owner in owners for owner in owners.values())
            and all(owners[name] == name for name in self.draft_layer_names)
        )
        _check(owners_ok, "Owners")
        self.main_caches = {name: _validate_mla_cache(main_caches[name], dev) for name in owners}

        layer_names = tuple(owners)
        self._next_follower = {cur: nxt for cur, nxt in zip(layer_names, layer_names[1:]) if owners[nxt] == owners[cur]}

        sc = runner.vllm_config.scheduler_config
        self._copy_stream = torch.npu.Stream(device=dev)
        q_cap = int(sc.max_num_batched_tokens)
        max_reqs = min(int(sc.max_num_seqs), q_cap)
        _check(0 < q_cap <= skv.MAX_BATCHED_TOKENS and 0 < max_reqs <= skv.MAX_REQUESTS, "Bounds exceeded")
        self.query_capacity = q_cap
        self.max_requests = max_reqs

        self._dev_meta: Tensor | None = torch.empty((2, max_reqs, 4), dtype=torch.int32, device=dev)
        self._host_meta: Tensor | None = torch.empty((2, max_reqs, 4), dtype=torch.int32, pin_memory=dev.type != "cpu")
        self._failed = False
        self._closed = False
        self._awaiting_draft = False
        self._reset_step_state()
        union_cap = min(skv.MAX_HISTORY_TOKENS, skv.TOPK * q_cap)
        self.selection_planner = planner or SelectedKVPlanner(dev, union_cap, q_cap, max_reqs)
        skv._ACTIVE_RUNTIMES.add(self)

    def _reset_step_state(self) -> None:
        self.source_step: Any | None = None
        self._compute_stream: Any | None = None
        self.current_owner: str | None = None
        self.current_plan: Any | None = None
        self.request_metadata: Tensor | None = None
        self._active = False
        self._selected_indices: tuple[int, ...] = ()
        self._num_reqs = 0
        self.histories: list[int] = []
        self.starts: list[int] = []
        self.queries: list[int] = []
        self._metadata_by_layer: dict[str, Any] = {}
        self._prefetched: dict[str, Any] = {}
        self._submitted_layers: set[str] = set()
        self._waited_evs: set[int] = set()

    def _check_stream(self, msg: str = "Selected execution must stay on bound compute stream") -> Any:
        stream = torch.npu.current_stream()
        _rcheck(stream.npu_stream == self._compute_stream, msg)
        return stream

    def _scrub_metas(self, metas: Iterable[Any] | None = None) -> None:
        targets = self._metadata_by_layer.values() if metas is None else metas
        for meta in targets:
            if getattr(meta, "selected_kv", None) is self:
                meta.selected_kv = None

    def fail(self, step: Any | None = None, err: BaseException | None = None) -> None:
        self._awaiting_draft = False
        self._active = False
        self._failed = True
        if step is None or getattr(step, "_finished", False):
            self._scrub_metas()
            return

        def finish_failed() -> None:
            if step is self.source_step:
                self.finish_step(failed=True)
            else:
                step.finish(failed=True)

        _run_cleanup(finish_failed, err)

    def _apply_step_layout(
        self,
        layer_names: Iterable[str],
        meta: Mapping[str, Any],
        qsl: Sequence[int],
        seq_lens: Sequence[int],
    ) -> SelectedKVRuntime:
        num_reqs = self._num_reqs
        qsl_ok = (
            len(qsl) == num_reqs + 1
            and len(seq_lens) == num_reqs
            and qsl[0] == 0
            and 0 < qsl[-1] <= self.query_capacity
        )
        _check(qsl_ok, "Bad QSL")
        queries = [qsl[i + 1] - qsl[i] for i in range(num_reqs)]
        _check(
            all(q > 0 and seq_lens[i] >= q for i, q in enumerate(queries)) and all(k in meta for k in layer_names),
            "Bad meta",
        )

        max_cols = max(_pages(length) for length in seq_lens)
        by_layer = {name: meta[name] for name in layer_names}
        for layer_meta in {id(val): val for val in by_layer.values()}.values():
            bt = layer_meta.block_table
            bt_ok = (
                (layer_meta.num_actual_tokens, layer_meta.block_size, bt.ndim, bt.dtype)
                == (qsl[-1], skv.PAGE_ROWS, 2, torch.int32)
                and bt.device == self.device
                and bt.shape[0] >= num_reqs
                and bt.shape[1] >= max_cols
                and bt.stride(1) == 1
                and bt.stride(0) >= bt.shape[1]
            )
            _check(bt_ok, "Bad block_table")

        sel_starts = [qsl[i] for i in self._selected_indices]
        sel_queries = [queries[i] for i in self._selected_indices]
        if self.request_metadata is None or (sel_starts, sel_queries) != (self.starts, self.queries):
            stage = int(self._awaiting_draft)
            num_sel = len(self._selected_indices)
            self.starts = sel_starts
            self.queries = sel_queries
            host_view = cast(Tensor, self._host_meta)[stage, :num_sel]
            host_view.numpy()[:] = list(zip(self._selected_indices, self.histories, sel_starts, sel_queries))
            self.request_metadata = cast(Tensor, self._dev_meta)[stage, :num_sel].copy_(host_view, non_blocking=True)

        self._metadata_by_layer.update(by_layer)
        for layer_meta in by_layer.values():
            layer_meta.selected_kv = self
        self._active = True
        return self

    def _copy_layer(
        self,
        step: Any,
        name: str,
        stream: Any,
        plan: Any,
        block_table: Tensor,
        caches: tuple[Tensor, Tensor],
        *events: Any,
    ) -> None:
        step.wait_main_slot_ready(name, stream)
        for ev in events:
            if ev is not None:
                stream.wait_event(ev)
        self.selection_planner.load_mla(
            *plan,
            step.page_bases(name),
            cast(Tensor, self.request_metadata),
            block_table,
            *caches,
        )

    def _retire_previous(self) -> None:
        step = self.source_step
        if step is None:
            return
        _rcheck(not self._failed, "Failed selected execution retains its workspace and cannot be reused")
        if self._awaiting_draft:
            self.fail(step)
            raise RuntimeError("Deferred selected draft pass did not complete before next step")
        step.wait_completion()
        _rcheck(step.try_retire(), "The previous selected source still has live readers")
        self.selection_planner.release_quiescent()
        self._scrub_metas()
        self._reset_step_state()

    def bind_step(self, step: Any | None, req_ids: Sequence[str], hists: Any, q_starts: Any, attn_meta: Any) -> Any:
        _rcheck(not self._closed, "Selected KV runtime is closed")
        self._retire_previous()
        if step is None:
            if isinstance(attn_meta, Mapping):
                self._scrub_metas(attn_meta.values())
            return None
        _rcheck(not self._failed, "Failed selected execution retains its workspace and cannot be reused")
        self.source_step = step
        _check(isinstance(attn_meta, Mapping), "Selected prefill requires one eager, unsliced metadata batch")

        ids = tuple(req_ids)
        histories = [int(x) for x in hists]
        starts = [int(x) for x in q_starts]
        reqs_by_id = {req.request_id: req for req in step.requests}
        batch_ok = (
            bool(ids)
            and len(ids) == len(set(ids)) == len(reqs_by_id) == len(step.requests) <= self.max_requests
            and set(ids) == reqs_by_id.keys()
            and len(histories) == len(ids)
            and len(starts) == len(ids) + 1
        )
        _check(batch_ok, "Bad batch")

        queries = [end - start for start, end in zip(starts, starts[1:])]
        sel_indices = tuple(idx for idx, rid in enumerate(ids) if reqs_by_id[rid].selected)
        hq_ok = all(
            (reqs_by_id[rid].history_tokens, reqs_by_id[rid].query_tokens) == (h, q)
            for rid, h, q in zip(ids, histories, queries)
        )
        hist_ok = bool(sel_indices) and all(
            skv.MIN_HISTORY_TOKENS <= histories[idx] <= skv.MAX_HISTORY_TOKENS for idx in sel_indices
        )
        _check(hq_ok and hist_ok, "Invalid selected batch H/Q")

        self._scrub_metas(attn_meta[name] for name in self.draft_layer_names if name in attn_meta)
        self._selected_indices = sel_indices
        self._num_reqs = len(ids)
        self.histories = [histories[idx] for idx in sel_indices]
        self._compute_stream = torch.npu.current_stream().npu_stream
        seq_lens = [h + q for h, q in zip(histories, queries)]
        return self._apply_step_layout(self.target_layer_names, attn_meta, starts, seq_lens)

    def defer_to_draft_step(self) -> None:
        if self.source_step is None:
            return
        ready_for_draft = (
            self._active
            and not self._failed
            and self._submitted_layers == self.target_layer_names
            and bool(self.draft_layer_names)
            and torch.npu.current_stream().npu_stream == self._compute_stream
        )
        if not ready_for_draft:
            self.finish_step(failed=True)
            raise RuntimeError("Target layers incomplete before draft")
        self._awaiting_draft = True

    def bind_draft_step(self, draft_meta: Mapping[str, Any], common_meta: Any) -> Any:
        _rcheck(not self._closed, "Selected KV runtime is closed")
        if not self._awaiting_draft:
            self._scrub_metas(draft_meta.values())
            return None
        _rcheck(self._active and not self._failed and self.source_step is not None, "Bad draft bind")
        _check(bool(self.draft_layer_names) and not (self._submitted_layers & self.draft_layer_names), "Dup draft")

        qsl_raw = getattr(common_meta, "query_start_loc_cpu", None)
        sl_raw = getattr(common_meta, "seq_lens_cpu", None)
        _check(qsl_raw is not None and sl_raw is not None, "Selected draft bind requires CPU query_start_loc/seq_lens")

        def to_ints(vals: Any, limit: int) -> list[int]:
            seq = vals.tolist() if hasattr(vals, "tolist") else list(vals)
            return [int(x) for x in seq[:limit]]

        num_reqs = self._num_reqs
        qsl = to_ints(qsl_raw, num_reqs + 1)
        seq_lens = to_ints(sl_raw, num_reqs)
        hist_ok = (
            len(qsl) == num_reqs + 1
            and len(seq_lens) == num_reqs
            and all(
                seq_lens[idx] - (qsl[idx + 1] - qsl[idx]) == hist
                for idx, hist in zip(self._selected_indices, self.histories)
            )
        )
        _check(hist_ok, "Bad draft history")
        self._check_stream("Selected draft bind must stay on bound compute stream")
        return self._apply_step_layout(self.draft_layer_names, draft_meta, qsl, seq_lens)

    def load(self, name: str, topk: Tensor, cache: Sequence[Tensor], metadata: Any) -> None:
        num_tokens = int(metadata.num_actual_tokens)
        if num_tokens <= 0:
            return
        layer_ok = (
            self._active
            and not (self._closed or self._failed)
            and name not in self._submitted_layers
            and name in self.owner_by_layer
            and metadata is self._metadata_by_layer.get(name)
        )
        _rcheck(layer_ok, "Bad layer")

        sliced = topk[:num_tokens] if isinstance(topk, Tensor) and topk.shape[0] > num_tokens else topk
        topk_3d = sliced.unsqueeze(1) if sliced.ndim == 2 else sliced
        topk_ok = (
            topk_3d.ndim == 3
            and topk_3d.dtype == torch.int32
            and topk_3d.shape[1:] == (1, skv.TOPK)
            and topk_3d.is_contiguous()
            and topk_3d.device == self.device
            and num_tokens <= topk_3d.shape[0] <= self.query_capacity
        )
        _check(topk_ok, "Invalid selected TopK")

        cache_planes = _validate_mla_cache(cache[:2] if isinstance(cache, (tuple, list)) else cache, self.device)
        stream = self._check_stream()
        owner = self.owner_by_layer[name]
        if name == owner:
            self.current_plan = self.selection_planner.build(topk_3d, self.histories, self.starts, self.queries)
            self.current_owner = owner
        else:
            _rcheck(owner == self.current_owner, "Selected TopK follower ran outside its current owner group")

        step = cast(Any, self.source_step)
        ready_ev = step.ready(name)
        if ready_ev is not None and id(ready_ev) not in self._waited_evs:
            stream.wait_event(ready_ev)
            self._waited_evs.add(id(ready_ev))

        if name in self._prefetched:
            comp_ev = self._prefetched[name]
            _rcheck(comp_ev is not None, "Native slot readiness did not submit prefetch")
            _check(
                all(a.data_ptr() == b.data_ptr() for a, b in zip(cache_planes, self.main_caches[name])),
                "Prefetch cache",
            )
            stream.wait_event(comp_ev)
        else:
            self._copy_layer(step, name, stream, self.current_plan, metadata.block_table, cache_planes)
        self._submitted_layers.add(name)

    def prefetch_next(self, name: str, metadata: Any = None) -> None:
        if metadata is not None and int(metadata.num_actual_tokens) <= 0:
            return
        _rcheck(self._active and not (self._closed or self._failed) and name in self._submitted_layers, "Bad prefetch")
        nxt = self._next_follower.get(name)
        if nxt is None:
            return
        _rcheck(nxt not in self._prefetched and nxt not in self._submitted_layers, "Duplicate follower prefetch")
        self._check_stream("Selected follower prefetch off bound compute stream")

        plan_ready_ev = torch.npu.Event()
        plan_ready_ev.record()
        plan = self.current_plan
        step = cast(Any, self.source_step)
        copy_stream = self._copy_stream
        follower_table = self._metadata_by_layer[nxt].block_table
        follower_caches = self.main_caches[nxt]
        self._prefetched[nxt] = None

        def submit() -> None:
            with torch.npu.stream(copy_stream):
                self._copy_layer(
                    step, nxt, copy_stream, plan, follower_table, follower_caches, step.ready(nxt), plan_ready_ev
                )
                done_ev = torch.npu.Event()
                done_ev.record(copy_stream)
                self._prefetched[nxt] = done_ev

        step.prefetch_main(nxt, submit)

    def finish_step(self, *, failed: bool = False) -> None:
        all_submitted = (
            self._active and not (self._failed or failed) and self._submitted_layers == self.owner_by_layer.keys()
        )
        self._awaiting_draft = False
        self._active = False
        step = self.source_step
        if step is None:
            return
        try:
            if not all_submitted:
                self._failed = True
                step.finish(failed=True)
                _rcheck(failed, "Selected execution did not successfully submit every layer")
                return
            self._check_stream("Selected completion must stay on bound compute stream")
            comp_ev = torch.npu.Event()
            comp_ev.record()
            step.finish(completion=comp_ev)
        except BaseException as err:
            self._failed = True
            if not getattr(step, "_finished", False):
                _run_cleanup(lambda: step.finish(failed=True), err)
            raise
        finally:
            self._active = False
            self._scrub_metas()

    def close(self) -> None:
        if self._closed:
            return
        _rcheck(not self._failed and not getattr(self.source, "_failed", False), "Workspace retained on failure")
        self._retire_previous()
        self.source.close()
        self.selection_planner.close_quiescent()
        skv._ACTIVE_RUNTIMES.discard(self)
        self._dev_meta = None
        self._host_meta = None
        self.request_metadata = None
        self._closed = True
