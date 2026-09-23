# SPDX-License-Identifier: Apache-2.0
"""Tool-only worker hooks: collect the ends of one fixed-budget prefill request.

Registered as a vLLM worker extension at startup, then installed by string
collective RPC AFTER model startup. No production worker changes,
tensor inspection, per-layer checks, or extra per-chunk synchronization.
"""

EDGE_CHUNKS = 3
PREFIX = "[PREFILL_PROFILE]"


class TransferAttribution:
    """Temporary CPU profiler ranges; never synchronize or inspect device data."""

    def __init__(self, record_function):
        self.record_function = record_function
        self.patches = []

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        def traced(*args, **kwargs):
            title = label(*args, **kwargs) if callable(label) else label
            with self.record_function("PREFILL_ATTR/" + title):
                return original(*args, **kwargs)

        self.patches.append((owner, name, original))
        setattr(owner, name, traced)

    def restore(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()


def install_dma_diagnostics(ranges, ops):
    import inspect
    import json
    import os
    import time

    if getattr(ops.layerwise_prefill_dma_copy, "dummy_dma", False):
        ranges.wrap(ops, "layerwise_prefill_dma_copy", lambda *a, **kw: "DUMMY/" + dma_range_label(*a, **kw))
        return

    diagnose = getattr(ops, "layerwise_prefill_dma_copy_diagnose", None)
    if diagnose is None:
        raise RuntimeError(
            "DMA diagnostics require rebuilding LMCache-Ascend (missing layerwise_prefill_dma_copy_diagnose)"
        )
    original = ops.layerwise_prefill_dma_copy

    def traced(copies, device_to_host):
        # Read only the Python caller's scalar bookkeeping, never tensors.
        frame = inspect.currentframe()
        try:
            caller = frame.f_back.f_locals
            context = {key: caller.get(key) for key in ("layer_id", "kv_group", "bank")}
        finally:
            del frame
            del caller
        started = time.perf_counter()
        with ranges.record_function("PREFILL_ATTR/" + dma_range_label(copies, device_to_host)):
            stream_us, stream, copy_us = diagnose(copies, device_to_host)
        elapsed_ms = (time.perf_counter() - started) * 1000
        slowest = sorted(range(len(copy_us)), key=copy_us.__getitem__, reverse=True)[:5]

        def segment(i):
            dst, src, size = copies[i]
            return dict(index=i, us=round(copy_us[i], 3), src=hex(src), dst=hex(dst), bytes=size)

        report = dict(
            pid=os.getpid(),
            direction="D2H" if device_to_host else "H2D",
            **context,
            stream=hex(stream),
            stream_us=round(stream_us, 3),
            native_call_ms=round(elapsed_ms, 3),
            memcpy_sum_us=round(sum(copy_us), 3),
            segments=len(copies),
            bytes=sum(c[2] for c in copies),
            first=segment(0) if copies else None,
            slowest=[segment(i) for i in slowest],
            calls_over_1ms=sum(t >= 1000 for t in copy_us),
            src_range=[hex(min(c[1] for c in copies)), hex(max(c[1] + c[2] for c in copies))] if copies else [],
            dst_range=[hex(min(c[0] for c in copies)), hex(max(c[0] + c[2] for c in copies))] if copies else [],
        )
        print("[PREFILL_DMA] " + json.dumps(report, separators=(",", ":")), flush=True)

    ranges.patches.append((ops, "layerwise_prefill_dma_copy", original))
    ops.layerwise_prefill_dma_copy = traced


def dma_range_label(copies, device_to_host):
    # These are Python address/size tuples already prepared by the connector.
    # One range per batch, NOT per segment. No tensor access or stream query.
    direction = "D2H" if device_to_host else "H2D"
    return f"dma_submit/{direction}/segments={len(copies)}/bytes={sum(c[2] for c in copies)}"


def install_dummy_dma():
    import lmcache_ascend.c_ops as ops

    original = ops.layerwise_prefill_dma_copy

    def dummy(copies, device_to_host):
        pass

    dummy.dummy_dma = True
    ops.layerwise_prefill_dma_copy = dummy
    return lambda: setattr(ops, "layerwise_prefill_dma_copy", original)


def install_dummy_prepare():
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    # Keep scheduler/model bank metadata; bypass only worker cache work.
    # The test generates one token, with no subsequent decode/cache consumer.
    def start(self, *args, **kwargs):
        self._wait_for_save_done = False

    def noop(self, *args, **kwargs):
        pass

    def finish(self):
        # Acknowledge the diagnostic request without publishing any real KV.
        # Necessary for normal request cleanup; these results are INVALID.
        metadata = self._parent._get_connector_metadata()
        for request in metadata.requests:
            self._mark_prefill_committed(request, len(request.token_ids))
        self._complete_worker_save_step()

    replacements = {
        "start_load_kv": start,
        "wait_for_layer_load": noop,
        "submit_layerwise_prefill_load": noop,
        "save_kv_layer": noop,
        "finish_layerwise_prefill_save": noop,
        "wait_for_save": finish,
    }
    originals = [(name, getattr(LMCacheConnectorV1Impl, name)) for name in replacements]
    for name, replacement in replacements.items():
        setattr(LMCacheConnectorV1Impl, name, replacement)

    def restore():
        for name, original in originals:
            setattr(LMCacheConnectorV1Impl, name, original)

    return restore


def install_dummy_submit_load():
    """Keep submit/cursor state and rely on the dummy native DMA hook.

    ``submit_layerwise_prefill_load`` is not a pure enqueue call: it advances
    the deferred retriever and the layer cursor. Replacing it with a no-op
    makes the final wait observe cursor=0 and aborts the request. The native
    DMA operation is already replaced by :func:`install_dummy_dma`, so there
    is no transfer to suppress here; retaining the callback is the only
    state-correct diagnostic behavior.
    """
    return lambda: None


PREFILL_STORE_STAGES = {
    0: "skip_all_store_work",
    1: "request_group_dispatch",
    2: "refresh_and_group_lookup",
    3: "build_store_inputs_and_mask",
    4: "resolve_bank_slots_and_block_ids",
    5: "construct_store_generator",
    6: "prime_store_generator",
    7: "run_layer_save_finish_callbacks",
    8: "finalize_storers_and_promote_request_state",
    9: "publish_frontier_and_release_request_pins",
}


def install_dummy_prefill_store(stage=0):
    """Run a cumulative prefix of the P-node store path for profiling.

    Native DMA is independently replaced by ``install_dummy_dma``.  Stages
    0..7 stop progressively earlier in the save path. Stage 8 performs only
    the store completion needed to carry the next chunk: it drains each
    storer, consumes its result, and keeps the result request-owned. Stage 9
    adds the frontier publication and lookup-pin release after stage 8. Both
    stages are diagnostic-only because native DMA is replaced separately.
    """
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    if stage not in PREFILL_STORE_STAGES:
        raise ValueError(
            f"Invalid prefill-store stage {stage}; expected 0..9"
        )

    original_prepare = LMCacheConnectorV1Impl._prepare_p_node_layerwise_save_storers
    original_create = LMCacheConnectorV1Impl._create_p_node_layerwise_save_storer
    original_save_layer = LMCacheConnectorV1Impl.save_kv_layer
    original_finish_layer = LMCacheConnectorV1Impl.finish_layerwise_prefill_save
    original_wait = LMCacheConnectorV1Impl.wait_for_save

    def noop(self, *args, **kwargs):
        return None

    def prime_only_prepare(self, metadata):
        """Run production preparation while keeping stage 6/7 diagnostic."""
        engine = getattr(self, "lmcache_engine", None)
        had_flag = hasattr(engine, "_layerwise_prefill_diagnostic_prime_only")
        old_flag = getattr(engine, "_layerwise_prefill_diagnostic_prime_only", False)
        if engine is not None:
            engine._layerwise_prefill_diagnostic_prime_only = True
        try:
            return original_prepare(self, metadata)
        finally:
            if engine is not None:
                if had_flag:
                    engine._layerwise_prefill_diagnostic_prime_only = old_flag
                else:
                    delattr(engine, "_layerwise_prefill_diagnostic_prime_only")

    def staged_create(self, request, save_spec, kv_group):
        """Mirror production setup only through the requested boundary."""
        assert self._layerwise_prefill_p_node
        if stage < 2:
            return None

        self._refresh_kvcaches_list()
        kvcaches = self._kvcaches_for_group(kv_group)
        if not kvcaches or stage < 3:
            return None

        store_inputs = self._prepare_layerwise_store_inputs(
            request,
            save_spec,
            kv_group,
            materialize_device_slot_mapping=not self._layerwise_prefill_dma,
        )
        if store_inputs is None:
            return None
        (
            token_ids,
            _,
            store_mask,
            skip_leading_tokens,
            store_kwargs,
            _,
        ) = store_inputs
        if request.is_sparse_decode:
            raise RuntimeError(
                "Staged P-node store received a decode request: "
                f"req_id={request.req_id}"
            )
        if stage < 4:
            return None

        slot_mapping = self._layerwise_prefill_slot_mapping(request, kv_group, 0)
        dma_kwargs = {}
        if self._layerwise_prefill_dma:
            dma_kwargs = {
                "prefill_dma_block_ids_by_bank": self._layerwise_prefill_dma_block_ids(
                    request, kv_group
                ),
                "prefill_dma_block_size": self._block_size,
            }
        if stage < 5:
            return None

        metadata = getattr(self.lmcache_engine, "metadata", None)
        world_size = getattr(metadata, "world_size", 1) if metadata else 1
        sync = kv_group == 0 or (self._is_dsa_two_groups() and world_size > 1)
        return self.lmcache_engine.store_layer(
            token_ids,
            mask=store_mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
            sync=sync,
            deferred_layerwise_put=True,
            layerwise_prefill_incremental=True,
            layerwise_prefill_bank_count=2,
            **dma_kwargs,
            req_id=request.req_id,
            **store_kwargs,
        )

    def finish_save_stage7(self):
        # Close staged generators and acknowledge request cleanup. No result
        # is published in stages 0..7.
        storers = getattr(self, "_layerwise_save_storers", {})
        for storer in tuple(storers.values()):
            close = getattr(storer, "close", None)
            if close is not None:
                close()
        storers.clear()
        getattr(self, "_layerwise_prefill_prepared_storer_keys", set()).clear()
        getattr(self, "_layerwise_prefill_pending_store_finishes", {}).clear()
        metadata = self._parent._get_connector_metadata()
        for request in metadata.requests:
            self._mark_prefill_committed(request, len(request.token_ids))
        self._complete_worker_save_step()

    def finish_save_stage8(self, publish=False):
        """Complete stores needed for the following chunk.

        This is the smallest post-forward operation that leaves the
        layerwise storer maps in a usable state. It deliberately leaves
        frontier publication and lookup-pin release to stage 9 so the two
        costs can be measured separately.
        """
        metadata = self._parent._get_connector_metadata()
        if not self.use_layerwise or self.kv_role == "kv_consumer":
            return original_wait(self)
        try:
            if self._should_defer_latent_save_under_tp():
                for request in metadata.requests:
                    key = self._layerwise_save_storer_key(request, 0)
                    if key in getattr(self, "_deferred_latent_pending", set()):
                        self._flush_deferred_latent_store(request, request.save_spec)

            for request in metadata.requests:
                for kv_group in (0, 1):
                    storer_key = self._layerwise_save_storer_key(request, kv_group)
                    pending_layer = getattr(
                        self, "_layerwise_prefill_pending_store_finishes", {}
                    ).get(storer_key)
                    if pending_layer is not None:
                        raise RuntimeError(
                            "diagnostic stage reached a storer before its "
                            f"post-HCOM finish hook: req_id={request.req_id}, "
                            f"kv_group={kv_group}, layer={pending_layer}"
                        )
                    storer = self._layerwise_save_storers.pop(storer_key, None)
                    getattr(
                        self, "_layerwise_prefill_prepared_storer_keys", set()
                    ).discard(storer_key)
                    if storer is not None:
                        completed, store_result = self._finalize_layerwise_storer(storer)
                        self._consume_completed_layerwise_store(
                            request, kv_group, completed, store_result
                        )

                if publish:
                    if self._is_decode_window_save_request(request):
                        self._mark_decode_window_save_completed(request)
                    self._mark_prefill_committed(request)
                    self._mark_initial_sparse_release_ready(request)
                    self._maybe_lookup_unpin_for_request(request)
            self._complete_worker_save_step()
        except BaseException:
            self._abort_save_step(metadata.requests)
            raise

    replacements = {}
    if stage == 0:
        replacements["_prepare_p_node_layerwise_save_storers"] = noop
    elif 6 <= stage <= 7:
        replacements["_prepare_p_node_layerwise_save_storers"] = prime_only_prepare
    elif stage <= 5:
        replacements["_create_p_node_layerwise_save_storer"] = staged_create
    if stage <= 6:
        replacements["save_kv_layer"] = noop
        replacements["finish_layerwise_prefill_save"] = noop
    if stage <= 7:
        replacements["wait_for_save"] = finish_save_stage7
    elif stage == 8:
        replacements["wait_for_save"] = finish_save_stage8
    elif stage == 9:
        replacements["wait_for_save"] = lambda self: finish_save_stage8(self, publish=True)

    for name, replacement in replacements.items():
        setattr(LMCacheConnectorV1Impl, name, replacement)

    def restore():
        originals = {
            "_prepare_p_node_layerwise_save_storers": original_prepare,
            "_create_p_node_layerwise_save_storer": original_create,
            "save_kv_layer": original_save_layer,
            "finish_layerwise_prefill_save": original_finish_layer,
            "wait_for_save": original_wait,
        }
        for name in replacements:
            setattr(LMCacheConnectorV1Impl, name, originals[name])
    return restore


def install_dummy_dma_bind():
    from lmcache_ascend.v1.npu_connector import npu_connectors
    from types import SimpleNamespace

    original_copy = npu_connectors.bind_copy_addresses
    original_incremental = npu_connectors.bind_incremental_copy_addresses

    def dummy(*args, **kwargs):
        return []

    def dummy_incremental(plan, source_objs, starts, ends, npu_ptrs,
                          plane_widths, element_bytes, *args, **kwargs):
        # Keep the connector's ownership/state protocol intact while removing
        # only Python address-row construction. Native DMA is also replaced by
        # install_dummy_dma, so the empty rows are never submitted.
        return SimpleNamespace(
            owners=tuple(source_objs),
            owner_ids=tuple(map(id, source_objs)),
            starts=tuple(starts),
            ends=tuple(ends),
            npu_ptrs=tuple(npu_ptrs),
            plane_widths=tuple(plane_widths),
            element_bytes=element_bytes,
            segment_chunks=getattr(plan, "chunk", ()),
            rows=[],
        )

    npu_connectors.bind_copy_addresses = dummy
    npu_connectors.bind_incremental_copy_addresses = dummy_incremental

    def restore():
        npu_connectors.bind_copy_addresses = original_copy
        npu_connectors.bind_incremental_copy_addresses = original_incremental

    return restore


def install_transfer_attribution():
    import sys

    import torch

    ranges = TransferAttribution(torch.profiler.record_function)
    try:
        sfa = sys.modules["vllm_ascend.attention.sfa_v1"]
        ranges.wrap(sfa.AscendSFAImpl, "forward", lambda self, layer_name, *a, **kw: f"mla/{layer_name}")
        for name in (
            "exec_kv",
            "indexer_select_pre_process",
            "indexer_select_post_process",
            "_update_indexcache_topk_indices",
            "_get_indexcache_topk_indices",
        ):
            ranges.wrap(sfa.AscendSFAImpl, name, name)
        for name in ("maybe_submit_layerwise_prefill_load", "wait_for_kv_layer_from_connector"):
            ranges.wrap(sfa, name, lambda layer_name, *a, _name=name, **kw: f"{_name}/{layer_name}")
        ranges.wrap(sfa.torch_npu, "npu_scatter_nd_update_", "indexer_cache_write")
        if hasattr(sfa.torch_npu, "npu_lightning_indexer"):
            ranges.wrap(sfa.torch_npu, "npu_lightning_indexer", "native_lightning_indexer")
        # The plain custom-op invocation is nested inside post_process, so
        # internal ACLNN copies can be distinguished from connector DMA.
        ops = torch.ops._C_ascend
        for name in ("npu_lightning_indexer", "npu_lightning_indexer_quant"):
            if hasattr(ops, name):
                ranges.wrap(ops, name, name)
        connector = sys.modules.get("lmcache_ascend.v1.npu_connector.npu_connectors")
        if connector is not None:
            install_dma_diagnostics(ranges, connector.lmc_ops)
            # Attribute preparation in the real request as well as the
            # isolated probe. These functions return normally (not generators).
            for owner, names in (
                (connector, ("_prefill_dma_plans", "_cached_layerwise_slot_mapping")),
                (
                    getattr(connector, "VLLMPagedMemLayerwiseNPUConnector", None),
                    (
                        "_append_sparse_chunk_ptr_rows",
                        "_layer_page_pointer_rows",
                        "_check_layerwise_transfer_invariants",
                    ),
                ),
                (
                    getattr(
                        sys.modules.get("lmcache.integration.vllm.vllm_v1_adapter"), "LMCacheConnectorV1Impl", None
                    ),
                    (
                        "start_load_kv",
                        "_materialize_layerwise_prefill_slot_mappings",
                        "_prime_dense_prefix_retrievers",
                        "submit_layerwise_prefill_load",
                        "_advance_deferred_layerwise_prefill_load",
                        "_advance_dense_layerwise_retriever",
                    ),
                ),
                (
                    getattr(sys.modules.get("lmcache_ascend.v1.cache_engine"), "AscendLMCacheEngine", None),
                    ("_append_retrieve_group_cache",),
                ),
            ):
                for name in names:
                    if owner is not None and hasattr(owner, name):
                        # Static helper needs to retain its descriptor semantics.
                        if name == "_prime_dense_prefix_retrievers":
                            continue
                        ranges.wrap(owner, name, "prepare/" + name)
        return ranges
    except BaseException:
        ranges.restore()
        raise


def synchronize_boundary():
    # Only four window boundaries, not every chunk/layer/kernel. Flush all
    # streams so unfinished middle-chunk work cannot leak into the tail trace.
    import torch_npu

    torch_npu.npu.synchronize()


def make_capture_plan(prompt_tokens, chunk_tokens):
    chunks = (prompt_tokens + chunk_tokens - 1) // chunk_tokens
    windows = (
        [{"name": "all", "first_chunk": 1, "last_chunk": chunks}]
        if chunks <= 2 * EDGE_CHUNKS
        else [
            {"name": "head", "first_chunk": 1, "last_chunk": EDGE_CHUNKS},
            {"name": "tail", "first_chunk": chunks - EDGE_CHUNKS + 1, "last_chunk": chunks},
        ]
    )
    return {"prompt_tokens": prompt_tokens, "chunk_tokens": chunk_tokens, "total_chunks": chunks, "windows": windows}


class ChunkProfileCapture:
    def __init__(self, worker, case, plan):
        self.worker = worker
        self.case = case
        self.plan = plan
        self.original_execute = worker.execute_model
        self.active = None
        self.recorded_windows = []
        self.chunks = []
        self.tokens = 0
        self.attribution = None
        # Install before the request, including all unprofiled middle chunks.
        dummy_prepare = plan.get("dummy_prepare", False)
        dummy_bind = plan.get("dummy_dma_bind", False)
        dummy_submit_load = plan.get("dummy_submit_load", False)
        dummy_prefill_store = plan.get("dummy_prefill_store", False)
        dummy_prefill_store_stage = plan.get("dummy_prefill_store_stage")
        if dummy_prefill_store and dummy_prefill_store_stage is not None:
            raise ValueError(
                "capture plan cannot contain both dummy_prefill_store and "
                "dummy_prefill_store_stage"
            )
        if dummy_prefill_store:
            dummy_prefill_store_stage = 0
        self.restore_dma = install_dummy_dma() if (
            plan.get("dummy_dma", False)
            or dummy_prepare
            or dummy_bind
            or dummy_submit_load
            or dummy_prefill_store_stage is not None
        ) else None
        self.restore_bind = None
        self.restore_prepare = None
        self.restore_submit_load = None
        self.restore_prefill_store = None
        try:
            if dummy_bind:
                self.restore_bind = install_dummy_dma_bind()
                print(
                    f"{PREFIX} rank={worker.rank}: DUMMY DMA BIND; "
                    "bind_copy_addresses returns []; other preparation retained",
                    flush=True,
                )
            if dummy_prepare:
                self.restore_prepare = install_dummy_prepare()
                print(
                    f"{PREFIX} rank={worker.rank}: DUMMY PREPARE enabled; "
                    "LMCache worker preparation/transfer skipped; outputs INVALID",
                    flush=True,
                )
            if dummy_submit_load:
                self.restore_submit_load = install_dummy_submit_load()
                print(
                    f"{PREFIX} rank={worker.rank}: DUMMY SUBMIT LOAD enabled; "
                    "submit/cursor retained, native load DMA disabled; outputs INVALID",
                    flush=True,
                )
            if dummy_prefill_store:
                self.restore_prefill_store = install_dummy_prefill_store(0)
                print(
                    f"{PREFIX} rank={worker.rank}: DUMMY PREFILL STORE enabled; "
                    "load path retained, save path skipped; outputs INVALID",
                    flush=True,
                )
            elif dummy_prefill_store_stage is not None:
                self.restore_prefill_store = install_dummy_prefill_store(
                    dummy_prefill_store_stage
                )
                description = PREFILL_STORE_STAGES[dummy_prefill_store_stage]
                print(
                    f"{PREFIX} rank={worker.rank}: DUMMY PREFILL STORE STAGE "
                    f"{dummy_prefill_store_stage} ({description}); "
                    "outputs INVALID",
                    flush=True,
                )
        except BaseException:
            if self.restore_prefill_store:
                self.restore_prefill_store()
            if self.restore_submit_load:
                self.restore_submit_load()
            if self.restore_bind:
                self.restore_bind()
            if self.restore_dma:
                self.restore_dma()
            raise
        if self.restore_dma:
            print(f"{PREFIX} rank={worker.rank}: DUMMY DMA enabled for entire request; outputs INVALID", flush=True)

    def stop_window(self):
        if self.active is not None:
            print(f"{PREFIX} rank={self.worker.rank}: {self.case}/{self.active} profiler stop begin", flush=True)
            try:
                synchronize_boundary()
                self.worker.profile(is_start=False)
            finally:
                if self.attribution is not None:
                    self.attribution.restore()
                    self.attribution = None
            # Worker.profile() otherwise restarts the OLD trace name. Each
            # segment needs a fresh profiler and a distinct head/tail handler.
            self.worker.profiler = None
            self.active = None

    def execute_model(self, scheduler_output, *args, **kwargs):
        count = scheduler_output.total_num_scheduled_tokens
        if count > 0:
            chunk = len(self.chunks) + 1
            window = next(
                (w["name"] for w in self.plan["windows"] if w["first_chunk"] <= chunk <= w["last_chunk"]), None
            )
            if window != self.active:
                # Stop before the next execute_model, not after the previous
                # one: its sample_tokens/MTP RPC and async copies belong to it.
                self.stop_window()
                if window is not None:
                    synchronize_boundary()
                    print(
                        f"{PREFIX} rank={self.worker.rank}: {self.case}/{window} profiler start; chunk={chunk}",
                        flush=True,
                    )
                    self.worker.profile(is_start=True, profile_prefix=f"{self.case}_{window}")
                    self.active = window
                    self.attribution = install_transfer_attribution()
                    print(f"{PREFIX} rank={self.worker.rank}: PREFILL_ATTR ranges enabled", flush=True)
                    self.recorded_windows.append(window)
            self.chunks.append(
                {"chunk": chunk, "token_start": self.tokens, "token_end": self.tokens + count, "window": window}
            )
            self.tokens += count
        return self.original_execute(scheduler_output, *args, **kwargs)

    def finish(self):
        try:
            self.stop_window()
        finally:
            self.worker.execute_model = self.original_execute
            if self.restore_bind is not None:
                self.restore_bind()
                self.restore_bind = None
            if self.restore_submit_load is not None:
                self.restore_submit_load()
                self.restore_submit_load = None
            if self.restore_prepare is not None:
                self.restore_prepare()
                self.restore_prepare = None
            if self.restore_prefill_store is not None:
                self.restore_prefill_store()
                self.restore_prefill_store = None
            if self.restore_dma is not None:
                self.restore_dma()
                self.restore_dma = None
        return {"rank": self.worker.rank, "windows": self.recorded_windows, "chunks": self.chunks}


def install_chunk_profile(worker, case, plan):
    """Callable collective RPC; the parent invokes this once, before generate."""
    # MultiprocExecutor passes WorkerWrapperBase. Mutate the actual worker,
    # not the proxy (whose __getattr__ forwards reads but NOT assignments).
    worker = getattr(worker, "worker", worker)
    capture = ChunkProfileCapture(worker, case, plan)
    worker._prefill_chunk_profile_capture = capture
    worker.execute_model = capture.execute_model


def finish_chunk_profile(worker):
    worker = getattr(worker, "worker", worker)
    capture = worker._prefill_chunk_profile_capture
    try:
        return capture.finish()
    finally:
        del worker._prefill_chunk_profile_capture


class ChunkProfileWorkerExtension:
    """Expose tool-only capture hooks through serializable RPC method names."""

    def install_chunk_profile(self, case, plan):
        return install_chunk_profile(self, case, plan)

    def finish_chunk_profile(self):
        return finish_chunk_profile(self)


def validate_capture(plan, workers):
    """Post-request coverage check, never on a production execution path."""
    expected = [
        (start, min(start + plan["chunk_tokens"], plan["prompt_tokens"]))
        for start in range(0, plan["prompt_tokens"], plan["chunk_tokens"])
    ]
    windows = [w["name"] for w in plan["windows"]]
    if not workers:
        raise RuntimeError("No worker capture reports; cannot verify chunk coverage")
    for report in workers:
        actual = [(c["token_start"], c["token_end"]) for c in report["chunks"]]
        if actual != expected or report["windows"] != windows:
            raise RuntimeError(
                f"rank={report['rank']}: actual prefill chunk layout differs from the capture plan; "
                "see capture_windows.json. Do not treat these traces as the first/last three chunks."
            )
