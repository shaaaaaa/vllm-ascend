# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch unchanged Q2 SFA batches before reconciling CPU acceptance counts.

Only device metadata is advanced early. All request/bookkeeping mutations still
run through the original runner, with real counts, after successful submission.
"""

from contextlib import ExitStack, contextmanager
from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, triton

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import _decode_window_save_window_size
from vllm_ascend.attention.utils import staged_sfa_metadata_sparse_route
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.spec_decode.async_mtp import prepare_async_mtp_kernel, prepare_async_mtp_tokens_kernel
from vllm_ascend.utils import StagedSFARouteReason, lmhead_tp_enable, sfa_full_graph_enabled
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner, _mtp_dw_diag_enabled, npu_content_diagnostics_enabled


@dataclass
class _Snapshot:
    ids: tuple
    requests: tuple
    bases: np.ndarray
    metadata: dict
    common: Any
    shape: tuple
    key: Any
    frontiers: tuple
    boundary: tuple
    window: int
    boundary_values: tuple
    target_name: str


def _shape(kwargs):
    return tuple(
        kwargs.get(k)
        for k in ("num_tokens", "num_tokens_padded", "num_reqs", "num_reqs_padded", "max_query_len", "full_graph")
    )


def _boundary(bases, frontiers, window):
    return tuple(min((int(c) + 1) // window * window, f) if window > 0 else f for c, f in zip(bases, frontiers))


class AsyncSFAModelRunner(NPUModelRunner):
    def __init__(self, *args, **kwargs):
        self._async_live_execute = False
        self._async_host_write = None
        self._async_query_layout = None
        self._async_padded_logits = None
        super().__init__(*args, **kwargs)
        if not (
            HAS_TRITON
            and sfa_full_graph_enabled(self.vllm_config)
            and self.use_async_scheduling
            and self._fixed_mtp_metadata is not None
            and self.parallel_config.pipeline_parallel_size == 1
            and not self.use_cp
        ):
            raise ValueError("SFA async MTP preparation requires full SFA graphs, async MTP=1 and PP=PCP=DCP=1")
        self._async_bases = torch.empty(self.max_num_reqs, dtype=torch.int32, device=self.device)
        self._async_snapshot = self._async_pending = self._async_built = None
        self._async_counts = None
        self._async_epoch, self._async_counts_epoch = 0, -1
        self.async_mtp_replays = 0

    @torch.inference_mode()
    def capture_model(self, *args, **kwargs):
        self._async_snapshot = self._async_query_layout = None
        result = super().capture_model(*args, **kwargs)
        # Compile each metadata specialization outside capture and before serving,
        # using private outputs. Never pay first-use compilation in a live batch.
        groups = self.input_batch.block_table.block_tables
        if len(groups) == 2 and self._sfa_full_graph is not None:
            tables = [group.block_table.gpu for group in groups]
            counts = torch.ones(self.max_num_reqs, dtype=torch.int64, device=self.device)
            for key in self._sfa_full_graph.entries:
                n, capacity = key.request_capacity, key.token_capacity
                self._async_bases.zero_()
                positions = self.positions.gpu.new_empty(capacity)
                lengths = [self.seq_lens.gpu.new_empty(n + extra) for extra in (0, 1)]
                slots = [group.slot_mapping.gpu.new_empty(capacity) for group in groups]
                prepare_async_mtp_kernel[(triton.cdiv(max(capacity // 2, n + 1), 32),)](
                    self._async_bases,
                    counts,
                    positions,
                    *lengths,
                    *tables,
                    *slots,
                    n,
                    capacity,
                    n,
                    n + 1,
                    tables[0].stride(0),
                    tables[1].stride(0),
                    groups[0].block_size,
                    groups[1].block_size,
                    BLOCK=32,
                )
            # Private token buffers: compile both supported proposal dtypes.
            tokens = self.input_ids.gpu.new_zeros(2 * self.max_num_reqs)
            for dtype in (torch.int32, torch.int64):
                draft = torch.zeros(self.max_num_reqs, dtype=dtype, device=self.device)
                prepare_async_mtp_tokens_kernel[(triton.cdiv(self.max_num_reqs, 32),)](
                    tokens[:self.max_num_reqs].clone(), draft, tokens, tokens.new_empty(self.max_num_reqs),
                    self.max_num_reqs, BLOCK=32,
                )
            self.drafter.warmup_next_mtp_tokens()
            torch.npu.current_stream().synchronize()
        return result

    def execute_model(self, *args, **kwargs):
        self._async_live_execute = True
        try:
            return super().execute_model(*args, **kwargs)
        except BaseException:
            self._async_snapshot = self._async_pending = self._async_query_layout = None
            raise
        finally:
            self._async_live_execute = False

    @contextmanager
    def synchronize_input_prep(self):
        if not self._async_live_execute:
            with super().synchronize_input_prep():
                yield
            return
        self._async_host_write = False
        try:
            yield
        finally:
            try:
                if self._async_host_write and self.prepare_inputs_event is not None:
                    self.prepare_inputs_event.record()
            finally:
                self._async_host_write = None

    def _ensure_host_staging_ready(self):
        # Lazy fence: read-only eligibility and device-only preparation need no
        # host-source protection. Every ordinary writer enters through here.
        if self._async_host_write is False:
            if self.prepare_inputs_event is not None:
                self.prepare_inputs_event.synchronize()
            self._async_host_write = True

    def _dummy_run(self, *args, **kwargs):
        live, self._async_live_execute = self._async_live_execute, False
        self._async_snapshot = self._async_query_layout = None
        try:
            with ExitStack() as input_prep:
                input_prep.enter_context(super().synchronize_input_prep())
                return super()._dummy_run(*args, _input_prep=input_prep, **kwargs)
        finally:
            self._async_live_execute = live
            self._async_snapshot = self._async_query_layout = None

    def _pad_query_start_loc_for_fia(
        self, num_tokens_padded, num_reqs_padded, num_reqs,
        cudagraph_runtime_mode=None, batch_desc_num_reqs=None, full_graph=False,
    ):
        args = (num_tokens_padded, num_reqs_padded, num_reqs, cudagraph_runtime_mode, batch_desc_num_reqs, full_graph)
        signature = (args, self.compilation_config.cudagraph_mode)
        cached = self._async_query_layout
        if self._async_pending is not None and cached is not None and cached[0] == signature:
            # The caller's SP adjustment can still write the host dummy row.
            if num_tokens_padded == 2 * num_reqs and cached[1] > num_reqs_padded:
                self._ensure_host_staging_ready()
            return cached[1]
        self._ensure_host_staging_ready()
        self._async_query_layout = None
        result = super()._pad_query_start_loc_for_fia(*args)
        self._async_query_layout = signature, result
        return result

    def _copy_valid_sampled_token_count(self, next_token_ids, valid_sampled_tokens_count):
        self._async_counts_epoch = -1
        super()._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)
        if self.valid_sampled_token_count_event is not None:
            self._async_counts = valid_sampled_tokens_count
            self._async_counts_epoch = self._async_epoch

    def _eligible(self, scheduled):
        s, batch = self._async_snapshot, self.input_batch
        if (
            s is None
            or self._async_counts_epoch != self._async_epoch
            or self._async_counts is None
            or self._async_counts.dtype != torch.int64
            or tuple(self._async_counts.shape) != (len(s.ids),)
            or scheduled.scheduled_new_reqs
            or scheduled.finished_req_ids
            or scheduled.scheduled_cached_reqs.resumed_req_ids
            or scheduled.new_block_ids_to_zero
            or scheduled.scheduled_encoder_inputs
            or scheduled.free_encoder_mm_hashes
            or self.num_discarded_requests
            or self.use_cp
            or self.uses_mrope
            or self.uses_xdrope_dim > 0
            or self.calculate_kv_scales
            or self.need_accepted_tokens
            or self.lora_config
            # Ascend enables this flag globally; zero shared-prefix blocks
            # cannot use cascade attention or need reconciled CPU positions.
            or (self.cascade_attn_enabled and any(scheduled.num_common_prefix_blocks))
            or self.enable_prompt_embeds
            or self.input_batch.req_prompt_embeds
            or self.is_multimodal_model
            or self.model_config.is_hybrid
            or self.model_config.enable_return_routed_experts
            or self.debugger is not None
            or self.num_prompt_logprobs
            or self.dynamic_eplb
            or _mtp_dw_diag_enabled()
            or npu_content_diagnostics_enabled()
            or not batch.sampling_metadata.no_penalties
            or batch.sampling_metadata.bad_words_token_ids
            or s.metadata[s.target_name].decode_remap_boundary_buffer._values != s.boundary_values
        ):
            return False
        ids = tuple(batch.req_ids)
        sampled, draft = batch.prev_sampled_token_ids, self._draft_token_ids
        if (
            ids != s.ids
            or set(scheduled.num_scheduled_tokens) != set(ids)
            or any(scheduled.num_scheduled_tokens[r] != 2 for r in ids)
            or any(self.requests[r] is not old for r, old in zip(ids, s.requests))
            or batch.prev_req_id_to_index != batch.req_id_to_index
            or any(len(scheduled.scheduled_spec_decode_tokens.get(r, ())) != 1 for r in ids)
            or any(self.requests[r].prev_num_draft_len != 1 for r in ids)
            or any(
                getattr(getattr(request, "sampling_params", None), field, None)
                for request in s.requests
                for field in ("structured_outputs", "logits_processors", "min_tokens")
            )
            or not isinstance(draft, torch.Tensor)
            or tuple(draft.shape) != (len(ids), 1)
            or not isinstance(sampled, torch.Tensor)
            or tuple(sampled.shape) != (len(ids), 1)
            or sampled.stride(0) != 1 or draft.stride(0) != 1
            or sampled.dtype != self.input_ids.gpu.dtype or sampled.dtype != torch.int32
            or draft.dtype not in (torch.int32, torch.int64)
            or sampled.device != draft.device or sampled.device != self.input_ids.gpu.device
        ):
            return False
        cached = scheduled.scheduled_cached_reqs
        if set(cached.req_ids) != set(ids) or any(blocks and any(blocks) for blocks in cached.new_block_ids):
            return False
        optimistic = dict(zip(cached.req_ids, cached.num_computed_tokens))
        if any(
            optimistic[r] != int(c) + 2 or int(c) < int(batch.num_prompt_tokens[i])
            for i, (r, c) in enumerate(zip(ids, s.bases))
        ):
            return False
        high = s.bases + 2
        if np.any(high + 2 > self.max_model_len):
            return False
        for group in batch.block_table.block_tables:
            if (
                group._block_table_dirty
                or not group.kernel_sizes
                or group.block_size != group.kernel_sizes[0]
                or np.any((high + 1) // group.block_size >= group.num_blocks_per_row[: len(ids)])
            ):
                return False
        reason, frontiers, cold = staged_sfa_metadata_sparse_route(scheduled.kv_connector_metadata, ids)
        if (
            reason != StagedSFARouteReason.ELIGIBLE
            or cold
            or frontiers != s.frontiers
            or any(getattr(r, "is_decode_window_save", False) for r in scheduled.kv_connector_metadata.requests)
        ):
            return False
        # Without window saving the remap boundary is independent of acceptance.
        if s.window == 0:
            return frontiers == s.boundary
        # Both possible acceptance counts must leave the existing boundary exact.
        return (
            _boundary(s.bases + 1, frontiers, s.window) == s.boundary
            and _boundary(high, frontiers, s.window) == s.boundary
        )

    def _update_states(self, scheduler_output):
        self._async_pending = scheduler_output if self._eligible(scheduler_output) else None
        if self._async_pending is None:
            self._ensure_host_staging_ready()
            self._async_snapshot = None
            super()._update_states(scheduler_output)

    def _staged_sfa_local_route(self, **kwargs):
        pending, snapshot = self._async_pending, self._async_snapshot
        if pending is not None and kwargs["request_ids"] is not None and tuple(kwargs["request_ids"]) == snapshot.ids:
            # _eligible validated this step; no CPU request mutation has run.
            kwargs["_validated_sparse_route"] = (
                pending.kv_connector_metadata,
                (StagedSFARouteReason.ELIGIBLE, snapshot.frontiers, ()),
            )
        return super()._staged_sfa_local_route(**kwargs)

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        if self._async_pending is None:
            self._ensure_host_staging_ready()
            self._async_query_layout = None
            return super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        n = self.input_batch.num_reqs
        self.attn_state, self.with_prefill = AscendAttentionState.SpecDecoding, False
        # _eligible has already validated request order, tensor layout and Q2 scheduling.
        draft_ids = self.input_ids.gpu.new_empty(n)
        prepare_async_mtp_tokens_kernel[(triton.cdiv(n, 32),)](
            self.input_batch.prev_sampled_token_ids, self._draft_token_ids,
            self.input_ids.gpu, draft_ids, n, BLOCK=32,
        )
        spec = self._fixed_spec_decode_metadata(n, self._fixed_decode_cu_num_tokens.dtype, draft_ids)
        self.logits_indices = spec.logits_indices
        logits_indices = spec.logits_indices
        if lmhead_tp_enable():
            key = (n, logits_indices.dtype, logits_indices.device, self.max_num_reqs * self.uniform_decode_query_len)
            cached = self._async_padded_logits
            if cached is None or cached[0] != key:
                cached = key, torch.nn.functional.pad(logits_indices, (0, key[3] - 2 * n))
                self._async_padded_logits = cached
            logits_indices = cached[1]
        return logits_indices, spec, 2 * n

    def _reconcile(self):
        scheduled, self._async_pending = self._async_pending, None
        super()._update_states(scheduled)
        n = self.input_batch.num_reqs
        self.seq_lens.np[:n] = self.input_batch.num_computed_tokens_cpu[:n] + 2
        self.seq_lens.np[n:].fill(0)
        return scheduled

    def _apply_staged_sfa_route(self, route):
        key = super()._apply_staged_sfa_route(route)
        if self._async_pending is not None and key != self._async_snapshot.key:
            self._ensure_host_staging_ready()
            scheduled = self._reconcile()
            self._async_snapshot = None
            self._prepare_inputs(scheduled, np.full(self.input_batch.num_reqs, 2, dtype=np.int32))
        return key

    def _build_attention_metadata(self, *args, **kwargs):
        s = self._async_snapshot
        if self._async_pending is None:
            result = super()._build_attention_metadata(*args, **kwargs)
        else:
            if args or _shape(kwargs) != s.shape:
                raise RuntimeError("Async SFA metadata layout changed after route validation")
            target = s.metadata[s.target_name]
            item = copy(target)
            metadata = {name: item if value is target else value for name, value in s.metadata.items()}
            common = copy(s.common)
            n, capacity = len(s.ids), s.key.token_capacity
            groups = self.input_batch.block_table.block_tables
            prepare_async_mtp_kernel[(triton.cdiv(max(capacity // 2, item.seq_lens.numel()), 32),)](
                self._async_bases,
                self._async_counts,
                self.positions.gpu,
                common.seq_lens,
                item.seq_lens,
                item.block_table,
                item.indexer_block_table,
                item.slot_mapping,
                item.indexer_slot_mapping,
                n,
                capacity,
                common.seq_lens.numel(),
                item.seq_lens.numel(),
                item.block_table.stride(0),
                item.indexer_block_table.stride(0),
                groups[0].block_size,
                groups[1].block_size,
                BLOCK=32,
            )
            common.seq_lens_cpu = common.num_computed_tokens_cpu = None
            item.seq_lens_cpu = None
            # The boundary is proven unchanged, so no host read/upload is needed.
            item.decode_remap_boundary_ready = True
            item.cos, item.sin = get_cos_and_sin_mla(self.positions.gpu[:capacity].long(), True)
            result = metadata, common
        self._async_built = (result, _shape(kwargs))
        return result

    def _model_forward(self, *args, **kwargs):
        deferred = self._async_pending is not None
        try:
            output = super()._model_forward(*args, **kwargs)
            if self._async_pending is not None:
                s = self._async_snapshot
                self._reconcile()
                metadata, common = self._async_built[0]
                item = metadata[s.target_name]
                # No CPU view below is a DMA source on this device-only path.
                cpu_lengths = s.metadata[s.target_name].seq_lens_cpu
                cpu_lengths[: len(s.ids)].copy_(self.seq_lens.cpu[: len(s.ids)])
                cpu_lengths[len(s.ids) :].zero_()
                item.seq_lens_cpu = cpu_lengths
                common.seq_lens_cpu = self.seq_lens.cpu[: common.num_reqs]
                common.num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[: common.num_reqs]
                common.max_seq_len = int(self.seq_lens.np[: len(s.ids)].max())
                self.async_mtp_replays += 1
                if self.async_mtp_replays == 1:
                    logger.info("[SFA async MTP] first launch-before-reconcile batch: requests=%d", len(s.ids))
            self._remember(seed=not deferred)
            return output
        except BaseException:
            self._async_snapshot = self._async_pending = None
            raise

    def _remember(self, *, seed=True):
        self._async_epoch += 1
        context = get_forward_context()
        self._async_snapshot = None
        if context.staged_sfa_graph_dummy_run or self._async_built is None or context.staged_sfa_graph_key is None:
            return
        (metadata, common), shape = self._async_built
        ids = tuple(self.input_batch.req_ids)
        names = self._staged_sfa_layer_names
        if not metadata or common is None or not ids or not names:
            return
        item = metadata.get(names[0])
        if item is None or any(metadata.get(name) is not item for name in names):
            return
        groups = self.input_batch.block_table.block_tables
        if (
            shape[0] != 2 * len(ids)
            or item.dsa_cp_context is not None
            or len(groups) != 2
            or item.indexer_slot_mapping is None
            or item.decode_remap_boundary_buffer is None
            or common.indexer_slot_mapping is None
            or item.decode_remap_boundary_buffer._values is None
            or item.slot_mapping.data_ptr() != common.slot_mapping.data_ptr()
            or item.indexer_slot_mapping.data_ptr() != common.indexer_slot_mapping.data_ptr()
            or tuple(item.req_ids) != ids
        ):
            return
        bases = self.input_batch.num_computed_tokens_cpu[: len(ids)].copy()
        frontiers = tuple(context.staged_sfa_route.frontiers)
        window = _decode_window_save_window_size()
        if seed:
            # Same-stream read after this target, before sampling/count production.
            # Non-graph and unusable layouts do not copy state they cannot reuse.
            self._async_bases[: len(ids)].copy_(self.positions.gpu[: 2 * len(ids) : 2], non_blocking=True)
        self._async_snapshot = _Snapshot(
            ids,
            tuple(self.requests[r] for r in ids),
            bases,
            dict(metadata),
            copy(common),
            shape,
            context.staged_sfa_graph_key,
            frontiers,
            _boundary(bases, frontiers, window),
            window,
            item.decode_remap_boundary_buffer._values,
            names[0],
        )
