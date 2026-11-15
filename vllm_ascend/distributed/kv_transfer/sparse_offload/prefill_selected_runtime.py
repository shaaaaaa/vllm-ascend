# SPDX-License-Identifier: Apache-2.0
"""Fill selected history in native pages before the unchanged whole-batch SFA."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsInt, cast

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prefill_selected_plan import SelectedKVPlanner
from vllm_ascend.distributed.kv_transfer.sparse_offload.prefill_selected_workspace import (
    MAX_HISTORY_TOKENS,
    MIN_HISTORY_TOKENS,
    PACKED_ROW_BYTES,
    PAGE_ROWS,
    TOPK,
    selected_request_metadata_shape,
    selected_union_capacity,
)

if TYPE_CHECKING:
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata
    from vllm_ascend.distributed.kv_transfer.sparse_offload.selected_prefill_test_source import (
        SelectedLoadStep,
    )
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner, PerLayerAttnMetadata


@dataclass(frozen=True, eq=False)
class SelectedKVBatch:
    """Handle for a final worker ordering, not an alternative capacity plan."""

    runtime: SelectedKVRuntime

    def load(
        self,
        layer_name: str,
        topk: torch.Tensor,
        cache: Sequence[torch.Tensor],
        metadata: AscendSFAMetadata,
    ) -> None:
        self.runtime.load(self, layer_name, topk, cache, metadata)

    def prefetch_next(self, layer_name: str) -> None:
        self.runtime.prefetch_next(self, layer_name)


class SelectedKVRuntime:
    def __init__(
        self,
        runner: NPUModelRunner,
        *,
        owner_by_layer: Mapping[str, str],
        main_caches: Mapping[str, torch.Tensor],
        source: object | None,
        query_capacity: int | None = None,
    ) -> None:
        self.device, self.source = runner.device, source
        self.owner_by_layer = dict(owner_by_layer)
        if not self.owner_by_layer or any(owner not in self.owner_by_layer for owner in self.owner_by_layer.values()):
            raise ValueError("Every selected main layer needs an actual TopK owner")
        if self.owner_by_layer.keys() != main_caches.keys():
            raise ValueError("Selected prefetch requires every native main cache")
        self.main_caches = dict(main_caches)
        names = tuple(self.owner_by_layer)
        self._next_follower = {
            name: following
            for name, following in zip(names, names[1:])
            if self.owner_by_layer[following] == self.owner_by_layer[name]
        }
        self._copy_stream = torch.npu.Stream(device=self.device)
        self._prefetched: dict[str, torch.npu.Event | None] = {}
        schedule = runner.vllm_config.scheduler_config
        requested_capacity = int(schedule.max_num_batched_tokens)
        self.query_capacity = (
            requested_capacity
            if query_capacity is None
            else min(requested_capacity, int(query_capacity))
        )
        self.max_requests = min(int(schedule.max_num_seqs), self.query_capacity)
        self.request_metadata_buffer: torch.Tensor | None = torch.empty(
            selected_request_metadata_shape(self.query_capacity, self.max_requests),
            dtype=torch.int32,
            device=self.device,
        )
        self.host_metadata: torch.Tensor | None = torch.empty_like(
            self.request_metadata_buffer, device="cpu", pin_memory=True
        )
        self.source_step: SelectedLoadStep | None = None
        self.batch: SelectedKVBatch | None = None
        self._compute_stream: int | None = None
        self._failed = self._closed = False
        self._metadata: tuple[AscendSFAMetadata, ...] = ()
        self._submitted_layers: set[str] = set()
        self.current_owner: str | None = None
        self.current_plan: tuple[torch.Tensor, torch.Tensor] | None = None
        self.prefetch_enabled = False
        self.histories: list[int] = []
        self.starts: list[int] = []
        self.queries: list[int] = []
        self.request_metadata: torch.Tensor | None = None
        # Open the ACL binary only after all other allocations have succeeded.
        self.selection_planner = SelectedKVPlanner(
            self.device,
            selected_union_capacity(self.query_capacity),
            self.query_capacity,
            self.max_requests,
        )

    def _retire_previous(self) -> None:
        if self.source_step is None:
            return
        if self._failed:
            raise RuntimeError("Failed selected execution retains its workspace and cannot be reused")
        # The source owns event failure handling as well as lease retirement.
        # Do not bypass it: connector.prepare runs before the next runtime bind.
        self.source_step.wait_completion()
        if not self.source_step.try_retire():
            raise RuntimeError("The previous selected source still has live readers")
        self.selection_planner.release_quiescent()
        self._prefetched.clear()
        self.current_owner = self.current_plan = None
        for metadata in self._metadata:
            if metadata.selected_kv is self.batch:
                metadata.selected_kv = None
        self.source_step = self.batch = self._compute_stream = None
        self._metadata = ()

    def bind_step(
        self,
        step: SelectedLoadStep | None,
        request_ids: Sequence[str],
        histories: Iterable[SupportsInt],
        query_starts: Iterable[SupportsInt],
        attention_metadata: PerLayerAttnMetadata | None,
    ) -> SelectedKVBatch | None:
        """Bind once after native connector preparation and final reordering."""
        if self._closed:
            raise RuntimeError("Selected KV runtime is closed")
        self._retire_previous()
        if step is None:
            if isinstance(attention_metadata, dict):
                for metadata in attention_metadata.values():
                    handle = getattr(metadata, "selected_kv", None)
                    if isinstance(handle, SelectedKVBatch) and handle.runtime is self:
                        metadata.selected_kv = None
            return None
        # The caller must mark a post-publication binding failure on this source.
        self.source_step = step
        if not isinstance(attention_metadata, dict):
            raise ValueError("Selected prefill requires one eager, unsliced metadata batch")
        sfa_metadata = cast("dict[str, AscendSFAMetadata]", attention_metadata)
        ids = tuple(request_ids)
        history_tokens, starts = tuple(map(int, histories)), tuple(map(int, query_starts))
        requests = {request.request_id: request for request in step.requests}
        id_set = set(ids)
        if (
            not ids
            or len(ids) != len(id_set)
            or len(requests) != len(step.requests)
            or id_set != requests.keys()
            or len(ids) > self.max_requests
            or len(history_tokens) != len(ids)
            or len(starts) != len(ids) + 1
            or starts[0] != 0
            or not 0 < starts[-1] <= self.query_capacity
        ):
            raise ValueError("Selected connector and final worker request sets differ")
        queries = tuple(end - start for start, end in zip(starts, starts[1:]))
        if any(
            q <= 0 or (requests[name].history_tokens, requests[name].query_tokens) != (h, q)
            for name, h, q in zip(ids, history_tokens, queries)
        ):
            raise ValueError("Selected connector and actual worker H/Q differ")
        selected = [i for i, name in enumerate(ids) if requests[name].selected]
        if not selected or any(not MIN_HISTORY_TOKENS <= history_tokens[i] <= MAX_HISTORY_TOKENS for i in selected):
            raise ValueError("The frozen selected request exceeds the validated execution bounds")
        request_metadata_buffer = cast(torch.Tensor, self.request_metadata_buffer)
        host_metadata = cast(torch.Tensor, self.host_metadata)
        columns = max((h + q + PAGE_ROWS - 1) // PAGE_ROWS for h, q in zip(history_tokens, queries))
        metadatas = tuple({id(sfa_metadata[name]): sfa_metadata[name] for name in self.owner_by_layer}.values())
        first = metadatas[0]
        if any(
            metadata.num_actual_tokens != starts[-1]
            or metadata.block_table.ndim != 2
            or metadata.block_table.dtype != torch.int32
            or metadata.block_table.device != self.device
            or metadata.block_table.shape[0] < len(ids)
            or metadata.block_table.shape[1] < columns
            or metadata.block_table.stride(1) != 1
            or metadata.block_table.stride(0) < metadata.block_table.shape[1]
            for metadata in metadatas
        ):
            raise ValueError("Actual attention metadata differs from selected execution bounds")
        # Target SFA metadata may materialize one block-table tensor per
        # layer even though every main layer carries the same values. The
        # selected copy only needs identical shape and stride.
        if any(
            metadata.block_table.shape != first.block_table.shape
            or metadata.block_table.stride() != first.block_table.stride()
            for metadata in metadatas
        ):
            raise ValueError("Selected prefill requires matching main KV tables")
        self.histories = [history_tokens[i] for i in selected]
        self.starts = [starts[i] for i in selected]
        self.queries = [queries[i] for i in selected]
        host = host_metadata[: len(selected)]
        host_array = host.numpy()
        host_array[:, 0] = selected
        host_array[:, 1] = self.histories
        host_array[:, 2] = self.starts
        host_array[:, 3] = self.queries
        self.request_metadata = request_metadata_buffer[: len(selected)]
        self._compute_stream = torch.npu.current_stream().npu_stream
        self.request_metadata.copy_(host, non_blocking=True)
        self.batch = SelectedKVBatch(self)
        self._metadata = metadatas
        self._submitted_layers = set()
        for metadata in metadatas:
            metadata.selected_kv = self.batch
        return self.batch

    def load(
        self,
        batch: SelectedKVBatch,
        name: str,
        topk: torch.Tensor,
        cache: Sequence[torch.Tensor],
        metadata: AscendSFAMetadata,
    ) -> None:
        if self._closed or self._failed or batch is not self.batch:
            raise RuntimeError("Stale or failed selected attention handle")
        if (
            name not in self.owner_by_layer
            or name in self._submitted_layers
            or not any(metadata is item for item in self._metadata)
        ):
            raise RuntimeError("Selected attention layer is missing, duplicated or stale")
        if (
            topk.dtype != torch.int32
            or topk.shape[1:] != (1, TOPK)
            or not metadata.num_actual_tokens <= topk.shape[0] <= self.query_capacity
            or not topk.is_contiguous()
            or len(cache) != 1
            or cache[0].dtype != torch.int8
            or not cache[0].is_contiguous()
            or cache[0].shape[1:] != (PAGE_ROWS, 1, PACKED_ROW_BYTES)
            or any(tensor.device != self.device for tensor in (topk, cache[0]))
        ):
            raise ValueError("Selected loading requires the unchanged packed C8 geometry")
        stream = torch.npu.current_stream()
        if stream.npu_stream != self._compute_stream:
            raise RuntimeError("Selected execution must stay on its bound compute stream")
        owner = self.owner_by_layer[name]
        if name == owner:
            self.current_plan = self.selection_planner.build(topk, self.histories, self.starts, self.queries)
            self.current_owner = owner
        elif owner != self.current_owner:
            raise RuntimeError("Selected TopK follower ran outside its current owner group")
        # One event publishes every layer's source addresses for this step.
        # Later copies inherit the first wait through the same compute stream.
        source_step = cast("SelectedLoadStep", self.source_step)
        if not self._submitted_layers:
            stream.wait_event(source_step.ready(name))
        if name in self._prefetched:
            completion = self._prefetched[name]
            if completion is None:
                raise RuntimeError("Native slot readiness did not submit the selected prefetch")
            if cache[0].data_ptr() != self.main_caches[name].data_ptr():
                raise ValueError("Selected prefetch destination differs from the actual SFA cache")
            stream.wait_event(completion)
        else:
            self.selection_planner.load(
                *cast("tuple[torch.Tensor, torch.Tensor]", self.current_plan),
                source_step.page_bases(name),
                cast(torch.Tensor, self.request_metadata),
                metadata.block_table,
                cache[0],
            )
            source_step.main_copy_submitted(name)
        self._submitted_layers.add(name)

    def prefetch_next(self, batch: SelectedKVBatch, name: str) -> None:
        """Overlap a TopK follower's history copy with post-SFA computation."""
        if not self.prefetch_enabled:
            return
        if self._closed or self._failed or batch is not self.batch or name not in self._submitted_layers:
            raise RuntimeError("Selected prefetch requires the current layer's submitted attention")
        following = self._next_follower.get(name)
        if following is None:
            return
        if following in self._prefetched or following in self._submitted_layers:
            raise RuntimeError("Selected follower prefetch was already submitted")
        if torch.npu.current_stream().npu_stream != self._compute_stream:
            raise RuntimeError("Selected prefetch must be published on its bound compute stream")
        # Recorded after SFA: its cache reader has finished before slot reuse.
        ready = torch.npu.Event()
        ready.record()
        plan = cast("tuple[torch.Tensor, torch.Tensor]", self.current_plan)
        step = cast("SelectedLoadStep", self.source_step)
        self._prefetched[following] = None

        def submit() -> None:
            # The source invokes this only after the native D2H/reuse fence.
            with torch.npu.stream(self._copy_stream):
                self._copy_stream.wait_event(ready)
                self.selection_planner.load(
                    *plan,
                    step.page_bases(following),
                    cast(torch.Tensor, self.request_metadata),
                    self._metadata[0].block_table,
                    self.main_caches[following],
                )
                completion = torch.npu.Event()
                completion.record()
                step.main_copy_submitted(following, completion=completion)
                self._prefetched[following] = completion

        step.prefetch_main(following, submit)

    def finish_step(self, *, failed: bool = False) -> None:
        if self.source_step is None:
            return
        incomplete = self.batch is None or self._submitted_layers != self.owner_by_layer.keys()
        self._failed = self._failed or failed or incomplete
        if self._failed:
            self.source_step.finish(failed=True)
            if not failed:
                raise RuntimeError("Selected execution did not successfully submit every layer")
            return
        try:
            if torch.npu.current_stream().npu_stream != self._compute_stream:
                raise RuntimeError("Selected completion must stay on its bound compute stream")
            completion = torch.npu.Event()
            completion.record()
            self.source_step.finish(completion=completion)
        except BaseException as error:
            self._failed = True
            try:
                self.source_step.finish(failed=True)
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Selected source failure retention also failed: {cleanup_error!r}")
            raise

    def close(self) -> None:
        if self._closed:
            return
        if self._failed:
            raise RuntimeError("Failed selected execution retains its workspace until process exit")
        self._retire_previous()
        self.selection_planner.close_quiescent()
        self.request_metadata_buffer = self.host_metadata = self.request_metadata = None
        self._closed = True
