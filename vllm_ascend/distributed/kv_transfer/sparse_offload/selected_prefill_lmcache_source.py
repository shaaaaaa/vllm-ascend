# SPDX-License-Identifier: Apache-2.0
"""Read selected history rows from real LMCache host pages into native MLA pages.

LMCache ``ffd1ac9`` stages the exact host objects a layerwise P-node retrieved
for one request inside a :class:`SelectedPrefillStep`.  The target Kaiwu model
keeps the native MLA latent cache (``(k_nope, k_pe)`` pages), so the packed C8
copy kernel cannot be used.  This module provides the minimal read side:

* build one step for an eligible single-request prefill step,
* expose the acquired source through the ``SelectedLoadStep`` protocol used by
  :class:`SelectedKVRuntime`,
* copy only the selected history rows into the existing native pages.

Nothing here resizes or copies the full history: the union of the indexer's
TopK rows (``ids[0, :count]``) is gathered from the LMCache page tensors and
scattered into ``block_table[0, row // block] * block + row % block``.

Scope: this is a *prefill* source.  It covers the final prefill query's TopK
union only, not the TopK rows selected by later decode steps.  On the tested
Kaiwu P-node the no-op connector load leaves non-union destination rows stale
(the final chunk's block mapping is not the earlier chunks' mapping), so a
multi-token generation run diverges after the first decode steps.  Keeping the
computed history resident or adding a decode-row restore is required before
this can be called generation-correct.
"""

from __future__ import annotations

import bisect
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.logger import init_logger

from vllm_ascend.distributed.kv_transfer.sparse_offload.prefill_selected_runtime import (
    SelectedKVRuntime,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.prefill_selected_workspace import (
    MIN_HISTORY_TOKENS,
    PAGE_ROWS,
    TOPK,
)

if TYPE_CHECKING:
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

SELECTED_LMCACHE_ENV = "VLLM_ASCEND_SELECTED_PREFILL_LMCACHE"
logger = init_logger(__name__)


def selected_lmcache_enabled() -> bool:
    """Explicit opt-in for the real-LMCache native selected source."""
    return os.getenv(SELECTED_LMCACHE_ENV, "0") == "1"


def lmcache_selected_supported(connector: object) -> bool:
    """Return whether this connector implements the selected-prefill handoff.

    LMCache's vLLM adapter accepts ``selected_prefill_step`` as a keyword in
    ``LMCacheConnectorV1Impl.start_load_kv``.  The Ascend adapter overrides
    ``start_load_kv`` with ``**kwargs`` and forwards to that base method, so
    walk the MRO instead of inspecting only the most derived signature.
    Older adapters that only drop unknown kwargs are rejected unless they
    expose an explicit support marker.
    """
    impl = getattr(connector, "_lmcache_engine", None)
    start = getattr(impl, "start_load_kv", None)
    if not callable(start):
        return False
    import inspect

    def exposes(method: object) -> bool:
        try:
            return "selected_prefill_step" in inspect.signature(method).parameters
        except (TypeError, ValueError):
            return False

    if exposes(start):
        return True
    for klass in type(impl).__mro__:
        method = klass.__dict__.get("start_load_kv")
        if method is not None and exposes(method):
            return True
    return bool(getattr(connector, "supports_selected_prefill_sources", False))


@dataclass(frozen=True)
class NativeSelectedRequest:
    """Request facts consumed by :meth:`SelectedKVRuntime.bind_step`."""

    request_id: str
    history_tokens: int
    query_tokens: int
    selected: bool = True


@dataclass(frozen=True)
class PendingSelectedPrefill:
    """One step created before the connector runs, acquired after it."""

    step: Any
    request_id: str
    history_tokens: int
    query_tokens: int


def build_selected_prefill_step(
    request_id: str, history_tokens: int, query_tokens: int = 0
) -> PendingSelectedPrefill | None:
    """Create the LMCache step for the whole history interval of one request."""
    if not request_id or history_tokens < MIN_HISTORY_TOKENS:
        return None
    from lmcache.v1.selected_prefill import (
        SelectedPrefillRequest,
        SelectedPrefillStep,
    )

    step = SelectedPrefillStep(
        [SelectedPrefillRequest(request_id, 0, int(history_tokens))]
    )
    return PendingSelectedPrefill(
        step, request_id, int(history_tokens), int(query_tokens)
    )


class NativeSelectedLoadStep:
    """``SelectedLoadStep`` protocol over one acquired ``SelectedPrefillSource``.

    The runtime consumes ``requests``, ``ready``, ``main_copy_submitted``,
    ``prefetch_main``, ``finish``, ``wait_completion`` and ``try_retire``.
    The actual copy is :meth:`load_native`, called from the SFA forward.
    """

    def __init__(
        self,
        step: Any,
        source: Any,
        layer_names: Sequence[str],
        layer_ids: dict[str, int],
        query_tokens: int,
    ) -> None:
        self.step = step
        self.source = source
        self.layer_names = tuple(layer_names)
        self.layer_ids = dict(layer_ids)
        interval = source.request
        history = int(interval.end - interval.start)
        self.requests = (
            NativeSelectedRequest(interval.request_id, history, int(query_tokens)),
        )
        self._ready: torch.npu.Event | None = None
        self._completion: torch.npu.Event | None = None
        self._failed = False
        self._retired = False
        self.last_count = 0
        self.copied_bytes = 0
        self.copy_ms = 0.0
        self.layer_rows: dict[str, int] = {}
        self.layer_bytes: dict[str, int] = {}
        self._slot_formula_checked = False

    # -- SelectedLoadStep protocol -------------------------------------
    def ready(self, layer_name: str) -> torch.npu.Event:
        """Publish the host pages; the source staged them before seal()."""
        if self._ready is None:
            self._ready = torch.npu.Event()
            self._ready.record(torch.npu.current_stream())
        return self._ready

    def main_copy_submitted(
        self, layer_name: str, completion: torch.npu.Event | None = None
    ) -> None:
        if completion is not None:
            self._completion = completion

    def prefetch_main(self, layer_name: str, submit) -> None:
        submit()

    def finish(
        self, completion: torch.npu.Event | None = None, failed: bool = False
    ) -> None:
        """Retire the LMCache lease; a reader event is required when sources exist."""
        self._failed = self._failed or failed
        if completion is not None:
            self._completion = completion
        if self._retired:
            return
        if self._completion is None:
            self._completion = torch.npu.Event()
            self._completion.record(torch.npu.current_stream())
        # release() synchronizes, so every submitted read has finished before
        # LMCache drops its retained object references.
        self.step.release(self._completion)
        self._retired = True

    def wait_completion(self) -> None:
        if self._completion is not None:
            self._completion.synchronize()

    def try_retire(self) -> bool:
        return self._retired

    # -- selected copy -------------------------------------------------
    def load_native(
        self,
        layer_name: str,
        ids: torch.Tensor,
        status: torch.Tensor,
        cache: Sequence[torch.Tensor] | torch.Tensor,
        block_table: torch.Tensor,
        metadata: AscendSFAMetadata | None = None,
    ) -> int:
        """Copy this layer's selected history rows into the native MLA pages."""
        layer_id = self.layer_ids.get(layer_name)
        if layer_id is None:
            raise ValueError(f"Selected layer {layer_name!r} is not a latent layer")
        if (
            ids.ndim != 2
            or status.ndim != 2
            or ids.shape[0] != 1
            or status.shape[0] != 1
        ):
            raise ValueError(
                "Selected plan must hold one request: "
                f"ids={tuple(ids.shape)}, status={tuple(status.shape)}"
            )
        count = int(status[0, 0].item())
        if count < 0 or count > ids.shape[1]:
            raise ValueError(
                f"Selected row count {count} exceeds ids capacity {ids.shape[1]}"
            )
        if count == 0:
            self.last_count = 0
            return 0
        rows = [int(value) for value in ids[0, :count].tolist()]
        if any(left >= right for left, right in zip(rows, rows[1:])):
            raise ValueError("Selected history rows must be strictly increasing")
        layer_source = self.source.layers[layer_id]
        started = time.perf_counter()
        source_rows = gather_layer_rows(
            layer_source,
            layer_id,
            self.source.spans,
            rows,
            native_part_widths(cache),
        )
        block_size = native_block_size(cache)
        table = block_table[0]
        if table.numel() * block_size <= rows[-1]:
            raise ValueError(
                "Selected destination is outside the native block table: "
                f"rows={count}, last={rows[-1]}, blocks={int(table.numel())}, block={block_size}"
            )
        destinations = destination_rows(
            table[: (rows[-1] // block_size) + 1].tolist(),
            block_size,
            rows,
            cache,
        )
        if not self._slot_formula_checked:
            self._check_slot_formula(metadata, table, block_size)
        nbytes = scatter_native_rows(cache, destinations, source_rows)
        self.copy_ms += (time.perf_counter() - started) * 1000.0
        self.last_count = count
        self.copied_bytes += nbytes
        self.layer_rows[layer_name] = count
        self.layer_bytes[layer_name] = nbytes
        return nbytes

    def _check_slot_formula(
        self,
        metadata: AscendSFAMetadata | None,
        table: torch.Tensor,
        block_size: int,
    ) -> None:
        """Prove the selected destination mapping against vLLM's slot_mapping."""
        if metadata is None:
            return
        slots = getattr(metadata, "slot_mapping", None)
        query = int(metadata.num_actual_tokens)
        if not isinstance(slots, torch.Tensor) or query <= 0 or slots.numel() < query:
            return
        history = int(self.requests[0].history_tokens)
        blocks = table[: (history + query + block_size - 1) // block_size].tolist()
        expected = [
            int(blocks[(history + index) // block_size]) * block_size
            + (history + index) % block_size
            for index in range(query)
        ]
        actual = [int(value) for value in slots[:query].tolist()]
        if expected != actual:
            mismatches = [
                (index, want, got)
                for index, (want, got) in enumerate(zip(expected, actual))
                if want != got
            ][:4]
            raise RuntimeError(
                "Selected destination mapping disagrees with vLLM slot_mapping: "
                f"{mismatches}"
            )
        self._slot_formula_checked = True
        print(f"SELECTED_LMCACHE_SLOT_CHECK ok query={query}", flush=True)


def _row_major(tensor: torch.Tensor, tokens: int) -> torch.Tensor:
    """View a page tensor as ``[tokens, width]`` without copying."""
    if tensor.dim() == 1:
        if tokens <= 0 or tensor.numel() % tokens:
            raise ValueError(
                f"Flat selected page of {tensor.numel()} values is not divisible by {tokens}"
            )
        tensor = tensor.view(tokens, -1)
    elif tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.dim() != 2 or tensor.shape[0] < tokens:
        raise ValueError(
            f"Selected page holds {tuple(tensor.shape)}, needs at least {tokens} rows"
        )
    return tensor[:tokens]


def _page_tensor(obj: object, layer_id: int | None) -> torch.Tensor:
    if not getattr(obj, "is_valid", lambda: True)():
        raise RuntimeError("Selected LMCache page is no longer valid")
    if layer_id is not None and hasattr(obj, "layer_tensor"):
        return obj.layer_tensor(layer_id)
    tensor = getattr(obj, "tensor", None)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Selected page {type(obj).__name__} does not expose a tensor")
    return tensor


def layer_objects(layer_source: Any) -> tuple[tuple[object, ...], int | None]:
    """Return the page objects and the container's layer id, if any."""
    pages = getattr(layer_source, "pages", None)
    if pages is None:
        return tuple(layer_source), None
    return tuple(pages) + tuple(getattr(layer_source, "suffix", ())), int(
        layer_source.layer_id
    )


def native_part_widths(
    cache: Sequence[torch.Tensor] | torch.Tensor,
) -> tuple[int, ...]:
    parts = cache if isinstance(cache, (tuple, list)) else (cache,)
    widths = tuple(int(part.shape[-1]) for part in parts)
    if not widths or any(width <= 0 for width in widths):
        raise ValueError(f"Native selected cache widths are invalid: {widths}")
    return widths


def _page_rows(
    tensor: torch.Tensor,
    tokens: int,
    locals_: Sequence[int],
    plane_widths: Sequence[int],
) -> torch.Tensor:
    """Select rows from one host page, supporting plane-stacked LMCache pages.

    LMCache's MLA_LATENT layer page is a flat ``N * (k_nope | k_pe)`` buffer
    whose planes are stacked per chunk (``[N*512 | N*64]``), not interleaved
    per token.  A row-major ``[N, 576]`` page is also accepted.
    """
    offsets = torch.as_tensor(locals_, dtype=torch.long, device=tensor.device)
    total = sum(plane_widths)
    if tensor.dim() == 1:
        if tensor.numel() != tokens * total:
            raise ValueError(
                "Selected flat page does not match its span and plane widths: "
                f"values={tensor.numel()}, tokens={tokens}, planes={tuple(plane_widths)}"
            )
        pieces = []
        base = 0
        for width in plane_widths:
            plane = tensor[base : base + tokens * width].view(tokens, width)
            pieces.append(plane.index_select(0, offsets))
            base += tokens * width
        return torch.cat(pieces, dim=1)
    if (
        tensor.dim() == 3
        and tensor.shape[0] == len(plane_widths)
        and int(tensor.shape[-1]) == plane_widths[0]
        and tensor.shape[1] >= tokens
    ):
        if tuple(int(value) for value in tensor.shape[1:]) != (tokens, plane_widths[0]):
            raise ValueError(
                "Selected plane page does not match its span: "
                f"{tuple(tensor.shape)}, tokens={tokens}, planes={tuple(plane_widths)}"
            )
        return torch.cat(
            [
                tensor[index].index_select(0, offsets)
                for index in range(len(plane_widths))
            ],
            dim=1,
        )
    page = _row_major(tensor, tokens)
    if int(page.shape[1]) != total:
        raise ValueError(
            "Selected row-major page width does not match its native planes: "
            f"page={tuple(page.shape)}, planes={tuple(plane_widths)}"
        )
    return page.index_select(0, offsets)


def gather_layer_rows(
    layer_source: Any,
    layer_id: int,
    spans: Sequence[tuple[int, int]],
    rows: Sequence[int],
    plane_widths: Sequence[int] = (576,),
) -> torch.Tensor:
    """Gather selected global history rows out of one layer's host pages.

    ``spans`` are half-open global token intervals; the combined
    ``pages + suffix`` sequence has exactly one object per span.  Only the
    requested rows are read, and their order is preserved.
    """
    objects, object_layer = layer_objects(layer_source)
    if not objects or len(objects) != len(spans):
        raise ValueError(
            "Selected source layers/spans differ: "
            f"objects={len(objects)}, spans={len(spans)}"
        )
    starts = [int(start) for start, _ in spans]
    if any(left >= right for left, right in zip(starts, starts[1:])):
        raise ValueError("Selected source spans must be strictly increasing")
    page_layer = object_layer if object_layer is not None else layer_id
    # One bucket per touched page object, preserving the requested row order.
    buckets: dict[int, tuple[list[int], list[int]]] = {}
    for out_index, row in enumerate(rows):
        span = bisect.bisect_right(starts, int(row)) - 1
        if span < 0 or not starts[span] <= row < spans[span][1]:
            raise ValueError(
                f"Selected row {row} is outside the staged source spans "
                f"(history={spans[0][0]}..{spans[-1][1]})"
            )
        local = int(row) - int(spans[span][0])
        bucket = buckets.setdefault(span, ([], []))
        bucket[0].append(out_index)
        bucket[1].append(local)
    width = None
    rows_out: torch.Tensor | None = None
    for span, (positions, locals_) in buckets.items():
        span_length = spans[span][1] - spans[span][0]
        page = _page_rows(
            _page_tensor(objects[span], page_layer),
            span_length,
            locals_,
            plane_widths,
        )
        if width is None:
            width = int(page.shape[1])
            rows_out = torch.empty(
                (len(rows), width), dtype=page.dtype, device=page.device
            )
        elif int(page.shape[1]) != width:
            raise ValueError(
                f"Selected source row width changed inside one layer: "
                f"{width} then {int(page.shape[1])}"
            )
        assert rows_out is not None
        targets = torch.as_tensor(positions, dtype=torch.long, device=page.device)
        rows_out[targets] = page
    assert rows_out is not None
    return rows_out


def native_block_size(cache: Sequence[torch.Tensor] | torch.Tensor) -> int:
    first = cache[0] if isinstance(cache, (tuple, list)) else cache
    if not isinstance(first, torch.Tensor) or first.dim() < 2:
        raise ValueError(f"Native selected cache has an unsupported shape: {type(first)}")
    block_size = int(first.shape[1])
    if block_size != PAGE_ROWS:
        raise ValueError(
            f"Native selected copy expects {PAGE_ROWS} rows per block, got {block_size}"
        )
    return block_size


def destination_rows(
    block_ids: Sequence[int],
    block_size: int,
    rows: Sequence[int],
    cache: Sequence[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Map global history rows to flattened native page rows."""
    device = cache[0].device if isinstance(cache, (tuple, list)) else cache.device
    destinations = []
    for row in rows:
        page = int(block_ids[int(row) // block_size])
        if page < 0:
            raise ValueError(f"Selected row {row} maps to an invalid block id {page}")
        destinations.append(page * block_size + int(row) % block_size)
    return torch.tensor(destinations, dtype=torch.long, device=device)


def scatter_native_rows(
    cache: Sequence[torch.Tensor] | torch.Tensor,
    destinations: torch.Tensor,
    source_rows: torch.Tensor,
) -> int:
    """Scatter ``[rows, width]`` host rows into one native MLA layer cache."""
    parts = tuple(cache) if isinstance(cache, (tuple, list)) else (cache,)
    widths = []
    for part in parts:
        if not isinstance(part, torch.Tensor) or part.dim() < 2:
            raise ValueError(f"Native selected cache part is invalid: {type(part)}")
        widths.append(int(part.shape[-1]))
    total = sum(widths)
    if int(source_rows.shape[1]) != total:
        raise ValueError(
            "Selected source row width does not match the native latent cache: "
            f"source={tuple(source_rows.shape)}, parts={[tuple(p.shape) for p in parts]}"
        )
    offset = 0
    for part, width in zip(parts, widths):
        flat = part.reshape(-1, width)
        destination = destinations.to(flat.device)
        rows = source_rows[:, offset : offset + width].to(
            device=flat.device, dtype=flat.dtype, non_blocking=True
        )
        flat.index_copy_(0, destination, rows)
        offset += width
    return int(source_rows.numel() * source_rows.element_size())


class NativeSelectedKVRuntime(SelectedKVRuntime):
    """Selected runtime that fills native MLA pages instead of packed C8 rows."""

    native_latent = True

    def __init__(
        self,
        runner: NPUModelRunner,
        *,
        owner_by_layer: dict[str, str],
        main_caches: dict[str, Any],
        layer_names: Sequence[str],
        query_capacity: int | None = None,
    ) -> None:
        super().__init__(
            runner,
            owner_by_layer=owner_by_layer,
            main_caches=main_caches,
            source=None,
            query_capacity=query_capacity,
        )
        self.layer_names = tuple(layer_names)
        self.layer_ids = {name: index for index, name in enumerate(self.layer_names)}
        if set(self.layer_names) != set(self.owner_by_layer):
            raise ValueError("Selected native layers differ from TopK owners")
        self.forward_ms = 0.0
        self._forward_started = 0.0
        self._bound_adapter: NativeSelectedLoadStep | None = None

    def bind_adapter(
        self,
        adapter: NativeSelectedLoadStep,
        attention_metadata: Any,
    ) -> None:
        """Bind an acquired LMCache source; raises before publication on mismatch."""
        request = adapter.requests[0]
        self.bind_step(
            adapter,
            [request.request_id],
            [request.history_tokens],
            [0, request.query_tokens],
            attention_metadata,
        )
        self._bound_adapter = adapter
        self._forward_started = time.perf_counter()

    def clear_binding(self, attention_metadata: Any) -> None:
        """Drop a previous step's attention handles without touching sources."""
        if self.source_step is not None or self.batch is not None:
            self._retire_previous()
        if isinstance(attention_metadata, dict):
            for metadata in attention_metadata.values():
                handle = getattr(metadata, "selected_kv", None)
                if handle is not None and getattr(handle, "runtime", None) is self:
                    metadata.selected_kv = None

    def load(
        self,
        batch: Any,
        name: str,
        topk: torch.Tensor | None,
        cache: Sequence[torch.Tensor] | torch.Tensor,
        metadata: AscendSFAMetadata,
    ) -> None:
        if self._closed or self._failed or batch is not self.batch:
            raise RuntimeError("Stale or failed selected attention handle")
        if name not in self.owner_by_layer or name in self._submitted_layers:
            raise RuntimeError("Selected attention layer is missing or duplicated")
        if not any(metadata is item for item in self._metadata):
            raise RuntimeError("Selected attention metadata is stale")
        if torch.npu.current_stream().npu_stream != self._compute_stream:
            raise RuntimeError("Selected execution must stay on its bound compute stream")
        owner = self.owner_by_layer[name]
        if name == owner:
            if topk is None:
                raise ValueError("Selected TopK owner ran without indexer output")
            if (
                topk.dtype != torch.int32
                or topk.shape[1:] != (1, TOPK)
                or not metadata.num_actual_tokens <= topk.shape[0] <= self.query_capacity
                or not topk.is_contiguous()
                or topk.device != self.device
            ):
                raise ValueError(
                    "Selected TopK must be contiguous int32 [T,1,2048]: "
                    f"{tuple(topk.shape)} {topk.dtype}"
                )
            self.current_plan = self.selection_planner.build(
                topk, self.histories, self.starts, self.queries
            )
            self.current_owner = owner
        elif owner != self.current_owner:
            raise RuntimeError("Selected TopK follower ran outside its owner group")
        if self.current_plan is None:
            raise RuntimeError("Selected follower ran without an owner plan")
        adapter = self._bound_adapter
        if adapter is None:
            raise RuntimeError("Selected native load has no acquired LMCache source")
        ids, status = self.current_plan
        adapter.load_native(name, ids, status, cache, metadata.block_table, metadata)
        adapter.main_copy_submitted(name)
        self._submitted_layers.add(name)

    def finish_step(self, *, failed: bool = False) -> None:
        super().finish_step(failed=failed)
        if self._forward_started:
            self.forward_ms = (time.perf_counter() - self._forward_started) * 1000.0
        self._bound_adapter = None


def latent_layer_names(kv_cache_config: Any) -> list[str]:
    """Latent (main MLA) layers in connector registration order."""
    names: list[str] = []
    for group in kv_cache_config.kv_cache_groups:
        names.extend(
            name for name in group.layer_names if "indexer" not in name.lower()
        )
    if not names:
        raise ValueError("Selected native source requires latent MLA layers")
    return names


def native_owners(runner: NPUModelRunner, names: Sequence[str]) -> dict[str, str]:
    """Map every latent layer to the last indexer TopK producer before it."""
    context = runner.compilation_config.static_forward_context
    owners: dict[str, str] = {}
    previous: str | None = None
    for name in names:
        impl = context[name].impl
        if getattr(impl, "kv_lora_rank", None) != 512 or getattr(
            impl, "qk_rope_head_dim", None
        ) != 64:
            raise ValueError(
                f"Selected native source requires kv_lora_rank=512 and RoPE=64 at {name}"
            )
        if not impl.skip_topk:
            if getattr(impl, "index_topk", None) != TOPK:
                raise ValueError(
                    f"Selected native source requires {TOPK}-wide TopK at {name}"
                )
            previous = name
        if previous is None:
            raise ValueError(f"Selected native consumer {name} has no earlier TopK owner")
        owners[name] = previous
    return owners


def initialize_selected_lmcache(
    runner: NPUModelRunner, kv_cache_config: Any, kv_caches: dict[str, Any]
) -> None:
    """Create the native selected runtime when the env gate is on."""
    if not selected_lmcache_enabled():
        return
    if runner.selected_kv_runtime is not None:
        raise RuntimeError("Selected runtime must be initialized once")
    connector = get_kv_transfer_group() if has_kv_transfer_group() else None
    ready = connector is not None and lmcache_selected_supported(connector)
    runner._selected_lmcache_connector_ready = ready
    if not ready:
        logger.warning(
            "Selected LMCache source is enabled but the connector does not "
            "accept selected_prefill_step; the native path stays in use."
        )
        return
    names = latent_layer_names(kv_cache_config)
    missing = [name for name in names if name not in kv_caches]
    if missing:
        raise ValueError(f"Selected native layers have no KV cache: {missing}")
    runtime = NativeSelectedKVRuntime(
        runner,
        owner_by_layer=native_owners(runner, names),
        main_caches={name: kv_caches[name] for name in names},
        layer_names=names,
        query_capacity=8192,
    )
    runner.selected_kv_runtime = runtime
    runner._selected_lmcache_runtime = runtime


def runner_selected_prefill(runner: NPUModelRunner) -> PendingSelectedPrefill | None:
    """Eligibility for the single chunked-prefill request of this step."""
    if runner.selected_kv_runtime is None:
        return None
    if not getattr(runner, "_selected_lmcache_connector_ready", False):
        return None
    batch = runner.input_batch
    if batch.num_reqs != 1:
        return None
    history = int(batch.num_computed_tokens_cpu[0])
    if history < MIN_HISTORY_TOKENS:
        return None
    starts = runner.query_start_loc.np[:2]
    query = int(starts[1] - starts[0])
    if query < 2:
        return None
    pending = build_selected_prefill_step(str(batch.req_ids[0]), history, query)
    return pending


def bind_acquired_selected(
    runner: NPUModelRunner, pending: PendingSelectedPrefill, attention_metadata: Any
) -> bool:
    """Acquire the sealed step after start_load_kv and bind it to the runtime.

    An empty source set is a normal miss: the connector already loaded the full
    history, so retire the step and keep the native attention path unchanged.
    """
    runtime = getattr(runner, "_selected_lmcache_runtime", None)
    if runtime is None:
        return False
    step = pending.step
    try:
        sources = step.acquire()
    except Exception as error:
        logger.warning(
            "Selected LMCache step was not sealed by the connector; "
            "keeping the full native load: %r",
            error,
        )
        try:
            step.release()
        except Exception:
            logger.exception("Failed to release an unsealed selected step")
        return False
    if not sources:
        step.release()
        return False
    adapter = NativeSelectedLoadStep(
        step,
        sources[0],
        runtime.layer_names,
        runtime.layer_ids,
        pending.query_tokens,
    )
    if len(sources[0].layers) != len(runtime.layer_names):
        adapter.finish(failed=True)
        raise ValueError(
            "Selected LMCache layer count differs from the native latent group: "
            f"source={len(sources[0].layers)}, native={len(runtime.layer_names)}"
        )
    if not getattr(runner, "_selected_lmcache_layout_logged", False):
        runner._selected_lmcache_layout_logged = True
        layout = {}
        for index, layer in enumerate(sources[0].layers):
            objects, layer_id = layer_objects(layer)
            page_id = layer_id if layer_id is not None else index
            page = objects[0] if objects else None
            try:
                tensor = _page_tensor(page, page_id) if page is not None else None
                page_desc = None if tensor is None else (tuple(tensor.shape), str(tensor.dtype))
            except Exception as error:  # noqa: BLE001 - report the exact mismatch
                page_desc = repr(error)
            layout[runtime.layer_names[index]] = dict(
                objects=len(objects),
                layer_id=layer_id,
                first_page=page_desc,
                cache=[
                    (tuple(part.shape), str(part.dtype))
                    for part in (
                        runtime.main_caches[runtime.layer_names[index]]
                        if isinstance(
                            runtime.main_caches[runtime.layer_names[index]], (tuple, list)
                        )
                        else (runtime.main_caches[runtime.layer_names[index]],)
                    )
                ],
            )
        print(
            "SELECTED_LMCACHE_LAYOUT "
            + repr(dict(spans=list(sources[0].spans), layers=layout)),
            flush=True,
        )
    try:
        runtime.bind_adapter(adapter, attention_metadata)
    except BaseException:
        adapter.finish(failed=True)
        raise
    runner._selected_lmcache_active = adapter
    return True


def clear_selected_binding(runner: NPUModelRunner, attention_metadata: Any) -> None:
    runtime = getattr(runner, "_selected_lmcache_runtime", None)
    if runtime is not None:
        runtime.clear_binding(attention_metadata)


def finish_selected_lmcache(runner: NPUModelRunner, *, failed: bool = False) -> None:
    """Record the completion event, release the LMCache lease, print metrics."""
    runtime = getattr(runner, "_selected_lmcache_runtime", None)
    adapter = getattr(runner, "_selected_lmcache_active", None)
    if runtime is None or adapter is None:
        return
    try:
        runtime.finish_step(failed=failed)
    finally:
        cache = next(iter(runtime.main_caches.values()))
        parts = cache if isinstance(cache, (tuple, list)) else (cache,)
        row_bytes = sum(
            int(part.shape[-1]) * part.element_size() for part in parts
        )
        metric = dict(
            status="failed" if failed else "ok",
            history=adapter.requests[0].history_tokens,
            query=adapter.requests[0].query_tokens,
            count=adapter.last_count,
            spans=list(getattr(adapter.source, "spans", ())),
            layers=dict(adapter.layer_rows),
            layer_bytes=dict(adapter.layer_bytes),
            selected_bytes=adapter.copied_bytes,
            full_bytes=adapter.requests[0].history_tokens
            * row_bytes
            * len(runtime.main_caches),
            copy_ms=round(adapter.copy_ms, 3),
            forward_ms=round(runtime.forward_ms, 3),
        )
        print("SELECTED_LMCACHE_METRIC " + repr(metric), flush=True)
        runner._selected_lmcache_active = None
