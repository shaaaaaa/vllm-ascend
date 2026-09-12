# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One target-model ACL graph, including device-selected LMCache transfers."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import torch
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform


def tensor_signature(value: Any) -> Any:
    """Describe inputs, inspecting each shared tensor once within THIS call.

    Attention layers often share builder-owned metadata tensors. Never retain
    this memo across forwards: in-place layout/storage changes must be caught.
    """
    tensors: dict[int, tuple[Any, Any]] = {}

    def visit(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            previous = tensors.get(id(item))
            if previous is not None:
                return previous[1]
            signature = (item.data_ptr(), tuple(item.shape), item.stride(), item.dtype, item.device)
            tensors[id(item)] = (item, signature)
            return signature
        if isinstance(item, dict):
            return tuple((k, visit(v)) for k, v in sorted(item.items()))
        if isinstance(item, (tuple, list)):
            return tuple(visit(v) for v in item)
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        raise TypeError(f"Unsupported full SFA graph input: {type(item)}")

    return visit(value)


@dataclass
class SFAFullGraphEntry:
    graph: Any
    output: Any
    signature: Any


@dataclass
class SFAValidatedCall:
    """Single-use, same-context handoff from error agreement to graph launch.

    Own the actual model kwargs rather than accepting a boolean 'skip checks'.
    The runner must not mutate input layouts between preparation and run.
    """

    owner: Any
    generation: int
    context: Any
    key: Hashable
    entry: SFAFullGraphEntry | None
    signature: Any
    kwargs: dict[str, Any]
    consumed: bool = False


@dataclass(frozen=True)
class SFASourceBinding:
    request_ids: tuple[str, ...]
    sources: tuple[Any, ...]

    def matches(self, request_ids: tuple[str, ...], sources: tuple[Any, ...]) -> bool:
        # PreparedSparseSource is an immutable, request-owned snapshot. A store,
        # restore or pointer-table replacement publishes a NEW snapshot. Never
        # compare its dataclass values: tensor equality would run device work.
        # Retaining the objects also prevents Python id/address reuse (ABA).
        return (
            self.request_ids == request_ids
            and len(self.sources) == len(sources)
            and all(old is new for old, new in zip(self.sources, sources))
        )


class SFAFullGraph:
    """Startup-only capture; exactly one replay per authorized target forward.

    FX partitions still exist for compilation, but their ACL wrappers are
    bypassed during root capture. They never execute during root replay.
    """

    def __init__(self) -> None:
        self.entries: dict[Hashable, SFAFullGraphEntry] = {}
        self.sealed = False
        self.replay_count = 0
        self.graph_pool = None
        self.source_bindings: dict[int, SFASourceBinding] = {}
        self.source_binding_count = 0
        self._generation = 0

    def clear(self) -> None:
        """Discard graphs before profiling's temporary KV storage is released."""
        self.entries.clear()
        self.sealed = False
        self.replay_count = 0
        self.source_bindings.clear()
        self.source_binding_count = 0
        self._generation += 1

    def bind_sources(
        self,
        sources: tuple[Any, ...],
        request_ids: tuple[str, ...],
        transfers: Callable[[], tuple[Any, ...]],
    ) -> bool:
        """Rebind layers only when the request-owned source batch changes.

        The warm path examines request snapshots, not layers or device tables.
        Transfer enumeration is deliberately lazy. Tables are shared by graph
        keys with the same request capacity, so cache the LAST binding per
        capacity, not per graph key. The existing post-replay fence remains
        mandatory before replacing/releasing the previous source lease.
        """
        capacity = get_forward_context().staged_sfa_graph_key.request_capacity
        if len(sources) > capacity or len(request_ids) != len(sources):
            raise ValueError("Full SFA source lanes do not match request IDs/capacity")
        previous = self.source_bindings.get(capacity)
        if previous is not None and previous.matches(request_ids, sources):
            return False
        # A failure can leave partially updated tables. Never allow retrying the
        # old batch to hit the old memoized binding after such a partial write.
        self.source_bindings.pop(capacity, None)
        for layer_id, transfer in enumerate(transfers()):
            transfer.bind_batch(sources, layer_id)
        self.source_bindings[capacity] = SFASourceBinding(request_ids, sources)
        self.source_binding_count += 1
        return True

    def seal(self, keys: tuple[Hashable, ...]) -> int:
        """Require precisely one startup graph per authorized shape."""
        if set(self.entries) != set(keys):
            raise RuntimeError(f"Incomplete full SFA capture: expected={keys}, actual={tuple(self.entries)}")
        self.sealed = True
        return len(self.entries)

    def validate_inputs(self, *, graph_inputs: Any = None, **kwargs: Any) -> Any:
        """Check replay inputs before entering captured model collectives."""
        context = get_forward_context()
        key = context.staged_sfa_graph_key
        signature = tensor_signature((kwargs, graph_inputs))
        entry = self.entries.get(key)
        if entry is None:
            if self.sealed or not context.staged_sfa_graph_dummy_run:
                raise RuntimeError(f"Full SFA graph missing at runtime: {key}; live capture is prohibited")
        elif entry.signature != signature:
            raise RuntimeError(f"Full SFA graph inputs changed address or layout: {key}")
        return signature

    def prepare_run(self, *, graph_inputs: Any = None, **kwargs: Any) -> SFAValidatedCall:
        """Validate once, before the runner's existing fail-stop/error agreement."""
        signature = self.validate_inputs(graph_inputs=graph_inputs, **kwargs)
        context = get_forward_context()
        key = context.staged_sfa_graph_key
        return SFAValidatedCall(self, self._generation, context, key, self.entries.get(key), signature, kwargs)

    def run(
        self,
        runnable: Callable[..., Any],
        *,
        prepared: SFAValidatedCall | None = None,
        graph_inputs: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Capture or replay a whole target forward using stable runner inputs."""
        context = get_forward_context()
        key = context.staged_sfa_graph_key
        if context.cudagraph_runtime_mode == CUDAGraphMode.NONE:
            if prepared is not None:
                raise RuntimeError("Cannot use validated graph inputs for eager execution")
            return runnable(**kwargs)
        if prepared is None:
            prepared = self.prepare_run(graph_inputs=graph_inputs, **kwargs)
        elif kwargs or graph_inputs is not None:
            raise RuntimeError("Cannot replace already validated full SFA graph inputs")
        if (
            not isinstance(prepared, SFAValidatedCall)
            or prepared.owner is not self
            or prepared.generation != self._generation
            or prepared.context is not context
            or prepared.key != key
            or prepared.consumed
            or self.entries.get(key) is not prepared.entry
        ):
            raise RuntimeError("Stale, foreign or consumed full SFA graph validation")
        prepared.consumed = True
        signature = prepared.signature
        kwargs = prepared.kwargs
        entry = self.entries.get(key)
        if entry is None:
            if self.sealed or not context.staged_sfa_graph_dummy_run:
                raise RuntimeError("Full SFA capture authorization changed after validation")
            validate_cudagraph_capturing_enabled()
            graph = torch.npu.NPUGraph()
            previous_capturing = context.capturing
            context.sfa_full_graph_active = True
            context.capturing = True
            try:
                pool = self.graph_pool if self.graph_pool is not None else current_platform.get_global_graph_pool()
                with torch.npu.graph(graph, pool=pool):
                    output = runnable(**kwargs)
            finally:
                context.sfa_full_graph_active = False
                context.capturing = previous_capturing
            self.entries[key] = SFAFullGraphEntry(graph, output, signature)
            compilation_counter.num_cudagraph_captured += 1
            return output
        with torch.profiler.record_function("sfa_full_graph::target_replay"):
            entry.graph.replay()
        # One model-boundary fence protects shared CPU source leases and staged
        # output/save consumers. There are no per-layer host waits or callbacks.
        torch.npu.current_stream().synchronize()
        self.replay_count += 1
        return entry.output
