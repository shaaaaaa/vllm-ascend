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
    """Describe nested fixed-address inputs, including keyword arguments."""
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), tuple(value.shape), value.stride(), value.dtype, value.device)
    if isinstance(value, dict):
        return tuple((k, tensor_signature(v)) for k, v in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(tensor_signature(v) for v in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported full SFA graph input: {type(value)}")


@dataclass
class SFAFullGraphEntry:
    graph: Any
    output: Any
    signature: Any


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

    def clear(self) -> None:
        """Discard graphs before profiling's temporary KV storage is released."""
        self.entries.clear()
        self.sealed = False
        self.replay_count = 0
        self.source_bindings.clear()
        self.source_binding_count = 0

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
        """Check replay inputs before workers agree to enter graph collectives."""
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

    def run(self, runnable: Callable[..., Any], *, graph_inputs: Any = None, **kwargs: Any) -> Any:
        """Capture or replay a whole target forward using stable runner inputs."""
        context = get_forward_context()
        key = context.staged_sfa_graph_key
        if context.cudagraph_runtime_mode == CUDAGraphMode.NONE:
            return runnable(**kwargs)
        signature = self.validate_inputs(graph_inputs=graph_inputs, **kwargs)
        entry = self.entries.get(key)
        if entry is None:
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
