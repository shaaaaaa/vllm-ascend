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

    def clear(self) -> None:
        """Discard graphs before profiling's temporary KV storage is released."""
        self.entries.clear()
        self.sealed = False
        self.replay_count = 0

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
