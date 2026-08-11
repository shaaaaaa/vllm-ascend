# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Benchmark-only timing for graph-mode chunked prefill.

The profiler times the complete target-model forward for every real prefill
scheduler step with NPU events. This boundary remains valid when the model is
executed by PIECEWISE or FULL ACL graph replay, unlike Python module hooks.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROFILE_DIR_ENV = "VLLM_ASCEND_CHUNKED_PREFILL_GRAPH_PROFILE_DIR"


@dataclass(frozen=True)
class GraphPrefillStep:
    request_ids: tuple[str, ...]
    query_tokens: tuple[int, ...]
    context_tokens_before: tuple[int, ...]
    prompt_tokens: tuple[int, ...]
    num_tokens_padded: int
    cudagraph_mode: str
    graph_capture_count_before: int

    @property
    def is_prefill(self) -> bool:
        return any(context < prompt for context, prompt in zip(self.context_tokens_before, self.prompt_tokens))


class ChunkedPrefillGraphProfiler:
    """Write one graph-compatible model-forward timing record per chunk."""

    def __init__(
        self,
        output_dir: Path,
        *,
        rank: int,
        dp_rank: int,
        num_hidden_layers: int,
        max_num_batched_tokens: int,
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        self.rank = int(rank)
        self.dp_rank = int(dp_rank)
        self.num_hidden_layers = int(num_hidden_layers)
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.hostname = socket.gethostname()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / (
            f"chunked-prefill-graph-{self.hostname}-pid{os.getpid()}-rank{self.rank}.jsonl"
        )
        self._event_factory = event_factory or (lambda: torch.npu.Event(enable_timing=True))
        self._active_step: GraphPrefillStep | None = None
        self._start_event: Any | None = None
        self._step_index = 0

    @classmethod
    def from_env(
        cls,
        *,
        enforce_eager: bool,
        rank: int,
        dp_rank: int,
        pipeline_parallel_size: int,
        num_hidden_layers: int,
        max_num_batched_tokens: int,
    ) -> ChunkedPrefillGraphProfiler | None:
        output_dir = os.getenv(PROFILE_DIR_ENV)
        if output_dir is None or not output_dir.strip():
            return None
        if enforce_eager:
            raise ValueError(f"{PROFILE_DIR_ENV} measures ACL graph execution and cannot be used with --enforce-eager")
        if pipeline_parallel_size != 1:
            raise ValueError(f"{PROFILE_DIR_ENV} currently requires pipeline_parallel_size=1")
        return cls(
            Path(output_dir).expanduser(),
            rank=rank,
            dp_rank=dp_rank,
            num_hidden_layers=num_hidden_layers,
            max_num_batched_tokens=max_num_batched_tokens,
        )

    def start_step(self, step: GraphPrefillStep) -> bool:
        if self._active_step is not None:
            raise RuntimeError("chunked-prefill graph profile step is active")
        lengths = {
            len(step.request_ids),
            len(step.query_tokens),
            len(step.context_tokens_before),
            len(step.prompt_tokens),
        }
        if len(lengths) != 1:
            raise ValueError("chunked-prefill graph metadata lengths differ")
        if not step.is_prefill:
            return False
        if step.cudagraph_mode == "NONE":
            raise RuntimeError(
                "Chunked-prefill graph benchmark reached an eager prefill "
                "step. Configure PIECEWISE/FULL graph capture for the tested "
                "chunk size."
            )
        self._active_step = step
        self._start_event = self._event_factory()
        self._start_event.record()
        return True

    def finish_step(self, graph_capture_count_after: int) -> dict[str, Any] | None:
        step = self._active_step
        start = self._start_event
        if step is None:
            return None
        if start is None:
            raise RuntimeError("missing chunked-prefill graph start event")
        self._active_step = None
        self._start_event = None
        try:
            end = self._event_factory()
            end.record()
            end.synchronize()
            model_forward_npu_ms = float(start.elapsed_time(end))
            record = {
                "schema_version": 1,
                "timestamp_ns": time.time_ns(),
                "step_index": self._step_index,
                "hostname": self.hostname,
                "pid": os.getpid(),
                "rank": self.rank,
                "dp_rank": self.dp_rank,
                "request_ids": list(step.request_ids),
                "query_tokens": list(step.query_tokens),
                "context_tokens_before": list(step.context_tokens_before),
                "prompt_tokens": list(step.prompt_tokens),
                "num_tokens_padded": step.num_tokens_padded,
                "configured_chunk_size": self.max_num_batched_tokens,
                "cudagraph_mode": step.cudagraph_mode,
                "graph_capture_count_delta": (int(graph_capture_count_after) - step.graph_capture_count_before),
                "num_hidden_layers": self.num_hidden_layers,
                "model_forward_npu_ms": model_forward_npu_ms,
            }
            with self.output_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, separators=(",", ":")))
                output.write("\n")
            self._step_index += 1
            return record
        finally:
            self._active_step = None
            self._start_event = None

    def abort_step(self) -> None:
        self._active_step = None
        self._start_event = None


def make_graph_prefill_step(
    *,
    request_ids: Sequence[str],
    query_tokens: Sequence[int],
    context_tokens_before: Sequence[int],
    prompt_tokens: Sequence[int],
    num_tokens_padded: int,
    cudagraph_mode: Any,
    graph_capture_count_before: int,
) -> GraphPrefillStep:
    """Snapshot mutable scheduler metadata immediately before model forward."""
    mode_name = getattr(cudagraph_mode, "name", str(cudagraph_mode))
    return GraphPrefillStep(
        request_ids=tuple(str(value) for value in request_ids),
        query_tokens=tuple(int(value) for value in query_tokens),
        context_tokens_before=tuple(int(value) for value in context_tokens_before),
        prompt_tokens=tuple(int(value) for value in prompt_tokens),
        num_tokens_padded=int(num_tokens_padded),
        cudagraph_mode=str(mode_name),
        graph_capture_count_before=int(graph_capture_count_before),
    )
