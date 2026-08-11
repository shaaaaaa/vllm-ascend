# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_ascend.worker.chunked_prefill_graph_profiler import (
    PROFILE_DIR_ENV,
    ChunkedPrefillGraphProfiler,
    GraphPrefillStep,
    make_graph_prefill_step,
)


class FakeEvent:
    def __init__(self, timestamp_ms: float) -> None:
        self.timestamp_ms = timestamp_ms
        self.recorded = False
        self.synchronized = False

    def record(self) -> None:
        self.recorded = True

    def synchronize(self) -> None:
        self.synchronized = True

    def elapsed_time(self, end: FakeEvent) -> float:
        assert self.recorded
        assert end.recorded
        return end.timestamp_ms - self.timestamp_ms


def _step(
    *,
    context_tokens: int = 0,
    prompt_tokens: int = 8192,
    mode: str = "PIECEWISE",
) -> GraphPrefillStep:
    return GraphPrefillStep(
        request_ids=("request-1",),
        query_tokens=(4096,),
        context_tokens_before=(context_tokens,),
        prompt_tokens=(prompt_tokens,),
        num_tokens_padded=4096,
        cudagraph_mode=mode,
        graph_capture_count_before=7,
    )


def test_records_graph_prefill_npu_event_time(tmp_path: Path) -> None:
    events = iter([FakeEvent(10.0), FakeEvent(22.5)])
    profiler = ChunkedPrefillGraphProfiler(
        tmp_path,
        rank=3,
        dp_rank=1,
        num_hidden_layers=80,
        max_num_batched_tokens=4096,
        event_factory=lambda: next(events),
    )

    assert profiler.start_step(_step())
    record = profiler.finish_step(graph_capture_count_after=7)

    assert record is not None
    assert record["model_forward_npu_ms"] == pytest.approx(12.5)
    assert record["configured_chunk_size"] == 4096
    assert record["num_hidden_layers"] == 80
    assert record["graph_capture_count_delta"] == 0
    assert record["rank"] == 3
    assert record["dp_rank"] == 1
    disk_record = json.loads(profiler.output_path.read_text(encoding="utf-8"))
    assert disk_record == record


def test_skips_decode_step_without_creating_events(tmp_path: Path) -> None:
    profiler = ChunkedPrefillGraphProfiler(
        tmp_path,
        rank=0,
        dp_rank=0,
        num_hidden_layers=2,
        max_num_batched_tokens=4096,
        event_factory=lambda: pytest.fail("decode must not create an NPU event"),
    )

    assert not profiler.start_step(_step(context_tokens=8192, prompt_tokens=8192))
    assert profiler.finish_step(graph_capture_count_after=0) is None
    assert not profiler.output_path.exists()


def test_rejects_eager_prefill_step(tmp_path: Path) -> None:
    profiler = ChunkedPrefillGraphProfiler(
        tmp_path,
        rank=0,
        dp_rank=0,
        num_hidden_layers=2,
        max_num_batched_tokens=4096,
        event_factory=lambda: pytest.fail("eager must fail before timing"),
    )

    with pytest.raises(RuntimeError, match="eager prefill"):
        profiler.start_step(_step(mode="NONE"))


def test_from_env_is_opt_in_and_rejects_enforce_eager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROFILE_DIR_ENV, raising=False)
    assert (
        ChunkedPrefillGraphProfiler.from_env(
            enforce_eager=False,
            rank=0,
            dp_rank=0,
            pipeline_parallel_size=1,
            num_hidden_layers=2,
            max_num_batched_tokens=4096,
        )
        is None
    )

    monkeypatch.setenv(PROFILE_DIR_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="enforce-eager"):
        ChunkedPrefillGraphProfiler.from_env(
            enforce_eager=True,
            rank=0,
            dp_rank=0,
            pipeline_parallel_size=1,
            num_hidden_layers=2,
            max_num_batched_tokens=4096,
        )

    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        ChunkedPrefillGraphProfiler.from_env(
            enforce_eager=False,
            rank=0,
            dp_rank=0,
            pipeline_parallel_size=2,
            num_hidden_layers=2,
            max_num_batched_tokens=4096,
        )


def test_make_step_snapshots_mutable_metadata() -> None:
    request_ids = ["request-1"]
    query_tokens = [4096]
    contexts = [0]
    prompts = [65536]

    step = make_graph_prefill_step(
        request_ids=request_ids,
        query_tokens=query_tokens,
        context_tokens_before=contexts,
        prompt_tokens=prompts,
        num_tokens_padded=4096,
        cudagraph_mode="PIECEWISE",
        graph_capture_count_before=1,
    )
    contexts[0] = 4096

    assert step.context_tokens_before == (0,)
    assert step.prompt_tokens == (65536,)
