# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _load_benchmark_module() -> ModuleType:
    source = Path(__file__).parents[3] / "benchmarks" / "chunked_prefill_graph" / "benchmark_chunk_sizes.py"
    spec = importlib.util.spec_from_file_location("chunked_prefill_graph_benchmark", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def test_offline_case_constructs_tokens_and_calls_local_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {"generate": []}

    class FakeSamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            calls["sampling_params"] = kwargs

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            calls["llm_kwargs"] = kwargs

        def generate(
            self,
            prompt: dict[str, Any],
            _sampling_params: FakeSamplingParams,
            *,
            use_tqdm: bool,
        ) -> list[SimpleNamespace]:
            calls["generate"].append((prompt, use_tqdm))
            return [SimpleNamespace(request_id=f"request-{len(calls['generate'])}")]

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_inputs = ModuleType("vllm.inputs")
    fake_inputs.TokensPrompt = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", fake_inputs)
    monkeypatch.setenv(BENCHMARK.PROFILE_DIR_ENV, "previous-value")

    BENCHMARK._run_offline_case(
        Namespace(
            profile_dir=str(tmp_path),
            model="model-path",
            additional_config_json='{"enable_npugraph_ex":true}',
            trust_remote_code=True,
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            quantization="ascend",
            gpu_memory_utilization=0.93,
            prompt_tokens=64,
            chunk_size=32,
            seed=1024,
            warmups=1,
            repeats=2,
            token_id_min=1000,
            token_id_max=2000,
        )
    )

    kwargs = calls["llm_kwargs"]
    assert kwargs["data_parallel_size"] == 1
    assert kwargs["max_num_batched_tokens"] == 32
    assert kwargs["enable_chunked_prefill"] is True
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["enforce_eager"] is False
    assert kwargs["compilation_config"] == {
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": [1, 32],
    }
    assert len(calls["generate"]) == 3
    assert all(len(prompt["prompt_token_ids"]) == 64 for prompt, _use_tqdm in calls["generate"])
    assert all(not use_tqdm for _prompt, use_tqdm in calls["generate"])
    assert (tmp_path / "offline_case.json").exists()


def _record(
    *,
    chunk_size: int,
    request_id: str,
    context: int,
    query: int,
    rank: int,
    elapsed_ms: float,
    timestamp_ns: int,
    capture_delta: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_ns": timestamp_ns,
        "rank": rank,
        "dp_rank": 0,
        "request_ids": [request_id],
        "query_tokens": [query],
        "context_tokens_before": [context],
        "prompt_tokens": [8192],
        "configured_chunk_size": chunk_size,
        "cudagraph_mode": "PIECEWISE",
        "graph_capture_count_delta": capture_delta,
        "num_hidden_layers": 2,
        "model_forward_npu_ms": elapsed_ms,
    }


def _request_records(
    *,
    chunk_size: int,
    request_id: str,
    timestamp_ns: int,
    slow_rank_chunk_times: list[float],
    capture_delta: int = 0,
) -> list[dict[str, Any]]:
    records = []
    for ordinal, slow_time in enumerate(slow_rank_chunk_times):
        context = ordinal * chunk_size
        query = min(chunk_size, 8192 - context)
        for rank, elapsed in ((0, slow_time - 2.0), (1, slow_time)):
            records.append(
                _record(
                    chunk_size=chunk_size,
                    request_id=request_id,
                    context=context,
                    query=query,
                    rank=rank,
                    elapsed_ms=elapsed,
                    timestamp_ns=timestamp_ns + ordinal,
                    capture_delta=capture_delta,
                )
            )
    return records


def test_summary_reduces_tp_ranks_and_compares_chunk_sizes() -> None:
    records = []
    records += _request_records(
        chunk_size=4096,
        request_id="4096-warmup",
        timestamp_ns=100,
        slow_rank_chunk_times=[90.0, 110.0],
        capture_delta=1,
    )
    records += _request_records(
        chunk_size=4096,
        request_id="4096-measured-a",
        timestamp_ns=200,
        slow_rank_chunk_times=[10.0, 20.0],
    )
    records += _request_records(
        chunk_size=4096,
        request_id="4096-measured-b",
        timestamp_ns=300,
        slow_rank_chunk_times=[14.0, 24.0],
    )
    records += _request_records(
        chunk_size=8192,
        request_id="8192-warmup",
        timestamp_ns=400,
        slow_rank_chunk_times=[200.0],
        capture_delta=1,
    )
    records += _request_records(
        chunk_size=8192,
        request_id="8192-measured-a",
        timestamp_ns=500,
        slow_rank_chunk_times=[25.0],
    )
    records += _request_records(
        chunk_size=8192,
        request_id="8192-measured-b",
        timestamp_ns=600,
        slow_rank_chunk_times=[29.0],
    )

    summary = BENCHMARK.build_summary(records, prompt_tokens=8192, warmup_requests=1)

    by_chunk = {case["configured_chunk_size"]: case for case in summary["cases"]}
    small = by_chunk[4096]
    assert small["num_chunks"] == 2
    assert small["min_rank_count"] == 2
    assert small["single_chunk_median_ms"] == pytest.approx(17.0)
    assert small["single_chunk_average_layer_median_ms"] == pytest.approx(8.5)
    assert small["total_model_forward_p50_ms"] == pytest.approx(34.0)
    assert small["curve"][0]["model_forward_p50_ms"] == pytest.approx(12.0)
    assert small["curve"][1]["model_forward_p50_ms"] == pytest.approx(22.0)

    large = by_chunk[8192]
    assert large["num_chunks"] == 1
    assert large["single_chunk_median_ms"] == pytest.approx(27.0)
    assert large["total_model_forward_p50_ms"] == pytest.approx(27.0)


def test_summary_rejects_capture_in_measured_request() -> None:
    records = _request_records(
        chunk_size=8192,
        request_id="warmup",
        timestamp_ns=100,
        slow_rank_chunk_times=[20.0],
    )
    records += _request_records(
        chunk_size=8192,
        request_id="measured",
        timestamp_ns=200,
        slow_rank_chunk_times=[25.0],
        capture_delta=1,
    )

    with pytest.raises(RuntimeError, match="capture occurred"):
        BENCHMARK.build_summary(records, prompt_tokens=8192, warmup_requests=1)


def test_prompt_ids_have_requested_64k_length_and_vary_by_request() -> None:
    first = BENCHMARK._prompt_token_ids(65536, 0, token_id_min=1000, token_id_max=100000)
    second = BENCHMARK._prompt_token_ids(65536, 1, token_id_min=1000, token_id_max=100000)

    assert len(first) == 65536
    assert min(first) >= 1000
    assert max(first) < 100000
    assert first != second
