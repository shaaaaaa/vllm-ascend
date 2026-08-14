# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_benchmark_module() -> ModuleType:
    source = Path(__file__).parents[3] / "benchmarks" / "layerwise_prefill_cache" / "benchmark_prefill_p.py"
    spec = importlib.util.spec_from_file_location(
        "layerwise_prefill_benchmark",
        source,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def test_both_modes_use_prompt_length_sweep_and_separate_output_paths() -> None:
    assert BENCHMARK.prompt_multipliers("off", 4) == (1, 2, 4, 8)
    assert BENCHMARK.prompt_multipliers("on", 3) == (1, 2, 4)
    assert BENCHMARK.prompt_multipliers("on", 4) == (1, 2, 4, 8)
    assert BENCHMARK.case_output_path(Path("on.json"), 8) == Path("on-8x.json")


def test_each_prompt_length_uses_a_distinct_cache_prefix() -> None:
    digests = []
    for multiplier in BENCHMARK.prompt_multipliers("on", 3):
        prompt = BENCHMARK.make_prompts(
            seed=BENCHMARK.case_seed(1234, multiplier),
            count=1,
            prompt_tokens=64 * multiplier,
            token_id_min=1000,
            token_id_max=2000,
            cache_chunk_tokens=16,
        )[0]
        digests.append(prompt.first_chunk_digest)
    assert len(digests) == len(set(digests))


def test_off_and_on_use_the_same_prompt_stream() -> None:
    multiplier = 4
    prompt_kwargs = {
        "seed": BENCHMARK.case_seed(1234, multiplier),
        "count": 3,
        "prompt_tokens": 256,
        "token_id_min": 1000,
        "token_id_max": 2000,
        "cache_chunk_tokens": 16,
    }
    off_digests = [
        prompt.digest for prompt in BENCHMARK.make_prompts(**prompt_kwargs)
    ]
    on_digests = [
        prompt.digest for prompt in BENCHMARK.make_prompts(**prompt_kwargs)
    ]
    assert off_digests == on_digests


def test_summary_reports_average_chunk_and_total_times() -> None:
    results = [
        BENCHMARK.RequestResult(
            request_index=0,
            request_id="prefill-off-1x-measure-0",
            prompt_digest="a",
            first_chunk_digest="aa",
            ttft_seconds=8.0,
            e2e_seconds=8.1,
            prompt_tokens_reported=16384,
            completion_tokens_reported=1,
            client_send_unix_ns=1,
            client_response_headers_unix_ns=2,
            client_first_token_unix_ns=3,
            client_done_unix_ns=4,
        ),
        BENCHMARK.RequestResult(
            request_index=1,
            request_id="prefill-off-1x-measure-1",
            prompt_digest="b",
            first_chunk_digest="bb",
            ttft_seconds=12.0,
            e2e_seconds=12.1,
            prompt_tokens_reported=16384,
            completion_tokens_reported=1,
            client_send_unix_ns=5,
            client_response_headers_unix_ns=6,
            client_first_token_unix_ns=7,
            client_done_unix_ns=8,
        ),
    ]

    summary = BENCHMARK.summarize(
        results,
        prompt_tokens=16384,
        chunk_size=4096,
    )

    assert summary["num_chunks"] == 4
    assert summary["average_chunk_ttft"]["mean_ms"] == pytest.approx(2500.0)
    assert summary["ttft"]["mean_ms"] == pytest.approx(10000.0)


@pytest.mark.parametrize(
    ("label", "expected_timeout"),
    (("off", 1200.0), ("on", 1200.0)),
)
def test_mode_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str,
    expected_timeout: float,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_prefill_p.py",
            "--label",
            label,
            "--output",
            str(tmp_path / f"{label}.json"),
        ],
    )

    args = BENCHMARK.parse_args()

    assert args.prompt_tokens == 65536
    assert args.rounds == 3
    assert args.chunk_size == 2048
    assert args.timeout == expected_timeout


@pytest.mark.parametrize("label", ("off", "on"))
def test_custom_rounds_control_cases_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_prefill_p.py",
            "--label",
            label,
            "--rounds",
            "4",
            "--output",
            str(tmp_path / f"{label}.json"),
        ],
    )

    args = BENCHMARK.parse_args()

    assert BENCHMARK.prompt_multipliers(args.label, args.rounds) == (1, 2, 4, 8)
    assert args.timeout == 2400.0
