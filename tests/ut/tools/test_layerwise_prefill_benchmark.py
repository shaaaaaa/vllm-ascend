# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
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


def test_run_case_label_does_not_change_requests() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            return 0.1, 0.2, 64, 1, 100, 200

    def run(label: str):
        client = RecordingClient()
        args = Namespace(
            label=label,
            prompt_tokens=64,
            warmups=1,
            repeats=2,
            seed=20260811,
            warmup_seed=10001,
            token_id_min=1000,
            token_id_max=2000,
            cache_chunk_tokens=16,
            chunk_size=16,
            model="model",
            base_url="http://127.0.0.1:9960",
            endpoint="/v1/completions",
        )
        BENCHMARK.run_case(
            args=args,
            client=client,
            multiplier=1,
            seen_first_chunk_digests=set(),
        )
        return [
            (call["request_id"], call["prompt"].token_ids)
            for call in client.calls
        ]

    assert run("off") == run("on")


def test_measured_output_records_request_id_and_wall_clock() -> None:
    class RecordingClient:
        def run(self, **kwargs):
            return 0.1, 0.2, 64, 1, 123, 456

    args = Namespace(
        label="on",
        prompt_tokens=64,
        warmups=0,
        repeats=1,
        seed=20260811,
        warmup_seed=10001,
        token_id_min=1000,
        token_id_max=2000,
        cache_chunk_tokens=16,
        chunk_size=16,
        model="model",
        base_url="http://127.0.0.1:9960",
        endpoint="/v1/completions",
    )

    output = BENCHMARK.run_case(
        args=args,
        client=RecordingClient(),
        multiplier=1,
        seen_first_chunk_digests=set(),
    )

    request = output["requests"][0]
    assert request["request_id"] == "prefill-1x-measure-0"
    assert request["client_start_unix_ns"] == 123
    assert request["first_token_unix_ns"] == 456


def test_summary_reports_average_chunk_and_total_times() -> None:
    results = [
        BENCHMARK.RequestResult(
            request_index=0,
            request_id="prefill-1x-measure-0",
            prompt_digest="a",
            first_chunk_digest="aa",
            ttft_seconds=8.0,
            e2e_seconds=8.1,
            prompt_tokens_reported=16384,
            completion_tokens_reported=1,
        ),
        BENCHMARK.RequestResult(
            request_index=1,
            request_id="prefill-1x-measure-1",
            prompt_digest="b",
            first_chunk_digest="bb",
            ttft_seconds=12.0,
            e2e_seconds=12.1,
            prompt_tokens_reported=16384,
            completion_tokens_reported=1,
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
