# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import layerwise_prefill_profile_smoke as smoke


def test_cli_defaults_to_one_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "layerwise_prefill_profile_smoke.py",
            "--label",
            "on",
            "--server-log",
            "/tmp/server.log",
            "--profile-dir",
            "/tmp/profile",
        ],
    )

    assert smoke.parse_args().concurrency == 1


def test_prompts_are_deterministic_and_warmup_has_distinct_cache_chunk():
    measured_a = smoke.make_prompt(smoke.DEFAULT_SEED)
    measured_b = smoke.make_prompt(smoke.DEFAULT_SEED)
    warmup = smoke.make_prompt(smoke.DEFAULT_WARMUP_SEED)

    assert measured_a.digest == measured_b.digest
    assert measured_a.first_chunk_digest == measured_b.first_chunk_digest
    assert measured_a.first_chunk_digest != warmup.first_chunk_digest
    assert len(measured_a.token_ids) == 65_536


def test_requires_worker_only_full_profile_configuration(tmp_path: Path):
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "Deferred full prefill profiler enabled: chunk_size=2048\n",
        encoding="utf-8",
    )

    smoke.require_worker_only_profiling(server_log)

    server_log.write_text("unconfigured\n", encoding="utf-8")
    with pytest.raises(smoke.SmokeFailure, match="all prefill chunks"):
        smoke.require_worker_only_profiling(server_log)


def test_expected_ranks_follow_active_dp_replicas(tmp_path: Path):
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "TP=4 DP=2 DP_LOCAL=2 DP_BACKEND=mp\n",
        encoding="utf-8",
    )

    assert smoke.resolve_expected_ranks(
        server_log,
        concurrency=1,
        configured=None,
    ) == 4
    assert smoke.resolve_expected_ranks(
        server_log,
        concurrency=2,
        configured=None,
    ) == 8
    assert smoke.resolve_expected_ranks(
        server_log,
        concurrency=1,
        configured=7,
    ) == 7


def test_expected_ranks_are_inferred_from_dp_worker_log_names(tmp_path: Path):
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "\n".join(
            f"(Worker_DP{dp}_TP{tp}_EP0 pid=1) ready"
            for dp in range(2)
            for tp in range(4)
        ),
        encoding="utf-8",
    )

    assert smoke.resolve_expected_ranks(
        server_log,
        concurrency=1,
        configured=None,
    ) == 4


def test_expected_chunk_markers_cover_all_32_chunks():
    markers = smoke.expected_chunk_markers()
    assert len(markers) == 32
    assert markers[0] == (
        "prefill_profile::all::chunk_1_of_32::tokens_0_2048"
    )
    assert markers[-1] == (
        "prefill_profile::all::chunk_32_of_32::tokens_63488_65536"
    )


def test_expected_chunk_markers_include_partial_final_chunk():
    markers = smoke.expected_chunk_markers(14_000)
    assert len(markers) == 7
    assert markers[-2] == (
        "prefill_profile::all::chunk_6_of_7::tokens_10240_12288"
    )
    assert markers[-1] == (
        "prefill_profile::all::chunk_7_of_7::tokens_12288_14000"
    )


def test_trace_requires_every_final_chunk_marker(tmp_path: Path):
    markers = smoke.expected_chunk_markers()
    trace = tmp_path / "trace_view.json"
    trace.write_text(" ".join(markers), encoding="utf-8")

    smoke.validate_chunk_markers(
        [trace],
        expected_ranks=1,
    )

    trace.write_text(" ".join(markers[:-1]), encoding="utf-8")
    with pytest.raises(smoke.SmokeFailure, match="chunk_32_of_32"):
        smoke.validate_chunk_markers(
            [trace],
            expected_ranks=1,
        )


def test_rejects_frontend_profile_even_with_deferred_workers(tmp_path: Path):
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "Deferred full prefill profiler enabled: chunk_size=2048\n"
        f"{smoke.FRONTEND_PROFILER_ENABLED} /tmp/profile\n",
        encoding="utf-8",
    )

    with pytest.raises(smoke.SmokeFailure, match="frontend profiling"):
        smoke.require_worker_only_profiling(server_log)


def _capture_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        label="off",
        base_url="http://127.0.0.1:9960",
        model="glm51-prefill",
        prompt_tokens=14_000,
        concurrency=1,
        profile_dir=tmp_path,
        expected_ranks=8,
        seed=smoke.DEFAULT_SEED,
        warmup_seed=smoke.DEFAULT_WARMUP_SEED,
        cache_chunk_tokens=256,
        request_timeout=10,
        profile_control_timeout=10,
        profile_analysis_timeout=10,
        trace_timeout=10,
    )


def test_capture_profiles_only_measured_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    events = []
    trace = tmp_path / "trace_view.json"

    class Client:
        def __init__(self, base_url, timeout):
            events.append(("client", base_url, timeout))

        def run(self, *, model, prompt, request_id):
            events.append(("request", request_id, prompt.digest))
            return smoke.RequestTiming(1.0, 2.0, 14_000)

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(smoke, "CompletionClient", Client)
    monkeypatch.setattr(
        smoke,
        "profile_control",
        lambda base_url, action, timeout: events.append(
            ("profile", action)
        ),
    )
    monkeypatch.setattr(
        smoke,
        "trace_snapshot",
        lambda profile_dir: events.append(("snapshot",)) or {},
    )
    monkeypatch.setattr(
        smoke,
        "analyse_profile_data",
        lambda *args, **kwargs: events.append(("analyse",)),
    )
    monkeypatch.setattr(
        smoke,
        "wait_for_new_traces",
        lambda *args, **kwargs: [trace],
    )
    monkeypatch.setattr(smoke, "validate_chunk_markers", lambda *a, **k: None)

    assert smoke.run_capture(_capture_args(tmp_path)) == [trace]
    event_names = [event[0] for event in events]
    assert event_names == [
        "client",
        "request",
        "close",
        "snapshot",
        "profile",
        "client",
        "request",
        "close",
        "profile",
        "analyse",
    ]
    assert events[1][1].endswith("-warmup-0")
    assert events[6][1].endswith("-measure-0")
    assert events[4] == ("profile", "start")
    assert events[8] == ("profile", "stop")
    assert events[1][2] != events[6][2]


def test_stop_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    actions = []

    class Client:
        def __init__(self, *args):
            pass

        def run(self, **kwargs):
            return smoke.RequestTiming(1.0, 2.0, 14_000)

        def close(self):
            pass

    def profile_control(base_url, action, timeout):
        actions.append(action)
        if action == "stop":
            raise smoke.SmokeFailure("response lost")

    monkeypatch.setattr(smoke, "CompletionClient", Client)
    monkeypatch.setattr(smoke, "profile_control", profile_control)
    monkeypatch.setattr(smoke, "trace_snapshot", lambda profile_dir: {})

    with pytest.raises(smoke.SmokeFailure, match="response lost"):
        smoke.run_capture(_capture_args(tmp_path))

    assert actions == ["start", "stop"]


def test_offline_analysis_uses_torch_npu_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls = []
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    smoke.analyse_profile_data(tmp_path, expected_ranks=8, timeout=17)

    command, kwargs = calls[0]
    assert "torch_npu.profiler.profiler import analyse" in command[2]
    assert command[3:] == [str(tmp_path), "8"]
    assert kwargs == {"check": True, "timeout": 17}
