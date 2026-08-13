# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import layerwise_prefill_profile_smoke as smoke


def test_prompts_are_deterministic_and_warmup_has_distinct_cache_chunk():
    measured_a = smoke.make_prompt(smoke.DEFAULT_SEED)
    measured_b = smoke.make_prompt(smoke.DEFAULT_SEED)
    warmup = smoke.make_prompt(smoke.DEFAULT_WARMUP_SEED)

    assert measured_a.digest == measured_b.digest
    assert measured_a.first_chunk_digest == measured_b.first_chunk_digest
    assert measured_a.first_chunk_digest != warmup.first_chunk_digest
    assert len(measured_a.token_ids) == 65_536


def test_requires_worker_only_deferred_profile_configuration(tmp_path: Path):
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "Deferred prefill profiler enabled: last_chunks=4, "
        "chunk_size=2048\n",
        encoding="utf-8",
    )

    smoke.require_worker_only_profiling(server_log, 4)

    with pytest.raises(smoke.SmokeFailure, match="requested final"):
        smoke.require_worker_only_profiling(server_log, 3)


def test_expected_chunk_markers_cover_final_four_2048_chunks():
    assert smoke.expected_chunk_markers(4) == (
        "prefill_profile::chunk_29_of_32::tokens_57344_59392",
        "prefill_profile::chunk_30_of_32::tokens_59392_61440",
        "prefill_profile::chunk_31_of_32::tokens_61440_63488",
        "prefill_profile::chunk_32_of_32::tokens_63488_65536",
    )


def test_trace_requires_every_final_chunk_marker(tmp_path: Path):
    markers = smoke.expected_chunk_markers(4)
    trace = tmp_path / "trace_view.json"
    trace.write_text(" ".join(markers), encoding="utf-8")

    smoke.validate_chunk_markers(
        [trace],
        expected_ranks=1,
        profile_last_chunks=4,
    )

    trace.write_text(" ".join(markers[:-1]), encoding="utf-8")
    with pytest.raises(smoke.SmokeFailure, match="chunk_32_of_32"):
        smoke.validate_chunk_markers(
            [trace],
            expected_ranks=1,
            profile_last_chunks=4,
        )


def test_rejects_frontend_profile_even_with_deferred_workers(tmp_path: Path):
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "Deferred prefill profiler enabled: last_chunks=4, "
        "chunk_size=2048\n"
        f"{smoke.FRONTEND_PROFILER_ENABLED} /tmp/profile\n",
        encoding="utf-8",
    )

    with pytest.raises(smoke.SmokeFailure, match="frontend profiling"):
        smoke.require_worker_only_profiling(server_log, 4)


def _capture_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        label="off",
        base_url="http://127.0.0.1:9960",
        model="glm51-prefill",
        profile_dir=tmp_path,
        expected_ranks=8,
        profile_last_chunks=4,
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
            return smoke.RequestTiming(1.0, 2.0, smoke.PROMPT_TOKENS)

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
        "snapshot",
        "profile",
        "request",
        "profile",
        "close",
        "analyse",
    ]
    assert events[1][1].endswith("-warmup")
    assert events[4][1].endswith("-measure")
    assert events[3] == ("profile", "start")
    assert events[5] == ("profile", "stop")
    assert events[1][2] != events[4][2]


def test_stop_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    actions = []

    class Client:
        def __init__(self, *args):
            pass

        def run(self, **kwargs):
            return smoke.RequestTiming(1.0, 2.0, smoke.PROMPT_TOKENS)

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
