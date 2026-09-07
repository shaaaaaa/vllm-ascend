import json
from types import SimpleNamespace

from vllm.logger import logger

import vllm_ascend.lmcache_cold_perf as cold_perf


def test_unknown_fields_are_not_stringified(monkeypatch):
    class NoReadback:
        def __str__(self):
            raise AssertionError("diagnostics must not stringify device data")

        __repr__ = __str__

    records = []
    monkeypatch.setattr(cold_perf, "_COLD_PERF_ENABLED", True)
    monkeypatch.setattr(
        cold_perf, "logger",
        SimpleNamespace(info=lambda _format, raw: records.append(json.loads(raw))),
    )
    cold_perf.log_cold_perf_process_event("safe", values=[NoReadback()])
    assert records[0]["values"] == ["<non-JSON value>"]


def test_forget_only_removes_one_requests_once_records(monkeypatch):
    class NoScan(dict):
        def __iter__(self):
            raise AssertionError("retirement must not scan all requests")

    monkeypatch.setattr(cold_perf, "_COLD_PERF_ENABLED", True)
    monkeypatch.setattr(cold_perf, "_cold_perf_request_ids", {"one", "two"})
    emitted = NoScan(one={"ready"}, two={"ready"})
    monkeypatch.setattr(cold_perf, "_cold_perf_emitted", emitted)
    cold_perf.forget_cold_perf_request("one")
    assert "one" not in emitted
    assert emitted["two"] == {"ready"}
    assert cold_perf._cold_perf_request_ids == {"two"}


def test_uses_configured_vllm_logger():
    assert cold_perf.logger is logger


def test_marks_only_lmcache_cold_compact_resumes(monkeypatch):
    monkeypatch.setattr(cold_perf, "_COLD_PERF_ENABLED", True)
    cold_perf._cold_perf_request_ids.clear()
    metadata = SimpleNamespace(
        requests=[
            SimpleNamespace(
                req_id="cold",
                load_spec=SimpleNamespace(dsa_cold_compact_resume=True),
            ),
            SimpleNamespace(
                req_id="ordinary",
                load_spec=SimpleNamespace(dsa_cold_compact_resume=False),
            ),
            SimpleNamespace(req_id="save-only", load_spec=None),
        ]
    )

    cold_perf.mark_cold_perf_connector_requests(metadata)

    assert cold_perf.is_cold_perf_request("cold")
    assert not cold_perf.is_cold_perf_request("ordinary")
    assert not cold_perf.is_cold_perf_request("save-only")
    cold_perf._cold_perf_request_ids.clear()


def test_process_event_does_not_require_a_marked_request(monkeypatch):
    records = []
    monkeypatch.setattr(cold_perf, "_COLD_PERF_ENABLED", True)
    monkeypatch.setattr(
        cold_perf,
        "logger",
        SimpleNamespace(
            info=lambda _format, payload: records.append(json.loads(payload))
        ),
    )

    cold_perf.log_cold_perf_process_event("decoder_execute_slow", elapsed_ms=800)

    assert records[0]["event"] == "decoder_execute_slow"
    assert records[0]["elapsed_ms"] == 800


def test_late_request_event_uses_captured_request_id(monkeypatch):
    records = []
    monkeypatch.setattr(cold_perf, "_COLD_PERF_ENABLED", True)
    cold_perf._cold_perf_request_ids.clear()
    monkeypatch.setattr(
        cold_perf,
        "logger",
        SimpleNamespace(
            info=lambda _format, payload: records.append(json.loads(payload))
        ),
    )

    cold_perf.log_cold_perf_event(
        "ordinary_request_event",
        request_ids=("completed-cold-request",),
    )
    assert records == []

    cold_perf.log_cold_perf_event(
        "decoder_async_output_slow",
        request_ids=("completed-cold-request",),
        require_active=False,
        total_ms=1200,
    )

    assert records[0]["event"] == "decoder_async_output_slow"
    assert records[0]["request_ids"] == ["completed-cold-request"]
    assert records[0]["total_ms"] == 1200
