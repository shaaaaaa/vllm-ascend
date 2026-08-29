from types import SimpleNamespace

from vllm.logger import logger

import vllm_ascend.lmcache_cold_perf as cold_perf


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
