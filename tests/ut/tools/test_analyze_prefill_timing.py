# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import analyze_prefill_timing as analyzer


def test_summarize_breaks_down_critical_rank(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "on-1x.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "request_id": "prefill-1x-measure-0",
                        "ttft_seconds": 0.05,
                        "client_start_unix_ns": 1_000_000_000,
                        "first_token_unix_ns": 1_050_000_000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "\n".join(
            (
                "(Worker_TP0 pid=1) [PREFILL_TIMING] worker_chunk rank=0 "
                "mode=on request=cmpl-prefill-1x-measure-0-0-deadbeef chunk=1/2 "
                "tokens=[0,2048) scheduled=2048 gap_ms=first execute_ms=10.000 "
                "start_unix_ns=1010000000 end_unix_ns=1020000000",
                "(Worker_TP0 pid=1) [PREFILL_TIMING] lmcache_save_fence mode=on "
                "request=cmpl-prefill-1x-measure-0-0-deadbeef prefix_tokens=4096 "
                "save_from=0 load_wait_count=0 load_wait_ms=0.000 "
                "callback_count=80 callback_ms=2.000 "
                "active_storers_before=2 pending_sync_before_finish=4 "
                "pending_sync_after_finish=0 wait_impl_ms=1.000 "
                "finish_batch_ms=3.000 total_ms=4.000",
                "(Worker_TP0 pid=1) [PREFILL_TIMING] lmcache_start_load mode=on "
                "request=cmpl-prefill-1x-measure-0-0-deadbeef prefix_tokens=4096 "
                "can_load=true load_tokens=256 elapsed_ms=0.500",
                "(Worker_TP0 pid=1) [PREFILL_TIMING] worker_chunk rank=0 "
                "mode=on request=cmpl-prefill-1x-measure-0-0-deadbeef chunk=2/2 "
                "tokens=[2048,4096) scheduled=2048 gap_ms=5.000 "
                "execute_ms=20.000 start_unix_ns=1025000000 "
                "end_unix_ns=1045000000",
                "(Worker_TP0 pid=1) [PREFILL_TIMING] worker_sample rank=0 mode=on "
                "request=cmpl-prefill-1x-measure-0-0-deadbeef prompt_tokens=4096 "
                "forward_to_sample_ms=1.000 sample_ms=2.000 "
                "start_unix_ns=1046000000 end_unix_ns=1048000000",
            )
        ),
        encoding="utf-8",
    )

    rows = analyzer.summarize(
        analyzer.load_requests(benchmark_path),
        analyzer.parse_log(log_path),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["server_request"] == "cmpl-prefill-1x-measure-0-0-deadbeef"
    assert row["rank"] == 0
    assert row["ttft_ms"] == pytest.approx(50)
    assert row["client_to_first_chunk_ms"] == pytest.approx(10)
    assert row["worker_execute_ms"] == pytest.approx(30)
    assert row["chunk_gap_ms"] == pytest.approx(5)
    assert row["last_chunk_to_sample_ms"] == pytest.approx(1)
    assert row["sample_ms"] == pytest.approx(2)
    assert row["sample_end_to_first_token_ms"] == pytest.approx(2)
    assert row["reconstructed_ttft_ms"] == pytest.approx(50)
    assert row["timeline_residual_ms"] == pytest.approx(0)
    assert row["lmcache_start_load_ms"] == pytest.approx(0.5)
    assert row["lmcache_max_load_tokens"] == 256
    assert row["lmcache_layer_load_wait_count"] == 0
    assert row["lmcache_layer_load_wait_ms"] == pytest.approx(0)
    assert row["lmcache_callback_count"] == 80
    assert row["lmcache_callback_ms"] == pytest.approx(2)
    assert row["lmcache_max_active_storers"] == 2
    assert row["lmcache_max_pending_sync_before_finish"] == 4
    assert row["lmcache_max_pending_sync_after_finish"] == 0
    assert row["lmcache_wait_impl_ms"] == pytest.approx(1)
    assert row["lmcache_finish_batch_ms"] == pytest.approx(3)
    assert row["lmcache_fence_total_ms"] == pytest.approx(4)


def test_load_requests_rejects_aggregate_output(tmp_path: Path) -> None:
    path = tmp_path / "on.json"
    path.write_text('{"cases": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="aggregate file"):
        analyzer.load_requests(path)
