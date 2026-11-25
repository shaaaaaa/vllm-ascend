# SPDX-License-Identifier: Apache-2.0
"""Hash-chain report tests without model, PyTorch or SSH dependencies."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tool():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tools"))
    spec = importlib.util.spec_from_file_location("kv_trace_report_test", root / "tools/prefill_kv_trace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_rows():
    base = dict(
        req="cmpl-target-0-suffix",
        rank=0,
        group=0,
        layer=0,
        frontier=4,
        component=0,
        start=0,
        end=4,
        role="P",
        algorithm="sum16x2-v1",
    )
    rows = [
        dict(base, stage="BEGIN", groups=[1]),
        dict(base, stage="CALIBRATION", match=True),
        dict(base, stage="PLAN", phase="P_LOAD", start=0, end=0, components=1),
    ]
    for stage in ("P_COMPUTE", "P_SCATTER"):
        rows.append(dict(base, stage="PLAN", phase=stage, components=1))
    for stage in ("P_COMPUTE", "P_PRODUCED", "P_SCATTER", "P_CPU_STORE", "P_FINAL", "D_CPU", "D_MIRROR"):
        rows.append(dict(base, stage=stage, hash="a" * 32, final=True, role="D" if stage.startswith("D_") else "P"))
    rows.append(dict(base, stage="D_SELECTED", match=True, hash="b" * 32, expected_hash="b" * 32, role="D"))
    rows.append(dict(base, stage="END"))
    rows.extend(dict(base, role="D", stage=stage, groups=[1], match=True) for stage in ("BEGIN", "CALIBRATION", "END"))
    return rows


def run(tool, tmp_path, rows):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return tool.analyze([path], "target", tmp_path)


def test_equal_chain_is_consistent(tool, tmp_path):
    report = run(tool, tmp_path, fixture_rows())
    assert report["status"] == "consistent"
    assert report["different"] == report["unmatched"] == report["errors"] == 0


def test_first_bad_mirror_is_located(tool, tmp_path):
    rows = fixture_rows()
    next(row for row in rows if row["stage"] == "D_MIRROR")["hash"] = "c" * 32
    report = run(tool, tmp_path, rows)
    assert report["status"] == "differences"
    assert report["first_difference"]["stage"] == "D_MIRROR"


def test_missing_layer_cannot_pass(tool, tmp_path):
    rows = fixture_rows()
    rows[0]["groups"] = [2]
    report = run(tool, tmp_path, rows)
    assert report["status"] == "incomplete"
    assert report["errors"] >= 2


def test_no_d_logs_cannot_pass(tool, tmp_path):
    rows = [row for row in fixture_rows() if not row["stage"].startswith("D_")]
    assert run(tool, tmp_path, rows)["status"] == "incomplete"


def test_invalid_algorithm_is_coverage_error(tool, tmp_path):
    rows = fixture_rows()
    next(row for row in rows if row["stage"] == "CALIBRATION")["match"] = False
    report = run(tool, tmp_path, rows)
    assert report["status"] == "invalid_algorithm"
    assert report["first_error"]["error"] == "cpu_npu_checksum_algorithm_disagrees"


def test_packed_device_hash_records_are_expanded(tool, tmp_path):
    rows = fixture_rows()
    row = next(row for row in rows if row["stage"] == "P_COMPUTE" and "hash" in row)
    row["hashes"] = [[row["start"], row["end"], row.pop("hash")]]
    assert run(tool, tmp_path, rows)["status"] == "consistent"


def test_tp_rank_is_read_from_service_prefix(tool, tmp_path):
    path = tmp_path / "service.log"
    row = fixture_rows()[0]
    row["rank"] = 8
    path.write_text("(Worker_DP2_TP0_EP8 pid=12) [KVTRACE] " + json.dumps(row) + "\n")
    assert next(tool.records([path], "target"))["rank"] == 0


def test_decoder_recomputed_boundary_uses_exact_prefill_slice(tool, tmp_path):
    rows = fixture_rows()
    boundary = dict(next(row for row in rows if row["stage"] == "P_FINAL"))
    boundary.update(stage="P_D_BOUNDARY", end=3)
    rows.append(boundary)
    for row in rows:
        if row["stage"] in ("D_CPU", "D_MIRROR"):
            row.update(end=3, frontier=3)
    assert run(tool, tmp_path, rows)["status"] == "consistent"


def test_remote_collect_filters_before_compressing(tool, tmp_path):
    import gzip
    import subprocess

    logs = tmp_path / "vllm-ascend/tp4_dp4_logs"
    logs.mkdir(parents=True)
    row = fixture_rows()[0]
    data = "(Worker_DP2_TP0_EP8 pid=20) [KVTRACE] " + json.dumps(row) + "\n"
    data += "[PFS] p20 q7 req=cmpl-target-0-suffix\n[PFS] p20 q7 g0 flat_put chunks=4\n"
    data += '[KVTRACE] {"req":"other","stage":"P_FINAL"}\n'
    (logs / "P_dp2.log").write_text(data)
    result = subprocess.run(
        [sys.executable, "-c", tool.REMOTE, str(tmp_path), "tp4_dp4_logs", "target"], check=True, capture_output=True
    )
    rows = [json.loads(line) for line in gzip.decompress(result.stdout).splitlines()]
    assert len(rows) == 3
    assert all("target" in row["req"] for row in rows)
    assert rows[0]["tp_rank"] == 0
    assert any("flat_put" in row.get("text", "") for row in rows)


def test_hash_database_keeps_full_counts_without_writing_equal_details(tool, tmp_path):
    report = run(tool, tmp_path, fixture_rows())
    assert report["comparisons"] > 0
    assert (tmp_path / "comparisons.jsonl").read_text() == ""


def test_calibration_failure_takes_priority_over_apparent_corruption(tool, tmp_path):
    rows = fixture_rows()
    next(row for row in rows if row["stage"] == "CALIBRATION")["match"] = False
    next(row for row in rows if row["stage"] == "D_MIRROR")["hash"] = "c" * 32
    report = run(tool, tmp_path, rows)
    assert report["different"] == 1
    assert report["status"] == "invalid_algorithm"


def test_partial_history_uses_prior_computation_not_current_self(tool, tmp_path):
    rows = fixture_rows()
    rows += [
        dict(rows[0], stage="P_COMPUTE", frontier=8, prefix=3, start=0, end=3, hash="c" * 32),
        dict(rows[0], stage="P_LOAD", frontier=8, prefix=3, start=0, end=3, hash="c" * 32),
    ]
    report = run(tool, tmp_path, rows)
    assert report["status"] == "incomplete"
    assert report["first_unmatched"]["stage"] == "P_LOAD"
    # An earlier actual observation of precisely this tail makes it comparable.
    rows.append(dict(rows[0], stage="P_COMPUTE", frontier=3, start=0, end=3, hash="c" * 32))
    assert run(tool, tmp_path, rows)["status"] == "consistent"


def test_absent_optional_sparse_probe_is_not_claimed_verified(tool, tmp_path):
    rows = [row for row in fixture_rows() if row["stage"] != "D_SELECTED"]
    report = run(tool, tmp_path, rows)
    assert report["status"] == "consistent"
    assert report["sparse_transfer_observed"] is False


def test_direct_indexer_import_uses_same_prefill_reference(tool, tmp_path):
    rows = fixture_rows()
    rows.append(dict(rows[0], role="D", stage="D_DIRECT", group=1, hash="c" * 32))
    rows.append(dict(rows[0], stage="P_FINAL", group=1, final=True, hash="c" * 32))
    assert run(tool, tmp_path, rows)["status"] == "consistent"
    rows[-2]["hash"] = "d" * 32
    report = run(tool, tmp_path, rows)
    assert report["first_difference"]["stage"] == "D_DIRECT"


def test_full_decoder_history_is_required_not_just_existing_segments(tool, tmp_path):
    rows = fixture_rows()
    next(row for row in rows if row["stage"] == "BEGIN" and row["role"] == "D")["history_end"] = 8
    report = run(tool, tmp_path, rows)
    assert report["status"] == "incomplete"
    assert report["first_error"]["error"] == "decode_history_prefix_short"


def test_missing_tp_rank_cannot_pass(tool, tmp_path):
    rows = fixture_rows()
    rows[0]["tp_size"] = 4
    report = run(tool, tmp_path, rows)
    assert report["status"] == "incomplete"
    assert report["errors"] == 3
    assert report["first_error"]["error"] == "tp_rank_manifest_absent"


def test_coverage_uses_metadata_frontier_with_extra_mtp_query_row(tool, tmp_path):
    rows = fixture_rows()
    for row in rows:
        if row["stage"] in ("PLAN", "P_COMPUTE", "P_PRODUCED", "P_SCATTER"):
            row.update(frontier=5, metadata_frontier=4)
    assert run(tool, tmp_path, rows)["status"] == "consistent"


def test_shared_cpu_decode_uses_publisher_rank_not_consumer_rank(tool, tmp_path):
    rows = fixture_rows()
    for row in rows:
        if row["role"] == "D":
            row.update(rank=1, source_rank=0)
    rows.append(dict(rows[0], stage="P_FINAL", rank=1, final=True, hash="c" * 32))
    assert run(tool, tmp_path, rows)["status"] == "consistent"


def test_stall_summary_is_informational_and_preserves_waiting_stack(tool, tmp_path):
    rows = fixture_rows()
    rows.append(
        dict(
            rows[0],
            stage="STALL",
            blocked_stage="P_CPU_LOAD",
            elapsed_s=60.0,
            main_stack=["model.py:10:forward"],
            hash_stack=["stream.py:20:synchronize"],
        )
    )
    report = run(tool, tmp_path, rows)
    assert report["status"] == "consistent"
    assert report["first_stall"]["blocked_stage"] == "P_CPU_LOAD"
    assert report["first_stall"]["hash_stack"] == ["stream.py:20:synchronize"]


def test_mixed_algorithms_cannot_report_data_corruption_or_consistency(tool, tmp_path):
    rows = fixture_rows()
    next(row for row in rows if row["stage"] == "D_MIRROR").update(algorithm="mix32x4-v1", hash="c" * 32)
    report = run(tool, tmp_path, rows)
    assert report["status"] == "invalid_algorithm"
    assert report["comparisons"] == 0 and report["different"] == 0
    assert report["first_error"]["error"] == "incompatible_hash_algorithms"


@pytest.mark.parametrize("algorithm", ["sum16x2-v1", "mix32x4-v1"])
def test_report_identifies_the_actual_algorithm_including_old_logs(tool, tmp_path, algorithm):
    rows = fixture_rows()
    for row in rows:
        row["algorithm"] = algorithm
    report = run(tool, tmp_path, rows)
    assert report["status"] == "consistent"
    assert report["algorithm"] == algorithm
