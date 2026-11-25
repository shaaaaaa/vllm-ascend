# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def tool(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("diagnose_prefill_ranks", directory / "diagnose_prefill_ranks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def device(name, start, duration, connection, task, stream=29):
    return {
        "name": name,
        "ph": "X",
        "ts": start,
        "dur": duration,
        "args": {
            "Physic Stream Id": stream,
            "Task Id": task,
            "Task Type": "PCIE_DMA_SQE" if name == "MEMCPY_ASYNC" else "EVENT_WAIT",
            "connection_id": connection,
        },
    }


def test_dma_gap_has_prior_host_submissions_and_separate_waits(tool, tmp_path):
    events = [
        device("MEMCPY_ASYNC", 10000, 20, 10, 1),
        device("MEMCPY_ASYNC", 200000, 20, 11, 2),
        device("EVENT_WAIT", 20000, 170000, 12, 3, stream=47),
    ]
    events.extend(
        {
            "name": "AscendCL@aclrtMemcpyAsync",
            "ph": "X",
            "ts": -300000 + i,
            "dur": 2,
            "args": {"connection_id": connection, "level": "acl"},
        }
        for i, connection in enumerate((10, 11))
    )
    path = tmp_path / "dp0_rank2_123_20261109140000000_ascend_pt" / "trace_view.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"traceEvents": events}))
    alignment = {"first_start_us": 0, "ranks": [{"start_offset_ms": 201}]}
    result = tool.trace_report(path, alignment)
    gap = result["copy_streams"][0]["largest_gaps"][0]
    assert result["rank"] == 2 and result["profile_pid"] == 123
    assert gap["duration_ms"] == pytest.approx(189.98)
    assert gap["after_host_calls"][0]["start_ms"] < -250
    assert gap["same_stream_control_tasks"] == []
    assert result["longest_device_waits"][0]["duration_ms"] == 170


def test_same_stream_event_wait_is_reported(tool, tmp_path):
    events = [
        device("MEMCPY_ASYNC", 0, 20, 1, 1),
        device("EVENT_WAIT", 20, 20000, 2, 2),
        device("MEMCPY_ASYNC", 20020, 20, 3, 3),
    ]
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps({"traceEvents": events}))
    result = tool.trace_report(path, {"first_start_us": 0, "ranks": [{"start_offset_ms": 100}]})
    gap = result["copy_streams"][0]["largest_gaps"][0]
    assert gap["same_stream_control_tasks"][0]["name"] == "EVENT_WAIT"
    assert result["copy_streams"][0]["copy_active_ms"] == pytest.approx(0.04)


@pytest.mark.parametrize("with_command", [False, True])
def test_process_numa_snapshot_and_missing_process(tool, tmp_path, with_command):
    root = tmp_path / "42"
    (root / "task" / "42").mkdir(parents=True)
    status = (
        "Name:\tVLLM::Worker_DP\nPid:\t42\nPPid:\t40\nNSpid:\t1042\t42\n"
        "Cpus_allowed_list:\t0-3\nMems_allowed_list:\t0,1\n"
    )
    (root / "status").write_text(status)
    (root / "task" / "42" / "status").write_text(status)
    (root / "numa_maps").write_text("1000 bind:1 anon=512 N1=512 kernelpagesize_kB=4\n")
    if with_command:
        (root / "cmdline").write_bytes(b"VLLM::Worker_DP0_TP2_EP2\0")
    result = tool.process_snapshot(42, tmp_path)
    assert result["process"]["NSpid"] == "1042\t42"
    assert result["process"]["PPid"] == "40"
    if with_command:
        assert result["command"] == ["VLLM::Worker_DP0_TP2_EP2"]
    else:
        assert "command" not in result
    assert result["memory_numa_MiB"] == {"1": 2}
    assert result["threads"][0]["Cpus_allowed_list"] == "0-3"
    assert tool.process_snapshot(43, tmp_path)["status"] == "unavailable"


def test_full_existing_profile_workflow(tool, monkeypatch, tmp_path):
    root = tmp_path / "profiles"
    paths = []
    for rank in range(2):
        capture = root / f"dp0_tp{rank}_rank{rank}_{100 + rank}_20261109140000000_ascend_pt"
        output = capture / "ASCEND_PROFILER_OUTPUT"
        output.mkdir(parents=True)
        communication = {
            "step": {
                "collective": {
                    "hcom_allGather__1@group": {
                        "Communication Time Info": {
                            "Start Timestamp(us)": 1000 + rank * 100000,
                            "Elapse Time(ms)": 100 if rank == 0 else 0.01,
                        }
                    }
                }
            }
        }
        (output / "communication.json").write_text(json.dumps(communication))
        (output / "trace_view.json").write_text(
            json.dumps(
                {"traceEvents": [device("MEMCPY_ASYNC", 100, 20, 10, 1), device("MEMCPY_ASYNC", 10000, 20, 11, 2)]}
            )
        )
        paths.append(str(capture))
    manifest = tmp_path / "prefill-profile-test"
    manifest.mkdir()
    (manifest / "result.json").write_text(json.dumps({"run_id": "test-run", "raw_profiles": paths}))
    observations = []

    def machine_snapshot(pids):
        observations.append("machine")
        return {"profile_pids": sorted(pids)}

    original_trace_report = tool.trace_report

    def trace_report(path, alignment):
        assert observations == ["machine"]
        return original_trace_report(path, alignment)

    monkeypatch.setattr(tool, "machine_snapshot", machine_snapshot)
    monkeypatch.setattr(tool, "trace_report", trace_report)
    destination = tmp_path / "diagnosis.json"
    tool.main(
        ["--profile-root", str(root), "--runs-dir", str(tmp_path), "--run", "test-run", "--output", str(destination)]
    )
    result = json.loads(destination.read_text())
    assert {row["rank"] for row in result["ranks"]} == {0, 1}
    assert result["current_machine"]["profile_pids"] == [100, 101]
    assert result["collective_alignment"][0]["ranks"][-1]["rank"] == 1
