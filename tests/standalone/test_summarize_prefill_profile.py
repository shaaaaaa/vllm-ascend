# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[2] / "tools/summarize_prefill_profile.py"
    spec = importlib.util.spec_from_file_location("summarize_prefill_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("chunk_size", [1, 7, 32, 1024])
@pytest.mark.parametrize("wrapped", [False, True])
def test_streaming_json_at_arbitrary_boundaries(tool, tmp_path, chunk_size, wrapped):
    rows = [{"name": 'value with } ] \\" escaped and 汉字', "args": {"nested": [1, 2]}}, {"dur": 60000}]
    path = tmp_path / "trace_view.json"
    value = {"traceEvents": rows, "metadata": "trailing"} if wrapped else rows
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8-sig")
    assert list(tool.events(path, chunk_size)) == rows


def test_truncated_event_reports_error(tool, tmp_path):
    path = tmp_path / "trace_view.json"
    path.write_text('{"traceEvents":[{"name":"partial', encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        list(tool.events(path, 4))


def test_preserves_device_tasks_and_cpu_ranges_separately(tool, tmp_path):
    rows = [
        {"name": "hcom_allReduce", "ph": "X", "pid": 1, "tid": 3, "ts": 10, "dur": 180000},
        {"name": "hcom_allReduce", "ph": "X", "pid": 2, "tid": 4, "ts": 20, "dur": 90000},
        {"name": "Memcpy H2D", "ph": "X", "pid": 2, "tid": 5, "ts": 5, "dur": 60000},
        {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "CPU"}},
        {"name": "process_name", "ph": "M", "pid": 2, "args": {"name": "Ascend Hardware"}},
        {"name": "thread_name", "ph": "M", "pid": 2, "tid": 5, "args": {"name": "Stream 5"}},
    ]
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps({"traceEvents": rows}))
    result = tool.trace_summary(path, 1)
    assert len(result["allreduce"]) == 2
    assert result["allreduce"][0]["duration_ms"] == 180
    assert result["copy"][0]["duration_ms"] == 60
    assert result["copy"][0]["thread"] == "Stream 5"


def test_hccl_summary_retains_wait_and_transit(tool, tmp_path):
    path = tmp_path / "communication.json"
    data = {
        "step0": {
            "hcom_allReduce__1@group": {
                "Communication Time Info": {"Elapse Time(ms)": 180, "Wait Time(ms)": 170, "Transit Time(ms)": 10},
                "Communication Bandwidth Info": {
                    "SDMA": {"Transit Size(MB)": 2, "Bandwidth(GB/s)": 0.2, "Size Distribution": {"2": [1, 10]}}
                },
            }
        }
    }
    path.write_text(json.dumps(data))
    result = tool.communication_summary(path, 1)["allreduce"][0]
    assert result["time"]["Wait Time(ms)"] == 170
    assert result["time"]["Transit Time(ms)"] == 10
    assert "Size Distribution" not in result["links"]["SDMA"]


def test_separates_gloo_and_preserves_other_collectives(tool, tmp_path):
    path = tmp_path / "trace_view.json"
    names = ["gloo:all_reduce", "hcom_allGather", "hcom_reduceScatter", "hcom_allToAll", "vllm::maybe_pad_and_reduce"]
    rows = [dict(name=name, ph="X", pid=1, tid=2, ts=10, dur=1000) for name in names]
    path.write_text(json.dumps({"traceEvents": rows}))
    result = tool.trace_summary(path, 10)
    assert result["counts"]["gloo"] == 1
    assert result["counts"]["collective"] == 3
    assert result["counts"]["pad_reduce"] == 1
    assert all("gloo" not in row["name"] for row in result["collective"])


def test_nested_begin_end_ranges_preserve_hcom_duration(tool, tmp_path):
    path = tmp_path / "trace_view.json"
    rows = [
        dict(name="vllm::maybe_pad_and_reduce", ph="B", pid=1, tid=2, ts=0),
        dict(name="hcom_allReduce", ph="B", pid=1, tid=2, ts=1000),
        dict(ph="E", pid=1, tid=2, ts=181000),
        dict(ph="E", pid=1, tid=2, ts=182000),
    ]
    path.write_text(json.dumps(rows))
    result = tool.trace_summary(path, 3)
    assert result["collective"][0]["duration_ms"] == 180
    assert result["pad_reduce"][0]["duration_ms"] == 182


def test_communication_summary_includes_reduce_scatter_and_skips_total(tool, tmp_path):
    path = tmp_path / "communication.json"
    data = {
        "step0": {
            "hcom_reduceScatter@group": {"Communication Time Info": {"Elapse Time(ms)": 12, "Wait Time(ms)": 10}},
            "hcom_allGather@group": {"Communication Time Info": {"Elapse Time(ms)": 8}},
            "Total Op Info": {"Communication Time Info": {"Elapse Time(ms)": 20}},
        }
    }
    path.write_text(json.dumps(data))
    result = tool.communication_summary(path, 3)
    assert result["counts"] == {"allreduce": 0, "collective": 2}
    assert result["allreduce"] == []
    assert result["collective"][0]["time"]["Wait Time(ms)"] == 10


def test_alignment_keeps_late_rank_outside_its_local_top_events(tool, tmp_path, capsys, monkeypatch):
    operation = "hcom_allGather__503_47_1@group"
    starts_and_durations = [(0, 249), (17, 232), (244, 5), (17, 232)]
    for rank, (start_ms, duration_ms) in enumerate(starts_and_durations):
        output = tmp_path / f"dp0_rank{rank}_123_20261109145349727_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
        output.mkdir(parents=True)
        data = {
            "step": {
                "collective": {
                    operation: {
                        "Communication Time Info": {
                            "Start Timestamp(us)": 10_000_000 + start_ms * 1000,
                            "Elapse Time(ms)": duration_ms,
                        }
                    },
                    f"hcom_allGather__other_{rank}@group": {
                        "Communication Time Info": {"Start Timestamp(us)": 9_000_000, "Elapse Time(ms)": 100}
                    },
                }
            }
        }
        (output / "communication.json").write_text(json.dumps(data))
        (output / "trace_view.json").write_text("This large trace must stay unread")
    monkeypatch.setattr(tool, "trace_summary", lambda *args: pytest.fail("Alignment read a trace"))
    tool.main([str(tmp_path), "--top", "1"])
    report = json.loads(capsys.readouterr().out)
    aligned = report["collective_alignment"][0]
    assert len(report["collective_alignment"]) == 1
    assert aligned["operation"] == "step/collective/" + operation
    assert aligned["matched_profiles"] == aligned["available_profiles"] == 4
    assert aligned["start_skew_ms"] == 244
    assert aligned["end_skew_ms"] == 0
    assert aligned["tail_after_last_start_ms"] == 5
    late_rank = aligned["ranks"][-1]
    assert late_rank == {"rank": 2, "start_offset_ms": 244, "end_offset_ms": 249, "duration_ms": 5}


def test_alignment_preserves_communicator_and_missing_rank(tool):
    summaries = [
        {
            "file": f"/profiles/rank{rank}_123_20261109145349727_ascend_pt/ASCEND_PROFILER_OUTPUT/communication.json",
            "collective": [
                {
                    "operation": f"step/hcom_allGather__1@group{rank}",
                    "time": {"Start Timestamp(us)": 1000, "Elapse Time(ms)": 10},
                },
                {"operation": "step/no_timestamp", "time": {"Elapse Time(ms)": 100}},
            ],
        }
        for rank in range(2)
    ]
    aligned = tool.collective_alignment(summaries, 3)
    assert len(aligned) == 2
    assert {row["operation"] for row in aligned} == {"step/hcom_allGather__1@group0", "step/hcom_allGather__1@group1"}
    assert all(row["matched_profiles"] == 1 and row["available_profiles"] == 2 for row in aligned)


def test_alignment_requires_communication_data(tool, tmp_path, capsys):
    (tmp_path / "trace_view.json").write_text("{}")
    with pytest.raises(SystemExit):
        tool.main([str(tmp_path / "trace_view.json")])
    assert "Rank alignment requires communication.json" in capsys.readouterr().err


@pytest.mark.parametrize("host_start_ms", [0, 249])
@pytest.mark.parametrize("first_start", [1_000_000, 1_794_207_232_640_317.75])
@pytest.mark.parametrize("host_name", ["AscendCL@hcom_allGather_", "Runtime@rtKernelLaunchWithHandle"])
def test_delay_details_match_connection_id_and_reuse_rank_trace(
    tool, tmp_path, monkeypatch, host_start_ms, first_start, host_name
):
    output = tmp_path / "dp0_rank2_123_20261109145349727_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
    output.mkdir(parents=True)
    communication = output / "communication.json"
    operation = "hcom_allGather__503_47_1"
    alignment = {
        "operation": "step/collective/" + operation + "@group",
        "first_start_us": first_start,
        "ranks": [{"rank": 0, "start_offset_ms": 0}, {"rank": 2, "start_offset_ms": 250}],
    }
    rows = [
        dict(name=operation, ph="X", ts=first_start + 250_000, dur=8, args={"connection_id": 77}),
        dict(name=operation, ph="X", ts=first_start + 500_000, dur=8, args={"connection_id": 99}),
        dict(
            name="AscendCL@hcom_allGather_",
            ph="X",
            ts=first_start + 249_900,
            dur=10,
            args={"connection_id": 88},
        ),
        dict(
            name=host_name,
            ph="B",
            pid=1,
            tid=12,
            ts=first_start + host_start_ms * 1000,
            args={"connection_id": "77"},
        ),
        dict(ph="E", pid=1, tid=12, ts=first_start + host_start_ms * 1000 + 10),
        dict(name="AscendCL@aclrtMemcpyAsync", ph="X", tid=22, ts=first_start + 20_000, dur=195_000),
        dict(name="AscendCL@aclrtMemcpyAsync", ph="X", tid=22, ts=first_start + 260_000, dur=300_000),
        dict(name="MEMCPY_ASYNC", ph="X", ts=first_start + 100_000, dur=100, args={"Physic Stream Id": 24}),
        dict(
            name="EVENT_WAIT",
            ph="X",
            ts=first_start + 14_000,
            dur=235_000,
            args={"Physic Stream Id": 47, "connection_id": 12},
        ),
        dict(
            name="Runtime@rtStreamWaitEvent",
            ph="X",
            ts=first_start + 10_000,
            dur=5,
            args={"level": "runtime", "connection_id": 12, "event_id": 123},
        ),
    ]
    (output / "trace_view.json").write_text(json.dumps({"traceEvents": rows}))
    calls = []
    real_events = tool.complete_events

    def traced_events(path):
        calls.append(path)
        return real_events(path)

    monkeypatch.setattr(tool, "complete_events", traced_events)
    details = tool.collective_delay_details([alignment, alignment], [communication], 3)
    assert len(calls) == 2
    assert len(details) == 2
    detail = details[0]
    assert detail["status"] == "matched"
    assert detail["host_launch"]["start_offset_ms"] == host_start_ms
    assert detail["host_launch"]["connection_id"] == 77
    assert detail["host_launch"]["name"] == host_name
    assert detail["launch_end_to_device_start_ms"] == pytest.approx(250 - host_start_ms - 0.01)
    assert [row["duration_ms"] for row in detail["long_host_copies"]] == [195]
    wait = detail["long_device_tasks"][0]
    assert wait["stream"] == 47
    assert wait["host_api_candidates"][0]["name"] == "Runtime@rtStreamWaitEvent"
    assert wait["host_api_candidates"][0]["args"]["event_id"] == 123


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([], "device_call_missing"),
        ([dict(name="hcom_allGather__1", ph="X", ts=102, dur=8)], "device_timestamp_mismatch"),
        ([dict(name="hcom_allGather__1", ph="X", ts=100, dur=8)], "host_launch_missing"),
    ],
)
def test_delay_details_report_missing_correlations(tool, tmp_path, rows, expected):
    output = tmp_path / "rank2_123_20261109145349727_ascend_pt"
    output.mkdir()
    (output / "trace_view.json").write_text(json.dumps(rows))
    alignment = {
        "operation": "step/hcom_allGather__1@group",
        "first_start_us": 0,
        "ranks": [{"rank": 2, "start_offset_ms": 0.1}],
    }
    detail = tool.collective_delay_details([alignment], [output / "communication.json"], 3)[0]
    assert detail["status"] == expected
    assert detail["host_launch"] is None


def test_locate_cli_uses_selected_capture(tool, tmp_path, monkeypatch, capsys):
    profile_root, _, workers = make_capture(tmp_path, "20261109-140000", "abcd")
    monkeypatch.chdir(tmp_path)
    output = workers[1] / "ASCEND_PROFILER_OUTPUT"
    rows = [
        dict(name="hcom_allGather__1", ph="X", ts=1_001_000, dur=10_000, args={"connection_id": 77}),
        dict(name="AscendCL@hcom_allGather_", ph="X", ts=1_000_000, dur=100, args={"connection_id": 77}),
    ]
    (output / "trace_view.json").write_text(json.dumps(rows))
    tool.main([str(profile_root), "1", "--locate"])
    printed = capsys.readouterr().out
    report = json.loads((tmp_path / "profile-summary.json").read_text())
    detail = report["collective_delay_details"][0]
    assert detail["rank"] == 1
    assert detail["status"] == "matched"
    assert detail["launch_end_to_device_start_ms"] == 0.9
    assert "traces" not in report
    artifact = detail["trace_slice"]
    fragment_path = Path(artifact["file"])
    fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
    assert len(fragment["traceEvents"]) == artifact["events"] == 2
    assert fragment_path.suffix == ".json"
    assert artifact["bytes"] == fragment_path.stat().st_size
    assert printed.splitlines() == [
        f"摘要已保存到 {(tmp_path / 'profile-summary.json').resolve()}",
        f"诊断片段已保存到 {fragment_path}",
    ]


def test_trace_slice_preserves_metadata_spanning_ranges_and_flow_events(tool, tmp_path):
    path = tmp_path / "trace_view.json"
    output = tmp_path / "fragment.json"
    rows = [
        dict(name="process_name", ph="M", pid=1, args={"name": "CANN"}),
        dict(name="outside", ph="X", ts=10, dur=20),
        dict(name="Runtime@record", ph="X", ts=5, dur=2, args={"connection_id": 7}),
        dict(name="spanning", ph="B", pid=1, tid=2, ts=50, args={"connection_id": 7}),
        dict(ph="E", pid=1, tid=2, ts=400),
        dict(name="Runtime@launch", ph="X", ts=200, dur=10, args={"connection_id": 7}),
        dict(name="ac2g", ph="s", ts=201, id=7),
        dict(name="kernel", ph="X", ts=240, dur=100, args={"Physic Stream Id": 47, "connection_id": 7}),
        dict(name="after", ph="X", ts=500, dur=10),
    ]
    path.write_text(json.dumps(rows))
    hosts, artifact = tool.trace_dependencies(path, {"7"}, (output, 100, 300))
    assert [event["name"] for event in hosts["7"]] == ["Runtime@record", "Runtime@launch"]
    fragment = json.loads(output.read_text(encoding="utf-8"))["traceEvents"]
    assert [row["name"] for row in fragment] == [
        "process_name",
        "Runtime@record",
        "spanning",
        "Runtime@launch",
        "ac2g",
        "kernel",
    ]
    assert fragment[2]["dur"] == 350
    assert artifact["events"] == 6


def write_communication(output, rank):
    data = {
        "step": {
            "hcom_allGather__1@group": {
                "Communication Time Info": {"Start Timestamp(us)": 1_000_000 + rank * 1000, "Elapse Time(ms)": 10}
            }
        }
    }
    (output / "communication.json").write_text(json.dumps(data))


def make_capture(root, stamp, run_id, prefix=512000, new=128):
    profile_root = root / "vllm_profile" / "P_dp0"
    workers = []
    for rank in range(2):
        worker = profile_root / f"rank{rank}_{stamp}_{run_id}_ascend_pt"
        output = worker / "ASCEND_PROFILER_OUTPUT"
        output.mkdir(parents=True)
        (output / "trace_view.json").write_text(json.dumps({"traceEvents": []}))
        write_communication(output, rank)
        workers.append(worker)
    result_dir = root / f"prefill-profile-{stamp}-{run_id}"
    result_dir.mkdir()
    data = {
        "run_id": run_id,
        "prefix_tokens": prefix,
        "new_prompt_tokens": new,
        "status": "exported",
        "raw_profiles": list(map(str, workers)),
    }
    (result_dir / "result.json").write_text(json.dumps(data))
    return profile_root, result_dir, workers


def test_lists_chronological_runs_without_parsing_traces(tool, tmp_path, monkeypatch, capsys):
    profile_root, _, _ = make_capture(tmp_path, "20261109-150000", "new", prefix=20480, new=4096)
    make_capture(tmp_path, "20261109-140000", "old")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tool, "trace_summary", lambda *args: pytest.fail("Listing parsed a trace"))
    tool.main([str(profile_root), "--list"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert lines[1].startswith("1\t2026-11-09 14:00:00\t512000\t128\t2/2")
    assert lines[2].startswith("2\t2026-11-09 15:00:00\t20480\t4096\t2/2")


@pytest.mark.parametrize("selector", [["2"], ["--run", "new"], ["--run", "ne"]])
def test_selects_exact_workers_for_one_run(tool, tmp_path, monkeypatch, capsys, selector):
    profile_root, _, old_workers = make_capture(tmp_path, "20261109-140000", "old")
    _, result_dir, new_workers = make_capture(tmp_path, "20261109-150000", "new", new=4096)
    monkeypatch.chdir(tmp_path)
    tool.main([str(profile_root), *selector])
    result = json.loads(capsys.readouterr().out)
    assert result["selection"]["number"] == 2
    assert result["selection"]["manifest"] == str(result_dir / "result.json")
    assert result["selection"]["new_prompt_tokens"] == 4096
    assert result["collective_alignment"][0]["matched_profiles"] == 2
    assert all(str(path) in result["selection"]["profiles"] for path in new_workers)
    assert all(str(path) not in json.dumps(result) for path in old_workers)


def test_selection_by_full_result_directory_name(tool, tmp_path):
    profile_root, result_dir, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    runs = tool.discover_runs(profile_root)
    index, run = tool.select_run(runs, result_dir.name)
    assert index == 1 and run["run_id"] == "abcd"


def test_missing_worker_profile_fails_selection(tool, tmp_path, capsys):
    profile_root, _, workers = make_capture(tmp_path, "20261109-140000", "abcd")
    trace = workers[1] / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    trace.unlink()
    (trace.parent / "communication.json").unlink()
    trace.parent.rmdir()
    workers[1].rmdir()
    with pytest.raises(SystemExit) as error:
        tool.main([str(profile_root), "1"])
    assert error.value.code == 2
    assert "1/2 worker profiles" in capsys.readouterr().err


def test_relocated_raw_paths_match_by_unique_basename(tool, tmp_path):
    profile_root, result_dir, workers = make_capture(tmp_path, "20261109-140000", "abcd")
    path = result_dir / "result.json"
    data = json.loads(path.read_text())
    data["raw_profiles"] = [str(tmp_path / "previous-host" / worker.name) for worker in workers]
    path.write_text(json.dumps(data))
    runs = tool.discover_runs(profile_root)
    assert len(runs) == 1
    assert runs[0]["paths"] == workers


def test_falls_back_to_raw_profile_manifest(tool, tmp_path):
    profile_root, result_dir, workers = make_capture(tmp_path, "20261109-140000", "abcd")
    path = result_dir / "result.json"
    data = json.loads(path.read_text())
    references = data.pop("raw_profiles")
    path.write_text(json.dumps(data))
    (result_dir / "raw_profiles.json").write_text(json.dumps(references))
    assert tool.discover_runs(profile_root)[0]["paths"] == workers


def test_custom_manifest_directory(tool, tmp_path):
    profile_root, result_dir, workers = make_capture(tmp_path, "20261109-140000", "abcd")
    custom_root = tmp_path / "other-client" / "results"
    custom_root.mkdir(parents=True)
    result_dir.rename(custom_root / "custom-output")
    runs = tool.discover_runs(profile_root, custom_root)
    assert len(runs) == 1 and runs[0]["paths"] == workers


def test_unrecorded_capture_is_listed_individually(tool, tmp_path):
    raw = tmp_path / "legacy_ascend_pt"
    raw.mkdir()
    runs = tool.discover_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["paths"] == [raw]
    assert runs[0]["prefix_tokens"] is None
    assert runs[0]["status"] == "raw"


@pytest.mark.parametrize("selector", ["0", "2", "missing"])
def test_invalid_selection_is_reported(tool, tmp_path, selector):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    with pytest.raises(ValueError):
        tool.select_run(tool.discover_runs(profile_root), selector)


def test_ambiguous_id_is_reported(tool, tmp_path):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd1")
    make_capture(tmp_path, "20261109-150000", "abcd2")
    with pytest.raises(ValueError, match="matched 2"):
        tool.select_run(tool.discover_runs(profile_root), "abcd")


def test_legacy_multiple_files_still_work(tool, tmp_path, capsys):
    paths = []
    for rank in range(2):
        directory = tmp_path / f"rank{rank}"
        directory.mkdir()
        path = directory / "trace_view.json"
        path.write_text(json.dumps({"traceEvents": []}))
        paths.append(path)
    tool.main([*map(str, paths), "--full"])
    result = json.loads(capsys.readouterr().out)
    assert len(result["traces"]) == 2
    assert "selection" not in result


def test_incomplete_manifest_leaves_other_captures_available(tool, tmp_path, capsys):
    profile_root, result_dir, _ = make_capture(tmp_path, "20261109-140000", "broken")
    _, _, workers = make_capture(tmp_path, "20261109-150000", "ready")
    (result_dir / "result.json").write_text('{"run_id":')
    runs = tool.discover_runs(profile_root)
    ready = next(run for run in runs if run["run_id"] == "ready")
    assert ready["paths"] == workers
    assert "Skipping unreadable capture record" in capsys.readouterr().err


def test_no_arguments_prompts_for_number_and_saves_summary(tool, tmp_path, monkeypatch, capsys):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "old")
    make_capture(tmp_path, "20261109-150000", "new", new=4096)
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", profile_root)
    monkeypatch.chdir(tmp_path)
    prompts = []

    def choose(prompt):
        prompts.append(prompt)
        return "2"

    monkeypatch.setattr("builtins.input", choose)
    tool.main([])
    assert prompts == ["请输入目录编号（PID_时间戳）："]
    report = json.loads((tmp_path / "profile-summary.json").read_text())
    assert report["selection"]["run_id"] == "new"
    assert report["selection"]["new_prompt_tokens"] == 4096
    output = capsys.readouterr().out
    assert "Profile 根目录" in output and "摘要已保存到" in output


def test_number_only_uses_default_directory_without_prompt(tool, tmp_path, monkeypatch):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", profile_root)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Unexpected prompt"))
    monkeypatch.chdir(tmp_path)
    tool.main(["1"])
    report = json.loads((tmp_path / "profile-summary.json").read_text())
    assert report["selection"]["run_id"] == "abcd"


def test_explicit_directory_interactive_and_output(tool, tmp_path, monkeypatch):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    output = tmp_path / "selected.json"
    monkeypatch.setattr("builtins.input", lambda prompt: "1")
    tool.main([str(profile_root), "--interactive", "--output", str(output)])
    assert json.loads(output.read_text())["selection"]["number"] == 1


def test_default_directory_list_does_not_prompt_or_write_summary(tool, tmp_path, monkeypatch):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", profile_root)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Unexpected prompt"))
    monkeypatch.chdir(tmp_path)
    tool.main(["--list"])
    assert not (tmp_path / "profile-summary.json").exists()


def test_number_is_supplied_once(tool, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        tool.main(["1", "--run", "2"])
    assert error.value.code == 2
    assert "Supply the capture identifier once" in capsys.readouterr().err


def test_interactive_eof_does_not_create_summary(tool, tmp_path, monkeypatch, capsys):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", profile_root)
    monkeypatch.chdir(tmp_path)

    def end_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)
    with pytest.raises(SystemExit) as error:
        tool.main([])
    assert error.value.code == 2
    assert "No profile identifier was entered" in capsys.readouterr().err
    assert not (tmp_path / "profile-summary.json").exists()


def test_interactive_script_entry_point_on_cpu(tool, tmp_path):
    make_capture(tmp_path, "20261109-140000", "abcd")
    script = tmp_path / "tools" / "summarize_prefill_profile.py"
    script.parent.mkdir()
    shutil.copyfile(tool.__file__, script)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script)],
        input="1\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=tmp_path,
        check=True,
        timeout=15,
    )
    assert "请输入目录编号" in result.stdout
    report = json.loads((tmp_path / "profile-summary.json").read_text())
    assert report["selection"]["run_id"] == "abcd"
    assert report["collective_alignment"][0]["matched_profiles"] == 2


def raw_captures(root):
    paths = []
    for pid in (162830, 162950):
        path = root / f"dp0_pp0_tp1_dcp0_ep1_rank1_{pid}_20261109154305160_ascend_pt"
        path.mkdir(parents=True)
        (path / "trace_view.json").write_text(json.dumps({"traceEvents": []}))
        write_communication(path, 1)
        paths.append(path)
    return paths


@pytest.mark.parametrize("arguments", [[], ["--run"]])
def test_raw_identifier_filters_one_directory_without_client_records(tool, tmp_path, monkeypatch, capsys, arguments):
    target, other = raw_captures(tmp_path)
    identifier = "162830_20261109154305160"
    monkeypatch.setattr(tool, "discover_runs", lambda *args: pytest.fail("Raw selection read other run records"))
    tool.main([str(tmp_path), *arguments, identifier])
    result = json.loads(capsys.readouterr().out)
    assert result["selection"]["profile_identifier"] == identifier
    assert result["selection"]["profiles"] == [str(target)]
    assert result["collective_alignment"][0]["matched_profiles"] == 1
    assert str(other) not in json.dumps(result)


def test_identifier_only_uses_default_root(tool, tmp_path, monkeypatch):
    target, _ = raw_captures(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", tmp_path)
    tool.main(["162830_20261109154305160"])
    result = json.loads((tmp_path / "profile-summary.json").read_text())
    assert result["selection"]["profiles"] == [str(target)]


def test_interactive_raw_identifier(tool, tmp_path, monkeypatch):
    target, _ = raw_captures(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: "162830_20261109154305160")
    tool.main([])
    result = json.loads((tmp_path / "profile-summary.json").read_text())
    assert result["selection"]["profiles"] == [str(target)]


def test_duplicate_raw_identifier_requires_unique_directory(tool, tmp_path):
    raw_captures(tmp_path / "copy1")
    raw_captures(tmp_path / "copy2")
    with pytest.raises(ValueError, match="matched 2 directories"):
        tool.select_profile_directory(tmp_path, "162830_20261109154305160")


def test_missing_raw_identifier_reports_zero_matches(tool, tmp_path):
    raw_captures(tmp_path)
    with pytest.raises(ValueError, match="matched 0 directories"):
        tool.select_profile_directory(tmp_path, "999999_20261109154305160")


def test_existing_raw_directory_path_preserves_direct_input(tool, tmp_path, capsys):
    target, _ = raw_captures(tmp_path)
    tool.main([str(target)])
    result = json.loads(capsys.readouterr().out)
    assert result["collective_alignment"][0]["matched_profiles"] == 1
    assert "selection" not in result


def test_raw_identifier_match_is_exact(tool, tmp_path):
    raw_captures(tmp_path)
    with pytest.raises(ValueError, match="matched 0 directories"):
        tool.select_profile_directory(tmp_path, "62830_20261109154305160")


@pytest.mark.parametrize("interactive", [False, True])
def test_user_pid_timestamp_through_real_script_entry_point(tool, tmp_path, interactive):
    target, other = raw_captures(tmp_path / "vllm_profile" / "P_dp0")
    identifier = "162830_20261109154305160"
    script = tmp_path / "tools" / "summarize_prefill_profile.py"
    script.parent.mkdir()
    shutil.copyfile(tool.__file__, script)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script), *([] if interactive else [identifier])],
        input=identifier + "\n" if interactive else "",
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=tmp_path,
        check=True,
        timeout=15,
    )
    report = json.loads((tmp_path / "profile-summary.json").read_text())
    assert report["selection"]["profiles"] == [str(target)]
    assert str(other) not in json.dumps(report)
    assert "摘要已保存到" in result.stdout
