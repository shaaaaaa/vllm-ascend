# SPDX-License-Identifier: Apache-2.0
"""Offline PD log summaries retain worker identity and nested failure causes."""

import importlib.util
from pathlib import Path


def load_tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "pd_startup_summary.py"
    spec = importlib.util.spec_from_file_location("pd_startup_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_lm_phase_inside_pd_with_prefixed_and_interleaved_logs():
    tool = load_tool()
    workers = tool.summarize(
        [
            "(Worker_DP2_TP1 pid=7) [PD_INIT] h=node1.lab p=7 d=2 t=1 g=9 kv_init begin ms=0.0",
            "[PD_INIT] h=node2 p=7 d=3 t=0 g=12 warmup begin ms=0.0",
            "[LMCACHE_INIT] host=node1.lab pid=7 rank=1 stage=shared_cache_setup state=begin",
            "[LMCACHE_INIT] host=node1.lab pid=7 rank=1 stage=shared_startup_receive state=begin src=0",
        ]
    )
    assert len(workers) == 2
    first = workers[("node1.lab", 7)]
    assert first["open"][-1] == "LM.shared_startup_receive"
    assert (first["dp"], first["tp"], first["global"]) == ("2", "1", "9")
    assert first["status"] == "waiting"
    assert "waiting=2" in tool.render(workers)[0]


def test_outer_error_does_not_hide_inner_cause_or_copy_exception_message():
    tool = load_tool()
    workers = tool.summarize(
        [
            "[PD_INIT] h=node p=8 d=0 t=0 kv_init begin ms=0.0",
            "[LMCACHE_INIT] host=node pid=8 rank=0 stage=mooncake_setup state=begin",
            "[LMCACHE_INIT] host=node pid=8 rank=0 stage=mooncake_setup state=error error=ValueError:secret",
            "[PD_INIT] h=node p=8 d=0 t=0 kv_init error ms=11.0 exc=RuntimeError",
        ]
    )
    worker = workers[("node", 8)]
    assert worker["error"] == ("LM.mooncake_setup", "ValueError")
    assert worker["status"] == "failed" and worker["open"] == []
    output = "\n".join(tool.render(workers))
    assert "secret" not in output and "err=LM.mooncake_setup:ValueError" in output


def test_successful_warmup_and_finished_subphase_are_distinct():
    tool = load_tool()
    workers = tool.summarize(
        [
            "[PD_INIT] h=node p=8 d=0 t=0 warmup begin ms=0.0",
            "[PD_INIT] h=node p=8 d=0 t=0 ep_barrier begin ms=0.0 n=16",
            "[PD_INIT] h=node p=8 d=0 t=0 ep_barrier end ms=3.0 n=16",
            "[PD_INIT] h=node p=8 d=0 t=0 warmup end ms=4.0",
            "[LMCACHE_INIT] host=node pid=9 rank=1 stage=shared_slab_attach state=end ms=1.0",
        ]
    )
    assert workers[("node", 8)]["status"] == "complete"
    assert workers[("node", 9)]["status"] == "partial"
    output = "\n".join(tool.render(workers))
    assert "complete=1" in output and "p=9 r=1 partial" in output


def test_unrelated_and_garbled_lines_are_ignored():
    tool = load_tool()
    workers = tool.summarize(["proxy password=secret", "[PD_INIT] broken", "[LMCACHE_INIT] stage=x state=error"])
    assert not workers
    assert tool.render(workers) == ["No valid [PD_INIT]/[LMCACHE_INIT] records found."]


def test_same_short_hostname_and_pid_in_distinct_domains_stay_separate():
    tool = load_tool()
    workers = tool.summarize(
        [
            "[PD_INIT] h=node.site1 p=8 d=0 t=0 kv_init begin ms=0.0",
            "[LMCACHE_INIT] host=node.site2 pid=8 rank=0 stage=mooncake_setup state=begin",
        ]
    )
    assert len(workers) == 2


def test_cli_reads_multiple_logs_without_modifying_them(tmp_path, capsys):
    tool = load_tool()
    first, second = tmp_path / "a.log", tmp_path / "b.log"
    first.write_text("[PD_INIT] h=node p=8 d=0 t=0 kv_init begin ms=0.0\n")
    second.write_text("[LMCACHE_INIT] host=node pid=8 rank=0 stage=native_pinned_alloc state=begin bytes=1\n")
    before = (first.read_bytes(), second.read_bytes())
    assert tool.main([str(first), str(second)]) == 0
    assert before == (first.read_bytes(), second.read_bytes())
    assert "open=LM.native_pinned_alloc" in capsys.readouterr().out


def test_large_worker_inventory_is_bounded():
    tool = load_tool()
    workers = tool.summarize(f"[PD_INIT] h=node p={pid} d=0 t=0 kv_init begin ms=0.0" for pid in range(1000))
    output = tool.render(workers)
    assert len(output) < 400
    assert "610 additional workers omitted." in output
