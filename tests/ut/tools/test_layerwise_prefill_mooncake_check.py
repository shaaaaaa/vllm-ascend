# SPDX-License-Identifier: Apache-2.0
"""No NPU required: isolation, reload proof and sequential-process contracts."""

import ast
import copy
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional

import pytest


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    return importlib.import_module("layerwise_prefill_mooncake_check")


def options(tool):
    args = tool.parser().parse_args([])
    # The launcher resolves its private master's port before constructing envs.
    args.master = "127.0.0.1:45678"
    args.local_hostname = "7.150.7.133"
    return args


def copy_evidence(tool, root):
    for stage in ("prefill", "holder0", "holder1"):
        (root / stage).mkdir(exist_ok=True)
    (root / "prefill/objects.jsonl").write_text('{"key":"page","size":4}\n', encoding="utf-8")
    report = {"objects": 1, "bytes": 4, "manifest_sha256": tool.manifest_digest({"page": 4})}
    tool.write_json(root / "prefill/persisted.json", {"device": 0, "segment": "P0", **report})
    tool.write_json(
        root / "holder1/ready.json", {"device": 1, "segment": "segment1", "source_segment": "P0", "copy": report}
    )
    tool.write_json(
        root / "holder0/ready.json", {"device": 0, "segment": "segment0", "source_segment": "segment1", "copy": report}
    )


def test_no_required_config_and_deployment_defaults(tool):
    defaults = tool.parser().parse_args([])
    assert defaults.master is None
    assert defaults.master_bin == "mooncake_master"
    assert defaults.with_baseline is False
    args = options(tool)
    assert not hasattr(args, "config")
    base = tool.deployment_config(args.master, args.local_hostname)
    assert base["chunk_size"] == 1024
    assert base["pin_timeout_sec"] == 1800
    assert base["shared_cpu_cache_numa_policy"] == "interleave"
    assert base["extra_config"]["local_hostname"] == "7.150.7.133"
    assert base["extra_config"]["transfer_timeout"] == 120
    assert base["extra_config"]["mooncake_dsa_raw_token_dims"] == {0: 576, 1: 128}


def test_local_hostname_comes_from_route_not_remote_master(tool, monkeypatch):
    calls = []

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect(self, destination):
            calls.append(destination)

        def getsockname(self):
            return ("7.150.7.133", 45000)

    monkeypatch.setattr(tool.socket, "socket", lambda *args: Probe())
    assert tool.detect_local_hostname("7.150.4.174:58888") == "7.150.7.133"
    assert calls == [("7.150.4.174", 58888)]


def base_config():
    return {
        "remote_url": "mooncakestore://localhost:58888/",
        "chunk_size": 1024,
        "shared_cpu_cache_name": "old-cache",
        "extra_config": {
            "global_segment_size": 100000000000,
            "local_hostname": "localhost",
            "master_server_address": "localhost:58888",
            "shared_cpu_cache_size_gb": 190,
        },
    }


@pytest.mark.parametrize("stage", ["baseline", "prefill", "decode"])
def test_only_prefill_model_owns_native_storage_without_reusing_old_cache(tool, stage):
    source = base_config()
    config = tool.stage_config(source, options(tool), stage)
    assert source == base_config()
    assert config["extra_config"]["global_segment_size"] == (8 * 1024**3 if stage == "prefill" else 0)
    assert config["extra_config"]["mooncake_prefer_local_alloc"] == (stage == "prefill")
    assert config["shared_cpu_cache_name"] is None
    assert config["extra_config"]["shared_cpu_cache_name"] is None
    assert config["extra_config"]["shared_cpu_cache_size_gb"] == 8
    assert config["enable_remote_lmcache_store"] == (stage != "baseline")
    assert bool(config["remote_url"]) == (stage != "baseline")


@pytest.mark.parametrize("stage", ["baseline", "prefill", "decode"])
def test_generated_config_passes_real_lmcache_validation(tool, stage, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "LMCache"))
    import lmcache.v1.config as config_module

    # Execute the actual Ascend config extension, without importing its NPU
    # plugin initializer. Restore global config/class state after each test.
    monkeypatch.setattr(config_module, "_CONFIG_DEFINITIONS", copy.deepcopy(config_module._CONFIG_DEFINITIONS))
    monkeypatch.setattr(config_module, "LMCacheEngineConfig", config_module.LMCacheEngineConfig)
    patch_path = Path(__file__).resolve().parents[4] / "LMCache-Ascend/lmcache_ascend/__init__.py"
    tree = ast.parse(patch_path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if getattr(node, "name", None) == "_patch_config")
    namespace = {"Optional": Optional, "sys": NS(modules={})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(patch_path), "exec"), namespace)
    namespace["_patch_config"]()

    for key in list(os.environ):
        if key.startswith(("LMCACHE_", "MOONCAKE_")):
            monkeypatch.delenv(key)
    env = tool.child_environment(options(tool), tmp_path, stage)
    for key, value in env.items():
        if key.startswith("LMCACHE_"):
            monkeypatch.setenv(key, value)
    obj = config_module.LMCacheEngineConfig.from_env()
    obj.validate()
    assert obj.chunk_size == 1024
    assert obj.extra_config["transfer_timeout"] == 120
    assert obj.remote_url == (None if stage == "baseline" else "mooncakestore://127.0.0.1:45678/")
    assert obj.extra_config["master_server_address"] == "127.0.0.1:45678"
    assert obj.pd_role == ("receiver" if stage == "decode" else "sender")


def test_no_archive_or_probe_environment_leaks(tool, tmp_path, monkeypatch):
    monkeypatch.setenv("LMCACHE_EXTRA_CONFIG", '{"validation_stage_dir":"stale"}')
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "other.json")
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "stale.yaml")
    env = tool.child_environment(options(tool), tmp_path, "prefill")
    assert "validation_stage_dir" not in json.loads(env["LMCACHE_EXTRA_CONFIG"])
    assert "MOONCAKE_CONFIG_PATH" not in env
    assert "LMCACHE_CONFIG_FILE" not in env
    assert env["LMCACHE_CHUNK_SIZE"] == "1024"
    assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == "true"


@pytest.mark.parametrize("stage", ["holder0", "holder1", "baseline", "prefill", "decode"])
def test_holder_and_prefill_environment_have_storage_segments(tool, tmp_path, stage):
    env = tool.child_environment(options(tool), tmp_path, stage)
    extra = json.loads(env["LMCACHE_EXTRA_CONFIG"])
    assert extra["global_segment_size"] == (8 * 1024**3 if stage.startswith("holder") or stage == "prefill" else 0)
    assert extra["protocol"] == "ascend"
    assert "LMCACHE_CONFIG_FILE" not in env


@pytest.mark.parametrize("p_cached,d_cached,valid", [(0, 8192, True), (1, 8192, False), (0, 0, False)])
def test_summary_requires_real_fresh_p_and_d_reload(tool, tmp_path, p_cached, d_cached, valid):
    tool.write_json(tmp_path / "prompt.json", {"length": 9000})
    for stage, cached, tokens in [("baseline", 0, [1, 2]), ("prefill", p_cached, [1]), ("decode", d_cached, [1, 3])]:
        (tmp_path / stage).mkdir()
        tool.write_json(tmp_path / stage / "output.json", {"num_cached_tokens": cached, "token_ids": tokens})
    copy_evidence(tool, tmp_path)
    if valid:
        tool.analyse(tmp_path, 1024, with_baseline=True)
    else:
        with pytest.raises(RuntimeError, match="Did not exercise"):
            tool.analyse(tmp_path, 1024, with_baseline=True)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["p_computed_and_d_reloaded"] is valid
    assert summary["baseline_ran"] is True
    assert summary["decode_first_difference"] == 1  # differences are reported, not raised


@pytest.mark.parametrize("with_baseline", [False, True])
def test_prefill_group_is_fully_stopped_before_decode_launch(tool, tmp_path, monkeypatch, with_baseline):
    actions = []

    def start(args, root, stage):
        actions.append(("start", stage))
        return NS(stage=stage, poll=lambda: None if stage.startswith("holder") else 0, returncode=0)

    monkeypatch.setattr(tool, "start_child", start)
    monkeypatch.setattr(tool, "finish_child", lambda proc: actions.append(("stop", proc.stage)))
    monkeypatch.setattr(tool, "back_up_prefill", lambda *args: actions.append(("backup", "prefill")))
    monkeypatch.setattr(tool, "wait_for_holder", lambda *args: actions.append(("ready", "holder0")))
    args = options(tool)
    args.with_baseline = with_baseline
    tool.run_models(args, tmp_path, {"holder1": NS(poll=lambda: None)})
    stages = ("baseline", "prefill", "decode") if with_baseline else ("prefill", "decode")
    expected = []
    for stage in stages[:-1]:
        expected.append(("start", stage))
        if stage == "prefill":
            expected.append(("backup", "prefill"))
        expected.append(("stop", stage))
    expected += [("start", "holder0"), ("ready", "holder0"), ("start", "decode"), ("stop", "decode")]
    assert actions == expected


@pytest.fixture
def holder_runtime(tool, tmp_path, monkeypatch):
    """Fake only native boundaries; execute the actual holder startup/close."""
    actions = []
    args = options(tool)
    args.devices = "4,5,6,7"
    args.child = "holder1"
    args.run_dir = tmp_path
    (tmp_path / "holder1").mkdir()
    copy_evidence(tool, tmp_path)
    (tmp_path / "holder1/ready.json").unlink()
    env = tool.child_environment(args, tmp_path, "holder1")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", env["ASCEND_RT_VISIBLE_DEVICES"])
    monkeypatch.setenv("LMCACHE_EXTRA_CONFIG", env["LMCACHE_EXTRA_CONFIG"])
    state = NS(protocol="ascend", setup_status=0, init_error=None)

    def set_device(device):
        # Physical 5 is logical 1; never pass the physical ordinal here.
        assert device == int(args.child[-1])
        actions.append("set_device")

    def init():
        actions.append("npu_init")
        if state.init_error:
            raise state.init_error

    class Store:
        def __init__(self):
            # Reproduce aclrtGetDevice failing without an initialized context.
            if state.protocol == "ascend":
                assert actions == ["set_device", "npu_init"], "Missing NPU initialization before Mooncake"
            actions.append("construct")

        def setup(self, *values):
            actions.append("setup")
            assert values[2] == 8 * 1024**3
            assert values[4] == state.protocol
            return state.setup_status

        def get_hostname(self):
            return "test-holder-segment"

        def close(self):
            actions.append("close")

    monkeypatch.setitem(sys.modules, "torch_npu", NS(npu=NS(set_device=set_device, init=init)))
    monkeypatch.setitem(sys.modules, "mooncake", NS())
    monkeypatch.setitem(sys.modules, "mooncake.store", NS(MooncakeDistributedStore=Store, ReplicateConfig=NS))
    monkeypatch.setattr(tool, "holder_preflight", lambda *args: actions.append("preflight"))

    def copy(*args, **kwargs):
        actions.append("copy")
        return {"objects": 1, "bytes": 4, "manifest_sha256": tool.manifest_digest({"page": 4})}

    monkeypatch.setattr(tool, "copy_objects", copy)
    monkeypatch.setattr(tool.signal, "signal", lambda *_: None)
    monkeypatch.setattr(tool.threading, "Event", lambda: NS(set=lambda: None, wait=lambda: actions.append("wait")))
    return args, actions, state


def test_holder_initializes_visible_npu_before_mooncake(tool, holder_runtime):
    args, actions, _ = holder_runtime
    tool.run_holder(args)
    assert actions == ["set_device", "npu_init", "construct", "setup", "preflight", "copy", "wait", "close"]
    ready = json.loads((args.run_dir / "holder1/ready.json").read_text())
    assert ready["segment"] == "test-holder-segment"


def test_holder_npu_init_failure_does_not_start_store_or_publish_ready(tool, holder_runtime):
    args, actions, state = holder_runtime
    state.init_error = RuntimeError("NPU initialization failed")
    with pytest.raises(RuntimeError, match="NPU initialization failed"):
        tool.run_holder(args)
    assert actions == ["set_device", "npu_init"]
    assert not (args.run_dir / "holder1/ready.json").exists()


def test_holder_setup_failure_closes_store_without_publishing_ready(tool, holder_runtime):
    args, actions, state = holder_runtime
    state.setup_status = -600
    with pytest.raises(RuntimeError, match="Mooncake holder setup failed: -600"):
        tool.run_holder(args)
    assert actions == ["set_device", "npu_init", "construct", "setup", "close"]
    assert not (args.run_dir / "holder1/ready.json").exists()


@pytest.mark.parametrize("copy_failed", [False, True])
def test_holder0_publishes_ready_only_after_copy(tool, holder_runtime, monkeypatch, copy_failed):
    args, actions, _ = holder_runtime
    args.child = "holder0"
    copy_evidence(tool, args.run_dir)
    ready_path = args.run_dir / "holder0/ready.json"
    ready_path.unlink()
    report = {"objects": 1, "bytes": 4, "manifest_sha256": tool.manifest_digest({"page": 4})}

    def copy(*values, **kwargs):
        assert not ready_path.exists()
        assert values[2] == {"page": 4}
        assert values[4] == "segment1"
        actions.append("copy")
        if copy_failed:
            raise RuntimeError("copy failed")
        return report

    monkeypatch.setattr(tool, "copy_objects", copy)
    if copy_failed:
        with pytest.raises(RuntimeError, match="copy failed"):
            tool.run_holder(args)
        assert not ready_path.exists()
        assert "wait" not in actions
    else:
        tool.run_holder(args)
        assert json.loads(ready_path.read_text())["copy"] == report
        assert actions.index("copy") < actions.index("wait")
    assert actions[-1] == "close"


@pytest.mark.parametrize("bad", ["digest", "device", "segment", "objects"])
def test_ready_with_wrong_copy_evidence_cannot_launch_decode(tool, tmp_path, monkeypatch, bad):
    copy_evidence(tool, tmp_path)
    path = tmp_path / "holder0/ready.json"
    record = json.loads(path.read_text())
    if bad == "device":
        record["device"] = 1
    elif bad == "segment":
        record["segment"] = "segment1"
    elif bad == "objects":
        record["copy"]["objects"] = 0
    else:
        record["copy"]["manifest_sha256"] = "wrong"
    tool.write_json(path, record)
    launched = []

    def start(args, root, stage):
        launched.append(stage)
        return NS(poll=lambda: 0 if stage == "prefill" else None, returncode=0)

    monkeypatch.setattr(tool, "start_child", start)
    monkeypatch.setattr(tool, "finish_child", lambda *_: None)
    monkeypatch.setattr(tool, "back_up_prefill", lambda *_: None)
    with pytest.raises(RuntimeError, match="refusing to start D"):
        tool.run_models(options(tool), tmp_path, {"holder1": NS(poll=lambda: None)})
    assert launched == ["prefill", "holder0"]


def test_holder_exit_during_copy_fails_without_waiting_for_timeout(tool, tmp_path):
    (tmp_path / "holder0").mkdir()
    with pytest.raises(RuntimeError, match="holder0"):
        tool.wait_for_holder(tmp_path, "holder0", {"holder0": NS(poll=lambda: 1)}, None)


def test_force_tcp_settings_are_not_inherited(tool, tmp_path, monkeypatch):
    monkeypatch.setenv("MC_FORCE_TCP", "1")
    monkeypatch.setenv("MC_FORCE_SHM", "1")
    for stage in ("holder0", "holder1", "prefill", "decode"):
        env = tool.child_environment(options(tool), tmp_path, stage)
        assert "MC_FORCE_TCP" not in env and "MC_FORCE_SHM" not in env
        assert json.loads(env["LMCACHE_EXTRA_CONFIG"])["protocol"] == "ascend"


def test_tcp_holder_is_rejected_in_ascend_validation(tool, holder_runtime, monkeypatch):
    args, actions, state = holder_runtime
    state.protocol = "tcp"
    extra = json.loads(os.environ["LMCACHE_EXTRA_CONFIG"])
    extra["protocol"] = state.protocol
    monkeypatch.setenv("LMCACHE_EXTRA_CONFIG", json.dumps(extra))
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    with pytest.raises(ValueError, match="requires the Ascend transport"):
        tool.run_holder(args)
    assert actions == []


def test_local_nic_detection_has_no_remote_master_dependency(tool, monkeypatch):
    calls = []

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect(self, address):
            calls.append(address)

        def getsockname(self):
            return ("7.150.7.133", 12345)

    monkeypatch.setattr(tool.socket, "socket", lambda *args: Probe())
    assert tool.detect_local_hostname() == "7.150.7.133"
    assert calls == [("192.0.2.1", 9)]  # routing query only, no send()


def test_local_master_ports_are_distinct_and_available(tool):
    ports = tool.local_master_ports()
    assert len(set(ports)) == 2
    for port in ports:
        with tool.socket.socket() as probe:
            probe.bind(("127.0.0.1", port))


@pytest.mark.parametrize("stage_failure", [False, True])
def test_managed_master_defaults_to_local_and_cleans_only_own_process(tool, tmp_path, monkeypatch, stage_failure):
    args = tool.parser().parse_args([])
    actions = []
    proc = NS(pid=123)
    (tmp_path / "master").mkdir()
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "remote.yaml")
    monkeypatch.setenv("MOONCAKE_MASTER", "old-remote:58888")
    monkeypatch.setattr(tool.shutil, "which", lambda _: "/usr/local/bin/mooncake_master")
    monkeypatch.setattr(tool, "local_master_ports", lambda: (45678, 45679))

    def start(command, env, log_path, label):
        actions.append("start_master")
        assert command == [
            "/usr/local/bin/mooncake_master",
            "--port=45678",
            "--rpc_address=127.0.0.1",
            "--metrics_port=45679",
            "--enable_metric_reporting=false",
            "--enable_http_metadata_server=false",
            "--logtostderr=true",
        ]
        assert "MOONCAKE_CONFIG_PATH" not in env
        assert "MOONCAKE_MASTER" not in env
        assert log_path == tmp_path / "master/server.log"
        return proc

    monkeypatch.setattr(tool, "start_logged_process", start)
    monkeypatch.setattr(tool, "wait_for_master", lambda *args: actions.append("ready"))
    monkeypatch.setattr(tool, "finish_child", lambda child: actions.append(("stop", child.pid)))

    def run():
        with tool.managed_master(args, tmp_path) as child:
            assert child is proc
            assert args.master == "127.0.0.1:45678"
            actions.append("run_models")
            if stage_failure:
                raise RuntimeError("prefill failed")

    if stage_failure:
        with pytest.raises(RuntimeError, match="prefill failed"):
            run()
    else:
        run()
    assert actions == ["start_master", "ready", "run_models", ("stop", 123)]


def test_explicit_master_is_never_started_or_stopped(tool, tmp_path, monkeypatch):
    args = options(tool)

    def forbidden(*args):
        pytest.fail("An existing master must not be managed")

    monkeypatch.setattr(tool, "start_logged_process", forbidden)
    monkeypatch.setattr(tool, "finish_child", forbidden)
    with tool.managed_master(args, tmp_path) as master:
        assert master is None


def test_missing_master_binary_fails_before_starting_any_process(tool, tmp_path, monkeypatch):
    args = tool.parser().parse_args(["--master-bin", str(tmp_path / "missing_master")])
    monkeypatch.setattr(tool.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="mooncake_master executable not found"):
        with tool.managed_master(args, tmp_path):
            pytest.fail("Must not start test without a local master")


def test_master_startup_failure_closes_its_process(tool, tmp_path, monkeypatch):
    args = tool.parser().parse_args([])
    closed = []
    proc = NS(pid=456)
    monkeypatch.setattr(tool.shutil, "which", lambda _: "/bin/mooncake_master")
    monkeypatch.setattr(tool, "start_logged_process", lambda *args: proc)

    def failed(*args):
        raise RuntimeError("master bind failed")

    monkeypatch.setattr(tool, "wait_for_master", failed)
    monkeypatch.setattr(tool, "finish_child", closed.append)
    with pytest.raises(RuntimeError, match="master bind failed"):
        with tool.managed_master(args, tmp_path):
            pytest.fail("Must not proceed after failed startup")
    assert closed == [proc]


def test_wait_for_master_checks_real_local_listener(tool):
    with tool.socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        tool.wait_for_master(NS(poll=lambda: None), f"127.0.0.1:{listener.getsockname()[1]}", "server.log")


def test_wait_for_master_fails_immediately_on_exit(tool):
    with pytest.raises(RuntimeError, match=r"master exited \(7\)"):
        tool.wait_for_master(NS(poll=lambda: 7, returncode=7), "127.0.0.1:45678", "server.log")


def test_wait_for_master_timeout_is_bounded(tool, monkeypatch):
    times = iter([0, 1, tool.MASTER_STARTUP_TIMEOUT_SECONDS + 1])
    monkeypatch.setattr(tool.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(tool.time, "sleep", lambda _: None)

    def refused(*args, **kwargs):
        raise ConnectionRefusedError()

    monkeypatch.setattr(tool.socket, "create_connection", refused)
    with pytest.raises(RuntimeError, match="did not listen"):
        tool.wait_for_master(NS(poll=lambda: None), "127.0.0.1:45678", "server.log")


def test_master_death_stops_current_model_instead_of_retrying(tool, tmp_path, monkeypatch):
    proc = NS(poll=lambda: None)
    stopped = []
    monkeypatch.setattr(tool, "start_child", lambda *args: proc)
    monkeypatch.setattr(tool, "finish_child", stopped.append)
    states = iter([None, 1])
    with pytest.raises(RuntimeError, match="Local Mooncake master exited"):
        tool.run_models(options(tool), tmp_path, {"holder1": NS(poll=lambda: None)}, NS(poll=lambda: next(states)))
    assert stopped == [proc]


@pytest.mark.parametrize("prefill_failed", [False, True])
@pytest.mark.parametrize("with_baseline", [False, True])
def test_run_check_keeps_storage_alive_across_p_exit(tool, tmp_path, monkeypatch, prefill_failed, with_baseline):
    args = options(tool)
    args.with_baseline = with_baseline
    args.prompt_file = tmp_path / "article.txt"
    args.prompt_file.write_text("Test article", encoding="utf-8")
    actions = []
    stages = ("baseline", "prefill", "decode") if with_baseline else ("prefill", "decode")
    for stage in ("holder1", "holder0", *stages):
        (tmp_path / stage).mkdir()
    monkeypatch.setattr(tool, "prepare_prompt", lambda *_: 9000)
    monkeypatch.setattr(tool, "copy_report", lambda *_: {})

    def start(args, root, stage):
        actions.append(("start", stage))
        if stage.startswith("holder"):
            assert (root / "prefill/persisted.json").exists()
            if stage == "holder0":
                assert ("stop", "prefill") in actions
            tool.write_json(root / stage / "ready.json", {"pid": 10})
            return NS(stage=stage, poll=lambda: None)
        if stage == "prefill" and not prefill_failed:
            tool.write_json(root / "prefill/persisted.json", {"device": 0, "segment": "P0"})
            return NS(stage=stage, poll=lambda: 0 if (root / "prefill/release.json").exists() else None, returncode=0)
        code = 1 if stage == "prefill" and prefill_failed else 0
        return NS(stage=stage, poll=lambda: code, returncode=code)

    monkeypatch.setattr(tool, "start_child", start)
    monkeypatch.setattr(tool, "finish_child", lambda proc: actions.append(("stop", proc.stage)))

    def analyse(root, chunk_size, baseline_enabled):
        assert baseline_enabled == with_baseline
        actions.append(("analyse", "outputs"))

    monkeypatch.setattr(tool, "analyse", analyse)
    if prefill_failed:
        with pytest.raises(RuntimeError, match="P exited before"):
            tool.run_check(args, tmp_path, NS(poll=lambda: None))
        expected_stages = stages[:-1]
    else:
        tool.run_check(args, tmp_path, NS(poll=lambda: None))
        expected_stages = stages
    expected = []
    for stage in expected_stages:
        if stage == "decode":
            expected.append(("start", "holder0"))
        expected.append(("start", stage))
        if stage == "prefill" and not prefill_failed:
            expected.append(("start", "holder1"))
        expected.append(("stop", stage))
    if not prefill_failed:
        expected.append(("analyse", "outputs"))
        expected.append(("stop", "holder0"))
        expected.append(("stop", "holder1"))
    assert actions == expected
    for stage in ("prefill", "decode", "holder1", "holder0"):
        env = json.loads((tmp_path / stage / "lmcache_env.json").read_text())
        assert env["LMCACHE_REMOTE_URL"] == "mooncakestore://127.0.0.1:45678/"
    assert (tmp_path / "baseline").exists() == with_baseline


def test_failed_backup_never_releases_prefill(tool, tmp_path, monkeypatch):
    copy_evidence(tool, tmp_path)
    holder = NS(poll=lambda: None)
    monkeypatch.setattr(tool, "start_child", lambda *_: holder)

    def fail(*args):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(tool, "wait_for_holder", fail)
    holders = {}
    with pytest.raises(RuntimeError, match="copy failed"):
        tool.back_up_prefill(options(tool), tmp_path, NS(poll=lambda: None), holders, None)
    assert holders == {"holder1": holder}
    assert not (tmp_path / "prefill/release.json").exists()


def test_backup_monitors_original_source_owner(tool, tmp_path, monkeypatch):
    copy_evidence(tool, tmp_path)
    (tmp_path / "holder1/ready.json").unlink()
    states = iter([None, 1])
    monkeypatch.setattr(tool, "start_child", lambda *_: NS(poll=lambda: None))
    with pytest.raises(RuntimeError, match="prefill/server.log"):
        tool.back_up_prefill(options(tool), tmp_path, NS(poll=lambda: next(states)), {}, None)
    assert not (tmp_path / "prefill/release.json").exists()


def test_prefill_exit_gate_validates_backup(tool, tmp_path):
    copy_evidence(tool, tmp_path)
    tool.write_json(tmp_path / "prefill/release.json", {})
    tool.wait_for_backup_release(tmp_path)
    target = tmp_path / "holder1/ready.json"
    record = json.loads(target.read_text())
    record["copy"]["bytes"] = 1
    tool.write_json(target, record)
    with pytest.raises(RuntimeError, match="copy does not match"):
        tool.wait_for_backup_release(tmp_path)


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_prefill_does_not_override_production_memcpy_setting(tool, tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MC_STORE_MEMCPY", raising=False)
    else:
        monkeypatch.setenv("MC_STORE_MEMCPY", value)
    assert tool.child_environment(options(tool), tmp_path, "prefill").get("MC_STORE_MEMCPY") == value
    assert tool.child_environment(options(tool), tmp_path, "holder1")["MC_STORE_MEMCPY"] == "1"


@pytest.mark.parametrize("failure", [None, "fence", "backup"])
def test_model_keeps_native_store_alive_until_backup_finishes(tool, tmp_path, monkeypatch, failure):
    args = options(tool)
    args.run_dir, args.child = tmp_path, "prefill"
    (tmp_path / "prefill").mkdir()
    tool.write_json(tmp_path / "prompt.json", {"length": 9000, "token_ids": [1, 2, 3]})
    actions = []

    def generate(*args, **kwargs):
        actions.append("generate")
        return [NS(outputs=[NS(text="test", token_ids=[3], finish_reason="length")], num_cached_tokens=0)]

    def rpc(method, **kwargs):
        assert method == "prefill_check_seal_source"
        actions.append("fence")
        if failure == "fence":
            raise RuntimeError("fence failed")
        return [{"device": 0, "segment": "P0"}, None]

    llm = NS(
        generate=generate,
        collective_rpc=rpc,
        llm_engine=NS(engine_core=NS(shutdown=lambda: actions.append("shutdown"))),
    )
    monkeypatch.setitem(sys.modules, "vllm", NS(LLM=lambda **_: llm, SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setattr(tool, "engine_options", lambda *_: {"worker_extension_cls": "old_probe"})

    def wait(root):
        assert actions == ["generate", "fence"]
        assert json.loads((root / "prefill/persisted.json").read_text())["segment"] == "P0"
        actions.append("backup")
        if failure == "backup":
            raise RuntimeError("backup failed")

    monkeypatch.setattr(tool, "wait_for_backup_release", wait)
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            tool.run_model(args)
    else:
        tool.run_model(args)
    assert actions == ["generate", "fence", *([] if failure == "fence" else ["backup"]), "shutdown"]


@pytest.mark.parametrize("stale_baseline", [False, True])
@pytest.mark.parametrize("cached,valid", [(8192, True), (0, False)])
def test_summary_without_baseline_never_reads_or_claims_baseline_comparison(
    tool, tmp_path, stale_baseline, cached, valid
):
    tool.write_json(tmp_path / "prompt.json", {"length": 9000})
    for stage, hits in (("prefill", 0), ("decode", cached)):
        (tmp_path / stage).mkdir()
        tool.write_json(tmp_path / stage / "output.json", {"num_cached_tokens": hits, "token_ids": [1]})
    copy_evidence(tool, tmp_path)
    if stale_baseline:
        (tmp_path / "baseline").mkdir()
        (tmp_path / "baseline/output.json").write_text("not valid JSON", encoding="utf-8")
    if valid:
        tool.analyse(tmp_path, 1024)
    else:
        with pytest.raises(RuntimeError, match="Did not exercise"):
            tool.analyse(tmp_path, 1024)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["baseline_ran"] is False
    assert "prefill_first_token_equal" not in summary
    assert "decode_first_difference" not in summary
    assert summary["p_computed_and_d_reloaded"] is valid


def test_baseline_requires_explicit_opt_in(tool):
    assert tool.model_stages(tool.parser().parse_args([])) == ("prefill", "decode")
    assert tool.model_stages(tool.parser().parse_args(["--with-baseline"])) == ("baseline", "prefill", "decode")


def test_clear_shared_memory_only_removes_visible_children(tool, tmp_path, monkeypatch, capsys):
    # Redirect the fixed production path to a pytest-owned directory. Never
    # touch this machine's /dev/shm, even when these tests run on Linux.
    root = tmp_path.resolve()
    (root / "cache").write_text("shared pages", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested/data").write_text("shared pages", encoding="utf-8")
    (root / ".hidden").write_text("untouched", encoding="utf-8")

    def fixed_path(value):
        assert value == "/dev/shm"
        return root

    monkeypatch.setattr(tool, "Path", fixed_path)
    tool.clear_shared_memory()
    assert root.is_dir()
    assert [entry.name for entry in root.iterdir()] == [".hidden"]
    assert "removed 2 entries" in capsys.readouterr().out


def test_clear_shared_memory_unlinks_symlinks_without_following_them(tool, monkeypatch):
    removed = []
    link = NS(name="external", is_symlink=lambda: True, unlink=lambda **_: removed.append("link"))
    root = NS(is_symlink=lambda: False, is_dir=lambda: True, iterdir=lambda: iter([link]))
    root.resolve = lambda: root
    monkeypatch.setattr(tool, "Path", lambda _: root)

    def forbidden(*args):
        pytest.fail("Must not recursively delete a symlink target")

    monkeypatch.setattr(tool.shutil, "rmtree", forbidden)
    tool.clear_shared_memory()
    assert removed == ["link"]


def test_clear_shared_memory_rejects_redirected_root(tool, monkeypatch):
    monkeypatch.setattr(tool, "Path", lambda _: NS(is_symlink=lambda: True))
    with pytest.raises(RuntimeError, match="expected a real directory"):
        tool.clear_shared_memory()


@pytest.mark.parametrize("stage", ["holder0", "holder1", "baseline", "prefill", "decode"])
def test_child_never_clears_shared_memory(tool, monkeypatch, stage):
    args = options(tool)
    args.child = stage
    actions = []
    monkeypatch.setattr(tool, "parser", lambda: NS(parse_args=lambda: args))
    monkeypatch.setattr(tool, "clear_shared_memory", lambda: pytest.fail("Child must never clear /dev/shm"))
    monkeypatch.setattr(tool, "run_holder", lambda _: actions.append("holder"))
    monkeypatch.setattr(tool, "run_model", lambda _: actions.append("model"))
    tool.main()
    assert actions == ["holder" if stage.startswith("holder") else "model"]


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_parent_cleans_once_before_starting_master(tool, tmp_path, monkeypatch, cleanup_fails):
    args = options(tool)
    args.run_dir = tmp_path
    actions = []
    monkeypatch.setattr(tool, "parser", lambda: NS(parse_args=lambda: args))
    monkeypatch.setattr(tool, "os", NS(name="posix"))

    def clean():
        actions.append("clean")
        if cleanup_fails:
            raise PermissionError("shared memory cleanup failed")

    @contextmanager
    def master(*args):
        actions.append("master")
        yield None

    monkeypatch.setattr(tool, "clear_shared_memory", clean)
    monkeypatch.setattr(tool, "managed_master", master)
    monkeypatch.setattr(tool, "run_check", lambda *args: actions.extend(["holder", "prefill", "decode"]))
    if cleanup_fails:
        with pytest.raises(PermissionError, match="cleanup failed"):
            tool.main()
        assert actions == ["clean"]
    else:
        tool.main()
        assert actions == ["clean", "master", "holder", "prefill", "decode"]
