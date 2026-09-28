# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the real-model smoke launcher and its failure boundaries."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    return importlib.import_module("pd_tensor_smoke")


def manifest(smoke, root, role, rank=0, **updates):
    row = dict(request_id="0", tp_rank=rank, records=10, complete=True, request_finished=True, errors=[])
    row.update(updates)
    path = root / "capture" / role / "0" / f"worker-{rank}" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    smoke.write_json(temporary, row)
    temporary.replace(path)
    return path


def test_dry_run_has_no_model_imports_or_files(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools" / "pd_tensor_smoke.py"
    code = """
import builtins, runpy, sys
real_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'vllm', 'torch', 'torch_npu', 'lmcache', 'mooncake', 'transformers'}:
        raise AssertionError('dry-run imported model/runtime package: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = checked_import
script, destination = sys.argv[1:]
sys.path.insert(0, str(__import__('pathlib').Path(script).parent))
sys.argv = [script, '--dry-run', '--run-dir', destination]
runpy.run_path(script, run_name='__main__')
"""
    destination = tmp_path / "not-created"
    result = subprocess.run([sys.executable, "-c", code, str(script), str(destination)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not destination.exists()
    plan = json.loads(result.stdout)
    assert [(row["case"], row["stage"]) for row in plan] == [
        ("off", "prefill"),
        ("off", "decode"),
        ("on", "prefill"),
        ("on", "decode"),
    ]
    assert [row["output_token_limit"] for row in plan] == [1, 16, 1, 16]
    for row in plan:
        options = row["engine_options"]
        assert options["tensor_parallel_size"] == 8
        assert options["data_parallel_size"] == 1
        assert options["compilation_config"] == {"mode": 0, "cudagraph_mode": "NONE"}
        assert options["enforce_eager"] and options["async_scheduling"] is False
        assert not options["additional_config"]["enable_npugraph_ex"]
        assert options["worker_extension_cls"] == "layerwise_prefill_file_worker.FileStoreWorker"
        assert "profiler_config" not in options


@pytest.mark.parametrize("case", ["off", "on"])
@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_environment_isolated_and_roles_use_correct_paths(smoke, tmp_path, monkeypatch, case, stage):
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "/stale/config.yaml")
    monkeypatch.setenv("LMCACHE_EXTRA_CONFIG", '{"poison":true}')
    monkeypatch.setenv("VLLM_ASCEND_PD_TENSOR_DUMP_DIR", "/stale/capture")
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")
    args = smoke.parser().parse_args([])
    root = tmp_path / case
    env = smoke.child_environment(args, root, case, stage)
    assert "LMCACHE_CONFIG_FILE" not in env
    assert "poison" not in env["LMCACHE_EXTRA_CONFIG"]
    assert env["VLLM_ASCEND_PD_TENSOR_DUMP_DIR"] == str((root / "capture").resolve())
    assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == str(case == "on" and stage == "prefill").lower()
    assert env["VLLM_ASCEND_ENABLE_FLASHCOMM1"] == ("1" if stage == "prefill" else "0")
    assert env["LMCACHE_STORE_ASYNC"] == str(stage == "prefill").lower()
    assert env["LMCACHE_USE_LAYERWISE"] == "true"
    assert env["LMCACHE_PD_ROLE"] == ("sender" if stage == "prefill" else "receiver")
    sdk = json.loads(env["LMCACHE_EXTRA_CONFIG"])["prefill_check_file_sdk"]
    assert sdk == {"root": str(root.resolve()), "stage": stage}
    assert env["VLLM_ASCEND_SFA_STAGED_GRAPH"] == "0"
    assert env["VLLM_ASCEND_SFA_STAGED_MTP_DRAFT_GRAPH"] == "0"
    assert not root.exists()


def test_child_command_preserves_run_settings(smoke, tmp_path):
    args = smoke.parser().parse_args(
        ["--model", "/checkpoint/custom", "--output-tokens", "7", "--mtp-tokens", "0", "--devices", "4,5"]
    )
    command = smoke.child_command(args, tmp_path, "off", "decode")
    child = smoke.parser().parse_args(command[3:])
    smoke.validate_args(child)
    assert child.model == args.model and child.output_tokens == 7
    assert child.mtp_tokens == 0 and child.devices == "4,5"
    assert child.run_dir == tmp_path and child.case == "off" and child.child == "decode"


@pytest.mark.parametrize(
    "options,match",
    [
        (["--devices", "0,0"], "distinct"),
        (["--devices", ""], "distinct"),
        (["--output-tokens", "1"], "two output"),
        (["--gpu-memory-utilization", "1"], "memory"),
        (["--cpu-cache-gb", "nan"], "memory"),
        (["--output-tokens", "100", "--max-model-len", "100"], "no room"),
        (["--child", "decode"], "child needs"),
    ],
)
def test_invalid_requests_fail_before_model_start(smoke, options, match):
    with pytest.raises(ValueError, match=match):
        smoke.validate_args(smoke.parser().parse_args(options))


def test_wait_observes_atomic_manifest_completion(smoke, tmp_path, monkeypatch):
    manifest(smoke, tmp_path, "P", 0)
    path = manifest(smoke, tmp_path, "P", 1, complete=False, request_finished=False)
    sleeps = []

    def finish(_seconds):
        sleeps.append(1)
        updated = json.loads(path.read_text())
        updated.update(complete=True, request_finished=True)
        temporary = path.with_suffix(".tmp")
        smoke.write_json(temporary, updated)
        temporary.replace(path)

    monkeypatch.setattr(smoke.time, "sleep", finish)
    rows = smoke.wait_capture_finished(tmp_path, "P", 2, "0")
    assert sleeps == [1]
    assert {row["tp_rank"] for row in rows} == {0, 1}
    assert all(row["request_finished"] and row["complete"] for row in rows)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"request_id": "other"}, "captured ID"),
        ({"tp_rank": 1}, "TP ranks"),
        ({"errors": ["missing KV tensor"]}, "missing KV tensor"),
        ({"complete": False}, "did not finish"),
        ({"request_finished": False}, "did not finish"),
    ],
)
def test_wait_rejects_bad_or_unfinished_archive(smoke, tmp_path, updates, match):
    manifest(smoke, tmp_path, "D", **updates)
    with pytest.raises(RuntimeError, match=match):
        smoke.wait_capture_finished(tmp_path, "D", 1, "0", timeout=0)


def test_wait_missing_rank_times_out(smoke, tmp_path):
    manifest(smoke, tmp_path, "D", 0)
    with pytest.raises(RuntimeError, match="all 2 ranks"):
        smoke.wait_capture_finished(tmp_path, "D", 2, "0", timeout=0)


def staged_processes(smoke, root, monkeypatch, *, p_code=0, seal_passed=True, timed_out=False):
    events = []

    class Process:
        pid = 100

        def __init__(self, stage):
            self.stage = stage
            self.returncode = None

        def wait(self, timeout):
            events.append(f"wait:{self.stage}")
            if timed_out:
                raise subprocess.TimeoutExpired("fake-model", timeout)
            self.returncode = p_code if self.stage == "prefill" else 0
            smoke.write_json(root / self.stage / "output.json", {"completed": self.returncode == 0})
            return self.returncode

    def start(command, env, logfile, label, prefix):
        stage = label.split("/")[-1]
        events.append(f"start:{stage}")
        return Process(stage)

    def seal(path):
        assert path == root
        assert (root / "prefill" / "process-exited.json").is_file()
        events.append("seal")
        return {"passed": seal_passed, "errors": [] if seal_passed else ["incomplete P write"]}

    store = importlib.import_module("layerwise_prefill_file_store")
    monkeypatch.setattr(store, "seal_store", seal)
    monkeypatch.setattr(smoke, "check_shm_capacity", lambda *_args: None)
    monkeypatch.setattr(smoke, "start_logged_process", start)
    monkeypatch.setattr(smoke, "finish_child", lambda proc: events.append(f"finish:{proc.stage}"))
    return events


def test_p_process_cleanup_and_seal_precede_d_start(smoke, tmp_path, monkeypatch):
    args = smoke.parser().parse_args([])
    events = staged_processes(smoke, tmp_path, monkeypatch)
    smoke.run_stages(args, tmp_path, "on")
    assert events == [
        "start:prefill",
        "wait:prefill",
        "finish:prefill",
        "seal",
        "start:decode",
        "wait:decode",
        "finish:decode",
    ]


@pytest.mark.parametrize(
    "failure,match,expected",
    [
        ({"p_code": 9}, "exited 9", ["start:prefill", "wait:prefill", "finish:prefill"]),
        ({"timed_out": True}, "timed out", ["start:prefill", "wait:prefill", "finish:prefill"]),
        ({"seal_passed": False}, "could not be sealed", ["start:prefill", "wait:prefill", "finish:prefill", "seal"]),
    ],
)
def test_p_failure_never_starts_d(smoke, tmp_path, monkeypatch, failure, match, expected):
    events = staged_processes(smoke, tmp_path, monkeypatch, **failure)
    with pytest.raises(RuntimeError, match=match):
        smoke.run_stages(smoke.parser().parse_args([]), tmp_path, "off")
    assert events == expected
    assert not (tmp_path / "decode").exists()


def fake_llm(smoke, root, stage, monkeypatch, *, fail=None):
    events = []
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "/stale/nonexistent-config.yaml")
    failures = {fail} if isinstance(fail, str) else set(fail or [])
    store = importlib.import_module("layerwise_prefill_file_store")
    monkeypatch.setattr(store, "install", lambda: events.append("sdk-install"))

    class LLM:
        def __init__(self, **options):
            # Even the internal entry point must ignore a stale external YAML.
            assert "LMCACHE_CONFIG_FILE" not in os.environ
            events.append("model-init")
            if "init" in failures:
                raise RuntimeError("init failure")
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=self.shutdown))

        def shutdown(self):
            events.append("shutdown")
            if "shutdown" in failures:
                raise RuntimeError("shutdown failure")

        def generate(self, prompt, params, use_tqdm):
            events.append("generate")
            if "generate" in failures:
                raise RuntimeError("generate failure")
            for rank in range(2):
                manifest(smoke, root, "P" if stage == "prefill" else "D", rank)
            cached = len(prompt["prompt_token_ids"]) - 1 if stage == "decode" else 0
            if "cache" in failures:
                cached += 1
            output = SimpleNamespace(token_ids=[42] * params.max_tokens, text="test output", finish_reason="length")
            return [SimpleNamespace(request_id="0", finished=True, outputs=[output], num_cached_tokens=cached)]

        def collective_rpc(self, method, **kwargs):
            assert method in {"flush_file_store", "close_file_store"}, "Old probes must never be installed or finalized"
            events.append(method)
            if method in failures:
                raise RuntimeError(f"{method} failure")
            return [True, True]

    vllm = ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = LLM, SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    samples = iter([{"vllm:spec_decode_num_draft_tokens": 0}, {"vllm:spec_decode_num_draft_tokens": 3}])
    monkeypatch.setattr(smoke, "mtp_snapshot", lambda _llm: next(samples))
    smoke.write_json(root / "prompt.json", {"length": 5, "token_ids": [1, 2, 3, 4, 5]})
    (root / stage).mkdir()
    args = smoke.parser().parse_args(["--case", "on", "--child", stage, "--run-dir", str(root), "--devices", "0,1"])
    return args, events


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_child_exercises_only_new_capture_then_closes_store(smoke, tmp_path, monkeypatch, stage):
    args, events = fake_llm(smoke, tmp_path, stage, monkeypatch)
    smoke.run_child(args)
    assert events == ["sdk-install", "model-init", "generate", "flush_file_store", "close_file_store", "shutdown"]
    output = json.loads((tmp_path / stage / "output.json").read_text())
    assert output["completed"]
    assert len(output["token_ids"]) == (1 if stage == "prefill" else 16)
    assert output["num_cached_tokens"] == (0 if stage == "prefill" else 4)
    captured = json.loads((tmp_path / stage / "capture-completion.json").read_text())
    assert len(captured) == 2


@pytest.mark.parametrize("failure", ["init", "generate", "cache", "flush_file_store", "close_file_store"])
def test_child_failures_persist_failure_and_release_created_engine(smoke, tmp_path, monkeypatch, failure):
    args, events = fake_llm(smoke, tmp_path, "decode", monkeypatch, fail=failure)
    with pytest.raises(RuntimeError):
        smoke.run_child(args)
    output = json.loads((tmp_path / "decode" / "output.json").read_text())
    assert not output["completed"]
    assert output.get("error") or output.get("close_error")
    if failure == "init":
        assert "close_file_store" not in events and "shutdown" not in events
    else:
        assert events[-2:] == ["close_file_store", "shutdown"]


@pytest.mark.parametrize("generate_fails", [False, True])
def test_shutdown_failure_is_recorded_without_replacing_original_error(smoke, tmp_path, monkeypatch, generate_fails):
    failures = {"shutdown", "generate"} if generate_fails else {"shutdown"}
    args, events = fake_llm(smoke, tmp_path, "decode", monkeypatch, fail=failures)
    with pytest.raises(RuntimeError, match="generate failure" if generate_fails else "shutdown failure"):
        smoke.run_child(args)
    output = json.loads((tmp_path / "decode" / "output.json").read_text())
    assert output["completed"] is False
    assert output["shutdown_error"] == "RuntimeError: shutdown failure"
    if generate_fails:
        assert output["error"] == "RuntimeError: generate failure"
    assert events[-2:] == ["close_file_store", "shutdown"]
