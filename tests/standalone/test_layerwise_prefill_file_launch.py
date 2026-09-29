# SPDX-License-Identifier: Apache-2.0
"""CPU launch regressions; actual model execution requires Linux Ascend."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import layerwise_prefill_file_check as runner
import layerwise_prefill_file_store as store


def test_real_connector_roles_and_inline_config_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "/deployment/production.yaml")
    monkeypatch.setenv("LMCACHE_REMOTE_URL", "mooncakestore://production:50051")
    monkeypatch.setenv("LMCACHE_EXTRA_CONFIG", '{"prefill_check_routing": "wrong"}')
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH", "1")
    args = runner.parser().parse_args([])
    for stage in runner.STAGES:
        env = runner.child_environment(args, tmp_path, stage)
        assert "LMCACHE_CONFIG_FILE" not in env
        extra = json.loads(env["LMCACHE_EXTRA_CONFIG"])
        assert "prefill_check_routing" not in extra
        assert extra["mooncake_layer_merged_page_objects"] is True
        assert extra["save_only_first_rank"] is True
        assert extra["save_chunk_meta"] is False
        assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == str(stage == "prefill").lower()
        assert env["LMCACHE_STORE_ASYNC"] == str(stage == "prefill").lower()
        assert env["LMCACHE_STORE_ASYNC_MAX_QUEUE_SIZE"] == "2"
        assert env["VLLM_ASCEND_SFA_STAGED_GRAPH"] == "0"
        if stage == "baseline":
            assert "LMCACHE_REMOTE_URL" not in env
            assert "prefill_check_file_sdk" not in extra
        else:
            assert extra["prefill_check_file_sdk"] == {"root": str(tmp_path.resolve()), "stage": stage}
        options = runner.engine_options(args, stage, 5)
        assert options["worker_extension_cls"] == "layerwise_prefill_file_worker.FileStoreWorker"
        assert (
            options["kv_transfer_config"]["kv_role"]
            == {
                "baseline": "kv_both",
                "prefill": "kv_producer",
                "decode": "kv_consumer",
            }[stage]
        )
        assert options["enforce_eager"] is True
        assert options["compilation_config"] == {"mode": 0, "cudagraph_mode": "NONE"}
        assert "profiler_config" not in options
        assert options["speculative_config"]["num_speculative_tokens"] == 1


def test_process_exit_precedes_seal_and_decode_launch(tmp_path, monkeypatch):
    args = runner.parser().parse_args(["--off-dir", str(tmp_path / "old")])
    events = []
    monkeypatch.setattr(runner, "check_shm_capacity", lambda *_: None)

    def launch(command, env, log_path, stage, **kwargs):
        events.append(f"start:{stage}")
        assert command[command.index("--output-tokens") + 1] == "64"
        runner.write_json(log_path.parent / "output.json", {"completed": True})
        return SimpleNamespace(pid=42, returncode=0, stage=stage, wait=lambda **_: 0)

    def seal(root):
        assert (root / "prefill" / "process-exited.json").is_file()
        events.append("seal")
        return {"passed": True}

    monkeypatch.setattr(runner, "start_logged_process", launch)
    monkeypatch.setattr(runner, "finish_child", lambda proc: events.append(f"exit:{proc.stage}"))
    monkeypatch.setattr(store, "seal_store", seal)
    runner.run_stages(args, tmp_path)
    assert events == ["start:prefill", "exit:prefill", "seal", "start:decode", "exit:decode"]


@pytest.mark.parametrize("failure", ["exit", "timeout", "seal", "unfinished"])
def test_failed_prefill_does_not_launch_decode(tmp_path, monkeypatch, failure):
    args = runner.parser().parse_args(["--off-dir", str(tmp_path / "old")])
    events = []
    monkeypatch.setattr(runner, "check_shm_capacity", lambda *_: None)

    def launch(command, env, log_path, stage, **kwargs):
        events.append(stage)
        runner.write_json(log_path.parent / "output.json", {"completed": failure != "unfinished"})

        def wait(**kwargs):
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 1)
            return 7 if failure == "exit" else 0

        return SimpleNamespace(pid=42, returncode=7 if failure == "exit" else 0, wait=wait)

    monkeypatch.setattr(runner, "start_logged_process", launch)
    monkeypatch.setattr(runner, "finish_child", lambda *_: events.append("stopped"))
    monkeypatch.setattr(store, "seal_store", lambda *_: {"passed": failure != "seal", "errors": ["no objects"]})
    with pytest.raises(RuntimeError):
        runner.run_stages(args, tmp_path)
    assert events == ["prefill", "stopped"]


@pytest.mark.parametrize("cached_tokens,passes", [(0, False), (4, True)])
def test_decode_reuses_full_short_prompt_and_rejects_silent_recompute(tmp_path, monkeypatch, cached_tokens, passes):
    args = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--child",
            "decode",
            "--devices",
            "0",
            "--mtp-tokens",
            "0",
            "--output-tokens",
            "3",
        ]
    )
    (tmp_path / "decode").mkdir()
    prompt = [10, 11, 12, 13, 14]
    runner.write_json(tmp_path / "prompt.json", {"token_ids": prompt, "length": len(prompt)})
    events = []

    class FakeLLM:
        def __init__(self, **kwargs):
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: events.append("shutdown")))

        def collective_rpc(self, method, **kwargs):
            events.append(method)
            if method == "finish_file_probe":
                summary = {"rank": 0, "complete": True, "errors": [], "records": 5}
                path = tmp_path / "decode" / "tensors" / "rank0" / "coverage.json"
                path.parent.mkdir(parents=True)
                runner.write_json(path, summary)
                return [{**summary, "coverage_path": "tensors/rank0/coverage.json"}]
            return [{"rank": 0, "complete": True}]

        def generate(self, payload, params, **kwargs):
            # D gets the exact original prompt, never the P completion appended.
            assert payload == {"prompt_token_ids": prompt}
            assert params.ignore_eos is True
            assert params.max_tokens == 3
            return [
                SimpleNamespace(
                    finished=True,
                    num_cached_tokens=cached_tokens,
                    outputs=[SimpleNamespace(token_ids=[99, 98, 97], text="answer", finish_reason="length")],
                )
            ]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=SimpleNamespace))
    monkeypatch.setattr(store, "install", lambda: None)
    if passes:
        runner.run_child(args)
    else:
        with pytest.raises(RuntimeError, match="must restore"):
            runner.run_child(args)
    saved = json.loads((tmp_path / "decode" / "output.json").read_text())
    assert saved["completed"] is passes
    assert events[-1] == "shutdown"
    assert events.index("flush_file_store") < events.index("finish_file_probe")
    assert events.index("finish_file_probe") < events.index("close_file_store") < events.index("shutdown")


def test_reused_off_does_not_depend_on_failed_on(tmp_path, monkeypatch):
    old_root = tmp_path / "old"
    baseline = old_root / "baseline"
    baseline.mkdir(parents=True)
    new_root = tmp_path / "new"
    new_root.mkdir()
    args = runner.parser().parse_args(["--off-dir", str(old_root)])
    options = {key: getattr(args, key) for key in runner.RUN_OPTIONS}
    options.update(model="/real/model", output_tokens=4, devices="0")
    runner.write_json(
        old_root / "run_config.json",
        {
            "schema_version": runner.SCHEMA_VERSION,
            "tool": "layerwise_prefill_file_check",
            "options": options,
        },
    )
    runner.write_json(old_root / "failure.json", {"error": "ON failed; OFF remains usable"})
    runner.write_json(old_root / "prompt.json", {"length": 5, "token_ids": [1, 2, 3, 4, 5]})
    runner.write_json(baseline / "output.json", {"completed": True, "prompt_token_ids": [1, 2, 3, 4, 5]})
    runner.write_json(old_root / "model_info.json", {"model": "/real/model", "num_hidden_layers": 8})
    saved_args = runner.parser().parse_args(["--model", "/real/model", "--output-tokens", "4", "--devices", "0"])
    runner.write_json(baseline / "engine_options.json", runner.engine_options(saved_args, "baseline", 5))
    fake_compare = SimpleNamespace(validate_baseline=lambda *_: {"valid": True})
    monkeypatch.setitem(sys.modules, "layerwise_prefill_file_compare", fake_compare)
    monkeypatch.setattr(runner, "record_model_identity", lambda *_: {"model": "/real/model", "num_hidden_layers": 8})
    before = {str(p): p.read_bytes() for p in old_root.rglob("*") if p.is_file()}
    assert runner.prepare_reused_off(args, new_root) == 5
    assert args.model == "/real/model"
    assert args.output_tokens == 4
    assert args.off_dir == baseline
    assert json.loads((new_root / "off_reference.json").read_text())["baseline_dir"] == str(baseline)
    assert before == {str(p): p.read_bytes() for p in old_root.rglob("*") if p.is_file()}
    mismatch = runner.parser().parse_args(["--off-dir", str(old_root), "--model", "/different/model"])
    with pytest.raises(ValueError, match="differs from OFF"):
        runner.prepare_reused_off(mismatch, new_root)


def test_bootstrap_has_no_native_fallback(tmp_path):
    runner.prepare_bootstrap(tmp_path)
    source = (tmp_path / "bootstrap" / "sitecustomize.py").read_text()
    assert "layerwise_prefill_file_store import install" in source
    assert "os._exit(1)" in source
    assert "/dev/shm" not in source


@pytest.mark.parametrize("reuse_off", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_parent_cleans_shm_once_before_models(tmp_path, monkeypatch, reuse_off, cleanup_fails):
    events = []
    monkeypatch.setattr(sys, "platform", "linux")

    def cleanup(command, *, check):
        assert command == ["/bin/sh", "-c", "rm -rf /dev/shm/*"]
        assert check is True
        events.append("cleanup")
        if cleanup_fails:
            raise subprocess.CalledProcessError(1, command)

    def prepare(args, root):
        runner.write_json(root / "model_info.json", {"num_hidden_layers": 78})
        return 5

    monkeypatch.setattr(subprocess, "run", cleanup)
    monkeypatch.setattr(runner, "record_model_identity", lambda *_: None)
    monkeypatch.setattr(runner, "prepare_prompt", prepare)
    monkeypatch.setattr(runner, "prepare_reused_off", prepare)
    monkeypatch.setattr(runner, "prepare_bootstrap", lambda *_: None)
    monkeypatch.setattr(runner, "run_stages", lambda *_: events.append("models"))
    monkeypatch.setattr(runner, "analyse", lambda *_: 0)
    argv = ["--run-dir", str(tmp_path / "run")]
    if reuse_off:
        argv.extend(["--off-dir", str(tmp_path / "old")])
    if cleanup_fails:
        with pytest.raises(subprocess.CalledProcessError):
            runner.main(argv)
        assert events == ["cleanup"]
    else:
        assert runner.main(argv) == 0
        assert events == ["cleanup", "models"]


@pytest.mark.parametrize("mode", ["child", "compare", "help"])
def test_non_parent_execution_never_cleans_shm(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Unexpected shared memory cleanup"))
    monkeypatch.setattr(runner, "run_child", lambda *_: None)
    monkeypatch.setattr(runner, "analyse", lambda *_: 0)
    if mode == "help":
        with pytest.raises(SystemExit) as exc:
            runner.main(["--help"])
        assert exc.value.code == 0
    else:
        argv = ["--child", "decode"] if mode == "child" else ["--compare-only", str(tmp_path)]
        assert runner.main(argv) == 0


@pytest.mark.parametrize("problem", ["rank_missing", "wrong_path", "incomplete_file"])
def test_rpc_manifest_cannot_hide_missing_worker_coverage(tmp_path, problem):
    summary = {"rank": 0, "complete": True, "errors": [], "records": 1}
    path = tmp_path / "tensors" / "rank0" / "coverage.json"
    path.parent.mkdir(parents=True)
    runner.write_json(path, {**summary, "complete": problem != "incomplete_file"})
    reply = {**summary, "coverage_path": "tensors/rank0/coverage.json"}
    if problem == "wrong_path":
        reply["coverage_path"] = "../another-run/coverage.json"
    with pytest.raises(RuntimeError):
        runner.load_worker_coverage(tmp_path, [] if problem == "rank_missing" else [reply], 1)
