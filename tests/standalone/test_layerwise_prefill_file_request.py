# SPDX-License-Identifier: Apache-2.0
"""CPU checks of LoCoMo shell invocation and the existing file probe contract."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import layerwise_prefill_file_check as runner
import layerwise_prefill_file_request as request
import layerwise_prefill_file_store as store


@pytest.mark.parametrize("reuse,fail", [(False, False), (True, False), (False, True)])
def test_locomo_subprocess_runs_after_server_ready_and_preserves_stage_order(tmp_path, monkeypatch, reuse, fail):
    args = runner.parser().parse_args(["--locomo", "--output-tokens", "16"])
    if reuse:
        args.off_dir = tmp_path / "old"
    events = []
    monkeypatch.setattr(runner, "check_shm_capacity", lambda *_: None)
    monkeypatch.setattr(request, "check_port", lambda: None)

    def launch(command, env, log, stage, **kwargs):
        assert "layerwise_prefill_file_request.py" in command[2]
        assert "--no-disable-log-stats" not in command
        assert command[command.index("--capture-output-tokens") + 1] == ("1" if stage == "prefill" else "16")
        assert "LMCACHE_CONFIG_FILE" not in env
        assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == str(stage == "prefill").lower()
        events.append((stage, "start"))
        runner.write_json(log.parent / "output.json", {"completed": True})
        return NS(pid=1, returncode=0, stage=stage)

    def shell(command, **kwargs):
        assert command == [sys.executable, str(runner.LOCOMO_SCRIPT), "--vllm_port", "8000", "--vllm_ip", "127.0.0.1"]
        assert kwargs["check"] is True and kwargs["timeout"] == args.stage_timeout_seconds
        assert "127.0.0.1" in kwargs["env"]["no_proxy"]
        stage, last = events[-1]
        assert last == "ready"
        events.append((stage, "locomo"))
        if fail:
            raise subprocess.CalledProcessError(1, command)

    def seal(root):
        assert events[-1] == ("prefill", "exit") and (root / "prefill/process-exited.json").is_file()
        events.append(("prefill", "seal"))
        return {"passed": True}

    monkeypatch.setattr(runner, "start_logged_process", launch)
    monkeypatch.setattr(request, "wait_for_server", lambda proc, timeout: events.append((proc.stage, "ready")))
    monkeypatch.setattr(runner.subprocess, "run", shell)
    monkeypatch.setattr(runner, "finish_child", lambda proc: events.append((proc.stage, "exit")))
    monkeypatch.setattr(store, "seal_store", seal)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            runner.run_stages(args, tmp_path)
        assert events == [("baseline", phase) for phase in ("start", "ready", "locomo", "exit")]
    else:
        runner.run_stages(args, tmp_path)
        stages = ("prefill", "decode") if reuse else runner.STAGES
        expected = []
        for stage in stages:
            expected.extend((stage, phase) for phase in ("start", "ready", "locomo", "exit"))
            if stage == "prefill":
                expected.append((stage, "seal"))
        assert events == expected


@pytest.mark.parametrize("stage", runner.STAGES)
@pytest.mark.parametrize("delta", [False, True])
def test_http_request_uses_existing_probe_and_deterministic_generation(tmp_path, monkeypatch, stage, delta):
    directory = tmp_path / stage
    directory.mkdir()
    options = dict(tensor_parallel_size=1, max_model_len=1024, speculative_config=None)
    runner.write_json(directory / "engine_options.json", options)
    runner.write_json(tmp_path / "prompt.json", dict(token_ids=[1, 2, 3]))
    monkeypatch.setattr(runner, "load_worker_coverage", lambda *args: [{"rank": 0, "complete": True, "errors": []}])
    calls = []
    count = 1 if stage == "prefill" else 2
    kind = NS(name="DELTA" if delta else "CUMULATIVE")

    async def rpc(method, **kwargs):
        calls.append((method, kwargs))
        return []

    def sampling(limit, output_kind):
        assert limit == count and output_kind is kind
        return NS(output_kind=kind, max_tokens=limit)

    async def generate(client, prompt, params, req, **kwargs):
        assert prompt == {"prompt_token_ids": [1, 2, 3]} and params.max_tokens == count
        yield NS(finished=False, outputs=[NS(token_ids=[9], text="a")])
        tail = [10] if count == 2 else []
        yield NS(
            finished=True,
            num_cached_tokens=2 if stage == "decode" else 0,
            outputs=[NS(token_ids=tail if delta else [9, *tail], text="b" if delta else "ab", finish_reason="length")],
        )

    wrapped = request.request_entry(
        generate,
        root=tmp_path,
        stage=stage,
        output_tokens=count,
        timeout=1800,
        extract_tokens=lambda *_: [1, 2, 3],
        build_sampling=sampling,
    )

    async def run():
        async for _ in wrapped(NS(collective_rpc=rpc), {"prompt_token_ids": [1, 2, 3]}, NS(n=1, output_kind=kind), "r"):
            pass

    asyncio.run(run())
    assert [name for name, _ in calls] == [
        "install_file_probe",
        "flush_file_store",
        "finish_file_probe",
        "close_file_store",
    ]
    assert calls[0][1]["args"] == (str(directory), [1, 2, 3])
    output = json.loads((directory / "output.json").read_text())
    assert output["completed"] and len(output["token_ids"]) == count
    assert output["num_cached_tokens"] == (2 if stage == "decode" else 0)


def test_changed_locomo_prompt_never_reaches_model(tmp_path):
    (tmp_path / "decode").mkdir()
    runner.write_json(tmp_path / "decode/engine_options.json", {"max_model_len": 100})
    runner.write_json(tmp_path / "prompt.json", {"token_ids": [1, 2, 3]})
    cleanup = []

    async def rpc(method, **kwargs):
        cleanup.append(method)

    def fail(*args, **kwargs):
        pytest.fail("Changed input must not run the model")

    wrapped = request.request_entry(
        fail,
        root=tmp_path,
        stage="decode",
        output_tokens=2,
        timeout=1800,
        extract_tokens=lambda *_: [4, 5, 6],
        build_sampling=fail,
    )

    async def run():
        async for _ in wrapped(NS(collective_rpc=rpc), {}, NS(n=1), "r"):
            pass

    with pytest.raises(ValueError, match="tokenized prompt differs"):
        asyncio.run(run())
    assert cleanup == ["close_file_store"]
