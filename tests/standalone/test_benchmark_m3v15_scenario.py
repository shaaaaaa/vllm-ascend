# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of input identity, queued bursts and worker observations."""

import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[2] / "tools/benchmark_m3v15_scenario.py"
    spec = importlib.util.spec_from_file_location("benchmark_m3v15_scenario", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(tool, tmp_path):
    tokens = list(range(4096))
    tool.write_json(tmp_path / "tokens.json", tokens)
    manifest = {
        "synthetic_padding_tokens": False,
        "sources": [
            {
                "task_name": "real",
                "token_file": "tokens.json",
                "glm52_tokens": len(tokens),
                "tokens_sha256_uint32_le": tool.token_digest(tokens),
            }
        ],
        "cases": [
            {
                "history": 256,
                "query": 128,
                "total_query_tokens": 2048,
                "requests": 2,
                "windows": [
                    {
                        "task": "real",
                        "start_token": start,
                        "token_sha256": tool.token_digest(tokens[start : start + 384]),
                    }
                    for start in (0, 1024)
                ],
            }
        ],
    }
    tool.write_json(tmp_path / "manifest.json", manifest)
    output = tmp_path / "inputs.json"
    args = SimpleNamespace(
        corpus=tmp_path / "manifest.json", tokens_dir=None, output=output, history=256, query=128, requests=2, rounds=3
    )
    tool.prepare_inputs(args)
    return tool.read_json(output), output


def test_reference_windows_and_followup_order(tool, prepared):
    data, _ = prepared
    assert data["requests"][0]["history"] == list(range(256))
    assert [row[:2] for row in data["requests"][0]["followups"]] == [[256, 257], [384, 385], [512, 513]]
    assert data["requests"][1]["history"] == list(range(1024, 1280))
    tool.validate_inputs(data)
    changed = copy.deepcopy(data)
    changed["requests"][0]["followups"][1][0] += 1
    assert tool.input_summary(changed) != tool.input_summary(data)


def test_manifest_hash_checked_before_export(tool, prepared, tmp_path):
    _, _ = prepared
    values = tool.read_json(tmp_path / "tokens.json")
    values[0] += 1
    tool.write_json(tmp_path / "tokens.json", values)
    args = SimpleNamespace(
        corpus=tmp_path / "manifest.json",
        tokens_dir=None,
        output=tmp_path / "bad.json",
        history=256,
        query=128,
        requests=2,
        rounds=2,
    )
    with pytest.raises(ValueError, match="hash/length"):
        tool.prepare_inputs(args)
    assert not args.output.exists()


@pytest.mark.parametrize("mutation", ["duplicate_prefix", "duplicate_suffix", "short_suffix", "negative_token"])
def test_bad_inputs_rejected(tool, prepared, mutation):
    data = copy.deepcopy(prepared[0])
    if mutation == "duplicate_prefix":
        data["requests"][1]["history"] = data["requests"][0]["history"]
    elif mutation == "duplicate_suffix":
        data["requests"][0]["followups"][1] = data["requests"][0]["followups"][0]
    elif mutation == "short_suffix":
        data["requests"][0]["followups"][0].pop()
    else:
        data["requests"][0]["history"][0] = -1
    with pytest.raises(ValueError):
        tool.validate_inputs(data)


def engine_config(model):
    return {
        "model": str(model),
        "tensor_parallel_size": 1,
        "data_parallel_size": 2,
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "block_size": 128,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 6144,
        "max_model_len": 1024,
        "async_scheduling": False,
        "kv_transfer_config": {"kv_connector": "TestConnector", "kv_role": "kv_both"},
    }


class FakeEngine:
    def __init__(self, bad_history=False):
        self.paused = False
        self.pending = []
        self.calls = []
        self.bad_history = bad_history

    async def generate(self, prompt, params, request_id, data_parallel_rank):
        assert len(prompt["prompt_token_ids"]) == 257
        assert params.max_tokens == 1
        self.calls.append(("seed", data_parallel_rank))
        yield SimpleNamespace(finished=True)

    async def pause_generation(self, **kwargs):
        assert kwargs["clear_cache"] is False
        self.paused = True
        self.calls.append(("pause",))

    async def resume_generation(self):
        assert len(self.pending) == 2
        self.paused = False
        self.calls.append(("resume",))
        for future in self.pending:
            future.set_result(
                SimpleNamespace(
                    finished=True,
                    outputs=[SimpleNamespace(token_ids=[77], logprobs=[{77: SimpleNamespace(logprob=-0.5)}])],
                )
            )
        self.pending.clear()

    async def add_request(self, request_id, prompt, params, data_parallel_rank):
        assert self.paused
        assert len(prompt["prompt_token_ids"]) == 384
        self.calls.append(("add", data_parallel_rank))
        future = asyncio.get_running_loop().create_future()
        self.pending.append(future)
        return SimpleNamespace(get=lambda: future)

    async def collective_rpc(self, method, timeout, args=()):
        self.calls.append((method,))
        rows = [{"dp_rank": rank, "tp_rank": 0} for rank in range(2)]
        if method == "cpb_collect":
            for row in rows:
                row.update(
                    steps=[{"H": [255 if self.bad_history else 256], "Q": [128]}],
                    model_forward_ms=10 + row["dp_rank"],
                    allocated_peak_bytes=123,
                )
        return rows

    def shutdown(self):
        self.calls.append(("shutdown",))


def install_fake_vllm(monkeypatch, engine):
    modules = {
        name: ModuleType(name)
        for name in (
            "vllm",
            "vllm.config",
            "vllm.engine",
            "vllm.engine.arg_utils",
            "vllm.v1",
            "vllm.v1.engine",
            "vllm.v1.engine.async_llm",
        )
    }
    modules["vllm"].SamplingParams = SimpleNamespace
    modules["vllm.config"].KVTransferConfig = SimpleNamespace
    modules["vllm.engine.arg_utils"].AsyncEngineArgs = SimpleNamespace

    def factory(args):
        assert args.worker_extension_cls.endswith(".CachedPrefillObserver")
        assert args.kv_transfer_config.kv_connector == "TestConnector"
        return engine

    modules["vllm.v1.engine.async_llm"].AsyncLLM = SimpleNamespace(from_engine_args=factory)
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)


@pytest.mark.parametrize("bad_history", [False, True])
def test_full_run_cpu(tool, prepared, tmp_path, monkeypatch, bad_history):
    model = tmp_path / "model"
    model.mkdir()
    tool.write_json(model / "config.json", {"model_type": "test", "num_hidden_layers": 78})
    tool.write_json(tmp_path / "engine.json", engine_config(model))
    engine = FakeEngine(bad_history)
    install_fake_vllm(monkeypatch, engine)
    monkeypatch.setenv("PYTHONPATH", "")
    args = SimpleNamespace(
        inputs=prepared[1],
        engine_args=tmp_path / "engine.json",
        model=None,
        allow_topology_change=True,
        warmups=1,
        output=tmp_path / "run",
        profile=False,
        label="cpu-test",
        timeout=3,
        settle_seconds=0,
    )
    if bad_history:
        with pytest.raises(ValueError, match="Actual worker H/Q"):
            asyncio.run(tool.run(args))
    else:
        asyncio.run(tool.run(args))
    result = tool.read_json(args.output / "result.json")
    assert result["status"] == ("failed" if bad_history else "complete")
    assert engine.calls[-1] == ("shutdown",)
    assert sum(call[0] == "resume" for call in engine.calls) == (1 if bad_history else 3)
    if not bad_history:
        assert [sample["max_rank_forward_ms"] for sample in result["samples"]] == [11, 11, 11]
        assert tool.main(["compare", str(args.output / "result.json"), str(args.output / "result.json")]) == 0
        changed = copy.deepcopy(result)
        changed["samples"][1]["completions"][0]["logprobs"]["77"] = -0.6
        tool.write_json(tmp_path / "changed.json", changed)
        assert tool.main(["compare", str(args.output / "result.json"), str(tmp_path / "changed.json")]) == 1
        changed["inputs"]["sha256"] = "different"
        tool.write_json(tmp_path / "changed.json", changed)
        with pytest.raises(ValueError, match="mismatch"):
            tool.main(["compare", str(args.output / "result.json"), str(tmp_path / "changed.json")])


def test_reference_config_and_rank_coverage(tool, prepared):
    config = engine_config("/model")
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        tool.validate_engine(config, prepared[0], False)
    config["max_num_batched_tokens"] = 127
    with pytest.raises(ValueError, match="Token budget"):
        tool.validate_engine(config, prepared[0], True)
    with pytest.raises(ValueError, match="distinct"):
        tool.validate_ranks([{"dp_rank": 0, "tp_rank": 0}] * 2, 1, 2)


@pytest.mark.parametrize("profile", [False, True])
def test_observer_records_real_forward_and_excludes_dummy(tool, monkeypatch, tmp_path, profile):
    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 2.5

    npu = SimpleNamespace(
        Event=Event, synchronize=lambda: None, reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 123
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=npu))
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", SimpleNamespace(get_tp_group=lambda: SimpleNamespace(rank_in_group=0))
    )
    profile_calls = []
    profiler = SimpleNamespace(start=lambda: profile_calls.append("start"), stop=lambda: profile_calls.append("stop"))
    profile_module = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        tensorboard_trace_handler=lambda *args, **kwargs: None,
        profile=lambda **kwargs: profiler,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profile_module))
    monkeypatch.setitem(sys.modules, "torch_npu.profiler", profile_module)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.profiler.experimental_config",
        SimpleNamespace(_ExperimentalConfig=lambda **kwargs: None, ProfilerLevel=SimpleNamespace(Level2=2)),
    )
    observer = tool.CachedPrefillObserver()
    observer.parallel_config = SimpleNamespace(data_parallel_rank=0)
    runner = SimpleNamespace()
    runner.model = SimpleNamespace(forward=lambda: "output")
    runner._dummy_run = lambda: runner.model.forward()
    runner.input_batch = SimpleNamespace(num_reqs=1, num_computed_tokens_cpu=SimpleNamespace(__getitem__=None))

    class Histories(list):
        def __getitem__(self, key):
            values = super().__getitem__(key)
            return SimpleNamespace(tolist=lambda: values)

    runner.input_batch.num_computed_tokens_cpu = Histories([256])
    runner.query_start_loc = SimpleNamespace(np=[0, 128])
    observer.model_runner = runner
    observer.cpb_install()
    observer.cpb_begin("measure", str(tmp_path / "profile") if profile else None)
    assert runner._dummy_run() == "output"
    assert runner.model.forward() == "output"
    result = observer.cpb_collect()
    assert len(result["steps"]) == 1
    assert result["steps"][0]["H"] == [256]
    assert result["steps"][0]["Q"] == [128]
    assert result["model_forward_ms"] == 2.5
    assert profile_calls == (["start", "stop"] if profile else [])


def test_installed_c8_patches_load_before_engineargs(tool, monkeypatch, tmp_path):
    root = tmp_path / "vllm_ascend"
    platform = root / "patch" / "platform"
    platform.mkdir(parents=True)
    for name in ("patch_kv_cache_dtype", "patch_indexer_kv_dtype"):
        (platform / f"{name}.py").write_text("initialized = True\n")
    original = tool.importlib.util.find_spec
    monkeypatch.setattr(
        tool.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=[str(root)]) if name == "vllm_ascend" else original(name)
        ),
    )
    config = {"kv_cache_dtype": "int8", "attention_config": {"indexer_kv_dtype": "int8"}}
    receipts = tool.load_dtype_schemas(config)
    assert len(receipts) == 2
    assert sys.modules["_cached_prefill_patch_kv_cache_dtype"].initialized
    assert sys.modules["_cached_prefill_patch_indexer_kv_dtype"].initialized
    assert tool.load_dtype_schemas({"kv_cache_dtype": "auto"}) == []
