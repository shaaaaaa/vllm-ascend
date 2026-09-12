# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only orchestration/timing tests; no mocked inference is an NPU pass."""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def driver(monkeypatch):
    tools = Path(__file__).resolve().parents[3] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("tested_sfa_graph_benchmark", tools / "sfa_graph_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    return SimpleNamespace(
        child="full",
        model=str(tmp_path),
        devices="0,1,2,3,4,5,6,7",
        prompt_tokens=4351,
        output_tokens=512,
        profile_tokens=32,
        warmups=1,
        repeats=5,
        profile=True,
        diagnose=False,
        profile_dir=tmp_path / "profile",
        run_dir=str(tmp_path),
        order="staged,full",
    )


def test_baseline_and_full_only_differ_in_production_switch(driver, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH", "0")
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "server.yaml")
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "1")
    staged = driver.benchmark_environment("staged", "0,1")
    full = driver.benchmark_environment("full", "0,1")
    assert {k for k in full if full[k] != staged[k]} == {"VLLM_ASCEND_SFA_FULL_GRAPH"}
    assert staged["VLLM_ASCEND_SFA_FULL_GRAPH"] == "0"
    assert full["VLLM_ASCEND_SFA_FULL_GRAPH"] == "1"
    assert full["VLLM_ASCEND_SFA_STAGED_GRAPH"] == "1"
    assert full["VLLM_ASCEND_MTP_DW_DEEP_DIAG"] == "0"
    assert "LMCACHE_CONFIG_FILE" not in full and "ASCEND_LAUNCH_BLOCKING" not in full
    assert full["LMCACHE_ENABLE_SHARED_CPU_CACHE"] == "false"
    assert full["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


@pytest.mark.parametrize("profile", [False, True])
def test_preserves_fixture_without_parity_worker_or_hooks(driver, args, profile):
    args.profile = profile
    options = driver.benchmark_options(args)
    assert options["worker_cls"].endswith("sfa_benchmark_worker.SFABenchmarkWorker")
    assert options["additional_config"] == {"sfa_benchmark": True}
    assert options["hf_overrides"] == {"num_hidden_layers": 8}
    assert options["tensor_parallel_size"] == 8 and options["data_parallel_size"] == 1
    assert options["load_format"] == "dummy" and options["quantization"] == "ascend"
    assert not options["enforce_eager"] and not options["async_scheduling"]
    assert options["speculative_config"] == {
        "num_speculative_tokens": 1,
        "method": "deepseek_mtp",
        "enforce_eager": True,
    }
    assert options["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
    assert options["compilation_config"]["pass_config"] == {"enable_sp": False}
    assert options["max_model_len"] > args.prompt_tokens + args.output_tokens
    assert options["max_num_batched_tokens"] == 512 and options["max_num_seqs"] == 1
    assert not options["enable_prefix_caching"] and not options["enable_expert_parallel"]
    assert ("profiler_config" in options) == profile
    if profile:
        assert options["profiler_config"]["ignore_frontend"]


def test_distinct_reproducible_prompts_do_not_share_prefix(driver):
    prompts = [driver.prompt_ids(4351, index) for index in range(8)]
    assert len({p[0] for p in prompts}) == 8
    assert all(len(p) == 4351 for p in prompts)
    assert prompts[3] == driver.prompt_ids(4351, 3)


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt_tokens": 4096},
        {"output_tokens": 1},
        {"profile_tokens": 1},
        {"warmups": -1},
        {"repeats": 0},
        {"devices": "0,0"},
        {"warmups": 256},
        {"model": "missing-sfa-model"},
    ],
)
def test_bad_config_fails_before_creating_directory_or_engine(driver, args, monkeypatch, changes):
    for key, value in changes.items():
        setattr(args, key, value)
    launch = Mock()
    monkeypatch.setattr(driver.subprocess, "run", launch)
    with pytest.raises((ValueError, FileNotFoundError)):
        driver.run_pair(args)
    launch.assert_not_called()
    assert not args.profile_dir.exists()


def test_tpot_excludes_prefill_and_handles_mtp_multi_token_emission(driver):
    result = driver.request_metrics([1, 2, 3, 4, 5], [(2, 10), (4, 11), (5, 13)], 0)
    assert result["ttft_ms"] == 10000
    assert result["decode_ms"] == 3000 and result["decode_tokens"] == 3
    assert result["tpot_ms"] == 1000 and result["decode_tokens_per_second"] == 1


@pytest.mark.parametrize("arrivals", [[], [(3, 1)], [(1, 1), (2, 2)], [(1, 2), (3, 1)], [(1, 1), (3, float("nan"))]])
def test_missing_or_invalid_timing_cannot_pass(driver, arrivals):
    with pytest.raises(RuntimeError):
        driver.request_metrics([1, 2, 3], arrivals, 0)


def fake_llm(monkeypatch, batches):
    params_type = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    vllm = ModuleType("vllm")
    vllm.SamplingParams = params_type
    sampling = ModuleType("vllm.sampling_params")
    sampling.RequestOutputKind = SimpleNamespace(CUMULATIVE="cumulative")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)

    class Engine:
        pending = False

        def add_request(self, request_id, prompt, params):
            self.pending = True
            self.params = params
            self.batches = iter(batches)
            self.external_id = request_id
            return request_id + "-internal-random-suffix"

        def has_unfinished_requests(self):
            return self.pending

        def step(self):
            current = next(self.batches)
            if isinstance(current, Exception):
                raise current
            if current is None:
                return []
            tokens, finished = current
            self.pending = not finished
            return [
                SimpleNamespace(
                    request_id=self.external_id,
                    finished=finished,
                    outputs=[SimpleNamespace(token_ids=tokens)],
                )
            ]

    return SimpleNamespace(llm_engine=Engine(), start_profile=Mock(), stop_profile=Mock(), collective_rpc=Mock())


def test_real_request_loop_times_stream_without_profiler_or_rpc(driver, args, monkeypatch):
    args.output_tokens = 5
    llm = fake_llm(monkeypatch, [None, ([1], False), ([1, 2, 3], False), ([1, 2, 3, 4, 5], True)])
    clock = iter([0, 1, 2, 4, 6])
    monkeypatch.setattr(driver.time, "perf_counter", lambda: next(clock))
    result = driver.generate_request(llm, args, 2)
    assert result["ttft_ms"] == 2000 and result["tpot_ms"] == 1000
    assert result["token_ids"] == [1, 2, 3, 4, 5]
    assert llm.llm_engine.params.output_kind == "cumulative"
    assert llm.llm_engine.params.ignore_eos and not llm.llm_engine.params.detokenize
    llm.start_profile.assert_not_called()
    llm.stop_profile.assert_not_called()
    llm.collective_rpc.assert_not_called()


def test_profile_starts_in_decode_and_never_returns_timing(driver, args, monkeypatch):
    args.profile_tokens = 4
    llm = fake_llm(monkeypatch, [([1], False), ([1] * 9, False), ([1] * 12, True)])
    result = driver.generate_request(llm, args, 2, profile=True)
    assert result == {"profiled": True, "output_tokens": 12}
    llm.start_profile.assert_called_once_with(profile_prefix="sfa_full")
    llm.stop_profile.assert_called_once_with()
    llm.collective_rpc.assert_not_called()


def test_failed_profiled_generation_stops_once_preserving_primary_error(driver, args, monkeypatch):
    llm = fake_llm(monkeypatch, [([1] * 9, False), RuntimeError("decode failed")])
    llm.stop_profile.side_effect = RuntimeError("stop failed")
    with pytest.raises(RuntimeError, match="decode failed") as error:
        driver.generate_request(llm, args, 2, profile=True)
    assert "stop failed" in error.value.__notes__[0]
    llm.stop_profile.assert_called_once_with()


@pytest.mark.parametrize(
    "batches,profile",
    [
        ([([1], True)], False),
        ([([1, 2], False), ([3, 2, 1], True)], False),
        ([([1] * 40, True)], True),
    ],
)
def test_early_stop_changed_prefix_and_empty_profile_fail(driver, args, monkeypatch, batches, profile):
    llm = fake_llm(monkeypatch, batches)
    with pytest.raises(RuntimeError):
        driver.generate_request(llm, args, 2, profile=profile)


def states(mode="full", count=0):
    return [
        {
            "rank": rank,
            "pid": 1000 + rank,
            "layers": 8,
            "staged": True,
            "full": mode == "full",
            "root_replays": count,
            "source_binding_updates": count // 5,
            "root_sealed": mode == "full",
            "root_keys": 1,
        }
        for rank in range(8)
    ]


@pytest.mark.parametrize("change", ["missing", "duplicate", "eager", "unsealed", "wrong-mode"])
def test_worker_state_never_accepts_wrong_engine(driver, args, change):
    reports = states()
    if change == "missing":
        reports.pop()
    elif change == "duplicate":
        reports[-1]["rank"] = 0
    elif change == "eager":
        reports[0]["staged"] = False
    elif change == "unsealed":
        reports[0]["root_sealed"] = False
    else:
        reports[0]["full"] = False
    with pytest.raises(RuntimeError):
        driver.worker_state(SimpleNamespace(collective_rpc=lambda *a, **kw: reports), args)


def test_replay_coverage_is_outside_requests_and_checks_every_rank(driver):
    assert driver.replay_delta(states(count=3), states(count=8), "full") == [5] * 8
    assert driver.replay_delta(states("staged"), states("staged"), "staged") == [0] * 8
    with pytest.raises(RuntimeError):
        driver.replay_delta(states(), states(), "full")
    with pytest.raises(RuntimeError):
        driver.replay_delta(states(), states(count=8), "staged")
    after = states(count=8)
    after[7]["root_replays"] = 7
    with pytest.raises(RuntimeError):
        driver.replay_delta(states(), after, "full")


def sample(tpot=10):
    return {"tpot_ms": tpot, "decode_tokens_per_second": 1000 / tpot, "token_ids": [1, 2, 3]}


def report(mode, values=(10, 20)):
    return {"mode": mode, "profiled_measurements": False, "samples": [sample(v) for v in values]}


def test_comparison_reports_gain_and_regression_without_threshold(driver):
    result = driver.compare_results(report("staged"), report("full", (5, 10)))
    assert result["tpot_reduction_percent"] == 50 and result["decode_speedup"] == 2
    assert result["tokens_equal"]
    assert result["staged_tpot_ms"]["std"] == 5
    assert driver.compare_results(report("staged"), report("full", (20, 40)))["tpot_reduction_percent"] == -100


@pytest.mark.parametrize("instrumented", [False, True])
def test_single_sample_statistics_are_explicit_about_count_variance_and_overhead(driver, instrumented):
    staged, full = report("staged", (10,)), report("full", (5,))
    for value in (staged, full):
        value["instrumented_measurements"] = instrumented
    result = driver.compare_results(staged, full)
    assert result["decode_speedup"] == 2 and result["tokens_equal"]
    assert result["staged_tpot_ms"] == {"count": 1, "mean": 10, "median": 10, "std": None, "min": 10, "max": 10}
    assert result["instrumented_measurements"] is instrumented
    assert ("includes diagnostic timing overhead" in result["scope"]) == instrumented


def test_comparison_rejects_mixed_timing_instrumentation(driver):
    staged, full = report("staged", (10,)), report("full", (5,))
    full["instrumented_measurements"] = True
    with pytest.raises(ValueError, match="instrumented and uninstrumented"):
        driver.compare_results(staged, full)


def test_changed_outputs_are_explicit_not_a_numerical_pass(driver):
    full = report("full")
    full["samples"][0]["token_ids"][1] = 99
    assert not driver.compare_results(report("staged"), full)["tokens_equal"]
    full["profiled_measurements"] = True
    with pytest.raises(ValueError):
        driver.compare_results(report("staged"), full)


def test_child_order_measurement_persistence_profile_and_cleanup(driver, args, monkeypatch):
    events = []
    llm = SimpleNamespace(collective_rpc=lambda *a, **kw: [{"rank": r["rank"], "pid": r["pid"]} for r in states()])
    stub = ModuleType("vllm")
    stub.LLM = lambda **kw: llm
    monkeypatch.setitem(sys.modules, "vllm", stub)
    counters = iter([0] + [value for i in range(5) for value in (i * 5, (i + 1) * 5)])
    monkeypatch.setattr(driver, "worker_state", lambda *a: states(count=next(counters)))
    monkeypatch.setattr(driver, "track_workers", lambda *a: [])
    monkeypatch.setattr(driver, "shutdown_engine", lambda *a: events.append("shutdown"))

    def generate(llm, args, ordinal, *, profile=False):
        events.append((ordinal, profile))
        if profile:
            saved = json.loads(Path(args.run_dir, "full.json").read_text())
            assert len(saved["samples"]) == 5 and not saved["profiled_measurements"]
            assert all(s["source_binding_updates_per_rank"] == [1] * 8 for s in saved["samples"])
        return sample()

    monkeypatch.setattr(driver, "generate_request", generate)
    driver.run_child(args)
    assert events == [(i, False) for i in range(6)] + [(6, True), "shutdown"]


def test_child_failure_always_shuts_engine_down(driver, args, monkeypatch):
    stub = ModuleType("vllm")
    llm = Mock()
    stub.LLM = lambda **kwargs: llm
    monkeypatch.setitem(sys.modules, "vllm", stub)
    monkeypatch.setattr(driver, "track_workers", lambda *a: ["owned worker"])
    monkeypatch.setattr(driver, "worker_state", Mock(side_effect=RuntimeError("bad state")))
    shutdown = Mock()
    monkeypatch.setattr(driver, "shutdown_engine", shutdown)
    with pytest.raises(RuntimeError, match="bad state"):
        driver.run_child(args)
    shutdown.assert_called_once_with(llm, ["owned worker"])


@pytest.mark.parametrize("order", ["staged,full", "full,staged"])
def test_parent_sequences_engines_then_analysis_and_keeps_previous_runs(driver, args, monkeypatch, order):
    args.order = order
    events = []
    args.profile_dir.mkdir()
    old = args.profile_dir / "old.txt"
    old.write_text("keep me")

    def launch(argv, **kwargs):
        if "--child" not in argv:
            assert events[:3] == ["preflight", *order.split(",")]
            events.append("analysis")
            return
        mode = argv[argv.index("--child") + 1]
        events.append(mode)
        if mode != "preflight":
            Path(args.run_dir, f"{mode}.json").write_text(json.dumps(report(mode)))

    monkeypatch.setattr(driver.subprocess, "run", launch)
    monkeypatch.setattr(driver, "analyse_traces", lambda *a: {"status": "UNVERIFIED", "ranks": 8})
    driver.run_pair(args)
    assert events == ["preflight", *order.split(","), "analysis", "analysis"]
    assert old.read_text() == "keep me"
    assert Path(args.run_dir, "comparison.json").is_file()


@pytest.mark.parametrize("failure", [1, 2, 3])
def test_parent_never_continues_after_preflight_or_engine_failure(driver, args, monkeypatch, failure):
    launch = Mock(side_effect=[None] * (failure - 1) + [subprocess.CalledProcessError(1, "child")])
    monkeypatch.setattr(driver.subprocess, "run", launch)
    with pytest.raises(subprocess.CalledProcessError):
        driver.run_pair(args)
    assert launch.call_count == failure


def test_worker_uses_only_fixture_loading_not_parity_execution(monkeypatch):
    events = []

    class Dummy:
        def load_weights(self, model, config):
            events.append("original weights")

    class NPU:
        def load_model(self):
            Dummy().load_weights(None, None)
            events.append("production load")

    class Parity:
        def load_model(self):
            raise AssertionError("Parity hooks must never be installed")

        def _prepare_quant_config(self):
            events.append("quant remap")

    def dummy_load(original, loader, model, config):
        original(loader, model, config)
        events.append("integer weights")

    modules = {
        "vllm": {},
        "vllm.distributed": {"get_tp_group": lambda: None},
        "vllm.model_executor.model_loader.dummy_loader": {"DummyModelLoader": Dummy},
        "vllm_ascend": {"envs": SimpleNamespace()},
        "vllm_ascend.attention.sfa_parity": {"coordinated_check": lambda check, **kw: check()},
        "vllm_ascend.worker.sfa_parity_worker": {"SFAParityWorker": Parity, "deterministic_dummy_load": dummy_load},
        "vllm_ascend.worker.worker": {"NPUWorker": NPU},
    }
    for name, attributes in modules.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/sfa_benchmark_worker.py"
    spec = importlib.util.spec_from_file_location("tested_benchmark_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.SFABenchmarkWorker()
    worker.vllm_config = SimpleNamespace(
        additional_config={"sfa_benchmark": True},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8, data_parallel_size=1, pipeline_parallel_size=1, enable_expert_parallel=False
        ),
    )
    original = Dummy.load_weights
    worker.load_model()
    assert events == ["quant remap", "original weights", "integer weights", "production load"]
    assert Dummy.load_weights is original
    assert not isinstance(worker, Parity)
    assert set(module.SFABenchmarkWorker.__dict__) >= {"load_model", "benchmark_state", "shutdown"}
    assert "execute_model" not in module.SFABenchmarkWorker.__dict__


def test_actual_sibling_vllm_request_id_contract():
    """Execute real ID/output methods on CPU; don't invent their return IDs."""
    root = Path(__file__).resolve().parents[4] / "vllm/vllm/v1/engine"
    if not root.is_dir():
        pytest.skip("Requires the matching sibling vllm checkout")

    def method(file, cls, name, namespace):
        tree = ast.parse((root / file).read_text(encoding="utf-8"))
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
        function = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
        function.decorator_list = []
        module = ast.parse("from __future__ import annotations")
        module.body.append(function)
        exec(compile(ast.fix_missing_locations(module), str(root / file), "exec"), namespace)
        return namespace[name]

    assign = method(
        "input_processor.py",
        "InputProcessor",
        "assign_request_id",
        {
            "envs": SimpleNamespace(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False),
            "random_uuid": lambda: "abcdefgh-more-random-data",
        },
    )
    request = SimpleNamespace(request_id="sfa-benchmark-2", external_req_id=None)
    assign(request)
    assert request.request_id == "sfa-benchmark-2-abcdefgh"
    assert request.external_req_id == "sfa-benchmark-2"

    make_output = method(
        "output_processor.py",
        "RequestState",
        "_new_request_output",
        {
            "PoolingOutput": type("PoolingOutput", (), {}),
            "RequestOutput": SimpleNamespace,
            "RequestOutputKind": SimpleNamespace(DELTA="delta"),
            "CompletionOutput": object,
            "cast": lambda typ, value: value,
        },
    )
    state = SimpleNamespace(
        prompt_token_ids=[100],
        prompt_embeds=None,
        prompt=None,
        logprobs_processor=SimpleNamespace(prompt_logprobs=None),
        output_kind="cumulative",
        lora_request=None,
        num_cached_tokens=0,
        stats=None,
    )
    output = make_output(state, request.external_req_id, [SimpleNamespace(token_ids=[1])], False)
    assert output.request_id == "sfa-benchmark-2"
    assert output.request_id != request.request_id


def test_default_long_context_cli_and_options(driver, args, monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "argv", ["sfa_graph_benchmark.py"])
    monkeypatch.setattr(driver, "run_pair", lambda options: seen.append(options))
    driver.main()
    assert seen[0].prompt_tokens == 30000 and seen[0].output_tokens == 512
    assert seen[0].warmups == 0 and seen[0].repeats == 1
    assert not seen[0].diagnose and not seen[0].profile
    args.prompt_tokens = seen[0].prompt_tokens
    driver.validate_args(args)
    options = driver.benchmark_options(args)
    assert options["max_model_len"] == 30544
    assert options["max_num_batched_tokens"] == 512  # Keep bounded prefill memory.
    assert len(driver.prompt_ids(args.prompt_tokens, 1)) == 30000


def test_short_context_remains_an_explicit_cli_override(driver, monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "argv", ["sfa_graph_benchmark.py", "--prompt-tokens", "4351"])
    monkeypatch.setattr(driver, "run_pair", lambda options: seen.append(options))
    driver.main()
    assert seen[0].prompt_tokens == 4351


def diagnostic_workers(mode="full", steps=3):
    metrics = {"count": steps, "total_ms": 12.0, "mean_ms": 4.0, "std_ms": 1.0, "max_ms": 6.0}
    names = ["worker.execute", "worker.sample", "target.forward"]
    names.extend(f"metadata.L{i}" if mode == "full" else f"retrieve.L{i}" for i in range(8))
    if mode == "full":
        names.append("root.replay_submit")
        names.append("signature.validate")
    return [
        dict(
            rank=rank,
            decode_steps=steps,
            root_replays=steps if mode == "full" else 0,
            source_binding_updates=1,
            query_tokens_histogram={2: steps},
            sampled_tokens_histogram={2: steps},
            prefill_steps_excluded=10,
            device_intervals_dropped=0,
            stages={name: {key: dict(metrics) for key in ("wall", "self_wall", "self_cpu")} for name in names},
        )
        for rank in range(8)
    ]


def test_diagnose_has_no_mid_request_rpcs_or_profiler(driver, args, monkeypatch):
    args.profile, args.diagnose = False, True
    events = []

    def rpc(method, **kwargs):
        events.append(method)
        if method == "benchmark_start_decode_timing":
            assert kwargs["args"] == (4351,)
        return diagnostic_workers()

    def generate(*a):
        events.append("request")
        return {"decode_ms": 80, "decode_tokens": 6, "tpot_ms": 80 / 6}

    llm = SimpleNamespace(collective_rpc=rpc)
    monkeypatch.setattr(driver, "generate_request", generate)
    result = driver.diagnose_request(llm, args, 6)
    assert events == ["benchmark_start_decode_timing", "request", "benchmark_stop_decode_timing"]
    assert len(result["workers"]) == 8
    assert "no profiler" in result["scope"]
    assert "profiler_config" not in driver.benchmark_options(args)


def test_diagnostic_generation_error_attempts_stop_and_preserves_original(driver, args, monkeypatch):
    llm = SimpleNamespace(collective_rpc=Mock(side_effect=[None, RuntimeError("stop failed")]))
    monkeypatch.setattr(driver, "generate_request", Mock(side_effect=ValueError("original failure")))
    with pytest.raises(ValueError, match="original failure") as error:
        driver.diagnose_request(llm, args, 6)
    assert "stop failed" in error.value.__notes__[0]
    assert llm.collective_rpc.call_count == 2


@pytest.mark.parametrize(
    "failure",
    ["missing_rank", "duplicate_rank", "missing_target", "missing_root", "rank_skew", "missing_layer", "callback"],
)
def test_diagnose_rejects_incomplete_coverage(driver, args, monkeypatch, failure):
    reports = diagnostic_workers()
    if failure == "missing_rank":
        reports.pop()
    elif failure == "duplicate_rank":
        reports[-1]["rank"] = 0
    elif failure == "missing_target":
        reports[-1]["stages"].pop("target.forward")
    elif failure == "missing_root":
        reports[-1]["root_replays"] = 0
    elif failure == "missing_layer":
        reports[-1]["stages"].pop("metadata.L7")
    elif failure == "callback":
        reports[-1]["stages"]["retrieve.L3"] = {"wall": {"count": 1}}
    else:
        reports[-1] = diagnostic_workers(steps=4)[-1]
    llm = SimpleNamespace(collective_rpc=Mock(side_effect=[None, reports]))
    monkeypatch.setattr(driver, "generate_request", lambda *a: sample())
    with pytest.raises(RuntimeError):
        driver.diagnose_request(llm, args, 6)


def test_diagnostic_log_is_compact_includes_all_ranks_and_no_tokens(driver, capsys):
    driver.print_decode_timing(
        {
            "mode": "full",
            "workers": diagnostic_workers(),
            "request": {"decode_ms": 80, "decode_tokens": 6, "token_ids": [999999] * 512},
        }
    )
    output = capsys.readouterr().out
    assert len(output) < 5000
    assert "999999" not in output
    assert all(f"rank={rank} " in output for rank in range(8))
    assert "wall=4.000(4.000) self=4.000(4.000)" in output
    assert "committed/forward=2.000" in output
    assert "MUST NOT be added" in output
    assert "not pure scheduler" in output


def test_diagnose_and_profile_are_mutually_exclusive(driver, args, monkeypatch):
    args.profile = args.diagnose = True
    with pytest.raises(ValueError, match="OR --profile"):
        driver.validate_args(args)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--diagnose", "--profile"])
    with pytest.raises(SystemExit):
        driver.main()


def test_diagnose_cli_never_runs_trace_export_or_analysis(driver, args, monkeypatch):
    args.profile, args.diagnose = False, True
    launches = []

    def launch(argv, **kwargs):
        assert "--child" in argv and "--diagnose" in argv and "--profile" not in argv
        mode = argv[argv.index("--child") + 1]
        launches.append(mode)
        if mode != "preflight":
            Path(args.run_dir, f"{mode}.json").write_text(json.dumps(report(mode)))

    monkeypatch.setattr(driver.subprocess, "run", launch)
    monkeypatch.setattr(driver, "analyse_traces", Mock(side_effect=AssertionError("No profiler analysis")))
    driver.run_pair(args)
    assert launches == ["preflight", "staged", "full"]


def test_diagnostics_installed_only_after_all_performance_samples(driver, args, monkeypatch):
    args.profile, args.diagnose = False, True
    events = []
    stub = ModuleType("vllm")
    stub.LLM = lambda **kw: SimpleNamespace(
        collective_rpc=lambda *a, **kw: [{"rank": r["rank"], "pid": r["pid"]} for r in states()]
    )
    monkeypatch.setitem(sys.modules, "vllm", stub)
    counters = iter([0] + [value for i in range(5) for value in (i * 5, (i + 1) * 5)])
    monkeypatch.setattr(driver, "worker_state", lambda *a: states(count=next(counters)))
    monkeypatch.setattr(driver, "track_workers", lambda *a: [])
    monkeypatch.setattr(driver, "shutdown_engine", lambda *a: events.append("shutdown"))
    monkeypatch.setattr(driver, "generate_request", lambda *a: (events.append("performance"), sample())[1])

    def diagnose(*a):
        saved = json.loads(Path(args.run_dir, "full.json").read_text())
        assert len(saved["samples"]) == 5
        assert events == ["performance"] * 6
        events.append("diagnose")
        return {"diagnostic_only": True}

    monkeypatch.setattr(driver, "diagnose_request", diagnose)
    monkeypatch.setattr(driver, "print_decode_timing", lambda *a: events.append("print"))
    driver.run_child(args)
    assert events == ["performance"] * 6 + ["diagnose", "print", "shutdown"]
    assert Path(args.run_dir, "full-timing.json").is_file()


@pytest.mark.parametrize("mode", ["staged", "full"])
@pytest.mark.parametrize("diagnose", [False, True])
def test_single_request_mode_never_generates_hidden_warmup_or_extra_diagnostics(
    driver, args, monkeypatch, capsys, mode, diagnose
):
    args.child, args.warmups, args.repeats = mode, 0, 1
    args.profile, args.diagnose = False, diagnose
    driver.validate_args(args)
    events, generations = [], []

    def rpc(name, **kwargs):
        events.append(name)
        if name == "benchmark_stop_decode_timing":
            return diagnostic_workers(mode)
        return [{"rank": r["rank"], "pid": r["pid"]} for r in states(mode)]

    stub = ModuleType("vllm")
    llm = SimpleNamespace(collective_rpc=rpc)
    stub.LLM = lambda **kw: llm
    monkeypatch.setitem(sys.modules, "vllm", stub)
    monkeypatch.setattr(driver, "track_workers", lambda *a: [])
    monkeypatch.setattr(
        driver, "worker_state", lambda *a: states(mode, count=3 * len(generations) if mode == "full" else 0)
    )
    monkeypatch.setattr(driver, "shutdown_engine", lambda *a: events.append("shutdown"))

    def generate(llm, args, ordinal, *, profile=False):
        assert not profile
        generations.append(ordinal)
        events.append("request")
        return {**sample(), "decode_ms": 20, "decode_tokens": 2}

    monkeypatch.setattr(driver, "generate_request", generate)
    # Execute the actual run_child -> diagnose_request -> generate_request
    # orchestration. One inline request must produce BOTH reports, not two runs.
    driver.run_child(args)
    assert generations == [0]
    expected = ["benchmark_process_info"]
    expected += (
        ["benchmark_start_decode_timing", "request", "benchmark_stop_decode_timing"] if diagnose else ["request"]
    )
    expected += ["benchmark_release_resources", "shutdown"]
    assert events == expected
    saved = json.loads(Path(args.run_dir, f"{mode}.json").read_text())
    assert len(saved["samples"]) == 1
    assert saved["tpot_ms"]["std"] is None and saved["tpot_ms"]["count"] == 1
    assert saved["instrumented_measurements"] is diagnose
    timing = Path(args.run_dir, f"{mode}-timing.json")
    assert timing.exists() == diagnose
    if diagnose:
        diagnostic = json.loads(timing.read_text())
        assert len(diagnostic["workers"]) == 8
        assert "single instrumented" in diagnostic["scope"]
        assert diagnostic["request"]["token_ids"] == saved["samples"][0]["token_ids"]
    output = capsys.readouterr().out
    assert f"[SFA_BENCH] {mode} 1/1" in output
    assert ("[SFA_TIMING]" in output) == diagnose


@pytest.mark.parametrize("diagnose", [False, True])
def test_single_request_parent_prints_and_saves_comparison(driver, args, monkeypatch, capsys, diagnose):
    args.warmups, args.repeats, args.profile, args.diagnose = 0, 1, False, diagnose
    launches = []

    def launch(argv, **kwargs):
        mode = argv[argv.index("--child") + 1]
        launches.append(mode)
        assert argv[argv.index("--warmups") + 1] == "0"
        assert argv[argv.index("--repeats") + 1] == "1"
        assert ("--diagnose" in argv) == diagnose
        if mode != "preflight":
            value = report(mode, (10 if mode == "staged" else 5,))
            value["instrumented_measurements"] = diagnose
            Path(args.run_dir, f"{mode}.json").write_text(json.dumps(value))

    monkeypatch.setattr(driver.subprocess, "run", launch)
    monkeypatch.setattr(driver, "analyse_traces", Mock(side_effect=AssertionError("No profile")))
    driver.run_pair(args)
    assert launches == ["preflight", "staged", "full"]
    result = json.loads(Path(args.run_dir, "comparison.json").read_text())
    assert result["config"]["repeats"] == 1 and result["config"]["warmups"] == 0
    assert result["instrumented_measurements"] is diagnose
    output = capsys.readouterr().out
    assert "std=n/a (one request)" in output and "speedup=2.000x" in output
    assert ("TPOT INCLUDES diagnostic overhead" in output) == diagnose
