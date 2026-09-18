# SPDX-License-Identifier: Apache-2.0
"""CPU-only orchestration tests; real Ascend profiling requires the server."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    return importlib.import_module("layerwise_prefill_profile")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "return_dict": False}
        # Mapping/nested-list form exercises the former len(BatchEncoding)==2 bug.
        ids = [1, *[ord(c) for c in messages[0]["content"]], 2, 3]
        return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}


@pytest.mark.parametrize("length", [8192, 10000])
def test_prompt_repeats_article_and_preserves_chat_markers(tool, length):
    text, ids = tool.build_prompt(Tokenizer(), "Example article.\n" * 320, length)
    assert len(ids) == length
    assert ids[0] == 1 and ids[-2:] == [2, 3]
    assert ids[1:-2] == [ord(c) for c in text]
    assert text.startswith("Example article.\n")


def test_prompt_can_crop_longer_source(tool):
    text, ids = tool.build_prompt(Tokenizer(), "Report " * 10000, 10000)
    assert len(ids) == 10000
    assert text == ("Report " * 10000)[:9997]


def test_short_pair_shares_one_saved_input(tool, monkeypatch, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Example article.\n" * 320, encoding="utf-8")
    args = tool.parser().parse_args(["--prompt-file", str(source)])
    monkeypatch.setitem(sys.modules, "transformers", NS(AutoTokenizer=NS(from_pretrained=lambda *a, **kw: Tokenizer())))
    tool.prepare_inputs(args, tmp_path, tool.CASES)
    assert sorted(p.name for p in tmp_path.glob("*_prompt.json")) == ["10k_prompt.json"]
    for name, count in (("10k", 10000),):
        prompt = json.loads((tmp_path / f"{name}_prompt.json").read_text())
        assert prompt["length"] == len(prompt["token_ids"]) == count
        text = (tmp_path / f"{name}_input.txt").read_text(encoding="utf-8")
        assert prompt["token_ids"] == [1, *[ord(c) for c in text], 2, 3]


def test_off_on_environment_diff_is_only_feature_switch(tool, monkeypatch):
    for key in ("LMCACHE_CONFIG_FILE", "LMCACHE_REMOTE_URL", "LMCACHE_EXTRA_CONFIG", "VLLM_PREFILL_CHECK_TRACE_DIR"):
        monkeypatch.setenv(key, "inherited-debug-config")
    args = tool.parser().parse_args([])
    off = tool.case_environment(args, "10k_off")
    on = tool.case_environment(args, "10k_on")
    assert {k for k in off if off[k] != on[k]} == {"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"}
    assert off["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == "false"
    assert on["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == "true"
    assert "LMCACHE_CONFIG_FILE" not in on and "LMCACHE_REMOTE_URL" not in on
    assert "VLLM_PREFILL_CHECK_TRACE_DIR" not in on
    assert json.loads(on["LMCACHE_EXTRA_CONFIG"]) == {"save_only_first_rank": True}
    assert on["MSMONITOR_USE_DAEMON"] == "0"


@pytest.mark.parametrize("prompt_len, expected_max", [(9999, 16384), (10000, 16384)])
def test_full_model_mtp_and_profile_options(tool, tmp_path, prompt_len, expected_max):
    args = tool.parser().parse_args([])
    options = tool.engine_options(args, tmp_path, prompt_len)
    assert "hf_overrides" not in options and options["load_format"] == "safetensors"
    assert "worker_extension_cls" not in options
    assert options["max_model_len"] == expected_max
    assert options["gpu_memory_utilization"] == 0.96
    assert options["tensor_parallel_size"] == 8
    assert options["max_num_seqs"] == 1 and options["max_num_batched_tokens"] == 4096
    assert options["enable_prefix_caching"] is False and options["enforce_eager"] is True
    assert options["speculative_config"]["num_speculative_tokens"] == 1
    assert options["kv_transfer_config"]["kv_role"] == "kv_producer"
    config = options["profiler_config"]
    assert config["profiler"] == "torch" and config["ignore_frontend"] is True
    assert config["torch_profiler_dir"] == str((tmp_path / "profile").resolve())
    assert not config["torch_profiler_with_stack"] and not config["torch_profiler_with_memory"]


@pytest.mark.parametrize("generate_fails, stop_fails", [(False, False), (True, False), (True, True), (False, True)])
def test_capture_brackets_whole_generate_and_preserves_error(tool, generate_fails, stop_fails):
    events = []

    def start_profile(**kwargs):
        events.append(("start", kwargs))

    def generate(prompt, params, **kwargs):
        assert prompt == {"prompt_token_ids": [1, 8, 2, 3]}
        assert params == "params" and kwargs == {"use_tqdm": False}
        events.append("generate")
        if generate_fails:
            raise ValueError("model failed")
        return ["output"]

    def stop_profile():
        events.append("stop")
        if stop_fails:
            raise RuntimeError("stop failed")

    llm = NS(start_profile=start_profile, generate=generate, stop_profile=stop_profile)
    if generate_fails:
        with pytest.raises(ValueError, match="model failed"):
            tool.capture_request(llm, [1, 8, 2, 3], "params", "10k_on")
    elif stop_fails:
        with pytest.raises(RuntimeError, match="stop failed"):
            tool.capture_request(llm, [1, 8, 2, 3], "params", "10k_on")
    else:
        output, elapsed = tool.capture_request(llm, [1, 8, 2, 3], "params", "10k_on")
        assert output == ["output"] and elapsed >= 0
    assert events == [("start", {"profile_prefix": "10k_on"}), "generate", "stop"]


def test_child_only_requests_first_token_and_shuts_down(tool, monkeypatch, tmp_path):
    args = tool.parser().parse_args(["--child", "10k_on", "--run-dir", str(tmp_path)])
    case_dir = tmp_path / "10k_on"
    case_dir.mkdir()
    tool.write_json(tmp_path / "10k_prompt.json", {"length": 10000, "token_ids": [9] * 10000})
    events = []

    def generate(prompt, params, **kwargs):
        events.append("generate")
        assert len(prompt["prompt_token_ids"]) == 10000
        assert params.max_tokens == 1 and params.temperature == 0
        return [NS(num_cached_tokens=0, outputs=[NS(text="hello", token_ids=[42])])]

    def llm(**options):
        events.append("load")
        assert options["max_model_len"] == 16384
        return NS(
            start_profile=lambda **kw: events.append("start"),
            stop_profile=lambda: events.append("stop"),
            generate=generate,
            llm_engine=NS(engine_core=NS(shutdown=lambda: events.append("shutdown"))),
        )

    monkeypatch.setitem(sys.modules, "vllm", NS(LLM=llm, SamplingParams=NS))
    tool.run_child(args)
    assert events == ["load", "start", "generate", "stop", "shutdown"]
    assert json.loads((case_dir / "result.json").read_text())["token_ids"] == [42]


def test_sequential_cases_release_model_before_analysis_and_next_launch(tool, monkeypatch, tmp_path):
    events = []
    args = tool.parser().parse_args(["--include-off"])
    assert args.case == "all"
    assert tool.CASES == ("10k_off", "10k_on")

    def launch(command, env, log, label, **kwargs):
        events.append((label, "start"))
        assert command[command.index("--child") + 1] == label
        assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == str(label.endswith("_on")).lower()
        assert log == tmp_path / label / "server.log"
        return NS(label=label, wait=lambda: events.append((label, "wait")) or 0)

    monkeypatch.setattr(tool, "start_logged_process", launch)
    monkeypatch.setattr(tool, "finish_child", lambda p: events.append((p.label, "finish")))
    monkeypatch.setattr(tool, "analyse_case", lambda p: events.append((p.name, "analyse")))
    tool.run_cases(args, tmp_path, tool.CASES)
    assert events == [(case, action) for case in tool.CASES for action in ("start", "wait", "finish", "analyse")]


@pytest.mark.parametrize(
    "options, expected",
    [
        ([], ("10k_on",)),
        (["--include-off"], ("10k_off", "10k_on")),
        (["--case", "all"], ("10k_off", "10k_on")),
        (["--case", "10k_off"], ("10k_off",)),
        (["--case", "10k_on"], ("10k_on",)),
    ],
)
def test_main_selects_requested_cases(tool, monkeypatch, tmp_path, options, expected):
    events = []
    monkeypatch.setattr(tool, "os", NS(name="posix"))
    monkeypatch.setattr(tool, "prepare_inputs", lambda args, root, cases: events.append(("prepare", cases)))
    monkeypatch.setattr(tool, "run_cases", lambda args, root, cases: events.append(("run", cases)))
    tool.main(["--run-dir", str(tmp_path), *options])
    assert events == [("prepare", expected), ("run", expected)]


def test_analyse_only_still_exports_existing_off_and_on(tool, monkeypatch, tmp_path):
    for case in tool.CASES:
        case_dir = tmp_path / case
        case_dir.mkdir()
        tool.write_json(case_dir / "engine_options.json", {})
    analysed = []
    monkeypatch.setattr(tool, "analyse_case", lambda path: analysed.append(path.name))
    monkeypatch.setattr(tool, "run_cases", lambda *args: pytest.fail("Analysis must not launch models"))
    tool.main(["--analyse-only", str(tmp_path)])
    assert analysed == list(tool.CASES)


@pytest.mark.parametrize("option", ["--case", "--child"])
def test_removed_long_case_cannot_be_launched(tool, option):
    with pytest.raises(SystemExit) as error:
        tool.parser().parse_args([option, "100k_on"])
    assert error.value.code == 2


def test_failed_case_is_cleaned_and_no_following_case_launches(tool, monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(tool, "start_logged_process", lambda *a, **kw: NS(wait=lambda: 1))
    monkeypatch.setattr(tool, "finish_child", lambda p: events.append("finish"))
    monkeypatch.setattr(tool, "analyse_case", lambda p: pytest.fail("Must not analyse failed request as success"))
    with pytest.raises(RuntimeError, match="10k_off failed"):
        tool.run_cases(tool.parser().parse_args([]), tmp_path, tool.CASES)
    assert events == ["finish"]
    assert not (tmp_path / "10k_on").exists()


def test_trace_export_writes_all_rank_paths(tool, monkeypatch, tmp_path):
    tool.write_json(tmp_path / "engine_options.json", {"tensor_parallel_size": 8})
    for rank in range(8):
        path = tmp_path / "profile" / f"rank{rank}" / "ASCEND_PROFILER_OUTPUT"
        path.mkdir(parents=True)
        (path / "trace_view.json").write_text("{}")
    calls = []
    monkeypatch.setattr(tool.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    tool.analyse_case(tmp_path)
    assert len(json.loads((tmp_path / "traces.json").read_text())) == 8
    assert calls[0][0][-1] == str(tmp_path / "profile")
    assert "max_process_number=2" in calls[0][0][-2]
    assert calls[0][1] == {"check": True}


def test_missing_trace_is_not_reported_as_success(tool, monkeypatch, tmp_path):
    tool.write_json(tmp_path / "engine_options.json", {"tensor_parallel_size": 8})
    monkeypatch.setattr(tool.subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="found 0"):
        tool.analyse_case(tmp_path)
