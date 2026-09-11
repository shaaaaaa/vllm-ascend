# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of output-mode orchestration, not simulated model correctness."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from test_sfa_full_graph_parity import driver as driver
from test_sfa_full_graph_parity import reports


def output_reports(mode):
    result = reports(mode)
    for report in result:
        # Different MTP acceptance can change forward count with equal output.
        steps = 8 if mode == "eager" else 12
        report.update(
            steps=9 + steps,
            decode_steps=steps,
            q2_steps=steps,
            draft_calls=steps,
            decode_observations=steps,
            compare_output=True,
        )
    return result


def output():
    return {"token_ids": list(range(16)), "text": "乱码\n<special>", "finish_reason": "length"}


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_real_greedy_sampling_without_target_mask(driver, monkeypatch, tmp_path, mode):
    generated = output()
    llm = Mock()
    llm.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(**generated)])]
    llm.collective_rpc.return_value = output_reports(mode)
    constructor = Mock(return_value=llm)
    stub = ModuleType("vllm")
    stub.LLM, stub.SamplingParams = constructor, Mock()
    monkeypatch.setitem(sys.modules, "vllm", stub)
    monkeypatch.setattr(driver, "track_workers", lambda reports, tp_size: [])
    driver.run_child(
        SimpleNamespace(
            child=mode,
            devices=driver.DEFAULT_DEVICES,
            model="local-model",
            reference=str(tmp_path),
            atol=1e-7,
            rtol=1e-2,
            compare_output=True,
        )
    )
    params = stub.SamplingParams.call_args.kwargs
    assert params["allowed_token_ids"] is None
    assert params["temperature"] == 0
    assert params["detokenize"] and not params["skip_special_tokens"]
    assert params["min_tokens"] == params["max_tokens"] == 16
    assert params["ignore_eos"]
    config = constructor.call_args.kwargs
    assert config["tensor_parallel_size"] == 8 and config["hf_overrides"] == {"num_hidden_layers": 8}
    assert config["additional_config"]["sfa_parity"]["compare_output"]
    assert json.loads((tmp_path / f"{mode}-output.json").read_text(encoding="utf-8")) == generated


def test_output_mode_accepts_independent_decode_lengths(driver):
    driver.validate_summaries(output_reports("eager"), output_reports("graph"), 8, compare_output=True)


@pytest.mark.parametrize(
    "field,value", [("compare_output", False), ("decode_observations", 0), ("prefill_model_calls", 1), ("q2_steps", 0)]
)
def test_output_mode_still_requires_correct_execution_on_every_rank(driver, field, value):
    graph = output_reports("graph")
    graph[7][field] = value
    with pytest.raises(AssertionError):
        driver.validate_summaries(output_reports("eager"), graph, 8, compare_output=True)


def test_equal_text_cannot_hide_different_token_ids(driver):
    ref, actual = output(), output()
    actual["token_ids"][3] = 42
    report = driver.compare_generated_outputs(ref, actual)
    assert report["text_equal"] and not report["tokens_equal"]
    assert report["first_token_difference"] == {"token_number": 4, "eager": 3, "graph": 42}


def test_identical_tokens_and_text_are_compared_in_full(driver):
    assert driver.compare_generated_outputs(output(), output()) == {
        "tokens_equal": True,
        "text_equal": True,
        "first_token_difference": None,
        "tokens_compared": 16,
    }


@pytest.mark.parametrize("corruption", ["short", "invalid_token", "missing_text"])
def test_missing_or_partial_outputs_cannot_match(driver, corruption):
    actual = output()
    if corruption == "short":
        actual["token_ids"].pop()
    elif corruption == "invalid_token":
        actual["token_ids"][0] = -1
    else:
        del actual["text"]
    with pytest.raises(AssertionError, match="complete generation"):
        driver.compare_generated_outputs(output(), actual)


@pytest.mark.parametrize("difference", [None, "tokens", "text"])
@pytest.mark.parametrize("trace_residual", [False, True])
def test_pair_completes_both_runs_then_reports_output_result(
    driver, monkeypatch, tmp_path, capsys, difference, trace_residual
):
    (tmp_path / "config.json").write_text("{}")
    calls = []

    def run(argv, **kwargs):
        mode = argv[argv.index("--child") + 1]
        calls.append(mode)
        assert "--compare-output" in argv
        assert ("--trace-residual" in argv) == trace_residual
        if mode == "preflight":
            return
        directory = Path(argv[argv.index("--reference") + 1])
        (directory / f"{mode}-summary.json").write_text(json.dumps(output_reports(mode)))
        generated = output()
        if mode == "graph":
            if difference == "tokens":
                generated["token_ids"][-1] = 42
            elif difference == "text":
                generated["text"] = "different"
        (directory / f"{mode}-output.json").write_text(json.dumps(generated), encoding="utf-8")

    monkeypatch.setattr(driver.subprocess, "run", run)
    statistics = Mock()
    monkeypatch.setattr(driver, "print_stage_statistics", statistics)
    if difference:
        with pytest.raises(AssertionError, match="OUTPUT DIFFERENT"):
            driver.run_pair(str(tmp_path), compare_output=True, trace_residual=trace_residual)
    else:
        driver.run_pair(str(tmp_path), compare_output=True, trace_residual=trace_residual)
    assert calls == ["preflight", "eager", "graph"]
    statistics.assert_called_once_with(output_reports("eager"), output_reports("graph"))
    printed = capsys.readouterr().out
    assert "[SFA_OUTPUT] eager:" in printed and "[SFA_OUTPUT] graph:" in printed
    assert ("OUTPUT MATCH" in printed) == (difference is None)
    assert "[SFA_PARITY] PASS" not in printed
    if difference == "tokens":
        assert '"token_number": 16' in printed


def test_cli_forwards_output_mode(driver, monkeypatch):
    run = Mock()
    monkeypatch.setattr(driver, "run_pair", run)
    monkeypatch.setattr(sys, "argv", ["driver", "--compare-output"])
    driver.main()
    assert run.call_args.kwargs["compare_output"] is True
    assert run.call_args.kwargs["trace_residual"] is False


def test_extra_probes_are_explicitly_opted_in_for_output_statistics(driver, monkeypatch):
    run = Mock()
    monkeypatch.setattr(driver, "run_pair", run)
    monkeypatch.setattr(sys, "argv", ["driver", "--compare-output", "--trace-residual"])
    driver.main()
    assert run.call_args.kwargs["compare_output"] is True
    assert run.call_args.kwargs["trace_residual"] is True
