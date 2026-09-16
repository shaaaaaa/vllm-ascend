# SPDX-License-Identifier: Apache-2.0
"""Real CPU tensor/file tests; not evidence of NPU model parity."""

import copy
import json
import sys

import pytest
import torch
from test_sfa_parity import checkpoint as checkpoint
from test_sfa_parity import parity as parity
from test_sfa_parity import worker as worker


@pytest.fixture
def statistics(worker):
    return sys.modules["vllm_ascend.attention.sfa_parity_stats"]


def state(value, step=9, token=0):
    return {
        "rank": 0,
        "tp_size": 8,
        "step": step,
        "decode": True,
        "rows": 2,
        "layers": 8,
        "trace_residual": False,
        "input_ids": torch.tensor([token, token]),
        "positions": torch.tensor([4351 + step - 9, 4352 + step - 9]),
        "seq_lens": torch.tensor([4353 + step - 9]),
        "query_ends": torch.tensor([2]),
        "tensors": {"layer=0 output.hidden": value},
    }


def test_absolute_error_is_not_difference_of_magnitudes(statistics):
    stages = {}
    statistics.add_step(stages, state(torch.tensor([-2.0, 0.0, 2.0])), state(torch.tensor([2.0, 0.0, 1.0])))
    result = stages["layer=0 output.hidden"]
    assert result["diff_abs"]["mean"] == pytest.approx(5 / 3)
    assert result["diff_abs"]["m2"] / 3 == pytest.approx(26 / 9)
    assert result["diff_abs"]["max"] == 4
    assert result["diff_abs"]["nonzero"] == 2
    assert result["eager_abs"]["mean"] == pytest.approx(4 / 3)
    assert result["graph_abs"]["mean"] == 1


def test_zero_and_tiny_differences_are_descriptive_not_a_failure(statistics):
    stages = {}
    statistics.add_step(stages, state(torch.zeros(1000)), state(torch.zeros(1000)))
    result = stages["layer=0 output.hidden"]
    for metric in ("diff_abs", "eager_abs", "graph_abs"):
        assert result[metric] == {"count": 1000, "mean": 0, "m2": 0, "max": 0, "nonzero": 0}
    statistics.add_step(stages, state(torch.zeros(1, dtype=torch.float64)), state(torch.tensor([1e-7]).double()))
    assert stages["layer=0 output.hidden"]["diff_abs"]["count"] == 1001
    assert stages["layer=0 output.hidden"]["diff_abs"]["nonzero"] == 1


def test_1000_element_population_statistics_and_weighted_rank_step_merging(statistics):
    reference = torch.arange(-500, 500, dtype=torch.float64) / 3
    actual = reference + torch.arange(1000, dtype=torch.float64).remainder(7) / 100
    ranks = []
    for start, end in ((0, 3), (3, 100), (100, 1000)):
        stages = {}
        for ref, val in zip(reference[start:end].split(29), actual[start:end].split(29)):
            statistics.add_step(stages, state(ref), state(val))
        ranks.append(stages)
    merged = statistics.merge_stages(ranks)["layer=0 output.hidden"]
    for key, data in (
        ("diff_abs", (actual - reference).abs()),
        ("eager_abs", reference.abs()),
        ("graph_abs", actual.abs()),
    ):
        assert merged[key]["count"] == 1000
        assert merged[key]["mean"] == pytest.approx(data.mean().item())
        assert merged[key]["m2"] / 1000 == pytest.approx(data.var(unbiased=False).item())
        assert merged[key]["max"] == data.max().item()
        assert merged[key]["nonzero"] == torch.count_nonzero(data).item()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("mode", ["reference", "actual"])
def test_nonfinite_observations_are_still_errors(statistics, bad, mode):
    reference, actual = state(torch.zeros(3)), state(torch.zeros(3))
    (reference if mode == "reference" else actual)["tensors"]["layer=0 output.hidden"][1] = bad
    with pytest.raises(ValueError, match="NaN/Inf"):
        statistics.add_step({}, reference, actual)


@pytest.mark.parametrize("issue", ["dtype", "shape", "empty", "probe"])
def test_incomparable_tensor_schema_cannot_produce_statistics(statistics, issue):
    reference, actual = state(torch.ones(3)), state(torch.ones(3))
    actual["tensors"]["layer=0 output.hidden"] = {
        "dtype": torch.ones(3).double(),
        "shape": torch.ones(4),
        "empty": torch.ones(0),
        "probe": torch.ones(3),
    }[issue]
    if issue == "probe":
        actual["tensors"]["unexpected"] = torch.ones(3)
    with pytest.raises(ValueError):
        statistics.add_step({}, reference, actual)


@pytest.mark.parametrize("no_valid", [False, True])
def test_kv_excludes_padding_and_different_logical_topk(statistics, no_valid):
    reference = state(torch.ones(2))
    reference["tensors"].update(
        {
            "layer=0 topk": torch.tensor([[1, 2, -1]]),
            "layer=0 valid": torch.tensor([[not no_valid, True, False]]),
            "layer=0 kv_nope": torch.tensor([[[2.0, 3.0], [4.0, 5.0], [0.0, 0.0]]]),
        }
    )
    actual = copy.deepcopy(reference)
    actual["tensors"]["layer=0 topk"][0, 1] = 3
    actual["tensors"]["layer=0 kv_nope"] += 1
    actual["tensors"]["layer=0 kv_nope"][0, 1:] = 1e6  # excluded, not a false giant KV error
    stages = {}
    statistics.add_step(stages, reference, actual)
    kv = stages["layer=0 kv_nope"]
    assert kv["excluded"] == (6 if no_valid else 4)
    assert kv["diff_abs"]["count"] == (0 if no_valid else 2)
    assert kv["diff_abs"]["mean"] == (0 if no_valid else 1)
    assert stages["layer=0 topk"]["diff_abs"]["nonzero"] == 1


def make_worker(worker, directory):
    subject = worker.SFAParityWorker()
    subject.parity_directory = directory
    directory.mkdir()
    subject.parity_rank = 0
    subject.parity_is_graph = False
    subject.parity_decode_observations = 0
    return subject


def save_eager(subject, states):
    for index, snapshot in enumerate(states):
        subject.parity_decode_observations = index
        subject._observe_output_statistics(snapshot)
    (subject.parity_directory.parent / "eager-summary.json").write_text(
        json.dumps([{"rank": 0, "decode_observations": len(states)}])
    )
    subject.parity_is_graph = True
    subject.parity_decode_observations = 0


def test_worker_uses_real_eager_snapshots_and_reports_all_stages_without_threshold(worker, tmp_path):
    subject = make_worker(worker, tmp_path / "rank-0")
    eager = [state(torch.tensor([-2.0, 0.0, 3.0]), step=9 + i) for i in range(3)]
    for snapshot in eager:
        for layer in range(8):
            snapshot["tensors"][f"layer={layer} input.hidden"] = torch.ones(2)
            snapshot["tensors"][f"layer={layer} output.hidden"] = torch.ones(2)
        snapshot["tensors"]["target.final_hidden"] = torch.ones(2)
    save_eager(subject, eager)
    for index, reference in enumerate(eager):
        actual = copy.deepcopy(reference)
        for value in actual["tensors"].values():
            value.add_(100)  # Even large finite differences are statistics, not a tolerance gate.
        subject.parity_decode_observations = index
        subject._observe_output_statistics(actual)
    report = subject.parity_absolute_statistics
    assert report["compared_steps"] == report["graph_steps"] == report["eager_steps"] == 3
    assert len(report["stages"]) == 17
    assert all(stage["diff_abs"]["mean"] == 100 for stage in report["stages"].values())
    assert len(list(subject.parity_directory.glob("*.pt"))) == 3  # no graph snapshot duplication


@pytest.mark.parametrize("field", ["input_ids", "positions", "seq_lens", "query_ends", "rows"])
def test_input_divergence_is_sticky_even_if_later_tokens_match(worker, tmp_path, field):
    subject = make_worker(worker, tmp_path / "rank-0")
    eager = [state(torch.zeros(3), step=9 + i) for i in range(3)]
    save_eager(subject, eager)
    for index, reference in enumerate(eager):
        actual = copy.deepcopy(reference)
        if index == 1:
            actual[field] += 1
        subject.parity_decode_observations = index
        subject._observe_output_statistics(actual)
    report = subject.parity_absolute_statistics
    assert report["compared_steps"] == 1
    assert report["graph_steps"] == 3
    assert report["first_unaligned"] == {"step": 10, "reason": field}


def test_missing_eager_file_is_not_misreported_as_normal_input_divergence(worker, tmp_path):
    subject = make_worker(worker, tmp_path / "rank-0")
    save_eager(subject, [state(torch.zeros(3))])
    (subject.parity_directory.parent / "eager-summary.json").write_text(
        json.dumps([{"rank": 0, "decode_observations": 2}])
    )
    with pytest.raises(worker.ParityError, match="snapshots"):
        subject._observe_output_statistics(state(torch.zeros(3)))


def test_extra_graph_step_is_marked_unaligned_without_stopping_generation(worker, tmp_path):
    subject = make_worker(worker, tmp_path / "rank-0")
    save_eager(subject, [state(torch.zeros(3))])
    subject._observe_output_statistics(state(torch.zeros(3)))
    subject.parity_decode_observations = 1
    subject._observe_output_statistics(state(torch.zeros(3), step=10))
    assert subject.parity_absolute_statistics["compared_steps"] == 1
    assert subject.parity_absolute_statistics["graph_steps"] == 2
    assert "no further" in subject.parity_absolute_statistics["first_unaligned"]["reason"]


def test_prints_three_absolute_distributions_and_partial_coverage(statistics, capsys):
    stages = {}
    statistics.add_step(stages, state(torch.zeros(1000)), state(torch.zeros(1000)))
    eager = [{"rank": 0, "decode_observations": 2}]
    graph = [
        {
            "rank": 0,
            "decode_observations": 1,
            "absolute_statistics": {
                "eager_steps": 2,
                "graph_steps": 1,
                "compared_steps": 1,
                "first_unaligned": None,
                "stages": stages,
            },
        }
    ]
    statistics.print_statistics(eager, graph)
    printed = capsys.readouterr().out
    assert "PARTIAL" in printed and "n=1000" in printed
    for metric in ("diff_abs", "eager_abs", "graph_abs"):
        assert f"{metric}(mean=0 std=0 var=0 max=0 nonzero=0)" in printed
    assert "PASS" not in printed


def test_zero_compared_steps_cannot_be_presented_as_successful_statistics(statistics):
    eager = [{"rank": 0, "decode_observations": 1}]
    graph = [
        {
            "rank": 0,
            "decode_observations": 1,
            "absolute_statistics": {
                "eager_steps": 1,
                "graph_steps": 1,
                "compared_steps": 0,
                "first_unaligned": "input_ids",
                "stages": {},
            },
        }
    ]
    with pytest.raises(ValueError, match="statistics"):
        statistics.print_statistics(eager, graph)


@pytest.mark.parametrize("field", ["rank", "tp_size", "decode", "layers", "trace_residual"])
def test_wrong_configuration_is_an_error_not_an_ordinary_alignment_difference(statistics, field):
    reference = state(torch.ones(3))
    actual = copy.deepcopy(reference)
    actual[field] += 1
    with pytest.raises(ValueError, match="configuration"):
        statistics.alignment_reason(reference, actual)


def test_report_keeps_a_single_bad_rank_in_pooled_maximum(statistics, capsys):
    eager, graph = [], []
    for rank in range(8):
        stages = {}
        statistics.add_step(stages, state(torch.ones(10)), state(torch.ones(10) * (9 if rank == 7 else 1)))
        eager.append({"rank": rank, "decode_observations": 1})
        graph.append(
            {
                "rank": rank,
                "decode_observations": 1,
                "absolute_statistics": {
                    "eager_steps": 1,
                    "graph_steps": 1,
                    "compared_steps": 1,
                    "first_unaligned": None,
                    "stages": stages,
                },
            }
        )
    statistics.print_statistics(eager, graph)
    printed = capsys.readouterr().out
    assert printed.count("COMPLETE") == 8
    assert "n=80" in printed
    assert "max=8 nonzero=10" in printed
