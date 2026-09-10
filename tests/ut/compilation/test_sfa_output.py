# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU wiring/guard tests for free generation; not real NPU inference."""

from types import SimpleNamespace

import pytest
import torch
from test_sfa_parity import checkpoint as checkpoint
from test_sfa_parity import parity as parity
from test_sfa_parity import worker as worker


@pytest.mark.parametrize("compare_output", [False, True])
def test_output_mode_keeps_actual_mtp_choices(worker, compare_output):
    subject = worker.SFAParityWorker()
    subject.parity_options = {"compare_output": compare_output, "token_id": 100}
    subject.parity_draft_calls = 0
    proposals = torch.tensor([[237], [19]], dtype=torch.int32)
    result = subject._parity_draft_tokens(proposals)
    assert subject.parity_draft_calls == 1
    assert proposals.tolist() == [[237], [19]]
    if compare_output:
        assert result is proposals
    else:
        assert result.tolist() == [[100], [100]]


@pytest.mark.parametrize("tokens", [None, [[1]], torch.tensor(1), torch.tensor([[1, 2]])])
def test_invalid_draft_result_still_fails(worker, tokens):
    subject = worker.SFAParityWorker()
    subject.parity_options = {"compare_output": True}
    subject.parity_draft_calls = 0
    with pytest.raises(worker.ParityError, match="one draft token"):
        subject._parity_draft_tokens(tokens)
    assert subject.parity_draft_calls == 0


@pytest.fixture
def output_worker(worker, tmp_path, monkeypatch):
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None)), raising=False
    )
    subject = worker.SFAParityWorker()
    subject.parity_options = {"compare_output": True, "atol": 0, "rtol": 0}
    subject.parity_step = 9
    subject.parity_is_graph = True
    subject.parity_directory = tmp_path
    subject.parity_decode_observations = 0
    subject.parity_transfers = [0] * 8
    subject.parity_layers = [
        SimpleNamespace(
            index=index,
            read=lambda rows, decode, index=index: (
                {f"layer={index} output.hidden": torch.full((rows, 4), 1000.0 + index)},
                {f"layer={index} miss_count": torch.tensor([1])},
            ),
        )
        for index in range(8)
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("Free generation must not compare/reload teacher-forced steps")

    monkeypatch.setattr(worker, "compare_step", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    return subject


@pytest.mark.parametrize("graph", [False, True])
def test_all_decode_steps_observed_without_intermediate_tolerance_abort(output_worker, graph):
    output_worker.parity_is_graph = graph
    for index in range(12):
        output_worker.parity_step = 9 + index
        output_worker._observe_step({"rows": 2, "decode": True}, torch.ones(2, 4) * index, int(graph))
    assert output_worker.parity_decode_observations == 12
    assert output_worker.parity_transfers == [12] * 8
    assert not list(output_worker.parity_directory.iterdir())


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("location", ["layer", "final"])
def test_output_mode_never_ignores_nonfinite_data(worker, output_worker, graph, bad, location):
    output_worker.parity_is_graph = graph
    final = torch.ones(2, 4)
    if location == "final":
        final[0, 0] = bad
    else:
        output_worker.parity_layers[5].read = lambda rows, decode: (
            {"layer=5 kv_nope": torch.tensor([[bad]])},
            {"layer=5 miss_count": torch.tensor([1])},
        )
    with pytest.raises(worker.ParityError, match="NaN/Inf"):
        output_worker._observe_step({"rows": 2, "decode": True}, final, int(graph))
    assert output_worker.parity_decode_observations == 0


@pytest.mark.parametrize("replays", [0, 2])
def test_output_mode_still_requires_one_root_replay(worker, output_worker, replays):
    with pytest.raises(worker.ParityError, match="expected 1 root replay"):
        output_worker._observe_step({"rows": 2, "decode": True}, torch.ones(2, 4), replays)
    assert output_worker.parity_decode_observations == 0


def test_stale_or_invalid_address_probe_failure_still_propagates(worker, output_worker):
    def broken(*args):
        raise worker.ParityError("attention is addressing an invalid KV location")

    output_worker.parity_layers[4].read = broken
    with pytest.raises(worker.ParityError, match="invalid KV location"):
        output_worker._observe_step({"rows": 2, "decode": True}, torch.ones(2, 4), 1)
    assert output_worker.parity_decode_observations == 0
