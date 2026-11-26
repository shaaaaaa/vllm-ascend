# SPDX-License-Identifier: Apache-2.0
"""Local handoff qualification excludes warmup, unrelated requests and dense decode."""

import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


@pytest.fixture
def handoff(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    import check_banked_dcp_handoff

    return check_banked_dcp_handoff


def test_observe_only_successful_real_forwards_and_restore_hook(handoff, monkeypatch):
    req = "cmpl-rr-0-11111111"
    graph = NS(replay_count=9)
    ctx = NS(staged_sfa_graph_dummy_run=False, staged_sfa_route=NS(frontiers=(0,)), attn_metadata={})
    mod = ModuleType("vllm.forward_context")
    mod.get_forward_context = lambda: ctx
    monkeypatch.setitem(sys.modules, "vllm.forward_context", mod)

    parallel = ModuleType("vllm.distributed.parallel_state")
    parallel.get_tp_group = lambda: NS(rank_in_group=1, world_size=4)
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)

    class Runner:
        layerwise_prefill_p_node = False
        dcp_rank = 1
        dcp_size = 4
        _sfa_full_graph = graph
        _dcp_indexer_owners = {req: (2, 17)}
        input_batch = NS(req_ids=[req], num_computed_tokens_cpu=[20002])

        def _model_forward(self, replay=False):
            graph.replay_count += int(replay)
            return 5

    owner = handoff.BankedDcpWitness()
    runner = owner.model_runner = Runner()
    owner.arm_banked_dcp_witness("a")
    assert owner.finish_banked_dcp_witness("other") is None
    with pytest.raises(RuntimeError, match="already armed"):
        owner.arm_banked_dcp_witness("other")
    meta = NS(sfa_dcp=None, prefill_dcp=NS(coverage={req: (0, 16384)}, sources=(1, 2, 3, 4), count=0))
    ctx.attn_metadata = {"L0": meta, "L1": meta}
    assert runner._model_forward() == 5
    meta.prefill_dcp.coverage = {req: (16384, 20000)}
    meta.prefill_dcp.count = 32
    runner._model_forward()
    meta.prefill_dcp = None
    meta.sfa_dcp = NS(owner_rows=object())
    runner._model_forward(replay=True)  # Resident decode has a zero frontier.
    ctx.staged_sfa_route.frontiers = (20000,)
    ctx.staged_sfa_graph_dummy_run = True
    runner._model_forward(replay=True)
    ctx.staged_sfa_graph_dummy_run = False
    runner._model_forward()  # A native fallback has no replay increment.
    assert owner._banked_dcp_witness[-1]["decode"] == {}
    runner._model_forward(replay=True)
    runner._model_forward(replay=True)
    report = owner.finish_banked_dcp_witness("a")
    assert report["prefill"] == {req: [[0, 16384, 4, 0], [16384, 20000, 4, 32]]}
    assert report["decode"] == {req: dict(frontier=20000, owner=2, epoch=17, replays=2, computed=20002)}
    assert "_model_forward" not in vars(runner)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "warm",
        "nocontinuation",
        "wrong_order",
        "rank",
        "owner",
        "zero",
        "bootstrap",
        "eos",
        "reallocated",
        "other",
    ],
)
def test_phase_evidence_across_workers(handoff, bad):
    key = "cmpl-r-0-abc123ef" if bad != "other" else "cmpl-r2-0-abc123ef"
    chunks = [[0, 128, 4, 0], [128, 55555, 4, 1]]
    decode = dict(frontier=55555, owner=1, epoch=21, replays=4, computed=55559)
    rows = [dict(rank=i, tp=2, world=2, prefill={key: chunks}, decode={key: decode}) for i in range(2)]
    if bad == "warm":
        chunks[0][0] = 128
    elif bad == "nocontinuation":
        chunks[1][-1] = 0
    elif bad == "wrong_order":
        chunks.reverse()
    elif bad == "rank":
        rows[1]["rank"] = 0
    elif bad == "owner":
        rows[1]["decode"] = {key: dict(decode, owner=0)}
    elif bad == "zero":
        decode["frontier"] = 0
    elif bad == "bootstrap":
        decode["replays"] = 1
    elif bad == "eos":
        decode["computed"] = 55555
    elif bad == "reallocated":
        decode["placement_changed"] = True

    def verify():
        assert handoff.verify(rows, "prefill", "cmpl-r", 55555) == chunks
        assert handoff.verify(rows[::-1], "decode", "cmpl-r", 55555) == decode

    if bad is None:
        verify()
    else:
        with pytest.raises(ValueError):
            verify()


def test_replicated_sender_control_still_checks_all_tp_workers(handoff):
    chunks = [[0, 128, 4, None], [128, 256, 4, None]]
    rows = [dict(rank=i, tp=4, world=1, prefill={"cmpl-r-0": chunks}) for i in range(4)]
    assert handoff.verify(rows, "prefill", "cmpl-r", 256, expected_dcp=1) == chunks
    with pytest.raises(ValueError, match="Incomplete"):
        handoff.verify(rows, "prefill", "cmpl-r", 256)


@pytest.mark.parametrize("arm_failure", [False, True])
def test_client_cleans_both_hooks_even_when_one_rank_fails(handoff, monkeypatch, capsys, arm_failure):
    import json

    import requests

    req = "cmpl-wire"
    armed = {}
    check_ids = set()

    def post(url, json, timeout):
        endpoint, method = url.split(":")[-1].split("/", 1)
        if method == "collective_rpc":
            (key,) = json["args"]
            check_ids.add(key)
            if json["method"] == "arm_banked_dcp_witness":
                armed[endpoint] = True
                if arm_failure and endpoint == "7920":
                    raise RuntimeError("partial arm")
                payload = {"results": [None, None]}
            else:
                armed[endpoint] = False
                rows = []
                for i in range(2):
                    values = (
                        {"prefill": [[0, 1024, 4, 0], [1024, 2048, 4, 4]]}
                        if endpoint == "7910"
                        else {"decode": dict(frontier=2047, owner=1, epoch=3, replays=2, computed=2049)}
                    )
                    rows.append(
                        {
                            "rank": i,
                            "tp": 2,
                            "world": 2,
                            **{stage: {req + "-0-" + endpoint: entries} for stage, entries in values.items()},
                        }
                    )
                payload = {"results": rows}
        else:
            assert endpoint == "8080" and method == "v1/completions"
            assert json["return_token_ids"] and len(json["prompt"]) == 2048
            payload = {
                "id": req,
                "usage": {"prompt_tokens": 2048, "completion_tokens": 3},
                "choices": [{"token_ids": [7, 11, 13]}],
            }
        return NS(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(sys, "argv", ["check", "--prompt-tokens", "2048", "--output-tokens", "3"])
    if arm_failure:
        with pytest.raises(RuntimeError, match="partial arm"):
            handoff.main()
    else:
        handoff.main()
        report = json.loads(capsys.readouterr().out)
        assert report["request"] == req and report["decode"]["frontier"] == 2047
    assert armed == {"7910": False, "7920": False} and len(check_ids) == 1
