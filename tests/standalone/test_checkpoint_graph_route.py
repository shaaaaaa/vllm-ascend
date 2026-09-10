# SPDX-License-Identifier: Apache-2.0
"""Run the production route classifier without initializing NPU collectives."""

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def route_api():
    ns = dict(Enum=Enum, dataclass=dataclass, Any=Any, np=np)
    utils = ast.parse((ROOT / "vllm_ascend/utils.py").read_text(encoding="utf-8"))
    definitions = [
        n
        for n in utils.body
        if isinstance(n, ast.ClassDef)
        and n.name in {"StagedSFARouteAction", "StagedSFARouteReason", "StagedSFARouteDecision"}
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "route_types", "exec"), ns)
    metadata = ast.parse((ROOT / "vllm_ascend/attention/utils.py").read_text(encoding="utf-8"))
    markers = next(n for n in metadata.body if isinstance(n, ast.ClassDef) and n.name == "ColdResumeMarkers")
    exec(compile(ast.Module(body=[markers], type_ignores=[]), "cold_markers", "exec"), ns)
    source = ast.parse((ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "_staged_sfa_local_route")
    ns.update(
        cold_perf_enabled=lambda: False,
        AscendAttentionState=NS(DecodeOnly="decode", SpecDecoding="spec"),
        unwrap_staged_sfa_connector_metadata=lambda x: x,
        staged_sfa_metadata_sparse_route=lambda meta, ids: (ns["StagedSFARouteReason"].ELIGIBLE, meta[0], meta[1]),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), "route", "exec"), ns)
    return ns


@pytest.mark.parametrize(
    "prompt,history,computed,expected",
    [
        (3000, 3000, 2999, "staged"),
        (3000, 3100, 3099, "staged"),
        (3000, 3100, 3000, "safe_native"),
        (3000, 3100, 3100, "safe_native"),
    ],
)
def test_only_last_token_cold_resume_can_enter_graph(prompt, history, computed, expected):
    ns = route_api()
    runner = NS(
        _staged_sfa_graph_capture_sizes=[8, 16, 24, 32],
        speculative_config=NS(num_speculative_tokens=1),
        attn_state="spec",
        decode_threshold=2,
        vllm_config=NS(lora_config=None),
        input_batch=NS(num_tokens_no_spec=np.array([history])),
    )
    outcome = ns["_staged_sfa_local_route"](
        runner,
        num_tokens_unpadded=2,
        num_reqs=1,
        num_scheduled_tokens=np.array([2]),
        index_topk=1024,
        has_cascade_attention=False,
        request_ids=["r"],
        kv_connector_metadata=((computed,), (True,)),
        num_computed_tokens=[computed],
        prompt_lens=[prompt],
    )
    assert outcome.action.value == expected
    if expected == "staged":
        assert outcome.cold_compact_resumes.computed_ends == (computed,)


def test_historical_recovery_is_not_relabelled_as_speculative_decode():
    ns = route_api()
    runner = NS(
        _staged_sfa_graph_capture_sizes=[8, 16, 24, 32],
        speculative_config=NS(num_speculative_tokens=1),
        attn_state="prefill",
        decode_threshold=2,
        vllm_config=NS(lora_config=None),
        input_batch=NS(num_tokens_no_spec=np.array([3100])),
    )
    outcome = ns["_staged_sfa_local_route"](
        runner,
        num_tokens_unpadded=32,
        num_reqs=1,
        num_scheduled_tokens=np.array([32]),
        index_topk=1024,
        has_cascade_attention=False,
        request_ids=["r"],
        kv_connector_metadata=((3000,), (True,)),
        num_computed_tokens=[3000],
        prompt_lens=[3000],
    )
    assert outcome.reason.value == "not_decode"


def test_cold_marker_proof_survives_existing_metadata_unpadding_and_copy():
    from copy import copy, deepcopy

    mask = route_api()["ColdResumeMarkers"]((True, False, True), (3099, 0, 4111))
    assert tuple(mask) == (True, False, True)
    assert mask[0] is True and mask[1] is False
    for cloned in (copy(mask), deepcopy(mask), mask[:2], mask[::2]):
        assert len(cloned.computed_ends) == len(cloned)
    assert mask[:2].computed_ends == (3099, 0)
    assert mask[::2].computed_ends == (3099, 4111)


def test_ordinary_graph_route_does_not_read_checkpoint_history():
    ns = route_api()
    runner = NS(
        _staged_sfa_graph_capture_sizes=[8, 16, 24, 32],
        speculative_config=NS(num_speculative_tokens=1),
        attn_state="spec",
        decode_threshold=2,
        vllm_config=NS(lora_config=None),
    )
    # No input_batch/history attribute: touching it on an ordinary step fails.
    result = ns["_staged_sfa_local_route"](
        runner,
        num_tokens_unpadded=2,
        num_reqs=1,
        num_scheduled_tokens=np.array([2]),
        index_topk=1024,
        has_cascade_attention=False,
        request_ids=["r"],
        kv_connector_metadata=((3000,), ()),
        num_computed_tokens=[3100],
        prompt_lens=[3000],
    )
    assert result.action.value == "staged"
    assert type(result.cold_compact_resumes) is tuple
