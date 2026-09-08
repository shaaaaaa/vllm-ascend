# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of the real routing methods, isolated from worker imports."""

import ast
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int


def definitions(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(body) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def routing():
    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    env = SimpleNamespace(VLLM_ASCEND_SFA_FULL_GRAPH=True, VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES="1")
    modes = SimpleNamespace(NONE="none", PIECEWISE="piecewise")
    states = SimpleNamespace(DecodeOnly="decode", SpecDecoding="spec", ChunkedPrefill="prefill")
    ns = dict(
        np=np,
        dataclass=dataclass,
        field=field,
        Enum=Enum,
        IntEnum=IntEnum,
        BatchDescriptor=BatchDescriptor,
        envs_ascend=env,
        CUDAGraphMode=modes,
        AscendAttentionState=states,
        staged_sfa_graph_configured=lambda _: True,
    )
    definitions(root / "ascend_forward_context.py", {"StagedSFAQueryProfile", "StagedSFAGraphKey"}, ns)
    definitions(
        root / "utils.py",
        {"StagedSFARouteAction", "StagedSFARouteReason", "StagedSFARouteDecision", "staged_sfa_graph_capture_sizes"},
        ns,
    )
    names = {
        "_staged_sfa_local_route",
        "_staged_sfa_live_route",
        "_staged_sfa_dummy_batch_size",
        "_apply_staged_sfa_route",
        "_build_attn_state",
    }
    definitions(root / "worker/model_runner_v1.py", names, ns, class_name="NPUModelRunner")
    runner_type = type("RunnerRouting", (), {name: ns[name] for name in names})
    runner = runner_type()
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=1)
    runner.decode_threshold = 2
    runner.vllm_config = SimpleNamespace(lora_config=None)
    runner._staged_sfa_graph_capture_sizes = (1, 2)
    ns["staged_sfa_metadata_sparse_route"] = lambda *args: (ns["StagedSFARouteReason"].ELIGIBLE, (4096,), (False,))
    return runner, ns, env, modes, states


@pytest.mark.parametrize("width", [1, 2])
def test_mtp_target_routes_both_query_widths_to_one_root(routing, width):
    runner, ns, _, modes, states = routing
    runner.attn_state = states.DecodeOnly if width == 1 else states.SpecDecoding
    local = runner._staged_sfa_local_route(
        num_tokens_unpadded=width,
        num_reqs=1,
        num_scheduled_tokens=np.array([width]),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=["r1"],
        kv_connector_metadata=None,
    )
    assert local.action == ns["StagedSFARouteAction"].STAGED
    live = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=local.action,
        cudagraph_mode=modes.PIECEWISE,
        batch_descriptor=BatchDescriptor(width),
        num_tokens_unpadded=width,
        num_tokens_padded=width,
        num_reqs=1,
        should_ubatch=False,
    )
    assert runner._apply_staged_sfa_route(live).max_query_len == width
    assert (
        runner._staged_sfa_dummy_batch_size(
            is_profile=False,
            cudagraph_runtime_mode=modes.PIECEWISE,
            allow_eager=True,
            num_active_loras=0,
            num_tokens_unpadded=width,
            num_tokens_padded=width,
            num_reqs=1,
            num_scheduled_tokens=np.array([width]),
            batch_descriptor=BatchDescriptor(width),
            dp_route_action=local.action,
        )
        == width
    )


def test_capture_sizes_include_q1_only_for_new_opt_in(routing):
    _, ns, env, _, _ = routing
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=4096),
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
    )
    assert ns["staged_sfa_graph_capture_sizes"](config) == (1, 2)
    env.VLLM_ASCEND_SFA_FULL_GRAPH = False
    assert ns["staged_sfa_graph_capture_sizes"](config) == (2,)


def test_decode_cannot_silently_fall_back(routing):
    runner, ns, _, _, states = routing
    runner.attn_state = states.SpecDecoding
    route = ns["StagedSFARouteDecision"](
        ns["StagedSFARouteAction"].SAFE_NATIVE, ns["StagedSFARouteReason"].UNSUPPORTED_BATCH
    )
    with pytest.raises(RuntimeError, match="cannot use a native fallback"):
        runner._apply_staged_sfa_route(route)


def test_mtp_single_token_attention_uses_q1_only_when_opted_in(routing):
    runner, _, env, _, states = routing
    runner.speculative_config.method = "mtp"
    runner.input_batch = SimpleNamespace(num_computed_tokens_cpu=np.array([4096]))
    runner.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    assert runner._build_attn_state(1, np.array([1]), np.array([1])) == states.DecodeOnly
    env.VLLM_ASCEND_SFA_FULL_GRAPH = False
    assert runner._build_attn_state(1, np.array([1]), np.array([1])) == states.SpecDecoding


def test_single_token_prefill_tail_is_not_misclassified_as_decode(routing):
    runner, ns, _, _, states = routing
    runner.attn_state = states.DecodeOnly
    ns["logger"] = SimpleNamespace(info=lambda *args: None)
    route = runner._staged_sfa_local_route(
        num_tokens_unpadded=1,
        num_reqs=1,
        num_scheduled_tokens=np.array([1]),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=["r1"],
        kv_connector_metadata=None,
        num_computed_tokens=np.array([4096]),
        prompt_lens=np.array([4097]),
    )
    assert route.reason == ns["StagedSFARouteReason"].NOT_DECODE
    assert runner._apply_staged_sfa_route(route) is None
