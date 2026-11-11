# SPDX-License-Identifier: Apache-2.0
"""Real host route selection and INFO serialization, with NPU imports isolated."""

import ast
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import test_sfa_full_graph_routing as route_tests
import torch
from sfa_test_support import load_module

routing = route_tests.routing

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def diagnostics(monkeypatch, caplog):
    log = logging.getLogger("test.sfa_route")
    caplog.set_level(logging.INFO, logger=log.name)
    monkeypatch.setenv("PD_SERVING_PERF", "0")
    monkeypatch.setitem(sys.modules, "vllm.logger", NS(logger=log))
    monkeypatch.setitem(sys.modules, "vllm_ascend", NS(envs=NS(PD_SERVING_PERF="0")))
    perf = load_module(ROOT / "vllm_ascend/serving_perf.py", "vllm_ascend.serving_perf", monkeypatch)
    assert not perf.cold_perf_enabled()
    # Unwrapping is a connector operation, independently covered by route tests.
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.attention.utils", NS(unwrap_staged_sfa_connector_metadata=lambda value: value)
    )
    module = load_module(ROOT / "vllm_ascend/worker/sfa_route_diagnostics.py", "route_diagnostics", monkeypatch)
    now = [100.0]
    monkeypatch.setattr(module, "time", NS(monotonic=lambda: now[0]))
    return (
        module,
        now,
        lambda: [
            json.loads(record.message[record.message.index("{") :])
            for record in caplog.records
            if "[SFA_ROUTE]" in record.message
        ],
    )


def batch(routing, *, computed=4096, prompt=4097, state=None):
    runner, ns, _, _, states = routing
    runner.attn_state = state or states.DecodeOnly
    runner.dp_rank = 1
    runner.dsa_shrink_latent = True
    runner.input_batch = NS(
        req_ids=["resumed"],
        num_computed_tokens_cpu=np.array([computed]),
        num_prompt_tokens=np.array([prompt]),
        num_tokens_no_spec=np.array([prompt]),
    )
    metadata = NS(
        requests=[
            NS(
                req_id="resumed",
                is_sparse_decode=True,
                dsa_current_released_frontier=0,
                dsa_nonresident_frontier=0,
                load_spec=NS(can_load=False, dsa_cold_compact_resume=False),
            )
        ]
    )
    output = NS(
        kv_connector_metadata=metadata,
        scheduled_cached_reqs=NS(resumed_req_ids={"resumed"}),
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
    )
    local = runner._staged_sfa_local_route(
        num_tokens_unpadded=1,
        num_reqs=1,
        num_scheduled_tokens=np.array([1]),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=["resumed"],
        kv_connector_metadata=metadata,
        num_computed_tokens=runner.input_batch.num_computed_tokens_cpu,
        prompt_lens=runner.input_batch.num_prompt_tokens,
    )
    return runner, ns, local, output


def apply(module, runner, route, output, **kwargs):
    return module.apply_sfa_route_with_diagnostics(
        runner,
        route,
        scheduler_output=output,
        num_reqs=1,
        scheduled=[1],
        origin="execute_model",
        **kwargs,
    )


def test_perf_zero_logs_actual_offending_row_and_cold_marker(diagnostics, routing):
    module, _, records = diagnostics
    runner, _, route, output = batch(routing)
    assert apply(module, runner, route, output) is None
    event = records()[0]
    assert event["event"] == "native_route_selected"
    assert event["local_failed_checks"] == ["computed_before_prompt_without_validated_cold_resume"]
    assert event["dp_rank"] == 1 and event["pid"] > 0 and event["wall_time_ns"] > 0
    row = event["rows"][0]
    assert row["resumed_request"] and row["request_id"] == "resumed"
    assert row["computed_tokens"] == 4096 and row["prompt_tokens"] == 4097
    assert row["computed_before_prompt_without_retained_marker"]
    assert not row["connector"][0]["dsa_cold_compact_resume"]


def test_not_decode_disambiguates_attention_state(diagnostics, routing):
    module, _, records = diagnostics
    runner, _, route, output = batch(routing, computed=4097, state=routing[-1].ChunkedPrefill)
    apply(module, runner, route, output)
    assert records()[0]["failed_checks"] == ["attention_state_not_decode"]


def test_early_not_decode_preserves_underlying_connector_rejection(diagnostics, routing):
    module, _, records = diagnostics
    ns = routing[1]
    ns["staged_sfa_metadata_sparse_route"] = lambda *args: (ns["StagedSFARouteReason"].INVALID_FRONTIER, (), ())
    runner, _, route, output = batch(routing)
    # Preserve the existing ordering: the prompt-row guard selects native.
    assert route.reason == ns["StagedSFARouteReason"].NOT_DECODE
    apply(module, runner, route, output)
    assert "connector_validation:invalid_frontier" in records()[0]["failed_checks"]


def test_repeat_summary_changed_batch_and_new_episode(diagnostics, routing, monkeypatch):
    module, now, records = diagnostics
    runner, _, route, output = batch(routing)
    original = module._snapshot
    snapshot = Mock(wraps=original)
    monkeypatch.setattr(module, "_snapshot", snapshot)
    for _ in range(4):
        apply(module, runner, route, output)
    assert len(records()) == snapshot.call_count == 1
    now[0] += 5
    apply(module, runner, route, output)
    assert records()[-1]["suppressed_steps"] == 3
    runner.input_batch.req_ids = ["other"]
    apply(module, runner, route, output)
    assert records()[-1]["request_ids"] == ["other"]
    runner._sfa_route_diagnostics.restored()
    assert records()[-1]["event"] == "graph_route_restored"
    assert records()[-1]["non_staged_route_steps"] == 6
    apply(module, runner, route, output)
    assert records()[-1]["episode"] == 2


def test_peer_override_logs_local_eligibility_separately(diagnostics, routing):
    module, _, records = diagnostics
    runner, ns, _, output = batch(routing, computed=4097)
    local = ns["StagedSFARouteDecision"](ns["StagedSFARouteAction"].STAGED, ns["StagedSFARouteReason"].ELIGIBLE)
    route = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=ns["StagedSFARouteAction"].SAFE_NATIVE,
        cudagraph_mode="none",
        batch_descriptor=NS(num_tokens=2),
        num_tokens_unpadded=1,
        num_tokens_padded=2,
        num_reqs=1,
        should_ubatch=False,
    )
    apply(module, runner, route, output, local_route=local, dp_route_action=ns["StagedSFARouteAction"].SAFE_NATIVE)
    event = records()[0]
    assert event["local_action"] == "staged" and event["local_reason"] == "eligible"
    assert event["failed_checks"] == ["dp_agreed_non_staged_action"]


@pytest.mark.parametrize("combined", [False, True])
def test_dp_votes_identify_peer_without_new_collectives_or_stale_buffers(diagnostics, routing, monkeypatch, combined):
    module, _, records = diagnostics
    runner, ns, _, output = batch(routing, computed=4097)
    actions = tuple(ns["StagedSFARouteAction"])
    staged, native = actions[:2]
    runner.dp_size, runner.dp_rank = 4, 0
    runner._dp_batch_sync_buffers = {}
    runner._skip_all_reduce_across_dp_group = lambda: False
    ns.update(
        torch=torch,
        _STAGED_SFA_ROUTE_ACTIONS=actions,
        get_dp_group=lambda: NS(cpu_group="dp"),
        _post_process_cudagraph_mode=lambda tensor: int(tensor[1].min()),
    )
    proposal = torch.tensor([[2, 8, 4, 2], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
    ns["dist"] = NS(all_reduce=Mock(side_effect=lambda tensor, group: tensor.copy_(proposal)))
    runner._sfa_full_graph = NS(sealed=True)
    vote = Mock(side_effect=lambda *args, **kwargs: proposal.clone())
    if combined:
        runner._sfa_single_agreement = NS(active=True, vote=vote, counts=torch.empty(4, dtype=torch.int32))
    _, _, mode, action = runner._sync_batch_across_dp(2, 1, True, staged)
    assert mode == 0 and action == native
    assert runner._staged_sfa_diagnostic_peer_votes["tokens_before_padding"] == [2, 8, 4, 2]
    route = ns["StagedSFARouteDecision"](native, ns["StagedSFARouteReason"].RUNTIME_PARALLELISM)
    apply(module, runner, route, output)
    assert records()[0]["dp_peer_votes"]["actions"] == [
        "staged_or_neutral",
        "safe_native",
        "staged_or_neutral",
        "staged_or_neutral",
    ]
    # A fresh all-graph agreement must clear the old peer attribution and make
    # no diagnostic tensor-to-list conversion (all tensors here are CPU).
    proposal[1].fill_(1)
    proposal[2:].zero_()
    monkeypatch.setattr(torch.Tensor, "tolist", lambda *args: pytest.fail("healthy agreement copied diagnostic data"))
    runner._sync_batch_across_dp(2, 1, True, staged)
    assert runner._staged_sfa_diagnostic_peer_votes is None
    assert (vote.call_count, ns["dist"].all_reduce.call_count) == ((2, 0) if combined else (0, 2))


def test_disabled_sfa_does_not_snapshot_ordinary_dp_mode_changes(routing, monkeypatch):
    runner, ns, env, _, _ = routing
    env.VLLM_ASCEND_SFA_FULL_GRAPH = False
    runner._staged_sfa_graph_capture_sizes = ()
    runner.dp_size, runner.dp_rank = 2, 0
    runner._dp_batch_sync_buffers = {}
    runner._skip_all_reduce_across_dp_group = lambda: False
    runner._staged_sfa_diagnostic_peer_votes = {"old": True}

    def reduce(tensor, group):
        assert tensor.shape == (2, 2)
        tensor[0].fill_(2)
        tensor[1] = torch.tensor([2, 1], dtype=torch.int32)

    ns.update(
        torch=torch,
        get_dp_group=lambda: NS(cpu_group="dp"),
        dist=NS(all_reduce=reduce),
        _post_process_cudagraph_mode=lambda tensor: int(tensor[1].min()),
    )
    monkeypatch.setattr(torch.Tensor, "tolist", lambda *args: pytest.fail("disabled SFA copied diagnostic votes"))
    assert runner._sync_batch_across_dp(2, 2, True, None)[2] == 1
    assert runner._staged_sfa_diagnostic_peer_votes is None


@pytest.mark.parametrize("field,value", [("uniform", True), ("has_lora", True), ("num_active_loras", 3)])
def test_real_descriptor_flag_mismatch_identifies_exact_field(diagnostics, routing, field, value):
    module, _, records = diagnostics
    runner, ns, local, output = batch(routing, computed=4097)

    @dataclass(frozen=True)
    class Descriptor:
        num_tokens: int
        num_reqs: int | None = None
        uniform: bool = False
        has_lora: bool = False
        num_active_loras: int = 0

    ns["BatchDescriptor"] = Descriptor
    route = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=local.action,
        cudagraph_mode=routing[3].PIECEWISE,
        batch_descriptor=Descriptor(2, **{field: value}),
        num_tokens_unpadded=1,
        num_tokens_padded=2,
        num_reqs=1,
        should_ubatch=False,
    )
    assert route.reason == ns["StagedSFARouteReason"].BATCH_DESCRIPTOR
    with pytest.raises(RuntimeError, match="cannot use a native fallback"):
        apply(module, runner, route, output, local_route=local)
    assert f"descriptor_field:{field}" in records()[0]["failed_checks"]


def test_rejected_route_is_logged_but_never_claimed_as_native_execution(diagnostics, routing):
    module, _, records = diagnostics
    runner, ns, _, output = batch(routing)
    route = ns["StagedSFARouteDecision"](
        ns["StagedSFARouteAction"].SAFE_NATIVE,
        ns["StagedSFARouteReason"].COLD_COMPACT_LAYOUT,
        failed_checks=("frontier_computed[0]",),
    )
    with pytest.raises(RuntimeError, match="cannot use a native fallback"):
        apply(module, runner, route, output)
    assert records()[0]["event"] == "route_rejected"
    assert records()[0]["failed_checks"] == ["frontier_computed[0]"]


def test_snapshot_precedes_async_reconciliation(diagnostics, routing):
    module, _, records = diagnostics
    runner, _, route, output = batch(routing)
    runner._async_pending = object()
    original = runner._apply_staged_sfa_route

    def reconcile(value):
        result = original(value)
        runner.input_batch.num_computed_tokens_cpu[0] = 999
        return result

    runner._apply_staged_sfa_route = reconcile
    apply(module, runner, route, output)
    assert records()[0]["async_acceptance_pending"]
    assert records()[0]["rows"][0]["computed_tokens"] == 4096


@pytest.mark.parametrize("operand", [None, np.array([[1, 2]])])
def test_malformed_host_operand_cannot_mask_original_rejection(diagnostics, routing, operand):
    module, _, records = diagnostics
    runner, ns, _, output = batch(routing)
    runner.input_batch.num_tokens_no_spec = operand
    route = ns["StagedSFARouteDecision"](
        ns["StagedSFARouteAction"].SAFE_NATIVE,
        ns["StagedSFARouteReason"].COLD_COMPACT_LAYOUT,
        failed_checks=("history_shape",),
    )
    with pytest.raises(RuntimeError, match="cannot use a native fallback"):
        apply(module, runner, route, output)
    assert records()[0]["failed_checks"] == ["history_shape"]
    assert records()[0]["rows"][0]["history_tokens"] is None


def test_malformed_connector_scalar_cannot_turn_native_selection_into_failure(diagnostics, routing):
    module, _, records = diagnostics
    runner, _, route, output = batch(routing)
    cycle = []
    cycle.append(cycle)
    output.kv_connector_metadata.requests[0].load_spec.dsa_committed_end = cycle
    assert apply(module, runner, route, output) is None
    assert records()[0]["rows"][0]["connector"][0]["dsa_committed_end"] == "<non-scalar list>"


def test_failed_marker_is_not_described_as_a_validated_resume(diagnostics, routing):
    module, _, records = diagnostics
    ns = routing[1]
    ns["staged_sfa_metadata_sparse_route"] = lambda *args: (ns["StagedSFARouteReason"].ELIGIBLE, (8191,), (True,))
    runner, _, route, output = batch(routing, computed=8192, prompt=8193)
    assert "frontier_computed[0]" in route.failed_checks
    with pytest.raises(RuntimeError, match="cannot use a native fallback"):
        apply(module, runner, route, output)
    row = records()[0]["rows"][0]
    assert "validated_cold_resume" not in row
    assert row["retained_cold_resume_marker"] is True


def test_unknown_connector_fields_never_format_or_read_back_device_data(diagnostics, routing):
    module, _, records = diagnostics
    runner, _, route, output = batch(routing)

    class NoReadback:
        def __str__(self):
            pytest.fail("tensor formatting synchronizes the device")

        __repr__ = __str__
        __int__ = __str__

    output.kv_connector_metadata.requests[0].load_spec.dsa_committed_end = NoReadback()
    apply(module, runner, route, output)
    assert records()[0]["rows"][0]["connector"][0]["dsa_committed_end"] == "<non-JSON value>"


def test_malformed_scheduled_shape_and_duplicate_metadata_still_log(diagnostics, routing):
    module, _, records = diagnostics
    runner, _, route, output = batch(routing)
    output.kv_connector_metadata.requests *= 2
    module.apply_sfa_route_with_diagnostics(runner, route, num_reqs=1, scheduler_output=output)
    assert records()[0]["rows"][0]["scheduled_tokens"] is None
    assert records()[0]["rows"][0]["connector_main_count"] == 2


@pytest.mark.parametrize(
    "case,reason,check",
    [
        ("scales", "runtime_mode", "calculate_kv_scales"),
        ("lora", "lora", "lora_configured"),
        ("spec", "speculative_decode", "query_width_not_one_or_two"),
        ("cascade", "cascade", "cascade_attention"),
        ("capacity", "unsupported_batch", "local_batch_capacity_or_shape"),
        ("shape", "non_q1", "scheduled_query_shape_or_width"),
        ("width", "non_q1", "scheduled_query_shape_or_width"),
    ],
)
def test_local_rejecting_checks_are_visible_at_perf_zero(diagnostics, routing, case, reason, check):
    module, _, records = diagnostics
    runner, ns, _, output = batch(routing, computed=4097)
    runner.calculate_kv_scales = case == "scales"
    if case == "lora":
        runner.vllm_config.lora_config = object()
    if case == "spec":
        runner.speculative_config.num_speculative_tokens = 2
    scheduled = [1, 1] if case == "shape" else [0] if case == "width" else [1]
    route = runner._staged_sfa_local_route(
        num_tokens_unpadded=3 if case == "capacity" else 1,
        num_reqs=1,
        num_scheduled_tokens=np.array(scheduled),
        index_topk=2048,
        has_cascade_attention=case == "cascade",
        request_ids=["resumed"],
        kv_connector_metadata=output.kv_connector_metadata,
        num_computed_tokens=runner.input_batch.num_computed_tokens_cpu,
        prompt_lens=runner.input_batch.num_prompt_tokens,
    )
    assert route.reason.value == reason
    with pytest.raises(RuntimeError, match="cannot use a native fallback"):
        module.apply_sfa_route_with_diagnostics(runner, route, scheduler_output=output, num_reqs=1, scheduled=scheduled)
    event = records()[0]
    assert event["reason"] == reason and check in event["failed_checks"]
    assert event["event"] == "route_rejected"
    assert event["num_reqs"] == 1 and event["scheduled_query_lengths"] == scheduled


@pytest.mark.parametrize(
    "case,check",
    [
        ("mode", "dispatcher_mode_not_piecewise"),
        ("ubatch", "microbatching"),
        ("multiple", "padded_capacity_not_multiple_of_query_width"),
        ("bucket", "padding_capacity_or_capture_bucket"),
    ],
)
def test_live_rejecting_checks_are_visible_at_perf_zero(diagnostics, routing, case, check):
    module, _, records = diagnostics
    runner, ns, local, output = batch(routing, computed=4097)
    route = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=local.action,
        cudagraph_mode=routing[3].NONE if case == "mode" else routing[3].PIECEWISE,
        batch_descriptor=NS(num_tokens=2),
        num_tokens_unpadded=1,
        num_tokens_padded=3 if case == "multiple" else 4 if case == "bucket" else 2,
        num_reqs=1,
        should_ubatch=case == "ubatch",
    )
    with pytest.raises(RuntimeError, match="cannot use a native fallback"):
        apply(module, runner, route, output, local_route=local)
    assert records()[0]["local_action"] == "staged"
    assert check in records()[0]["failed_checks"]


def runner_dispatch_block():
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "execute_model")
    block = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and "staged_sfa_route.action" in ast.unparse(node.test)
    )
    return compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec")


def test_actual_serving_call_logs_and_healthy_graph_skips_all_diagnostic_work(diagnostics, routing):
    module, _, records = diagnostics
    runner, ns, route, output = batch(routing)
    code = runner_dispatch_block()
    scope = dict(
        self=runner,
        staged_sfa_route=route,
        staged_sfa_local_route=route,
        dp_route_action=route.action,
        scheduler_output=output,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([1]),
        cudagraph_mode="piecewise",
        batch_desc=NS(num_tokens=2, num_reqs=1, uniform=True, has_lora=True, num_active_loras=3),
        num_tokens_unpadded=1,
        num_tokens_padded=2,
        should_ubatch=False,
        cascade_attn_prefix_lens=None,
        sfa_full_graph_enabled=ns["sfa_full_graph_enabled"],
        StagedSFARouteAction=ns["StagedSFARouteAction"],
        apply_sfa_route_with_diagnostics=module.apply_sfa_route_with_diagnostics,
    )
    exec(code, scope)
    assert records()[0]["num_tokens_padded"] == 2
    assert records()[0]["descriptor_uniform"] is True
    assert records()[0]["descriptor_has_lora"] is True
    assert records()[0]["descriptor_num_active_loras"] == 3
    scope["staged_sfa_route"] = ns["StagedSFARouteDecision"](
        ns["StagedSFARouteAction"].STAGED, ns["StagedSFARouteReason"].ELIGIBLE, graph_key=object()
    )
    exec(code, scope)
    assert records()[-1]["event"] == "graph_route_restored"
    scope["apply_sfa_route_with_diagnostics"] = lambda *args, **kwargs: pytest.fail("healthy graph built diagnostics")
    runner.input_batch = None  # Any subsequent CPU metadata scan would fail.
    module.time.monotonic = lambda: pytest.fail("healthy graph read the clock")
    for _ in range(100):
        exec(code, scope)
    assert len(records()) == 2


def test_dummy_call_has_no_stale_request_rows(diagnostics, routing):
    module, _, records = diagnostics
    runner, ns, _, _ = batch(routing)
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dummy = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run")
    call = next(
        node
        for node in ast.walk(dummy)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "apply_sfa_route_with_diagnostics"
    )
    scope = dict(
        self=runner,
        fallback_action=ns["StagedSFARouteAction"].SAFE_NATIVE,
        dp_route_action=ns["StagedSFARouteAction"].STAGED,
        dp_idle=True,
        is_profile=False,
        num_tokens=1,
        num_tokens_unpadded=8,
        num_tokens_padded=8,
        num_reqs=4,
        num_scheduled_tokens=np.array([2, 2, 2, 2]),
        staged_capture=False,
        num_active_loras=0,
        batch_desc=NS(num_tokens=8, num_reqs=4, uniform=True, has_lora=False, num_active_loras=0),
        cudagraph_runtime_mode="piecewise",
        sfa_full_graph_enabled=ns["sfa_full_graph_enabled"],
        StagedSFARouteDecision=ns["StagedSFARouteDecision"],
        StagedSFARouteReason=ns["StagedSFARouteReason"],
        apply_sfa_route_with_diagnostics=module.apply_sfa_route_with_diagnostics,
    )
    exec(compile(ast.Module(body=[call], type_ignores=[]), str(path), "exec"), scope)
    event = records()[0]
    assert event["origin"] == "dummy_run" and event["dp_idle"]
    assert event["rows"] == [] and event["request_ids"] == []
    assert event["num_tokens_unpadded"] == 8
    assert event["requested_dummy_tokens"] == 1
    assert event["scheduled_query_lengths"] == [2, 2, 2, 2]
    assert event["allow_eager"] is False and event["num_active_loras"] == 0
    assert event["descriptor_uniform"] is True
