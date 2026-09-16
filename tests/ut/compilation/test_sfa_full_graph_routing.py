# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of the real routing methods, isolated from worker imports."""

import ast
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


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
    modes = SimpleNamespace(NONE="none", PIECEWISE="piecewise", FULL="full")
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
        has_kv_transfer_group=lambda: False,
        cold_perf_enabled=lambda: False,
        parent_process=lambda: object(),  # Fixture represents a worker child.
    )
    definitions(root / "attention/utils.py", {"unwrap_staged_sfa_connector_metadata", "ColdResumeMarkers"}, ns)
    definitions(root / "compilation/sfa_fail_stop.py", {"uses_local_sfa_fail_stop"}, ns)
    definitions(root / "ascend_forward_context.py", {"StagedSFAQueryProfile", "StagedSFAGraphKey"}, ns)
    definitions(
        root / "utils.py",
        {
            "StagedSFARouteAction",
            "StagedSFARouteReason",
            "StagedSFARouteDecision",
            "staged_sfa_graph_capture_sizes",
            "sfa_full_graph_enabled",
        },
        ns,
    )
    names = {
        "_staged_sfa_local_route",
        "_staged_sfa_live_route",
        "_staged_sfa_dummy_batch_size",
        "_apply_staged_sfa_route",
        "_build_attn_state",
        "_pad_query_start_loc_for_fia",
        "_validate_sfa_layerwise_connector_cudagraph_mode",
        "_sync_batch_across_dp",
        "_determine_batch_execution_and_padding",
        "_coordinate_sfa_full_graph_preparation",
        "_model_forward",
        "_staged_sfa_dummy_seq_len",
        "_prepare_staged_sfa_dummy_block_tables",
        "_staged_sfa_capture_keys",
        "_staged_sfa_dummy_graph_key",
        "_warmup_and_capture",
    }
    definitions(root / "worker/model_runner_v1.py", names, ns, class_name="NPUModelRunner")
    runner_type = type("RunnerRouting", (), {name: ns[name] for name in names})
    runner = runner_type()
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=1)
    runner.decode_threshold = 2
    runner.vllm_config = SimpleNamespace(lora_config=None, model_config=SimpleNamespace(enforce_eager=False))
    runner.model_config = runner.vllm_config.model_config
    runner.parallel_config = SimpleNamespace(data_parallel_size=4)
    runner.vllm_config.parallel_config = runner.parallel_config
    # Local-layout tests represent an already agreed all-uniform DP cohort.
    # Protocol tests below execute the actual collective method instead.
    runner._staged_sfa_dp_bounded_decode = False
    runner._staged_sfa_graph_capture_sizes = (2,)
    ns["logger"] = SimpleNamespace(info=lambda *args: None)
    ns["staged_sfa_metadata_sparse_route"] = lambda *args: (ns["StagedSFARouteReason"].ELIGIBLE, (4096,), (False,))
    return runner, ns, env, modes, states


@pytest.fixture
def dummy_capture(routing):
    runner, ns, _, _, _ = routing
    runner.max_model_len = 4399  # parity driver: 4351 prompt + 16 output + 32 margin
    runner.kv_cache_config = SimpleNamespace(num_blocks=4, num_blocks_per_group=[4, 4])

    def table(width, block_size):
        result = SimpleNamespace(
            max_num_reqs=4,
            max_num_blocks_per_req=width,
            block_size=block_size,
            block_table=SimpleNamespace(np=np.zeros((4, width), dtype=np.int32)),
            num_blocks_per_row=np.zeros(4, dtype=np.int32),
            slot_mapping=SimpleNamespace(np=np.full(8, -1, dtype=np.int64)),
            commit_block_table=Mock(),
            commit_slot_mapping=Mock(),
        )

        def map_slots(requests, positions):
            # Simple CP=1 table storage; execute the actual runner's validation,
            # row assignment and mapping calls, without initializing a device.
            blocks = result.block_table.np[requests, positions // block_size]
            result.slot_mapping.np[: positions.size] = blocks * block_size + positions % block_size

        result.compute_slot_mapping = map_slots
        return result

    runner.input_batch = SimpleNamespace(block_table=SimpleNamespace(block_tables=[table(36, 128), table(18, 256)]))
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dummy = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run")
    # Run the real length-selection branch from _dummy_run, not a duplicated
    # implementation. This catches fixing the helper but forgetting the caller.
    selection = next(
        node
        for node in ast.walk(dummy)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "staged_sfa_graph_dummy_run"
        and any(
            isinstance(child, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "seq_lens" for target in child.targets)
            for child in node.body
        )
    )
    code = compile(ast.fix_missing_locations(ast.Module(body=[selection], type_ignores=[])), str(path), "exec")

    def select(**overrides):
        values = dict(
            ns,
            self=runner,
            staged_sfa_graph_dummy_run=True,
            staged_query_width=2,
            profile_seq_lens=None,
            is_graph_capturing=True,
            max_query_len=2,
            num_tokens=2,
            using_paged_attention=lambda *args: True,
            SEQ_LEN_WITH_MAX_PA_WORKSPACE=6144,
        )
        values.update(overrides)
        exec(code, values)
        return values["seq_lens"]

    return runner, select


@pytest.mark.parametrize("query_width", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_short_context_capture_positions_fit_both_groups(dummy_capture, query_width, batch_size):
    runner, select = dummy_capture
    seq_len = select(staged_query_width=query_width)
    assert seq_len == 4399
    positions = np.tile(np.arange(seq_len - query_width, seq_len, dtype=np.int64), batch_size)
    runner._prepare_staged_sfa_dummy_block_tables(batch_size=batch_size, positions=positions)
    for table in runner.input_batch.block_table.block_tables:
        slots = table.slot_mapping.np[: positions.size]
        assert np.unique(slots).size == positions.size
        assert np.all((slots >= 0) & (slots < batch_size * table.block_size))
        table.commit_block_table.assert_called_once_with(batch_size, force=True)


def test_capture_length_honors_smaller_second_kv_group(dummy_capture):
    runner, select = dummy_capture
    runner.max_model_len = 10000
    table = runner.input_batch.block_table.block_tables[1]
    table.block_table.np = np.zeros((4, 12), dtype=np.int32)
    table.max_num_blocks_per_req = 12
    assert select() == 3072  # not group 0's 4608 or the default 6144
    runner._prepare_staged_sfa_dummy_block_tables(batch_size=1, positions=np.array([3070, 3071]))


def test_large_context_retains_original_capture_heuristic(dummy_capture):
    runner, select = dummy_capture
    runner.max_model_len = 10000
    for table in runner.input_batch.block_table.block_tables:
        table.block_table.np = np.zeros((4, 128), dtype=np.int32)
    assert select() == 6144


@pytest.mark.parametrize("dcp,pcp", [(2, 1), (1, 2), (2, 2)])
def test_capture_length_uses_logical_table_width_and_context_parallelism(dummy_capture, dcp, pcp):
    runner, select = dummy_capture
    runner.max_model_len = 10000
    for table in runner.input_batch.block_table.block_tables:
        table.block_table.np = np.zeros((4, 4), dtype=np.int32)
        table.max_num_blocks_per_req = 1  # kernel-split table width is authoritative
        table.dcp_world_size, table.pcp_world_size = dcp, pcp
    assert select() == 4 * 128 * dcp * pcp


@pytest.mark.parametrize("requested,expected", [(1, 2), (256, 256), (4399, 4399), (6144, 4399)])
def test_profiling_override_is_bounded_and_holds_complete_query(dummy_capture, requested, expected):
    _, select = dummy_capture
    assert select(profile_seq_lens=requested) == expected


@pytest.mark.parametrize("kind", ["model_too_short", "empty_table", "missing_group", "zero_length", "zero_query"])
def test_invalid_dummy_capacity_fails_before_mapping(dummy_capture, kind):
    runner, select = dummy_capture
    overrides = {}
    if kind == "model_too_short":
        runner.max_model_len = 1
    elif kind == "empty_table":
        runner.input_batch.block_table.block_tables[1].block_table.np = np.zeros((4, 0), dtype=np.int32)
    elif kind == "missing_group":
        runner.input_batch.block_table.block_tables.pop()
    elif kind == "zero_length":
        overrides["profile_seq_lens"] = 0
    else:
        overrides["staged_query_width"] = 0
    with pytest.raises((ValueError, RuntimeError), match="positive|two KV|capacity"):
        select(**overrides)


def test_original_out_of_range_dummy_positions_still_fail(dummy_capture):
    runner, _ = dummy_capture
    with pytest.raises(RuntimeError, match=r"max_position=6143, logical_capacity=4608"):
        runner._prepare_staged_sfa_dummy_block_tables(batch_size=1, positions=np.array([6142, 6143]))


@pytest.mark.parametrize("profile,capturing,expected", [(None, True, 6144), (None, False, 2), (123, True, 123)])
def test_non_sfa_paged_attention_and_profile_lengths_are_unchanged(dummy_capture, profile, capturing, expected):
    _, select = dummy_capture
    assert select(staged_sfa_graph_dummy_run=False, profile_seq_lens=profile, is_graph_capturing=capturing) == expected


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
        batch_descriptor=BatchDescriptor(2),
        num_tokens_unpadded=width,
        num_tokens_padded=2,
        num_reqs=1,
        should_ubatch=False,
    )
    assert runner._apply_staged_sfa_route(live).max_query_len == 2
    assert live.graph_key.query_profile == (
        ns["StagedSFAQueryProfile"].SPEC_FIXED if width == 2 else ns["StagedSFAQueryProfile"].DECODE_BOUNDED
    )
    assert (
        runner._staged_sfa_dummy_batch_size(
            is_profile=False,
            cudagraph_runtime_mode=modes.PIECEWISE,
            allow_eager=True,
            num_active_loras=0,
            num_tokens_unpadded=2,
            num_tokens_padded=2,
            num_reqs=1,
            num_scheduled_tokens=np.array([2]),
            batch_descriptor=BatchDescriptor(2),
            dp_route_action=local.action,
        )
        == 2
    )


def test_capture_token_capacities_stay_shared_between_profiles(routing):
    _, ns, env, _, _ = routing
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=4096),
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
    )
    assert ns["staged_sfa_graph_capture_sizes"](config) == (2,)
    env.VLLM_ASCEND_SFA_FULL_GRAPH = False
    assert ns["staged_sfa_graph_capture_sizes"](config) == (2,)


@pytest.mark.parametrize("full_graph,width", [(True, 1), (True, 2), (False, 2)])
@pytest.mark.parametrize("configured", [False, True])
def test_cold_resume_keeps_markers_and_mtp_route_after_merge(routing, full_graph, width, configured):
    runner, ns, env, modes, states = routing
    runner.input_batch = SimpleNamespace(num_tokens_no_spec=np.array([8193]))
    env.VLLM_ASCEND_SFA_FULL_GRAPH = full_graph
    runner.attn_state = states.DecodeOnly if width == 1 else states.SpecDecoding
    runner._staged_sfa_graph_capture_sizes = (2,) if configured else ()
    metadata = object()
    wrapped = SimpleNamespace(child=metadata)
    unwrap = Mock(side_effect=lambda value: value.child)
    ns.update(
        has_kv_transfer_group=lambda: True,
        is_v1_kv_transfer_group=lambda: True,
        get_kv_transfer_group=lambda: SimpleNamespace(_unwrap_staged_sfa_connector_metadata=unwrap),
    )
    route_metadata = Mock(return_value=(ns["StagedSFARouteReason"].ELIGIBLE, (8192,), (True,)))
    ns["staged_sfa_metadata_sparse_route"] = route_metadata
    local = runner._staged_sfa_local_route(
        num_tokens_unpadded=width,
        num_reqs=1,
        num_scheduled_tokens=np.array([width]),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=["cold"],
        kv_connector_metadata=wrapped,
        num_computed_tokens=np.array([8192]),
        prompt_lens=np.array([8193]),
    )
    unwrap.assert_called_once_with(wrapped)
    route_metadata.assert_called_once_with(metadata, ["cold"])
    assert local.frontiers == (8192,)
    assert local.cold_compact_resumes == (True,)
    if not configured:
        assert local.action == ns["StagedSFARouteAction"].SAFE_NATIVE
        assert local.reason == ns["StagedSFARouteReason"].NOT_CONFIGURED
        return
    assert local.action == ns["StagedSFARouteAction"].STAGED
    live = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=local.action,
        cudagraph_mode=modes.PIECEWISE,
        batch_descriptor=BatchDescriptor(2),
        num_tokens_unpadded=width,
        num_tokens_padded=2,
        num_reqs=1,
        should_ubatch=False,
    )
    expected_profile = (
        ns["StagedSFAQueryProfile"].DECODE_BOUNDED if width == 1 else ns["StagedSFAQueryProfile"].SPEC_FIXED
    )
    assert live.graph_key.query_profile == expected_profile
    assert live.cold_compact_resumes == (True,)


@pytest.mark.parametrize("failure", ["frontier", "computed", "missing_lengths", "resume_count"])
def test_full_graph_rejects_invalid_cold_resume_markers(routing, failure):
    runner, ns, _, _, states = routing
    runner.input_batch = SimpleNamespace(num_tokens_no_spec=np.array([8193]))
    runner.attn_state = states.SpecDecoding
    frontiers = (8191,) if failure == "frontier" else (8192,)
    resumes = (True, True) if failure == "resume_count" else (True,)
    ns["staged_sfa_metadata_sparse_route"] = lambda *args: (ns["StagedSFARouteReason"].ELIGIBLE, frontiers, resumes)
    route = runner._staged_sfa_local_route(
        num_tokens_unpadded=2,
        num_reqs=1,
        num_scheduled_tokens=np.array([2]),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=["cold"],
        kv_connector_metadata=None,
        num_computed_tokens=None
        if failure == "missing_lengths"
        else np.array([8191 if failure == "computed" else 8192]),
        prompt_lens=np.array([8193]),
    )
    assert route.reason == ns["StagedSFARouteReason"].COLD_COMPACT_LAYOUT
    assert route.action == ns["StagedSFARouteAction"].SAFE_NATIVE
    # Full graph must fail closed, not silently introduce per-layer splits.
    with pytest.raises(RuntimeError, match="cannot use a native fallback"):
        runner._apply_staged_sfa_route(route)


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


@pytest.mark.parametrize("capacity", [4, 8, 12, 16])
@pytest.mark.parametrize("pattern", ["q1", "q2", "mixed"])
def test_dp4_bounded_decode_buckets(routing, capacity, pattern):
    runner, ns, env, modes, states = routing
    env.VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES = "4,8,12,16"
    runner.vllm_config.scheduler_config = SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=4096)
    runner.vllm_config.speculative_config = runner.speculative_config
    runner._staged_sfa_graph_capture_sizes = ns["staged_sfa_graph_capture_sizes"](runner.vllm_config)
    assert runner._staged_sfa_graph_capture_sizes == (8, 16, 24, 32)
    count = capacity - 1
    widths = np.array([1 if pattern == "q1" or (pattern == "mixed" and i % 2) else 2 for i in range(count)])
    runner.attn_state = states.DecodeOnly if pattern == "q1" else states.SpecDecoding
    ns["staged_sfa_metadata_sparse_route"] = lambda *args: (
        ns["StagedSFARouteReason"].ELIGIBLE,
        (4096,) * count,
        (False,) * count,
    )
    local = runner._staged_sfa_local_route(
        num_tokens_unpadded=int(widths.sum()),
        num_reqs=count,
        num_scheduled_tokens=widths,
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=[str(i) for i in range(count)],
        kv_connector_metadata=None,
    )
    assert local.action == ns["StagedSFARouteAction"].STAGED
    live = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=local.action,
        cudagraph_mode=modes.PIECEWISE,
        batch_descriptor=BatchDescriptor(capacity * 2),
        num_tokens_unpadded=int(widths.sum()),
        num_tokens_padded=capacity * 2,
        num_reqs=count,
        should_ubatch=False,
    )
    key = runner._apply_staged_sfa_route(live)
    expected = ns["StagedSFAGraphKey"].fixed_spec if pattern == "q2" else ns["StagedSFAGraphKey"].bounded_decode
    assert key == expected(capacity, 2)


@pytest.mark.parametrize("count", [1, 2, 8, 64])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("pattern", ["q2", "q1", "mixed"])
def test_layout_classification_uses_every_request_not_mtp_toggle(routing, count, padded, pattern):
    runner, ns, _, modes, states = routing
    capacity = count + int(padded)
    runner._staged_sfa_graph_capture_sizes = (capacity * 2,)
    widths = np.full(count, 2, dtype=np.int32)
    if pattern == "q1":
        widths[:] = 1
    elif pattern == "mixed":
        widths[-1] = 1  # A late Q1 must not be hidden by the first request.
    runner.attn_state = states.SpecDecoding if np.any(widths == 2) else states.DecodeOnly
    ns["staged_sfa_metadata_sparse_route"] = lambda *args: (
        ns["StagedSFARouteReason"].ELIGIBLE,
        (4096,) * count,
        (False,) * count,
    )
    local = runner._staged_sfa_local_route(
        num_tokens_unpadded=int(widths.sum()),
        num_reqs=count,
        num_scheduled_tokens=widths,
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=list(map(str, range(count))),
        kv_connector_metadata=None,
    )
    assert local.uniform_query_len == (2 if pattern == "q2" else 0)
    live = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=local.action,
        cudagraph_mode=modes.PIECEWISE,
        batch_descriptor=BatchDescriptor(capacity * 2),
        num_tokens_unpadded=int(widths.sum()),
        num_tokens_padded=capacity * 2,
        num_reqs=count,
        should_ubatch=False,
    )
    key = runner._apply_staged_sfa_route(live)
    assert key in runner._staged_sfa_capture_keys(capacity * 2)
    assert key.query_profile == (
        ns["StagedSFAQueryProfile"].SPEC_FIXED if pattern == "q2" else ns["StagedSFAQueryProfile"].DECODE_BOUNDED
    )


@pytest.mark.parametrize("widths", [[2, 0], [2, 3], [2, -1], [2], [2, 2, 2]])
def test_invalid_scheduled_widths_cannot_select_uniform(routing, widths):
    runner, ns, _, _, states = routing
    runner.attn_state = states.SpecDecoding
    runner._staged_sfa_graph_capture_sizes = (4,)
    local = runner._staged_sfa_local_route(
        num_tokens_unpadded=4,
        num_reqs=2,
        num_scheduled_tokens=np.array(widths),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=["a", "b"],
        kv_connector_metadata=None,
    )
    assert local.reason == ns["StagedSFARouteReason"].NON_Q1
    assert local.uniform_query_len == 0


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("full", [False, True])
def test_capture_warms_both_profiles_and_restores_selector(routing, width, full):
    runner, ns, env, modes, _ = routing
    env.VLLM_ASCEND_SFA_FULL_GRAPH = full
    runner.decode_threshold = width
    runner._staged_sfa_graph_capture_sizes = (8 * width,)
    captured = []

    def parent_capture(owner, desc, **kwargs):
        # The parent calls dummy_run for warmup then capture. Both must select
        # the SAME topology, including graph-memory profiling's seq-len override.
        warm = owner._staged_sfa_dummy_graph_key(desc.num_tokens, dp_idle=False)
        capture = owner._staged_sfa_dummy_graph_key(desc.num_tokens, dp_idle=False)
        assert warm == capture
        assert kwargs["profile_seq_lens"] == 6144
        captured.append(capture)

    ns["GPUModelRunner"] = SimpleNamespace(_warmup_and_capture=parent_capture)
    runner._warmup_and_capture(
        SimpleNamespace(num_tokens=8 * width, uniform=False), modes.PIECEWISE, profile_seq_lens=6144
    )
    assert tuple(captured) == runner._staged_sfa_capture_keys(8 * width)
    assert len(captured) == (2 if full else 1)
    assert len(set(captured)) == len(captured)
    assert getattr(runner, "_sfa_capture_graph_key", None) is None
    if full:
        runner._sfa_capture_graph_key = captured[0]
        assert runner._staged_sfa_dummy_graph_key(8 * width, dp_idle=True) == captured[1]
        with pytest.raises(RuntimeError, match="capacity"):
            runner._staged_sfa_dummy_graph_key(9 * width, dp_idle=False)


def test_capture_failure_restores_selector(routing):
    runner, ns, _, modes, _ = routing
    ns["GPUModelRunner"] = SimpleNamespace(_warmup_and_capture=Mock(side_effect=RuntimeError("capture error")))
    with pytest.raises(RuntimeError, match="capture error"):
        runner._warmup_and_capture(SimpleNamespace(num_tokens=2, uniform=False), modes.PIECEWISE)
    assert runner._sfa_capture_graph_key is None


@pytest.mark.parametrize("kind", ["not_configured", "uniform_descriptor", "other_mode"])
def test_non_sfa_capture_delegates_once(routing, kind):
    runner, ns, _, modes, _ = routing
    parent = Mock(return_value="original")
    ns["GPUModelRunner"] = SimpleNamespace(_warmup_and_capture=parent)
    if kind == "not_configured":
        runner._staged_sfa_graph_capture_sizes = ()
    desc = SimpleNamespace(num_tokens=2, uniform=kind == "uniform_descriptor")
    assert runner._warmup_and_capture(desc, modes.FULL if kind == "other_mode" else modes.PIECEWISE) == "original"
    parent.assert_called_once()
    assert not hasattr(runner, "_sfa_capture_graph_key")


def test_no_mtp_uses_original_q1_layout(routing):
    runner, ns, _, modes, states = routing
    runner.speculative_config = None
    runner.decode_threshold = 1
    runner._staged_sfa_graph_capture_sizes = (8,)
    runner.attn_state = states.DecodeOnly
    ns["staged_sfa_metadata_sparse_route"] = lambda *args: (
        ns["StagedSFARouteReason"].ELIGIBLE,
        (4096,) * 8,
        (False,) * 8,
    )
    local = runner._staged_sfa_local_route(
        num_tokens_unpadded=8,
        num_reqs=8,
        num_scheduled_tokens=np.ones(8, dtype=np.int32),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=list(map(str, range(8))),
        kv_connector_metadata=None,
    )
    live = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=local.action,
        cudagraph_mode=modes.PIECEWISE,
        batch_descriptor=BatchDescriptor(8),
        num_tokens_unpadded=8,
        num_tokens_padded=8,
        num_reqs=8,
        should_ubatch=False,
    )
    assert live.graph_key == ns["StagedSFAGraphKey"].exact_q1(8)


def test_live_runner_passes_bounded_flag_only_to_ragged_metadata(routing):
    runner, ns, _, _, _ = routing
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    expressions = [
        kw.value
        for n in ast.walk(method)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("_pad_query_start_loc_for_fia", "_build_attention_metadata")
        for kw in n.keywords
        if kw.arg == "full_graph"
    ]
    assert len(expressions) == 2
    for key, expected in zip((*runner._staged_sfa_capture_keys(8), None), (False, True, False)):
        ns["staged_sfa_graph_key"] = key
        for expr in expressions:
            assert eval(compile(ast.Expression(expr), str(path), "eval"), ns) is expected


def test_seal_uses_both_captured_profiles(routing):
    runner, ns, _, _, _ = routing
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "capture_model")
    expr = next(
        n.value
        for n in ast.walk(method)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "graph_keys" for t in n.targets)
    )
    ns.update(self=runner, capture_sizes=(2, 8, 16))
    keys = eval(compile(ast.Expression(expr), str(path), "eval"), ns)
    assert keys == tuple(k for size in (2, 8, 16) for k in runner._staged_sfa_capture_keys(size))
    assert len(set(keys)) == 6
    draft = next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "seal_staged_mtp_draft_graphs"
    )
    ns["graph_keys"] = keys
    assert eval(compile(ast.Expression(draft.args[0]), str(path), "eval"), ns) == (1, 4, 8)


@pytest.mark.parametrize("count,capacity", [(1, 1), (1, 4), (8, 8), (8, 16), (64, 64)])
def test_uniform_query_padding_only_uploads_when_padding_needed(routing, count, capacity):
    runner, _, _, modes, _ = routing
    data = np.full(capacity + 1, -99, dtype=np.int32)
    data[: count + 1] = np.arange(count + 1) * 2
    upload = Mock()
    runner.uniform_decode_query_len = 2
    runner.arange_np = np.arange(capacity + 1)
    runner.query_start_loc = SimpleNamespace(np=data, copy_to_gpu=upload)
    assert runner._pad_query_start_loc_for_fia(capacity * 2, count, count, modes.PIECEWISE, capacity) == capacity
    assert data.tolist() == (np.arange(capacity + 1) * 2).tolist()
    assert upload.call_count == int(count < capacity)


def test_p_worker_explicit_eager_ignores_global_full_flag(routing):
    runner, ns, _, _, _ = routing
    runner.model_config.enforce_eager = True
    assert not ns["sfa_full_graph_enabled"](runner.vllm_config)
    runner._validate_sfa_layerwise_connector_cudagraph_mode()


def test_query_padding_preserves_last_real_request(routing):
    runner, _, _, modes, _ = routing
    data = np.array([0, 1, 3, 4, -99], dtype=np.int32)
    runner.query_start_loc = SimpleNamespace(np=data, copy_to_gpu=lambda: None)
    assert runner._pad_query_start_loc_for_fia(8, 3, 3, modes.PIECEWISE, 4, full_graph=True) == 4
    assert data.tolist() == [0, 1, 3, 4, 4]


def test_dp_peer_prefill_uses_coordinated_native_route(routing):
    runner, ns, _, modes, states = routing
    runner.attn_state = states.SpecDecoding
    local = ns["StagedSFARouteDecision"](ns["StagedSFARouteAction"].STAGED, ns["StagedSFARouteReason"].ELIGIBLE)
    route = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=ns["StagedSFARouteAction"].SAFE_NATIVE,
        cudagraph_mode=modes.NONE,
        batch_descriptor=BatchDescriptor(8),
        num_tokens_unpadded=2,
        num_tokens_padded=8,
        num_reqs=1,
        should_ubatch=False,
    )
    assert route.reason == ns["StagedSFARouteReason"].RUNTIME_PARALLELISM
    assert runner._apply_staged_sfa_route(route) is None


def test_full_graph_overrides_recompute_dp_sync_bypass(routing):
    runner, ns, _, _, _ = routing
    actions = tuple(ns["StagedSFARouteAction"])
    runner.dp_size, runner.dp_rank = 4, 0
    runner._dp_batch_sync_buffers = {}
    runner._skip_all_reduce_across_dp_group = lambda: True
    ns["torch"] = torch
    ns["_STAGED_SFA_ROUTE_ACTIONS"] = actions
    ns["get_dp_group"] = lambda: SimpleNamespace(cpu_group="dp")
    ns["_post_process_cudagraph_mode"] = lambda tensor: int(tensor[1].min())

    def all_reduce(tensor, group):
        assert group == "dp"
        tensor[0] = torch.tensor([8, 16, 24, 32])
        tensor[1].fill_(1)
        tensor[2].fill_(actions.index(ns["StagedSFARouteAction"].STAGED))

    ns["dist"] = SimpleNamespace(all_reduce=Mock(side_effect=all_reduce))
    _, sizes, mode, action = runner._sync_batch_across_dp(
        8,
        1,
        True,
        ns["StagedSFARouteAction"].STAGED,
    )
    ns["dist"].all_reduce.assert_called_once()
    assert sizes.tolist() == [32, 32, 32, 32]
    assert mode == 1 and action == ns["StagedSFARouteAction"].STAGED


def test_active_q2_and_idle_dp_peer_select_the_same_full_graph(routing):
    runner, ns, _, modes, _ = routing
    runner._staged_sfa_graph_capture_sizes = (8,)
    # The idle peer needs private, zero-length attention tables. Its vote must
    # also select the bounded graph on the active Q2 peer, not just its size.
    runner._staged_sfa_dp_bounded_decode = True
    local = ns["StagedSFARouteDecision"](
        ns["StagedSFARouteAction"].STAGED,
        ns["StagedSFARouteReason"].ELIGIBLE,
        uniform_query_len=2,
    )
    active = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=local.action,
        cudagraph_mode=modes.PIECEWISE,
        batch_descriptor=BatchDescriptor(8),
        num_tokens_unpadded=2,
        num_tokens_padded=8,
        num_reqs=1,
        should_ubatch=False,
    )
    idle = runner._staged_sfa_dummy_graph_key(8, dp_idle=True)
    assert active.graph_key == idle
    assert idle == ns["StagedSFAGraphKey"].bounded_decode(4, 2)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("votes", [[False] * 4, [False, True, False, False], [True, False, True, False]])
def test_dp_layout_vote_reuses_one_collective_and_refreshes_each_step(routing, rank, votes):
    runner, ns, _, modes, _ = routing
    actions = tuple(ns["StagedSFARouteAction"])
    staged = ns["StagedSFARouteAction"].STAGED
    runner.dp_size, runner.dp_rank = 4, rank
    runner._dp_batch_sync_buffers = {}
    runner._skip_all_reduce_across_dp_group = lambda: True
    runner._staged_sfa_graph_capture_sizes = (8,)
    ns.update(
        torch=torch,
        _STAGED_SFA_ROUTE_ACTIONS=actions,
        get_dp_group=lambda: SimpleNamespace(cpu_group="dp"),
        _post_process_cudagraph_mode=lambda tensor: int(tensor[1].min()),
    )
    peer_votes = votes
    buffers = []

    def all_reduce(tensor, group):
        assert group == "dp" and tensor.shape == (4, 4)
        assert tensor[3, rank].item() == int(peer_votes[rank])
        # Other columns must be cleared, not left over from the previous step.
        assert torch.count_nonzero(tensor[3]).item() == int(peer_votes[rank])
        tensor[0].fill_(8)
        tensor[1].fill_(1)
        tensor[2].fill_(actions.index(staged))
        tensor[3] = torch.tensor(peer_votes, dtype=torch.int32)
        buffers.append(tensor.data_ptr())

    ns["dist"] = SimpleNamespace(all_reduce=Mock(side_effect=all_reduce))
    # Idle -> busy and ragged -> uniform transitions must restore the fast
    # graph, rather than permanently forcing bounded after one idle step.
    for peer_votes in (votes, [False] * 4, [False, False, False, True]):
        _, _, _, action = runner._sync_batch_across_dp(8, 1, True, staged, staged_sfa_bounded_decode=peer_votes[rank])
        assert runner._staged_sfa_dp_bounded_decode == any(peer_votes)
        local = ns["StagedSFARouteDecision"](
            staged,
            ns["StagedSFARouteReason"].ELIGIBLE,
            uniform_query_len=0 if peer_votes[rank] else 2,
        )
        route = runner._staged_sfa_live_route(
            local_route=local,
            dp_route_action=action,
            cudagraph_mode=modes.PIECEWISE,
            batch_descriptor=BatchDescriptor(8),
            num_tokens_unpadded=2,
            num_tokens_padded=8,
            num_reqs=1,
            should_ubatch=False,
        )
        expected = ns["StagedSFAGraphKey"].bounded_decode if any(peer_votes) else ns["StagedSFAGraphKey"].fixed_spec
        assert route.graph_key == expected(4, 2)
    assert ns["dist"].all_reduce.call_count == 3  # One existing collective/step.
    assert len(set(buffers)) == 1


@pytest.mark.parametrize("idle", [False, True])
@pytest.mark.parametrize("widths", [[2, 2], [1, 1], [2, 1]])
def test_dispatch_sends_real_layout_and_idle_vote(routing, idle, widths):
    runner, ns, _, _, _ = routing
    runner.model_config.is_encoder_decoder = False
    runner.uniform_decode_query_len = 2
    runner.input_batch = SimpleNamespace(num_computed_tokens_cpu=np.ones(2), lora_id_to_lora_request={})
    runner._pad_for_sequence_parallelism = lambda value: value
    runner.cudagraph_dispatcher = SimpleNamespace(
        dispatch=lambda **kwargs: (SimpleNamespace(value=1), BatchDescriptor(8))
    )
    ns["enable_sp"] = lambda *_: False
    observed = []

    class StopAtCollective(Exception):
        pass

    def sync(**kwargs):
        observed.append(kwargs["staged_sfa_bounded_decode"])
        raise StopAtCollective

    runner._sync_batch_across_dp = sync
    with pytest.raises(StopAtCollective):
        runner._determine_batch_execution_and_padding(
            sum(widths),
            2,
            np.array(widths),
            max(widths),
            False,
            force_uniform_decode=True if idle else None,
            staged_sfa_route_action=ns["StagedSFARouteAction"].STAGED,
            staged_sfa_dp_idle=idle,
        )
    assert observed == [idle or widths != [2, 2]]


def test_dummy_dispatch_passes_idle_status_to_layout_agreement():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dummy = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run")
    dispatch = next(
        node
        for node in ast.walk(dummy)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_determine_batch_execution_and_padding"
    )
    vote = next(keyword.value for keyword in dispatch.keywords if keyword.arg == "staged_sfa_dp_idle")
    assert isinstance(vote, ast.Name) and vote.id == "dp_idle"


@pytest.mark.parametrize("configured,full", [(False, False), (False, True), (True, False)])
def test_non_full_sync_preserves_existing_wire_format(routing, configured, full):
    runner, ns, env, _, _ = routing
    env.VLLM_ASCEND_SFA_FULL_GRAPH = full
    runner._staged_sfa_graph_capture_sizes = (8,) if configured else ()
    runner.dp_size, runner.dp_rank = 2, 0
    runner._dp_batch_sync_buffers = {}
    runner._skip_all_reduce_across_dp_group = lambda: False
    actions = tuple(ns["StagedSFARouteAction"])
    staged = ns["StagedSFARouteAction"].STAGED

    def all_reduce(tensor, group):
        assert tensor.shape == (3 if configured else 2, 2)
        tensor[0].fill_(8)
        tensor[1].fill_(1)
        if configured:
            tensor[2].fill_(actions.index(staged))

    ns.update(
        torch=torch,
        _STAGED_SFA_ROUTE_ACTIONS=actions,
        get_dp_group=lambda: SimpleNamespace(cpu_group="dp"),
        _post_process_cudagraph_mode=lambda tensor: int(tensor[1].min()),
        dist=SimpleNamespace(all_reduce=Mock(side_effect=all_reduce)),
    )
    runner._sync_batch_across_dp(8, 1, True, staged if configured else None)
    ns["dist"].all_reduce.assert_called_once()


def test_tp_only_q2_does_not_consume_dp_layout_state(routing):
    runner, ns, _, modes, _ = routing
    runner.parallel_config.data_parallel_size = 1
    runner._staged_sfa_dp_bounded_decode = True
    local = ns["StagedSFARouteDecision"](
        ns["StagedSFARouteAction"].STAGED,
        ns["StagedSFARouteReason"].ELIGIBLE,
        uniform_query_len=2,
    )
    route = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=None,
        cudagraph_mode=modes.PIECEWISE,
        batch_descriptor=BatchDescriptor(2),
        num_tokens_unpadded=2,
        num_tokens_padded=2,
        num_reqs=1,
        should_ubatch=False,
    )
    assert route.graph_key == ns["StagedSFAGraphKey"].fixed_spec(1, 2)


@pytest.mark.parametrize("rank", range(4))
def test_prefill_and_decode_ranks_agree_on_the_same_padding_vector(routing, rank):
    runner, ns, _, _, _ = routing
    actions = tuple(ns["StagedSFARouteAction"])
    staged, native = ns["StagedSFARouteAction"].STAGED, ns["StagedSFARouteAction"].SAFE_NATIVE
    runner.dp_size, runner.dp_rank = 4, rank
    runner._dp_batch_sync_buffers = {}
    runner._skip_all_reduce_across_dp_group = lambda: True
    ns["torch"] = torch
    ns["_STAGED_SFA_ROUTE_ACTIONS"] = actions
    ns["get_dp_group"] = lambda: SimpleNamespace(cpu_group="dp")
    ns["_post_process_cudagraph_mode"] = lambda tensor: int(tensor[1].min())
    sizes = [8, 4096, 24, 32]
    modes = [1, 0, 1, 1]
    routes = [staged, native, staged, staged]

    def all_reduce(tensor, group):
        tensor[0] = torch.tensor(sizes)
        tensor[1] = torch.tensor(modes)
        tensor[2] = torch.tensor([actions.index(route) for route in routes])

    ns["dist"] = SimpleNamespace(all_reduce=all_reduce)
    _, counts, mode, action = runner._sync_batch_across_dp(sizes[rank], modes[rank], rank != 1, routes[rank])
    assert counts.tolist() == [4096] * 4
    assert mode == 0 and action == native


def test_ragged_dispatch_reserves_request_capacity_not_just_token_sum(routing):
    runner, ns, _, _, _ = routing
    reserved = []

    class StopBeforeDispatch(Exception):
        pass

    def pad(tokens):
        reserved.append(tokens)
        raise StopBeforeDispatch

    runner._pad_for_sequence_parallelism = pad
    # Sixteen Q1 requests need sixteen planner lanes even with MTP enabled.
    with pytest.raises(StopBeforeDispatch):
        runner._determine_batch_execution_and_padding(
            16,
            16,
            np.ones(16, dtype=np.int32),
            1,
            False,
            staged_sfa_route_action=ns["StagedSFARouteAction"].STAGED,
        )
    assert reserved == [32]


@pytest.mark.parametrize("failure", [None, "source", "binding", "signature", "peer"])
def test_preparation_and_signatures_are_agreed_before_collective_replay(routing, failure):
    runner, ns, _, modes, _ = routing
    runner.model = Mock()
    runner.model_config.max_model_len = 140000
    runner.input_batch = SimpleNamespace(req_ids=["r1"], num_reqs=1)
    layer = SimpleNamespace(prepare_full_graph_layer=Mock(return_value={}))
    runner._staged_sfa_impls = [("layer0", layer)]
    runner._run_sfa_full_graph_target = Mock()
    runner._sfa_full_graph = SimpleNamespace(bind_sources=Mock(), prepare_run=Mock(), run=Mock(return_value="output"))
    connector = SimpleNamespace(prepare_sparse_graph_step=Mock(return_value=(None,)))
    context = SimpleNamespace(
        staged_sfa_graph_key="r4q2",
        cudagraph_runtime_mode=modes.PIECEWISE,
        staged_sfa_graph_dummy_run=False,
        staged_sfa_route=SimpleNamespace(frontiers=(0,)),
    )
    ns.update(
        torch=torch,
        get_forward_context=lambda: context,
        get_kv_transfer_group=lambda: connector,
        get_tp_group=lambda: SimpleNamespace(world_size=4, cpu_group="tp"),
        get_dp_group=lambda: SimpleNamespace(world_size=4, cpu_group="dp"),
    )
    if failure == "source":
        connector.prepare_sparse_graph_step.side_effect = ValueError("missing source")
    elif failure == "binding":
        runner._sfa_full_graph.bind_sources.side_effect = ValueError("bad source table")
    elif failure == "signature":
        runner._sfa_full_graph.prepare_run.side_effect = ValueError("changed address")

    groups = []

    def agree(failed, *, op, group):
        groups.append(group)
        runner._sfa_full_graph.run.assert_not_called()
        assert bool(failed.item()) == (failure in ("source", "binding", "signature"))
        if failure == "peer" and group == "dp":
            failed.fill_(1)

    ns["dist"] = SimpleNamespace(all_reduce=agree, ReduceOp=SimpleNamespace(MAX="max"))
    if failure:
        with pytest.raises(RuntimeError, match="preparation failed"):
            runner._model_forward(8)
        runner._sfa_full_graph.run.assert_not_called()
    else:
        assert runner._model_forward(8) == "output"
        runner._sfa_full_graph.prepare_run.assert_called_once()
        runner._sfa_full_graph.run.assert_called_once()
        layer.prepare_full_graph_layer.assert_called_once_with("layer0", 140000, bind_source=False, metadata_checks={})
        assert runner._sfa_full_graph.run.call_args.kwargs == {
            "prepared": runner._sfa_full_graph.prepare_run.return_value
        }
        assert runner._sfa_full_graph.bind_sources.call_args.args[:2] == ((None,), ("r1",))
    assert groups == ["tp", "dp"]


@pytest.mark.parametrize("failure", [None, "source", "binding", "signature"])
def test_local_supervised_decode_has_no_per_step_error_collective(routing, failure):
    runner, ns, _, modes, _ = routing
    runner.model = Mock()
    runner.model_config.max_model_len = 30544
    runner.vllm_config.parallel_config = SimpleNamespace(
        distributed_executor_backend="mp", nnodes=1, data_parallel_size=1, pipeline_parallel_size=1
    )
    runner.input_batch = SimpleNamespace(req_ids=["r"], num_reqs=1)
    layer = SimpleNamespace(prepare_full_graph_layer=Mock(return_value={}))
    runner._staged_sfa_impls = [("layer0", layer)]
    runner._run_sfa_full_graph_target = Mock()
    runner._sfa_full_graph = SimpleNamespace(bind_sources=Mock(), prepare_run=Mock(), run=Mock(return_value="output"))
    connector = SimpleNamespace(prepare_sparse_graph_step=Mock(return_value=(None,)))
    context = SimpleNamespace(
        staged_sfa_graph_key="r1q2",
        cudagraph_runtime_mode=modes.PIECEWISE,
        staged_sfa_graph_dummy_run=False,
        staged_sfa_route=SimpleNamespace(frontiers=(0,)),
    )
    forbidden = Mock(side_effect=AssertionError("no failure tensor, group query, or collective on the live fast path"))
    original = ValueError("injected source preparation failure")

    def exit_worker(error):
        assert error is original
        raise SystemExit(1) from error

    ns.update(
        torch=SimpleNamespace(tensor=forbidden),
        get_forward_context=lambda: context,
        get_kv_transfer_group=lambda: connector,
        get_tp_group=forbidden,
        get_dp_group=forbidden,
        dist=SimpleNamespace(all_reduce=forbidden),
        logger=Mock(),
        exit_failed_sfa_worker=Mock(side_effect=exit_worker),
    )
    if failure:
        target = {
            "source": connector.prepare_sparse_graph_step,
            "binding": runner._sfa_full_graph.bind_sources,
            "signature": runner._sfa_full_graph.prepare_run,
        }[failure]
        target.side_effect = original
        with pytest.raises(SystemExit) as error:
            runner._model_forward(2)
        assert error.value.__cause__ is original
        runner._sfa_full_graph.run.assert_not_called()
        ns["exit_failed_sfa_worker"].assert_called_once_with(original)
        assert ns["logger"].critical.call_args.kwargs["exc_info"][1] is original
    else:
        for _ in range(300):
            assert runner._model_forward(2) == "output"
        assert runner._sfa_full_graph.run.call_count == 300
        assert runner._sfa_full_graph.prepare_run.call_count == 300
        memos = [c.kwargs["metadata_checks"] for c in layer.prepare_full_graph_layer.call_args_list]
        assert len({id(memo) for memo in memos}) == 300
        ns["exit_failed_sfa_worker"].assert_not_called()
    forbidden.assert_not_called()


@pytest.mark.parametrize("failed", [False, True])
def test_local_startup_capture_retains_error_agreement(routing, failed):
    runner, ns, _, modes, _ = routing
    runner.model = Mock()
    runner.model_config.max_model_len = 30544
    runner.vllm_config.parallel_config = SimpleNamespace(
        distributed_executor_backend="mp", nnodes=1, data_parallel_size=1, pipeline_parallel_size=1
    )
    layer = SimpleNamespace(prepare_full_graph_layer=Mock(return_value={}))
    runner._staged_sfa_impls = [("layer0", layer)]
    runner._run_sfa_full_graph_target = Mock()
    runner._sfa_full_graph = SimpleNamespace(
        bind_sources=Mock(side_effect=ValueError("bad startup binding") if failed else None),
        prepare_run=Mock(),
        run=Mock(return_value="capture"),
    )
    context = SimpleNamespace(
        staged_sfa_graph_key="r1q2", cudagraph_runtime_mode=modes.PIECEWISE, staged_sfa_graph_dummy_run=True
    )
    policy = Mock(side_effect=AssertionError("startup must not rely on a ready worker supervisor"))
    calls = []

    def agree(tensor, *, op, group):
        calls.append(group)
        assert tensor.item() == int(failed)

    ns.update(
        torch=torch,
        get_forward_context=lambda: context,
        uses_local_sfa_fail_stop=policy,
        get_tp_group=lambda: SimpleNamespace(world_size=8, cpu_group="tp"),
        get_dp_group=lambda: SimpleNamespace(world_size=1),
        dist=SimpleNamespace(all_reduce=agree, ReduceOp=SimpleNamespace(MAX="max")),
    )
    if failed:
        with pytest.raises(RuntimeError, match="preparation failed"):
            runner._model_forward(2)
        runner._sfa_full_graph.run.assert_not_called()
    else:
        assert runner._model_forward(2) == "capture"
    assert calls == ["tp"]
    policy.assert_not_called()
