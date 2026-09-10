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
    )
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
        "_model_forward",
        "_staged_sfa_dummy_seq_len",
        "_prepare_staged_sfa_dummy_block_tables",
    }
    definitions(root / "worker/model_runner_v1.py", names, ns, class_name="NPUModelRunner")
    runner_type = type("RunnerRouting", (), {name: ns[name] for name in names})
    runner = runner_type()
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=1)
    runner.decode_threshold = 2
    runner.vllm_config = SimpleNamespace(lora_config=None, model_config=SimpleNamespace(enforce_eager=False))
    runner.model_config = runner.vllm_config.model_config
    runner.parallel_config = SimpleNamespace(data_parallel_size=4)
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
    assert live.graph_key.query_profile == ns["StagedSFAQueryProfile"].DECODE_BOUNDED
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


def test_capture_sizes_use_one_bounded_graph_for_q1_and_q2(routing):
    _, ns, env, _, _ = routing
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=4096),
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
    )
    assert ns["staged_sfa_graph_capture_sizes"](config) == (2,)
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
    assert key == ns["StagedSFAGraphKey"].bounded_decode(capacity, 2)


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


@pytest.mark.parametrize("failure", [None, "source", "signature", "peer"])
def test_preparation_and_signatures_are_agreed_before_collective_replay(routing, failure):
    runner, ns, _, modes, _ = routing
    runner.model = Mock()
    runner.model_config.max_model_len = 140000
    runner.input_batch = SimpleNamespace(req_ids=["r1"], num_reqs=1)
    layer = SimpleNamespace(prepare_full_graph_layer=Mock(return_value={}))
    runner._staged_sfa_impls = [("layer0", layer)]
    runner._run_sfa_full_graph_target = Mock()
    runner._sfa_full_graph = SimpleNamespace(validate_inputs=Mock(), run=Mock(return_value="output"))
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
    elif failure == "signature":
        runner._sfa_full_graph.validate_inputs.side_effect = ValueError("changed address")

    groups = []

    def agree(failed, *, op, group):
        groups.append(group)
        runner._sfa_full_graph.run.assert_not_called()
        assert bool(failed.item()) == (failure in ("source", "signature"))
        if failure == "peer" and group == "dp":
            failed.fill_(1)

    ns["dist"] = SimpleNamespace(all_reduce=agree, ReduceOp=SimpleNamespace(MAX="max"))
    if failure:
        with pytest.raises(RuntimeError, match="preparation failed"):
            runner._model_forward(8)
        runner._sfa_full_graph.run.assert_not_called()
    else:
        assert runner._model_forward(8) == "output"
        runner._sfa_full_graph.validate_inputs.assert_called_once()
        runner._sfa_full_graph.run.assert_called_once()
    assert groups == ["tp", "dp"]
