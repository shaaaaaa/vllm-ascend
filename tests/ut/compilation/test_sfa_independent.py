# SPDX-License-Identifier: Apache-2.0
"""CPU tests of independent routing; device/transport behavior is not mocked into a pass."""

import gc
import importlib.util
import sys
from contextlib import nullcontext
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import extract
from test_sfa_full_graph_routing import routing

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/worker/sfa_independent.py"
SPEC = importlib.util.spec_from_file_location("sfa_independent_tested", SOURCE)
independent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(independent)
RUNNER = ROOT / "vllm_ascend/worker/model_runner_v1.py"


def policy(monkeypatch):
    calls = []

    def all_reduce(packet, *, group):
        calls.append(group)
        packet.copy_(packet[:, :1].expand_as(packet).clone())

    monkeypatch.setattr(independent.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(independent.dist, "new_group", Mock(return_value="tp"))
    tp = NS(cpu_group="cache-tp", ranks=list(range(4)), rank_in_group=0, world_size=4)
    return independent.IndependentDP(tp, 4, 32), calls


def test_target_uses_its_local_bucket_and_layout(monkeypatch):
    instance, calls = policy(monkeypatch)
    fn = extract(RUNNER, "_sync_batch_across_dp", dict(_STAGED_SFA_ROUTE_ACTIONS=["native", "graph"]))
    runner = NS(dp_size=4, _sfa_independent=instance, input_batch=NS(num_reqs=2))
    for size, bounded in ((8, False), (32, True), (4, True)):
        result = fn(runner, size, 1, True, "graph", bounded)
        assert result[1].tolist() == [size] * 4
        assert result[2:] == (1, "graph")
        assert runner._staged_sfa_dp_bounded_decode == bounded
    assert calls == ["tp"] * 3


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "idle,query_width,native", [(False, 1, False), (False, 2, False), (True, 2, False), (False, 1, True)]
)
def test_dispatch_only_reads_peer_padding_for_coordinated_modes(monkeypatch, enabled, idle, query_width, native):
    runner, ns, *_ = routing.__wrapped__()
    mode = IntEnum("Mode", dict(NONE=0, PIECEWISE=1, FULL=2))
    instance, calls = policy(monkeypatch)
    runner._sfa_independent = instance if enabled else None
    runner.dp_size, runner.dp_rank = 4, 0
    runner._dp_batch_sync_buffers = {}
    runner._skip_all_reduce_across_dp_group = lambda: False
    runner.parallel_config.data_parallel_rank = 0
    runner.model_config.is_encoder_decoder = False
    runner.uniform_decode_query_len = 2
    runner.input_batch = NS(num_computed_tokens_cpu=torch.ones(2).numpy(), num_reqs=2, lora_id_to_lora_request={})
    runner._pad_for_sequence_parallelism = lambda value: value
    runner.vllm_config.observability_config = NS(cudagraph_metrics=False)
    runner.cudagraph_dispatcher = NS(dispatch=Mock(
        side_effect=lambda **kw: (mode.PIECEWISE, NS(num_tokens=max(8, kw["num_tokens"])))
    ))

    def peer_growth(packet, *, group):
        calls.append(group)
        packet.copy_(packet[:, :1].expand_as(packet).clone())
        packet[0].fill_(16)

    ns.update(
        CUDAGraphMode=mode, enable_sp=lambda _: False, torch=torch,
        dist=NS(all_reduce=peer_growth), get_dp_group=lambda: NS(cpu_group="dp"),
        _post_process_cudagraph_mode=lambda tensor: int(tensor[1].min()),
    )
    ns["_STAGED_SFA_ROUTE_ACTIONS"] = tuple(ns["StagedSFARouteAction"])
    reads = Mock(wraps=torch.Tensor.item)
    monkeypatch.setattr(torch.Tensor, "item", lambda tensor: reads(tensor))

    result = runner._determine_batch_execution_and_padding(
        2 * query_width, 2, [query_width] * 2, query_width, False,
        staged_sfa_route_action=ns["StagedSFARouteAction"].SAFE_NATIVE if native else ns["StagedSFARouteAction"].STAGED,
        staged_sfa_dp_idle=idle, force_eager=native,
    )

    capacity = (2 * query_width if native else 8) if enabled else 16
    assert result[0] == (mode.NONE if native else mode.PIECEWISE)
    assert result[1].num_tokens == capacity and result[3].tolist() == [capacity] * 4
    assert runner._staged_sfa_dp_bounded_decode == (idle or query_width == 1)
    assert calls == ["tp" if enabled else "dp"]
    assert runner.cudagraph_dispatcher.dispatch.call_count == (0 if native else 1 if enabled else 2)
    if enabled:
        reads.assert_not_called()
    else:
        reads.assert_called()


def test_draft_and_idle_metadata_do_not_enter_dp_vote(monkeypatch):
    instance, calls = policy(monkeypatch)
    fn = extract(RUNNER, "_sync_metadata_across_dp", {})
    runner = NS(dp_size=4, _sfa_independent=instance, input_batch=NS(num_reqs=2))
    for count in (4, 8, 16):
        tokens, counts, prefill = fn(runner, count, is_draft_model=True)
        assert tokens == count and counts.tolist() == [count] * 4 and not prefill
    assert calls == ["tp"] * 3


@pytest.mark.parametrize("tokens", [0, -1, 33])
def test_out_of_contract_extent_fails_before_any_collective(monkeypatch, tokens):
    instance, calls = policy(monkeypatch)
    with pytest.raises(ValueError):
        instance.agree(tokens, 1, 1, True)
    assert not calls


def test_tp_disagreement_is_detected(monkeypatch):
    instance, _ = policy(monkeypatch)

    def mismatch(packet, **kwargs):
        packet.copy_(packet[:, :1].expand_as(packet).clone())
        packet[0, 2] += 1

    monkeypatch.setattr(independent.dist, "all_reduce", mismatch)
    with pytest.raises(RuntimeError, match="within TP"):
        instance.agree(8, 1, 1, True)


@pytest.mark.parametrize("error", [None, RuntimeError("preparation failed")])
def test_actual_preparation_guard_defers_failure_to_supervisor(error):
    fn = extract(RUNNER, "_coordinate_sfa_full_graph_preparation", {})
    runner = NS(_sfa_independent=object())
    if error is None:
        fn(runner, error, True)  # Includes live idle graph participation.
    else:
        with pytest.raises(RuntimeError, match="preparation failed"):
            fn(runner, error, False)


def test_disabled_preparation_keeps_original_fail_stop_policy():
    exit_worker = Mock(side_effect=SystemExit(1))
    fn = extract(
        RUNNER,
        "_coordinate_sfa_full_graph_preparation",
        dict(
            uses_local_sfa_fail_stop=lambda config: True,
            logger=Mock(),
            exit_failed_sfa_worker=exit_worker,
        ),
    )
    runner = NS(_sfa_independent=None, vllm_config=NS(parallel_config=object()))
    with pytest.raises(SystemExit):
        fn(runner, RuntimeError("source"), False)
    exit_worker.assert_called_once()


@pytest.fixture
def contract_runner(monkeypatch):
    monkeypatch.setattr(independent, "_native_identity", lambda: {"mock-native.so": "a" * 64})
    parallel = NS(
        data_parallel_size=4,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        nnodes_within_dp=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_dbo=False,
        enable_elastic_ep=False,
    )
    spec = NS(num_speculative_tokens=1, method="mtp", draft_tensor_parallel_size=4)
    config = NS(
        parallel_config=parallel,
        speculative_config=spec,
        additional_config={},
        model_config=NS(dtype=torch.bfloat16, hf_text_config=NS(to_dict=lambda: dict(num_hidden_layers=79))),
        scheduler_config=NS(mc2_recovery_token_budget=32, async_scheduling=True),
    )
    key = NS(token_capacity=32, request_capacity=16, query_profile=NS(value="bounded"), max_query_len=2)
    # The real startup validator iterates structural keys; no device capture is simulated.
    graph = NS(sealed=True, entries=[key])
    runner = NS(
        vllm_config=config,
        _sfa_full_graph=graph,
        _can_skip_dp_metadata=lambda draft: True,
        drafter=NS(needs_extra_input_slots=False),
    )
    calls = []

    def selector(count, config, draft):
        calls.append((count, draft))
        return "MC2"

    modules = {
        "torch_npu": NS(
            __version__="test-8.5.1-binding",
            npu_moe_distribute_dispatch_v2=lambda: None,
            npu_moe_distribute_combine_v2=lambda: None,
        ),
        "vllm.distributed": NS(
            get_ep_group=lambda: NS(world_size=16, ranks=range(16)), get_tp_group=lambda: NS(world_size=4)
        ),
        "vllm_ascend.ascend_forward_context": NS(
            MoECommType=NS(MC2="MC2"), get_mc2_tokens_capacity=lambda: 32, select_moe_comm_method=selector
        ),
        "vllm_ascend.ops.fused_moe.moe_comm_method": NS(
            get_moe_comm_method=lambda mode: NS(
                token_dispatcher=NS(enable_dispatch_v2=True, need_comm_alg=False, global_bs=128, ep_world_size=16)
            )
        ),
        "vllm_ascend.utils": NS(
            AscendDeviceType=NS(A2="A2", A3="A3"),
            enable_sp=lambda config: False,
            enable_dsa_cp=lambda: False,
            enable_dsa_cp_decoder=lambda: False,
            shared_expert_dp_enabled=lambda: False,
            get_ascend_device_type=lambda: "A3",
            is_drafter_moe_model=lambda config: True,
            is_moe_model=lambda config: True,
            is_hierarchical_communication_enabled=lambda: False,
            sfa_full_graph_enabled=lambda config: True,
        ),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    return runner, modules, calls


def test_startup_contract_checks_every_target_and_draft_extent(contract_runner):
    runner, _, calls = contract_runner
    result = independent.execution_contract(runner)
    assert len(result["digest"]) == 64 and result["capacity"] == 32
    assert calls == [(n, draft) for draft in (False, True) for n in range(1, 33)]


@pytest.fixture
def dcp_contract_runner(contract_runner, monkeypatch):
    runner, modules, calls = contract_runner
    config = runner.vllm_config
    parallel = config.parallel_config
    parallel.decode_context_parallel_size = parallel.local_world_size = parallel.tensor_parallel_size
    parallel.cp_kv_cache_interleave_size = 128
    config.model_config.use_mla, config.model_config.enforce_eager = True, False
    config.model_config.hf_text_config.index_topk = 2048
    config.kv_transfer_config = NS(is_kv_consumer=True)
    config.cache_config = NS(block_size=128)
    config._dsa_decoder_dcp_policy = (4096, 0)
    runner.dsa_decoder_dcp, runner.use_cp = True, False
    tp = NS(world_size=4, ranks=[0, 1, 2, 3], rank_in_group=0)
    dcp = NS(**vars(tp))
    modules["vllm.distributed"].get_tp_group = lambda: tp
    modules["vllm.distributed"].get_dcp_group = lambda: dcp
    modules["vllm_ascend.utils"].get_ascend_device_type = lambda: "A2"
    for name in ("SFA_INDEPENDENT_DP", "SFA_FULL_GRAPH", "DSA_UNBUNDLE", "DSA_TWO_GROUPS"):
        monkeypatch.setenv("VLLM_ASCEND_" + name, "1")
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "false")
    for name in ("DCP_HCCL_HOST", "SFA_DCP_EARLY_GATHER", "SFA_DCP_QUERY_FIRST", "SFA_DCP_PREFETCH_AFTER_PACK"):
        monkeypatch.setenv("VLLM_ASCEND_" + name, "0")
    return runner, modules, tp, dcp


def test_dcp_contract_uses_replica_relative_geometry(dcp_contract_runner):
    runner, _, tp, dcp = dcp_contract_runner
    first = independent.execution_contract(runner)
    assert len(first["digest"]) == 64 and first["capacity"] == 32
    tp.ranks = dcp.ranks = [4, 5, 6, 7]
    tp.rank_in_group = dcp.rank_in_group = 3
    assert independent.execution_contract(runner) == first


@pytest.mark.parametrize("case", ["unvalidated", "cp", "device", "order", "rank", "world"])
def test_dcp_contract_rejects_unvalidated_or_misaligned_layout(dcp_contract_runner, case):
    runner, modules, tp, dcp = dcp_contract_runner
    if case == "unvalidated":
        runner.dsa_decoder_dcp = False
    elif case == "cp":
        runner.use_cp = True
    elif case == "device":
        modules["vllm_ascend.utils"].get_ascend_device_type = lambda: "A3"
    elif case == "order":
        dcp.ranks = list(reversed(tp.ranks))
    elif case == "rank":
        dcp.rank_in_group = 1
    else:
        dcp.world_size = 2
    result = independent.execution_contract(runner)
    assert "DCP layout/membership" in result["reason"] and "digest" not in result


@pytest.mark.parametrize(
    "name", ["DCP_HCCL_HOST", "SFA_DCP_EARLY_GATHER", "SFA_DCP_QUERY_FIRST", "SFA_DCP_PREFETCH_AFTER_PACK"]
)
def test_dcp_communication_schedule_changes_contract(dcp_contract_runner, monkeypatch, name):
    runner, *_ = dcp_contract_runner
    first = independent.execution_contract(runner)
    monkeypatch.setenv("VLLM_ASCEND_" + name, "1")
    assert independent.execution_contract(runner)["digest"] != first["digest"]


def test_dcp_unpack_contract_matches_active_schedule(dcp_contract_runner, monkeypatch):
    runner, *_ = dcp_contract_runner
    monkeypatch.setenv("VLLM_ASCEND_SFA_DCP_EARLY_GATHER", "1")
    monkeypatch.setenv("VLLM_ASCEND_SFA_DCP_EARLY_UNPACK", "off")
    first = independent.execution_contract(runner)["digest"]
    monkeypatch.setenv("VLLM_ASCEND_SFA_DCP_UNPACK_CORES", "1")
    assert independent.execution_contract(runner)["digest"] == first
    monkeypatch.setenv("VLLM_ASCEND_SFA_DCP_EARLY_UNPACK", "producer")
    second = independent.execution_contract(runner)["digest"]
    assert second != first
    monkeypatch.setenv("VLLM_ASCEND_SFA_DCP_UNPACK_CORES", "12")
    assert independent.execution_contract(runner)["digest"] != second


def test_dcp_packet_contract_records_native_io_mode(dcp_contract_runner, monkeypatch):
    runner, *_ = dcp_contract_runner
    monkeypatch.setenv("VLLM_ASCEND_SFA_DCP_EARLY_GATHER", "1")
    monkeypatch.setenv("VLLM_ASCEND_SFA_DCP_EARLY_UNPACK", "producer")
    digests = set()
    for mode in ("off", "output", "direct", "fused"):
        monkeypatch.setenv("VLLM_ASCEND_SFA_DCP_PACKET_IO", mode)
        digests.add(independent.execution_contract(runner)["digest"])
    assert len(digests) == 4
    before = independent.execution_contract(runner)["digest"]
    monkeypatch.setenv("VLLM_ASCEND_SFA_DCP_UNPACK_CORES", "3")
    assert independent.execution_contract(runner)["digest"] == before


@pytest.mark.parametrize(
    "name,before,after", [("HCCL_OP_EXPANSION_MODE", "AIV", "HOST"), ("TASK_QUEUE_ENABLE", "1", "0")]
)
def test_dcp_runtime_mode_changes_contract(dcp_contract_runner, monkeypatch, name, before, after):
    runner, *_ = dcp_contract_runner
    monkeypatch.setenv(name, before)
    first = independent.execution_contract(runner)
    monkeypatch.setenv(name, after)
    assert independent.execution_contract(runner)["digest"] != first["digest"]


def test_independent_dcp_idle_clears_local_kv_and_request_ownership():
    clear = extract(RUNNER, "_clear_sfa_idle_metadata", {})
    dcp = NS(local_seq_lens=torch.tensor([128, 256]), row_requests=torch.tensor([0, 0, 1, 1]))
    metadata = NS(sfa_dcp=dcp, slot_mapping=torch.tensor([129, 257]), indexer_slot_mapping=torch.tensor([384, 512]))
    clear(metadata)
    assert dcp.local_seq_lens.tolist() == [0, 0]
    assert dcp.row_requests.tolist() == [-1] * 4
    assert metadata.slot_mapping.tolist() == metadata.indexer_slot_mapping.tolist() == [-1, -1]


@pytest.mark.parametrize("filename", ["mc2_recovery.py", "sfa_dcp.py", "sfa_query.py", "sfa_retrieval_overlap.py"])
def test_dcp_source_changes_contract(dcp_contract_runner, monkeypatch, filename):
    runner, *_ = dcp_contract_runner
    first = independent.execution_contract(runner)
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: b"different build" if path.name == filename else original(path)
    )
    assert independent.execution_contract(runner)["digest"] != first["digest"]


@pytest.mark.parametrize("case", ["disabled", "tp", "local", "prefiller", "generic", "pcp", "dbo"])
def test_dcp_startup_topology_is_restricted_to_decoder_layout(dcp_contract_runner, monkeypatch, case):
    runner, *_ = dcp_contract_runner
    config = runner.vllm_config
    assert independent.topology_reason(config) == ""
    if case == "disabled":
        monkeypatch.setenv("VLLM_ASCEND_SFA_INDEPENDENT_DP", "0")
    elif case == "prefiller":
        config.kv_transfer_config.is_kv_consumer = False
    elif case == "generic":
        del config.model_config.hf_text_config.index_topk
    else:
        field = dict(
            tp="tensor_parallel_size", local="local_world_size", pcp="prefill_context_parallel_size", dbo="enable_dbo"
        )[case]
        setattr(config.parallel_config, field, True if case == "dbo" else 2)
    assert independent.topology_reason(config)


@pytest.mark.parametrize(
    "case", ["hierarchy", "missing_combine", "draft_tp", "recovery", "graph_capacity", "middle_extent", "other_runner"]
)
def test_unsupported_contract_preserves_collective_legacy_choice(contract_runner, case):
    runner, modules, _ = contract_runner
    if case == "hierarchy":
        modules["vllm_ascend.utils"].is_hierarchical_communication_enabled = lambda: True
    elif case == "missing_combine":
        del modules["torch_npu"].npu_moe_distribute_combine_v2
    elif case == "draft_tp":
        runner.vllm_config.speculative_config.draft_tensor_parallel_size = 1
    elif case == "recovery":
        runner._can_skip_dp_metadata = lambda draft: False
    elif case == "graph_capacity":
        runner._sfa_full_graph.entries[0].token_capacity = 33
    elif case == "other_runner":
        del runner._sfa_full_graph
    else:
        modules["vllm_ascend.ascend_forward_context"].select_moe_comm_method = (
            lambda n, config, draft: "ALLGATHER" if n == 7 else "MC2"
        )
    result = independent.execution_contract(runner)
    assert result["reason"] and "digest" not in result


def test_unqualified_runner_can_install_common_fallback(contract_runner, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.sfa_independent", independent)
    activate = extract(ROOT / "vllm_ascend/worker/worker.py", "activate_independent_dp", {})
    guard = Mock()
    create = Mock()
    monkeypatch.setattr(independent.dist, "new_group", create)
    worker = NS(model_runner=NS(), dp_guardian=guard)
    activate(worker, False)
    guard.close.assert_called_once()
    assert worker.dp_guardian is None
    create.assert_not_called()


def test_failed_single_fallback_keeps_guardian_attached(contract_runner, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.sfa_independent", independent)
    activate = extract(ROOT / "vllm_ascend/worker/worker.py", "activate_independent_dp", {})
    single = NS(activate=Mock(side_effect=ValueError("capture keys differ")))
    runner = NS(_sfa_single_agreement=single, _sfa_full_graph=NS(entries=[]))
    guard = Mock()
    worker = NS(model_runner=runner, dp_guardian=guard)
    with pytest.raises(ValueError, match="capture keys differ"):
        activate(worker, False)
    guard.close.assert_not_called()
    assert worker.dp_guardian is guard


@pytest.mark.parametrize("single_requested", [False, True])
@pytest.mark.parametrize("qualification", ["disabled", "static_fallback", "dynamic_fallback", "qualified"])
def test_startup_selects_one_agreement_protocol(monkeypatch, single_requested, qualification):
    from test_sfa_single_agreement import policy as single_policy

    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.decode_warmup", NS(warmup_decode_kernels=Mock()))
    runner, ns = single_policy()
    tp = NS(cpu_group="cache-tp", ranks=[0, 1], rank_in_group=0, world_size=2)
    ep = NS(cpu_group="ep", rank_in_group=0, world_size=4)
    votes, signatures = [], []

    def vote(value, *, group):
        votes.append(group)
        if value.ndim == 1:
            value.mul_(ep.world_size)
        else:
            value.copy_(value[:, :1].expand_as(value).clone())

    def gather(rows, value, *, group):
        signatures.append(group)
        rows[:] = [value] * len(rows)

    monkeypatch.setattr(independent.dist, "all_reduce", vote)
    monkeypatch.setattr(independent.dist, "all_gather_object", gather)
    create = Mock(return_value="routing-tp")
    monkeypatch.setattr(independent.dist, "new_group", create)
    monkeypatch.setitem(sys.modules, "vllm.distributed", NS(get_tp_group=lambda: tp))
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.sfa_independent", independent)
    ns["envs"].VLLM_ASCEND_SFA_SINGLE_AGREEMENT = single_requested
    runner._sfa_preparation_groups = (("sfa_full_graph::prepare_agreement_ep", ep.cpu_group),)
    runner._async_snapshot = None
    runner.parallel_config.tensor_parallel_size = tp.world_size
    runner._sfa_single_agreement = runner._sfa_independent = None
    ns["initialize_single_agreement"](runner, ep)
    single = runner._sfa_single_agreement
    runner._sfa_full_graph = NS(entries=[ns["StagedSFAGraphKey"].fixed_spec(4, 2)])
    runner.capture_model = Mock()
    pending = qualification in ("dynamic_fallback", "qualified")
    guard = Mock() if pending else None
    config = NS(compile_sizes=[], cudagraph_mode=0, get_compile_ranges=lambda: [], compilation_time=0)
    worker = NS(
        model_runner=runner,
        dp_guardian=guard,
        vllm_config=NS(compilation_config=config),
        model_config=NS(enforce_eager=False, seed=0),
        parallel_config=NS(data_parallel_size=2),
        _wait_for_decoder_ep_startup=Mock(),
        _independent_dp_contract=dict(digest="a" * 64, capacity=32),
    )
    source = ROOT / "vllm_ascend/worker/worker.py"
    warmup = extract(
        source,
        "compile_or_warm_up_model",
        dict(
            CUDAGraphMode=NS(NONE=0),
            cold_perf_enabled=lambda: False,
            startup_phase=lambda *a, **kw: nullcontext(),
            get_ascend_device_type=lambda: "a5",
            AscendDeviceType=NS(A5="a5"),
            set_random_seed=Mock(),
        ),
    )
    warmup(worker)
    runner.capture_model.assert_called_once()
    assert signatures == (["ep"] if single_requested and not pending else [])
    if single is not None:
        assert single.active == (not pending)
    if pending:
        extract(source, "activate_independent_dp", {})(worker, qualification == "qualified")

    if qualification == "qualified":
        assert runner._sfa_single_agreement is None and runner._sfa_independent is not None
        assert not signatures  # No EP activation vote, even when both flags are on.
        guard.close.assert_not_called()
        create.assert_called_once()
        runner._sfa_independent.agree(8, 1, 1, False, requests=4)
        assert votes == ["ep", "routing-tp"]  # Only the startup vote crosses DP.
    else:
        assert runner._sfa_independent is None
        assert signatures == (["ep"] if single_requested else [])
        assert single is None or single.active
        create.assert_not_called()
        if pending:
            guard.close.assert_called_once()
            assert worker.dp_guardian is None


@pytest.mark.parametrize("field,value", [("global_bs", 64), ("enable_dispatch_v2", False), ("need_comm_alg", True)])
def test_contract_checks_loaded_dispatcher_not_only_config_estimate(contract_runner, field, value):
    runner, modules, _ = contract_runner
    module = modules["vllm_ascend.ops.fused_moe.moe_comm_method"]
    method = module.get_moe_comm_method("MC2")
    setattr(method.token_dispatcher, field, value)
    module.get_moe_comm_method = lambda mode: method
    result = independent.execution_contract(runner)
    assert "loaded MC2 dispatcher" in result["reason"] and "digest" not in result


def test_native_build_change_changes_execution_contract(contract_runner, monkeypatch):
    runner, _, _ = contract_runner
    before = independent.execution_contract(runner)
    monkeypatch.setattr(independent, "_native_identity", lambda: {"mock-native.so": "b" * 64})
    assert independent.execution_contract(runner)["digest"] != before["digest"]


@pytest.mark.parametrize("flag", ["enable_sp", "enable_dsa_cp", "shared_expert_dp_enabled", "enable_dsa_cp_decoder"])
def test_effective_layout_changes_contract_even_with_same_additional_config(contract_runner, flag):
    runner, modules, _ = contract_runner
    before = independent.execution_contract(runner)
    setattr(modules["vllm_ascend.utils"], flag, lambda *args: True)
    assert independent.execution_contract(runner)["digest"] != before["digest"]


@pytest.mark.parametrize("filename", ["eagle_proposer.py", "sfa_v1.py", "fused_moe.py", "register_custom_ops.py"])
def test_fc1_source_changes_fail_startup_contract(contract_runner, monkeypatch, filename):
    runner, _, _ = contract_runner
    before = independent.execution_contract(runner)
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: b"different build" if path.name == filename else original(path)
    )
    assert independent.execution_contract(runner)["digest"] != before["digest"]


def test_different_draft_layer_schedule_changes_execution_contract(contract_runner):
    runner, _, _ = contract_runner
    layers = {"num_hidden_layers": 1}
    runner.vllm_config.speculative_config.draft_model_config = NS(
        hf_text_config=NS(to_dict=lambda: dict(layers)), dtype=torch.bfloat16, quantization="ascend"
    )
    before = independent.execution_contract(runner)
    layers["num_hidden_layers"] = 2
    assert independent.execution_contract(runner)["digest"] != before["digest"]


def test_missing_native_identity_uses_common_legacy_decision(contract_runner, monkeypatch):
    runner, _, _ = contract_runner
    monkeypatch.setattr(independent, "_native_identity", Mock(side_effect=OSError("unreadable mapping")))
    result = independent.execution_contract(runner)
    assert "native runtime identity unavailable" in result["reason"] and "digest" not in result


def test_native_identity_uses_mapped_file_identity_and_detects_replacement(tmp_path, monkeypatch):
    # Synthetic mappings test the actual file verifier, not loaded NPU binaries.
    monkeypatch.setattr(independent.os, "major", lambda device: 0, raising=False)
    monkeypatch.setattr(independent.os, "minor", lambda device: 1, raising=False)
    backing = Mock(return_value=(0, 1, -1))
    monkeypatch.setattr(independent, "_mapped_file_identity", backing, raising=False)
    lines = []
    for name in ("libtorch_npu.so", "libhccl.so", "libruntime.so", "vllm_ascend_C.so"):
        library = tmp_path / name
        library.write_bytes(name.encode())
        lines.append(f"1-2 r-xp 0000 00:01 {library.stat().st_ino} {library}")
    maps = tmp_path / "maps"
    maps.write_text("\n".join(lines))
    identities = independent._native_identity(maps)
    assert len(identities) == 4 and len(set(identities.values())) == 4
    backing.assert_not_called()
    maps.write_text("\n".join(lines).replace("00:01", "00:02"))
    with pytest.raises(ValueError, match="replaced"):
        independent._native_identity(maps)


@pytest.mark.parametrize("replacement", [None, "device", "inode"])
def test_native_identity_accepts_overlay_only_when_backing_identity_matches(tmp_path, monkeypatch, replacement):
    # Reproduce the container's fstat=00:33 versus maps=08:02 discrepancy.
    monkeypatch.setattr(independent.os, "major", lambda device: 0, raising=False)
    monkeypatch.setattr(independent.os, "minor", lambda device: 0x33, raising=False)
    lines = []
    for name in ("libtorch_npu.so", "libhccl.so", "libruntime.so", "vllm_ascend_C.so"):
        library = tmp_path / name
        library.write_bytes(name.encode())
        lines.append(f"1-2 r-xp 0000 08:02 {library.stat().st_ino} {library}")
    maps = tmp_path / "maps"
    maps.write_text("\n".join(lines))
    backing = Mock(side_effect=lambda fd: (
        9 if replacement == "device" else 8, 2,
        independent.os.fstat(fd).st_ino + int(replacement == "inode"),
    ))
    monkeypatch.setattr(independent, "_mapped_file_identity", backing, raising=False)
    if replacement is None:
        assert len(independent._native_identity(maps)) == 4
        assert backing.call_count == 4
    else:
        with pytest.raises(ValueError, match="replaced"):
            independent._native_identity(maps)


@pytest.mark.parametrize("failure", [None, "missing", "unreadable"])
def test_backing_identity_probe_closes_real_mapping_without_gc(tmp_path, monkeypatch, failure):
    library = tmp_path / "native.so"
    library.write_bytes(b"unchanged native bytes")
    mappings = []
    mmap_file, read_text = independent.mmap.mmap, Path.read_text

    def mapping(*args, **kwargs):
        value = mmap_file(*args, **kwargs)
        mappings.append(value)
        return value

    def maps(path, *args, **kwargs):
        if path != Path("/proc/self/maps"):
            return read_text(path, *args, **kwargs)
        if failure == "unreadable":
            raise PermissionError("maps unavailable")
        address = independent.ctypes.addressof(independent.ctypes.c_char.from_buffer(mappings[-1]))
        address += int(failure == "missing")
        return f"{address:x}-{address + 1:x} rw-p 0000 08:02 15620327 /native.so"

    monkeypatch.setattr(independent.mmap, "mmap", mapping)
    monkeypatch.setattr(Path, "read_text", maps)
    enabled = gc.isenabled()
    gc.disable()
    try:
        with library.open("rb") as file:
            if failure:
                with pytest.raises(PermissionError if failure == "unreadable" else ValueError):
                    independent._mapped_file_identity(file.fileno())
            else:
                assert independent._mapped_file_identity(file.fileno()) == (8, 2, 15620327)
        assert len(mappings) == 1 and mappings[0].closed
        assert library.read_bytes() == b"unchanged native bytes"
    finally:
        if enabled:
            gc.enable()


@pytest.mark.skipif(sys.platform != "linux", reason="Requires real Linux /proc mappings")
def test_backing_identity_tracks_open_file_across_path_replacement(tmp_path):
    library, replacement = tmp_path / "native.so", tmp_path / "replacement.so"
    library.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    with library.open("rb") as old:
        identity = independent._mapped_file_identity(old.fileno())
        replacement.replace(library)
        assert independent._mapped_file_identity(old.fileno()) == identity
        with library.open("rb") as new:
            assert independent._mapped_file_identity(new.fileno()) != identity


@pytest.mark.parametrize("entry", ["idle", "graph", "native"])
@pytest.mark.parametrize("dcp_size", [1, 4, 8])
def test_actual_ascend_draft_entry_reaches_local_policy_once(monkeypatch, entry, dcp_size):
    instance, calls = policy(monkeypatch)
    sync = extract(RUNNER, "_sync_metadata_across_dp", {})
    runner = NS(dp_size=4, _sfa_independent=instance, input_batch=NS(num_reqs=3), use_cp=False)

    class ReachedDeviceBoundary(Exception):
        pass

    def observe(count, **kwargs):
        result = sync(runner, count, **kwargs)
        assert result[1].tolist() == [8 if entry != "native" else 3] * 4
        raise ReachedDeviceBoundary

    runner._sync_metadata_across_dp = observe
    source = ROOT / "vllm_ascend/spec_decode/eagle_proposer.py"
    if entry == "idle":
        fn = extract(source, "dummy_run", dict(CUDAGraphMode=NS(NONE=0)))
        with pytest.raises(ReachedDeviceBoundary):
            fn(NS(runner=runner), 8)
    else:
        fn = extract(source, "_propose", {})
        metadata = NS(batch_size=lambda: 3)
        proposer = NS(
            runner=runner,
            method="mtp",
            pcp_size=1,
            dcp_size=dcp_size,
            use_staged_mtp_draft_graph=entry == "graph",
            use_cuda_graph=False,
            set_inputs_first_pass=lambda **kwargs: (3, None, metadata, None),
        )
        with pytest.raises(ReachedDeviceBoundary):
            fn(
                proposer,
                None,
                None,
                None,
                None,
                torch.arange(3),
                metadata,
                None,
                None,
                target_staged_sfa_graph_key=NS(request_capacity=8),
            )
    assert instance.sequence == 1 and calls == ["tp"]
    assert instance.packet[4].tolist() == [1] * 4
