# SPDX-License-Identifier: Apache-2.0
"""CPU tests of independent routing; device/transport behavior is not mocked into a pass."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import extract

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
    return independent.IndependentDP(NS(cpu_group="tp", rank_in_group=0, world_size=4), 4, 32), calls


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
    worker = NS(model_runner=NS(), dp_guardian=guard)
    activate(worker, False)
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
    lines = []
    for name in ("libtorch_npu.so", "libhccl.so", "libruntime.so", "vllm_ascend_C.so"):
        library = tmp_path / name
        library.write_bytes(name.encode())
        lines.append(f"1-2 r-xp 0000 00:01 {library.stat().st_ino} {library}")
    maps = tmp_path / "maps"
    maps.write_text("\n".join(lines))
    identities = independent._native_identity(maps)
    assert len(identities) == 4 and len(set(identities.values())) == 4
    maps.write_text("\n".join(lines).replace("00:01", "00:02"))
    with pytest.raises(ValueError, match="replaced"):
        independent._native_identity(maps)


@pytest.mark.parametrize("entry", ["idle", "graph", "native"])
def test_actual_ascend_draft_entry_reaches_local_policy_once(monkeypatch, entry):
    instance, calls = policy(monkeypatch)
    sync = extract(RUNNER, "_sync_metadata_across_dp", {})
    runner = NS(dp_size=4, _sfa_independent=instance, input_batch=NS(num_reqs=3))

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
            dcp_size=1,
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
