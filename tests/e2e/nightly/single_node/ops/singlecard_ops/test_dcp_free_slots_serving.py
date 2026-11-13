# SPDX-License-Identifier: Apache-2.0
"""Qualify the actual serving extension; run policies in separate processes."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu", reason="serving kernels require CANN/torch-npu")

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.sparse_offload import resident_sorted_cache as resident
from vllm_ascend.utils import enable_custom_op

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "benchmarks/ops/resident_kernel_experiment"))
import benchmark_dcp as benchmark  # noqa: E402 -- reuse the standalone oracle
from dcp_workload import cold_case, make_trace  # noqa: E402
from resident_experiment import _put_state, assert_result, make_case, membership_reference, reference  # noqa: E402


def serving_run(case, world, rank, policy):
    state = resident.SortedResidentState(
        case["state_tokens"], case["state_slots"], case["state_counts"], case["state_generations"], case.dummy_base
    )
    workspace = resident.SortedResidentWorkspace(*case.tensors[3:6], *case.tensors[12:19])
    dcp = resident.ResidentDcpLayout(world, rank, *case.tensors[20:22]) if world > 1 else None
    resident.prepare_resident_sharded_union_(
        case["topk"],
        case["boundary"],
        case["row_requests"],
        case["request_states"],
        case["request_generations"],
        state,
        workspace,
        mtp=case.mtp,
        query_membership=True,
        dcp=dcp,
    )
    resident.prepare_sorted_resident_cache_fused_(
        case["topk"],
        case["block_table"],
        case["request_states"],
        case["request_generations"],
        state,
        workspace,
        block_size=128,
        query_membership=True,
        dcp=dcp,
    )


@pytest.fixture(scope="module")
def policy():
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    assert enable_custom_op(), "rebuild the serving extension"
    resident.configure_resident_kernels()
    return int(envs.VLLM_ASCEND_DSA_DCP_FREE_SLOT_FIRST)


@pytest.mark.parametrize("world,rank", [(2, 1), (4, 2), (8, 0), (8, 7)])
def test_serving_stateful_graph_replay(monkeypatch, policy, world, rank):
    monkeypatch.setattr(benchmark, "run", serving_run)
    trace = make_trace(requests=2, steps=24, world=world, rank=rank)
    benchmark.validate(cold_case(2, 65536), trace, world, rank, policy, "npu")


@pytest.mark.parametrize("occupancy", [2047, 2048, 2049, 4095, 4096])
def test_serving_skew_capacity_and_generation(monkeypatch, policy, occupancy):
    monkeypatch.setattr(benchmark, "run", serving_run)
    snapshot = cold_case(1, 1 << 20)
    tokens = [t for t in range(1 << 20) if t // 128 % 8 == 0 and t % 8 == 0]
    _put_state(snapshot, 0, dict(zip(tokens[:occupancy], reversed(range(occupancy)), strict=True)))
    snapshot["topk"].fill_(-1)
    snapshot["topk"][:, 0, :64] = torch.tensor(tokens[occupancy - 32 : occupancy + 32])
    trace = torch.stack([snapshot["topk"].clone()] * 3)
    benchmark.validate(snapshot, trace, 8, 0, policy, "npu")
    snapshot["request_generations"].add_(1)
    snapshot["row_requests"][1] = -1
    benchmark.validate(snapshot, trace, 8, 0, policy, "npu")


def test_serving_dcp1_remains_old_policy(policy):
    snapshot = cold_case(1, 65536)
    _put_state(snapshot, 0, {t: t for t in range(512)})
    snapshot["topk"].copy_(make_trace(steps=2)[0])
    # The candidate's setting must have no effect on the non-DCP membership op.
    expected = membership_reference(snapshot, reference(snapshot)[0])
    actual = snapshot.clone("npu")
    serving_run(actual, 1, 0, policy)
    torch.npu.synchronize()
    assert_result(actual, expected, membership=True)


def test_serving_live_tokens_and_foreign_owner_filter(policy):
    snapshot = cold_case(1, 65536)
    snapshot["topk"].fill_(-1)
    snapshot["topk"][:, 0, :4] = torch.tensor([0, 128, 65536, 65664])
    actual = snapshot.clone("npu")
    serving_run(actual, 8, 0, policy)
    torch.npu.synchronize()
    assert actual.tensors[21][:, 0].cpu().tolist() == [2, 2]
    assert actual["topk"][:, 0, :3].cpu().tolist() == [[0, 12288, -1]] * 2
    assert actual["miss_counts"][0, 0].item() == 1
    assert actual["miss_tokens"][0, 0].item() == 0


def test_serving_single_query_row(policy):
    snapshot = make_case(1, mtp=1, scenario="cold")
    snapshot["boundary"].fill_(65536)
    snapshot["topk"].copy_(make_trace(steps=2)[0, :1])
    snapshot.tensors.extend([torch.tensor([2048], dtype=torch.int32), torch.zeros((1, 16), dtype=torch.int32)])
    old = [t for t in range(65536) if t // 128 % 8 == 0][:512]
    _put_state(snapshot, 0, dict(zip(old, reversed(range(len(old))), strict=True)))
    filtered = snapshot.clone("cpu")
    owned = [t for t in snapshot["topk"].flatten().tolist() if t // 128 % 8 == 0]
    filtered["topk"][0, 0] = torch.tensor(owned + [-1] * (2048 - len(owned)), dtype=torch.int32)
    expected = membership_reference(filtered, reference(filtered, free_first=bool(policy))[0])
    actual = snapshot.clone("npu")
    serving_run(actual, 8, 0, policy)
    torch.npu.synchronize()
    assert_result(actual, expected, membership=True)
    assert int(actual.tensors[21][0, 0]) == len(owned)


def test_serving_policy_cannot_change_after_use(policy):
    configure = torch.ops._C_ascend.configure_dsa_resident_dcp_free_slots
    configure(bool(policy))
    with pytest.raises(RuntimeError, match="frozen.*restart"):
        configure(not policy)
