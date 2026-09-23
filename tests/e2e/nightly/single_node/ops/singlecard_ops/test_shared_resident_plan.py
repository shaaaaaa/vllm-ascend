# SPDX-License-Identifier: Apache-2.0
"""NPU planner parity and layer-specific K/PE reachability; no serving claim."""

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.attention import sfa_v1
from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sorted_cache import (
    SharedResidentPlan,
    allocate_sorted_resident_state,
    allocate_sorted_resident_workspace,
)
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("captured", [False, True])
def test_shared_plan_matches_independent_layers_and_reaches_own_kv(mtp, captured, monkeypatch):
    assert enable_custom_op()
    monkeypatch.setenv("VLLM_ASCEND_MTP_DRAFT_DEBUG", "0")
    requests, members, block_size = 2, 3, 128
    capacity = mtp * 2048

    def implementation():
        impl = object.__new__(sfa_v1.AscendSFAImpl)
        impl.dsa_resident_cache = True
        impl.decode_threshold = mtp
        impl.block_size = block_size
        impl.skip_topk = False
        impl.shared_resident_plan = None
        impl._sorted_resident_state = allocate_sorted_resident_state(
            requests, requests, mtp, device=torch.device("npu")
        )
        impl._sorted_resident_workspace = allocate_sorted_resident_workspace(requests, mtp, device=torch.device("npu"))
        impl._sorted_resident_workspace_views = {}
        return impl

    baseline = [implementation() for _ in range(members)]
    producer = implementation()
    group = SharedResidentPlan(
        tuple(str(i) for i in range(members)),
        producer._sorted_resident_state,
        producer._sorted_resident_workspace,
        torch.empty((requests * mtp, 1, 2048), dtype=torch.int32, device="npu"),
        mtp,
        block_size,
        active=True,
    )
    shared = [producer]
    for _ in range(members - 1):
        consumer = object.__new__(sfa_v1.AscendSFAImpl)
        consumer.__dict__.update(producer.__dict__)
        consumer.skip_topk = True
        shared.append(consumer)
    for impl in shared:
        impl.shared_resident_plan = group
        impl._sorted_resident_workspace_views = group.workspace_views

    # Fragmented physical blocks, with no overlap between requests.
    table = torch.randperm(requests * capacity // block_size, generator=torch.Generator().manual_seed(1))
    table = table.reshape(requests, -1).to(dtype=torch.int32, device="npu")
    rows = torch.arange(requests, dtype=torch.int32, device="npu").repeat_interleave(mtp)
    states = torch.arange(requests, dtype=torch.int32, device="npu")
    generations = torch.ones(requests, dtype=torch.int64, device="npu")
    boundaries = torch.full((requests * mtp,), 100_000, dtype=torch.int32, device="npu")
    caches = [torch.full((requests * capacity, 2), -1.0, device="npu") for _ in shared]

    def run(impl, raw):
        return impl._prepare_decode_sparse_indices(
            raw,
            boundaries,
            rows,
            table,
            None,
            None,
            None,
            states,
            generations,
            local_to_union_workspace=None,
            shard_packed_workspace=None,
            shard_mapping_workspace=None,
            shard_counts_workspace=None,
            staged_mtp=mtp,
            need_packed=True,
            clear_invalid_rows=True,
        )

    def source(offset):
        return (
            torch.stack(
                [
                    torch.roll(torch.arange(2048, dtype=torch.int32) + offset + request * 5000 + lane * 1024, 137)
                    for request in range(requests)
                    for lane in range(mtp)
                ]
            )
            .unsqueeze(1)
            .npu()
        )

    raw = source(0)
    bridges = [tuple(torch.empty_like(t) for t in group.plans[requests]) for _ in shared]

    def shared_step():
        for impl, bridge in zip(shared, bridges):
            for value, destination in zip(run(impl, raw), bridge):
                destination.copy_(value)

    if captured:
        shared_step()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            shared_step()
        # Capture/warmup must not seed the independent comparison's real state.
        group.state.counts.zero_()
        group.state.generations.fill_(-1)
        torch.npu.synchronize()

    for offset in (0, 512, 0, 0):
        raw.copy_(source(offset))
        reference = [run(impl, raw.clone()) for impl in baseline]
        if captured:
            graph.replay()
        else:
            with patch.object(
                sfa_v1, "prepare_resident_sharded_union_", wraps=sfa_v1.prepare_resident_sharded_union_
            ) as union:
                shared_step()
                assert union.call_count == 1
        actual = bridges
        for layer, (expected, result) in enumerate(zip(reference, actual)):
            assert torch.equal(expected[0], result[0])
            assert torch.equal(expected[2], result[2])
            for request in range(requests):
                count = int(result[2][request].cpu())
                assert torch.equal(expected[1][request, :count], result[1][request, :count])
                assert torch.equal(expected[3][request, :count], result[3][request, :count])
                tokens = result[1][request, :count].long()
                slots = result[3][request, :count].long()
                # Distinct layer/request/token values in both K and PE planes.
                values = tokens[:, None] * 4 + torch.arange(2, device="npu")
                values = (values + layer * 1_000_000 + request * 100_000).float()
                caches[layer].index_copy_(0, slots, values)
                logical = result[0][request * mtp : (request + 1) * mtp].reshape(-1).long()
                physical = table[request].long()[logical // block_size] * block_size + logical % block_size
                reached = caches[layer][physical]
                original = raw[request * mtp : (request + 1) * mtp].reshape(-1).long()
                expected_kv = original[:, None] * 4 + torch.arange(2, device="npu")
                expected_kv = (expected_kv + layer * 1_000_000 + request * 100_000).float()
                assert torch.equal(reached, expected_kv)
