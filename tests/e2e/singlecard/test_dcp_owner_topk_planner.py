# SPDX-License-Identifier: Apache-2.0
"""Compare the actual owner and contiguous native planners with changing replay inputs."""
import numpy as np
import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("position_dtype,reject", [(None, False), (torch.int64, False), (torch.int32, True)])
@torch.inference_mode()
def test_direct_owner_planner_replay(bounded, width, position_dtype, reject):
    from vllm_ascend.distributed.kv_transfer.sparse_offload import resident_sorted_cache as resident
    from vllm_ascend.utils import enable_custom_op
    from vllm_ascend.worker.dcp_indexer import GatheredTopk, planner_owner_rows

    assert enable_custom_op()
    resident.configure_resident_kernels()
    requests, k = 3, 2048
    rows = requests * width
    packet = torch.empty(8, 1, k, dtype=torch.int32, device="npu")
    output = torch.empty_like(packet[:rows])
    raw = torch.empty_like(output[:, 0])
    inverse = torch.full((rows - 1,), -1, dtype=torch.long, device="npu")
    lanes = (torch.arange(rows, dtype=torch.long) << 32).npu() if bounded else None
    positions = torch.zeros(rows, dtype=position_dtype, device="npu") if position_dtype else None
    owner = GatheredTopk(packet, inverse, positions, raw, rows, reject, lanes)
    req_ids = torch.empty(rows, dtype=torch.int32, device="npu")
    bounds = torch.empty_like(req_ids)
    states = torch.arange(requests, dtype=torch.int32, device="npu")
    generations = torch.ones(requests, dtype=torch.int64, device="npu")
    table = torch.arange(requests * 40, dtype=torch.int32, device="npu").view(requests, 40)

    def buffers():
        state = resident.allocate_sorted_resident_state(requests, requests, width, device="npu")
        workspace = resident.allocate_sorted_resident_workspace(requests, width, device="npu")
        layout = resident.ResidentDcpLayout(8, 3, torch.full_like(states, 4096),
                                            torch.zeros(rows, 16, dtype=torch.int32, device="npu"))
        return state, workspace, layout

    def run(ids, boundary, requests_by_row, bufs, source=None):
        state, workspace, layout = bufs
        resident.prepare_resident_sharded_union_(ids, boundary, requests_by_row, states, generations,
                                                 state, workspace, mtp=width, dcp=layout, owner_topk=source)
        resident.prepare_sorted_resident_cache_fused_(ids, table, states, generations, state, workspace,
                                                      block_size=128, dcp=layout, owner_topk=source)

    candidate, reference = buffers(), buffers()
    req_ids.fill_(-1)
    bounds.zero_()
    run(output, bounds, req_ids, candidate, owner)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            run(output, bounds, req_ids, candidate, owner)
        candidate[0].counts.zero_()
        candidate[0].generations.fill_(-1)
        for step, lengths in enumerate(((1, width, 1), (width, 1, width), (0, 0, 0), (width, width, width))):
            starts = np.cumsum((0, *lengths))
            active = int(starts[-1])
            req = (torch.tensor([r for r, n in enumerate(lengths) for _ in range(n)] + [-1] * (rows - active))
                   if bounded else torch.arange(requests).repeat_interleave(width))
            if not bounded:
                req[active:] = -1
            perm = (torch.arange(rows - 1) * 3 + step) % len(packet)
            perm[active:] = -1
            if step == 0:
                perm[1] = -1
            elif step == 3:
                perm[0] = len(packet)
            packet_lanes = torch.from_numpy(planner_owner_rows(starts, np.append(perm.numpy(), -1), width))
            order = packet_lanes >> 32 if bounded else torch.arange(rows)
            data = (torch.arange(packet.numel(), dtype=torch.int32).reshape(packet.shape) + step * 137) % 8261
            pos = torch.tensor([8101, -1, 5900, 7309, 4050, 6700])[:rows]
            boundary = torch.arange(rows, dtype=torch.int32) * 57 + 5100
            selected = torch.full((rows, 1, k), -1, dtype=torch.int32)
            for row, src in enumerate(perm.tolist()):
                if 0 <= src < len(packet) and not (reject and pos[row] < 0):
                    selected[row] = data[src]
            filtered = selected if positions is None else torch.where(selected <= pos[:, None, None], selected, -1)
            original = filtered[order].npu()
            packet.copy_(data)
            inverse.copy_(perm)
            req_ids.copy_(req)
            bounds.copy_(boundary)
            if lanes is not None:
                lanes.copy_(packet_lanes)
            if positions is not None:
                positions.copy_(pos)
            graph.replay()
            run(original, boundary[order].npu(), req[order].int().npu(), reference)
            expected = original.cpu()[order.argsort()]
            if bounded:
                expected[req < 0] = -1
            assert torch.equal(output.cpu(), expected) and torch.equal(raw.cpu(), selected[:, 0])
            assert torch.equal(packet.cpu(), data)
            torch.testing.assert_close(candidate[2].selected_counts[:, 0].cpu(),
                                       reference[2].selected_counts[:, 0].cpu()[order.argsort()])
            c, r = candidate[1], reference[1]
            assert torch.equal(c.miss_counts[:, 0], r.miss_counts[:, 0])
            for row, count in enumerate(r.miss_counts[:, 0].cpu().tolist()):
                assert torch.equal(c.miss_tokens[row, :count], r.miss_tokens[row, :count])
                assert torch.equal(c.target_slots[row, :count], r.target_slots[row, :count])
            assert torch.equal(candidate[0].counts, reference[0].counts)
            assert torch.equal(candidate[0].generations, reference[0].generations)
            for state_row, shard in torch.nonzero(reference[0].counts[..., 0].cpu()).tolist():
                n = int(reference[0].counts[state_row, shard, 0].item())
                for name in ("tokens", "slots"):
                    assert torch.equal(getattr(candidate[0], name)[state_row, shard, :n],
                                       getattr(reference[0], name)[state_row, shard, :n])
    finally:
        torch.npu.synchronize()
        graph.reset()
