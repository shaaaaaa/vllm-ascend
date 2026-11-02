# SPDX-License-Identifier: Apache-2.0
"""Installed MC2 V2 unequal-input/native-versus-capture probe, on 16 NPUs.

Launch pytest under torchrun on both hosts; see sfa_independent_dp.md. This is an
operator/graph contract test, NOT a full-SFA server, KV lifetime or throughput test.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

torch_npu = pytest.importorskip("torch_npu")


@torch.inference_mode()
def test_unequal_mc2_inputs_with_independent_native_and_graph_routes():
    assert int(os.environ.get("WORLD_SIZE", "1")) == 16, "Launch this qualification probe with 16 torchrun ranks"
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    # The operator documentation requires a domain dedicated to this pair.
    mc2 = dist.new_group(backend="hccl", timeout=timedelta(seconds=120))
    group = mc2._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    capacities, hidden = (1, 2, 4, 8), 2048
    global_bs = max(capacities) * world
    assert hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
    assert hasattr(torch_npu, "npu_moe_distribute_combine_v2")

    def transfer(x, ids, weights, quant):
        output = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=ids,
            scales=None,
            quant_mode=quant,
            group_ep=group,
            ep_world_size=world,
            ep_rank_id=rank,
            group_tp=group,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=world,
            global_bs=global_bs,
            expert_token_nums_type=0,
        )
        expanded, scales, assist, _, ep_counts, tp_counts, expand_scales = output[:7]
        if quant:
            # Identity expert computation after the dispatch's per-token quantization.
            expanded = (expanded.float() * scales.reshape(-1, 1)).to(x.dtype)
            tp_counts = torch.empty(1, dtype=torch.int32, device=x.device)
        return torch_npu.npu_moe_distribute_combine_v2(
            expand_x=expanded,
            expert_ids=ids,
            expert_scales=weights,
            assist_info_for_combine=assist,
            ep_send_counts=ep_counts,
            tp_send_counts=tp_counts,
            expand_scales=expand_scales,
            group_ep=group,
            ep_world_size=world,
            ep_rank_id=rank,
            group_tp=group,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=world,
            global_bs=global_bs,
        )

    try:
        for quant in (0, 2):
            captures = {}
            for size in capacities:
                x = torch.zeros((size, hidden), dtype=torch.bfloat16, device="npu")
                ids = (
                    torch.arange(size, device="npu", dtype=torch.int32)[:, None]
                    + torch.tensor([[rank, rank + 3]], device="npu", dtype=torch.int32)
                ) % world
                weights = torch.full((size, 2), 0.5, dtype=torch.float32, device="npu")
                for _ in range(2):
                    transfer(x, ids, weights, quant)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    # Two target-like coupled layers, followed by one draft-like
                    # layer. All routes must preserve this same phase sequence.
                    value = transfer(x, ids, weights, quant)
                    value = transfer(value, ids, weights, quant)
                    result = transfer(value, ids, weights, quant)
                captures[size] = (graph, x, ids, weights, result)
                torch.npu.synchronize()
            for iteration in range(8):
                dp = rank // 4
                size = capacities[(dp + iteration) % len(capacities)]
                graph, x, ids, weights, result = captures[size]
                values = torch.arange(size * hidden, device="npu", dtype=torch.float32).reshape(size, hidden)
                x.copy_(torch.sin(values * 0.17 + rank * 0.71 + iteration * 0.3).to(x.dtype))
                if (dp + iteration) % 2:
                    value = transfer(x, ids, weights, quant)
                    value = transfer(value, ids, weights, quant)
                    observed = transfer(value, ids, weights, quant)
                else:
                    graph.replay()
                    observed = result
                torch.npu.synchronize()
                torch.testing.assert_close(observed.cpu(), x.cpu(), rtol=0, atol=0.04 if quant else 0)
            dist.barrier()  # Default group, never the MC2 communication domain.
    finally:
        dist.destroy_process_group()
