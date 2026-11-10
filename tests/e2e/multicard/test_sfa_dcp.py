# SPDX-License-Identifier: Apache-2.0
"""Qualify counted SFA and the head exchange under a real HCCL replay."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")


@torch.inference_mode()
def test_counted_sfa_dcp_collectives_replay():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (4, 8):
        pytest.skip("launch with torchrun and 4 or 8 local NPU ranks")
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm_ascend.ops.triton.sfa_dcp import merge_sfa_dcp

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    comparisons = []
    try:
        rank = dist.get_rank()
        rows, heads = 2, 64
        local_heads = heads // world
        owned = slice(rank * local_heads, (rank + 1) * local_heads)
        rng = torch.Generator().manual_seed(174)
        all_q = torch.randn((rows, heads, 576), generator=rng, dtype=torch.bfloat16)
        cache = torch.randn((world, 128, 576), generator=rng, dtype=torch.bfloat16)
        q = all_q[:, owned].transpose(0, 1).contiguous().npu()
        k = cache[rank, :, :512].contiguous().view(1, 128, 1, 512).npu()
        pe = cache[rank, :, 512:].contiguous().view(1, 128, 1, 64).npu()
        table = torch.tensor([[0]], dtype=torch.int32, device="npu")
        q_lens = torch.tensor([2], dtype=torch.int32, device="npu")
        kv_lens = torch.tensor([128], dtype=torch.int32, device="npu")
        indices = torch.full((rows, 1, 2048), -1, dtype=torch.int32, device="npu")
        counts = torch.zeros((rows, 16), dtype=torch.int32, device="npu")
        scale = 576**-0.5

        def forward():
            gathered = torch.empty((heads, rows, 576), dtype=q.dtype, device=q.device)
            dist.all_gather_into_tensor(gathered, q)
            nope, rope = gathered.transpose(0, 1).split((512, 64), dim=-1)
            result = torch.ops._C_ascend.npu_sparse_flash_attention_dcp(
                nope.contiguous(), rope.contiguous(), k, pe, indices, counts, table, q_lens, kv_lens, scale
            )
            return merge_sfa_dcp(*result, dist.group.WORLD)

        forward()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = forward()
        for sizes in (
            [(0 if peer == 0 else 19 + peer, 11 * peer) for peer in range(world)],
            [(0, 0) for _ in range(world)],
            [(128 - peer, 128) for peer in range(world)],
        ):
            indices.fill_(-1)
            counts[:, 0].copy_(torch.tensor(sizes[rank], dtype=torch.int32))
            for row, count in enumerate(sizes[rank]):
                indices[row, 0, :count] = torch.arange(count, dtype=torch.int32, device="npu")
            graph.replay()
            reference = torch.zeros((rows, local_heads, 512), dtype=torch.float32)
            for row in range(rows):
                keys = torch.cat([cache[peer, :sizes[peer][row]] for peer in range(world)]).float()
                if keys.numel():
                    weights = (all_q[row, owned].float() @ keys.T * scale).softmax(dim=-1)
                    reference[row] = weights @ keys[:, :512]
            comparisons.append((output.cpu(), reference.bfloat16()))
        torch.npu.synchronize()
        graph.reset()
    finally:
        dist.destroy_process_group()
    # A rank-local parity failure must never strand peers in a later replay.
    for actual, expected in comparisons:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)
