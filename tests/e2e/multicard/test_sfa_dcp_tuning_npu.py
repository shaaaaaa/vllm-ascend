# SPDX-License-Identifier: Apache-2.0
"""Qualify DCP tuning on NPUs with changing native SFA inputs and graph replay."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")


@pytest.fixture(scope="module")
def tuning_world():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (2, 4, 8):
        pytest.skip("launch with torchrun and 2, 4 or 8 local NPUs")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    try:
        yield world, control
    finally:
        dist.destroy_process_group()


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("early_unpack", [False, True])
def test_native_dcp_early_gather_replay(tuning_world, dtype, early_unpack):
    from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

    from vllm_ascend.ops.triton.sfa_dcp import execute_sfa_dcp, start_sfa_dcp_query_gather
    from vllm_ascend.utils import create_hccl_pg_options

    world, control = tuning_world
    rank, rows, heads = dist.get_rank(), 3, 64
    local_heads = heads // world
    owned = slice(rank * local_heads, (rank + 1) * local_heads)
    device_group = dist.new_group(backend="hccl", pg_options=create_hccl_pg_options("dcp"))
    group = SimpleNamespace(world_size=world, device_group=device_group)
    group.all_gather = lambda packet, dim: DeviceCommunicatorBase.all_gather(group, packet, dim)
    rng = torch.Generator().manual_seed(913)
    all_queries = torch.randn((rows, heads, 576), generator=rng, dtype=dtype)
    keys = torch.randn((world, 128, 576), generator=rng, dtype=dtype)
    query = all_queries[:, owned, :512].transpose(0, 1).contiguous().npu().transpose(0, 1)
    rope = all_queries[:, owned, 512:].contiguous().npu()
    caches = tuple(
        keys[rank, :, first:last].contiguous().reshape(1, 128, 1, last - first).npu()
        for first, last in ((0, 512), (512, 576))
    )
    topk = torch.full((rows, 1, 2048), -1, dtype=torch.int32, device="npu")
    count_input = torch.zeros((rows, 16), dtype=torch.int32, device="npu")
    counts = torch.empty_like(count_input)
    table = torch.zeros((1, 1), dtype=torch.int32, device="npu")
    ends = torch.tensor([rows], dtype=torch.int32, device="npu")
    lengths = torch.tensor([128], dtype=torch.int32, device="npu")
    graphs, outputs = [], []
    unpack_stream = torch.npu.Stream(device=query.device) if early_unpack else None

    def forward(candidate):
        pending = start_sfa_dcp_query_gather(query, rope, group) if candidate else None
        try:
            if pending is not None and unpack_stream is not None:
                pending.start_unpack(query.shape[-1], unpack_stream, 12)
            # Real data dependency: attention must see the latest counts while
            # query gathering can execute independently of their publication.
            counts.copy_(count_input)
            return execute_sfa_dcp(
                query,
                rope,
                caches,
                topk,
                counts,
                table,
                ends,
                lengths,
                576**-0.5,
                group,
                query_gather=pending,
            )
        finally:
            if pending is not None:
                pending.wait()

    try:
        for candidate in (False, True):
            forward(candidate)
        torch.npu.synchronize()
        for candidate in (False, True):
            graph = torch.npu.NPUGraph()
            graphs.append(graph)
            with torch.npu.graph(graph):
                outputs.append(forward(candidate))
        for step in range(4):
            values = all_queries * (-1 if step % 2 else 1)
            query.copy_(values[:, owned, :512])
            rope.copy_(values[:, owned, 512:])
            sizes = [(0, 0, 0) if step == 1 else (128 - peer, peer * 11, 0) for peer in range(world)]
            count_input[:, 0].copy_(torch.tensor(sizes[rank], dtype=torch.int32))
            topk.fill_(-1)
            for row, count in enumerate(sizes[rank]):
                topk[row, 0, :count] = torch.arange(count - 1, -1, -1, dtype=torch.int32, device="npu")
            for graph in graphs:
                graph.replay()
            torch.npu.synchronize()
            error = None
            try:
                baseline, candidate = (tensor.cpu() for tensor in outputs)
                torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
                reference = torch.zeros_like(candidate, dtype=torch.float32)
                for row in range(rows):
                    selected = torch.cat([keys[peer, : sizes[peer][row]] for peer in range(world)]).float()
                    if selected.numel():
                        weights = (values[row, owned].float() @ selected.T * 576**-0.5).softmax(-1)
                        reference[row] = weights @ selected[:, :512]
                torch.testing.assert_close(candidate, reference.to(dtype), rtol=3e-2, atol=2e-2)
            except Exception as exc:
                error = f"rank={rank} replay={step}: {exc}"
            errors = [None] * world
            dist.all_gather_object(errors, error, group=control)
            if any(errors):
                raise AssertionError("\n".join(item for item in errors if item))
    finally:
        torch.npu.synchronize()
        for graph in graphs:
            graph.reset()
        dist.destroy_process_group(device_group)
