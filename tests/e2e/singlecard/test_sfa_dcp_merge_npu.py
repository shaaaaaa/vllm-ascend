# SPDX-License-Identifier: Apache-2.0
"""Compile and replay DCP exchange kernels on one NPU, without HCCL peers."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")


@torch.inference_mode()
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_query_unpack_bit_exact_capture_without_hccl(world, dtype):
    from vllm_ascend.ops.triton.sfa_dcp import _gather_sfa_dcp_queries

    # Isolate Triton compilation from HCCL. The multicard benchmark qualifies
    # the real collective separately with the same helper and strided inputs.
    for rows, heads in ((1, 8), (3, 1), (8, 8)):
        raw = torch.empty((world, rows, heads, 576), dtype=dtype, device="npu")
        group = SimpleNamespace(world_size=world, all_gather=lambda packet, dim, raw=raw: raw.flatten(0, 1))
        query, rope = raw[0].split((512, 64), dim=-1)
        raw.zero_()
        _gather_sfa_dcp_queries(query, rope, group)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = _gather_sfa_dcp_queries(query, rope, group)
        try:
            for step in range(3):
                bits = (torch.arange(raw.numel(), dtype=torch.int32) + step * 7919) % 65536 - 32768
                bits = bits.to(torch.int16).reshape(raw.shape)
                raw.view(torch.int16).copy_(bits)
                graph.replay()
                expected = bits.permute(1, 0, 2, 3).reshape(rows, world * heads, 576).split((512, 64), dim=-1)
                for actual, wanted in zip(outputs, expected):
                    assert torch.equal(actual.view(torch.int16).cpu(), wanted)
        finally:
            torch.npu.synchronize()
            graph.reset()


@torch.inference_mode()
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dcp_exchange_pack_combine_capture(world, dtype):
    from vllm_ascend.ops.triton.sfa_dcp import _combine, _pack

    rows, local_heads, dim = 4, 2, 512
    heads = world * local_heads
    generator = torch.Generator().manual_seed(915)
    partial = torch.randn((world, rows, heads, dim), generator=generator, dtype=dtype)
    maximum = torch.randn((world, rows, heads), generator=generator) * 12
    total = torch.rand((world, rows, heads), generator=generator) * 2048 + 1
    total[0, 0].zero_()
    total[:, 1].zero_()
    maximum[:, 2].fill_(2**24)
    total[:, 2].zero_()
    total[0, 2].fill_(1)
    total[1, 2].fill_(2)
    partial[0, 2].fill_(1)
    partial[1, 2].fill_(7)
    maximum[:, 3].zero_()
    partial[:, 3].fill_(1e36 if dtype == torch.bfloat16 else 60000)
    partial[total == 0] = 0
    partial_npu, maximum_npu, total_npu = (t.npu() for t in (partial, maximum, total))
    send = torch.empty((world, world, rows, local_heads, dim + 16), dtype=dtype, device="npu")
    recv = torch.empty_like(send)
    output = torch.empty((world, rows, local_heads, dim), dtype=dtype, device="npu")

    def forward():
        for peer in range(world):
            _pack[(1,)](partial_npu[peer], maximum_npu[peer], total_npu[peer], send[peer],
                        rows, heads, local_heads, dim)
        # Simulate only packet routing; both production Triton kernels run unchanged.
        recv.copy_(send.transpose(0, 1))
        for peer in range(world):
            _combine[(1,)](recv[peer], output[peer], rows * local_heads, world, dim)

    forward()  # Compile outside capture; fails here on the old unsigned casts.
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            forward()
        for all_empty in (False, True):
            if all_empty:
                total_npu.zero_()
                partial_npu.zero_()
            graph.replay()
            actual = output.cpu()
            if all_empty:
                assert not actual.any()
            else:
                lse = maximum.double() + total.double().log()
                lse[:, 1].zero_()
                expected = (lse.softmax(dim=0)[..., None] * partial.double()).sum(dim=0).to(dtype)
                expected = expected.reshape(rows, world, local_heads, dim).transpose(0, 1)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
    finally:
        torch.npu.synchronize()
        graph.reset()
