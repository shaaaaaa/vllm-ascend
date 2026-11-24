# SPDX-License-Identifier: Apache-2.0
"""Byte parity at both packet boundaries, including changing counts on graph replay."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


@torch.inference_mode()
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packet_io_replay(world, dtype):
    from vllm_ascend.ops.triton.sfa_dcp import _pack
    from vllm_ascend.ops.triton.sfa_query import rope_sfa_dcp_query_packet
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    torch.npu.set_device(0)
    rows, heads, pages = 5, 64, 16
    local = heads // world
    rng = torch.Generator().manual_seed(154)

    def data(*shape):
        return torch.randn(shape, generator=rng, dtype=dtype).npu()

    # As in serving, latent is a head-major BMM view and raw RoPE is a projection slice.
    q = data(heads, rows, 512).transpose(0, 1)
    raw = data(rows, heads, 192)[..., -64:]
    cos, sin = (data(rows, 1, 1, 64) for _ in range(2))
    # Force both signs of zero in the rotated output, independently of the random inputs.
    raw[0, 0, :4].zero_()
    cos[0, 0, 0, :2].fill_(-1)
    sin[0, 0, 0, :2] = sin.new_tensor([1, -1])
    k, kr = (data(2 * pages, 128, 1, dim) for dim in (512, 64))
    table = torch.arange(2 * pages, dtype=torch.int32, device="npu").view(2, pages)
    ends = table.new_tensor([2, 4])
    lengths = table.new_full((2,), 2048)
    counts = table.new_zeros((rows, 16))
    indices = table.new_full((rows, 1, 2048), -1)
    ops = torch.ops._C_ascend
    meta_args = [v.to("meta") for v in (k, kr, indices, counts, table, ends, lengths)]
    for args in ((q.to("meta"), raw.to("meta")), (q.new_empty((world, rows, local, 576), device="meta"), None)):
        out = ops.npu_sparse_flash_attention_dcp_packet(*args, *meta_args, 576**-0.5, world)
        assert out.shape == (world, rows, local, 528) and out.dtype == dtype

    def run():
        packets, contiguous_packets = [], []
        for peer in range(world):
            part = slice(peer * local, (peer + 1) * local)
            packets.append(rope_sfa_dcp_query_packet(q[:, part], raw[:, part], cos, sin))
            contiguous_packets.append(rope_sfa_dcp_query_packet(q[:, part], raw[:, part], cos, sin, contiguous=True))
        wire = torch.stack(packets)
        contiguous_wire = torch.stack(contiguous_packets)
        r = torch_npu.npu_interleave_rope(raw.unsqueeze(2).contiguous(), cos, sin).squeeze(2)
        expected_q = torch.cat((q, r), -1).reshape(rows, world, local, 576).transpose(0, 1).contiguous()
        meta = (k, kr, indices, counts, table, ends, lengths, 576**-0.5)
        partial, maximum, total = ops.npu_sparse_flash_attention_dcp(q.contiguous(), r, *meta)
        expected = q.new_empty((world, rows, local, 528))
        _pack[(8,)](partial, maximum, total, expected, rows, heads, local, 512)
        output_only = ops.npu_sparse_flash_attention_dcp_packet(q.contiguous(), r, *meta, world)
        direct = ops.npu_sparse_flash_attention_dcp_packet(expected_q, None, *meta, world)
        fused = ops.npu_sparse_flash_attention_dcp_packet(wire, None, *meta, world)
        fused_contiguous = ops.npu_sparse_flash_attention_dcp_packet(contiguous_wire, None, *meta, world)
        return wire, contiguous_wire, expected_q, expected, output_only, direct, fused, fused_contiguous

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            wire, contiguous_wire, expected_q, expected, *sends = run()
        for sizes, active_end, first_length in (
            ([1, 513, 0, 2048, 0], 4, 2048),
            ([0] * rows, 0, 2048),
            ([128, 17, 1025, 0, 0], 3, 2048),
            ([0, 0, 13, 0, 7], 5, 0),
        ):
            lengths.copy_(torch.tensor([first_length, 2048], dtype=torch.int32))
            ends.copy_(torch.tensor([min(2, active_end), active_end], dtype=torch.int32))
            counts[:, 0].copy_(torch.tensor(sizes, dtype=torch.int32))
            ids = torch.full((rows, 1, 2048), -1, dtype=torch.int32)
            for row, count in enumerate(sizes):
                ids[row, 0, :count] = torch.randperm(2048, generator=rng, dtype=torch.int32)[:count]
            indices.copy_(ids)
            raw.neg_()
            graph.replay()
            torch.npu.synchronize()
            for query_packet in (wire, contiguous_wire):
                assert torch.equal(query_packet.cpu().view(torch.int16), expected_q.cpu().view(torch.int16))
            for send in sends:
                assert torch.equal(send.cpu().view(torch.int16), expected.cpu().view(torch.int16))
    finally:
        torch.npu.synchronize()
        graph.reset()
