# SPDX-License-Identifier: Apache-2.0
"""Real NPU cache-miss and numerical checks after compile-only startup."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")


def forbid_compile(*args, **kwargs):
    raise AssertionError("decode kernel requested compilation after startup warmup")


def guard(monkeypatch, *kernels):
    for kernel in kernels:
        monkeypatch.setattr(kernel, "compile", forbid_compile)


@pytest.fixture(autouse=True)
def device():
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    torch.npu.set_device(0)
    init_device_properties_triton()
    yield
    torch.npu.synchronize()


@torch.inference_mode()
def test_owner_topk_replay_changes_owners_and_rejected_rows(monkeypatch):
    from vllm_ascend.ops.triton import sfa_dcp as ops
    from vllm_ascend.ops.triton.decode_warmup import warmup_owner_topk

    width, rows = 2048, 6
    warmup_owner_topk(width, torch.int32, (width, 1), torch.int64, draft=True)
    guard(monkeypatch, ops._restore_owner_topk)
    packets = (torch.arange(8 * width, dtype=torch.int32).reshape(8, 1, width) % 2063 - 2).npu()
    inverse = torch.zeros(rows - 1, dtype=torch.int64, device="npu")
    positions = torch.zeros(rows, dtype=torch.int32, device="npu")
    private = torch.empty_like(packets[:rows])
    raw = torch.empty_like(private[:, 0])
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            ops.restore_owner_topk(packets, inverse, private, positions, rows - 2, raw, causal=True, reject=True)
        for owners, pos in (([4, -1, 3, 1, 7], [74, 0, -1, 19, 600, -1]),
                            ([7, 0, -1, 2, 1], [-1, 500, 30, 7, 350, 10])):
            inverse.copy_(torch.tensor(owners))
            positions.copy_(torch.tensor(pos, dtype=torch.int32))
            graph.replay()
            expected = torch.full(private.shape, -1, dtype=torch.int32)
            cpu_packets = packets.cpu()
            for row, src in enumerate(owners):
                if src >= 0 and pos[row] >= 0:
                    expected[row] = cpu_packets[src]
            torch.testing.assert_close(raw.cpu(), expected[:, 0])
            expected[(expected < 0) | (expected > torch.tensor(pos)[:, None, None])] = -1
            expected[rows - 2:] = -1
            torch.testing.assert_close(private.cpu(), expected)
    finally:
        torch.npu.synchronize()
        graph.reset()


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_dcp_exchange_and_query_layouts(monkeypatch, world, dtype):
    from vllm_ascend.ops.triton import sfa_dcp as ops
    from vllm_ascend.ops.triton import sfa_query as query_ops
    from vllm_ascend.ops.triton.decode_warmup import warmup_dcp, warmup_query

    heads, dim, rope_dim = 64 // world, 512, 64
    warmup_dcp(2048, heads, dtype, world, world - 1, 128)
    warmup_query(heads, dim, rope_dim, 192)
    guard(monkeypatch, ops._unpack_queries, ops._pack, ops._combine, query_ops._copy_query_bridge)
    for rows in (1, 2, 3, 6, 17, 32, 33):
        for transposed in (False, True):
            query = torch.randn((heads, rows, dim) if transposed else (rows, heads, dim), dtype=dtype, device="npu")
            if transposed:
                query = query.transpose(0, 1)
            for rope_stride in (64, 192):
                rope = torch.randn((rows, heads, rope_stride), dtype=dtype, device="npu")[..., -rope_dim:]
                packet = query_ops.pack_sfa_dcp_queries(query, rope)
                torch.testing.assert_close(packet.cpu(), torch.cat((query.cpu(), rope.cpu()), -1), rtol=0, atol=0)
                qout, rout = torch.empty(query.shape, dtype=dtype, device="npu"), torch.empty_like(rope.contiguous())
                query_ops.copy_sfa_query_bridge(query, rope, qout, rout)
                torch.testing.assert_close(qout.cpu(), query.cpu(), rtol=0, atol=0)
                torch.testing.assert_close(rout.cpu(), rope.cpu(), rtol=0, atol=0)
        gathered = packet.repeat(world, 1, 1)
        qout, rout = ops._unpack_sfa_dcp_queries(gathered, rows, heads, dim, rope_dim, world)
        expected = packet.cpu().repeat(1, world, 1)
        torch.testing.assert_close(qout.cpu(), expected[..., :dim], rtol=0, atol=0)
        torch.testing.assert_close(rout.cpu(), expected[..., dim:], rtol=0, atol=0)
        partial = torch.randn((rows, world * heads, dim), dtype=dtype, device="npu")
        maximum = torch.zeros((rows, world * heads), device="npu")
        total = torch.full_like(maximum, 3)
        send = torch.empty((world, rows, heads, dim + 16), dtype=dtype, device="npu")
        ops._pack[(7,)](partial, maximum, total, send, rows, world * heads, heads, dim)
        expected = partial.cpu().reshape(rows, world, heads, dim).permute(1, 0, 2, 3)
        torch.testing.assert_close(send[..., :dim].cpu(), expected, rtol=0, atol=0)
        # Simulate equal peer contributions without a communicator or graph.
        recv, output = send[:1].repeat(world, 1, 1, 1), torch.empty((rows, heads, dim), dtype=dtype, device="npu")
        ops._combine[(7,)](recv, output, rows * heads, world, dim)
        torch.testing.assert_close(output.cpu(), partial[:, :heads].cpu(), rtol=0.01, atol=0.001)
    # Replay must retain the dynamic-loop behavior and newly produced packets.
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            ops._pack[(7,)](partial, maximum, total, send, rows, world * heads, heads, dim)
            recv.copy_(send[:1].expand_as(recv))
            ops._combine[(7,)](recv, output, rows * heads, world, dim)
        for value in (0.25, -2):
            partial.fill_(value)
            graph.replay()
            torch.testing.assert_close(output.cpu(), torch.full_like(output.cpu(), value), rtol=0.01, atol=0.001)
    finally:
        torch.npu.synchronize()
        graph.reset()


@torch.inference_mode()
def test_dcp_dense_mixed_and_draft_metadata(monkeypatch):
    from vllm_ascend.ops.triton import sfa_dcp as ops
    from vllm_ascend.ops.triton.decode_warmup import warmup_dcp
    from vllm_ascend.ops.triton.spec_decode.utils import prepare_dcp_mtp_draft_kernel

    world, rank, block = 4, 3, 128
    warmup_dcp(2048, 16, torch.bfloat16, world, rank, block, draft=True)
    guard(monkeypatch, ops._mask_topk, ops._prepare_topk, prepare_dcp_mtp_draft_kernel)
    for rows in (1, 2, 3, 6, 17, 33):
        raw_cpu = (torch.arange(rows * 2048, dtype=torch.int32) % 4097 - 1).reshape(rows, 1, 2048)
        raw, counts = raw_cpu.npu(), torch.zeros((rows, 16), dtype=torch.int32, device="npu")
        for dtype in (torch.int32, torch.int64):
            positions_cpu = torch.arange(rows, dtype=dtype) * 113 + 1000
            positions = positions_cpu.npu()
            for mode in ("global", "local", "mixed"):
                owners = torch.arange(rows, dtype=torch.int32) % 2 - 1 if mode == "mixed" else None
                for active in {0, 1, rows}:
                    output = torch.empty_like(raw)
                    ops.prepare_sfa_dcp_topk(
                        raw,
                        positions,
                        output,
                        counts,
                        active,
                        world,
                        rank,
                        block,
                        local=mode == "local",
                        row_requests=None if owners is None else owners.npu(),
                    )
                    actual, local_counts = output.cpu(), counts.cpu()
                    for row in range(rows):
                        tokens = raw_cpu[row, 0]
                        valid = (tokens >= 0) & (tokens <= positions_cpu[row]) & (row < active)
                        local = mode == "local" or (mode == "mixed" and (row >= active or owners[row] < 0))
                        if local:
                            valid &= tokens // block % world == rank
                            expected = torch.full_like(tokens, -1)
                            selected = tokens[valid]
                            expected[: len(selected)] = selected // (world * block) * block + selected % block
                            assert local_counts[row, 0] == len(selected)
                        else:
                            expected = torch.where(valid, tokens, -1)
                        assert torch.equal(actual[row, 0], expected)
        starts = torch.arange(rows + 1, dtype=torch.int32, device="npu") * 2
        lengths = torch.arange(rows, dtype=torch.int32, device="npu") + 1024
        origins = torch.full_like(lengths, 4096)
        pos = torch.arange(2 * rows, dtype=torch.int32, device="npu")
        for slot_dtype in (torch.int32, torch.int64):
            slots = pos.to(slot_dtype)
            for reject in (None, torch.ones_like(lengths)):
                outputs = [torch.empty_like(pos) for _ in range(3)]
                local_lengths = torch.empty_like(lengths)
                prepare_dcp_mtp_draft_kernel[(min(rows, 7),)](
                    starts,
                    lengths,
                    reject,
                    origins,
                    pos,
                    slots,
                    slots,
                    *outputs,
                    local_lengths,
                    rows,
                    WORLD=world,
                    RANK=rank,
                    BLOCK_SIZE=block,
                    TILE=64,
                )
                expected = pos.cpu()
                if reject is not None:
                    expected[1::2] = -1
                assert all(torch.equal(output.cpu(), expected) for output in outputs)
                visible = lengths.cpu() - int(reject is not None)
                expected_lengths = (
                    4096
                    + visible // (world * block) * block
                    + (visible % (world * block) - rank * block).clamp(0, block)
                )
                assert torch.equal(local_lengths.cpu(), expected_lengths)


@torch.inference_mode()
def test_paired_metadata_counts_and_slot_types(monkeypatch):
    from vllm_ascend.ops.triton.decode_warmup import warmup_indexer_metadata
    from vllm_ascend.ops.triton.spec_decode.indexer_c8_metadata import _map_indexer_metadata, map_indexer_metadata

    warmup_indexer_metadata(18, 36)
    guard(monkeypatch, _map_indexer_metadata)
    mapping = torch.arange(128, dtype=torch.int32, device="npu") * 2
    for rows in (1, 2, 3, 17, 33):
        for stride in (18, 36):
            table = (torch.arange(rows * stride, dtype=torch.int32, device="npu") % 128).reshape(rows, stride)[:, :18]
            for slot_dtype in (torch.int32, torch.int64):
                slots = torch.arange(rows * 2 + 1, dtype=slot_dtype, device="npu") * 129 - 1
                out_table, out_slots = (
                    torch.empty_like(table),
                    torch.empty(slots.shape, dtype=torch.int64, device="npu"),
                )
                map_indexer_metadata(table, slots, mapping, out_table, out_slots)
                assert torch.equal(out_table.cpu(), table.cpu() * 2)
                expected = torch.where(slots.cpu() < 0, -1, slots.cpu() // 128 * 256 + slots.cpu() % 128)
                assert torch.equal(out_slots.cpu(), expected)


@pytest.mark.parametrize("heads", [1, 64])
@pytest.mark.parametrize("neox", [False, True])
@torch.inference_mode()
def test_indexer_rope_batch_changes(monkeypatch, heads, neox):
    from vllm_ascend.ops.triton.decode_warmup import warmup_indexer_rope
    from vllm_ascend.ops.triton.rope import _triton_rope_siso, rope_forward_triton_siso

    warmup_indexer_rope(heads, 128, 64, neox, torch.bfloat16)
    guard(monkeypatch, _triton_rope_siso)
    for rows in (1, 2, 3, 16, 17, 33):
        original = torch.randn((rows, heads, 128), dtype=torch.bfloat16)
        cos, sin = (torch.randn((rows, 64), dtype=torch.bfloat16) for _ in range(2))
        expected = original.clone()
        first = slice(0, 32) if neox else slice(0, 64, 2)
        second = slice(32, 64) if neox else slice(1, 64, 2)
        a, b = original[..., first].float(), original[..., second].float()
        c, s = cos[:, None, :32].float(), sin[:, None, :32].float()
        expected[..., first] = a * c - b * s
        expected[..., second] = b * c + a * s
        result = rope_forward_triton_siso(original.npu(), cos.npu(), sin.npu(), rope_dim=64, is_neox_style=neox)
        torch.testing.assert_close(result.cpu(), expected, rtol=0.02, atol=0.02)


@torch.inference_mode()
def test_penalties_and_sampling_no_compile_after_warmup(monkeypatch):
    from vllm_ascend.ops.triton import reject_sample as ops
    from vllm_ascend.ops.triton.bincount import token_bin_counts_and_mask_kernel
    from vllm_ascend.ops.triton.decode_warmup import warmup_sampling, warmup_spec_decode
    from vllm_ascend.ops.triton.penalty import apply_all_penalties_kernel, apply_penalties_triton
    from vllm_ascend.ops.triton.spec_decode.utils import prepare_inputs_padded_kernel, prepare_next_mtp_tokens_kernel
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

    vocab, max_requests = 257, get_vectorcore_num() * 2 + 1
    warmup_sampling(vocab, max_requests, 3)
    warmup_spec_decode(vocab, True)
    guard(
        monkeypatch,
        token_bin_counts_and_mask_kernel,
        apply_all_penalties_kernel,
        ops.rejection_greedy_sample_spec_len_1_triton,
        ops.rejection_greedy_sample_triton,
        ops.rejection_random_sample_kernel,
        ops.expand_kernel,
        ops.sample_recovered_tokens_kernel,
        prepare_inputs_padded_kernel,
        prepare_next_mtp_tokens_kernel,
    )
    for n in (1, 2, 3, 16, 17, max_requests):
        for history in (1, 17, 256):
            tokens_cpu = (torch.arange(n * history).reshape(n, history) % vocab).long()
            tokens = tokens_cpu.npu()
            logits = torch.ones((n, vocab), device="npu")
            penalty = torch.full((n,), 0.25, device="npu")
            apply_penalties_triton(logits, tokens, tokens, penalty, penalty, penalty * 5)
            bins = torch.zeros((n, vocab)).scatter_add_(1, tokens_cpu, torch.ones_like(tokens_cpu, dtype=torch.float32))
            expected = torch.where(bins > 0, 0.8, 1.0) - bins * 0.25 - (bins > 0) * 0.25
            torch.testing.assert_close(logits.cpu(), expected)
        draft = torch.zeros(n, dtype=torch.int32, device="npu")
        bonus = torch.full_like(draft, 7)
        cu = torch.arange(1, n + 1, dtype=torch.int32, device="npu")
        argmax = (cu % 2).long()
        output = torch.full((n, 2), -1, dtype=torch.int32, device="npu")
        grid, tile = ops.cal_grid_and_block_size(n)
        ops.rejection_greedy_sample_with_triton(output, [1] * n, cu, draft, argmax, bonus, None, 1, grid, tile)
        expected = torch.stack((argmax.cpu().int(), torch.where(argmax.cpu() == 0, 7, -1).int()), -1)
        assert torch.equal(output.cpu(), expected)
        for greedy in (None, torch.ones(n, dtype=torch.bool, device="npu")):
            output.fill_(-1)
            ops.rejection_greedy_sample_triton[(grid,)](output, cu, draft, argmax, bonus, greedy, n, 1, BLOCK_SIZE=tile)
            assert torch.equal(output.cpu(), expected)
        for value_dtype in (torch.int32, torch.float32):
            values = cu.to(value_dtype)
            expanded = torch.empty_like(values)
            ops.expand_triton(n, expanded, values, cu, 0, 1, 128)
            assert torch.equal(expanded.cpu(), values.cpu())
        probs = torch.zeros((n, vocab), device="npu")
        probs[:, 1] = 1
        recovered = torch.empty_like(draft)
        ops.sample_recovered_tokens_kernel[(n, 1)](
            recovered,
            cu,
            draft,
            None,
            probs,
            torch.ones_like(probs),
            vocab,
            512,
            NO_DRAFT_PROBS=True,
            SUB_BLOCK=4096,
            multibuffer=False,
        )
        assert recovered.cpu().eq(1).all()
        ops.rejection_random_sample_kernel[(grid,)](
            output,
            cu,
            draft,
            None,
            probs,
            bonus,
            recovered,
            torch.ones(n, device="npu"),
            torch.zeros(n, dtype=torch.bool, device="npu"),
            1,
            vocab,
            n,
            NO_DRAFT_PROBS=True,
            BLOCK_SIZE=tile,
        )
        assert output[:, 0].cpu().eq(1).all()
        counts = torch.empty(n, dtype=torch.int64, device="npu")
        next_ids, indices, rejected = [torch.empty_like(draft) for _ in range(3)]
        for width in (1, 2):
            sampled = bonus[:, None].repeat(1, width)
            for fixed in (False, True):
                prepare_next_mtp_tokens_kernel[(1,)](
                    sampled,
                    draft,
                    next_ids,
                    counts,
                    n,
                    vocab,
                    width,
                    1,
                    WIDTH=width,
                    BLOCK_SIZE=4,
                    sample_indices=indices if fixed else None,
                    rejected_counts=rejected if fixed else None,
                    FIXED_Q2=fixed,
                )
                assert next_ids.cpu().eq(7).all() and counts.cpu().eq(width).all()
        starts = torch.arange(n + 1, dtype=torch.int32, device="npu") * 2
        prepare_inputs_padded_kernel[(1,)](cu, counts, starts, indices, rejected, n, BLOCK_SIZE=4)
        assert indices.cpu().tolist() == list(range(1, n * 2, 2))
        assert rejected.cpu().eq(0).all()
