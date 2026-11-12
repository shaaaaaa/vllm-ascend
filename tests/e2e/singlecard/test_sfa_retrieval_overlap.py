# SPDX-License-Identifier: Apache-2.0
"""Native fork/join and real registered-CPU transfer qualification."""
import gc
import os
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def test_native_capture_contract():
    from tests.sfa_capture_probe import CaptureProbe
    torch.npu.set_device(0)
    stream = torch.npu.Stream()
    probes = [CaptureProbe(stream, torch.device("npu:0")) for _ in range(2)]
    try:
        for probe in probes:
            probe.verify()
        for iteration in range(32):
            probe = probes[iteration % 2]
            probe.source.fill_(iteration)
            probe.graph.replay()
            done = torch.npu.Event()
            done.record()
            done.synchronize()  # Main completion must cover side-stream work.
            assert torch.equal(probe.observed.cpu(), torch.full((64,), float(iteration)))
    finally:
        torch.npu.synchronize()


@pytest.mark.parametrize("requests", [1, 8, 16])
@pytest.mark.parametrize("query_rows", [1, 2])
def test_registered_sources_consumer_boundary_and_replay(requests, query_rows):
    root = Path(os.environ.get("LMCACHE_ASCEND_SOURCE_DIR", str(ROOT.parent / "LMCache-Ascend")))
    if not (root / "benchmark/v1/kv_transfer/load_benchmark_utils.py").exists():
        pytest.fail("set LMCACHE_ASCEND_SOURCE_DIR to the matching LMCache-Ascend source checkout")
    sys.path.insert(0, str(ROOT / "tools"))
    from sfa_retrieval_overlap_benchmark import Workload
    workload = Workload(root, requests, query_rows)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        graphs = [workload.capture(mode, observe=True) for mode in (False, True)]
        for active, shift in ((0, 0), (1, 4096), (410, 3), (4096, 17), (37, 299), (0, 0)):
            for graph in graphs:
                tokens = workload.set_payload(active, shift, idle=requests > 1)
                graph.replay()
                workload.check(tokens, active, idle=requests > 1)
        # A different immutable lane-to-source binding models request replacement.
        workload.sources.reverse()
        for layer, transfer in enumerate(workload.transfers):
            transfer.bind_batch(workload.sources, layer)
        tokens = workload.set_payload(37, 21)
        for graph in graphs:
            graph.replay()
            for layer in range(workload.layers):
                for plane in range(2):
                    observed = workload.observed[layer][plane].cpu()
                    for lane in range(requests):
                        expected = (workload.values(tokens[:37], requests - lane - 1, layer) + 32 * plane).bfloat16()
                        torch.testing.assert_close(observed[lane, :37], expected, rtol=0, atol=0)
    finally:
        workload.close()
        if was_enabled:
            gc.enable()


@torch.inference_mode()
def test_counted_sfa_replay_ignores_stale_pair_tail():
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.npu.set_device(0)
    query = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="npu")
    rope = query.new_zeros((1, 64, 64))
    key = query.new_zeros((1, 128, 1, 512))
    key[0, 3].fill_(7)
    key_rope = query.new_zeros((1, 128, 1, 64))
    indices = torch.zeros((1, 1, 2048), dtype=torch.int32, device="npu")
    indices[0, 0, 0] = 3
    counts = torch.zeros((1, 16), dtype=torch.int32, device="npu")
    counts[0, 0] = 2
    table = torch.zeros((1, 1), dtype=torch.int32, device="npu")
    q_lens = torch.ones(1, dtype=torch.int32, device="npu")
    kv_lens = torch.full_like(q_lens, 128)

    def attend():
        return torch.ops._C_ascend.npu_sparse_flash_attention_dcp(
            query, rope, key, key_rope, indices, counts, table, q_lens, kv_lens, 1.0
        )[0]

    attend()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            output = attend()
        graph.replay()
        torch.testing.assert_close(output.cpu(), torch.full_like(output, 3.5, device="cpu"), rtol=0, atol=0)
        # The second entry is still resident but is no longer selected.
        counts[0, 0] = 1
        graph.replay()
        torch.testing.assert_close(output.cpu(), torch.full_like(output, 7, device="cpu"), rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        graph.reset()


@torch.inference_mode()
@pytest.mark.parametrize("mixed", [False, True])
def test_native_dcp_topk_filters_causal_rows_on_npu(mixed):
    from vllm_ascend.ops.triton.sfa_dcp import prepare_sfa_dcp_topk

    torch.npu.set_device(0)
    topk = torch.full((3, 1, 2048), -1, dtype=torch.int32, device="npu")
    topk[:, 0, :5] = torch.tensor([127, 128, 129, 256, 640], dtype=torch.int32, device="npu")
    positions = torch.tensor([128, 640, 700], dtype=torch.int64, device="npu")
    output = torch.empty_like(topk)
    counts = torch.empty((3, 16), dtype=torch.int32, device="npu")
    rows = torch.tensor([0, -1, -1], dtype=torch.int32, device="npu") if mixed else None
    prepare_sfa_dcp_topk(topk, positions, output, counts, 2, 4, 1, 128, local=not mixed, row_requests=rows)
    expected_first = [127, 128, -1, -1] if mixed else [0, -1, -1, -1]
    assert output[:, 0, :4].cpu().tolist() == [expected_first, [0, 1, 128, -1], [-1, -1, -1, -1]]
    assert counts[(1 if mixed else 0):, 0].cpu().tolist() == ([3, 0] if mixed else [1, 3, 0])
    assert output[:, :, 4:].cpu().eq(-1).all()


@torch.inference_mode()
def test_dcp_draft_tail_mask_handles_ragged_request_boundaries_on_npu():
    from vllm_ascend.ops.triton.spec_decode.utils import prepare_dcp_mtp_draft_kernel

    torch.npu.set_device(0)
    starts = torch.tensor([0, 1, 3, 5, 8], dtype=torch.int32, device="npu")
    ends = torch.tensor([128, 130, 513, 1048576], dtype=torch.int32, device="npu")
    rejected = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device="npu")
    origins = torch.tensor([0, 4096, 4096, 0], dtype=torch.int32, device="npu")
    positions = torch.tensor([127, 128, 129, 511, 512, 1048573, 1048574, 1048575], device="npu")
    slots0 = torch.tensor([-1, 1280, 1281, -1, -1, -1, -1, -1], device="npu")
    slots1 = torch.arange(8, device="npu") + 4096
    outputs = [torch.empty(8, dtype=torch.int32, device="npu") for _ in range(3)]
    lengths = torch.empty_like(ends)
    prepare_dcp_mtp_draft_kernel[(2,)](
        starts, ends, rejected, origins, positions, slots0, slots1, *outputs, lengths, 4,
        WORLD=4, RANK=1, BLOCK_SIZE=128, TILE=64,
    )
    for src, dst in zip((positions, slots0, slots1), outputs):
        expected = src.cpu().int()
        expected[2] = -1
        torch.testing.assert_close(dst.cpu(), expected)
    assert lengths.cpu().tolist() == [0, 4097, 4224, 262144]


@pytest.mark.parametrize("world,rank", [(4, 0), (4, 3), (8, 7)])
def test_dcp_planner_replays_owned_misses_into_counted_attention(world, rank):
    root = Path(os.environ.get("LMCACHE_ASCEND_SOURCE_DIR", str(ROOT.parent / "LMCache-Ascend")))
    sys.path.insert(0, str(ROOT / "tools"))
    from sfa_retrieval_overlap_benchmark import Workload

    from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sorted_cache import (
        ResidentDcpLayout,
        allocate_sorted_resident_state,
        allocate_sorted_resident_workspace,
        prepare_resident_sharded_union_,
        prepare_sorted_resident_cache_fused_,
    )

    workload = Workload(root, requests=2, query_rows=2, layers=1)
    try:
        device = workload.device
        k, pe = workload.caches[0]
        origin = 4096
        live = (workload.history // (128 * world) + 1) * 128 * world + rank * 128
        local_live = origin + live // (128 * world) * 128
        long_shard = [t for t in range(workload.history) if t // 128 % world == rank][:600]
        ids = [
            sorted({*long_shard, 0, 127, 128, 256, 511, 512, 640, 1023, 1024, 2048, 4096, live}),
            [0, 128, 639, 640, 4095, 4096],
        ]
        raw = torch.full((4, 1, 2048), -1, dtype=torch.int32)
        for i, row in enumerate(ids):
            raw[i, 0, : len(row)] = torch.tensor(row, dtype=torch.int32)
        raw = raw.to(device)
        topk = torch.empty_like(raw)
        state = allocate_sorted_resident_state(2, 2, 2, device=device)
        workspace = allocate_sorted_resident_workspace(2, 2, device=device)
        counts = torch.empty((4, 16), dtype=torch.int32, device=device)
        layout = ResidentDcpLayout(
            world, rank, torch.tensor([origin, 0], dtype=torch.int32, device=device), counts
        )
        boundaries = torch.tensor([workload.history, workload.history, 0, 0], dtype=torch.int32, device=device)
        rows = torch.tensor([0, 0, -1, -1], dtype=torch.int32, device=device)
        indices = torch.tensor([0, -1], dtype=torch.int32, device=device)
        generations = torch.ones(2, dtype=torch.int64, device=device)
        table = torch.arange(1, 64, dtype=torch.int32, device=device).repeat(2, 1)
        k.view(-1, 512)[local_live + 128].fill_(3)
        pe.view(-1, 64)[local_live + 128].fill_(35)
        query = torch.zeros((4, 64, 512), dtype=torch.bfloat16, device=device)
        rope = torch.zeros((4, 64, 64), dtype=torch.bfloat16, device=device)
        query[..., 0] = 2
        rope[..., 0] = 1
        q_lens = torch.tensor([2, 2], dtype=torch.int32, device=device)
        kv_lens = torch.tensor([local_live + 1, 0], dtype=torch.int32, device=device)

        def step():
            topk.copy_(raw)
            prepare_resident_sharded_union_(
                topk, boundaries, rows, indices, generations, state, workspace, mtp=2, dcp=layout
            )
            misses, miss_counts, slots = prepare_sorted_resident_cache_fused_(
                topk, table, indices, generations, state, workspace, block_size=128, dcp=layout
            )
            workload.transfers[0].load(misses, miss_counts, slots)
            return torch.ops._C_ascend.npu_sparse_flash_attention_dcp(
                query, rope, k, pe, topk, counts, table, q_lens, kv_lens, 576**-0.5
            )

        step()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        workload.graphs.append(graph)
        with torch.npu.graph(graph):
            output, softmax_max, softmax_sum = step()
        owned = [[t for t in row if t // 128 % world == rank] for row in ids]
        needed = {t for row in owned for t in row if t < workload.history}
        generations.add_(1)
        graph.replay()
        miss_counts = workspace.miss_counts[:, 0].cpu().tolist()
        assert miss_counts == [len(needed), 0]
        tokens = workspace.miss_tokens[0, : len(needed)].cpu()
        slots = workspace.target_slots[0, : len(needed)]
        assert set(tokens.tolist()) == needed
        for plane, add in ((k, 0), (pe, 32)):
            copied = plane.view(-1, plane.size(-1)).index_select(0, slots).cpu()
            expected = (workload.values(tokens, 0, 0) + add).to(plane.dtype)
            torch.testing.assert_close(copied, expected[:, None].expand_as(copied), rtol=0, atol=0)

        graph.replay()
        assert workspace.miss_counts[:, 0].cpu().tolist() == [0, 0]
        expected_counts = torch.tensor([len(row) for row in owned] + [0, 0])
        assert counts[:, 0].cpu().tolist() == expected_counts.tolist()
        reference = torch.zeros((4, 3))
        for i, row in enumerate(owned):
            if row:
                values = workload.values(torch.tensor(row), 0, 0)
                values[torch.tensor(row) == live] = 3
                logits = (3 * values + 32) * 576**-0.5
                maximum = logits.max()
                weights = (logits - maximum).exp()
                reference[i] = torch.stack((maximum, weights.sum(), weights @ values / weights.sum()))
        torch.testing.assert_close(softmax_max.cpu(), reference[:, 0, None].expand(4, 64), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(softmax_sum.cpu(), reference[:, 1, None].expand(4, 64), rtol=1e-2, atol=1e-2)
        expected = reference[:, 2, None, None].to(k.dtype).expand_as(output)
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)

        raw[1].fill_(-1)
        graph.replay()
        assert counts[1, 0].cpu().item() == 0
        assert torch.count_nonzero(output[1].cpu()).item() == 0
        assert torch.count_nonzero(softmax_sum[1].cpu()).item() == 0
    finally:
        workload.close()
