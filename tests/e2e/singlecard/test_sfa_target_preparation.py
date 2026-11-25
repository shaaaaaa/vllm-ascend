# SPDX-License-Identifier: Apache-2.0
"""Actual indexer contract and complete registered-source preparation chain."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_target_indexer_replay_is_causal(quantized):
    from vllm_ascend.attention.sfa_graph_layout import pack_decode_lanes
    from vllm_ascend.ops.triton.sfa_dcp import prepare_sfa_dcp_topk
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    assert enable_custom_op()
    heads, rows, block, blocks = 64, 6, 128, 130
    dtype = torch.int8 if quantized else torch.bfloat16
    q = torch.randint(-3, 4, (rows, heads, 128), device="npu").to(dtype)
    k = torch.randint(-3, 4, (blocks, block, 1, 128), device="npu").to(dtype)
    weights = torch.ones((rows, heads), dtype=torch.float16 if quantized else dtype, device="npu")
    qs = torch.ones((rows, heads), dtype=torch.float16, device="npu")
    ks = torch.ones((blocks, block, 1), dtype=torch.float16, device="npu")
    ends = torch.tensor([2, 4, 6], dtype=torch.int32, device="npu")
    lengths = torch.tensor([16384, 129, 0], dtype=torch.int32, device="npu")
    table = torch.arange(blocks, dtype=torch.int32, device="npu").repeat(3, 1)
    positions = torch.zeros(rows, dtype=torch.int64, device="npu")
    owners = torch.full((rows,), -1, dtype=torch.int32, device="npu")
    output = torch.empty((rows, 1, 2048), dtype=torch.int32, device="npu")
    counts = torch.full((rows, 16), -77, dtype=torch.int32, device="npu")

    def run():
        kwargs = dict(
            actual_seq_lengths_query=ends,
            actual_seq_lengths_key=lengths,
            block_table=table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
        if quantized:
            raw = torch.ops._C_ascend.npu_lightning_indexer_quant(
                q,
                k,
                weights,
                qs,
                ks,
                query_quant_mode=0,
                key_quant_mode=0,
                **kwargs,
            )
        else:
            raw = torch.ops._C_ascend.npu_lightning_indexer(q, k, weights, **kwargs)
        prepare_sfa_dcp_topk(raw, positions, output, counts, rows, 8, 0, 128, local=False)
        return raw

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            raw = run()
        for widths, seq in (
            ((2, 2), (16384, 129)),
            ((1, 2), (2047, 2049)),
            ((2, 1), (128, 1)),
            ((1, 1), (2, 16385)),
            ((0, 0), (0, 0)),
        ):
            real = sum(widths)
            ends.copy_(torch.tensor([widths[0], real, rows], dtype=torch.int32))
            lengths.copy_(torch.tensor([*seq, 0], dtype=torch.int32))
            p = [length - width + j for length, width in zip(seq, widths) for j in range(width)]
            positions.copy_(torch.tensor(p + [0] * (rows - real), dtype=torch.int64))
            ids = [i for i, width in enumerate(widths) for _ in range(width)] + [-1] * (rows - real)
            owners.copy_(torch.tensor(ids, dtype=torch.int32))
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(raw[:real].cpu(), output[:real].cpu())
            for a, b in zip(
                pack_decode_lanes(raw, owners, owners, ends[:2], 2),
                pack_decode_lanes(output, owners, owners, ends[:2], 2),
            ):
                assert torch.equal(a.cpu(), b.cpu())
            assert counts.cpu().eq(-77).all()
    finally:
        torch.npu.synchronize()
        graph.reset()


@pytest.mark.parametrize("layout,padding", [("uniform", 0), ("mixed", 1)])
@torch.inference_mode()
def test_full_chain_with_real_cpu_sources(layout, padding, tmp_path):
    path = ROOT / "benchmarks/ops/benchmark_sfa_target_preparation.py"
    spec = importlib.util.spec_from_file_location("target_preparation_npu", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    assert enable_custom_op()
    companion = Path(os.environ.get("LMCACHE_ASCEND_SOURCE_DIR", ROOT.parent / "LMCache-Ascend"))
    assert (companion / "benchmark/v1/kv_transfer/load_benchmark_utils.py").exists(), "set LMCACHE_ASCEND_SOURCE_DIR"
    result = benchmark.run_case(
        SimpleNamespace(
            requests=2,
            steps=3,
            warmup_steps=1,
            history=4096,
            live_tokens=128,
            overlap=0.9,
            layout=layout,
            padding=padding,
            seed=3,
            device=0,
            heads=8,
            resident=False,
            shared_plan=True,
            dcp_size=8,
            rank=0,
            lmcache_ascend_dir=companion,
            trials=2,
            iterations=1,
            profile_dir=None,
        )
    )
    assert result["parity"] == "passed" and result["copied_bytes"] > 0


@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_owner_scatter_masks_foreign_padding_and_rejected_draft_rows(quantized):
    from vllm_ascend.ops.triton.sfa_dcp import store_owner_keys

    dtype = torch.int8 if quantized else torch.bfloat16
    keys = torch.arange(4).to("npu", dtype)[:, None].expand(-1, 128).contiguous()
    scales = torch.arange(4, device="npu", dtype=torch.float16) if quantized else None
    cache = torch.full((8, 128), 71, device="npu", dtype=dtype)
    scale_cache = torch.full((8,), -3, device="npu", dtype=torch.float16)
    take = torch.tensor([3, 0, 1, 2], device="npu")
    valid = torch.tensor([True, True, True, False], device="npu")
    slots = torch.tensor([-1, 4, 7, 2], device="npu")
    planes = (None, None, cache, scale_cache) if quantized else (None, None, cache)
    store_owner_keys(take, valid, slots, keys, scales, planes)
    assert cache[:, 0].cpu().tolist() == [71, 71, 0, 71, 2, 71, 71, 71]
    if quantized:
        assert scale_cache.cpu().tolist() == [-3, -3, 0, -3, 2, -3, -3, -3]
