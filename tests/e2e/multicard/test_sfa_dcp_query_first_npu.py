# SPDX-License-Identifier: Apache-2.0
"""Real current-KV, sparse H2D and DCP replay parity for overlap schedules."""

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")


@pytest.fixture(scope="module")
def query_first_world():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    root = Path(os.environ.get("LMCACHE_ASCEND_DIR", "../LMCache-Ascend")).resolve()
    if world not in (2, 4, 8):
        pytest.skip("launch with torchrun and 2, 4 or 8 NPUs")
    if not (root / "benchmark/v1/kv_transfer/load_benchmark_utils.py").is_file():
        pytest.skip("set LMCACHE_ASCEND_DIR to the companion checkout")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    try:
        yield world, control, root
    finally:
        dist.destroy_process_group()


@torch.inference_mode()
@pytest.mark.parametrize("overlap_retrieval", [False, True])
@pytest.mark.parametrize("layout,padding", [("uniform", 0), ("mixed", 1)])
@pytest.mark.parametrize(
    "comparison,query_first",
    [
        ("query-first", False),
        ("prefetch-after-pack", False),
        ("prefetch-after-pack", True),
        ("early-unpack", True),
    ],
)
def test_query_first_group_replay(
    query_first_world, monkeypatch, overlap_retrieval, layout, padding, comparison, query_first
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmarks/ops"))
    from benchmark_sfa_copy_reduction import measure_case
    from benchmark_sfa_dcp_early_unpack import UnpackPipeline
    from benchmark_sfa_dcp_prefetch import prefetch_pipeline
    from benchmark_sfa_dcp_query_first import QueryFirstPipeline

    world, control, root = query_first_world
    args = SimpleNamespace(
        lmcache_ascend_dir=root,
        dcp_size=world,
        rank=dist.get_rank(),
        device=int(os.environ["LOCAL_RANK"]),
        requests=2,
        heads=64 // world,
        layers=4,
        steps=3,
        warmup_steps=1,
        trials=1,
        iterations=1,
        history=4096,
        live_tokens=127,
        overlap=0.9,
        layout=layout,
        padding=padding,
        seed=19,
        resident=False,
        shared_plan=True,
        overlap_retrieval=overlap_retrieval,
        profile_dir=None,
        hidden_size=256,
        q_rank=128,
        qk_nope=128,
        value_dim=128,
        query_first=query_first,
        unpack_placement="both",
        unpack_cores=6,
        indexer="c8",
        prefetch_after_pack=True,
    )
    # measure_case checks eager and three reset/replay cycles, exact planner,
    # retrieved/current KV, attention and local output-projection parity.
    factory = {
        "query-first": QueryFirstPipeline,
        "prefetch-after-pack": prefetch_pipeline,
        "early-unpack": UnpackPipeline,
    }[comparison]
    result = measure_case(args, control, pipeline_factory=factory)
    assert result["parity"] == "passed"
    assert all(value > 0 for value in result["slowest_rank_median"].values())
