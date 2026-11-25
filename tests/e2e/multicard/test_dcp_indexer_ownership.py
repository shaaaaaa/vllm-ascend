# SPDX-License-Identifier: Apache-2.0
"""Real C8 producer, shared consumers, sparse H2D and HCCL replay parity."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")
from test_sfa_dcp_query_first_npu import query_first_world  # noqa: F401


@pytest.mark.parametrize("placement,layout,padding", [("balanced", "uniform", 0), ("skewed", "mixed", 1)])
@pytest.mark.parametrize("baseline", ["replicated", "restore"])
@torch.inference_mode()
def test_owner_cache_group_replay(query_first_world, monkeypatch, placement, layout, padding, baseline):  # noqa: F811
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmarks/ops"))
    from benchmark_sfa_copy_reduction import measure_case
    from benchmark_sfa_dcp_indexer_sharding import IndexerPipeline

    world, control, root = query_first_world
    args = SimpleNamespace(
        lmcache_ascend_dir=root,
        dcp_size=world,
        rank=dist.get_rank(),
        device=int(os.environ["LOCAL_RANK"]),
        requests=4,
        heads=64 // world,
        layers=4,
        steps=3,
        warmup_steps=1,
        trials=1,
        iterations=1,
        history=8192,
        live_tokens=127,
        overlap=0.9,
        layout=layout,
        padding=padding,
        seed=31,
        resident=False,
        shared_plan=True,
        overlap_retrieval=True,
        profile_dir=None,
        hidden_size=256,
        q_rank=128,
        qk_nope=128,
        value_dim=128,
        owner_layout=placement,
        owner_baseline=baseline,
        skew_history=True,
        replicate_owner_storage=False,
    )
    result = measure_case(args, control, pipeline_factory=IndexerPipeline)
    assert result["parity"] == "passed"
