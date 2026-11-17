# SPDX-License-Identifier: Apache-2.0
"""Run the complete attention benchmark's eager/replay checks on real NPUs."""

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")


@pytest.fixture(scope="module")
def exchange_world():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (2, 4, 8):
        pytest.skip("launch with torchrun and 2, 4 or 8 local NPU ranks")
    device = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(device)
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    try:
        yield world, device, control
    finally:
        dist.destroy_process_group()


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("layout", ["balanced", "skewed", "empty"])
def test_direct_exchange_native_attention_capture(exchange_world, monkeypatch, dtype, layout):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmarks/ops"))
    from benchmark_sfa_dcp_exchange import measure_case

    world, device, control = exchange_world
    args = SimpleNamespace(
        world=world,
        device=device,
        rank=dist.get_rank(),
        requests=2,
        query_rows=2,
        padding=1,
        heads=64,
        local_kv_tokens=2048,
        dtype=dtype,
        layout=layout,
        seed=913,
        layers=2,
        steps=2,
        check_replays=3,
        warmup=1,
        trials=1,
        iterations=1,
        profile_dir=None,
    )
    result = measure_case(args, control)
    assert result["parity"] == "passed"
    assert result["output_collectives_per_layer"] == {"packed": 1, "direct": 15}
