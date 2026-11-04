# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual Triton/NPU execution, including graph replay; no model is loaded.

Standalone invocation (avoids the unrelated model fixtures in e2e/conftest.py):
  pytest --confcutdir=tests/e2e/nightly/single_node/ops/singlecard_ops/triton \
    tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_async_mtp_metadata.py -xv

Missing Torch-NPU, Triton or NPU hardware causes skips, never a mocked pass.
"""

import gc
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def metadata_kernel():
    pytest.importorskip("torch_npu")
    pytest.importorskip("triton")
    if not torch.npu.is_available():
        pytest.skip("Requires an NPU and the native Triton-Ascend compiler")
    # Load the repository source unchanged without importing unrelated custom
    # ops through ops/__init__.py. No substitutions or fallback kernel.
    path = Path(__file__).resolve().parents[7] / "vllm_ascend/ops/triton/spec_decode/async_mtp.py"
    spec = importlib.util.spec_from_file_location("native_async_mtp_metadata", path)
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module.prepare_async_mtp_kernel


@pytest.fixture(autouse=True)
def gc_disabled():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize(
    "n,pad,mode,block1",
    (pytest.param(14, 0, "plain", 128, id="startup-14-requests"),)
    + tuple(
        pytest.param(n, pad, mode, 128, id=f"n{n}-pad{pad}-{mode}")
        for n in (1, 2, 4, 6, 8, 10, 12, 14, 31, 32, 33)
        for pad in (0, 2)
        for mode in ("plain", "paired", "table")
        if (n, pad, mode) != (14, 0, "plain")
    )
    + (pytest.param(14, 2, "plain", 64, id="unequal-group-block-sizes"),),
)
def test_native_metadata_eager_and_graph(metadata_kernel, n, pad, mode, block1):
    device = "npu"
    cap = 2 * (n + pad)
    guard = 8 if pad else 0
    start = torch.arange(n, dtype=torch.int32) % 3 + (126 if pad else 0)
    if pad and n > 1:
        start[-1] = 1048512 if block1 == 128 else 524224
    bases = start.to(device)
    counts = torch.ones(n, dtype=torch.int64, device=device)
    host_tables = [
        torch.arange(n * stride, dtype=torch.int32).view(n, stride) + bias
        for stride, bias in ((8192, 11), (8208, 101), (8224, 1001))
    ]
    tables = [table.to(device) for table in host_tables]
    positions = torch.full((cap + guard,), -77, dtype=torch.int64, device=device)
    common = torch.full((n + guard,), -77, dtype=torch.int32, device=device)
    target = torch.full((n + pad + 1 + guard,), -77, dtype=torch.int32, device=device)
    slots = [
        torch.full((cap + guard,), -77, dtype=torch.int32, device=device) for _ in range(2 if mode == "plain" else 3)
    ]
    extra = {} if mode == "plain" else dict(slots_c8=slots[2], MIXED_C8=True)
    if mode == "table":
        extra.update(table_c8=tables[2], stride_c8=8224)

    def launch():
        metadata_kernel[((n + pad + 1 + 31) // 32,)](
            bases,
            counts,
            positions,
            common,
            target,
            *tables[:2],
            *slots[:2],
            n,
            cap,
            n,
            n + pad + 1,
            8192,
            8208,
            128,
            block1,
            BLOCK=32,
            **extra,
        )

    def check(actual, values, capacity, padding):
        expected = torch.full(actual.shape, -77, dtype=actual.dtype)
        expected[:capacity] = padding
        expected[: len(values)] = values.to(actual.dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def exercise(call):
        bases.copy_(start)
        expected = start.clone()
        for step in range(6):
            accepted = torch.ones(n, dtype=torch.int64) if step == 0 else 1 + (torch.arange(n) + step) % 2
            counts.copy_(accepted)
            torch.npu.synchronize()
            call()
            torch.npu.synchronize()
            expected += accepted.int()
            torch.testing.assert_close(bases.cpu(), expected, rtol=0, atol=0)
            expected_positions = (expected[:, None].long() + torch.arange(2)).flatten()
            check(positions, expected_positions, cap, 0)
            check(common, expected + 2, n, 0)
            check(target, expected + 2, n + pad + 1, 0)
            owners = torch.arange(n).repeat_interleave(2)
            for group, slot in enumerate(slots):
                block = 128 if group == 0 else block1
                table = host_tables[1] if group == 2 and mode == "paired" else host_tables[group]
                physical = table[owners, expected_positions // block]
                if group == 2 and mode == "paired":
                    physical = physical * 2
                check(slot, physical * block + expected_positions % block, cap, -1)

    exercise(launch)  # Compile and warm outside capture.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    torch.npu.synchronize()
    exercise(graph.replay)
    del graph  # All work completed; source and destination tensors are still held.
