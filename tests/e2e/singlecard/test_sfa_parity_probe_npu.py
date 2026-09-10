# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small real-NPU probe replay test; full model parity lives under multicard/.

Run with --confcutdir=tests/e2e/singlecard to avoid unrelated model dependencies.
"""

import pytest


def test_device_snapshots_are_refreshed_by_real_replay():
    pytest.importorskip("torch_npu")
    import torch

    from vllm_ascend.attention.sfa_parity import DeviceSnapshot

    if not torch.npu.is_available():
        pytest.skip("Requires an Ascend NPU")
    torch.npu.set_device(0)
    probe = DeviceSnapshot((2, 4), dtype=torch.float32, device=torch.device("npu:0"))
    source = torch.ones((2, 4), device="npu:0")
    probe.write(source * 2)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        probe.write(source * 2)
    for value in (3, 7, 11):
        probe.reset()
        source.fill_(value)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(probe.read(2, label="real replay"), torch.full((2, 4), value * 2.0))
    del graph, probe, source
    torch.npu.empty_cache()
