# SPDX-License-Identifier: Apache-2.0
"""Native contract used by the staged SFA graph-to-connector handoff."""

import pytest
import torch

pytest.importorskip("torch_npu")


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")
def test_graph_replay_handoff_multiple_consumers():
    iterations, width = 32, 16
    compute = torch.npu.current_stream()
    loader, diagnostic = torch.npu.Stream(), torch.npu.Stream()
    inputs = torch.zeros(width, device="npu", dtype=torch.int32)
    payload = torch.empty_like(inputs)
    loaded = torch.empty((iterations, width), device="npu", dtype=inputs.dtype)
    observed = torch.empty_like(loaded)
    event = torch.npu.Event()
    event.record(compute)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    capture = torch.npu.Stream()
    capture.wait_stream(compute)
    with torch.npu.graph(graph, stream=capture):
        payload.copy_(inputs)
    compute.wait_stream(capture)
    for i in range(iterations):
        inputs.fill_(i)
        graph.replay()
        event.record(compute)
        with torch.npu.stream(loader):
            loader.wait_event(event)
            loaded[i].copy_(payload)
        with torch.npu.stream(diagnostic):
            diagnostic.wait_event(event)
            observed[i].copy_(payload)
        # Protect the next producer overwrite without a host synchronization.
        compute.wait_stream(loader)
        compute.wait_stream(diagnostic)
    torch.npu.synchronize()
    expected = torch.arange(iterations, dtype=inputs.dtype)[:, None].expand(iterations, width)
    assert torch.equal(loaded.cpu(), expected)
    assert torch.equal(observed.cpu(), expected)
