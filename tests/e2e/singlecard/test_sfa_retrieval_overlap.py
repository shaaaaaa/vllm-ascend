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
    from vllm_ascend.compilation.sfa_retrieval_overlap import CaptureProbe
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
