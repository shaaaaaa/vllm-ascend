# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small real-NPU probe replay test; full model parity lives under multicard/.

Run with --confcutdir=tests/e2e/singlecard to avoid unrelated model dependencies.
"""

import pytest


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int8", "packed_int32"])
@pytest.mark.parametrize("shape", [(64, 128), (3, 64, 128)])
def test_fingerprint_reads_real_nz_weights_without_format_conversion(dtype, shape):
    torch_npu = pytest.importorskip("torch_npu")
    import torch

    from vllm_ascend.attention.sfa_parity import weight_fingerprint

    if not torch.npu.is_available():
        pytest.skip("Requires an Ascend NPU")
    torch.npu.set_device(0)
    source_dtype = torch.int8 if dtype == "packed_int32" else getattr(torch, dtype)
    source = (torch.arange(torch.tensor(shape).prod().item()) % 16).to(source_dtype).reshape(shape)

    def make_model(cpu_weight):
        weight = torch_npu.npu_format_cast(cpu_weight.npu(), 29)
        if dtype == "packed_int32":
            # Match ModelSlim's new W4A8 packing after NZ conversion.
            weight = weight.view(torch.int32).contiguous()
        model = torch.nn.Module()
        model.register_buffer("weight", weight)
        return model

    a, b = make_model(source), make_model(source)
    pointer, layout = a.weight.data_ptr(), torch_npu.get_npu_format(a.weight)
    torch.npu.synchronize()
    fingerprint = weight_fingerprint(a)
    assert fingerprint == weight_fingerprint(b)
    source.reshape(-1)[-1] += 1
    changed = make_model(source)
    torch.npu.synchronize()
    assert fingerprint != weight_fingerprint(changed)
    assert a.weight.data_ptr() == pointer
    assert torch_npu.get_npu_format(a.weight) == layout == 29
    assert weight_fingerprint(a) == fingerprint


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
