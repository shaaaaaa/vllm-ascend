# SPDX-License-Identifier: Apache-2.0
"""Real TorchAir/root-capture probe with synthetic SFA-schema operators.

Run in an NPU container with matching vLLM-Ascend installed:
python -m pytest --noconftest -q tests/e2e/singlecard/compile/test_sfa_npugraph_ex_npu.py
This checks the compiler contract and the actual RetrievalEdge, not GLM accuracy
or CPU-to-NPU sparse DMA. Distributed server qualification is still required.
"""

from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("torchair")

from torch._inductor import config as inductor_config
from vllm.config import CUDAGraphMode

from vllm_ascend.compilation.compiler_interface import COMPILATION_PASS_KEY, npugraph_ex_compile
from vllm_ascend.compilation.sfa_retrieval_overlap import RetrievalEdge


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("alias_output", [False, True])
@torch.inference_mode()
def test_real_backend_preserves_resident_storage_and_captured_overlap(monkeypatch, v2, alias_output):
    for name, value in {
        "VLLM_ASCEND_SFA_STAGED_GRAPH": "1",
        "VLLM_ASCEND_SFA_FULL_GRAPH": "1",
        "VLLM_ASCEND_DSA_UNBUNDLE": "1",
        "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
        "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
        "VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD": "0",
        "VLLM_ASCEND_DSA_USE_ADAPTER_CACHE": "0",
    }.items():
        monkeypatch.setenv(name, value)
    torch.npu.set_device(0)
    cfg = NS(
        model_config=NS(use_mla=True, hf_text_config=NS(index_topk=2048), enforce_eager=False),
        compilation_config=NS(cudagraph_mode=CUDAGraphMode.PIECEWISE),
        parallel_config=NS(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=object(),
        lora_config=None,
    )
    backing = torch.zeros(12, device="npu")
    resident = backing[4::2]  # Strided, with unused storage to check for overwrite.
    output = backing[:4] if alias_output else torch.zeros(4, device="npu")
    x, destination = torch.ones(4, device="npu"), torch.zeros(4, device="npu")
    edge = RetrievalEdge(
        torch.npu.Stream(), NS(load=lambda selected, counts, slots: destination.copy_(selected)), (destination,)
    )
    active = NS(value=False)
    seen = []
    lib = torch.library.Library(f"npu_sfa_ex_probe_{int(v2)}_{int(alias_output)}", "DEF")
    lib.define("pre(Tensor x, Tensor(a!) output, Tensor(b!)[] resident_writes) -> Tensor")
    lib.define("post(Tensor resident, Tensor(a!) output) -> ()")

    def pre(x, output, resident_writes):
        seen.append(resident_writes[0].data_ptr())
        output.copy_(x * 2)
        resident_writes[0].add_(x)
        return x.clone()

    def post(resident, output):
        if active.value:
            edge.launch(resident, None, None, (destination,))
        output.copy_(resident)

    lib.impl("pre", pre, "PrivateUse1")
    lib._register_fake("pre", lambda x, output, resident_writes: torch.empty_like(x))
    lib.impl("post", post, "PrivateUse1")
    lib._register_fake("post", lambda resident, output: None)
    ops = getattr(torch.ops, f"npu_sfa_ex_probe_{int(v2)}_{int(alias_output)}")
    # Test-local alias; the installed native operator registry is not changed.
    monkeypatch.setattr(torch.ops.vllm, "sfa_forward_pre_shared", ops.pre, raising=False)
    compilations = []

    def backend(gm, inputs):
        compilations.append(1)
        return npugraph_ex_compile(
            gm, inputs, {COMPILATION_PASS_KEY: lambda g: g}, cfg, NS(enable_static_kernel=False), None
        )[0]

    def producer(x, output, resident):
        return ops.pre(x, output, [resident]) + output

    def launch(resident, output):
        ops.post(resident, output)
        return output

    try:
        with inductor_config.patch(enable_auto_functionalized_v2=v2):
            write = torch.compile(producer, backend=backend, fullgraph=True)
            load = torch.compile(launch, backend=backend, fullgraph=True)
            write(x, output, resident)
            load(resident, output)
            torch.npu.synchronize()
            assert edge.payload is None  # Initial profiling has no active edge.
            active.value = True  # Late binding before root capture.
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = write(x, output, resident)
                load(resident, output)
                edge.join()
                observed = destination.clone()
            torch.npu.synchronize()
            count = len(compilations)
            assert count == 2
            resident.zero_()
            for i in (1, 2, 3):
                x.fill_(i)
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(result.cpu(), torch.full((4,), float(3 * i)))
                torch.testing.assert_close(observed.cpu(), torch.full((4,), float(i * (i + 1) // 2)))
                torch.testing.assert_close(backing[5::2].cpu(), torch.zeros(4))
            assert len(compilations) == count
            if not alias_output:
                assert seen and set(seen) == {resident.data_ptr()}
    finally:
        torch.npu.synchronize()
        torch._dynamo.reset()
