# SPDX-License-Identifier: Apache-2.0
"""NPU padding/capture qualification; this does not qualify EP serving."""

from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.fused_moe import prepare_finalize


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("tokens", [3, 4, 7, 8])
@torch.inference_mode()
def test_maskless_mc2_prepare_eager_and_graph(monkeypatch, rank, tokens):
    torch.npu.set_device(0)
    context = NS(padded_num_tokens=8, mc2_mask=None)
    monkeypatch.setattr(prepare_finalize, "_EXTRA_CTX", context)
    owner = NS(tp_size=4, tp_rank=rank)
    hidden = torch.arange(tokens * 16, device="npu", dtype=torch.float32).to(torch.bfloat16).reshape(tokens, 16)
    router = hidden[:, :4].contiguous()

    def prepare():
        return prepare_finalize.PrepareAndFinalizeWithMC2.prepare(owner, hidden, router)

    context.mc2_mask = torch.arange(8, device="npu") < tokens
    expected = prepare()
    context.mc2_mask = None
    for _ in range(2):
        eager = prepare()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = prepare()
        outputs = captured.hidden_states.clone(), captured.router_logits.clone()
    graph.replay()
    torch.npu.synchronize()
    for result in (eager, captured):
        assert result.mc2_mask is None
        assert result.padded_hidden_states_shape == expected.padded_hidden_states_shape
        torch.testing.assert_close(result.hidden_states, expected.hidden_states, rtol=0, atol=0)
        torch.testing.assert_close(result.router_logits, expected.router_logits, rtol=0, atol=0)
    hidden.add_(10)
    router.copy_(hidden[:, :4])
    context.mc2_mask = torch.arange(8, device="npu") < tokens
    expected = prepare()
    context.mc2_mask = None
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(outputs[0], expected.hidden_states, rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], expected.router_logits, rtol=0, atol=0)
