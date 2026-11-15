# SPDX-License-Identifier: Apache-2.0
"""Pure bounds/protocol checks for the selected-prefill performance harness."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prefill_selected_workspace import (
    MAX_HISTORY_TOKENS,
    selected_request_metadata_shape,
    selected_runtime_storage_bytes,
    selected_union_capacity,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.selected_prefill_test_binding import (
    _owners,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.selected_prefill_test_source import (
    SelectedLoadStep,
    SelectedRequest,
)


def test_union_capacity_and_metadata_bounds():
    assert selected_union_capacity(1) == 2048
    assert selected_union_capacity(8192) == MAX_HISTORY_TOKENS
    assert selected_request_metadata_shape(8192, 256) == (256, 4)
    assert selected_runtime_storage_bytes(8192, 256) == 256 * 4 * 4


def test_selected_metadata_rejects_oversized_batches():
    with pytest.raises(ValueError, match="8192"):
        selected_request_metadata_shape(8193, 1)
    with pytest.raises(ValueError, match="256"):
        selected_request_metadata_shape(1, 257)


def test_load_step_protocol_returns_page_bases_and_ready_event():
    event = torch.npu.Event() if hasattr(torch, "npu") else object()
    step = SelectedLoadStep(
        SelectedRequest("req", 16384, 2, True),
        {"main.0": torch.zeros(2, dtype=torch.uint64)},
        event,
    )
    assert step.requests[0].history_tokens == 16384
    assert step.page_bases("main.0").dtype == torch.uint64
    assert step.ready("main.0") is event
    step.finish(completion=None)
    assert step.try_retire()


def test_owner_mapping_uses_the_last_producer_for_consumers():
    context = {
        "main.0": SimpleNamespace(impl=SimpleNamespace(skip_topk=False, kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048)),
        "main.1": SimpleNamespace(impl=SimpleNamespace(skip_topk=True, kv_lora_rank=512, qk_rope_head_dim=64)),
        "main.2": SimpleNamespace(impl=SimpleNamespace(skip_topk=False, kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048)),
        "main.3": SimpleNamespace(impl=SimpleNamespace(skip_topk=True, kv_lora_rank=512, qk_rope_head_dim=64)),
    }
    runner = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=context))
    assert _owners(runner, list(context)) == {
        "main.0": "main.0",
        "main.1": "main.0",
        "main.2": "main.2",
        "main.3": "main.2",
    }
