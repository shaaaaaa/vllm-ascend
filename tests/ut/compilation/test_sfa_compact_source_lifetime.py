# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lease tests against real source/graph code with controlled NPU events."""

import gc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from lmcache.v1.gpu_connector.sparse import build_compact_prepared_sparse_source
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.shared_cpu_cache import LocalLeaseHandle
from sfa_test_support import load_module
from test_sfa_full_graph import graph_module as graph_module


@pytest.fixture
def lifetime(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/compilation/sfa_source_lifetime.py"
    return load_module(path, "tested_compact_source_lifetime", monkeypatch)


def source_and_lease():
    retired = []
    lease = LocalLeaseHandle(
        lease_id="source-1", mapping=object(),
        retire=lambda *args: retired.append(args),
    )
    source = build_compact_prepared_sparse_source(
        [torch.tensor([101, 102])], num_layers=1, total_tokens=6,
        chunk_token_counts=(4, 2), owner_lease=lease, chunk_size=4,
    )
    return source, lease, retired


def test_graph_hold_outlives_request_hold_and_retires_once(lifetime):
    source, request, retired = source_and_lease()
    binding = lifetime.SFASourceLease((source, source))
    request.close()
    assert not retired
    binding.close()
    binding.close()
    assert retired == [("source-1", "fenced")]


def test_compact_lease_preserves_allocator_owner_fallback(lifetime):
    source, request, retired = source_and_lease()
    tensor = torch.empty(8)
    owner = TensorMemoryObj(
        tensor,
        MemoryObjMetadata(tensor.shape, tensor.dtype, 0, 32, 1),
        None,
    )
    legacy = SimpleNamespace(graph_owners=(owner,))
    binding = lifetime.SFASourceLease((source, legacy, legacy))
    assert owner.meta.ref_count == 2
    request.close()
    binding.close()
    assert owner.meta.ref_count == 1
    assert retired == [("source-1", "fenced")]


def test_partial_bind_rolls_back_compact_hold(lifetime):
    source, request, retired = source_and_lease()
    proxy = SimpleNamespace(
        graph_owners=None,
        layers=(SimpleNamespace(memory_objs=(object(),), chunk_ptrs_npu=torch.ones(1)),),
    )
    with pytest.raises(RuntimeError, match="not proxies"):
        lifetime.SFASourceLease((source, proxy))
    assert not retired
    request.close()
    assert retired == [("source-1", "fenced")]


@pytest.mark.parametrize("query_error", [False, True])
def test_graph_retirement_waits_for_exact_event_gc_off(
    lifetime, graph_module, monkeypatch, query_error
):
    module, context, _, _ = graph_module
    monkeypatch.setattr(module, "SFASourceLease", lifetime.SFASourceLease)
    event = SimpleNamespace(record=Mock(), query=Mock(return_value=False))
    monkeypatch.setattr(torch.npu, "Event", lambda: event)
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1)
    graph = module.SFAFullGraph()
    source, request, retired = source_and_lease()
    enabled = gc.isenabled()
    gc.disable()
    try:
        graph.bind_sources((source,), ("request",), transfers=lambda: ())
        request.close()
        if query_error:
            event.query.side_effect = RuntimeError("unknown transfer completion")
            with pytest.raises(RuntimeError, match="unknown transfer"):
                graph.release_requests({"request"})
        else:
            graph.release_requests({"request"})
        assert not retired
        event.query.side_effect = None
        event.query.return_value = True
        graph.collect_retired_sources()
        assert retired == [("source-1", "fenced")]
    finally:
        if enabled:
            gc.enable()
