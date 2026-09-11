# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-check real SFA, LMCache source and Ascend transfer code on CPU.

Requires sibling LMCache/LMCache-Ascend checkouts. NPU imports/native calls and
the already-validated attention metadata are fixtures, not inference results.
The caller, constructor, source dataclasses, bind/load and native Python
wrappers are real code: a permissive transfer Mock cannot hide API drift.
"""

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def extract(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(body) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def contract(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    lmcache = root.parent / "LMCache/lmcache"
    ascend = root.parent / "LMCache-Ascend/lmcache_ascend"
    if (
        not (lmcache / "v1/gpu_connector/sparse.py").is_file()
        or not (ascend / "v1/npu_connector/sparse_graph.py").is_file()
    ):
        pytest.skip("Cross-repo CPU contract requires sibling LMCache and LMCache-Ascend checkouts")

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, value)
        spec.loader.exec_module(value)
        return value

    module("lmcache.v1.memory_management", MemoryObj=object)
    module("lmcache.logging", init_logger=logging.getLogger)
    source = load("lmcache.v1.gpu_connector.sparse", lmcache / "v1/gpu_connector/sparse.py")
    load("lmcache_ascend.v1.kv_format", ascend / "v1/kv_format.py")
    allocations, copies = [], []

    def prepare(caches, slots, fmt, k, v, dsa):
        assert len(caches) == 1 and isinstance(caches, tuple)
        state = SimpleNamespace(cache=caches[0], slots=slots, fmt=fmt)
        allocations.append(state)
        return state

    def copy(state, slots, selected, ptrs, chunk_size, total_tokens, interleaved, counts=None, diagnostic_layer_id=-1):
        assert interleaved is False and chunk_size == 256
        assert diagnostic_layer_id == -1
        copies.append(
            SimpleNamespace(
                state=state,
                slots=slots.clone(),
                selected=selected.clone(),
                ptrs=ptrs.clone(),
                total_tokens=total_tokens,
                counts=counts.clone(),
            )
        )

    utils = module(
        "lmcache_ascend.v1.npu_connector.utils",
        torch=torch,
        lmc_ops=SimpleNamespace(
            prepare_sparse_direct_destination_state=prepare, sparse_mla_dsa_batched_direct_kv_transfer_prepared=copy
        ),
    )
    extract(
        ascend / "v1/npu_connector/utils.py",
        {
            "_normalize_vllm_kv_caches",
            "prepare_sparse_direct_destination_state",
            "sparse_mla_dsa_batched_direct_kv_transfer_prepared",
        },
        utils.__dict__,
    )
    transfer_module = load("lmcache_ascend.v1.npu_connector.sparse_graph", ascend / "v1/npu_connector/sparse_graph.py")
    module("lmcache.integration.vllm.utils", lmcache_get_or_create_config=lambda: SimpleNamespace(chunk_size=256))

    context = SimpleNamespace(staged_sfa_graph_dummy_run=True, staged_sfa_route=SimpleNamespace(frontiers=()))
    namespace = {"get_forward_context": lambda: context, "torch": torch}
    path = root / "vllm_ascend/attention/sfa_v1.py"
    extract(path, {"_prepare_sfa_remap_boundary"}, namespace)
    extract(path, {"prepare_full_graph_layer"}, namespace, class_name="AscendSFAImpl")
    impl_type = type("RealLayerPreparation", (), {"prepare_full_graph_layer": namespace["prepare_full_graph_layer"]})
    return SimpleNamespace(
        context=context,
        impl_type=impl_type,
        source=source,
        allocations=allocations,
        copies=copies,
        transfer_type=transfer_module.SparseGraphTransfer,
    )


def make_layer(contract, request_capacity):
    caches = (torch.zeros((2, 16, 1, 512)), torch.zeros((2, 16, 1, 64)))
    slots = torch.zeros((request_capacity, 4), dtype=torch.int64)
    fields = (
        "cos",
        "sin",
        "slot_mapping",
        "indexer_slot_mapping",
        "cum_query_lens",
        "seq_lens",
        "block_table",
        "indexer_block_table",
        "decode_req_indices",
        "decode_selected_tokens",
        "decode_selected_counts",
        "decode_union_mapping_workspace",
        "decode_shard_packed_workspace",
        "decode_shard_mapping_workspace",
        "decode_shard_counts_workspace",
        "resident_state_indices",
        "resident_state_generations",
    )
    metadata = SimpleNamespace(
        **{name: torch.zeros(1, dtype=torch.int32) for name in fields},
        req_ids=tuple(f"request-{lane}" for lane in range(request_capacity)),
        decode_target_slot_mapping=slots,
        decode_remap_boundary=torch.zeros(request_capacity, dtype=torch.int32),
        decode_remap_boundary_ready=True,
        reshape_cache_event=object(),
    )
    impl = contract.impl_type()
    impl.index_topk = 4
    impl._staged_sfa_capture_state = SimpleNamespace(runtime=(None, caches))
    impl._staged_sfa_bridge_buffers = (torch.zeros(request_capacity * 2, 1),)
    # Graph routing/layout eligibility is covered separately; this test starts
    # from validated metadata and checks the cross-repo allocation/binding edge.
    impl._cross_layer_ineligible_reason = lambda *args: None
    contract.context.staged_sfa_graph_key = SimpleNamespace(
        request_capacity=request_capacity, token_capacity=request_capacity * 2
    )
    contract.context.attn_metadata = {"target": metadata}
    return impl, metadata


def make_source(contract, base, counts):
    layers = tuple(
        contract.source.PreparedSparseSourceLayer(
            tensors=(),
            chunk_ptrs_npu=torch.tensor([base + layer * 10000 + i * 1000 for i in range(len(counts))]),
        )
        for layer in range(8)
    )
    return contract.source.PreparedSparseSource(layers, sum(counts), tuple(counts), torch.device("cpu"))


@pytest.mark.parametrize("request_capacity", [1, 2, 4])
@pytest.mark.parametrize("layer_id", range(8))
def test_real_layer_startup_and_source_rebinding(contract, request_capacity, layer_id):
    impl, metadata = make_layer(contract, request_capacity)
    inputs = impl.prepare_full_graph_layer("target", 1024, (), layer_id)
    transfer = impl._full_graph_transfer
    assert type(transfer) is contract.transfer_type
    assert transfer.request_capacity == request_capacity
    assert len(contract.allocations) == 2
    assert inputs["decode_target_slot_mapping"] is metadata.decode_target_slot_mapping
    assert inputs["kv_caches"] is impl._staged_sfa_capture_state.runtime[1]
    assert metadata.reshape_cache_event is None
    assert not transfer.valid_tokens.any()
    addresses = (transfer.ptrs.data_ptr(), transfer.valid_tokens.data_ptr())

    contract.context.staged_sfa_graph_dummy_run = False
    for counts, base in (([13], 1000), ([256, 17], 2000), ([256, 256, 1], 3000)):
        sources = tuple(make_source(contract, base + lane * 100000, counts) for lane in range(request_capacity))
        impl.prepare_full_graph_layer("target", 1024, sources, layer_id)
        assert impl._full_graph_transfer is transfer
        assert addresses == (transfer.ptrs.data_ptr(), transfer.valid_tokens.data_ptr())
        assert transfer.valid_tokens.eq(sum(counts)).all()
        for lane, source in enumerate(sources):
            start = lane * transfer.capacity
            ptrs = source.layers[layer_id].chunk_ptrs_npu
            torch.testing.assert_close(transfer.ptrs[0, start : start + len(counts)], ptrs)
            torch.testing.assert_close(
                transfer.ptrs[1, start : start + len(counts)], ptrs + torch.tensor(counts) * 512 * 4
            )
        selected = torch.tensor([[0, sum(counts) - 1, sum(counts), -1]]).repeat(request_capacity, 1)
        transfer.load(selected, torch.full((request_capacity,), 4), torch.arange(4).repeat(request_capacity, 1))
        for call in contract.copies[-2:]:
            assert call.total_tokens == request_capacity * 1024
            torch.testing.assert_close(call.slots, torch.tensor([[0, 1, -1, -1]]).repeat(request_capacity, 1))
            assert call.selected.dtype == call.counts.dtype == torch.int32
            for lane in range(request_capacity):
                assert call.selected[lane].tolist() == [lane * 1024, lane * 1024 + sum(counts) - 1, 0, 0]

    # An empty/finished/padded request cannot reuse the last step's pointers.
    impl.prepare_full_graph_layer("target", 1024, (None,) * request_capacity, layer_id)
    transfer.load(selected, torch.full((request_capacity,), 4), metadata.decode_target_slot_mapping)
    assert not transfer.ptrs.any() and not transfer.valid_tokens.any()
    assert contract.copies[-1].slots.eq(-1).all() and not contract.copies[-1].counts.any()
    assert len(contract.allocations) == 2


def test_real_layer_cannot_allocate_new_transfer_during_live_decode(contract):
    impl, _ = make_layer(contract, 1)
    contract.context.staged_sfa_graph_dummy_run = False
    with pytest.raises(RuntimeError, match="not allocated at startup"):
        impl.prepare_full_graph_layer("target", 1024)
    assert not contract.allocations


def test_real_layer_rejects_incompatible_singleton_transfer(contract, monkeypatch):
    class LegacyTransfer:
        def __init__(self, caches, slots, chunk_size, max_tokens):
            raise AssertionError("Constructor body must not run")

    monkeypatch.setattr(
        sys.modules["lmcache_ascend.v1.npu_connector.sparse_graph"], "SparseGraphTransfer", LegacyTransfer
    )
    impl, _ = make_layer(contract, 1)
    with pytest.raises(TypeError, match="request_capacity"):
        impl.prepare_full_graph_layer("target", 1024)
