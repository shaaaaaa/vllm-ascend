# SPDX-License-Identifier: Apache-2.0
"""Run the production group/dispatch code on CPU, without importing the NPU stack."""

import ast
import gc
import importlib.util
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def load_source(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


resident = load_source(
    "vllm_ascend/distributed/kv_transfer/sparse_offload/resident_sorted_cache.py", "shared_plan_test_resident"
)
registry = load_source(
    "vllm_ascend/distributed/kv_transfer/sparse_offload/resident_sparse_cache.py", "shared_plan_test_registry"
)


def methods(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    module = ast.parse("from __future__ import annotations")
    module.body.extend(nodes)
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return type("SourceMethods", (), {name: namespace[name] for name in names})


@pytest.fixture
def api():
    ns = dict(vars(resident), np=np, envs=NS(VLLM_ASCEND_MTP_DRAFT_DEBUG=False))
    ns["_configured_resident_shards"] = lambda mtp: (4, 4 * mtp)
    ns["prepare_sparse_indices"] = Mock(return_value="legacy")
    ns["prepare_resident_sharded_union_"] = Mock()

    def finalize(topk, table, indices, generations, state, workspace, **kwargs):
        # A second planner would observe the producer's new state as a hit.
        workspace.miss_counts[:, 0].fill_(3 if not state.counts.any() else 0)
        state.counts.fill_(1)
        topk.add_(100)
        workspace.miss_tokens.fill_(7)
        workspace.target_slots.fill_(5)
        return workspace.miss_tokens, workspace.miss_counts[:, 0], workspace.target_slots

    ns["prepare_sorted_resident_cache_fused_"] = Mock(side_effect=finalize)
    cls = methods(
        "vllm_ascend/attention/sfa_v1.py",
        {
            "initialize_sorted_resident_cache",
            "_prepare_sorted_resident_sparse_cache",
            "_prepare_decode_sparse_indices",
            "_get_indexcache_topk_indices",
            "_update_indexcache_topk_indices",
            "_copy_to_staged_sfa_bridge",
        },
        ns,
    )
    return cls, ns


def impl(api, index, raw, *, shared=False, mtp=1):
    cls, _ = api
    obj = cls()
    obj.__dict__.update(
        layer_name=f"model.layers.{index}.self_attn.attn",
        has_indexer=not shared,
        skip_topk=shared,
        shared_resident_candidate=True,
        shared_resident_plan=None,
        dsa_resident_cache=True,
        decode_threshold=mtp,
        index_topk=2048,
        block_size=128,
        dsa_shrink_latent=2,
        topk_indices_buffer=raw,
        _indexcache_topk_staging=torch.empty_like(raw),
        _sorted_resident_state=None,
        _sorted_resident_workspace=None,
        _sorted_resident_workspace_views={},
        q_b_proj=NS(weight=torch.empty(1)),
        vllm_config=NS(scheduler_config=NS(max_num_seqs=2)),
        enable_mlapo=False,
        enable_dsa_cp=False,
        enable_dsa_cp_with_o_proj_tp=False,
        use_sparse_c8_indexer=False,
    )
    return obj


def runner_api(implementations, kinds):
    enum = NS(NONE=0, PIECEWISE=1)
    states = NS(DecodeOnly=1, SpecDecoding=2)
    ns = dict(
        torch=torch,
        np=np,
        SharedResidentPlan=resident.SharedResidentPlan,
        AttentionLayerBase=object,
        envs_ascend=NS(VLLM_ASCEND_SFA_FULL_GRAPH=False),
        StagedSFAConfigReason=NS(CUDAGRAPH_MODE="mode"),
        CUDAGraphMode=enum,
        AscendAttentionState=states,
        logger=Mock(),
        staged_sfa_graph_configuration_reasons=lambda config: (),
        parse_layer_idx=lambda name: int(name.split(".layers.")[1].split(".")[0]),
    )
    methods("vllm_ascend/attention/sfa_v1.py", {"_fixed_staged_decode_mtp", "_get_indexer_types"}, ns)
    layers = {i.layer_name: NS(layer_name=i.layer_name, impl=i) for i in implementations}
    ns["get_layers_from_vllm_config"] = lambda *args: layers
    cls = methods(
        "vllm_ascend/worker/model_runner_v1.py",
        {
            "_bind_shared_resident_plans",
            "_validate_shared_resident_layout",
            "_prepare_shared_resident_plans",
            "_model_forward",
        },
        ns,
    )
    runner = cls()
    runner.vllm_config = NS(
        model_config=NS(hf_text_config=NS(indexer_types=kinds)),
        compilation_config=NS(cudagraph_mode=1),
        parallel_config=NS(enable_dbo=False),
    )
    runner.dsa_resident_cache = True
    runner.max_num_reqs = 2
    runner._shared_resident_groups = []
    runner._shared_resident_failed = False
    runner._resident_state_registry = registry.ResidentRequestStateRegistry(2)
    return runner, ns


def grouped(api, mtp=1, bounded=False):
    raw = torch.zeros((4, 2048), dtype=torch.int32)
    layers = [impl(api, 0, raw, mtp=mtp), impl(api, 1, raw, shared=True, mtp=mtp)]
    runner, ns = runner_api(layers, ["full", "shared"])
    ns["envs_ascend"].VLLM_ASCEND_SFA_FULL_GRAPH = bounded
    runner._bind_shared_resident_plans()
    group = runner._shared_resident_groups[0]
    group.active = True
    return runner, layers, group, ns


def plan(layer, raw, *, requests=1, writes=None, reads=None, packed=True):
    return layer._prepare_decode_sparse_indices(
        raw,
        torch.full((raw.shape[0],), 4096),
        torch.arange(requests).repeat_interleave(layer.decode_threshold),
        torch.tensor([[3, 1]] * requests),
        torch.empty((requests, 2048), dtype=torch.int32),
        torch.empty((requests, 16), dtype=torch.int32),
        torch.empty((requests, 2048), dtype=torch.int64),
        torch.arange(requests, dtype=torch.int32),
        torch.ones(requests, dtype=torch.int64),
        local_to_union_workspace=None,
        shard_packed_workspace=None,
        shard_mapping_workspace=None,
        shard_counts_workspace=None,
        staged_mtp=layer.decode_threshold,
        need_packed=packed,
        clear_invalid_rows=True,
        resident_reads=reads,
        resident_writes=writes,
    )


@pytest.mark.parametrize("mtp", [1, 2])
def test_one_plan_raw_publication_and_capacity_views(api, mtp):
    _, (producer, consumer), group, _ = grouped(api, mtp)
    state = group.state
    for requests in (1, 2, 1):
        raw = torch.full((requests * mtp, 1, 2048), requests, dtype=torch.int32)
        producer._update_indexcache_topk_indices(raw)
        expected = plan(producer, raw, requests=requests)
        selected = consumer._get_indexcache_topk_indices(requests * mtp)
        assert selected.data_ptr() == producer.topk_indices_buffer.data_ptr()
        actual = plan(consumer, selected, requests=requests)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        assert actual[0].data_ptr() == group.topk.data_ptr()
        assert actual[2].stride() == (16,)
        assert torch.equal(raw, selected)  # Producer remapping never corrupts raw publication.
        assert torch.all(actual[0] == requests + 100)
        assert group.state is state
    assert api[1]["prepare_resident_sharded_union_"].call_count == 3
    assert api[1]["prepare_sorted_resident_cache_fused_"].call_count == 3
    api[1]["prepare_sparse_indices"].assert_not_called()


def test_consumer_keeps_misses_despite_updated_state_and_private_bridge(api):
    _, (producer, consumer), group, _ = grouped(api)
    expected = plan(producer, torch.zeros((1, 1, 2048), dtype=torch.int32))
    actual = plan(consumer, torch.zeros_like(expected[0]))
    assert actual[2].tolist() == [3]
    bridge = tuple(torch.empty_like(t) for t in actual)
    consumer._ensure_staged_sfa_bridge_buffers = lambda hidden: bridge
    copied = consumer._copy_to_staged_sfa_bridge(torch.empty(1), actual)
    for source, destination in zip(actual, copied):
        assert torch.equal(source, destination)
        assert source.data_ptr() != destination.data_ptr()
    copied[0].zero_()
    assert group.topk[0].eq(100).all()


def test_explicit_operator_storage_is_authoritative(api):
    _, (producer, consumer), group, _ = grouped(api)
    writes = [t.clone() for t in group.writes]
    actual = plan(producer, torch.ones((1, 1, 2048), dtype=torch.int32), writes=writes)
    assert actual[0].data_ptr() == writes[0].data_ptr()
    assert group.state.counts.eq(0).all()
    assert writes[3].eq(1).all()
    reads = [t.clone() for t in actual]
    reads[0].fill_(999)
    consumed = plan(consumer, torch.ones_like(actual[0]), reads=reads)
    assert consumed[0].eq(999).all()


@pytest.mark.parametrize("metadata_present", [False, True])
def test_staged_native_fallback_forwards_explicit_buffers(metadata_present):
    ns = dict(torch=torch, get_forward_context=lambda: NS(staged_sfa_graph_key=None))
    cls = methods("vllm_ascend/attention/sfa_v1.py", {"cross_layer_graph_pre"}, ns)
    obj = cls()
    obj.forward = Mock()
    obj._cross_layer_empty_outputs = lambda hidden: "empty"
    obj._cross_layer_kv_cache = lambda name, caches: (caches, None, False)
    writes, reads = [torch.empty(1)], [torch.empty(1)]
    assert (
        obj.cross_layer_graph_pre(
            "layer",
            torch.empty(1),
            (),
            NS() if metadata_present else None,
            False,
            torch.empty(1),
            resident_reads=reads,
            resident_writes=writes,
        )
        == "empty"
    )
    assert obj.forward.call_args.kwargs == dict(resident_reads=reads, resident_writes=writes)


def test_inactive_group_preserves_legacy_staging_and_avoids_shared_state(api):
    _, (_, consumer), group, _ = grouped(api)
    group.active = False
    raw = consumer._get_indexcache_topk_indices(1)
    assert raw.data_ptr() != consumer.topk_indices_buffer.data_ptr()
    assert plan(consumer, raw) == "legacy"
    assert group.state.counts.eq(0).all()


def test_no_group_keeps_independent_planner(api):
    layer = impl(api, 0, torch.zeros((2, 2048), dtype=torch.int32))
    layer.initialize_sorted_resident_cache()
    plan(layer, torch.zeros((1, 1, 2048), dtype=torch.int32))
    api[1]["prepare_resident_sharded_union_"].assert_called_once()


@pytest.mark.parametrize("outer_config", [False, True])
def test_topology_uses_order_not_reused_buffer_and_excludes_draft(api, outer_config):
    raw = torch.zeros((4, 2048), dtype=torch.int32)
    layers = [impl(api, i, raw, shared=i in (1, 3)) for i in range(6)]
    layers[-1].shared_resident_candidate = False
    runner, _ = runner_api(layers, ["full", "shared", "full", "shared", "full"])
    if outer_config:
        runner.vllm_config.model_config.hf_config = runner.vllm_config.model_config.hf_text_config
        runner.vllm_config.model_config.hf_text_config = NS()
    runner._bind_shared_resident_plans()
    assert len(runner._shared_resident_groups) == 2
    assert layers[0]._sorted_resident_state is layers[1]._sorted_resident_state
    assert layers[0]._sorted_resident_state is not layers[2]._sorted_resident_state
    assert layers[4].shared_resident_plan is None
    assert layers[4]._sorted_resident_state is not None
    assert layers[5]._sorted_resident_state is None


@pytest.mark.parametrize("fault", ["missing", "leading", "role", "buffer", "geometry", "duplicate"])
def test_incompatible_topology_rejected(api, fault):
    raw = torch.zeros((4, 2048), dtype=torch.int32)
    layers = [impl(api, 0, raw), impl(api, 1, raw, shared=True)]
    kinds = ["full", "shared"]
    if fault == "missing":
        layers.pop()
    elif fault == "leading":
        kinds[0] = "shared"
    elif fault == "role":
        layers[1].skip_topk = False
    elif fault == "buffer":
        layers[1].topk_indices_buffer = raw.clone()
    elif fault == "duplicate":
        duplicate = impl(api, 1, raw, shared=True)
        duplicate.layer_name = "other.layers.1.self_attn.attn"
        layers.append(duplicate)
    else:
        layers[1].block_size = 64
    runner, _ = runner_api(layers, kinds)
    with pytest.raises(ValueError):
        runner._bind_shared_resident_plans()


def metadata(group):
    return NS(
        need_sparse_lmcache_payload=True,
        split_boundary=torch.tensor([4096] * group.mtp),
        num_decode_tokens=group.mtp,
        resident_state_indices=torch.tensor([0]),
        resident_state_generations=torch.tensor([1]),
        block_table=torch.tensor([[2, 1]]),
        decode_req_indices_cpu=np.array([0] * group.mtp),
        attn_state=1,
        req_ids=["r"],
    )


def test_preflight_fallback_and_generation_reuse(api):
    runner, _, group, ns = grouped(api)
    common = metadata(group)
    context = NS(attn_metadata=dict.fromkeys(group.members, common), staged_sfa_graph_key=None)
    ns["get_forward_context"] = lambda: context
    rows, before = runner._resident_state_registry.bind(["r"], [(2, 1)])
    runner._prepare_shared_resident_plans(1)
    assert group.active
    _, same = runner._resident_state_registry.bind(["r"], [(2, 1)])
    assert np.array_equal(before, same)
    common.attn_state = 3  # Mixed/prefill must invalidate any prior scratch ownership.
    runner._prepare_shared_resident_plans(1)
    assert not group.active
    after_rows, after = runner._resident_state_registry.bind(["r"], [(2, 1)])
    assert np.array_equal(rows, after_rows) and np.all(after > before)
    common.attn_state = 1
    runner._prepare_shared_resident_plans(1)
    assert group.active


def test_preflight_rejects_separate_metadata_before_model_execution(api):
    runner, _, group, ns = grouped(api)
    ns["get_forward_context"] = lambda: NS(attn_metadata={name: metadata(group) for name in group.members})
    with pytest.raises(ValueError, match="same live"):
        runner._prepare_shared_resident_plans(1)


@pytest.mark.parametrize("fault", [None, "builder", "block_size", "stride", "dtype"])
def test_destination_slot_layout_qualification(api, fault):
    runner, _, group, _ = grouped(api)
    runner.attn_groups = [[NS(layer_names=group.members)]]
    caches = {name: (torch.empty(32, 128, 1, 4), torch.empty(32, 128, 1, 2)) for name in group.members}
    if fault == "builder":
        runner.attn_groups = [[NS(layer_names=[name]) for name in group.members]]
    elif fault == "block_size":
        caches[group.members[1]] = (torch.empty(32, 64, 1, 4), torch.empty(32, 64, 1, 2))
    elif fault == "stride":
        caches[group.members[1]] = (torch.empty(32, 128, 1, 8)[..., ::2], caches[group.members[1]][1])
    elif fault == "dtype":
        caches[group.members[1]] = tuple(c.double() for c in caches[group.members[1]])
    if fault:
        with pytest.raises(ValueError):
            runner._validate_shared_resident_layout(caches)
    else:
        runner._validate_shared_resident_layout(caches)


def test_capture_without_live_connector_still_captures_shared_planning(api):
    runner, _, group, ns = grouped(api)
    common = metadata(group)
    common.need_sparse_lmcache_payload = False
    ns["get_forward_context"] = lambda: NS(
        attn_metadata=dict.fromkeys(group.members, common),
        staged_sfa_graph_key=object(),
        staged_sfa_graph_dummy_run=True,
    )
    runner._prepare_shared_resident_plans(1)
    assert group.active


@pytest.mark.parametrize("staged", [False, True])
def test_missing_request_state_fails_before_producer(api, staged):
    runner, _, group, ns = grouped(api)
    common = metadata(group)
    common.resident_state_generations = None
    ns["get_forward_context"] = lambda: NS(
        attn_metadata=dict.fromkeys(group.members, common), staged_sfa_graph_key=object() if staged else None
    )
    with pytest.raises(RuntimeError, match="request state"):
        runner._prepare_shared_resident_plans(1)


def test_partial_forward_is_fail_stop(api):
    runner, _, group, ns = grouped(api)
    ns["get_forward_context"] = lambda: NS(attn_metadata=dict.fromkeys(group.members, metadata(group)))
    runner.model = Mock(side_effect=RuntimeError("consumer fill failed"))
    with pytest.raises(RuntimeError, match="consumer fill"):
        runner._model_forward(1)
    with pytest.raises(RuntimeError, match="restart"):
        runner._model_forward(1)
    assert runner.model.call_count == 1


def test_interrupted_forward_cannot_leave_reusable_planned_state(api):
    runner, _, group, ns = grouped(api)
    ns["get_forward_context"] = lambda: NS(attn_metadata=dict.fromkeys(group.members, metadata(group)))
    runner.model = Mock(side_effect=KeyboardInterrupt("interrupted after producer planning"))
    with pytest.raises(KeyboardInterrupt):
        runner._model_forward(1)
    assert runner._shared_resident_failed
    with pytest.raises(RuntimeError, match="restart"):
        runner._model_forward(1)


def test_runner_retains_plan_storage_with_gc_disabled(api):
    enabled = gc.isenabled()
    gc.disable()
    try:
        runner, layers, group, ns = grouped(api)
        refs = [weakref.ref(t) for t in group.writes]
        addresses = [t.data_ptr() for t in group.writes]
        ns["get_layers_from_vllm_config"] = lambda *args: {}
        del layers, group
        assert [ref().data_ptr() for ref in refs] == addresses
        owner = runner._shared_resident_groups[0]
        assert owner.plans[1][0].data_ptr() == owner.plans[2][0].data_ptr()
        assert owner.state.counts.eq(0).all()
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("enabled", [False, True])
def test_preemption_with_recycled_block_ids_changes_shared_generation(enabled):
    tree = ast.parse((ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_update_states")
    module = ast.parse("from __future__ import annotations\nclass Runner(Base): pass")
    module.body[1].body = [method]
    base = type("Base", (), {"_update_states": lambda self, output: None})
    ns = dict(Base=base)
    exec(compile(ast.fix_missing_locations(module), "resident_preemption", "exec"), ns)
    runner = ns["Runner"]()
    runner._sfa_full_graph = None
    runner._shared_resident_groups = [object()] if enabled else []
    runner._resident_state_registry = registry.ResidentRequestStateRegistry(2)
    rows, before = runner._resident_state_registry.bind(["r"], [(2, 1)])
    runner._update_states(NS(finished_req_ids=set(), preempted_req_ids={"r"}))
    # Resume can reallocate exactly the same block numbers, after their old
    # contents were overwritten; request ID + block signature cannot detect it.
    resumed_rows, after = runner._resident_state_registry.bind(["r"], [(2, 1)])
    assert np.array_equal(rows, resumed_rows)
    assert bool(np.all(after > before)) is enabled


def test_shared_operator_schema_preserves_old_api_and_declares_writes():
    ns = dict(
        torch=torch,
        StagedSFABridge=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    )
    methods(
        "vllm_ascend/ops/mla.py",
        {
            "sfa_forward_pre",
            "sfa_forward_pre_fake",
            "sfa_forward_pre_shared",
            "sfa_forward_pre_shared_fake",
        },
        ns,
    )
    old = torch.library.infer_schema(ns["sfa_forward_pre"], mutates_args=["output"])
    shared = torch.library.infer_schema(ns["sfa_forward_pre_shared"], mutates_args=["output", "resident_writes"])
    assert "resident_" not in old
    assert "Tensor[] resident_reads" in shared and "!)[] resident_writes" in shared
    lib = torch.library.Library("shared_resident_schema_test", "DEF")
    lib.define("pre" + shared)  # Catch unsupported optional-list/default schemas.
    args = (torch.empty(2, 8), False, torch.empty(2, 8), "layer", 1, 4, 2, 2048, 2, 1, 4096)
    fake = ns["sfa_forward_pre_shared_fake"](*args, [], [])
    legacy = ns["sfa_forward_pre_fake"](*args)
    assert [(t.shape, t.dtype) for t in fake] == [(t.shape, t.dtype) for t in legacy]


@pytest.mark.parametrize("mtp", [1, 2])
def test_compiled_producer_consumer_observe_explicit_mutation(api, mtp):
    _, (producer, consumer), group, _ = grouped(api, mtp)
    ns = dict(
        torch=torch,
        StagedSFABridge=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    )
    methods(
        "vllm_ascend/ops/mla.py",
        {
            "sfa_forward_pre_fake",
            "sfa_forward_pre_shared",
            "sfa_forward_pre_shared_fake",
        },
        ns,
    )

    def pre(layer, hidden, cache, metadata, gather, output, *, resident_reads, resident_writes):
        current = producer if layer == "producer" else consumer
        raw = hidden[:, :1].to(torch.int32).expand(-1, 2048).unsqueeze(1)
        values = plan(current, raw, writes=resident_writes, reads=resident_reads)
        # Match the production contract: six private outputs, no input aliases.
        return (hidden[:, None, :].clone(), hidden[:, None, :2].clone(), *(t.clone() for t in values))

    producer.cross_layer_graph_pre = pre
    consumer.cross_layer_graph_pre = pre
    ns["_mla_runtime_state"] = lambda name: (producer if name == "producer" else consumer, name, (), None)
    # Dynamo retains operator handles after the local library is destroyed.
    # Keep parameterized cases in separate namespaces, as separate workers are.
    namespace = f"shared_resident_compile_test_{mtp}"
    lib = torch.library.Library(namespace, "DEF")
    lib.define(
        "pre" + torch.library.infer_schema(ns["sfa_forward_pre_shared"], mutates_args=["output", "resident_writes"])
    )
    lib.impl("pre", ns["sfa_forward_pre_shared"], "CPU")
    lib._register_fake("pre", ns["sfa_forward_pre_shared_fake"])
    op = getattr(torch.ops, namespace).pre

    def forward(x, writes, reads):
        output = torch.empty_like(x)
        first = op(x, False, output, "producer", 1, 4, 2, 2048, mtp, 1, mtp * 2048, [], writes)
        second = op(first[0][:, 0], False, output, "consumer", 1, 4, 2, 2048, mtp, 1, mtp * 2048, reads, [])
        return first[2], second[2], first[4], second[4]

    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True, dynamic=False)
    for value in (2, 9):
        first, second, first_counts, second_counts = compiled(
            torch.full((mtp, 4), float(value)), group.writes, group.reads
        )
        assert first.eq(value + 100).all() and torch.equal(first, second)
        assert torch.equal(first_counts, second_counts)
        assert group.state.counts.eq(1).all()
        assert group.topk[:mtp].eq(value + 100).all()


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("fault", [None, "missing_scale", "scale_dtype", "scale_capacity"])
def test_c8_producer_shares_resident_plan_and_validates_scales(api, mixed, fault):
    runner, layers, group, _ = grouped(api)
    layers[0].use_sparse_c8_indexer = True
    runner._shared_resident_groups = []
    runner._bind_shared_resident_plans()
    group = runner._shared_resident_groups[0]
    group.active = True
    runner.use_sparse_c8_indexer = True
    index_name = group.members[0].rsplit(".", 1)[0] + ".indexer.k_cache"
    runner._mixed_indexer_c8_names = frozenset([index_name]) if mixed else None
    runner.attn_groups = [[NS(layer_names=group.members)]]
    caches = {name: (torch.empty(32, 128, 1, 4), torch.empty(32, 128, 1, 2)) for name in group.members}
    blocks = 64 if mixed else 32
    key = torch.empty(blocks, 128, 1, 128, dtype=torch.int8)
    scale = torch.empty(blocks, 128, 1, 1, dtype=torch.float16)
    if fault == "scale_dtype":
        scale = scale.float()
    if fault == "scale_capacity":
        scale = scale[:-1]
    caches[index_name] = (key,) if fault == "missing_scale" else (key, scale)
    if fault:
        with pytest.raises(ValueError, match="paired key/scale"):
            runner._validate_shared_resident_layout(caches)
        return
    runner._validate_shared_resident_layout(caches)
    raw = torch.zeros((1, 1, 2048), dtype=torch.int32)
    produced = plan(layers[0], raw)
    consumed = plan(layers[1], raw)
    assert all(torch.equal(a, b) for a, b in zip(produced, consumed))
    assert not layers[1].use_sparse_c8_indexer
