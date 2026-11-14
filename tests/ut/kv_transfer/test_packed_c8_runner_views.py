# SPDX-License-Identifier: Apache-2.0

"""Test the guarded packed view conversion with real CPU storage."""

import ast
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.worker.dsa_shared_pool import reshape_dsa_shared_pool_raw, reshape_packed_dsa_shared_pool_raw

from test_packed_c8_spec import ROOT, _load_definitions
from test_packed_c8_spec import api, bf16_main, indexer, initialize, packed, packed_shared_pool, vllm_root  # noqa: F401

MAIN_LAYERS = [f"model.layers.{layer}.self_attn.attn" for layer in range(3)]
INDEXER_LAYERS = [f"model.layers.{layer}.self_attn.indexer.k_cache" for layer in (0, 2)]
ALIGNMENT = 2 * 1024 * 1024
RAW_BUNDLE_BYTES = 2_686_976
SCALE_BUNDLE_BYTES = 41_984


@pytest.fixture
def runner_views(api, vllm_root, packed_shared_pool):
    namespace = {
        "dataclass": dataclass,
        "AttentionSpec": api.AttentionSpec,
        "MLAAttentionSpec": api.MLAAttentionSpec,
        "UniformTypeKVCacheSpecs": api.UniformTypeKVCacheSpecs,
        "dsa_shared_block_layout": packed_shared_pool["dsa_shared_block_layout"],
        "get_dtype_size": lambda dtype: dtype.itemsize,
        "reshape_dsa_shared_pool_raw": reshape_dsa_shared_pool_raw,
        "reshape_packed_dsa_shared_pool_raw": reshape_packed_dsa_shared_pool_raw,
        "torch": torch,
    }
    _load_definitions(
        vllm_root / "v1/kv_cache_interface.py", {"KVCacheGroupSpec", "KVCacheConfig", "KVCacheTensor"}, namespace
    )
    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    runner_class = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    names = {"_get_layer_kv_cache_specs", "_reshape_packed_dsa_shared_pool", "_reshape_kv_cache_tensors"}
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(node for node in runner_class.body if getattr(node, "name", None) in names)
    exec(compile(tree, str(source), "exec"), namespace)

    def make_runner(main, li, raw_bytes, scale_bytes, *, parent_capacity=1):
        group, tensor = namespace["KVCacheGroupSpec"], namespace["KVCacheTensor"]
        groups = [group(list(MAIN_LAYERS), main), group(list(INDEXER_LAYERS), li)]
        # Use an actual shared_by allocation declaration and Runner raw tuples.
        layers = MAIN_LAYERS + INDEXER_LAYERS
        config = namespace["KVCacheConfig"](parent_capacity, [tensor(raw_bytes + scale_bytes, layers)], groups)
        # Reserve space for the following aligned address even if the
        # allocation starts aligned, plus room for the misalignment cases.
        backing = torch.zeros(raw_bytes + scale_bytes + ALIGNMENT + 4, dtype=torch.int8)
        offset = ALIGNMENT - backing.data_ptr() % ALIGNMENT
        raw = backing[offset : offset + raw_bytes]
        scale = backing[offset + raw_bytes : offset + raw_bytes + scale_bytes]
        c8_owners = frozenset(li.indexer_c8_layer_names or ())
        raws = {name: (raw, scale) if name in c8_owners else (raw,) for name in layers}
        runner = SimpleNamespace(
            device=torch.device("cpu"),
            dsa_shared_pool=True,
            dsa_unbundle=True,
            layerwise_prefill_p_node=True,
            use_sparse=True,
            use_sparse_c8_indexer=li.cache_sparse_c8,
            runner_only_attn_layers=set(),
            _mixed_indexer_c8_names=c8_owners or None,
            sparse_head_dim=(512, 64, 128),
            kv_cache_dtype=torch.bfloat16,
            c8_k_cache_dtype=torch.int8,
            kv_cache_config=config,
            _kv_cache_spec_attn_group_iterator=lambda: (
                SimpleNamespace(backend=None, layer_names=g.layer_names) for g in groups
            ),
        )
        for name in names:
            setattr(runner, name, namespace[name].__get__(runner))
        return SimpleNamespace(runner=runner, config=config, raws=raws, raw=raw, scale=scale, backing=backing)

    return make_runner


@pytest.fixture
def packed_views(runner_views, packed, indexer):
    mixed = replace(indexer, indexer_c8_layer_names=(INDEXER_LAYERS[1],), shared_pool_alignment_bytes=ALIGNMENT)
    parent_capacity = 2
    slots = parent_capacity * len(MAIN_LAYERS) + 1
    return runner_views(
        packed, mixed, slots * RAW_BUNDLE_BYTES, slots * SCALE_BUNDLE_BYTES, parent_capacity=parent_capacity
    )


def test_packed_runner_views_match_child_pool_addresses(packed_views, packed_shared_pool):
    state = packed_views
    main_group, indexer_group = state.config.kv_cache_groups
    parent_layout = packed_shared_pool["dsa_shared_block_layout"](
        main_group.kv_cache_spec, indexer_group.kv_cache_spec, state.config.num_blocks, prefill_child=True
    )
    pool = packed_shared_pool["PrefillLayerBundlePool"](
        packed_shared_pool["DSASharedBundleAllocator"](parent_layout), len(MAIN_LAYERS), bank_count=2
    )
    layout = pool.layout
    owner = packed_shared_pool["DSASharedBlockOwner"]
    assert layout.slot_count == 7
    views = state.runner._reshape_kv_cache_tensors(state.config, state.raws)
    (main,) = views[MAIN_LAYERS[0]]
    # The view exists even for the main-only follower. Resolving which
    # indexer owner that follower uses is outside the reshape contract.
    assert all(len(views[name]) == 1 and views[name][0] is main for name in MAIN_LAYERS)
    (bf16_li,) = views[INDEXER_LAYERS[0]]
    c8_li, scale = views[INDEXER_LAYERS[1]]
    assert (main.dtype, bf16_li.dtype, c8_li.dtype, scale.dtype) == (
        torch.int8, torch.bfloat16, torch.int8, torch.float16
    )
    assert main.shape == (224, 128, 1, 656)
    assert bf16_li.shape == (574, 128, 1, 128)
    assert c8_li.shape == (1148, 128, 1, 128)
    assert scale.shape == (1148, 128, 1, 1)
    assert state.raw.storage_offset() > 0
    assert state.raw.data_ptr() % ALIGNMENT == 0
    for view, region in ((main, state.raw), (bf16_li, state.raw), (c8_li, state.raw), (scale, state.scale)):
        assert view.is_contiguous()
        assert view.untyped_storage().data_ptr() == state.backing.data_ptr()
        assert view.storage_offset() * view.element_size() == region.storage_offset()
    # Check every child, including the transition (slot=0,parent=2) to
    # (slot=1,parent=1) and the last child in the allocation.
    for physical_slot in range(len(MAIN_LAYERS)):
        for parent_id in (1, 2):
            child = pool.child_bundle_id(physical_slot, parent_id)
            assert (pool.physical_slot(child), pool.parent_bundle_id(child)) == (physical_slot, parent_id)
            main_ids = layout.block_ids_for_bundle(owner.LATENT, child)
            li_ids = layout.block_ids_for_bundle(owner.INDEXER, child)
            assert main[main_ids[0]].data_ptr() == state.raw.data_ptr() + child * RAW_BUNDLE_BYTES
            assert bf16_li[li_ids[0]].data_ptr() == main[main_ids[0]].data_ptr()
            assert scale[2 * li_ids[0]].data_ptr() == state.scale.data_ptr() + child * SCALE_BUNDLE_BYTES
            for li_id in (li_ids[0], li_ids[-1]):
                assert c8_li[2 * li_id].data_ptr() == bf16_li[li_id].data_ptr()
                assert c8_li[2 * li_id + 1].data_ptr() == bf16_li[li_id].data_ptr() + 16_384
            assert main[main_ids[-1]].data_ptr() + 83_968 == state.raw.data_ptr() + (child + 1) * RAW_BUNDLE_BYTES
            scale_end = scale[2 * li_ids[-1] + 1].data_ptr() + 256
            assert scale_end == state.scale.data_ptr() + (child + 1) * SCALE_BUNDLE_BYTES


def test_native_runner_keeps_bf16_split_main_planes(runner_views, bf16_main, indexer):
    indexer = replace(indexer, cache_sparse_c8=False)
    slots = len(MAIN_LAYERS) + 1
    state = runner_views(bf16_main, indexer, slots * 9 * 32_768, 0)
    state.runner._reshape_packed_dsa_shared_pool = Mock(side_effect=AssertionError("packed conversion called"))
    views = state.runner._reshape_kv_cache_tensors(state.config, state.raws)
    nope, rope = views[MAIN_LAYERS[1]]
    (li,) = views[INDEXER_LAYERS[0]]
    assert nope.shape == (2 * slots, 128, 1, 512)
    assert rope.shape == (2 * slots, 128, 1, 64)
    assert li.shape == (9 * slots, 128, 1, 128)
    assert nope.dtype == rope.dtype == li.dtype == torch.bfloat16
    assert nope.data_ptr() == li.data_ptr() == state.raw.data_ptr()
    assert rope.data_ptr() == nope.data_ptr() + slots * 8 * 32_768
    state.runner._reshape_packed_dsa_shared_pool.assert_not_called()


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("unknown_marker", "explicitly packed"),
        ("wrong_owners", "mixed BF16/C8"),
        ("wrong_budget", "global raw and scale allocation"),
        ("missing_raw", "cover the configured layer groups"),
        ("wrong_raw", "one shared raw tensor"),
        ("missing_scale", "Invalid packed main C8 raw tensors"),
        ("foreign_device", "Invalid packed main C8 raw tensors"),
        ("wrong_scale_dtype", "scale must be a contiguous"),
        ("wrong_scale_size", "scale must expose exactly"),
        ("scale_other_backing", "slices of one aligned backing"),
        ("misaligned_backing", "slices of one aligned backing"),
        ("scale_gap", "slices of one aligned backing"),
        ("paired_bank_slots", "two layerwise prefill shared-pool groups"),
    ],
)
def test_invalid_packed_runner_inputs_fail_before_binding(packed_views, change, error):
    state = packed_views
    owner = INDEXER_LAYERS[1]
    if change == "unknown_marker":
        group = state.config.kv_cache_groups[0]
        group.kv_cache_spec = replace(group.kv_cache_spec, cache_sparse_sfa_c8="packed")
    elif change == "wrong_owners":
        state.runner._mixed_indexer_c8_names = frozenset([MAIN_LAYERS[1]])
    elif change == "wrong_budget":
        state.config.kv_cache_tensors[0].size -= SCALE_BUNDLE_BYTES
    elif change == "missing_raw":
        del state.raws[MAIN_LAYERS[1]]
    elif change == "wrong_raw":
        state.raws[MAIN_LAYERS[1]] = (state.raw.clone(),)
    elif change == "missing_scale":
        state.raws[owner] = (state.raw,)
    elif change == "foreign_device":
        state.raws[owner] = (state.raw, state.scale.to("meta"))
    elif change == "wrong_scale_dtype":
        state.raws[owner] = (state.raw, state.scale.view(torch.float16))
    elif change == "wrong_scale_size":
        state.raws[owner] = (state.raw, state.scale[:-2])
    elif change == "scale_other_backing":
        state.raws[owner] = (state.raw, state.scale.clone())
    elif change == "misaligned_backing":
        start = state.raw.storage_offset() + 4
        split = start + state.raw.numel()
        raw = state.backing[start:split]
        scale = state.backing[split : split + state.scale.numel()]
        assert raw.data_ptr() % ALIGNMENT == 4
        assert scale.data_ptr() == raw.data_ptr() + raw.numel()
        state.raws = {name: (raw, scale) if name == owner else (raw,) for name in state.raws}
    elif change == "scale_gap":
        start = state.scale.storage_offset() + 4
        scale = state.backing[start : start + state.scale.numel()]
        assert scale.untyped_storage().data_ptr() == state.raw.untyped_storage().data_ptr()
        assert scale.data_ptr() == state.raw.data_ptr() + state.raw.numel() + 4
        state.raws[owner] = (state.raw, scale)
    elif change == "paired_bank_slots":
        state.config.dsa_paired_bank_slots = 1
    with pytest.raises(ValueError, match=error):
        state.runner._reshape_kv_cache_tensors(state.config, state.raws)


def test_production_guard_blocks_the_same_packed_config(initialize, packed_views):
    state = packed_views
    state.runner.vllm_config = None
    state.runner._validate_dsa_compact_startup_policy = Mock()
    state.runner.initialize_attn_backend = Mock()
    state.runner._allocate_kv_cache_tensors = Mock()
    with pytest.raises(ValueError, match="not enabled in the production runner"):
        initialize(state.runner, state.config)
    state.runner._validate_dsa_compact_startup_policy.assert_not_called()
    state.runner.initialize_attn_backend.assert_not_called()
    state.runner._allocate_kv_cache_tensors.assert_not_called()
