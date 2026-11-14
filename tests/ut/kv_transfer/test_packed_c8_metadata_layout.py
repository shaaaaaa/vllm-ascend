# SPDX-License-Identifier: Apache-2.0

"""Check SFA metadata sizing and Runner dispatch without loading NPU extensions."""

import ast
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_packed_c8_spec import ROOT, _load_definitions
from test_packed_c8_spec import api, bf16_main, indexer, initialize, packed, packed_shared_pool, vllm_root  # noqa: F401
from vllm_ascend.attention.selected_prefill_policy import packed_main_c8_enabled


def _load_method(source, class_name, method_name, namespace, *, keep_class=False):
    cls = next(node for node in ast.parse(source.read_text()).body if getattr(node, "name", None) == class_name)
    method = next(node for node in cls.body if getattr(node, "name", None) == method_name)
    tree = ast.parse("from __future__ import annotations")
    if keep_class:
        cls.bases = [ast.Name(id="BuilderBase", ctx=ast.Load())]
        cls.body = [method]
        tree.body.append(cls)
    else:
        tree.body.append(method)
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), namespace)
    return namespace[class_name if keep_class else method_name]


@pytest.fixture
def metadata(monkeypatch, packed_shared_pool):
    # Use the same factory's class identity for the builder's lazy import.
    pool_module = ModuleType("vllm.v1.core.dsa_shared_pool")
    pool_module.DSAPackedMainC8PrefillLayout = packed_shared_pool["DSAPackedMainC8PrefillLayout"]
    monkeypatch.setitem(sys.modules, pool_module.__name__, pool_module)

    class BuilderBase:
        def __init__(self, spec, layers, config, device, metadata_cls, supports_dcp_with_varlen):
            self.model_config = config.model_config
            self.device = device
            self.vllm_config = config

    owner_mask = Mock(side_effect=lambda names: [name == "main.1" for name in names])
    ascend_config = SimpleNamespace(
        indexer_c8_shared_block_factor=2, indexer_hbm_block_map=None, indexer_c8_layer_mask=owner_mask
    )
    namespace = {"torch": torch}
    _load_definitions(ROOT / "vllm_ascend/worker/dsa_shared_pool.py", {"MixedIndexerMetadata"}, namespace)
    namespace.update(
        BuilderBase=BuilderBase,
        envs=SimpleNamespace(VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=True, VLLM_ASCEND_DSA_UNBUNDLE=False),
        AscendSFAMetadata=object,
        AttentionMaskBuilder=Mock(),
        SFARemapBoundaryBuffer=Mock(),
        get_ascend_config=lambda: ascend_config,
        enable_dsa_cp=lambda: False,
        packed_main_c8_enabled=packed_main_c8_enabled,
    )
    builder = _load_method(
        ROOT / "vllm_ascend/attention/sfa_v1.py",
        "AscendSFAMetadataBuilder",
        "__init__",
        namespace,
        keep_class=True,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=4096 * 128, hf_config=None, hf_text_config=SimpleNamespace(qk_rope_head_dim=64)
        ),
        cache_config=SimpleNamespace(block_size=128),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=128),
        kv_transfer_config=SimpleNamespace(is_kv_producer=True),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        speculative_config=None,
    )
    return SimpleNamespace(builder=builder, config=config, ascend=ascend_config, device=torch.device("cpu"))


@pytest.fixture
def mixed_indexer(indexer):
    return replace(indexer, indexer_c8_layer_names=("li.1",))


@pytest.mark.parametrize(("max_blocks", "packed_capacity"), [(4096, 4100), (4101, 4182)])
def test_prefill_capacity_and_canonical_to_c8_mapping(
    metadata, packed_shared_pool, bf16_main, packed, mixed_indexer, max_blocks, packed_capacity
):
    factory = packed_shared_pool["dsa_shared_block_layout"]
    native_layout = factory(bf16_main, mixed_indexer, prefill_child=True)
    packed_layout = factory(packed, mixed_indexer, prefill_child=True)
    assert (native_layout.indexer_blocks_per_bundle, packed_layout.indexer_blocks_per_bundle) == (18, 82)
    metadata.config.model_config.max_model_len = max_blocks * 128
    for owner in ("main.0", "main.1"):
        native = metadata.builder(bf16_main, [owner], metadata.config, metadata.device)
        candidate = metadata.builder(
            packed, [owner], metadata.config, metadata.device, shared_pool_layout=packed_layout
        )
        assert native.max_blocks == candidate.max_blocks == max_blocks
        assert native._mixed_indexer_metadata.max_blocks == 4104
        workspace = candidate._mixed_indexer_metadata
        assert workspace.max_blocks == packed_capacity
        assert workspace.tables.numel() == 3 * packed_capacity
        assert workspace.slots.numel() == 16
        canonical_table = torch.full((1, packed_capacity), 82, dtype=torch.int32)
        canonical_slots = torch.tensor([82 * 128 + 7, -1])
        overflow_table = torch.zeros((1, packed_capacity + 1), dtype=torch.int32)
        with pytest.raises(ValueError, match="preallocated capacity"):
            workspace.update(overflow_table, canonical_slots)
        if max_blocks == 4101:
            with pytest.raises(ValueError, match="preallocated capacity"):
                native._mixed_indexer_metadata.update(canonical_table, canonical_slots)
        physical_table, physical_slots = workspace.update(canonical_table, canonical_slots)
        assert physical_table.tolist() == [[164] * packed_capacity]
        assert physical_slots.tolist() == [164 * 128 + 7, -1]
        assert canonical_table[0, 0].item() == 82
        assert canonical_slots.tolist() == [82 * 128 + 7, -1]


def test_packed_builder_rejects_missing_or_incompatible_layout(
    metadata, packed_shared_pool, bf16_main, packed, mixed_indexer
):
    with pytest.raises(ValueError, match="requires a shared-pool layout"):
        metadata.builder(packed, ["main.1"], metadata.config, metadata.device)
    factory = packed_shared_pool["dsa_shared_block_layout"]
    native_layout = factory(bf16_main, mixed_indexer, prefill_child=True)
    packed_layout = factory(packed, mixed_indexer, prefill_child=True)
    with pytest.raises(ValueError, match="matching mixed-indexer prefill"):
        metadata.builder(packed, ["main.1"], metadata.config, metadata.device, shared_pool_layout=native_layout)
    with pytest.raises(ValueError, match="matching mixed-indexer prefill"):
        metadata.builder(
            mixed_indexer, ["li.0", "li.1"], metadata.config, metadata.device, shared_pool_layout=native_layout
        )
    metadata.config.cache_config.block_size = 64
    with pytest.raises(ValueError, match="matching mixed-indexer prefill"):
        metadata.builder(packed, ["main.1"], metadata.config, metadata.device, shared_pool_layout=packed_layout)
    metadata.config.cache_config.block_size = 128
    metadata.config.kv_transfer_config.is_kv_producer = False
    with pytest.raises(ValueError, match="matching mixed-indexer prefill"):
        metadata.builder(packed, ["main.1"], metadata.config, metadata.device, shared_pool_layout=packed_layout)


def test_ordinary_and_paired_decode_capacities(metadata, packed_shared_pool, bf16_main):
    metadata.config.kv_transfer_config.is_kv_producer = False
    metadata.config.model_config.max_model_len = 4090 * 128
    ordinary = metadata.builder(bf16_main, ["main.1"], metadata.config, metadata.device)
    assert ordinary._mixed_indexer_metadata.max_blocks == 4095

    block_map = packed_shared_pool["paired_bank_indexer_block_map"](3)
    metadata.ascend.indexer_hbm_block_map = block_map
    bf16 = metadata.builder(bf16_main, ["main.0"], metadata.config, metadata.device)
    c8 = metadata.builder(bf16_main, ["main.1"], metadata.config, metadata.device)
    assert bf16._mixed_indexer_metadata is None
    assert c8._mixed_indexer_metadata.max_blocks == 4104
    blocks = torch.tensor([[8, 24, 49]], dtype=torch.int32)
    slots = torch.tensor([8 * 128 + 4, 49 * 128 + 3, -1])
    physical_table, physical_slots = c8._mixed_indexer_metadata.update(blocks, slots)
    assert physical_table.tolist() == [[block_map[8], block_map[24], block_map[49]]]
    assert physical_slots.tolist() == [block_map[8] * 128 + 4, block_map[49] * 128 + 3, -1]


@pytest.fixture
def cp_builder(metadata):
    return _load_method(
        ROOT / "vllm_ascend/attention/context_parallel/sfa_cp.py",
        "AscendSFACPMetadataBuilder",
        "__init__",
        {
            "BuilderBase": metadata.builder,
            "enabling_mlapo": lambda config: False,
            "get_pcp_group": lambda: SimpleNamespace(world_size=2, rank_in_group=0, device_group=None),
            "get_dcp_group": lambda: SimpleNamespace(world_size=1),
            "np": np,
            "torch": torch,
        },
        keep_class=True,
    )


@pytest.fixture
def runner_backend(api, vllm_root, metadata, packed_shared_pool, mixed_indexer):
    class OtherBuilder:
        def __init__(self, spec, layers, config, device):
            self.args = (spec, layers, config, device)

    class SFAWithInheritedConstructor(metadata.builder):
        pass

    def backend(name, builder):
        class Backend:
            @staticmethod
            def full_cls_name():
                return name

            @staticmethod
            def get_builder_cls():
                return builder

        return Backend

    layer_backends = {
        "main.1": backend("sfa", SFAWithInheritedConstructor),
        "li.0": backend("indexer", SFAWithInheritedConstructor),
        "li.1": backend("indexer", SFAWithInheritedConstructor),
    }
    factory = Mock(wraps=packed_shared_pool["dsa_shared_block_layout"])
    namespace = {
        "__name__": __name__,
        "NamedTuple": NamedTuple,
        "defaultdict": defaultdict,
        "dataclass": dataclass,
        "AttentionLayerBase": object,
        "AttentionGroup": lambda backend, names, spec, group, builders: builders[0],
        "MLAAttentionSpec": api.MLAAttentionSpec,
        "UniformTypeKVCacheSpecs": api.UniformTypeKVCacheSpecs,
        "AscendSFAMetadataBuilder": metadata.builder,
        "dsa_shared_block_layout": factory,
        "get_layers_from_vllm_config": lambda config, layer_cls, names: {
            name: SimpleNamespace(get_attn_backend=lambda name=name: layer_backends[name]) for name in names
        },
    }
    _load_definitions(vllm_root / "v1/kv_cache_interface.py", {"KVCacheGroupSpec", "KVCacheConfig"}, namespace)
    init = _load_method(
        ROOT / "vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", "initialize_attn_backend", namespace
    )
    runner = SimpleNamespace(
        attn_groups=[],
        dsa_shared_pool=True,
        layerwise_prefill_p_node=True,
        vllm_config=metadata.config,
        device=metadata.device,
        _check_and_update_cudagraph_mode=Mock(),
        calculate_reorder_batch_threshold=Mock(),
    )

    def make_config(main_spec):
        group = namespace["KVCacheGroupSpec"]
        return namespace["KVCacheConfig"](
            16, [], [group(["main.1"], main_spec), group(["li.0", "li.1"], mixed_indexer)]
        )

    return SimpleNamespace(
        initialize=init,
        runner=runner,
        make_config=make_config,
        backends=layer_backends,
        make_backend=backend,
        other_backend=backend("legacy", OtherBuilder),
        factory=factory,
    )


def test_packed_component_dispatches_layout_to_both_sfa_builders(runner_backend, packed, mixed_indexer):
    runner_backend.initialize(runner_backend.runner, runner_backend.make_config(packed))
    runner_backend.factory.assert_called_once_with(packed, mixed_indexer, prefill_child=True)
    main_group, indexer_group = runner_backend.runner.attn_groups
    assert main_group[0]._mixed_indexer_metadata.max_blocks == 4100
    assert indexer_group[0]._mixed_indexer_metadata.max_blocks == 4100


def test_packed_indexer_metadata_capacity_covers_padded_block_table(runner_backend, packed):
    # At 4101 blocks the 18-rounded indexer workspace is 4104 entries while
    # may_reinitialize_input_batch pads the indexer table to 82-rounded 4182.
    runner_backend.runner.vllm_config.model_config.max_model_len = 4101 * 128
    runner_backend.initialize(runner_backend.runner, runner_backend.make_config(packed))
    padded_blocks = -(-4101 // 82) * 82
    assert padded_blocks == 4182
    for group in runner_backend.runner.attn_groups:
        workspace = group[0]._mixed_indexer_metadata
        assert workspace.max_blocks == padded_blocks
        canonical_table = torch.full((1, padded_blocks), 82, dtype=torch.int32)
        canonical_slots = torch.tensor([82 * 128 + 7, -1])
        physical_table, physical_slots = workspace.update(canonical_table, canonical_slots)
        assert physical_table.tolist() == [[164] * padded_blocks]
        assert physical_slots.tolist() == [164 * 128 + 7, -1]


@pytest.mark.parametrize("prefill", [True, False])
def test_ordinary_runner_keeps_four_argument_dispatch(runner_backend, bf16_main, prefill):
    runner_backend.runner.layerwise_prefill_p_node = prefill
    runner_backend.backends["main.1"] = runner_backend.other_backend
    runner_backend.initialize(runner_backend.runner, runner_backend.make_config(bf16_main))
    runner_backend.factory.assert_not_called()
    assert runner_backend.runner.attn_groups[0][0].args[:2] == (bf16_main, ["main.1"])


def test_packed_component_rejects_incompatible_constructor(runner_backend, packed):
    runner_backend.backends["main.1"] = runner_backend.other_backend
    with pytest.raises(ValueError, match="compatible SFA metadata builder"):
        runner_backend.initialize(runner_backend.runner, runner_backend.make_config(packed))
    assert not runner_backend.runner.attn_groups


def test_ordinary_cp_runner_keeps_existing_constructor(runner_backend, bf16_main, cp_builder):
    runner_backend.backends["main.1"] = runner_backend.make_backend("sfa_cp", cp_builder)
    runner_backend.initialize(runner_backend.runner, runner_backend.make_config(bf16_main))
    runner_backend.factory.assert_not_called()
    builder = runner_backend.runner.attn_groups[0][0]
    assert type(builder) is cp_builder
    assert builder.pcp_size == 2
    assert builder._mixed_indexer_metadata.max_blocks == 4104
    assert builder.block_arange_buffer.tolist() == [0, 1]


def test_packed_component_rejects_cp_constructor(runner_backend, packed, cp_builder):
    runner_backend.backends["main.1"] = runner_backend.make_backend("sfa_cp", cp_builder)
    with pytest.raises(ValueError, match="compatible SFA metadata builder"):
        runner_backend.initialize(runner_backend.runner, runner_backend.make_config(packed))
    assert not runner_backend.runner.attn_groups


def test_production_startup_rejects_before_building_metadata_or_allocating(initialize, runner_backend, packed):
    runner = runner_backend.runner
    runner.vllm_config = None
    runner._validate_dsa_compact_startup_policy = Mock()
    runner.initialize_attn_backend = Mock()
    runner.initialize_kv_cache_tensors = Mock()
    with pytest.raises(ValueError, match="not enabled in the production runner"):
        initialize(runner, runner_backend.make_config(packed))
    runner._validate_dsa_compact_startup_policy.assert_not_called()
    runner.initialize_attn_backend.assert_not_called()
    runner.initialize_kv_cache_tensors.assert_not_called()
