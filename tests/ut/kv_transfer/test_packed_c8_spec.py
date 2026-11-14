# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual spec classes and Runner entry on a CPU-only host."""

import ast
import runpy
from dataclasses import dataclass, replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.selected_prefill_policy import packed_main_c8_enabled

ROOT = Path(__file__).resolve().parents[3]


def _load_definitions(source: Path, names: set[str], namespace: dict) -> None:
    """Load real definitions without importing the NPU backend or applying patches."""
    nodes = [node for node in ast.parse(source.read_text()).body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == names
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(nodes)
    exec(compile(tree, str(source), "exec"), namespace)


@pytest.fixture(scope="module")
def vllm_root():
    vllm = find_spec("vllm")
    assert vllm is not None and vllm.origin, "Put the companion vLLM checkout on PYTHONPATH."
    return Path(vllm.origin).parent


@pytest.fixture(scope="module")
def api(vllm_root):
    namespace = {
        "dataclass": dataclass,
        "torch": torch,
        "get_dtype_size": lambda dtype: dtype.itemsize,
        "packed_main_c8_enabled": packed_main_c8_enabled,
    }
    _load_definitions(
        vllm_root / "v1/kv_cache_interface.py",
        {"KVCacheSpec", "AttentionSpec", "FullAttentionSpec", "MLAAttentionSpec", "UniformTypeKVCacheSpecs"},
        namespace,
    )
    _load_definitions(
        ROOT / "vllm_ascend/patch/platform/patch_kv_cache_interface.py", {"AscendMLAAttentionSpec"}, namespace
    )
    return SimpleNamespace(**namespace)


@pytest.fixture
def packed(api):
    return api.AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=656,
        dtype=torch.int8,
        sparse_head_dim=(512, 64),
        cache_sparse_sfa_c8=True,
    )


def test_packed_main_page_and_merge(api, packed):
    assert packed.page_size_bytes == 83_968
    assert packed.indexer_scale_page_size_bytes == 0
    assert api.AscendMLAAttentionSpec.merge([packed, replace(packed)]) == packed
    with pytest.raises(ValueError, match="no split KV cache ratio"):
        _ = packed.sparse_kv_cache_ratio


@pytest.fixture
def bf16_main(api):
    return api.AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, sparse_head_dim=(512, 64)
    )


@pytest.fixture
def indexer(api):
    return api.AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sparse_head_dim=(128,),
        cache_sparse_c8=True,
    )


def test_existing_main_and_indexer_pages(api, bf16_main, indexer):
    assert not bf16_main.cache_sparse_sfa_c8
    assert bf16_main.page_size_bytes == 147_456
    assert indexer.page_size_bytes == 16_640
    assert indexer.indexer_scale_page_size_bytes == 256
    assert api.AscendMLAAttentionSpec.merge([bf16_main, bf16_main]) == bf16_main
    assert api.AscendMLAAttentionSpec.merge([indexer, indexer]) == indexer


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("block_size", 64),
        ("num_kv_heads", 2),
        ("head_size", 576),
        ("dtype", torch.bfloat16),
        ("sparse_head_dim", (512, 64, 128)),
        ("page_size_padded", 83_968),
        ("cache_sparse_c8", True),
        ("indexer_c8_layer_names", ("layer",)),
        ("indexer_paired_banks", True),
        ("dsa_compact_startup_scratch_tokens", 128),
        ("dsa_compact_startup_dense_tokens", 128),
    ],
)
def test_rejects_invalid_packed_spec(api, packed, field, value):
    spec = replace(packed, **{field: value})
    with pytest.raises(ValueError, match="Packed main C8"):
        _ = spec.page_size_bytes
    with pytest.raises(ValueError, match="Packed main C8"):
        api.AscendMLAAttentionSpec.merge([spec])


def test_merge_preserves_format_distinction(api, packed):
    ordinary = replace(packed, cache_sparse_sfa_c8=False)
    with pytest.raises(AssertionError, match="same packed main C8 format"):
        api.AscendMLAAttentionSpec.merge([ordinary, packed])


def test_merge_preserves_indexer_precision_error(api, indexer):
    ordinary = replace(indexer, cache_sparse_c8=False)
    with pytest.raises(AssertionError, match="share one indexer precision policy"):
        api.AscendMLAAttentionSpec.merge([ordinary, indexer])


@pytest.mark.parametrize("marker", [1, None, "false"])
def test_rejects_non_boolean_packed_marker(api, packed, marker):
    spec = replace(packed, cache_sparse_sfa_c8=marker)
    with pytest.raises(ValueError, match="marker must be a boolean"):
        _ = spec.page_size_bytes
    with pytest.raises(ValueError, match="marker must be a boolean"):
        _ = spec.sparse_kv_cache_ratio
    with pytest.raises(ValueError, match="marker must be a boolean"):
        api.AscendMLAAttentionSpec.merge([packed, spec])


@pytest.fixture(scope="module")
def packed_shared_pool(vllm_root):
    source = vllm_root / "v1/core/dsa_shared_pool.py"
    skip_reason = "Installed vLLM does not provide the dsa_packed_main_c8_prefill_layout prototype."
    try:
        source.stat()
    except FileNotFoundError:
        pytest.skip(skip_reason)
    shared_pool = runpy.run_path(str(source))
    if "dsa_packed_main_c8_prefill_layout" not in shared_pool:
        pytest.skip(skip_reason)
    return shared_pool


def test_shared_pool_factory_uses_real_specs(packed_shared_pool, api, packed, indexer):
    factory = packed_shared_pool["dsa_shared_block_layout"]
    main_spec = api.AscendMLAAttentionSpec.merge([packed, packed])
    indexer_spec = api.AscendMLAAttentionSpec.merge([indexer, indexer])
    layout = factory(main_spec, indexer_spec, prefill_child=True)

    assert isinstance(layout, packed_shared_pool["DSAPackedMainC8PrefillLayout"])
    assert layout.latent_page_size_bytes == main_spec.page_size_bytes == 83_968
    assert layout.indexer_page_size_bytes == 32_768
    assert layout.indexer_scale_page_size_bytes == 512
    assert layout.bundle_page_size_bytes == 2_686_976

    mixed_indexer = replace(indexer, indexer_c8_layer_names=("indexer.2", "indexer.0"))
    mixed_spec = api.AscendMLAAttentionSpec.merge([mixed_indexer, mixed_indexer])
    assert mixed_spec.indexer_scale_layer_count == 2
    assert mixed_spec.page_size_bytes == 33_280
    assert mixed_spec.indexer_scale_page_size_bytes == 512

    mixed_layout = factory(main_spec, mixed_spec, prefill_child=True)
    assert mixed_layout == layout
    assert mixed_layout.indexer_blocks_per_bundle == 82
    assert mixed_layout.indexer_scale_layer_count == 1
    assert mixed_layout.scale_bytes_per_bundle == 82 * 512
    assert mixed_layout.allocation_bytes_per_bundle(4, 3) == 4 * 2_686_976 + 82 * 512


@pytest.mark.parametrize(
    ("owners", "paired", "error"),
    [
        ((), False, "Invalid packed main C8 mixed indexer owner declaration"),
        (("indexer.0", "indexer.0"), False, "Invalid packed main C8 mixed indexer owner declaration"),
        (("indexer.0", "indexer.2"), True, "unpadded, unpaired prefill specs"),
    ],
)
def test_shared_pool_factory_rejects_invalid_mixed_specs(packed_shared_pool, packed, indexer, owners, paired, error):
    mixed_indexer = replace(indexer, indexer_c8_layer_names=owners, indexer_paired_banks=paired)
    with pytest.raises(ValueError, match=error):
        packed_shared_pool["dsa_shared_block_layout"](packed, mixed_indexer, prefill_child=True)


@pytest.fixture
def initialize(api):
    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    runner_class = next(
        node for node in ast.parse(source.read_text()).body if getattr(node, "name", None) == "NPUModelRunner"
    )
    method = next(node for node in runner_class.body if getattr(node, "name", None) == "initialize_kv_cache")
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(method)
    namespace = {
        "staged_sfa_graph_configured": lambda config: False,
        "UniformTypeKVCacheSpecs": api.UniformTypeKVCacheSpecs,
        "packed_main_c8_enabled": packed_main_c8_enabled,
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["initialize_kv_cache"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_runner_rejects_packed_before_initialization(api, initialize, packed, wrapped):
    spec = api.UniformTypeKVCacheSpecs(128, {"main": packed}) if wrapped else packed
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
    runner = SimpleNamespace(vllm_config=None, _validate_dsa_compact_startup_policy=Mock())
    with pytest.raises(ValueError, match="Packed main C8 KV cache"):
        initialize(runner, config)
    runner._validate_dsa_compact_startup_policy.assert_not_called()


@pytest.mark.parametrize("marker", [1, None, "false"])
def test_runner_rejects_non_boolean_marker(api, initialize, packed, marker):
    spec = api.UniformTypeKVCacheSpecs(128, {"main": replace(packed, cache_sparse_sfa_c8=marker)})
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
    runner = SimpleNamespace(vllm_config=None, _validate_dsa_compact_startup_policy=Mock())
    with pytest.raises(ValueError, match="marker must be a boolean"):
        initialize(runner, config)
    runner._validate_dsa_compact_startup_policy.assert_not_called()


def test_runner_continues_with_ordinary_spec(initialize, bf16_main):
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=bf16_main)])
    validation = Mock(side_effect=RuntimeError("native startup policy reached"))
    runner = SimpleNamespace(vllm_config=None, _validate_dsa_compact_startup_policy=validation)
    with pytest.raises(RuntimeError, match="native startup policy reached"):
        initialize(runner, config)
    validation.assert_called_once_with(config)
