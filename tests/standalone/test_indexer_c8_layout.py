# SPDX-License-Identifier: Apache-2.0
"""C8 cache specs and typed shared views; execute production code on CPU."""

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def api():
    @dataclass(frozen=True)
    class BaseSpec:
        block_size: int
        num_kv_heads: int
        head_size: int
        dtype: torch.dtype
        cache_dtype_str: str = "auto"

    ns = dict(torch=torch, dataclass=dataclass, MLAAttentionSpec=BaseSpec, get_dtype_size=lambda dtype: dtype.itemsize)
    source = ROOT / "vllm_ascend/patch/platform/patch_kv_cache_interface.py"
    node = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendMLAAttentionSpec"
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(node)
    exec(compile(tree, str(source), "exec"), ns)
    source = ROOT / "vllm_ascend/worker/dsa_shared_pool.py"
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), ns)
    return ns


@pytest.mark.parametrize("c8,expected", [(False, 32768), (True, 16640)])
def test_unbundled_indexer_spec_counts_keys_and_scales(api, c8, expected):
    spec = api["AscendMLAAttentionSpec"](128, 1, 128, torch.bfloat16, sparse_head_dim=(128,), cache_sparse_c8=c8)
    assert spec.page_size_bytes == expected
    assert spec.indexer_scale_page_size_bytes == (256 if c8 else 0)


@pytest.mark.parametrize("c8", [False, True])
def test_shared_views_and_sidecar_use_existing_logical_ids(api, c8):
    slots, latent_page = 4, 147456
    bundle = latent_page if c8 else latent_page * 2
    raw = torch.zeros(slots * bundle, dtype=torch.int8)
    scale = torch.zeros(slots * 9 * 128 * 2, dtype=torch.int8) if c8 else None
    kwargs = dict(indexer_dtype=torch.int8 if c8 else None)
    view = api["reshape_dsa_shared_pool_raw"]
    latent = view(raw, torch.bfloat16, 128, 1, 512, 64, 128, is_indexer=False, **kwargs)
    index = view(raw, torch.bfloat16, 128, 1, 512, 64, 128, is_indexer=True, indexer_scale=scale, **kwargs)
    assert len(index) == (2 if c8 else 1)
    assert index[0].shape == (slots * 9, 128, 1, 128)
    assert index[0].dtype == (torch.int8 if c8 else torch.bfloat16)
    assert sum(t.numel() * t.element_size() for t in latent) == raw.numel()
    assert index[0].data_ptr() == raw.data_ptr()
    if c8:
        assert index[1].dtype == torch.float16
        assert index[1].shape == (slots * 9, 128, 1, 1)
        index[1][9].fill_(1)
        assert torch.count_nonzero(raw) == 0  # Scales never alias shared key/latent storage.
        assert index[1].data_ptr() == scale.data_ptr()


@pytest.mark.parametrize("scale_bytes", [None, 0, 256])
def test_missing_or_short_sidecar_rejected(api, scale_bytes):
    raw = torch.zeros(2 * 147456, dtype=torch.int8)
    scale = None if scale_bytes is None else torch.zeros(scale_bytes, dtype=torch.int8)
    with pytest.raises(ValueError, match="sidecar"):
        api["reshape_dsa_shared_pool_raw"](
            raw,
            torch.bfloat16,
            128,
            1,
            512,
            64,
            128,
            is_indexer=True,
            indexer_dtype=torch.int8,
            indexer_scale=scale,
        )


def test_mixed_pool_uses_common_ownership_with_c8_physical_block_mapping(api):
    slots = 4
    raw = torch.zeros(slots * 294912, dtype=torch.int8)
    scales = torch.zeros(slots * 9 * 2 * 128 * 2, dtype=torch.int8)
    view = api["reshape_dsa_shared_pool_raw"]
    common = dict(shared_indexer_dtype=torch.bfloat16)
    latent = view(raw, torch.bfloat16, 128, 1, 512, 64, 128,
                  is_indexer=False, indexer_dtype=torch.int8, **common)
    keys, scale = view(raw, torch.bfloat16, 128, 1, 512, 64, 128,
                       is_indexer=True, indexer_dtype=torch.int8, indexer_scale=scales, **common)
    assert keys.shape == (slots * 18, 128, 1, 128)
    assert scale.shape == (slots * 18, 128, 1, 1)
    # Bundle 2 belongs to latent; bundle 1 belongs to the indexer. A C8 key
    # block occupies the first half of each common BF16-sized logical page.
    for tensor in latent:
        tensor[4:6].fill_(3)
    logical_indexer_ids = [*range(8, 16), slots * 8 + 1]
    for block in logical_indexer_ids:
        keys[block * 2].fill_(17)
        scale[block * 2].fill_(0.25)
    for tensor in latent:
        assert torch.all(tensor[4:6] == 3)
    for block in logical_indexer_ids:
        assert torch.all(keys[block * 2] == 17)
        assert torch.all(keys[block * 2 + 1] == 0)
        assert torch.all(scale[block * 2] == 0.25)


@pytest.mark.parametrize("shared", [False, True, "paired"])
@pytest.mark.parametrize("aligned", [False, True])
def test_runner_allocates_mixed_indexers_and_shared_consumers(api, monkeypatch, shared, aligned):
    core_path = ROOT.parent / "vllm/vllm/v1/core/dsa_shared_pool.py"
    spec = importlib.util.spec_from_file_location("mixed_core_layout", core_path)
    core = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, core)
    spec.loader.exec_module(core)
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    cls = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    names = {"_allocate_kv_cache_tensors", "_reshape_kv_cache_tensors", "_align_memory"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods, decorator_list=[]))
    ns = dict(api, AttentionSpec=api["MLAAttentionSpec"], MambaSpec=type("Mamba", (), {}),
              dsa_shared_block_layout=core.dsa_shared_block_layout)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    latent_names = [f"model.layers.{i}.self_attn.attn" for i in range(3)]
    index_names = [f"model.layers.{i}.self_attn.indexer.k_cache" for i in range(2)]
    c8_names = (index_names[1],)
    latent_spec = api["AscendMLAAttentionSpec"](128, 1, 576, torch.bfloat16, sparse_head_dim=(512, 64))
    index_spec = api["AscendMLAAttentionSpec"](
        128,
        1,
        128,
        torch.bfloat16,
        sparse_head_dim=(128,),
        cache_sparse_c8=True,
        indexer_c8_layer_names=c8_names,
        indexer_paired_banks=shared == "paired",
    )
    layer_specs = {**dict.fromkeys(latent_names, latent_spec), **dict.fromkeys(index_names, index_spec)}
    group0 = NS(layer_names=latent_names, backend=None)
    group1 = NS(layer_names=index_names, backend=None)
    tensors = []
    for i, name in enumerate(latent_names):
        paired = [name, index_names[i]] if shared and i < 2 else [name]
        size = 3 * 294912 * (2 if shared == "paired" and i == 0 else 1) + (3 * 4608 if shared and i == 1 else 0)
        tensors.append(NS(shared_by=paired, size=size))
    if not shared:
        tensors += [NS(shared_by=[name], size=5 * index_spec.page_size_bytes) for name in index_names]
    config = NS(
        num_blocks=2,
        kv_cache_tensors=tensors,
        kv_cache_groups=[group0, group1],
        dsa_paired_bank_slots=3 if shared == "paired" else 0,
    )
    runner = ns["Runner"]()
    runner.kv_cache_config = config
    runner.dsa_shared_pool, runner.dsa_unbundle, runner.use_sparse = shared, True, True
    runner.use_sparse_c8_indexer = True
    runner._mixed_indexer_c8_names = frozenset(c8_names)
    runner.vllm_config = NS(kv_transfer_config=object() if aligned else None)
    runner.device = torch.device("cpu")
    runner.kv_cache_dtype = torch.bfloat16
    runner.c8_k_cache_dtype, runner.c8_k_scale_cache_dtype = torch.int8, torch.float16
    runner.sparse_head_dim = (512, 64, 128)
    runner.runner_only_attn_layers = set()
    runner._get_layer_kv_cache_specs = lambda _: layer_specs
    runner._kv_cache_spec_attn_group_iterator = lambda: iter((group0, group1))
    raw = runner._allocate_kv_cache_tensors(config)
    views = runner._reshape_kv_cache_tensors(config, raw)
    bf16, c8 = views[index_names[0]], views[index_names[1]]
    assert len(bf16) == 1 and bf16[0].dtype == torch.bfloat16
    assert len(c8) == 2 and [t.dtype for t in c8] == [torch.int8, torch.float16]
    blocks = 54 if shared == "paired" else 27 if shared else 5
    assert bf16[0].shape[0] == blocks
    assert c8[0].shape[0] == c8[1].shape[0] == blocks * (2 if shared is True else 1)
    if shared == "paired":
        assert all(t.shape[0] == 6 and t.is_contiguous() for name in latent_names for t in views[name])
    assert all(len(views[name]) == 2 for name in latent_names)
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for row in raw.values() for t in row}
    assert sum(storages.values()) <= sum(t.size for t in tensors) + (len(storages) * 2 * 1024**2 if aligned else 0)
    if shared == "paired":
        expected_padding = 4 * 2 * 1024**2 if aligned else 0  # two paired slabs, two consumer planes
        assert sum(storages.values()) == sum(t.size for t in tensors) + expected_padding


def test_mixed_indexer_metadata_keeps_stable_contiguous_buffers(api):
    builder = api["MixedIndexerMetadata"](4, 18, 32, torch.device("cpu"))
    original = torch.tensor([[0, 1, 33], [4, 5, 6]], dtype=torch.int32)
    slots = torch.tensor([-1, 0, 1, 127, 128, 129, 255, 256], dtype=torch.int64)
    table, physical = builder.update(original, slots)
    assert torch.equal(table, original * 2)
    assert physical.tolist() == [-1, 0, 1, 127, 256, 257, 383, 512]
    assert table.is_contiguous() and physical.is_contiguous()
    pointers = table.data_ptr(), physical.data_ptr()
    next_table, next_slots = builder.update(original[:, :2], slots[:3])
    assert (next_table.data_ptr(), next_slots.data_ptr()) == pointers
    assert next_table.is_contiguous()
    assert next_slots.tolist() == [-1, 0, 1]
    assert original.tolist() == [[0, 1, 33], [4, 5, 6]]
    with pytest.raises(ValueError, match="capacity"):
        builder.update(torch.empty(5, 1, dtype=torch.int32), slots)


def test_merging_specs_preserves_paired_geometry_and_alignment(api):
    from dataclasses import replace

    spec = api["AscendMLAAttentionSpec"](
        128,
        1,
        128,
        torch.bfloat16,
        sparse_head_dim=(128,),
        cache_sparse_c8=True,
        indexer_c8_layer_names=("c8",),
        indexer_paired_banks=True,
        shared_pool_alignment_bytes=2 * 1024**2,
    )
    merged = type(spec).merge([spec, replace(spec)])
    assert merged.indexer_paired_banks and merged.shared_pool_alignment_bytes == 2 * 1024**2
    assert merged.page_size_bytes == 32768 + 256
    for other in (replace(spec, indexer_paired_banks=False), replace(spec, shared_pool_alignment_bytes=0)):
        with pytest.raises(AssertionError):
            type(spec).merge([spec, other])
