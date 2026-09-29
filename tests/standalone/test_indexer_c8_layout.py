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


@pytest.mark.parametrize("shared", [False, True, "paired", "banked"])
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
    if shared == "banked":
        layout = core.dsa_shared_block_layout(latent_spec, index_spec, 2, prefill_child=True)
        tensors = [NS(shared_by=latent_names + index_names,
                      size=7 * (layout.bundle_page_size_bytes + layout.scale_bytes_per_bundle))]
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
    runner.layerwise_prefill_p_node = shared == "banked"
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
    blocks = 126 if shared == "banked" else 54 if shared == "paired" else 27 if shared else 5
    assert bf16[0].shape[0] == blocks
    assert c8[0].shape[0] == c8[1].shape[0] == blocks * (2 if shared is True or shared == "banked" else 1)
    if shared == "paired":
        assert all(t.shape[0] == 6 and t.is_contiguous() for name in latent_names for t in views[name])
    assert all(len(views[name]) == 2 for name in latent_names)
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for row in raw.values() for t in row}
    assert sum(storages.values()) <= sum(t.size for t in tensors) + (len(storages) * 2 * 1024**2 if aligned or shared == "banked" else 0)
    if shared == "banked":
        assert len(storages) == 1
        assert sum(storages.values()) == tensors[0].size + 2 * 1024**2
        assert views[latent_names[0]] is views[latent_names[2]]
        assert bf16[0].data_ptr() == c8[0].data_ptr()
        # Discontinuous child IDs in two temporal banks preserve null and gaps.
        for block, value in ((18, 11), (72, 23)):
            c8[0][2 * block].fill_(value)
            c8[1][2 * block].fill_(value)
            logical_bytes = bf16[0][block].view(torch.int8).flatten()
            assert torch.all(logical_bytes[:16384] == value)
            assert torch.count_nonzero(logical_bytes[16384:]) == 0
            assert torch.count_nonzero(c8[1][2 * block + 1]) == 0
        assert torch.count_nonzero(bf16[0][0]) == 0
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


def test_prefill_rebind_owns_physical_metadata_for_each_bank(api):
    from copy import copy

    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    cls = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "AscendSFAMetadataBuilder")
    method = next(n for n in cls.body if getattr(n, "name", None) == "rebind_layerwise_prefill_metadata")
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(method)
    states = NS(PrefillNoCache=0, PrefillCacheHit=1, ChunkedPrefill=2)
    ns = dict(copy=copy, AscendAttentionState=states)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    builder = NS(enable_dsa_cp=False,
                 _mixed_indexer_metadata=api["MixedIndexerMetadata"](2, 18, 8, torch.device("cpu")))
    template = NS(attn_state=0, num_decode_tokens=0, indexer_c8_block_table=torch.tensor([999]),
                  indexer_c8_slot_mapping=torch.tensor([999]))
    results = []
    for bank in (18, 72, 18):
        common = NS(num_reqs=1, num_input_tokens=3,
                    block_table_tensor=torch.tensor([[bank]], dtype=torch.int32),
                    slot_mapping=torch.tensor([bank * 128, bank * 128 + 127, -1]),
                    indexer_block_table_tensor=torch.tensor([[bank]], dtype=torch.int32),
                    indexer_slot_mapping=torch.tensor([bank * 128, bank * 128 + 127, -1]))
        metadata = ns[method.name](builder, template, common)
        results.append(metadata)
        assert metadata.indexer_c8_block_table.tolist() == [[bank * 2]]
        assert metadata.indexer_c8_slot_mapping.tolist() == [bank * 256, bank * 256 + 127, -1]
    assert results[0].indexer_c8_block_table.tolist() == [[36]]
    assert results[1].indexer_c8_block_table.tolist() == [[144]]
    assert len({m.indexer_c8_block_table.data_ptr() for m in results}) == 3
    common.indexer_block_table_tensor = None
    cleared = ns[method.name](builder, template, common)
    assert cleared.indexer_c8_block_table is cleared.indexer_c8_slot_mapping is None


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("connector", [False, True])
def test_bf16_prefill_spec_charges_retained_alignment(api, monkeypatch, prefill, connector):
    from types import ModuleType
    from unittest.mock import Mock

    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    cls = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    methods = [n for n in cls.body if getattr(n, "name", None) in {
        "get_kv_cache_spec", "_get_dsa_compact_startup_policy", "_allocate_kv_cache_tensors", "_align_memory"}]
    module = ast.parse("from __future__ import annotations")
    module.body.append(ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods, decorator_list=[]))
    spec_type = api["AscendMLAAttentionSpec"]
    mla_type = type("MLAAttention", (), {})
    layers = {"model.layers.0.self_attn.attn": mla_type(),
              "model.layers.0.self_attn.indexer.k_cache": type("DeepseekV32IndexerCache", (), {})()}
    interface = ModuleType("vllm.v1.kv_cache_interface")
    interface.MLAAttentionSpec = spec_type
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    ns = dict(api, has_ec_transfer=lambda: False, get_layers_from_vllm_config=lambda *_: layers,
              AttentionLayerBase=object, Attention=type("UnusedAttention", (), {}),
              MLAAttention=mla_type, MLAAttentionSpec=spec_type, AttentionSpec=spec_type,
              MambaBase=type("UnusedMamba", (), {}), MambaSpec=type("UnusedMambaSpec", (), {}),
              logger=Mock())
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    runner = ns["Runner"]()
    runner.layerwise_prefill_p_node = prefill
    runner.use_sparse = runner.dsa_unbundle = runner.dsa_shared_pool = True
    runner.use_sparse_c8_indexer = runner.dsa_free_paged = False
    runner.block_size, runner.sparse_head_dim = 128, (512, 64, 128)
    runner.kv_cache_dtype, runner.device = torch.bfloat16, torch.device("cpu")
    runner.vllm_config = NS(cache_config=NS(cache_dtype="auto"),
                            kv_transfer_config=object() if connector else None)
    runner.model_config = NS(hf_text_config=NS(num_hidden_layers=1))
    runner.runner_only_attn_layers = set()
    specs = runner.get_kv_cache_spec()
    index = specs["model.layers.0.self_attn.indexer.k_cache"]
    if not prefill:
        assert index.shared_pool_alignment_bytes == 0  # Keep ordinary/D policy unchanged.
        return
    config = NS(kv_cache_tensors=[NS(size=2 * 589824, shared_by=list(specs))],
                kv_cache_groups=[NS(layer_names=list(specs))])
    runner._get_layer_kv_cache_specs = lambda _: specs
    raw = runner._allocate_kv_cache_tensors(config)
    storage = next(iter(raw.values()))[0].untyped_storage()
    retained_overhead = storage.nbytes() - config.kv_cache_tensors[0].size
    assert retained_overhead == 2 * 1024**2
    assert index.shared_pool_alignment_bytes == retained_overhead


def test_compact_startup_markers_stay_on_latent_specs_and_survive_merge(api):
    from dataclasses import replace

    latent = api["AscendMLAAttentionSpec"](128, 1, 576, torch.bfloat16, sparse_head_dim=(512, 64))
    indexer = api["AscendMLAAttentionSpec"](128, 1, 128, torch.bfloat16, sparse_head_dim=(128,))
    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    method = next(n for n in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
                  if isinstance(n, ast.FunctionDef) and n.name == "get_kv_cache_spec")
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "compact_policy" for t in n.targets))
    ns = dict(self=NS(_get_dsa_compact_startup_policy=lambda: (4096, 8192), sparse_head_dim=(512, 64, 128)),
              kv_cache_spec={"model.layers.0.self_attn.attn": latent,
                             "model.layers.0.self_attn.indexer.k_cache": indexer})
    exec(compile(ast.Module(body=method.body[start:start + 2], type_ignores=[]), str(source), "exec"), ns)
    assert (latent.dsa_compact_startup_scratch_tokens, latent.dsa_compact_startup_dense_tokens) == (4096, 8192)
    assert indexer.dsa_compact_startup_scratch_tokens == indexer.dsa_compact_startup_dense_tokens == 0
    merged = type(latent).merge([latent, replace(latent)])
    assert (merged.dsa_compact_startup_scratch_tokens, merged.dsa_compact_startup_dense_tokens) == (4096, 8192)
    for other in (replace(latent, dsa_compact_startup_scratch_tokens=0),
                  replace(latent, dsa_compact_startup_dense_tokens=0)):
        with pytest.raises(AssertionError):
            type(latent).merge([latent, other])
