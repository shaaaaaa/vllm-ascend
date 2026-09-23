# SPDX-License-Identifier: Apache-2.0
"""C8 cache specs and typed shared views; execute production code on CPU."""

import ast
from dataclasses import dataclass
from pathlib import Path

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
