# SPDX-License-Identifier: Apache-2.0
"""Validate artifact layer selection before allocating the two-group cache."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


@pytest.fixture
def validate(monkeypatch):
    source = Path(__file__).parents[2] / "vllm_ascend/ascend_config.py"
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendConfig"
    )
    ns = {}
    selector = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_c8_layer_mask")
    exec(compile(ast.Module(body=[selector], type_ignores=[]), str(source), "exec"), ns)
    # Isolate the upstream layer-index utility from model/device imports.
    import sys

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.utils",
        SimpleNamespace(extract_layer_index=lambda name: int(name.split("layers.")[1].split(".")[0])),
    )

    def run(description, enabled=True, names=None, resolve=False):
        config = SimpleNamespace(
            enable_sparse_li_c8=enabled,
            vllm_config=SimpleNamespace(quant_config=SimpleNamespace(quant_description=description)),
        )
        config.indexer_c8_layer_mask = MethodType(ns[selector.name], config)
        names = names or ["model.layers.0.self_attn.indexer.k_cache", "model.layers.3.self_attn.indexer.k_cache"]
        return config.indexer_c8_layer_mask(names)

    return run


@pytest.mark.parametrize("description", [None, {}, {"some.weight": "INT8_DYNAMIC"}])
def test_absent_filter_keeps_global_c8_selection(validate, description):
    validate(description)


def test_only_physical_cache_owners_need_c8_and_aliases_match_by_layer(validate):
    validate(
        {
            "other.layers.0.self_attn.indexer.quant_type": "INT8_DYNAMIC",
            "other.layers.3.self_attn.indexer.quant_type": "INT8_DYNAMIC",
            "other.layers.1.self_attn.indexer.quant_type": "FLOAT",
        }
    )


def test_partial_layer_policy_preserved(validate):
    assert validate({"model.layers.0.self_attn.indexer.quant_type": "INT8_DYNAMIC"}) == (True, False)


def test_all_bf16_layer_policy_preserved(validate):
    assert validate({"model.layers.0.self_attn.indexer.quant_type": "FLOAT"}) == (False, False)


def test_weight_annotation_follows_upstream_selection(validate):
    validate(
        {
            "model.layers.0.self_attn.indexer.quant_type": "INT8_DYNAMIC",
            "model.layers.3.self_attn.indexer.wq_b_weight": "W8A8_MXFP8",
        }
    )


def test_disabled_c8_does_not_enforce_artifact_filter(validate):
    validate({"model.layers.0.self_attn.indexer.quant_type": "FLOAT"}, enabled=False)


def test_glm53_mixed_policy_preserves_six_bf16_owners(validate):
    selected = list(range(6, 67, 4))
    owners = [0, 1, 2, *selected, 70, 74, 78]
    description = {f"model.layers.{i}.self_attn.indexer.quant_type": "INT8_DYNAMIC" for i in selected}
    names = [f"model.layers.{i}.self_attn.indexer.k_cache" for i in owners]
    assert validate(description, names=names, resolve=True) == (False,) * 3 + (True,) * 16 + (False,) * 3


@pytest.mark.parametrize(
    "settings,device,error,enabled",
    [
        ({}, "A2", None, False),
        ({"enable_sparse_li_c8": True}, "A2", None, True),
        ({"enable_sparse_c8": True}, "A3", None, True),
        ({"enable_sparse_li_c8": False, "enable_sparse_c8": True}, "A2", ValueError, False),
        ({"enable_sparse_sfa_c8": True}, "A2", NotImplementedError, False),
        ({"enable_sparse_li_c8": True}, "A5", NotImplementedError, False),
        ({"enable_sparse_c8": True}, "A5", None, False),
        ({"enable_sparse_li_c8": True}, "310P", NotImplementedError, False),
        ({"enable_sparse_li_c8": True, "c8_enable_reshape_optim": True}, "A2", NotImplementedError, False),
    ],
)
def test_startup_c8_flags_and_supported_hardware(settings, device, error, enabled):
    source = Path(__file__).parents[2] / "vllm_ascend/ascend_config.py"
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendConfig"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    start = next(
        i
        for i, node in enumerate(init.body)
        if isinstance(node, ast.If)
        and any(isinstance(n, ast.Constant) and n.value == "enable_sparse_sfa_c8" for n in ast.walk(node.test))
    )
    end = next(
        i
        for i, node in enumerate(init.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "enable_sp_by_pass" for t in node.targets)
    )
    obj = SimpleNamespace()
    ns = dict(
        self=obj,
        additional_config=settings,
        use_sparse=True,
        get_ascend_device_type=lambda: device,
        AscendDeviceType=SimpleNamespace(A2="A2", A3="A3", A5="A5"),
    )
    code = compile(ast.Module(body=init.body[start:end], type_ignores=[]), str(source), "exec")
    if error:
        with pytest.raises(error):
            exec(code, ns)
    else:
        exec(code, ns)
        assert obj.enable_sparse_c8 is obj.enable_sparse_li_c8 is enabled


@pytest.mark.parametrize("mask", [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize("shared", [False, True])
def test_runner_startup_uses_resolved_policy(mask, shared):
    source = Path(__file__).parents[2] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_kv_cache_spec")
    selection = next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Attribute)
        and n.test.attr == "use_sparse_c8_indexer"
        and any(
            isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "indexer_c8_layer_mask"
            for x in ast.walk(n)
        )
    )
    names = ["layer.0.indexer", "layer.1.indexer"]
    specs = {name: SimpleNamespace(cache_sparse_c8=True, indexer_c8_layer_names=None) for name in names}
    config = SimpleNamespace(indexer_c8_layer_mask=lambda _: mask)
    runner = SimpleNamespace(
        use_sparse_c8_indexer=True,
        _mixed_indexer_c8_names=None,
        ascend_config=config,
        dsa_shared_pool=shared,
        vllm_config=SimpleNamespace(kv_transfer_config=None),
    )
    exec(
        compile(ast.Module(body=[selection], type_ignores=[]), str(source), "exec"),
        dict(self=runner, indexer_names=names, kv_cache_spec=specs),
    )
    assert runner.use_sparse_c8_indexer == any(mask)
    mixed = any(mask) and not all(mask)
    assert config.indexer_c8_shared_block_factor == (2 if shared and mixed else 1)
    for spec in specs.values():
        assert spec.cache_sparse_c8 == any(mask)
        assert spec.indexer_c8_layer_names == ((names[1],) if mixed else None)
