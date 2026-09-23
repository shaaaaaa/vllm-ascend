"""Startup selector tests without vLLM/CANN imports."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def load_planner():
    path = ROOT / "vllm_ascend/distributed/kv_transfer/sparse_offload/resident_sorted_cache.py"
    spec = importlib.util.spec_from_file_location("serving_resident_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value,expected", [(None, True), ("0", False), ("1", True)])
def test_startup_selection_and_no_reread(monkeypatch, value, expected):
    name = "VLLM_ASCEND_DSA_RESIDENT_EXACT_KERNELS"
    monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(name, value)
    configure = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels",
                        configure, raising=False)
    planner = load_planner()
    planner.configure_resident_kernels()
    monkeypatch.setenv(name, "0" if expected else "1")
    planner.configure_resident_kernels()
    configure.assert_called_once_with(expected)


def test_missing_extension_fails_before_capture(monkeypatch):
    monkeypatch.delattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels",
                        raising=False)
    with pytest.raises(RuntimeError, match="Rebuild"):
        load_planner().configure_resident_kernels()


def test_configuration_failure_is_not_cached(monkeypatch):
    configure = Mock(side_effect=[RuntimeError("restart"), None])
    monkeypatch.setattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels",
                        configure, raising=False)
    planner = load_planner()
    with pytest.raises(RuntimeError, match="restart"):
        planner.configure_resident_kernels()
    planner.configure_resident_kernels()
    assert configure.call_count == 2


@pytest.mark.parametrize("resident,shared", [(False, False), (False, True), (True, False), (True, True)])
def test_ascend_startup_selection_precedes_deferred_allocation(resident, shared):
    """Exercise the actual merged constructor section, including shared deferral."""
    import ast
    from types import SimpleNamespace

    source = ROOT / "vllm_ascend/attention/sfa_v1.py"
    cls = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    begin = next(i for i, n in enumerate(init.body)
                 if isinstance(n, ast.If) and any(
                     isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                     and call.func.id == "configure_resident_kernels" for call in ast.walk(n)))
    end = next(i for i in range(begin, len(init.body))
               if isinstance(init.body[i], ast.If) and any(
                   isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                   and call.func.attr == "initialize_sorted_resident_cache"
                   for call in ast.walk(init.body[i])))
    events = []
    impl = SimpleNamespace(dsa_resident_cache=resident, index_cache_enabled=True,
                           layer_name="model.layers.0", initialize_sorted_resident_cache=lambda: events.append("allocate"))
    ns = dict(self=impl, configure_resident_kernels=lambda: events.append("configure"),
              envs=SimpleNamespace(VLLM_ASCEND_SFA_SHARED_RESIDENT_PLAN=shared),
              hf_text_config=SimpleNamespace(num_hidden_layers=2), hf_config=None,
              parse_layer_idx=lambda _: 0)
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(init.body[begin:end + 1])
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    assert events == (["configure"] if resident else []) + ([] if shared else ["allocate"])
    assert impl.shared_resident_candidate == shared
