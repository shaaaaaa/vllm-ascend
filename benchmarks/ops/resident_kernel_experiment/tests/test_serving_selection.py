"""Startup selector tests without vLLM/CANN imports."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

MEMBERSHIP_ENV = "VLLM_ASCEND_DSA_RESIDENT_QUERY_MEMBERSHIP"
MEMBERSHIP_OPS = ("npu_dsa_resident_membership_union_", "npu_dsa_resident_membership_plan_")


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
    monkeypatch.delenv(MEMBERSHIP_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(name, value)
    configure = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels",
                        configure, raising=False)
    planner = load_planner()
    assert planner.configure_resident_kernels() is False
    monkeypatch.setenv(name, "0" if expected else "1")
    planner.configure_resident_kernels()
    configure.assert_called_once_with(expected)


@pytest.mark.parametrize("exact", ["0", "1"])
def test_membership_selection_is_startup_static(monkeypatch, exact):
    monkeypatch.setenv(MEMBERSHIP_ENV, "1")
    monkeypatch.setenv("VLLM_ASCEND_DSA_RESIDENT_EXACT_KERNELS", exact)
    configure = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels", configure, raising=False)
    for name in MEMBERSHIP_OPS:
        monkeypatch.setattr(torch.ops._C_ascend, name, Mock(), raising=False)
    planner = load_planner()
    assert planner.configure_resident_kernels() is True
    monkeypatch.setenv(MEMBERSHIP_ENV, "0")
    assert planner.configure_resident_kernels() is True
    configure.assert_called_once_with(exact == "1")


@pytest.mark.parametrize("missing", MEMBERSHIP_OPS)
def test_membership_requires_both_native_ops_before_capture(monkeypatch, missing):
    monkeypatch.setenv(MEMBERSHIP_ENV, "1")
    configure = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels", configure, raising=False)
    for name in MEMBERSHIP_OPS:
        if name == missing:
            monkeypatch.delattr(torch.ops._C_ascend, name, raising=False)
        else:
            monkeypatch.setattr(torch.ops._C_ascend, name, Mock(), raising=False)
    with pytest.raises(RuntimeError, match="Rebuild.*query membership"):
        load_planner().configure_resident_kernels()
    configure.assert_not_called()


@pytest.mark.parametrize("membership", [False, True])
def test_serving_wrappers_dispatch_matching_ops_and_keep_split_remap(monkeypatch, membership):
    planner = load_planner()
    calls = []
    for name in (
        "npu_dsa_resident_sharded_union_", "npu_dsa_resident_sorted_plan_",
        "npu_dsa_resident_sorted_plan_no_remap_", "npu_dsa_resident_sorted_remap_",
        *MEMBERSHIP_OPS,
    ):
        monkeypatch.setattr(
            torch.ops._C_ascend, name, lambda *args, name=name: calls.append((name, args)), raising=False
        )
    workspace = planner.allocate_sorted_resident_workspace(1, 2, device=torch.device("cpu"))
    state = planner.allocate_sorted_resident_state(1, 1, 2, device=torch.device("cpu"))
    topk = torch.empty((2, 1, planner.INDEX_TOPK), dtype=torch.int32)
    boundary = torch.tensor([4096, 4097], dtype=torch.int32)
    rows = torch.zeros(2, dtype=torch.int32)
    request_states = torch.zeros(1, dtype=torch.int32)
    generations = torch.ones(1, dtype=torch.int64)
    table = torch.arange(32, dtype=torch.int32).view(1, 32)
    planner.prepare_resident_sharded_union_(
        topk, boundary, rows, request_states, generations, state, workspace, mtp=2, query_membership=membership,
    )
    planner.prepare_sorted_resident_cache_fused_(
        topk, table, request_states, generations, state, workspace, block_size=128, query_membership=membership,
    )
    expected = MEMBERSHIP_OPS if membership else ("npu_dsa_resident_sharded_union_", "npu_dsa_resident_sorted_plan_")
    assert tuple(name for name, _ in calls) == expected
    assert calls[0][1][5] is calls[1][1][3] is workspace.shard_counts
    calls.clear()
    # The public split path still receives position ranks and its old operators.
    planner.prepare_resident_sharded_union_(
        topk, boundary, rows, request_states, generations, state, workspace, mtp=2,
    )
    planner.prepare_sorted_resident_cache_(
        topk, table, request_states, generations, state, workspace, block_size=128,
    )
    assert tuple(name for name, _ in calls) == (
        "npu_dsa_resident_sharded_union_", "npu_dsa_resident_sorted_plan_no_remap_", "npu_dsa_resident_sorted_remap_",
    )


def test_missing_extension_fails_before_capture(monkeypatch):
    monkeypatch.delenv(MEMBERSHIP_ENV, raising=False)
    monkeypatch.delattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels",
                        raising=False)
    with pytest.raises(RuntimeError, match="Rebuild"):
        load_planner().configure_resident_kernels()


def test_configuration_failure_is_not_cached(monkeypatch):
    monkeypatch.delenv(MEMBERSHIP_ENV, raising=False)
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
    begin = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr == "_resident_query_membership" for t in n.targets))
    end = next(i for i in range(begin, len(init.body))
               if isinstance(init.body[i], ast.If) and any(
                   isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                   and call.func.attr == "initialize_sorted_resident_cache"
                   for call in ast.walk(init.body[i])))
    events = []

    def configure():
        events.append("configure")
        return True

    impl = SimpleNamespace(
        dsa_resident_cache=resident,
        index_cache_enabled=True,
        layer_name="model.layers.0",
        initialize_sorted_resident_cache=lambda: events.append("allocate"),
    )
    ns = dict(self=impl, configure_resident_kernels=configure,
              envs=SimpleNamespace(VLLM_ASCEND_SFA_SHARED_RESIDENT_PLAN=shared),
              hf_text_config=SimpleNamespace(num_hidden_layers=2), hf_config=None,
              parse_layer_idx=lambda _: 0)
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(init.body[begin:end + 1])
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    assert events == (["configure"] if resident else []) + ([] if shared else ["allocate"])
    assert impl.shared_resident_candidate == shared
    assert impl._resident_query_membership is resident
