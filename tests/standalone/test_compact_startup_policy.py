# SPDX-License-Identifier: Apache-2.0
"""Execute real startup capability probes without constructing services."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def definitions(path, names, namespace, owner=None, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if owner:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == owner)
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    module = ast.parse("from __future__ import annotations")
    if class_name:
        module.body.append(ast.ClassDef(name=class_name, bases=[], keywords=[], body=nodes, decorator_list=[]))
    else:
        module.body.extend(nodes)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def policy(monkeypatch):
    events = []
    config = NS(pd_role="receiver", dsa_group1_load_mode="persistent_direct_hbm",
                enable_dsa_cold_compact_load=True, enable_sparse_attention=True,
                dsa_two_groups=True, enable_shared_cpu_cache=True, use_layerwise=True,
                extra_config={})
    vllm = NS(kv_transfer_config=NS(kv_role="kv_both", kv_load_failure_policy="fail", kv_connector_extra_config={}),
              cache_config=NS(enable_prefix_caching=False), num_speculative_tokens=1,
              model_config=NS(hf_text_config=NS(index_topk=2048)),
              parallel_config=NS(pipeline_parallel_size=1, prefill_context_parallel_size=1,
                                 decode_context_parallel_size=1))
    validator_ns = dict(json=json, logger=Mock())
    definitions(ROOT.parent / "LMCache-NPU/lmcache/v1/config_base.py",
                {"validate_and_set_config_value"}, validator_ns)
    config_module = ModuleType("lmcache.v1.config_base")
    config_module.validate_and_set_config_value = validator_ns["validate_and_set_config_value"]
    utils = ModuleType("lmcache.integration.vllm.utils")
    utils.lmcache_get_or_create_config = lambda: events.append("config") or config
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    ns = dict(deepcopy=deepcopy, os=os)
    definitions(ROOT.parent / "LMCache-Ascend/lmcache_ascend/integration/vllm/lmcache_ascend_connector_v1.py",
                {"get_dsa_compact_startup_policy", "_compact_policy_from_config", "get_dsa_compact_runtime_policy"},
                ns, "LMCacheAscendConnectorV1Dynamic", "Connector")
    connector_cls = ns["Connector"]
    connector_cls.supports_dsa_compact_external_load = True
    connector_cls.uses_layerwise_model_callbacks = True
    factory = ModuleType("vllm.distributed.kv_transfer.kv_connector.factory")
    factory.KVConnectorFactory = NS(get_connector_class=lambda _: events.append("class") or connector_cls)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    runtime = connector_cls()
    runtime._lmcache_engine = NS(config=config, _dsa_scratch_capacity=4096, _dsa_kv_policy_threshold=0)
    ns.update(has_kv_transfer_group=lambda: True, get_kv_transfer_group=lambda: runtime)
    definitions(ROOT / "vllm_ascend/worker/model_runner_v1.py",
                {"_get_dsa_compact_startup_policy", "_validate_dsa_compact_startup_policy"},
                ns, "NPUModelRunner", "Runner")
    runner = ns["Runner"]()
    runner.vllm_config = vllm
    runner.cache_config, runner.parallel_config = vllm.cache_config, vllm.parallel_config
    runner.use_sparse = runner.dsa_unbundle = runner.dsa_shared_pool = True
    runner.dsa_shrink_latent, runner.layerwise_prefill_p_node = 2, False
    monkeypatch.delenv("LMCACHE_DSA_KV_POLICY_THRESHOLD", raising=False)
    return NS(runner=runner, config=config, vllm=vllm, connector=runtime,
              cls=connector_cls, events=events, ns=ns, factory=factory)


@pytest.mark.parametrize("load_mode", ["persistent_direct_hbm", "p2p_preferred"])
def test_startup_orders_class_import_before_config_and_revalidates(policy, load_mode):
    policy.config.dsa_group1_load_mode = load_mode
    assert policy.runner._get_dsa_compact_startup_policy() == (4096, 0)
    assert policy.events == ["class", "config"]
    config = NS(kv_cache_groups=[NS(kv_cache_spec=NS(dsa_compact_startup_scratch_tokens=4096,
                                                  dsa_compact_startup_dense_tokens=0))])
    policy.runner._validate_dsa_compact_startup_policy(config)
    policy.connector._lmcache_engine._dsa_scratch_capacity = 2048
    with pytest.raises(RuntimeError, match="disagrees"):
        policy.runner._validate_dsa_compact_startup_policy(config)


@pytest.mark.parametrize("field,value", [("pd_role", "sender"), ("use_layerwise", False),
    ("enable_sparse_attention", False), ("dsa_two_groups", False),
    ("enable_shared_cpu_cache", False), ("enable_dsa_cold_compact_load", False),
    ("dsa_group1_load_mode", "unknown")])
def test_config_rejections_and_validated_overrides_do_not_mutate_singleton(policy, field, value):
    before = deepcopy(policy.config.__dict__)
    policy.vllm.kv_transfer_config.kv_connector_extra_config = {"lmcache." + field: value}
    assert policy.runner._get_dsa_compact_startup_policy() is None
    assert policy.config.__dict__ == before


def test_extra_shared_cpu_override_matches_effective_split_contract(policy):
    policy.vllm.kv_transfer_config.kv_connector_extra_config = {
        "lmcache.extra_config": json.dumps({"enable_shared_cpu_cache": False})}
    assert policy.runner._get_dsa_compact_startup_policy() is None
    assert policy.config.extra_config == {}


@pytest.mark.parametrize("field,value", [("layerwise_prefill_p_node", True), ("dsa_shrink_latent", 1),
    ("dsa_shared_pool", False), ("dsa_unbundle", False), ("use_sparse", False)])
def test_noncompact_runners_do_not_load_connector_config(policy, field, value):
    setattr(policy.runner, field, value)
    assert policy.runner._get_dsa_compact_startup_policy() is None
    assert policy.events == []


@pytest.mark.parametrize("field", ["pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"])
def test_parallel_configs_do_not_claim_compact_startup(policy, field):
    setattr(policy.runner.parallel_config, field, 2)
    assert policy.runner._get_dsa_compact_startup_policy() is None
    assert policy.events == []


def test_ordinary_connector_and_prefix_cache_keep_full_requirement(policy):
    policy.factory.KVConnectorFactory.get_connector_class = lambda _: type("Ordinary", (), {})
    assert policy.runner._get_dsa_compact_startup_policy() is None
    policy.vllm.cache_config.enable_prefix_caching = True
    assert policy.runner._get_dsa_compact_startup_policy() is None
    assert policy.events == []


@pytest.mark.parametrize("threshold,expected", [("8192", 8192), ("0", 0), ("-5", 0), ("invalid", 0)])
def test_dense_threshold_uses_existing_adapter_semantics(policy, monkeypatch, threshold, expected):
    monkeypatch.setenv("LMCACHE_DSA_KV_POLICY_THRESHOLD", threshold)
    assert policy.runner._get_dsa_compact_startup_policy() == (4096, expected)


def test_unmarked_live_config_never_looks_up_connector(policy):
    policy.ns["has_kv_transfer_group"] = lambda: (_ for _ in ()).throw(AssertionError("unexpected connector lookup"))
    policy.runner._validate_dsa_compact_startup_policy(NS(kv_cache_groups=[NS(kv_cache_spec=NS())]))


def test_failed_load_cannot_fall_back_to_dense_recomputation(policy):
    policy.vllm.kv_transfer_config.kv_load_failure_policy = "recompute"
    assert policy.runner._get_dsa_compact_startup_policy() is None
    assert policy.cls._compact_policy_from_config(policy.vllm, policy.config) is None


def test_minimal_graph_profiling_defers_live_check_until_real_initialization(policy):
    config = NS(kv_cache_groups=[NS(kv_cache_spec=NS(dsa_compact_startup_scratch_tokens=4096,
                                                  dsa_compact_startup_dense_tokens=0))])
    policy.ns["has_kv_transfer_group"] = lambda: False
    policy.runner._profiling_cudagraph_memory = True
    policy.runner._validate_dsa_compact_startup_policy(config)
    policy.runner._profiling_cudagraph_memory = False
    with pytest.raises(RuntimeError, match="verified active connector"):
        policy.runner._validate_dsa_compact_startup_policy(config)


def test_graph_pool_setup_failure_cannot_leave_profiling_validation_bypass(policy):
    definitions(ROOT / "vllm_ascend/worker/model_runner_v1.py", {"profile_cudagraph_memory"},
                policy.ns, "NPUModelRunner", "Profiler")
    runner = policy.ns["Profiler"]()
    runner.vllm_config = policy.vllm
    runner._profiling_cudagraph_memory = False
    runner._sfa_full_graph = NS(graph_pool=object())
    policy.ns.update(staged_sfa_graph_configured=lambda _: True,
                     ACLGraphWrapper=NS(_all_instances=[]),
                     compilation_counter=NS(num_cudagraph_captured=0),
                     envs_ascend=NS(VLLM_ASCEND_SFA_FULL_GRAPH=True),
                     current_platform=NS(graph_pool_handle=Mock(side_effect=RuntimeError("pool failed"))))
    with pytest.raises(RuntimeError, match="pool failed"):
        runner.profile_cudagraph_memory()
    assert runner._profiling_cudagraph_memory is False
