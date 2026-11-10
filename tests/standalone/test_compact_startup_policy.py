# SPDX-License-Identifier: Apache-2.0
"""Execute real startup capability probes without constructing services."""
import ast
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

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


@pytest.fixture
def decoder_dcp(policy, monkeypatch):
    config, vllm = policy.config, policy.vllm
    config.decoder_storage_mode, config.enable_async_loading = "process", True
    config.local_cpu = config.shared_cpu_cache_strict = True
    config.decode_preemption_checkpoint = False
    config.chunk_size = 1024
    config.get_extra_config_value = config.extra_config.get
    vllm.kv_transfer_config.kv_connector = "LMCacheAscendConnectorV1Dynamic"
    vllm.kv_transfer_config.kv_role = "kv_consumer"
    vllm.kv_transfer_config.is_kv_consumer = True
    vllm.num_speculative_tokens = 0
    vllm.cache_config.block_size = 128
    vars(vllm.model_config).update(use_mla=True, enforce_eager=True, dtype=torch.bfloat16, seed=7)
    vars(vllm.model_config.hf_text_config).update(kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128)
    parallel = vllm.parallel_config
    vars(parallel).update(
        decode_context_parallel_size=4, tensor_parallel_size=4, local_world_size=4,
        cp_kv_cache_interleave_size=128, enable_dbo=False, enable_elastic_ep=False,
        data_parallel_size=1, distributed_executor_backend="mp",
    )
    vllm.additional_config = {}
    ops = NS(npu_sparse_flash_attention_dcp=True, npu_dsa_resident_dcp_union_=True, npu_dsa_resident_dcp_plan_=True)
    ns = policy.ns
    envs = NS(
        VLLM_ASCEND_DSA_UNBUNDLE=True, VLLM_ASCEND_DSA_TWO_GROUPS=True,
        VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=False, VLLM_ASCEND_SFA_ASYNC_MTP_PREP=False,
        VLLM_ASCEND_SFA_FULL_GRAPH=False,
        VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP=False, VLLM_ASCEND_SFA_INDEPENDENT_DP=False,
        VLLM_ASCEND_DSA_SHARED_POOL=True, VLLM_ASCEND_DSA_SHRINK_LATENT=2,
        VLLM_ASCEND_DSA_RESIDENT_CACHE=True, VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE=False,
        VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=False, VLLM_ASCEND_DSA_USE_ADAPTER_CACHE=False,
    )
    ns.update(
        envs_ascend=envs, INDEX_TOPK=2048, AscendDeviceType=NS(A2="A2"), CUDAGraphMode=NS(NONE=None),
        staged_sfa_graph_configured=lambda _: False,
        get_ascend_device_type=lambda: "A2", enable_sp=lambda _: False, enable_custom_op=lambda: True,
        torch=NS(float16=torch.float16, bfloat16=torch.bfloat16, device=str, ops=NS(_C_ascend=ops),
                 npu=NS(set_device=Mock(), empty_cache=Mock())),
        gc=NS(collect=Mock()), MemorySnapshot=lambda: NS(total_memory=100, free_memory=100),
        get_ascend_config=lambda: NS(enable_cpu_binding=False),
        set_random_seed=Mock(), init_device_properties_triton=Mock(),
    )
    definitions(ROOT / "vllm_ascend/worker/worker.py", {"_configure_dsa_decoder_dcp", "_init_device"},
                ns, "NPUWorker", "Worker")
    worker = ns["Worker"]()
    worker.vllm_config, worker.parallel_config, worker.model_config = vllm, parallel, vllm.model_config
    worker.cache_config, worker.cache_dtype = vllm.cache_config, torch.bfloat16
    worker.cache_config.gpu_memory_utilization = 0.9
    worker.speculative_config, worker.use_v2_model_runner, worker.local_rank = None, False, 0
    worker.compilation_config = NS(cudagraph_mode=None)
    worker._init_worker_distributed_environment = lambda: policy.events.append("distributed")
    triton = ModuleType("vllm.triton_utils")
    triton.HAS_TRITON = False
    monkeypatch.setitem(sys.modules, triton.__name__, triton)
    bootstrap = ModuleType("lmcache.v1.rpc.storage_bootstrap")

    def start_storage(owner):
        assert owner.vllm_config._dsa_decoder_dcp_policy == (2048, 0)
        policy.events.append("storage")

    bootstrap.early_bootstrap_decoder_storage = start_storage
    monkeypatch.setitem(sys.modules, bootstrap.__name__, bootstrap)
    policy.worker, policy.envs, policy.ops = worker, envs, ops
    return policy


def test_decoder_dcp_selects_split_backend_before_storage_and_rechecks_live_policy(decoder_dcp, monkeypatch):
    ctx = decoder_dcp
    ctx.worker._init_device()
    assert ctx.events == ["class", "config", "storage", "distributed"]
    assert ctx.runner._get_dsa_compact_startup_policy() == (2048, 0)
    ctx.runner.dcp_size = 4
    config = NS(kv_cache_groups=[
        NS(kv_cache_spec=NS(dsa_compact_startup_scratch_tokens=2048,
                            dsa_compact_startup_dense_tokens=0, dsa_dcp_size=4)),
        NS(kv_cache_spec=NS(dsa_dcp_size=1)),
    ])
    engine = ctx.connector._lmcache_engine
    engine._dsa_scratch_capacity = 2048
    with pytest.raises(RuntimeError, match="disagrees"):
        ctx.runner._validate_dsa_compact_startup_policy(config)
    engine._dsa_dcp_size = 4
    ctx.runner._validate_dsa_compact_startup_policy(config)
    config.kv_cache_groups[1].kv_cache_spec.dsa_dcp_size = 4
    with pytest.raises(RuntimeError, match="startup layout"):
        ctx.runner._validate_dsa_compact_startup_policy(config)

    cp = ModuleType("vllm_ascend.attention.context_parallel.sfa_cp")
    cp.AscendSFACPMetadataBuilder, cp.AscendSFACPImpl = object(), object()
    monkeypatch.setitem(sys.modules, cp.__name__, cp)
    builder, impl = object(), object()
    ctx.ns.update(
        enable_cp=lambda: True, get_current_vllm_config=lambda: ctx.vllm,
        AscendSFAMetadataBuilder=builder, AscendSFAImpl=impl,
    )
    definitions(ROOT / "vllm_ascend/attention/sfa_v1.py", {"get_builder_cls", "get_impl_cls"},
                ctx.ns, "AscendSFABackend", "Backend")
    backend = ctx.ns["Backend"]
    assert backend.get_builder_cls() is builder
    assert backend.get_impl_cls() is impl
    del ctx.vllm._dsa_decoder_dcp_policy
    assert backend.get_builder_cls() is cp.AscendSFACPMetadataBuilder
    assert backend.get_impl_cls() is cp.AscendSFACPImpl


@pytest.mark.parametrize(("owner", "field", "value", "message"), [
    ("parallel_config", "tensor_parallel_size", 8, "local DCP=TP"),
    ("parallel_config", "cp_kv_cache_interleave_size", 64, "128-token interleave"),
    ("envs", "VLLM_ASCEND_SFA_ASYNC_MTP_PREP", True, "requires full SFA graphs"),
    ("envs", "VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP", True, "without retrieval overlap"),
    ("envs", "VLLM_ASCEND_SFA_INDEPENDENT_DP", True, "coordinated"),
    ("config", "decode_preemption_checkpoint", True, "cooperative"),
    ("config", "decoder_storage_mode", "local", "async process storage"),
    ("config", "enable_async_loading", False, "async process storage"),
    ("ops", "npu_sparse_flash_attention_dcp", None, "Rebuild vllm-ascend"),
])
def test_decoder_dcp_refuses_incompatible_routes_before_storage(decoder_dcp, owner, field, value, message):
    ctx = decoder_dcp
    target = ctx.vllm.parallel_config if owner == "parallel_config" else getattr(ctx, owner)
    if owner == "ops":
        delattr(target, field)
    else:
        setattr(target, field, value)
    with pytest.raises(ValueError, match=message):
        ctx.worker._init_device()
    assert "storage" not in ctx.events and "distributed" not in ctx.events
    assert not hasattr(ctx.vllm, "_dsa_decoder_dcp_policy")


def test_decoder_dcp_requires_effective_shared_cpu_strictness(decoder_dcp):
    ctx = decoder_dcp
    ctx.config.extra_config["shared_cpu_cache_strict"] = False
    with pytest.raises(ValueError, match="strict shared CPU source"):
        ctx.worker._init_device()
    assert "storage" not in ctx.events


@pytest.mark.parametrize("variant", ["full", "full_async", "staged_only", "other_cp", "other_connector"])
def test_decoder_dcp_full_graph_policy_is_available_before_worker_selection(decoder_dcp, monkeypatch, variant):
    ctx = decoder_dcp
    config = ModuleType("vllm.config")
    config.CUDAGraphMode = ctx.ns["CUDAGraphMode"]
    config.CUDAGraphMode.PIECEWISE = "piecewise"
    monkeypatch.setitem(sys.modules, config.__name__, config)
    ctx.vllm.compilation_config = ctx.worker.compilation_config
    ctx.vllm.compilation_config.cudagraph_mode = "piecewise"
    ctx.vllm.model_config.enforce_eager = False
    ctx.vllm.speculative_config = None
    ctx.envs.VLLM_ASCEND_SFA_STAGED_GRAPH = True
    ctx.envs.VLLM_ASCEND_SFA_FULL_GRAPH = variant != "staged_only"
    ctx.ns["StagedSFAConfigReason"] = NS(CONTEXT_PARALLEL="cp")
    definitions(ROOT / "vllm_ascend/utils.py",
                {"staged_sfa_graph_configuration_reasons", "staged_sfa_graph_configured"}, ctx.ns)
    if variant == "other_cp":
        ctx.vllm.parallel_config.prefill_context_parallel_size = 2
    if variant == "other_connector":
        ctx.vllm.kv_transfer_config.kv_connector = "DifferentConnector"
    if variant == "full_async":
        ctx.envs.VLLM_ASCEND_SFA_ASYNC_MTP_PREP = True
        ctx.vllm.num_speculative_tokens = 1
        ctx.vllm.model_config.uses_mrope, ctx.vllm.model_config.uses_xdrope_dim = False, 0
        ctx.worker.speculative_config = ctx.vllm.speculative_config = NS(
            method="mtp", num_speculative_tokens=1, draft_tensor_parallel_size=4, parallel_drafting=False,
            disable_padded_drafter_batch=False, draft_model_config=ctx.vllm.model_config,
        )
    eligible = variant in ("full", "full_async")
    assert ctx.ns["staged_sfa_graph_configured"](ctx.vllm) is eligible
    if eligible:
        ctx.worker._configure_dsa_decoder_dcp()
        assert ctx.vllm._dsa_decoder_dcp_policy == (2048 * (1 + ctx.vllm.num_speculative_tokens), 0)


@pytest.mark.parametrize("draft_tp,draft_count,padded", [(4, 1, True), (2, 1, True), (4, 2, True), (4, 1, False)])
def test_decoder_dcp_uses_same_tp_and_two_row_scratch_for_eager_mtp(decoder_dcp, draft_tp, draft_count, padded):
    ctx = decoder_dcp
    ctx.vllm.model_config.uses_mrope = False
    ctx.vllm.model_config.uses_xdrope_dim = 0
    ctx.worker.speculative_config = NS(
        method="mtp", num_speculative_tokens=draft_count, draft_tensor_parallel_size=draft_tp,
        parallel_drafting=False, disable_padded_drafter_batch=not padded, draft_model_config=ctx.vllm.model_config,
    )
    ctx.vllm.num_speculative_tokens = draft_count
    if (draft_tp, draft_count, padded) == (4, 1, True):
        ctx.worker._configure_dsa_decoder_dcp()
        assert ctx.vllm._dsa_decoder_dcp_policy == (4096, 0)
    else:
        with pytest.raises(ValueError, match="one padded MTP draft token"):
            ctx.worker._configure_dsa_decoder_dcp()


@pytest.mark.parametrize("marked", [False, True])
def test_decoder_dcp_charges_measured_attention_peak_before_kv_sizing(decoder_dcp, monkeypatch, marked):
    ctx = decoder_dcp
    worker = ctx.worker
    if marked:
        worker._configure_dsa_decoder_dcp()
    calls = []
    sync = Mock()
    group = object()
    npu = ModuleType("vllm_ascend.ops.triton.sfa_dcp")

    def execute(query, rope, cache, indices, counts, table, ends, lengths, scale, actual_group):
        assert query.shape == (16, 16, 512) and rope.shape == (16, 16, 64)
        assert [part.shape for part in cache] == [(1, 128, 1, 512), (1, 128, 1, 64)]
        assert indices.shape == (16, 1, 2048) and indices.eq(-1).all()
        assert counts.shape == (16, 16) and not counts.any()
        assert table.tolist() == [[0]] and ends.tolist() == [16] and lengths.tolist() == [0]
        assert actual_group is group
        calls.append("sfa")

    npu.execute_sfa_dcp = execute
    monkeypatch.setitem(sys.modules, npu.__name__, npu)
    worker.init_snapshot = NS(free_memory=9500)
    worker.model_runner = NS(model_memory_usage=1200, profile_run=lambda: calls.append("model"))
    worker.requested_memory = 9000
    worker.device = torch.device("cpu")
    worker.scheduler_config = NS(max_num_batched_tokens=16)
    worker.model_config.get_num_attention_heads = lambda parallel: 16
    after_model = NS(free_memory=6900)

    @contextmanager
    def profile(baseline, weights_memory=0):
        if baseline is worker.init_snapshot:
            assert weights_memory == 1200
            yield NS(after_profile=after_model, non_kv_cache_memory=2900)
        else:
            assert baseline is after_model and weights_memory == 0
            yield NS(after_profile=NS(free_memory=6000), torch_peak_increase=700, non_torch_increase=200)

    ctx.ns.update(
        torch=NS(empty=torch.empty, full=torch.full, int32=torch.int32,
                 npu=NS(synchronize=sync), inference_mode=torch.inference_mode),
        startup_stage=lambda _: lambda fn: fn, staged_sfa_graph_configured=lambda _: False,
        GiB_bytes=1 << 30, memory_profiling=profile, logger=Mock(),
        RESIDENT_COUNT_CACHELINE_INTS=16, get_dcp_group=lambda: group,
    )
    definitions(ROOT / "vllm_ascend/worker/worker.py",
                {"_profile_dsa_decoder_dcp", "determine_available_memory"},
                ctx.ns, "NPUWorker", "Profiler")
    profiler = ctx.ns["Profiler"]()
    vars(profiler).update(vars(worker))
    assert profiler.determine_available_memory() == (5200 if marked else 6100)
    assert calls == (["model", "sfa"] if marked else ["model"])
    assert sync.call_count == int(marked)
