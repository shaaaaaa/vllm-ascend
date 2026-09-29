# SPDX-License-Identifier: Apache-2.0
"""Execute real startup methods with host-only dependency stubs."""

import ast
import copy
import importlib.util
import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def trace(monkeypatch):
    path = ROOT / "vllm_ascend/worker/startup_trace.py"
    spec = importlib.util.spec_from_file_location("pd_startup_integration_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = io.StringIO()
    monkeypatch.setattr(module, "sys", NS(stderr=output, modules={}))
    return module, output


def method(filename, class_name, name, trace, **dependencies):
    path = ROOT / "vllm_ascend/worker" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    target = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, target], type_ignores=[]))
    scope = dict(startup_phase=trace.startup_phase, startup_stage=trace.startup_stage, **dependencies)
    exec(compile(code, str(path), "exec"), scope)
    return scope[name]


def worker():
    parallel = NS(data_parallel_rank=2, tensor_parallel_size=4, data_parallel_size=4)
    parallel.enable_expert_parallel = True
    parallel.enable_elastic_ep = False
    model = NS(enable_sleep_mode=False, is_moe=True)
    config = NS(parallel_config=parallel, model_config=model, kv_transfer_config=NS(is_kv_consumer=True))
    return NS(vllm_config=config, model_config=model, rank=1)


@pytest.mark.parametrize("failure", [None, "connector", "cache"])
def test_kv_initialization_keeps_call_order_and_original_exception(trace, failure):
    tracing, output = trace
    owner, calls = worker(), []
    sentinel = RuntimeError("original startup failure")

    def connector(*args):
        assert "kv_connector begin" in output.getvalue()
        calls.append("connector")
        if failure == "connector":
            raise sentinel

    def initialize(config):
        assert "kv_connector end" in output.getvalue()
        calls.append("cache")
        if failure == "cache":
            raise sentinel

    owner.model_runner = NS(initialize_kv_cache=initialize)
    initialize = method(
        "worker.py",
        "NPUWorker",
        "initialize_from_config",
        tracing,
        ensure_kv_transfer_initialized=connector,
        envs_ascend=NS(VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=True),
    )
    if failure:
        with pytest.raises(RuntimeError) as caught:
            initialize(owner, NS())
        assert caught.value is sentinel
        assert "kv_init error" in output.getvalue()
    else:
        initialize(owner, NS())
        assert "kv_init end" in output.getvalue()
    assert calls == (["connector"] if failure == "connector" else ["connector", "cache"])


@pytest.mark.parametrize("fail", [False, True])
def test_ep_barrier_is_called_once_and_visible_without_perf_flag(trace, fail):
    tracing, output = trace
    calls, sentinel = [], TimeoutError("same Gloo timeout")

    def barrier():
        assert "ep_barrier begin" in output.getvalue()
        assert "members=0-15" in output.getvalue()
        calls.append("barrier")
        if fail:
            raise sentinel

    group = NS(rank_in_group=0, world_size=16, ranks=list(range(16)), barrier=barrier)
    wait = method(
        "worker.py",
        "NPUWorker",
        "_wait_for_decoder_ep_startup",
        tracing,
        get_ep_group=lambda: group,
        cold_perf_enabled=lambda: False,
        time=time,
    )
    if fail:
        with pytest.raises(TimeoutError) as caught:
            wait(worker())
        assert caught.value is sentinel and "ep_barrier error" in output.getvalue()
    else:
        wait(worker())
        assert "ep_barrier end" in output.getvalue()
    assert calls == ["barrier"]
    owner = worker()
    owner.vllm_config.parallel_config.enable_elastic_ep = True
    wait(owner)
    assert calls == ["barrier"]  # Existing skip conditions are unchanged.


@pytest.mark.parametrize("failure", [None, "allocate", "reshape"])
def test_kv_allocation_reports_phase_without_extra_device_work(trace, monkeypatch, failure):
    tracing, output = trace
    owner, calls = worker(), []
    sentinel, raw, cache = RuntimeError("original allocator failure"), object(), {"layer": object()}

    def allocate(config):
        assert "kv_alloc begin" in output.getvalue() and "bytes=3072" in output.getvalue()
        calls.append("allocate")
        if failure == "allocate":
            raise sentinel
        return raw

    def reshape(config, buffers):
        assert buffers is raw and "kv_reshape begin" in output.getvalue()
        calls.append("reshape")
        if failure == "reshape":
            raise sentinel
        return cache

    monkeypatch.setitem(sys.modules, "vllm.v1.worker.utils", NS(bind_kv_cache=lambda *args: calls.append("bind")))
    owner._allocate_kv_cache_tensors = allocate
    owner._reshape_kv_cache_tensors = reshape
    owner.shared_kv_cache_layers = {}
    owner.model_config.hf_text_config = NS(model_type="glm")
    owner.compilation_config = NS(static_forward_context={})
    owner.kv_caches = []
    initialize = method("model_runner_v1.py", "NPUModelRunner", "initialize_kv_cache_tensors", tracing)
    config = NS(kv_cache_tensors=[NS(size=1024), NS(size=2048)])
    if failure:
        with pytest.raises(RuntimeError) as caught:
            initialize(owner, config)
        assert caught.value is sentinel
        assert f"kv_{'alloc' if failure == 'allocate' else 'reshape'} error" in output.getvalue()
    else:
        assert initialize(owner, config) is cache
    assert (
        calls
        == {None: ["allocate", "reshape", "bind"], "allocate": ["allocate"], "reshape": ["allocate", "reshape"]}[
            failure
        ]
    )


@pytest.mark.parametrize("fail", [False, True])
def test_kv_registration_boundary_surrounds_real_connector_call(trace, fail):
    tracing, output = trace
    owner, calls = worker(), []
    sentinel, cache = RuntimeError("original register failure"), {"layer": object()}

    def register(actual):
        assert actual is cache and "kv_register begin" in output.getvalue()
        calls.append("register")
        if fail:
            raise sentinel

    owner._validate_sfa_layerwise_connector_cudagraph_mode = lambda: None
    owner.may_add_encoder_only_layers_to_kv_cache_config = lambda: None
    owner.maybe_add_kv_sharing_layers_to_kv_cache_groups = lambda _: None
    owner.initialize_attn_backend = lambda _: calls.append("attention")
    owner.attn_groups = [[NS(kv_cache_spec=object())]]
    owner.may_reinitialize_input_batch = lambda _: None
    owner.initialize_kv_cache_tensors = lambda _: cache
    owner.speculative_config = None
    owner._profiling_cudagraph_memory = False
    owner.dsa_unbundle = False
    owner._maybe_init_dsa_latent_offload = lambda: calls.append("offload")
    owner.model_config.enable_return_routed_experts = False
    initialize = method(
        "model_runner_v1.py",
        "NPUModelRunner",
        "initialize_kv_cache",
        tracing,
        staged_sfa_graph_configured=lambda _: False,
        deepcopy=copy.deepcopy,
        MambaSpec=type("MambaSpec", (), {}),
        has_kv_transfer_group=lambda: True,
        get_kv_transfer_group=lambda: NS(register_kv_caches=register),
        envs_ascend=NS(VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE=False),
    )
    if fail:
        with pytest.raises(RuntimeError) as caught:
            initialize(owner, NS(kv_cache_groups=[object()]))
        assert caught.value is sentinel and "kv_register error" in output.getvalue()
    else:
        initialize(owner, NS(kv_cache_groups=[object()]))
        assert "kv_register end" in output.getvalue()
    assert calls == (["attention", "register"] if fail else ["attention", "register", "offload"])
