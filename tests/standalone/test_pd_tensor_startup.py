# SPDX-License-Identifier: Apache-2.0
"""Exercise production startup boundaries without importing the NPU runtime."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]


def runner_methods():
    tree = ast.parse((ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    return {node.name: node for node in owner.body if isinstance(node, ast.FunctionDef)}


def startup(monkeypatch, *, enabled=True, profiling=False):
    """Execute the real KV initializer; replace only hardware/backend boundaries."""
    events = []
    state = NS(connector_ready=False)
    env = NS(VLLM_ASCEND_PD_TENSOR_DUMP_DIR="/capture" if enabled else "", VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE=False)
    connector = NS(register_kv_caches=lambda caches: events.append("register-kv"))
    probe = object()

    def install(runner, root):
        # Reading LMCache before its connector imports Ascend captures the old
        # configuration class. This is the failing boundary from the server log.
        assert state.connector_ready, "PD probe read LMCache config before connector initialization"
        assert root == "/capture"
        events.append("install-probe")
        return probe

    module = ModuleType("vllm_ascend.pd_tensor_dump")
    module.install_pd_tensor_dump = install
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = NS(
        vllm_config=NS(),
        model_config=NS(enable_return_routed_experts=False),
        speculative_config=None,
        _profiling_cudagraph_memory=profiling,
        dsa_unbundle=False,
        attn_groups=[[NS(kv_cache_spec=object())]],
        _validate_sfa_layerwise_connector_cudagraph_mode=lambda: None,
        may_add_encoder_only_layers_to_kv_cache_config=lambda: None,
        maybe_add_kv_sharing_layers_to_kv_cache_groups=lambda config: None,
        initialize_attn_backend=lambda config: None,
        may_reinitialize_input_batch=lambda config: None,
        initialize_kv_cache_tensors=lambda config: {"layer0": object()},
        _maybe_init_dsa_latent_offload=lambda: events.append("latent-init"),
    )
    namespace = dict(
        envs_ascend=env,
        KVCacheConfig=object,
        MambaSpec=type("MambaSpec", (), {}),
        deepcopy=lambda value: value,
        startup_phase=lambda *args, **kwargs: nullcontext(),
        staged_sfa_graph_configured=lambda config: False,
        has_kv_transfer_group=lambda: state.connector_ready,
        get_kv_transfer_group=lambda: connector,
    )
    methods = runner_methods()
    # Execute any diagnostic installation at model-load time, before the worker
    # creates the connector. The previous version fails here, before KV init.
    for statement in methods["load_model"].body:
        if "VLLM_ASCEND_PD_TENSOR_DUMP_DIR" in ast.unparse(statement):
            exec(
                compile(ast.Module(body=[statement], type_ignores=[]), "model-load", "exec"),
                namespace | {"self": runner},
            )
    method = methods["initialize_kv_cache"]
    exec(compile(ast.Module(body=[method], type_ignores=[]), "kv-init", "exec"), namespace)
    return runner, state, events, probe, namespace["initialize_kv_cache"]


@pytest.mark.parametrize("p_node", [False, True])
def test_recorder_installs_after_connector_and_before_requests(monkeypatch, p_node):
    runner, state, events, probe, initialize = startup(monkeypatch)
    runner.layerwise_prefill_p_node = p_node
    state.connector_ready = True  # Worker.initialize_from_config creates it first.
    initialize(runner, NS(kv_cache_groups=[object()]))
    assert events == ["register-kv", "latent-init", "install-probe"]
    assert runner._pd_tensor_dump is probe
    # A second allocation must not stack duplicate recording wrappers.
    initialize(runner, NS(kv_cache_groups=[object()]))
    assert events.count("install-probe") == 1


@pytest.mark.parametrize("enabled,profiling", [(False, False), (False, True), (True, True)])
def test_disabled_or_memory_profile_does_not_initialize_recorder(monkeypatch, enabled, profiling):
    runner, state, events, _, initialize = startup(monkeypatch, enabled=enabled, profiling=profiling)
    state.connector_ready = not profiling
    initialize(runner, NS(kv_cache_groups=[object()]))
    assert "install-probe" not in events
    assert not hasattr(runner, "_pd_tensor_dump")
