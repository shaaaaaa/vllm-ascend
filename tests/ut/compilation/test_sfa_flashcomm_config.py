# SPDX-License-Identifier: Apache-2.0
"""Actual config helpers/initialization block and target tail, with CPU stand-ins."""

import ast
import os
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import ROOT, extract


@pytest.fixture
def switches(monkeypatch):
    cfg = NS(
        additional_config={},
        model_config=NS(hf_text_config=NS(index_topk=2048)),
        parallel_config=NS(enable_expert_parallel=True, tensor_parallel_size=4),
    )
    env, ascend = NS(VLLM_ASCEND_ENABLE_FLASHCOMM1=False), NS(enable_shared_expert_dp=False)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM", "0")
    monkeypatch.setitem(sys.modules, "vllm.config", NS(get_current_vllm_config=lambda: cfg))
    ns = dict(
        _ENABLE_SP=None,
        envs_ascend=env,
        os=os,
        logger=Mock(),
        get_ascend_config=lambda: ascend,
        enable_sp_by_pass=lambda: False,
    )
    for name in ("enable_sp", "enable_dsa_cp", "shared_expert_dp_enabled"):
        extract(ROOT / "vllm_ascend/utils.py", name, ns)
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", NS(enable_sp=ns["enable_sp"]))

    def initialize_shared():
        # Execute the complete consecutive placement-initialization block from
        # the real constructor without importing unrelated NPU configuration.
        path = ROOT / "vllm_ascend/ascend_config.py"
        cls = next(
            n
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "AscendConfig"
        )
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        positions = [
            i
            for i, n in enumerate(init.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "enable_shared_expert_dp" for t in n.targets)
        ]
        assert len(positions) == 2
        tree = ast.Module(body=init.body[positions[0] : positions[-1] + 1], type_ignores=[])
        exec(
            compile(ast.fix_missing_locations(tree), str(path), "exec"),
            dict(self=ascend, additional_config=cfg.additional_config, vllm_config=cfg),
        )
        return ns["shared_expert_dp_enabled"]()

    return cfg, env, ns, initialize_shared


@pytest.mark.parametrize("environment,legacy", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("explicit", [None, False, True])
def test_fc1_override_and_environment_compatibility(switches, monkeypatch, environment, legacy, explicit):
    cfg, env, ns, _ = switches
    env.VLLM_ASCEND_ENABLE_FLASHCOMM1 = environment
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM", str(int(legacy)))
    if explicit is not None:
        cfg.additional_config["enable_flashcomm1"] = explicit
    assert ns["enable_sp"]() == (environment or legacy if explicit is None else explicit)


@pytest.mark.parametrize("fc1", [False, True])
@pytest.mark.parametrize("cp", [None, False, True])
@pytest.mark.parametrize("indexer", [False, True])
def test_dsa_cp_decoupling_preserves_omitted_prefiller_setting(switches, fc1, cp, indexer):
    cfg, _, ns, _ = switches
    cfg.additional_config["enable_flashcomm1"] = fc1
    if cp is not None:
        cfg.additional_config["enable_dsa_cp"] = cp
    if not indexer:
        del cfg.model_config.hf_text_config.index_topk
    if indexer and cp and not fc1:
        with pytest.raises(ValueError, match="requires FlashComm1"):
            ns["enable_dsa_cp"]()
    else:
        assert ns["enable_dsa_cp"]() == (indexer and (fc1 if cp is None else cp))


@pytest.mark.parametrize("fc1", [False, True])
@pytest.mark.parametrize("shared", [None, False, True])
def test_shared_placement_is_independent_with_legacy_default(switches, fc1, shared):
    cfg, _, ns, initialize = switches
    cfg.additional_config["enable_flashcomm1"] = fc1
    if shared is not None:
        cfg.additional_config["enable_shared_expert_dp"] = shared
    assert initialize() == (fc1 if shared is None else shared)
    # Existing explicit shared-DP opt-in can enable FC1 before its cache is set.
    assert ns["enable_sp"]() == (fc1 or shared is True)


def test_none_additional_config_keeps_environment_default(switches):
    cfg, env, ns, _ = switches
    cfg.additional_config = None
    env.VLLM_ASCEND_ENABLE_FLASHCOMM1 = True
    assert ns["enable_sp"]() and ns["enable_dsa_cp"]()


@pytest.mark.parametrize("ep,tp", [(False, 4), (True, 1)])
def test_explicit_shared_dp_alone_keeps_existing_topology_guard(switches, ep, tp):
    cfg, _, ns, initialize = switches
    cfg.parallel_config.enable_expert_parallel = ep
    cfg.parallel_config.tensor_parallel_size = tp
    cfg.additional_config["enable_shared_expert_dp"] = True
    assert not initialize() and not ns["enable_sp"]()


@pytest.mark.parametrize("fc1", [False, True])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_actual_target_tail_gathers_once_inside_capture_callable(fc1, auxiliary):
    class Intermediate:
        pass

    events = []
    gathered = object()
    local = torch.arange(6).reshape(2, 3)
    output = (local, [local + 1]) if auxiliary else local
    fn = extract(
        ROOT / "vllm_ascend/worker/model_runner_v1.py",
        "_run_sfa_full_graph_target",
        dict(
            get_forward_context=lambda: NS(flash_comm_v1_enabled=fc1),
            IntermediateTensors=Intermediate,
        ),
    )

    def model(**kwargs):
        events.append("model")
        assert kwargs == {"positions": "positions"}
        return output

    def gather(value):
        events.append("gather")
        assert value is output
        return gathered

    runner = NS(model=model, _all_gather_hidden_states_and_aux=gather)
    assert fn(runner, positions="positions") is (gathered if fc1 else output)
    assert events == (["model", "gather"] if fc1 else ["model"])
    runner.model = lambda **kwargs: Intermediate()
    assert isinstance(fn(runner), Intermediate)
