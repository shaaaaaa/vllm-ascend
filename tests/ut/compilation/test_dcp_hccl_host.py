# SPDX-License-Identifier: Apache-2.0
"""CPU checks of production communicator options and mixed-mode graph budgets."""

import os
import sys
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from sfa_test_support import ROOT, extract
from test_sfa_capture_budget import config, sizing  # noqa: F401


@pytest.fixture
def options(request, monkeypatch):
    module = request.getfixturevalue("sizing")
    monkeypatch.delenv("VLLM_ASCEND_DCP_HCCL_HOST", raising=False)
    monkeypatch.setattr(module, "calculate_dp_buffer_size", lambda: 64)
    backend = NS(ProcessGroupHCCL=NS(Options=lambda: NS(hccl_config=None)))
    monkeypatch.setattr(module.torch_npu, "_C", NS(_distributed_c10d=backend), raising=False)
    return module


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("name", ["dcp", "tp", "ep", "dp", "pcp", "dynamic_eplb", "mc2", "mc2_dcp"])
def test_override_is_only_applied_to_dcp(options, monkeypatch, enabled, name):
    monkeypatch.setenv("VLLM_ASCEND_DCP_HCCL_HOST", str(int(enabled)))
    result = options.create_hccl_pg_options(name).hccl_config
    if "mc2" in name:
        assert result is None
    else:
        expected = {"hccl_buffer_size": {"dp": 64, "dynamic_eplb": options._DYNAMIC_EPLB_BUFFER_SIZE}.get(name, 200)}
        if name == "dcp" and enabled:
            expected["hccl_op_expansion_mode"] = 1
        assert result == expected
    assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AIV"


def test_default_and_invalid_override(options, monkeypatch):
    assert options.create_hccl_pg_options("dcp").hccl_config == {"hccl_buffer_size": 200}
    monkeypatch.setenv("VLLM_ASCEND_DCP_HCCL_HOST", "invalid")
    with pytest.raises(ValueError):
        options.create_hccl_pg_options("dcp")


def test_deployed_group_constructor_receives_host_options(options, monkeypatch, caplog):
    monkeypatch.setenv("VLLM_ASCEND_DCP_HCCL_HOST", "1")
    caplog.set_level("INFO")
    device_group, cpu_group = object(), object()
    new_group = Mock(side_effect=[device_group, cpu_group])
    communicator = Mock()
    message_queue = ModuleType("vllm.distributed.device_communicators.shm_broadcast")
    message_queue.MessageQueue = object
    monkeypatch.setitem(sys.modules, message_queue.__name__, message_queue)
    init = extract(ROOT / "vllm_ascend/patch/worker/patch_distributed.py", "__init__", {
        "torch": NS(distributed=NS(get_rank=lambda: 0, new_group=new_group), npu=NS(current_device=lambda: 0)),
        "_get_unique_name": lambda name: name + ":0", "_register_group": lambda group: None,
        "create_hccl_pg_options": options.create_hccl_pg_options, "NPUCommunicator": communicator,
    })
    state, ranks = NS(), list(range(8))
    init(state, [ranks], 0, "hccl", True, group_name="dcp")
    hccl, gloo = new_group.call_args_list
    assert hccl.args == (ranks,)
    assert hccl.kwargs["backend"] == "hccl"
    assert hccl.kwargs["pg_options"].hccl_config == {"hccl_buffer_size": 200, "hccl_op_expansion_mode": 1}
    assert gloo.kwargs == {"backend": "gloo"}
    assert state.device_group is device_group and state.cpu_group is cpu_group
    communicator.assert_called_once_with(cpu_group=cpu_group, device=0, device_group=device_group, unique_name="dcp:0")
    assert "DCP HCCL expansion mode override: HOST (global=AIV, buffer=200 MiB)" in caplog.text


def test_full_graph_budget_reserves_host_dcp_streams(options, monkeypatch):
    monkeypatch.setattr(options, "_MAX_ACL_GRAPH_ENTRIES", 200)
    monkeypatch.setenv("VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP", "1")
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "1,2,3")
    cfg = config(tp=8, dp=1)
    cfg.parallel_config.decode_context_parallel_size = 8
    cfg.kv_transfer_config = NS(kv_connector="LMCacheAscendConnectorV1Dynamic", is_kv_consumer=True)
    options.update_aclgraph_sizes(cfg)
    monkeypatch.setenv("VLLM_ASCEND_DCP_HCCL_HOST", "1")
    # Six units per size, with one HOST DCP plus an AIV TP/EP path:
    # floor((200 - 100 - 40) / (6 * 5)) = 2 supported sizes.
    with pytest.raises(ValueError, match="estimated stream budget of 2 sizes"):
        options.update_aclgraph_sizes(cfg)
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "1,2")
    options.update_aclgraph_sizes(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes == [2, 4]


@pytest.mark.parametrize("mode,dcp", [("AIV", 1), ("HOST", 1), ("HOST", 8)])
def test_dcp1_and_existing_host_budgets_are_unchanged(options, monkeypatch, mode, dcp):
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", mode)
    cfg = config(tp=8, dp=1)
    cfg.parallel_config.decode_context_parallel_size = dcp
    results = []
    for enabled in (0, 1):
        monkeypatch.setenv("VLLM_ASCEND_DCP_HCCL_HOST", str(enabled))
        results.append(options._max_aclgraph_keys(cfg, 6, 1 + int(dcp > 1)))
    assert results[0] == results[1]


@pytest.mark.parametrize("full", [False, True])
def test_mixed_mode_counts_dcp_group_once(options, monkeypatch, full):
    monkeypatch.setenv("VLLM_ASCEND_DCP_HCCL_HOST", "1")
    monkeypatch.setenv("VLLM_ASCEND_SFA_FULL_GRAPH", str(int(full)))
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "1")
    budget = Mock(wraps=options._max_aclgraph_keys)
    monkeypatch.setattr(options, "_max_aclgraph_keys", budget)
    cfg = config(tp=8, dp=1)
    cfg.parallel_config.decode_context_parallel_size = 8
    cfg.kv_transfer_config = NS(kv_connector="LMCacheAscendConnectorV1Dynamic", is_kv_consumer=True)
    options.update_aclgraph_sizes(cfg)
    assert budget.call_args.args[2] == 2
