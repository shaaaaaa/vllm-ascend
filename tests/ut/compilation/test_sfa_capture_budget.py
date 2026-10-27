# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the complete sizing module; no CANN resource emulation."""

import atexit
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from sfa_test_support import ROOT, definitions, load_module


class CUDAGraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL_DECODE_ONLY = 2


@pytest.fixture
def sizing(monkeypatch):
    # Only unavailable import dependencies are replaced. Configuration gates,
    # environment parsing, resource accounting and size updates are all real.
    modules = {
        "torch_npu": {},
        "vllm.logger": {"logger": logging.getLogger(__name__)},
        "vllm.sequence": {"IntermediateTensors": object},
        "vllm.config": {"CUDAGraphMode": CUDAGraphMode},
        "vllm_ascend.ascend_config": {"WeightPrefetchConfig": object, "get_ascend_config": lambda: None},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    for name, value in {
        "VLLM_ASCEND_SFA_STAGED_GRAPH": "1",
        "VLLM_ASCEND_SFA_FULL_GRAPH": "1",
        "VLLM_ASCEND_DSA_UNBUNDLE": "1",
        "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
        "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
        "VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD": "0",
        "VLLM_ASCEND_DSA_USE_ADAPTER_CACHE": "0",
        "VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP": "0",
        "VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": "2,4,6,8,10,12,14,16",
        "HCCL_OP_EXPANSION_MODE": "AIV",
    }.items():
        monkeypatch.setenv(name, value)
    module = load_module(ROOT / "vllm_ascend/utils.py", "tested_sfa_capture_utils", monkeypatch)
    try:
        yield module
    finally:
        atexit.unregister(module._unregister_print_streams_on_exit)


def config(*, speculative_tokens=1, draft_layers=1, tp=4, dp=4):
    return NS(
        model_config=NS(
            hf_text_config=NS(num_hidden_layers=78, index_topk=2048, to_dict=lambda: {"num_experts": 256}),
            use_mla=True,
            enforce_eager=False,
            architecture="GLM",
        ),
        parallel_config=NS(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            distributed_executor_backend="mp",
        ),
        speculative_config=(
            NS(
                method="mtp",
                num_speculative_tokens=speculative_tokens,
                draft_model_config=NS(get_total_num_hidden_layers=lambda: draft_layers),
            )
            if speculative_tokens
            else None
        ),
        scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=4096),
        compilation_config=NS(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            cudagraph_capture_sizes=list(range(1, 9)),
            max_cudagraph_capture_size=None,
            post_init_cudagraph_sizes=Mock(),
        ),
        additional_config={},
        kv_transfer_config=object(),
        lora_config=None,
    )


@pytest.mark.parametrize("speculative_tokens", [0, 1])
@pytest.mark.parametrize("overlap", [False, True])
def test_eight_full_sfa_request_buckets_reach_capture(sizing, monkeypatch, speculative_tokens, overlap):
    monkeypatch.setenv("VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP", str(int(overlap)))
    cfg = config(speculative_tokens=speculative_tokens)
    sizing.update_aclgraph_sizes(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes == [n * (1 + speculative_tokens) for n in range(2, 17, 2)]
    assert cfg.compilation_config.max_cudagraph_capture_size == 16 * (1 + speculative_tokens)
    cfg.compilation_config.post_init_cudagraph_sizes.assert_called_once_with()


@pytest.mark.parametrize("tp,dp,count", [(4, 4, 4), (8, 1, 7)])
def test_true_staged_path_preserves_original_limit(sizing, monkeypatch, tp, dp, count):
    monkeypatch.setenv("VLLM_ASCEND_SFA_FULL_GRAPH", "0")
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", ",".join(map(str, range(1, count + 1))))
    sizing.update_aclgraph_sizes(config(tp=tp, dp=dp))
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", ",".join(map(str, range(1, count + 2))))
    with pytest.raises(ValueError, match=f"device quota of {count} keys"):
        sizing.update_aclgraph_sizes(config(tp=tp, dp=dp))


@pytest.mark.parametrize(
    "speculative_tokens,draft_layers,overlap,budget",
    [(0, 0, False, 20), (0, 0, True, 40), (1, 1, False, 40), (1, 3, False, 60), (1, 1, True, 60)],
)
def test_root_overlap_and_draft_reservations_remain_bounded(
    sizing, monkeypatch, speculative_tokens, draft_layers, overlap, budget
):
    # Constrained test budgets fit exactly two request buckets. A third must
    # fail; overlap and multi-layer draft reservations cannot be dropped.
    monkeypatch.setattr(sizing, "_MAX_ACL_GRAPH_ENTRIES", budget)
    monkeypatch.setenv("VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP", str(int(overlap)))
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "1,2")
    sizing.update_aclgraph_sizes(config(speculative_tokens=speculative_tokens, draft_layers=draft_layers))
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "1,2,3")
    with pytest.raises(ValueError, match="estimated stream budget of 2 sizes"):
        sizing.update_aclgraph_sizes(config(speculative_tokens=speculative_tokens, draft_layers=draft_layers))


@pytest.mark.parametrize("aiv", [False, True])
def test_existing_communication_reserves_remain_active(sizing, monkeypatch, aiv):
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV" if aiv else "AICPU")
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "1,2")
    # AICPU retains its 40-stream reservation for each of the two domains.
    monkeypatch.setattr(sizing, "_MAX_ACL_GRAPH_ENTRIES", 20 if aiv else 100)
    sizing.update_aclgraph_sizes(config(speculative_tokens=0))
    monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "1,2,3")
    with pytest.raises(ValueError, match="estimated stream budget of 2 sizes"):
        sizing.update_aclgraph_sizes(config(speculative_tokens=0))


def test_eager_worker_does_not_receive_full_graph_budget(sizing):
    cfg = config()
    cfg.model_config.enforce_eager = True
    with pytest.raises(ValueError, match="device quota of 4 keys"):
        sizing.update_aclgraph_sizes(cfg)


@pytest.mark.parametrize("disabled", ["staged_flag", "unrelated_model"])
def test_full_flag_does_not_change_ordinary_path(sizing, monkeypatch, disabled):
    results = []
    for full in (False, True):
        cfg = config()
        monkeypatch.setenv("VLLM_ASCEND_SFA_FULL_GRAPH", str(int(full)))
        if disabled == "staged_flag":
            monkeypatch.setenv("VLLM_ASCEND_SFA_STAGED_GRAPH", "0")
        else:
            cfg.model_config.use_mla = False
        sizing.update_aclgraph_sizes(cfg)
        results.append(cfg.compilation_config.cudagraph_capture_sizes)
    assert results[0] == results[1]
    assert len(results[0]) == 4


@pytest.mark.parametrize("limit", ["max_num_seqs", "max_num_batched_tokens"])
def test_scheduler_capacity_still_bounds_full_graph_sizes(sizing, limit):
    cfg = config()
    setattr(cfg.scheduler_config, limit, 15)
    with pytest.raises(ValueError, match="exceed scheduler capacity"):
        sizing.update_aclgraph_sizes(cfg)


def test_full_graph_still_rejects_unsupported_mtp_width(sizing):
    with pytest.raises(ValueError, match="at most one MTP draft token"):
        sizing.update_aclgraph_sizes(config(speculative_tokens=2))


@pytest.mark.parametrize("query_width", [1, 2])
def test_budgeted_target_profiles_match_runner_capture_keys(sizing, query_width):
    namespace = {"dataclass": dataclass, "Enum": Enum, "sfa_full_graph_enabled": sizing.sfa_full_graph_enabled}
    definitions(
        ROOT / "vllm_ascend/ascend_forward_context.py", {"StagedSFAQueryProfile", "StagedSFAGraphKey"}, namespace
    )
    definitions(
        ROOT / "vllm_ascend/worker/model_runner_v1.py",
        {"_staged_sfa_capture_keys"},
        namespace,
        class_name="NPUModelRunner",
    )
    cfg = config(speculative_tokens=query_width - 1)
    runner = NS(vllm_config=cfg, decode_threshold=query_width)
    keys = namespace["_staged_sfa_capture_keys"](runner, 8 * query_width)
    assert len(set(keys)) == 2
    assert {key.query_profile.value for key in keys} == {
        "decode_bounded",
        "decode_q1" if query_width == 1 else "spec_fixed",
    }
    assert all(key.request_capacity == 8 for key in keys)
