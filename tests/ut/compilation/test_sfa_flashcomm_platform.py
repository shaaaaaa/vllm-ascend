# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the platform's size filter and existing staged size budgeting."""

import ast
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from sfa_test_support import ROOT, extract

PLATFORM = ROOT / "vllm_ascend/platform.py"
UTILS = ROOT / "vllm_ascend/utils.py"


def config(max_requests, speculative=1, *, cp=False, flashcomm=True, staged=True):
    cfg = NS(
        additional_config={"enable_dsa_cp": cp},
        parallel_config=NS(tensor_parallel_size=4, data_parallel_size=4),
        scheduler_config=NS(max_num_seqs=max_requests, max_num_batched_tokens=4096),
        model_config=NS(enforce_eager=False, hf_text_config=NS(num_hidden_layers=78)),
        speculative_config=NS(num_speculative_tokens=speculative, draft_model_config=None),
        compilation_config=NS(
            max_cudagraph_capture_size=None, cudagraph_capture_sizes=None, post_init_cudagraph_sizes=Mock()
        ),
        staged=staged,
        flashcomm=flashcomm,
        kv_transfer_config=None,
    )
    maximum = extract(PLATFORM, "_get_default_max_cudagraph_capture_size", {})(None, cfg)
    cfg.compilation_config.max_cudagraph_capture_size = maximum
    cfg.compilation_config.cudagraph_capture_sizes = [size for size in (1, 2, 4, 8, 16, 24) if size <= maximum]
    core = ROOT.parent / "vllm/vllm/config/vllm.py"
    filter_sizes = extract(core, "update_sizes_for_sequence_parallelism", dict(logger=Mock()))
    cfg.update_sizes_for_sequence_parallelism = lambda sizes: filter_sizes(cfg, sizes)
    return cfg


def size_namespace(request_sizes, capacity=100):
    ns = dict(
        logger=Mock(),
        enable_sp=lambda cfg: cfg.flashcomm,
        staged_sfa_graph_configured=lambda cfg: cfg.staged,
        sfa_full_graph_enabled=lambda _: True,
        _SFA_FULL_GRAPH_LAYOUTS=2,
        _max_aclgraph_keys=lambda *args: capacity,
        envs_ascend=NS(
            VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES=request_sizes,
            VLLM_ASCEND_SFA_FULL_GRAPH=True,
            VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP=True,
            VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=False,
        ),
    )
    for name in (
        "dsa_cp_decoder_configured", "update_cudagraph_capture_sizes",
        "staged_sfa_graph_capture_sizes", "update_aclgraph_sizes",
    ):
        extract(UTILS, name, ns)
    return ns


def run_platform_filter(cfg, ns):
    tree = ast.parse(PLATFORM.read_text(encoding="utf-8"))
    # Execute the real pre-filter block from check_and_update_config. Importing
    # the complete platform would require an installed Ascend native runtime.
    block = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "enable_sp(vllm_config)" in ast.unparse(node.test)
        and any("sp_aclgraph_sizes" in ast.unparse(child) for child in node.body)
    )
    ns.update(vllm_config=cfg, compilation_config=cfg.compilation_config)
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(PLATFORM), "exec"), ns)


@pytest.mark.parametrize("requests,speculative,sizes", [(1, 0, "1"), (1, 1, "1"), (12, 1, "2,4,6,8,10,12")])
def test_fc_only_staged_sizes_survive_initial_platform_filter(requests, speculative, sizes):
    cfg, ns = config(requests, speculative), size_namespace(sizes)
    run_platform_filter(cfg, ns)
    ns["update_aclgraph_sizes"](cfg)
    expected = [int(size) * (1 + speculative) for size in sizes.split(",")]
    assert cfg.compilation_config.cudagraph_capture_sizes == expected
    assert cfg.compilation_config.max_cudagraph_capture_size == max(expected)


@pytest.mark.parametrize("requests,speculative,sizes", [(1, 0, "1"), (1, 1, "1"), (12, 1, "2,4,6,8,10,12")])
def test_decoder_cp_preserves_logical_capture_buckets(requests, speculative, sizes):
    cfg, ns = config(requests, speculative, cp=True), size_namespace(sizes)
    cfg.kv_transfer_config = NS(is_kv_consumer=True, is_kv_producer=True)
    run_platform_filter(cfg, ns)
    ns["update_aclgraph_sizes"](cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes == [int(s) * (1 + speculative) for s in sizes.split(",")]


@pytest.mark.parametrize("cp,staged", [(True, True), (None, True), (False, False)])
def test_legacy_and_nonstaged_flashcomm_still_require_divisible_sizes(cp, staged):
    cfg, ns = config(1, cp=cp, staged=staged), size_namespace("1")
    if cp is None:
        cfg.additional_config.clear()
    with pytest.raises(AssertionError, match="multiples of tp_size"):
        run_platform_filter(cfg, ns)
    cfg = config(12, cp=cp, staged=staged)
    if cp is None:
        cfg.additional_config.clear()
    run_platform_filter(cfg, ns)
    assert cfg.compilation_config.cudagraph_capture_sizes == [4, 8, 16, 24]


def test_disabled_flashcomm_keeps_ordinary_initial_sizes():
    cfg, ns = config(1, flashcomm=False), size_namespace("1")
    run_platform_filter(cfg, ns)
    assert cfg.compilation_config.cudagraph_capture_sizes == [1, 2]


@pytest.mark.parametrize("sizes,tokens,capacity", [("2", 4096, 100), ("1", 1, 100), ("1", 4096, 0)])
def test_staged_request_token_and_stream_limits_are_not_bypassed(sizes, tokens, capacity):
    cfg, ns = config(1), size_namespace(sizes, capacity)
    cfg.scheduler_config.max_num_batched_tokens = tokens
    run_platform_filter(cfg, ns)
    with pytest.raises(ValueError, match="exceed"):
        ns["update_aclgraph_sizes"](cfg)
