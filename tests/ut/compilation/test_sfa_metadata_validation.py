# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of real SFA eligibility checks and per-forward memoization."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sfa_test_support import definitions, extract


@dataclass(frozen=True)
class GraphKey:
    request_capacity: int = 1
    max_query_len: int = 2
    query_profile: str = "bounded"

    @property
    def token_capacity(self):
        return self.request_capacity * self.max_query_len

    def to_legacy_batch_descriptor(self):
        return SimpleNamespace(
            num_tokens=self.token_capacity, uniform=False, has_lora=False, num_reqs=None, num_active_loras=0
        )


@pytest.fixture
def checks():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_v1.py"
    names = ("_cross_layer_ineligible_reason", "_cross_layer_metadata_ineligible_reason")
    context = SimpleNamespace(
        staged_sfa_graph_key=GraphKey(),
        cudagraph_runtime_mode="piecewise",
        staged_sfa_graph_dummy_run=False,
        batch_descriptor=GraphKey().to_legacy_batch_descriptor(),
        dsa_offload_manager=None,
        dsa_adapter_cache=None,
    )
    namespace = {
        "np": np,
        "torch": torch,
        "get_forward_context": lambda: context,
        "CUDAGraphMode": SimpleNamespace(PIECEWISE="piecewise", NONE="none"),
        "StagedSFAQueryProfile": SimpleNamespace(DECODE_Q1="q1", SPEC_FIXED="fixed", DECODE_BOUNDED="bounded"),
        "AscendAttentionState": SimpleNamespace(DecodeOnly="decode", SpecDecoding="spec"),
        "envs": SimpleNamespace(VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY=False),
        "staged_sfa_connector_supports_sparse_load": lambda: True,
        "get_weight_prefetch_method": lambda: None,
    }
    definitions(path, names, namespace, class_name="AscendSFAImpl")
    impl_type = type("RealChecks", (), {name: namespace[name] for name in names})

    def make_impl():
        impl = impl_type()
        for name, value in dict(
            has_indexer=True,
            vllm_config=SimpleNamespace(lora_config=None, cache_config=SimpleNamespace(block_size=4)),
            _staged_sfa_graph_capture_sizes=(2,),
            decode_threshold=2,
            dsa_shrink_latent=2,
            enable_mlapo=False,
            enable_dsa_cp=False,
            enable_dsa_cp_with_o_proj_tp=False,
            use_sparse_c8_indexer=False,
            dsa_offload_free_paged=False,
            num_kv_heads=1,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            head_dim=3,
            q_lora_rank=4,
            fused_qkv_a_proj=object(),
            q_a_layernorm=object(),
            dsa_resident_cache=False,
        ).items():
            setattr(impl, name, value)
        impl._cross_layer_metadata_ineligible_reason = Mock(wraps=impl._cross_layer_metadata_ineligible_reason)
        return impl

    def make_metadata():
        return SimpleNamespace(
            attn_state="spec",
            num_actual_tokens=2,
            num_input_tokens=2,
            num_decode_tokens=2,
            decode_request_ids_compact=["r"],
            req_ids=["r"],
            need_sparse_lmcache_payload=True,
            cos=torch.zeros(2, 2),
            sin=torch.zeros(2, 2),
            slot_mapping=torch.zeros(2),
            indexer_slot_mapping=torch.zeros(2),
            cum_query_lens=torch.zeros(2),
            seq_lens=torch.zeros(2),
            block_table=torch.zeros(2, 4),
            indexer_block_table=torch.zeros(2, 4),
            decode_req_indices=torch.zeros(2),
            decode_selected_tokens=torch.zeros(1, 4),
            decode_selected_counts=torch.zeros(1),
            decode_target_slot_mapping=torch.zeros(1, 4),
            decode_union_mapping_workspace=torch.zeros(1, 4),
            decode_shard_packed_workspace=torch.zeros(1, 2, 4),
            decode_shard_mapping_workspace=torch.zeros(1, 2, 4),
            decode_shard_counts_workspace=torch.zeros(1, 2, 16),
            resident_state_indices=None,
            resident_state_generations=None,
            prompt_lens_cpu_rows=[5000, 5000],
            decode_req_indices_cpu=[0, 0],
            seq_lens_cpu=torch.tensor([5002, 0]),
            decode_remap_boundary=torch.zeros(2),
        )

    def check(impl, metadata, memo=None, caches=None):
        if caches is None:
            caches = tuple(torch.zeros(5, 4, 1, dim, dtype=torch.float16) for dim in (4, 2, 3))
        hidden = torch.zeros(context.staged_sfa_graph_key.token_capacity, 4)
        return impl._cross_layer_ineligible_reason(hidden, caches, metadata, metadata_checks=memo)

    return make_impl, make_metadata, check, context


def test_missing_constructor_capture_validation_is_still_enforced(checks, monkeypatch):
    make_impl, make_metadata, check, _ = checks
    impl, metadata = make_impl(), make_metadata()
    validation = Mock(side_effect=ValueError("invalid capture configuration"))
    monkeypatch.setitem(
        type(impl)._cross_layer_ineligible_reason.__globals__, "staged_sfa_graph_capture_sizes", validation
    )
    assert check(impl, metadata) is None
    validation.assert_not_called()
    del impl._staged_sfa_graph_capture_sizes
    with pytest.raises(ValueError, match="invalid capture configuration"):
        check(impl, metadata)
    validation.assert_called_once_with(impl.vllm_config)


@pytest.mark.parametrize("layers", [1, 8, 80])
def test_shared_metadata_checked_once_but_again_next_forward(checks, layers):
    make_impl, make_metadata, check, _ = checks
    impls = [make_impl() for _ in range(layers)]
    metadata = make_metadata()
    for _ in range(3):
        memo = {}
        for impl in impls:
            assert check(impl, metadata, memo) is None
    assert sum(i._cross_layer_metadata_ineligible_reason.call_count for i in impls) == 3
    # Mutable contents of this same CPU metadata must be rechecked next step.
    metadata.decode_req_indices_cpu = [0, 1]
    assert check(impls[0], metadata, {}) == "invalid bounded-decode request ownership"


@pytest.mark.parametrize("consumer_first", [False, True])
def test_glm52_shared_consumer_keeps_layer_guards_with_shared_metadata_memo(checks, consumer_first):
    make_impl, make_metadata, check, _ = checks
    producer, consumer = make_impl(), make_impl()
    consumer.has_indexer = False
    consumer.topk_indices_buffer = torch.zeros(2, 4, dtype=torch.int32)
    caches = tuple(torch.zeros(5, 4, 1, dim, dtype=torch.float16) for dim in (4, 2, 3))
    metadata, memo = make_metadata(), {}
    layers = [(producer, caches), (consumer, caches[:2])]
    for impl, planes in reversed(layers) if consumer_first else layers:
        assert check(impl, metadata, memo, planes) is None
    assert sum(x._cross_layer_metadata_ineligible_reason.call_count for x in (producer, consumer)) == 1
    # Memoization cannot authorize a consumer without its raw top-k source.
    consumer.topk_indices_buffer = None
    assert "shared top-k buffer" in check(consumer, metadata, memo, caches[:2])
    # Nor can a consumer's successful check authorize a producer without Group 1.
    assert "indexer plane" in check(producer, metadata, memo, caches[:2])


def test_distinct_metadata_objects_are_not_merged(checks):
    make_impl, make_metadata, check, _ = checks
    impls = [make_impl() for _ in range(8)]
    first, second = make_metadata(), make_metadata()
    memo = {}
    for index, impl in enumerate(impls):
        assert check(impl, first if index % 2 else second, memo) is None
    assert sum(i._cross_layer_metadata_ineligible_reason.call_count for i in impls) == 2
    assert len(memo) == 2


def test_other_implementation_type_cannot_reuse_base_policy(checks):
    make_impl, make_metadata, check, _ = checks
    first, second = make_impl(), make_impl()
    second.__class__ = type("DifferentAttention", (type(second),), {})
    metadata, memo = make_metadata(), {}
    assert check(first, metadata, memo) is None
    assert check(second, metadata, memo) is None
    assert first._cross_layer_metadata_ineligible_reason.call_count == 1
    assert second._cross_layer_metadata_ineligible_reason.call_count == 1


@pytest.mark.parametrize("change", ["cache_shape", "cache_dtype", "layer_flag", "resident_required"])
def test_shared_metadata_hit_does_not_hide_later_layer_incompatibility(checks, change):
    make_impl, make_metadata, check, _ = checks
    first, second = make_impl(), make_impl()
    metadata, memo = make_metadata(), {}
    assert check(first, metadata, memo) is None
    caches = None
    if change == "cache_shape":
        caches = tuple(torch.zeros(5, 4, 1, dim, dtype=torch.float16) for dim in (4, 2, 5))
    elif change == "cache_dtype":
        caches = tuple(torch.zeros(5, 4, 1, dim, dtype=torch.float32) for dim in (4, 2, 3))
    elif change == "layer_flag":
        second.enable_mlapo = True
    else:
        second.dsa_resident_cache = True
    assert check(second, metadata, memo, caches=caches) is not None
    assert second._cross_layer_metadata_ineligible_reason.call_count == int(change == "resident_required")


@pytest.mark.parametrize(
    "field,value",
    [
        ("attn_state", "prefill"),
        ("num_actual_tokens", 0),
        ("num_input_tokens", 4),
        ("num_decode_tokens", 1),
        ("cos", None),
        ("sin", torch.zeros(1)),
        ("cum_query_lens", None),
        ("block_table", None),
        ("indexer_block_table", torch.zeros(1, 4)),
        ("need_sparse_lmcache_payload", False),
        ("decode_selected_tokens", None),
        ("decode_shard_counts_workspace", torch.zeros(1, 2, 4)),
        ("req_ids", ["other"]),
        ("decode_remap_boundary", None),
        ("prompt_lens_cpu_rows", []),
        ("decode_req_indices_cpu", [1, 0]),
        ("seq_lens_cpu", torch.zeros(1)),
    ],
)
def test_invalid_shared_inputs_still_rejected_and_failure_can_be_memoized(checks, field, value):
    make_impl, make_metadata, check, _ = checks
    impl = make_impl()
    metadata = make_metadata()
    setattr(metadata, field, value)
    memo = {}
    reason = check(impl, metadata, memo)
    assert reason is not None
    assert check(impl, metadata, memo) == reason
    assert impl._cross_layer_metadata_ineligible_reason.call_count == 1


def test_unmemoized_staged_checks_and_graph_context_guards_remain(checks):
    make_impl, make_metadata, check, context = checks
    impl, metadata = make_impl(), make_metadata()
    for _ in range(4):
        assert check(impl, metadata) is None
    assert impl._cross_layer_metadata_ineligible_reason.call_count == 4
    memo = {}
    assert check(impl, metadata, memo) is None
    context.cudagraph_runtime_mode = "none"
    assert check(impl, metadata, memo) == "the runtime graph mode is not PIECEWISE"


@pytest.mark.parametrize("profile", ["q1", "fixed", "bounded_q1"])
def test_other_authorized_decode_shapes_keep_validation(checks, profile):
    make_impl, make_metadata, check, context = checks
    impl, metadata = make_impl(), make_metadata()
    key = (
        GraphKey(max_query_len=1, query_profile="q1")
        if profile == "q1"
        else GraphKey(query_profile="fixed" if profile == "fixed" else "bounded")
    )
    context.staged_sfa_graph_key = key
    context.batch_descriptor = key.to_legacy_batch_descriptor()
    impl._staged_sfa_graph_capture_sizes = (key.token_capacity,)
    if profile != "bounded_q1":
        metadata.seq_lens_cpu = metadata.seq_lens_cpu[:1]
        for name in ("seq_lens", "cum_query_lens", "block_table", "indexer_block_table"):
            setattr(metadata, name, getattr(metadata, name)[:1])
    if profile != "fixed":
        metadata.attn_state = "decode"
        metadata.num_actual_tokens = metadata.num_decode_tokens = 1
        if profile == "q1":
            metadata.num_input_tokens = 1
            metadata.prompt_lens_cpu_rows = [5000]
            metadata.decode_req_indices_cpu = [0]
            for name in (
                "cos",
                "sin",
                "slot_mapping",
                "indexer_slot_mapping",
                "decode_req_indices",
                "decode_remap_boundary",
            ):
                setattr(metadata, name, getattr(metadata, name)[:1])
        else:
            metadata.prompt_lens_cpu_rows = [5000, 0]
            metadata.decode_req_indices_cpu = [0, -1]
    memo = {}
    assert check(impl, metadata, memo) is None
    assert check(impl, metadata, memo) is None
    assert impl._cross_layer_metadata_ineligible_reason.call_count == 1


def test_shared_checker_depends_on_no_unkeyed_layer_attributes(checks):
    # A future policy added to this helper must be reflected in the memo key.
    make_impl, _, _, _ = checks
    helper = make_impl()._cross_layer_metadata_ineligible_reason._mock_wraps
    path = Path(helper.__code__.co_filename)
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl"
    )
    method = next(n for n in cls.body if getattr(n, "name", "") == helper.__name__)
    attributes = {
        n.attr
        for n in ast.walk(method)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
    }
    assert attributes == {"dsa_resident_cache"}


@pytest.mark.parametrize("dummy", [False, True])
def test_full_graph_static_checks_only_run_at_startup(dummy):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_v1.py"
    # Deliberately no tensor metadata attributes: replay must not enumerate them.
    metadata = SimpleNamespace(req_ids=["request"], reshape_cache_event=object())
    context = SimpleNamespace(
        attn_metadata={"L0": metadata},
        staged_sfa_graph_dummy_run=dummy,
        staged_sfa_graph_key=GraphKey(),
        staged_sfa_route=SimpleNamespace(frontiers=(4096,)),
    )
    boundary = Mock(return_value=object())
    namespace = {"get_forward_context": lambda: context, "_prepare_sfa_remap_boundary": boundary}
    extract(path, "prepare_full_graph_layer", namespace)
    extract(path, "prepare_full_graph_metadata", namespace)
    transfer = Mock()
    impl = SimpleNamespace(
        _staged_sfa_capture_state=SimpleNamespace(runtime=(None, [object(), object()])),
        _staged_sfa_bridge_buffers=(torch.empty(2),),
        _cross_layer_ineligible_reason=Mock(return_value="static failure"),
        _full_graph_transfers={1: transfer},
        index_topk=2048,
    )
    impl.prepare_full_graph_metadata = lambda metadata, ctx: namespace["prepare_full_graph_metadata"](
        impl, metadata, ctx
    )
    prepare = namespace["prepare_full_graph_layer"]
    if dummy:
        with pytest.raises(RuntimeError, match="static failure"):
            prepare(impl, "L0", 65536, bind_source=False)
        impl._cross_layer_ineligible_reason.assert_called_once()
        boundary.assert_not_called()
    else:
        for frontier in (4096, 4352):
            context.staged_sfa_route.frontiers = (frontier,)
            assert prepare(impl, "L0", 65536, bind_source=False) is None
            assert boundary.call_args.kwargs["cached_tokens"] == (frontier,)
            assert impl._full_graph_transfer is transfer
            assert metadata.reshape_cache_event is None
        impl._cross_layer_ineligible_reason.assert_not_called()
        transfer.bind_batch.assert_not_called()
        impl._full_graph_transfers.clear()
        with pytest.raises(RuntimeError, match="not allocated at startup"):
            prepare(impl, "L0", 65536, bind_source=False)
