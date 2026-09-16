# SPDX-License-Identifier: Apache-2.0
"""Exercise the real SFA builder on CPU without importing the NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from test_checkpoint_graph_route import route_api


class HostMetadataParent:
    """Replace only upstream model/runtime initialization, not SFA metadata."""

    @classmethod
    def __class_getitem__(cls, _):
        return cls

    def __init__(self, spec, layers, config, device, metadata_cls, supports_dcp):
        self.metadata_cls = metadata_cls
        self.model_config = config.model_config
        self.device = device


@pytest.fixture(scope="module")
def api():
    ns = route_api()
    ns.update(
        torch=torch,
        MLACommonMetadataBuilder=HostMetadataParent,
        AscendSFAMetadata=NS,
        AttentionMaskBuilder=lambda device: NS(get_attention_mask=lambda config: None),
        enable_dsa_cp=lambda: False,
        envs=NS(VLLM_ASCEND_DSA_UNBUNDLE=True, VLLM_ASCEND_DSA_SHRINK_LATENT=2, VLLM_ASCEND_MTP_DW_DEEP_DIAG=False),
        get_cos_and_sin_mla=lambda positions, _: (torch.zeros_like(positions), torch.zeros_like(positions)),
        staged_sfa_connector_supports_sparse_load=lambda: True,
    )
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in {"AscendSFAMetadataBuilder", "_update_dsa_split_boundary_in_place", "_fixed_staged_decode_mtp"}
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"), ns)
    return ns


@pytest.fixture
def builder(api):
    model = NS(
        max_model_len=178000,
        get_head_size=lambda: 576,
        hf_config=None,
        hf_text_config=NS(qk_rope_head_dim=64, topk_tokens=2048),
    )
    config = NS(
        cache_config=NS(block_size=128),
        model_config=model,
        speculative_config=NS(num_speculative_tokens=1),
        scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=32),
    )
    return api["AscendSFAMetadataBuilder"](None, [], config, torch.device("cpu"))


def common(api, widths, cold_indices, *, computed=None, frontier=None, state="spec", padded=None):
    count = len(widths)
    computed = [22684] * count if computed is None else computed
    frontier = computed if frontier is None else frontier
    actual = sum(widths)
    padded = actual if padded is None else padded
    starts = torch.tensor([0, *np.cumsum(widths)], dtype=torch.int32)
    markers = api["ColdResumeMarkers"](tuple(i in cold_indices for i in range(count)), tuple(frontier))
    return NS(
        num_reqs=count,
        num_actual_tokens=actual,
        num_input_tokens=padded,
        max_query_len=max(widths),
        block_table_tensor=torch.zeros((count, 16), dtype=torch.int32),
        indexer_block_table_tensor=torch.ones((count, 16), dtype=torch.int32),
        slot_mapping=torch.arange(padded),
        indexer_slot_mapping=torch.arange(padded) + 100,
        positions=torch.arange(padded),
        prompt_lens_cpu=[22684] * count,
        query_start_loc_cpu=starts,
        query_start_loc=starts,
        num_computed_tokens_cpu=torch.tensor(computed, dtype=torch.int32),
        seq_lens=torch.tensor(np.array(computed) + widths, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(np.array(computed) + widths, dtype=torch.int32),
        request_ids=[f"r{i}" for i in range(count)],
        attn_state=state,
        cold_compact_resumes=markers,
    )


def route(api, metadata, widths):
    runner = NS(
        _staged_sfa_graph_capture_sizes=[4, 8, 16, 32],
        speculative_config=NS(num_speculative_tokens=1),
        attn_state="spec",
        decode_threshold=2,
        vllm_config=NS(lora_config=None),
        input_batch=NS(num_tokens_no_spec=metadata.num_computed_tokens_cpu.numpy() + 1),
    )
    return api["_staged_sfa_local_route"](
        runner,
        num_tokens_unpadded=sum(widths),
        num_reqs=len(widths),
        num_scheduled_tokens=np.array(widths),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=metadata.request_ids,
        kv_connector_metadata=(metadata.cold_compact_resumes.computed_ends, metadata.cold_compact_resumes),
        num_computed_tokens=metadata.num_computed_tokens_cpu.numpy(),
        prompt_lens=metadata.prompt_lens_cpu,
    )


def test_reported_recovery_keeps_native_route_and_valid_sparse_rows(api, builder):
    builder.build(0, common(api, [2] * 13, {11}))
    assert builder._dsa_fixed_layout_signature is not None
    widths = [2] * 11 + [1, 2]
    metadata = common(api, widths, {11}, padded=32)
    decision = route(api, metadata, widths)
    assert decision.action.value == "safe_native"
    assert decision.reason.value == "unsupported_batch"
    metadata.cold_compact_resumes = decision.cold_compact_resumes
    result = builder.build(0, metadata)

    owners = np.repeat(np.arange(13), widths).tolist()
    offsets = [offset for width in widths for offset in range(width)]
    assert result.num_decode_tokens == 25
    assert result.decode_req_indices.tolist() == owners + [-1] * 7
    assert result.decode_req_indices_compact.tolist() == owners
    assert result.decode_row_offsets.tolist() == offsets + [0] * 7
    assert result.decode_valid_row_indices.tolist() == list(range(25))
    assert not result.decode_valid_rows_all
    assert result.decode_request_ids_compact == metadata.request_ids
    assert api["_fixed_staged_decode_mtp"](result.decode_req_indices_cpu, 13, 32, pure_decode=True) is None
    assert result.block_table.data_ptr() == metadata.block_table_tensor.data_ptr()
    assert result.indexer_block_table.data_ptr() == metadata.indexer_block_table_tensor.data_ptr()
    assert torch.equal(result.block_table, metadata.block_table_tensor)
    assert torch.equal(result.indexer_block_table, metadata.indexer_block_table_tensor)
    assert torch.equal(result.slot_mapping, metadata.slot_mapping)
    assert torch.equal(result.indexer_slot_mapping, metadata.indexer_slot_mapping)
    boundary = api["_update_dsa_split_boundary_in_place"](result, [22684] * 13, 0)
    assert boundary.tolist() == [22684] * 25 + [0] * 7
    assert builder._dsa_fixed_layout_signature is None

    # Reuse the same storage on a subsequent uniform step; stale native rows
    # and padding must not leak into graph-compatible metadata.
    uniform = common(api, [2] * 13, {11})
    assert route(api, uniform, [2] * 13).action.value == "staged"
    next_result = builder.build(0, uniform)
    assert next_result.decode_req_indices.data_ptr() == result.decode_req_indices.data_ptr()
    assert next_result.decode_req_indices.tolist() == np.repeat(np.arange(13), 2).tolist()
    assert next_result.decode_row_offsets.tolist() == [0, 1] * 13
    assert builder._dsa_fixed_layout_signature is not None


@pytest.mark.parametrize("widths,cold", [([1], {0}), ([1, 2], {0}), ([2, 1], {1}), ([1, 2, 1], {0, 2}), ([2], {0})])
@pytest.mark.parametrize("frontier", [22683, 22684, 22780])
def test_resume_frontiers_cover_prompt_tail_and_generated_history(api, builder, widths, cold, frontier):
    metadata = common(api, widths, cold, computed=[frontier] * len(widths))
    # Keep the non-resuming requests strictly in decode.
    for i in range(len(widths)):
        if i not in cold:
            metadata.num_computed_tokens_cpu[i] = 23000
            metadata.seq_lens_cpu[i] = 23000 + widths[i]
    result = builder.build(0, metadata)
    assert result.num_decode_tokens == sum(widths)
    assert result.decode_req_indices.tolist() == np.repeat(np.arange(len(widths)), widths).tolist()
    assert result.decode_row_offsets.tolist() == [j for width in widths for j in range(width)]
    boundary = api["_update_dsa_split_boundary_in_place"](result, [frontier] * len(widths), 0)
    assert boundary.tolist() == [frontier] * sum(widths)


@pytest.mark.parametrize(
    "width,computed,frontier,state",
    [
        (0, 22684, 22684, "spec"),
        (3, 22684, 22684, "spec"),
        (1, 22683, 22684, "spec"),
        (1, 22685, 22684, "spec"),
        (2, 22683, 22684, "spec"),
        (2, 22684, 22684, "decode"),
    ],
)
def test_invalid_widths_and_stale_frontiers_still_fail(api, builder, width, computed, frontier, state):
    metadata = common(api, [width], {0}, computed=[computed], frontier=[frontier], state=state)
    with pytest.raises(RuntimeError, match="Invalid cold-compact resume layout"):
        builder.build(0, metadata)


def test_legacy_marker_still_requires_last_prompt_token(api, builder):
    metadata = common(api, [1], {0}, computed=[22683])
    metadata.cold_compact_resumes = (True,)
    assert builder.build(0, metadata).num_decode_tokens == 1
    metadata.num_computed_tokens_cpu[0] = 22684
    with pytest.raises(RuntimeError, match="Invalid cold-compact resume layout"):
        builder.build(0, metadata)


def test_non_speculative_single_row_keeps_fixed_metadata(api, builder):
    result = builder.build(0, common(api, [1], {0}, state="decode"))
    assert result.num_decode_tokens == 1
    assert result.decode_req_indices.tolist() == [0]
    assert result.decode_row_offsets.tolist() == [0]
    assert builder._dsa_fixed_layout_signature is not None


@pytest.mark.parametrize("markers,frontiers", [((True,), (22684,)), ((True, False), (22684,))])
def test_misaligned_proofs_still_fail(api, builder, markers, frontiers):
    metadata = common(api, [1, 2], {0})
    metadata.cold_compact_resumes = api["ColdResumeMarkers"](markers, (22684,) * len(markers))
    metadata.cold_compact_resumes.computed_ends = frontiers
    with pytest.raises(RuntimeError, match="(markers|frontiers).*(match|requests)"):
        builder.build(0, metadata)


@pytest.fixture
def native_api():
    ns = api.__wrapped__()
    ns["AscendAttentionState"] = NS(
        DecodeOnly="decode",
        SpecDecoding="spec",
        ChunkedPrefill="chunked",
        PrefillNoCache="empty",
        PrefillCacheHit="prefix",
    )
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "vllm_ascend/attention/utils.py").read_text(encoding="utf-8"))
    names = {"_dsa_remap_frontier", "staged_sfa_metadata_sparse_route", "native_sfa_cold_resume_layout"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    tree = ast.parse((root / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8"))
    nodes.append(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_build_attn_state"))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), "native_route", "exec"),
        ns,
    )
    return ns


def native_case(ns, widths, computed, prompts, histories, drafts, cold, captures=(4, 8, 16, 32)):
    metadata = common(ns, widths, cold, computed=computed)
    metadata.prompt_lens_cpu = prompts
    requests = []
    for i in range(len(widths)):
        spec = (
            NS(
                can_load=True,
                lmcache_cached_tokens=histories[i],
                dsa_committed_end=computed[i],
                dsa_remap_frontier=computed[i],
                dsa_scratch_capacity=4096,
                dsa_cold_compact_resume=True,
            )
            if i in cold
            else None
        )
        requests.append(
            NS(
                req_id=f"r{i}",
                is_sparse_decode=i in cold,
                dsa_current_released_frontier=0,
                dsa_nonresident_frontier=computed[i] if i in cold else 0,
                load_spec=spec,
            )
        )
    runner = NS(
        _staged_sfa_graph_capture_sizes=captures,
        speculative_config=NS(num_speculative_tokens=1, method="mtp"),
        decode_threshold=2,
        vllm_config=NS(lora_config=None),
        scheduler_config=NS(enable_chunked_prefill=True),
        input_batch=NS(num_tokens_no_spec=np.array(histories), num_computed_tokens_cpu=np.array(computed)),
    )
    ns["_build_attn_state"](runner, len(widths), np.array(widths), np.array(widths) - drafts)
    metadata.attn_state = runner.attn_state
    kwargs = dict(
        num_tokens_unpadded=sum(widths),
        num_reqs=len(widths),
        num_scheduled_tokens=np.array(widths),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=metadata.request_ids,
        kv_connector_metadata=NS(requests=requests),
        num_computed_tokens=computed,
        prompt_lens=prompts,
    )
    return runner, metadata, kwargs


@pytest.mark.parametrize("captures", [(4, 8, 16, 32), ()])
@pytest.mark.parametrize("prefill_metadata", [True, False])
def test_cold_admission_mixed_with_prefill_keeps_first_sparse_row(native_api, captures, prefill_metadata):
    # The scheduler can admit a full-hit request (one pending token + one
    # draft) alongside four uncached prompt tokens of another request.
    runner, metadata, kwargs = native_case(
        native_api,
        [2, 4],
        [22683, 0],
        [22684, 4],
        [22684, 4],
        np.array([1, 0]),
        {0},
        captures,
    )
    assert metadata.attn_state == "chunked"
    if not prefill_metadata:
        # A prefill chunk with no connector load/save has no ReqMeta entry.
        kwargs["kv_connector_metadata"].requests.pop()
    decision = native_api["_staged_sfa_local_route"](runner, **kwargs)
    assert decision.action.value == "safe_native"
    assert decision.reason.value == ("not_decode" if captures else "not_configured")
    metadata.cold_compact_resumes = decision.cold_compact_resumes
    result = builder.__wrapped__(native_api).build(0, metadata)
    assert result.decode_req_indices.tolist() == [0, 0, -1, -1, -1, -1]
    assert result.decode_row_offsets.tolist() == [0, 1, 0, 0, 0, 0]
    assert result.decode_valid_row_indices.tolist() == [0, 1]
    boundary = native_api["_update_dsa_split_boundary_in_place"](result, [22683, 0], 0)
    assert boundary.tolist() == [22683, 22683, 0, 0, 0, 0]


def test_graph_disabled_preserves_distinct_initial_and_resumed_frontiers(native_api):
    runner, metadata, kwargs = native_case(
        native_api,
        [2, 1],
        [22683, 22684],
        [22684, 22684],
        [22684, 22685],
        np.array([1, 0]),
        {0, 1},
        (),
    )
    decision = native_api["_staged_sfa_local_route"](runner, **kwargs)
    assert decision.reason.value == "not_configured"
    metadata.cold_compact_resumes = decision.cold_compact_resumes
    result = builder.__wrapped__(native_api).build(0, metadata)
    assert result.num_decode_tokens == 3
    assert decision.cold_compact_resumes.computed_ends == (22683, 22684)
    boundary = native_api["_update_dsa_split_boundary_in_place"](result, [22683, 22684], 0)
    assert boundary.tolist() == [22683, 22683, 22684]


def test_mixed_native_history_recovery_does_not_acquire_a_decode_width_limit(native_api):
    runner, metadata, kwargs = native_case(
        native_api,
        [2, 8],
        [22683, 22684],
        [22684, 22684],
        [22684, 22692],
        np.array([1, 0]),
        {0, 1},
    )
    assert metadata.attn_state == "chunked"
    decision = native_api["_staged_sfa_local_route"](runner, **kwargs)
    metadata.cold_compact_resumes = decision.cold_compact_resumes
    result = builder.__wrapped__(native_api).build(0, metadata)
    assert result.decode_req_indices.tolist() == [0, 0] + [1] * 8
    assert result.decode_row_offsets.tolist() == [0, 1] + list(range(8))
    assert result.num_decode_tokens == 10


def test_graph_route_does_not_enter_native_metadata_helper(native_api):
    def unexpected(*args, **kwargs):
        raise AssertionError("graph route entered native fallback metadata")

    native_api["native_sfa_cold_resume_layout"] = unexpected
    runner, metadata, kwargs = native_case(
        native_api,
        [2],
        [22683],
        [22684],
        [22684],
        np.array([1]),
        {0},
    )
    assert native_api["_staged_sfa_local_route"](runner, **kwargs).action.value == "staged"


@pytest.mark.parametrize("invalid", ["duplicate", "not_loadable", "beyond_cache", "wrong_computed"])
def test_native_cold_sources_retain_existing_validation(native_api, invalid):
    runner, metadata, kwargs = native_case(
        native_api,
        [2, 4],
        [22683, 0],
        [22684, 4],
        [22684, 4],
        np.array([1, 0]),
        {0},
    )
    requests = kwargs["kv_connector_metadata"].requests
    if invalid == "duplicate":
        requests.append(requests[0])
    elif invalid == "not_loadable":
        requests[0].load_spec.can_load = False
    elif invalid == "beyond_cache":
        requests[0].load_spec.lmcache_cached_tokens = 20000
    else:
        kwargs["num_computed_tokens"] = [22682, 0]
    with pytest.raises(RuntimeError, match="[Nn]ative cold-resume"):
        native_api["_staged_sfa_local_route"](runner, **kwargs)


def test_native_marker_mapping_ignores_inactive_and_non_cold_rows(native_api):
    runner, metadata, kwargs = native_case(
        native_api,
        [4, 2, 1],
        [0, 22683, 22684],
        [4, 22684, 22684],
        [4, 22684, 22685],
        np.array([0, 1, 0]),
        {1, 2},
    )
    requests = kwargs["kv_connector_metadata"].requests
    requests[:] = [requests[2], NS(req_id="inactive", load_spec=NS(dsa_cold_compact_resume=True)), requests[1]]
    decision = native_api["_staged_sfa_local_route"](runner, **kwargs)
    assert decision.frontiers == (0, 22683, 22684)
    assert decision.cold_compact_resumes == (False, True, True)
    metadata.cold_compact_resumes = decision.cold_compact_resumes
    result = builder.__wrapped__(native_api).build(0, metadata)
    assert result.decode_req_indices.tolist() == [-1, -1, -1, -1, 1, 1, 2]


def test_ordinary_native_decode_still_skips_connector_inspection(native_api):
    def unexpected(*args, **kwargs):
        raise AssertionError("ordinary native decode inspected connector metadata")

    native_api["unwrap_staged_sfa_connector_metadata"] = unexpected
    native_api["native_sfa_cold_resume_layout"] = unexpected
    runner, metadata, kwargs = native_case(
        native_api,
        [2],
        [22690],
        [22684],
        [22691],
        np.array([1]),
        set(),
        (),
    )
    decision = native_api["_staged_sfa_local_route"](runner, **kwargs)
    assert decision.reason.value == "not_configured"
    assert not decision.frontiers and not decision.cold_compact_resumes


def test_native_markers_follow_completed_load_handoff(native_api):
    root = Path(__file__).resolve().parents[3]
    source = root / "LMCache-NPU/lmcache/integration/vllm/vllm_v1_adapter.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_take_completed_cold_load")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(source), "exec"),
        native_api,
    )
    take_completed = native_api["_take_completed_cold_load"]
    runner, metadata, kwargs = native_case(
        native_api,
        [2, 4],
        [22683, 0],
        [22684, 4],
        [22684, 4],
        np.array([1, 0]),
        {0},
    )
    spec = kwargs["kv_connector_metadata"].requests[0].load_spec
    del spec.dsa_cold_compact_resume
    spec.dsa_cold_compact_load = True
    spec.can_load = False
    connector = NS(_dsa_cold_loaded_req_ids=set())
    assert not take_completed(connector, "r0", spec)
    assert native_api["native_sfa_cold_resume_layout"](
        kwargs["kv_connector_metadata"],
        metadata.request_ids,
        kwargs["num_computed_tokens"],
    ) == ((), ())
    assert not spec.can_load

    # The scheduler receives completion from the workers before this handoff.
    connector._dsa_cold_loaded_req_ids.add("r0")
    assert take_completed(connector, "r0", spec)
    assert spec.can_load and not hasattr(spec, "dsa_cold_compact_load")
    decision = native_api["_staged_sfa_local_route"](runner, **kwargs)
    assert decision.cold_compact_resumes.computed_ends == (22683, 0)
    metadata.cold_compact_resumes = decision.cold_compact_resumes
    result = builder.__wrapped__(native_api).build(0, metadata)
    assert result.decode_req_indices.tolist() == [0, 0, -1, -1, -1, -1]
    assert not take_completed(connector, "r0", spec)


def test_native_resume_ignores_same_request_save_control(native_api):
    runner, metadata, kwargs = native_case(
        native_api,
        [2, 4],
        [22683, 0],
        [22684, 4],
        [22684, 4],
        np.array([1, 0]),
        {0},
    )
    kwargs["kv_connector_metadata"].requests.append(
        NS(req_id="r0", is_decode_window_save=True, load_spec=NS(dsa_cold_compact_resume=True))
    )
    decision = native_api["_staged_sfa_local_route"](runner, **kwargs)
    assert decision.cold_compact_resumes == (True, False)
    assert decision.cold_compact_resumes.computed_ends == (22683, 0)
