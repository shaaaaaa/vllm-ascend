# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real builder row-layout execution on CPU, without loading the NPU runtime."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sfa_test_support import definitions, extract
from torch.utils._python_dispatch import TorchDispatchMode

PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_v1.py"


def builder():
    namespace = {
        "np": np,
        "torch": torch,
        "AscendAttentionState": SimpleNamespace(DecodeOnly="decode", SpecDecoding="spec"),
        "envs": SimpleNamespace(VLLM_ASCEND_MTP_DW_DEEP_DIAG=False),
        "get_cos_and_sin_mla": lambda pos, _: (pos[:, None], pos[:, None]),
        "staged_sfa_connector_supports_sparse_load": lambda: True,
    }
    spec = importlib.util.spec_from_file_location("row_graph_layout", PATH.with_name("sfa_graph_layout.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace["FullGraphAttentionBuffers"] = module.FullGraphAttentionBuffers
    definitions(PATH, {"build"}, namespace, class_name="AscendSFAMetadataBuilder")
    result = type("RealRowBuilder", (), {"build": namespace["build"]})()
    result.dsa_shrink_latent, result.decode_threshold, result.scratch_capacity = 2, 2, 16
    result._dsa_max_num_rows, result._dsa_max_num_reqs = 32, 8
    result.enable_dsa_cp = False
    result._full_graph_tables = None
    result._dsa_fixed_layout_signature = None
    result._dsa_general_layout_signature = None
    result._dsa_general_decode_rows = 0
    result._dsa_fixed_query_starts_cpu = np.stack((np.arange(9), np.arange(9) * 2))
    result.metadata_cls = SimpleNamespace
    result.model_config = SimpleNamespace(get_head_size=lambda: 576)
    result.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda _: None)
    for field in (
        "prompt_lens",
        "split_boundary",
        "req_indices",
        "row_offsets",
        "current_positions",
        "valid_row_indices",
        "compact_req_indices",
    ):
        array = np.empty(32, dtype=np.int64 if field == "current_positions" else np.int32)
        setattr(result, "_dsa_" + field + "_cpu", array)
        setattr(result, "_dsa_" + field + "_cpu_tensor", torch.from_numpy(array))
    for field in (
        "_dsa_prompt_lens",
        "_dsa_split_boundary",
        "_dsa_req_indices",
        "_dsa_row_offsets",
        "decode_valid_row_indices",
        "decode_req_indices_compact",
        "decode_remap_boundary",
        "decode_scratch_base",
    ):
        setattr(result, field, torch.empty(32, dtype=torch.int32))
    for field in (
        "_dsa_selected_tokens",
        "_dsa_selected_counts",
        "_dsa_target_slots",
        "_dsa_union_mapping",
        "_dsa_shard_packed",
        "_dsa_shard_mapping",
        "_dsa_shard_counts",
    ):
        setattr(result, field, torch.empty((8, 16), dtype=torch.int32))
    result.decode_remap_boundary_buffer = object()
    return result, namespace


def common(widths=(1, 2, 1), *, padded=6, computed=(1024, 2048, 3072), prompts=(1024, 2048, 3072)):
    qsl = torch.tensor([0, *np.cumsum(widths)], dtype=torch.int32)
    reqs = len(widths)
    return SimpleNamespace(
        num_reqs=reqs,
        num_actual_tokens=sum(widths),
        num_input_tokens=padded,
        block_table_tensor=torch.zeros((reqs, 8), dtype=torch.int32),
        slot_mapping=torch.arange(padded),
        positions=torch.arange(padded),
        indexer_block_table_tensor=None,
        indexer_slot_mapping=None,
        prompt_lens_cpu=np.asarray(prompts),
        request_ids=[f"r{i}" for i in range(reqs)],
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        num_computed_tokens_cpu=torch.tensor(computed, dtype=torch.int32),
        seq_lens=torch.tensor(np.asarray(computed) + widths),
        seq_lens_cpu=torch.tensor(np.asarray(computed) + widths),
        cold_compact_resumes=(),
        attn_state="spec",
    )


def row_buffers(subject):
    return tuple(
        getattr(subject, name)
        for name in (
            "_dsa_prompt_lens",
            "_dsa_split_boundary",
            "_dsa_req_indices",
            "_dsa_row_offsets",
            "decode_valid_row_indices",
            "decode_req_indices_compact",
        )
    )


def test_ragged_decode_reuses_all_six_row_uploads():
    subject, _ = builder()
    item = common()
    first = subject.build(0, item)
    versions = tuple(t._version for t in row_buffers(subject))
    item.num_computed_tokens_cpu.add_(1)
    item.seq_lens.add_(1)
    item.seq_lens_cpu.add_(1)
    second = subject.build(0, item)
    assert second.num_decode_tokens == first.num_decode_tokens == 4
    assert second.decode_req_indices.tolist() == [0, 1, 1, 2, -1, -1]
    assert second.decode_row_offsets.tolist() == [0, 0, 1, 0, 0, 0]
    assert second.seq_lens.tolist() == item.seq_lens.tolist()
    assert tuple(t._version for t in row_buffers(subject)) == versions


def snapshot(item):
    fields = (
        "prompt_lens",
        "decode_req_indices",
        "decode_row_offsets",
        "decode_valid_row_indices",
        "decode_req_indices_compact",
        "split_boundary",
        "cum_query_lens",
        "seq_lens",
        "slot_mapping",
    )
    return {name: None if getattr(item, name) is None else getattr(item, name).clone() for name in fields}


@pytest.mark.parametrize("full_graph", [False, True])
def test_cache_matches_fresh_build_across_layout_switches(full_graph):
    rng = np.random.default_rng(738)
    subject, _ = builder()
    for step in range(100):
        widths = rng.integers(1, 3, size=3).tolist()
        prompts = [1024, 2048, 3072]
        rng.shuffle(prompts)
        item = common(widths, padded=int(sum(widths) + step % 3), computed=np.asarray(prompts) + step, prompts=prompts)
        if full_graph:
            item.sfa_full_graph = True
            item.indexer_block_table_tensor = item.block_table_tensor + 100
            item.indexer_slot_mapping = item.slot_mapping + 100
        fresh, _ = builder()
        expected = fresh.build(0, item)
        expected_fields = snapshot(expected)
        for _ in range(2):
            actual = subject.build(0, item)
            assert actual.num_decode_tokens == expected.num_decode_tokens
            for name, value in snapshot(actual).items():
                if value is None:
                    assert expected_fields[name] is None
                else:
                    torch.testing.assert_close(value, expected_fields[name], rtol=0, atol=0)


@pytest.mark.parametrize("path", ["prefill", "cold", "wide", "diagnostic"])
def test_noneligible_layouts_keep_rebuilding(path):
    subject, namespace = builder()
    item = common()
    subject.build(0, item)
    if path == "prefill":
        item.num_computed_tokens_cpu[1] = 2047
    elif path == "cold":
        item.num_computed_tokens_cpu[1] = 2047
        item.cold_compact_resumes = (False, True, False)
    elif path == "wide":
        item = common((3, 1, 1))
    else:
        namespace["envs"].VLLM_ASCEND_MTP_DW_DEEP_DIAG = True
    subject.build(0, item)
    versions = tuple(t._version for t in row_buffers(subject))
    actual = subject.build(0, item)
    assert subject._dsa_general_layout_signature is None
    assert all(t._version > version for t, version in zip(row_buffers(subject), versions))
    if path == "diagnostic":
        assert actual.decode_current_positions_cpu[:4].tolist() == [1024, 2048, 2049, 3072]


def test_legacy_native_boundary_writes_remain_consistent_on_cache_hit():
    subject, _ = builder()
    item = common()
    namespace = {"np": np, "torch": torch}
    update = extract(PATH, "_update_dsa_split_boundary_in_place", namespace)
    for step in range(3):
        actual = subject.build(0, item)
        fresh, _ = builder()
        expected = fresh.build(0, item)
        frontiers = [512 + step * 128, 1024 + step * 128, 2048 + step * 128]
        torch.testing.assert_close(update(actual, frontiers, 0), update(expected, frontiers, 0))
        assert actual.prompt_lens.tolist() == [1024, 2048, 2048, 3072, 0, 0]


def test_request_change_and_cold_validation_cannot_hit_prior_signature():
    subject, _ = builder()
    item = common()
    subject.build(0, item)
    versions = tuple(t._version for t in row_buffers(subject))
    item.request_ids[0] = "replacement"
    subject.build(0, item)
    assert all(t._version > version for t, version in zip(row_buffers(subject), versions))
    item.cold_compact_resumes = (True, False, False)
    with pytest.raises(RuntimeError, match="Invalid cold-compact"):
        subject.build(0, item)
    assert subject._dsa_general_layout_signature is None


def test_failed_upload_cannot_reuse_the_old_layout_signature():
    subject, _ = builder()
    old = common()
    subject.build(0, old)

    class FailRowCopy(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.copy_.default and args[0].data_ptr() == subject._dsa_req_indices.data_ptr():
                raise RuntimeError("upload failed")
            return func(*args, **(kwargs or {}))

    with FailRowCopy(), pytest.raises(RuntimeError, match="upload failed"):
        subject.build(0, common((2, 1, 1)))
    assert subject._dsa_general_layout_signature is None
    actual = subject.build(0, old)
    fresh, _ = builder()
    expected = fresh.build(0, old)
    for name, value in snapshot(actual).items():
        torch.testing.assert_close(value, snapshot(expected)[name])
