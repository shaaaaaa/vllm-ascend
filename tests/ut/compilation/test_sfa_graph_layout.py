# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical regression tests for graph-executed ragged Q1/Q2 mapping."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_graph_layout.py"
_SPEC = importlib.util.spec_from_file_location("graph_layout_under_test", _PATH)
layout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(layout)


@pytest.mark.parametrize("capacity", [1, 4, 8, 12, 16])
@pytest.mark.parametrize("pattern", ["q1", "q2", "mixed"])
@pytest.mark.parametrize("padded", [False, True])
def test_topk_request_ownership_and_inverse(capacity, pattern, padded):
    count = max(1, capacity - 1) if padded else capacity
    widths = [1 if pattern == "q1" or (pattern == "mixed" and i % 2) else 2 for i in range(count)]
    total = sum(widths)
    ends = torch.tensor(widths + [0] * (capacity - count), dtype=torch.int32).cumsum(0)
    rows = torch.full((capacity * 2,), -1, dtype=torch.int32)
    rows[:total] = torch.repeat_interleave(torch.arange(count, dtype=torch.int32), torch.tensor(widths))
    topk = torch.arange(capacity * 2 * 7, dtype=torch.int32).reshape(capacity * 2, 1, 7)
    boundaries = torch.arange(capacity * 2, dtype=torch.int32) + 4096
    packed, boundary, owners, inverse = layout.pack_decode_lanes(topk, boundaries, rows, ends, 2)
    offset = 0
    for request, width in enumerate(widths):
        assert torch.equal(packed[request * 2 : request * 2 + width], topk[offset : offset + width])
        assert owners[request * 2 : request * 2 + width].eq(request).all()
        if width == 1:
            assert packed[request * 2 + 1].eq(-1).all()
        offset += width
    assert packed[owners < 0].eq(-1).all()
    assert boundary[owners < 0].eq(0).all()
    # Simulate a request-specific sparse planner changing the returned indices.
    result = layout.unpack_decode_lanes(packed + owners[:, None, None] * 1000, inverse, rows)
    assert torch.equal(result[:total], topk[:total] + rows[:total, None, None] * 1000)
    assert result[total:].eq(-1).all()


def test_attention_padding_preserves_causality_and_addresses():
    table = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    ends = torch.tensor([1, 3, 4, 4], dtype=torch.int32)
    lengths = torch.tensor([4097, 8194, 2001, 0], dtype=torch.int32)
    buffers = layout.FullGraphAttentionBuffers(16, table, table, ends, lengths)
    first = buffers.update(table, table, ends, lengths, lengths, 8)
    addresses = [tensor.data_ptr() for tensor in first]
    assert first[2].tolist() == [1, 3, 4, 4, 8]
    assert first[3].tolist() == [4097, 8194, 2001, 0, 0]
    assert first[0][-1].eq(0).all()
    # Replace requests and change Q1/Q2 lengths without changing graph inputs.
    second = buffers.update(table + 50, table + 80, ends + 1, lengths + 5, lengths + 5, 8)
    assert [tensor.data_ptr() for tensor in second] == addresses
    assert torch.equal(second[0][:-1], table + 50)
    assert second[3][-1] == 0


@pytest.mark.parametrize("requests", [1, 2, 8, 64])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("width", [1, 2])
def test_uniform_planner_bypasses_pack_and_matches_bounded(requests, padded, width):
    """Execute the actual pre-compute planner section, not a copied branch."""
    path = _PATH.with_name("sfa_v1.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_cross_layer_pre_compute")
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "graph_key" for t in n.targets)
    )
    # Graph-profile selection now precedes the shared consumer's early return.
    # Exercise the original planner section independently of indexer/query work.
    pack_start = next(i for i, n in enumerate(method.body)
                      if isinstance(n, ast.If) and ast.unparse(n.test) == "bounded_decode")
    code = compile(ast.Module(body=method.body[start:start + 3] + method.body[pack_start:-1], type_ignores=[]),
                   str(path), "exec")
    capacity = requests + int(padded)
    tokens = capacity * width
    owners = torch.arange(capacity, dtype=torch.int32).repeat_interleave(width)
    owners[requests * width :] = -1
    topk = torch.arange(tokens * 7, dtype=torch.int32).reshape(tokens, 1, 7)
    topk[owners < 0] = -1
    boundary = torch.full((tokens,), 4096, dtype=torch.int32)
    boundary[owners < 0] = 0
    outputs = []
    for profile in ("fixed", "bounded"):
        pack = Mock(wraps=layout.pack_decode_lanes)
        unpack = Mock(wraps=layout.unpack_decode_lanes)
        table = torch.zeros(capacity + int(profile == "bounded"), 3, dtype=torch.int32)

        def plan(indices, boundaries, request_rows, blocks, selected, counts, slots, *args, **kwargs):
            assert kwargs["staged_mtp"] == width
            assert blocks.shape[0] == capacity
            assert torch.equal(request_rows, owners)
            assert torch.equal(boundaries, boundary)
            # Request-specific update makes ownership mistakes observable.
            result = torch.where(request_rows[:, None, None] >= 0, indices + request_rows[:, None, None] * 1000, -1)
            return result, selected, counts, slots

        key = SimpleNamespace(query_profile=profile, request_capacity=capacity, max_query_len=width)
        ns = dict(
            get_forward_context=lambda key=key: SimpleNamespace(staged_sfa_graph_key=key),
            StagedSFAQueryProfile=SimpleNamespace(DECODE_BOUNDED="bounded"),
            pack_decode_lanes=pack,
            unpack_decode_lanes=unpack,
            self=SimpleNamespace(_prepare_decode_sparse_indices=plan),
            topk_indices=topk,
            resident_reads=None,
            resident_writes=None,
            remap_boundary=boundary,
            row_req_indices=owners,
            request_block_table=table,
            actual_seq_lengths_query=torch.arange(1, capacity + 1).clamp(max=requests) * width,
            **{
                name: torch.zeros(1)
                for name in (
                    "selected_packed",
                    "selected_counts",
                    "target_slot_mapping",
                    "request_state_indices",
                    "request_state_generations",
                    "local_to_union_workspace",
                    "shard_packed_workspace",
                    "shard_mapping_workspace",
                    "shard_counts_workspace",
                )
            },
        )
        exec(code, ns)
        assert pack.call_count == unpack.call_count == int(profile == "bounded")
        outputs.append(ns["topk_indices"])
    assert torch.equal(*outputs)


@pytest.mark.parametrize("bounded", [False, True])
def test_only_bounded_metadata_updates_extra_attention_tables(bounded):
    path = _PATH.with_name("sfa_v1.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and isinstance(n.test.func, ast.Name)
        and n.test.func.id == "getattr"
        and any(isinstance(a, ast.Constant) and a.value == "sfa_full_graph" for a in n.test.args)
    )
    values = tuple(torch.ones(1) for _ in range(5))
    buffers = SimpleNamespace(update=Mock(return_value=values))
    ns = dict(
        self=SimpleNamespace(_full_graph_tables=buffers),
        common_attn_metadata=SimpleNamespace(sfa_full_graph=bounded),
        num_input_tokens=2,
        **dict(zip(("block_table", "indexer_block_table", "cum_query_lens", "seq_lens", "seq_lens_cpu"), values)),
    )
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec"), ns)
    assert buffers.update.call_count == int(bounded)
