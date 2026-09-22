# SPDX-License-Identifier: Apache-2.0
"""Staged capture bounds, independent of the full-graph feature."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def dummy_capture():
    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    ns = {"np": np}
    runner.max_model_len = 4399
    runner.kv_cache_config = SimpleNamespace(num_blocks=4, num_blocks_per_group=[4, 4])

    def table(width, block_size):
        result = SimpleNamespace(
            max_num_reqs=4,
            max_num_blocks_per_req=width,
            block_size=block_size,
            block_table=SimpleNamespace(np=np.zeros((4, width), dtype=np.int32)),
            num_blocks_per_row=np.zeros(4, dtype=np.int32),
            slot_mapping=SimpleNamespace(np=np.full(8, -1, dtype=np.int64)),
            commit_block_table=Mock(),
            commit_slot_mapping=Mock(),
        )

        def map_slots(requests, positions):
            blocks = result.block_table.np[requests, positions // block_size]
            result.slot_mapping.np[: positions.size] = blocks * block_size + positions % block_size

        result.compute_slot_mapping = map_slots
        return result

    runner.input_batch = SimpleNamespace(block_table=SimpleNamespace(block_tables=[table(36, 128), table(18, 256)]))
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_staged_sfa_dummy_seq_len", "_prepare_staged_sfa_dummy_block_tables"}
    ]
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    exec(compile(module, str(path), "exec"), ns)
    for method in methods:
        setattr(runner, method.name, ns[method.name].__get__(runner))
    dummy = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run")
    selection = next(
        node
        for node in ast.walk(dummy)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "seq_lens" for target in child.targets)
            for child in node.body
        )
        and any(isinstance(child, ast.Name) and child.id == "SEQ_LEN_WITH_MAX_PA_WORKSPACE" for child in ast.walk(node))
    )
    code = compile(ast.fix_missing_locations(ast.Module(body=[selection], type_ignores=[])), str(path), "exec")

    def select(**overrides):
        values = dict(
            ns,
            self=runner,
            staged_sfa_graph_dummy_run=True,
            staged_query_width=2,
            profile_seq_lens=None,
            is_graph_capturing=True,
            max_query_len=2,
            num_tokens=2,
            using_paged_attention=lambda *args: True,
            SEQ_LEN_WITH_MAX_PA_WORKSPACE=6144,
        )
        values.update(overrides)
        values["max_query_len"] = values["staged_query_width"]
        exec(code, values)
        return values["seq_lens"]

    return (runner, select)


@pytest.mark.parametrize("query_width", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_short_context_capture_positions_fit_both_groups(dummy_capture, query_width, batch_size):
    runner, select = dummy_capture
    seq_len = select(staged_query_width=query_width)
    assert seq_len == 4399
    positions = np.tile(np.arange(seq_len - query_width, seq_len, dtype=np.int64), batch_size)
    runner._prepare_staged_sfa_dummy_block_tables(batch_size=batch_size, positions=positions)
    for table in runner.input_batch.block_table.block_tables:
        slots = table.slot_mapping.np[: positions.size]
        assert np.unique(slots).size == positions.size
        assert np.all((slots >= 0) & (slots < batch_size * table.block_size))
        table.commit_block_table.assert_called_once_with(batch_size, force=True)


def test_capture_length_honors_smaller_second_kv_group(dummy_capture):
    runner, select = dummy_capture
    runner.max_model_len = 10000
    table = runner.input_batch.block_table.block_tables[1]
    table.block_table.np = np.zeros((4, 12), dtype=np.int32)
    table.max_num_blocks_per_req = 12
    assert select() == 3072
    runner._prepare_staged_sfa_dummy_block_tables(batch_size=1, positions=np.array([3070, 3071]))


def test_large_context_retains_original_capture_heuristic(dummy_capture):
    runner, select = dummy_capture
    runner.max_model_len = 10000
    for table in runner.input_batch.block_table.block_tables:
        table.block_table.np = np.zeros((4, 128), dtype=np.int32)
    assert select() == 6144


@pytest.mark.parametrize("dcp,pcp", [(2, 1), (1, 2), (2, 2)])
def test_capture_length_uses_logical_table_width_and_context_parallelism(dummy_capture, dcp, pcp):
    runner, select = dummy_capture
    runner.max_model_len = 10000
    for table in runner.input_batch.block_table.block_tables:
        table.block_table.np = np.zeros((4, 4), dtype=np.int32)
        table.max_num_blocks_per_req = 1
        table.dcp_world_size, table.pcp_world_size = (dcp, pcp)
    assert select() == 4 * 128 * dcp * pcp


@pytest.mark.parametrize("requested,expected", [(1, 2), (256, 256), (4399, 4399), (6144, 4399)])
def test_profiling_override_is_bounded_and_holds_complete_query(dummy_capture, requested, expected):
    _, select = dummy_capture
    assert select(profile_seq_lens=requested) == expected


@pytest.mark.parametrize("kind", ["model_too_short", "empty_table", "missing_group", "zero_length", "zero_query"])
def test_invalid_dummy_capacity_fails_before_mapping(dummy_capture, kind):
    runner, select = dummy_capture
    overrides = {}
    if kind == "model_too_short":
        runner.max_model_len = 1
    elif kind == "empty_table":
        runner.input_batch.block_table.block_tables[1].block_table.np = np.zeros((4, 0), dtype=np.int32)
    elif kind == "missing_group":
        runner.input_batch.block_table.block_tables.pop()
    elif kind == "zero_length":
        overrides["profile_seq_lens"] = 0
    else:
        overrides["staged_query_width"] = 0
    with pytest.raises((ValueError, RuntimeError), match="positive|two KV|capacity"):
        select(**overrides)


def test_original_out_of_range_dummy_positions_still_fail(dummy_capture):
    runner, _ = dummy_capture
    with pytest.raises(RuntimeError, match="max_position=6143, logical_capacity=4608"):
        runner._prepare_staged_sfa_dummy_block_tables(batch_size=1, positions=np.array([6142, 6143]))


@pytest.mark.parametrize("profile,capturing,expected", [(None, True, 6144), (None, False, 2), (123, True, 123)])
def test_non_sfa_paged_attention_and_profile_lengths_are_unchanged(dummy_capture, profile, capturing, expected):
    _, select = dummy_capture
    assert select(staged_sfa_graph_dummy_run=False, profile_seq_lens=profile, is_graph_capturing=capturing) == expected
