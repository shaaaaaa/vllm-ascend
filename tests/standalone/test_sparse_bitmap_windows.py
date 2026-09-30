# SPDX-License-Identifier: Apache-2.0
"""CPU model of the windowed algorithm, checked against the production oracle.

This checks arithmetic/in-place semantics, not AscendC execution or barriers.
"""
import ast
from pathlib import Path
import re

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "csrc/kernels/prepare_sparse_indices_limits.h"
WORDS = int(re.search(r"DSA_BITMAP_WINDOW_WORDS = (\d+)", HEADER.read_text()).group(1))


def reference():
    path = ROOT / "vllm_ascend/distributed/kv_transfer/sparse_offload/prepare_sparse_indices.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_prepare_sparse_indices_torch")
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def windowed(source, boundaries, rows, table, need_packed, clear, work=None):
    result = source.copy()
    capacity = source.shape[1] * max(1, max(sum(rows == r) for r in range(len(table))))
    assert capacity <= WORDS * 32
    selected = np.zeros((len(table), capacity), np.int32)
    targets = np.zeros((len(table), capacity), np.int64)
    counts = np.zeros(len(table), np.int32)
    bitmap_words = (table.shape[1] * 128 + 31) // 32
    for req in range(len(table)):
        count = 0
        request_rows = np.flatnonzero(rows == req)
        boundary = max(0, max((int(boundaries[r]) for r in request_rows), default=0))
        active_words = min(bitmap_words, (boundary + 31) // 32)
        for base in range(0, active_words, WORDS):
            size = min(WORDS, active_words - base)
            if work is not None:
                work["windows"] = work.get("windows", 0) + 1
                work["prefix_words"] = work.get("prefix_words", 0) + size
                work["clear_words"] = work.get("clear_words", 0) + (size + 7) // 8 * 8
            bitmap = [0] * size
            for row in request_rows:
                if boundaries[row] <= base * 32:
                    continue
                for token in result[row]:
                    token = int(token)
                    word = ((token >> 5) - base) & 0xFFFFFFFF
                    if 0 <= token < boundaries[row] and word < size:
                        bitmap[word] |= 1 << (token & 31)
            prefix = [None] * size  # Empty words must never read stale prefix data.
            for word, bits in enumerate(bitmap):
                if bits:
                    prefix[word] = count
                    count += bits.bit_count()
                    if work is not None:
                        work["popcounts"] = work.get("popcounts", 0) + 1
            for row in request_rows:
                if boundaries[row] <= base * 32:
                    continue
                for col, token in enumerate(result[row]):
                    token = int(token)
                    word = ((token >> 5) - base) & 0xFFFFFFFF
                    if 0 <= token < boundaries[row] and word < size:
                        rank = prefix[word] + (bitmap[word] & ((1 << (token & 31)) - 1)).bit_count()
                        result[row, col] = rank
                        if need_packed:
                            selected[req, rank] = token
                            targets[req, rank] = int(table[req, rank // 128]) * 128 + rank % 128
        counts[req] = count
    if clear:
        result[rows < 0] = 0
    return result, selected if need_packed else None, counts if need_packed else None, targets if need_packed else None


@pytest.mark.parametrize("context", [140032, 200192, 524288, 524416, 1048576])
@pytest.mark.parametrize("need_packed,clear", [(False, False), (False, True), (True, False), (True, True)])
def test_windowed_union_matches_oracle(context, need_packed, clear):
    rng = np.random.default_rng(context)
    rows = np.array([0, 1, 0, -1, 1], np.int32)
    source = rng.integers(-1, context, (5, 64), dtype=np.int32)
    points = [-1, 0, 31, 32, min(context - 1, WORDS * 32 - 1), min(context - 1, WORDS * 32), context - 1]
    source[:, :len(points)] = points
    source[:, 8:15] = points
    table = rng.integers(1, 100, (2, context // 128), dtype=np.int32)
    for boundaries in (np.zeros(5, np.int32), np.array([context, context // 2, context, 0, 0], np.int32)):
        expected = reference()(*(torch.from_numpy(x) for x in (source, boundaries, rows, table)), 128, need_packed, clear)
        actual = windowed(source, boundaries, rows, table, need_packed, clear)
        for x, y in zip(expected, actual):
            assert (x is None and y is None) or np.array_equal(x.numpy(), y)


def test_one_million_workspace_is_bounded_and_short_context_is_unchanged():
    assert WORDS * 4 * 2 == 128 * 1024 < 192 * 1024
    for words in (4376, 6256, 16384):
        assert min((words + 7) // 8 * 8, WORDS) == (words + 7) // 8 * 8
    source = (ROOT / "csrc/kernels/prepare_sparse_indices.cpp").read_text()
    assert "ProcessWindow<false>" in source and "ProcessWindow<true>" in source
    assert "SyncPipeline<AscendC::HardEvent::S_V>();" in source


def test_full_q2_scratch_union_spans_both_windows():
    context = 1048576
    source = np.stack((np.arange(2048), np.arange(context - 2048, context))).astype(np.int32)
    boundaries = np.full(2, context, np.int32)
    rows = np.zeros(2, np.int32)
    table = np.arange(context // 128, dtype=np.int32)[None, :] + 19
    expected = reference()(*(torch.from_numpy(x) for x in (source, boundaries, rows, table)), 128, True, True)
    actual = windowed(source, boundaries, rows, table, True, True)
    assert actual[2].tolist() == [4096]
    for x, y in zip(expected, actual):
        assert np.array_equal(x.numpy(), y)


@pytest.mark.parametrize("boundary", [0, 1, 31, 32, 33, 255, 256, 257, 20000, 524287, 524288, 524289, 1048576])
@pytest.mark.parametrize("need_packed,clear", [(False, False), (False, True), (True, False), (True, True)])
def test_short_live_ranges_in_padded_one_million_table(boundary, need_packed, clear):
    rng = np.random.default_rng(boundary)
    rows = np.array([0, 1, -1, 0, 1, 0], np.int32)
    table = rng.integers(1, 100, (3, 8208), dtype=np.int32)  # Third request has no rows.
    boundaries = np.array([boundary, 20000, 1048576, max(0, boundary - 1), -1, 0], np.int32)
    source = rng.integers(-1, 1048576, (6, 64), dtype=np.int32)
    source[:, :10] = [-1, 0, 31, 32, 19999, 20000, max(0, boundary - 1), boundary, 524287, 524288]
    source[:, 10:20] = source[:, :10]
    expected = reference()(*(torch.from_numpy(x) for x in (source, boundaries, rows, table)), 128, need_packed, clear)
    actual = windowed(source, boundaries, rows, table, need_packed, clear)
    for x, y in zip(expected, actual):
        assert (x is None and y is None) or np.array_equal(x.numpy(), y)


def test_short_request_work_is_independent_of_unused_table_capacity():
    source = np.array([[0, 31, 19999, 19999, 20000, -1]], np.int32)
    work = {}
    actual = windowed(source, np.array([20000]), np.array([0]), np.ones((1, 8208), np.int32), True, True, work)
    assert actual[2].tolist() == [3]
    assert work == dict(windows=1, prefix_words=625, clear_words=632, popcounts=2)
