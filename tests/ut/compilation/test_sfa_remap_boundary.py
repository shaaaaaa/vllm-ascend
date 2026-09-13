# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU comparison against the original remap helper, including cache lifetime."""

import ast
import importlib.util
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("remap_under_test", ROOT / "vllm_ascend/attention/sfa_remap_boundary.py")
remap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remap)


@pytest.fixture
def original():
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_prepare_sfa_remap_boundary", "_validate_dsa_scratch_capacity"}
    code = ast.parse("from __future__ import annotations")
    code.body.extend(n for n in tree.body if getattr(n, "name", None) in names)
    config = SimpleNamespace(window=256)
    namespace = {
        "torch": torch,
        "np": np,
        "_decode_window_save_window_size": lambda: config.window,
        "get_lmcache_sparse_cached_tokens": Mock(side_effect=AssertionError("unexpected connector lookup")),
    }
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    prepare = namespace["_prepare_sfa_remap_boundary"]
    return partial(prepare, reuse_unchanged=True), config


def metadata(rows, prompts, lengths, buffer=None, capacity=4096):
    tensor = torch.zeros(len(rows), dtype=torch.int32) if buffer is None else buffer.tensor[: len(rows)]
    return SimpleNamespace(
        decode_remap_boundary=tensor,
        decode_remap_boundary_ready=False,
        decode_remap_boundary_buffer=buffer,
        decode_req_indices_cpu=np.asarray(rows, dtype=np.int32),
        prompt_lens_cpu_rows=np.asarray(prompts, dtype=np.int32),
        seq_lens_cpu=torch.tensor(lengths),
        decode_scratch_capacity=capacity,
        decode_scratch_base_cpu=None,
    )


@pytest.mark.parametrize("window", [0, 256, 512])
@pytest.mark.parametrize("requests", [1, 4, 16])
def test_random_layouts_exactly_match_uncached_helper(original, window, requests):
    prepare, config = original
    config.window = window
    rng = np.random.default_rng(19)
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(requests * 2 + 3, dtype=torch.int32))
    for _ in range(40):
        # Includes non-contiguous request indices, Q1/Q2, lane reorder and pads.
        rows = [row for row in range(requests) for _ in range(int(rng.integers(0, 3)))]
        rows += [-1] * 3
        rng.shuffle(rows)
        prompts = [int(rng.choice((0, 4096, 8192))) for _ in rows]
        lengths = rng.integers(8193, 30000, size=requests).tolist()
        frontiers = tuple(int(rng.choice((0, 8192, 12288))) for _ in sorted(set(rows) - {-1}))
        old = metadata(rows, prompts, lengths)
        new = metadata(rows, prompts, lengths, buffer)
        expected = prepare(old, None, is_dummy_run=False, index_topk=2048, cached_tokens=frontiers)
        actual = prepare(new, None, is_dummy_run=False, index_topk=2048, cached_tokens=frontiers)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_511_decode_steps_upload_only_three_window_changes(original):
    prepare, _ = original
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32))
    address = buffer.tensor.data_ptr()
    for seq in range(5001, 5512):
        item = metadata([0, 0], [5000, 5000], [seq], buffer)
        prepare(item, None, is_dummy_run=False, index_topk=2048, cached_tokens=(30000,))
        assert item.decode_remap_boundary.tolist() == [(seq - 1) // 256 * 256] * 2
        assert item.decode_remap_boundary_ready
    assert buffer.upload_count == 3
    assert buffer.tensor.data_ptr() == address


def test_warm_update_uses_no_tensor_operators_or_copy():
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32))
    lengths = torch.tensor([5001])
    args = ([0, 0], [5000, 5000], lengths, (5000,), 256, 2048, 4096)
    buffer.update(*args)

    class NoTensorOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            pytest.fail(f"Unchanged boundary issued a tensor operation: {func}")

    with NoTensorOps():
        buffer.update(*args)
    assert buffer.upload_count == 1


def test_mode_handoff_invalidates_shadow_before_external_write(original):
    prepare, _ = original
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32))
    live = metadata([0, 0], [5000, 5000], [5200], buffer)
    prepare(live, None, is_dummy_run=False, index_topk=2048, cached_tokens=(5120,))
    assert buffer.tensor.tolist() == [5120, 5120]
    dummy = metadata([0, 0], [5000, 5000], [7000], buffer)
    prepare(dummy, None, is_dummy_run=True, index_topk=2048)
    assert buffer.tensor.tolist() == [6912, 6912]
    live.decode_remap_boundary_ready = False
    prepare(live, None, is_dummy_run=False, index_topk=2048, cached_tokens=(5120,))
    assert buffer.tensor.tolist() == [5120, 5120]
    assert buffer.upload_count == 2


def test_frontier_shrink_padding_prompts_and_capacity_switches():
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(4, dtype=torch.int32))
    buffer.update([0, 0, 1, -1], [10000] * 4, [15000, 17000], (12288, 16384), 256, 2048, 4096)
    assert buffer.tensor.tolist() == [12288, 12288, 16384, 10000]
    buffer.update([0], [10000], [15000], (8192,), 256, 2048, 4096)
    buffer.update([0, 0, 1, -1], [10000, 10000, 10000, 123], [15000, 17000], (8192, 0), 256, 2048, 4096)
    assert buffer.tensor.tolist() == [8192, 8192, 0, 123]
    assert buffer.upload_count == 3


@pytest.mark.parametrize("change", ["overlap", "too_many_rows", "scratch_growth", "frontier_count", "missing_request"])
def test_invalid_dynamic_inputs_cannot_hit_previous_value_cache(change):
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(3, dtype=torch.int32))
    args = [[0, 0], [5000, 5000], [5200], (5120,), 256, 2048, 4096]
    buffer.update(*args)
    if change == "overlap":
        args[3] = (2048,)
    elif change == "too_many_rows":
        args[0], args[1] = [0, 0, 0], [5000] * 3
    elif change == "scratch_growth":
        args[-1] = 8192
    elif change == "frontier_count":
        args[3] = ()
    else:
        args[2] = []
    with pytest.raises(RuntimeError):
        buffer.update(*args)
    assert buffer.tensor[:2].tolist() == [5120, 5120]


def test_partial_failed_upload_invalidates_old_shadow():
    class Destination:
        def __init__(self):
            self.fail = False
            self.writes = 0

        def numel(self):
            return 2

        def __getitem__(self, key):
            return self

        def copy_(self, value):
            self.writes += 1
            if self.fail:
                raise RuntimeError("copy failed")

    destination = Destination()
    buffer = remap.SFARemapBoundaryBuffer(destination)
    args = ([0, 0], [5000, 5000], [5200], (5120,), 256, 2048, 4096)
    buffer.update(*args)
    destination.fail = True
    with pytest.raises(RuntimeError, match="copy failed"):
        buffer.update([0, 0], [5000, 5000], [6000], (5632,), 256, 2048, 4096)
    destination.fail = False
    buffer.update(*args)
    assert destination.writes == 3 and buffer.upload_count == 2


def test_original_path_still_updates_each_step_and_invalidates_full_cache(original):
    prepare, _ = original
    # Use the underlying function, without the fixture's full-graph opt-in.
    uncached = prepare.func
    buffer = remap.SFARemapBoundaryBuffer(torch.zeros(2, dtype=torch.int32))
    item = metadata([0, 0], [5000, 5000], [5200], buffer)
    prepare(item, None, is_dummy_run=False, index_topk=2048, cached_tokens=(5120,))
    version = buffer.tensor._version
    for _ in range(2):
        item.decode_remap_boundary_ready = False
        uncached(item, None, is_dummy_run=False, index_topk=2048, cached_tokens=(5120,))
    assert buffer.tensor._version == version + 2
    item.decode_remap_boundary_ready = False
    prepare(item, None, is_dummy_run=False, index_topk=2048, cached_tokens=(5120,))
    assert buffer.upload_count == 2
