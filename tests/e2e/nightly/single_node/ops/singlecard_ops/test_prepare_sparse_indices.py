# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    _prepare_sparse_indices_torch,
    prepare_sparse_indices,
)
from vllm_ascend.utils import enable_custom_op

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is required",
)


def _reference(
    topk,
    split_boundary,
    valid_rows,
    scratch_base,
    row_req_indices=None,
):
    return _prepare_sparse_indices_torch(
        topk,
        split_boundary,
        scratch_base=scratch_base,
        valid_row_indices=valid_rows,
        row_req_indices=row_req_indices,
    )


def _measure_ms(fn, warmup=20, iterations=200):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


@pytest.mark.parametrize(
    "rows,k,valid",
    [
        pytest.param(4, 64, [2, 0], id="narrow-ordered-subset"),
        (32, 512, [0, 3, 7, 12, 20, 31]),
        pytest.param(
            4,
            2048,
            [0, 1, 2, 3],
            id="typical-four-decode-rows",
        ),
        (4096, 2048, [0, 1]),
        pytest.param(3, 4096, [2], id="maximum-row-width"),
    ],
)
def test_npu_dsa_prepare_sparse_indices_matches_torch_reference_exactly(
    rows, k, valid
):
    enable_custom_op()
    try:
        fused_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_
    except AttributeError:
        pytest.skip(
            "vllm_ascend_C was built without npu_dsa_prepare_sparse_indices_"
        )

    generator = torch.Generator().manual_seed(20260716)
    topk_cpu = torch.randint(
        low=-8,
        high=131646,
        size=(rows, 1, k),
        dtype=torch.int32,
        generator=generator,
    )
    split_boundary_cpu = torch.zeros(rows, dtype=torch.int32)
    scratch_base_cpu = torch.zeros(rows, dtype=torch.int32)
    for offset, row in enumerate(valid):
        split_boundary_cpu[row] = 131614
        scratch_base_cpu[row] = offset * k
        topk_cpu[row, 0, :9] = torch.tensor(
            [-7, -1, 0, 0, 42, 42, 131613, 131614, 131621],
            dtype=torch.int32,
        )
    valid_rows_cpu = torch.tensor(valid, dtype=torch.int32)

    topk_npu = topk_cpu.npu()
    split_boundary_npu = split_boundary_cpu.npu()
    valid_rows_npu = valid_rows_cpu.npu()
    scratch_base_npu = scratch_base_cpu.npu()
    expected_topk, expected_packed = _reference(topk_npu.clone(), split_boundary_npu, valid_rows_npu, scratch_base_npu)
    actual_topk = topk_npu.clone()
    actual_packed = fused_op(
        actual_topk,
        split_boundary_npu,
        valid_rows_npu,
        scratch_base_npu,
        True,
    )
    torch.npu.synchronize()

    assert torch.equal(actual_topk, expected_topk)
    assert torch.equal(actual_packed, expected_packed)


def test_npu_dsa_prepare_sparse_indices_typical_shape_performance():
    enable_custom_op()
    try:
        fused_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_
    except AttributeError:
        pytest.skip(
            "vllm_ascend_C was built without npu_dsa_prepare_sparse_indices_"
        )

    rows = 4
    decode_rows = 2
    topk = 2048
    generator = torch.Generator().manual_seed(20260717)
    topk_cpu = torch.randint(
        -8,
        131646,
        (rows, 1, topk),
        dtype=torch.int32,
        generator=generator,
    )
    split_boundary_cpu = torch.zeros(rows, dtype=torch.int32)
    split_boundary_cpu[:decode_rows] = 131614
    split_boundary_npu = split_boundary_cpu.npu()
    valid_rows_npu = torch.arange(decode_rows, dtype=torch.int32).npu()
    row_req_indices_cpu = torch.full((rows,), -1, dtype=torch.int32)
    row_req_indices_cpu[:decode_rows] = torch.arange(
        decode_rows, dtype=torch.int32
    )
    row_req_indices_npu = row_req_indices_cpu.npu()
    scratch_base_cpu = torch.zeros(rows, dtype=torch.int32)
    scratch_base_cpu[:decode_rows] = (
        torch.arange(decode_rows, dtype=torch.int32) * topk
    )
    scratch_base_npu = scratch_base_cpu.npu()
    topk_npu = topk_cpu.npu()

    expected_topk, expected_packed = _reference(
        topk_npu.clone(),
        split_boundary_npu,
        valid_rows_npu,
        scratch_base_npu,
        row_req_indices=row_req_indices_npu,
    )
    actual_topk = topk_npu.clone()
    actual_packed = fused_op(
        actual_topk,
        split_boundary_npu,
        valid_rows_npu,
        scratch_base_npu,
        True,
        row_req_indices_npu,
    )
    torch.npu.synchronize()
    assert torch.equal(actual_topk, expected_topk)
    assert torch.equal(actual_packed, expected_packed)

    fused_topk = topk_npu.clone()

    def fused():
        fused_op(
            fused_topk,
            split_boundary_npu,
            valid_rows_npu,
            scratch_base_npu,
            True,
            row_req_indices_npu,
        )

    def torch_reference():
        _reference(
            topk_npu,
            split_boundary_npu,
            valid_rows_npu,
            scratch_base_npu,
            row_req_indices=row_req_indices_npu,
        )

    fused_ms = _measure_ms(fused)
    torch_ms = _measure_ms(torch_reference)
    speedup = torch_ms / fused_ms
    print(
        "[prepare_sparse_indices_perf] "
        f"shape=({rows}, 1, {topk}) "
        f"decode_rows={decode_rows} "
        f"padding_rows={rows - decode_rows} "
        "exact_match=True "
        f"fused_ms={fused_ms:.6f} "
        f"torch_ms={torch_ms:.6f} "
        f"speedup={speedup:.2f}x"
    )
    assert fused_ms < torch_ms


def test_public_prepare_sparse_indices_dispatch_matches_torch_reference_exactly():
    enable_custom_op()
    if not hasattr(torch.ops._C_ascend, "npu_dsa_prepare_sparse_indices_"):
        pytest.skip(
            "vllm_ascend_C was built without npu_dsa_prepare_sparse_indices_"
        )

    topk_cpu = torch.arange(4 * 64, dtype=torch.int32).reshape(4, 1, 64)
    topk_cpu[1, 0, :8] = torch.tensor([-3, -1, 0, 31, 32, 63, 64, 127], dtype=torch.int32)
    topk_cpu[3, 0, :8] = torch.tensor([-2, 0, 1, 1, 95, 96, 127, 128], dtype=torch.int32)
    split_boundary_cpu = torch.tensor([0, 64, 0, 96], dtype=torch.int32)
    valid_rows_cpu = torch.tensor([3, 1], dtype=torch.int32)
    scratch_base_cpu = torch.tensor([0, 64, 0, 128], dtype=torch.int32)
    topk_npu = topk_cpu.npu()
    split_boundary_npu = split_boundary_cpu.npu()
    valid_rows_npu = valid_rows_cpu.npu()
    scratch_base_npu = scratch_base_cpu.npu()
    expected_topk, expected_packed = _reference(topk_npu.clone(), split_boundary_npu, valid_rows_npu, scratch_base_npu)

    actual_input = topk_npu.clone()
    assert actual_input.data_ptr() % 256 == 0
    input_ptr = actual_input.data_ptr()
    actual_topk, actual_packed = prepare_sparse_indices(
        actual_input,
        split_boundary_npu,
        scratch_base=scratch_base_npu,
        valid_row_indices=valid_rows_npu,
    )
    torch.npu.synchronize()

    # The fused path updates complete source rows in place. A fallback would
    # return the cloned index-copy result instead.
    assert actual_topk.data_ptr() == input_ptr
    assert torch.equal(actual_topk, expected_topk)
    assert torch.equal(actual_packed, expected_packed)


def test_npu_dsa_prepare_sparse_indices_all_and_none_selected():
    enable_custom_op()
    try:
        fused_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_
    except AttributeError:
        pytest.skip(
            "vllm_ascend_C was built without npu_dsa_prepare_sparse_indices_"
        )

    topk_cpu = torch.stack(
        (
            torch.arange(64, dtype=torch.int32),
            torch.arange(64, 128, dtype=torch.int32),
        )
    )
    split_boundary_cpu = torch.tensor([64, 0], dtype=torch.int32)
    valid_rows_cpu = torch.tensor([0, 1], dtype=torch.int32)
    scratch_base_cpu = torch.tensor([128, 0], dtype=torch.int32)
    topk_npu = topk_cpu.npu()
    split_boundary_npu = split_boundary_cpu.npu()
    valid_rows_npu = valid_rows_cpu.npu()
    scratch_base_npu = scratch_base_cpu.npu()
    expected_topk, expected_packed = _reference(topk_npu.clone(), split_boundary_npu, valid_rows_npu, scratch_base_npu)

    actual_topk = topk_npu.clone()
    actual_packed = fused_op(
        actual_topk,
        split_boundary_npu,
        valid_rows_npu,
        scratch_base_npu,
        True,
    )
    torch.npu.synchronize()

    assert torch.equal(actual_topk, expected_topk)
    assert torch.equal(actual_packed, expected_packed)


def test_npu_dsa_prepare_sparse_indices_without_packed_matches_reference():
    enable_custom_op()
    try:
        fused_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_
    except AttributeError:
        pytest.skip(
            "vllm_ascend_C was built without npu_dsa_prepare_sparse_indices_"
        )

    topk_cpu = torch.arange(3 * 64, dtype=torch.int32).reshape(3, 64)
    split_boundary_cpu = torch.tensor([0, 80, 160], dtype=torch.int32)
    valid_rows_cpu = torch.tensor([2, 1], dtype=torch.int32)
    scratch_base_cpu = torch.tensor([0, 128, 256], dtype=torch.int32)
    topk_npu = topk_cpu.npu()
    split_boundary_npu = split_boundary_cpu.npu()
    valid_rows_npu = valid_rows_cpu.npu()
    scratch_base_npu = scratch_base_cpu.npu()
    expected_topk, expected_packed = _prepare_sparse_indices_torch(
        topk_npu.clone(),
        split_boundary_npu,
        need_packed=False,
        scratch_base=scratch_base_npu,
        valid_row_indices=valid_rows_npu,
    )

    direct_topk = topk_npu.clone()
    direct_packed = fused_op(
        direct_topk,
        split_boundary_npu,
        valid_rows_npu,
        scratch_base_npu,
        False,
    )
    actual_topk, actual_packed = prepare_sparse_indices(
        topk_npu.clone(),
        split_boundary_npu,
        need_packed=False,
        scratch_base=scratch_base_npu,
        valid_row_indices=valid_rows_npu,
    )
    torch.npu.synchronize()

    assert expected_packed is None
    assert direct_packed.shape == (0, 64)
    assert torch.equal(direct_topk, expected_topk)
    assert actual_packed is None
    assert torch.equal(actual_topk, expected_topk)


def test_npu_dsa_prepare_sparse_indices_large_int32_boundary_matches_exactly():
    enable_custom_op()
    try:
        fused_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_
    except AttributeError:
        pytest.skip(
            "vllm_ascend_C was built without npu_dsa_prepare_sparse_indices_"
        )

    split_boundary_value = 16_777_217
    topk_cpu = torch.full((1, 64), split_boundary_value + 8, dtype=torch.int32)
    topk_cpu[0, :8] = torch.tensor(
        [
            -1,
            0,
            split_boundary_value - 2,
            split_boundary_value - 1,
            split_boundary_value,
            split_boundary_value + 1,
            42,
            42,
        ],
        dtype=torch.int32,
    )
    split_boundary_npu = torch.tensor([split_boundary_value], dtype=torch.int32).npu()
    valid_rows_npu = torch.tensor([0], dtype=torch.int32).npu()
    scratch_base_npu = torch.tensor([64], dtype=torch.int32).npu()
    topk_npu = topk_cpu.npu()
    expected_topk, expected_packed = _reference(
        topk_npu.clone(),
        split_boundary_npu,
        valid_rows_npu,
        scratch_base_npu,
    )

    actual_topk = topk_npu.clone()
    actual_packed = fused_op(
        actual_topk,
        split_boundary_npu,
        valid_rows_npu,
        scratch_base_npu,
        True,
    )
    torch.npu.synchronize()

    assert torch.equal(actual_topk, expected_topk)
    assert torch.equal(actual_packed, expected_packed)


def test_npu_dsa_prepare_sparse_indices_empty_valid_rows_matches_reference():
    enable_custom_op()
    try:
        fused_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_
    except AttributeError:
        pytest.skip(
            "vllm_ascend_C was built without npu_dsa_prepare_sparse_indices_"
        )

    topk_cpu = torch.arange(3 * 64, dtype=torch.int32).reshape(3, 64)
    split_boundary_cpu = torch.zeros(3, dtype=torch.int32)
    valid_rows_cpu = torch.empty(0, dtype=torch.int32)
    scratch_base_cpu = torch.tensor([0, 64, 128], dtype=torch.int32)
    topk_npu = topk_cpu.npu()
    split_boundary_npu = split_boundary_cpu.npu()
    valid_rows_npu = valid_rows_cpu.npu()
    scratch_base_npu = scratch_base_cpu.npu()
    expected_topk, expected_packed = _reference(topk_npu.clone(), split_boundary_npu, valid_rows_npu, scratch_base_npu)

    actual_topk = topk_npu.clone()
    actual_packed = fused_op(
        actual_topk,
        split_boundary_npu,
        valid_rows_npu,
        scratch_base_npu,
        True,
    )

    assert torch.equal(actual_topk, expected_topk)
    assert torch.equal(actual_packed, expected_packed)
    assert actual_packed.shape == (0, 64)


@pytest.mark.parametrize("actual_rows", [0, 3])
def test_npu_dsa_prepare_sparse_indices_fuses_graph_padding_zero(actual_rows):
    enable_custom_op()
    try:
        fused_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_
    except AttributeError:
        pytest.skip(
            "vllm_ascend_C was built without npu_dsa_prepare_sparse_indices_"
        )

    rows, topk = 8, 64
    topk_cpu = torch.arange(rows * topk, dtype=torch.int32).reshape(rows, 1, topk)
    split_boundary_cpu = torch.zeros(rows, dtype=torch.int32)
    split_boundary_cpu[:actual_rows] = 128
    valid_rows_cpu = torch.arange(actual_rows, dtype=torch.int32)
    row_req_indices_cpu = torch.full((rows,), -1, dtype=torch.int32)
    row_req_indices_cpu[:actual_rows] = torch.arange(
        actual_rows, dtype=torch.int32
    )
    scratch_base_cpu = torch.arange(rows, dtype=torch.int32) * topk

    topk_npu = topk_cpu.npu()
    split_boundary_npu = split_boundary_cpu.npu()
    valid_rows_npu = valid_rows_cpu.npu()
    row_req_indices_npu = row_req_indices_cpu.npu()
    scratch_base_npu = scratch_base_cpu.npu()
    expected_topk, expected_packed = _reference(
        topk_npu.clone(),
        split_boundary_npu,
        valid_rows_npu,
        scratch_base_npu,
        row_req_indices=row_req_indices_npu,
    )

    actual_topk = topk_npu.clone()
    actual_packed = fused_op(
        actual_topk,
        split_boundary_npu,
        valid_rows_npu,
        scratch_base_npu,
        True,
        row_req_indices_npu,
    )
    torch.npu.synchronize()

    assert torch.equal(actual_topk, expected_topk)
    assert torch.equal(actual_packed, expected_packed)
    assert torch.count_nonzero(actual_topk[actual_rows:]).item() == 0
