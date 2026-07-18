# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    _prepare_sparse_indices_torch,
)
from vllm_ascend.utils import enable_custom_op


def _measure(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--decode-rows", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if not 0 < args.decode_rows <= args.rows:
        parser.error("--decode-rows must be in [1, --rows]")

    enable_custom_op()
    try:
        fused_op = torch.ops._C_ascend.npu_dsa_prepare_sparse_indices_
    except AttributeError as exc:
        raise RuntimeError(
            "vllm_ascend_C does not contain "
            "npu_dsa_prepare_sparse_indices_; rebuild the extension before "
            "running this benchmark"
        ) from exc

    device = torch.device("npu")
    topk = torch.randint(
        -1,
        131646,
        (args.rows, 1, args.topk),
        dtype=torch.int32,
        device=device,
    )
    split_boundary = torch.zeros(args.rows, dtype=torch.int32, device=device)
    split_boundary[: args.decode_rows] = 131614
    valid_rows = torch.arange(args.decode_rows, dtype=torch.int32, device=device)
    row_req_indices = torch.full(
        (args.rows,), -1, dtype=torch.int32, device=device
    )
    row_req_indices[: args.decode_rows] = torch.arange(
        args.decode_rows, dtype=torch.int32, device=device
    )
    scratch_base = torch.zeros(args.rows, dtype=torch.int32, device=device)
    scratch_base[: args.decode_rows] = torch.arange(args.decode_rows, dtype=torch.int32, device=device) * args.topk

    expected_topk, expected_packed = _prepare_sparse_indices_torch(
        topk.clone(),
        split_boundary,
        scratch_base=scratch_base,
        valid_row_indices=valid_rows,
        row_req_indices=row_req_indices,
    )
    actual_topk = topk.clone()
    actual_packed = fused_op(
        actual_topk,
        split_boundary,
        valid_rows,
        scratch_base,
        True,
        row_req_indices,
    )
    torch.npu.synchronize()
    if not torch.equal(actual_topk, expected_topk):
        raise AssertionError("fused topk_indices differ from the Torch reference")
    if not torch.equal(actual_packed, expected_packed):
        raise AssertionError("fused selected_packed differs from the Torch reference")

    fused_topk = topk.clone()

    def fused():
        # Re-remapping is stable after the first iteration and keeps the same
        # selected count, so an input clone does not pollute each measurement.
        fused_op(
            fused_topk,
            split_boundary,
            valid_rows,
            scratch_base,
            True,
            row_req_indices,
        )

    def torch_reference():
        _prepare_sparse_indices_torch(
            topk,
            split_boundary,
            scratch_base=scratch_base,
            valid_row_indices=valid_rows,
            row_req_indices=row_req_indices,
        )

    fused_ms = _measure(fused, args.warmup, args.iterations)
    torch_reference_ms = _measure(torch_reference, args.warmup, args.iterations)
    print(
        f"shape=({args.rows}, 1, {args.topk}) "
        f"decode_rows={args.decode_rows} "
        f"padding_rows={args.rows - args.decode_rows} "
        "exact_match=True "
        f"fused_ms={fused_ms:.6f} "
        f"torch_reference_ms={torch_reference_ms:.6f} "
        f"speedup={torch_reference_ms / fused_ms:.2f}x"
    )


if __name__ == "__main__":
    main()
