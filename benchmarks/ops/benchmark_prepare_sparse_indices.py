import argparse
import time

import torch

from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
    prepare_sparse_indices,
)


def main(
    rows=4,
    requests=2,
    topk=2048,
    iterations=200,
    reuse=False,
    cold_reuse=False,
):
    reuse = reuse or cold_reuse
    source = torch.randint(
        0, 131072, (rows, 1, topk), dtype=torch.int32, device="npu"
    )
    if rows <= 0 or requests <= 0 or rows < requests:
        raise ValueError(
            "rows and requests must be positive, with at least one row per "
            f"request: rows={rows}, requests={requests}"
        )
    boundaries = torch.full((rows,), 131072, dtype=torch.int32, device="npu")
    row_requests = torch.arange(rows, dtype=torch.int32, device="npu") % requests
    rows_per_request = (rows + requests - 1) // requests
    scratch_capacity = rows_per_request * topk
    block_table = torch.arange(
        1,
        requests * 1024 + 1,
        dtype=torch.int32,
        device="npu",
    ).reshape(requests, 1024)
    selected = torch.empty(
        (requests, scratch_capacity), dtype=torch.int32, device="npu"
    )
    counts = torch.empty((requests, 16), dtype=torch.int32, device="npu")
    targets = torch.empty(
        (requests, scratch_capacity), dtype=torch.long, device="npu"
    )
    state_indices = torch.arange(requests, dtype=torch.int32, device="npu")
    generations = torch.ones(requests, dtype=torch.long, device="npu")
    resident = torch.full(
        (requests, scratch_capacity), -1, dtype=torch.int32, device="npu"
    )
    resident_generations = torch.full(
        (requests, 8), -1, dtype=torch.long, device="npu"
    )

    def run(values, launch_generations):
        prepare_sparse_indices(
            values,
            boundaries,
            row_req_indices=row_requests,
            request_block_table=block_table,
            selected_packed=selected,
            selected_counts=counts,
            target_slot_mapping=targets,
            block_size=128,
            request_state_indices=state_indices if reuse else None,
            request_generations=launch_generations if reuse else None,
            resident_token_ids=resident if reuse else None,
            resident_generations=resident_generations if reuse else None,
        )

    # The operator remaps top-k in place. Give each launch a distinct input so
    # the timed region measures the operator rather than an input-reset copy.
    warmup_inputs = [source.clone() for _ in range(20)]
    measured_inputs = [source.clone() for _ in range(iterations)]
    if cold_reuse:
        generation_inputs = [
            torch.full(
                (requests,), generation, dtype=torch.long, device="npu"
            )
            for generation in range(1, 20 + iterations + 1)
        ]
    else:
        generation_inputs = [generations] * (20 + iterations)
    for index, values in enumerate(warmup_inputs):
        run(values, generation_inputs[index])
    torch.npu.synchronize()
    started = time.perf_counter()
    for index, values in enumerate(measured_inputs, start=20):
        run(values, generation_inputs[index])
    torch.npu.synchronize()
    mode = (
        "reuse-cold-miss"
        if cold_reuse
        else ("reuse-steady-hit" if reuse else "baseline")
    )
    print(
        f"{mode}: "
        f"{(time.perf_counter() - started) * 1000 / iterations:.6f} ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--cold-reuse",
        action="store_true",
        help="change request generation every launch to measure all-miss reset",
    )
    args = parser.parse_args()
    main(
        rows=args.rows,
        requests=args.requests,
        topk=args.topk,
        iterations=args.iterations,
        reuse=args.reuse,
        cold_reuse=args.cold_reuse,
    )
