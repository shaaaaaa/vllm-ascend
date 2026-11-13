# SPDX-License-Identifier: Apache-2.0
"""Single-NPU old/new SFA query-copy and DCP-mask timings; excludes H2D and attention."""

import argparse
import json
import statistics
from functools import partial

import torch


def copy_planes(query, rope, query_out, rope_out):
    query_out.copy_(query)
    rope_out.copy_(rope)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--heads", type=int, default=8, help="rank-local query heads")
    parser.add_argument(
        "--dcp-size",
        type=int,
        choices=[1, 2, 4, 8],
        default=8,
        help="select applicable preparation paths; no collectives are run",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--unroll", type=int, default=16)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if min(*args.rows, args.heads, args.iterations, args.unroll, args.trials) < 1:
        parser.error("sizes and iteration counts must be positive")
    import torch_npu

    from vllm_ascend.ops.triton.sfa_dcp import _prepare_topk, prepare_sfa_dcp_topk
    from vllm_ascend.ops.triton.sfa_query import copy_sfa_query_bridge
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

    torch.npu.set_device(args.device)
    init_device_properties_triton()
    print(json.dumps({"torch": torch.__version__, "torch_npu": torch_npu.__version__, **vars(args)}), flush=True)
    for rows in args.rows:
        raw = torch.arange(rows * 2048, dtype=torch.int32, device="npu").reshape(rows, 1, 2048)
        positions = torch.full((rows,), 1024, dtype=torch.int64, device="npu")
        counts = torch.full((rows, 16), -77, dtype=torch.int32, device="npu")
        dtype = getattr(torch, args.dtype)
        # Match BMM's [heads, rows, latent] storage and the split RoPE view.
        query = torch.zeros((args.heads, rows, 512), device="npu", dtype=dtype).transpose(0, 1)
        rope_storage = torch.zeros((rows, args.heads, 576), device="npu", dtype=dtype)
        rope = rope_storage[..., 512:]
        outputs = {
            "topk": {name: (torch.empty_like(raw),) for name in ("old", "new")},
            "bridge": {
                name: (
                    torch.empty(query.shape, dtype=dtype, device="npu"),
                    torch.empty(rope.shape, dtype=dtype, device="npu"),
                )
                for name in ("old", "new")
            },
        }
        actions = {
            "topk": {
                "old": partial(
                    _prepare_topk[(min(rows, get_vectorcore_num()),)],
                    raw,
                    positions,
                    outputs["topk"]["old"][0],
                    counts,
                    None,
                    rows,
                    2048,
                    rows,
                    args.dcp_size,
                    0,
                    128,
                    False,
                ),
                "new": partial(
                    prepare_sfa_dcp_topk,
                    raw,
                    positions,
                    outputs["topk"]["new"][0],
                    counts,
                    rows,
                    args.dcp_size,
                    0,
                    128,
                    local=False,
                ),
            },
            "bridge": {
                "old": partial(copy_planes, query, rope, *outputs["bridge"]["old"]),
                "new": partial(copy_sfa_query_bridge, query, rope, *outputs["bridge"]["new"]),
            },
        }
        if args.dcp_size == 1:
            # Ordinary SFA has no separate DCP mask/copy to optimize.
            actions.pop("topk")
        for case, functions in actions.items():
            graphs = {}
            try:
                for name, action in functions.items():
                    for _ in range(3):
                        action()
                    torch.npu.synchronize()
                    graphs[name] = torch.npu.NPUGraph()
                    with torch.npu.graph(graphs[name]):
                        for _ in range(args.unroll):
                            action()
                for step in range(3):
                    positions.fill_(step * 512 - 1)
                    # Nonuniform values expose incorrect row/head permutations.
                    query.copy_(((torch.arange(query.numel()) + step) % 257).reshape(query.shape))
                    rope.copy_(((torch.arange(rope.numel()) + step) % 31).reshape(rope.shape))
                    for graph in graphs.values():
                        graph.replay()
                    torch.npu.synchronize()
                    for old, new in zip(outputs[case]["old"], outputs[case]["new"]):
                        assert torch.equal(old, new), f"{case} parity failed: rows={rows}, step={step}"
                    assert torch.all(counts.cpu() == -77)
                samples = {name: [] for name in functions}
                for trial in range(args.trials):
                    for name in tuple(functions) if trial % 2 == 0 else tuple(reversed(functions)):
                        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                        start.record()
                        for _ in range(args.iterations):
                            graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 1000 / (args.iterations * args.unroll))
                print(
                    json.dumps(
                        {
                            "case": case,
                            "rows": rows,
                            "parity": "passed",
                            "unit": "us",
                            "median": {n: statistics.median(s) for n, s in samples.items()},
                            "samples": samples,
                        }
                    ),
                    flush=True,
                )
            finally:
                torch.npu.synchronize()
                for graph in graphs.values():
                    graph.reset()


if __name__ == "__main__":
    main()
