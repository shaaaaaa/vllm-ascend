# SPDX-License-Identifier: Apache-2.0
"""Compare legacy/fused DCP query preparation with real HCCL and ACL replay.

Run in a separate idle TP host with torchrun --nproc-per-node=8. No model,
KV retrieval, native attention, or MoE is included in these timings.
"""

import argparse
import json
import os
import statistics
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist


def legacy(query, rope, group):
    packed = group.all_gather(torch.cat((query, rope), dim=-1), dim=1)
    return tuple(t.contiguous() for t in packed.split((query.shape[-1], rope.shape[-1]), dim=-1))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--heads", type=int, default=64, help="global query heads")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--execution", choices=["graph", "eager"], default="graph")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    import torch_npu
    from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

    from vllm_ascend.ops.triton.sfa_dcp import _gather_sfa_dcp_queries

    world = int(os.environ["WORLD_SIZE"])
    if (
        world not in (2, 4, 8)
        or args.heads < 1
        or args.heads % world
        or min(*args.rows, args.iterations, args.trials) < 1
    ):
        parser.error("use DCP=2/4/8, divisible global heads and positive test sizes")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    from vllm_ascend.utils import create_hccl_pg_options

    device_group = dist.new_group(
        backend="hccl", pg_options=create_hccl_pg_options("dcp"), timeout=timedelta(seconds=120)
    )
    group = SimpleNamespace(world_size=world, device_group=device_group)
    group.all_gather = lambda packet, dim: DeviceCommunicatorBase.all_gather(group, packet, dim)
    if rank == 0:
        print(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "torch_npu": torch_npu.__version__,
                    "torch_npu_git": torch_npu.version.git_version,
                    "HCCL_OP_EXPANSION_MODE": os.getenv("HCCL_OP_EXPANSION_MODE"),
                    **vars(args),
                }
            ),
            flush=True,
        )
    functions = {"legacy": legacy, "fused": _gather_sfa_dcp_queries}
    for rows in args.rows:
        print(f"rank={rank} rows={rows} phase=warmup", flush=True)
        storage = torch.empty((rows, args.heads // world, 1152), dtype=getattr(torch, args.dtype), device="npu")
        query, rope = storage[..., :1024:2], storage[..., 1024::2]
        storage.zero_()
        graphs, outputs = {}, {}
        for name, function in functions.items():
            for _ in range(5):
                function(query, rope, group)
            torch.npu.synchronize()
            if args.execution == "graph":
                graph = torch.npu.NPUGraph()
                print(f"rank={rank} rows={rows} phase=capture variant={name}", flush=True)
                with torch.npu.graph(graph):
                    outputs[name] = function(query, rope, group)
                graphs[name] = graph

        def run(name, graphs=graphs, outputs=outputs, query=query, rope=rope):
            if graphs:
                graphs[name].replay()
                return outputs[name]
            return functions[name](query, rope, group)

        failures = []
        # Change input storage between replays. Raw halfwords cover signed zero,
        # NaN payloads and subnormals as well as rank/row/head permutations.
        for step in range(3):
            bits = torch.arange(storage.numel(), dtype=torch.int32) + rank * 1103 + step * 7919
            bits = (bits % 65536 - 32768).to(torch.int16).reshape(storage.shape)
            storage.view(torch.int16).copy_(bits)
            old, new = run("legacy"), run("fused")
            torch.npu.synchronize()
            failures.append(
                any(not torch.equal(a.view(torch.int16).cpu(), b.view(torch.int16).cpu()) for a, b in zip(old, new))
            )
        failed = torch.tensor([int(any(failures))], dtype=torch.int32)
        dist.all_reduce(failed, group=control)
        if failed.item():
            raise AssertionError(f"query parity failed: rows={rows} local_failures={failures}")

        measurements = {name: [] for name in functions}
        for trial in range(args.trials):
            # Alternate order to reduce systematic warm/cache ordering bias.
            for name in tuple(functions) if trial % 2 == 0 else tuple(reversed(functions)):
                dist.barrier(group=control)
                start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(args.iterations):
                    run(name)
                end.record()
                end.synchronize()
                measurements[name].append(start.elapsed_time(end) * 1000 / args.iterations)
        all_ranks = [None] * world
        dist.all_gather_object(all_ranks, measurements, group=control)
        if rank == 0:
            slowest = {
                name: [max(item[name][trial] for item in all_ranks) for trial in range(args.trials)]
                for name in functions
            }
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "parity": "bit_exact",
                        "unit": "us",
                        "slowest_rank_median": {n: statistics.median(v) for n, v in slowest.items()},
                        "slowest_rank_trials": slowest,
                        "per_rank_trials": all_ranks,
                    }
                ),
                flush=True,
            )
        torch.npu.synchronize()
        for graph in graphs.values():
            graph.reset()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
