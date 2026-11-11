# SPDX-License-Identifier: Apache-2.0
"""Isolate DCP-shaped HCCL capture/replay failures without a model or native SFA.

Use a fresh torchrun invocation per case. Phase markers identify submission,
capture, replay, synchronization, and parity failures on each rank. This probe
does not reproduce MC2 dispatch, native sparse attention, or LMCache DMA.
"""

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["gather", "exchange", "sequence", "alternating"], default="sequence")
    parser.add_argument("--execution", choices=["eager", "graph"], default="graph")
    parser.add_argument("--dcp-mode", choices=["inherit", "HOST", "AI_CPU", "AIV"], default="inherit")
    parser.add_argument(
        "--exchange-api",
        choices=["fixed", "variable"],
        default="fixed",
        help="equal payloads through HcclAlltoAll or HcclAlltoAllV",
    )
    parser.add_argument("--overlap-copy", action="store_true", help="add an event-joined side-stream copy")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--heads", type=int, default=64, help="global heads")
    parser.add_argument("--buffer-mb", type=int, default=200, help="production default non-MC2 HCCL buffer")
    args = parser.parse_args()
    import torch_npu

    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    if world not in (2, 4, 8) or args.heads < 1 or args.heads % world or args.rows < 1:
        parser.error("use 2/4/8 ranks, divisible global heads and positive rows")

    def mark(phase):
        print(json.dumps({"rank": rank, "phase": phase, **vars(args)}), flush=True)

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    mark("initialize")
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    # Match the production non-MC2 group buffer override; do not change the
    # default group's global mode when isolating the DCP communicator.
    config = {"hccl_buffer_size": args.buffer_mb}
    if args.dcp_mode != "inherit":
        config["hccl_op_expansion_mode"] = {"HOST": 1, "AI_CPU": 2, "AIV": 3}[args.dcp_mode]
    options.hccl_config = config
    dcp = dist.new_group(backend="hccl", pg_options=options, timeout=timedelta(seconds=120))
    print(
        json.dumps(
            {
                "rank": rank,
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "git": torch_npu.version.git_version,
                "global_mode": os.getenv("HCCL_OP_EXPANSION_MODE"),
                "task_queue": os.getenv("TASK_QUEUE_ENABLE"),
                "dcp_options": options.hccl_config,
            }
        ),
        flush=True,
    )
    rows, heads = args.rows, args.heads // world
    packet = torch.empty((rows, heads, 576), dtype=torch.bfloat16, device="npu")
    gathered = packet.new_empty((world * rows, heads, 576))
    send = packet.new_empty((world, rows, heads, 528))
    recv = torch.empty_like(send)
    reduced = packet.new_empty((512,))
    stream = torch.npu.Stream() if args.overlap_copy else None
    copied = torch.empty_like(packet)
    start, done = (torch.npu.Event() for _ in range(2))
    graph = None

    def forward():
        if args.case != "exchange":
            dist.all_gather_into_tensor(gathered, packet, group=dcp)
        if stream is not None:
            start.record(torch.npu.current_stream())
            with torch.npu.stream(stream):
                stream.wait_event(start)
                copied.copy_(packet)
                done.record(stream)
        if args.case != "gather":
            if args.case != "exchange":
                send.copy_(gathered.view(world, rows, heads, 576)[..., :528])
                send.add_(rank * 16)
            splits = [1] * world if args.exchange_api == "variable" else None
            dist.all_to_all_single(recv, send, splits, splits, group=dcp)
        if args.case == "alternating":
            reduced.copy_(recv.flatten()[:512])
            dist.all_reduce(reduced)  # Same ranks, separate communicator.
        if stream is not None:
            torch.npu.current_stream().wait_event(done)

    def update(value):
        packet.fill_(value + rank)
        if args.case == "exchange":
            for peer in range(world):
                send[peer].fill_(value + peer + rank * 16)

    update(1)
    mark("warmup_submit")
    forward()
    mark("warmup_synchronize")
    torch.npu.synchronize()
    if args.execution == "graph":
        mark("capture_begin")
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            forward()
        mark("capture_end")
    failures = []
    for value in (1, 7, 2):
        update(value)
        mark(f"execute_{value}")
        graph.replay() if graph is not None else forward()
        mark(f"synchronize_{value}")
        torch.npu.synchronize()
        if args.case != "exchange":
            expected = (torch.arange(world) + value).repeat_interleave(rows * heads * 576)
            failures.append(not torch.equal(gathered.cpu().flatten(), expected.to(gathered.dtype)))
        if args.case != "gather":
            expected = (torch.arange(world) * 16 + rank + value).repeat_interleave(rows * heads * 528)
            failures.append(not torch.equal(recv.cpu().flatten(), expected.to(recv.dtype)))
        if args.case == "alternating":
            failures.append(not reduced.cpu().eq(world * value + world * (world - 1) // 2).all().item())
        if stream is not None:
            failures.append(not copied.cpu().eq(value + rank).all().item())
        mark(f"verified_{value}")
    failed = torch.tensor([int(any(failures))], dtype=torch.int32)
    dist.all_reduce(failed, group=control)
    if failed.item():
        raise AssertionError(f"HCCL replay parity failure: rank={rank} local_checks={failures}")
    if graph is not None:
        graph.reset()
    mark("passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
