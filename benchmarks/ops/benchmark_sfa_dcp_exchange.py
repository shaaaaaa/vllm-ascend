# SPDX-License-Identifier: Apache-2.0
"""Compare full DCP attention with packed and direct output exchange.

Run with torchrun on idle NPUs; no model or LMCache checkout is needed.
Times include query packing/gather/unpack, native counted attention, output
communication and weighted combination, not just the exchange kernels.
"""

import argparse
import json
import os
import statistics
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
BLOCK, LATENT, ROPE, TOPK = 128, 512, 64, 2048


def make_counts(requests, query_rows, padding, world, rank, step, layout, capacity):
    """Distribute a global Top-K; preserve empty and heavily skewed shards."""
    counts = torch.zeros(requests * query_rows + padding, dtype=torch.int32)
    for row in range(requests * query_rows):
        selected = TOPK - (step + row) % 17
        if layout == "balanced":
            count = selected // world + (rank < selected % world)
        elif layout == "skewed":
            count = selected if rank == (step + row) % world else 0
        elif layout == "empty":
            count = 0
        else:
            raise ValueError(f"unknown layout: {layout}")
        counts[row] = min(count, capacity)
    return counts


class AttentionCase:
    def __init__(self, args, device):
        self.args = args
        rows = args.requests * args.query_rows + args.padding
        local_heads = args.heads // args.world
        rng = torch.Generator().manual_seed(args.seed + args.rank)
        dtype = getattr(torch, args.dtype)
        blocks = args.local_kv_tokens // BLOCK
        self.cache = tuple(
            torch.randn((args.requests * blocks, BLOCK, 1, width), generator=rng, dtype=dtype).to(device)
            for width in (LATENT, ROPE)
        )
        table = torch.arange(args.requests * blocks, dtype=torch.int32).reshape(args.requests, blocks)
        ends = torch.arange(1, args.requests + 1, dtype=torch.int32) * args.query_rows
        lengths = torch.full((args.requests,), args.local_kv_tokens, dtype=torch.int32)
        if args.padding:
            table = torch.cat((table, torch.zeros_like(table[:1])))
            ends = torch.cat((ends, ends.new_tensor([rows])))
            lengths = torch.cat((lengths, lengths.new_zeros(1)))
        self.metadata = tuple(t.to(device) for t in (table, ends, lengths))
        self.frames = []
        for step in range(args.steps):
            # Match the head-major projection view entering production DCP.
            q = torch.randn((local_heads, rows, LATENT), generator=rng, dtype=dtype).to(device).transpose(0, 1)
            rope = torch.randn((rows, local_heads, ROPE), generator=rng, dtype=dtype).to(device)
            indices = torch.full((rows, 1, TOPK), -1, dtype=torch.int32)
            counts = torch.zeros((rows, 16), dtype=torch.int32)
            counts[:, 0] = make_counts(
                args.requests,
                args.query_rows,
                args.padding,
                args.world,
                args.rank,
                step,
                args.layout,
                args.local_kv_tokens,
            )
            for row, count in enumerate(counts[:, 0].tolist()):
                # Sparse scratch positions need not be in input-sequence order.
                indices[row, 0, :count] = torch.randperm(args.local_kv_tokens, generator=rng)[:count]
            self.frames.append((q, rope, indices.to(device), counts.to(device)))

    def forward(self, group, direct):
        from vllm_ascend.ops.triton.sfa_dcp import execute_sfa_dcp

        outputs = []
        for q, rope, indices, counts in self.frames:
            for _ in range(self.args.layers):
                outputs.append(
                    execute_sfa_dcp(
                        q,
                        rope,
                        self.cache,
                        indices,
                        counts,
                        *self.metadata,
                        (LATENT + ROPE) ** -0.5,
                        group,
                        _direct_exchange=direct,
                    )
                )
        return outputs


def check_outputs(actual, expected, control, phase):
    """Every rank agrees on failures before anyone enters the next collective."""
    error = None
    try:
        assert len(actual) == len(expected)
        for index, (got, want) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(
                got.cpu(), want.cpu(), rtol=0, atol=0, msg=lambda message, index=index: f"output={index}: {message}"
            )
    except Exception as exc:
        error = f"rank={dist.get_rank()} {phase}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size(control)
    dist.all_gather_object(errors, error, group=control)
    if any(errors):
        raise AssertionError("\n".join(e for e in errors if e))


def summarize(samples):
    slowest = {
        name: [max(rank[name][i] for rank in samples) for i in range(len(samples[0][name]))] for name in samples[0]
    }
    return {
        name: {"median_us": statistics.median(values), "mean_us": statistics.mean(values), "samples_us": values}
        for name, values in slowest.items()
    }


def measure_case(args, control):
    from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

    from vllm_ascend.utils import create_hccl_pg_options

    # Keep both real attention graphs alive, and destroy their communicator
    # before any later case. Do not introduce disposable synthetic probes.
    device_group = dist.new_group(
        backend="hccl", timeout=timedelta(seconds=120), pg_options=create_hccl_pg_options("dcp")
    )
    group = SimpleNamespace(world_size=args.world, device_group=device_group)
    group.all_gather = lambda packet, dim: DeviceCommunicatorBase.all_gather(group, packet, dim)
    graphs = []
    try:
        case = AttentionCase(args, torch.device(f"npu:{args.device}"))
        eager = [case.forward(group, direct) for direct in (False, True)]
        torch.npu.synchronize()
        check_outputs(eager[1], eager[0], control, "eager direct vs packed")
        outputs = []
        for direct in (False, True):
            graph = torch.npu.NPUGraph()
            graphs.append(graph)
            with torch.npu.graph(graph):
                outputs.append(case.forward(group, direct))
        for replay in range(args.check_replays):
            # Change graph inputs in place: stale captured values must fail.
            # Eager references are computed outside timed/captured regions.
            for q, rope, _, _ in case.frames:
                q.mul_(-1)
                rope.mul_(-1)
            reference = case.forward(group, False)
            for graph in graphs:
                graph.replay()
            torch.npu.synchronize()
            for name, result in zip(("packed", "direct"), outputs):
                check_outputs(result, reference, control, f"{name} replay={replay}")
        for _ in range(args.warmup):
            for graph in graphs:
                graph.replay()
        torch.npu.synchronize()
        samples = {"packed": [], "direct": []}
        for trial in range(args.trials):
            # Paired, alternating trials keep thermal/order effects visible.
            for variant in (0, 1) if trial % 2 == 0 else (1, 0):
                for _ in range(args.iterations):
                    dist.barrier(group=control)
                    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    graphs[variant].replay()
                    end.record()
                    end.synchronize()
                    samples[("packed", "direct")[variant]].append(start.elapsed_time(end) * 1000 / args.steps)
        if args.profile_dir:
            import torch_npu

            for name, graph in zip(samples, graphs):
                dist.barrier(group=control)
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
                ) as prof:
                    graph.replay()
                    torch.npu.synchronize()
                path = args.profile_dir / f"{args.layout}-b{args.requests}-{name}-rank{args.rank}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(path))
        all_samples = [None] * args.world
        dist.all_gather_object(all_samples, samples, group=control)
        summary = summarize(all_samples)
        rows = args.requests * args.query_rows + args.padding
        return {
            "requests": args.requests,
            "query_rows": args.query_rows,
            "padding": args.padding,
            "layout": args.layout,
            "world": args.world,
            "dtype": args.dtype,
            "steps": args.steps,
            "layers": args.layers,
            "parity": "passed",
            "unit": "us_per_step_for_whole_layer_group",
            "output_collectives_per_layer": {"packed": 1, "direct": rows * 3},
            "slowest_rank": summary,
            "per_rank_samples": all_samples,
            "median_reduction_percent": 100 * (1 - summary["direct"]["median_us"] / summary["packed"]["median_us"]),
        }
    finally:
        torch.npu.synchronize()
        # No communication on this group after releasing a captured graph.
        for graph in graphs:
            graph.reset()
        dist.destroy_process_group(device_group)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--layouts", nargs="+", choices=("balanced", "skewed", "empty"), default=["balanced"])
    parser.add_argument("--query-rows", type=int, default=2, help="target plus MTP rows per request")
    parser.add_argument("--padding", type=int, default=0, help="additional graph rows with zero local counts")
    parser.add_argument("--heads", type=int, default=64, help="global query heads")
    parser.add_argument("--local-kv-tokens", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--check-replays", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--json", type=Path, default=Path("sfa-dcp-exchange.json"))
    args = parser.parse_args()
    args.world, args.rank, args.device = (int(os.environ[name]) for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if (
        args.world not in (2, 4, 8)
        or args.heads % args.world
        or args.heads % 8
        or min(
            *args.requests,
            args.query_rows,
            args.heads,
            args.local_kv_tokens,
            args.layers,
            args.steps,
            args.check_replays,
            args.trials,
            args.iterations,
        )
        < 1
        or args.local_kv_tokens % BLOCK
        or args.padding < 0
        or args.warmup < 0
    ):
        parser.error("use positive sizes, DCP=2/4/8, heads aligned to 8 and KV capacity aligned to 128")
    sys.path.insert(0, str(ROOT))
    import torch_npu

    from vllm_ascend import envs
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend native operators are unavailable")
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    result = {
        "scope": "query pack/gather/unpack -> native counted SFA -> output exchange -> weighted combine",
        "excludes": "projections, indexer, resident planning, H2D, MoE, sampling and server scheduling",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "HCCL_OP_EXPANSION_MODE": os.getenv("HCCL_OP_EXPANSION_MODE"),
        "dcp_hccl_host": bool(envs.VLLM_ASCEND_DCP_HCCL_HOST),
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases": [],
    }
    try:
        for layout in args.layouts:
            for requests in args.requests:
                if not args.rank:
                    print(f"Starting {layout} batch={requests} DCP={args.world}", flush=True)
                case_args = SimpleNamespace(**{**vars(args), "requests": requests, "layout": layout})
                case = measure_case(case_args, control)
                if not args.rank:
                    print(json.dumps(case), flush=True)
                    result["cases"].append(case)
                    args.json.parent.mkdir(parents=True, exist_ok=True)
                    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
