# SPDX-License-Identifier: Apache-2.0
"""Compare complete shared-plan preparation/retrieval/attention groups.

Use torchrun on idle NPUs. Both arms use the current mask-free planner, real
registered host sources, native attention and production DCP collectives.
Only the graph bridge copies differ. Indexer/projection compute is excluded.
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
from benchmark_sfa_target_preparation import (
    BLOCK,
    LATENT,
    ROOT,
    ROPE,
    Pipeline,
    Sources,
    assert_snapshots,
    make_inputs,
)


class AttentionPipeline(Pipeline):
    def __init__(self, args, inputs, sources, group, direct, *, early_gather=False):
        from lmcache_ascend.v1.npu_connector.sparse_graph import SparseGraphTransfer

        from vllm_ascend.compilation.sfa_retrieval_overlap import RetrievalEdge

        # optimized=True in BOTH arms: do not attribute the earlier mask change
        # to bridge removal. The reference reproduces the parent commit's path.
        super().__init__(args, inputs, sources, optimized=True)
        self.direct, self.group = direct, group
        self.early_gather = early_gather
        if args.dcp_size == 1:
            self.dcp.layout = None
        self.layer_caches = [self.caches]
        self.transfers, self.bridges = [self.transfer], [self.bridge]
        self.queries = []
        for layer in range(args.layers):
            q = (self.query.transpose(0, 1).contiguous() / 128 + (args.rank + layer) / 64).transpose(0, 1)
            self.queries.append((q, self.rope.contiguous()))
            if layer:
                caches = tuple(torch.empty_like(t) for t in self.caches)
                self.layer_caches.append(caches)
                bridge = tuple(torch.empty_like(t) for t in self.bridge)
                self.bridges.append(bridge)
                transfer = SparseGraphTransfer(caches, bridge[-1], sources.chunk, args.history, self.capacity)
                transfer.bind_batch([None] * args.requests if args.resident else sources.sources, 0)
                self.transfers.append(transfer)
        # Give graph padding its own zero-KV request, as the serving builder does.
        self.attention_table = torch.cat((self.table, torch.zeros_like(self.table[:1])))
        self.attention_ends = torch.cat((self.ends, self.ends.new_full((1,), self.tokens)))
        extent = self.table.shape[1] * BLOCK
        self.local_lens = self.ends.new_full((self.capacity + 1,), extent)
        self.local_lens[-1] = 0
        self.local_lens[args.requests : self.capacity] = 0
        self.edges = []
        self.copy_stream = torch.npu.Stream() if args.overlap_retrieval and args.layers > 1 else None
        if self.copy_stream is not None:
            self.edges = [
                RetrievalEdge(self.copy_stream, transfer, caches, max_aiv_cores=12)
                for transfer, caches in zip(self.transfers[1:], self.layer_caches[1:])
            ]
        self.attention_outputs = []
        self.reset()

    def reset(self):
        super().reset()
        for caches in getattr(self, "layer_caches", ())[1:]:
            for cache in caches:
                cache.fill_(-7)

    def step(self, frame):
        from vllm_ascend.ops.triton.sfa_dcp import execute_sfa_dcp

        self.attention_outputs = []
        for layer, (queries, caches, transfer, bridge) in enumerate(
            zip(self.queries, self.layer_caches, self.transfers, self.bridges)
        ):
            if layer and self.edges:
                self.edges[layer - 1].join()
            pending = None
            if getattr(self, "early_gather", False) and (not layer or not self.edges):
                from vllm_ascend.ops.triton.sfa_dcp import start_sfa_dcp_query_gather

                pending = start_sfa_dcp_query_gather(*queries, self.group)
            try:
                # Only the producer runs the planner. Consumer layers have
                # little independent work after their queries become ready.
                if layer == 0:
                    prepared = self.prepare(frame)
                self._attend_layer(layer, queries, caches, transfer, bridge, prepared, pending, execute_sfa_dcp)
            finally:
                if pending is not None:
                    pending.wait()

    def _attend_layer(self, layer, queries, caches, transfer, bridge, prepared, pending, execute_sfa_dcp):
        outputs = (*queries, *prepared[2:])
        # Use the real bridge implementation for both the baseline and
        # DCP=1's query-only copy. The DCP direct arm packs at all-gather.
        if not self.direct or self.args.dcp_size == 1:
            owner = SimpleNamespace(_ensure_staged_sfa_bridge_buffers=lambda _, bridge=bridge: bridge)
            outputs = self.copy_bridge(owner, queries[0], outputs, copy_payload=not self.direct)
        query, rope, topk, selected, counts, slots = outputs
        if not layer or not self.edges:
            transfer.load(selected, counts, slots)
        prefetch = None
        if layer < len(self.edges):
            edge = self.edges[layer]
            prefetch = (edge, selected, counts, slots, edge.destinations)
        if self.args.dcp_size > 1:
            result = execute_sfa_dcp(
                query,
                rope,
                caches,
                topk,
                self.counts,
                self.attention_table,
                self.attention_ends,
                self.local_lens,
                (LATENT + ROPE) ** -0.5,
                self.group,
                prefetch=prefetch,
                **({"query_gather": pending} if pending is not None else {}),
                **(
                    {"prefetch_after_pack": True}
                    if prefetch is not None and getattr(self, "prefetch_after_pack", False)
                    else {}
                ),
            )
        else:
            result = torch.ops._C_ascend.npu_sparse_flash_attention(
                query,
                caches[0],
                caches[0],
                topk,
                (LATENT + ROPE) ** -0.5,
                1,
                self.attention_table,
                self.attention_ends,
                self.local_lens,
                rope,
                caches[1],
                "TND",
                "PA_BSND",
                3,
            )
            if prefetch is not None:
                prefetch[0].launch(*prefetch[1:])
        self.attention_outputs.append(result)
        self.current_outputs = outputs

    def snapshot(self, *, include_cache=True):
        values = super().snapshot(include_cache=include_cache)
        if include_cache:
            values += [cache.cpu() for caches in self.layer_caches[1:] for cache in caches]
        return values


def verify(pipelines, control, golden=None, *, phase="eager"):
    """Agree on validation failures before any rank starts another collective."""
    error, misses = None, None
    try:
        assert_snapshots(*(p.snapshot(include_cache=False) for p in pipelines))
        left, right = (p.check_transfer() for p in pipelines)
        assert left == right
        misses = left
        for i, pipeline in enumerate(pipelines):
            for caches in pipeline.layer_caches[1:]:
                assert pipeline.check_transfer(caches=caches) == left
            if golden is not None:
                assert_snapshots(golden[i][0], pipeline.snapshot(include_cache=False))
                assert_snapshots(golden[i][1], [t.cpu() for t in pipeline.attention_outputs])
        for a, b in zip(pipelines[0].attention_outputs, pipelines[1].attention_outputs):
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
    except Exception as exc:
        error = f"rank={dist.get_rank()} phase={phase}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error, group=control)
    if any(errors):
        raise AssertionError("\n".join(e for e in errors if e))
    return misses


def measure_case(args, control, *, pipeline_factory=AttentionPipeline):
    from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

    from vllm_ascend.utils import create_hccl_pg_options

    # Never reuse a communicator after releasing one of its captured graphs.
    device_group = dist.new_group(backend="hccl", pg_options=create_hccl_pg_options("dcp"))
    group = SimpleNamespace(world_size=args.dcp_size, device_group=device_group)
    group.all_gather = lambda packet, dim: DeviceCommunicatorBase.all_gather(group, packet, dim)
    device = torch.device(f"npu:{args.device}")
    raw, rows, ends = make_inputs(
        args.requests,
        args.steps + args.warmup_steps,
        args.history,
        args.live_tokens,
        args.overlap,
        args.layout,
        args.padding,
        args.seed,
    )
    frames = [(t.to(device), p.to(device)) for t, p in raw]
    inputs = frames, rows.to(device), ends.to(device)
    sources, pipelines = None, []
    try:
        sources = Sources(args.lmcache_ascend_dir, args.requests, args.history, device)
        for direct in (False, True):
            pipelines.append(pipeline_factory(args, inputs, sources, group, direct))
        misses = []
        for step, frame in enumerate(frames):
            for pipeline in pipelines:
                pipeline.step(frame)
            torch.npu.synchronize()
            misses.append(verify(pipelines, control, phase=f"eager_step_{step}"))
        golden = [(p.snapshot(include_cache=False), [t.cpu() for t in p.attention_outputs]) for p in pipelines]
        graphs = []
        graph_bytes = [] if hasattr(pipelines[0], "indexer_bytes") else None
        # Both variants' real graphs stay alive throughout parity and timing.
        for index, pipeline in enumerate(pipelines):
            if not args.rank:
                print(f"Capturing {'candidate' if index else 'baseline'} layer group", flush=True)
            if graph_bytes is not None:
                before = torch.npu.memory_allocated(), torch.npu.memory_reserved()
            warm = pipeline.capture(frames[: args.warmup_steps])
            timed = pipeline.capture(frames[args.warmup_steps :])
            graphs.append((warm, timed))
            if graph_bytes is not None:
                graph_bytes.append((torch.npu.memory_allocated() - before[0], torch.npu.memory_reserved() - before[1]))
        for replay in range(3):
            for pipeline, (warm, timed) in zip(pipelines, graphs):
                pipeline.reset()
                if warm is not None:
                    warm.replay()
                timed.replay()
            torch.npu.synchronize()
            verify(pipelines, control, golden, phase=f"replay_{replay}")
        samples = {"baseline": [], "direct": []}
        for trial in range(args.trials):
            for variant in (0, 1) if trial % 2 == 0 else (1, 0):
                pipeline, (warm, timed) = pipelines[variant], graphs[variant]
                for _ in range(args.iterations):
                    pipeline.reset()
                    if warm is not None:
                        warm.replay()
                    torch.npu.synchronize()
                    dist.barrier(group=control)
                    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    timed.replay()
                    end.record()
                    end.synchronize()
                    samples[("baseline", "direct")[variant]].append(start.elapsed_time(end) * 1000 / args.steps)
        if args.profile_dir:
            import torch_npu

            for name, pipeline, (warm, timed) in zip(samples, pipelines, graphs):
                pipeline.reset()
                if warm is not None:
                    warm.replay()
                torch.npu.synchronize()
                dist.barrier(group=control)
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
                ) as prof:
                    timed.replay()
                    torch.npu.synchronize()
                path = args.profile_dir / f"{args.layout}-b{args.requests}-{name}-rank{args.rank}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(path))
        all_samples = [None] * args.dcp_size
        dist.all_gather_object(all_samples, samples, group=control)
        slowest = {
            name: [max(rank[name][i] for rank in all_samples) for i in range(len(samples[name]))] for name in samples
        }
        memory = {}
        if hasattr(pipelines[0], "indexer_bytes"):
            local = tuple(p.indexer_bytes for p in pipelines), graph_bytes
            ranked = [None] * args.dcp_size
            dist.all_gather_object(ranked, local, group=control)
            cache_bytes, capture_bytes = zip(*ranked)
            memory = {"indexer_key_and_scale_bytes_per_rank": cache_bytes,
                      "indexer_key_and_scale_bytes_total": list(map(sum, zip(*cache_bytes))),
                      "capture_allocated_and_reserved_byte_deltas_per_rank": capture_bytes}
        return {
            "requests": args.requests,
            "layout": args.layout,
            "padding": args.padding,
            "dcp_size": args.dcp_size,
            "layers": args.layers,
            "overlap_retrieval": args.overlap_retrieval,
            "steps": args.steps,
            "parity": "passed",
            "unit": "us_per_step_for_whole_layer_group",
            "slowest_rank_median": {name: statistics.median(values) for name, values in slowest.items()},
            "per_rank_samples": all_samples,
            "rank0_misses_per_step": misses[args.warmup_steps :],
            **memory,
        }
    finally:
        for pipeline in pipelines:
            pipeline.close()
        if sources is not None:
            sources.close()
        dist.destroy_process_group(device_group)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-ascend-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--layouts", nargs="+", choices=("uniform", "bounded", "mixed"), default=["uniform"])
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--heads", type=int, default=64, help="global heads; DCP=1 uses --local-heads")
    parser.add_argument("--local-heads", type=int, default=8, help="TP-local heads for the single-NPU DCP=1 case")
    parser.add_argument("--layers", type=int, default=2, help="producer plus shared-plan consumers")
    parser.add_argument("--overlap-retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.9)
    parser.add_argument("--history", type=int, default=16384)
    parser.add_argument("--live-tokens", type=int, default=128)
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--json", type=Path, default=Path("sfa-copy-reduction.json"))
    args = parser.parse_args()
    args.dcp_size, args.rank, args.device = (int(os.environ[name]) for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if (
        args.dcp_size not in (1, 2, 4, 8)
        or args.heads % args.dcp_size
        or args.heads % 8
        or args.local_heads % 8
        or min(
            *args.requests,
            args.layers,
            args.steps,
            args.trials,
            args.iterations,
            args.history,
            args.heads,
            args.local_heads,
        )
        < 1
        or args.warmup_steps < 0
    ):
        parser.error("use positive sizes, DCP=1/2/4/8 and aligned, divisible heads")
    for layout in args.layouts:
        try:
            make_inputs(1, 1, args.history, args.live_tokens, args.overlap, layout, args.padding)
        except ValueError as exc:
            parser.error(str(exc))
    if not (args.lmcache_ascend_dir / "benchmark/v1/kv_transfer/load_benchmark_utils.py").is_file():
        parser.error("--lmcache-ascend-dir must be the companion LMCache-Ascend checkout")
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
        "scope": "raw Top-K/projected queries through planner, real sparse H2D and completed attention merge",
        "excludes": "indexer/projections, MoE and server scheduling; reset/warmup outside timing",
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
                case_args = SimpleNamespace(
                    **{
                        **vars(args),
                        "layout": layout,
                        "requests": requests,
                        "shared_plan": True,
                        "heads": args.heads // args.dcp_size if args.dcp_size > 1 else args.local_heads,
                    }
                )
                if not args.rank:
                    print(f"Starting {layout} batch={requests} DCP={args.dcp_size}", flush=True)
                case = measure_case(case_args, control)
                if not args.rank:
                    result["cases"].append(case)
                    print(json.dumps(case), flush=True)
                    args.json.parent.mkdir(parents=True, exist_ok=True)
                    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
