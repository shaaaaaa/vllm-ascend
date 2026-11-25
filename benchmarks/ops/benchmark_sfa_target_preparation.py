# SPDX-License-Identifier: Apache-2.0
"""Single-NPU indexer-output -> real sparse H2D benchmark, including sfa_copy.

Inputs are precomputed causal Top-K sequences. Indexer compute, attention and
collectives are outside the interval. Both variants use the serving planner,
bridge and LMCache transfer; no mock copy or startup probe is used.
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
TOPK, QUERY_ROWS, BLOCK, LATENT, ROPE = 2048, 2, 128, 512, 64
SCRATCH = TOPK * QUERY_ROWS


def make_inputs(requests, steps, history, live, overlap, layout, padding=0, seed=0):
    """Exact consecutive-step overlap per row; query rows have independent sets."""
    if not 0 <= overlap <= 1 or history + live < 2 * TOPK or min(requests, steps, live) < 1 or padding < 0:
        raise ValueError("invalid input geometry or overlap; history + live must be >= 4096")
    if layout not in ("uniform", "bounded", "mixed") or (layout == "uniform" and padding):
        raise ValueError("padding requires bounded/mixed layout")
    widths = [1 if layout == "mixed" and i % 2 == 0 else QUERY_ROWS for i in range(requests)]
    capacity = requests + padding
    owners = [i for i, width in enumerate(widths) for _ in range(width)]
    tokens = capacity * QUERY_ROWS
    rows = torch.tensor(owners + [-1] * (tokens - len(owners)), dtype=torch.int32)
    ends = torch.tensor(widths + [0] * padding, dtype=torch.int32).cumsum(0).int()
    rng = random.Random(seed)
    pool = range(history + live)
    previous = [rng.sample(pool, TOPK) for _ in owners]
    keep = round(TOPK * overlap)
    frames = []
    for step in range(steps):
        raw = torch.full((tokens, 1, TOPK), -1, dtype=torch.int32)
        positions = torch.zeros(tokens, dtype=torch.int64)
        lane = 0
        for width in widths:
            for query in range(width):
                if step:
                    before = set(previous[lane])
                    selected = rng.sample(previous[lane], keep)
                    selected += rng.sample([t for t in pool if t not in before], TOPK - keep)
                    rng.shuffle(selected)
                    previous[lane] = selected
                raw[lane, 0] = torch.tensor(previous[lane], dtype=torch.int32)
                positions[lane] = history + live + step * QUERY_ROWS + query
                lane += 1
        frames.append((raw, positions))
    return frames, rows, ends


def source_values(tokens, lane, layer=0):
    return ((tokens + lane * 7 + layer * 11) % 127).float() / 8


class Sources:
    """Own registered host pages until every graph referencing them is gone."""

    def __init__(self, root, requests, history, device, *, layers=1):
        sys.path.insert(0, str(root / "benchmark/v1/kv_transfer"))
        import lmcache_ascend.c_ops  # noqa: F401 -- require real host registration, never a CUDA fallback
        from lmcache.v1.gpu_connector.sparse import PreparedSparseSource, PreparedSparseSourceLayer
        from lmcache.v1.memory_management import PinMemoryAllocator
        from load_benchmark_utils import build_chunk_ptrs_npu, ensure_ascend_host_memory_registered

        ensure_ascend_host_memory_registered()
        self.owners, self.sources = [], []
        self.chunk = 1024
        self.allocator = PinMemoryAllocator(requests * layers * history * (LATENT + ROPE) * 2 + 16 * 1024**2)
        try:
            sizes = tuple(min(self.chunk, history - start) for start in range(0, history, self.chunk))
            for lane in range(requests):
                source_layers = []
                for layer in range(layers):
                    chunks = []
                    for start, count in zip(range(0, history, self.chunk), sizes):
                        obj = self.allocator.allocate(torch.Size([count * (LATENT + ROPE)]), torch.bfloat16)
                        if obj is None or obj.tensor is None:
                            raise RuntimeError("registered host allocation failed")
                        self.owners.append(obj)
                        values = source_values(torch.arange(start, start + count), lane, layer)
                        obj.tensor[: count * LATENT].view(count, LATENT).copy_(values[:, None].expand(-1, LATENT))
                        obj.tensor[count * LATENT :].view(count, ROPE).copy_((values + 32)[:, None].expand(-1, ROPE))
                        chunks.append(obj.tensor)
                    source_layers.append(PreparedSparseSourceLayer(tuple(chunks), build_chunk_ptrs_npu(chunks, device)))
                self.sources.append(
                    PreparedSparseSource(
                        layers=tuple(source_layers),
                        total_tokens=history,
                        chunk_token_counts=sizes,
                        pointer_device=device,
                    )
                )
        except BaseException:
            self.close()
            raise

    def close(self):
        torch.npu.synchronize()
        self.sources.clear()
        for obj in self.owners:
            obj.ref_count_down()
        self.owners.clear()


class Pipeline:
    def __init__(self, args, inputs, sources, optimized):
        from lmcache_ascend.v1.npu_connector.sparse_graph import SparseGraphTransfer

        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
        from vllm_ascend.distributed.kv_transfer.sparse_offload.resident_sorted_cache import (
            ResidentDcpLayout,
            SharedResidentPlan,
            allocate_sorted_resident_state,
            allocate_sorted_resident_workspace,
            configure_resident_kernels,
        )

        configure_resident_kernels()
        frames, self.rows, self.ends = inputs
        self.frames, self.optimized, self.args = frames, optimized, args
        self.capacity = len(self.ends)
        self.tokens = len(self.rows)
        self.device = self.rows.device
        self.graphs = []
        self.state = allocate_sorted_resident_state(self.capacity, self.capacity, QUERY_ROWS, device=self.device)
        self.workspace = allocate_sorted_resident_workspace(self.capacity, QUERY_ROWS, device=self.device)
        self.counts = torch.zeros((self.tokens, 16), dtype=torch.int32, device=self.device)
        self.planner_counts = torch.zeros_like(self.counts) if args.layout != "uniform" else self.counts
        self.origin = 0 if args.resident else SCRATCH
        self.dcp = SimpleNamespace(
            layout=ResidentDcpLayout(
                args.dcp_size,
                args.rank,
                torch.full((self.capacity,), self.origin, dtype=torch.int32, device=self.device),
                self.planner_counts,
            ),
            topk=torch.empty_like(frames[0][0]),
            counts=self.counts,
        )
        max_end = args.history + args.live_tokens + len(frames) * QUERY_ROWS + QUERY_ROWS
        blocks = SCRATCH // BLOCK + (max_end + BLOCK * args.dcp_size - 1) // (BLOCK * args.dcp_size)
        self.table = torch.arange(self.capacity * blocks, dtype=torch.int32, device=self.device).view(
            self.capacity, blocks
        )
        self.caches = tuple(
            torch.empty((self.capacity * blocks, BLOCK, 1, d), dtype=torch.bfloat16, device=self.device)
            for d in (LATENT, ROPE)
        )
        self.boundaries = torch.where(self.rows >= 0, 0 if args.resident else args.history, 0).int()
        self.states = torch.arange(self.capacity, dtype=torch.int32, device=self.device)
        self.generations = torch.ones(self.capacity, dtype=torch.int64, device=self.device)
        self.published = torch.empty_like(frames[0][0])
        self.union_mapping = torch.empty_like(self.workspace.miss_tokens)
        # Match BMM's head-major latent and noncontiguous split RoPE storage.
        self.query = (
            torch.arange(args.heads * self.tokens * LATENT, device=self.device).remainder(127).to(torch.bfloat16)
        )
        self.query = self.query.reshape(args.heads, self.tokens, LATENT).transpose(0, 1)
        self.rope = torch.ones((self.tokens, args.heads, LATENT + ROPE), dtype=torch.bfloat16, device=self.device)[
            ..., LATENT:
        ]
        self.bridge = (
            torch.empty(self.query.shape, dtype=self.query.dtype, device=self.device),
            torch.empty(self.rope.shape, dtype=self.rope.dtype, device=self.device),
            torch.empty_like(self.published),
            torch.empty_like(self.workspace.miss_tokens),
            torch.empty(self.capacity, dtype=torch.int32, device=self.device),
            torch.empty_like(self.workspace.target_slots),
        )
        self.impl = SimpleNamespace(
            _sorted_resident_state=self.state,
            _sorted_resident_workspace=self.workspace,
            _sorted_resident_workspace_views={self.capacity: self.workspace},
            _resident_query_membership=True,
            block_size=BLOCK,
            dsa_resident_cache=True,
            decode_threshold=QUERY_ROWS,
            skip_topk=False,
            shared_resident_plan=None,
            _ensure_staged_sfa_bridge_buffers=lambda _: self.bridge,
        )
        self.impl._prepare_sorted_resident_sparse_cache = MethodType(
            AscendSFAImpl._prepare_sorted_resident_sparse_cache,
            self.impl,
        )
        if args.shared_plan:
            self.impl.shared_resident_plan = SharedResidentPlan(
                ("producer", "consumer"),
                self.state,
                self.workspace,
                torch.empty_like(self.published),
                QUERY_ROWS,
                BLOCK,
                active=True,
                bounded=args.layout != "uniform",
            )
        self.plan = AscendSFAImpl._prepare_decode_sparse_indices
        self.copy_bridge = AscendSFAImpl._copy_to_staged_sfa_bridge
        self.transfer = SparseGraphTransfer(self.caches, self.bridge[-1], sources.chunk, args.history, self.capacity)
        self.transfer.bind_batch([None] * args.requests if args.resident else sources.sources, 0)
        self.reset()

    def reset(self):
        self.state.counts.zero_()
        self.state.generations.fill_(-1)
        self.counts.zero_()
        self.planner_counts.zero_()
        for cache in self.caches:
            cache.fill_(-7)

    def prepare(self, frame, *, owner_prepared=False, owner_source=None):
        from dataclasses import replace

        from vllm_ascend.attention.sfa_graph_layout import pack_decode_lanes, unpack_decode_lanes
        from vllm_ascend.ops.triton.sfa_dcp import prepare_sfa_dcp_topk

        raw, positions = frame
        if not owner_prepared:
            self.published.copy_(raw)
        bounded = self.args.layout != "uniform"
        topk = raw
        if not owner_prepared and not (self.optimized and (bounded or self.args.shared_plan)):
            prepare_sfa_dcp_topk(
                raw,
                positions,
                self.dcp.topk,
                self.counts,
                self.tokens,
                self.args.dcp_size,
                self.args.rank,
                BLOCK,
                local=False,
                _causal_global_ids=self.optimized,
            )
            topk = self.dcp.topk
        boundary, rows = self.boundaries, self.rows
        if bounded and owner_source is None:
            topk, boundary, rows, inverse = pack_decode_lanes(topk, boundary, rows, self.ends, QUERY_ROWS)
        planned, selected, counts, slots = self.plan(
            self.impl,
            topk,
            boundary,
            rows,
            self.table,
            self.workspace.miss_tokens,
            self.workspace.miss_counts[:, 0],
            self.workspace.target_slots,
            self.states,
            self.generations,
            local_to_union_workspace=self.union_mapping,
            shard_packed_workspace=None,
            shard_mapping_workspace=None,
            shard_counts_workspace=None,
            staged_mtp=QUERY_ROWS,
            need_packed=True,
            clear_invalid_rows=True,
            dcp=replace(self.dcp.layout, selected_counts=self.counts) if bounded and owner_source is not None
                else self.dcp.layout,
            shared_topk_ready=owner_prepared and (not bounded or owner_source is not None) and self.args.shared_plan,
            **({"owner_topk": owner_source} if owner_source is not None else {}),
        )
        if bounded and owner_source is None:
            planned = unpack_decode_lanes(planned, inverse, self.rows)
            self.counts[:, 0].copy_(
                unpack_decode_lanes(
                    self.planner_counts[:, 0],
                    inverse,
                    self.rows,
                    fill_value=0,
                )
            )
            if self.args.shared_plan:
                storage = self.impl.shared_resident_plan.attention_topk
                storage.copy_(planned)
                planned = storage
        return self.query, self.rope, planned, selected, counts, slots

    def step(self, frame):
        outputs = self.copy_bridge(self.impl, self.query, self.prepare(frame))
        self.transfer.load(*outputs[3:])  # Real sfa_copy_i32_s64, K and PE.

    def snapshot(self, *, include_cache=True):
        # Read only defined prefixes of resident state and variable-size payloads.
        bridge = getattr(self, "current_outputs", self.bridge)
        counts = self.state.counts.cpu()[..., 0]
        state_tokens, state_slots = self.state.tokens.cpu(), self.state.slots.cpu()
        result = [counts, self.state.generations.cpu(), self.counts[:, 0].cpu()]
        for row in range(counts.shape[0]):
            for shard in range(counts.shape[1]):
                n = int(counts[row, shard])
                assert 0 <= n <= SCRATCH
                result += [state_tokens[row, shard, :n], state_slots[row, shard, :n]]
        result += [value.cpu() for value in (*bridge[:3], bridge[4])]
        if include_cache:
            result += [cache.cpu() for cache in self.caches]
        selected, misses, slots = (value.cpu() for value in bridge[3:])
        for lane, count in enumerate(misses.tolist()):
            result += [selected[lane, :count], slots[lane, :count]]
        return result

    def check_transfer(self, *, caches=None, layer=0):
        bridge = getattr(self, "current_outputs", self.bridge)
        selected, counts, slots = (value.cpu() for value in bridge[3:])
        caches = [cache.view(-1, cache.shape[-1]) for cache in (self.caches if caches is None else caches)]
        for lane, count in enumerate(counts.tolist()):
            tokens, destinations = selected[lane, :count].long(), slots[lane, :count].long()
            assert 0 <= count <= SCRATCH
            if not count:
                continue
            assert bool(((tokens >= 0) & (tokens < self.args.history)).all())
            assert bool((tokens // BLOCK % self.args.dcp_size == self.args.rank).all())
            assert bool(((destinations >= 0) & (destinations < caches[0].shape[0])).all())
            for plane, cache in enumerate(caches):
                expected = (source_values(tokens, lane, layer) + plane * 32).to(cache.dtype)
                actual = cache.index_select(0, bridge[-1][lane, :count]).cpu()
                torch.testing.assert_close(actual, expected[:, None].expand(-1, cache.shape[-1]), rtol=0, atol=0)
        return counts.tolist()

    def capture(self, frames):
        if not frames:
            return None
        for frame in frames:
            self.step(frame)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        self.graphs.append(graph)
        with torch.npu.graph(graph):
            for frame in frames:
                self.step(frame)
        return graph

    def close(self):
        torch.npu.synchronize()
        for graph in self.graphs:
            graph.reset()
        self.graphs.clear()


def assert_snapshots(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@torch.inference_mode()
def run_case(args):
    device = torch.device(f"npu:{args.device}")
    cpu_inputs = make_inputs(
        args.requests,
        args.steps + args.warmup_steps,
        args.history,
        args.live_tokens,
        args.overlap,
        args.layout,
        args.padding,
        args.seed,
    )
    frames = [(raw.to(device), positions.to(device)) for raw, positions in cpu_inputs[0]]
    inputs = frames, cpu_inputs[1].to(device), cpu_inputs[2].to(device)
    sources = Sources(args.lmcache_ascend_dir, args.requests, args.history, device)
    pipelines = []
    try:
        for optimized in (False, True):
            pipelines.append(Pipeline(args, inputs, sources, optimized))
        misses = []
        # Cold-to-warm trajectory, exact native state/output parity and real DMA.
        for frame in frames:
            for pipeline in pipelines:
                pipeline.step(frame)
            torch.npu.synchronize()
            assert_snapshots(*(pipeline.snapshot(include_cache=False) for pipeline in pipelines))
            copied = [pipeline.check_transfer() for pipeline in pipelines]
            assert copied[0] == copied[1]
            misses.append(copied[0])
        golden = pipelines[0].snapshot()
        graphs = []
        for pipeline in pipelines:
            warm = pipeline.capture(frames[: args.warmup_steps])
            timed = pipeline.capture(frames[args.warmup_steps :])
            graphs.append((warm, timed))
            pipeline.reset()
            if warm is not None:
                warm.replay()
            timed.replay()
            torch.npu.synchronize()
            assert_snapshots(golden, pipeline.snapshot())
        samples = {"old": [], "new": []}
        for trial in range(args.trials):
            for variant in (0, 1) if trial % 2 == 0 else (1, 0):
                pipeline, (warm, timed) = pipelines[variant], graphs[variant]
                for _ in range(args.iterations):
                    pipeline.reset()
                    if warm is not None:
                        warm.replay()
                    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    timed.replay()
                    end.record()
                    end.synchronize()
                    samples[("old", "new")[variant]].append(start.elapsed_time(end) * 1000 / args.steps)
        if args.profile_dir:
            import torch_npu

            for name, pipeline, (warm, timed) in zip(("old", "new"), pipelines, graphs):
                pipeline.reset()
                if warm is not None:
                    warm.replay()
                torch.npu.synchronize()
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
                ) as prof:
                    timed.replay()
                    torch.npu.synchronize()
                path = args.profile_dir / f"{args.layout}-b{args.requests}-{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(path))
        measured = misses[args.warmup_steps :]
        return {
            "requests": args.requests,
            "layout": args.layout,
            "padding": args.padding,
            "dcp_size": args.dcp_size,
            "rank": args.rank,
            "resident": args.resident,
            "shared_plan": args.shared_plan,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "overlap": args.overlap,
            "parity": "passed",
            "misses_per_step_per_request": measured,
            "copied_bytes": sum(map(sum, measured)) * (LATENT + ROPE) * 2,
            "timings": {
                name: {"median_us_per_step": statistics.median(values), "samples_us_per_step": values}
                for name, values in samples.items()
            },
        }
    finally:
        # Captured host addresses outlive graphs, including failed qualification.
        for pipeline in pipelines:
            pipeline.close()
        sources.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-ascend-dir", type=Path, required=True)
    parser.add_argument("--requests", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--layouts", nargs="+", choices=("uniform", "bounded", "mixed"), default=["uniform"])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.9)
    parser.add_argument("--history", type=int, default=16384)
    parser.add_argument("--live-tokens", type=int, default=128)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dcp-size", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--resident", action="store_true", help="zero retrieval frontier; still executes the copy kernel"
    )
    parser.add_argument("--shared-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--json", type=Path, default=Path("sfa-target-preparation.json"))
    args = parser.parse_args()
    if min(*args.requests, args.steps, args.history, args.heads, args.trials, args.iterations) < 1:
        parser.error("sizes and iteration counts must be positive")
    if args.warmup_steps < 0 or not 0 <= args.rank < args.dcp_size:
        parser.error("invalid warmup count or DCP rank")
    for layout in args.layouts:
        try:
            make_inputs(1, 1, args.history, args.live_tokens, args.overlap, layout, args.padding)
        except ValueError as exc:
            parser.error(str(exc))
    if not (args.lmcache_ascend_dir / "benchmark/v1/kv_transfer/load_benchmark_utils.py").is_file():
        parser.error("--lmcache-ascend-dir must point to the companion LMCache-Ascend source checkout")
    sys.path.insert(0, str(ROOT))
    import torch_npu

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend native operators are unavailable")
    result = {
        "scope": "raw indexer output through completed real sparse H2D; includes publication, planner and bridge",
        "excludes": "indexer compute, attention, collectives, model CPU scheduling; reset/warmup outside timing",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cases": [],
    }
    for layout in args.layouts:
        for requests in args.requests:
            case = run_case(SimpleNamespace(**{**vars(args), "layout": layout, "requests": requests}))
            result["cases"].append(case)
            print(json.dumps(case), flush=True)
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
