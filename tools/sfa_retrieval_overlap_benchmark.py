# SPDX-License-Identifier: Apache-2.0
"""Registered-CPU retrieval + production cube projection, not a model benchmark.

Uses the serving fork/join helper and SparseGraphTransfer unchanged. Real SFA,
O-projection and MoE interference must additionally be measured in serving.
"""
import argparse
import json
import statistics
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_ascend.compilation.sfa_retrieval_overlap import CaptureProbe, CopyGate, RetrievalEdge, SharedPrefetchGroup


def scattered_token_order(width, history):
    # A stride of 17 modulo 4097 only visits 241 host rows. A seeded
    # permutation visits each row once before wrapping at the history length.
    order = torch.randperm(history, generator=torch.Generator().manual_seed(17), dtype=torch.int32)
    return order[torch.arange(width) % history]


class Workload:
    def __init__(self, root, requests=8, query_rows=2, layers=4, width=4096, history=4097, chunk=1024, *,
                 main_gate=False, pieces=None, multi_layer=False):
        import vllm_ascend.vllm_ascend_C  # noqa: F401
        from vllm_ascend import envs

        helpers = Path(root) / "benchmark/v1/kv_transfer"
        if not (helpers / "load_benchmark_utils.py").is_file():
            raise FileNotFoundError(f"LMCache-Ascend benchmark helpers missing under {helpers}")
        sys.path.insert(0, str(helpers))
        from load_benchmark_utils import build_chunk_ptrs_npu, ensure_ascend_host_memory_registered

        ensure_ascend_host_memory_registered()
        from lmcache.v1.gpu_connector.sparse import PreparedSparseSource, PreparedSparseSourceLayer
        from lmcache.v1.memory_management import PinMemoryAllocator
        from lmcache_ascend.v1.npu_connector.sparse_graph import SparseGraphTransfer

        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        self.requests, self.layers, self.width, self.history = requests, layers, width, history
        self.token_order = scattered_token_order(width, history)
        self.owners, self.sources, self.transfers, self.graphs = [], [], [], []
        self.stream = torch.npu.Stream(device=self.device)
        self.probe = CaptureProbe(self.stream, self.device, main_gate=main_gate)
        self.probe.verify()
        self.allocator = PinMemoryAllocator(requests * layers * history * 576 * 2 + 16 * 1024 * 1024)
        self.selected = torch.zeros((requests, width), dtype=torch.int32, device=self.device)
        self.slots = torch.arange(requests * width, dtype=torch.int64, device=self.device).view(requests, width)
        self.count_storage = torch.zeros((requests, 16), dtype=torch.int32, device=self.device)
        self.counts = self.count_storage[:, 0]
        self.caches = [tuple(torch.full(((requests * width + 127) // 128, 128, 1, w), -7,
                                       device=self.device, dtype=torch.bfloat16) for w in (512, 64))
                       for _ in range(layers)]
        try:
            for request in range(requests):
                plane_layers = []
                sizes = tuple(min(chunk, history - start) for start in range(0, history, chunk))
                for layer in range(layers):
                    chunks = []
                    for start, count in zip(range(0, history, chunk), sizes):
                        obj = self.allocator.allocate(torch.Size([count * 576]), torch.bfloat16)
                        if obj is None or obj.tensor is None:
                            raise RuntimeError("registered CPU allocation failed")
                        self.owners.append(obj)
                        values = self.values(torch.arange(start, start + count), request, layer)
                        obj.tensor[:count * 512].view(count, 512).copy_(values[:, None].expand(-1, 512))
                        obj.tensor[count * 512:].view(count, 64).copy_((values + 32)[:, None].expand(-1, 64))
                        chunks.append(obj.tensor)
                    plane_layers.append(
                        PreparedSparseSourceLayer(tuple(chunks), build_chunk_ptrs_npu(chunks, self.device))
                    )
                self.sources.append(PreparedSparseSource(layers=tuple(plane_layers), total_tokens=history,
                                    chunk_token_counts=sizes, pointer_device=self.device))
            for layer, cache in enumerate(self.caches):
                transfer = SparseGraphTransfer(cache, self.slots, chunk, history, request_capacity=requests)
                transfer.bind_batch(self.sources, layer)
                self.transfers.append(transfer)
            gate = CopyGate(self.device) if main_gate else None
            copy_pieces = (16 if main_gate else 1) if pieces is None else pieces
            self.edges = [RetrievalEdge(self.stream, self.transfers[i], self.caches[i], pieces=copy_pieces, gate=gate)
                          for i in range(1, layers)]
            self.group = None
            if multi_layer:
                group_gate = CopyGate(self.device) if main_gate else None
                self.group = SharedPrefetchGroup(tuple(
                    RetrievalEdge(self.stream, e.transfer, e.destinations, pieces=e.pieces, gate=group_gate)
                    for e in self.edges
                ))
            self.set_copy_core_limit(envs.VLLM_ASCEND_SFA_COPY_MAX_AIV_CORES)
            self.router_x = torch.ones((requests * query_rows, 16), dtype=torch.float32,
                                       device=self.device) if main_gate else None
            router_weight = torch.ones((8, 16), dtype=torch.float32, device=self.device) if main_gate else None
            self.router_weights = [router_weight] * layers
            self.x = torch.ones((requests * query_rows, 16, 512), dtype=torch.bfloat16, device=self.device)
            weight = torch.ones((16, 512, 128), dtype=torch.bfloat16, device=self.device)
            self.vup_weights = [weight] * layers
            self.qkv_projections = [None] * layers
            self.glm_qkv = None
            if multi_layer:
                # Use the same BF16 projection on all three benchmark schedules.
                qkv = partial(torch.mm, self.x[:, 0, :].contiguous(), weight[0],
                              out=self.x.new_empty((requests * query_rows, 128)))
                self.qkv_projections = [qkv] * layers
            self.products = [None] * layers
            self.observed = [tuple(torch.empty((requests, width), dtype=torch.bfloat16, device=self.device)
                                   for _ in range(2)) for _ in range(layers)]
        except BaseException:
            self.close()
            raise

    @staticmethod
    def values(tokens, request, layer):
        return ((tokens + request * 7 + layer * 11) % 127).float() / 8

    def set_payload(self, active, shift=0, idle=False):
        tokens = (self.token_order + shift) % self.history
        self.selected.copy_(tokens.expand(self.requests, -1))
        self.counts.fill_(active)
        if idle:
            self.counts[-1].zero_()
        for cache in self.caches:
            for plane in cache:
                plane.fill_(-7)
        return tokens

    def use_glm_projections(self, config, tp):
        """Set up the W8A8 dynamic QKV, BF16 V-up and FP32 router of one TP rank."""
        from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
        from vllm_ascend.utils import maybe_trans_nz

        hidden, kv, rope = config["hidden_size"], config["kv_lora_rank"], config["qk_rope_head_dim"]
        heads = config["num_attention_heads"]
        if (kv, rope) != (512, 64) or tp < 1 or heads % tp:
            raise ValueError("this KV fixture requires K=512, PE=64 and attention heads divisible by TP")
        width = config["q_lora_rank"] + kv + rope  # The fused A projection is replicated across TP.
        rows = self.x.shape[0]
        x = self.x.new_ones((rows, hidden))
        scheme = AscendW8A8DynamicLinearMethod()
        self.qkv_projections = []
        self.glm_qkv = (x, [])
        for _ in range(self.layers):
            layer = SimpleNamespace(
                weight=maybe_trans_nz(torch.ones((hidden, width), dtype=torch.int8, device=self.device)),
                weight_scale=x.new_ones((width,)),
            )
            self.qkv_projections.append(partial(scheme.apply, layer, x))
            self.glm_qkv[1].append(layer)
        self.x = x.new_ones((rows, heads // tp, kv))
        self.vup_weights = [x.new_ones((heads // tp, kv, config["v_head_dim"])) for _ in range(self.layers)]
        self.router_x = torch.ones((rows, hidden), dtype=torch.float32, device=self.device)
        self.router_weights = [self.router_x.new_ones((config["n_routed_experts"], hidden)) for _ in range(self.layers)]

    def snapshot_layer(self, layer, targets):
        for plane, target in zip(self.caches[layer], targets):
            rows = plane.view(-1, plane.shape[-1])[:self.requests * self.width, 0]
            target.copy_(rows.view_as(target))

    def set_copy_core_limit(self, limit):
        """Select the launch grid for subsequent captures on the same host KV and slots."""
        self.max_aiv_cores = limit
        group_edges = self.group.edges if self.group is not None else ()
        for edge in (*self.edges, *group_edges):
            edge.max_aiv_cores = limit

    def chain(self, overlap, observe, *, multi_layer=False, after_first_qkv=None, bridge_counts=None):
        edges = self.group.edges if multi_layer else self.edges
        for layer, transfer in enumerate(self.transfers):
            qkv = self.qkv_projections[layer]
            if overlap and layer:
                edge = edges[layer - 1]
                join_projection = multi_layer
                if getattr(self, "glm_qkv", None) is not None and edge.gate is not None:
                    from vllm_ascend.compilation.sfa_copy_windows import qkv_grant_fits, w8a8_with_gate

                    x, weights = self.glm_qkv
                    gate = edge.gate if edge.gate.supports_decoder_windows else None
                    projection = weights[layer]
                    join_projection = multi_layer and qkv_grant_fits(gate, x, projection.weight)
                    if not join_projection and gate is not None:
                        # After the consumer joins, only the next layer(s) can still have work.
                        qkv = partial(w8a8_with_gate, gate if gate.pending > 1 else None, x,
                                      projection.weight, projection.weight_scale, None, "qkv", False)
                if join_projection:
                    edge.join(qkv)
                else:
                    edge.join()
                    if qkv is not None:
                        qkv()
            else:
                if qkv is not None:
                    qkv()
                if multi_layer and layer == 0:
                    if bridge_counts is None:
                        self.group.launch(self.selected, self.counts, self.slots)
                    else:
                        self.group.launch(self.selected, bridge_counts, self.slots, producer_counts=self.counts)
                counts = self.counts if bridge_counts is None else bridge_counts
                transfer.load(self.selected, counts, self.slots, max_aiv_cores=self.max_aiv_cores)
            if layer == 1 and after_first_qkv is not None:
                # The gate has taken its token back. Later copies cannot change a
                # destination while this optional observation reads it.
                after_first_qkv()
            if observe:
                # Read at the consumer boundary, not merely after the whole graph.
                self.snapshot_layer(layer, self.observed[layer])
            edge = edges[layer] if overlap and layer + 1 < self.layers else None
            if edge is not None and not multi_layer:
                edge.launch(self.selected, self.counts, self.slots, self.caches[layer + 1])
            gated = edge is not None and edge.gate is not None
            with edge.cube_window("vup") if gated else nullcontext():
                self.products[layer] = torch_npu.npu_transpose_batchmatmul(
                    self.x, self.vup_weights[layer], perm_x1=(1, 0, 2), perm_y=(1, 0, 2)
                )
            if self.router_x is not None:
                with edge.cube_window("router") if gated else nullcontext():
                    torch.nn.functional.linear(self.router_x, self.router_weights[layer])

    def capture(self, overlap, observe=False, *, multi_layer=False, after_first_qkv=None, bridge_counts=None):
        if after_first_qkv is not None and (not overlap or self.edges[0].gate is None):
            raise ValueError("a stable first-QKV snapshot requires gated overlap")
        return self._capture(partial(
            self.chain, overlap, observe, multi_layer=multi_layer, after_first_qkv=after_first_qkv,
            bridge_counts=bridge_counts,
        ))

    def capture_copy(self):
        """One real layer transfer with the same gate, piece count and stream join as serving."""
        edge = self.edges[0]

        def enqueue():
            edge.launch(self.selected, self.counts, self.slots, edge.destinations)
            edge.join()

        return self._capture(enqueue)

    def _capture(self, enqueue):
        enqueue()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        self.graphs.append(graph)  # Retain even on a partial capture failure.
        with torch.npu.graph(graph):
            enqueue()
        return graph

    def check_copied_layer(self, layer, tokens, active):
        """Check every K/PE element and the unused slots after an untimed replay."""
        lanes = torch.arange(self.requests)[:, None]
        for index, plane in enumerate(self.caches[layer]):
            expected = torch.full((self.requests, self.width), -7, dtype=plane.dtype)
            expected[:, :active] = self.values(tokens[:active], lanes, layer) + 32 * index
            rows = plane.view(-1, plane.shape[-1])[:self.requests * self.width]
            rows = rows.view(self.requests, self.width, -1).cpu()
            if not torch.equal(rows, expected.unsqueeze(-1).expand_as(rows)):
                raise AssertionError(f"incorrect KV copy in layer {layer}, plane {index}, misses {active}")

    def check(self, tokens, active, idle=False):
        for layer in range(self.layers):
            for plane in range(2):
                actual = self.observed[layer][plane].cpu()
                for request in range(self.requests):
                    n = 0 if idle and request == self.requests - 1 else active
                    expected = torch.full((self.width,), -7, dtype=torch.bfloat16)
                    expected[:n] = (self.values(tokens[:n], request, layer) + 32 * plane).bfloat16()
                    torch.testing.assert_close(actual[request], expected, rtol=0, atol=0)

    def close(self):
        torch.npu.synchronize()  # Never release registered pages before side-stream completion.
        self.graphs.clear()
        for obj in self.owners:
            obj.ref_count_down()
        self.owners.clear()


def ready_rows(snapshot, expected):
    """Count rows whose K and PE match the verified destination at its own join."""
    k, pe = (t.cpu() for t in snapshot)
    mask = k != -7
    for observed, reference in zip((k, pe), expected):
        torch.testing.assert_close(observed, torch.where(mask, reference.cpu(), -7), rtol=0, atol=0)
    return int(mask.sum())


def paired_graph_times(first, second, *, names=("adjacent", "grouped"), pairs=8, replays=4, prepare_pair=None,
                       include_means=False):
    """Paired device timings in us per replay. Keep setup and observations outside the measurement."""
    graphs = (first, second)
    timers = [[(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in graphs]
              for _ in range(pairs)]
    for graph in graphs:
        for _ in range(replays):
            graph.replay()
    for pair in timers:
        for start, end in pair:
            start.record()
            end.record()
    torch.npu.synchronize()
    for i, pair in enumerate(timers):
        if prepare_pair is not None:
            prepare_pair(i)
        for mode in ((0, 1) if i % 2 == 0 else (1, 0)):
            start, end = pair[mode]
            start.record()
            for _ in range(replays):
                graphs[mode].replay()
            end.record()
    torch.npu.synchronize()
    elapsed = [[start.elapsed_time(end) * 1000 / replays for start, end in pair] for pair in timers]
    if any(t <= 0 for pair in elapsed for t in pair):
        raise RuntimeError(f"invalid device timing in {elapsed!r}")
    gains = [old - new for old, new in elapsed]
    result = {
        f"{names[0]}_us": statistics.median(old for old, _ in elapsed),
        f"{names[1]}_us": statistics.median(new for _, new in elapsed),
        "paired_saved_us": statistics.median(gains),
        f"{names[1]}_faster_pairs": sum(gain > 0 for gain in gains),
        "pairs": pairs,
    }
    if include_means:
        result[f"{names[0]}_mean_us"] = statistics.mean(old for old, _ in elapsed)
        result[f"{names[1]}_mean_us"] = statistics.mean(new for _, new in elapsed)
        result["paired_mean_saved_us"] = statistics.mean(gains)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-ascend-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--query-rows", type=int, choices=(1, 2), default=2)
    parser.add_argument("--misses", type=int, default=410)
    parser.add_argument("--main-gate", action="store_true", help="Open the V-up and FP32 router windows")
    parser.add_argument("--multi-layer", action="store_true", help="Also measure queuing all three consumers together")
    parser.add_argument("--model-config", type=Path, help="Use GLM dimensions and W8A8 dynamic QKV from config.json")
    parser.add_argument("--tp-size", type=int, default=4, help="TP degree for the per-rank V-up projection")
    parser.add_argument("--pieces", type=int, help="Total pieces per prefetch (gated default 16, ordinary default 1)")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--json", type=Path, default=Path("retrieval-overlap.json"))
    args = parser.parse_args()
    if args.requests <= 0 or args.requests * args.query_rows > 1024 or not 0 <= args.misses <= 4096:
        parser.error("invalid request count or misses (0..4096)")
    if args.iterations <= 0 or args.repeats < 2:
        parser.error("iterations must be positive and repeats >= 2")
    if args.pieces is not None and not 1 <= args.pieces <= args.requests * 4096:
        parser.error("pieces must be in [1, packed token capacity]")
    if args.model_config:
        torch.npu.config.allow_internal_format = True
    workload = Workload(args.lmcache_ascend_dir, args.requests, args.query_rows, main_gate=args.main_gate,
                        pieces=args.pieces, multi_layer=args.multi_layer)
    try:
        if args.model_config:
            config = json.loads(args.model_config.read_text(encoding="utf-8"))
            workload.use_glm_projections(config.get("text_config") or config, args.tp_size)
        modes = ("serial", "overlap", "multi_layer") if args.multi_layer else ("serial", "overlap")
        checks = [workload.capture(m != "serial", observe=True, multi_layer=m == "multi_layer") for m in modes]
        for active, shift in ((0, 0), (args.misses, 3), (4096, 4096)):
            for graph in checks:
                tokens = workload.set_payload(active, shift, idle=args.requests > 1)
                graph.replay()
                workload.check(tokens, active, idle=args.requests > 1)
        workload.set_payload(args.misses)
        graphs = [workload.capture(m != "serial", multi_layer=m == "multi_layer") for m in modes]
        samples, host = [[] for _ in modes], [[] for _ in modes]
        for graph in graphs:
            for _ in range(10):
                graph.replay()
        torch.npu.synchronize()
        timers = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                  for _ in range(args.iterations)]
        for start, end in timers:
            start.record()
            end.record()
        torch.npu.synchronize()
        for repeat in range(args.repeats):
            order = range(len(modes))
            for mode in (order if repeat % 2 == 0 else reversed(order)):
                begin = time.perf_counter()
                for start, end in timers:
                    start.record()
                    graphs[mode].replay()
                    end.record()
                host[mode].append((time.perf_counter() - begin) * 1e6 / args.iterations)
                timers[-1][1].synchronize()
                samples[mode].extend(start.elapsed_time(end) * 1000 for start, end in timers)
        result = {"scope": "registered-CPU retrieval plus projection microbenchmark",
                  "requests": args.requests, "query_rows": args.query_rows,
                  "main_gate": args.main_gate, "pieces": workload.edges[0].pieces,
                  "max_aiv_cores": workload.max_aiv_cores,
                  "copied_bytes_per_chain": args.requests * args.misses * 576 * 2 * 4,
                  "methods": {}}
        for mode, name in enumerate(modes):
            result["methods"][name] = {"samples_us": samples[mode],
                                       "median_us": statistics.median(samples[mode]),
                                       "p95_us": sorted(samples[mode])[int(.95 * (len(samples[mode]) - 1))],
                                       "host_enqueue_including_events_us": host[mode]}
        if args.profile_dir:
            import torch_npu
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            for mode, name in enumerate(modes):
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                                           torch_npu.profiler.ProfilerActivity.NPU]) as prof:
                    for _ in range(10):
                        graphs[mode].replay()
                    torch.npu.synchronize()
                prof.export_chrome_trace(str(args.profile_dir / f"{name}.json"))
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        workload.close()


if __name__ == "__main__":
    main()
