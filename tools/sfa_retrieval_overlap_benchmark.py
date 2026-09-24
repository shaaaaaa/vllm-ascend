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
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_ascend.compilation.sfa_retrieval_overlap import CaptureProbe, RetrievalEdge


class Workload:
    def __init__(self, root, requests=8, query_rows=2, layers=4, width=4096, history=4097, chunk=1024):
        import torch_npu  # noqa: F401
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        sys.path.insert(0, str(Path(root) / "benchmark/v1/kv_transfer"))
        from load_benchmark_utils import build_chunk_ptrs_npu, ensure_ascend_host_memory_registered
        from lmcache.v1.gpu_connector.sparse import PreparedSparseSource, PreparedSparseSourceLayer
        from lmcache.v1.memory_management import PinMemoryAllocator
        from lmcache_ascend.v1.npu_connector.sparse_graph import SparseGraphTransfer

        ensure_ascend_host_memory_registered()
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        self.requests, self.layers, self.width, self.history = requests, layers, width, history
        self.owners, self.sources, self.transfers, self.graphs = [], [], [], []
        self.stream = torch.npu.Stream(device=self.device)
        self.probe = CaptureProbe(self.stream, self.device)
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
                    plane_layers.append(PreparedSparseSourceLayer(tuple(chunks), build_chunk_ptrs_npu(chunks, self.device)))
                self.sources.append(PreparedSparseSource(layers=tuple(plane_layers), total_tokens=history,
                                    chunk_token_counts=sizes, pointer_device=self.device))
            for layer, cache in enumerate(self.caches):
                transfer = SparseGraphTransfer(cache, self.slots, chunk, history, request_capacity=requests)
                transfer.bind_batch(self.sources, layer)
                self.transfers.append(transfer)
            self.edges = [RetrievalEdge(self.stream, self.transfers[i], self.caches[i]) for i in range(1, layers)]
            self.x = torch.ones((requests * query_rows, 16, 512), dtype=torch.bfloat16, device=self.device)
            self.weight = torch.ones((16, 512, 128), dtype=torch.bfloat16, device=self.device)
            self.products = [torch.empty((requests * query_rows, 16, 128), dtype=torch.bfloat16, device=self.device)
                             for _ in range(layers)]
            self.observed = [tuple(torch.empty((requests, width), dtype=torch.bfloat16, device=self.device)
                                   for _ in range(2)) for _ in range(layers)]
        except BaseException:
            self.close()
            raise

    @staticmethod
    def values(tokens, request, layer):
        return ((tokens + request * 7 + layer * 11) % 127).float() / 8

    def set_payload(self, active, shift=0, idle=False):
        tokens = ((torch.arange(self.width) * 17 + shift) % self.history).int()
        self.selected.copy_(tokens.expand(self.requests, -1))
        self.counts.fill_(active)
        if idle:
            self.counts[-1].zero_()
        for cache in self.caches:
            for plane in cache:
                plane.fill_(-7)
        return tokens

    def chain(self, overlap, observe):
        for layer, transfer in enumerate(self.transfers):
            if overlap and layer:
                self.edges[layer - 1].join()
            else:
                transfer.load(self.selected, self.counts, self.slots)
            if observe:
                # Read at the consumer boundary, not merely after the whole graph.
                for plane, dst in zip(self.caches[layer], self.observed[layer]):
                    dst.copy_(plane.view(-1, plane.shape[-1])[:self.requests * self.width, 0].view_as(dst))
            if overlap and layer + 1 < self.layers:
                self.edges[layer].launch(self.selected, self.counts, self.slots, self.caches[layer + 1])
            torch.ops._C_ascend.batch_matmul_transpose(self.x, self.weight, self.products[layer])

    def capture(self, overlap, observe=False):
        self.chain(overlap, observe)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        self.graphs.append(graph)  # Retain even on a partial capture failure.
        with torch.npu.graph(graph):
            self.chain(overlap, observe)
        return graph

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-ascend-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--query-rows", type=int, choices=(1, 2), default=2)
    parser.add_argument("--misses", type=int, default=410)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--json", type=Path, default=Path("retrieval-overlap.json"))
    args = parser.parse_args()
    if args.requests <= 0 or args.requests * args.query_rows > 1024 or not 0 <= args.misses <= 4096:
        parser.error("invalid request count or misses (0..4096)")
    if args.iterations <= 0 or args.repeats < 2:
        parser.error("iterations must be positive and repeats >= 2")
    workload = Workload(args.lmcache_ascend_dir, args.requests, args.query_rows)
    try:
        checks = [workload.capture(mode, observe=True) for mode in (False, True)]
        for active, shift in ((0, 0), (args.misses, 3), (4096, 4096)):
            for graph in checks:
                tokens = workload.set_payload(active, shift, idle=args.requests > 1)
                graph.replay()
                workload.check(tokens, active, idle=args.requests > 1)
        workload.set_payload(args.misses)
        graphs = [workload.capture(mode) for mode in (False, True)]
        samples, host = [[], []], [[], []]
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
            for mode in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                begin = time.perf_counter()
                for start, end in timers:
                    start.record()
                    graphs[mode].replay()
                    end.record()
                host[mode].append((time.perf_counter() - begin) * 1e6 / args.iterations)
                timers[-1][1].synchronize()
                samples[mode].extend(start.elapsed_time(end) * 1000 for start, end in timers)
        result = {"scope": "registered-CPU retrieval plus cube projection; excludes SFA/O-proj/MoE",
                  "requests": args.requests, "query_rows": args.query_rows,
                  "copied_bytes_per_chain": args.requests * args.misses * 576 * 2 * 4,
                  "methods": {}}
        for mode, name in enumerate(("serial", "overlap")):
            result["methods"][name] = {"samples_us": samples[mode],
                                       "median_us": statistics.median(samples[mode]),
                                       "p95_us": sorted(samples[mode])[int(.95 * (len(samples[mode]) - 1))],
                                       "host_enqueue_including_events_us": host[mode]}
        if args.profile_dir:
            import torch_npu
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            for mode, name in enumerate(("serial", "overlap")):
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
