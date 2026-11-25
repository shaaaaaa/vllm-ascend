# SPDX-License-Identifier: Apache-2.0
"""Compare serial and side-stream query unpack across a complete DCP layer group."""

import math

import torch
from benchmark_sfa_dcp_query_first import QueryFirstPipeline, main
from benchmark_sfa_target_preparation import BLOCK, QUERY_ROWS, TOPK


class UnpackPipeline(QueryFirstPipeline):
    def __init__(self, args, inputs, sources, group, candidate):
        super().__init__(args, inputs, sources, group, candidate=True)  # Both arms gather Q early.
        self.prefetch_after_pack = args.prefetch_after_pack
        self.unpack_placement = args.unpack_placement if candidate else "off"
        if candidate:
            self.unpack_stream = torch.npu.Stream(device=self.device)
        if args.indexer == "c8":
            self._setup_indexer()

    def _setup_indexer(self):
        args, device = self.args, self.device
        rng = torch.Generator().manual_seed(args.seed + 991)  # Replicated index keys and queries on every rank.
        heads, dim = 64, 128
        blocks = math.ceil((args.history + args.live_tokens + len(self.frames) * QUERY_ROWS) / BLOCK)
        table = torch.arange(args.requests * blocks, dtype=torch.int32).view(args.requests, blocks)
        ends = self.ends[: args.requests].cpu()
        padded = int(ends[-1]) < self.tokens
        query_ends = torch.cat((ends, ends.new_tensor([self.tokens]))) if padded else ends
        keys = torch.randint(-16, 17, (table.numel(), BLOCK, 1, dim), generator=rng, dtype=torch.int8)
        self.index_inputs = dict(
            key=keys.to(device),
            weights=torch.ones((self.tokens, heads), dtype=torch.float16, device=device),
            query_dequant_scale=torch.full((self.tokens, heads), 1 / 16, dtype=torch.float16, device=device),
            key_dequant_scale=torch.full(keys.shape[:3], 1 / 16, dtype=torch.float16, device=device),
            actual_seq_lengths_query=query_ends.to(device),
            block_table=(torch.cat((table, table[:1])) if padded else table).to(device),
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=3,
        )
        base = torch.randint(-16, 17, (self.tokens, heads, dim), generator=rng, dtype=torch.int8)
        self.index_frames = {}
        for raw, positions in self.frames:
            lengths = positions.cpu()[ends.long() - 1].int() + 1
            if padded:
                lengths = torch.cat((lengths, lengths.new_zeros(1)))
            query = base + torch.randint(-2, 3, base.shape, generator=rng, dtype=torch.int8)
            self.index_frames[raw.data_ptr()] = query.to(device), lengths.to(device)

    def prepare(self, frame):
        if self.args.indexer == "c8":
            raw, positions = frame
            query, lengths = self.index_frames[raw.data_ptr()]
            topk = torch.ops._C_ascend.npu_lightning_indexer_quant(
                query=query, actual_seq_lengths_key=lengths, **self.index_inputs
            )
            frame = topk, positions
        return super().prepare(frame)


if __name__ == "__main__":
    main(comparison="early-unpack", pipeline_factory=UnpackPipeline, description=__doc__)
