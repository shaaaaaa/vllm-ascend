# SPDX-License-Identifier: Apache-2.0
"""Compare replicated and request-owned C8 indexing through a complete DCP layer group."""

from types import MethodType, SimpleNamespace

import torch
from benchmark_sfa_dcp_query_first import Projection, QueryFirstPipeline, main
from benchmark_sfa_target_preparation import BLOCK, QUERY_ROWS, TOPK


class IndexerPipeline(QueryFirstPipeline):
    def __init__(self, args, inputs, sources, group, candidate):
        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
        from vllm_ascend.worker.dcp_indexer import OwnerRows, owner_rows

        frames, rows, ends = inputs
        history = [args.history // 2 if args.skew_history and i % 2 else args.history for i in range(args.requests)]
        delta = torch.tensor(history, device=rows.device)[rows.clamp_min(0).long()] - args.history
        frames = [(raw.clone(), pos + delta) for raw, pos in frames]
        super().__init__(args, (frames, rows, ends), sources, group, True)
        if not args.resident:
            self.boundaries = self.boundaries + delta.int()
        owners = [
            (i % args.dcp_size if args.owner_layout == "balanced" else (1 % args.dcp_size))
            for i in range(args.requests)
        ]
        self.owned_lanes = torch.tensor(
            [i for i, row in enumerate(rows.cpu().tolist()) if row >= 0 and owners[row] == args.rank],
            device=self.device,
            dtype=torch.long,
        )
        packed = max(owners.count(r) for r in range(args.dcp_size))
        starts = [0, *ends.cpu().tolist()[: args.requests]]
        arrays = owner_rows(owners, starts, args.rank, args.dcp_size, packed, packed * QUERY_ROWS, len(rows))
        self.owner_rows = OwnerRows(*(torch.from_numpy(a).to(self.device) for a in arrays)) if candidate else None

        g = torch.Generator().manual_seed(args.seed + 193)

        def weight(out, inp):
            return Projection((torch.randn(out, inp, generator=g) * inp**-0.5).to(self.device, torch.bfloat16))

        heads, dim, rope = 64, 128, 64
        h = torch.ones(1, 1)
        while h.shape[0] < dim:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)))
        h = (h / dim**0.5).to(self.device, torch.bfloat16)
        self.indexer = SimpleNamespace(
            has_indexer=True,
            use_sparse_c8_indexer=True,
            head_dim=dim,
            n_head=heads,
            qk_rope_head_dim=rope,
            wk=weight(dim, args.hidden_size),
            wq_b=weight(heads * dim, args.q_rank),
            weights_proj=weight(heads, args.hidden_size),
            k_norm=torch.nn.LayerNorm(dim).to(self.device, torch.bfloat16),
            q_hadamard=h,
            k_hadamard=h,
            c8_k_cache_dtype=torch.int8,
            c8_k_scale_cache_dtype=torch.float16,
            is_rope_neox_style=False,
            index_topk=TOPK,
            _mask_staged_index_scatter_padding=AscendSFAImpl._mask_staged_index_scatter_padding,
        )
        for method in ("indexer_select_pre_process", "indexer_select_post_process"):
            setattr(self.indexer, method, MethodType(getattr(AscendSFAImpl, method), self.indexer))

        pages = [(v + args.live_tokens + len(frames) * QUERY_ROWS + QUERY_ROWS + BLOCK - 1) // BLOCK for v in history]
        table = torch.zeros(self.capacity + 1, max(pages), dtype=torch.int32)
        chunks = [torch.zeros(1, BLOCK, 1, dim, dtype=torch.int8)]
        scales = [torch.ones(1, BLOCK, 1, 1, dtype=torch.float16)]
        cursor = 1
        for i, n in enumerate(pages):
            if candidate and not args.replicate_owner_storage and owners[i] != args.rank:
                continue
            rng = torch.Generator().manual_seed(args.seed + 313 + i)
            table[i, :n] = torch.arange(cursor, cursor + n)
            chunks.append(torch.randint(-64, 65, (n, BLOCK, 1, dim), dtype=torch.int8, generator=rng))
            scales.append((0.01 + torch.rand(n, BLOCK, 1, 1, generator=rng) * 0.01).half())
            cursor += n
        self.idx_cache = (None, None, torch.cat(chunks).to(self.device), torch.cat(scales).to(self.device))
        self.idx_table = table.to(self.device)
        if self.owner_rows is not None:
            self.owner_rows = self.owner_rows.bind_tables([], self.idx_table, None, self.capacity)
        self.idx_frames = {}
        # Graph capture is driven by the original frames in measure_case.
        for (original, _), (raw, pos) in zip(inputs[0], frames, strict=True):
            self.frame_inputs[original.data_ptr()] = self.frame_inputs[raw.data_ptr()]
            seq = self.ends.new_zeros(self.capacity + 1)
            for i in range(args.requests):
                seq[i] = (
                    history[i]
                    + args.live_tokens
                    + (len(self.idx_frames) + 1) * QUERY_ROWS
                    - (QUERY_ROWS - (starts[i + 1] - starts[i]))
                )
            slots = self.idx_table[rows.clamp_min(0).long(), (pos // BLOCK).long()].long() * BLOCK + pos % BLOCK
            self.idx_frames[original.data_ptr()] = pos, seq, torch.where(rows >= 0, slots, 0)
        self.indexer_bytes = sum(t.numel() * t.element_size() for t in self.idx_cache[2:])

    def prepare_producer(self, frame, hidden, q_c, cos, sin):
        import torch_npu

        pos, seq, slots = self.idx_frames[frame[0].data_ptr()]
        self.current_index_slots = slots
        owner = self.owner_rows
        ix, ic, isn = (hidden, cos, sin) if owner is None else tuple(owner.pack(t) for t in (hidden, cos, sin))
        keys, scales = self.indexer.indexer_select_pre_process(ix, ic, isn)
        if owner is None:
            for cache, values in ((self.idx_cache[3], scales), (self.idx_cache[2], keys)):
                flat = cache.view(-1, cache.shape[-1])
                safe_slots, values = self.indexer._mask_staged_index_scatter_padding(slots, values, self.rows, flat)
                torch_npu.npu_scatter_nd_update_(
                    flat, safe_slots.view(-1, 1), values.reshape(-1, cache.shape[-1])
                )
        else:
            owner.store(slots, keys, scales, self.idx_cache)
        ids = self.indexer.indexer_select_post_process(
            ix,
            q_c,
            self.idx_cache,
            None,
            ic,
            isn,
            self.attention_ends,
            seq,
            self.idx_table,
            owner_rows=owner,
            model_tokens=self.tokens,
            dcp_group=self.group,
        )
        ids = torch.where((self.rows >= 0).view(-1, 1, 1), ids, -1)
        return self.prepare((ids, pos))

    def snapshot(self, *, include_cache=True):
        slots = self.current_index_slots.index_select(0, self.owned_lanes)
        assert self.idx_cache[2][0].count_nonzero().item() == 0
        assert self.idx_cache[3][0].eq(1).all().item()
        current = [c.view(-1, c.shape[-1]).index_select(0, slots).cpu() for c in self.idx_cache[2:]]
        return super().snapshot(include_cache=include_cache) + [self.published.cpu(), *current]


if __name__ == "__main__":
    main(comparison="indexer-sharding", pipeline_factory=IndexerPipeline, description=__doc__)
