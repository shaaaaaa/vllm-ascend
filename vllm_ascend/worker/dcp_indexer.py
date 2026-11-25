# SPDX-License-Identifier: Apache-2.0
"""Request-major indexer rows on a rank-local DCP owner."""

from dataclasses import dataclass, replace

import numpy as np
import torch


def bind_block_tables(output, owners, rank):
    """Resolve scheduler handles before either persistent worker table sees them."""
    for req in output.finished_req_ids | (output.preempted_req_ids or set()):
        owners.pop(req, None)

    def local(req, ids):
        owner, epoch, latent = output.dcp_indexer_placements[req]
        owners[req] = (owner, epoch)
        return list(latent[rank]), ids[1] if owner == rank else [0] * len(ids[1])

    for item in output.scheduled_new_reqs:
        item.block_ids = local(item.req_id, item.block_ids)
    cached = output.scheduled_cached_reqs
    for i, req in enumerate(cached.req_ids):
        if req in output.dcp_indexer_placements:
            cached.new_block_ids[i] = local(req, cached.new_block_ids[i])


def owner_rows(owners, starts, rank, world, requests, tokens, model_tokens, row_ids=()):
    """Pack whole sequences in worker order and invert the rank-major gather."""
    take = np.zeros(tokens, np.int64)
    valid = np.zeros(tokens, np.bool_)
    seq = np.zeros(requests + 1, np.int64)
    present = np.zeros(requests + 1, np.bool_)
    ends = np.zeros(requests + 1, np.int32)
    inverse = np.full(model_tokens, -1, np.int64)
    used = [0] * world
    n = 0
    # The packet order must agree even when workers condensed their local
    # rows differently. The inverse still addresses this worker's model order.
    order = sorted(range(len(owners)), key=row_ids.__getitem__) if row_ids else range(len(owners))
    for row in order:
        owner = owners[row]
        start, end = int(starts[row]), int(starts[row + 1])
        if not 0 <= owner < world or not 0 <= start <= end <= model_tokens:
            raise ValueError("Invalid DCP indexer owner or query interval")
        a, b = used[owner], used[owner] + end - start
        if b > tokens:
            raise ValueError("DCP indexer rows exceed their graph bucket")
        inverse[start:end] = owner * tokens + np.arange(a, b)
        if owner == rank:
            if n >= requests:
                raise ValueError("DCP indexer requests exceed their graph bucket")
            take[a:b], valid[a:b] = np.arange(start, end), True
            seq[n], present[n], ends[n] = row, True, b
            n += 1
        used[owner] = b
    ends[n:-1] = used[rank]
    ends[-1] = tokens  # Positive-sized native invocation even on an empty owner.
    return take, valid, seq, present, ends, inverse


@dataclass
class OwnerRows:
    take: torch.Tensor
    valid: torch.Tensor
    seq: torch.Tensor
    present: torch.Tensor
    ends: torch.Tensor
    inverse: torch.Tensor
    tables: tuple[torch.Tensor, ...] = ()

    def pack(self, value):
        rows = value.index_select(0, self.take)
        return torch.where(self.valid.view((-1,) + (1,) * (rows.ndim - 1)), rows, 0)

    def store(self, mapping, keys, scales, cache):
        from vllm_ascend.ops.triton.sfa_dcp import store_owner_keys

        store_owner_keys(self.take, self.valid, mapping, keys, scales, cache)

    def bind_tables(self, arena, table, c8, max_requests):
        # Called before target/draft forward. Each builder owns its buffers
        # and each graph shape sees a stable address. Async metadata may
        # reuse the result only while the order and allocation are unchanged.
        tables = []
        for i, src in enumerate((table, c8)):
            if src is None:
                continue
            if i == len(arena):
                arena.append(src.new_empty((max_requests + 1, src.shape[1])))
            out = arena[i][: self.seq.numel()]
            torch.index_select(src, 0, self.seq, out=out)
            out.masked_fill_(~self.present[:, None], 0)
            tables.append(out)
        return replace(self, tables=tuple(tables))

    def score_inputs(self, table, lengths, c8=False):
        local = self.tables[-1 if c8 else 0] if self.tables else table.index_select(0, self.seq)
        return (local, self.ends, torch.where(self.present, lengths.index_select(0, self.seq), 0))

    def restore(self, topk, group, model_tokens):
        import torch.distributed as dist

        packet = topk.contiguous()
        gathered = packet.new_empty((group.world_size * packet.shape[0], *packet.shape[1:]))
        dist.all_gather_into_tensor(gathered, packet, group=group.device_group)
        # Draft running eagerly can have a different padded extent from its target.
        inverse = self.inverse[:model_tokens]
        rows = gathered.index_select(0, inverse.clamp_min(0))
        rows = torch.where((inverse >= 0).view(-1, 1, 1), rows, -1)
        if rows.shape[0] < model_tokens:
            rows = torch.nn.functional.pad(rows, (0, 0, 0, 0, 0, model_tokens - rows.shape[0]), value=-1)
        return rows


class OwnerWorkspace:
    def __init__(self, runner):
        t, r = runner.max_num_tokens, runner.max_num_reqs + 1
        self.signature = None
        self.view = None
        self.buffers = tuple(
            runner._make_buffer(n, dtype=dtype)
            for n, dtype in (
                (t, torch.int64),
                (t, torch.bool),
                (r, torch.int64),
                (r, torch.bool),
                (r, torch.int32),
                (t, torch.int64),
            )
        )

    def update(self, owners, starts, rank, world, requests, tokens, model_tokens, row_ids=()):
        signature = (tuple(row_ids), tuple(owners), tuple(starts) if owners else (), requests, tokens, model_tokens)
        if signature == self.signature:
            return self.view
        arrays = owner_rows(owners, starts, rank, world, requests, tokens, model_tokens, row_ids)
        views = []
        for buffer, array in zip(self.buffers, arrays, strict=True):
            buffer.np[: len(array)] = array
            views.append(buffer.copy_to_gpu(len(array)))
        self.view = OwnerRows(*views)
        self.signature = signature
        return self.view
