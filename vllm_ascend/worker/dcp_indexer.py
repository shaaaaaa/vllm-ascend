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


def planner_owner_rows(starts, inverse, width):
    """One scalar read per bounded lane: high word = model row, low word = packet row + 1."""
    tokens = len(inverse)
    if width not in (1, 2) or tokens % width or len(starts) == 0 or starts[0] != 0 or starts[-1] > tokens:
        raise ValueError("Invalid bounded DCP owner query extents")
    lanes = np.full(tokens, -1, np.int64)
    for req, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        if not 0 <= end - start <= width or req * width + width > tokens:
            raise ValueError("DCP owner query rows exceed the bounded planner bucket")
        lanes[req * width:req * width + end - start] = np.arange(start, end)
    lanes[lanes < 0] = np.arange(starts[-1], tokens)
    return (lanes << 32) | (inverse[lanes] + 1)


@dataclass
class GatheredTopk:
    gathered: torch.Tensor
    inverse: torch.Tensor
    positions: torch.Tensor | None
    published: torch.Tensor | None
    actual_rows: int
    reject: bool
    lanes: torch.Tensor | None

    def supports_native_publication(self):
        # The native writer transfers contiguous blocks of int32 IDs from UB.
        raw = self.published
        return raw is None or (raw.dtype == torch.int32 and raw.stride(-1) == 1
                               and raw.stride(0) >= self.gathered.shape[-1] and raw.stride(0) % 8 == 0
                               and raw.storage_offset() % 8 == 0)

    def materialize(self, output):
        from vllm_ascend.ops.triton.sfa_dcp import restore_owner_topk

        restore_owner_topk(self.gathered, self.inverse, output, self.positions, self.actual_rows,
                           self.published, causal=self.positions is not None, reject=self.reject)


@dataclass
class OwnerRows:
    take: torch.Tensor
    valid: torch.Tensor
    seq: torch.Tensor
    present: torch.Tensor
    ends: torch.Tensor
    inverse: torch.Tensor
    tables: tuple[torch.Tensor, ...] = ()
    planner_lanes: torch.Tensor | None = None

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

    @staticmethod
    def gather(topk, group):
        import torch.distributed as dist

        packet = topk.contiguous()
        gathered = packet.new_empty((group.world_size * packet.shape[0], *packet.shape[1:]))
        dist.all_gather_into_tensor(gathered, packet, group=group.device_group)
        return gathered

    def restore(self, topk, group, model_tokens, *, output=None, positions=None,
                actual_rows=None, published=None, reject=False):
        """Publish ordered raw IDs and fill the private planner output in one pass.

        Native dense/mixed batches request raw rows for subsequent local-stripe compaction.
        """
        from vllm_ascend.ops.triton.sfa_dcp import restore_owner_topk

        gathered = self.gather(topk, group)
        causal = output is not None and positions is not None
        if output is None:
            output = topk.new_empty((model_tokens, *topk.shape[1:]))
        elif output.shape[0] != model_tokens:
            raise ValueError("DCP planner and model disagree on the Top-K row count")
        restore_owner_topk(
            gathered, self.inverse, output, positions if causal or reject else None,
            model_tokens if actual_rows is None else actual_rows, published, causal=causal, reject=reject,
        )
        return output


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
                (t + min(t, runner.max_num_reqs * runner.decode_threshold), torch.int64),
            )
        )

    def update(self, owners, starts, rank, world, requests, tokens, model_tokens, row_ids=(), bounded_width=0):
        signature = (tuple(row_ids), tuple(owners), tuple(starts) if owners else (), requests, tokens, model_tokens,
                     bounded_width)
        if signature == self.signature:
            return self.view
        arrays = owner_rows(owners, starts, rank, world, requests, tokens, model_tokens, row_ids)
        if bounded_width:
            # Transfer the two maps together. Fixed decode only sends the inverse.
            lanes = planner_owner_rows(starts[:len(owners) + 1], arrays[-1], bounded_width)
            arrays = (*arrays[:-1], np.concatenate((arrays[-1], lanes)))
        views = []
        for buffer, array in zip(self.buffers, arrays, strict=True):
            buffer.np[: len(array)] = array
            views.append(buffer.copy_to_gpu(len(array)))
        if bounded_width:
            combined = views[-1]
            self.view = OwnerRows(*views[:-1], combined[:model_tokens], planner_lanes=combined[model_tokens:])
        else:
            self.view = OwnerRows(*views)
        self.signature = signature
        return self.view
