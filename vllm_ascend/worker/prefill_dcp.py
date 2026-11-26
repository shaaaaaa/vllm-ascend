# SPDX-License-Identifier: Apache-2.0
"""Group-0 history exchange for the banked prefill sender."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


def workspace_blocks(max_len, requests, block, world, banks, page_bytes, budget):
    """Bound the send and dense views by both scheduling and four-bank shard capacity."""
    pages = (max_len + block - 1) // block
    send = requests * ((pages + world - 1) // world)
    padding = 1 + requests * (world - 1)
    if budget is not None:
        # Each admitted historical column consumes at least one G0 page in *each* bank.
        # After reserving (world+1)*columns + padding pages, the remaining pool
        # can admit at most columns such stripes. G1 shares that same pool.
        divisor = banks + world + 1
        by_pool = max(1, (budget // page_bytes - padding + divisor - 1) // divisor)
        if by_pool < send:
            return padding + world * by_pool, by_pool
    return padding + requests * pages, send


class PrefillDcpWorkspace:
    def __init__(self, runner, group, budget):
        from vllm_ascend.ops.triton.sfa_dcp import pack_prefill_history, store_prefill_rows

        self.pack = pack_prefill_history
        self.store_rows = store_prefill_rows
        self.world = group.world_size
        self.rank = group.rank_in_group
        self.group = group.device_group
        self.block_size = block = runner.block_size
        page_bytes = block * sum(runner.sparse_head_dim[:2]) * runner.kv_cache_dtype.itemsize
        dense, send = workspace_blocks(
            runner.max_model_len,
            runner.max_num_reqs,
            block,
            self.world,
            runner.layerwise_prefill_bank_count,
            page_bytes,
            budget,
        )

        def planes(count):
            return tuple(
                torch.empty((count, block, 1, width), dtype=runner.kv_cache_dtype, device=runner.device)
                for width in runner.sparse_head_dim[:2]
            )

        self.stream = torch.npu.Stream(device=runner.device)
        self.planes = planes(dense)
        self.send = planes(send)
        self.save_events = {}
        self.last_read = None
        self.physical_capacity = 0
        self.unfenced = False

    def bind_banks(self, layers):
        """Validate the native plane strides once, before accepting raw block addresses."""
        capacity = None
        for cache in layers:
            if len(cache) != 2 or any(
                plane.shape[1:] != dense.shape[1:] or plane.dtype != dense.dtype or not plane.is_contiguous()
                for plane, dense in zip(cache, self.planes)
            ):
                raise ValueError("Invalid banked prefill latent/RoPE planes")
            if capacity is None:
                capacity = len(cache[0])
            if any(len(plane) != capacity for plane in cache):
                raise ValueError("Inconsistent banked prefill slab capacity")
        if not capacity:
            raise ValueError("Missing banked prefill Group-0 cache")
        self.physical_capacity = capacity

    def exchange(self, cache, count, source):
        works = []
        try:
            if isinstance(source, int):
                inputs = (plane.narrow(0, source, count) for plane in cache[:2])
            else:
                inputs = tuple(plane[:count] for plane in self.send)
                self.pack(cache, source, inputs)
            for out, local in zip(self.planes, inputs):
                works.append(
                    dist.all_gather_into_tensor(
                        out.narrow(0, 1, count * self.world), local, group=self.group, async_op=True
                    )
                )
            for work in works:
                work.wait()
            return self.stream.record_event()
        except BaseException:
            # An earlier pack/gather may still own these buffers. A failed peer
            # cannot be relied on to complete a new collective or a host fence.
            self.unfenced = True
            self._failed_exchange = (cache, source, works)
            raise

    def warmup(self):
        # Initialize HCCL on the production stream before sizing the shared slab.
        with torch.npu.stream(self.stream):
            ids = torch.zeros(1, dtype=torch.int32, device=self.planes[0].device)
            self.exchange(self.planes, 1, ids)
            packet = self.planes[0].new_zeros((1, sum(p.shape[-1] for p in self.planes)))
            for slots in (ids, ids.long()):
                self.store_rows(packet, slots, ids, self.planes, self.send, 1, self.rank, self.world)
        torch.npu.current_stream().wait_stream(self.stream)


@dataclass
class PrefillDcpBatch:
    workspace: PrefillDcpWorkspace
    save_blocks: dict[str, np.ndarray]
    coverage: dict[str, tuple[int, int]]
    table: torch.Tensor
    slots: torch.Tensor
    sources: tuple[int | torch.Tensor, ...]
    physical_blocks: torch.Tensor
    refs: dict
    count: int
    partial_history: bool
    maps_ready: object

    @property
    def rank(self):
        return self.workspace.rank

    @classmethod
    def build(cls, workspace, req_ids, history, ends, banks, refs, num_rows, num_tokens):
        """Build a shared dense table and four compact send maps in worker row order."""
        block, world = workspace.block_size, workspace.world
        hist = (history + block - 1) // block
        pages = (ends + block - 1) // block
        columns = (hist + world - 1) // world
        count = int(columns.sum())
        if (
            len(req_ids) != len(history)
            or len(req_ids) > num_rows
            or (ends - history).sum() > num_tokens
            or (history < 0).any()
            or (ends < history).any()
            or 1 + world * count + (pages - hist).sum() > len(workspace.planes[0])
            or count > len(workspace.send[0])
        ):
            raise ValueError("Scheduled prefill exceeds the reserved DCP workspace")
        table = np.zeros((num_rows, max(1, int(pages.max(initial=0)))), dtype=np.int32)
        slots = np.full(num_tokens, -1, dtype=np.int64)
        source = np.zeros((len(banks), count + int((pages - hist).sum())), dtype=np.int32)
        save_blocks = {}
        coverage = {}
        offset = row_start = 0
        fresh = 1 + world * count
        for i, req_id in enumerate(req_ids):
            h, end, m = int(hist[i]), int(pages[i]), int(columns[i])
            logical = np.arange(h, dtype=np.int64)
            ids = table[i, :end]
            ids[:h] = 1 + logical % world * count + offset + logical // world
            ids[h:] = np.arange(fresh, fresh + end - h)
            save_blocks[req_id] = ids
            coverage[req_id] = int(history[i]), int(ends[i])
            positions = np.arange(history[i], ends[i], dtype=np.int64)
            slots[row_start : row_start + len(positions)] = (
                ids[positions // block] * np.int64(block) + positions % block
            )
            owned = (h + world - 1 - workspace.rank) // world
            # Reuse the history map for the boundary block. Only new-only blocks need extra IDs.
            suffix = np.arange(h + (workspace.rank - h) % world, end, world)
            needed = (end + world - 1) // world
            for j, (bank, lengths) in enumerate(banks):
                # add_row() leaves old IDs after the live prefix. All peers must
                # validate the same columns before any rank enters the gather.
                blocks = bank[i, :needed]
                if (
                    lengths[i] < needed
                    or len(blocks) != needed
                    or (blocks <= 0).any()
                    or (blocks >= workspace.physical_capacity).any()
                ):
                    raise ValueError(f"Missing Group-0 bank {j} allocation for {req_id}")
                source[j, offset : offset + owned] = blocks[:owned]
                source[j, fresh - 1 - (world - 1) * count + suffix - h] = blocks[suffix // world]
            row_start += len(positions)
            offset += m
            fresh += end - h
        device = workspace.planes[0].device

        def upload(array):
            return torch.from_numpy(array).to(device=device, non_blocking=True)

        physical = upload(source)
        sources = tuple(
            int(ids[0]) if count and ids[0] > 0 and np.all(np.diff(ids[:count]) == 1) else physical[j, :count]
            for j, ids in enumerate(source)
        )
        maps_ready = None
        if count:
            physical.record_stream(workspace.stream)
            maps_ready = torch.npu.current_stream().record_event()
        return cls(
            workspace,
            save_blocks,
            coverage,
            upload(table),
            upload(slots),
            sources,
            physical,
            refs,
            count,
            bool((history % block != 0).any()),
            maps_ready,
        )

    def start(self, layer_name, cache, wait_for_layer):
        """Return the dependencies for fresh writes and for reading history."""
        ws = self.workspace
        if ws.unfenced:
            raise RuntimeError("Cannot reuse prefill DCP workspace after an unfenced transfer")
        reuse = tuple(ws.save_events.values())
        if ws.last_read is not None:
            reuse += (ws.last_read,)
        if not self.count:
            return reuse, None
        with torch.npu.stream(ws.stream):
            ws.stream.wait_event(self.maps_ready)
            for event in reuse:
                ws.stream.wait_event(event)
            wait_for_layer(layer_name)
            # A recalc-last load can include the first fresh token even on a block boundary.
            bank_ready = None if self.partial_history else ws.stream.record_event()
            ready = ws.exchange(cache, self.count, self.sources[self.refs[layer_name].bank])
        # A full-block receive would overwrite fresh rows sharing the historical tail.
        return ((ready,), None) if self.partial_history else ((bank_ready,), ready)

    def wait_for_writes(self, events):
        for event in events:
            torch.npu.current_stream().wait_event(event)
        self.workspace.save_events.clear()

    def store(self, name, packet, slots, cache):
        self.workspace.store_rows(
            packet,
            slots,
            self.physical_blocks[self.refs[name].bank],
            self.workspace.planes,
            cache,
            self.count,
            self.rank,
            self.workspace.world,
        )
