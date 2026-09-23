# SPDX-License-Identifier: Apache-2.0
"""Prepared LMCache transfer adapter; imports serving dependencies only on NPU."""
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

from candidates import Query, Snapshot
from matched_data import initial_slots, payload
from native import NativeCase


def bind_transfer_api(helpers, native):
    # utils.py may be newer than the installed c_ops binary. Both native versions
    # accept eight arguments; newer bindings default diagnostic_layer_id to -1.
    # Keep this compatibility choice local to the experiment, without monkeypatching
    # the production module or retrying a possibly already-submitted operation.
    names = ('__file__', 'MlaDsaDims', 'compute_chunk_partition', 'create_pin_memory_allocator',
             'release_pin_memory_objects', 'build_chunk_ptrs_npu',
             'prepare_sparse_direct_destination_state', 'KV_FORMAT_MLA_LATENT')
    return SimpleNamespace(**{name: getattr(helpers, name) for name in names},
        sparse_mla_dsa_batched_direct_kv_transfer_prepared=native.sparse_mla_dsa_batched_direct_kv_transfer_prepared)


def load_transfer_api(root):
    helpers = Path(root) / 'benchmark/v1/kv_transfer'
    if not (helpers / 'load_benchmark_utils.py').is_file():
        raise FileNotFoundError(f'missing production transfer helpers: {helpers}')
    sys.path.insert(0, str(helpers))
    import load_benchmark_utils as api
    import lmcache_ascend.c_ops as native_ops
    if Path(api.__file__).resolve() != (helpers / 'load_benchmark_utils.py').resolve():
        raise RuntimeError('a different load_benchmark_utils module is already loaded; use a fresh process')
    return bind_transfer_api(api, native_ops)


class Sources:
    def __init__(self, api, trace, device, chunk_size=1024, widths=(512, 64)):
        self.api, self.trace, self.device = api, trace, device
        self.widths, self.chunk_size = widths, chunk_size
        self.allocators, self.objects, self.chunks, self.pointers = [], [], [], []
        self.live, self.live_ptrs = [], []
        self.registered_payload_bytes = 0
        self.pages, self.seed_slots, self.slots = initial_slots(trace)
        self.r, self.n = trace['initial_tokens'].shape
        self.prefix = trace['boundary'].tolist()
        self.tail = (trace['lengths'] - trace['boundary']).tolist()
        dims = api.MlaDsaDims(widths[0], widths[1], 0, sum(widths))
        try:
            for r in range(self.r):
                partition = api.compute_chunk_partition(self.prefix[r], chunk_size)
                allocator = api.create_pin_memory_allocator(partition, dims, torch.bfloat16)
                self.allocators.append(allocator)
                # Allocate one chunk at a time so failures retain ownership for cleanup.
                chunks = []
                for start, count in zip(partition.chunk_offsets, partition.chunk_sizes, strict=True):
                    obj = allocator.allocate(torch.Size([count * sum(widths)]), torch.bfloat16)
                    if obj is None or obj.tensor is None:
                        raise RuntimeError('registered CPU allocation failed')
                    self.objects.append(obj)
                    chunk = obj.tensor
                    tokens = torch.arange(start, start + count)
                    chunk[:count*widths[0]].view(count, widths[0]).copy_(payload(tokens, r, widths[0], 0))
                    chunk[count*widths[0]:].view(count, widths[1]).copy_(payload(tokens, r, widths[1], 1))
                    chunks.append(chunk)
                    self.registered_payload_bytes += chunk.numel() * chunk.element_size()
                self.chunks.append(chunks)
                self.pointers.append(api.build_chunk_ptrs_npu(chunks, device))
                length = max(1, self.tail[r])
                tokens = torch.arange(self.prefix[r], self.prefix[r]+length)
                live = torch.cat([payload(tokens, r, width, plane).flatten()
                                  for plane, width in enumerate(widths)]).to(device)
                self.live.append(live)
                self.live_ptrs.append(torch.tensor([live.data_ptr()], dtype=torch.long, device=device))
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.allocators:
            torch.npu.synchronize()
        self.api.release_pin_memory_objects(self.objects)
        self.objects.clear()
        for allocator in self.allocators:
            allocator.close()
        self.allocators.clear()

    def bank(self):
        return torch.empty((self.r, self.n * sum(self.widths)), dtype=torch.bfloat16, device=self.device)

    def planes(self, bank, r):
        offset = self.n * self.widths[0]
        return [bank[r, :offset].view(-1, 128, 1, self.widths[0]),
                bank[r, offset:].view(-1, 128, 1, self.widths[1])]

    def seed(self, bank):
        bank.zero_()
        for r in range(self.r):
            ready = self.trace['initial_ready'][r]
            slots = self.seed_slots[r, ready].to(self.device)
            tokens = self.trace['initial_tokens'][r, ready]
            for plane, view in enumerate(self.planes(bank, r)):
                view.flatten(0, 2).index_copy_(0, slots, payload(tokens, r, self.widths[plane], plane).to(self.device))

    def states(self, bank, slots):
        return [self.api.prepare_sparse_direct_destination_state(self.planes(bank, r), slots[r],
                self.api.KV_FORMAT_MLA_LATENT, *self.widths, 0) for r in range(self.r)]

    def transfer(self, state, slots, selected, counts, pointers, chunk_size, total):
        self.api.sparse_mla_dsa_batched_direct_kv_transfer_prepared(
            state, slots, selected, pointers, chunk_size, total, False, counts)

    def expected(self, tokens, r):
        valid = tokens >= 0
        return [torch.where(valid[..., None], payload(tokens.clamp_min(0), r, width, plane), 0)
                for plane, width in enumerate(self.widths)]


class Original:
    name = 'original'
    def __init__(self, source, trace, variant='baseline'):
        if variant not in ('baseline', 'vector_intersection', 'vector_state_update', 'exact_combined'):
            raise ValueError('unsupported exact resident variant')
        self.variant = variant
        self.name = 'original' if variant == 'baseline' else variant
        from resident_experiment import make_case, _put_state
        self.source, self.trace = source, trace
        self.s, self.r, self.q, self.k = trace['steps'].shape
        self.cpu = make_case(self.r, self.q, 4, 0, block_size=128)
        self.cpu['state_counts'].zero_()
        self.cpu['block_table'].copy_(source.pages)
        self.cpu['boundary'].copy_(trace['boundary'].repeat_interleave(self.q))
        for r in range(self.r):
            row = int(self.cpu['request_states'][r])
            tokens = trace['initial_tokens'][r, trace['initial_ready'][r]].tolist()
            _put_state(self.cpu, row, dict(zip(tokens, range(len(tokens)), strict=True)))
        self.case = self.cpu.clone(source.device)
        self.reset_tensors = [t.clone() for t in self.case.tensors]
        self.inputs = trace['steps'].to(source.device).reshape(self.s, self.r*self.q, 1, self.k)
        self.bank = source.bank()
        source.seed(self.bank)
        self.seed_bank = self.bank.clone()
        self.states = source.states(self.bank, self.case['target_slots'])
        self.loads = [(self.states[r], self.case['target_slots'][r].view(1, -1),
                       self.case['miss_tokens'][r].view(1, -1), self.case['miss_counts'][r, :1],
                       source.pointers[r], source.chunk_size, source.prefix[r]) for r in range(self.r)]

    def reset(self):
        for dst, src in zip(self.case.tensors, self.reset_tensors, strict=True):
            dst.copy_(src)
        self.bank.copy_(self.seed_bank)

    def step(self, index):
        self.case['topk'].copy_(self.inputs[index])
        self.case.run(self.variant)
        for args in self.loads:
            self.source.transfer(*args)

    def check(self, index):
        counts = self.case['miss_counts'][:, 0].cpu()
        mapped = self.case['topk'].cpu().reshape(self.r, self.q, self.k)
        self.last_misses = int(counts.sum())
        self.last_hits, self.last_tail = 0, 0
        miss_tokens = self.case['miss_tokens'].cpu()
        for r in range(self.r):
            raw = self.trace['steps'][index, r]
            prefix = (raw >= 0) & (raw < self.source.prefix[r])
            missed = set(miss_tokens[r, :int(counts[r])].tolist())
            self.last_hits += sum(token not in missed for token in raw[prefix].tolist())
            self.last_tail += int((raw >= self.source.prefix[r]).sum())
            expected = self.source.expected(raw, r)
            for plane, view in enumerate(self.source.planes(self.bank, r)):
                actual = torch.zeros_like(expected[plane])
                # Existing attention consumes the resident slots without copying hits.
                physical = self.source.slots[r, mapped[r][prefix].long()]
                actual[prefix] = view.flatten(0, 2).cpu()[physical]
                tail = raw >= self.source.prefix[r]
                length = max(1, self.source.tail[r])
                offset = 0 if plane == 0 else length * self.source.widths[0]
                live = self.source.live[r][offset:offset+length*self.source.widths[plane]].view(length, -1).cpu()
                actual[tail] = live[(raw[tail]-self.source.prefix[r]).long()]
                torch.testing.assert_close(actual, expected[plane], rtol=0, atol=0)


class Replacement:
    def __init__(self, source, trace, variant):
        self.name, self.source, self.trace = variant, source, trace
        self.s, self.r, self.q, self.k = trace['steps'].shape
        n, device = source.n, source.device
        self.inputs = trace['steps'].to(device)
        self.boundary = trace['boundary'].to(device)
        query = Query(self.inputs[0], torch.zeros_like(self.inputs[0]),
                      torch.ones(self.r, self.q, dtype=torch.bool, device=device),
                      self.boundary[:, None].expand(-1, self.q),
                      trace['lengths'].to(device)[:, None].expand(-1, self.q),
                      torch.ones(self.r, dtype=torch.int64, device=device), max(source.prefix)+max(source.tail))
        # Zero-width payloads invoke only metadata kernels; there is NO dense HBM prefix.
        snap = Snapshot(trace['initial_tokens'].to(device), torch.zeros((self.r, n), dtype=torch.int32, device=device),
                        trace['initial_ready'].to(device), query.epochs.clone(),
                        torch.empty(self.r, n, 0, dtype=torch.bfloat16, device=device))
        self.case = NativeCase(query, snap, torch.empty(self.r, query.universe, 0, dtype=torch.bfloat16, device=device),
                               variant, copy_mode='batched', table_mode='wide' if variant == 'direct_directory' else 'baseline')
        self.seed_tags, self.seed_ready = self.case.tensors[2].clone(), self.case.tensors[4].clone()
        self.banks = [source.bank(), source.bank()]
        source.seed(self.banks[0])
        self.seed_bank = self.banks[0].clone()
        self.seed_slots = source.seed_slots.to(device)
        self.current_slots = source.slots.to(device)
        self.old_slots = self.seed_slots.clone()
        self.selected = torch.empty(3, self.r, n//256, 256, dtype=torch.int32, device=device)
        self.targets = torch.empty_like(self.selected, dtype=torch.long)
        self.counts = torch.empty(3, self.r, n//256, 16, dtype=torch.int32, device=device)
        self.pack = [self.case.tensors[8].view(self.r, self.q, self.k), self.case.tensors[0],
                     self.old_slots, self.current_slots, self.boundary, self.selected, self.targets, self.counts]
        self.states = [source.states(bank, self.targets[0]) for bank in self.banks]
        self.bank_ptrs = [[torch.tensor([bank[r].data_ptr()], dtype=torch.long, device=device)
                           for r in range(self.r)] for bank in self.banks]
        self.ready_steps = [((tokens.flatten(1) >= 0) & (tokens.flatten(1) < self.boundary[:, None])).int()
                            for tokens in self.inputs]
        self.loads = []
        for index in range(self.s):
            old, new = index % 2, 1-index % 2
            args = []
            for r in range(self.r):
                for kind, ptr, chunk, total in ((0, source.pointers[r], source.chunk_size, source.prefix[r]),
                                                (1, self.bank_ptrs[old][r], n, n),
                                                (2, source.live_ptrs[r], max(1, source.tail[r]), max(1, source.tail[r]))):
                    args.append((self.states[new][r], self.targets[kind, r], self.selected[kind, r],
                                 self.counts[kind, r, :, 0], ptr, chunk, total))
            self.loads.append(args)

    def reset(self):
        self.case.tensors[2].copy_(self.seed_tags)
        self.case.tensors[4].copy_(self.seed_ready)
        self.old_slots.copy_(self.seed_slots)
        self.banks[0].copy_(self.seed_bank)
        self.banks[1].zero_()

    def step(self, index):
        self.case.tensors[0].copy_(self.inputs[index])
        self.case.run(False)
        torch.ops.resident_redesign.pack_sources_(self.pack)
        for args in self.loads[index]:
            self.source.transfer(*args)
        # Publication follows every transfer on this same stream. Immutable old bank
        # is never overwritten by its own misses; next step reads the completed bank.
        self.case.tensors[2].copy_(self.inputs[index].flatten(1))
        self.case.tensors[4].copy_(self.ready_steps[index])
        self.old_slots.copy_(self.current_slots)

    def check(self, index):
        self.last_misses = int(self.counts[0, :, :, 0].sum().cpu())
        self.last_hits = int(self.counts[1, :, :, 0].sum().cpu())
        self.last_tail = int(self.counts[2, :, :, 0].sum().cpu())
        bank = self.banks[1-index % 2]
        for r in range(self.r):
            raw = self.trace['steps'][index, r]
            expected = self.source.expected(raw, r)
            for plane, view in enumerate(self.source.planes(bank, r)):
                actual = view.flatten(0, 2).cpu()[self.source.slots[r]].reshape(self.q, self.k, -1)
                actual[raw < 0] = 0
                torch.testing.assert_close(actual, expected[plane], rtol=0, atol=0)
