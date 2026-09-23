# SPDX-License-Identifier: Apache-2.0
from decimal import Decimal

import pytest
import torch

from matched import device_timing, ORIGINAL_SYMBOLS
from matched_data import make_trace, validate_trace, initial_slots, trace_digest, payload


def test_measurement_order_is_balanced_in_each_four_repeat_block():
    from matched import METHODS, measurement_order
    for start in (0, 4):
        orders = [measurement_order(i) for i in range(start, start+4)]
        for position in range(4):
            assert set(order[position] for order in orders) == set(METHODS)


@pytest.mark.parametrize('scenario', ('stable', 'rank_shift', 'permuted', 'cold'))
def test_shared_trace_contract(scenario):
    trace = make_trace(2, 3, topk=256, prefix=1024, overlap=128, scenario=scenario)
    pages, old, new = initial_slots(trace)
    assert pages.shape == (2, 4)
    assert torch.equal(new.sort(-1).values, torch.arange(512).expand(2, -1))
    if scenario != 'cold':
        assert torch.equal(old, new)
    assert trace_digest(trace) == trace_digest({k: v.clone() for k, v in trace.items()})
    trace['steps'][0, 0, 0, 0] = 2000
    with pytest.raises(ValueError, match='valid length'):
        validate_trace(trace)


def test_partial_initial_readiness_maps_to_dense_original_slots():
    trace = make_trace(1, 1, topk=256, prefix=1024, overlap=128)
    trace['initial_ready'][:, ::3] = False
    _, old, new = initial_slots(trace)
    ready = trace['initial_ready'][0]
    assert torch.equal(old[0, ready], new[0, :int(ready.sum())])


def test_original_reference_evolves_from_the_same_initial_state():
    from resident_experiment import make_case, _put_state, reference, _state
    trace = make_trace(1, 3, prefix=8192)
    pages, _, _ = initial_slots(trace)
    old = make_case(1, 2, 4, 0)
    old['state_counts'].zero_()
    old['block_table'].copy_(pages)
    old['boundary'].copy_(trace['boundary'].repeat_interleave(2))
    row = int(old['request_states'][0])
    _put_state(old, row, dict(zip(trace['initial_tokens'][0].tolist(), range(4096), strict=True)))
    for step in trace['steps']:
        old['topk'].copy_(step.reshape(2, 1, 2048))
        updated, _ = reference(old)
        state = _state(updated, row)
        raw = step.flatten()
        mapped = updated['topk'].flatten()
        for token, slot in zip(raw.tolist(), mapped.tolist(), strict=True):
            assert slot == (state[token] if token < 8192 else token)
        old = updated


def test_payload_planes_request_identity_and_chunk_independence():
    tokens = torch.arange(2048)
    full = payload(tokens, 0, 64, 1)
    assert torch.equal(full, torch.cat([payload(tokens[:1024], 0, 64, 1), payload(tokens[1024:], 0, 64, 1)]))
    assert not torch.equal(full, payload(tokens, 1, 64, 1))
    assert not torch.equal(full, payload(tokens, 0, 64, 0))


def event(name, start, dur, tid=1):
    return dict(ph='X', cat='kernel', name=name, pid=1, tid=tid, ts=Decimal(start), dur=Decimal(dur))


def test_full_timing_includes_adapter_transfer_and_publication():
    trace = [event(name, str(i*10), '5') for i, name in enumerate(ORIGINAL_SYMBOLS)]
    trace += [event('prepared_transfer', '30', '20'), event('publication', '50', '3')]
    timing = device_timing(trace, 'original', 1)
    assert timing['original_three_us_per_step'] == 15
    assert timing['complete_sum_us_per_step'] == 38
    assert timing['complete_span_us_per_step'] == 53
    with pytest.raises(RuntimeError, match='expected'):
        device_timing(trace, 'original', 2)


def test_timing_preserves_large_timestamp_precision_and_overlap():
    trace = [event('resident_pack_sources_redesign', '18000000000000000.00', '2.25'),
             event('transfer', '18000000000000002.25', '2.25')]
    result = device_timing(trace, 'bounded_position', 1)
    assert result['complete_span_us_per_step'] == 4.5


def test_partial_host_allocation_failure_releases_owned_chunks(monkeypatch):
    from types import SimpleNamespace
    from matched_runtime import Sources
    events = []
    class Allocator:
        calls = 0
        def allocate(self, shape, dtype):
            self.calls += 1
            return SimpleNamespace(tensor=torch.empty(shape, dtype=dtype)) if self.calls == 1 else None
        def close(self):
            events.append('close')
    api = SimpleNamespace(
        MlaDsaDims=lambda *args: args,
        compute_chunk_partition=lambda *args: SimpleNamespace(chunk_offsets=[0, 512], chunk_sizes=[512, 512]),
        create_pin_memory_allocator=lambda *args: Allocator(),
        release_pin_memory_objects=lambda objects: events.append(len(objects)),
    )
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=lambda: events.append('sync')), raising=False)
    with pytest.raises(RuntimeError, match='allocation failed'):
        Sources(api, make_trace(1, 1, topk=256, prefix=1024, overlap=128), torch.device('cpu'), 512)
    assert events == ['sync', 1, 'close']


def test_matched_adapter_state_and_payload_lifecycle_on_cpu(monkeypatch):
    """Exercise orchestration, independently of CANN, with exact CPU operations."""
    from types import SimpleNamespace
    from matched_runtime import Sources, Original, Replacement
    from resident_experiment import Case, reference
    from candidates import Query, Snapshot, lookup
    from native import NativeCase
    trace = make_trace(1, 3, prefix=8192, tail=128)
    trace['steps'][:, 0, 0, 0] = -1
    trace['steps'][:, 0, 0, 1] = 8195
    source = Sources.__new__(Sources)
    source.trace, source.device, source.widths = trace, torch.device('cpu'), (16, 16)
    source.r, source.n, source.prefix, source.tail, source.chunk_size = 1, 4096, [8192], [128], 1024
    source.pages, source.seed_slots, source.slots = initial_slots(trace)
    buffers = {}
    def register(tensor):
        buffers[tensor.data_ptr()] = tensor
        return tensor.data_ptr()
    chunks = [torch.cat([payload(torch.arange(start, start+1024), 0, 16, plane).flatten()
                         for plane in (0, 1)]) for start in range(0, 8192, 1024)]
    source.pointers = [torch.tensor([register(chunk) for chunk in chunks])]
    source.live = [torch.cat([payload(torch.arange(8192, 8320), 0, 16, plane).flatten() for plane in (0, 1)])]
    source.live_ptrs = [torch.tensor([register(source.live[0])])]
    real_bank = Sources.bank
    def bank(self):
        value = real_bank(self)
        for row in value:
            register(row)
        return value
    monkeypatch.setattr(Sources, 'bank', bank)
    def transfer(planes, destinations, selected, pointers, chunk_size, total, interleaved, counts):
        assert not interleaved
        for row, count in enumerate(counts.tolist()):
            for i in range(count):
                token, slot = int(selected[row, i]), int(destinations[row, i])
                assert 0 <= token < total and 0 <= slot < 4096
                chunk, offset = divmod(token, chunk_size)
                size = min(chunk_size, total-chunk*chunk_size)
                raw = buffers[int(pointers[chunk])]
                for plane, view in enumerate(planes):
                    values = raw[plane*size*16:(plane+1)*size*16].view(size, 16)
                    view.flatten(0, 2)[slot].copy_(values[offset])
    source.api = SimpleNamespace(KV_FORMAT_MLA_LATENT=5,
        prepare_sparse_direct_destination_state=lambda planes, *args: planes,
        sparse_mla_dsa_batched_direct_kv_transfer_prepared=transfer)
    def original_run(self, *args):
        result, _ = reference(self)
        self.reset_from(result)
    monkeypatch.setattr(Case, 'run', original_run)
    def metadata_run(self, fused=False):
        assert not fused
        t = self.tensors
        query = Query(t[0], t[1], self.meta[..., 0].bool(), self.meta[..., 1], self.meta[..., 2], self.epochs[:, 0], self.universe)
        snapshot = Snapshot(t[2], t[3], t[4].bool(), self.epochs[:, 1], t[9])
        t[8].copy_(lookup(query, snapshot, self.variant, radius=self.radius).source)
    monkeypatch.setattr(NativeCase, 'run', metadata_run)
    def pack(t):
        plan, raw, old, new, boundary, selected, slots, counts = t
        counts.zero_()
        for r in range(plan.shape[0]):
            for pos, src in enumerate(plan[r].flatten().tolist()):
                if src == -2:
                    continue
                kind = 0 if src == -1 else 2 if src == -3 else 1
                tile = pos // 256
                i = int(counts[kind, r, tile, 0])
                token = int(raw[r].flatten()[pos])
                selected[kind, r, tile, i] = token if kind == 0 else token-int(boundary[r]) if kind == 2 else old[r, src]
                slots[kind, r, tile, i] = new[r, pos]
                counts[kind, r, tile, 0] += 1
    monkeypatch.setattr(torch.ops, 'resident_redesign', SimpleNamespace(pack_sources_=pack))
    for name in ('original', 'bounded_position', 'hash_snapshot', 'direct_directory'):
        runtime = Original(source, trace) if name == 'original' else Replacement(source, trace, name)
        counts = []
        for repeat in range(2):
            runtime.reset()
            observed = []
            for step in range(3):
                runtime.step(step)
                runtime.check(step)
                observed.append(runtime.last_misses)
            counts.append(observed)
        assert counts[0] == counts[1]
