# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_ascend.worker.dcp_indexer import OwnerRows, bind_block_tables, owner_rows


@pytest.mark.parametrize("owners,widths", [([2, 0, 2, 2], [2, 1, 2, 2]), ([3], [17]), ([], [])])
def test_owner_exchange_and_causality(owners, widths, monkeypatch):
    starts = np.cumsum([0, *widths])
    live = int(starts[-1])
    capacity = max(1, max(sum(n for r, n in zip(owners, widths) if r == rank) for rank in range(4)))
    reqs = max(1, *(owners.count(r) for r in range(4)))
    n = live + 3
    table = torch.arange((len(owners) + 1) * 5, dtype=torch.int32).view(-1, 5)
    lengths = torch.arange(len(owners) + 1) * 100 + 300
    data = torch.arange(n).view(n, 1, 1)
    packets, plans = [], []
    for rank in range(4):
        plan = OwnerRows(*(torch.from_numpy(a) for a in owner_rows(owners, starts, rank, 4, reqs, capacity, n)))
        plans.append(plan)
        packet = plan.pack(data)
        assert torch.all(packet[~plan.valid] == 0)
        packets.append(packet)
        local_table, query_ends, key_lengths = plan.score_inputs(table, lengths)
        assert query_ends[-1] == capacity and key_lengths[-1] == 0
        previous = 0
        for j, row in enumerate(i for i, r in enumerate(owners) if r == rank):
            end = int(query_ends[j])
            assert end - previous == widths[row] and torch.equal(local_table[j], table[row])
            assert key_lengths[j] == lengths[row]
            assert torch.equal(packet[previous:end], data[starts[row] : starts[row + 1]])
            previous = end
    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", lambda out, inp, group: out.copy_(torch.cat(packets))
    )
    for plan, packet in zip(plans, packets):
        restored = plan.restore(packet, NS(world_size=4, device_group=None), n)
        assert torch.equal(restored[:live], data[:live])
        assert torch.all(restored[live:] == -1)


def test_local_block_binding_survives_row_movement_and_replaces_resumed_owner():
    owners = {"old": (0, 1)}
    new = NS(req_id="x", block_ids=([99], [8, 9]))
    cached = NS(req_ids=["old"], new_block_ids=[([88], [5])])
    output = NS(
        finished_req_ids=set(),
        preempted_req_ids={"old"},
        scheduled_new_reqs=[new],
        scheduled_cached_reqs=cached,
        dcp_indexer_placements={"x": (2, 4, ((2,), (4,), (6,), (8,))), "old": (1, 9, ((3,), (5,), (7,), (9,)))},
    )
    bind_block_tables(output, owners, 2)
    assert new.block_ids == ([6], [8, 9])
    assert cached.new_block_ids == [([7], [0])]
    assert owners == {"old": (1, 9), "x": (2, 4)}


def test_packed_table_rebinds_in_place_and_keeps_c8_addressing():
    table = torch.arange(12).reshape(3, 4)
    arena = []
    addresses = None
    for owners in ([1, 1, 0], [0, 1, 0], [0, 0, 0]):
        arrays = owner_rows(owners, [0, 1, 3, 4], 1, 2, 3, 6, 6)
        plan = OwnerRows(*(torch.from_numpy(a) for a in arrays))
        bound = plan.bind_tables(arena, table, table + 50, 3)
        pointers = tuple(t.data_ptr() for t in bound.tables)
        assert addresses is None or pointers == addresses
        addresses = pointers
        for c8 in (False, True):
            result, ends, lengths = bound.score_inputs(table, torch.tensor([9, 5, 7]), c8)
            indices = [i for i, owner in enumerate(owners) if owner == 1]
            assert torch.equal(result[: len(indices)], table[indices] + int(c8) * 50)
            assert not result[len(indices) :].any() and not lengths[len(indices) :].any()
            assert ends[-1] == 6


def test_different_worker_orders_restore_the_same_request_values(monkeypatch):
    batches = (["b", "a", "c"], ["c", "b", "a"])
    ranks = {"a": 1, "b": 1, "c": 0}
    packets, plans = [], []
    for rank, ids in enumerate(batches):
        rows = owner_rows([ranks[k] for k in ids], [0, 1, 2, 3], rank, 2, 2, 2, 3, ids)
        plan = OwnerRows(*(torch.from_numpy(r) for r in rows))
        plans.append(plan)
        packets.append(plan.pack(torch.tensor([ord(k) for k in ids]).view(-1, 1, 1)))
    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", lambda out, inp, group: out.copy_(torch.cat(packets))
    )
    for plan, packet, ids in zip(plans, packets, batches):
        restored = plan.restore(packet, NS(world_size=2, device_group=None), 3)
        assert restored.flatten().tolist() == [ord(k) for k in ids]
