# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tests.ut.compilation.sfa_test_support import ROOT, HostTL, Pointer, extract
from vllm_ascend.worker.dcp_indexer import OwnerRows, bind_block_tables, owner_rows


@pytest.fixture
def restore_launches(monkeypatch):
    path = ROOT / "vllm_ascend/ops/triton/sfa_dcp.py"
    calls = []

    class Kernel(HostTL):
        def __getitem__(self, grid):
            def launch(*args):
                calls.append(grid)
                self.programs = grid[0]
                for pid in range(grid[0]):
                    self.pid = pid
                    body(*(Pointer(x) if isinstance(x, torch.Tensor) else x for x in args))
            return launch

    kernel = Kernel()
    body = extract(path, "_restore_owner_topk", {"tl": kernel})
    restore = extract(path, "restore_owner_topk", {
        "_restore_owner_topk": kernel, "get_vectorcore_num": lambda: 3, "init_device_properties_triton": lambda: None,
        "triton": NS(cdiv=lambda a, b: (a + b - 1) // b),
    })
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.sfa_dcp", NS(restore_owner_topk=restore))
    return calls


@pytest.mark.parametrize("owners,widths", [([2, 0, 2, 2], [2, 1, 2, 2]), ([3], [17]), ([], [])])
def test_owner_exchange_and_causality(owners, widths, monkeypatch, restore_launches):
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


def test_different_worker_orders_restore_the_same_request_values(monkeypatch, restore_launches):
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


@pytest.mark.parametrize("extent,width", [(3, 513), (5, 513), (8, 2048)])
@pytest.mark.parametrize("causal,reject,private", [(False, False, True), (True, False, True),
                                                  (True, True, True), (False, True, False)])
def test_one_owner_launch_publishes_raw_and_prepares_private_topk(extent, width, causal, reject, private,
                                                                monkeypatch, restore_launches):
    inverse = torch.tensor([4, 1, -1, 3, 0])
    rows = OwnerRows(None, None, None, None, None, inverse)
    packets = (torch.arange(6 * width, dtype=torch.int32).view(6, 1, width) % 541) - 4
    gathers = []

    def gather(out, packet, group):
        gathers.append(packet.shape)
        out.copy_(packets)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    pos = torch.tensor([50, -1, 120, 375, 515, -1, 70, 500], dtype=torch.int32)[:extent]
    active = extent - 1
    storage = torch.full((extent + 1, 2 * (width + 5)), -555, dtype=torch.int32)
    published = storage[:, ::2]
    out = torch.empty(extent, 1, width, dtype=torch.int32) if private else None
    result = rows.restore(packets[:3], NS(world_size=2, device_group=None), extent, output=out,
                          positions=pos if reject else pos[:active] if causal else None, actual_rows=active,
                          published=published, reject=reject)
    raw = torch.full_like(result, -1)
    for i, src in enumerate(inverse[:extent].tolist()):
        if src >= 0 and (not reject or pos[i] >= 0):
            raw[i] = packets[src]
    expected = raw.clone()
    if causal:
        expected[(raw < 0) | (raw > pos[:, None, None])] = -1
        expected[active:] = -1
    assert torch.equal(result, expected)
    assert out is None or result is out
    assert gathers == [packets[:3].shape] and len(restore_launches) == 1
    result.fill_(-998)  # Later resident remapping must leave the publication untouched.
    assert torch.equal(published[:extent, :width], raw[:, 0])
    assert published[extent:].eq(-555).all() and published[:, width:].eq(-555).all()
    assert storage[:, 1::2].eq(-555).all()


@pytest.mark.parametrize("native,layout", [(False, "staged"), (True, "uniform"), (True, "mixed"), (True, "dense")])
def test_sharded_indexer_chooses_private_planner_only_for_global_rows(native, layout, monkeypatch, restore_launches):
    n, width = 4, 8
    packet = torch.arange(n * width, dtype=torch.int32).reshape(n, 1, width)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", lambda out, local, group: out.copy_(local))
    def projection(t):
        return t, None

    impl = NS(has_indexer=True, wq_b=projection, weights_proj=projection, n_head=1, head_dim=1,
              qk_rope_head_dim=1, is_rope_neox_style=False, use_sparse_c8_indexer=False,
              use_torch_npu_lightning_indexer=True, index_topk=width, index_cache_enabled=True,
              topk_indices_buffer=torch.empty_like(packet[:, 0]))
    sfa = ROOT / "vllm_ascend/attention/sfa_v1.py"
    select = extract(sfa, "indexer_select_post_process", {
        "HAS_TRITON": True, "rope_forward_triton_siso": lambda q, *a, **kw: q,
        "torch_npu": NS(npu_lightning_indexer=lambda **kw: (packet, None)),
    })
    pos = torch.tensor([30, -1, 17, 40])
    seq = torch.arange(n)
    owner = OwnerRows(seq, seq >= 0, seq, seq >= 0, seq, torch.tensor([3, 0, 1, 2]))
    dcp = NS(topk=torch.empty_like(packet), positions=pos, mask_indexer_writes=True,
             row_requests=None if layout == "dense" else seq, compact_rows=seq if layout == "mixed" else None)
    metadata = NS(sfa_dcp=dcp, num_actual_tokens=3) if native else None
    result = select(impl, seq, seq, (None, None, None), metadata, None, None, seq, seq, seq, owner, n,
                    NS(world_size=1, device_group=None), owner_output=None if native else dcp.topk,
                    owner_positions=None if native else pos)
    raw = packet[[3, 0, 1, 2]]
    if native:
        raw[1].fill_(-1)  # Draft rejection affects both outputs.
    assert torch.equal(raw[:, 0], impl.topk_indices_buffer)
    if layout in ("staged", "uniform"):
        expected = torch.where(raw <= pos[:, None, None], raw, -1)
        if native:
            expected[3].fill_(-1)
        assert result is dcp.topk and torch.equal(result, expected)
    else:
        assert result is not dcp.topk and torch.equal(result, raw)
    assert len(restore_launches) == 1

    if native and layout == "uniform":
        def unexpected_prepare(*args, **kwargs):
            pytest.fail("the native planner launched a second global mask")

        monkeypatch.setattr(sys.modules["vllm_ascend.ops.triton.sfa_dcp"], "prepare_sfa_dcp_topk",
                            unexpected_prepare, raising=False)
        dcp.layout, dcp.local_seq_lens, dcp.split_boundary = NS(), seq, seq
        metadata.block_table, metadata.resident_state_indices, metadata.resident_state_generations = seq, seq, seq
        impl.decode_threshold = 1
        impl._prepare_sorted_resident_sparse_cache = lambda t, *args, **kwargs: (t, None, None, None)
        extract(sfa, "_plan_native_dcp_topk", {})(impl, result, metadata, dcp)
