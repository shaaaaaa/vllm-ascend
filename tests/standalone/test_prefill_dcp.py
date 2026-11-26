# SPDX-License-Identifier: Apache-2.0
"""Exercise the producer's real layout and stream dependencies on CPU."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tests.ut.compilation.sfa_test_support import ROOT, HostTL, Pointer, extract
from vllm_ascend.worker.prefill_dcp import PrefillDcpBatch, PrefillDcpWorkspace, workspace_blocks


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("hist", [(0, 0, 0), (128, 384, 1024), (1025, 0, 127)])
def test_gather_and_writer_follow_logical_rows(monkeypatch, world, hist):
    block = 128
    hist = np.array(hist)
    ends = hist + [3, 129, 2]
    names = ["last", "first", "middle"]
    ranks = []
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_: None)
    monkeypatch.setattr(torch, "npu", NS(current_stream=lambda: NS(record_event=lambda: object())), raising=False)
    for rank in range(world):
        ws = NS(
            world=world,
            rank=rank,
            block_size=block,
            planes=(torch.empty(96),),
            send=(torch.empty(96),),
            stream=object(),
            physical_capacity=65,
        )
        # Different allocator IDs per bank, request and local column.
        banks = [np.arange(15, 0, -1).reshape(3, 5) + bank * 16 for bank in range(4)]
        batch = PrefillDcpBatch.build(
            ws, names, hist, ends, [(b, [5] * 3) for b in banks], {}, 4, int(sum(ends - hist)) + 5
        )
        ranks.append(batch)
    for bank in range(4):
        send = []
        for rank, batch in enumerate(ranks):
            physical = np.full((65, block), -2)
            for i, h in enumerate(hist):
                for pos in range(h):
                    b = pos // block
                    if b % world == rank:
                        physical[banks[bank][i, b // world], pos % block] = i * 10000 + pos
            source = batch.sources[bank]
            if batch.count:
                ids = np.arange(source, source + batch.count) if isinstance(source, int) else source.numpy()
                send.append(physical[ids])
        dense = np.full((96, block), -1)
        if send:
            dense[1 : 1 + world * batch.count] = np.concatenate(send)
        fresh = np.concatenate([i * 10000 + np.arange(h, e) for i, (h, e) in enumerate(zip(hist, ends))])
        dense.ravel()[batch.slots[: len(fresh)].numpy()] = fresh
        assert (batch.slots[len(fresh) :] == -1).all() and (batch.table[-1] == 0).all()
        for i, (req, end) in enumerate(zip(names, ends)):
            for rank, result in enumerate(ranks):
                for pos in range(hist[i], end):
                    page = int(result.save_blocks[req][pos // block]) - 1
                    if page < world * result.count and page // result.count != rank:
                        assert pos // block % world != rank
                        continue
                    idx = page % result.count if page < world * result.count else page - (world - 1) * result.count
                    expected = banks[bank][i, pos // (world * block)] if pos // block % world == rank else 0
                    assert int(result.physical_blocks[bank, idx]) == expected
                np.testing.assert_array_equal(result.table[i, : len(result.save_blocks[req])], result.save_blocks[req])
                np.testing.assert_array_equal(dense[result.save_blocks[req]].ravel()[:end], i * 10000 + np.arange(end))


@pytest.mark.parametrize("history,partial", [(0, False), (1, False), (1, True)])
def test_waits_follow_last_consumer(monkeypatch, history, partial):
    ops = []
    stream = NS(wait_event=ops.append, record_event=lambda: "bank-ready-for-writes")
    monkeypatch.setattr(torch, "npu", NS(stream=lambda _: nullcontext(), current_stream=lambda: stream), raising=False)

    def exchange(*args):
        ops.extend(("gather", "gather"))
        return "history-ready"

    ws = NS(
        stream=stream,
        last_read="previous-mtp",
        save_events={0: "last-save"},
        unfenced=False,
        exchange=exchange,
    )
    batch = PrefillDcpBatch(ws, {}, {}, None, None, (7,), None, {"L": NS(bank=0)}, history, partial, "maps")
    writes, reads = batch.start("L", (), lambda _: ops.append("bank-ready"))
    if not history:
        assert ops == [] and writes == ("last-save", "previous-mtp") and reads is None
    else:
        assert ops == ["maps", "last-save", "previous-mtp", "bank-ready", "gather", "gather"]
        assert (writes, reads) == (
            (("history-ready",), None) if partial else (("bank-ready-for-writes",), "history-ready")
        )
    assert ws.save_events == {0: "last-save"}
    batch.wait_for_writes(writes)
    assert not ws.save_events


def test_budget_covers_every_admissible_batch():
    rng = np.random.default_rng(123)
    b, w, banks, max_len, reqs = 128, 8, 4, 2048, 8
    for budget in (60, 110, 150, 210, 1000):
        dense, send = workspace_blocks(max_len, reqs, b, w, banks, 1, budget)
        for _ in range(200):
            ends = rng.integers(1, max_len + 1, size=rng.integers(1, reqs + 1))
            hist = rng.integers(0, ends + 1)
            pages = (ends + b - 1) // b
            if banks * ((pages + w - 1) // w).sum() > budget - dense - send:
                continue
            h = (hist + b - 1) // b
            m = ((h + w - 1) // w).sum()
            assert m <= send and 1 + w * m + (pages - h).sum() <= dense


class KernelOps(HostTL):
    int64 = torch.int64
    maximum = staticmethod(lambda a, b: torch.maximum(torch.as_tensor(a), torch.as_tensor(b)))
    cdiv = staticmethod(lambda a, b: (a + b - 1) // b)
    load = staticmethod(lambda ptr, mask=True, other=0: HostTL.load(ptr, mask, other))


def invoke_kernel(name, tensors, args):
    tl = KernelOps()
    code = ROOT / "vllm_ascend/ops/triton/sfa_dcp.py"
    body = extract(code, name, {"tl": tl, "range": lambda a, b, c: map(torch.tensor, range(a, b, c))})
    tl.programs = 3
    for pid in range(tl.programs):
        tl.pid = pid
        body(*(Pointer(x) for x in tensors), *args)


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_fused_fresh_store_raw_packet(count, slot_dtype):
    block, world, rank, latent, rope = 8, 4, 2, 8, 2
    fresh = 1 + world * count
    pages = ([1 + rank * count, 2] if count else []) + [fresh, fresh + 1]
    slots = torch.tensor([b * block + 3 for b in pages] + [-1, 0], dtype=slot_dtype)
    # Include the BF16 indexer key trailing the two G0 planes.
    packet = torch.arange(len(slots) * (rope + latent + 4)).reshape(len(slots), -1)
    mapping = torch.tensor(([3, 2] if count else []) + [4, 0], dtype=torch.int32)
    out = [torch.full((16 * block, w), -3) for w in (latent, rope, latent, rope)]
    expected = [p.clone() for p in out]
    for i, p in enumerate(pages):
        owner = 3 if count and i == 0 else 4 if p == fresh else None
        for j, (start, width) in enumerate(((rope, latent), (0, rope))):
            expected[j][p * block + 3] = packet[i, start : start + width]
            if owner:
                expected[2 + j][owner * block + 3] = packet[i, start : start + width]
    invoke_kernel(
        "_store_prefill_rows",
        [packet, slots, mapping, *out],
        [len(slots), packet.stride(0), count, rank, world, block, latent, rope, 4, latent, rope],
    )
    for actual, reference in zip(out, expected):
        torch.testing.assert_close(actual, reference)


def test_fused_history_pack_zeros_missing_peer_stripe():
    ids = torch.tensor([3, 0, 1], dtype=torch.int32)
    block, rows = 8, 4
    src = [torch.arange(5 * block * w).reshape(5, block, 1, w) for w in (8, 2)]
    dst = [torch.empty_like(plane[:3]) for plane in src]
    invoke_kernel(
        "_pack_prefill_history", [*src, ids, *dst], [len(ids) * block // rows, block, 8, 2, rows, rows * 8, rows * 2]
    )
    for p, received in zip(src, dst):
        expected = p[ids.long()]
        expected[1].zero_()
        torch.testing.assert_close(received, expected)


@pytest.mark.parametrize("replacement", [[7], [7, 0], [7, 16]])
def test_every_rank_rejects_missing_column_after_row_reuse(replacement):
    # A short row can inherit valid-looking IDs from its previous occupant.
    path = ROOT / "vllm_ascend/worker/block_table.py"
    row = NS(
        num_blocks_per_row=np.zeros(1, np.int32), block_table=NS(np=np.zeros((1, 3), np.int32)), use_hybrid_blocks=False
    )
    append = extract(path, "append_row", {"np": np})
    row.append_row = lambda ids, i: append(row, ids, i)
    add = extract(path, "add_row", {})
    add(row, [7, 12], 0)
    add(row, replacement, 0)
    if len(replacement) == 1:
        assert row.block_table.np[0, 1] == 12
    for rank in range(4):
        ws = NS(
            rank=rank, world=4, block_size=128, planes=[torch.empty(12)], send=[torch.empty(2)], physical_capacity=16
        )
        with pytest.raises(ValueError, match="Missing Group-0 bank"):
            PrefillDcpBatch.build(
                ws, ["r"], np.array([512]), np.array([513]), [(row.block_table.np, row.num_blocks_per_row)], {}, 1, 1
            )


@pytest.mark.parametrize("fault", ["second_launch", "wait", "record"])
def test_failed_exchange_retains_queued_work_and_prevents_reuse(monkeypatch, fault):
    ws = object.__new__(PrefillDcpWorkspace)
    ws.world, ws.rank, ws.group = 2, 0, object()
    ws.planes = tuple(torch.empty(3, 1) for _ in range(2))
    ws.unfenced, ws.last_read, ws.save_events = False, None, {0: "d2h"}
    pending = []

    def step(stage):
        if stage == fault:
            raise RuntimeError(stage)

    def launch(*args, **kwargs):
        if pending:
            step("second_launch")
        work = NS(wait=lambda: step("wait"))
        pending.append(work)
        return work

    ws.stream = NS(wait_event=lambda event: None, record_event=lambda: step("record"))
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", launch)
    monkeypatch.setattr(torch, "npu", NS(stream=lambda _: nullcontext()), raising=False)
    cache = tuple(torch.empty(1, 1) for _ in range(2))
    batch = PrefillDcpBatch(ws, {}, {}, None, None, (0,), None, {"L": NS(bank=0)}, 1, True, "maps")
    with pytest.raises(RuntimeError, match=fault):
        batch.start("L", cache, lambda name: None)
    assert ws._failed_exchange[0] is cache and ws._failed_exchange[2] == pending
    assert len(pending) == (1 if fault == "second_launch" else 2)
    assert ws.save_events == {0: "d2h"}
    with pytest.raises(RuntimeError, match="unfenced"):
        batch.start("L", cache, lambda name: pytest.fail("re-entered collectives"))


@pytest.mark.parametrize("p_node,world", [(True, 8), (True, 1), (False, 8)])
def test_only_banked_dcp_uses_runner_workspace(p_node, world):
    path = ROOT / "vllm_ascend/attention"
    standard = extract(path / "utils.py", "ascend_chunked_prefill_workspace_size", {})
    effective = extract(
        path / "sfa_v1.py",
        "determine_chunked_prefill_workspace_size",
        {"envs": NS(VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=p_node), "ascend_chunked_prefill_workspace_size": standard},
    )
    cfg = NS(
        model_config=NS(max_model_len=1048576),
        cache_config=NS(block_size=128),
        scheduler_config=NS(max_num_seqs=8),
        parallel_config=NS(decode_context_parallel_size=world),
    )
    assert effective(cfg) == (0 if p_node and world == 8 else standard(cfg))


def test_workspace_rejects_invalid_bank_before_dma(monkeypatch):
    ws = NS(
        world=4,
        rank=0,
        block_size=128,
        planes=(torch.empty(32),),
        send=(torch.empty(8),),
        stream=object(),
        physical_capacity=8,
    )
    # A cold miss also writes the physical bank. Check fresh-only as well as historic IDs.
    for prefix in (0, 1):
        with pytest.raises(ValueError, match="allocation for r"):
            PrefillDcpBatch.build(ws, ["r"], np.array([prefix]), np.array([2]), [(np.array([[8]]), [1])], {}, 1, 2)
    failed = PrefillDcpBatch(NS(unfenced=True), {}, {}, None, None, (), None, {}, 0, False, None)
    with pytest.raises(RuntimeError, match="unfenced"):
        failed.start("unused", (), lambda _: pytest.fail("unfenced source was reused"))
