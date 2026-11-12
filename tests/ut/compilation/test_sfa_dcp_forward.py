# SPDX-License-Identifier: Apache-2.0
"""Exercise native DCP metadata and forward ordering with host device shims."""

import sys
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sfa_test_support import ROOT, HostTL, Pointer, definitions, extract, load_module
from test_sfa_dcp_merge import DcpTL

RUNNER = ROOT / "vllm_ascend/worker/model_runner_v1.py"
SFA = ROOT / "vllm_ascend/attention/sfa_v1.py"
OPS = ROOT / "vllm_ascend/ops/triton/sfa_dcp.py"


@pytest.fixture
def gather_queries():
    tl = HostTL()
    kernel = extract(OPS, "_unpack_queries", {"tl": tl})

    class Launch:
        def __getitem__(self, grid):
            def run(packed, query, rope, *args):
                tl.programs = grid[0]
                for pid in range(tl.programs):
                    tl.pid = pid
                    kernel(Pointer(packed), Pointer(query), Pointer(rope), *args)
            return run

    return extract(OPS, "_gather_sfa_dcp_queries", {
        "torch": torch, "_unpack_queries": Launch(),
        "init_device_properties_triton": lambda: None, "get_vectorcore_num": lambda: 7,
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
    })


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,heads", [(1, 8), (2, 8), (3, 1), (8, 8), (32, 1), (3, 16)])
@pytest.mark.parametrize("strided", [False, True])
def test_gather_query_unpack_preserves_rank_row_head_and_bits(gather_queries, world, dtype, rows, heads, strided):
    packets, inputs = [], []
    for rank in range(world):
        bits = ((torch.arange(rows * heads * 1152) + rank * 1103) % 65536 - 32768).to(torch.int16)
        values = bits.view(dtype).reshape(rows, heads, 1152)
        query, rope = values[..., :1024:2], values[..., 1024::2]
        if not strided:
            query, rope = query.contiguous(), rope.contiguous()
        inputs.append((query, rope))
        packets.append(torch.cat((query, rope), -1))
    snapshots = [value.view(torch.int16).clone() for value in inputs[0]]
    calls = []
    raw = torch.cat(packets, dim=0)

    def collective(output, packet, group):
        calls.append((packet.shape, packet.dtype, packet.numel()))
        assert torch.equal(packet.view(torch.int16), packets[0].view(torch.int16))
        output.copy_(raw)

    # Use the actual rank-major communicator layout, not a head-major test double.
    all_gather = extract(
        ROOT.parent / "vllm/vllm/distributed/device_communicators/base_device_communicator.py",
        "all_gather", {"torch": torch, "dist": SimpleNamespace(all_gather_into_tensor=collective)},
    )
    group = SimpleNamespace(world_size=world, device_group=object())
    group.all_gather = lambda packet, dim: all_gather(group, packet, dim)
    actual = gather_queries(*inputs[0], group)
    expected = torch.cat(packets, dim=1).split((512, 64), dim=-1)
    for result, reference in zip(actual, expected):
        assert result.is_contiguous() and result.dtype == dtype
        assert torch.equal(result.view(torch.int16), reference.contiguous().view(torch.int16))
    for value, snapshot in zip(inputs[0], snapshots):
        assert torch.equal(value.view(torch.int16), snapshot)
    assert calls == [(packets[0].shape, dtype, packets[0].numel())]


class TopkTL(DcpTL):
    cumsum = staticmethod(torch.cumsum)
    where = staticmethod(lambda mask, yes, no: torch.where(torch.as_tensor(mask), yes, no))


@pytest.fixture
def prepare_topk():
    tl = TopkTL()
    kernel = extract(OPS, "_prepare_topk", {"tl": tl})

    def prepare(raw, positions, output, counts, actual_rows, world, rank, block, *, local, row_requests=None):
        tl.programs = 2
        for pid in range(tl.programs):
            tl.pid = pid
            kernel(
                Pointer(raw), Pointer(positions), Pointer(output), Pointer(counts),
                Pointer(row_requests) if row_requests is not None else None,
                raw.shape[0], raw.shape[-1], actual_rows, world, rank, block, local,
            )

    return prepare


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_counted_topk_masks_future_and_padding_before_local_attention(prepare_topk, rank):
    positions = torch.tensor([128, 1048575, 999])
    topk = torch.full((3, 1, 2048), -1, dtype=torch.int32)
    topk[:, 0, :8] = torch.tensor([129, 128, 127, 0, -1, 512, 1048575, 1048576])
    counts = torch.full((3, 16), -9, dtype=torch.int32)
    output = torch.empty_like(topk)
    prepare_topk(topk, positions, output, counts, 2, 4, rank, 128, local=True)
    for row in range(3):
        selected = [
            t // 512 * 128 + t % 128 for t in topk[row, 0].tolist()
            if row < 2 and 0 <= t <= positions[row] and t // 128 % 4 == rank
        ]
        assert output[row, 0].tolist() == selected + [-1] * (2048 - len(selected))
        assert counts[row, 0] == len(selected)
    assert torch.all(counts[:, 1:] == -9)
    assert topk[0, 0, 0] == 129


@pytest.mark.parametrize("bounded", [False, True])
def test_dcp_graph_metadata_reuses_fixed_storage_across_idle_and_admission(monkeypatch, bounded):
    resident = load_module(
        ROOT / "vllm_ascend/distributed/kv_transfer/sparse_offload/resident_sorted_cache.py",
        "dcp_graph_resident_test", monkeypatch,
    )
    ns = dict(dataclass=dataclass, np=np, torch=torch, ResidentDcpLayout=resident.ResidentDcpLayout,
              StagedSFAQueryProfile=SimpleNamespace(DECODE_BOUNDED="bounded"))
    definitions(ROOT / "vllm_ascend/attention/utils.py", {"DcpSFABatch", "dcp_local_seq_lens"}, ns)
    build = extract(RUNNER, "_prepare_sfa_dcp_batch", ns)
    clear = extract(RUNNER, "_clear_sfa_idle_metadata", ns)
    signature = extract(SFA, "full_graph_metadata_inputs", ns)

    def buffer(values, dtype=torch.int32):
        cpu = torch.tensor(values, dtype=dtype)
        gpu = torch.empty_like(cpu)
        return SimpleNamespace(np=cpu.numpy(), gpu=gpu, copy_to_gpu=lambda n: gpu[:n].copy_(cpu[:n]))

    def signature_fields(dcp):
        class Metadata:
            sfa_dcp = dcp

            def __getattr__(self, name):
                return None

        return signature(Metadata())["dcp"]

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=[]),
        seq_lens=buffer([16384, 16384, 16384]), query_start_loc=buffer([0, 2, 4, 6]),
        positions=buffer([0] * 6, torch.int64),
        block_size=128, dcp_size=4, dcp_rank=1, decode_threshold=2,
        _dsa_dcp_origins=buffer([4096] * 3), _dsa_dcp_seq_lens=buffer([0] * 3),
        _dsa_dcp_rows=buffer([0] * 6), _dsa_dcp_boundaries=buffer([0] * 6),
        _dsa_dcp_compact_rows=buffer([0] * 6, torch.int64),
        _dsa_dcp_topk=torch.empty(6, 1, 8, dtype=torch.int32),
        _dsa_dcp_counts=torch.empty(6, 16, dtype=torch.int32),
        _dsa_dcp_planner_counts=torch.empty(6, 16, dtype=torch.int32),
        _dsa_dcp_zero_counts=torch.zeros(3, dtype=torch.int32),
        _dsa_dcp_empty_slots=torch.empty(3, 0, dtype=torch.int64),
    )
    key = SimpleNamespace(request_capacity=3, query_profile="bounded" if bounded else "fixed")
    capture, _ = build(runner, 3, 6, 6, None, graph_key=key)
    assert capture.row_requests.tolist() == [0, 0, 1, 1, 2, 2]
    captured_fields = signature_fields(capture)
    clear(SimpleNamespace(sfa_dcp=capture))
    assert capture.local_seq_lens.tolist() == [0, 0, 0] and capture.row_requests.tolist() == [-1] * 6

    runner.input_batch.req_ids = ["short", "compact"]
    runner._dsa_dcp_origins.np[:] = [0, 4096, 999]
    runner.seq_lens.np[:] = [129, 8194, 999]
    widths = [1, 2] if bounded else [2, 2]
    runner.query_start_loc.np[:3] = [0, widths[0], sum(widths)]
    output = SimpleNamespace(dsa_dcp_compact_req_ids={"compact"})
    active, frontier = build(runner, 2, sum(widths), 6, output, graph_key=key, graph_frontiers=(0, 8191))
    assert frontier == (0, 8191)
    assert active.layout.live_origins.tolist() == [0, 4096, 0]
    assert active.local_seq_lens.tolist() == [1, 6144, 0]
    assert active.row_requests.tolist() == [0] * widths[0] + [1, 1] + [-1] * (6 - sum(widths))
    assert active.split_boundary.tolist() == [0] * widths[0] + [8191, 8191] + [0] * (6 - sum(widths))
    assert (active.layout.selected_counts.data_ptr() != active.counts.data_ptr()) == bounded
    for before, after in zip(captured_fields, signature_fields(active), strict=True):
        if isinstance(before, torch.Tensor):
            assert (before.data_ptr(), before.shape, before.stride()) == (after.data_ptr(), after.shape, after.stride())
        else:
            assert before == after
    with pytest.raises(RuntimeError, match="source coverage"):
        build(runner, 2, sum(widths), 6, output, graph_key=key, graph_frontiers=(8191, 0))


@pytest.mark.parametrize("dp_padding", [0, 2])
def test_native_dcp_mixed_rows_reuse_one_shared_plan_and_copy_each_layer(monkeypatch, prepare_topk, dp_padding):
    geometry = load_module(
        ROOT / "vllm_ascend/attention/sfa_graph_layout.py", "dcp_graph_layout_test", monkeypatch
    )
    resident = load_module(
        ROOT / "vllm_ascend/distributed/kv_transfer/sparse_offload/resident_sorted_cache.py",
        "dcp_resident_test", monkeypatch,
    )
    dcp_metadata = {"dataclass": dataclass, "np": np, "torch": torch}
    definitions(
        ROOT / "vllm_ascend/attention/utils.py", {"DcpSFABatch", "dcp_local_seq_lens"}, dcp_metadata
    )

    def buffer(values, dtype=torch.int32):
        cpu = torch.tensor(values, dtype=dtype)
        gpu = torch.empty_like(cpu)
        return SimpleNamespace(np=cpu.numpy(), gpu=gpu, copy_to_gpu=lambda n: gpu[:n].copy_(cpu[:n]))

    eligible = object()
    connector_meta = object()

    def prove(metadata, requests):
        assert metadata is connector_meta and requests == ["compact"]
        return eligible, (256,), ()

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["compact", "dense"]),
        seq_lens=buffer([257, 131]), query_start_loc=buffer([0, 1, 4]),
        positions=SimpleNamespace(gpu=torch.tensor([256, 128, 129, 130])),
        block_size=128, dcp_size=4, dcp_rank=1, decode_threshold=2,
        _dsa_dcp_origins=buffer([4096, 0]), _dsa_dcp_seq_lens=buffer([0, 0]),
        _dsa_dcp_rows=buffer([0] * 4), _dsa_dcp_boundaries=buffer([0] * 4),
        _dsa_dcp_compact_rows=buffer([0] * 4, torch.int64),
        _dsa_dcp_topk=torch.empty((4, 1, 8), dtype=torch.int32),
        _dsa_dcp_counts=torch.empty((4, 16), dtype=torch.int32),
        _dsa_dcp_planner_counts=torch.empty((4, 16), dtype=torch.int32),
        _dsa_dcp_zero_counts=torch.zeros(2, dtype=torch.int32),
        _dsa_dcp_empty_slots=torch.empty((2, 0), dtype=torch.int64),
    )
    build = extract(RUNNER, "_prepare_sfa_dcp_batch", {
        "np": np, **dcp_metadata, "ResidentDcpLayout": resident.ResidentDcpLayout,
        "unwrap_staged_sfa_connector_metadata": lambda metadata: metadata,
        "staged_sfa_metadata_sparse_route": prove,
        "StagedSFARouteReason": SimpleNamespace(ELIGIBLE=eligible),
    })
    dcp, frontiers = build(runner, 2, 4, 4, SimpleNamespace(
        dsa_dcp_compact_req_ids={"compact"}, kv_connector_metadata=connector_meta,
    ))
    assert frontiers == (256, 0)
    assert dcp.local_seq_lens.tolist() == [4224, 3]
    assert dcp.compact_rows.tolist() == [0]
    assert dcp.row_requests.tolist() == [0, -1, -1, -1]

    if dp_padding:
        dcp = replace(
            dcp, positions=torch.nn.functional.pad(dcp.positions, (0, dp_padding)),
            topk=torch.empty((4 + dp_padding, 1, 8), dtype=torch.int32),
            counts=torch.empty((4 + dp_padding, 16), dtype=torch.int32),
        )
    raw = torch.full((4 + dp_padding, 1, 8), -1, dtype=torch.int32)
    raw[:, 0, :5] = torch.tensor([128, 255, 257, 129, 127])
    snapshot = raw.clone()
    block_table = torch.tensor([[19, 25], [27, 0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=4, cum_query_lens=torch.tensor([1, 4]), block_table=block_table,
        resident_state_indices=torch.tensor([8, -1]), resident_state_generations=torch.tensor([4, -1]),
        req_ids=runner.input_batch.req_ids,
    )
    trace = []
    misses = torch.tensor([[128, 255, 129], [-1, -1, -1]], dtype=torch.int32)
    slots = torch.tensor([[2432, 2433, 2434], [-1, -1, -1]])
    counts = torch.tensor([3, 0], dtype=torch.int32)

    def planner(topk, split, rows, table, state, generations, *, mtp, dcp):
        assert mtp == 2 and table.equal(block_table)
        assert state.tolist() == [8, -1] and generations.tolist() == [4, -1]
        assert rows.tolist() == [0, -1, -1, -1]
        assert split.tolist() == [256, 0, 0, 0]
        assert topk[0, 0].tolist() == [128, 255, -1, 129, 127, -1, -1, -1]
        topk[0, 0, :] = torch.tensor([0, 1, 2, -1, -1, -1, -1, -1])
        dcp.selected_counts[:, 0] = counts.repeat_interleave(2)
        trace.append("plan")
        return topk, misses, counts, slots

    def wait(layer, selected_tokens, request_ids, target_slot_mapping, selected_token_counts):
        assert selected_tokens is misses and target_slot_mapping is slots
        assert selected_token_counts is counts and request_ids == ["compact", "dense"]
        trace.append(f"load:{layer}")

    def attend(query, rope, kv, topk, selected, table, ends, local_seq_lens, scale, group):
        assert selected[:, 0].tolist() == [3, 1, 2, 2] + [0] * dp_padding
        assert topk[:, 0, :3].tolist() == (
            [[0, 1, 2], [0, -1, -1], [0, 1, -1], [0, 1, -1]] + [[-1, -1, -1]] * dp_padding
        )
        assert local_seq_lens is dcp.local_seq_lens and table is block_table
        trace.append("attention")
        return query

    def prepare(*args, **kwargs):
        trace.append("prepare")
        return prepare_topk(*args, **kwargs)

    triton_ops = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    triton_ops.prepare_sfa_dcp_topk, triton_ops.execute_sfa_dcp = prepare, attend
    monkeypatch.setitem(sys.modules, triton_ops.__name__, triton_ops)
    namespace = {
        "pack_decode_lanes": geometry.pack_decode_lanes,
        "wait_for_kv_layer_from_connector": wait,
        "get_dcp_group": lambda: object(),
    }
    plan = extract(SFA, "_plan_native_dcp_topk", namespace)
    forward = extract(SFA, "_execute_native_dcp_attention", namespace)
    impl = SimpleNamespace(
        shared_resident_plan=object(), skip_topk=False, block_size=128,
        decode_threshold=2, scale=0.25, _prepare_sorted_resident_sparse_cache=planner,
    )
    impl._plan_native_dcp_topk = lambda *args: plan(impl, *args)
    query = torch.empty((4, 2, 512))
    forward(impl, "producer", query, None, (), raw, metadata, dcp)
    impl.skip_topk = True
    forward(impl, "consumer", query, None, (), raw, metadata, dcp)
    assert raw.equal(snapshot)
    assert trace == ["prepare", "plan", "load:producer", "attention", "load:consumer", "attention"]


@pytest.mark.parametrize("staged", [False, True])
def test_native_dcp_attention_gathers_heads_then_merges_local_outputs(monkeypatch, staged, gather_queries):
    query, rope = torch.ones((3, 2, 512), dtype=torch.bfloat16), torch.ones((3, 2, 64), dtype=torch.bfloat16)
    kv = (torch.empty(1), torch.empty(2))
    counts, topk = torch.empty((3, 16)), torch.empty((3, 1, 8))
    table, query_ends, local_lens = torch.zeros((3, 2)), torch.tensor([2, 3, 3]), torch.tensor([0, 19])
    sequence = []

    def gather(packet, *, dim):
        assert packet.shape == (3, 2, 576) and dim == 0
        sequence.append("gather")
        return torch.cat((packet, packet * 2), dim=dim)

    def local(nope, pe, k, key_rope, indices, selected, table, ends, lengths, scale):
        assert nope.is_contiguous() and pe.is_contiguous()
        assert nope[0, :, 0].tolist() == pe[0, :, 0].tolist() == [1, 1, 2, 2]
        assert k is kv[0] and key_rope is kv[1] and selected is counts and indices is topk
        assert table.shape == (2, 2) and ends.tolist() == [2, 3] and lengths is local_lens
        sequence.append("local")
        return nope, pe, counts

    def merge(partial, maximum, total, group):
        assert group == "dcp" and total is counts
        sequence.append("merge")
        return partial[:, :2]

    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_attention_dcp", local, raising=False)
    run = extract(OPS, "execute_sfa_dcp", {
        "torch": torch, "merge_sfa_dcp": merge, "_gather_sfa_dcp_queries": gather_queries,
    })
    group = SimpleNamespace(all_gather=gather, device_group="dcp", world_size=2)
    if staged:
        ops = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
        ops.execute_sfa_dcp = run
        monkeypatch.setitem(sys.modules, ops.__name__, ops)
        post = extract(SFA, "_cross_layer_post_compute", {
            "get_dcp_group": lambda: group, "MAX_O_PROJ_PREFETCH_SIZE": 0,
            "get_weight_prefetch_method": lambda: SimpleNamespace(
                maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kwargs: None,
            ),
        })
        projection = Mock(side_effect=lambda values: (values,))
        owner = SimpleNamespace(scale=0.25, _v_up_proj=lambda values: values, dsa_cp_decoder=False, o_proj=projection)
        edge = SimpleNamespace(launch=lambda *args: sequence.append("prefetch"))
        output = post(
            owner, query, rope, topk, *kv, query_ends, None, table, torch.empty_like(query),
            trace_label="cross_layer", prefetch=(edge, None, None, None, None),
            dcp=SimpleNamespace(counts=counts, local_seq_lens=local_lens),
        )
        assert sequence == ["gather", "local", "prefetch", "merge"]
    else:
        output = run(query, rope, kv, topk, counts, table, query_ends, local_lens, 0.25, group)
        assert sequence == ["gather", "local", "merge"]
    assert output.equal(query)


def test_dcp_draft_dp_padding_is_excluded_from_the_fixed_width_planner(monkeypatch, prepare_topk):
    raw = torch.tensor([[[3, 2, 1, -1]], [[4, 3, 1, -1]], [[0, 1, 2, 3]]], dtype=torch.int32)
    batch = SimpleNamespace(
        layout=SimpleNamespace(world_size=4, rank=0), row_requests=torch.tensor([0, 0]),
        split_boundary=torch.tensor([4, 4]), compact_rows=None, local_seq_lens=torch.tensor([4096 + 5]),
        positions=torch.tensor([3, -1, 0]), topk=torch.empty_like(raw),
        counts=torch.empty((3, 16), dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=2, block_table=torch.tensor([[2]]), cum_query_lens=torch.tensor([2]),
        resident_state_indices=torch.tensor([0]), resident_state_generations=torch.tensor([1]),
    )
    ops = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    ops.prepare_sfa_dcp_topk = prepare_topk
    monkeypatch.setitem(sys.modules, ops.__name__, ops)

    def planner(topk, boundary, rows, *args, mtp, dcp):
        assert mtp == 2 and topk.shape[0] == rows.numel() == boundary.numel() == 2
        assert topk[:, 0].tolist() == [[3, 2, 1, -1], [-1, -1, -1, -1]]
        return topk, None, None, None

    plan = extract(SFA, "_plan_native_dcp_topk", {})
    plan(SimpleNamespace(block_size=128, decode_threshold=2, _prepare_sorted_resident_sparse_cache=planner),
         raw, metadata, batch)
    assert batch.topk[2].eq(-1).all()


def test_dense_dcp_does_not_advance_resident_state_and_returns_empty_transfer(monkeypatch, prepare_topk):
    raw = torch.tensor([[[127, 128, 129, -1]], [[0, 1, 2, -1]]], dtype=torch.int32)
    empty_payload = (
        raw[:, 0, :0], torch.zeros(2, dtype=torch.int32),
        torch.empty((2, 0), dtype=torch.int64),
    )
    batch = SimpleNamespace(
        layout=SimpleNamespace(world_size=4, rank=1),
        row_requests=None, split_boundary=None, compact_rows=None,
        local_seq_lens=torch.tensor([1, 0]), positions=torch.tensor([128, 2]),
        topk=torch.empty_like(raw), counts=torch.empty((2, 16), dtype=torch.int32),
        empty_payload=empty_payload,
    )
    triton_ops = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    triton_ops.prepare_sfa_dcp_topk = prepare_topk
    monkeypatch.setitem(sys.modules, triton_ops.__name__, triton_ops)
    plan = extract(SFA, "_plan_native_dcp_topk", {})
    assert plan(SimpleNamespace(block_size=128), raw, SimpleNamespace(num_actual_tokens=2), batch) is empty_payload
    assert batch.counts[:, 0].tolist() == [1, 0]
    assert batch.topk[:, 0].tolist() == [[0, -1, -1, -1], [-1, -1, -1, -1]]
