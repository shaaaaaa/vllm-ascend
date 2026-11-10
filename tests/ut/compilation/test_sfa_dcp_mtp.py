# SPDX-License-Identifier: Apache-2.0
"""Exercise eager DCP draft staging with the production proposer and kernel."""

import ast
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS

import pytest
import torch
from sfa_test_support import ROOT, HostTL, Pointer, definitions, extract, load_module

PROPOSER = ROOT / "vllm_ascend/spec_decode/eagle_proposer.py"
DRAFT_KERNEL = ROOT / "vllm_ascend/ops/triton/spec_decode/utils.py"
SFA = ROOT / "vllm_ascend/attention/sfa_v1.py"


class DraftTL(HostTL):
    minimum = staticmethod(lambda a, b: torch.minimum(torch.as_tensor(a), torch.as_tensor(b)))
    maximum = staticmethod(lambda a, b: torch.maximum(torch.as_tensor(a), torch.as_tensor(b)))
    load = staticmethod(lambda ptr, mask=True, other=0: HostTL.load(ptr, mask, other))
    store = staticmethod(lambda ptr, value, mask=True: HostTL.store(ptr, value, mask))


@pytest.mark.parametrize("world,rank", [(4, 0), (4, 1), (8, 7)])
@pytest.mark.parametrize("has_rejections", [False, True])
def test_dcp_mtp_draft_masks_rejected_rows_and_preserves_target_buffers(monkeypatch, world, rank, has_rejections):
    tl = DraftTL()
    kernel = extract(DRAFT_KERNEL, "prepare_dcp_mtp_draft_kernel", {"tl": tl})

    class Launch:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                tl.programs = grid[0]
                for tl.pid in range(tl.programs):
                    kernel(*(Pointer(arg) if isinstance(arg, torch.Tensor) else arg for arg in args), **kwargs)
            return run

    resident = load_module(
        ROOT / "vllm_ascend/distributed/kv_transfer/sparse_offload/resident_sorted_cache.py",
        "dcp_mtp_resident_test", monkeypatch,
    )
    ns = {"dataclass": dataclass, "torch": torch}
    definitions(ROOT / "vllm_ascend/attention/utils.py", {"DcpSFABatch"}, ns)
    # A dense Q1, compact Q2 rows crossing stripe/virtual-block boundaries,
    # an ignored Q2 request, and a short recovery suffix at the context limit.
    query_starts = torch.tensor([0, 1, 3, 5, 7, 10], dtype=torch.int32)
    ends = torch.tensor([128, 130, world * 128 + 1, 257, 1048576], dtype=torch.int32)
    rejected = torch.tensor([0, 1, 0, 2, 0], dtype=torch.int32) if has_rejections else None
    origins = torch.tensor([0, 4096, 4096, 4096, 0], dtype=torch.int32)
    rows = query_starts[-1].item()
    padded = rows + 3
    positions = torch.cat([torch.arange(end - width, end) for end, width in zip(ends, query_starts.diff())])
    owners = positions // 128 % world == rank
    slots0 = torch.where(owners, positions % 128 + 1280, -1)
    slots1 = positions % 128 + 2560
    topk = torch.empty((padded, 1, 8), dtype=torch.int32)
    counts = torch.empty((padded, 16), dtype=torch.int32)
    layout = resident.ResidentDcpLayout(world, rank, origins, counts[:rows])
    compact_rows = torch.tensor([1, 2, 3, 4, 5, 6])
    dcp = ns["DcpSFABatch"](
        layout, ends.clone(), positions, topk[:rows], counts[:rows], (),
        torch.tensor([-1, 1, 1, 2, 2, 3, 3, -1, -1, -1]), torch.zeros(rows), compact_rows,
        last_plan=object(),
    )
    common = NS(
        sfa_dcp=dcp, indexer_slot_mapping=slots1, slot_mapping=slots0,
        num_reqs=ends.numel(), seq_lens=ends, query_start_loc=query_starts, positions=positions,
    )
    proposer_positions = torch.nn.functional.pad(positions.int(), (0, 3), value=999)
    proposer = NS(
        runner=NS(block_size=128, decode_threshold=2, _dsa_dcp_topk=topk,
                  _dsa_dcp_counts=counts, _dsa_dcp_planner_counts=layout.selected_counts),
        _get_positions=lambda n: proposer_positions[:n],
        slot_mapping_group=[torch.empty(padded, dtype=torch.int32)],
        indexer_slot_mapping_group=[torch.empty(padded, dtype=torch.int32)],
        _dcp_draft_positions=torch.empty(padded, dtype=torch.int32),
        _dcp_draft_lengths=torch.empty(ends.numel(), dtype=torch.int32),
    )
    prepare = extract(PROPOSER, "_prepare_dcp_draft_metadata", {
        "replace": replace, "get_vectorcore_num": lambda: 2,
        "prepare_dcp_mtp_draft_kernel": Launch(), "PADDING_SLOT_ID": -1,
    })
    originals = [tensor.clone() for tensor in (positions, slots0, slots1, ends)]
    prepare(proposer, common, rows, padded, rejected)
    draft = common.sfa_dcp
    visible_ends = ends if rejected is None else ends - rejected
    visible = torch.repeat_interleave(visible_ends, query_starts.diff())
    for actual, source in ((common.slot_mapping, slots0), (common.indexer_slot_mapping, slots1),
                           (draft.positions, positions)):
        assert actual[:rows].tolist() == torch.where(positions < visible, source, -1).tolist()
    assert common.slot_mapping[rows:].eq(-1).all() and common.indexer_slot_mapping[rows:].eq(-1).all()
    assert common.positions.tolist() == positions.tolist() + [0, 0, 0]
    assert draft.local_seq_lens.tolist() == [
        int(origin) + sum((t // 128) % world == rank for t in range(int(end)))
        for origin, end in zip(origins, visible_ends)
    ]
    assert draft.last_plan is None and dcp.last_plan is not None
    assert draft.mask_indexer_writes == has_rejections and not dcp.mask_indexer_writes
    assert draft.layout.live_origins.data_ptr() == dcp.layout.live_origins.data_ptr()
    assert draft.layout.selected_counts.data_ptr() == dcp.layout.selected_counts.data_ptr()
    assert draft.compact_rows is compact_rows and draft.topk.shape[0] == padded
    assert all(current.equal(original) for current, original in zip((positions, slots0, slots1, ends), originals))


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize(
    "slots,masked",
    [([2, 4, 5], False), ([2, -1, 5], True), ([-1, 4, -1], True), ([-1, -1, -1], True)],
)
def test_dcp_draft_group1_scatter_uses_in_bounds_slots(quantized, slots, masked):
    tree = ast.parse(SFA.read_text(encoding="utf8"))
    impl = next(node for node in tree.body if getattr(node, "name", None) == "AscendSFAImpl")
    forward = next(node for node in impl.body if getattr(node, "name", None) == "forward")
    scatter = next(node for node in forward.body if isinstance(node, ast.If)
                   and any(isinstance(stmt, ast.Assign)
                           and ast.unparse(stmt.targets[0]) == "actual_tokens" for stmt in node.body))
    start = next(i for i, stmt in enumerate(scatter.body)
                 if isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == "actual_tokens")
    end = next(i for i, stmt in enumerate(scatter.body[start:], start)
               if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == "self.is_kv_producer")
    launches = []

    def strict_scatter(cache, indices, values):
        indices = indices.flatten().long()
        assert bool(((indices >= 0) & (indices < cache.shape[0])).all()), "out-of-bounds indexer scatter"
        launches.append(indices)
        cache.index_copy_(0, indices, values)

    keys = torch.arange(16).view(4, 1, 4).to(torch.int8 if quantized else torch.bfloat16)
    scales = torch.arange(4, dtype=torch.float16).view(-1, 1)
    key_cache = torch.full((8, 4), 51, dtype=keys.dtype)
    scale_cache = torch.full((8, 1), 1.25, dtype=scales.dtype)
    reference_keys, reference_scales = key_cache.clone(), scale_cache.clone()
    for row, slot in enumerate(slots):
        if slot >= 0:
            reference_keys[slot] = keys[row, 0]
            reference_scales[slot] = scales[row]
    namespace = dict(
        self=NS(use_sparse_c8_indexer=quantized,
                _mask_staged_index_scatter_padding=extract(SFA, "_mask_staged_index_scatter_padding", {"torch": torch})),
        attn_metadata=NS(num_actual_tokens=3), dcp=NS(mask_indexer_writes=masked),
        idx_slot_mapping=torch.tensor([*slots, -1]), k_li=keys, k_li_scale=scales,
        kv_cache=(None, None, key_cache, scale_cache),
        torch_npu=NS(npu_scatter_nd_update_=strict_scatter),
    )
    exec(compile(ast.Module(body=scatter.body[start:end], type_ignores=[]), str(SFA), "exec"), namespace)
    torch.testing.assert_close(key_cache, reference_keys)
    assert len(launches) == (2 if quantized else 1)
    if quantized:
        torch.testing.assert_close(scale_cache, reference_scales)
        assert launches[0].equal(launches[1])
