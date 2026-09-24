# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811 -- imported pytest fixtures
"""Execute production ordinary/async mapping with bank-boundary inputs."""

from pathlib import Path

import pytest
import torch
from sfa_test_support import HostTL, Pointer, extract, load_module
from test_sfa_async_mtp import setup as setup

ROOT = Path(__file__).resolve().parents[3]


def block_map(slots):
    mapping = torch.empty(18 * slots, dtype=torch.int32)
    for u in range(slots):
        ids = [*range(8 * u, 8 * u + 8), *range(8 * (slots + u), 8 * (slots + u) + 8), 16 * slots + u, 17 * slots + u]
        physical = [*range(16 * u, 16 * u + 16), 16 * slots + 2 * u, 16 * slots + 2 * u + 1]
        mapping[ids] = torch.tensor(physical, dtype=torch.int32)
    return mapping


@pytest.mark.parametrize("width", [9, 18, 36])
def test_fused_metadata_kernel_matches_builder_and_keeps_padding(monkeypatch, width):
    m = load_module(ROOT / "vllm_ascend/worker/dsa_shared_pool.py", "paired_views", monkeypatch)
    mapping = block_map(4)
    builder = m.MixedIndexerMetadata(4, 36, 64, torch.device("cpu"), block_map=tuple(mapping.tolist()))
    # Noncontiguous source table has an explicit stride in the kernel.
    backing = torch.arange(4 * 72, dtype=torch.int32).reshape(4, 72) % 72
    table = backing[:3, :width]
    table[2].zero_()
    slots = torch.tensor([0, 127, 128, 31 * 128 + 127, 32 * 128, 63 * 128 + 127, 64 * 128, 71 * 128 + 127, -1])
    expected = builder.update(table, slots)
    out_table = torch.empty_like(expected[0])
    out_slots = torch.empty_like(expected[1])
    tl = HostTL()
    fn = extract(
        ROOT / "vllm_ascend/ops/triton/spec_decode/indexer_c8_metadata.py", "_map_indexer_metadata", {"tl": tl}
    )
    for pid in range((max(table.numel(), slots.numel()) + 31) // 32):
        tl.pid = pid
        fn(
            *(Pointer(t) for t in (table, slots, mapping, out_table, out_slots)),
            table.numel(),
            slots.numel(),
            table.shape[1],
            table.stride(0),
            32,
        )
    assert torch.equal(out_table, expected[0]) and torch.equal(out_slots, expected[1])
    assert torch.equal(out_table, mapping[table.long()])
    assert out_slots[-1] == -1
    pointers = tuple(t.data_ptr() for t in expected)
    assert tuple(t.data_ptr() for t in builder.update(table, slots)) == pointers


@pytest.mark.parametrize("accepted", [1, 2])
def test_async_mtp_uses_physical_table_not_legacy_double_mapping(setup, accepted):
    r, s, shape, events, _ = setup
    for group in r.input_batch.block_table.block_tables:
        group.block_size = 128
        group.kernel_sizes = [128]
    item = r._async_snapshot.metadata["l0"]
    item.indexer_block_table.fill_(25)
    # canonical block 25 maps to a different, non-double physical block.
    item.indexer_c8_block_table = torch.full_like(item.indexer_block_table, 9)
    item.indexer_c8_slot_mapping = torch.full_like(item.indexer_slot_mapping, -1)
    r._async_counts.fill_(accepted)
    r._update_states(s)
    metadata, _ = r._build_attention_metadata(**shape)
    actual = metadata["l0"].indexer_c8_slot_mapping
    pos = r.positions.gpu[:6]
    assert torch.equal(actual[:6], 9 * 128 + pos % 128)
    assert actual[6:].eq(-1).all()
    assert events.count("kernel") == 1 and "sync" not in events


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_builder_does_not_allocate_unused_paired_map(paired, selected):
    import ast
    from types import SimpleNamespace as NS
    from unittest.mock import Mock

    source = ROOT / "vllm_ascend/attention/sfa_v1.py"
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendSFAMetadataBuilder"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    start = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_mixed_indexer_metadata" for t in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "speculative_config" for t in n.targets)
    )
    factory = Mock()
    mask = Mock(return_value=(selected,))
    mapping = tuple(range(36)) if paired else None
    config = NS(indexer_c8_shared_block_factor=2, indexer_hbm_block_map=mapping, indexer_c8_layer_mask=mask)
    subject = NS(max_blocks=1391)
    exec(
        compile(ast.Module(body=init.body[start:end], type_ignores=[]), str(source), "exec"),
        dict(
            self=subject,
            get_ascend_config=lambda: config,
            layer_names=["draft"],
            MixedIndexerMetadata=factory,
            vllm_config=NS(scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=32)),
            device="cpu",
        ),
    )
    assert factory.call_count == int(not paired or selected)
    if paired and not selected:
        assert subject._mixed_indexer_metadata is None
    if factory.called:
        assert factory.call_args.args[1] == (1404 if paired else 1395)


def test_paired_metadata_on_npu_matches_host_mapping():
    pytest.importorskip("torch_npu")
    pytest.importorskip("triton")
    if not torch.npu.is_available():
        pytest.skip("Requires NPU")
    from vllm_ascend.ops.triton.spec_decode.indexer_c8_metadata import map_indexer_metadata

    mapping = block_map(4)
    table = torch.tensor([[0, 31, 32, 63, 64, 71], [8, 40, 65, 67, 70, 0]], dtype=torch.int32)
    slots = torch.tensor([-1, 0, 127, 31 * 128 + 127, 32 * 128, 63 * 128 + 127, 64 * 128, 71 * 128 + 127])
    device_table, device_slots = table.npu(), slots.npu()
    out_table = torch.empty_like(device_table)
    out_slots = torch.empty_like(device_slots)
    map_indexer_metadata(device_table, device_slots, mapping.npu(), out_table, out_slots)
    torch.npu.synchronize()
    assert torch.equal(out_table.cpu(), mapping[table.long()])
    expected = torch.where(slots >= 0, mapping[(slots.clamp(min=0) // 128).long()].long() * 128 + slots % 128, -1)
    assert torch.equal(out_slots.cpu(), expected)
