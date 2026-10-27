# SPDX-License-Identifier: Apache-2.0
"""Execute production bank-table selection on CPU, with fixed per-layer banks."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / "vllm_ascend/worker/model_runner_v1.py"


def method(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if getattr(n, "name", "") == "NPUModelRunner")
    return next(n for n in cls.body if getattr(n, "name", "") == name)


@pytest.mark.parametrize("banks", [1, 2, 3, 8])
def test_all_shadow_bank_tables_follow_request_row_order(banks):
    rows = []
    tables = tuple(NS(add_row=lambda ids, row, b=b: rows.append((b, row, ids))) for b in range(banks))
    requests = {
        name: NS(
            block_allocation_mode="prefill_child",
            block_ids_by_bank=tuple(([100 * req + b], [1000 + b]) for b in range(banks)),
        )
        for req, name in enumerate(("a", "b"))
    }
    for request in requests.values():
        request.block_ids = request.block_ids_by_bank[0]
    obj = NS(
        layerwise_prefill_p_node=True,
        layerwise_prefill_bank_count=banks,
        input_batch=NS(req_ids=["b", "a"], layerwise_prefill_block_tables=tables, block_table=tables[0]),
        requests=requests,
    )
    namespace = dict(DSABlockAllocationMode=NS(PREFILL_CHILD="prefill_child"))
    module = ast.parse("from __future__ import annotations")
    module.body.append(method("_refresh_layerwise_prefill_block_tables"))
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    assert namespace["_refresh_layerwise_prefill_block_tables"](obj) == tables
    assert rows == [
        (b, row, requests[name].block_ids_by_bank[b]) for row, name in enumerate(("b", "a")) for b in range(1, banks)
    ]


@pytest.mark.parametrize("banks", [1, 2, 3, 4, 8])
def test_unequal_prefixes_use_fixed_bank_and_disjoint_request_slots(banks):
    owner = method("_build_attention_metadata")
    getter = next(
        n for n in ast.walk(owner) if isinstance(n, ast.FunctionDef) and n.name == "_get_block_table_and_slot_mapping"
    )
    wrapper = ast.parse(
        "def make(self, kv_cache_groups):\n"
        "    num_reqs = 2\n"
        "    num_reqs_padded = 3\n"
        "    num_tokens = 4096\n"
        "    num_tokens_padded = 4100\n"
        "    prefill_bank_views = {}\n"
    ).body[0]
    wrapper.body.append(getter)
    wrapper.body.append(ast.parse("return _get_block_table_and_slot_mapping").body[0])
    module = ast.parse("from __future__ import annotations")
    module.body.append(wrapper)
    namespace = dict(torch=torch, EncoderOnlyAttentionSpec=type("Unused", (), {}))
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    tables, expected = [], {}
    for bank in range(banks):
        groups = []
        for group in range(2):
            base = bank * 100000 + group * 1000000
            blocks = torch.arange(6).reshape(3, 2) + base
            # A: prefix 20000 + 2000 new; B: prefix 10000 + 2096 new.
            # B owns a different physical region even though bank IDs match.
            slots = torch.cat((torch.arange(20000, 22000),
                               torch.arange(60000, 62096), torch.zeros(4))).long() + base
            expected[bank, group] = (blocks.clone(), slots.clone())
            groups.append(NS(get_device_tensor=lambda blocks=blocks: blocks, slot_mapping=NS(gpu=slots)))
        tables.append(tuple(groups))
    obj = NS(
        layerwise_prefill_p_node=True, layerwise_prefill_bank_count=banks,
        input_batch=NS(layerwise_prefill_block_tables=tables, block_table=tables[0]),
        pcp_size=1, use_cp=False, device="cpu",
        model_config=NS(enable_return_routed_experts=False),
    )
    get = namespace["make"](obj, [NS(kv_cache_spec=object())] * 2)
    for group in range(2):
        for bank in range(banks):
            blocks, slots = get(group, bank)
            want_blocks, want_slots = expected[bank, group]
            assert torch.equal(blocks[:2], want_blocks[:2])
            assert torch.equal(slots[:4096], want_slots[:4096])
            assert blocks[2].tolist() == [0, 0]
            assert slots[4096:].tolist() == [-1] * 4
            cached = get(group, bank)
            assert cached[0] is blocks and cached[1] is slots
