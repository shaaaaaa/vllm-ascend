# SPDX-License-Identifier: Apache-2.0
"""Execute production bank-table selection on CPU, including mixed phases."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
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


@pytest.mark.parametrize("banks,phases", [(2, [0, 1, 0]), (3, [2, 0, 1]), (8, [0, 7, 3]), (8, [7, 7, 7])])
def test_mixed_request_phases_select_correct_physical_rows_and_slots(banks, phases):
    owner = method("_build_attention_metadata")
    getter = next(
        n for n in ast.walk(owner) if isinstance(n, ast.FunctionDef) and n.name == "_get_block_table_and_slot_mapping"
    )
    wrapper = ast.parse(
        "def make(self, kv_cache_groups, phase_by_request_cpu, phase_by_token_cpu):\n"
        "    num_reqs = 3\n"
        "    num_reqs_padded = 4\n"
        "    num_tokens = 6\n"
        "    num_tokens_padded = 8\n"
        "    prefill_bank_views = {}\n"
        "    mixed_bank_phase = bool(np.any(phase_by_request_cpu != phase_by_request_cpu[0]))\n"
        "    phase_by_request_device = None\n"
        "    phase_by_token_device = None\n"
    ).body[0]
    wrapper.body.append(getter)
    wrapper.body.append(ast.parse("return _get_block_table_and_slot_mapping").body[0])
    module = ast.parse("from __future__ import annotations")
    module.body.append(wrapper)
    namespace = dict(np=np, torch=torch, EncoderOnlyAttentionSpec=type("Unused", (), {}))
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    tables = []
    for bank in range(banks):
        groups = []
        for group in range(2):
            blocks = torch.arange(8).reshape(4, 2) + bank * 100 + group * 10000
            slots = torch.arange(8) + bank * 1000 + group * 10000
            groups.append(NS(get_device_tensor=lambda blocks=blocks: blocks, slot_mapping=NS(gpu=slots)))
        tables.append(tuple(groups))
    obj = NS(
        layerwise_prefill_p_node=True,
        layerwise_prefill_bank_count=banks,
        input_batch=NS(layerwise_prefill_block_tables=tables, block_table=tables[0]),
        pcp_size=1,
        use_cp=False,
        device="cpu",
        model_config=NS(enable_return_routed_experts=False),
    )
    request_phases = np.array(phases, dtype=np.int32)
    token_phases = np.repeat(request_phases, 2)
    get = namespace["make"](obj, [NS(kv_cache_spec=object())] * 2, request_phases, token_phases)
    for group in range(2):
        for logical_bank in range(banks):
            blocks, slots = get(group, logical_bank)
            expected_blocks = torch.tensor(
                [
                    [10000 * group + 100 * ((logical_bank + phase) % banks) + row * 2 + col for col in range(2)]
                    for row, phase in enumerate(phases)
                ]
            )
            expected_slots = torch.tensor(
                [
                    10000 * group + 1000 * ((logical_bank + int(phase)) % banks) + row
                    for row, phase in enumerate(token_phases)
                ]
            )
            assert torch.equal(blocks[:3], expected_blocks)
            assert torch.equal(slots[:6], expected_slots)
            assert blocks[3].tolist() == [0, 0]
            assert slots[6:].tolist() == [-1, -1]
            cached = get(group, logical_bank)
            assert cached[0] is blocks and cached[1] is slots
