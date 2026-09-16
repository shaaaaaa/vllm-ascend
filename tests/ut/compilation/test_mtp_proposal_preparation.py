# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-derived checks of proposal selection, ownership and startup warmup."""

import ast
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from sfa_test_support import ROOT, extract
from torch.utils._python_dispatch import TorchDispatchMode

PROPOSER = ROOT / "vllm_ascend/spec_decode/eagle_proposer.py"
RUNNER = ROOT / "vllm_ascend/worker/model_runner_v1.py"


def compile_body(body, args, namespace):
    tree = ast.parse(f"def run({args}):\n    pass\n")
    tree.body[0].body = deepcopy(body)
    exec(compile(ast.fix_missing_locations(tree), str(PROPOSER), "exec"), namespace)
    return namespace["run"]


@pytest.mark.parametrize("drafts,staged", [(1, False), (2, False), (1, True), (2, True)])
def test_only_later_drafts_clone_tables_and_only_consumed_positions_are_gathered(drafts, staged):
    tree = ast.parse(PROPOSER.read_text(encoding="utf8"))
    propose = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_propose")
    start = next(
        i
        for i, n in enumerate(propose.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "used_update_positions"
    )
    clone = next(
        n
        for n in propose.body
        if isinstance(n, ast.If) and "common_attn_metadata.block_table_tensor.clone()" in ast.unparse(n)
    )
    body = [*propose.body[start : start + 2], clone, ast.parse("return used_update_positions").body[0]]
    run = compile_body(body, "self, common_attn_metadata, use_staged_mtp_draft_graph, token_indices_to_sample", {})
    subject = SimpleNamespace(num_speculative_tokens=drafts, uses_mrope=False, positions=torch.arange(8))
    tables = [torch.arange(16).view(2, 8), torch.arange(24).view(2, 12)]
    common = SimpleNamespace(block_table_tensor=tables[0], indexer_block_table_tensor=tables[1])
    positions = run(subject, common, staged, torch.tensor([1, 3]))
    assert (positions is not None) == (drafts > 1 or staged)
    if positions is not None:
        assert positions.tolist() == [1, 3]
    for original, current in zip(tables, (common.block_table_tensor, common.indexer_block_table_tensor)):
        assert torch.equal(original, current)
        assert (original.data_ptr() != current.data_ptr()) == (drafts > 1 and not staged)
        if drafts > 1 and not staged:
            current.zero_()
            assert torch.count_nonzero(original) > 0


class Operations(TorchDispatchMode):
    def __init__(self):
        self.indices = 0

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        self.indices += int(op == torch.ops.aten.index.Tensor)
        return op(*args, **(kwargs or {}))


def selection():
    tree = ast.parse(RUNNER.read_text(encoding="utf8"))
    # Execute the actual enclosing selection branch, including its fallback.
    parent = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and n.orelse
        and any(isinstance(c, ast.If) and "self.drafter.method == 'mtp'" in ast.unparse(c.test) for c in n.orelse)
    )
    body = [*parent.orelse, ast.parse("return target_token_ids, target_positions, target_hidden_states").body[0]]
    return compile_body(body, "self, token_indices, hidden_states, aux_hidden_states=None", {"torch": torch})


@pytest.mark.parametrize("n", [1, 4, 16])
@pytest.mark.parametrize("accepted", [1, 2])
@pytest.mark.parametrize("strided", [False, True])
def test_padded_identity_selection_matches_gather_and_draft_owned_outputs(n, accepted, strided):
    prepare = extract(
        PROPOSER,
        "prepare_inputs_padded",
        {
            "torch": torch,
            "HAS_TRITON": False,
            "AscendCommonAttentionMetadata": SimpleNamespace,
        },
    )
    first = extract(PROPOSER, "set_inputs_first_pass", {"torch": torch})
    total = n * 2
    ids, positions = torch.arange(total + 8, dtype=torch.int32), torch.arange(total + 8)
    hidden = torch.arange((total + 8) * 32, dtype=torch.float32).view(total + 8, 32)
    if strided:
        hidden = hidden[:, ::2]
    common = SimpleNamespace(
        num_reqs=n,
        query_start_loc=torch.arange(n + 1) * 2,
        query_start_loc_cpu=torch.arange(n + 1) * 2,
        seq_lens_cpu=torch.full((n,), 100),
        seq_lens=torch.full((n,), 100),
        num_actual_tokens=total,
        num_input_tokens=total,
        block_table_tensor=torch.ones(n, 4),
        slot_mapping=torch.arange(total),
        positions=positions,
        num_computed_tokens_cpu=torch.full((n,), 98),
    )
    proposer = SimpleNamespace(
        arange=torch.arange(total + 8),
        pcp_size=1,
        runner=SimpleNamespace(actual_seq_lengths_q=None, attn_state="spec", decode_token_per_req=2),
    )
    cad, indices, sample_indices, rejected = prepare(
        proposer,
        common,
        SimpleNamespace(cu_num_draft_tokens=torch.arange(1, n + 1)),
        torch.full((n,), accepted, dtype=torch.int64),
    )
    runner = SimpleNamespace(
        drafter=SimpleNamespace(method="mtp", needs_extra_input_slots=False),
        num_spec_tokens=1,
        use_cp=False,
        vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(disable_padded_drafter_batch=False)),
        input_ids=SimpleNamespace(gpu=ids),
        _get_positions=lambda index: positions[index],
        use_aux_hidden_state_outputs=False,
    )
    ops = Operations()
    with ops:
        selected = selection()(runner, indices, hidden)
    assert ops.indices == 0
    snapshots = []
    for inputs in ((ids[indices], positions[indices], hidden[indices]), selected):
        worker = SimpleNamespace(
            needs_extra_input_slots=False,
            runner=object(),
            pcp_size=1,
            dcp_size=1,
            uses_xdrope_dim=0,
            input_ids=torch.full_like(ids, -9),
            positions=torch.full_like(positions, -9),
            hidden_states=torch.full_like(hidden, -9),
        )
        worker._set_positions = lambda count, values, target=worker: target.positions[:count].copy_(values)
        first(
            worker,
            inputs[0],
            torch.arange(n, dtype=torch.int32) + 900,
            inputs[1],
            inputs[2],
            sample_indices,
            cad,
            rejected,
        )
        snapshots.append((worker.input_ids.clone(), worker.positions.clone(), worker.hidden_states.clone()))
    assert all(torch.equal(a, b) for a, b in zip(*snapshots))
    ids.zero_()
    positions.zero_()
    hidden.zero_()
    assert all(
        torch.equal(a, b) for a, b in zip((worker.input_ids, worker.positions, worker.hidden_states), snapshots[1])
    )


@pytest.mark.parametrize("reason", ["compact", "cp", "multi_draft", "other_proposer", "extra_slots"])
def test_nonidentity_cases_keep_indexing(reason):
    values = torch.arange(12)
    runner = SimpleNamespace(
        drafter=SimpleNamespace(method="mtp", needs_extra_input_slots=False),
        num_spec_tokens=1,
        use_cp=False,
        vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(disable_padded_drafter_batch=False)),
        input_ids=SimpleNamespace(gpu=values),
        _get_positions=lambda index: values[index],
        use_aux_hidden_state_outputs=False,
    )
    if reason == "compact":
        runner.vllm_config.speculative_config.disable_padded_drafter_batch = True
    elif reason == "cp":
        runner.use_cp = True
    elif reason == "multi_draft":
        runner.num_spec_tokens = 2
    elif reason == "extra_slots":
        runner.drafter.needs_extra_input_slots = True
    else:
        runner.drafter.method = "eagle"
    indices = torch.tensor([4, 1, 7])
    assert all(t.tolist() == [4, 1, 7] for t in selection()(runner, indices, values))


def test_warmup_uses_private_buffers_and_serving_specializations():
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def call(sampled, backup, output, counts, n, vocab, row_stride, token_stride, **kwargs):
                calls.append(
                    (
                        grid,
                        sampled.shape,
                        backup.data_ptr(),
                        output.data_ptr(),
                        counts.dtype,
                        n,
                        vocab,
                        row_stride,
                        token_stride,
                        kwargs,
                    )
                )

            return call

    warmup = extract(
        PROPOSER,
        "warmup_next_mtp_tokens",
        {
            "torch": torch,
            "triton": SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
            "get_vectorcore_num": lambda: 8,
            "_PREPARE_INPUTS_BLOCK_SIZE": 4,
            "prepare_next_mtp_tokens_kernel": Kernel(),
        },
    )
    live = torch.full((5,), 99, dtype=torch.int32)
    subject = SimpleNamespace(
        device="cpu",
        runner=SimpleNamespace(max_num_reqs=5, input_batch=SimpleNamespace(vocab_size=1000)),
        backup_next_token_ids=live,
    )
    warmup(subject)
    assert len(calls) == 2 and torch.all(live == 99)
    for call, width in zip(calls, (1, 2)):
        grid, shape, backup_ptr, output_ptr, count_type, n, vocab, row_stride, token_stride, kwargs = call
        assert grid == (2,) and shape == (5, width) and n == 5 and vocab == 1000
        assert row_stride == width and token_stride == 1 and count_type == torch.int64
        assert live.data_ptr() not in (backup_ptr, output_ptr)
        assert kwargs == {"WIDTH": width, "BLOCK_SIZE": 4}
