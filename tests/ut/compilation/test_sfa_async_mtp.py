# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ordering/gate contracts and an optional actual NPU metadata-kernel test."""

import ast
import importlib.util
import sys
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from test_mtp_next_tokens import HostTL, Pointer, extract
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "vllm_ascend/ops/triton/spec_decode/async_mtp.py"
RUNNER = ROOT / "vllm_ascend/worker/sfa_async_mtp.py"


class Kernel:
    def __init__(self, events):
        self.tl = HostTL()
        self.tl.static_range = range
        self.body = extract(KERNEL, "prepare_async_mtp_kernel", {"tl": self.tl})
        self.events = events

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.events.append("kernel")
            ptrs = [Pointer(t) for t in args[:9]]
            self.tl.programs = grid[0]
            for pid in range(grid[0]):
                self.tl.pid = pid
                self.body(*ptrs, *args[9:], **kwargs)

        return launch


class Buffer:
    def __init__(self, size, dtype=torch.int32):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros_like(self.cpu)


@pytest.fixture
def setup():
    events = []
    context = SimpleNamespace(staged_sfa_graph_dummy_run=False)
    source = ROOT.parent / "vllm/vllm/v1/worker/gpu_model_runner.py"
    getter = extract(source, "_get_valid_sampled_token_count", {})

    class Base:
        _get_valid_sampled_token_count = getter

        def _update_states(self, scheduled):
            counts = self._get_valid_sampled_token_count()
            events.append("state_update")
            for rid, optimistic in zip(
                scheduled.scheduled_cached_reqs.req_ids, scheduled.scheduled_cached_reqs.num_computed_tokens
            ):
                row = self.input_batch.req_id_to_index[rid]
                actual = optimistic - (2 - counts[row])
                self.input_batch.num_computed_tokens_cpu[row] = actual
                self.requests[rid].num_computed_tokens = actual

        def _prepare_inputs(self, scheduled, widths):
            events.append("normal_prepare")
            n = self.input_batch.num_reqs
            bases = self.input_batch.num_computed_tokens_cpu[:n]
            self.positions.gpu[: 2 * n] = torch.from_numpy(np.repeat(bases, 2) + np.tile([0, 1], n))
            self.seq_lens.np[:n] = bases + 2
            self.seq_lens.gpu[:n] = torch.from_numpy(bases + 2)
            return torch.arange(2 * n), object(), 2 * n

        def _prepare_input_ids(self, scheduled, count, ends):
            events.append("input_ids")

        def _calc_spec_decode_metadata(self, *args):
            return SimpleNamespace(logits_indices=torch.arange(2 * self.input_batch.num_reqs))

        def _apply_staged_sfa_route(self, route):
            return route

        def _build_attention_metadata(self, **kwargs):
            events.append("normal_build")
            return self.normal_metadata

        def _model_forward(self):
            events.append("replay")
            self.submitted = tuple(
                t.clone()
                for t in (
                    self.positions.gpu,
                    self.seq_lens.gpu,
                    self.item.seq_lens,
                    self.item.slot_mapping,
                    self.item.indexer_slot_mapping,
                )
            )
            return "output"

        def _copy_valid_sampled_token_count(self, tokens, counts):
            self.valid_sampled_token_count_cpu = counts.clone()
            self.input_batch.prev_sampled_token_ids = tokens.view(-1, 1)

        def capture_model(self):
            events.append("capture")
            return "captured"

    def rope(positions, cache):
        events.append("rope")
        return positions.clone(), -positions.clone()

    ns = dict(
        np=np,
        logger=SimpleNamespace(info=lambda *args: None),
        torch=torch,
        copy=copy,
        contextmanager=contextmanager,
        dataclass=dataclass,
        Any=Any,
        NPUModelRunner=Base,
        get_forward_context=lambda: context,
        HAS_TRITON=True,
        sfa_full_graph_enabled=lambda c: True,
        lmhead_tp_enable=lambda: False,
        _mtp_dw_diag_enabled=lambda: False,
        npu_content_diagnostics_enabled=lambda: False,
        _decode_window_save_window_size=lambda: 0,
        staged_sfa_metadata_sparse_route=lambda m, ids: (m.reason, m.frontiers, m.cold),
        StagedSFARouteReason=SimpleNamespace(ELIGIBLE="eligible"),
        AscendAttentionState=SimpleNamespace(SpecDecoding="decode"),
        prepare_async_mtp_kernel=Kernel(events),
        get_cos_and_sin_mla=rope,
        triton=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
    )
    tree = ast.parse(RUNNER.read_text(encoding="utf8"))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.fix_missing_locations(tree), str(RUNNER), "exec"), ns)
    runner = ns["AsyncSFAModelRunner"].__new__(ns["AsyncSFAModelRunner"])
    n, capacity = 3, 8
    ids = tuple(f"r{i}" for i in range(n))
    bases = np.array([5000, 5100, 5200], dtype=np.int32)
    groups = []
    for block in (128, 64):
        table = torch.arange(5 * 128, dtype=torch.int32).view(5, 128) + 10
        groups.append(
            SimpleNamespace(
                _block_table_dirty=False,
                block_size=block,
                kernel_sizes=[block],
                num_blocks_per_row=np.full(4, 128),
                table=table,
                slot_mapping=Buffer(capacity),
            )
        )
    runner.input_batch = SimpleNamespace(
        num_reqs=n,
        req_ids=list(ids),
        req_id_to_index=dict(zip(ids, range(n))),
        prev_req_id_to_index=dict(zip(ids, range(n))),
        num_computed_tokens_cpu=bases.copy(),
        num_computed_tokens_cpu_tensor=torch.zeros(4, dtype=torch.int32),
        num_prompt_tokens=np.full(n, 4096),
        req_prompt_embeds={},
        block_table=SimpleNamespace(block_tables=groups),
        prev_sampled_token_ids=torch.ones((n, 1), dtype=torch.int32),
        sampling_metadata=SimpleNamespace(no_penalties=True, bad_words_token_ids={}),
    )
    runner.input_batch.num_computed_tokens_cpu_tensor[:n] = torch.from_numpy(bases)
    runner.input_batch.num_computed_tokens_cpu = runner.input_batch.num_computed_tokens_cpu_tensor.numpy()[:n]
    runner.requests = {
        rid: SimpleNamespace(prev_num_draft_len=1, num_computed_tokens=int(base)) for rid, base in zip(ids, bases)
    }
    runner.seq_lens, runner.positions = Buffer(4), Buffer(capacity, torch.int64)
    runner.input_ids = Buffer(capacity)
    runner._async_live_execute = False
    runner._async_host_write = runner._async_query_layout = None
    runner.prepare_inputs_event = None
    runner.positions.gpu[: 2 * n] = torch.from_numpy(np.repeat(bases, 2) + np.tile([0, 1], n))
    runner.seq_lens.np[:n] = bases + 2
    runner.seq_lens.gpu.copy_(runner.seq_lens.cpu)
    runner.item = item = SimpleNamespace(
        seq_lens=torch.zeros(5, dtype=torch.int32),
        seq_lens_cpu=torch.zeros(5, dtype=torch.int32),
        block_table=groups[0].table,
        indexer_block_table=groups[1].table,
        slot_mapping=groups[0].slot_mapping.gpu,
        indexer_slot_mapping=groups[1].slot_mapping.gpu,
        dsa_cp_context=None,
        req_ids=ids,
        decode_remap_boundary_buffer=SimpleNamespace(_values=(4096,) * 6 + (0, 0)),
        decode_remap_boundary_ready=True,
    )
    item.seq_lens_cpu[:n] = torch.from_numpy(bases + 2)
    common = SimpleNamespace(
        seq_lens=runner.seq_lens.gpu,
        seq_lens_cpu=runner.seq_lens.cpu,
        num_computed_tokens_cpu=runner.input_batch.num_computed_tokens_cpu_tensor,
        num_reqs=4,
        slot_mapping=item.slot_mapping,
        indexer_slot_mapping=item.indexer_slot_mapping,
    )
    runner.normal_metadata = ({"l0": item, "l1": item}, common)
    shape = dict(num_tokens=6, num_tokens_padded=8, num_reqs=3, num_reqs_padded=4, max_query_len=2, full_graph=True)
    key = SimpleNamespace(token_capacity=8, request_capacity=4)
    context.staged_sfa_graph_key = key
    context.staged_sfa_route = SimpleNamespace(frontiers=(4096,) * n)
    runner._async_snapshot = ns["_Snapshot"](
        ids,
        tuple(runner.requests.values()),
        bases.copy(),
        runner.normal_metadata[0],
        common,
        ns["_shape"](shape),
        key,
        (4096,) * n,
        (4096,) * n,
        0,
        item.decode_remap_boundary_buffer._values,
        "l0",
    )
    runner._staged_sfa_impls = (("l0", object()), ("l1", object()))
    runner._async_pending = runner._async_built = None
    runner._async_epoch = runner._async_counts_epoch = 1
    runner._async_counts = torch.tensor([1, 2, 1], dtype=torch.int64)
    runner._async_bases = torch.from_numpy(bases.copy())
    runner.valid_sampled_token_count_cpu = runner._async_counts.clone()
    runner.valid_sampled_token_count_event = SimpleNamespace(synchronize=lambda: events.append("sync"))
    runner._draft_token_ids = torch.ones((n, 1), dtype=torch.int64)
    runner._fixed_decode_cu_num_tokens = np.arange(2, 33, 2)
    runner.async_mtp_replays = 0
    for flag in (
        "num_discarded_requests",
        "use_cp",
        "uses_mrope",
        "uses_xdrope_dim",
        "calculate_kv_scales",
        "need_accepted_tokens",
        "lora_config",
        "cascade_attn_enabled",
        "enable_prompt_embeds",
        "is_multimodal_model",
        "num_prompt_logprobs",
        "dynamic_eplb",
    ):
        setattr(runner, flag, False)
    runner.debugger = None
    runner.max_model_len = 16384
    runner.model_config = SimpleNamespace(is_hybrid=False, enable_return_routed_experts=False)
    scheduled = SimpleNamespace(
        scheduled_new_reqs=[],
        finished_req_ids=set(),
        new_block_ids_to_zero=[],
        scheduled_encoder_inputs={},
        free_encoder_mm_hashes=[],
        num_scheduled_tokens=dict.fromkeys(ids, 2),
        scheduled_spec_decode_tokens={rid: [-1] for rid in ids},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(ids), new_block_ids=[None] * n, num_computed_tokens=(bases + 2).tolist(), resumed_req_ids=set()
        ),
        kv_connector_metadata=SimpleNamespace(reason="eligible", frontiers=(4096,) * n, cold=(), requests=[]),
    )
    return runner, scheduled, shape, events, ns


def test_replay_precedes_real_count_readback_and_cpu_update(setup):
    runner, scheduled, shape, events, _ = setup
    original = runner.input_batch.num_computed_tokens_cpu.copy()
    assert runner._eligible(scheduled)
    runner._update_states(scheduled)
    runner._prepare_inputs(scheduled, np.array([2, 2, 2]))
    runner._apply_staged_sfa_route(runner._async_snapshot.key)
    metadata, common = runner._build_attention_metadata(**shape)
    assert common.seq_lens_cpu is None and metadata["l0"].seq_lens_cpu is None
    assert np.array_equal(runner.input_batch.num_computed_tokens_cpu, original)
    assert runner._model_forward() == "output"
    assert events == ["input_ids", "kernel", "rope", "replay", "sync", "state_update"]
    expected = original + np.array([1, 2, 1])
    assert runner.input_batch.num_computed_tokens_cpu.tolist() == expected.tolist()
    assert runner.positions.gpu[:6].tolist() == (np.repeat(expected, 2) + [0, 1, 0, 1, 0, 1]).tolist()
    assert metadata["l0"].cos[:6].tolist() == runner.positions.gpu[:6].tolist()
    assert metadata["l0"].seq_lens_cpu.tolist() == [5003, 5104, 5203, 0, 0]
    assert common.seq_lens_cpu.tolist() == [5003, 5104, 5203, 0]
    assert runner.async_mtp_replays == 1 and runner._async_pending is None
    for group, slots in zip(
        runner.input_batch.block_table.block_tables, (metadata["l0"].slot_mapping, metadata["l0"].indexer_slot_mapping)
    ):
        oracle = [
            int(group.table[r, p // group.block_size]) * group.block_size + p % group.block_size
            for r, base in enumerate(expected)
            for p in (base, base + 1)
        ]
        assert slots.tolist() == oracle + [-1, -1]


@pytest.mark.parametrize(
    "case",
    [
        "new",
        "finished",
        "resume",
        "zero",
        "blocks",
        "rollback",
        "frontier",
        "cold",
        "store",
        "order",
        "identity",
        "counts",
        "draft",
        "boundary",
        "limit",
    ],
)
def test_ineligible_transitions_read_real_counts_before_preparation(setup, case):
    r, s, _, events, _ = setup
    if case == "new":
        s.scheduled_new_reqs = [object()]
    elif case == "finished":
        s.finished_req_ids = {"r0"}
    elif case == "resume":
        s.scheduled_cached_reqs.resumed_req_ids = {"r0"}
    elif case == "zero":
        s.new_block_ids_to_zero = [1]
    elif case == "blocks":
        s.scheduled_cached_reqs.new_block_ids[0] = ([1], [2])
    elif case == "rollback":
        s.scheduled_cached_reqs.num_computed_tokens[0] -= 1
    elif case == "frontier":
        s.kv_connector_metadata.frontiers = (4000, 4096, 4096)
    elif case == "cold":
        s.kv_connector_metadata.cold = (True, False, False)
    elif case == "store":
        s.kv_connector_metadata.requests = [SimpleNamespace(is_decode_window_save=True)]
    elif case == "order":
        r.input_batch.req_ids.reverse()
    elif case == "identity":
        r.requests["r0"] = copy(r.requests["r0"])
    elif case == "counts":
        r._async_counts_epoch -= 1
    elif case == "draft":
        s.scheduled_spec_decode_tokens["r0"] = []
    elif case == "limit":
        r.max_model_len = 5203
    elif case == "boundary":
        r._async_snapshot.window = 4096
        r._async_snapshot.boundary = (0, 0, 0)
    assert not r._eligible(s)
    r._update_states(s)
    assert events[:2] == ["sync", "state_update"] and r._async_pending is None


def test_changed_graph_key_reconciles_without_submitting_metadata_kernel(setup):
    r, s, _, events, _ = setup
    r._update_states(s)
    r._prepare_inputs(s, np.array([2, 2, 2]))
    assert r._apply_staged_sfa_route(None) is None
    assert events == ["input_ids", "sync", "state_update", "normal_prepare"]
    assert r._async_pending is None and r._async_snapshot is None


def test_missing_new_count_never_reuses_previous_step_count(setup):
    r, s, shape, _, _ = setup
    r._update_states(s)
    r._prepare_inputs(s, np.array([2, 2, 2]))
    r._build_attention_metadata(**shape)
    r._model_forward()
    s.scheduled_cached_reqs.num_computed_tokens = (r._async_snapshot.bases + 2).tolist()
    assert not r._eligible(s)
    r._copy_valid_sampled_token_count(torch.ones(3, dtype=torch.int32), torch.tensor([2, 1, 2]))
    assert r._eligible(s)


def test_partial_forward_error_does_not_apply_cpu_update_or_retry(setup, monkeypatch):
    r, s, shape, events, ns = setup
    r._update_states(s)
    r._prepare_inputs(s, np.array([2, 2, 2]))
    r._build_attention_metadata(**shape)

    def fail(self):
        raise RuntimeError("partial submission")

    monkeypatch.setattr(ns["NPUModelRunner"], "_model_forward", fail)
    with pytest.raises(RuntimeError, match="partial submission"):
        r._model_forward()
    assert "sync" not in events and r._async_snapshot is None and r._async_pending is None


def test_boundary_shadow_invalidation_uses_normal_path(setup):
    r, scheduled, _, events, _ = setup
    r.item.decode_remap_boundary_buffer._values = None
    r._update_states(scheduled)
    assert events == ["sync", "state_update"] and r._async_pending is None


def test_lmhead_tp_keeps_baseline_logits_index_padding(setup):
    r, scheduled, _, _, ns = setup
    ns["lmhead_tp_enable"] = lambda: True
    r.max_num_reqs, r.uniform_decode_query_len = 16, 2
    r._update_states(scheduled)
    indices, spec, count = r._prepare_inputs(scheduled, np.array([2, 2, 2]))
    expected = torch.nn.functional.pad(spec.logits_indices, (0, 32 - len(spec.logits_indices)))
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)
    assert r.logits_indices is spec.logits_indices and count == 6


@pytest.mark.parametrize("field,value", [("uses_mrope", True), ("uses_xdrope_dim", 3)])
def test_nonstandard_positions_do_not_take_single_axis_kernel(setup, field, value):
    r, scheduled, _, events, _ = setup
    setattr(r, field, value)
    r._update_states(scheduled)
    assert r._async_pending is None and events == ["sync", "state_update"]


def test_non_graph_steps_do_not_copy_unused_device_state(setup):
    r, scheduled, shape, _, ns = setup
    r._async_snapshot = None
    r._async_bases.fill_(-77)
    ns["get_forward_context"]().staged_sfa_graph_key = None
    r._prepare_inputs(scheduled, np.array([2, 2, 2]))
    r._build_attention_metadata(**shape)
    r._model_forward()
    assert r._async_bases.tolist() == [-77, -77, -77]


def test_usable_graph_seeds_after_submission_before_next_count(setup):
    r, scheduled, shape, events, _ = setup
    r._async_snapshot = None
    r._update_states(scheduled)
    r._prepare_inputs(scheduled, np.array([2, 2, 2]))
    r._build_attention_metadata(**shape)
    base_pointer = r._async_bases.data_ptr()

    class ObserveSeed(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.copy_.default and args[0].data_ptr() == base_pointer:
                events.append("seed")
            return func(*args, **(kwargs or {}))

    with ObserveSeed():
        r._model_forward()
    assert events.index("replay") < events.index("seed")
    assert r._async_bases.tolist() == r._async_snapshot.bases.tolist()
    assert r._async_counts_epoch != r._async_epoch
    r._copy_valid_sampled_token_count(torch.ones(3, dtype=torch.int32), torch.tensor([2, 1, 2]))
    assert r._async_counts_epoch == r._async_epoch


@pytest.mark.parametrize("field", ["structured_outputs", "logits_processors", "min_tokens"])
def test_cpu_sampling_constraints_keep_original_preparation(setup, field):
    r, scheduled, _, events, _ = setup
    r.requests["r0"].sampling_params = SimpleNamespace(**{field: 1})
    r._update_states(scheduled)
    assert events == ["sync", "state_update"] and r._async_pending is None


@pytest.mark.parametrize("unsupported", ["none", "triton", "async", "mtp", "pp", "cp"])
def test_startup_contract_rejects_unsupported_modes(setup, unsupported):
    r, _, _, _, ns = setup
    r.vllm_config, r.device = object(), torch.device("cpu")
    r.max_num_reqs, r.use_async_scheduling, r._fixed_mtp_metadata = 3, True, ()
    r.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    if unsupported == "triton":
        ns["HAS_TRITON"] = False
    elif unsupported == "async":
        r.use_async_scheduling = False
    elif unsupported == "mtp":
        r._fixed_mtp_metadata = None
    elif unsupported == "pp":
        r.parallel_config.pipeline_parallel_size = 2
    elif unsupported == "cp":
        r.use_cp = True
    if unsupported == "none":
        r.__init__()
        assert r._async_bases.shape == (3,)
    else:
        with pytest.raises(ValueError, match="requires"):
            r.__init__()


def test_non_target_indexer_metadata_does_not_disable_async_entry(setup):
    r, s, shape, _, _ = setup
    r._staged_sfa_impls = (("l0", object()), ("l1", object()))
    r.normal_metadata[0]["indexer"] = None
    r._async_built = (r.normal_metadata, tuple(shape.values()))
    r._remember()
    assert r._async_snapshot is not None
    r._copy_valid_sampled_token_count(torch.ones(3, dtype=torch.int32), r._async_counts)
    r._update_states(s)
    r._prepare_inputs(s, np.array([2, 2, 2]))
    metadata, common = r._build_attention_metadata(**shape)
    assert metadata["indexer"] is None
    assert metadata["l0"] is metadata["l1"]
    r._model_forward()


def test_unpadded_draft_lengths_and_padded_target_lengths(setup):
    r, s, shape, events, _ = setup
    common = r._async_snapshot.common
    common.seq_lens = common.seq_lens[:3]
    common.seq_lens_cpu = common.seq_lens_cpu[:3]
    common.num_reqs = 3
    common.slot_mapping = common.slot_mapping[:6]
    common.indexer_slot_mapping = common.indexer_slot_mapping[:6]
    r._update_states(s)
    r._prepare_inputs(s, np.array([2, 2, 2]))
    metadata, common = r._build_attention_metadata(**shape)
    r._model_forward()
    assert common.seq_lens_cpu.tolist() == [5003, 5104, 5203]
    assert metadata["l0"].seq_lens_cpu.tolist() == [5003, 5104, 5203, 0, 0]
    assert events.index("replay") < events.index("sync")


def test_uniform_graph_cpu_lengths_may_alias_runner_buffer(setup):
    r, s, shape, _, ns = setup
    r.item.seq_lens = r.seq_lens.gpu
    r.item.seq_lens_cpu = r.seq_lens.cpu
    shape["full_graph"] = False
    r._async_snapshot.shape = ns["_shape"](shape)
    r._update_states(s)
    r._prepare_inputs(s, np.array([2, 2, 2]))
    metadata, common = r._build_attention_metadata(**shape)
    r._model_forward()
    assert metadata["l0"].seq_lens_cpu.tolist() == [5003, 5104, 5203, 0]
    assert common.seq_lens_cpu.tolist() == [5003, 5104, 5203, 0]


def test_kernel_warmup_is_after_capture_and_uses_private_outputs(setup, monkeypatch):
    r, _, _, events, _ = setup
    r.max_num_reqs, r.device = 3, torch.device("cpu")
    r._sfa_full_graph = SimpleNamespace(entries=[r._async_snapshot.key])
    r._async_bases = torch.zeros(4, dtype=torch.int32)
    r.max_num_reqs = 4
    r.drafter = SimpleNamespace(warmup_next_mtp_tokens=lambda: events.append("next_token_warmup"))
    for group in r.input_batch.block_table.block_tables:
        group.block_table = SimpleNamespace(gpu=group.table)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: events.append("startup_sync"))),
        raising=False,
    )
    before = tuple(t.clone() for t in (r.positions.gpu, r.seq_lens.gpu, r.item.slot_mapping))
    assert r.capture_model() == "captured"
    assert events == ["capture", "kernel", "next_token_warmup", "startup_sync"]
    for current, old in zip((r.positions.gpu, r.seq_lens.gpu, r.item.slot_mapping), before):
        torch.testing.assert_close(current, old, rtol=0, atol=0)


def test_deferred_update_matches_actual_vllm_request_bookkeeping(setup, monkeypatch):
    r, s, shape, events, ns = setup
    upstream = ROOT.parent / "vllm/vllm/v1/worker"
    update = extract(
        upstream / "gpu_model_runner.py", "_update_states", {"get_pp_group": lambda: SimpleNamespace(is_last_rank=True)}
    )
    update_spec = extract(upstream / "gpu_input_batch.py", "update_req_spec_token_ids", {})
    r.speculative_config = SimpleNamespace(use_ngram_gpu=lambda: False)
    r.use_async_scheduling = True
    r.late_interaction_runner = SimpleNamespace(on_requests_finished=lambda ids: None)
    r.num_prompt_logprobs, r.encoder_cache = {}, {}
    r._may_reorder_batch = lambda scheduled: None
    batch = r.input_batch
    batch.spec_token_ids = [[] for _ in batch.req_ids]
    batch.num_tokens_no_spec = batch.num_computed_tokens_cpu.copy() + 1
    batch.num_tokens = batch.num_tokens_no_spec.copy()
    batch.token_ids_cpu = np.zeros((3, 16384), dtype=np.int32)
    batch.condense = batch.refresh_metadata = lambda: None
    batch.update_req_spec_token_ids = lambda request, tokens: update_spec(batch, request, tokens)
    s.scheduled_cached_reqs.num_output_tokens = []
    for rid in batch.req_ids:
        req = r.requests[rid]
        req.req_id = rid
        req.output_token_ids = [-1] * (req.num_computed_tokens - 4096 + 1)
        s.scheduled_cached_reqs.num_output_tokens.append(len(req.output_token_ids) + 1)
    baseline = copy(r)
    baseline.requests = deepcopy(r.requests)
    baseline.input_batch = copy(batch)
    for name in ("num_tokens_no_spec", "num_tokens", "token_ids_cpu", "num_computed_tokens_cpu"):
        setattr(baseline.input_batch, name, getattr(batch, name).copy())
    baseline.input_batch.spec_token_ids = deepcopy(batch.spec_token_ids)
    baseline.input_batch.update_req_spec_token_ids = lambda request, tokens: update_spec(
        baseline.input_batch, request, tokens
    )
    update(baseline, s)
    events.clear()
    monkeypatch.setattr(ns["NPUModelRunner"], "_update_states", update)
    r._update_states(s)
    r._prepare_inputs(s, np.array([2, 2, 2]))
    r._build_attention_metadata(**shape)
    r._model_forward()
    assert events.index("replay") < events.index("sync")
    for name in ("num_tokens_no_spec", "num_tokens", "token_ids_cpu", "num_computed_tokens_cpu"):
        assert np.array_equal(getattr(batch, name), getattr(baseline.input_batch, name))
    assert batch.spec_token_ids == baseline.input_batch.spec_token_ids
    for rid in batch.req_ids:
        assert vars(r.requests[rid]) == vars(baseline.requests[rid])


def test_many_rejections_never_accumulate_position_drift(setup):
    r, s, shape, events, _ = setup
    expected = r._async_snapshot.bases.copy()
    for step in range(60):
        counts = torch.tensor([1 + step % 2, 1, 2], dtype=torch.int64)
        r._copy_valid_sampled_token_count(torch.ones(3, dtype=torch.int32), counts)
        s.scheduled_cached_reqs.num_computed_tokens = (expected + 2).tolist()
        events.clear()
        r._update_states(s)
        assert r._async_pending is s
        r._prepare_inputs(s, np.array([2, 2, 2]))
        r._build_attention_metadata(**shape)
        r._model_forward()
        expected += counts.numpy().astype(np.int32)
        assert np.array_equal(r.input_batch.num_computed_tokens_cpu, expected)
        assert r._async_bases.tolist() == expected.tolist()
        assert events.index("replay") < events.index("sync")
    assert r.async_mtp_replays == 60


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_selection_happens_only_at_worker_startup(monkeypatch, enabled):
    baseline, optimized = object(), object()
    module = SimpleNamespace(AsyncSFAModelRunner=lambda *args: optimized)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.sfa_async_mtp", module)
    ns = dict(
        envs_ascend=SimpleNamespace(VLLM_ASCEND_SFA_ASYNC_MTP_PREP=enabled),
        NPUModelRunner=lambda *args: baseline,
        init_workspace_manager=lambda *args: None,
    )
    initialize = extract(ROOT / "vllm_ascend/worker/worker.py", "init_device", ns)
    worker = SimpleNamespace(_init_device=lambda: "npu", use_v2_model_runner=False, vllm_config=object())
    initialize(worker)
    assert worker.model_runner is (optimized if enabled else baseline)
    if enabled:
        worker.use_v2_model_runner = True
        with pytest.raises(ValueError, match="V1"):
            initialize(worker)


def test_actual_npu_metadata_kernel_matches_cpu_slots_and_padding(monkeypatch):
    pytest.importorskip("torch_npu")
    pytest.importorskip("triton")
    if not torch.npu.is_available():
        pytest.skip("Requires NPU")
    spec = importlib.util.spec_from_file_location("async_mtp_metadata_npu", KERNEL)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    held = []
    for n, capacity in ((1, 8), (3, 8), (4, 8), (16, 32)):
        common_capacity = capacity // 2
        bases = torch.arange(n, dtype=torch.int32) * 127 + 4095
        actual_base = bases.to("npu")
        seq_common = torch.full((common_capacity,), -77, dtype=torch.int32, device="npu")
        seq_target = torch.full((common_capacity + 1,), -77, dtype=torch.int32, device="npu")
        positions = torch.full((capacity,), -77, dtype=torch.int64, device="npu")
        tables = [
            torch.arange(common_capacity * 128, dtype=torch.int32).view(common_capacity, 128) + shift
            for shift in (100, 900)
        ]
        device_tables = [t.to("npu") for t in tables]
        slots = [torch.full((capacity,), -77, dtype=torch.int32, device="npu") for _ in range(2)]
        for step in range(4):
            counts = (torch.arange(n) + step) % 2 + 1
            module.prepare_async_mtp_kernel[((common_capacity + 32) // 32,)](
                actual_base,
                counts.to("npu"),
                positions,
                seq_common,
                seq_target,
                *device_tables,
                *slots,
                n,
                capacity,
                common_capacity,
                common_capacity + 1,
                128,
                128,
                128,
                64,
                BLOCK=32,
            )
            bases += counts.int()
            expected_pos = torch.cat(
                ((bases[:, None].long() + torch.arange(2)).flatten(), torch.zeros(capacity - 2 * n, dtype=torch.int64))
            )
            expected_slots = []
            for table, block in zip(tables, (128, 64)):
                p = expected_pos[: 2 * n]
                slot = table[torch.arange(n).repeat_interleave(2), p // block] * block + p % block
                expected_slots.append(torch.cat((slot.int(), torch.full((capacity - 2 * n,), -1, dtype=torch.int32))))
            held.append(
                (
                    positions.clone(),
                    seq_target.clone(),
                    [t.clone() for t in slots],
                    expected_pos,
                    bases.clone() + 2,
                    expected_slots,
                )
            )
    torch.npu.synchronize()
    for positions, lengths, slots, expected_pos, expected_len, expected_slots in held:
        torch.testing.assert_close(positions.cpu(), expected_pos, rtol=0, atol=0)
        assert torch.equal(lengths[: len(expected_len)].cpu(), expected_len)
        assert torch.all(lengths[len(expected_len) :] == 0)
        for actual, expected in zip(slots, expected_slots):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
