# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise host-source fences and cached layouts across real runner overrides."""

import ast
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sfa_test_support import ROOT, extract, load_module
from test_sfa_async_mtp import setup  # noqa: F401
from test_sfa_full_graph_routing import routing  # noqa: F401


@pytest.fixture
def staging(request, monkeypatch):
    r, scheduled, shape, events, ns = request.getfixturevalue("setup")
    r.prepare_inputs_event = SimpleNamespace(
        synchronize=lambda: events.append("staging_wait"), record=lambda: events.append("staging_record")
    )
    source = ROOT.parent / "vllm/vllm/v1/worker/gpu_model_runner.py"
    guard = extract(source, "synchronize_input_prep", {})
    monkeypatch.setattr(ns["NPUModelRunner"], "synchronize_input_prep", contextmanager(guard), raising=False)
    padding = extract(
        ROOT / "vllm_ascend/worker/model_runner_v1.py",
        "_pad_query_start_loc_for_fia",
        {"CUDAGraphMode": SimpleNamespace(FULL="full")},
    )
    monkeypatch.setattr(ns["NPUModelRunner"], "_pad_query_start_loc_for_fia", padding, raising=False)
    cpu = np.arange(18, dtype=np.int32) * 2
    gpu = cpu.copy()
    r.query_start_loc = SimpleNamespace(np=cpu, gpu=gpu)

    def upload():
        events.append("query_upload")
        gpu[:] = cpu

    r.query_start_loc.copy_to_gpu = upload
    r.arange_np = np.arange(32)
    r.uniform_decode_query_len = 2
    r.compilation_config = SimpleNamespace(cudagraph_mode="piecewise")
    return r, scheduled, shape, events, ns


@pytest.mark.parametrize("bounded", [False, True])
def test_fast_steps_reuse_layout_without_wait_or_record(staging, monkeypatch, bounded):
    r, s, shape, events, ns = staging
    args = (8, 4, 3, "piecewise", 4)
    # First normal preparation owns the host upload and establishes the fence.
    with r.synchronize_input_prep():
        expected = r._pad_query_start_loc_for_fia(*args, full_graph=bounded)
    before = r.query_start_loc.gpu.copy()
    assert events == ["staging_wait", "query_upload", "staging_record"]

    def execute(self, scheduled):
        with self.synchronize_input_prep():
            self._update_states(scheduled)
            assert self._async_pending is scheduled
            self._prepare_inputs(scheduled, np.full(3, 2, dtype=np.int32))
            assert self._pad_query_start_loc_for_fia(*args, full_graph=bounded) == expected
            self._build_attention_metadata(**shape)
        return self._model_forward()

    monkeypatch.setattr(ns["NPUModelRunner"], "execute_model", execute, raising=False)
    for _ in range(3):
        events.clear()
        r._copy_valid_sampled_token_count(torch.ones(3, dtype=torch.int32), torch.tensor([1, 2, 1]))
        s.scheduled_cached_reqs.num_computed_tokens = (r.input_batch.num_computed_tokens_cpu + 2).tolist()
        assert r.execute_model(s) == "output"
        assert "staging_wait" not in events and "staging_record" not in events and "query_upload" not in events
        assert events.index("replay") < events.index("sync") < events.index("state_update")
        assert np.array_equal(before, r.query_start_loc.gpu)
        assert not r._async_live_execute and r._async_host_write is None


@pytest.mark.parametrize("failure", [False, True])
def test_transition_waits_before_host_preparation_and_records_on_exception(staging, monkeypatch, failure):
    r, s, _, events, ns = staging
    s.scheduled_spec_decode_tokens["r0"] = []

    def execute(self, scheduled):
        with self.synchronize_input_prep():
            self._update_states(scheduled)
            self._prepare_inputs(scheduled, np.full(3, 2))
            self._ensure_host_staging_ready()
            events.append("host_write")
            if failure:
                raise RuntimeError("failed host prep")

    monkeypatch.setattr(ns["NPUModelRunner"], "execute_model", execute, raising=False)
    if failure:
        with pytest.raises(RuntimeError, match="host prep"):
            r.execute_model(s)
    else:
        r.execute_model(s)
    assert events == ["staging_wait", "sync", "state_update", "normal_prepare", "host_write", "staging_record"]
    assert not r._async_live_execute and r._async_host_write is None
    assert r._async_snapshot is None and r._async_query_layout is None


def test_route_cancellation_waits_before_reconciliation(staging):
    r, s, _, events, _ = staging
    r._async_live_execute = True
    with r.synchronize_input_prep():
        r._update_states(s)
        r._apply_staged_sfa_route(None)
    assert events == ["staging_wait", "sync", "state_update", "normal_prepare", "staging_record"]
    assert r._async_pending is None


def test_padding_miss_waits_once_and_invalidates_old_signature(staging):
    r, s, _, events, _ = staging
    r._async_live_execute = True
    with r.synchronize_input_prep():
        r._update_states(s)
        r._pad_query_start_loc_for_fia(8, 4, 3, "piecewise", 4)
        r._pad_query_start_loc_for_fia(16, 8, 3, "piecewise", 8)
        r._pad_query_start_loc_for_fia(16, 8, 3, "piecewise", 8)
    assert events == ["staging_wait", "query_upload", "query_upload", "staging_record"]
    assert r.query_start_loc.gpu[:9].tolist() == list(range(0, 18, 2))


def test_cached_padding_protects_callers_host_dummy_row_adjustment(staging):
    r, s, _, events, _ = staging
    args = (8, 4, 4, "piecewise", 8, True)
    with r.synchronize_input_prep():
        assert r._pad_query_start_loc_for_fia(*args) == 8
    events.clear()
    r._async_live_execute, r._async_pending = True, s
    with r.synchronize_input_prep():
        assert r._pad_query_start_loc_for_fia(*args) == 8
        events.append("caller_host_write")
        r.query_start_loc.np[5] = 0
    assert events == ["staging_wait", "caller_host_write", "staging_record"]


def test_failed_exit_event_clears_pending_state(staging, monkeypatch):
    r, s, _, events, ns = staging

    def fail_record():
        raise RuntimeError("record failed")

    r.prepare_inputs_event.record = fail_record

    def execute(self, scheduled):
        with self.synchronize_input_prep():
            self._async_pending = scheduled
            self._ensure_host_staging_ready()

    monkeypatch.setattr(ns["NPUModelRunner"], "execute_model", execute, raising=False)
    with pytest.raises(RuntimeError, match="record failed"):
        r.execute_model(s)
    assert events == ["staging_wait"]
    assert r._async_pending is None and r._async_snapshot is None and r._async_query_layout is None
    assert r._async_host_write is None and not r._async_live_execute


@pytest.fixture
def real_dummy(staging, request, monkeypatch):
    r, _, _, events, ns = staging
    route_runner, route_ns, _, modes, _ = request.getfixturevalue("routing")
    namespace = dict(
        route_ns, torch=torch, cdiv=lambda n, d: (n + d - 1) // d,
        SEQ_LEN_WITH_MAX_PA_WORKSPACE=512, update_cos_sin=lambda positions: None,
        get_pp_group=lambda: SimpleNamespace(is_first_rank=True),
        set_ascend_forward_context=lambda *args, **kwargs: nullcontext(),
        lmhead_tp_enable=lambda: False,
    )
    method = extract(ROOT / "vllm_ascend/worker/model_runner_v1.py", "_dummy_run", namespace)
    monkeypatch.setattr(ns["NPUModelRunner"], "_dummy_run", method, raising=False)
    r.vllm_config = route_runner.vllm_config
    r.vllm_config.model_config.use_mla = True
    r.speculative_config = SimpleNamespace(method="mtp")
    r.lora_config = None
    r.scheduler_config = SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=16)
    r.max_num_tokens, r.dp_size, r.decode_threshold = 4096, 4, 2
    r._staged_sfa_graph_capture_sizes = (8, 16, 24, 32)
    r._staged_sfa_dp_route_action = route_ns["StagedSFARouteAction"].STAGED
    r._determine_batch_execution_and_padding = lambda **kw: (
        modes.PIECEWISE, SimpleNamespace(num_tokens=8, num_reqs=4), False, None, None
    )
    r._staged_sfa_dummy_batch_size = lambda **kw: 8
    r._staged_sfa_dummy_graph_key = lambda *args, **kw: route_ns["StagedSFAGraphKey"].bounded_decode(4, 2)
    r._should_build_dummy_attn_metadata = lambda *args: True
    r._staged_sfa_dummy_seq_len = lambda **kw: 512
    r._staged_sfa_query_start_locs = lambda n, query_width, dtype: np.arange(n + 1, dtype=dtype) * query_width
    r.maybe_dummy_run_with_lora = lambda *args, **kwargs: nullcontext()
    r.use_aux_hidden_state_outputs = False
    r.model = object()
    r._staged_sfa_impls = [("l0", SimpleNamespace(bootstrap_cross_layer=lambda name: events.append("bootstrap")))]
    r.drafter = SimpleNamespace(dummy_run=lambda **kw: events.append("draft"))
    r._model_forward = lambda *args: (events.append("forward"), torch.zeros(8, 4))[1]
    r.seq_lens.copy_to_gpu = lambda: events.append("sequence_upload")
    return r, events


@pytest.mark.parametrize("event_enabled", [False, True])
@pytest.mark.parametrize("failure", [None, "upload", "forward"])
def test_real_dummy_fences_uploads_before_compute(real_dummy, event_enabled, failure):
    r, events = real_dummy
    r._async_query_layout = (((), ()), 4)
    r._async_live_execute = True
    if not event_enabled:
        r.prepare_inputs_event = None

    class GuardedHostArray(np.ndarray):
        def __setitem__(self, index, value):
            if event_enabled:
                assert events and events[0] == "staging_wait"
            super().__setitem__(index, value)

    r.seq_lens.np = r.seq_lens.np.view(GuardedHostArray)

    def upload():
        assert r.seq_lens.np.tolist() == [512] * 4
        if event_enabled:
            assert events[0] == "staging_wait"
        events.append("sequence_upload")
        if failure == "upload":
            raise RuntimeError("upload failed")

    def forward(*args):
        events.append("forward")
        if failure == "forward":
            raise RuntimeError("forward failed")
        return torch.zeros(8, 4)

    r.seq_lens.copy_to_gpu, r._model_forward = upload, forward
    if failure:
        with pytest.raises(RuntimeError, match=failure + " failed"):
            r._dummy_run(2, uniform_decode=True)
    else:
        r._dummy_run(2, uniform_decode=True)
        assert events[-1] == "draft"
    if event_enabled:
        assert events.count("staging_wait") == events.count("staging_record") == 1
        assert events.index("sequence_upload") < events.index("staging_record")
        if failure != "upload":
            assert events.index("query_upload") < events.index("staging_record") < events.index("forward")
    else:
        assert "staging_wait" not in events and "staging_record" not in events
    assert r._async_live_execute and r._async_query_layout is None and r._async_snapshot is None


@pytest.mark.parametrize("kind", ["sample_none", "sample_shape", "sample_dtype", "draft_dtype", "draft_device"])
def test_inputs_that_could_take_host_fallback_are_not_eligible(staging, kind):
    r, s, _, _, _ = staging
    if kind == "sample_none":
        r.input_batch.prev_sampled_token_ids = None
    elif kind == "sample_shape":
        r.input_batch.prev_sampled_token_ids = torch.zeros(2, 1, dtype=torch.int32)
    elif kind == "sample_dtype":
        r.input_batch.prev_sampled_token_ids = torch.zeros(3, 1, dtype=torch.int64)
    elif kind == "draft_dtype":
        r._draft_token_ids = torch.zeros(3, 1)
    else:
        r._draft_token_ids = torch.empty(3, 1, dtype=torch.int64, device="meta")
    assert not r._eligible(s)


def test_real_ascend_update_and_device_interleave_remain_in_the_mro(staging, monkeypatch):
    r, s, shape, events, ns = staging
    source = ast.parse((ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf8"))
    ascend = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    tree = ast.parse("from __future__ import annotations\nclass Ascend(GPU): pass")
    tree.body[1].body = [
        n
        for n in ascend.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in ("_update_states", "_prepare_input_ids", "_prepare_fixed_mtp_input_ids", "_fixed_spec_decode_metadata")
    ]
    ns["SpecDecodeMetadata"] = load_module(
        ROOT.parent / "vllm/vllm/v1/spec_decode/metadata.py", "mtp_staging_metadata", monkeypatch
    ).SpecDecodeMetadata
    ns["GPU"] = ns["NPUModelRunner"]
    exec(compile(ast.fix_missing_locations(tree), "ascend_dispatch", "exec"), ns)
    ns["NPUModelRunner"] = ns["Ascend"]
    source = ast.parse((ROOT / "vllm_ascend/worker/sfa_async_mtp.py").read_text(encoding="utf8"))
    source.body = [n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "AsyncSFAModelRunner"]
    exec(compile(ast.fix_missing_locations(source), "async_dispatch", "exec"), ns)
    actual = ns["AsyncSFAModelRunner"].__new__(ns["AsyncSFAModelRunner"])
    actual.__dict__.update(vars(r))
    actual._fixed_mtp_metadata = extract(
        ROOT / "vllm_ascend/worker/model_runner_v1.py", "_fixed_mtp_metadata_arrays", {"torch": torch}
    )(3, "cpu")
    actual.use_async_scheduling = True
    actual.input_batch.prev_sampled_token_ids = torch.tensor([[11], [22], [33]], dtype=torch.int32)
    actual._draft_token_ids = torch.tensor([[101], [202], [303]], dtype=torch.int64)
    actual._sfa_full_graph = SimpleNamespace(release_requests=lambda ids: events.append("ascend_graph_release"))
    actual._resident_state_registry = SimpleNamespace(release=lambda ids: events.append("ascend_resident_release"))
    actual._async_live_execute = True
    with actual.synchronize_input_prep():
        actual._update_states(s)
        logits, spec, count = actual._prepare_inputs(s, np.full(3, 2, dtype=np.int32))
        actual._build_attention_metadata(**shape)
    assert actual.input_ids.gpu[:6].tolist() == [11, 101, 22, 202, 33, 303]
    assert spec.draft_token_ids.tolist() == [101, 202, 303]
    assert spec.num_draft_tokens == [1, 1, 1] and count == 6
    assert logits.tolist() == list(range(6))
    assert logits.dtype == torch.int64
    assert "input_ids" not in events  # The GPU parent's fallback stub was not called.
    assert "staging_wait" not in events and "staging_record" not in events
    actual._model_forward()
    assert events.index("replay") < events.index("ascend_graph_release") < events.index("ascend_resident_release")
    assert events.index("ascend_resident_release") < events.index("sync") < events.index("state_update")


def test_ordinary_preparation_invalidates_same_shaped_query_layout(staging):
    r, s, _, events, _ = staging
    args = (8, 4, 3, "piecewise", 4)
    with r.synchronize_input_prep():
        r._pad_query_start_loc_for_fia(*args)
    r._prepare_inputs(s, np.full(3, 2))
    assert r._async_query_layout is None
    events.clear()
    r._async_live_execute = True
    with r.synchronize_input_prep():
        r._update_states(s)
        r._pad_query_start_loc_for_fia(*args)
    assert events == ["staging_wait", "query_upload", "staging_record"]
