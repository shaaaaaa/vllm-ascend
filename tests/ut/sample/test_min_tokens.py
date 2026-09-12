# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU semantics and deferred-DMA contracts; no NPU performance claims."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def modules(monkeypatch):
    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    diagnostics = load("vllm_ascend.sample.rejection_diagnostics", "vllm_ascend/sample/rejection_diagnostics.py")
    masks = load("tested_ascend_min_tokens", "vllm_ascend/sample/min_tokens.py")
    return masks, diagnostics


def processor(min_toks):
    return SimpleNamespace(min_toks=min_toks, neg_inf_tensor=torch.tensor(-float("inf")))


def expected_mask(logits, min_toks, drafts):
    result = logits.clone()
    offset = 0
    for req, count in enumerate(drafts):
        if req in min_toks:
            minimum, output, stops = min_toks[req]
            for draft in range(count):
                if len(output) + draft < minimum:
                    for token in stops:
                        result[offset + draft, token] = -float("inf")
        offset += count
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "drafts,min_toks",
    [
        ([1], {}),
        ([1], {0: (5, [], set())}),
        ([1], {0: (0, [], {2})}),
        ([2], {0: (2, [5, 6], {1})}),
        ([0], {0: (3, [], {1})}),
        ([3], {0: (3, [5], {1, 4})}),
        ([2, 3, 1], {0: (5, [6], {0, 4}), 1: (2, [3], {1}), 2: (1, [], {2})}),
        ([0, 2, 1], {0: (3, [], {0}), 1: (2, [1], {2}), 2: (8, [], {0, 6})}),
        ([2, 0, 2], {2: (1, [], {2}), 0: (2, [], {0, 1})}),
    ],
)
def test_mask_semantics_and_inplace_result(modules, dtype, drafts, min_toks):
    masks, _ = modules
    # RejectionSampler converts target logits to FP32 before applying these
    # processors (including its existing FP32 neg_inf_tensor).
    logits = torch.arange(sum(drafts) * 7, dtype=dtype).reshape(sum(drafts), 7).to(torch.float32)
    expected = expected_mask(logits, min_toks, drafts)
    actual = masks.apply_with_spec_decode(processor(min_toks), logits, drafts)
    assert actual is logits
    assert torch.equal(actual, expected)


def test_random_multirequest_masks(modules):
    masks, _ = modules
    rng = np.random.default_rng(15)
    owner = processor({})
    for _ in range(150):
        drafts = rng.integers(0, 7, size=int(rng.integers(1, 9))).tolist()
        owner.min_toks = {
            req: (
                int(rng.integers(0, 12)),
                [0] * int(rng.integers(0, 12)),
                set(rng.choice(17, size=int(rng.integers(0, 6)), replace=False).tolist()),
            )
            for req in rng.permutation(len(drafts)).tolist()
        }
        # Noncontiguous logits are supported by the original index_put_.
        logits = torch.arange(sum(drafts) * 34, dtype=torch.float32).reshape(sum(drafts), 34)[:, ::2]
        expected = expected_mask(logits, owner.min_toks, drafts)
        assert torch.equal(masks.apply_with_spec_decode(owner, logits, drafts), expected)


def test_unchanged_mask_reuses_both_buffers_as_history_grows(modules, monkeypatch):
    masks, _ = modules
    history = []
    owner = processor({0: (512, history, {1, 3})})
    masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
    cached = owner._ascend_spec_mask_indices
    monkeypatch.setattr(masks, "_allocate_indices", Mock(side_effect=AssertionError("unexpected reallocation")))
    monkeypatch.setattr(masks, "_fill_indices", Mock(side_effect=AssertionError("unexpected host write")))
    for _ in range(50):
        history.append(0)
        result = masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
        assert owner._ascend_spec_mask_indices is cached
        assert torch.isneginf(result[0, [1, 3]]).all()
    assert cached.host.shape == cached.indices.shape == (2, 2)


def test_boundary_stop_ids_draft_lengths_and_request_moves_invalidate_cache(modules):
    masks, _ = modules
    owner = processor({0: (4, [], {1})})
    steps = [
        ([3, 2], {0: (4, [], {1})}),
        ([3, 2], {0: (4, [8, 8], {1})}),  # Mask shrinks to two positions.
        ([3, 2], {0: (4, [8, 8], {2})}),  # Same shape, different stop token.
        ([3, 2], {1: (4, [8, 8], {2})}),  # Request moves to another row range.
        ([1, 2], {1: (4, [8, 8], {2})}),  # Earlier draft length changes offset.
        ([1, 2], {1: (4, [8, 8, 8], {2})}),
        ([1, 2], {}),  # Request removed / minimum reached.
        ([1, 2], {0: (6, [], {5})}),  # New request in the reused slot.
    ]
    previous = None
    for drafts, state in steps:
        owner.min_toks = state
        logits = torch.zeros(sum(drafts), 7)
        expected = expected_mask(logits, state, drafts)
        assert torch.equal(masks.apply_with_spec_decode(owner, logits, drafts), expected)
        cached = owner._ascend_spec_mask_indices
        assert cached is not previous
        previous = cached
    owner.min_toks = {0: (1, [0], {5})}
    masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
    assert owner._ascend_spec_mask_indices is None


class DeferredNPU:
    """Emulate async H2D and masks: no device work runs until drain()."""

    def __init__(self):
        self.device = SimpleNamespace(type="npu", index=0)
        self.stream = SimpleNamespace(npu_stream=1)
        self.pending = []
        self.host_allocations = []
        self.uploads = 0
        self.torch = SimpleNamespace(
            int64=torch.int64,
            empty=self.empty,
            npu=SimpleNamespace(current_stream=lambda device: self.stream),
        )

    def empty(self, shape, *, dtype, device, pin_memory=None):
        if device == "cpu":
            assert pin_memory is True  # Regression: pageable H2D is forbidden.
            host = torch.empty(shape, dtype=dtype)
            self.host_allocations.append(host)
            return host
        assert device == self.device
        return self.Indices(self, shape, dtype)

    class Indices:
        def __init__(self, backend, shape, dtype):
            self.backend = backend
            self.data = torch.empty(shape, dtype=dtype)

        def copy_(self, host, *, non_blocking):
            assert non_blocking is True
            assert any(host is allocation for allocation in self.backend.host_allocations)
            self.backend.uploads += 1
            # Keep the original host tensor, NOT a snapshot: early overwrite
            # before DMA runs would corrupt the resulting mask in this test.
            self.backend.pending.append(lambda: self.data.copy_(host))
            return self

        def __getitem__(self, row):
            return self, row

    class Logits:
        def __init__(self, backend, rows):
            self.backend = backend
            self.device = backend.device
            self.data = torch.zeros(rows, 7)

        def index_put_(self, indices, value):
            def execute():
                actual = tuple(owner.data[row] for owner, row in indices)
                self.data.index_put_(actual, value)

            self.backend.pending.append(execute)
            return self

    def drain(self):
        for operation in self.pending:
            operation()
        self.pending.clear()


def test_pending_upload_is_never_overwritten_or_waited_on(modules, monkeypatch):
    masks, _ = modules
    backend = DeferredNPU()
    monkeypatch.setattr(masks, "torch", backend.torch)
    owner = processor({0: (4, [], {1})})
    first = backend.Logits(backend, 2)
    masks.apply_with_spec_decode(owner, first, [2])
    old_cache = owner._ascend_spec_mask_indices
    old_host = old_cache.host.clone()
    owner.min_toks = {0: (4, [0, 0, 0], {2})}
    second, third = backend.Logits(backend, 2), backend.Logits(backend, 2)
    masks.apply_with_spec_decode(owner, second, [2])
    masks.apply_with_spec_decode(owner, third, [2])
    assert owner._ascend_spec_mask_indices is not old_cache
    assert torch.equal(old_cache.host, old_host)
    assert backend.uploads == 2  # Third call is a cache hit with no H2D.
    assert torch.count_nonzero(first.data) == 0  # No implicit completion.
    backend.drain()
    assert torch.isneginf(first.data[:, 1]).all()
    assert torch.isneginf(second.data[0, 2]) and second.data[1, 2] == 0
    assert torch.equal(second.data, third.data)


@pytest.mark.parametrize("change", ["stream", "device"])
def test_cache_is_not_consumed_on_another_stream_or_device(modules, monkeypatch, change):
    masks, _ = modules
    backend = DeferredNPU()
    monkeypatch.setattr(masks, "torch", backend.torch)
    owner = processor({0: (4, [], {1})})
    masks.apply_with_spec_decode(owner, backend.Logits(backend, 1), [1])
    previous = owner._ascend_spec_mask_indices
    if change == "stream":
        backend.stream = SimpleNamespace(npu_stream=2)
    else:
        backend.device = SimpleNamespace(type="npu", index=1)
    masks.apply_with_spec_decode(owner, backend.Logits(backend, 1), [1])
    assert owner._ascend_spec_mask_indices is not previous
    assert backend.uploads == 2
    backend.drain()


def test_stages_split_upload_and_mask_and_leave_normal_path_untimed(modules):
    masks, diagnostics = modules
    stages = []

    def record(name, operation, args, kwargs):
        stages.append(name)
        return operation(*args, **kwargs)

    owner = processor({0: (4, [], {1})})
    token = diagnostics.set_stage_recorder(record)
    try:
        masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
        assert stages == [
            "min_tokens.layout",
            "min_tokens.allocate",
            "min_tokens.fill",
            "min_tokens.h2d",
            "min_tokens.mask",
        ]
        stages.clear()
        masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
        assert stages == ["min_tokens.layout", "min_tokens.mask"]
    finally:
        diagnostics.reset_stage_recorder(token)
    stages.clear()
    masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
    assert not stages


def test_failed_upload_is_propagated_without_publishing_bad_cache(modules, monkeypatch):
    masks, _ = modules
    backend = DeferredNPU()
    monkeypatch.setattr(masks, "torch", backend.torch)
    owner = processor({0: (4, [], {1})})
    masks.apply_with_spec_decode(owner, backend.Logits(backend, 1), [1])
    previous = owner._ascend_spec_mask_indices
    owner.min_toks = {0: (4, [], {2})}
    monkeypatch.setattr(backend.Indices, "copy_", Mock(side_effect=RuntimeError("copy failed")))
    with pytest.raises(RuntimeError, match="copy failed"):
        masks.apply_with_spec_decode(owner, backend.Logits(backend, 1), [1])
    assert owner._ascend_spec_mask_indices is previous


@pytest.fixture
def sibling_processor():
    """Load the actual baseline class without importing the accelerator stack."""
    path = ROOT.parent / "vllm/vllm/v1/sample/logits_processor/builtin.py"
    if not path.is_file():
        pytest.skip("Requires matching sibling vllm checkout")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module = ast.parse("from __future__ import annotations")
    module.body.extend(
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("MinTokensLogitsProcessor", "process_dict_updates")
    )
    namespace = dict(torch=torch, np=np, LogitsProcessor=object, MoveDirectionality=SimpleNamespace(SWAP="swap"))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["MinTokensLogitsProcessor"]


def test_actual_sibling_baseline_batch_updates_and_masks_match(modules, sibling_processor):
    masks, _ = modules
    baseline = sibling_processor(None, torch.device("cpu"), False)
    patched = sibling_processor(None, torch.device("cpu"), False)
    history_a, history_b = [], [5]
    params_a = SimpleNamespace(min_tokens=3, all_stop_token_ids={1, 2})
    params_b = SimpleNamespace(min_tokens=5, all_stop_token_ids={3})

    def step(update, drafts):
        for owner in (baseline, patched):
            owner.update_state(update)
        raw = torch.arange(sum(drafts) * 7, dtype=torch.float32).reshape(sum(drafts), 7)
        expected = baseline.apply_with_spec_decode(raw.clone(), drafts)
        actual = masks.apply_with_spec_decode(patched, raw.clone(), drafts)
        assert torch.equal(actual, expected)

    def update(*, added=(), removed=(), moved=()):
        return SimpleNamespace(added=added, removed=removed, moved=moved)

    step(update(added=[(0, params_a, [], history_a), (1, params_b, [], history_b)]), [2, 3])
    history_a.extend([6, 6])
    step(None, [2, 3])
    step(update(moved=[(0, 1, "swap")]), [3, 2])
    history_a.append(6)  # The normal update_state removes the attained minimum.
    step(None, [1, 2])
    step(update(removed=[1], moved=[(0, 1, "move")]), [0, 2])
    step(update(removed=[1]), [0, 0])
    step(update(added=[(0, params_b, [], [])]), [1])


def test_real_worker_patch_installs_only_spec_decode_method(modules, sibling_processor, monkeypatch):
    masks, _ = modules
    # Execute the real worker patch against an isolated vllm package hierarchy.
    for name in ("vllm", "vllm.v1", "vllm.v1.sample", "vllm.v1.sample.logits_processor"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    baseline_module = ModuleType("vllm.v1.sample.rejection_sampler")
    builtin = ModuleType("vllm.v1.sample.logits_processor.builtin")
    builtin.MinTokensLogitsProcessor = sibling_processor
    replacement = ModuleType("vllm_ascend.sample.rejection_sampler")
    for name in ("apply_sampling_constraints", "expand_batch_to_tokens", "rejection_sample"):
        setattr(replacement, name, object())
    monkeypatch.setitem(sys.modules, baseline_module.__name__, baseline_module)
    monkeypatch.setitem(sys.modules, builtin.__name__, builtin)
    monkeypatch.setitem(sys.modules, replacement.__name__, replacement)
    monkeypatch.setitem(sys.modules, "vllm_ascend.sample.min_tokens", masks)
    original_apply, original_update = sibling_processor.apply, sibling_processor.update_state
    path = ROOT / "vllm_ascend/patch/worker/patch_rejection_sampler.py"
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {})
    assert sibling_processor.apply_with_spec_decode is masks.apply_with_spec_decode
    assert sibling_processor.apply is original_apply
    assert sibling_processor.update_state is original_update
    assert baseline_module.rejection_sample is replacement.rejection_sample


def test_no_new_completion_waits_or_device_readbacks():
    source = (ROOT / "vllm_ascend/sample/min_tokens.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"item", "tolist", "cpu", "synchronize", "query", "Event", "wait_event", "wait_stream"}
    assert not [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden
    ]
