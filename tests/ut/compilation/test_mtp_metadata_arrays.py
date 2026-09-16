# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare immutable MTP layouts against the unchanged general calculation."""

import ast
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from sfa_test_support import AsyncMTPTokenKernel, extract
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]
FIELDS = (
    "draft_token_ids",
    "cu_num_draft_tokens",
    "cu_num_sampled_tokens",
    "target_logits_indices",
    "bonus_logits_indices",
    "logits_indices",
)


@pytest.fixture(scope="module")
def methods():
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    metadata_path = ROOT.parent / "vllm/vllm/v1/spec_decode/metadata.py"
    if not metadata_path.is_file():
        metadata_path = importlib.metadata.distribution("vllm").locate_file("vllm/v1/spec_decode/metadata.py")
    spec = importlib.util.spec_from_file_location("mtp_metadata_contract", metadata_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cls = next(n for n in tree.body if getattr(n, "name", "") == "NPUModelRunner")
    calc = next(n for n in cls.body if getattr(n, "name", "") == "_calc_spec_decode_metadata")
    factory = next(n for n in tree.body if getattr(n, "name", "") == "_fixed_mtp_metadata_arrays")
    init = next(n for n in cls.body if getattr(n, "name", "") == "__init__")
    assignment = next(
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_fixed_mtp_metadata" for t in n.targets)
    )
    code = ast.parse("from __future__ import annotations")
    code.body.extend(
        [factory, calc, next(n for n in cls.body if getattr(n, "name", "") == "_fixed_spec_decode_metadata")]
    )
    ns = dict(torch=torch, np=np, SpecDecodeMetadata=module.SpecDecodeMetadata)
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), ns)
    initialize = compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec")
    return ns, initialize


def runner(methods, *, method="mtp", width=2, cp=False, capacity=16):
    ns, initialize = methods
    subject = SimpleNamespace(
        speculative_config=SimpleNamespace(method=method) if method else None,
        decode_threshold=width,
        use_cp=cp,
        pcp_size=2 if cp else 1,
        max_num_reqs=capacity,
        device=torch.device("cpu"),
        input_ids=SimpleNamespace(gpu=torch.arange(512, dtype=torch.int32)),
        arange_np=np.arange(512, dtype=np.int64),
        _fixed_decode_cu_num_tokens=np.arange(1, capacity + 1, dtype=np.int64) * width,
    )
    subject._fixed_spec_decode_metadata = MethodType(ns["_fixed_spec_decode_metadata"], subject)
    exec(initialize, dict(ns, self=subject))
    return subject


@pytest.fixture
def pin_calls(monkeypatch):
    calls = []

    # CPU-only reference execution; record calls rather than emulate NPU pinning.
    def pin(tensor, *args, **kwargs):
        calls.append(tensor.numel())
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    return calls


def compare(actual, expected):
    assert actual.num_draft_tokens == expected.num_draft_tokens
    assert actual.max_spec_len == expected.max_spec_len
    for name in FIELDS:
        a, b = getattr(actual, name), getattr(expected, name)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.is_contiguous() and a.stride() == b.stride()


@pytest.mark.parametrize("requests", [1, 2, 3, 4, 7, 8, 15, 16])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_fixed_layout_exactly_matches_general_path(methods, pin_calls, requests, dtype):
    ns, _ = methods
    actual, baseline = runner(methods), runner(methods)
    baseline._fixed_mtp_metadata = None
    drafts = np.ones(requests, dtype=np.int32)
    scheduled = np.arange(2, 2 * requests + 1, 2, dtype=dtype)
    expected = ns["_calc_spec_decode_metadata"](baseline, drafts, scheduled, None)
    assert len(pin_calls) == 5
    pin_calls.clear()
    result = ns["_calc_spec_decode_metadata"](actual, drafts, scheduled, None)
    assert pin_calls == []
    compare(result, expected)


def test_only_fresh_token_gather_runs_on_repeated_layout(methods, pin_calls):
    ns, _ = methods
    subject = runner(methods)
    arrays = subject._fixed_mtp_metadata
    addresses = tuple(t.data_ptr() for t in arrays)
    versions = tuple(t._version for t in arrays)
    ops = []

    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            ops.append(func)
            return func(*args, **(kwargs or {}))

    drafts, scheduled = np.ones(4, dtype=np.int32), np.arange(2, 9, 2, dtype=np.int64)
    old = ns["_calc_spec_decode_metadata"](subject, drafts, scheduled, None)
    subject.input_ids.gpu.add_(100)
    with Record():
        new = ns["_calc_spec_decode_metadata"](subject, drafts, scheduled, None)
    assert ops.count(torch.ops.aten.index.Tensor) == 1
    assert set(ops) <= {torch.ops.aten.index.Tensor, torch.ops.aten.slice.Tensor}
    assert old.draft_token_ids.tolist() == [1, 3, 5, 7]
    assert new.draft_token_ids.tolist() == [101, 103, 105, 107]
    assert old.draft_token_ids.data_ptr() != new.draft_token_ids.data_ptr()
    assert addresses == tuple(t.data_ptr() for t in arrays)
    assert versions == tuple(t._version for t in arrays)
    assert not pin_calls


@pytest.mark.parametrize(
    "drafts,scheduled,cp,dtype",
    [
        ([1, 0, 1], [2, 3, 5], False, np.int32),
        ([1, 1], [2, 5], False, np.int32),
        ([0, 0], [1, 2], False, np.int32),
        ([2, 1], [3, 5], False, np.int32),
        ([1, 1], [2, 4], True, np.int32),
        ([1, 1], [2, 4], False, np.int64),
    ],
)
def test_nonmatching_layouts_keep_general_behavior(methods, pin_calls, drafts, scheduled, cp, dtype):
    ns, _ = methods
    subject, baseline = runner(methods, cp=cp), runner(methods, cp=cp)
    baseline._fixed_mtp_metadata = None
    args = (
        np.array(drafts, dtype=dtype),
        np.array(scheduled, dtype=np.int64),
        np.zeros(len(drafts), dtype=np.int32) if cp else None,
    )
    expected = ns["_calc_spec_decode_metadata"](baseline, *args)
    pin_calls.clear()
    actual = ns["_calc_spec_decode_metadata"](subject, *args)
    assert len(pin_calls) == (6 if cp else 5)
    compare(actual, expected)


@pytest.mark.parametrize(
    "method,width,cp", [(None, 1, False), ("eagle", 2, False), ("mtp", 3, False), ("mtp", 2, True)]
)
def test_templates_only_allocated_for_supported_configuration(methods, method, width, cp):
    assert runner(methods, method=method, width=width, cp=cp)._fixed_mtp_metadata is None


def test_capacity_changes_and_fallback_do_not_mutate_held_metadata(methods, pin_calls):
    ns, _ = methods
    subject = runner(methods)
    held = []
    for count in (16, 1, 7, 4, 16):
        subject.input_ids.gpu.add_(1)
        result = ns["_calc_spec_decode_metadata"](
            subject,
            np.ones(count, dtype=np.int32),
            np.arange(2, 2 * count + 1, 2),
            None,
        )
        held.append((result, {name: getattr(result, name).clone() for name in FIELDS}))
        ns["_calc_spec_decode_metadata"](
            subject,
            np.array([1, 0], dtype=np.int32),
            np.array([2, 7], dtype=np.int64),
            None,
        )
    for result, snapshot in held:
        for name in FIELDS:
            torch.testing.assert_close(getattr(result, name), snapshot[name], rtol=0, atol=0)


def test_request_row_reordering_uses_current_inputs(methods, pin_calls):
    ns, _ = methods
    subject = runner(methods)
    drafts, scheduled = np.ones(3, dtype=np.int32), np.array([2, 4, 6], dtype=np.int64)
    first = ns["_calc_spec_decode_metadata"](subject, drafts, scheduled, None)
    subject.input_ids.gpu[:6] = torch.tensor([4, 5, 0, 1, 2, 3], dtype=torch.int32)
    second = ns["_calc_spec_decode_metadata"](subject, drafts, scheduled, None)
    assert first.draft_token_ids.tolist() == [1, 3, 5]
    assert second.draft_token_ids.tolist() == [5, 1, 3]
    assert not pin_calls


def test_real_npu_fixed_metadata_matches_general_path(methods):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires an NPU")
    ns, _ = methods
    subject, baseline = runner(methods), runner(methods)
    subject.device = baseline.device = torch.device("npu")
    subject.input_ids.gpu = torch.arange(512, dtype=torch.int32, device="npu")
    baseline.input_ids.gpu = subject.input_ids.gpu
    subject._fixed_mtp_metadata = ns["_fixed_mtp_metadata_arrays"](16, subject.device)
    baseline._fixed_mtp_metadata = None
    held = []
    for requests in (16, 1, 7, 4, 16):
        subject.input_ids.gpu.add_(100)
        for dtype in (np.int32, np.int64):
            args = (np.ones(requests, dtype=np.int32), np.arange(2, 2 * requests + 1, 2, dtype=dtype), None)
            actual = ns["_calc_spec_decode_metadata"](subject, *args)
            expected = ns["_calc_spec_decode_metadata"](baseline, *args)
            held.append((actual, expected))
    torch.npu.synchronize()
    for actual, expected in held:
        compare(actual, expected)


@pytest.mark.parametrize("requests", [1, 3, 16])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("draft_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("lmhead_tp", [False, True])
def test_validated_async_preparation_reuses_fixed_helpers(
    methods, monkeypatch, pin_calls, requests, index_dtype, draft_dtype, lmhead_tp
):
    subject = runner(methods)
    baseline = runner(methods)
    baseline._fixed_mtp_metadata = None
    subject._async_pending = object()
    subject._async_padded_logits = None
    subject.uniform_decode_query_len = 2
    subject._fixed_decode_cu_num_tokens = subject._fixed_decode_cu_num_tokens.astype(index_dtype)
    subject.input_batch = SimpleNamespace(
        num_reqs=requests,
        prev_sampled_token_ids=torch.arange(requests, dtype=torch.int32).view(-1, 1),
    )
    subject._draft_token_ids = torch.arange(requests, dtype=draft_dtype).view(-1, 1) + 100
    prepare = extract(
        ROOT / "vllm_ascend/worker/sfa_async_mtp.py",
        "_prepare_inputs",
        dict(
            torch=torch,
            AscendAttentionState=SimpleNamespace(SpecDecoding="spec"),
            lmhead_tp_enable=lambda: lmhead_tp,
            prepare_async_mtp_tokens_kernel=AsyncMTPTokenKernel(),
            triton=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("validated preparation must not allocate ones or repeat generic validation")

    subject._prepare_input_ids = subject._calc_spec_decode_metadata = forbidden
    monkeypatch.setattr(np, "ones", forbidden)
    padding = subject.input_ids.gpu[2 * requests :].clone()
    previous_logits = previous_spec = previous_draft = None
    for step in range(2):
        subject.input_batch.prev_sampled_token_ids.add_(7)
        subject._draft_token_ids.add_(11)
        baseline.input_ids.gpu[: 2 * requests : 2] = subject.input_batch.prev_sampled_token_ids[:, 0]
        baseline.input_ids.gpu[1 : 2 * requests : 2] = subject._draft_token_ids[:, 0]
        expected = methods[0]["_calc_spec_decode_metadata"](
            baseline, np.full(requests, 1, dtype=np.int32), subject._fixed_decode_cu_num_tokens[:requests], None
        )
        with monkeypatch.context() as patch:
            if step:
                patch.setattr(torch.nn.functional, "pad", forbidden)
            logits, actual, count = prepare(subject, None, None)
        if step:
            if lmhead_tp:
                assert logits is previous_logits
            torch.testing.assert_close(previous_spec.draft_token_ids, previous_draft)
        previous_logits, previous_spec = logits, actual
        previous_draft = actual.draft_token_ids.clone()
        compare(actual, expected)
        assert count == 2 * requests
        torch.testing.assert_close(subject.input_ids.gpu[2 * requests :], padding)
        if lmhead_tp:
            expected_logits = torch.nn.functional.pad(
                expected.logits_indices, (0, 2 * (subject.max_num_reqs - requests))
            )
        else:
            expected_logits = expected.logits_indices
        torch.testing.assert_close(logits, expected_logits)
