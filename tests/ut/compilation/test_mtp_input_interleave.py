# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the Ascend override against the actual inherited input-ID path."""

import ast
import copy
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]


def method(path, name):
    return next(
        n
        for n in ast.walk(ast.parse(path.read_text(encoding="utf8")))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


@pytest.fixture(scope="module")
def runner_class():
    upstream = ROOT.parent / "vllm/vllm/v1/worker/gpu_model_runner.py"
    if not upstream.is_file():
        upstream = importlib.metadata.distribution("vllm").locate_file("vllm/v1/worker/gpu_model_runner.py")
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse("from __future__ import annotations\nclass Base: pass\nclass Subject(Base): pass")
    tree.body[1].body = [method(upstream, "_prepare_input_ids")]
    tree.body[2].body = [method(path, "_prepare_input_ids")]
    ns = dict(torch=torch, np=np)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    base = ns["Base"]._prepare_input_ids

    def observed(self, *args):
        self.fallback_calls += 1
        self.before_fallback = self.input_ids.gpu.clone()
        return base(self, *args)

    ns["Base"]._prepare_input_ids = observed
    return ns["Subject"]


class Buffer:
    def __init__(self, n, device="cpu"):
        self.cpu = torch.arange(n, dtype=torch.int32) + 9000
        self.gpu = torch.full((n,), -77, dtype=torch.int32, device=device)

    def copy_to_gpu(self, n):
        self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)


def setup(cls, n=4, draft_dtype=torch.int64, device="cpu"):
    subject = cls()
    ids = [f"r{i}" for i in range(n)]
    subject.input_batch = SimpleNamespace(
        num_reqs=n,
        req_id_to_index=dict(zip(ids, range(n))),
        prev_req_id_to_index=dict(zip(ids, range(n))),
        prev_sampled_token_ids=torch.arange(n, dtype=torch.int32, device=device).view(n, 1) + 100,
    )
    subject._draft_token_ids = torch.arange(n, dtype=draft_dtype, device=device).view(n, 1) + 200
    subject.input_ids = Buffer(2 * n + 5, device)
    subject.enable_prompt_embeds = False
    subject.use_async_scheduling = True
    subject._fixed_mtp_metadata = ()  # Startup gate already tested in test_mtp_metadata_arrays.
    subject._fixed_decode_cu_num_tokens = np.arange(2, 2 * max(16, n) + 1, 2, dtype=np.int64)
    subject.num_spec_tokens = 1
    subject.device = torch.device(device)
    subject.pin_memory = device != "cpu"
    subject.fallback_calls = 0
    sched = SimpleNamespace(
        scheduled_new_reqs=[],
        finished_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        scheduled_spec_decode_tokens={rid: [-1] for rid in ids},
    )
    return subject, sched, np.arange(2, 2 * n + 1, 2, dtype=np.int64)


@pytest.mark.parametrize("n", [1, 2, 4, 7, 16, 32])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_exact_tokens_and_padding_across_steps(runner_class, n, dtype):
    subject, sched, cu = setup(runner_class, n, dtype)
    baseline = copy.deepcopy(subject)
    baseline._fixed_mtp_metadata = None
    addresses = (subject.input_ids.gpu.data_ptr(), subject.input_batch.prev_sampled_token_ids.data_ptr())
    for _ in range(3):
        subject._prepare_input_ids(sched, 2 * n, cu)
        baseline._prepare_input_ids(sched, 2 * n, cu)
        torch.testing.assert_close(subject.input_ids.gpu, baseline.input_ids.gpu, rtol=0, atol=0)
        assert subject.fallback_calls == 0
        for owner in (subject, baseline):
            owner.input_batch.prev_sampled_token_ids.add_(13)
            owner._draft_token_ids.add_(17)
    assert addresses == (subject.input_ids.gpu.data_ptr(), subject.input_batch.prev_sampled_token_ids.data_ptr())


def test_fast_path_only_submits_two_copies(runner_class):
    subject, sched, cu = setup(runner_class)
    sampled = subject.input_batch.prev_sampled_token_ids
    draft = subject._draft_token_ids
    versions = sampled._version, draft._version
    ops = []

    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            ops.append(func)
            return func(*args, **(kwargs or {}))

    with Record():
        subject._prepare_input_ids(sched, 8, cu)
    assert ops.count(torch.ops.aten.copy_.default) == 2
    assert set(ops) <= {
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.select.int,
        torch.ops.aten.view.default,
        torch.ops.aten.copy_.default,
    }
    assert versions == (sampled._version, draft._version)


@pytest.mark.parametrize(
    "case",
    [
        "config",
        "sync",
        "embeddings",
        "new",
        "finished",
        "resumed",
        "reorder",
        "missing_previous",
        "no_common",
        "dropped_draft",
        "wide",
        "extra_previous",
        "multidraft",
    ],
)
def test_transitions_delegate_before_any_write(runner_class, case):
    subject, sched, cu = setup(runner_class)
    if case == "config":
        subject._fixed_mtp_metadata = None
    elif case == "sync":
        subject.use_async_scheduling = False
    elif case == "embeddings":
        subject.enable_prompt_embeds = True
    elif case == "new":
        sched.scheduled_new_reqs = [SimpleNamespace(req_id="r0")]
    elif case == "finished":
        sched.finished_req_ids = {"r0"}
    elif case == "resumed":
        sched.scheduled_cached_reqs.resumed_req_ids = {"r0"}
    elif case == "reorder":
        subject.input_batch.req_id_to_index = {f"r{i}": 3 - i for i in range(4)}
    elif case == "missing_previous":
        subject.input_batch.prev_sampled_token_ids = None
    elif case == "no_common":
        subject.input_batch.prev_req_id_to_index = {f"old{i}": i for i in range(4)}
    elif case == "dropped_draft":
        sched.scheduled_spec_decode_tokens.pop("r0")
        cu -= 1
    elif case == "wide":
        cu += 1
    elif case == "extra_previous":
        subject.input_batch.prev_req_id_to_index["old"] = 4
    elif case == "multidraft":
        subject._draft_token_ids = subject._draft_token_ids.expand(4, 2).clone()
    before = subject.input_ids.gpu.clone()
    baseline = copy.deepcopy(subject)
    baseline._fixed_mtp_metadata = None
    subject._prepare_input_ids(sched, int(cu[-1]), cu)
    baseline._prepare_input_ids(sched, int(cu[-1]), cu)
    assert subject.fallback_calls == 1
    torch.testing.assert_close(subject.before_fallback, before, rtol=0, atol=0)
    torch.testing.assert_close(subject.input_ids.gpu, baseline.input_ids.gpu, rtol=0, atol=0)


def test_invalid_sampled_dtype_keeps_original_error(runner_class):
    subject, sched, cu = setup(runner_class)
    subject.input_batch.prev_sampled_token_ids = subject.input_batch.prev_sampled_token_ids.long()
    with pytest.raises(RuntimeError, match="dtype"):
        subject._prepare_input_ids(sched, 8, cu)
    assert subject.fallback_calls == 1
    assert torch.all(subject.before_fallback == -77)


def test_npu_interleave_preserves_queued_consumers(runner_class):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires an NPU")
    held = []
    for n in (1, 4, 16):
        subject, sched, cu = setup(runner_class, n, device="npu")
        for _ in range(3):
            previous = subject.input_ids.gpu.clone()
            sampled = subject.input_batch.prev_sampled_token_ids
            draft = subject._draft_token_ids
            expected = subject.input_ids.gpu.clone()
            expected[: 2 * n] = torch.stack((sampled[:, 0], draft[:, 0].int()), dim=1).flatten()
            subject._prepare_input_ids(sched, 2 * n, cu)
            held.append((subject.input_ids.gpu.clone(), expected, previous))
            sampled.add_(13)
            draft.add_(17)
    torch.npu.synchronize()
    for actual, expected, previous in held:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert previous.data_ptr() != actual.data_ptr()
