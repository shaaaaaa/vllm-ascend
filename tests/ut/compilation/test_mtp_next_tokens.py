# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare fused MTP selection with the unchanged Ascend tensor implementation.

CPU tests execute the kernel body through a small masked-memory adapter. They
verify arithmetic/addressing, not Triton compilation or NPU execution, which is
covered by the hardware test at the end of this file.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sfa_test_support import HostTL, Pointer, extract, load_module

ROOT = Path(__file__).resolve().parents[3]
KERNEL_PATH = ROOT / "vllm_ascend/ops/triton/spec_decode/utils.py"


class HostKernel:
    def __init__(self):
        self.tl = HostTL()
        self.body = extract(KERNEL_PATH, "prepare_next_mtp_tokens_kernel", {"tl": self.tl})
        self.calls = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls += 1
            pointers = [Pointer(t) for t in args[:4]]
            self.tl.programs = grid[0]
            for pid in range(grid[0]):
                self.tl.pid = pid
                self.body(*pointers, *args[4:], **kwargs)

        return launch


@pytest.fixture
def methods():
    kernel = HostKernel()
    ns = dict(
        torch=torch,
        np=np,
        HAS_TRITON=True,
        prepare_next_mtp_tokens_kernel=kernel,
        triton=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
        get_vectorcore_num=lambda: 2,
        _PREPARE_INPUTS_BLOCK_SIZE=4,
    )
    run = extract(ROOT / "vllm_ascend/spec_decode/eagle_proposer.py", "prepare_next_token_ids_padded", ns)
    return run, ns, kernel


class Backup:
    def __init__(self, n, device="cpu"):
        self.cpu = torch.zeros(n + 3, dtype=torch.int32, pin_memory=device != "cpu")
        self.np = self.cpu.numpy()
        self.gpu = torch.full((n + 3,), -77, dtype=torch.int32, device=device)
        self.copies = 0

    def copy_to_gpu(self, n):
        self.copies += 1
        self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)


def arguments(n, width, discarded, *, strided=False, device="cpu", dtype=torch.int32):
    patterns = torch.tensor([[7, -1], [7, 8], [-1, -1], [100, -1], [-1, 8], [100, 8], [-2, 8], [0, 101]], dtype=dtype)
    values = patterns[torch.arange(n) % len(patterns), :width].to(device)
    if strided:
        storage = torch.full((n, 6), -999, dtype=dtype, device=device)
        storage[:, 1 : 1 + 2 * width : 2] = values
        values = storage[:, 1 : 1 + 2 * width : 2]
    subject = SimpleNamespace(method="mtp", num_speculative_tokens=1, backup_next_token_ids=Backup(n, device))
    batch = SimpleNamespace(num_reqs=n, req_ids=[f"r{i}" for i in range(n)], vocab_size=100)
    requests = {
        rid: SimpleNamespace(get_token_id=lambda pos, i=i: 1000 + i + pos) for i, rid in enumerate(batch.req_ids)
    }
    common = SimpleNamespace(seq_lens_cpu=torch.arange(n, dtype=torch.int32) + 20)
    # Entries beyond num_discarded must never be consumed.
    indices = torch.tensor(discarded + [999, 999], dtype=torch.int64, device=device)
    return subject, (common, values, requests, batch, indices, len(discarded))


@pytest.mark.parametrize("n", [1, 3, 4, 5, 16, 17])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("discard_mode", ["none", "first", "all", "duplicate", "negative"])
def test_exact_selection_counts_masking_and_lifetime(methods, n, width, discard_mode):
    run, ns, kernel = methods
    discarded = {"none": [], "first": [0], "all": list(range(n)), "duplicate": [0, 0], "negative": [-1]}[discard_mode]
    owner, args = arguments(n, width, discarded, strided=True)
    original_tokens = args[1].clone()
    ns["HAS_TRITON"] = False
    expected = run(owner, *args)
    ns["HAS_TRITON"] = True
    actual = run(owner, *args)
    assert kernel.calls == (0 if discarded else 1) and owner.backup_next_token_ids.copies == 2
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.dtype == b.dtype and a.is_contiguous()
    torch.testing.assert_close(args[1], original_tokens, rtol=0, atol=0)
    held = tuple(x.clone() for x in actual)
    args[1].fill_(9)
    run(owner, *args)
    for a, b in zip(actual, held):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.all(owner.backup_next_token_ids.gpu[n:] == -77)


@pytest.mark.parametrize("mode", ["no_triton", "eagle", "multidraft", "int64", "wide", "empty"])
def test_other_modes_keep_tensor_path(methods, mode):
    run, ns, kernel = methods
    owner, args = arguments(4, 2, [1], dtype=torch.int64 if mode == "int64" else torch.int32)
    if mode == "eagle":
        owner.method = "eagle"
    elif mode == "multidraft":
        owner.num_speculative_tokens = 2
    elif mode == "wide":
        args = (args[0], torch.cat((args[1], args[1][:, :1]), 1), *args[2:])
    elif mode == "empty":
        owner, args = arguments(0, 2, [])
    ns["HAS_TRITON"] = False
    expected = run(owner, *args)
    ns["HAS_TRITON"] = mode != "no_triton"
    actual = run(owner, *args)
    assert kernel.calls == 0
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_kernel_runs_after_backup_upload(methods):
    run, ns, _ = methods
    owner, args = arguments(4, 2, [])

    class CheckLaunch:
        def __getitem__(self, grid):
            def launch(*inputs, **kwargs):
                assert owner.backup_next_token_ids.copies == 1
                assert inputs[1][0].item() == 1020
                assert inputs[2].dtype == torch.int32 and inputs[3].dtype == torch.int64

            return launch

    ns["prepare_next_mtp_tokens_kernel"] = CheckLaunch()
    run(owner, *args)


def test_npu_fused_selection_and_count_readback(methods, monkeypatch):
    pytest.importorskip("torch_npu")
    pytest.importorskip("triton")
    if not torch.npu.is_available():
        pytest.skip("Requires an NPU")
    module = load_module(KERNEL_PATH, "mtp_next_tokens_npu_kernel", monkeypatch)
    run, ns, _ = methods
    ns["prepare_next_mtp_tokens_kernel"] = module.prepare_next_mtp_tokens_kernel
    copy_stream = torch.npu.Stream()
    held = []
    for n in (1, 4, 17):
        for width in (1, 2):
            for discarded in ([], [0], list(range(n))):
                owner, args = arguments(n, width, discarded, strided=True, device="npu")
                ns["HAS_TRITON"] = False
                expected = run(owner, *args)
                ns["HAS_TRITON"] = True
                actual = run(owner, *args)
                host = torch.empty(n, dtype=torch.int64, pin_memory=True)
                producer = torch.npu.current_stream()
                with torch.npu.stream(copy_stream):
                    copy_stream.wait_stream(producer)
                    host.copy_(actual[1], non_blocking=True)
                held.append((actual, expected, host, owner, args))
    copy_stream.synchronize()
    for actual, expected, host, _, _ in held:
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(host, expected[1].cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_backup_ids_keep_real_request_lookup_and_active_length_order(methods, monkeypatch, dtype, strided):
    run, _, _ = methods
    owner, args = arguments(6, 2, [])
    common, sampled, requests, batch, *_ = args
    get_token_id = extract(ROOT.parent / "vllm/vllm/v1/worker/gpu_input_batch.py", "get_token_id", {})
    request_type = type("Request", (), {"get_token_id": get_token_id})
    for rid in batch.req_ids:
        request = request_type()
        request.num_prompt_tokens, request.prompt_token_ids, request.output_token_ids = 3, [11, 12, 13], [21, 22]
        requests[rid] = request
    lengths = torch.tensor([0, 2, 3, 4, 5, -1, 999], dtype=dtype)
    if strided:
        storage = torch.zeros(2 * len(lengths), dtype=dtype)
        storage[::2] = lengths
        lengths = storage[::2]
    common.seq_lens_cpu = lengths
    # Independent reference: exact old preparation, including the missing-token -1.
    expected = np.array([requests[rid].get_token_id(lengths[i].item()) for i, rid in enumerate(batch.req_ids)])
    sampled.fill_(-1)

    def forbidden(*args, **kwargs):
        raise AssertionError("backup preparation must not read per-row scalars or allocate np.array")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(np, "array", forbidden)
        next_ids, counts = run(owner, *args)
    assert next_ids.tolist() == expected.tolist()
    assert counts.tolist() == [0] * 6
    assert owner.backup_next_token_ids.np[:6].tolist() == expected.tolist()
    assert owner.backup_next_token_ids.copies == 1
    assert owner.backup_next_token_ids.gpu[6:].tolist() == [-77] * 3


@pytest.mark.parametrize("failure", ["short_lengths", "unknown_prompt_id"])
def test_backup_failure_does_not_upload_or_partly_overwrite_host_buffer(methods, failure):
    run, _, _ = methods
    owner, args = arguments(3, 2, [])
    if failure == "short_lengths":
        args[0].seq_lens_cpu = torch.tensor([20, 21])
        error = IndexError
    else:
        def unavailable(index):
            raise ValueError("prompt token ID is unknown")
        args[2]["r1"].get_token_id = unavailable
        error = ValueError
    owner.backup_next_token_ids.cpu.fill_(-99)
    with pytest.raises(error):
        run(owner, *args)
    assert owner.backup_next_token_ids.copies == 0
    assert owner.backup_next_token_ids.cpu.tolist() == [-99] * 6


def test_empty_batch_does_not_require_cpu_lengths(methods):
    run, _, _ = methods
    owner, args = arguments(0, 2, [])
    args[0].seq_lens_cpu = None
    result = run(owner, *args)
    assert all(t.numel() == 0 for t in result)
    assert owner.backup_next_token_ids.copies == 1
