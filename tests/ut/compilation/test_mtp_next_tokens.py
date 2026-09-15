# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare fused MTP selection with the unchanged Ascend tensor implementation.

CPU tests execute the kernel body through a small masked-memory adapter. They
verify arithmetic/addressing, not Triton compilation or NPU execution, which is
covered by the hardware test at the end of this file.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
KERNEL_PATH = ROOT / "vllm_ascend/ops/triton/spec_decode/utils.py"


def extract(path, name, namespace):
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text(encoding="utf8")))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(node)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


class Pointer:
    def __init__(self, tensor, offset=0):
        count = tensor.untyped_storage().nbytes() // tensor.element_size() - tensor.storage_offset()
        self.data = tensor.as_strided((count,), (1,))
        self.offset = offset

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)


class HostTL:
    int32 = torch.int32
    range = staticmethod(range)
    arange = staticmethod(torch.arange)
    where = staticmethod(torch.where)
    full = staticmethod(lambda shape, value, dtype: torch.full(shape, value, dtype=dtype))

    def program_id(self, axis):
        return self.pid

    def num_programs(self, axis):
        return self.programs

    @staticmethod
    def load(ptr, mask, other):
        offsets, mask = torch.broadcast_tensors(torch.as_tensor(ptr.offset), torch.as_tensor(mask))
        result = torch.full(offsets.shape, other, dtype=ptr.data.dtype)
        result[mask] = ptr.data[offsets[mask].long()]
        return result

    @staticmethod
    def store(ptr, value, mask):
        offsets, values, mask = torch.broadcast_tensors(
            torch.as_tensor(ptr.offset), torch.as_tensor(value), torch.as_tensor(mask)
        )
        ptr.data[offsets[mask].long()] = values[mask].to(ptr.data.dtype)


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
    spec = importlib.util.spec_from_file_location("mtp_next_tokens_npu_kernel", KERNEL_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
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
