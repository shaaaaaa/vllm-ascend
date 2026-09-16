# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source loading and masked CPU memory helpers; no production-result oracles.

These helpers avoid NPU import-time dependencies. HostTL checks arithmetic and
addressing only: it does not emulate device concurrency or compile Triton.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_module(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


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


def definitions(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(body) == len(names)
    module = ast.parse("from __future__ import annotations")
    module.body.extend(body)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


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


class AsyncMTPTokenKernel:
    """Run the production token kernel's address arithmetic on CPU tensors."""

    def __init__(self, events=None):
        self.tl = HostTL()
        self.body = extract(
            ROOT / "vllm_ascend/ops/triton/spec_decode/async_mtp.py", "prepare_async_mtp_tokens_kernel", {"tl": self.tl}
        )
        self.events = events

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if self.events is not None:
                self.events.append("token_kernel")
            for pid in range(grid[0]):
                self.tl.pid = pid
                self.body(*(Pointer(t) for t in args[:4]), *args[4:], **kwargs)

        return launch
