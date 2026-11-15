# SPDX-License-Identifier: Apache-2.0
"""Host-arena source used by the selected-prefill performance harness.

This is deliberately test-only: source pages live in ACL-mapped host memory,
not in LMCache or AscendStore.  It exercises the real selected-copy kernel and
the target packed-main pages while keeping the connector path out of the
measurement.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

PAGE_ROWS = 128
ROW_BYTES = 656
PAGE_BYTES = PAGE_ROWS * ROW_BYTES

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


@dataclass(frozen=True)
class SelectedRequest:
    request_id: str
    history_tokens: int
    query_tokens: int
    selected: bool


class SelectedLoadStep:
    """Protocol-compatible load step for :class:`SelectedKVRuntime`."""

    def __init__(
        self,
        request: SelectedRequest,
        page_bases: dict[str, torch.Tensor],
        ready: torch.npu.Event,
    ) -> None:
        self.requests = (request,)
        self._page_bases = page_bases
        self._ready = ready
        self._completion: torch.npu.Event | None = None
        self._failed = False

    def ready(self, layer_name: str) -> torch.npu.Event:
        return self._ready

    def page_bases(self, layer_name: str) -> torch.Tensor:
        return self._page_bases[layer_name]

    def main_copy_submitted(
        self, layer_name: str, completion: torch.npu.Event | None = None
    ) -> None:
        if completion is not None:
            self._completion = completion

    def prefetch_main(self, layer_name: str, submit) -> None:
        submit()

    def finish(
        self, completion: torch.npu.Event | None = None, failed: bool = False
    ) -> None:
        self._failed = failed
        if completion is not None:
            self._completion = completion

    def wait_completion(self) -> None:
        if self._completion is not None:
            self._completion.synchronize()

    def try_retire(self) -> bool:
        return not self._failed


class _HostArena:
    def __init__(self, pages: int) -> None:
        self.pages = pages
        self.bytes = pages * PAGE_BYTES
        self.acl = ctypes.CDLL("libascendcl.so")
        self.acl.aclrtMallocHost.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.acl.aclrtMallocHost.restype = ctypes.c_int32
        self.acl.aclrtHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.acl.aclrtHostRegister.restype = ctypes.c_int32
        self.acl.aclrtHostGetDevicePointer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
        ]
        self.acl.aclrtHostGetDevicePointer.restype = ctypes.c_int32
        self.acl.aclrtFreeHost.argtypes = [ctypes.c_void_p]
        self.acl.aclrtFreeHost.restype = ctypes.c_int32
        self.acl.aclrtMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        self.acl.aclrtMemcpyAsync.restype = ctypes.c_int32
        self.host = ctypes.c_void_p()
        rc = self.acl.aclrtMallocHost(ctypes.byref(self.host), self.bytes)
        if rc:
            raise RuntimeError(f"aclrtMallocHost failed: {rc}")
        self.device = ctypes.c_void_p()
        rc = self.acl.aclrtHostRegister(
            self.host, self.bytes, 0, ctypes.byref(self.device)
        )
        if rc:
            raise RuntimeError(f"aclrtHostRegister failed: {rc}")
        mapped = ctypes.c_void_p()
        rc = self.acl.aclrtHostGetDevicePointer(
            self.host, ctypes.byref(mapped), 0
        )
        if rc:
            raise RuntimeError(f"aclrtHostGetDevicePointer failed: {rc}")
        self.mapped = mapped.value

    def close(self) -> None:
        if self.host:
            self.acl.aclrtFreeHost(self.host)
            self.host = ctypes.c_void_p()

    def write_from_cpu(self, tensor: torch.Tensor) -> None:
        data = tensor.detach().cpu().contiguous()
        if data.nbytes > self.bytes:
            raise ValueError("host arena overflow")
        ctypes.memmove(self.host, data.data_ptr(), data.nbytes)

    def read_to_cpu(self, nbytes: int) -> torch.Tensor:
        if nbytes > self.bytes:
            raise ValueError("host arena overflow")
        out = torch.empty(nbytes, dtype=torch.uint8)
        ctypes.memmove(out.data_ptr(), self.host, nbytes)
        return out

    def d2h(self, dst_ptr: int, nbytes: int, stream) -> None:
        rc = self.acl.aclrtMemcpyAsync(
            self.host, ctypes.c_void_p(dst_ptr), nbytes,
            ACL_MEMCPY_DEVICE_TO_HOST, ctypes.c_void_p(stream),
        )
        if rc:
            raise RuntimeError(f"aclrtMemcpyAsync D2H failed: {rc}")

    def h2d(self, src_offset: int, dst_ptr: int, nbytes: int, stream) -> None:
        rc = self.acl.aclrtMemcpyAsync(
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(self.host.value + src_offset),
            nbytes,
            ACL_MEMCPY_HOST_TO_DEVICE,
            ctypes.c_void_p(stream),
        )
        if rc:
            raise RuntimeError(f"aclrtMemcpyAsync H2D failed: {rc}")


class HostArenaSelectedStore:
    """One ACL-mapped host arena per packed main layer."""

    def __init__(
        self,
        device: torch.device,
        main_caches: dict[str, torch.Tensor],
        *,
        max_pages: int = 1024,
    ) -> None:
        self.device = device
        self.main_caches = dict(main_caches)
        self.max_pages = int(max_pages)
        if self.max_pages <= 0:
            raise ValueError("selected test store needs at least one page")
        self.arenas = {name: _HostArena(self.max_pages) for name in self.main_caches}
        self.saved_pages: dict[str, int] = {}

    def close(self) -> None:
        for arena in self.arenas.values():
            arena.close()

    def save_pages(
        self,
        layer_name: str,
        cache: torch.Tensor,
        block_ids: Sequence[int],
        count: int,
    ) -> None:
        count = int(count)
        if count <= 0 or count > self.max_pages or len(block_ids) < count:
            raise ValueError("selected test store page count is out of range")
        ids = torch.tensor(block_ids[:count], dtype=torch.long, device=self.device)
        pages = cache.index_select(0, ids)
        self.arenas[layer_name].write_from_cpu(pages)
        self.saved_pages[layer_name] = count

    def page_bases(
        self,
        layer_name: str,
        block_ids: Sequence[int],
        count: int,
        num_blocks: int,
    ) -> torch.Tensor:
        count = int(count)
        arena = self.arenas[layer_name]
        if count > arena.pages or count > len(block_ids):
            raise ValueError("selected test store page count is out of range")
        bases = np.zeros((int(num_blocks),), dtype=np.uint64)
        for j in range(count):
            bases[int(block_ids[j])] = arena.mapped + j * PAGE_BYTES
        return torch.from_numpy(bases).to(self.device, non_blocking=True)

    def full_load(
        self,
        layer_name: str,
        cache: torch.Tensor,
        block_ids: Sequence[int],
        count: int,
    ) -> None:
        count = int(count)
        arena = self.arenas[layer_name]
        if count > arena.pages or count > len(block_ids):
            raise ValueError("selected test store page count is out of range")
        ids_list = [int(v) for v in block_ids[:count]]
        host = arena.read_to_cpu(count * PAGE_BYTES).view(
            count, PAGE_ROWS, 1, ROW_BYTES
        ).to(torch.int8)
        staging = host.to(self.device)
        contiguous = ids_list == list(range(ids_list[0], ids_list[0] + count)) if ids_list else False
        if contiguous:
            cache[ids_list[0] : ids_list[0] + count].copy_(staging)
        else:
            ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
            cache.index_copy_(0, ids, staging)
