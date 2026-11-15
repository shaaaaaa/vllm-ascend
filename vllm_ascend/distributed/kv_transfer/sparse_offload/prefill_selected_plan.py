# SPDX-License-Identifier: Apache-2.0
"""Device-count union and selected C8 copies into original native KV pages."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

PAGE_ROWS = 128
MAX_HISTORY = 524288
MAX_BATCHED_TOKENS = 8192
MAX_REQUESTS = 256
CORES = 32
COUNTER_STRIDE = 8
COHORT_ROWS = 2
COHORT_CAPACITY = 4096
BITMAP_INTS = MAX_HISTORY // 32


def selected_plan_shapes(
    capacity: int, query_capacity: int, max_requests: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """One same-stream workspace, including the current owner's IDs and status."""
    values = (capacity, query_capacity, max_requests)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Selection workspace bounds must be positive integers")
    if (
        capacity > MAX_HISTORY
        or capacity % PAGE_ROWS
        or query_capacity > MAX_BATCHED_TOKENS
        or max_requests > MAX_REQUESTS
    ):
        raise ValueError("Selection workspace exceeds the fused kernel bounds")
    cohorts = (query_capacity + COHORT_ROWS - 1) // COHORT_ROWS
    scratch = (
        (cohorts, COHORT_CAPACITY),
        (cohorts, COUNTER_STRIDE),
        (MAX_HISTORY,),
        (CORES, COUNTER_STRIDE),
        (CORES, COUNTER_STRIDE),
        (BITMAP_INTS,),
    )
    return scratch, ((max_requests, capacity), (max_requests, COUNTER_STRIDE))


def selected_plan_storage_bytes(capacity: int, query_capacity: int, max_requests: int) -> int:
    """Exact device bytes; source addresses and request metadata are separate."""
    from math import prod

    scratch, outputs = selected_plan_shapes(capacity, query_capacity, max_requests)
    return 4 * (sum(map(prod, scratch)) + sum(map(prod, outputs)))


class SelectedKVPlanner:
    def __init__(self, device: torch.device | str, capacity: int, query_capacity: int, max_requests: int) -> None:
        # Keep configuration/accounting usable without loading NPU libraries.
        import torch

        scratch, outputs = selected_plan_shapes(capacity, query_capacity, max_requests)
        self.device = torch.device(device)
        self.capacity, self.query_capacity = capacity, query_capacity
        self.max_requests = max_requests
        self.pending: list[tuple[torch.Tensor | list[torch.Tensor], ...]] = []
        self._closed = False
        root = Path(__file__).resolve().parents[3] / "lib"
        torch.ops.load_library(str(root / "libsparse_prefill_fused_fixed_launch.so"))
        self.ops = torch.ops.dsa_prefill_queue
        self.library = ctypes.CDLL(str(root / "libsparse_prefill_fused_fixed_launch.so"))
        self.copy_library = ctypes.CDLL(str(root / "libdsa_packed_inplace.so"))
        self.copy_launch = ctypes.cast(self.copy_library.launch_sparse_prefill_packed_inplace, ctypes.c_void_p).value
        create = self.library.create_sparse_prefill_fixed_handle
        create.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int32
        self.handle = ctypes.c_void_p()
        self.fused_launch = ctypes.cast(self.library.launch_sparse_prefill_fused_plan, ctypes.c_void_p).value
        self._destroy = self.library.destroy_sparse_prefill_fused_handle
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = ctypes.c_int32
        allocate = lambda shape: torch.empty(shape, dtype=torch.int32, device=self.device)
        # No live ACL handle exists if a workspace allocation fails.
        self.scratch = [allocate(shape) for shape in scratch]
        self.outputs = [allocate(shape) for shape in outputs]
        with torch.npu.device(self.device):
            status = create(str(root / "sparse_prefill_fused_fixed.o").encode(), capacity, ctypes.byref(self.handle))
        if status:
            raise RuntimeError(f"Cannot initialize fused selected-KV planning: ACL status {status}")

    def build(
        self,
        topk: torch.Tensor,
        histories: Sequence[int],
        starts: Sequence[int],
        queries: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Overwrite outputs after their prior readers on the caller's stream."""
        n = len(histories)
        if self._closed:
            raise RuntimeError("Selected-KV planning requires a live device-count workspace")
        if not 0 < n <= self.max_requests or any(
            type(value) is not int for values in (histories, starts, queries) for value in values
        ):
            raise ValueError("Selected request metadata exceeds configured capacity or is not integer")
        if topk.ndim != 3 or not 0 < topk.shape[0] <= self.query_capacity or topk.device != self.device:
            raise ValueError("Selected TopK exceeds its configured rows or device")
        # The C++ entry validates raw tensor ABI and H/Q spans before enqueue.
        # Keep configured bounds here: native global limits are not reservations.
        buffers = self.outputs
        self.pending.append((topk, buffers, self.scratch))
        self.ops.batch_plan(
            topk,
            self.scratch,
            buffers,
            list(histories),
            list(starts),
            list(queries),
            self.capacity,
            self.fused_launch,
            self.handle.value,
        )
        return buffers[0][:n], buffers[1][:n]

    def load(
        self,
        ids: torch.Tensor,
        status: torch.Tensor,
        page_bases: torch.Tensor,
        request_metadata: torch.Tensor,
        block_table: torch.Tensor,
        native_cache: torch.Tensor,
    ) -> None:
        """Copy history only; zero page bases were proven FULL by the source."""
        n = ids.shape[0] if ids.ndim == 2 else 0
        if self._closed:
            raise RuntimeError("Selected copies require a live device-count workspace")
        if not 0 < n <= self.max_requests or ids.shape != (n, self.capacity) or ids.device != self.device:
            raise ValueError("Selected IDs exceed their configured shape or device")
        tensors = (ids, status, page_bases, request_metadata, block_table, native_cache)
        self.pending.append(tensors)
        self.ops.inplace_copy(*tensors, self.copy_launch)

    def release_quiescent(self) -> None:
        """The caller must first retire every real producer/consumer event."""
        self.pending.clear()

    def close_quiescent(self) -> None:
        if not self._closed:
            import torch

            with torch.npu.device(self.device):
                status = self._destroy(self.handle)
                if status:
                    raise RuntimeError(f"Cannot unload selected-KV planning: ACL status {status}")
                self.handle = ctypes.c_void_p()
                self.release_quiescent()
                self._closed = True
