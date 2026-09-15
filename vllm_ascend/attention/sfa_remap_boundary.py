# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builder-owned history boundaries; upload only changed decode values."""

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import torch

MAX_PENDING_BOUNDARY_UPLOADS = 2


def prepare_native_sparse_boundaries(impls: Iterable[tuple[str, Any]], metadata: Any) -> None:
    """Prepare each native metadata object once, immediately before its forward."""
    if not isinstance(metadata, dict):
        return
    prepared = set()
    for name, impl in impls:
        item = metadata.get(name)
        prepare = getattr(impl, "prepare_native_sparse_boundary", None)
        if item is not None and id(item) not in prepared and callable(prepare):
            prepare(item)
            prepared.add(id(item))


class SFARemapBoundaryBuffer:
    """Own the CPU shadow of one stable device allocation.

    All writes must go through update(), or call invalidate() before an external
    write (eager/staged/dummy preparation). A prefix with a different row count
    is a different value, so switching graph capacities cannot reuse stale tails.
    This caches boundaries, never selected tokens or request source ownership.
    Changed NPU values use bounded pinned staging; copy events protect CPU reuse,
    while same-stream ordering protects the stable device destination.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self._layout: tuple | None = None
        self._requests: tuple[int, ...] = ()
        self._values: tuple[int, ...] | None = None
        self._stream = None
        self._uploads: list[tuple[torch.Tensor, Any]] = []
        self._next_upload = 0
        self._failed = False
        self.upload_count = 0

    def invalidate(self) -> None:
        """Forget the last upload before another path writes this allocation."""
        self._values = None

    def _upload(self, values: tuple[int, ...]) -> None:
        if self.tensor.device.type != "npu":
            self.tensor[: len(values)].copy_(torch.tensor(values, dtype=torch.int32))
            return
        index = self._next_upload
        if index == len(self._uploads):
            host = torch.empty_like(self.tensor, device="cpu", pin_memory=True)
            event = torch.npu.Event()
            self._uploads.append((host, event))
        else:
            host, event = self._uploads[index]
            # Only pool exhaustion waits, and only for this slot's H2D copy.
            # Unchanged values never query or record an event.
            if not event.query():
                event.synchronize()
        try:
            host[: len(values)].copy_(torch.tensor(values, dtype=torch.int32))
            self.tensor[: len(values)].copy_(host[: len(values)], non_blocking=True)
            event.record(self._stream)
        except BaseException:
            # Retain staging after uncertain submission; never reuse it on retry.
            self._failed = True
            raise
        self._next_upload = (index + 1) % MAX_PENDING_BOUNDARY_UPLOADS

    def update(
        self,
        row_requests: Sequence[int],
        prompt_rows: Sequence[int],
        seq_lens_cpu: torch.Tensor | Sequence[int],
        frontiers: tuple[int, ...],
        window: int,
        index_topk: int,
        scratch_capacity: int | None,
    ) -> None:
        """Refresh from CPU metadata, preserving the original remap formula.

        Frontiers follow sorted unique nonnegative request indices, as in the
        staged helper. Padding retains its prompt boundary. Layout checks run
        on layout changes; the live-KV overlap guard runs on changed values.
        """
        if self._failed:
            raise RuntimeError("SFA boundary upload failed; staging cannot be reused")
        if self.tensor.device.type == "npu":
            stream = torch.npu.current_stream(self.tensor.device)
            if self._stream is not None and self._stream != stream:
                raise RuntimeError("SFA boundary upload and consumption must stay on one stream")
            self._stream = stream
        rows = tuple(map(int, row_requests))
        prompts = tuple(map(int, prompt_rows))
        layout = (rows, len(prompts), index_topk, scratch_capacity)
        if layout != self._layout:
            if len(rows) != len(prompts) or len(rows) > self.tensor.numel():
                raise RuntimeError("SFA remap boundary shapes differ")
            if index_topk <= 0 or scratch_capacity is None or scratch_capacity < index_topk:
                raise RuntimeError("SFA remap scratch reservation is missing or too small")
            counts = Counter(row for row in rows if row >= 0)
            if any(count * index_topk > scratch_capacity for count in counts.values()):
                raise RuntimeError("SFA request-union scratch reservation is too small")
            self._requests = tuple(sorted(counts))
            self._layout = layout
            self.invalidate()
        if len(frontiers) != len(self._requests):
            raise RuntimeError("SFA remap frontier count does not match decode requests")
        if isinstance(seq_lens_cpu, torch.Tensor):
            if seq_lens_cpu.device.type != "cpu":
                raise RuntimeError("SFA remap lengths must be CPU metadata")
            lengths = seq_lens_cpu.tolist()
        else:
            lengths = seq_lens_cpu
        if self._requests and self._requests[-1] >= len(lengths):
            raise RuntimeError("SFA remap row references an unavailable request")
        boundaries = {
            request: min(max(int(lengths[request]) - 1, 0) // window * window, int(frontier))
            if window > 0
            else int(frontier)
            for request, frontier in zip(self._requests, frontiers)
        }
        values = tuple(prompts[i] if row < 0 else boundaries[row] for i, row in enumerate(rows))
        if values == self._values:
            return
        if any(value != 0 and value < scratch_capacity for value in boundaries.values()):
            raise RuntimeError("SFA request-union scratch would alias live KV positions")
        # A failed/partial copy must not leave the previous cache entry usable.
        self.invalidate()
        self._upload(values)
        self._values = values
        self.upload_count += 1
