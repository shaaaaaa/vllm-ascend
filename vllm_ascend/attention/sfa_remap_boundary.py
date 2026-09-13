# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builder-owned history boundaries; upload only changed decode values."""

from collections import Counter
from collections.abc import Sequence

import torch


class SFARemapBoundaryBuffer:
    """Own the CPU shadow of one stable device allocation.

    All writes must go through update(), or call invalidate() before an external
    write (eager/staged/dummy preparation). A prefix with a different row count
    is a different value, so switching graph capacities cannot reuse stale tails.
    This caches boundaries, never selected tokens or request source ownership.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self._layout: tuple | None = None
        self._requests: tuple[int, ...] = ()
        self._values: tuple[int, ...] | None = None
        self.upload_count = 0

    def invalidate(self) -> None:
        """Forget the last upload before another path writes this allocation."""
        self._values = None

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
        self.tensor[: len(values)].copy_(torch.tensor(values, dtype=torch.int32))
        self._values = values
        self.upload_count += 1
