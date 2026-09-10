# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device snapshots and CPU comparisons for the opt-in SFA parity worker.

There are no host reads in ``DeviceSnapshot.write`` or ``gather_sparse_kv``.
Their device operations can be captured by the target's existing root graph.
Nothing in this module is enabled by the normal serving worker.
"""

import math
from dataclasses import dataclass

import torch


class ParityError(AssertionError):
    """A mismatch, invalid reference, or incomplete observation."""

    def __init__(self, message: str, *, label: str = "", index: tuple[int, ...] = ()) -> None:
        super().__init__(message)
        self.label = label
        self.index = index


class DeviceSnapshot:
    """Fixed-address tensor storage, allocated before capture.

    The device counter detects missing/stale probes on replay. Host code resets
    it at the forward boundary, never between layers. Unwritten padding is not
    read back or compared.
    """

    def __init__(self, shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> None:
        self.value = torch.empty(shape, dtype=dtype, device=device)
        self.writes = torch.zeros((), dtype=torch.int32, device=device)

    def write(self, value: torch.Tensor) -> None:
        rows = min(value.shape[0], self.value.shape[0])
        self.value[:rows].copy_(value[:rows])
        self.writes.add_(1)

    def write_absent(self) -> None:
        self.value.zero_()
        self.writes.add_(1)

    def write_padded(self, value: torch.Tensor, fill: int = -1) -> None:
        """Copy a variable-width planner prefix into a fixed-width snapshot."""
        self.value.fill_(fill)
        self.value[: value.shape[0], : value.shape[1]].copy_(value)
        self.writes.add_(1)

    def reset(self) -> None:
        """Reset only the counter, outside forward/capture."""
        self.writes.zero_()

    def read(self, rows: int, *, label: str) -> torch.Tensor:
        """Read after the complete forward; reject missing or duplicate writes."""
        count = int(self.writes.cpu())
        if count != 1:
            raise ParityError(f"{label}: expected one fresh device snapshot, got {count}")
        if not 0 < rows <= self.value.shape[0]:
            raise ParityError(f"{label}: snapshot capacity exceeded: rows={rows}, capacity={self.value.shape[0]}")
        return self.value[:rows].detach().cpu().clone()


def gather_sparse_kv(
    cache: torch.Tensor,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    query_ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the KV that attention will consume, in top-k order, not slot order.

    Accepts PA_BSND with one KV head and sparse_block_size=1. Invalid nonnegative
    indices are reported separately instead of silently clamping them into
    valid data. Negative top-k entries are masked padding. Physical slots are
    returned for diagnosis only: independent runs may allocate different blocks.
    """
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("SFA parity requires PA_BSND with one KV head")
    indices = topk.reshape(topk.shape[0], -1).to(torch.long)
    rows = torch.arange(indices.shape[0], device=indices.device)
    owners = (rows[:, None] >= query_ends[None, :]).sum(dim=1)
    blocks = torch.div(indices.clamp_min(0), cache.shape[1], rounding_mode="floor")
    addressable = (owners[:, None] < block_table.shape[0]) & (blocks < block_table.shape[1])
    physical_blocks = block_table[
        owners.clamp(max=block_table.shape[0] - 1)[:, None], blocks.clamp(max=block_table.shape[1] - 1)
    ].to(torch.long)
    slots = physical_blocks * cache.shape[1] + indices.clamp_min(0) % cache.shape[1]
    flat = cache.reshape(-1, cache.shape[-1])
    addressable = addressable & (slots >= 0) & (slots < flat.shape[0])
    valid = (indices >= 0) & addressable
    invalid = (indices >= 0) & ~addressable
    values = flat[slots.clamp(0, flat.shape[0] - 1)]
    values = torch.where(valid[..., None], values, torch.zeros_like(values))
    return values, valid, invalid, torch.where(valid, slots, -1)


@dataclass(frozen=True)
class TensorDifference:
    max_abs: float
    max_rel: float
    mismatches: int
    elements: int


def compare_tensor(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    label: str,
    atol: float = 1e-7,
    rtol: float = 1e-2,
) -> TensorDifference:
    """Compare all elements; integers exactly, floats with explicit tolerance.

    Even matching NaNs/infinities fail: a broken dummy model is not a reference.
    Reports the first bad coordinate and the maximum absolute/relative errors.
    """
    if any(not math.isfinite(tolerance) or tolerance < 0 for tolerance in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise ParityError(
            f"{label}: shape/dtype mismatch: eager={reference.shape}/{reference.dtype}, "
            f"graph={actual.shape}/{actual.dtype}"
        )
    if not reference.numel():
        raise ParityError(f"{label}: empty tensor is not a parity observation")
    if reference.is_floating_point():
        for name, value in (("eager", reference), ("graph", actual)):
            if not torch.isfinite(value).all():
                raise ParityError(f"{label}: {name} contains NaN/Inf; numerical parity is invalid")
        ref, val = reference.double(), actual.double()
        error = (ref - val).abs()
        relative = error / ref.abs().clamp_min(torch.finfo(torch.float64).tiny)
        different = error > atol + rtol * ref.abs()
        result = TensorDifference(float(error.max()), float(relative.max()), int(different.sum()), ref.numel())
    else:
        different = reference != actual
        result = TensorDifference(0.0, 0.0, int(different.sum()), reference.numel())
    if result.mismatches:
        index = tuple(different.nonzero()[0].tolist())
        raise ParityError(
            f"{label}: first_index={index} eager={reference[index].item()} graph={actual[index].item()} "
            f"mismatches={result.mismatches}/{result.elements} "
            f"max_abs={result.max_abs:.6g} max_rel={result.max_rel:.6g} atol={atol} rtol={rtol}",
            label=label,
            index=index,
        )
    return result


def compare_step(reference: dict, actual: dict, *, atol: float, rtol: float) -> None:
    """Fail at the earliest layer/phase, checking step alignment first."""
    label = f"step={actual['step']}"
    for key in ("step", "decode", "rows", "layers"):
        if reference[key] != actual[key]:
            raise ParityError(f"{label}: incomparable {key}: eager={reference[key]}, graph={actual[key]}")
    for key in ("input_ids", "positions", "seq_lens", "query_ends"):
        compare_tensor(reference[key], actual[key], label=f"{label} {key}", atol=0, rtol=0)
    if not reference["tensors"] or list(reference["tensors"]) != list(actual["tensors"]):
        raise ParityError(f"{label}: different probe coverage between eager and graph")
    for key, ref in reference["tensors"].items():
        compare_tensor(ref, actual["tensors"][key], label=f"{label} {key}", atol=atol, rtol=rtol)
