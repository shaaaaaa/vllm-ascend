# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device snapshots and CPU comparisons for the opt-in SFA parity worker.

There are no host reads in ``DeviceSnapshot.write`` or ``gather_sparse_kv``.
Their device operations can be captured by the target's existing root graph.
Nothing in this module is enabled by the normal serving worker.
"""

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

WEIGHT_HASH_CHUNK_BYTES = 8 * 1024 * 1024
NPU_BASE_FORMATS = (0, 2)  # NCHW, ND


class ParityError(AssertionError):
    """A mismatch, invalid reference, or incomplete observation."""

    def __init__(self, message: str, *, label: str = "", index: tuple[int, ...] = ()) -> None:
        super().__init__(message)
        self.label = label
        self.index = index


def _weight_bytes_on_cpu(value: torch.Tensor, npu_format: int | None) -> torch.Tensor:
    """Read without slicing or converting packed NZ weights on the device."""
    if npu_format is not None and npu_format not in NPU_BASE_FORMATS:
        storage = value.untyped_storage()
        # Internal-format padding is not model data and may be uninitialized.
        # Only hash raw storage when every byte belongs to this tensor. Never
        # silently include padding, adjacent tensors or skip such a weight.
        if (
            not value.is_contiguous()
            or value.storage_offset() != 0
            or storage.nbytes() != value.numel() * value.element_size()
        ):
            raise ParityError("Internal-format weight must cover complete unpadded storage for bytewise hashing")
        host_storage = torch.UntypedStorage(storage.nbytes(), device="cpu")
        # Do NOT use storage.cpu(): torch_npu overrides it to construct a typed
        # tensor and invoke Tensor.cpu(), reintroducing Identity/TransData.
        host_storage.copy_(storage, non_blocking=False)
        return torch.empty(0, dtype=torch.uint8, device="cpu").set_(host_storage)
    # Ordinary layouts may be views. Copy the whole logical tensor first; all
    # contiguous/reshape/dtype-view operations below execute on the CPU.
    return value.cpu().contiguous().reshape(-1).view(torch.uint8)


def weight_fingerprint(model: torch.nn.Module) -> str:
    """Hash every state byte, using at most one host tensor plus a hash chunk.

    Eager and graph must have identical logical metadata and device formats.
    Packed internal layouts are compared byte-for-byte without a format cast.
    This startup diagnostic never mutates model weights or executes in replay.
    """
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        npu_format = None
        try:
            value = value.detach()
            if value.device.type == "npu":
                import torch_npu  # Lazy: CPU-only tests need no NPU runtime.

                npu_format = torch_npu.get_npu_format(value)
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}:format={npu_format}".encode())
            raw = _weight_bytes_on_cpu(value, npu_format)
            for chunk in raw.split(WEIGHT_HASH_CHUNK_BYTES):
                digest.update(chunk.numpy().tobytes())
            # Release the whole host tensor before allocating the next one.
            del chunk, raw
        except Exception as exc:
            raise ParityError(
                f"weight fingerprint name={name} shape={tuple(value.shape)} dtype={value.dtype} "
                f"device={value.device} format={npu_format}: {type(exc).__name__}: {exc}"
            ) from exc
    return digest.hexdigest()


def coordinated_check(check: Callable[[], Any], *, group: Any, phase: str) -> Any:
    """Agree diagnostic failures at model boundaries before further collectives.

    ``check`` must not itself enter distributed collectives. This coordinates
    reference I/O/shape/numerical failures, not a failed or hung NPU/HCCL kernel;
    the worker supervisor remains responsible for device/process failures.
    """
    result, error = None, None
    try:
        result = check()
    except Exception as exc:
        error = f"rank={group.rank_in_group} {type(exc).__name__}: {exc}"
    statuses = [(phase, error)]
    if group.world_size > 1:
        statuses = [None] * group.world_size
        torch.distributed.all_gather_object(statuses, (phase, error), group=group.cpu_group)
    if any(item[0] != phase for item in statuses):
        raise ParityError(f"TP parity diagnostic phases diverged: {statuses}")
    failures = [item[1] for item in statuses if item[1] is not None]
    if failures:
        raise ParityError(f"{phase}: " + " | ".join(failures))
    return result


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
    label = f"rank={actual['rank']} step={actual['step']}"
    for key in ("rank", "tp_size", "step", "decode", "rows", "layers"):
        if reference[key] != actual[key]:
            raise ParityError(f"{label}: incomparable {key}: eager={reference[key]}, graph={actual[key]}")
    for key in ("input_ids", "positions", "seq_lens", "query_ends"):
        compare_tensor(reference[key], actual[key], label=f"{label} {key}", atol=0, rtol=0)
    if not reference["tensors"] or list(reference["tensors"]) != list(actual["tensors"]):
        raise ParityError(f"{label}: different probe coverage between eager and graph")
    for key, ref in reference["tensors"].items():
        compare_tensor(ref, actual["tensors"][key], label=f"{label} {key}", atol=atol, rtol=rtol)
