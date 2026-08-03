# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent, graph-external snapshots for target SFA crash localization."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from vllm_ascend.diagnostic_utils import (
    atomic_torch_save,
    cpu_snapshot,
    snapshot_cache_components,
)

MTP_DRAFT_DIAG_ROOT = (
    Path(tempfile.gettempdir()) / "vllm_ascend_mtp_draft_diag"
)

TARGET_SFA_DIAG_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_target_step_lock = Lock()
_target_step_id = 0


@dataclass(frozen=True)
class TargetSFADiagnosticSession:
    step_id: int
    rank: int
    output_dir: Path


def layer_index(layer_name: str) -> int:
    match = _LAYER_INDEX.search(layer_name)
    return int(match.group(1)) if match is not None else -1


def target_sfa_session(
    context: Any,
    layer_name: str,
    rank: int,
    *,
    begin: bool = False,
) -> tuple[TargetSFADiagnosticSession, int]:
    """Return one session shared by every target SFA layer in this forward."""
    global _target_step_id

    index = layer_index(layer_name)
    session = getattr(context, "_target_sfa_diag_session", None)
    if session is None or (begin and index == 0):
        with _target_step_lock:
            _target_step_id += 1
            step_id = _target_step_id
        rank_dir = MTP_DRAFT_DIAG_ROOT / (
            f"rank_{rank}_pid_{os.getpid()}"
        )
        output_dir = rank_dir / "target" / f"slot_{step_id % 2}"
        session = TargetSFADiagnosticSession(
            step_id=step_id,
            rank=rank,
            output_dir=output_dir,
        )
        context._target_sfa_diag_session = session
        atomic_torch_save(
            {
                "schema_version": TARGET_SFA_DIAG_SCHEMA_VERSION,
                "step_id": step_id,
                "rank": rank,
                "pid": os.getpid(),
                "output_dir": str(output_dir),
            },
            rank_dir / "latest_target.pt",
        )
    return session, index


def target_sfa_path(
    session: TargetSFADiagnosticSession,
    layer: int,
    phase: str,
) -> Path:
    return session.output_dir / f"layer_{layer:02d}_{phase}.pt"


def target_metadata_snapshot(metadata: Any) -> dict[str, Any]:
    """Copy the metadata needed to replay resident planning and SFA."""
    names = (
        "attn_state",
        "num_actual_tokens",
        "num_decode_tokens",
        "req_ids",
        "decode_request_ids_compact",
        "seq_lens",
        "seq_lens_cpu",
        "cum_query_lens",
        "slot_mapping",
        "indexer_slot_mapping",
        "block_table",
        "indexer_block_table",
        "split_boundary",
        "decode_split_boundary",
        "decode_remap_boundary",
        "decode_req_indices",
        "decode_req_indices_cpu",
        "decode_current_positions_cpu",
        "decode_scratch_base",
        "decode_scratch_capacity",
        "resident_state_indices",
        "resident_state_generations",
    )
    return {
        name: cpu_snapshot(getattr(metadata, name, None))
        for name in names
    }


def active_resident_state_snapshot(
    state: Any,
    metadata: Any,
) -> dict[str, Any] | None:
    """Copy only active and request-dummy rows from persistent resident state."""
    if state is None:
        return None
    state_indices = getattr(metadata, "resident_state_indices", None)
    if state_indices is None:
        return None
    active = [
        int(value)
        for value in state_indices.detach().cpu().reshape(-1).tolist()
    ]
    request_count = len(
        getattr(metadata, "decode_request_ids_compact", None) or []
    )
    active = active[:request_count]
    rows = sorted(
        {
            value
            for value in active
            if 0 <= value < int(state.tokens.shape[0])
        }
        | {
            int(state.dummy_state_base) + request
            for request in range(request_count)
            if int(state.dummy_state_base) + request
            < int(state.tokens.shape[0])
        }
    )
    indices = torch.tensor(
        rows,
        dtype=torch.long,
        device=state.tokens.device,
    )
    return {
        "row_indices": rows,
        "dummy_state_base": int(state.dummy_state_base),
        "tokens": state.tokens.index_select(0, indices).detach().cpu().clone(),
        "slots": state.slots.index_select(0, indices).detach().cpu().clone(),
        "counts": state.counts.index_select(0, indices).detach().cpu().clone(),
        "generations": (
            state.generations.index_select(0, indices).detach().cpu().clone()
        ),
    }


def topk_physical_block_ids(
    topk_indices: torch.Tensor,
    row_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    target_slots: torch.Tensor,
    *,
    block_size: int,
    actual_rows: int,
    request_count: int,
) -> list[int]:
    """Resolve the physical latent blocks used or written by this SFA call."""
    topk = topk_indices[:actual_rows].detach().cpu().reshape(actual_rows, -1)
    row_requests = (
        row_req_indices[:actual_rows].detach().cpu().reshape(-1).tolist()
    )
    table = block_table[:request_count].detach().cpu()
    block_ids: set[int] = set()
    for row, request in enumerate(row_requests):
        request = int(request)
        if request < 0 or request >= table.shape[0]:
            continue
        for token in topk[row].tolist():
            logical_block = int(token) // block_size
            if 0 <= logical_block < table.shape[1]:
                physical_block = int(table[request, logical_block])
                if physical_block >= 0:
                    block_ids.add(physical_block)
    slots = target_slots[:request_count].detach().cpu().reshape(-1)
    block_ids.update(
        int(slot) // block_size
        for slot in slots.tolist()
        if int(slot) >= 0
    )
    return sorted(block_ids)


def target_cache_snapshot(
    cache: Any,
    block_ids: list[int],
) -> Any:
    return snapshot_cache_components(
        cache,
        latent_block_ids=block_ids,
        indexer_block_ids=[],
    )


def target_tail_boundary(
    session: TargetSFADiagnosticSession | None,
    phase: str,
    value: Any,
) -> None:
    """Fence and save target work after the last per-layer SFA boundary."""
    if session is None:
        return
    logger.warning(
        "[TARGET_SFA_DIAG] step=%d phase=%s sync started",
        session.step_id,
        phase,
    )
    try:
        torch.npu.synchronize()
    except Exception as error:
        payload = {
            "schema_version": TARGET_SFA_DIAG_SCHEMA_VERSION,
            "step_id": session.step_id,
            "rank": session.rank,
            "pid": os.getpid(),
            "phase": phase,
            "error_type": type(error).__qualname__,
            "error": str(error),
        }
        atomic_torch_save(
            payload,
            session.output_dir / f"target_{phase}_failure.pt",
        )
        atomic_torch_save(
            payload,
            session.output_dir.parent / "latest_failure.pt",
        )
        logger.exception(
            "[TARGET_SFA_DIAG] step=%d phase=%s sync failed",
            session.step_id,
            phase,
        )
        raise
    path = session.output_dir / f"target_{phase}.pt"
    atomic_torch_save(
        {
            "schema_version": TARGET_SFA_DIAG_SCHEMA_VERSION,
            "step_id": session.step_id,
            "rank": session.rank,
            "pid": os.getpid(),
            "phase": phase,
            "value": cpu_snapshot(value),
        },
        path,
    )
    logger.warning(
        "[TARGET_SFA_DIAG] step=%d phase=%s sync passed; saved=%s",
        session.step_id,
        phase,
        path,
    )
