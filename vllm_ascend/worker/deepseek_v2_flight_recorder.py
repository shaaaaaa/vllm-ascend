#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Host-side writer for the opt-in GLM-5.1/DeepSeek flight recorder."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch
from vllm.model_executor.models.deepseek_v2_diagnostics import DeepseekV2Trace

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def flight_recorder_enabled() -> bool:
    return (
        os.getenv("VLLM_ASCEND_GL51_DEEP_DIAG", "0").strip().lower()
        in _TRUE_VALUES
    )


def flight_recorder_run_name() -> str:
    value = os.getenv("VLLM_ASCEND_GL51_DEEP_DIAG_RUN", "").strip()
    if not value:
        raise RuntimeError(
            "VLLM_ASCEND_GL51_DEEP_DIAG_RUN must name this run "
            "(for example baseline or candidate)"
        )
    return _SAFE_COMPONENT.sub("_", value)


def flight_recorder_output_dir() -> Path:
    value = os.getenv("VLLM_ASCEND_GL51_DEEP_DIAG_DIR", "").strip()
    if not value:
        raise RuntimeError(
            "VLLM_ASCEND_GL51_DEEP_DIAG_DIR must be set when "
            "VLLM_ASCEND_GL51_DEEP_DIAG=1"
        )
    return Path(value).expanduser().resolve()


def _cpu_exact(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


def write_flight_record(
    trace: DeepseekV2Trace,
    *,
    request_id: str,
    dp_rank: int,
    tp_rank: int,
) -> Path:
    """Transfer all snapshots to CPU and atomically save one worker bundle."""
    output_dir = flight_recorder_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = flight_recorder_run_name()
    safe_request_id = _SAFE_COMPONENT.sub("_", request_id)

    seen_labels: set[str] = set()
    tensor_records: list[dict[str, Any]] = []
    for label, tensor in trace.tensors:
        if label in seen_labels:
            raise RuntimeError(f"Duplicate flight-recorder tensor label: {label}")
        seen_labels.add(label)
        tensor_records.append(
            {
                "label": label,
                "dtype": str(tensor.dtype),
                "shape": tuple(tensor.shape),
                "stride": tuple(tensor.stride()),
                "tensor": _cpu_exact(tensor),
            }
        )

    bundle = {
        "schema_version": 1,
        "run_name": run_name,
        "request_id": request_id,
        "dp_rank": int(dp_rank),
        "tp_rank": int(tp_rank),
        "metadata": trace.metadata,
        "values": list(trace.values),
        "tensors": tensor_records,
    }
    filename = (
        f"gl51-deep-{run_name}-dp{dp_rank}-tp{tp_rank}-"
        f"{safe_request_id}.pt"
    )
    final_path = output_dir / filename
    temporary_path = final_path.with_suffix(
        f"{final_path.suffix}.tmp-{os.getpid()}"
    )
    torch.save(bundle, temporary_path)
    os.replace(temporary_path, final_path)
    return final_path
