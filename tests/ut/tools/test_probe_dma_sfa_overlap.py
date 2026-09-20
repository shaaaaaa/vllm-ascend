# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the standalone NPU DMA/SFA profile probe."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[3] / "tools" / "probe_dma_sfa_overlap.py"
    spec = importlib.util.spec_from_file_location("probe_dma_sfa_overlap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validates_sfa_dimensions(probe):
    args = SimpleNamespace(
        query_tokens=512,
        context_tokens=16384,
        topk=2048,
        query_heads=16,
        copy_mib=64,
        warmups=3,
        repeats=5,
        profile_repeats=2,
    )
    probe.validate(args)
    args.context_tokens = 16383
    with pytest.raises(ValueError, match="multiple of 128"):
        probe.validate(args)


def test_trace_candidate_names_do_not_count_cpu_annotations(probe, tmp_path):
    path = tmp_path / "trace_view.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"ph": "X", "name": "SparseFlashAttention", "cat": "NPU", "pid": 1},
                    {"ph": "X", "name": "MemcpyAsync", "cat": "NPU", "pid": 1},
                    {"ph": "X", "name": "SFA_COMPUTE", "cat": "CPU", "pid": 2},
                    {"ph": "X", "name": "PINNED_DMA_H2D", "cat": "CPU", "pid": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    found = probe.trace_candidates(path)
    assert sum(found["sfa"].values()) == 1
    assert sum(found["memcpy"].values()) == 1
