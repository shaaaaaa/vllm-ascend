# SPDX-License-Identifier: Apache-2.0
"""Exercise the startup selector without a native extension or vLLM import."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ENV = "VLLM_ASCEND_DSA_DCP_FREE_SLOT_FIRST"
OP = "configure_dsa_resident_dcp_free_slots"


def planner():
    path = ROOT / "vllm_ascend/distributed/kv_transfer/sparse_offload/resident_sorted_cache.py"
    spec = importlib.util.spec_from_file_location("free_slot_selector_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("VLLM_ASCEND_DSA_RESIDENT_QUERY_MEMBERSHIP", "0")
    monkeypatch.setattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels", Mock(), raising=False)
    return planner()


@pytest.mark.parametrize("value,enabled", [(None, False), ("0", False), ("1", True)])
def test_selection_is_default_off_and_cached_before_capture(setup, monkeypatch, value, enabled):
    configure = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, OP, configure, raising=False)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    setup.configure_resident_kernels()
    monkeypatch.setenv(ENV, "0" if enabled else "1")
    setup.configure_resident_kernels()
    configure.assert_called_once_with(enabled)


def test_old_binary_works_when_disabled_and_fails_when_requested(setup, monkeypatch):
    monkeypatch.delattr(torch.ops._C_ascend, OP, raising=False)
    setup.configure_resident_kernels()
    setup.configure_resident_kernels.cache_clear()
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(RuntimeError, match="Rebuild.*free-slot-first"):
        setup.configure_resident_kernels()
    configure = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, OP, configure, raising=False)
    setup.configure_resident_kernels()
    configure.assert_called_once_with(True)


def test_native_freeze_failure_is_propagated_and_not_cached(setup, monkeypatch):
    configure = Mock(side_effect=[RuntimeError("policy is frozen"), None])
    monkeypatch.setattr(torch.ops._C_ascend, OP, configure, raising=False)
    with pytest.raises(RuntimeError, match="frozen"):
        setup.configure_resident_kernels()
    setup.configure_resident_kernels()
    assert configure.call_count == 2
