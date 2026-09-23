"""Startup selector tests without vLLM/CANN imports."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def load_planner():
    path = ROOT / "vllm_ascend/distributed/kv_transfer/sparse_offload/resident_sorted_cache.py"
    spec = importlib.util.spec_from_file_location("serving_resident_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value,expected", [(None, True), ("0", False), ("1", True)])
def test_startup_selection_and_no_reread(monkeypatch, value, expected):
    name = "VLLM_ASCEND_DSA_RESIDENT_EXACT_KERNELS"
    monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(name, value)
    configure = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels",
                        configure, raising=False)
    planner = load_planner()
    planner.configure_resident_kernels()
    monkeypatch.setenv(name, "0" if expected else "1")
    planner.configure_resident_kernels()
    configure.assert_called_once_with(expected)


def test_missing_extension_fails_before_capture(monkeypatch):
    monkeypatch.delattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels",
                        raising=False)
    with pytest.raises(RuntimeError, match="Rebuild"):
        load_planner().configure_resident_kernels()


def test_configuration_failure_is_not_cached(monkeypatch):
    configure = Mock(side_effect=[RuntimeError("restart"), None])
    monkeypatch.setattr(torch.ops._C_ascend, "configure_dsa_resident_exact_kernels",
                        configure, raising=False)
    planner = load_planner()
    with pytest.raises(RuntimeError, match="restart"):
        planner.configure_resident_kernels()
    planner.configure_resident_kernels()
    assert configure.call_count == 2
