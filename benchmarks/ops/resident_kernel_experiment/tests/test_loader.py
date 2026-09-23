"""Dependency loading order without a device or compiled extension."""
import json
import sys
from types import ModuleType

import pytest
import resident_experiment as experiment


@pytest.mark.parametrize("subdir", ["lib", ""])
def test_kernel_dependency_loaded_before_binding(tmp_path, monkeypatch, subdir):
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    stamp = {"source_sha256": experiment.source_digest()}
    (tmp_path / "build-info.json").write_text(json.dumps(stamp))
    kernel = tmp_path / subdir / "libresident_experiment_kernels.so"
    kernel.parent.mkdir(exist_ok=True)
    kernel.touch()
    binding = tmp_path / "libresident_experiment_ops.so"
    binding.touch()
    loaded = []
    monkeypatch.setattr(experiment.torch.ops, "load_library", loaded.append)
    assert experiment.load_library(tmp_path) == stamp
    assert loaded == [str(kernel.resolve()), str(binding.resolve())]


def test_missing_kernel_fails_before_loading_binding(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    (tmp_path / "build-info.json").write_text(json.dumps({"source_sha256": experiment.source_digest()}))
    loaded = []
    monkeypatch.setattr(experiment.torch.ops, "load_library", loaded.append)
    with pytest.raises(FileNotFoundError, match="kernel library is missing"):
        experiment.load_library(tmp_path)
    assert loaded == []
