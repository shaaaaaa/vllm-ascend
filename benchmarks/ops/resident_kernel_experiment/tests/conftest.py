import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def pytest_addoption(parser):
    parser.addoption("--resident-build-dir", default=str(HERE / "build"))
    parser.addoption("--resident-device", type=int, default=0)


@pytest.fixture(scope="session")
def native(request):
    pytest.importorskip("torch_npu", reason="native resident tests require torch-npu/CANN")
    import torch
    from resident_experiment import load_library

    if not torch.npu.is_available():
        pytest.skip("no NPU is available")
    torch.npu.set_device(request.config.getoption("--resident-device"))
    load_library(Path(request.config.getoption("--resident-build-dir")))
    return torch.device("npu", torch.npu.current_device())
