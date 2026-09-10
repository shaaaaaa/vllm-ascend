# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real eight-NPU TP8/DP1/MTP1 model test, not eight independent replicas.

Use --confcutdir=tests/e2e/multicard to avoid unrelated model dependencies.
"""

import importlib.util
from pathlib import Path

import pytest


def test_eight_layer_tp8_eager_full_graph_parity():
    pytest.importorskip("torch_npu")
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("sfa_parity_driver", root / "tools/sfa_full_graph_parity.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    if not Path(driver.DEFAULT_MODEL, "config.json").is_file():
        pytest.skip(f"Requires local model source: {driver.DEFAULT_MODEL}")
    # Device selection happens in fresh children, before importing torch_npu.
    # Do not initialize a device or count a parent's restricted visible set here.
    driver.run_pair(devices="0,1,2,3,4,5,6,7")
