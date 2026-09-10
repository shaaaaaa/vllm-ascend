# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP8 diagnosis of a residual mismatch, separate from graph acceptance.

Additional observations can affect compiler fusion. This test cannot replace
test_sfa_full_graph_parity.py even when its instrumented comparison succeeds.
"""

import importlib.util
from pathlib import Path

import pytest


def test_eight_layer_tp8_residual_trace():
    pytest.importorskip("torch_npu")
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("sfa_parity_driver", root / "tools/sfa_full_graph_parity.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    if not Path(driver.DEFAULT_MODEL, "config.json").is_file():
        pytest.skip(f"Requires local model source: {driver.DEFAULT_MODEL}")
    driver.run_pair(devices="0,1,2,3,4,5,6,7", trace_residual=True)
