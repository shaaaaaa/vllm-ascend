# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser):
    parser.addoption('--redesign-build-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'build')
    parser.addoption('--matched-original-build-dir', type=Path)
    parser.addoption('--matched-lmcache-ascend-dir', type=Path)
