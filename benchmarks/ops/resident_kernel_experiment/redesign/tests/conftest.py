# SPDX-License-Identifier: Apache-2.0
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption('--redesign-build-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'build')
