# SPDX-License-Identifier: Apache-2.0
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def test_configure_passes_nonempty_build_type(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('ASCEND_HOME_PATH', str(tmp_path / 'cann'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        __version__='test', utils=SimpleNamespace(cmake_prefix_path='torch')))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(
        __version__='test', __file__=str(tmp_path / 'torch_npu/__init__.py')))
    monkeypatch.setitem(sys.modules, 'native', SimpleNamespace(
        HERE=root, source_digest=lambda: 'test'))
    monkeypatch.setattr(sys, 'argv', [str(root / 'build.py'), '--soc',
                                    'ascend910b3', '--build-dir', str(tmp_path)])
    commands = []
    monkeypatch.setattr(subprocess, 'run', lambda args, **kwargs: commands.append(args))
    runpy.run_path(str(root / 'build.py'), run_name='__main__')
    # The explicit cache argument also overrides a reused empty CMake cache entry.
    assert '-DCMAKE_BUILD_TYPE=Release' in commands[0]
    assert commands[1][:2] == ['cmake', '--build']
