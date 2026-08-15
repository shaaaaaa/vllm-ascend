# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "benchmarks" / "layerwise_prefill_cache" / "run_prefill_p.sh"
BASH = shutil.which("bash")


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_launcher(
    tmp_path: Path,
    mode: str,
    **overrides: str,
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command in ("rm", "mkdir"):
        _write_executable(fake_bin / command, "#!/bin/sh\nexit 0\n")
    _write_executable(fake_bin / "date", "#!/bin/sh\nprintf 'test-run\\n'\n")
    _write_executable(
        fake_bin / "vllm",
        "#!/bin/sh\nprintf 'VLLM_ARG=<%s>\\n' \"$@\"\n",
    )
    _write_executable(
        fake_bin / "tee",
        "#!/bin/sh\nwhile IFS= read -r line; do printf '%s\\n' \"$line\"; done\n",
    )

    env = os.environ.copy()
    env.update(overrides)
    env["PATH"] = str(fake_bin)
    return subprocess.run(
        [BASH, str(SCRIPT), mode],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.skipif(BASH is None, reason="bash is required")
@pytest.mark.parametrize("chunk_size", [2048, 4096, 8192, 16384])
def test_layerwise_prefill_on_captures_configured_chunk_graph(
    tmp_path: Path,
    chunk_size: int,
) -> None:
    result = _run_launcher(tmp_path, "on", CHUNK_SIZE=str(chunk_size))

    assert result.returncode == 0, result.stderr
    assert f"CHUNK_SIZE={chunk_size}" in result.stdout
    assert "GPU_MEMORY_UTILIZATION=0.94" in result.stdout
    assert "PREFILL_GRAPH=true" in result.stdout
    expected = f'{{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[{chunk_size}]}}'
    assert f"COMPILATION_CONFIG={expected}" in result.stdout
    assert f"VLLM_ARG=<{expected}>" in result.stdout


@pytest.mark.skipif(BASH is None, reason="bash is required")
def test_layerwise_prefill_off_keeps_prefill_eager(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, "off")

    assert result.returncode == 0, result.stderr
    assert "GPU_MEMORY_UTILIZATION=0.96" in result.stdout
    assert "PREFILL_GRAPH=false" in result.stdout
    expected = '{"cudagraph_mode":"PIECEWISE"}'
    assert f"COMPILATION_CONFIG={expected}" in result.stdout
    assert f"VLLM_ARG=<{expected}>" in result.stdout


@pytest.mark.skipif(BASH is None, reason="bash is required")
def test_layerwise_prefill_graph_defaults_can_be_overridden(
    tmp_path: Path,
) -> None:
    result = _run_launcher(
        tmp_path,
        "on",
        PREFILL_GRAPH="false",
        GPU_MEMORY_UTILIZATION="0.91",
    )

    assert result.returncode == 0, result.stderr
    assert "GPU_MEMORY_UTILIZATION=0.91" in result.stdout
    assert "PREFILL_GRAPH=false" in result.stdout
    assert 'COMPILATION_CONFIG={"cudagraph_mode":"PIECEWISE"}' in result.stdout


@pytest.mark.skipif(BASH is None, reason="bash is required")
def test_layerwise_prefill_graph_rejects_non_boolean_value(
    tmp_path: Path,
) -> None:
    result = _run_launcher(tmp_path, "on", PREFILL_GRAPH="1")

    assert result.returncode == 2
    assert "PREFILL_GRAPH must be true or false" in result.stderr
