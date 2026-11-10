# SPDX-License-Identifier: Apache-2.0
"""Verify the native build's ACLNN source selection without a CANN SDK."""

import shutil
import subprocess

import pytest
from sfa_test_support import ROOT


@pytest.mark.parametrize("dcp", [False, True], ids=["original", "a2-with-dcp"])
def test_sfa_aclnn_sources_match_selected_operators(tmp_path, dcp):
    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    if not cmake or not ninja:
        pytest.skip("requires CMake and Ninja")

    native = ROOT / "csrc"
    ops = ["sparse_flash_attention", "sparse_flash_attention_dcp"]
    for op in ops:
        shutil.copytree(native / op / "op_host", tmp_path / op / "op_host")
    selected = ops if dcp else ops[:1]
    build = (native / "CMakeLists.txt").read_text(encoding="utf-8")
    source_selection = build[build.index("set(OP_LIST)"):build.index("if (base_aclnn_inner_srcs)")]
    (tmp_path / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.20)
project(sfa_aclnn_sources NONE)
set(BUILD_OPEN_PROJECT ON)
set(ASCEND_OP_NAME "{';'.join(selected)}")
include("{(native / 'cmake/func.cmake').as_posix()}")
function(add_ops_compile_options)
endfunction()
function(target_include_directories)
endfunction()
add_custom_target(op_host_aclnn)
add_custom_target(op_host_aclnnInner)
add_custom_target(op_host_aclnnExc)
add_custom_target(optiling)
add_custom_target(opsproto)
set(ASCEND_AUTOGEN_DIR autogen)
{source_selection}
file(WRITE "${{CMAKE_BINARY_DIR}}/aclnn_sources.txt" "${{generate_aclnn_srcs}}")
""",
        encoding="utf-8",
    )
    out = tmp_path / "build"
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(out), "-G", "Ninja"], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    sources = (out / "aclnn_sources.txt").read_text(encoding="utf-8").split(";")
    assert sources == [f"autogen/aclnn_{op}.cpp" for op in selected]
