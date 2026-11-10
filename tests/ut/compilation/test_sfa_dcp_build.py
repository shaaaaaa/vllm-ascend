# SPDX-License-Identifier: Apache-2.0
"""Verify ACLNN source selection and host binding arguments without a CANN SDK."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
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


def test_sfa_torch_adapter_arguments_compile(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("requires a host C++ compiler")

    native = ROOT / "csrc"
    bridge = (native / "aclnn_torch_adapter/op_api_common.h").read_text(encoding="utf-8")
    start = bridge.index("template <typename... Ts>\nconstexpr auto ConvertTypes(")
    converter = bridge[start : bridge.index("\n}", start) + 2]
    source = tmp_path / "sfa_adapter.cpp"
    # Keep the real Torch types and ConvertTypes contract in the launch stub.
    source.write_text(
        f"""#include <ATen/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/util/Optional.h>
#include <c10/util/string_view.h>
#include <tuple>

template <typename T>
T ConvertType(T value) {{ return value; }}

{converter}

#define EXEC_NPU_CMD(op, ...) static_cast<void>(ConvertTypes(__VA_ARGS__))
#include "sparse_flash_attention_torch_adpt.h"
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-c",
            "-I",
            str(Path(torch.__file__).resolve().parent / "include"),
            "-I",
            str(native / "sparse_flash_attention"),
            str(source),
            "-o",
            str(tmp_path / "sfa_adapter.o"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
