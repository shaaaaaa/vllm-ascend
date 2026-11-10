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


def test_sfa_query_tiles_use_only_cores_with_work(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("requires a host C++ compiler")

    kernel = ROOT / "csrc/sparse_flash_attention/op_kernel/sparse_flash_attention_kernel_mla.h"
    header = kernel.read_text(encoding="utf-8")
    functions = []
    for name in ("InitCalcParamsEach", "GetBalanceActualSeqLengths", "GetAxisStartIdx"):
        signature = header.index(f"SparseFlashAttentionMla<SFAT>::{name}(")
        start = header.rindex("template <typename SFAT>", 0, signature)
        functions.append(header[start : header.index("\n}\n", signature) + 3])
    source = tmp_path / "sfa_split.cpp"
    source.write_text(
        """#include <cstdint>
#include <cstdio>
#include <vector>
#define __aicore__

enum class SFA_LAYOUT { TND };
template <typename T> struct GlobalTensor {
    std::vector<T> data;
    T GetValue(uint32_t row) { return data[row]; }
};

template <typename SFAT>
struct SparseFlashAttentionMla {
    static constexpr auto LAYOUT_T = SFA_LAYOUT::TND;
    struct {
        uint32_t batchSize = 0, qSeqSize = 0, gSize = 64, mBaseSize = 64, kvHeadNum = 1;
        uint32_t bN2Start = 0, gS1Start = 0, bN2End = 0, gS1End = 0;
        uint32_t s2Start = 0, s2End = 0, coreStartKVSplitPos = 0, actualLenDimsQ = 0;
        bool needInit = false;
    } constInfo;
    uint32_t aiCoreIdx = 0, usedCoreNum = 0, cores = 0;
    uint32_t kvHeadNum = 1;
    GlobalTensor<int32_t> actualSeqLengthsQGm;
    uint32_t GetBlockNum() { return cores; }
    uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t>&, uint32_t);
    void GetAxisStartIdx(uint32_t, uint32_t, uint32_t);
    void InitCalcParamsEach();
};
"""
        + "\n".join(functions)
        + """
int main() {
    for (const auto &lengths : std::vector<std::vector<uint32_t>>{
             {0, 0}, {2}, {0, 2, 1, 0, 29, 0}, {256}}) {
        for (uint32_t cores : {1u, 2u, 20u, 24u}) {
            GlobalTensor<int32_t> ends;
            uint32_t total = 0, emptyCores = 0;
            for (auto count : lengths) {
                total += count;
                ends.data.push_back(total);
            }
            std::vector<uint32_t> visits(total);
            for (uint32_t core = 0; core < cores; ++core) {
                SparseFlashAttentionMla<void> op;
                op.constInfo.batchSize = lengths.size();
                op.constInfo.qSeqSize = total;
                op.aiCoreIdx = core;
                op.cores = op.usedCoreNum = cores;
                op.actualSeqLengthsQGm = ends;
                op.InitCalcParamsEach();
                if (core >= op.usedCoreNum) continue;
                const auto &info = op.constInfo;
                uint32_t prefix = 0, assigned = 0;
                for (uint32_t batch = 0; batch < lengths.size(); ++batch) {
                    if (info.bN2Start <= batch && batch <= info.bN2End && lengths[batch]) {
                        uint32_t first = batch == info.bN2Start ? info.gS1Start : 0;
                        uint32_t end = batch == info.bN2End ? info.gS1End + 1 : lengths[batch];
                        if (first >= end || end > lengths[batch]) return 2;
                        for (uint32_t row = first; row < end; ++row) ++visits[prefix + row];
                        assigned += end - first;
                    }
                    prefix += lengths[batch];
                }
                emptyCores += assigned == 0;
            }
            for (auto count : visits) {
                if (count == 1) continue;
                std::fprintf(stderr, "%u query rows on %u cores: row visited %u times\\n",
                             total, cores, count);
                return 1;
            }
            if (emptyCores != 0) {
                std::fprintf(stderr, "%u query rows on %u cores: %u active cores had no work\\n",
                             total, cores, emptyCores);
                return 3;
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "sfa_split.exe"
    built = subprocess.run(
        [*compiler, "-std=c++17", str(source), "-o", str(executable)],
        capture_output=True, text=True, timeout=120,
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([executable], capture_output=True, text=True, timeout=10)
    assert ran.returncode == 0, ran.stderr
