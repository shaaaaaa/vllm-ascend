# SPDX-License-Identifier: Apache-2.0
"""Execute the changed union headers at the boundaries of the removed statistics.

The scalar host model executes the real PartitionUnion and ReadPair/Publish code.
ReadPair's unchanged MTP sort is an observation boundary, not a modeled device
sort. Bitmap/finish execute their real header. This checks arithmetic, indexing,
flags and final IDs/status; it does not validate CANN codegen or NPU barriers.
An optional Git reference compiles the original headers without vendoring them.
"""

import argparse
import ctypes
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_selected_bitmap import GuardedArray

ROOT = Path(__file__).resolve().parents[3]
KERNEL_DIR = Path("csrc/sparse_prefill_native")
HEADERS = (
    "sparse_prefill_plan_services.h",
    "sparse_prefill_local_union_services.h",
    "sparse_prefill_bitmap_services.h",
)
STUB = Path(__file__).with_name("selected_bitmap_host_stub.h")
SENTINEL = 0x5A5A5A5A
SOURCE_KEY_BASE = 0x401FFFFF

HOST_EXTRA = r"""
namespace AscendC {
inline void GatherMask(LocalTensor<uint32_t> dst, LocalTensor<uint32_t> src, uint8_t pattern,
                       bool reduce, uint32_t mask, GatherMaskParams params, uint64_t& selected) {
    if (pattern != 1 || reduce || mask || params.src0BlockStride != 1 ||
        params.src0RepeatStride != 8 || params.src1RepeatStride != 0)
        throw std::runtime_error("Unexpected score GatherMask arguments");
    selected = params.repeatTimes * 32U;
    for (uint32_t i = 0; i < selected; ++i) dst.SetValue(i, src.GetValue(i * 2U));
}
template <typename T, bool D, bool S>
void DataCopy(Tensor<T, D> dst, Tensor<T, S> src, DataCopyParams params) {
    const uint32_t count = params.bytes * 32U / sizeof(T);
    for (uint32_t i = 0; i < params.blocks; ++i) {
        DataCopy(dst[i * (count + params.destinationStride * 32U / sizeof(T))],
                 src[i * (count + params.sourceStride * 32U / sizeof(T))], count);
    }
}
struct MrgSort4Info {
    uint32_t elementLengths[4], validBit, repeatTimes;
    bool ifExhaustedSuspension;
};
template <typename T> struct MrgSortSrcList { LocalTensor<T> src1, src2, src3, src4; };
// These unchanged local merge methods are parsed but not invoked by this test.
template <typename T> void MrgSort(LocalTensor<T>, MrgSortSrcList<T>, MrgSort4Info);
template <typename T> void Relu(LocalTensor<T>, LocalTensor<T>, uint32_t);
template <typename T> void Compare(LocalTensor<uint8_t>, LocalTensor<T>, LocalTensor<T>, CMPMODE, uint32_t);
}
"""

MTP_BOUNDARY = r"""
#pragma once
#include "kernel_operator.h"
namespace LIMtpC8ServiceVec {
using namespace AscendC;
constexpr uint32_t LONG_SOURCE_MASK = (1U << 21U) - 1U;
inline std::vector<uint32_t> observedSortInput;
inline void SortAll(LocalTensor<float>& src, LocalTensor<float>&, int64_t count) {
    auto bits = src.ReinterpretCast<uint32_t>();
    observedSortInput.clear();
    for (int64_t i = 0; i < count; ++i) observedSortInput.push_back(bits.GetValue(i));
}
void ExtractScoreBits(const LocalTensor<uint32_t>&, const LocalTensor<uint32_t>&, int64_t);
void ExtractIndex(const LocalTensor<uint32_t>&, const LocalTensor<uint32_t>&, int64_t);
}
"""

WRAPPER = r"""
#include "kernel_operator.h"
#include <string>
#define private public
#include "sparse_prefill_local_union_services.h"
#undef private
#include "sparse_prefill_bitmap_services.h"

namespace {
std::string lastError;
void Reset() {
    AscendC::regions.clear();
    AscendC::watched = {};
    AscendC::progressions = 0;
    lastError.clear();
}
}
extern "C" {
const char* union_error() { return lastError.c_str(); }
uint32_t union_progressions() { return AscendC::progressions; }
int union_partition(int32_t* lists, int32_t* descriptors, int32_t* membership,
                    int32_t* counters, uint32_t history, uint32_t query,
                    uint32_t width, uint32_t adaptive) {
    using namespace AscendC;
    Reset();
    const uint32_t cohorts = (query + 1U) / 2U;
    Register(lists, cohorts * 4096U * sizeof(int32_t));
    Register(descriptors, cohorts * 8U * sizeof(int32_t));
    Register(membership, 32U * width * sizeof(int32_t));
    Register(counters, 32U * 8U * sizeof(int32_t));
    try {
        for (int block = 31; block >= 0; --block) {
            blockIndex = block;
            TPipe pipe;
            SparsePrefillPlan::PartitionUnion op;
            op.Init(reinterpret_cast<GM_ADDR>(lists), reinterpret_cast<GM_ADDR>(descriptors),
                    reinterpret_cast<GM_ADDR>(membership), reinterpret_cast<GM_ADDR>(counters),
                    history, query, width, pipe, adaptive != 0U);
            op.Process();
        }
        return 0;
    } catch (const std::exception& error) { lastError = error.what(); return 1; }
}
int union_read_pair(int32_t* topk, int32_t* lists, int32_t* descriptors, uint32_t* scores,
                    uint32_t history, uint32_t query, uint32_t group) {
    using namespace AscendC;
    Reset();
    const uint32_t cohorts = (query + 1U) / 2U;
    Register(topk, query * 2048U * sizeof(int32_t));
    Register(lists, cohorts * 4096U * sizeof(int32_t));
    Register(descriptors, cohorts * 8U * sizeof(int32_t));
    try {
        TPipe pipe;
        SparsePrefillLocal::LocalQueryUnion op;
        op.Init(reinterpret_cast<GM_ADDR>(topk), reinterpret_cast<GM_ADDR>(lists),
                reinterpret_cast<GM_ADDR>(descriptors), history, query, pipe);
        const uint32_t count = op.ReadPair(group);
        if (LIMtpC8ServiceVec::observedSortInput.size() != count)
            throw std::runtime_error("ReadPair did not reach the unchanged sort boundary");
        std::copy(LIMtpC8ServiceVec::observedSortInput.begin(),
                  LIMtpC8ServiceVec::observedSortInput.end(), scores);
        op.Publish(0U, 0U, 0U);
        return 0;
    } catch (const std::exception& error) { lastError = error.what(); return 1; }
}
int union_finish(float* membership, int32_t* counters, int32_t* groups, uint32_t* bitmap,
                 int32_t* sources, int32_t* status, uint32_t width,
                 uint32_t query, uint32_t capacity) {
    using namespace AscendC;
    Reset();
    const uint32_t words = ((width + 31U) / 32U + 63U) / 64U * 64U;
    Register(membership, 32U * width * sizeof(float));
    Register(counters, 32U * 8U * sizeof(int32_t));
    Register(groups, 32U * 8U * sizeof(int32_t));
    Register(bitmap, words * 32U * sizeof(uint32_t));
    Register(sources, capacity * sizeof(int32_t));
    Register(status, 8U * sizeof(int32_t));
    try {
        for (int block = 31; block >= 0; --block) {
            blockIndex = block;
            TPipe pipe;
            SparsePrefillBitmap::BuildMembershipWords op;
            op.Init(reinterpret_cast<GM_ADDR>(membership), reinterpret_cast<GM_ADDR>(counters),
                    reinterpret_cast<GM_ADDR>(groups), reinterpret_cast<GM_ADDR>(bitmap), width, pipe);
            op.Process();
        }
        for (int block = 31; block >= 0; --block) {
            blockIndex = block;
            TPipe pipe;
            SparsePrefillBitmap::FinishSortedBitmap op;
            op.Init(reinterpret_cast<GM_ADDR>(groups), reinterpret_cast<GM_ADDR>(bitmap),
                    reinterpret_cast<GM_ADDR>(sources), reinterpret_cast<GM_ADDR>(status),
                    query, capacity, pipe);
            op.Process();
        }
        return 0;
    } catch (const std::exception& error) { lastError = error.what(); return 1; }
}
}
"""


def make_host_stub():
    """Extend the existing scalar model only for the newly executed operations."""
    source = STUB.read_text()
    replacements = {
        "MTE3_S, MTE2_S, V_MTE3, S_V": "MTE3_S, MTE2_S, V_MTE3, S_V, S_MTE2",
        "enum class CMPMODE { GT, GE }": "enum class CMPMODE { GT, GE, EQ, LE, LT, NE }",
        "enum class RoundMode { CAST_NONE }": "enum class RoundMode { CAST_NONE, CAST_RINT }",
        "inline uint32_t blockIndex = 0;": "inline uint32_t blockIndex = 0;\ninline uint32_t progressions = 0;",
        "void ArithProgression(LocalTensor<T> dst, T first, T step, size_t count) {": (
            "void ArithProgression(LocalTensor<T> dst, T first, T step, size_t count) {\n    ++progressions;"
        ),
        "mode == CMPMODE::GT ? src.GetValue(i) > value : src.GetValue(i) >= value": (
            "mode == CMPMODE::GT ? src.GetValue(i) > value :\n"
            "            mode == CMPMODE::GE ? src.GetValue(i) >= value :\n"
            "            mode == CMPMODE::EQ ? src.GetValue(i) == value :\n"
            "            mode == CMPMODE::LE ? src.GetValue(i) <= value :\n"
            "            mode == CMPMODE::LT ? src.GetValue(i) < value : src.GetValue(i) != value"
        ),
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise AssertionError(f"The common host stub changed at {before!r}")
        source = source.replace(before, after)
    return source + HOST_EXTRA


def write_indexer_model(build, operations):
    # The optional historical reference still includes the old MTP headers.
    mtp = build / "csrc/attention/fused_quant_lightning_indexer_manage/op_kernel"
    mtp.mkdir(parents=True)
    (mtp / "fused_quant_lightning_indexer_manage_common.h").write_text("#pragma once\n")
    (mtp / "fused_quant_lightning_indexer_manage_vector.h").write_text(operations)

    indexer = build / "csrc/lightning_indexer_vllm/op_kernel"
    indexer.mkdir(parents=True)
    (indexer / "lightning_indexer_common.h").write_text("#pragma once\n")
    (indexer / "lightning_indexer_vector.h").write_text(
        '#pragma once\n'
        '#include "../../attention/fused_quant_lightning_indexer_manage/op_kernel/'
        'fused_quant_lightning_indexer_manage_vector.h"\n'
        'namespace LIServiceVec {\n'
        'using LIMtpC8ServiceVec::ExtractIndex;\n'
        'using LIMtpC8ServiceVec::SortAll;\n'
        'constexpr uint32_t VEC_REPEAT_BYTES = 256;\n'
        'constexpr uint32_t B32_VEC_REPEAT_STRIDE = 8;\n'
        '}\n'
    )


def copy_indexer_bridge(headers, revision):
    if revision is None:
        name = "sparse_prefill_indexer_primitives.h"
        shutil.copyfile(ROOT / KERNEL_DIR / name, headers / name)


class SelectedUnionStatisticsTest(unittest.TestCase):
    reference_commit = None

    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++")
        if cls.compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required")
        directory = tempfile.TemporaryDirectory(prefix="selected-union-statistics-")
        cls.addClassCleanup(directory.cleanup)
        cls.build = Path(directory.name)
        cls.library = cls.compile_headers("current")
        cls.reference = cls.compile_headers("reference", cls.reference_commit) if cls.reference_commit else None

    @classmethod
    def compile_headers(cls, name, revision=None):
        build = cls.build / name
        headers = build / KERNEL_DIR
        headers.mkdir(parents=True)
        for filename in HEADERS:
            path = KERNEL_DIR / filename
            contents = (
                subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
                if revision
                else (ROOT / path).read_text()
            )
            (headers / filename).write_text(contents)
        (headers / "kernel_operator.h").write_text(make_host_stub())
        copy_indexer_bridge(headers, revision)
        write_indexer_model(build, MTP_BOUNDARY)
        (build / "wrapper.cpp").write_text(WRAPPER)
        library_path = build / "libunion.so"
        command = [
            cls.compiler,
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{headers}",
            str(build / "wrapper.cpp"),
            "-o",
            str(library_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise AssertionError(f"Real union headers did not compile with the host stub:\n{completed.stderr}")
        library = ctypes.CDLL(str(library_path))
        library.union_partition.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 4
        library.union_read_pair.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 3
        library.union_finish.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 3
        library.union_error.restype = ctypes.c_char_p
        library.union_progressions.restype = ctypes.c_uint32
        return library

    def partition(self, history, query, runs, *, adaptive=False, run_layout=False, errors=None, holes=0):
        width = (history + 2047) // 2048 * 64
        cohorts = (query + 1) // 2
        lists = GuardedArray((cohorts * 4096,), np.int32, -1)
        descriptors = GuardedArray((cohorts, 8), np.int32)
        selected = []
        offset = 0
        for group, values in runs.items():
            values = np.asarray(values, dtype=np.int32)
            begin = offset if run_layout else group * 4096
            self.assertLessEqual(len(values), 8192 if run_layout else 4096)
            lists.value[begin : begin + len(values)] = values
            descriptors.value[group, :4] = [len(values), (errors or {}).get(group, 0), holes, begin]
            selected.extend(values[values >= 0])
            offset = begin + len(values)
        if run_layout and adaptive:
            for core in range(32):
                first, limit = cohorts * core // 32, cohorts * (core + 1) // 32
                if first != limit:
                    descriptors.value[first, 4] = 1
        expected = np.zeros((32, width), dtype=np.float32)
        selected = np.unique(selected)
        if len(selected):
            expected[selected % 32, selected // 32] = 1
        error = int(np.bitwise_or.reduce(descriptors.value[:, 1]))
        result = []
        for library in (self.library, self.reference):
            if library is None:
                continue
            membership = GuardedArray((32, width), np.float32, SENTINEL)
            counters = GuardedArray((32, 8), np.int32, SENTINEL)
            rc = library.union_partition(
                lists.address,
                descriptors.address,
                membership.address,
                counters.address,
                history,
                query,
                width,
                int(adaptive),
            )
            self.assertEqual(rc, 0, library.union_error().decode())
            expected_scans = 0 if library is self.library else 32 * ((width + 4095) // 4096)
            self.assertEqual(library.union_progressions(), expected_scans)
            np.testing.assert_array_equal(membership.value, expected)
            np.testing.assert_array_equal(counters.value[:, 2], error)
            if library is self.library:
                np.testing.assert_array_equal(counters.value[:, [0, 1, 3, 4, 5, 6, 7]], 0)
            words = ((width + 31) // 32 + 63) // 64 * 64
            groups = GuardedArray((32, 8), np.int32, SENTINEL)
            bitmap = GuardedArray((words * 32,), np.uint32, SENTINEL)
            sources = GuardedArray((history,), np.int32, SENTINEL)
            status = GuardedArray((8,), np.int32, SENTINEL)
            rc = library.union_finish(
                membership.address,
                counters.address,
                groups.address,
                bitmap.address,
                sources.address,
                status.address,
                width,
                query,
                history,
            )
            self.assertEqual(rc, 0, library.union_error().decode())
            total = len(selected)
            np.testing.assert_array_equal(status.value, [total, query, total + query, error, 0, 0, 0, 0])
            np.testing.assert_array_equal(sources.value[:total], SENTINEL if error else selected)
            np.testing.assert_array_equal(sources.value[total:], SENTINEL)
            for array in (lists, descriptors, membership, counters, groups, bitmap, sources, status):
                array.assert_guards()
            result.append((sources.value.copy(), status.value.copy()))
        if self.reference:
            for current, previous in zip(*result):
                np.testing.assert_array_equal(current, previous)

    def read_pair(self, history, query, group, rows):
        topk = GuardedArray((query, 2048), np.int32, -1)
        start = group * 2
        topk.value[start : start + len(rows)] = rows
        cohorts = (query + 1) // 2
        observed = []
        expected = np.where((rows >= 0) & (rows < history), SOURCE_KEY_BASE - rows, 0).astype(np.uint32)
        bad = bool(np.any((rows < -1) | (rows >= history + query)))
        for library in (self.library, self.reference):
            if library is None:
                continue
            lists = GuardedArray((cohorts * 4096,), np.int32, SENTINEL)
            descriptors = GuardedArray((cohorts, 8), np.int32, SENTINEL)
            scores = GuardedArray(rows.shape, np.uint32, SENTINEL)
            rc = library.union_read_pair(
                topk.address,
                lists.address,
                descriptors.address,
                scores.address,
                history,
                query,
                group,
            )
            self.assertEqual(rc, 0, library.union_error().decode())
            np.testing.assert_array_equal(scores.value, expected)
            self.assertEqual(descriptors.value[0, 1], int(bad))
            np.testing.assert_array_equal(descriptors.value[0, [0, 3, 4, 5, 6, 7]], 0)
            if library is self.library:
                self.assertEqual(descriptors.value[0, 2], 0)
            np.testing.assert_array_equal(descriptors.value[1:], SENTINEL)
            np.testing.assert_array_equal(lists.value, SENTINEL)
            for array in (topk, lists, descriptors, scores):
                array.assert_guards()
            observed.append(scores.value.copy())
        if self.reference:
            np.testing.assert_array_equal(*observed)

    def test_partition_supported_history_extents_and_duplicate_ids(self):
        for history in (65536, 131072, 262144, 524288):
            with self.subTest(history=history):
                ids = np.array([0, 1, 31, 32, 2047, 2048, history - 1, -1] * 16)
                self.partition(history, 128, {0: ids, 63: ids[::-1]}, adaptive=True)

    def test_partition_adaptive_fixed_and_merged_run_layouts(self):
        for query in (256, 512, 1024):
            for run_layout in (False, True):
                with self.subTest(query=query, run_layout=run_layout):
                    count = 8192 if run_layout else 4096
                    ids = np.arange(count, dtype=np.int32) * 31 % 262144
                    self.partition(262144, query, {0: ids}, adaptive=True, run_layout=run_layout)

    def test_partition_empty_future_only_prefix_and_last_record(self):
        self.partition(65536, 128, {}, adaptive=True)
        self.partition(65536, 128, {63: [0, 65535]}, run_layout=True)

    def test_partition_directory_tiles_cover_large_odd_queries(self):
        for query in (1025, 2048, 4097, 7984, 8192):
            for run_layout in (False, True):
                with self.subTest(query=query, run_layout=run_layout):
                    runs = {0: [0, 31], 511: [32, 128], 512: [2048, 4095], (query - 1) // 2: [524286]}
                    self.partition(524287, query, runs, adaptive=True, run_layout=run_layout)

    def test_dead_descriptor_holes_never_change_ids_or_error_publication(self):
        for holes in (0, 1, SENTINEL):
            with self.subTest(holes=holes):
                self.partition(65536, 128, {0: [0, 31, 65535]}, adaptive=True, holes=holes)
                self.partition(
                    65536,
                    128,
                    {0: [0, 31], 63: [65535]},
                    adaptive=True,
                    errors={0: 4, 63: 8},
                    holes=holes,
                )

    def test_read_pair_holes_preserve_sort_input_and_error_flags(self):
        rows = np.full((2, 2048), -1, dtype=np.int32)
        rows[:, [0, 2, 31, 32, 64, 95, 1024, 2047]] = np.arange(8)
        for query in (128, 256, 512, 1024):
            with self.subTest(query=query):
                self.read_pair(65536, query, 0, rows)
                self.read_pair(65536, query, query // 2 - 1, rows)

    def test_odd_query_tail_causal_and_history_masks(self):
        for history in (65537, 65661, 65663, 524287):
            for query in (1, 125, 127, 129, 1023):
                with self.subTest(history=history, query=query):
                    rows = np.full((1, 2048), -1, dtype=np.int32)
                    rows[0, :6] = [0, history - 1, history, history + query - 1, history + query, -2]
                    self.read_pair(history, query, query // 2, rows)

    def test_read_pair_invalid_and_future_only_inputs(self):
        history, query = 65536, 128
        for values in ((-1,), (history, history + 1, history + 127), (-2,), (history + query,)):
            with self.subTest(values=values):
                rows = np.resize(np.asarray(values, dtype=np.int32), (2, 2048))
                self.read_pair(history, query, 0, rows)
        rows = np.full((1, 2048), -1, dtype=np.int32)
        rows[0, [0, 31, 32, 2047]] = [0, history - 1, history, -2]
        self.read_pair(history, 129, 64, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-commit", help="Optionally compile and compare the pre-cleanup real headers")
    options, remaining = parser.parse_known_args()
    SelectedUnionStatisticsTest.reference_commit = options.reference_commit
    unittest.main(argv=[sys.argv[0], *remaining])
