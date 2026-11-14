# SPDX-License-Identifier: Apache-2.0
"""Run the real local-union header with scalar sort/merge instruction models.

The MrgSort model enforces the SDK's per-source limit and sorted-input contract.
This checks the actual call construction, bounds and run publication, not CANN
code generation, instruction alignment requirements or device synchronization.
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
from test_selected_union_statistics import (
    HEADERS,
    KERNEL_DIR,
    ROOT,
    copy_indexer_bridge,
    make_host_stub,
    write_indexer_model,
)

HOST_OPERATIONS = r"""
namespace AscendC {
inline uint32_t mergeCalls = 0, largestSource = 0, threeWayCalls = 0;
template <typename T> void Relu(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) dst.SetValue(i, std::max<T>(0, src.GetValue(i)));
}
template <typename T> void Compare(LocalTensor<uint8_t> dst, LocalTensor<T> a,
                                  LocalTensor<T> b, CMPMODE mode, uint32_t count) {
    if (mode != CMPMODE::NE) throw std::runtime_error("Unexpected comparison");
    Duplicate(dst, uint8_t{0}, (count + 7U) / 8U);
    for (uint32_t i = 0; i < count; ++i)
        if (a.GetValue(i) != b.GetValue(i)) dst.SetValue(i / 8U, dst.GetValue(i / 8U) | (1U << (i % 8U)));
}
template <typename T> void MrgSort(LocalTensor<T> dst, MrgSortSrcList<T> src, MrgSort4Info params) {
    ++mergeCalls;
    if ((params.validBit != 3U && params.validBit != 7U && params.validBit != 15U) ||
        params.repeatTimes != 1U || params.ifExhaustedSuspension)
        throw std::runtime_error("Unsupported MrgSort parameters");
    threeWayCalls += params.validBit == 7U;
    const LocalTensor<T> sources[] = {src.src1, src.src2, src.src3, src.src4};
    std::vector<std::pair<float, uint32_t>> pairs;
    for (uint32_t s = 0; s < 4U; ++s) {
        if (!(params.validBit & (1U << s))) continue;
        const uint32_t count = params.elementLengths[s];
        largestSource = std::max(largestSource, count);
        if (count > 4095U) throw std::runtime_error("MrgSort source exceeds 4095 pairs");
        if (reinterpret_cast<uintptr_t>(sources[s].Access(0, false)) % 8U)
            throw std::runtime_error("MrgSort source is not 8-byte aligned");
        for (uint32_t i = 0; i < count; ++i) {
            const float score = sources[s].GetValue(i * 2U);
            if (i && sources[s].GetValue((i - 1U) * 2U) < score)
                throw std::runtime_error("MrgSort source is not sorted");
            pairs.emplace_back(score, sources[s].template ReinterpretCast<uint32_t>().GetValue(i * 2U + 1U));
        }
    }
    std::stable_sort(pairs.begin(), pairs.end(), [](auto a, auto b) { return a.first > b.first; });
    for (uint32_t i = 0; i < pairs.size(); ++i) {
        dst.SetValue(i * 2U, pairs[i].first);
        dst.template ReinterpretCast<uint32_t>().SetValue(i * 2U + 1U, pairs[i].second);
    }
}
}
"""

MTP_OPERATIONS = r"""
#pragma once
#include "kernel_operator.h"
namespace LIMtpC8ServiceVec {
using namespace AscendC;
constexpr uint32_t LONG_SOURCE_MASK = (1U << 21U) - 1U;
inline void SortAll(LocalTensor<float>& src, LocalTensor<float>&, int64_t count) {
    std::vector<std::pair<float, uint32_t>> pairs;
    for (int64_t i = 0; i < count; ++i)
        pairs.emplace_back(src.GetValue(i), src.ReinterpretCast<uint32_t>().GetValue(count + i));
    std::stable_sort(pairs.begin(), pairs.end(), [](auto a, auto b) { return a.first > b.first; });
    for (int64_t i = 0; i < count; ++i) {
        src.SetValue(i * 2, pairs[i].first);
        src.ReinterpretCast<uint32_t>().SetValue(i * 2 + 1, pairs[i].second);
    }
}
inline void ExtractScoreBits(const LocalTensor<uint32_t>& dst, const LocalTensor<uint32_t>& src, int64_t count) {
    for (int64_t i = 0; i < (count + 31) / 32 * 32; ++i) dst.SetValue(i, src.GetValue(i * 2));
}
inline void ExtractIndex(const LocalTensor<uint32_t>& dst, const LocalTensor<uint32_t>& src, int64_t count) {
    for (int64_t i = 0; i < (count + 31) / 32 * 32; ++i) dst.SetValue(i, src.GetValue(i * 2 + 1));
}
}
"""

WRAPPER = r"""
#include "sparse_prefill_local_union_services.h"
#include <string>
namespace {
std::string error;
void Reset() {
    AscendC::regions.clear();
    AscendC::watched = {};
    AscendC::mergeCalls = AscendC::largestSource = AscendC::threeWayCalls = 0;
    error.clear();
}
}
extern "C" {
const char* local_error() { return error.c_str(); }
uint32_t local_merges() { return AscendC::mergeCalls; }
uint32_t local_largest_source() { return AscendC::largestSource; }
uint32_t local_three_way() { return AscendC::threeWayCalls; }
#if __has_include("sparse_prefill_indexer_primitives.h")
int local_extract_pairs(const uint32_t* words, uint32_t* scores, uint32_t* indices, uint32_t count) {
    using namespace AscendC;
    Reset();
    try {
        const uint32_t aligned = (count + 31U) / 32U * 32U;
        TPipe pipe;
        TBuf<QuePosition::VECCALC> pairsBuf, scoreBuf, indexBuf;
        pipe.InitBuffer(pairsBuf, aligned * 2U * sizeof(uint32_t));
        pipe.InitBuffer(scoreBuf, aligned * sizeof(uint32_t));
        pipe.InitBuffer(indexBuf, aligned * sizeof(uint32_t));
        auto pairs = pairsBuf.Get<uint32_t>();
        auto score = scoreBuf.Get<uint32_t>();
        auto index = indexBuf.Get<uint32_t>();
        for (uint32_t i = 0; i < aligned * 2U; ++i) pairs.SetValue(i, words[i]);
        SparsePrefillIndexer::ExtractScoreBits(score, pairs, count);
        SparsePrefillIndexer::ExtractIndex(index, pairs, count);
        for (uint32_t i = 0; i < aligned; ++i) {
            scores[i] = score.GetValue(i);
            indices[i] = index.GetValue(i);
        }
        return 0;
    } catch (const std::exception& e) { error = e.what(); return 1; }
}
#endif
int local_process(int32_t* topk, int32_t* lists, int32_t* descriptors, uint32_t h, uint32_t q) {
    using namespace AscendC;
    Reset();
    const uint32_t cohorts = (q + 1U) / 2U;
    Register(topk, q * 2048U * sizeof(int32_t));
    Register(lists, cohorts * 4096U * sizeof(int32_t));
    Register(descriptors, cohorts * 8U * sizeof(int32_t));
    try {
        for (uint32_t core = 0; core < 32U; ++core) {
            blockIndex = core;
            TPipe pipe;
            SparsePrefillLocal::LocalQueryUnion op;
            op.Init(reinterpret_cast<GM_ADDR>(topk), reinterpret_cast<GM_ADDR>(lists),
                    reinterpret_cast<GM_ADDR>(descriptors), h, q, pipe);
            op.Process();
        }
        return 0;
    } catch (const std::exception& e) { error = e.what(); return 1; }
}
int local_merge(uint32_t leftCount, uint32_t rightCount, uint32_t* output) {
    using namespace AscendC;
    Reset();
#if HAS_MERGE_HELPER
    try {
        TPipe pipe;
        TBuf<QuePosition::VECCALC> leftBuf, rightBuf, outBuf;
        pipe.InitBuffer(leftBuf, 8192U * 8U);
        pipe.InitBuffer(rightBuf, 4096U * 8U);
        pipe.InitBuffer(outBuf, 8192U * 8U);
        auto left = leftBuf.Get<float>(), right = rightBuf.Get<float>(), out = outBuf.Get<float>();
        for (uint32_t i = 0; i < leftCount; ++i) {
            left.ReinterpretCast<uint32_t>().SetValue(i * 2U, 0x401fffffU - i * 2U);
            left.ReinterpretCast<uint32_t>().SetValue(i * 2U + 1U, i * 2U);
        }
        for (uint32_t i = 0; i < rightCount; ++i) {
            right.ReinterpretCast<uint32_t>().SetValue(i * 2U, 0x401fffffU - i * 2U - 1U);
            right.ReinterpretCast<uint32_t>().SetValue(i * 2U + 1U, i * 2U + 1U);
        }
        SparsePrefillLocal::MergeSortedRuns(out, left, right, leftCount, rightCount);
        for (uint32_t i = 0; i < leftCount + rightCount; ++i)
            output[i] = out.ReinterpretCast<uint32_t>().GetValue(i * 2U + 1U);
        return 0;
    } catch (const std::exception& e) { error = e.what(); return 1; }
#else
    (void)leftCount; (void)rightCount; (void)output;
    return -1;
#endif
}
}
"""


class SelectedLocalUnionTest(unittest.TestCase):
    reference_commit = None

    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++")
        if cls.compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required")
        directory = tempfile.TemporaryDirectory(prefix="selected-local-union-")
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
        has_helper = "MergeSortedRuns(" in (headers / "sparse_prefill_local_union_services.h").read_text()
        (headers / "kernel_operator.h").write_text(make_host_stub() + HOST_OPERATIONS)
        copy_indexer_bridge(headers, revision)
        write_indexer_model(build, MTP_OPERATIONS)
        (build / "wrapper.cpp").write_text(WRAPPER)
        output = build / "liblocal.so"
        result = subprocess.run(
            [
                cls.compiler,
                "-shared",
                "-fPIC",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-DHAS_MERGE_HELPER={int(has_helper)}",
                f"-I{headers}",
                str(build / "wrapper.cpp"),
                "-o",
                str(output),
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise AssertionError(result.stderr)
        library = ctypes.CDLL(str(output))
        library.local_process.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32] * 2
        library.local_merge.argtypes = [ctypes.c_uint32] * 2 + [ctypes.c_void_p]
        library.local_error.restype = ctypes.c_char_p
        if revision is None:
            library.local_extract_pairs.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32]
        return library

    def process(
        self, query, cohort_counts, *, history=65536, first_id=0, reference_rejects=False, duplicate_runs=False
    ):
        topk = GuardedArray((query, 2048), np.int32, -1)
        offset = first_id
        for group, count in enumerate(cohort_counts):
            if count:
                first = first_id if duplicate_runs else offset
                rows = min(2, query - group * 2)
                topk.value[group * 2 : group * 2 + rows] = np.resize(np.arange(first, first + count), (rows, 2048))
                offset += count
        expected = np.unique(topk.value[(topk.value >= 0) & (topk.value < history)])
        for library in (self.library, self.reference):
            if library is None:
                continue
            lists = GuardedArray(((query + 1) // 2 * 4096,), np.int32, -1)
            descriptors = GuardedArray(((query + 1) // 2, 8), np.int32, -1)
            rc = library.local_process(topk.address, lists.address, descriptors.address, history, query)
            if library is self.reference and reference_rejects:
                self.assertEqual(rc, 1)
                self.assertIn("MrgSort source exceeds 4095", library.local_error().decode())
                continue
            self.assertEqual(rc, 0, library.local_error().decode())
            use_runs = query <= 128 or np.any(descriptors.value[:, 4])
            published = []
            for group, record in enumerate(descriptors.value):
                count, error, _, begin = record[:4]
                self.assertEqual(error, 0)
                self.assertGreaterEqual(count, 0)
                self.assertLessEqual(count, 8128)
                begin = begin if use_runs else group * 4096
                self.assertLessEqual(begin + count, lists.value.size)
                ids = lists.value[begin : begin + count]
                np.testing.assert_array_equal(ids, np.unique(ids))
                published.extend(ids)
            np.testing.assert_array_equal(np.unique(published).astype(np.int32), expected)
            self.assertLessEqual(library.local_largest_source(), 4095)
            for array in (topk, lists, descriptors):
                array.assert_guards()
        return self.library.local_merges()

    def test_extract_pairs_preserves_raw_words_at_repeat_boundaries(self):
        for count in (1, 31, 32, 33, 64, 65):
            with self.subTest(count=count):
                aligned = (count + 31) // 32 * 32
                pairs = GuardedArray((aligned, 2), np.uint32)
                pairs.value[:, 0] = 0x401FFFFF - np.arange(aligned, dtype=np.uint32)
                pairs.value[:, 1] = 0x80000000 + np.arange(aligned, dtype=np.uint32)
                scores = GuardedArray((aligned,), np.uint32)
                indices = GuardedArray((aligned,), np.uint32)
                rc = self.library.local_extract_pairs(pairs.address, scores.address, indices.address, count)
                self.assertEqual(rc, 0, self.library.local_error().decode())
                np.testing.assert_array_equal(scores.value, pairs.value[:, 0])
                np.testing.assert_array_equal(indices.value, pairs.value[:, 1])
                for array in (pairs, scores, indices):
                    array.assert_guards()

    def test_odd_query_last_cohort_and_irregular_history_keep_exact_union(self):
        for history in (65537, 65661, 65663, 524287):
            for query in (1, 125, 127, 129, 1023):
                counts = [0] * ((query + 1) // 2)
                counts[0] = 127
                counts[-1] = 125
                with self.subTest(history=history, query=query):
                    self.process(query, counts, history=history, first_id=history - sum(counts))

    def test_merge_source_boundaries_preserve_every_pair(self):
        for left, right in (
            (4092, 1),
            (4093, 1),
            (4095, 1),
            (4096, 1),
            (6000, 1000),
            (8127, 1),
            (4092, 4036),
            (4032, 4096),
            (1, 4092),
            (1, 4093),
            (1, 4095),
            (1, 4096),
        ):
            with self.subTest(left=left, right=right):
                output = GuardedArray((left + right,), np.uint32)
                rc = self.library.local_merge(left, right, output.address)
                self.assertEqual(rc, 0, self.library.local_error().decode())
                expected = np.sort(np.concatenate((np.arange(left) * 2, np.arange(right) * 2 + 1)))
                np.testing.assert_array_equal(output.value, expected)
                self.assertEqual(self.library.local_merges(), 1)
                self.assertEqual(self.library.local_three_way(), int(max(left, right) > 4095))
                self.assertLessEqual(self.library.local_largest_source(), 4095)
                output.assert_guards()

    def test_large_query_multiple_directory_tiles_and_last_cohort(self):
        for query in (1025, 2048, 4097, 7984, 8192):
            counts = [0] * ((query + 1) // 2)
            # Include an interior metadata-tile boundary and the final cohort.
            counts[0], counts[511], counts[512], counts[-1] = 127, 3000, 1000, 125
            with self.subTest(query=query):
                self.process(query, counts, history=524287, first_id=500000)

    def test_short_query_full_cohort_merges_with_small_run(self):
        self.process(128, (4096, 1000), reference_rejects=True)
        self.process(128, (1000, 4096), reference_rejects=True)

    def test_accumulated_run_exceeds_single_source_limit(self):
        self.process(256, (3000, 3000, 1000, 0), reference_rejects=True)

    def test_exact_merge_capacity_and_next_run_flush(self):
        self.process(128, (4096, 4032), reference_rejects=True)
        self.process(256, (3000, 3000, 2128, 1), reference_rejects=True)

    def test_empty_inputs_keep_original_no_merge_paths(self):
        for query in (128, 256):
            for counts in ((0, 0), (0, 1000), (1000, 0)):
                with self.subTest(query=query, counts=counts):
                    self.assertEqual(self.process(query, counts), 0)

    def test_low_overlap_direct_and_high_overlap_merge(self):
        self.assertEqual(self.process(256, (4096, 4096, 4096, 4096)), 0)
        self.assertGreater(self.process(256, (3000, 3000, 3000, 3000), duplicate_runs=True), 0)

    def test_history_tail_across_supported_query_and_overlap_ranges(self):
        patterns = (
            ("high", (1000, 1000), True),
            ("medium", (3000, 3000, 1000, 0), False),
            ("low", (4096, 4096), False),
        )
        for history in (65536, 131072, 262144, 524288):
            for query in (128, 256, 512, 1024):
                for overlap, counts, duplicate in patterns:
                    with self.subTest(history=history, query=query, overlap=overlap):
                        span = max(counts) if duplicate else sum(counts)
                        self.process(
                            query,
                            counts,
                            history=history,
                            first_id=history - span,
                            duplicate_runs=duplicate,
                            reference_rejects=overlap == "medium" and query >= 256,
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-commit", help="Verify that the previous real header violates the SDK source bound")
    options, remaining = parser.parse_known_args()
    SelectedLocalUnionTest.reference_commit = options.reference_commit
    unittest.main(argv=[sys.argv[0], *remaining])
