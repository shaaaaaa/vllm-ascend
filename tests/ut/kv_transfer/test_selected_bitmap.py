# SPDX-License-Identifier: Apache-2.0
"""Execute the real bitmap/finish header using scalar AscendC host operations.

This validates C++ arithmetic, indexing, output bounds and publication semantics.
It does not validate SDK compilation, device alignment, barriers or NPU execution.
The optional v5 comparison compiles the old real header; no old header is vendored.
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

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "csrc/sparse_prefill_native/sparse_prefill_bitmap_services.h"
STUB = Path(__file__).with_name("selected_bitmap_host_stub.h")
SENTINEL = 0x5A5A5A5A

WRAPPER = r"""
#include "bitmap_under_test.h"
#include <string>

namespace {
std::string lastError;
}
extern "C" {
const char* bitmap_error() { return lastError.c_str(); }
size_t bitmap_writes() { return AscendC::watchedWrites; }
int bitmap_phase(uint32_t phase, uint32_t block, float* membership, int32_t* residue,
                 int32_t* groups, uint32_t* bitmap, int32_t* sources, int32_t* status,
                 uint32_t width, uint32_t query, uint32_t capacity) {
    using namespace AscendC;
    const uint32_t words = ((width + 31U) / 32U + 63U) / 64U * 64U;
    const uint32_t columns = OLD_REFERENCE ? 2U : 1U;
    regions.clear();
    Register(membership, 32U * width * sizeof(float));
    Register(residue, 32U * 8U * sizeof(int32_t));
    Register(groups, 32U * 8U * sizeof(int32_t));
    Register(bitmap, 32U * words * columns * sizeof(uint32_t));
    Register(sources, capacity * sizeof(int32_t));
    Register(status, 8U * sizeof(int32_t));
    watched = {reinterpret_cast<uintptr_t>(bitmap),
               reinterpret_cast<uintptr_t>(bitmap) + 32U * words * columns * sizeof(uint32_t)};
    watchedWrites = 0;
    blockIndex = block;
    lastError.clear();
    try {
        TPipe pipe;
        if (phase == 0) {
            SparsePrefillBitmap::BuildMembershipWords op;
#if OLD_REFERENCE
            op.Init(reinterpret_cast<GM_ADDR>(membership), reinterpret_cast<GM_ADDR>(residue),
                    reinterpret_cast<GM_ADDR>(groups), reinterpret_cast<GM_ADDR>(bitmap),
                    width, pipe, true, true);
#else
            op.Init(reinterpret_cast<GM_ADDR>(membership), reinterpret_cast<GM_ADDR>(residue),
                    reinterpret_cast<GM_ADDR>(groups), reinterpret_cast<GM_ADDR>(bitmap), width, pipe);
#endif
            op.Process();
        } else {
            SparsePrefillBitmap::FinishSortedBitmap op;
#if OLD_REFERENCE
            op.Init(reinterpret_cast<GM_ADDR>(groups), reinterpret_cast<GM_ADDR>(bitmap),
                    reinterpret_cast<GM_ADDR>(sources), reinterpret_cast<GM_ADDR>(status),
                    query, capacity, pipe, true, true, true);
#else
            op.Init(reinterpret_cast<GM_ADDR>(groups), reinterpret_cast<GM_ADDR>(bitmap),
                    reinterpret_cast<GM_ADDR>(sources), reinterpret_cast<GM_ADDR>(status),
                    query, capacity, pipe);
#endif
            op.Process();
        }
        return 0;
    } catch (const std::exception& error) {
        lastError = error.what();
        return 1;
    }
}
}
"""


class GuardedArray:
    """Keep GM guards separate from the address range the host stub permits."""

    def __init__(self, shape, dtype, fill=0):
        count = int(np.prod(shape))
        self.storage = np.full(count + 16, SENTINEL, dtype=dtype)
        self.value = self.storage[8:-8].reshape(shape)
        self.value.fill(fill)
        self.guards = (self.storage[:8].copy(), self.storage[-8:].copy())

    @property
    def address(self):
        return self.value.ctypes.data_as(ctypes.c_void_p)

    def assert_guards(self):
        for actual, expected in zip((self.storage[:8], self.storage[-8:]), self.guards):
            np.testing.assert_array_equal(actual, expected)


class BitmapCase:
    def __init__(self, width, selected, *, query=128, capacity=None, errors=None, old=False):
        self.width, self.query = width, query
        self.words = ((width + 31) // 32 + 63) // 64 * 64
        self.capacity = width * 32 if capacity is None else capacity
        self.selected = np.unique(np.asarray(selected, dtype=np.int64))
        self.membership = GuardedArray((32, width), np.float32)
        self.membership.value[self.selected % 32, self.selected // 32] = 1.0
        self.residue = GuardedArray((32, 8), np.int32)
        if errors:
            for block, error in errors.items():
                self.residue.value[block, 2] = error
        self.groups = GuardedArray((32, 8), np.int32, SENTINEL)
        self.bitmap = GuardedArray((self.words * 32, 2) if old else (self.words * 32,), np.uint32, SENTINEL)
        self.sources = GuardedArray((self.capacity,), np.int32, SENTINEL)
        self.status = GuardedArray((8,), np.int32, SENTINEL)
        self.arrays = (self.membership, self.residue, self.groups, self.bitmap, self.sources, self.status)

    def arguments(self, phase, block):
        return [phase, block, *(array.address for array in self.arrays), self.width, self.query, self.capacity]

    def expected_bitmap(self):
        expected = np.zeros(self.words * 32, dtype=np.uint32)
        np.bitwise_or.at(
            expected, self.selected // 32, np.left_shift(np.uint32(1), self.selected % 32).astype(np.uint32)
        )
        return expected


class SelectedBitmapTest(unittest.TestCase):
    reference_commit = None

    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++")
        if cls.compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required")
        directory = tempfile.TemporaryDirectory(prefix="selected-bitmap-")
        cls.addClassCleanup(directory.cleanup)
        cls.build = Path(directory.name)
        cls.library = cls.compile_header(HEADER, old=False)
        cls.reference = None
        if cls.reference_commit is not None:
            result = subprocess.run(
                ["git", "show", f"{cls.reference_commit}:{HEADER.relative_to(ROOT)}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            old_header = cls.build / "v5.h"
            old_header.write_text(result.stdout)
            cls.reference = cls.compile_header(old_header, old=True)

    @classmethod
    def compile_header(cls, header, *, old):
        build = cls.build / ("v5" if old else "current")
        build.mkdir()
        shutil.copyfile(STUB, build / "kernel_operator.h")
        (build / "bitmap_under_test.h").write_text(f'#include "{header}"\n')
        (build / "wrapper.cpp").write_text(WRAPPER)
        library_path = build / "libbitmap.so"
        command = [
            cls.compiler,
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-DOLD_REFERENCE={int(old)}",
            f"-I{build}",
            str(build / "wrapper.cpp"),
            "-o",
            str(library_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise AssertionError(f"Real bitmap header did not compile with the host stub:\n{completed.stderr}")
        library = ctypes.CDLL(str(library_path))
        library.bitmap_phase.argtypes = [ctypes.c_uint32] * 2 + [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 3
        library.bitmap_phase.restype = ctypes.c_int
        library.bitmap_error.restype = ctypes.c_char_p
        library.bitmap_writes.restype = ctypes.c_size_t
        return library

    def phase(self, case, phase, *, library=None):
        library = library or self.library
        # Exercise launch order independence, including core 0 running last.
        for block in range(31, -1, -1):
            result = library.bitmap_phase(*case.arguments(phase, block))
            self.assertEqual(result, 0, library.bitmap_error().decode())
            if phase == 1 and library is self.library:
                self.assertEqual(library.bitmap_writes(), 0, "Finish must not write input bitmap/rank data")
        for array in case.arrays:
            array.assert_guards()

    def check(self, width, selected, *, query=128, capacity=None, errors=None):
        case = BitmapCase(width, selected, query=query, capacity=capacity, errors=errors)
        original_membership = case.membership.value.copy()
        original_residue = case.residue.value.copy()
        self.phase(case, 0)
        expected_bitmap = case.expected_bitmap()
        np.testing.assert_array_equal(case.bitmap.value, expected_bitmap)
        for block in range(32):
            begin, end = block * case.words * 32, (block + 1) * case.words * 32
            expected_count = np.count_nonzero((case.selected >= begin) & (case.selected < end))
            self.assertEqual(case.groups.value[block, 0], expected_count)
            self.assertEqual(case.groups.value[block, 1], case.residue.value[block, 2])
            self.assertEqual(case.groups.value[block, 2], case.words)
            np.testing.assert_array_equal(case.groups.value[block, 3:], 0)
        bitmap_before, groups_before = case.bitmap.value.copy(), case.groups.value.copy()
        self.phase(case, 1)
        error = int(np.bitwise_or.reduce(case.residue.value[:, 2]))
        total = len(case.selected)
        np.testing.assert_array_equal(
            case.status.value, [total, query, total + query, error, int(total > case.capacity), 0, 0, 0]
        )
        if error or total > case.capacity:
            np.testing.assert_array_equal(case.sources.value, SENTINEL)
        else:
            np.testing.assert_array_equal(case.sources.value[:total], case.selected)
            np.testing.assert_array_equal(case.sources.value[total:], SENTINEL)
        np.testing.assert_array_equal(case.bitmap.value, bitmap_before)
        np.testing.assert_array_equal(case.groups.value, groups_before)
        np.testing.assert_array_equal(case.membership.value, original_membership)
        np.testing.assert_array_equal(case.residue.value, original_residue)
        if self.reference:
            reference = BitmapCase(width, selected, query=query, capacity=capacity, errors=errors, old=True)
            self.phase(reference, 0, library=self.reference)
            np.testing.assert_array_equal(reference.bitmap.value[:, 0], expected_bitmap)
            np.testing.assert_array_equal(reference.groups.value, case.groups.value)
            self.phase(reference, 1, library=self.reference)
            np.testing.assert_array_equal(reference.sources.value, case.sources.value)
            np.testing.assert_array_equal(reference.status.value, case.status.value)
        return case

    def test_empty_all_supported_history_extents(self):
        for width in (2048, 4096, 8192, 16384):
            with self.subTest(width=width):
                self.check(width, [])

    def test_transpose_every_residue_and_high_bit(self):
        for residue in range(32):
            with self.subTest(residue=residue):
                self.check(2048, np.arange(2048) * 32 + residue)

    def test_irregular_history_word_rounding_has_no_phantom_tail_ids(self):
        for history in (65537, 65661, 65663, 131073, 524287):
            width = ((history + 4095) // 4096) * 128
            for query in (1, 125, 127, 1023):
                with self.subTest(history=history, query=query):
                    case = self.check(width, [0, 31, 32, history - 1], query=query)
                    self.assertEqual(case.status.value[0], 4)

    def test_every_word_popcount_with_high_bits(self):
        selected = [word * 32 + (bit + 17) % 32 for word in range(33) for bit in range(word)]
        self.check(2048, selected)

    def test_full_groups_and_exact_float_reduce_bound(self):
        for width in (2048, 4096, 8192, 16384):
            with self.subTest(width=width):
                case = self.check(width, np.arange(width * 32), query=1024)
                self.assertEqual(case.groups.value[0, 0], case.words * 32)
        self.assertEqual(case.groups.value[0, 0], 16384)

    def test_word_group_gathermask_chunk_and_history_boundaries(self):
        width = 16384
        boundaries = (0, 1, 31, 32, 33, 8191, 8192, 8193, 16383, 16384, 16385, width * 32 - 1)
        self.check(width, boundaries, query=256)

    def test_holes_random_popcount_and_non_power_of_two_extent(self):
        random = np.random.default_rng(20261031)
        for width in (2049, 2176, 4097, 8320, 16383):
            with self.subTest(width=width):
                selected = random.choice(width * 32, 4097, replace=False)
                self.check(width, selected, query=512)

    def test_exact_capacity_succeeds_and_one_less_writes_no_partial_ids(self):
        selected = np.arange(0, 65536, 31)
        self.check(2048, selected, capacity=len(selected))
        self.check(2048, selected, capacity=len(selected) - 1)
        self.check(2048, [0], capacity=0)

    def test_error_bits_publish_without_writing_any_ids(self):
        self.check(2048, [0, 31, 2048, 65535], errors={0: 1, 15: 4, 31: 8})
        self.check(2048, [0, 31, 2048, 65535], capacity=3, errors={31: 2})
        self.check(2048, [], errors={16: 16})

    def test_bad_bitmap_extent_is_rejected_before_writes(self):
        case = BitmapCase(2048, [0, 31, 65535])
        self.phase(case, 0)
        for extent in (0, 16, 63, 65, 513, 1024):
            with self.subTest(extent=extent):
                case.groups.value[0, 2] = extent
                snapshots = [array.value.copy() for array in case.arrays]
                result = self.library.bitmap_phase(*case.arguments(1, 0))
                self.assertEqual(result, 1)
                self.assertIn(b"Invalid compact bitmap extent", self.library.bitmap_error())
                self.assertEqual(self.library.bitmap_writes(), 0)
                for array, before in zip(case.arrays, snapshots):
                    np.testing.assert_array_equal(array.value, before)
                    array.assert_guards()

    def test_finish_input_bitmap_guard_is_not_its_rank_output(self):
        case = self.check(2048, [31, 32, 65535])
        self.assertEqual(case.bitmap.value[0], 1 << 31)
        self.assertEqual(case.bitmap.value[1], 1)
        self.assertEqual(case.bitmap.value[-1], 1 << 31)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-commit", help="Optionally compile and compare the pre-v6 bitmap header")
    options, remaining = parser.parse_known_args()
    SelectedBitmapTest.reference_commit = options.reference_commit
    unittest.main(argv=[sys.argv[0], *remaining])
