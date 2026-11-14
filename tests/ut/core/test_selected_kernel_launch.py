# SPDX-License-Identifier: Apache-2.0
"""Compile and exercise the target host launcher against a recording ACL stub.

These tests check the host ABI and ownership paths. They do not compile the
AscendC kernels or establish NPU numerical correctness.
"""

import ctypes
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

ACL_HEADER = r"""
#pragma once
#include <cstddef>
#include <cstdint>
using aclrtBinHandle = void*;
using aclrtFuncHandle = void*;
using aclrtMemMallocPolicy = int32_t;
constexpr int32_t ACL_SUCCESS = 0;
constexpr aclrtMemMallocPolicy ACL_MEM_MALLOC_NORMAL_ONLY = 2;
constexpr int32_t ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE = 9;
struct aclrtLaunchKernelAttr {
    int32_t id;
    union { uint32_t schemMode; } value;
};
struct aclrtLaunchKernelCfg { aclrtLaunchKernelAttr* attrs; size_t numAttrs; };
extern "C" {
int32_t aclrtBinaryLoadFromFile(const char*, const void*, aclrtBinHandle*);
int32_t aclrtBinaryGetFunction(aclrtBinHandle, const char*, aclrtFuncHandle*);
int32_t aclrtGetHardwareSyncAddr(void**);
int32_t aclrtBinaryUnLoad(aclrtBinHandle);
int32_t aclrtMalloc(void**, size_t, aclrtMemMallocPolicy);
int32_t aclrtFree(void*);
int32_t aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, uint32_t, void*,
    const aclrtLaunchKernelCfg*, const void*, size_t, const void*, size_t);
}
"""

ACL_STUB = r"""
#include <acl/acl.h>
#include <array>
#include <cstring>

namespace {
std::array<int32_t, 9> status{};
uint64_t sequence = 0;
std::array<uint8_t, 256> arguments{};
size_t argumentSize = 0;
std::array<uint64_t, 6> prefix{};
uint32_t cores = 0, attributeId = 0, schema = 0, configCount = 0;
char functionName[128]{};
size_t dumpBytes = 0;
int32_t mallocPolicy = -1;
void* freed = nullptr;
int32_t Record(size_t operation) {
    sequence = sequence * 10 + operation;
    return status[operation];
}
}
extern "C" {
void test_reset() {
    status.fill(0); sequence = 0; arguments.fill(0); prefix.fill(0);
    argumentSize = 0; cores = attributeId = schema = configCount = 0;
    functionName[0] = '\0';
    dumpBytes = 0; mallocPolicy = -1; freed = nullptr;
}
void test_status(uint32_t operation, int32_t value) { status.at(operation) = value; }
uint64_t test_sequence() { return sequence; }
size_t test_argument_size() { return argumentSize; }
const void* test_arguments() { return arguments.data(); }
const char* test_function_name() { return functionName; }
uint64_t test_prefix(uint32_t index) { return prefix.at(index); }
uint32_t test_cores() { return cores; }
uint32_t test_attribute_id() { return attributeId; }
uint32_t test_schema() { return schema; }
uint32_t test_config_count() { return configCount; }
size_t test_dump_bytes() { return dumpBytes; }
int32_t test_malloc_policy() { return mallocPolicy; }
void* test_freed() { return freed; }
int32_t aclrtBinaryLoadFromFile(const char*, const void*, aclrtBinHandle* output) {
    const auto rc = Record(1); if (!rc) *output = reinterpret_cast<void*>(0x111);
    return rc;
}
int32_t aclrtBinaryGetFunction(aclrtBinHandle, const char* name, aclrtFuncHandle* output) {
    std::strncpy(functionName, name, sizeof(functionName) - 1);
    const auto rc = Record(2); if (!rc) *output = reinterpret_cast<void*>(0x222);
    return rc;
}
int32_t aclrtGetHardwareSyncAddr(void** output) {
    const auto rc = Record(3); if (!rc) *output = reinterpret_cast<void*>(0x333);
    return rc;
}
int32_t aclrtBinaryUnLoad(aclrtBinHandle) { return Record(4); }
int32_t aclrtMalloc(void** output, size_t bytes, aclrtMemMallocPolicy policy) {
    dumpBytes = bytes; mallocPolicy = policy;
    const auto rc = Record(7); if (!rc) *output = reinterpret_cast<void*>(0x444);
    return rc;
}
int32_t aclrtFree(void* pointer) { freed = pointer; return Record(8); }
int32_t launch_sparse_prefill_adaptive_prefix(void* stream, void* topk, void* lists,
    void* descriptors, uint32_t history, uint32_t query) {
    prefix = {reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(topk),
        reinterpret_cast<uintptr_t>(lists), reinterpret_cast<uintptr_t>(descriptors),
        history, query};
    return Record(5);
}
int32_t aclrtLaunchKernelWithHostArgs(aclrtFuncHandle function, uint32_t n, void* stream,
    const aclrtLaunchKernelCfg* config, const void* args, size_t size,
    const void* extra, size_t extraSize) {
    if (function != reinterpret_cast<void*>(0x222) || !stream ||
        extra || extraSize || size > arguments.size()) return -99;
    cores = n; attributeId = config->attrs[0].id;
    schema = config->attrs[0].value.schemMode; configCount = config->numAttrs;
    argumentSize = size; std::memcpy(arguments.data(), args, size);
    return Record(6);
}
}
"""


class KernelArguments(ctypes.Structure):
    """The 104-byte ABI generated for the target fixed kernel."""

    _fields_ = (
        [
            (name, ctypes.c_void_p)
            for name in (
                "dump",
                "ffts",
                "lists",
                "descriptors",
                "membership",
                "counters",
                "groups",
                "bitmap",
                "sources",
                "lengths",
            )
        ]
        + [(name, ctypes.c_uint32) for name in ("history", "query", "width", "capacity")]
        + [("overflow", ctypes.c_void_p)]
    )


class SelectedKernelLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required")
        directory = tempfile.TemporaryDirectory(prefix="selected-kv-launch-")
        cls.addClassCleanup(directory.cleanup)
        build = Path(directory.name)
        (build / "acl").mkdir()
        (build / "acl/acl.h").write_text(ACL_HEADER)
        (build / "stub.cpp").write_text(ACL_STUB)
        library = build / "liblaunch.so"
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                "-std=c++17",
                "-Werror",
                f"-I{build}",
                str(ROOT / "csrc/sparse_prefill_native/sparse_prefill_fused_fixed_launch.cpp"),
                str(build / "stub.cpp"),
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
        )
        cls.library = ctypes.CDLL(str(library))
        cls.create = cls.library.create_sparse_prefill_fixed_handle
        cls.create.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        cls.create.restype = ctypes.c_int32
        cls.destroy = cls.library.destroy_sparse_prefill_fused_handle
        cls.destroy.argtypes = [ctypes.c_void_p]
        cls.destroy.restype = ctypes.c_int32
        cls.launch = cls.library.launch_sparse_prefill_fused_plan
        cls.launch.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_uint32] * 4
        cls.launch.restype = ctypes.c_int32
        cls.library.test_reset.argtypes = []
        cls.library.test_reset.restype = None
        cls.library.test_status.argtypes = [ctypes.c_uint32, ctypes.c_int32]
        cls.library.test_status.restype = None
        cls.library.test_sequence.restype = ctypes.c_uint64
        cls.library.test_argument_size.restype = ctypes.c_size_t
        cls.library.test_arguments.restype = ctypes.c_void_p
        cls.library.test_function_name.restype = ctypes.c_char_p
        cls.library.test_prefix.argtypes = [ctypes.c_uint32]
        cls.library.test_prefix.restype = ctypes.c_uint64
        cls.library.test_dump_bytes.restype = ctypes.c_size_t
        cls.library.test_malloc_policy.restype = ctypes.c_int32
        cls.library.test_freed.restype = ctypes.c_void_p

    def setUp(self):
        self.library.test_reset()
        self.handles = []

    def tearDown(self):
        self.library.test_status(4, 0)
        self.library.test_status(8, 0)
        for handle in self.handles:
            self.assertEqual(self.destroy(handle), 0)

    def make_handle(self, capacity=524288):
        handle = ctypes.c_void_p()
        self.assertEqual(self.create(b"device.o", capacity, ctypes.byref(handle)), 0)
        self.handles.append(handle)
        return handle

    @staticmethod
    def launch_values(handle):
        return [handle, 0x100, *(0x200 + 0x100 * i for i in range(9)), 65536, 256, 2048, 524288]

    def test_create_selects_target_kernel_and_reserves_assertion_dump(self):
        self.make_handle()
        self.assertEqual(self.library.test_sequence(), 1237)
        self.assertEqual(self.library.test_function_name(), b"sparse_prefill_partition_finish_fixed_0")
        self.assertEqual(self.library.test_dump_bytes(), 75 * 1024)
        self.assertEqual(self.library.test_malloc_policy(), 2)

    def test_prefix_precedes_launch_and_argument_bytes_match_target_abi(self):
        handle = self.make_handle()
        self.library.test_reset()
        values = self.launch_values(handle)
        self.assertEqual(self.launch(*values), 0)
        self.assertEqual(self.library.test_sequence(), 56)
        self.assertEqual(
            [self.library.test_prefix(i) for i in range(6)],
            [0x100, 0x200, 0x300, 0x400, 65536, 256],
        )
        expected = KernelArguments(0x444, 0x333, *values[3:11], *values[11:], None)
        size = self.library.test_argument_size()
        self.assertEqual(size, ctypes.sizeof(expected))
        self.assertEqual(ctypes.string_at(self.library.test_arguments(), size), bytes(expected))
        self.assertEqual(self.library.test_cores(), 32)
        self.assertEqual(self.library.test_attribute_id(), 9)
        self.assertEqual((self.library.test_schema(), self.library.test_config_count()), (1, 1))

    def test_maximum_history_and_query_keep_exact_pointer_and_scalar_offsets(self):
        handle = self.make_handle()
        values = self.launch_values(handle)
        values[-4:] = [524288, 8192, 16384, 524288]
        self.assertEqual(self.launch(*values), 0)
        actual = KernelArguments.from_buffer_copy(
            ctypes.string_at(self.library.test_arguments(), ctypes.sizeof(KernelArguments))
        )
        self.assertEqual((actual.history, actual.query, actual.width, actual.capacity), tuple(values[-4:]))
        self.assertEqual(actual.dump, 0x444)
        self.assertIsNone(actual.overflow)
        # The target kernel has no topk argument; the generated entry is dump,
        # FFTS, eight user pointers, four uint32 scalars, hidden overflow.
        self.assertEqual(ctypes.sizeof(KernelArguments), 104)
        self.assertEqual((KernelArguments.dump.offset, KernelArguments.ffts.offset), (0, 8))
        self.assertEqual((KernelArguments.history.offset, KernelArguments.overflow.offset), (80, 96))

    def test_irregular_history_and_query_reach_prefix_and_fixed_abi(self):
        handle = self.make_handle()
        for history in (65537, 65661, 65663, 524287):
            for query in (1, 2, 125, 127, 129, 1023):
                with self.subTest(history=history, query=query):
                    values = self.launch_values(handle)
                    width = ((history + 4095) // 4096) * 128
                    values[-4:] = [history, query, width, 524288]
                    self.assertEqual(self.launch(*values), 0)
                    self.assertEqual(
                        [self.library.test_prefix(i) for i in (4, 5)], [history, query]
                    )
                    actual = KernelArguments.from_buffer_copy(
                        ctypes.string_at(self.library.test_arguments(), ctypes.sizeof(KernelArguments))
                    )
                    self.assertEqual((actual.history, actual.query, actual.width), (history, query, width))

    def test_prefix_error_stops_fixed_launch(self):
        handle = self.make_handle()
        self.library.test_reset()
        self.library.test_status(5, 71)
        self.assertEqual(self.launch(*self.launch_values(handle)), 71)
        self.assertEqual(self.library.test_sequence(), 5)
        self.assertEqual(self.library.test_argument_size(), 0)

    def test_launch_error_is_returned(self):
        handle = self.make_handle()
        self.library.test_reset()
        self.library.test_status(6, 72)
        self.assertEqual(self.launch(*self.launch_values(handle)), 72)
        self.assertEqual(self.library.test_sequence(), 56)

    def test_failed_create_cleans_binary_without_publishing_handle(self):
        for operation, sequence in ((1, 1), (2, 124), (3, 1234), (7, 12374)):
            with self.subTest(operation=operation):
                self.library.test_reset()
                self.library.test_status(operation, 70 + operation)
                handle = ctypes.c_void_p(0xBAD)
                self.assertEqual(self.create(b"device.o", 524288, ctypes.byref(handle)), 70 + operation)
                self.assertIsNone(handle.value)
                self.assertEqual(self.library.test_sequence(), sequence)

    def test_destroy_error_preserves_handle_for_retry(self):
        handle = self.make_handle()
        self.library.test_status(4, 74)
        self.assertEqual(self.destroy(handle), 74)
        self.assertEqual(self.launch(*self.launch_values(handle)), 0)
        self.library.test_status(4, 0)
        self.assertEqual(self.destroy(handle), 0)
        self.handles.remove(handle)
        self.assertEqual(self.destroy(None), 0)

    def test_dump_release_failure_keeps_storage_for_retry_without_unloaded_binary(self):
        handle = self.make_handle()
        self.library.test_reset()
        self.library.test_status(8, 78)
        self.assertEqual(self.destroy(handle), 78)
        self.assertEqual(self.library.test_sequence(), 48)
        self.assertEqual(self.library.test_freed(), 0x444)
        self.assertEqual(self.launch(*self.launch_values(handle)), -1)
        self.assertEqual(self.library.test_sequence(), 48)
        self.library.test_status(8, 0)
        self.assertEqual(self.destroy(handle), 0)
        self.assertEqual(self.library.test_sequence(), 488)
        self.handles.remove(handle)

    def test_successful_close_releases_assertion_storage_once(self):
        handle = self.make_handle()
        self.library.test_reset()
        self.assertEqual(self.destroy(handle), 0)
        self.assertEqual(self.library.test_sequence(), 48)
        self.assertEqual(self.library.test_freed(), 0x444)
        self.handles.remove(handle)

    def test_invalid_create_never_calls_acl(self):
        for base in (0, 1, 127, 524416, 524289):
            with self.subTest(base=base):
                handle = ctypes.c_void_p(0xBAD)
                self.assertEqual(self.create(b"device.o", base, ctypes.byref(handle)), -1)
                self.assertIsNone(handle.value)
        self.assertEqual(self.create(None, 128, ctypes.byref(ctypes.c_void_p())), -1)
        self.assertEqual(self.create(b"device.o", 128, None), -1)
        self.assertEqual(self.library.test_sequence(), 0)

    def test_invalid_launch_never_submits_prefix(self):
        handle = self.make_handle(capacity=65536)
        for index, invalid in (
            (1, None),
            (2, None),
            (3, None),
            (11, 524289),
            (12, 0),
            (12, 8193),
            (13, 127),
            (13, 129),
            (13, 16385),
            (14, 131072),
        ):
            with self.subTest(index=index, value=invalid):
                self.library.test_reset()
                values = self.launch_values(handle)
                values[14] = 65536
                values[index] = invalid
                self.assertEqual(self.launch(*values), -1)
                self.assertEqual(self.library.test_sequence(), 0)

    def test_width_must_cover_history(self):
        handle = self.make_handle()
        self.library.test_reset()
        values = self.launch_values(handle)
        values[-4:] = [65536, 256, 128, 524288]
        self.assertEqual(self.launch(*values), -1)
        self.assertEqual(self.library.test_sequence(), 0)

    def test_launch_without_handle_is_rejected(self):
        self.assertEqual(self.launch(*self.launch_values(None)), -1)
        self.assertEqual(self.library.test_sequence(), 0)

    def test_cmake_registers_host_launcher_and_installs_kernel_object(self):
        build = (ROOT / "csrc/sparse_prefill_native/CMakeLists.txt").read_text()
        self.assertIn("add_library(sparse_prefill_fused_fixed_launch SHARED", build)
        self.assertIn("ASCENDC_DUMP=1 ONE_CORE_DUMP_SIZE=1024", build)
        self.assertIn("-DASCENDC_DUMP=1 -DONE_CORE_DUMP_SIZE=1024", build)
        self.assertIn("sparse_prefill_fused_fixed.o", build)
        self.assertIn('"${SELECTED_KV_SOURCE_DIR}/li_page_gather_queued.cpp"', build)
        self.assertIn("dsa_selected_li_pages ${TORCH_LIBRARIES}", build)
        self.assertEqual(build.count("ascendc_compile_definitions("), 2)
        self.assertEqual(build.count("ascendc_compile_options("), 2)


QUEUE_SOURCE = r"""
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "sparse_prefill_batch_queued.cpp"

int unionCalls = 0, copyCalls = 0, opCommands = 0;
int32_t* topkBase = nullptr;
void* scratchPointers[6]{};
int32_t* sourceOutput = nullptr;
int32_t* statusOutput = nullptr;
void* const expectedStream = reinterpret_cast<void*>(0x123);

at::Tensor Make(at::ScalarType dtype, std::vector<int64_t> shape, std::vector<int64_t> stride = {}) {
    int64_t bytes = 8;
    for (auto n : shape) bytes *= n;
    if (!stride.empty()) bytes += 8 * shape[0] * stride[0];
    auto options = at::TensorOptions().dtype(dtype).device(c10::DeviceType::PrivateUse1, 0);
    auto maker = at::for_blob(std::malloc(bytes), shape);
    maker.options(options).target_device(options.device()).deleter([](void* p) { std::free(p); });
    if (!stride.empty()) maker.strides(stride);
    return maker.make_tensor();
}

int32_t Union(void* handle, void* stream, void* topk, void* lists, void* descriptors, void* membership,
              void* counters, void* groups, void* bitmap, void* sources, void* lengths,
              uint32_t h, uint32_t q, uint32_t width, uint32_t capacity) {
    assert(handle == reinterpret_cast<void*>(1) && stream == expectedStream);
    assert(h == 65536 && q == 128 && width == 2048 && capacity == 128);
    assert(static_cast<int32_t*>(topk) == topkBase + (unionCalls == 0 ? 0 : 768) * 2048);
    void* actual[] = {lists, descriptors, membership, counters, groups, bitmap};
    for (int i = 0; i < 6; ++i) assert(actual[i] == scratchPointers[i]);
    assert(sources == sourceOutput + unionCalls * capacity && lengths == statusOutput + unionCalls * 8);
    ++unionCalls;
    return 0;
}

int32_t CopyEntry(void* stream, void*, void*, void*, void*, void*, void*, uint32_t n, uint32_t c,
                  uint32_t sourcePages, uint32_t rows, uint32_t columns, uint32_t stride, uint32_t pages) {
    assert(stream == expectedStream);
    assert(n == 2 && c == 128 && sourcePages == 8 && rows == 4 && columns == 8 && stride == 12 && pages == 8);
    ++copyCalls;
    return 0;
}

template <class F> void Reject(F fn) {
    const auto before = opCommands;
    bool rejected = false;
    try { fn(); } catch (const c10::Error&) { rejected = true; }
    assert(rejected && opCommands == before);
}

int main() {
    auto topk = Make(at::kInt, {1024, 1, 2048});
    topkBase = topk.data_ptr<int32_t>();
    TensorList scratch;
    for (auto n : {64 * 4096, 64 * 8, 524288, 32 * 8, 32 * 8, 16384}) scratch.push_back(Make(at::kInt, {n}));
    TensorList outputs{Make(at::kInt, {2, 128}), Make(at::kInt, {2, 8})};
    for (int i = 0; i < 6; ++i) scratchPointers[i] = scratch.get(i).data_ptr();
    sourceOutput = outputs.get(0).data_ptr<int32_t>();
    statusOutput = outputs.get(1).data_ptr<int32_t>();
    const auto unionFn = static_cast<int64_t>(reinterpret_cast<uintptr_t>(&Union));
    Plan(topk, scratch, outputs, {65536, 65536}, {0, 768}, {128, 128}, 128, unionFn, 1);
    assert(unionCalls == 2 && opCommands == 1);
    Reject([&] { Plan(topk, scratch, outputs, {65536}, {0}, {128}, 128, 0, 1); });
    Reject([&] { Plan(Make(at::kInt, {1024, 2, 2048}), scratch, outputs, {65536}, {0}, {128}, 128, unionFn, 1); });
    Reject([&] { Plan(topk, scratch, outputs, {65536}, {1024}, {1}, 128, unionFn, 1); });
    Reject([&] { Plan(topk, scratch, outputs, {65536}, {0}, {8193}, 128, unionFn, 1); });

    auto ids = Make(at::kInt, {2, 128}), state = Make(at::kInt, {2, 8}), pages = Make(at::kUInt64, {8});
    auto metadata = Make(at::kInt, {2, 4}), table = Make(at::kInt, {4, 8}, {12, 1});
    auto cache = Make(at::kChar, {8, 128, 1, 656});
    const auto copyFn = static_cast<int64_t>(reinterpret_cast<uintptr_t>(&CopyEntry));
    InplaceCopy(ids, state, pages, metadata, table, cache, copyFn);
    assert(copyCalls == 1 && opCommands == 2);
    Reject([&] { InplaceCopy(ids, state, pages, metadata, table, Make(at::kChar, {8, 128, 1, 672}), copyFn); });
    std::cout << "queue contracts passed\n";
}
"""


class SelectedQueueBindingTest(unittest.TestCase):
    """Compile the real queue wrapper with a torched 15-argument launch stub."""

    def test_queue_launches_target_plan_signature_without_topk_or_overflow_kernel_args(self):
        import torch
        from torch.utils import cpp_extension

        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        with tempfile.TemporaryDirectory(prefix="selected-queue-") as directory:
            build = Path(directory)
            (build / "acl").mkdir()
            (build / "acl/acl.h").write_text(ACL_HEADER)
            npu = build / "torch_npu/csrc"
            (npu / "core/npu").mkdir(parents=True)
            (npu / "framework").mkdir()
            (npu / "core/npu/NPUStream.h").write_text(
                "#pragma once\nnamespace c10_npu { struct StubStream { void* stream(bool) "
                "{ return reinterpret_cast<void*>(0x123); } };"
                "inline StubStream getCurrentNPUStream() { return {}; } }\n"
            )
            (npu / "core/npu/NPUGuard.h").write_text(
                "#pragma once\nnamespace c10_npu { struct NPUGuard { explicit NPUGuard(c10::Device) {} }; }\n"
            )
            (npu / "framework/OpCommand.h").write_text(
                "#pragma once\nextern int opCommands;\nnamespace at_npu::native { struct OpCommand { "
                "template<class F> static void RunOpApi(const char*, F fn) { ++opCommands; fn(); } }; }\n"
            )
            source = build / "queue.cpp"
            source.write_text(QUEUE_SOURCE)
            flags = [f"-I{path}" for path in cpp_extension.include_paths()]
            flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}")
            library_dir = Path(torch.__file__).parent / "lib"
            executable = build / "queue"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wno-unused-variable",
                    f"-I{build}",
                    f"-I{ROOT / 'csrc/sparse_prefill_native'}",
                    *flags,
                    str(source),
                    f"-L{library_dir}",
                    f"-Wl,-rpath,{library_dir}",
                    *(["-Wl,--allow-shlib-undefined"] if sys.platform.startswith("linux") else []),
                    "-ltorch_cpu",
                    "-ltorch",
                    "-lc10",
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
            )
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout, "queue contracts passed\n")


if __name__ == "__main__":
    unittest.main()
