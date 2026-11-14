# SPDX-License-Identifier: Apache-2.0
"""Exercise the real host batching helper without initializing an SDK/NPU."""

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class AddressExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required")
        cls.directory = tempfile.TemporaryDirectory(prefix="selected-kv-host-")
        cls.addClassCleanup(cls.directory.cleanup)
        source = Path(__file__).resolve().parents[3] / "csrc/sparse_prefill_native/hybm_address_export.cpp"
        library = Path(cls.directory.name) / "libexport.so"
        subprocess.run(
            [compiler, "-shared", "-fPIC", "-std=c++17", str(source), "-o", str(library)],
            check=True,
            capture_output=True,
        )
        cls.owner = ctypes.CDLL(str(library))
        cls.export = cls.owner.dsa_export_hybm_components
        array = ctypes.POINTER(ctypes.c_uint64)
        cls.export.argtypes = [
            ctypes.c_void_p,
            array,
            array,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            array,
            array,
            ctypes.POINTER(ctypes.c_int32),
        ]
        cls.export.restype = ctypes.c_int32

    def run_export(self, bases, sizes, offset, length, *, fail_at=None, discontinuous_at=None):
        calls = []
        signature = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64))

        def mapper(address, memory_type, result):
            calls.append((address, memory_type))
            if address == fail_at:
                return 7
            result[0] = address + 4096 + (1 if address == discontinuous_at else 0)
            return 0

        callback = signature(mapper)
        values = ctypes.c_uint64 * len(bases)
        result, failed, status = values(), ctypes.c_uint64(), ctypes.c_int32()
        code = self.export(
            ctypes.cast(callback, ctypes.c_void_p),
            values(*bases),
            values(*sizes),
            len(bases),
            offset,
            length,
            result,
            ctypes.byref(failed),
            ctypes.byref(status),
        )
        return code, list(result), failed.value, status.value, calls

    def test_offsets_and_full_component_spans(self):
        code, result, _, status, calls = self.run_export([8192, 32768], [4096, 8192], 128, 656)
        self.assertEqual((code, result, status), (0, [12416, 36992], 0))
        self.assertEqual(calls, [(8320, 1), (8975, 1), (32896, 1), (33551, 1)])

    def test_mixed_allocation_extents_map_only_the_required_prefix(self):
        code, result, _, status, calls = self.run_export(
            [8192, 32768], [656, 8192], 0, 656, discontinuous_at=32768 + 8192 - 1
        )
        self.assertEqual((code, result, status), (0, [12288, 36864], 0))
        self.assertEqual(calls, [(8192, 1), (8847, 1), (32768, 1), (33423, 1)])

    def test_required_prefix_endpoint_is_still_validated_for_each_object(self):
        for fault in ("mapping", "continuity"):
            with self.subTest(fault=fault):
                kwargs = {"fail_at" if fault == "mapping" else "discontinuous_at": 33423}
                code, _, failed, _, calls = self.run_export([8192, 32768], [656, 8192], 0, 656, **kwargs)
                self.assertEqual((code, failed), (2 if fault == "mapping" else 3, 1))
                self.assertEqual(calls[-1], (33423, 1))

    def test_each_object_keeps_its_extent_check_in_one_batch(self):
        code, _, failed, _, calls = self.run_export([8192, 32768], [8192, 655], 0, 656)
        self.assertEqual((code, failed), (1, 1))
        self.assertEqual(calls, [(8192, 1), (8847, 1)])

    def test_mapping_failure_is_not_a_valid_source(self):
        code, _, failed, status, _ = self.run_export([8192, 32768], [4096, 4096], 0, 656, fail_at=32768)
        self.assertEqual((code, failed, status), (2, 1, 7))

    def test_discontinuous_mapping_is_rejected(self):
        code, _, failed, _, _ = self.run_export([8192], [4096], 0, 656, discontinuous_at=8847)
        self.assertEqual((code, failed), (3, 0))

    def test_invalid_component_never_calls_mapper(self):
        for bases, sizes, offset, count in (
            ([0], [4096], 0, 1),
            ([8192], [64], 0, 65),
            ([8192], [64], 65, 1),
            ([(1 << 64) - 8], [16], 0, 1),
        ):
            with self.subTest(bases=bases, sizes=sizes, offset=offset, count=count):
                code, _, _, _, calls = self.run_export(bases, sizes, offset, count)
                self.assertEqual(code, 1)
                self.assertEqual(calls, [])

    def test_empty_batch_does_not_map_anything(self):
        code, result, _, _, calls = self.run_export([], [], 0, 656)
        self.assertEqual((code, result, calls), (0, [], []))


if __name__ == "__main__":
    unittest.main()
