# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from vllm_ascend.worker.packed_c8_shared_views import (
    packed_c8_prefill_mixed_shared_views,
    packed_c8_prefill_shared_views,
)

_RAW_BUNDLE_BYTES = 2_686_976
_SCALE_BUNDLE_BYTES = 41_984


class TestPackedC8PrefillSharedViews(unittest.TestCase):
    def test_slices_share_storage_and_preserve_reserved_bundle_offsets(self):
        # Runner's raw and sidecar regions can be adjacent in one allocation.
        raw_bytes = 2 * _RAW_BUNDLE_BYTES
        scale_bytes = 2 * _SCALE_BUNDLE_BYTES
        prefix = 8
        backing = torch.zeros(prefix + raw_bytes + scale_bytes, dtype=torch.int8)
        raw = backing[prefix : prefix + raw_bytes]
        scale = backing[prefix + raw_bytes :]

        main, keys, scales = packed_c8_prefill_shared_views(raw, scale, 1)
        self.assertEqual(main.shape, (64, 128, 1, 656))
        self.assertEqual(keys.shape, (328, 128, 1, 128))
        self.assertEqual(scales.shape, (328, 128, 1, 1))
        self.assertEqual((main.dtype, keys.dtype, scales.dtype), (torch.int8, torch.int8, torch.float16))
        for view, region in ((main, raw), (keys, raw), (scales, scale)):
            self.assertTrue(view.is_contiguous())
            self.assertEqual(view.data_ptr(), region.data_ptr())
            self.assertEqual(view.untyped_storage().data_ptr(), backing.untyped_storage().data_ptr())
            self.assertEqual(view.storage_offset() * view.element_size(), region.storage_offset())

        self.assertEqual(main[1].data_ptr() - raw.data_ptr(), 83_968)
        self.assertEqual(main[32].data_ptr() - raw.data_ptr(), _RAW_BUNDLE_BYTES)
        self.assertEqual(keys[2].data_ptr() - raw.data_ptr(), 32_768)
        self.assertEqual(keys[164].data_ptr() - raw.data_ptr(), _RAW_BUNDLE_BYTES)
        self.assertEqual(scales[2].data_ptr() - scale.data_ptr(), 512)
        self.assertEqual(scales[164].data_ptr() - scale.data_ptr(), _SCALE_BUNDLE_BYTES)

        # The first byte of bundle one aliases both possible owner views.
        main[32, 0, 0, 0] = 23
        self.assertEqual(keys[164, 0, 0, 0].item(), 23)
        self.assertEqual(backing[prefix + _RAW_BUNDLE_BYTES].item(), 23)
        self.assertEqual(main[0, 0, 0, 0].item(), 0)
        scales[164, 0, 0, 0] = 1.5
        self.assertEqual(scale[_SCALE_BUNDLE_BYTES : _SCALE_BUNDLE_BYTES + 2].view(torch.float16).item(), 1.5)
        self.assertEqual(scales[0, 0, 0, 0].item(), 0)

    def test_mixed_indexer_views_keep_logical_block_addresses(self):
        prefix = 8
        raw_bytes = 2 * _RAW_BUNDLE_BYTES
        scale_bytes = 2 * _SCALE_BUNDLE_BYTES
        backing = torch.zeros(prefix + raw_bytes + scale_bytes, dtype=torch.int8)
        raw = backing[prefix : prefix + raw_bytes]
        scale = backing[prefix + raw_bytes :]
        main, c8_keys, c8_scales, bf16_keys = packed_c8_prefill_mixed_shared_views(raw, scale, 1)

        self.assertEqual(bf16_keys.shape, (164, 128, 1, 128))
        self.assertEqual(bf16_keys.dtype, torch.bfloat16)
        self.assertTrue(bf16_keys.is_contiguous())
        self.assertEqual(bf16_keys.stride(0) * bf16_keys.element_size(), 32_768)
        self.assertEqual(c8_keys.stride(0) * c8_keys.element_size(), 16_384)
        self.assertEqual(bf16_keys.storage_offset() * bf16_keys.element_size(), prefix)
        for view in (main, c8_keys, bf16_keys):
            self.assertEqual(view.data_ptr(), raw.data_ptr())
            self.assertEqual(view.untyped_storage().data_ptr(), backing.untyped_storage().data_ptr())
        self.assertEqual(c8_scales.data_ptr(), raw.data_ptr() + raw_bytes)
        self.assertEqual(c8_scales.untyped_storage().data_ptr(), backing.untyped_storage().data_ptr())

        # Check bundle zero, the first usable bundle, and the final page.
        for logical in (0, 1, 82, 163):
            with self.subTest(logical=logical):
                self.assertEqual(bf16_keys[logical].data_ptr() - raw.data_ptr(), logical * 32_768)
                self.assertEqual(c8_keys[2 * logical].data_ptr(), bf16_keys[logical].data_ptr())
                self.assertEqual(c8_keys[2 * logical + 1].data_ptr(), bf16_keys[logical].data_ptr() + 16_384)
                self.assertEqual(c8_scales[2 * logical].data_ptr() - scale.data_ptr(), logical * 512)
                self.assertEqual(c8_scales[2 * logical + 1].data_ptr() - scale.data_ptr(), logical * 512 + 256)
        self.assertEqual(main[32].data_ptr(), bf16_keys[82].data_ptr())
        self.assertEqual(bf16_keys[-1].data_ptr() + 32_768, scale.data_ptr())
        self.assertEqual(c8_keys[-1].data_ptr() + 16_384, scale.data_ptr())
        self.assertEqual(c8_scales[-1].data_ptr() + 256, scale.data_ptr() + scale_bytes)

        # Read the same BF16 writes as bytes through each C8 extent.
        bf16_keys[82, 0, 0, 0] = 1.5
        bf16_keys[82, 64, 0, 0] = -2.5
        bf16_bytes = bf16_keys[82].view(torch.int8)
        torch.testing.assert_close(c8_keys[164, 0, 0, :2], bf16_bytes[0, 0, :2])
        torch.testing.assert_close(c8_keys[165, 0, 0, :2], bf16_bytes[64, 0, :2])
        self.assertEqual(bf16_keys[0, 0, 0, 0].item(), 0)
        self.assertEqual(c8_scales[164, 0, 0, 0].item(), 0)

    def test_zero_capacity_has_only_reserved_bundle(self):
        raw = torch.empty(_RAW_BUNDLE_BYTES, dtype=torch.int8)
        scale = torch.empty(_SCALE_BUNDLE_BYTES, dtype=torch.int8)
        main, keys, scales = packed_c8_prefill_shared_views(raw, scale, 0)
        self.assertEqual((main.shape[0], keys.shape[0], scales.shape[0]), (32, 164, 164))
        self.assertEqual(main[-1].data_ptr() + 83_968, raw.data_ptr() + raw.numel())
        self.assertEqual(keys[-1].data_ptr() + 16_384, raw.data_ptr() + raw.numel())
        self.assertEqual(scales[-1].data_ptr() + 256, scale.data_ptr() + scale.numel())

    def test_rejects_visible_size_and_invalid_capacity(self):
        raw = torch.empty(_RAW_BUNDLE_BYTES, dtype=torch.int8)
        scale = torch.empty(_SCALE_BUNDLE_BYTES, dtype=torch.int8)
        for capacity in (-1, True, 1.0):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                packed_c8_prefill_shared_views(raw, scale, capacity)
        with self.assertRaisesRegex(ValueError, "raw must expose exactly"):
            packed_c8_prefill_shared_views(raw[:-1], scale, 0)
        with self.assertRaisesRegex(ValueError, "scale must expose exactly"):
            packed_c8_prefill_shared_views(raw, scale[:-1], 0)
        with self.assertRaisesRegex(ValueError, "raw must expose exactly"):
            packed_c8_prefill_shared_views(raw, scale, 1)

    def test_rejects_incompatible_tensor_metadata_and_overlap(self):
        raw = torch.empty(2 * _RAW_BUNDLE_BYTES, dtype=torch.int8)
        scale = torch.empty(_SCALE_BUNDLE_BYTES + 2, dtype=torch.int8)
        raw_slice = raw[:_RAW_BUNDLE_BYTES]
        scale_slice = scale[:_SCALE_BUNDLE_BYTES]
        for invalid_raw in (raw[::2], raw_slice.view(torch.uint8), raw_slice.view(1, -1)):
            with self.subTest(dtype=invalid_raw.dtype, stride=invalid_raw.stride()):
                with self.assertRaisesRegex(ValueError, "contiguous one-dimensional INT8"):
                    packed_c8_prefill_shared_views(invalid_raw, scale_slice, 0)
        with self.assertRaisesRegex(ValueError, "scale must be a contiguous"):
            packed_c8_prefill_shared_views(raw_slice, scale_slice.view(torch.float16), 0)
        with self.assertRaisesRegex(ValueError, "same device"):
            packed_c8_prefill_shared_views(raw_slice, scale_slice.to("meta"), 0)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            packed_c8_prefill_shared_views(raw_slice, raw_slice[:_SCALE_BUNDLE_BYTES], 0)
        with self.assertRaisesRegex(ValueError, "aligned to 4 bytes"):
            packed_c8_prefill_shared_views(raw[1 : 1 + _RAW_BUNDLE_BYTES], scale_slice, 0)
        with self.assertRaisesRegex(ValueError, "aligned to 2 bytes"):
            packed_c8_prefill_shared_views(raw_slice, scale[1 : 1 + _SCALE_BUNDLE_BYTES], 0)
