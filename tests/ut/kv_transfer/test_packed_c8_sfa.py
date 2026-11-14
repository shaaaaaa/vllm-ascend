# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.attention.packed_c8_sfa import packed_c8_sparse_flash_attention


class TestPackedC8SparseFlashAttention(unittest.TestCase):
    def setUp(self):
        self.result = torch.zeros(3, 2, 512, dtype=torch.bfloat16)
        self.op = Mock(return_value=self.result)
        self.backend = SimpleNamespace(npu_kv_quant_sparse_flash_attention=self.op)
        self.inputs = {
            "ql_nope": torch.full((3, 2, 512), 2, dtype=torch.bfloat16),
            "q_pe": torch.full((3, 2, 64), 3, dtype=torch.bfloat16),
            "packed_main": torch.zeros(2, 128, 1, 656, dtype=torch.int8),
            "topk_indices": torch.zeros(3, 1, 16, dtype=torch.int32),
            "scale": 0.25,
            "block_table": torch.tensor([[0], [1]], dtype=torch.int32),
            "actual_seq_lengths_query": torch.tensor([1, 3], dtype=torch.int32),
            "actual_seq_lengths_kv": torch.tensor([72, 117], dtype=torch.int32),
        }

    def test_calls_official_operator_with_packed_cache_and_tnd_metadata(self):
        with patch.dict("sys.modules", {"torch_npu": self.backend}):
            self.assertIs(packed_c8_sparse_flash_attention(**self.inputs), self.result)

        self.op.assert_called_once()
        args, kwargs = self.op.call_args
        self.assertEqual(args, ())
        query = kwargs.pop("query")
        self.assertEqual(query.shape, (3, 2, 576))
        self.assertEqual(query.dtype, torch.bfloat16)
        self.assertTrue(query.is_contiguous())
        torch.testing.assert_close(query[..., :512], self.inputs["ql_nope"])
        torch.testing.assert_close(query[..., 512:], self.inputs["q_pe"])
        for argument, source in (
            ("key", "packed_main"),
            ("value", "packed_main"),
            ("sparse_indices", "topk_indices"),
            ("block_table", "block_table"),
            ("actual_seq_lengths_query", "actual_seq_lengths_query"),
            ("actual_seq_lengths_kv", "actual_seq_lengths_kv"),
        ):
            self.assertIs(kwargs.pop(argument), self.inputs[source])
        self.assertEqual(
            kwargs,
            {
                "scale_value": 0.25,
                "key_quant_mode": 2,
                "value_quant_mode": 2,
                "sparse_block_size": 1,
                "layout_query": "TND",
                "layout_kv": "PA_BSND",
                "sparse_mode": 3,
                "attention_mode": 2,
                "quant_scale_repo_mode": 1,
                "tile_size": 128,
                "rope_head_dim": 64,
            },
        )

    def test_rejects_wrong_page_geometry_and_padded_page_pitch(self):
        # Each page's rows are packed, but the gap between pages retains the
        # old BF16 pitch. The operator derives its pitch from the C8 row size.
        padded_cache = torch.empty(2 * 128 * 1152, dtype=torch.int8).as_strided(
            (2, 128, 1, 656),
            (128 * 1152, 656, 656, 1),
        )
        incompatible_caches = (
            torch.zeros(2, 64, 1, 656, dtype=torch.int8),
            torch.zeros(2, 128, 1, 655, dtype=torch.int8),
            torch.zeros(2, 128, 1, 656, dtype=torch.bfloat16),
            padded_cache,
        )
        with patch.dict("sys.modules", {"torch_npu": self.backend}):
            for cache in incompatible_caches:
                with self.subTest(shape=cache.shape, stride=cache.stride(), dtype=cache.dtype):
                    with self.assertRaisesRegex(ValueError, "contiguous INT8"):
                        packed_c8_sparse_flash_attention(**{**self.inputs, "packed_main": cache})
        self.op.assert_not_called()

    def test_rejects_incompatible_query_parts(self):
        incompatible_rope = (
            torch.zeros(3, 2, 63, dtype=torch.bfloat16),
            torch.zeros(3, 1, 64, dtype=torch.bfloat16),
            torch.zeros(3, 2, 64, dtype=torch.float16),
        )
        with patch.dict("sys.modules", {"torch_npu": self.backend}):
            for rope in incompatible_rope:
                with self.subTest(shape=rope.shape, dtype=rope.dtype):
                    with self.assertRaisesRegex(ValueError, "matching BF16 TND"):
                        packed_c8_sparse_flash_attention(**{**self.inputs, "q_pe": rope})
        self.op.assert_not_called()

    def test_rejects_mismatched_batch_metadata(self):
        with (
            patch.dict("sys.modules", {"torch_npu": self.backend}),
            self.assertRaisesRegex(ValueError, "same batch"),
        ):
            packed_c8_sparse_flash_attention(**{**self.inputs, "block_table": self.inputs["block_table"][:1]})
        self.op.assert_not_called()

    def test_requires_official_operator(self):
        with (
            patch.dict("sys.modules", {"torch_npu": SimpleNamespace()}),
            self.assertRaisesRegex(RuntimeError, "does not provide npu_kv_quant_sparse_flash_attention"),
        ):
            packed_c8_sparse_flash_attention(**self.inputs)

    def test_rejects_metadata_on_a_different_device(self):
        wrong_device = self.inputs["topk_indices"].to(device="meta")
        with (
            patch.dict("sys.modules", {"torch_npu": self.backend}),
            self.assertRaisesRegex(ValueError, "same device"),
        ):
            packed_c8_sparse_flash_attention(**{**self.inputs, "topk_indices": wrong_device})
        self.op.assert_not_called()
