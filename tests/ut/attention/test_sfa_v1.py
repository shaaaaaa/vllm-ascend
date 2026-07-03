import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import torch

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm.distributed.parallel_state import GroupCoordinator

if 'torch_npu._inductor' not in sys.modules:
    sys.modules['torch_npu._inductor'] = MagicMock()

from vllm_ascend.attention.sfa_v1 import (
    AscendSFABackend,
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    _dense_prefix_capture_loaded_snapshot,
    _dense_prefix_compare_build_sample,
    _dense_prefix_compare_cache,
    _dense_prefix_compare_diff,
    _dense_prefix_compare_direct_call,
    _dense_prefix_compare_enabled,
    _dense_prefix_compare_load_sample,
    _dense_prefix_compare_save_sample,
    _dense_prefix_full_snapshot_build,
    _dense_prefix_full_snapshot_diff,
    _dense_prefix_full_snapshot_load,
    _dense_prefix_full_snapshot_replay,
    _dense_prefix_full_snapshot_save_payload,
    _dense_prefix_replay_dev_lmy_if_loaded_mismatch,
    _dsa_env_flag,
    _sfa_path_trace_enabled,
    _sfa_path_trace_should_wrap,
    _sfa_trace_lmcache_call,
)
from vllm_ascend.utils import enable_dsa_cp


class TestAscendSFABackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFABackend.get_builder_cls(),
                         AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)


class TestAscendSFAMetadata(TestBase):

    def test_ascend_sfa_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        seq_lens = torch.tensor([30, 50])
        cum_query_lens = torch.tensor([0, 30, 80])
        block_table = torch.randint(0, 100, (100, 4))

        rope_dim = 32
        max_seq_len = int(seq_lens.max().item())
        sin = torch.randn(max_seq_len, rope_dim)
        cos = torch.randn(max_seq_len, rope_dim)

        num_input_tokens = 2
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        metadata = AscendSFAMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            num_input_tokens=num_input_tokens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
        )

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertTrue(torch.equal(metadata.cum_query_lens, cum_query_lens))
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.sin, sin)
        self.assertIs(metadata.cos, cos)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertIs(metadata.head_dim, head_dim)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)


class TestDensePrefixCompareHelpers(TestBase):

    def test_dense_prefix_compare_build_sample_excludes_recalc_last_slot(self):
        cache = torch.arange(3 * 4 * 1 * 2, dtype=torch.float32).reshape(3, 4, 1, 2)
        block_table = torch.tensor([[0, 1, 2]], dtype=torch.long)

        with patch.dict(os.environ, {"VLLM_ASCEND_DENSE_PREFIX_COMPARE_SAMPLES": "8"}):
            sample, error = _dense_prefix_compare_build_sample(cache, block_table, 10)

        self.assertIsNone(error)
        assert sample is not None
        self.assertEqual(sample["positions"], [0, 1, 2, 3, 5, 6, 7, 8])
        self.assertEqual(sample["slots"], [0, 1, 2, 3, 5, 6, 7, 8])
        expected = cache.reshape(-1, 1, 2).index_select(
            0, torch.tensor(sample["slots"], dtype=torch.long)
        )
        self.assertTrue(torch.equal(sample["values"], expected))

    def test_dense_prefix_compare_diff_detects_loaded_value_mismatch(self):
        cache = torch.arange(2 * 4 * 1 * 2, dtype=torch.float32).reshape(2, 4, 1, 2)
        block_table = torch.tensor([[0, 1]], dtype=torch.long)

        with patch.dict(os.environ, {"VLLM_ASCEND_DENSE_PREFIX_COMPARE_SAMPLES": "4"}):
            baseline, error = _dense_prefix_compare_build_sample(cache, block_table, 8)
            current, current_error = _dense_prefix_compare_build_sample(
                cache.clone(), block_table, 8
            )

        self.assertIsNone(error)
        self.assertIsNone(current_error)
        assert baseline is not None
        assert current is not None
        self.assertTrue(_dense_prefix_compare_diff(baseline, current)["match"])

        current["slots"] = [slot + 100 for slot in current["slots"]]
        moved_slots_diff = _dense_prefix_compare_diff(baseline, current)
        self.assertTrue(moved_slots_diff["match"])
        self.assertFalse(moved_slots_diff["same_slots"])

        current["values"] = current["values"].clone()
        current["values"].reshape(-1)[0] += 3.0
        diff = _dense_prefix_compare_diff(baseline, current)
        self.assertFalse(diff["match"])
        self.assertEqual(diff["max_abs_diff"], 3.0)
        self.assertEqual(diff["first_diff_flat_index"], 0)

    def test_dense_prefix_compare_sample_uses_rank_aware_file(self):
        sample = {
            "seq_len": 4,
            "positions": [0, 1],
            "slots": [0, 1],
            "values": torch.tensor([[1.0], [2.0]]),
            "summary": {"shape": (2, 1)},
        }

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_DIR": tmpdir,
                    "RANK": "3",
                    "LOCAL_RANK": "3",
                    "WORLD_SIZE": "8",
                },
                clear=False,
            ),
            patch(
                "vllm_ascend.attention.sfa_v1.get_tensor_model_parallel_world_size",
                return_value=8,
            ),
            patch(
                "vllm_ascend.attention.sfa_v1.torch.distributed.is_available",
                return_value=False,
            ),
            patch(
                "vllm_ascend.attention.sfa_v1.torch.distributed.is_initialized",
                return_value=False,
            ),
        ):
            tp_group = MagicMock()
            tp_group.rank_in_group = MagicMock(return_value=3)
            tp_group.world_size = MagicMock(return_value=8)
            path = _dense_prefix_compare_save_sample(
                layer_name="model.layers.0.self_attn.attn",
                label="latent_nope",
                stage="lmcache_saved",
                sample=sample,
                preserve_longer=True,
            )
            with patch("vllm_ascend.attention.sfa_v1.get_tp_group", return_value=tp_group):
                loaded = _dense_prefix_compare_load_sample(
                    "model.layers.0.self_attn.attn",
                    "latent_nope",
                    "lmcache_saved",
                )

        assert path is not None
        self.assertIn("rank3_", path)
        self.assertIn("_tp3_of8", path)
        assert loaded is not None
        self.assertEqual(loaded["seq_len"], sample["seq_len"])
        self.assertTrue(torch.equal(loaded["values"], sample["values"]))

    def test_dense_prefix_full_snapshot_diff_and_replay(self):
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.ChunkedPrefill
        metadata.num_decode_tokens = 0
        metadata.block_table = torch.tensor([[1, 3]], dtype=torch.long)
        metadata.indexer_block_table = torch.tensor([[0, 2]], dtype=torch.long)
        metadata.seq_lens = torch.tensor([6], dtype=torch.long)
        metadata.seq_lens_cpu = metadata.seq_lens

        kv0 = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4, 1, 1)
        kv1 = kv0 + 100
        kv2 = kv0 + 200
        payload, error = _dense_prefix_full_snapshot_build(
            source="dev_lmy",
            layer_name="model.layers.0.self_attn.attn",
            kv_cache=[kv0, kv1, kv2],
            attn_metadata=metadata,
            include_latent=True,
            include_index=True,
            require_complete=True,
            allow_decode=False,
        )
        self.assertIsNone(error)
        assert payload is not None
        self.assertTrue(_dense_prefix_full_snapshot_diff(payload, payload)["match"])

        changed_payload, changed_error = _dense_prefix_full_snapshot_build(
            source="lmcache_loaded",
            layer_name="model.layers.0.self_attn.attn",
            kv_cache=[kv0 + 1, kv1, kv2],
            attn_metadata=metadata,
            include_latent=True,
            include_index=True,
            require_complete=True,
            allow_decode=False,
        )
        self.assertIsNone(changed_error)
        assert changed_payload is not None
        self.assertFalse(
            _dense_prefix_full_snapshot_diff(payload, changed_payload)["match"]
        )
        length_changed_payload = {
            **payload,
            "meta": {
                **payload["meta"],
                "lengths": [payload["meta"]["lengths"][0] + 1],
            },
        }
        length_diff = _dense_prefix_full_snapshot_diff(
            payload, length_changed_payload
        )
        self.assertFalse(length_diff["match"])
        self.assertFalse(length_diff["same_lengths"])

        target0 = torch.zeros_like(kv0)
        target1 = torch.zeros_like(kv1)
        target2 = torch.zeros_like(kv2)
        class Owner:
            pass

        owner = Owner()
        fake_npu = MagicMock()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {
                    "VLLM_ASCEND_DENSE_PREFIX_COMPARE": "1",
                    "VLLM_ASCEND_DENSE_PREFIX_COMPARE_LAYER": "all",
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_DIR": tmpdir,
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_SYNC": "1",
                },
                clear=False,
            ),
            patch("vllm_ascend.attention.sfa_v1.torch.npu", fake_npu, create=True),
        ):
            _dense_prefix_full_snapshot_save_payload(
                layer_name="model.layers.0.self_attn.attn",
                source="dev_lmy",
                payload=payload,
                preserve_longer=False,
            )
            self.assertIsNotNone(
                _dense_prefix_full_snapshot_load(
                    "model.layers.0.self_attn.attn",
                    "dev_lmy",
                )
            )
            copied = _dense_prefix_full_snapshot_replay(
                owner,
                source="dev_lmy",
                layer_name="model.layers.0.self_attn.attn",
                kv_cache=[target0, target1, target2],
                attn_metadata=metadata,
            )

        self.assertEqual(copied, {0: 2, 1: 2, 2: 2})
        fake_npu.synchronize.assert_called_once()
        self.assertTrue(torch.equal(target0[1], kv0[1]))
        self.assertTrue(torch.equal(target0[3], kv0[3]))
        self.assertTrue(torch.equal(target1[1], kv1[1]))
        self.assertTrue(torch.equal(target1[3], kv1[3]))
        self.assertTrue(torch.equal(target2[0], kv2[0]))
        self.assertTrue(torch.equal(target2[2], kv2[2]))

    def test_dense_prefix_replay_captures_loaded_snapshot_when_missing(self):
        class Owner:
            pass

        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.num_decode_tokens = 1
        metadata.block_table = torch.tensor([[1, 3]], dtype=torch.long)
        metadata.indexer_block_table = torch.tensor([[0, 2]], dtype=torch.long)
        metadata.seq_lens = torch.tensor([6], dtype=torch.long)
        metadata.seq_lens_cpu = metadata.seq_lens

        kv0 = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4, 1, 1)
        kv1 = kv0 + 100
        kv2 = kv0 + 200
        payload, error = _dense_prefix_full_snapshot_build(
            source="dev_lmy",
            layer_name="model.layers.0.self_attn.attn",
            kv_cache=[kv0, kv1, kv2],
            attn_metadata=metadata,
            include_latent=True,
            include_index=True,
            require_complete=False,
            allow_decode=True,
        )
        self.assertIsNone(error)
        assert payload is not None

        target0 = torch.zeros_like(kv0)
        target1 = torch.zeros_like(kv1)
        target2 = torch.zeros_like(kv2)
        forward_context = MagicMock()
        forward_context.lmcache_dense_prefix_loaded = True

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {
                    "VLLM_ASCEND_DENSE_PREFIX_COMPARE": "1",
                    "VLLM_ASCEND_DENSE_PREFIX_COMPARE_LAYER": "all",
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_DIR": tmpdir,
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_PARTIAL_REPLAY": "1",
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_SYNC": "0",
                },
                clear=False,
            ),
            patch(
                "vllm_ascend.attention.sfa_v1.get_forward_context",
                return_value=forward_context,
            ),
        ):
            _dense_prefix_full_snapshot_save_payload(
                layer_name="model.layers.0.self_attn.attn",
                source="dev_lmy",
                payload=payload,
                preserve_longer=False,
            )
            self.assertIsNone(
                _dense_prefix_full_snapshot_load(
                    "model.layers.0.self_attn.attn",
                    "lmcache_loaded",
                )
            )
            _dense_prefix_replay_dev_lmy_if_loaded_mismatch(
                Owner(),
                layer_name="model.layers.0.self_attn.attn",
                kv_cache=[target0, target1, target2],
                attn_metadata=metadata,
            )
            loaded_payload = _dense_prefix_full_snapshot_load(
                "model.layers.0.self_attn.attn",
                "lmcache_loaded",
            )

        self.assertIsNotNone(loaded_payload)
        self.assertTrue(torch.equal(target0[1], kv0[1]))
        self.assertTrue(torch.equal(target0[3, :2], kv0[3, :2]))
        self.assertTrue(torch.equal(target0[3, 2:], torch.zeros_like(target0[3, 2:])))
        self.assertTrue(torch.equal(target1[1], kv1[1]))
        self.assertTrue(torch.equal(target1[3, :2], kv1[3, :2]))
        self.assertTrue(torch.equal(target1[3, 2:], torch.zeros_like(target1[3, 2:])))
        self.assertTrue(torch.equal(target2[0], kv2[0]))
        self.assertTrue(torch.equal(target2[2, :2], kv2[2, :2]))
        self.assertTrue(torch.equal(target2[2, 2:], torch.zeros_like(target2[2, 2:])))

    def test_dense_prefix_capture_loaded_snapshot_writes_current_cache(self):
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.num_decode_tokens = 1
        metadata.block_table = torch.tensor([[1, 3]], dtype=torch.long)
        metadata.indexer_block_table = torch.tensor([[0, 2]], dtype=torch.long)
        metadata.seq_lens = torch.tensor([6], dtype=torch.long)
        metadata.seq_lens_cpu = metadata.seq_lens

        kv0 = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4, 1, 1)
        forward_context = MagicMock()
        forward_context.lmcache_dense_prefix_loaded = True
        forward_context.dsa_prompt_lens = None

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {
                    "VLLM_ASCEND_DENSE_PREFIX_COMPARE": "1",
                    "VLLM_ASCEND_DENSE_PREFIX_COMPARE_LAYER": "all",
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_DIR": tmpdir,
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_SYNC": "0",
                },
                clear=False,
            ),
            patch(
                "vllm_ascend.attention.sfa_v1.get_forward_context",
                return_value=forward_context,
            ),
        ):
            payload = _dense_prefix_capture_loaded_snapshot(
                MagicMock(),
                layer_name="model.layers.0.self_attn.attn",
                kv_cache=[kv0, kv0 + 100, kv0 + 200],
                attn_metadata=metadata,
                reason="unit_test",
                overwrite=True,
            )
            loaded = _dense_prefix_full_snapshot_load(
                "model.layers.0.self_attn.attn",
                "lmcache_loaded",
            )

        assert payload is not None
        assert loaded is not None
        self.assertEqual(loaded["meta"]["source"], "lmcache_loaded")
        self.assertTrue(
            torch.equal(
                loaded["caches"][0]["rows"][0]["data"],
                payload["caches"][0]["rows"][0]["data"],
            )
        )

    def test_dense_prefix_capture_loaded_snapshot_preserves_existing_without_overwrite(self):
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.num_decode_tokens = 1
        metadata.block_table = torch.tensor([[1, 3]], dtype=torch.long)
        metadata.indexer_block_table = torch.tensor([[0, 2]], dtype=torch.long)
        metadata.seq_lens = torch.tensor([6], dtype=torch.long)
        metadata.seq_lens_cpu = metadata.seq_lens

        pre_scatter = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4, 1, 1)
        post_scatter = pre_scatter + 1000
        forward_context = MagicMock()
        forward_context.lmcache_dense_prefix_loaded = True
        forward_context.dsa_prompt_lens = None

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {
                    "VLLM_ASCEND_DENSE_PREFIX_COMPARE": "1",
                    "VLLM_ASCEND_DENSE_PREFIX_COMPARE_LAYER": "all",
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_DIR": tmpdir,
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_SYNC": "0",
                },
                clear=False,
            ),
            patch(
                "vllm_ascend.attention.sfa_v1.get_forward_context",
                return_value=forward_context,
            ),
        ):
            first = _dense_prefix_capture_loaded_snapshot(
                MagicMock(),
                layer_name="model.layers.0.self_attn.attn",
                kv_cache=[pre_scatter, pre_scatter + 100, pre_scatter + 200],
                attn_metadata=metadata,
                reason="pre_scatter",
                overwrite=True,
            )
            second = _dense_prefix_capture_loaded_snapshot(
                MagicMock(),
                layer_name="model.layers.0.self_attn.attn",
                kv_cache=[post_scatter, post_scatter + 100, post_scatter + 200],
                attn_metadata=metadata,
                reason="fallback",
                overwrite=False,
            )
            loaded = _dense_prefix_full_snapshot_load(
                "model.layers.0.self_attn.attn",
                "lmcache_loaded",
            )

        assert first is not None
        assert second is not None
        assert loaded is not None
        self.assertTrue(
            torch.equal(
                loaded["caches"][0]["rows"][0]["data"],
                first["caches"][0]["rows"][0]["data"],
            )
        )
        self.assertTrue(
            torch.equal(
                second["caches"][0]["rows"][0]["data"],
                first["caches"][0]["rows"][0]["data"],
            )
        )

    def test_dense_prefix_snapshot_uses_metadata_prompt_lens_fallback(self):
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.num_decode_tokens = 1
        metadata.block_table = torch.tensor([[1, 3]], dtype=torch.long)
        metadata.indexer_block_table = torch.tensor([[0, 2]], dtype=torch.long)
        metadata.seq_lens = torch.tensor([7], dtype=torch.long)
        metadata.seq_lens_cpu = metadata.seq_lens
        metadata.prompt_lens = torch.tensor([6], dtype=torch.long)

        kv0 = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4, 1, 1)
        forward_context = MagicMock()
        forward_context.dsa_prompt_lens = None
        forward_context.dsa_req_ids = ["r0"]

        with patch(
            "vllm_ascend.attention.sfa_v1.get_forward_context",
            return_value=forward_context,
        ):
            payload, error = _dense_prefix_full_snapshot_build(
                source="lmcache_loaded",
                layer_name="model.layers.0.self_attn.attn",
                kv_cache=[kv0, kv0 + 100, kv0 + 200],
                attn_metadata=metadata,
                include_latent=True,
                include_index=True,
                require_complete=False,
                allow_decode=True,
            )

        self.assertIsNone(error)
        assert payload is not None
        self.assertEqual(payload["meta"]["lengths"], [6])
        self.assertTrue(
            torch.equal(payload["meta"]["prompt_lens"], torch.tensor([6]))
        )

    def test_dense_prefix_compare_cache_synchronizes_before_sampling(self):
        class Owner:
            pass

        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.ChunkedPrefill
        metadata.num_actual_tokens = 4
        metadata.num_decode_tokens = 0
        metadata.seq_lens = torch.tensor([4], dtype=torch.long)
        metadata.prompt_lens = None
        metadata.block_table = torch.tensor([[0]], dtype=torch.long)

        kv_cache = [
            torch.zeros(1, 4, 1, 1),
            torch.ones(1, 4, 1, 1),
        ]
        fake_npu = MagicMock()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {
                    "VLLM_ASCEND_DENSE_PREFIX_COMPARE": "1",
                    "VLLM_ASCEND_DENSE_PREFIX_COMPARE_LAYER": "all",
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_DIR": tmpdir,
                    "VLLM_ASCEND_DENSE_PREFIX_FILE_SYNC": "1",
                },
                clear=True,
            ),
            patch("vllm_ascend.attention.sfa_v1.torch.npu", fake_npu, create=True),
        ):
            _dense_prefix_compare_cache(
                Owner(),
                stage="capture_before_store",
                layer_name="model.layers.0.self_attn.attn",
                kv_cache=kv_cache,
                attn_metadata=metadata,
                include_latent=True,
                include_index=False,
            )

        fake_npu.synchronize.assert_called_once()

    def test_diagnostics_are_opt_in_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_sfa_path_trace_enabled())
            self.assertFalse(_dense_prefix_compare_enabled())

        with patch.dict(os.environ, {"VLLM_ASCEND_SFA_V1_PATH_TRACE": "1"}, clear=True):
            self.assertTrue(_sfa_path_trace_enabled())
            self.assertFalse(_dense_prefix_compare_enabled())

        with patch.dict(os.environ, {"VLLM_ASCEND_DENSE_PREFIX_COMPARE": "1"}, clear=True):
            self.assertFalse(_sfa_path_trace_enabled())
            self.assertTrue(_dense_prefix_compare_enabled())

    def test_dense_prefix_compare_direct_call_logs_path_before_compare(self):
        class Owner:
            pass

        owner = Owner()
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.num_actual_tokens = 1
        metadata.num_decode_tokens = 1
        metadata.seq_lens = torch.tensor([8])
        metadata.prompt_lens = None
        metadata.block_table = torch.tensor([[0, 1]], dtype=torch.long)
        kv_cache = [
            torch.zeros(2, 4, 1, 2),
            torch.zeros(2, 4, 1, 2),
            torch.zeros(2, 4, 1, 2),
        ]
        forward_context = MagicMock()
        forward_context.lmcache_dense_prefix_loaded = True
        forward_context.lmcache_dense_prefix_loaded_reqs = [
            {"req_id": "r0", "token_count": 8}
        ]

        env = {
            "LMCACHE_DENSE_PREFIX_DIAG": "1",
            "VLLM_ASCEND_DENSE_PREFIX_COMPARE_LAYER": "all",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "vllm_ascend.attention.sfa_v1.get_forward_context",
                return_value=forward_context,
            ),
            patch(
                "vllm_ascend.attention.sfa_v1._dense_prefix_compare_cache"
            ) as compare_cache,
            patch("vllm_ascend.attention.sfa_v1.logger") as mock_logger,
        ):
            _dense_prefix_compare_direct_call(
                owner,
                note="before_attention",
                stage="compare_before_attention",
                layer_name="model.layers.0.self_attn.attn",
                kv_cache=kv_cache,
                attn_metadata=metadata,
                include_latent=True,
                include_index=True,
            )

        warning_messages = [
            call.args[0]
            for call in mock_logger.warning.call_args_list
            if call.args
        ]
        self.assertTrue(
            any(
                "[DENSE_PREFIX_COMPARE_PATH] direct_call" in message
                for message in warning_messages
            )
        )
        compare_cache.assert_called_once()

    def test_dense_prefix_compare_direct_call_can_suppress_path_log(self):
        class Owner:
            pass

        owner = Owner()
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.num_actual_tokens = 1
        metadata.num_decode_tokens = 1
        metadata.seq_lens = torch.tensor([8])
        metadata.prompt_lens = None
        metadata.block_table = torch.tensor([[0, 1]], dtype=torch.long)
        kv_cache = [
            torch.zeros(2, 4, 1, 2),
            torch.zeros(2, 4, 1, 2),
            torch.zeros(2, 4, 1, 2),
        ]
        forward_context = MagicMock()
        forward_context.lmcache_dense_prefix_loaded = True
        forward_context.lmcache_dense_prefix_loaded_reqs = [
            {"req_id": "r0", "token_count": 8}
        ]

        env = {
            "LMCACHE_DENSE_PREFIX_DIAG": "1",
            "VLLM_ASCEND_DENSE_PREFIX_COMPARE_LAYER": "all",
            "VLLM_ASCEND_DENSE_PREFIX_COMPARE_PATH_LOG_DISABLE": "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "vllm_ascend.attention.sfa_v1.get_forward_context",
                return_value=forward_context,
            ),
            patch(
                "vllm_ascend.attention.sfa_v1._dense_prefix_compare_cache"
            ) as compare_cache,
            patch("vllm_ascend.attention.sfa_v1.logger") as mock_logger,
        ):
            _dense_prefix_compare_direct_call(
                owner,
                note="before_attention",
                stage="compare_before_attention",
                layer_name="model.layers.0.self_attn.attn",
                kv_cache=kv_cache,
                attn_metadata=metadata,
                include_latent=True,
                include_index=True,
            )

        warning_messages = [
            call.args[0]
            for call in mock_logger.warning.call_args_list
            if call.args
        ]
        self.assertFalse(
            any(
                "[DENSE_PREFIX_COMPARE_PATH] direct_call" in message
                for message in warning_messages
            )
        )
        compare_cache.assert_called_once()

    def test_sfa_path_trace_does_not_wrap_env_flag_helper(self):
        self.assertFalse(_sfa_path_trace_should_wrap("_dsa_env_flag", _dsa_env_flag))
        self.assertTrue(
            _sfa_path_trace_should_wrap(
                "_dense_prefix_compare_enabled",
                _dense_prefix_compare_enabled,
            )
        )

    def test_sfa_trace_lmcache_call_logs_when_path_trace_enabled(self):
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.num_actual_tokens = 1
        metadata.num_decode_tokens = 1

        with (
            patch.dict(
                os.environ,
                {"VLLM_ASCEND_SFA_V1_PATH_TRACE": "1"},
                clear=True,
            ),
            patch(
                "vllm_ascend.attention.sfa_v1.has_kv_transfer_group",
                return_value=False,
            ),
            patch("vllm_ascend.attention.sfa_v1.logger") as mock_logger,
        ):
            _sfa_trace_lmcache_call(
                site="before_dense_latent_wait",
                layer_name="model.layers.0.self_attn.attn",
                attn_metadata=metadata,
            )

        mock_logger.warning.assert_called_once()
        self.assertIn(
            "[SFA_V1_LMCACHE_TRACE]",
            mock_logger.warning.call_args.args[0],
        )


class TestAscendSFAMetadataBuilder(TestBase):

    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def setUp(self, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        self.mock_cfg = MagicMock()

        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.prefill_context_parallel_size = 1
        self.mock_cfg.parallel_config.decode_context_parallel_size = 1

        self.mock_cfg.compilation_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config.enable_sp = False

        self.mock_cfg.speculative_config.num_speculative_tokens = 0

        self.patcher = patch("vllm.config.get_current_vllm_config",
                             return_value=self.mock_cfg)
        self.patcher.start()

        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(self, kv_cache_spec, layer_names, vllm_config,
                             device, metadata_cls, supports_dcp_with_varlen):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__",
            mock_parent_init)
        self.parent_init_patcher.start()

        if hasattr(enable_dsa_cp, "cache_clear"):
            enable_dsa_cp.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        assert builder.device == device
        assert builder.vllm_config == vllm_config

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100),
                                                 torch.randn(100))

        metadata = builder.build(
            common_prefix_len=10,
            common_attn_metadata=common_attn_metadata,
        )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_for_graph_capture(
            self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100),
                                                 torch.randn(100))

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly
