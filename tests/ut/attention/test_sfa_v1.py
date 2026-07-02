import os
import sys
from unittest.mock import MagicMock, patch

import torch

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm.distributed.parallel_state import GroupCoordinator

if 'torch_npu._inductor' not in sys.modules:
    sys.modules['torch_npu._inductor'] = MagicMock()

from vllm_ascend.attention.sfa_v1 import (AscendSFABackend, AscendSFAImpl,
                                           AscendSFAMetadata,
                                           AscendSFAMetadataBuilder,
                                           _dense_prefix_compare_enabled,
                                           _dense_prefix_compare_build_sample,
                                           _dense_prefix_compare_diff,
                                           _dense_prefix_compare_direct_call,
                                           _dsa_env_flag,
                                           _sfa_path_trace_should_wrap,
                                           _sfa_trace_lmcache_call)
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

    def test_dense_prefix_compare_build_sample_uses_logical_head_tail_slots(self):
        cache = torch.arange(3 * 4 * 1 * 2, dtype=torch.float32).reshape(3, 4, 1, 2)
        block_table = torch.tensor([[0, 1, 2]], dtype=torch.long)

        with patch.dict(os.environ, {"VLLM_ASCEND_DENSE_PREFIX_COMPARE_SAMPLES": "8"}):
            sample, error = _dense_prefix_compare_build_sample(cache, block_table, 10)

        self.assertIsNone(error)
        assert sample is not None
        self.assertEqual(sample["positions"], [0, 1, 2, 3, 6, 7, 8, 9])
        self.assertEqual(sample["slots"], [0, 1, 2, 3, 6, 7, 8, 9])
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

        mock_logger.warning.assert_called_once()
        self.assertIn(
            "[DENSE_PREFIX_COMPARE_PATH] direct_call",
            mock_logger.warning.call_args.args[0],
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
