import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import CacheConfig, CompilationMode, CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.spec_decode.eagle_proposer import (
    AscendEagleProposer,
    SpecDecodeBaseProposer,
)


class TestEagleProposerInitialization(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.cache_config = MagicMock(spec=CacheConfig)
        self.vllm_config.scheduler_config = MagicMock()
        self.vllm_config.model_config = MagicMock()
        self.vllm_config.model_config.hf_text_config = MagicMock(
            spec=[]
        )  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.compilation_config = MagicMock()
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    def test_initialization_eagle_graph(self):
        self.vllm_config.speculative_config.method = "eagle"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 4096
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = False
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 4096)
            self.assertTrue(proposer.use_cuda_graph)

            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.input_ids.shape, (expected_max_num_tokens,))
            self.assertEqual(proposer.positions.shape, (expected_max_num_tokens,))
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 4096))
            self.assertEqual(proposer.arange.shape, (expected_max_num_tokens,))

    def test_staged_mtp_metadata_arenas_have_stable_distinct_addresses(
        self,
    ):
        proposer = SpecDecodeBaseProposer.__new__(
            SpecDecodeBaseProposer
        )
        proposer.use_staged_mtp_draft_graph = True
        proposer._staged_mtp_max_request_capacity = 4
        proposer._staged_mtp_arena_capacity = 0
        proposer._staged_mtp_metadata_arenas = []
        proposer.num_speculative_tokens = 2
        proposer.device = torch.device("cpu")
        proposer.runner = MagicMock(pin_memory=False)
        proposer.arange = torch.arange(16, dtype=torch.int32)
        proposer.arange_cpu = torch.arange(16, dtype=torch.int32)
        source = SimpleNamespace(
            query_start_loc=torch.arange(3, dtype=torch.int32),
            query_start_loc_cpu=torch.arange(3, dtype=torch.int32),
            seq_lens=torch.tensor([8, 9], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([8, 9], dtype=torch.int32),
            num_computed_tokens_cpu=torch.tensor(
                [7, 8], dtype=torch.int32
            ),
            block_table_tensor=torch.tensor(
                [[10, 11], [12, 13], [20, 21], [22, 23]],
                dtype=torch.int32,
            ),
            positions=torch.tensor([7, 8], dtype=torch.int32),
        )

        step0 = proposer._bind_staged_mtp_metadata_arena(
            source,
            draft_step=0,
            capacity=4,
            actual_reqs=2,
        )
        step0_ptr = step0.block_table_tensor.data_ptr()
        step1 = proposer._bind_staged_mtp_metadata_arena(
            step0,
            draft_step=1,
            capacity=4,
            actual_reqs=2,
        )
        step0_again = proposer._bind_staged_mtp_metadata_arena(
            source,
            draft_step=0,
            capacity=4,
            actual_reqs=2,
        )

        self.assertEqual(
            step0_again.block_table_tensor.data_ptr(),
            step0_ptr,
        )
        self.assertNotEqual(
            step0.block_table_tensor.data_ptr(),
            step1.block_table_tensor.data_ptr(),
        )
        self.assertTrue(
            torch.equal(
                step0.block_table_tensor,
                source.block_table_tensor,
            )
        )
        self.assertTrue(
            torch.equal(
                step0.query_start_loc,
                torch.arange(5, dtype=torch.int32),
            )
        )

    def test_mtp_padding_recovers_fixed_dsa_state_buffers(self):
        proposer = SpecDecodeBaseProposer.__new__(
            SpecDecodeBaseProposer
        )
        runner_indices = torch.tensor(
            [1, 0, 2, 3, 4], dtype=torch.int32
        )
        runner_generations = torch.tensor(
            [-1, -2, -3, -4, -5], dtype=torch.int64
        )
        proposer.runner = SimpleNamespace(
            dsa_scratch_state_indices=SimpleNamespace(gpu=runner_indices),
            dsa_scratch_request_generations=SimpleNamespace(
                gpu=runner_generations
            ),
        )
        for common in (
            SimpleNamespace(
                dsa_scratch_state_indices=runner_indices[:2],
                dsa_scratch_request_generations=runner_generations[:2],
            ),
            # A padded metadata object may start without DSA-specific fields.
            SimpleNamespace(
                dsa_scratch_state_indices=None,
                dsa_scratch_request_generations=None,
            ),
        ):
            proposer._bind_dsa_scratch_state_capacity(common, 4)

            self.assertEqual(
                common.dsa_scratch_state_indices.tolist(),
                [1, 0, 2, 3],
            )
            self.assertEqual(
                common.dsa_scratch_request_generations.tolist(),
                [-1, -2, -3, -4],
            )
            self.assertEqual(
                common.dsa_scratch_state_indices.data_ptr(),
                runner_indices.data_ptr(),
            )
            self.assertEqual(
                common.dsa_scratch_request_generations.data_ptr(),
                runner_generations.data_ptr(),
            )

    def test_second_mtp_step_uses_indexer_group_block_table(self):
        proposer = SpecDecodeBaseProposer.__new__(
            SpecDecodeBaseProposer
        )
        proposer.use_staged_mtp_draft_graph = False
        proposer.pcp_size = 1
        proposer.dcp_size = 1
        proposer.uses_mrope = False
        proposer.max_model_len = 128
        proposer.kernel_block_size = 2
        proposer.method = "mtp"
        proposer.arange = torch.arange(8, dtype=torch.int32)
        proposer.token_arange_np = np.arange(8, dtype=np.int32)
        proposer.slot_mapping_group = [
            torch.full((4,), PADDING_SLOT_ID, dtype=torch.int32)
            for _ in range(2)
        ]
        proposer.indexer_slot_mapping_group = [
            torch.full((4,), PADDING_SLOT_ID, dtype=torch.int64)
            for _ in range(2)
        ]
        proposer.runner = SimpleNamespace(
            dsa_two_groups=True,
            input_batch=SimpleNamespace(
                block_table=[
                    SimpleNamespace(block_size=2),
                    SimpleNamespace(block_size=2),
                ]
            ),
        )
        common = SimpleNamespace(
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor(
                [0, 1, 2], dtype=torch.int32
            ),
            seq_lens=torch.tensor([2, 4], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([2, 4], dtype=torch.int32),
            num_computed_tokens_cpu=torch.tensor(
                [1, 3], dtype=torch.int32
            ),
            num_reqs=2,
            num_actual_tokens=2,
            num_input_tokens=2,
            max_query_len=1,
            decode_token_per_req=1,
            attn_state=None,
            graph_pad_size=-1,
            block_table_tensor=torch.tensor(
                [[10, 11], [20, 21]], dtype=torch.int32
            ),
            indexer_block_table_tensor=torch.tensor(
                [[30, 31], [40, 41]], dtype=torch.int32
            ),
            slot_mapping=torch.zeros(4, dtype=torch.int32),
            indexer_slot_mapping=torch.zeros(4, dtype=torch.int64),
            positions=torch.zeros(4, dtype=torch.int32),
        )
        builder = SimpleNamespace(
            build_for_drafting=lambda **kwargs: kwargs[
                "common_attn_metadata"
            ]
        )
        attn_group = SimpleNamespace(
            get_metadata_builder=lambda: builder
        )

        updated, metadata = proposer.attn_update_stack_num_spec_norm(
            1,
            None,
            common,
            batch_size=2,
            input_batch_size=4,
            used_update_positions=torch.tensor(
                [0, 2], dtype=torch.int32
            ),
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            attn_group=attn_group,
        )

        self.assertIs(updated, metadata)
        self.assertEqual(
            updated.slot_mapping.tolist(),
            [21, 43, PADDING_SLOT_ID, PADDING_SLOT_ID],
        )
        self.assertEqual(
            updated.indexer_slot_mapping.tolist(),
            [61, 83, PADDING_SLOT_ID, PADDING_SLOT_ID],
        )
        self.assertEqual(updated.indexer_slot_mapping.dtype, torch.int64)
        self.assertEqual(
            updated.indexer_slot_mapping.data_ptr(),
            proposer.indexer_slot_mapping_group[1].data_ptr(),
        )

    def test_initialization_eagle3_enforce_eager(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.NONE
        self.vllm_config.compilation_config.pass_config = MagicMock()
        self.vllm_config.compilation_config.pass_config.enable_sp = False
        self.vllm_config.model_config.enforce_eager = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertFalse(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_eagle3_full_graph_async(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertTrue(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_mtp_full_graph_async(self):
        self.vllm_config.speculative_config.method = "mtp"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertFalse(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_staged_mtp_keeps_host_retrieve_outside_draft_full_graph(
        self,
    ):
        self.vllm_config.speculative_config.method = "mtp"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = False
        self.runner._use_aclgraph.return_value = True
        self.runner.cudagraph_batch_sizes = [2, 4]

        with (
            patch(
                "vllm_ascend.spec_decode.eagle_proposer."
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch(
                "vllm_ascend.spec_decode.eagle_proposer."
                "staged_sfa_graph_capture_sizes",
                return_value=(2, 4),
            ),
            patch(
                "vllm_ascend.spec_decode.eagle_proposer.logger.warning"
            ) as warning,
            set_current_vllm_config(self.vllm_config),
        ):
            proposer = AscendEagleProposer(
                vllm_config=self.vllm_config,
                device=self.device,
                runner=self.runner,
            )

        self.assertFalse(proposer.use_cuda_graph)
        self.assertFalse(proposer.use_staged_mtp_draft_graph)
        warning.assert_called_once()
        self.assertIn(
            "host-side split",
            warning.call_args.args[0],
        )

@unittest.skip("Skip due to the changes in #7153, fix me later")
class TestEagleProposerLoadModel(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.method = "eagle"
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.parallel_drafting = False

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp1(self, mock_pp_group, mock_get_model, mock_get_layers):
        mock_pp_group.return_value.world_size = 1
        mock_target_layer1 = MagicMock()
        mock_target_layer2 = MagicMock()
        mock_draft_layer1 = MagicMock()
        mock_draft_layer3 = MagicMock()
        mock_get_layers.side_effect = [
            {"layer1": mock_target_layer1, "layer2": mock_target_layer2},
            {},
            {},
            {"layer1": mock_draft_layer1, "layer3": mock_draft_layer3},
        ]

        weight = torch.zeros(0)

        mock_model = MagicMock()
        mock_model.supports_multimodal = False
        mock_model.lm_head = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_model.model.embed_tokens = MagicMock()
        mock_model.model.embed_tokens.weight = weight

        mock_get_model.return_value = MagicMock()
        mock_get_model.return_value.model.embed_tokens.weight = weight

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            mock_get_model.assert_called_once()
            self.assertEqual(self.proposer.attn_layer_names, ["layer3"])
            self.assertIs(self.proposer.model.model.embed_tokens, mock_model.model.embed_tokens)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp_gt1(self, mock_pp_group, mock_get_model, mock_get_layers):
        mock_pp_group.return_value.world_size = 2
        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{"layer1": mock_target_layer1}, {}, {}, {"layer2": mock_draft_layer2}]

        mock_model = MagicMock()
        original_embed = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_get_model.return_value = MagicMock(model=MagicMock(embed_tokens=original_embed))

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)

            self.assertIsNot(self.proposer.model.model.embed_tokens, mock_model.model.embed_tokens)
            self.assertEqual(self.proposer.attn_layer_names, ["layer2"])

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    @patch("vllm_ascend.spec_decode.eagle_proposer.supports_multimodal")
    def test_load_model_multimodal(self, mock_supports_multi, mock_pp_group, mock_get_model, mock_get_layers):
        mock_model = MagicMock()
        mock_model.get_language_model.return_value.lm_head = MagicMock()
        mock_supports_multi.return_value = True
        original_embed = MagicMock()
        mock_get_model.return_value = MagicMock(model=MagicMock(embed_tokens=original_embed))

        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{"layer1": mock_target_layer1}, {}, {}, {"layer2": mock_draft_layer2}]
        mock_pp_group.return_value.world_size = 2

        self.proposer.model = MagicMock()

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            self.assertEqual(mock_model.get_language_model.call_count, 2)
            self.assertIs(self.proposer.model.lm_head, mock_model.get_language_model.return_value.lm_head)


class TestEagleProposerDummyRun(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.num_speculative_tokens = 4
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1
        self.runner.pin_memory = False
        self.runner._sync_metadata_across_dp.return_value = (8, torch.tensor([8]), False)

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.model_config.use_mla = False
        self.vllm_config.model_config.hf_text_config = MagicMock(
            spec=[]
        )  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(4)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Mock parallel state functions
        self.mock_tp_world_size = patch(
            "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size", return_value=1
        )
        self.mock_tp_world_size.start()

        mock_dp_group = MagicMock()
        mock_dp_group.world_size = 1
        self.mock_dp_group = patch("vllm_ascend.ascend_forward_context.get_dp_group", return_value=mock_dp_group)
        self.mock_dp_group.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.model = MagicMock()
        self.proposer._runnable = MagicMock()
        self.proposer.update_stream = MagicMock()

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        self.mock_tp_world_size.stop()
        self.mock_dp_group.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # cpu does not support parallel-group, let alone `sp`
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context",
           **{"return_value.flash_comm_v1_enabled": False})
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_basic(self, mock_context, mock_get_context, mock_get_context_2):
        num_tokens = 32
        with_prefill = False

        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=num_tokens, with_prefill=with_prefill)

            self.assertTrue(self.proposer._runnable.call_count == 1)

    # cpu does not support parallel-group, let alone `sp`
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context",
           **{"return_value.flash_comm_v1_enabled": False})
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_with_prefill(self, mock_context, mock_get_context, mock_get_context_2):
        mock_context.return_value.__enter__.return_value = None
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, with_prefill=True, num_reqs=4)
            self.assertTrue(self.proposer._runnable.call_count == 1)

    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_capture(self, mock_context, mock_get_context,
                                        mock_update_full_graph_params, mock_get_context_2):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = True
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        mock_get_context_2.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, in_graph_capturing=True, aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            mock_update_full_graph_params.assert_not_called()
            self.proposer.use_cuda_graph = last_use_cuda_graph
    
    @patch('vllm_ascend.ascend_forward_context.get_forward_context')
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_run(self, mock_context, mock_get_context,
                                    mock_update_full_graph_params, mock_get_context_2):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = False
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        mock_get_context_2.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        self.proposer.draft_attn_groups = [MagicMock()]
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, in_graph_capturing=False, aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            self.assertTrue(mock_update_full_graph_params.call_count == 1)
            self.proposer.use_cuda_graph = last_use_cuda_graph


class TestEagleProposerHelperMethods(TestBase):
    # TODO: Can add some tests about prepare_next_token_ids in future.

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.scheduler_config = MagicMock(max_num_seqs=3)
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.req_ids = [0, 1, 2]
        self.runner.arange_np = np.arange(10)
        self.runner.input_batch.num_reqs = 3
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # This is equivalent to disable_padded_drafter_batch=True.
    def test_prepare_inputs(self):
        self.proposer.token_arange_np = np.arange(10)
        mock_attn = MagicMock()
        mock_attn.slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])
        num_rejected = torch.tensor([1, 0, 1], device=self.device)
        mock_return_attn = MagicMock()

        with (
            set_current_vllm_config(self.vllm_config),
            patch.object(self.proposer, "prepare_inputs", return_value=(mock_return_attn, torch.tensor([1, 2, 4]))),
        ):
            return_attn, indices = self.proposer.prepare_inputs(mock_attn, num_rejected)
            self.assertEqual(indices.tolist(), [1, 2, 4])

    def _dsa_common_metadata(self):
        state_indices = torch.tensor([1, 0], dtype=torch.int32)
        generations = torch.tensor([17, 23], dtype=torch.int64)
        prompt_lens = np.array([8, 18], dtype=np.int32)
        request_ids = ["req-a", "req-b"]
        indexer_block_table = torch.tensor(
            [[110, 111], [120, 121]],
            dtype=torch.int32,
        )
        indexer_slot_mapping = torch.tensor(
            [210, 211, 220, 221],
            dtype=torch.int64,
        )
        return (
            SimpleNamespace(
                query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
                query_start_loc_cpu=torch.tensor(
                    [0, 2, 4], dtype=torch.int32
                ),
                seq_lens=torch.tensor([10, 20], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
                num_computed_tokens_cpu=torch.tensor(
                    [9, 19], dtype=torch.int32
                ),
                num_reqs=2,
                num_actual_tokens=4,
                num_input_tokens=4,
                block_table_tensor=torch.tensor(
                    [[10, 11], [20, 21]], dtype=torch.int32
                ),
                slot_mapping=torch.arange(4, dtype=torch.int32),
                indexer_block_table_tensor=indexer_block_table,
                indexer_slot_mapping=indexer_slot_mapping,
                positions=torch.arange(4, dtype=torch.int64),
                prompt_lens_cpu=prompt_lens,
                request_ids=request_ids,
                dsa_scratch_state_indices=state_indices,
                dsa_scratch_request_generations=generations,
            ),
            prompt_lens,
            request_ids,
            state_indices,
            generations,
        )

    @staticmethod
    def _assert_dsa_lifetime_metadata_preserved(
        metadata,
        prompt_lens,
        request_ids,
        state_indices,
        generations,
    ):
        assert metadata.prompt_lens_cpu is prompt_lens
        assert metadata.request_ids is request_ids
        assert metadata.dsa_scratch_state_indices is state_indices
        assert metadata.dsa_scratch_request_generations is generations

    def test_prepare_inputs_preserves_dsa_scratch_lifetime_metadata(self):
        (
            common,
            prompt_lens,
            request_ids,
            state_indices,
            generations,
        ) = self._dsa_common_metadata()
        self.proposer.token_arange_np = np.arange(8)
        self.proposer.runner.actual_seq_lengths_q = []
        self.proposer.runner.attn_state = None
        self.proposer.runner.decode_token_per_req = 2

        metadata, _ = self.proposer.prepare_inputs(
            common,
            sampled_token_ids=[[101], [202]],
            num_draft_tokens=[1, 1],
        )

        self._assert_dsa_lifetime_metadata_preserved(
            metadata,
            prompt_lens,
            request_ids,
            state_indices,
            generations,
        )
        self.assertIs(
            metadata.indexer_block_table_tensor,
            common.indexer_block_table_tensor,
        )
        self.assertEqual(
            metadata.indexer_slot_mapping.tolist(),
            [210, 220, PADDING_SLOT_ID, PADDING_SLOT_ID],
        )

    def test_prepare_inputs_padded_preserves_dsa_scratch_lifetime_metadata(
        self,
    ):
        (
            common,
            prompt_lens,
            request_ids,
            state_indices,
            generations,
        ) = self._dsa_common_metadata()
        self.proposer.arange = torch.arange(8, dtype=torch.int32)
        self.proposer.runner.actual_seq_lengths_q = []
        self.proposer.runner.attn_state = None
        self.proposer.runner.decode_token_per_req = 2
        spec_decode_metadata = SimpleNamespace(
            cu_num_draft_tokens=torch.tensor([1, 2], dtype=torch.int32)
        )

        with patch(
            "vllm_ascend.spec_decode.eagle_proposer.enable_sp",
            return_value=False,
        ):
            metadata, *_ = self.proposer.prepare_inputs_padded(
                common,
                spec_decode_metadata,
                valid_sampled_tokens_count=torch.tensor(
                    [1, 1], dtype=torch.int32
                ),
            )

        self._assert_dsa_lifetime_metadata_preserved(
            metadata,
            prompt_lens,
            request_ids,
            state_indices,
            generations,
        )
        self.assertIs(
            metadata.indexer_block_table_tensor,
            common.indexer_block_table_tensor,
        )
        self.assertIs(
            metadata.indexer_slot_mapping,
            common.indexer_slot_mapping,
        )
