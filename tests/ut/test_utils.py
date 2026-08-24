#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import math
import os
from unittest import mock

import pytest
import torch
from vllm.config import CompilationConfig, CUDAGraphMode, ModelConfig, ParallelConfig, VllmConfig

from tests.ut.base import TestBase
from vllm_ascend import utils
from vllm_ascend.utils import REGISTERED_ASCEND_OPS


class TestUtils(TestBase):
    def setUp(self):
        import importlib

        from vllm_ascend import platform

        importlib.reload(platform)

    def test_nd_to_nz_2d(self):
        # can be divided by 16
        input_tensor = torch.randn(32, 64)
        output = utils.nd_to_nz_2d(input_tensor)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 64 // 16)
        self.assertEqual(output.shape[2], 32)
        self.assertEqual(output.shape[3], 16)

        # cannot be divided by 16
        input_tensor = torch.randn(30, 62)
        output = utils.nd_to_nz_2d(input_tensor)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], math.ceil(62 / 16))
        self.assertEqual(output.shape[2], 32)
        self.assertEqual(output.shape[3], 16)

        # pad to 16
        input_tensor = torch.randn(8, 12)
        output = utils.nd_to_nz_2d(input_tensor)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 1)  # 12->16, 16//16=1
        self.assertEqual(output.shape[2], 16)  # 8->16
        self.assertEqual(output.shape[3], 16)

        # check if the output is contiguous
        input_tensor = torch.randn(32, 64)
        output = utils.nd_to_nz_2d(input_tensor)
        self.assertTrue(output.is_contiguous())

        # check if the output values are preserved
        input_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        output = utils.nd_to_nz_2d(input_tensor)
        expected = torch.tensor(
            [
                [
                    [
                        [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ]
            ]
        )
        self.assertTrue(torch.allclose(output, expected))

    def test_aligned_16(self):
        # align to 16
        input_tensor = torch.randn(15, 64)
        output_tensor = utils.aligned_16(input_tensor)
        self.assertEqual(output_tensor.shape[0], 16)

        # align to 16
        input_tensor = torch.randn(16, 64)
        output_tensor = utils.aligned_16(input_tensor)
        self.assertEqual(output_tensor.shape[0], 16)
        self.assertTrue(torch.equal(input_tensor, output_tensor))

        # align to 32
        input_tensor = torch.randn(17, 64)
        output_tensor = utils.aligned_16(input_tensor)
        self.assertEqual(output_tensor.shape[0], 32)

    @pytest.mark.skip("Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_enable_custom_op(self):
        result = utils.enable_custom_op()
        self.assertTrue(result)

        utils._CUSTOM_OP_ENABLED = None

        with mock.patch("builtins.__import__") as mock_import_module:
            mock_import_module.side_effect = ImportError("import error")
            self.assertFalse(utils.enable_custom_op())

    def test_find_hccl_library(self):
        with mock.patch.dict(os.environ, {"HCCL_SO_PATH": "/path/to/hccl/libhccl.so"}):
            self.assertEqual(utils.find_hccl_library(), "/path/to/hccl/libhccl.so")
        with mock.patch("torch.version.cann", None):
            self.assertRaises(ValueError, utils.find_hccl_library)
        with mock.patch("torch.version.cann", "Ascend910"):
            self.assertEqual(utils.find_hccl_library(), "libhccl.so")

    def test_current_stream(self):
        with mock.patch("torch.npu.current_stream") as mock_current_stream:
            self.assertEqual(utils.current_stream(), mock_current_stream())

    def test_vllm_version_is(self):
        with mock.patch.dict(os.environ, {"VLLM_VERSION": "1.0.0"}):
            with mock.patch("vllm.__version__", "1.0.0"):
                self.assertTrue(utils.vllm_version_is.__wrapped__("1.0.0"))
                self.assertFalse(utils.vllm_version_is.__wrapped__("2.0.0"))
            with mock.patch("vllm.__version__", "2.0.0"):
                self.assertTrue(utils.vllm_version_is.__wrapped__("1.0.0"))
                self.assertFalse(utils.vllm_version_is.__wrapped__("2.0.0"))
        with mock.patch("vllm.__version__", "1.0.0"):
            self.assertTrue(utils.vllm_version_is.__wrapped__("1.0.0"))
            self.assertFalse(utils.vllm_version_is.__wrapped__("2.0.0"))
        with mock.patch("vllm.__version__", "2.0.0"):
            self.assertTrue(utils.vllm_version_is.__wrapped__("2.0.0"))
            self.assertFalse(utils.vllm_version_is.__wrapped__("1.0.0"))
        # Test caching takes effect
        utils.vllm_version_is.cache_clear()
        utils.vllm_version_is("1.0.0")
        misses = utils.vllm_version_is.cache_info().misses
        hits = utils.vllm_version_is.cache_info().hits
        self.assertEqual(misses, 1)
        self.assertEqual(hits, 0)
        utils.vllm_version_is("1.0.0")
        hits = utils.vllm_version_is.cache_info().hits
        self.assertEqual(hits, 1)

    def test_get_max_hidden_layers(self):
        from transformers import PretrainedConfig

        class SimpleConfig(PretrainedConfig):
            def __init__(self, num_hidden_layers=12):
                self.num_hidden_layers = num_hidden_layers

            def to_dict(self):
                return {"num_hidden_layers": self.num_hidden_layers}

        self.assertEqual(utils.get_max_hidden_layers(SimpleConfig()), 12)
        self.assertEqual(utils.get_max_hidden_layers(SimpleConfig(24)), 24)

        class NestedConfig(PretrainedConfig):
            def to_dict(self):
                return {
                    "model": {"encoder": {"num_hidden_layers": 8}, "decoder": {"num_hidden_layers": 12}},
                    "other_setting": True,
                }

        self.assertEqual(utils.get_max_hidden_layers(NestedConfig()), 12)

        class MultiValueConfig(PretrainedConfig):
            def to_dict(self):
                return {
                    "num_hidden_layers": 6,
                    "submodule": {"num_hidden_layers": 18, "subsub": {"num_hidden_layers": 9}},
                }

        self.assertEqual(utils.get_max_hidden_layers(MultiValueConfig()), 18)

        class NoLayerConfig(PretrainedConfig):
            def to_dict(self):
                return {"attention_heads": 8}

        with self.assertRaises(ValueError) as context:
            utils.get_max_hidden_layers(NoLayerConfig())
        self.assertIn("num_hidden_layers", str(context.exception))

    def test_update_aclgraph_sizes(self):
        test_compilation_config = CompilationConfig(cudagraph_capture_sizes=[i for i in range(150)])
        model_path = os.path.join(os.path.dirname(__file__), "fake_weight")
        test_model_config = ModelConfig(model=model_path, enforce_eager=True)
        test_parallel_config = ParallelConfig()
        test_vllm_config = VllmConfig(
            model_config=test_model_config,
            compilation_config=test_compilation_config,
            parallel_config=test_parallel_config,
        )
        utils.update_aclgraph_sizes(test_vllm_config)
        os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
        utils.update_aclgraph_sizes(test_vllm_config)
        del os.environ["HCCL_OP_EXPANSION_MODE"]

        self.assertEqual(0, len(test_vllm_config.compilation_config.cudagraph_capture_sizes))

    def test_staged_sfa_graph_capture_sizes_accepts_exact_q1_batches(self):
        vllm_config = mock.MagicMock()
        vllm_config.scheduler_config.max_num_seqs = 64
        vllm_config.scheduler_config.max_num_batched_tokens = 128

        with (
            mock.patch.object(
                utils,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            mock.patch.dict(
                os.environ,
                {"VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": " 4,1,4,2 "},
            ),
        ):
            capture_sizes = utils.staged_sfa_graph_capture_sizes(vllm_config)

        self.assertEqual(capture_sizes, (1, 2, 4))

    def test_staged_sfa_graph_capture_sizes_scale_request_buckets_for_mtp(self):
        vllm_config = mock.MagicMock()
        vllm_config.speculative_config.num_speculative_tokens = 2
        vllm_config.scheduler_config.max_num_seqs = 64
        vllm_config.scheduler_config.max_num_batched_tokens = 128
        with (
            mock.patch.object(
                utils,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            mock.patch.dict(
                os.environ,
                {"VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": "1,2,4"},
            ),
        ):
            self.assertEqual(
                utils.staged_sfa_graph_capture_sizes(vllm_config),
                (3, 6, 12),
            )

    def test_staged_sfa_dsa_cp_capture_sizes_allow_fragmented_requests(self):
        vllm_config = mock.MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 8
        vllm_config.model_config.hf_text_config.index_topk = 2048
        vllm_config.speculative_config.num_speculative_tokens = 1
        vllm_config.scheduler_config.max_num_seqs = 64
        vllm_config.scheduler_config.max_num_batched_tokens = 128
        with (
            mock.patch.object(
                utils,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            mock.patch.object(
                utils.envs_ascend,
                "VLLM_ASCEND_ENABLE_FLASHCOMM1",
                True,
            ),
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": "10,20,25,32"
                },
            ),
        ):
            self.assertEqual(
                utils.staged_sfa_graph_capture_sizes(vllm_config),
                (20, 40, 50, 64),
            )

    def test_staged_sfa_configuration_accepts_model_specific_mtp_method(self):
        vllm_config = mock.MagicMock()
        vllm_config.speculative_config.method = "glm4_moe_mtp"
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        vllm_config.model_config.use_mla = True
        vllm_config.model_config.hf_text_config.index_topk = 2048
        vllm_config.kv_transfer_config = mock.MagicMock()
        vllm_config.lora_config = None
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1
        with (
            mock.patch.multiple(
                utils.envs_ascend,
                VLLM_ASCEND_DSA_UNBUNDLE=True,
                VLLM_ASCEND_DSA_TWO_GROUPS=True,
                VLLM_ASCEND_DSA_SHRINK_LATENT=2,
                VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=False,
                VLLM_ASCEND_DSA_USE_ADAPTER_CACHE=False,
            ),
            mock.patch.dict(os.environ, {"HCCL_OP_EXPANSION_MODE": ""}),
        ):
            reasons = utils.staged_sfa_graph_configuration_reasons(
                vllm_config
            )
        self.assertNotIn(
            utils.StagedSFAConfigReason.SPECULATIVE_DECODE,
            reasons,
        )

    def test_staged_sfa_graph_capture_sizes_ignored_when_disabled(self):
        with (
            mock.patch.object(
                utils,
                "staged_sfa_graph_configured",
                return_value=False,
            ),
            mock.patch.dict(
                os.environ,
                {"VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": "not-an-integer"},
            ),
        ):
            capture_sizes = utils.staged_sfa_graph_capture_sizes(mock.MagicMock())

        self.assertEqual(capture_sizes, ())

    def test_staged_sfa_graph_capture_sizes_rejects_invalid_values(self):
        vllm_config = mock.MagicMock()
        vllm_config.scheduler_config.max_num_seqs = 64
        vllm_config.scheduler_config.max_num_batched_tokens = 128
        invalid_values = (
            ("", "must contain positive integers"),
            ("1,not-an-integer", "comma-separated list"),
            ("0,1", "must contain positive integers"),
        )

        with mock.patch.object(
            utils,
            "staged_sfa_graph_configured",
            return_value=True,
        ):
            for raw_sizes, expected_message in invalid_values:
                with (
                    self.subTest(raw_sizes=raw_sizes),
                    mock.patch.dict(
                        os.environ,
                        {"VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": raw_sizes},
                    ),
                    self.assertRaisesRegex(ValueError, expected_message),
                ):
                    utils.staged_sfa_graph_capture_sizes(vllm_config)

    def test_staged_sfa_graph_capture_sizes_respects_q1_capacity(self):
        for max_num_seqs, max_num_batched_tokens in ((16, 64), (64, 16)):
            vllm_config = mock.MagicMock()
            vllm_config.scheduler_config.max_num_seqs = max_num_seqs
            vllm_config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens

            with (
                self.subTest(
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                ),
                mock.patch.object(
                    utils,
                    "staged_sfa_graph_configured",
                    return_value=True,
                ),
                mock.patch.dict(
                    os.environ,
                    {"VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": "1,16,32"},
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "scheduler capacity.*\\[32\\]",
                ),
            ):
                utils.staged_sfa_graph_capture_sizes(vllm_config)

    def test_update_aclgraph_sizes_restricts_cross_layer_capture(self):
        compilation_config = mock.MagicMock()
        compilation_config.cudagraph_capture_sizes = list(range(1, 101))
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        model_config = mock.MagicMock()
        model_config.hf_text_config.num_hidden_layers = 80
        model_config.hf_text_config.index_topk = 2048
        model_config.use_mla = True
        model_config.architecture = "StagedSFAResourceTest"
        parallel_config = mock.MagicMock()
        parallel_config.data_parallel_size = 1
        parallel_config.tensor_parallel_size = 8
        parallel_config.prefill_context_parallel_size = 1
        parallel_config.decode_context_parallel_size = 1
        vllm_config = mock.MagicMock()
        vllm_config.compilation_config = compilation_config
        vllm_config.model_config = model_config
        vllm_config.parallel_config = parallel_config
        vllm_config.speculative_config = None

        vllm_config.kv_transfer_config = mock.MagicMock()
        vllm_config.lora_config = None

        def apply_capture_sizes(config, capture_sizes):
            config.compilation_config.cudagraph_capture_sizes = capture_sizes

        with (
            mock.patch.object(
                utils,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            mock.patch.dict(
                os.environ,
                {"HCCL_OP_EXPANSION_MODE": ""},
            ),
            mock.patch.object(
                utils,
                "update_cudagraph_capture_sizes",
                side_effect=apply_capture_sizes,
            ) as update_sizes,
        ):
            utils.update_aclgraph_sizes(vllm_config)

        sampled_sizes = update_sizes.call_args.args[1]
        self.assertEqual(sampled_sizes, [1])

    def test_staged_sfa_resource_budget_keeps_size_one(self):
        compilation_config = mock.MagicMock()
        compilation_config.cudagraph_capture_sizes = list(range(1, 101))
        model_config = mock.MagicMock()
        model_config.hf_text_config.num_hidden_layers = 170
        model_config.hf_text_config.index_topk = 2048
        model_config.use_mla = True
        model_config.architecture = "DeepStagedSFAResourceTest"
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        parallel_config = mock.MagicMock()
        parallel_config.data_parallel_size = 1
        parallel_config.tensor_parallel_size = 8
        parallel_config.prefill_context_parallel_size = 1
        parallel_config.decode_context_parallel_size = 1
        vllm_config = mock.MagicMock()
        vllm_config.compilation_config = compilation_config
        vllm_config.model_config = model_config
        vllm_config.parallel_config = parallel_config
        vllm_config.speculative_config = None

        def apply_capture_sizes(config, capture_sizes):
            config.compilation_config.cudagraph_capture_sizes = capture_sizes

        vllm_config.kv_transfer_config = mock.MagicMock()
        vllm_config.lora_config = None
        with (
            mock.patch.object(
                utils,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            mock.patch.dict(
                os.environ,
                {"HCCL_OP_EXPANSION_MODE": ""},
            ),
            mock.patch.object(
                utils,
                "update_cudagraph_capture_sizes",
                side_effect=apply_capture_sizes,
            ) as update_sizes,
        ):
            utils.update_aclgraph_sizes(vllm_config)

        update_sizes.assert_called_once()
        self.assertEqual(update_sizes.call_args.args[1], [1])

    def test_staged_sfa_cross_layer_capture_uses_requested_key(self):
        compilation_config = mock.MagicMock()
        compilation_config.cudagraph_capture_sizes = list(range(1, 101))
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        model_config = mock.MagicMock()
        model_config.hf_text_config.num_hidden_layers = 90
        model_config.hf_text_config.index_topk = 2048
        model_config.use_mla = True
        model_config.architecture = "TwoKeyStagedSFAResourceTest"
        parallel_config = mock.MagicMock()
        parallel_config.data_parallel_size = 1
        parallel_config.tensor_parallel_size = 8
        parallel_config.prefill_context_parallel_size = 1
        parallel_config.decode_context_parallel_size = 1
        vllm_config = mock.MagicMock()
        vllm_config.compilation_config = compilation_config
        vllm_config.model_config = model_config
        vllm_config.parallel_config = parallel_config
        vllm_config.speculative_config = None
        vllm_config.kv_transfer_config = mock.MagicMock()
        vllm_config.lora_config = None

        def apply_capture_sizes(config, capture_sizes):
            config.compilation_config.cudagraph_capture_sizes = capture_sizes

        with (
            mock.patch.object(
                utils,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            mock.patch.object(
                utils,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 32),
            ),
            mock.patch.dict(
                os.environ,
                {"HCCL_OP_EXPANSION_MODE": ""},
            ),
            mock.patch.object(
                utils,
                "update_cudagraph_capture_sizes",
                side_effect=apply_capture_sizes,
            ) as update_sizes,
        ):
            utils.update_aclgraph_sizes(vllm_config)

        update_sizes.assert_called_once()
        self.assertEqual(update_sizes.call_args.args[1], [1, 32])

    def test_staged_sfa_capture_rejects_graph_entry_overcommit(self):
        compilation_config = mock.MagicMock(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
        model_config = mock.MagicMock()
        model_config.hf_text_config.num_hidden_layers = 80
        parallel_config = mock.MagicMock()
        parallel_config.data_parallel_size = 1
        parallel_config.tensor_parallel_size = 8
        parallel_config.prefill_context_parallel_size = 1
        parallel_config.decode_context_parallel_size = 1
        vllm_config = mock.MagicMock(
            compilation_config=compilation_config,
            model_config=model_config,
            parallel_config=parallel_config,
            speculative_config=None,
        )

        with (
            mock.patch.object(
                utils,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            mock.patch.object(
                utils,
                "staged_sfa_graph_capture_sizes",
                return_value=tuple(range(1, 33)),
            ),
            mock.patch.dict(
                os.environ,
                {"HCCL_OP_EXPANSION_MODE": ""},
            ),
            self.assertRaisesRegex(
                ValueError,
                "exceeding the device quota",
            ),
        ):
            utils.update_aclgraph_sizes(vllm_config)

    def test_staged_sfa_capture_uses_aiv_resource_budget(self):
        compilation_config = mock.MagicMock(
            cudagraph_capture_sizes=[1],
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
        model_config = mock.MagicMock()
        model_config.hf_text_config.num_hidden_layers = 78
        model_config.architecture = "StagedSFAAIVResourceTest"
        parallel_config = mock.MagicMock(
            data_parallel_size=2,
            tensor_parallel_size=2,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
        )
        vllm_config = mock.MagicMock(
            compilation_config=compilation_config,
            model_config=model_config,
            parallel_config=parallel_config,
            speculative_config=None,
            additional_config={},
        )

        for key_count, should_fail in ((5, False), (6, True)):
            compilation_config.cudagraph_capture_sizes = [1]
            capture_sizes = tuple(range(1, key_count + 1))
            with (
                self.subTest(key_count=key_count),
                mock.patch.object(
                    utils,
                    "staged_sfa_graph_configured",
                    return_value=True,
                ),
                mock.patch.object(
                    utils,
                    "staged_sfa_graph_capture_sizes",
                    return_value=capture_sizes,
                ),
                mock.patch.object(
                    utils,
                    "is_moe_model",
                    return_value=True,
                ),
                mock.patch.dict(
                    os.environ,
                    {"HCCL_OP_EXPANSION_MODE": "AIV"},
                ),
                mock.patch.object(
                    utils,
                    "update_cudagraph_capture_sizes",
                ) as update_sizes,
            ):
                if should_fail:
                    with self.assertRaisesRegex(
                        ValueError,
                        "device quota of 5 keys",
                    ):
                        utils.update_aclgraph_sizes(vllm_config)
                    update_sizes.assert_not_called()
                else:
                    utils.update_aclgraph_sizes(vllm_config)
                    update_sizes.assert_called_once_with(
                        vllm_config,
                        list(capture_sizes),
                    )

    def test_staged_sfa_cross_layer_capture_drops_native_buckets(self):
        compilation_config = mock.MagicMock()
        compilation_config.cudagraph_capture_sizes = list(range(1, 101))
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        model_config = mock.MagicMock()
        model_config.hf_text_config.num_hidden_layers = 80
        model_config.hf_text_config.index_topk = 2048
        model_config.use_mla = True
        model_config.architecture = "StagedSFAMaximumBucketTest"
        parallel_config = mock.MagicMock()
        parallel_config.data_parallel_size = 1
        parallel_config.tensor_parallel_size = 8
        parallel_config.prefill_context_parallel_size = 1
        parallel_config.decode_context_parallel_size = 1
        vllm_config = mock.MagicMock()
        vllm_config.compilation_config = compilation_config
        vllm_config.model_config = model_config
        vllm_config.parallel_config = parallel_config
        vllm_config.speculative_config = None
        vllm_config.kv_transfer_config = mock.MagicMock()
        vllm_config.lora_config = None

        def apply_capture_sizes(config, capture_sizes):
            config.compilation_config.cudagraph_capture_sizes = capture_sizes

        with (
            mock.patch.object(
                utils,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            mock.patch.object(
                utils,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 32),
            ),
            mock.patch.dict(
                os.environ,
                {"HCCL_OP_EXPANSION_MODE": ""},
            ),
            mock.patch.object(
                utils,
                "update_cudagraph_capture_sizes",
                side_effect=apply_capture_sizes,
            ) as update_sizes,
        ):
            utils.update_aclgraph_sizes(vllm_config)

        update_sizes.assert_called_once()
        self.assertEqual(update_sizes.call_args.args[1], [1, 32])

    def test_staged_sfa_no_sampling_updates_derived_maximum(self):
        compilation_config = mock.MagicMock()
        compilation_config.cudagraph_capture_sizes = [1, 8]
        compilation_config.max_cudagraph_capture_size = 8
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        compilation_config.post_init_cudagraph_sizes = mock.Mock()
        model_config = mock.MagicMock()
        model_config.hf_text_config.num_hidden_layers = 1
        model_config.hf_text_config.index_topk = 2048
        model_config.use_mla = True
        model_config.architecture = "StagedSFADerivedMaximumTest"
        parallel_config = mock.MagicMock()
        parallel_config.data_parallel_size = 1
        parallel_config.tensor_parallel_size = 1
        parallel_config.prefill_context_parallel_size = 1
        parallel_config.decode_context_parallel_size = 1
        vllm_config = mock.MagicMock()
        vllm_config.compilation_config = compilation_config
        vllm_config.model_config = model_config
        vllm_config.parallel_config = parallel_config
        vllm_config.speculative_config = None
        vllm_config.kv_transfer_config = mock.MagicMock()
        vllm_config.lora_config = None

        with (
            mock.patch.object(
                utils,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            mock.patch.object(
                utils,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 32),
            ),
            mock.patch.dict(
                os.environ,
                {"HCCL_OP_EXPANSION_MODE": ""},
            ),
        ):
            utils.update_aclgraph_sizes(vllm_config)

        self.assertEqual(compilation_config.cudagraph_capture_sizes, [1, 32])
        self.assertEqual(
            compilation_config.max_cudagraph_capture_size,
            32,
        )
        compilation_config.post_init_cudagraph_sizes.assert_called_once_with()

    def test_staged_sfa_graph_configured_prerequisites(self):
        compilation_config = mock.MagicMock()
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        model_config = mock.MagicMock()
        model_config.use_mla = True
        model_config.hf_text_config.index_topk = 2048
        vllm_config = mock.MagicMock()
        vllm_config.compilation_config = compilation_config
        vllm_config.model_config = model_config
        vllm_config.kv_transfer_config = mock.MagicMock()
        vllm_config.speculative_config = None
        vllm_config.lora_config = None

        with (
            mock.patch.multiple(
                utils.envs_ascend,
                VLLM_ASCEND_SFA_STAGED_GRAPH=True,
                VLLM_ASCEND_DSA_UNBUNDLE=True,
                VLLM_ASCEND_DSA_TWO_GROUPS=True,
                VLLM_ASCEND_DSA_SHRINK_LATENT=2,
                VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=False,
                VLLM_ASCEND_DSA_USE_ADAPTER_CACHE=False,
            ),
            mock.patch.dict(
                os.environ,
                {"HCCL_OP_EXPANSION_MODE": "AIV"},
            ),
        ):
            self.assertTrue(utils.staged_sfa_graph_configured(vllm_config))

            with mock.patch.object(
                utils.envs_ascend,
                "VLLM_ASCEND_DSA_SHRINK_LATENT",
                0,
            ):
                self.assertFalse(utils.staged_sfa_graph_configured(vllm_config))

            compilation_config.cudagraph_mode = CUDAGraphMode.FULL
            self.assertFalse(utils.staged_sfa_graph_configured(vllm_config))

            compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
            vllm_config.kv_transfer_config = None
            self.assertFalse(utils.staged_sfa_graph_configured(vllm_config))

    def test_staged_sfa_configuration_errors_report_unsupported_modes(self):
        vllm_config = mock.MagicMock()
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        vllm_config.model_config.use_mla = True
        vllm_config.model_config.hf_text_config.index_topk = 2048
        vllm_config.kv_transfer_config = mock.MagicMock()
        vllm_config.speculative_config = mock.MagicMock()
        vllm_config.lora_config = mock.MagicMock()
        vllm_config.parallel_config.data_parallel_size = 2
        vllm_config.parallel_config.distributed_executor_backend = "external_launcher"
        vllm_config.parallel_config.pipeline_parallel_size = 2
        vllm_config.parallel_config.prefill_context_parallel_size = 2
        vllm_config.parallel_config.decode_context_parallel_size = 1

        with (
            mock.patch.multiple(
                utils.envs_ascend,
                VLLM_ASCEND_DSA_UNBUNDLE=True,
                VLLM_ASCEND_DSA_TWO_GROUPS=True,
                VLLM_ASCEND_DSA_SHRINK_LATENT=2,
                VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=False,
                VLLM_ASCEND_DSA_USE_ADAPTER_CACHE=False,
            ),
            mock.patch.dict(
                os.environ,
                {"HCCL_OP_EXPANSION_MODE": ""},
            ),
        ):
            reasons = utils.staged_sfa_graph_configuration_reasons(vllm_config)
            errors = utils.staged_sfa_graph_configuration_errors(vllm_config)

        self.assertIn(utils.StagedSFAConfigReason.SPECULATIVE_DECODE, reasons)
        self.assertIn(utils.StagedSFAConfigReason.DATA_PARALLEL, reasons)
        self.assertIn(
            "only fixed-width MTP speculative decoding is supported",
            errors,
        )
        self.assertIn("LoRA is not implemented", errors)
        self.assertIn(
            "external-launcher data parallel staged graphs are not implemented",
            errors,
        )
        self.assertIn("pipeline parallel staged graphs are not implemented", errors)
        self.assertIn("context parallel staged graphs are not implemented", errors)

        vllm_config.parallel_config.distributed_executor_backend = "mp"
        reasons = utils.staged_sfa_graph_configuration_reasons(vllm_config)
        self.assertNotIn(utils.StagedSFAConfigReason.DATA_PARALLEL, reasons)

    def test_staged_sfa_aiv_keeps_other_profiling_guards(self):
        vllm_config = mock.MagicMock()
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        vllm_config.model_config.use_mla = True
        vllm_config.model_config.hf_text_config.index_topk = 2048
        vllm_config.kv_transfer_config = mock.MagicMock()
        vllm_config.speculative_config = None
        vllm_config.lora_config = None
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1

        with (
            mock.patch.multiple(
                utils.envs_ascend,
                VLLM_ASCEND_DSA_UNBUNDLE=True,
                VLLM_ASCEND_DSA_TWO_GROUPS=True,
                VLLM_ASCEND_DSA_SHRINK_LATENT=2,
                VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=True,
                VLLM_ASCEND_DSA_USE_ADAPTER_CACHE=True,
            ),
            mock.patch.dict(
                os.environ,
                {"HCCL_OP_EXPANSION_MODE": "AIV"},
            ),
        ):
            reasons = utils.staged_sfa_graph_configuration_reasons(
                vllm_config
            )

        self.assertIn(
            utils.StagedSFAConfigReason.LEGACY_DSA_OFFLOAD,
            reasons,
        )
        self.assertIn(utils.StagedSFAConfigReason.ADAPTER_CACHE, reasons)

    @mock.patch("vllm.model_executor.custom_op.CustomOp")
    @mock.patch("vllm_ascend.ops.activation.AscendQuickGELU")
    @mock.patch("vllm_ascend.ops.activation.AscendSiluAndMul")
    @mock.patch("vllm_ascend.ops.layernorm.AscendRMSNorm")
    def test_register_ascend_customop(
        self, mock_ascend_rmsnorm, mock_ascend_silu_and_mul, mock_ascend_quick_gelu, mock_customop
    ):
        utils._ASCEND_CUSTOMOP_IS_REIGISTERED = False

        # ascend custom op is not registered
        utils.register_ascend_customop()
        self.assertEqual(mock_customop.register_oot.call_count, len(REGISTERED_ASCEND_OPS))
        self.assertTrue(utils._ASCEND_CUSTOMOP_IS_REIGISTERED)

        # ascend custom op is already registered
        utils.register_ascend_customop()
        self.assertEqual(mock_customop.register_oot.call_count, len(REGISTERED_ASCEND_OPS))

    @mock.patch("torch_npu.npu_format_cast")
    def test_maybe_trans_nz(self, mock_npu_format_cast):
        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

        mock_npu_format_cast.side_effect = lambda weight, fmt: weight

        def assert_nz_cast(weight):
            mock_npu_format_cast.assert_called_once()
            args, kwargs = mock_npu_format_cast.call_args
            self.assertIs(args[0], weight)
            self.assertEqual(args[1], ACL_FORMAT_FRACTAL_NZ)
            self.assertEqual(kwargs, {})

        # Test case 1: non-310P, NZ is disabled
        with (
            mock.patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"}),
            mock.patch("vllm_ascend.utils.is_310p", return_value=False),
        ):
            weight = torch.randn(32, 64, dtype=torch.float16)
            result = utils.maybe_trans_nz(weight)
            self.assertIs(result, weight)
            mock_npu_format_cast.assert_not_called()

        # Test case 2: 310P always converts non-fp32 weights, even when NZ=0
        mock_npu_format_cast.reset_mock()
        with (
            mock.patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"}),
            mock.patch("vllm_ascend.utils.is_310p", return_value=True),
        ):
            weight = torch.randn(32, 64, dtype=torch.float16)
            result = utils.maybe_trans_nz(weight)
            self.assertIs(result, weight)
            assert_nz_cast(weight)

        # Test case 3: fp32 never converts, including on 310P
        mock_npu_format_cast.reset_mock()
        with (
            mock.patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "1"}),
            mock.patch("vllm_ascend.utils.is_310p", return_value=True),
        ):
            weight = torch.randn(32, 64, dtype=torch.float32)
            result = utils.maybe_trans_nz(weight)
            self.assertIs(result, weight)
            mock_npu_format_cast.assert_not_called()

        # Test case 4: non-310P fp16 converts only when NZ=2
        mock_npu_format_cast.reset_mock()
        with (
            mock.patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "1"}),
            mock.patch("vllm_ascend.utils.is_310p", return_value=False),
        ):
            weight = torch.randn(32, 64, dtype=torch.float16)
            result = utils.maybe_trans_nz(weight)
            self.assertIs(result, weight)
            mock_npu_format_cast.assert_not_called()

        # Test case 5: non-310P fp16 converts when NZ=2
        mock_npu_format_cast.reset_mock()
        with (
            mock.patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "2"}),
            mock.patch("vllm_ascend.utils.is_310p", return_value=False),
        ):
            weight = torch.randn(32, 64, dtype=torch.float16)
            result = utils.maybe_trans_nz(weight)
            self.assertIs(result, weight)
            assert_nz_cast(weight)

        # Test case 6: non-310P bf16 converts when NZ=2
        mock_npu_format_cast.reset_mock()
        with (
            mock.patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "2"}),
            mock.patch("vllm_ascend.utils.is_310p", return_value=False),
        ):
            weight = torch.randn(32, 64, dtype=torch.bfloat16)
            result = utils.maybe_trans_nz(weight)
            self.assertIs(result, weight)
            assert_nz_cast(weight)

        # Test case 7: non-310P quantized weights still convert by default
        mock_npu_format_cast.reset_mock()
        with (
            mock.patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "1"}),
            mock.patch("vllm_ascend.utils.is_310p", return_value=False),
        ):
            weight = torch.zeros(32, 64, dtype=torch.int8)
            result = utils.maybe_trans_nz(weight)
            self.assertIs(result, weight)
            assert_nz_cast(weight)
