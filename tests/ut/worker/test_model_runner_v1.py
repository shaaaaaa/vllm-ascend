import unittest
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

import vllm_ascend.worker.model_runner_v1 as model_runner_module
from vllm_ascend.ascend_forward_context import (
    STAGED_SFA_SINGLETON_GRAPH_KEY,
    StagedSFAGraphKey,
    StagedSFAQueryProfile,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import (
    StagedSFARouteAction,
    StagedSFARouteReason,
)
from vllm_ascend.worker.block_table import MultiGroupBlockTable
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.dsa_shared_pool = False
        runner.dsa_unbundle = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestStagedSFAGraphKey(unittest.TestCase):
    def test_structural_dimensions_do_not_collide(self):
        base = STAGED_SFA_SINGLETON_GRAPH_KEY
        variants = (
            StagedSFAGraphKey(
                token_capacity=1,
                request_capacity=2,
                query_profile=StagedSFAQueryProfile.DECODE_Q1,
                max_query_len=1,
            ),
            StagedSFAGraphKey(
                token_capacity=1,
                request_capacity=1,
                query_profile=StagedSFAQueryProfile.SPEC_FIXED,
                max_query_len=1,
            ),
            StagedSFAGraphKey(
                token_capacity=1,
                request_capacity=1,
                query_profile=StagedSFAQueryProfile.DECODE_Q1,
                max_query_len=2,
            ),
        )
        self.assertEqual(
            len({base, StagedSFAGraphKey(**base.__dict__)}),
            1,
        )
        self.assertTrue(all(variant != base for variant in variants))
        self.assertEqual(len(set(variants)), len(variants))

    def test_key_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            STAGED_SFA_SINGLETON_GRAPH_KEY.token_capacity = 2

    def test_only_singleton_adapts_to_legacy_descriptor(self):
        descriptor = STAGED_SFA_SINGLETON_GRAPH_KEY.to_legacy_batch_descriptor()
        self.assertEqual(descriptor, BatchDescriptor(num_tokens=1))
        self.assertIsNone(descriptor.num_reqs)
        self.assertFalse(descriptor.uniform)
        self.assertFalse(descriptor.has_lora)

        batch_descriptor = StagedSFAGraphKey(
            token_capacity=2,
            request_capacity=2,
            query_profile=StagedSFAQueryProfile.DECODE_Q1,
            max_query_len=1,
        ).to_legacy_batch_descriptor()
        self.assertEqual(batch_descriptor, BatchDescriptor(num_tokens=2))

        invalid_keys = (
            StagedSFAGraphKey(
                token_capacity=2,
                request_capacity=1,
                query_profile=StagedSFAQueryProfile.DECODE_Q1,
                max_query_len=1,
            ),
            StagedSFAGraphKey(
                token_capacity=2,
                request_capacity=2,
                query_profile=StagedSFAQueryProfile.SPEC_FIXED,
                max_query_len=2,
            ),
        )
        for key in invalid_keys:
            with (
                self.subTest(key=key),
                self.assertRaises(NotImplementedError),
            ):
                key.to_legacy_batch_descriptor()


class TestQueryStartPadding(unittest.TestCase):
    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.uniform_decode_query_len = 1
        runner.compilation_config = SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner.query_start_loc = SimpleNamespace(
            np=np.array([0, 1, 2, -1, -1, -1, -1, -1], dtype=np.int32),
            copy_to_gpu=MagicMock(),
        )
        return runner

    def test_exact_q1_capacity_reuses_uploaded_query_starts(self):
        runner = self._build_runner()

        padded_reqs = runner._pad_query_start_loc_for_fia(
            num_tokens_padded=2,
            num_reqs_padded=2,
            num_reqs=2,
            batch_desc_num_reqs=2,
        )

        self.assertEqual(padded_reqs, 2)
        runner.query_start_loc.copy_to_gpu.assert_not_called()

    def test_padded_q1_capacity_uploads_padding(self):
        runner = self._build_runner()

        padded_reqs = runner._pad_query_start_loc_for_fia(
            num_tokens_padded=4,
            num_reqs_padded=4,
            num_reqs=2,
            batch_desc_num_reqs=4,
        )

        self.assertEqual(padded_reqs, 4)
        np.testing.assert_array_equal(
            runner.query_start_loc.np[:5],
            np.arange(5, dtype=np.int32),
        )
        runner.query_start_loc.copy_to_gpu.assert_called_once_with()


class TestStagedSFADummyBatch(unittest.TestCase):
    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.vllm_config = object()
        runner.speculative_config = None
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)
        runner.attn_state = AscendAttentionState.DecodeOnly
        runner._staged_sfa_graph_capture_sizes = (1, 4)
        runner._dp_batch_sync_buffers = {}
        return runner

    @staticmethod
    def _eligibility_kwargs(batch_size=4):
        return {
            "is_profile": False,
            "cudagraph_runtime_mode": CUDAGraphMode.PIECEWISE,
            "allow_eager": True,
            "num_active_loras": 0,
            "num_tokens_unpadded": batch_size,
            "num_tokens_padded": batch_size,
            "num_reqs": batch_size,
            "num_scheduled_tokens": np.ones(batch_size, dtype=np.int32),
            "batch_descriptor": BatchDescriptor(num_tokens=batch_size),
            "dp_route_action": StagedSFARouteAction.STAGED,
        }

    def test_exact_q1_capture_sizes_are_staged(self):
        runner = self._build_runner()
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            for runtime_mode in (
                CUDAGraphMode.NONE,
                CUDAGraphMode.PIECEWISE,
            ):
                kwargs = self._eligibility_kwargs()
                kwargs["cudagraph_runtime_mode"] = runtime_mode
                with self.subTest(runtime_mode=runtime_mode):
                    self.assertEqual(
                        runner._staged_sfa_dummy_batch_size(**kwargs),
                        4,
                    )

    def test_dp_dummy_uses_agreed_graph_capacity(self):
        runner = self._build_runner()
        kwargs = self._eligibility_kwargs(batch_size=1)
        kwargs.update(
            num_tokens_padded=4,
            batch_descriptor=BatchDescriptor(num_tokens=4),
            allow_eager=False,
        )
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            self.assertEqual(
                runner._staged_sfa_dummy_batch_size(**kwargs),
                4,
            )
            kwargs["cudagraph_runtime_mode"] = CUDAGraphMode.NONE
            self.assertIsNone(
                runner._staged_sfa_dummy_batch_size(**kwargs)
            )

    def test_dp_sync_agrees_route_in_existing_collective(self):
        runner = self._build_runner()
        runner.dp_size = 2
        runner.dp_rank = 0
        runner._skip_all_reduce_across_dp_group = MagicMock(return_value=False)

        def peer(action):
            def all_reduce(tensor, group):
                tensor[0, 1] = 4
                tensor[1, 1] = CUDAGraphMode.PIECEWISE.value
                tensor[2, 1] = tuple(StagedSFARouteAction).index(action)

            return all_reduce

        buffer_addresses = set()
        for peer_action in StagedSFARouteAction:
            with (
                self.subTest(peer_action=peer_action),
                patch.object(
                    model_runner_module,
                    "get_dp_group",
                    return_value=SimpleNamespace(cpu_group=object()),
                ),
                patch.object(
                    model_runner_module.dist,
                    "all_reduce",
                    side_effect=peer(peer_action),
                ),
            ):
                _, sizes, _, action = runner._sync_batch_across_dp(
                    num_tokens_padded=1,
                    cudagraph_mode=CUDAGraphMode.PIECEWISE.value,
                    allow_dp_padding=True,
                    staged_sfa_route_action=StagedSFARouteAction.STAGED,
                )
                self.assertEqual(sizes.tolist(), [4, 4])
                self.assertEqual(action, peer_action)
                buffer_addresses.add(sizes.data_ptr())

        self.assertEqual(len(buffer_addresses), 1)
        self.assertEqual(
            runner._skip_all_reduce_across_dp_group.call_count,
            len(StagedSFARouteAction),
        )

    def test_dp_sync_keeps_non_staged_collective_shape(self):
        runner = self._build_runner()
        runner.dp_size = 2
        runner.dp_rank = 0
        runner._staged_sfa_graph_capture_sizes = ()
        runner._skip_all_reduce_across_dp_group = MagicMock(return_value=False)

        def all_reduce(tensor, group):
            self.assertEqual(tuple(tensor.shape), (2, 2))
            tensor[0, 1] = 4
            tensor[1, 1] = CUDAGraphMode.PIECEWISE.value

        with (
            patch.object(
                model_runner_module,
                "get_dp_group",
                return_value=SimpleNamespace(cpu_group=object()),
            ),
            patch.object(
                model_runner_module.dist,
                "all_reduce",
                side_effect=all_reduce,
            ),
        ):
            _, sizes, mode, action = runner._sync_batch_across_dp(
                num_tokens_padded=1,
                cudagraph_mode=CUDAGraphMode.PIECEWISE.value,
                allow_dp_padding=True,
            )

        self.assertEqual(sizes.tolist(), [4, 4])
        self.assertEqual(mode, CUDAGraphMode.PIECEWISE.value)
        self.assertIsNone(action)

    def test_dp_sync_keeps_staged_shape_for_neutral_bootstrap_rank(self):
        runner = self._build_runner()
        runner.dp_size = 2
        runner.dp_rank = 0
        runner._skip_all_reduce_across_dp_group = MagicMock(return_value=False)

        def all_reduce(tensor, group):
            self.assertEqual(tuple(tensor.shape), (3, 2))
            self.assertEqual(tensor[2, 0].item(), 0)
            tensor[0, 1] = 4
            tensor[1, 1] = CUDAGraphMode.PIECEWISE.value
            tensor[2, 1] = tuple(StagedSFARouteAction).index(
                StagedSFARouteAction.STAGED
            )

        with (
            patch.object(
                model_runner_module,
                "get_dp_group",
                return_value=SimpleNamespace(cpu_group=object()),
            ),
            patch.object(
                model_runner_module.dist,
                "all_reduce",
                side_effect=all_reduce,
            ),
        ):
            _, sizes, mode, action = runner._sync_batch_across_dp(
                num_tokens_padded=1,
                cudagraph_mode=CUDAGraphMode.NONE.value,
                allow_dp_padding=False,
            )

        self.assertEqual(sizes.tolist(), [1, 4])
        self.assertEqual(mode, CUDAGraphMode.NONE.value)
        self.assertIsNone(action)

    def test_dp_redispatches_only_when_agreement_changes_the_key(self):
        runner = self._build_runner()
        runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                tensor_parallel_size=2,
            ),
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        )
        runner.parallel_config = SimpleNamespace(data_parallel_rank=0)
        runner.input_batch = SimpleNamespace(
            num_computed_tokens_cpu=np.ones(1, dtype=np.int32),
            lora_id_to_lora_request={},
        )
        runner.model_config = SimpleNamespace(is_encoder_decoder=False)
        runner.uniform_decode_query_len = 1
        runner._pad_for_sequence_parallelism = MagicMock(
            side_effect=lambda value: value
        )
        runner.cudagraph_dispatcher = SimpleNamespace(
            dispatch=MagicMock(
                side_effect=lambda num_tokens, **_: (
                    CUDAGraphMode.PIECEWISE,
                    BatchDescriptor(num_tokens=num_tokens),
                )
            )
        )

        for agreed_size, expected_calls in ((1, 1), (4, 2)):
            runner.cudagraph_dispatcher.dispatch.reset_mock()
            runner._sync_batch_across_dp = MagicMock(
                return_value=(
                    False,
                    torch.tensor([agreed_size, agreed_size]),
                    CUDAGraphMode.PIECEWISE.value,
                    StagedSFARouteAction.STAGED,
                )
            )
            with (
                self.subTest(agreed_size=agreed_size),
                patch.object(model_runner_module, "enable_sp", return_value=False),
            ):
                _, descriptor, _, _, _ = (
                    runner._determine_batch_execution_and_padding(
                        num_tokens=1,
                        num_reqs=1,
                        num_scheduled_tokens_np=np.ones(1, dtype=np.int32),
                        max_num_scheduled_tokens=1,
                        use_cascade_attn=False,
                        staged_sfa_route_action=StagedSFARouteAction.STAGED,
                    )
                )

            self.assertEqual(descriptor.num_tokens, agreed_size)
            self.assertEqual(
                runner.cudagraph_dispatcher.dispatch.call_count,
                expected_calls,
            )

    def test_padded_non_q1_and_unsupported_batches_fall_back(self):
        runner = self._build_runner()
        cases = {
            "unsupported_size": {
                "num_tokens_unpadded": 2,
                "num_tokens_padded": 2,
                "num_reqs": 2,
                "num_scheduled_tokens": np.ones(2, dtype=np.int32),
                "batch_descriptor": BatchDescriptor(num_tokens=2),
            },
            "token_padding": {
                "num_tokens_padded": 8,
                "batch_descriptor": BatchDescriptor(num_tokens=8),
            },
            "non_q1": {
                "num_scheduled_tokens": np.array([1, 1, 2, 0]),
            },
            "uniform_descriptor": {
                "batch_descriptor": BatchDescriptor(
                    num_tokens=4,
                    uniform=True,
                ),
            },
            "dp_fallback": {
                "dp_route_action": StagedSFARouteAction.SAFE_NATIVE,
            },
            "profile": {"is_profile": True},
        }
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            for case_name, overrides in cases.items():
                kwargs = self._eligibility_kwargs()
                kwargs.update(overrides)
                with self.subTest(case=case_name):
                    self.assertIsNone(runner._staged_sfa_dummy_batch_size(**kwargs))

    def test_live_route_accepts_dp_padding_before_mutation(self):
        runner = self._build_runner()
        request_ids = [f"req-{index}" for index in range(4)]
        local_kwargs = {
            "num_tokens_unpadded": 4,
            "num_reqs": 4,
            "num_scheduled_tokens": np.ones(4, dtype=np.int32),
            "prompt_lens": np.full(4, 4096, dtype=np.int32),
            "index_topk": 2048,
            "has_cascade_attention": False,
            "request_ids": request_ids,
            "kv_connector_metadata": SimpleNamespace(
                requests=[
                    SimpleNamespace(
                        req_id=req_id,
                        is_sparse_decode=True,
                        load_spec=SimpleNamespace(
                            can_load=True,
                            lmcache_cached_tokens=4096,
                        ),
                    )
                    for req_id in request_ids
                ]
            ),
        }
        final_kwargs = {
            "dp_route_action": StagedSFARouteAction.STAGED,
            "cudagraph_mode": CUDAGraphMode.PIECEWISE,
            "batch_descriptor": BatchDescriptor(num_tokens=4),
            "num_tokens_unpadded": 4,
            "num_tokens_padded": 4,
            "num_reqs": 4,
            "should_ubatch": False,
        }
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            local_route = runner._staged_sfa_local_route(**local_kwargs)
            route = runner._staged_sfa_live_route(
                local_route=local_route,
                **final_kwargs,
            )
            self.assertEqual(route.action, StagedSFARouteAction.STAGED)
            self.assertEqual(route.reason, StagedSFARouteReason.ELIGIBLE)
            self.assertEqual(route.graph_key, StagedSFAGraphKey.exact_q1(4))
            self.assertEqual(route.frontiers, (4096,) * 4)

            one_request_kwargs = dict(local_kwargs)
            one_request_kwargs.update(
                num_tokens_unpadded=1,
                num_reqs=1,
                num_scheduled_tokens=np.ones(1, dtype=np.int32),
                prompt_lens=np.full(1, 4096, dtype=np.int32),
                request_ids=request_ids[:1],
                kv_connector_metadata=SimpleNamespace(requests=local_kwargs["kv_connector_metadata"].requests[:1]),
            )
            padded_route = runner._staged_sfa_live_route(
                local_route=runner._staged_sfa_local_route(**one_request_kwargs),
                **{
                    **final_kwargs,
                    "num_tokens_unpadded": 1,
                    "num_reqs": 1,
                },
            )
            self.assertEqual(
                padded_route.graph_key,
                StagedSFAGraphKey.exact_q1(4),
            )

            for name, overrides in {
                "multi_token": {
                    "num_scheduled_tokens": np.array(
                        [1, 1, 2, 0],
                        dtype=np.int32,
                    ),
                },
                "cascade": {"has_cascade_attention": True},
                "short_prompt": {
                    "prompt_lens": np.array([4096, 2047, 4096, 4096]),
                },
                "dense_prefix_hit": {
                    "prompt_lens": np.full(4, 1, dtype=np.int32),
                    "kv_connector_metadata": SimpleNamespace(
                        requests=[
                            SimpleNamespace(
                                req_id=req_id,
                                is_sparse_decode=False,
                                load_spec=SimpleNamespace(can_load=True),
                            )
                            for req_id in request_ids
                        ]
                    ),
                },
                "mixed_connector_load": {
                    "kv_connector_metadata": SimpleNamespace(
                        requests=[
                            SimpleNamespace(
                                req_id=req_id,
                                is_sparse_decode=index >= 2,
                                load_spec=SimpleNamespace(
                                    can_load=True,
                                    lmcache_cached_tokens=4096,
                                ),
                            )
                            for index, req_id in enumerate(request_ids)
                        ]
                    ),
                },
                "short_frontier": {
                    "kv_connector_metadata": SimpleNamespace(
                        requests=[
                            SimpleNamespace(
                                req_id=req_id,
                                is_sparse_decode=True,
                                load_spec=SimpleNamespace(
                                    can_load=True,
                                    lmcache_cached_tokens=1024,
                                ),
                            )
                            for req_id in request_ids
                        ]
                    ),
                },
            }.items():
                rejected = dict(local_kwargs)
                rejected.update(overrides)
                with self.subTest(name=name):
                    route = runner._staged_sfa_local_route(**rejected)
                    self.assertEqual(
                        route.action,
                        StagedSFARouteAction.FATAL
                        if name in ("short_prompt", "short_frontier")
                        else StagedSFARouteAction.SAFE_NATIVE,
                    )
                    if name == "dense_prefix_hit":
                        self.assertEqual(
                            route.reason,
                            StagedSFARouteReason.DENSE_PREFIX_HIT,
                        )
                    elif name == "mixed_connector_load":
                        self.assertEqual(
                            route.reason,
                            StagedSFARouteReason.MIXED_CONNECTOR_LOAD,
                        )
                    elif name == "short_frontier":
                        self.assertEqual(
                            route.reason,
                            StagedSFARouteReason.FRONTIER_TOO_SHORT,
                        )

            ubatch_route = runner._staged_sfa_live_route(
                local_route=local_route,
                **{**final_kwargs, "should_ubatch": True},
            )
            self.assertEqual(
                ubatch_route.reason,
                StagedSFARouteReason.UBATCH,
            )

            peer_fallback = runner._staged_sfa_live_route(
                local_route=local_route,
                **{
                    **final_kwargs,
                    "dp_route_action": StagedSFARouteAction.SAFE_NATIVE,
                },
            )
            self.assertEqual(
                peer_fallback.reason,
                StagedSFARouteReason.RUNTIME_PARALLELISM,
            )

            runner.attn_state = AscendAttentionState.PrefillNoCache
            route = runner._staged_sfa_local_route(**local_kwargs)
            self.assertEqual(route.action, StagedSFARouteAction.SAFE_NATIVE)
            self.assertEqual(route.reason, StagedSFARouteReason.NOT_DECODE)

    def test_non_native_routes_report_stable_action_and_reason(self):
        runner = self._build_runner()
        for action in (
            StagedSFARouteAction.RECOMPUTE,
            StagedSFARouteAction.FATAL,
        ):
            route = model_runner_module.StagedSFARouteDecision(
                action,
                StagedSFARouteReason.SPARSE_LOAD_UNAVAILABLE,
            )

            with (
                self.subTest(action=action),
                self.assertRaisesRegex(
                    RuntimeError,
                    rf"\[SFA_ROUTE\] action={action.value} "
                    r"reason=sparse_load_unavailable",
                ),
            ):
                runner._apply_staged_sfa_route(route)

    def test_native_route_logs_once_per_reason(self):
        runner = self._build_runner()
        route = model_runner_module.StagedSFARouteDecision(
            StagedSFARouteAction.SAFE_NATIVE,
            StagedSFARouteReason.DENSE_PREFIX_HIT,
        )

        with patch.object(model_runner_module.logger, "info") as log:
            self.assertIsNone(runner._apply_staged_sfa_route(route))
            self.assertIsNone(runner._apply_staged_sfa_route(route))

        log.assert_called_once_with("[SFA_ROUTE] action=safe_native reason=dense_prefix_hit")

    def test_native_q1_rows_have_unique_ids_and_query_starts(self):
        request_ids = NPUModelRunner._staged_sfa_dummy_request_ids(4)
        query_start_locs = NPUModelRunner._staged_sfa_q1_query_start_locs(
            4,
            dtype=np.dtype(np.int32),
        )

        self.assertEqual(
            request_ids,
            [
                "staged-sfa-graph-dummy-0",
                "staged-sfa-graph-dummy-1",
                "staged-sfa-graph-dummy-2",
                "staged-sfa-graph-dummy-3",
            ],
        )
        self.assertEqual(len(set(request_ids)), 4)
        np.testing.assert_array_equal(
            query_start_locs,
            np.arange(5, dtype=np.int32),
        )

    def test_speculative_dummy_batches_fall_back(self):
        runner = self._build_runner()
        kwargs = self._eligibility_kwargs()
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            runner.speculative_config = SimpleNamespace(method="mtp")
            self.assertIsNone(runner._staged_sfa_dummy_batch_size(**kwargs))

    def test_two_group_dummy_rows_use_noncolliding_physical_slots(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        positions = np.array([8, 9, 10, 11], dtype=np.int64)
        runner._prepare_staged_sfa_dummy_block_tables(
            batch_size=4,
            positions=positions,
        )

        expected_slots = (
            np.array([0, 5, 10, 15], dtype=np.int64),
            np.array([0, 9, 18, 27], dtype=np.int64),
        )
        for group_index, block_table in enumerate(runner.input_batch.block_table.block_tables):
            expected_rows = np.broadcast_to(
                np.arange(4, dtype=np.int32).reshape(-1, 1),
                (4, block_table.max_num_blocks_per_req),
            )
            np.testing.assert_array_equal(
                block_table.block_table.np[:4],
                expected_rows,
            )
            np.testing.assert_array_equal(
                block_table.slot_mapping.np[:4],
                expected_slots[group_index],
            )
            self.assertEqual(
                np.unique(block_table.slot_mapping.np[:4]).size,
                4,
            )

    def test_dummy_block_rows_require_enough_physical_blocks(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(num_blocks=3)
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        with self.assertRaisesRegex(RuntimeError, "one physical block"):
            runner._prepare_staged_sfa_dummy_block_tables(
                batch_size=4,
                positions=np.arange(4),
            )

    def test_dummy_block_rows_reject_asymmetric_group_pool(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[3, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"KV group 0: .*available_blocks=3",
        ):
            runner._prepare_staged_sfa_dummy_block_tables(
                batch_size=4,
                positions=np.arange(4),
            )

    def test_dummy_position_rejects_logical_row_overflow(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "logical block-table capacity for KV group 0",
        ):
            runner._prepare_staged_sfa_dummy_block_tables(
                batch_size=4,
                positions=np.array([16, 1, 2, 3], dtype=np.int64),
            )

    def test_dummy_position_capacity_includes_cp_world_size(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        block_table = MultiGroupBlockTable(
            max_num_reqs=4,
            max_model_len=16,
            max_num_batched_tokens=4,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[4, 8],
            kernel_sizes=[[4], [8]],
            max_num_blocks=[4, 2],
        )
        for group_table in block_table.block_tables:
            group_table.dcp_world_size = 2
            group_table.dcp_rank = 0
            group_table.pcp_world_size = 1
            group_table.pcp_rank = 0
        runner.input_batch = SimpleNamespace(block_table=block_table)

        runner._prepare_staged_sfa_dummy_block_tables(
            batch_size=4,
            positions=np.array([16, 18, 20, 22], dtype=np.int64),
        )

        for group_table in block_table.block_tables:
            self.assertEqual(
                np.unique(group_table.slot_mapping.np[:4]).size,
                4,
            )


class TestSFALayerwiseGraphModeCompatibility(unittest.TestCase):
    @staticmethod
    def _build_runner(mode, *, use_sparse=True):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.use_sparse = use_sparse
        runner._profiling_cudagraph_memory = False
        runner.compilation_config = SimpleNamespace(cudagraph_mode=mode)
        return runner

    def test_internal_data_parallel_staged_graph_is_accepted(self):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=2)

        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_connector_supports_sparse_load",
                return_value=True,
            ),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_explicit_unsupported_staged_graph_request_is_rejected(self):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)

        with (
            patch.object(
                model_runner_module.envs_ascend,
                "VLLM_ASCEND_SFA_STAGED_GRAPH",
                True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=False,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configuration_errors",
                return_value=("speculative decoding/MTP is not implemented",),
            ),
            self.assertRaisesRegex(ValueError, "MTP"),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_staged_graph_requires_sparse_load_connector_capability(self):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)

        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_connector_supports_sparse_load",
                return_value=False,
            ),
            self.assertRaisesRegex(
                ValueError,
                "batched sparse selective loads",
            ),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_connector_supports_sparse_load",
                return_value=True,
            ),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_profiling_defers_connector_capability_validation(self):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)
        runner._profiling_cudagraph_memory = True

        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_connector_supports_sparse_load",
                return_value=False,
            ),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_full_graph_modes_are_rejected_for_layerwise_connector(self):
        connector = SimpleNamespace(uses_layerwise_model_callbacks=True)
        for mode in (
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ):
            with (
                self.subTest(mode=mode),
                patch.object(
                    model_runner_module,
                    "has_kv_transfer_group",
                    return_value=True,
                ),
                patch.object(
                    model_runner_module,
                    "get_kv_transfer_group",
                    return_value=connector,
                ),
                self.assertRaisesRegex(ValueError, "PIECEWISE"),
            ):
                runner = self._build_runner(mode)
                runner.vllm_config = object()
                runner.parallel_config = SimpleNamespace(data_parallel_size=1)
                with patch.object(
                    model_runner_module,
                    "staged_sfa_graph_configured",
                    return_value=False,
                ):
                    runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_compatible_modes_and_connectors_are_accepted(self):
        cases = (
            (
                CUDAGraphMode.PIECEWISE,
                True,
                True,
                True,
            ),
            (
                CUDAGraphMode.FULL,
                False,
                True,
                True,
            ),
            (
                CUDAGraphMode.FULL,
                True,
                False,
                True,
            ),
            (
                CUDAGraphMode.FULL,
                True,
                True,
                False,
            ),
        )
        for mode, use_sparse, has_connector, uses_layerwise in cases:
            with (
                self.subTest(
                    mode=mode,
                    use_sparse=use_sparse,
                    has_connector=has_connector,
                    uses_layerwise=uses_layerwise,
                ),
                patch.object(
                    model_runner_module,
                    "has_kv_transfer_group",
                    return_value=has_connector,
                ),
                patch.object(
                    model_runner_module,
                    "get_kv_transfer_group",
                    return_value=SimpleNamespace(
                        uses_layerwise_model_callbacks=uses_layerwise,
                    ),
                ),
            ):
                runner = self._build_runner(
                    mode,
                    use_sparse=use_sparse,
                )
                runner.vllm_config = object()
                runner.parallel_config = SimpleNamespace(data_parallel_size=1)
                with patch.object(
                    model_runner_module,
                    "staged_sfa_graph_configured",
                    return_value=False,
                ):
                    runner._validate_sfa_layerwise_connector_cudagraph_mode()


class TestStagedSFAStartupCaptureValidation(unittest.TestCase):
    def setUp(self):
        configured = patch.object(
            model_runner_module,
            "staged_sfa_graph_configured",
            return_value=True,
        )
        capture_sizes = patch.object(
            model_runner_module,
            "staged_sfa_graph_capture_sizes",
            return_value=(1, 2),
        )
        configured.start()
        capture_sizes.start()
        self.addCleanup(configured.stop)
        self.addCleanup(capture_sizes.stop)

    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.compilation_config = SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
        runner.vllm_config = SimpleNamespace(
            compilation_config=runner.compilation_config,
            model_config=SimpleNamespace(
                use_mla=True,
                hf_text_config=SimpleNamespace(index_topk=2048),
            ),
            kv_transfer_config=object(),
            speculative_config=None,
            lora_config=None,
        )
        runner._staged_sfa_startup_capture_attempted = False
        runner._profiling_cudagraph_memory = False
        return runner

    def test_capture_model_preserves_result_and_validates_cross_layer_warmup(self):
        runner = self._build_runner()
        calls = []
        impl = SimpleNamespace(
            seal_staged_sfa_capture=MagicMock(),
        )
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
                side_effect=lambda: calls.append("reset"),
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
                side_effect=lambda _runner: calls.append("parent") or 123,
            ) as parent_capture,
            patch.object(
                model_runner_module.ACLGraphWrapper,
                "seal_staged_entries",
                return_value=4,
            ) as seal_entries,
            patch.object(
                runner,
                "_collect_staged_sfa_impls",
                return_value=(("layer-0", impl),),
            ),
        ):
            result = runner.capture_model()

        self.assertEqual(result, 123)
        self.assertEqual(calls, ["reset", "parent"])
        self.assertTrue(runner._staged_sfa_startup_capture_attempted)
        self.assertEqual(runner._staged_sfa_impls, (("layer-0", impl),))
        parent_capture.assert_called_once_with(runner)
        graph_keys = tuple(
            StagedSFAGraphKey.exact_q1(size) for size in (1, 2)
        )
        impl.seal_staged_sfa_capture.assert_called_once_with(graph_keys)
        seal_entries.assert_called_once_with(graph_keys, 2)

    def test_capture_model_rejects_a_missing_configured_key(self):
        runner = self._build_runner()
        impl = SimpleNamespace(
            seal_staged_sfa_capture=MagicMock(
                side_effect=RuntimeError("missing_keys=(2,)"),
            ),
        )
        with (
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(runner, "_reset_staged_sfa_startup_capture"),
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
                return_value=0,
            ),
            patch.object(
                runner,
                "_collect_staged_sfa_impls",
                return_value=(("layer-0", impl),),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                r"layer-0.*missing_keys=.*2",
            ),
        ):
            runner.capture_model()

    def test_failed_parent_capture_cannot_retry_stale_outer_graphs(self):
        runner = self._build_runner()
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
                side_effect=RuntimeError("capture failed"),
            ) as parent_capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                runner.capture_model()
            with self.assertRaisesRegex(
                RuntimeError,
                "startup graph capture was already attempted",
            ):
                runner.capture_model()

        self.assertTrue(runner._staged_sfa_startup_capture_attempted)
        parent_capture.assert_called_once_with(runner)

    def test_second_capture_attempt_is_rejected_before_parent(self):
        runner = self._build_runner()
        runner._staged_sfa_startup_capture_attempted = True
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ) as reset,
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
            ) as parent_capture,
            self.assertRaisesRegex(
                RuntimeError,
                "startup graph capture was already attempted",
            ),
        ):
            runner.capture_model()

        reset.assert_not_called()
        parent_capture.assert_not_called()

    def test_graph_memory_profile_resets_temporary_capture_state(self):
        runner = self._build_runner()
        with (
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "profile_cudagraph_memory",
                side_effect=lambda _runner: (
                    self.assertTrue(runner._profiling_cudagraph_memory)
                    or 123
                ),
            ) as parent_profile,
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ) as reset,
            patch.object(
                model_runner_module,
                "reset_graph_params",
            ) as reset_params,
        ):
            result = runner.profile_cudagraph_memory()

        self.assertEqual(result, 123)
        self.assertFalse(runner._profiling_cudagraph_memory)
        parent_profile.assert_called_once_with(runner)
        reset.assert_called_once_with()
        reset_params.assert_called_once_with()

    def test_graph_memory_profile_cleans_up_after_failure(self):
        runner = self._build_runner()
        runner.cudagraph_dispatcher = SimpleNamespace(
            cudagraph_keys={
                CUDAGraphMode.PIECEWISE: {object()},
            },
            keys_initialized=True,
        )
        with (
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "profile_cudagraph_memory",
                side_effect=RuntimeError("profile failed"),
            ),
            patch.object(
                model_runner_module.ACLGraphWrapper,
                "clear_all_graphs",
            ) as clear_graphs,
            patch.object(
                runner,
                "_cleanup_profiling_kv_cache",
            ) as cleanup_cache,
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ) as reset,
            patch.object(
                model_runner_module,
                "set_cudagraph_capturing_enabled",
            ) as set_capture_enabled,
            patch.object(
                model_runner_module,
                "reset_graph_params",
            ) as reset_params,
            self.assertRaisesRegex(RuntimeError, "profile failed"),
        ):
            runner.profile_cudagraph_memory()

        self.assertFalse(runner._profiling_cudagraph_memory)
        clear_graphs.assert_called_once_with()
        cleanup_cache.assert_called_once_with()
        reset.assert_called_once_with()
        reset_params.assert_called_once_with()
        set_capture_enabled.assert_called_once_with(False)
        self.assertFalse(runner.cudagraph_dispatcher.keys_initialized)
        self.assertFalse(
            runner.cudagraph_dispatcher.cudagraph_keys[
                CUDAGraphMode.PIECEWISE
            ]
        )

    def test_kv_cache_reinitialization_after_capture_is_rejected(self):
        runner = self._build_runner()
        runner._staged_sfa_startup_capture_attempted = True

        with (
            patch.object(
                runner,
                "_validate_sfa_layerwise_connector_cudagraph_mode",
            ) as validate,
            self.assertRaisesRegex(
                RuntimeError,
                "KV cache cannot be reinitialized after graph capture",
            ),
        ):
            runner.initialize_kv_cache(object())

        validate.assert_not_called()

if __name__ == "__main__":
    unittest.main()
