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
from vllm_ascend.worker.model_runner_v1 import (
    NPUModelRunner,
    _dsa_require_full_scratch_rows,
    _DSAScratchStateSlotManager,
)


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


class TestDSAScratchStateSlots(unittest.TestCase):
    @staticmethod
    def _runner(capacity=4):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.pin_memory = False
        runner._dsa_scratch_state_slot_manager = (
            _DSAScratchStateSlotManager(capacity)
        )
        runner.dsa_scratch_state_indices = runner._make_buffer(
            capacity + 1, dtype=torch.int32
        )
        runner.dsa_scratch_request_generations = runner._make_buffer(
            capacity + 1, dtype=torch.int64
        )
        runner._dsa_scratch_blocks_per_request = 2
        runner.input_batch = SimpleNamespace(
            req_ids=[],
            num_reqs=0,
        )
        runner.requests = {}
        return runner

    @staticmethod
    def _scheduler_output(
        *,
        finished=(),
        preempted=(),
        resumed=(),
    ):
        return SimpleNamespace(
            finished_req_ids=set(finished),
            preempted_req_ids=set(preempted),
            scheduled_cached_reqs=SimpleNamespace(
                resumed_req_ids=list(resumed)
            ),
        )

    @staticmethod
    def _request(blocks):
        return SimpleNamespace(block_ids=(list(blocks),))

    def test_metadata_follows_reordered_batch_rows(self):
        runner = self._runner()
        manager = runner._dsa_scratch_state_slot_manager
        a = manager.bind("a", (10, 11))
        b = manager.bind("b", (20, 21))
        runner.input_batch.req_ids = ["b", "a"]
        runner.input_batch.num_reqs = 2

        indices, generations = (
            runner._prepare_dsa_scratch_state_metadata(
                num_reqs=2,
                request_capacity=4,
                dummy_metadata=False,
            )
        )

        self.assertEqual(
            indices.tolist(),
            [b.state_index, a.state_index, 2, 3],
        )
        self.assertEqual(
            generations.tolist(),
            [b.generation, a.generation, -3, -4],
        )

    def test_metadata_rejects_duplicate_real_state_rows(self):
        runner = self._runner()
        manager = runner._dsa_scratch_state_slot_manager
        binding = manager.bind("a", (10, 11))
        # Corrupt the manager deliberately: this is the condition that would
        # make two request AIVs write one resident-state cache line.
        manager._bindings["b"] = binding
        runner.input_batch.req_ids = ["a", "b"]
        runner.input_batch.num_reqs = 2

        with self.assertRaisesRegex(
            RuntimeError,
            "must map to distinct scratch-state rows",
        ):
            runner._prepare_dsa_scratch_state_metadata(
                num_reqs=2,
                request_capacity=2,
                dummy_metadata=False,
            )

    def test_metadata_rejects_out_of_range_real_state_row(self):
        runner = self._runner(capacity=2)
        manager = runner._dsa_scratch_state_slot_manager
        binding = manager.bind("a", (10, 11))
        # Corrupt the manager deliberately. The device kernel indexes resident
        # state directly with this row, so uniqueness alone is insufficient.
        manager._bindings["a"] = type(binding)(
            state_index=manager.capacity,
            generation=binding.generation,
            scratch_block_prefix=binding.scratch_block_prefix,
        )
        runner.input_batch.req_ids = ["a"]
        runner.input_batch.num_reqs = 1

        with self.assertRaisesRegex(
            RuntimeError,
            r"must be in range \[0, 2\).*rows=\[2\].*capacity=2",
        ):
            runner._prepare_dsa_scratch_state_metadata(
                num_reqs=1,
                request_capacity=1,
                dummy_metadata=False,
            )

    def test_unscheduled_request_keeps_binding(self):
        runner = self._runner()
        manager = runner._dsa_scratch_state_slot_manager
        binding = manager.bind("unscheduled", (10, 11))
        runner.input_batch.req_ids = []
        runner.input_batch.num_reqs = 0

        with patch.object(
            model_runner_module.GPUModelRunner,
            "_update_states",
            return_value=None,
        ):
            runner._update_states(self._scheduler_output())

        self.assertEqual(manager.get("unscheduled"), binding)

    def test_preempt_resume_gets_new_generation(self):
        runner = self._runner()
        manager = runner._dsa_scratch_state_slot_manager
        old = manager.bind("a", (10, 11))
        runner.input_batch.req_ids = ["a"]
        runner.input_batch.num_reqs = 1
        runner.requests["a"] = self._request((10, 11))

        with patch.object(
            model_runner_module.GPUModelRunner,
            "_update_states",
            return_value=None,
        ):
            runner._update_states(
                self._scheduler_output(
                    preempted=("a",),
                    resumed=("a",),
                )
            )

        resumed = manager.get("a")
        self.assertIsNotNone(resumed)
        self.assertNotEqual(resumed.generation, old.generation)

    def test_scratch_prefix_change_invalidates_same_state_row(self):
        runner = self._runner()
        manager = runner._dsa_scratch_state_slot_manager
        runner.input_batch.req_ids = ["a"]
        runner.input_batch.num_reqs = 1
        runner.requests["a"] = self._request((10, 11, 99))

        with patch.object(
            model_runner_module.GPUModelRunner,
            "_update_states",
            return_value=None,
        ):
            runner._update_states(self._scheduler_output())
            old = manager.get("a")
            runner.requests["a"].block_ids[0][1] = 12
            runner._update_states(self._scheduler_output())
            changed = manager.get("a")

        self.assertIsNotNone(old)
        self.assertIsNotNone(changed)
        self.assertEqual(changed.state_index, old.state_index)
        self.assertNotEqual(changed.generation, old.generation)
        self.assertEqual(changed.scratch_block_prefix, (10, 12))

    def test_empty_and_partial_prefix_expand_without_error(self):
        runner = self._runner()
        manager = runner._dsa_scratch_state_slot_manager
        runner.input_batch.req_ids = ["short"]
        runner.input_batch.num_reqs = 1
        runner.requests["short"] = self._request(())

        with patch.object(
            model_runner_module.GPUModelRunner,
            "_update_states",
            return_value=None,
        ):
            runner._update_states(self._scheduler_output())
            empty = manager.get("short")
            runner.requests["short"].block_ids[0].append(10)
            runner._update_states(self._scheduler_output())
            partial = manager.get("short")
            runner.requests["short"].block_ids[0].append(11)
            runner._update_states(self._scheduler_output())
            complete = manager.get("short")

        self.assertIsNotNone(empty)
        self.assertIsNotNone(partial)
        self.assertIsNotNone(complete)
        self.assertEqual(empty.scratch_block_prefix, ())
        self.assertEqual(partial.scratch_block_prefix, (10,))
        self.assertEqual(complete.scratch_block_prefix, (10, 11))
        self.assertEqual(
            empty.state_index,
            partial.state_index,
        )
        self.assertEqual(
            partial.state_index,
            complete.state_index,
        )
        self.assertNotEqual(empty.generation, partial.generation)
        self.assertNotEqual(partial.generation, complete.generation)

    def test_update_rejects_null_block_in_existing_scratch_prefix(self):
        runner = self._runner()
        runner.input_batch.req_ids = ["bad"]
        runner.input_batch.num_reqs = 1
        runner.requests["bad"] = self._request((10, 0, 99))

        with (
            patch.object(
                model_runner_module.GPUModelRunner,
                "_update_states",
                return_value=None,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                r"positive block IDs.*block_id=0",
            ),
        ):
            runner._update_states(self._scheduler_output())

    def test_positive_boundary_readiness_requires_full_scratch_prefix(self):
        runner = self._runner()
        runner._dsa_scratch_state_slot_manager.bind("short", (10,))
        runner.input_batch.req_ids = ["short"]
        runner.input_batch.num_reqs = 1

        with self.assertRaisesRegex(
            RuntimeError,
            r"incomplete.*positive-boundary sparse decode.*"
            r"allocated_blocks=1, required_blocks=2",
        ):
            runner._prepare_dsa_scratch_state_metadata(
                num_reqs=1,
                request_capacity=1,
                dummy_metadata=False,
                require_full_scratch_rows=np.asarray([True]),
            )

    def test_zero_boundary_readiness_allows_partial_scratch_prefix(self):
        runner = self._runner()
        binding = runner._dsa_scratch_state_slot_manager.bind(
            "short",
            (10,),
        )
        runner.input_batch.req_ids = ["short"]
        runner.input_batch.num_reqs = 1

        state_indices, generations = (
            runner._prepare_dsa_scratch_state_metadata(
                num_reqs=1,
                request_capacity=1,
                dummy_metadata=False,
                require_full_scratch_rows=np.asarray([False]),
            )
        )

        self.assertEqual(state_indices.tolist(), [binding.state_index])
        self.assertEqual(generations.tolist(), [binding.generation])

    def test_scratch_readiness_uses_total_sequence_length_threshold(self):
        capacity = 4096
        required = _dsa_require_full_scratch_rows(
            np.asarray([capacity - 1, capacity, capacity + 1]),
            num_reqs=3,
            scratch_tokens=capacity,
        )

        self.assertEqual(required.tolist(), [False, True, True])

    def test_scratch_readiness_includes_scheduled_mtp_lookahead(self):
        capacity = 4096
        computed = np.asarray([capacity - 2], dtype=np.int64)
        scheduled_mtp_rows = np.asarray([2], dtype=np.int64)

        required = _dsa_require_full_scratch_rows(
            computed + scheduled_mtp_rows,
            num_reqs=1,
            scratch_tokens=capacity,
        )

        self.assertEqual(required.tolist(), [True])

    def test_short_full_hit_recompute_does_not_require_full_scratch(self):
        required = _dsa_require_full_scratch_rows(
            np.asarray([2047], dtype=np.int64),
            num_reqs=1,
            scratch_tokens=4096,
        )

        self.assertEqual(required.tolist(), [False])

    def test_finished_same_id_is_new_incarnation(self):
        runner = self._runner()
        manager = runner._dsa_scratch_state_slot_manager
        old = manager.bind("same", (10, 11))
        runner.input_batch.req_ids = ["same"]
        runner.input_batch.num_reqs = 1
        runner.requests["same"] = self._request((10, 11))

        with patch.object(
            model_runner_module.GPUModelRunner,
            "_update_states",
            return_value=None,
        ):
            runner._update_states(
                self._scheduler_output(finished=("same",))
            )

        reincarnated = manager.get("same")
        self.assertIsNotNone(reincarnated)
        self.assertNotEqual(reincarnated.generation, old.generation)

    def test_released_state_row_is_reused_with_new_generation(self):
        manager = _DSAScratchStateSlotManager(1)
        old = manager.bind("a", (10, 11))
        manager.release("a")
        new = manager.bind("b", (20, 21))

        self.assertEqual(new.state_index, old.state_index)
        self.assertNotEqual(new.generation, old.generation)

    def test_dummy_rows_have_valid_indices_and_reserved_generations(self):
        runner = self._runner(capacity=3)

        indices, generations = (
            runner._prepare_dsa_scratch_state_metadata(
                num_reqs=2,
                request_capacity=3,
                dummy_metadata=True,
            )
        )

        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(generations.tolist(), [-1, -2, -3])

    def test_fia_padding_row_uses_in_range_state_index(self):
        runner = self._runner(capacity=3)

        indices, generations = (
            runner._prepare_dsa_scratch_state_metadata(
                num_reqs=3,
                request_capacity=4,
                dummy_metadata=True,
            )
        )

        self.assertEqual(indices.tolist(), [0, 1, 2, 0])
        self.assertEqual(generations.tolist(), [-1, -2, -3, -4])


class TestStagedSFAGraphKey(unittest.TestCase):
    def test_structural_dimensions_do_not_collide(self):
        base = STAGED_SFA_SINGLETON_GRAPH_KEY
        variants = (
            StagedSFAGraphKey.exact_q1(2),
            StagedSFAGraphKey.fixed_spec(1, 2),
            StagedSFAGraphKey.fixed_spec(1, 3),
        )
        self.assertEqual(
            len({base, StagedSFAGraphKey(**base.__dict__)}),
            1,
        )
        self.assertTrue(all(variant != base for variant in variants))
        self.assertEqual(len(set(variants)), len(variants))

    def test_invalid_structural_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            StagedSFAGraphKey(
                token_capacity=2,
                request_capacity=1,
                query_profile=StagedSFAQueryProfile.DECODE_Q1,
                max_query_len=1,
            )
        with self.assertRaises(ValueError):
            StagedSFAGraphKey(
                token_capacity=2,
                request_capacity=2,
                query_profile=StagedSFAQueryProfile.SPEC_FIXED,
                max_query_len=2,
            )

    def test_key_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            STAGED_SFA_SINGLETON_GRAPH_KEY.token_capacity = 2

    def test_structural_keys_adapt_to_legacy_descriptor(self):
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

        spec_key = StagedSFAGraphKey.fixed_spec(2, 3)
        self.assertEqual(
            spec_key.to_legacy_batch_descriptor(),
            BatchDescriptor(num_tokens=6),
        )

    def test_fixed_spec_keeps_request_and_token_capacity_distinct(self):
        key = StagedSFAGraphKey.fixed_spec(4, 3)
        self.assertEqual(key.request_capacity, 4)
        self.assertEqual(key.token_capacity, 12)
        self.assertEqual(key.max_query_len, 3)
        self.assertEqual(key.query_profile, StagedSFAQueryProfile.SPEC_FIXED)

    def test_mtp_fia_padding_uses_request_not_token_capacity(self):
        key = StagedSFAGraphKey.fixed_spec(1, 2)
        batch_desc = BatchDescriptor(num_tokens=2)

        self.assertEqual(
            NPUModelRunner._fia_request_capacity(key, batch_desc),
            1,
        )

    def test_fixed_spec_rejects_q1_width(self):
        with self.assertRaises(ValueError):
            StagedSFAGraphKey.fixed_spec(4, 1)


class TestStagedSFADummyBatch(unittest.TestCase):
    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.vllm_config = object()
        runner.speculative_config = None
        runner.decode_threshold = 1
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
                "resident_short_prompt": {
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
                        (
                            StagedSFARouteAction.FATAL
                            if name == "short_frontier"
                            else StagedSFARouteAction.STAGED
                            if name == "resident_short_prompt"
                            else StagedSFARouteAction.SAFE_NATIVE
                        ),
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

    def test_fixed_mtp_query_starts_preserve_request_rows(self):
        np.testing.assert_array_equal(
            NPUModelRunner._staged_sfa_query_start_locs(
                3,
                query_width=2,
                dtype=np.dtype(np.int32),
            ),
            np.array([0, 2, 4, 6], dtype=np.int32),
        )

    def test_fixed_width_mtp_dummy_batches_are_staged(self):
        runner = self._build_runner()
        runner.speculative_config = SimpleNamespace(
            method="mtp",
            num_speculative_tokens=1,
        )
        runner.decode_threshold = 2
        runner.attn_state = AscendAttentionState.SpecDecoding
        kwargs = self._eligibility_kwargs()
        kwargs.update(
            num_reqs=2,
            num_scheduled_tokens=np.full(2, 2, dtype=np.int32),
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
            self.assertEqual(runner._staged_sfa_dummy_batch_size(**kwargs), 4)

    def test_fixed_width_mtp_live_route_uses_request_capacity(self):
        runner = self._build_runner()
        runner.speculative_config = SimpleNamespace(
            method="mtp",
            num_speculative_tokens=1,
        )
        runner.decode_threshold = 2
        runner.attn_state = AscendAttentionState.SpecDecoding
        request_ids = ["req-0", "req-1"]
        local = runner._staged_sfa_local_route(
            num_tokens_unpadded=4,
            num_reqs=2,
            num_scheduled_tokens=np.full(2, 2, dtype=np.int32),
            prompt_lens=np.full(2, 4096, dtype=np.int32),
            index_topk=2048,
            has_cascade_attention=False,
            request_ids=request_ids,
            kv_connector_metadata=SimpleNamespace(
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
        )
        route = runner._staged_sfa_live_route(
            local_route=local,
            dp_route_action=StagedSFARouteAction.STAGED,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=4),
            num_tokens_unpadded=4,
            num_tokens_padded=4,
            num_reqs=2,
            should_ubatch=False,
        )
        self.assertEqual(route.action, StagedSFARouteAction.STAGED)
        self.assertEqual(route.graph_key, StagedSFAGraphKey.fixed_spec(2, 2))

    def test_zero_frontier_uses_graph_with_all_kv_resident(self):
        runner = self._build_runner()
        request_ids = ["req-0"]
        local = runner._staged_sfa_local_route(
            num_tokens_unpadded=1,
            num_reqs=1,
            num_scheduled_tokens=np.ones(1, dtype=np.int32),
            prompt_lens=np.array([257], dtype=np.int32),
            index_topk=2048,
            has_cascade_attention=False,
            request_ids=request_ids,
            kv_connector_metadata=SimpleNamespace(
                requests=[
                    SimpleNamespace(
                        req_id="req-0",
                        is_sparse_decode=True,
                        load_spec=SimpleNamespace(
                            can_load=True,
                            lmcache_cached_tokens=257,
                            dsa_committed_end=0,
                        ),
                    )
                ]
            ),
        )
        self.assertEqual(local.action, StagedSFARouteAction.STAGED)
        self.assertEqual(local.frontiers, (0,))

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

    def test_two_group_mtp_dummy_rows_get_one_slot_per_token(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=8,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )
        runner._prepare_staged_sfa_dummy_block_tables(
            batch_size=2,
            positions=np.array([8, 9, 10, 11], dtype=np.int64),
        )
        expected_slots = (
            np.array([0, 1, 6, 7], dtype=np.int64),
            np.array([0, 1, 10, 11], dtype=np.int64),
        )
        for group_index, block_table in enumerate(
            runner.input_batch.block_table.block_tables
        ):
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
                return_value=("only fixed-width MTP speculative decoding is supported",),
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
        draft_seal = MagicMock(return_value=2)
        runner.drafter = SimpleNamespace(
            use_staged_mtp_draft_graph=False,
            seal_staged_mtp_draft_graphs=draft_seal,
        )
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
        draft_seal.assert_not_called()

    def test_collect_staged_sfa_impls_excludes_mtp_draft_layer(self):
        runner = self._build_runner()
        runner.vllm_config.model_config.hf_text_config.num_hidden_layers = 78
        target_impl = SimpleNamespace(enable_staged_sfa_graph=True)
        draft_impl = SimpleNamespace(enable_staged_sfa_graph=True)
        target_layer = SimpleNamespace(
            impl=target_impl,
            layer_name="model.layers.77.self_attn.attn",
        )
        draft_layer = SimpleNamespace(
            impl=draft_impl,
            layer_name="model.layers.78.self_attn.attn",
        )

        with patch.object(
            model_runner_module,
            "get_layers_from_vllm_config",
            return_value={
                target_layer.layer_name: target_layer,
                draft_layer.layer_name: draft_layer,
            },
        ):
            collected = runner._collect_staged_sfa_impls()

        self.assertEqual(
            collected,
            ((target_layer.layer_name, target_impl),),
        )

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
