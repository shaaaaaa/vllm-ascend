import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import BatchDescriptor

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import (
    STAGED_SFA_SINGLETON_GRAPH_KEY,
    StagedSFAGraphKey,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

import vllm_ascend.attention.sfa_v1 as sfa_v1
import vllm_ascend.attention.utils as attention_utils
from vllm_ascend.attention.sfa_v1 import (
    AscendSFABackend,
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    _update_dsa_split_boundary_in_place,
)
from vllm_ascend.utils import (
    StagedSFARouteAction,
    StagedSFARouteDecision,
    StagedSFARouteReason,
    enable_dsa_cp,
)


def test_sfa_metadata_declares_cached_decode_split_boundary() -> None:
    field = AscendSFAMetadata.__dataclass_fields__["decode_split_boundary"]
    assert field.default is None


def test_dsa_topk_resolver_accepts_one_consistent_value() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(index_topk=2048),
        hf_text_config=SimpleNamespace(
            index_topk=2048,
            topk_tokens=2048,
        ),
    )

    assert (
        attention_utils.resolve_dsa_index_topk(
            model_config,
            runtime_topk=2048,
        )
        == 2048
    )


def test_dsa_topk_resolver_rejects_disagreeing_sources() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(index_topk=2048),
        hf_text_config=SimpleNamespace(
            index_topk=2048,
            topk_tokens=4096,
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"one identical top-k value.*index_topk=2048.*topk_tokens=4096",
    ):
        attention_utils.resolve_dsa_index_topk(
            model_config,
            runtime_topk=2048,
        )


def test_sparse_boundary_updates_preallocated_storage_in_place():
    boundary_cpu = torch.tensor([9, 9, 19, 0], dtype=torch.int32)
    boundary = torch.empty(4, dtype=torch.int32)
    boundary.copy_(boundary_cpu)
    metadata = SimpleNamespace(
        split_boundary=boundary,
        decode_split_boundary_cpu=boundary_cpu.numpy(),
        decode_split_boundary_cpu_tensor=boundary_cpu,
        decode_req_indices_cpu=torch.tensor([0, 0, 1, -1], dtype=torch.int32).numpy(),
        seq_lens_cpu=torch.tensor([513, 770], dtype=torch.int32),
        num_decode_tokens=3,
        decode_split_boundary=None,
        decode_scratch_base_cpu=None,
        decode_scratch_capacity=256,
        block_table=torch.zeros((2, 4), dtype=torch.int32),
    )
    address = metadata.split_boundary.data_ptr()

    with (
        patch.object(
            sfa_v1.torch,
            "tensor",
            side_effect=AssertionError("unexpected torch.tensor"),
        ),
        patch.object(
            sfa_v1.torch,
            "arange",
            side_effect=AssertionError("unexpected torch.arange"),
        ),
        patch.object(
            sfa_v1.torch.nn.functional,
            "pad",
            side_effect=AssertionError("unexpected pad"),
        ),
    ):
        actual = _update_dsa_split_boundary_in_place(
            metadata,
            cached_tokens=[512, 768],
            decode_window_size=256,
            index_topk=128,
            block_size=256,
        )

    assert actual.data_ptr() == address
    assert metadata.decode_split_boundary.data_ptr() == address
    assert actual.tolist() == [512, 512, 768, 0]


def test_sparse_boundary_short_frontier_preserves_zero_pad_semantics():
    boundary_cpu = torch.tensor([9, 19], dtype=torch.int32)
    boundary = boundary_cpu.clone()
    metadata = SimpleNamespace(
        split_boundary=boundary,
        decode_split_boundary_cpu=boundary_cpu.numpy(),
        decode_split_boundary_cpu_tensor=boundary_cpu,
        decode_req_indices_cpu=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
        num_decode_tokens=2,
        decode_split_boundary=None,
        decode_scratch_base_cpu=None,
        decode_scratch_capacity=4,
        block_table=torch.zeros((2, 1), dtype=torch.int32),
    )

    actual = _update_dsa_split_boundary_in_place(
        metadata,
        cached_tokens=[8],
        decode_window_size=0,
        index_topk=4,
        block_size=32,
    )

    assert actual.tolist() == [8, 0]


def test_sparse_boundary_prefers_explicit_committed_end():
    from vllm_ascend.attention import utils as attention_utils

    metadata = SimpleNamespace(
        requests=[
            SimpleNamespace(
                req_id="resident",
                is_sparse_decode=True,
                load_spec=SimpleNamespace(
                    can_load=True,
                    lmcache_cached_tokens=3072,
                    dsa_committed_end=0,
                ),
            ),
            SimpleNamespace(
                req_id="offloaded",
                is_sparse_decode=True,
                load_spec=SimpleNamespace(
                    can_load=True,
                    lmcache_cached_tokens=8192,
                    dsa_committed_end=8192,
                ),
            ),
        ]
    )
    connector = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    with (
        patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
    ):
        assert attention_utils.get_lmcache_sparse_cached_tokens(["resident", "offloaded"]) == [0, 8192]


def _staged_route(frontiers=(4096,)):
    return StagedSFARouteDecision(
        StagedSFARouteAction.STAGED,
        StagedSFARouteReason.ELIGIBLE,
        STAGED_SFA_SINGLETON_GRAPH_KEY,
        frontiers,
    )


class TestLMCacheSparseWaitSync(TestBase):
    def setUp(self):
        self.original_once_done = sfa_v1._lmcache_sparse_wait_sync_once_done
        sfa_v1._lmcache_sparse_wait_sync_once_done = False

    def tearDown(self):
        sfa_v1._lmcache_sparse_wait_sync_once_done = self.original_once_done

    def test_once_mode_synchronizes_only_first_sparse_wait(self):
        stream = MagicMock()
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", True),
            patch.object(
                sfa_v1.torch.npu,
                "current_stream",
                return_value=stream,
            ) as current_stream,
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_called_once_with()
        stream.synchronize.assert_called_once_with()
        self.assertTrue(sfa_v1._lmcache_sparse_wait_sync_once_done)

    def test_completed_mode_does_not_touch_npu_stream(self):
        sfa_v1._lmcache_sparse_wait_sync_once_done = True
        with patch.object(sfa_v1.torch.npu, "current_stream") as current_stream:
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_not_called()

    def test_disabled_mode_does_not_synchronize(self):
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", False),
            patch.object(sfa_v1.torch.npu, "current_stream") as current_stream,
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_not_called()
        self.assertFalse(sfa_v1._lmcache_sparse_wait_sync_once_done)

    def test_sync_compute_stream_skips_when_npu_unavailable(self):
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", True),
            patch.object(sfa_v1.torch, "npu", None),
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        self.assertFalse(sfa_v1._lmcache_sparse_wait_sync_once_done)


class TestDSASparsePadding(TestBase):
    def test_trailing_graph_padding_is_zeroed_in_place(self):
        topk = torch.arange(4 * 64, dtype=torch.int32).reshape(4, 1, 64)
        original_actual = topk[:2].clone()
        input_ptr = topk.data_ptr()

        result, result_2d = sfa_v1._dsa_mask_padding_sparse_rows(
            topk,
            torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        )

        self.assertEqual(result.data_ptr(), input_ptr)
        self.assertEqual(result_2d.data_ptr(), input_ptr)
        self.assertTrue(torch.equal(result[:2], original_actual))
        self.assertEqual(torch.count_nonzero(result[2:]).item(), 0)


class TestLMCacheSparseFrontier(TestBase):
    def test_invalid_or_duplicate_request_identity_fails_closed(self):
        sparse = SimpleNamespace(
            req_id="req-0",
            is_sparse_decode=True,
            load_spec=SimpleNamespace(
                can_load=True,
                lmcache_cached_tokens=128,
            ),
        )
        metadata = SimpleNamespace(requests=[sparse])
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0", "req-0"],
            ),
            (StagedSFARouteReason.INVALID_REQUEST_IDS, ()),
        )

        metadata.requests.append(sparse)
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0"],
            ),
            (StagedSFARouteReason.DUPLICATE_SPARSE_LOAD, ()),
        )

    def test_missing_active_request_frontier_fails_closed(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="req-0",
                    is_sparse_decode=True,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=128,
                    ),
                )
            ]
        )
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0", "req-1"],
            ),
            (StagedSFARouteReason.MISSING_CONNECTOR_METADATA, ()),
        )

    def test_frontiers_preserve_native_request_order(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="req-1",
                    is_sparse_decode=True,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=256,
                    ),
                ),
                SimpleNamespace(
                    req_id="req-0",
                    is_sparse_decode=True,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=128,
                    ),
                ),
            ]
        )
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["req-0", "req-1"],
            ),
            (StagedSFARouteReason.ELIGIBLE, (128, 256)),
        )

    def test_dense_prefix_hit_is_not_a_sparse_graph_step(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="req-0",
                    is_sparse_decode=False,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=18879,
                    ),
                )
            ]
        )

        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(metadata, ["req-0"]),
            (StagedSFARouteReason.DENSE_PREFIX_HIT, ()),
        )
        metadata.requests[0].is_sparse_decode = True
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(metadata, ["req-0"]),
            (StagedSFARouteReason.ELIGIBLE, (18879,)),
        )
        metadata.requests[0].load_spec.can_load = False
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(metadata, ["req-0"]),
            (StagedSFARouteReason.ELIGIBLE, (0,)),
        )

    def test_sparse_route_prefers_committed_boundary_over_load_length(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="resident",
                    is_sparse_decode=True,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=1024,
                        dsa_committed_end=0,
                    ),
                ),
                SimpleNamespace(
                    req_id="offloaded",
                    is_sparse_decode=True,
                    load_spec=SimpleNamespace(
                        can_load=True,
                        lmcache_cached_tokens=8192,
                        dsa_committed_end=8192,
                    ),
                ),
            ]
        )
        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["resident", "offloaded"],
            ),
            (StagedSFARouteReason.ELIGIBLE, (0, 8192)),
        )

    def test_mixed_load_requires_every_row_to_be_loadable(self):
        metadata = SimpleNamespace(
            requests=[
                SimpleNamespace(
                    req_id="dense",
                    is_sparse_decode=False,
                    load_spec=SimpleNamespace(can_load=True),
                ),
                SimpleNamespace(
                    req_id="sparse",
                    is_sparse_decode=True,
                    load_spec=SimpleNamespace(can_load=False),
                ),
            ]
        )

        self.assertEqual(
            attention_utils.staged_sfa_metadata_sparse_load(
                metadata,
                ["dense", "sparse"],
            ),
            (StagedSFARouteReason.MIXED_CONNECTOR_LOAD, ()),
        )

    def test_sparse_wait_forwards_existing_payload_event(self):
        connector = MagicMock()
        event = object()
        selected = torch.ones(1, 4, dtype=torch.int32)
        target_slots = torch.arange(4).view(1, 4)
        with (
            patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
            patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
            patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
            patch.object(attention_utils, "_dsa_lmcache_log_layer", return_value=False),
        ):
            attention_utils.wait_for_kv_layer_from_connector(
                "layer-0",
                selected_tokens=selected,
                request_ids=["req-0"],
                target_slot_mapping=target_slots,
                payload_event=event,
            )

        connector.wait_for_layer_load.assert_called_once_with(
            "layer-0",
            selected,
            None,
            ["req-0"],
            target_slot_mapping=target_slots,
            payload_event=event,
        )

    def test_scratch_reservation_and_table_capacity_fail_closed(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "reservation is too small",
        ):
            sfa_v1._validate_dsa_scratch_capacity(
                [8, 8],
                [0, 0],
                [0, 4],
                4,
                scratch_capacity=7,
            )

        # Boundary zero means all KV stays resident. A nonzero boundary must
        # begin after the complete fixed-width request scratch reservation.
        sfa_v1._validate_dsa_scratch_capacity(
            [0, 0],
            [0, 0],
            None,
            4,
            scratch_capacity=8,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "must use the same sparse boundary",
        ):
            sfa_v1._validate_dsa_scratch_capacity(
                [0, 8],
                [0, 0],
                None,
                4,
                scratch_capacity=8,
            )
        with self.assertRaisesRegex(RuntimeError, "alias live KV"):
            sfa_v1._validate_dsa_scratch_capacity(
                [4, 4],
                [0, 0],
                None,
                4,
                scratch_capacity=8,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "outside the request sequence",
        ):
            sfa_v1._validate_dsa_scratch_capacity(
                [17],
                [0],
                None,
                4,
                scratch_capacity=8,
                seq_lens=[16],
                block_table_width=2,
                block_size=8,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "exceeds its block-table capacity",
        ):
            sfa_v1._validate_dsa_scratch_capacity(
                [0],
                [0],
                None,
                4,
                scratch_capacity=8,
                seq_lens=[17],
                block_table_width=2,
                block_size=8,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "block-table capacity",
        ):
            sfa_v1._dsa_build_target_slot_mapping(
                torch.tensor([[0]], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([4], dtype=torch.int64),
                4,
                4,
                scratch_capacity=8,
            )


class TestStagedSFAGraphPoc(TestBase):
    def setUp(self):
        super().setUp()
        capture_sizes = patch.object(
            sfa_v1,
            "staged_sfa_graph_capture_sizes",
            return_value=(1, 4),
        )
        connector_support = patch.object(
            sfa_v1,
            "staged_sfa_connector_supports_sparse_load",
            return_value=True,
        )
        capture_sizes.start()
        connector_support.start()
        self.addCleanup(capture_sizes.stop)
        self.addCleanup(connector_support.stop)

    @staticmethod
    def _make_eligible_impl():
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.dsa_shrink_latent = 2
        impl.num_kv_heads = 1
        impl.kv_lora_rank = 2
        impl.qk_rope_head_dim = 2
        impl.head_dim = 2
        impl.index_topk = 4
        impl.decode_threshold = 1
        impl.enable_mlapo = False
        impl.enable_dsa_cp = False
        impl.enable_dsa_cp_with_o_proj_tp = False
        impl.use_sparse_c8_indexer = False
        impl.dsa_offload_free_paged = False
        impl.q_lora_rank = 4
        impl.fused_qkv_a_proj = MagicMock()
        impl.q_a_layernorm = MagicMock()
        impl.vllm_config = MagicMock()
        impl.vllm_config.cache_config.block_size = 128
        impl.block_size = 128
        impl.vllm_config.speculative_config = None
        impl.vllm_config.lora_config = None
        impl._staged_sfa_capture_state = sfa_v1._StagedSFACaptureState()
        impl._staged_sfa_graph_capture_sizes = (1, 4)
        impl._staged_sfa_bridge_buffers = None
        impl._dsa_scratch_resident_tokens = torch.full(
            (4, 4),
            -1,
            dtype=torch.int32,
        )
        impl._dsa_scratch_resident_generations = torch.full(
            (4, 8),
            -1,
            dtype=torch.int64,
        )
        return impl

    @staticmethod
    def _make_eligible_kv_cache(
        *,
        dtype=torch.bfloat16,
        device="cpu",
        block_size=128,
        num_blocks=2,
    ):
        return (
            torch.empty(num_blocks, block_size, 1, 2, dtype=dtype, device=device),
            torch.empty(num_blocks, block_size, 1, 2, dtype=dtype, device=device),
            torch.empty(num_blocks, block_size, 1, 2, dtype=dtype, device=device),
        )

    @staticmethod
    def _make_pre_outputs(token_rows: int = 1, request_rows: int = 1):
        return (
            torch.empty(token_rows, 2, 2),
            torch.empty(token_rows, 2, 2),
            torch.empty(token_rows, 1, 4, dtype=torch.int32),
            torch.empty(request_rows, 4, dtype=torch.int32),
            torch.empty(request_rows, dtype=torch.int32),
            torch.empty(request_rows, 4, dtype=torch.long),
        )

    @staticmethod
    def _make_decode_metadata(batch_size: int = 1):
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.num_input_tokens = batch_size
        metadata.num_actual_tokens = batch_size
        metadata.num_decode_tokens = batch_size
        metadata.cos = torch.ones(batch_size, 2)
        metadata.sin = torch.zeros(batch_size, 2)
        metadata.slot_mapping = torch.arange(batch_size)
        metadata.indexer_slot_mapping = torch.arange(batch_size)
        metadata.cum_query_lens = torch.arange(1, batch_size + 1)
        metadata.seq_lens = torch.full((batch_size,), 9)
        metadata.seq_lens_cpu = torch.full((batch_size,), 9)
        metadata.block_table = torch.arange(
            batch_size * 32
        ).view(batch_size, 32)
        metadata.indexer_block_table = torch.arange(
            batch_size * 32
        ).view(
            batch_size,
            32,
        )
        metadata.prompt_lens = torch.full(
            (batch_size,),
            8,
            dtype=torch.int32,
        )
        metadata.prompt_lens_cpu_rows = [8] * batch_size
        metadata.decode_req_indices = torch.arange(
            batch_size,
            dtype=torch.int32,
        )
        metadata.decode_req_indices_cpu = list(range(batch_size))
        metadata.decode_req_indices_compact_cpu = np.arange(
            batch_size,
            dtype=np.int64,
        )
        metadata.need_sparse_lmcache_payload = True
        metadata.decode_valid_rows_all = True
        metadata.decode_valid_row_indices = torch.arange(
            batch_size,
            dtype=torch.int32,
        )
        metadata.decode_scratch_base = torch.zeros(
            batch_size,
            dtype=torch.int32,
        )
        metadata.decode_scratch_base_compact = None
        metadata.decode_scratch_base_cpu = [0] * batch_size
        metadata.decode_scratch_capacity = 4
        metadata.decode_selected_tokens = torch.empty(
            batch_size, 4, dtype=torch.int32
        )
        metadata.decode_selected_counts = torch.empty(
            batch_size, 16, dtype=torch.int32
        )
        metadata.decode_target_slot_mapping = torch.empty(
            batch_size, 4, dtype=torch.long
        )
        metadata.decode_scratch_state_indices = torch.arange(
            batch_size,
            dtype=torch.int32,
        )
        metadata.decode_scratch_request_generations = torch.arange(
            1,
            batch_size + 1,
            dtype=torch.int64,
        )
        metadata.decode_request_ids_compact = [f"req-{row}" for row in range(batch_size)]
        metadata.req_ids = list(metadata.decode_request_ids_compact)
        metadata.decode_remap_boundary = torch.empty(
            batch_size,
            dtype=torch.int32,
        )
        metadata.decode_remap_boundary_ready = False
        return metadata

    def test_cross_layer_pre_uses_native_path_without_authorized_key(self):
        impl = self._make_eligible_impl()
        impl.local_num_heads = 2
        impl.forward = MagicMock()
        impl._cross_layer_kv_cache = MagicMock(return_value=(self._make_eligible_kv_cache(), "index-0", True))
        context = SimpleNamespace(staged_sfa_graph_key=None)
        hidden_states = torch.empty(16, 4)
        output = torch.empty_like(hidden_states)

        with patch.object(sfa_v1, "get_forward_context", return_value=context):
            outputs = impl.cross_layer_graph_pre(
                "layer-0",
                hidden_states,
                self._make_eligible_kv_cache(),
                self._make_decode_metadata(),
                False,
                output,
            )

        impl.forward.assert_called_once()
        self.assertEqual(
            [tuple(tensor.shape[:1]) for tensor in outputs],
            [(4,)] * 6,
        )
        self.assertTrue(all(tensor.is_contiguous() for tensor in outputs))

    def test_cross_layer_pre_fails_if_authorized_key_becomes_ineligible(self):
        impl = self._make_eligible_impl()
        impl._cross_layer_kv_cache = MagicMock(return_value=(self._make_eligible_kv_cache(), "index-0", True))
        impl._cross_layer_ineligible_reason = MagicMock(return_value="changed metadata")
        context = SimpleNamespace(
            staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
            staged_sfa_route=_staged_route(),
        )

        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            self.assertRaisesRegex(RuntimeError, "changed metadata"),
        ):
            impl.cross_layer_graph_pre(
                "layer-0",
                torch.empty(1, 4),
                self._make_eligible_kv_cache(),
                self._make_decode_metadata(),
                False,
                torch.empty(1, 4),
            )

    def test_cross_layer_capture_reuses_eager_boundary_storage(self):
        impl = self._make_eligible_impl()
        kv_cache = self._make_eligible_kv_cache()
        impl._staged_sfa_capture_state.initialized_cache_capacity = 1
        impl._staged_sfa_capture_state.producer_event = MagicMock()
        impl._cross_layer_kv_cache = MagicMock(return_value=(kv_cache, "index-0", True))
        impl._cross_layer_ineligible_reason = MagicMock(return_value=None)
        impl._cross_layer_pre_compute = MagicMock(
            return_value=self._make_pre_outputs()
        )
        eager_metadata = self._make_decode_metadata()
        capture_metadata = self._make_decode_metadata()
        capture_metadata.decode_remap_boundary = eager_metadata.decode_remap_boundary
        contexts = (
            SimpleNamespace(
                staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
                staged_sfa_route=_staged_route(),
                staged_sfa_graph_dummy_run=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ),
            SimpleNamespace(
                staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
                staged_sfa_route=_staged_route(),
                staged_sfa_graph_dummy_run=True,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            ),
        )

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                side_effect=contexts,
            ),
            patch.object(
                sfa_v1,
                "_prepare_sfa_remap_boundary",
                return_value=eager_metadata.decode_remap_boundary,
            ) as prepare_boundary,
        ):
            outputs = [
                impl.cross_layer_graph_pre(
                    "layer-0",
                    torch.empty(1, 4),
                    kv_cache,
                    metadata,
                    False,
                    torch.empty(1, 4),
                )
                for metadata in (eager_metadata, capture_metadata)
            ]

        prepare_boundary.assert_called_once_with(
            eager_metadata,
            eager_metadata.req_ids,
            is_dummy_run=True,
            index_topk=impl.index_topk,
            block_size=impl.block_size,
            cached_tokens=(4096,),
        )
        self.assertIs(
            impl._staged_sfa_capture_state.remap_boundary,
            eager_metadata.decode_remap_boundary,
        )
        self.assertIs(
            impl._cross_layer_pre_compute.call_args_list[1].args[11],
            eager_metadata.decode_remap_boundary,
        )
        self.assertEqual(
            impl._staged_sfa_capture_state.bindings.keys(),
            {STAGED_SFA_SINGLETON_GRAPH_KEY},
        )
        binding = impl._staged_sfa_capture_state.bindings[
            STAGED_SFA_SINGLETON_GRAPH_KEY
        ]
        self.assertEqual(
            binding.request_state_indices.address,
            capture_metadata.decode_scratch_state_indices.data_ptr(),
        )
        self.assertEqual(
            binding.request_generations.address,
            capture_metadata.decode_scratch_request_generations.data_ptr(),
        )
        self.assertTrue(all(tensor.shape[0] == 4 for result in outputs for tensor in result))

    def test_cross_layer_replay_rejects_scratch_state_pointer_drift(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        kv_cache = self._make_eligible_kv_cache()
        state = impl._staged_sfa_capture_state
        state.producer_event = MagicMock()
        state.remap_boundary = metadata.decode_remap_boundary
        state.runtime = ("layer-0", kv_cache, "index-0", True)
        state.register(
            STAGED_SFA_SINGLETON_GRAPH_KEY,
            self._make_pre_outputs(),
            kv_cache,
            metadata.decode_scratch_state_indices,
            metadata.decode_scratch_request_generations,
            impl._dsa_scratch_resident_tokens,
            impl._dsa_scratch_resident_generations,
        )
        metadata.decode_scratch_state_indices = (
            metadata.decode_scratch_state_indices.clone()
        )
        context = SimpleNamespace(
            staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
            staged_sfa_route=_staged_route(),
            staged_sfa_graph_dummy_run=False,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        )

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=context,
            ),
            patch.object(
                impl,
                "_cross_layer_kv_cache",
                return_value=(kv_cache, "index-0", True),
            ),
            patch.object(
                impl,
                "_cross_layer_ineligible_reason",
                return_value=None,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "scratch-state input binding changed",
            ),
        ):
            impl.cross_layer_graph_pre(
                "layer-0",
                torch.empty(1, 4),
                kv_cache,
                metadata,
                False,
                torch.empty(1, 4),
            )

    def test_selected_counts_wait_view_uses_one_value_per_cacheline_row(
        self,
    ):
        padded_counts = torch.arange(
            48,
            dtype=torch.int32,
        ).view(3, 16)

        logical_counts = sfa_v1._dsa_selected_counts_for_wait(
            padded_counts,
            2,
        )

        self.assertEqual(logical_counts.shape, (2,))
        self.assertEqual(logical_counts.tolist(), [0, 16])
        self.assertEqual(padded_counts.shape, (3, 16))
        logical_input = torch.tensor([5, 7, 9], dtype=torch.int32)
        logical_view = sfa_v1._dsa_selected_counts_for_wait(logical_input, 2)
        self.assertEqual(logical_view.shape, (2,))
        self.assertEqual(logical_view.tolist(), [5, 7])
        with self.assertRaisesRegex(
            RuntimeError,
            "selected-count storage",
        ):
            sfa_v1._dsa_selected_counts_for_wait(
                torch.ones(3, 0, dtype=torch.int32),
                2,
            )

    def test_cross_layer_replay_rejects_scratch_state_layout_drift(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        state_indices_storage = torch.empty(2, dtype=torch.int32)
        metadata.decode_scratch_state_indices = state_indices_storage[:1]
        kv_cache = self._make_eligible_kv_cache()
        state = impl._staged_sfa_capture_state
        state.producer_event = MagicMock()
        state.remap_boundary = metadata.decode_remap_boundary
        state.runtime = ("layer-0", kv_cache, "index-0", True)
        state.register(
            STAGED_SFA_SINGLETON_GRAPH_KEY,
            self._make_pre_outputs(),
            kv_cache,
            metadata.decode_scratch_state_indices,
            metadata.decode_scratch_request_generations,
            impl._dsa_scratch_resident_tokens,
            impl._dsa_scratch_resident_generations,
        )
        metadata.decode_scratch_state_indices = (
            state_indices_storage.as_strided((1,), (2,))
        )
        context = SimpleNamespace(
            staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
            staged_sfa_route=_staged_route(),
            staged_sfa_graph_dummy_run=False,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        )

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=context,
            ),
            patch.object(
                impl,
                "_cross_layer_kv_cache",
                return_value=(kv_cache, "index-0", True),
            ),
            patch.object(
                impl,
                "_cross_layer_ineligible_reason",
                return_value=None,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "scratch-state input binding changed",
            ),
        ):
            impl.cross_layer_graph_pre(
                "layer-0",
                torch.empty(1, 4),
                kv_cache,
                metadata,
                False,
                torch.empty(1, 4),
            )

    def test_cross_layer_padding_uses_fixed_graph_rows(self):
        impl = self._make_eligible_impl()
        graph_key = StagedSFAGraphKey.exact_q1(4)
        metadata = self._make_decode_metadata(4)
        metadata.num_actual_tokens = metadata.num_decode_tokens = 1
        metadata.decode_req_indices[1:] = -1
        impl._staged_sfa_capture_state.producer_event = MagicMock()
        impl._cross_layer_kv_cache = MagicMock(
            return_value=(
                self._make_eligible_kv_cache(num_blocks=4),
                "index-0",
                True,
            )
        )
        impl._cross_layer_ineligible_reason = MagicMock(return_value=None)
        impl._cross_layer_pre_compute = MagicMock(
            return_value=self._make_pre_outputs(4, 4)
        )
        context = SimpleNamespace(
            staged_sfa_graph_key=graph_key,
            staged_sfa_route=StagedSFARouteDecision(
                StagedSFARouteAction.STAGED,
                StagedSFARouteReason.ELIGIBLE,
                graph_key,
                (4096,),
            ),
            staged_sfa_graph_dummy_run=False,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )

        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            patch.object(
                sfa_v1,
                "_prepare_sfa_remap_boundary",
                return_value=metadata.decode_remap_boundary,
            ),
        ):
            impl.cross_layer_graph_pre(
                "layer-0",
                torch.empty(4, 4),
                self._make_eligible_kv_cache(num_blocks=4),
                metadata,
                False,
                torch.empty(4, 4),
            )

        args = impl._cross_layer_pre_compute.call_args.args
        self.assertEqual(args[13].tolist(), [0, 1, 2, 3])

    def test_bridge_storage_is_preallocated_and_reused_for_q1(self):
        impl = self._make_eligible_impl()
        hidden_states = torch.empty(1, 4)
        outputs = self._make_pre_outputs()
        context = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.NONE
        )
        with patch.object(
            sfa_v1,
            "get_forward_context",
            return_value=context,
        ):
            first = impl._copy_to_staged_sfa_bridge(
                hidden_states,
                outputs,
            )
            first_addresses = tuple(tensor.data_ptr() for tensor in first)
            second = impl._copy_to_staged_sfa_bridge(
                hidden_states,
                outputs,
            )
        self.assertEqual(
            tuple(tensor.data_ptr() for tensor in second),
            first_addresses,
        )
        self.assertEqual(
            [tensor.shape[0] for tensor in second],
            [4, 4, 4, 4, 4, 4],
        )

    def test_bridge_storage_separates_mtp_token_and_request_capacity(self):
        impl = self._make_eligible_impl()
        impl.decode_threshold = 2
        impl._staged_sfa_graph_capture_sizes = (2, 8)
        hidden_states = torch.empty(4, 4)
        outputs = (
            torch.empty(4, 2, 2),
            torch.empty(4, 2, 2),
            torch.empty(4, 1, 4, dtype=torch.int32),
            torch.empty(2, 8, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, 8, dtype=torch.long),
        )
        with patch.object(
            sfa_v1,
            "get_forward_context",
            return_value=SimpleNamespace(
                cudagraph_runtime_mode=CUDAGraphMode.NONE
            ),
        ):
            bridge = impl._copy_to_staged_sfa_bridge(
                hidden_states,
                outputs,
            )
        self.assertEqual(
            [tensor.shape[0] for tensor in bridge],
            [8, 8, 8, 4, 4, 4],
        )
        self.assertEqual(tuple(bridge[3].shape), (4, 8))
        self.assertEqual(tuple(bridge[5].shape), (4, 8))

    def test_capture_state_seals_exact_keys(self):
        state = sfa_v1._StagedSFACaptureState(
            producer_event=object(),
            remap_boundary=torch.empty(1, dtype=torch.int32),
            runtime=("layer-0",),
        )
        key = STAGED_SFA_SINGLETON_GRAPH_KEY
        state.register(
            key,
            tuple(torch.empty(1) for _ in range(6)),
            self._make_eligible_kv_cache(),
            torch.empty(1, dtype=torch.int32),
            torch.empty(1, dtype=torch.int64),
            torch.empty((1, 4), dtype=torch.int32),
            torch.empty((1, 8), dtype=torch.int64),
        )
        state.seal((key,))

        with self.assertRaisesRegex(RuntimeError, "missing_keys=.*2"):
            state.seal((key, StagedSFAGraphKey.exact_q1(2)))

    def test_capture_state_rejects_binding_drift(self):
        state = sfa_v1._StagedSFACaptureState(
            producer_event=object(),
            remap_boundary=torch.empty(4, dtype=torch.int32),
            runtime=("layer-0",),
        )
        bridge = tuple(torch.empty(4) for _ in range(6))
        state_indices = torch.empty(1, dtype=torch.int32)
        request_generations = torch.empty(1, dtype=torch.int64)
        resident_tokens = torch.empty((1, 4), dtype=torch.int32)
        resident_generations = torch.empty((1, 8), dtype=torch.int64)
        state.register(
            STAGED_SFA_SINGLETON_GRAPH_KEY,
            bridge,
            self._make_eligible_kv_cache(),
            state_indices,
            request_generations,
            resident_tokens,
            resident_generations,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "bindings changed between graph keys",
        ):
            state.register(
                StagedSFAGraphKey.exact_q1(2),
                bridge,
                self._make_eligible_kv_cache(),
                state_indices,
                request_generations,
                resident_tokens,
                resident_generations,
            )

    def test_capture_state_allows_fixed_state_buffer_views_across_keys(self):
        state = sfa_v1._StagedSFACaptureState(
            producer_event=object(),
            remap_boundary=torch.empty(4, dtype=torch.int32),
            runtime=("layer-0",),
        )
        bridge = tuple(torch.empty(4) for _ in range(6))
        kv_cache = self._make_eligible_kv_cache()
        state_indices = torch.empty(4, dtype=torch.int32)
        request_generations = torch.empty(4, dtype=torch.int64)
        resident_tokens = torch.empty((4, 4), dtype=torch.int32)
        resident_generations = torch.empty((4, 8), dtype=torch.int64)

        state.register(
            STAGED_SFA_SINGLETON_GRAPH_KEY,
            bridge,
            kv_cache,
            state_indices[:1],
            request_generations[:1],
            resident_tokens,
            resident_generations,
        )
        state.register(
            StagedSFAGraphKey.exact_q1(2),
            bridge,
            kv_cache,
            state_indices[:2],
            request_generations[:2],
            resident_tokens,
            resident_generations,
        )

        self.assertEqual(
            set(state.bindings),
            {
                STAGED_SFA_SINGLETON_GRAPH_KEY,
                StagedSFAGraphKey.exact_q1(2),
            },
        )

    def test_capture_reset_discards_cached_index_tensor(self):
        impl = self._make_eligible_impl()
        old_state = impl._staged_sfa_capture_state
        impl._dsa_idx_cache_t = torch.empty(1)

        impl.reset_staged_sfa_capture()

        self.assertIsNot(impl._staged_sfa_capture_state, old_state)
        self.assertIsNone(impl._dsa_idx_cache_t)

    def test_dummy_cache_initialization_grows_with_capture_key(self):
        impl = self._make_eligible_impl()
        kv_cache = tuple(torch.ones_like(cache) for cache in self._make_eligible_kv_cache(num_blocks=4))
        impl._staged_sfa_capture_state.producer_event = MagicMock()
        impl._cross_layer_kv_cache = MagicMock(return_value=(kv_cache, "index-0", True))
        impl._cross_layer_ineligible_reason = MagicMock(return_value=None)
        impl._cross_layer_pre_compute = MagicMock(
            return_value=self._make_pre_outputs()
        )

        def run(batch_size: int) -> None:
            key = StagedSFAGraphKey.exact_q1(batch_size)
            metadata = self._make_decode_metadata(batch_size)
            context = SimpleNamespace(
                staged_sfa_graph_key=key,
                staged_sfa_route=StagedSFARouteDecision(
                    StagedSFARouteAction.STAGED,
                    StagedSFARouteReason.ELIGIBLE,
                    key,
                ),
                staged_sfa_graph_dummy_run=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with (
                patch.object(sfa_v1, "get_forward_context", return_value=context),
                patch.object(
                    sfa_v1,
                    "_prepare_sfa_remap_boundary",
                    return_value=metadata.decode_remap_boundary,
                ),
            ):
                impl.cross_layer_graph_pre(
                    "layer-0",
                    torch.empty(batch_size, 4),
                    kv_cache,
                    metadata,
                    False,
                    torch.empty(batch_size, 4),
                )

        run(2)
        self.assertTrue(all(torch.count_nonzero(cache[:2]) == 0 for cache in kv_cache))
        self.assertTrue(all(torch.count_nonzero(cache[2:]) > 0 for cache in kv_cache))
        run(4)
        self.assertTrue(all(torch.count_nonzero(cache) == 0 for cache in kv_cache))
        self.assertEqual(
            impl._staged_sfa_capture_state.initialized_cache_capacity,
            4,
        )

    def test_cross_layer_retrieve_prefetches_next_index(self):
        impl = self._make_eligible_impl()
        graph_key = StagedSFAGraphKey.exact_q1(4)
        impl._cross_layer_kv_cache = MagicMock(return_value=(self._make_eligible_kv_cache(), "index-0", True))
        impl._staged_sfa_capture_state.producer_event = object()
        impl._staged_sfa_capture_state.runtime = (None, None, None, True)
        metadata = self._make_decode_metadata()
        next_metadata = self._make_decode_metadata()
        context = SimpleNamespace(
            staged_sfa_graph_key=graph_key,
            staged_sfa_route=StagedSFARouteDecision(
                StagedSFARouteAction.STAGED,
                StagedSFARouteReason.ELIGIBLE,
                graph_key,
                (4096,),
            ),
            staged_sfa_graph_dummy_run=False,
            attn_metadata={"layer-1.attn": next_metadata},
        )
        waits = []
        with (
            patch.object(
                sfa_v1,
                "wait_for_kv_layer_from_connector",
                side_effect=lambda name, *args, **kwargs: waits.append(name),
            ) as wait_for_layer,
            patch.object(sfa_v1, "_sync_compute_stream_after_lmcache_sparse_wait"),
            patch.object(sfa_v1, "_prepare_sfa_remap_boundary") as prepare_boundary,
        ):
            impl.cross_layer_lmcache_retrieve(
                "layer-0",
                "layer-1.attn",
                torch.ones(4, 4, dtype=torch.int32),
                torch.arange(
                    64,
                    dtype=torch.int32,
                ).view(4, 16),
                torch.zeros(4, 4, dtype=torch.long),
                metadata,
                context,
            )

        self.assertEqual(waits, ["layer-0", "layer-1.indexer.k_cache"])
        self.assertIs(
            metadata.reshape_cache_event,
            impl._staged_sfa_capture_state.producer_event,
        )
        self.assertIs(
            wait_for_layer.call_args_list[0].kwargs["payload_event"],
            impl._staged_sfa_capture_state.producer_event,
        )
        self.assertEqual(
            wait_for_layer.call_args_list[0]
            .kwargs["selected_token_counts"]
            .tolist(),
            [0],
        )
        self.assertEqual(
            wait_for_layer.call_args_list[0]
            .kwargs["selected_token_counts"]
            .dim(),
            1,
        )
        prepare_boundary.assert_called_once_with(
            next_metadata,
            next_metadata.req_ids,
            is_dummy_run=False,
            index_topk=impl.index_topk,
            block_size=impl.block_size,
            cached_tokens=(4096,),
        )

    def test_cross_layer_dummy_retrieve_only_prepares_next_boundary(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        next_metadata = self._make_decode_metadata()
        context = SimpleNamespace(
            staged_sfa_graph_key=StagedSFAGraphKey.exact_q1(4),
            staged_sfa_graph_dummy_run=True,
            attn_metadata={"layer-1.attn": next_metadata},
        )
        with (
            patch.object(sfa_v1, "_prepare_sfa_remap_boundary") as prepare_boundary,
            patch.object(sfa_v1, "wait_for_kv_layer_from_connector") as wait_for_layer,
        ):
            impl.cross_layer_lmcache_retrieve(
                "layer-0",
                "layer-1.attn",
                torch.ones(4, 4, dtype=torch.int32),
                torch.ones(4, 16, dtype=torch.int32),
                torch.zeros(4, 4, dtype=torch.long),
                metadata,
                context,
            )

        prepare_boundary.assert_called_once_with(
            next_metadata,
            next_metadata.req_ids,
            is_dummy_run=True,
            index_topk=impl.index_topk,
            block_size=impl.block_size,
        )
        wait_for_layer.assert_not_called()

    def test_cross_layer_post_ignores_padded_bridge_rows(self):
        impl = self._make_eligible_impl()
        kv_cache = self._make_eligible_kv_cache()
        impl._cross_layer_kv_cache = MagicMock(return_value=(kv_cache, "index-0", True))
        impl._cross_layer_post_compute = MagicMock()
        context = SimpleNamespace(
            staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
        )

        with patch.object(sfa_v1, "get_forward_context", return_value=context):
            impl.cross_layer_graph_post(
                "layer-0",
                torch.empty(4, 2, 4),
                torch.empty(4, 2, 2),
                torch.empty(4, 1, 16, dtype=torch.int32),
                kv_cache,
                self._make_decode_metadata(),
                torch.empty(1, 4),
            )

        args = impl._cross_layer_post_compute.call_args.args
        self.assertEqual([tensor.shape[0] for tensor in args[:3]], [1] * 3)

    def test_cross_layer_bootstrap_prepares_boundary_before_index_wait(self):
        impl = self._make_eligible_impl()
        impl._staged_sfa_capture_state.runtime = (
            None,
            None,
            "layer-0.indexer.k_cache",
            True,
        )
        metadata = self._make_decode_metadata()
        context = SimpleNamespace(
            attn_metadata={"layer-0.attn": metadata},
            staged_sfa_route=_staged_route(),
            staged_sfa_graph_key=STAGED_SFA_SINGLETON_GRAPH_KEY,
        )
        order = []
        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            patch.object(
                sfa_v1,
                "_prepare_sfa_remap_boundary",
                side_effect=lambda *args, **kwargs: order.append("boundary"),
            ),
            patch.object(
                sfa_v1,
                "wait_for_kv_layer_from_connector",
                side_effect=lambda *args, **kwargs: order.append("index"),
            ) as wait_for_layer,
        ):
            impl.bootstrap_cross_layer("layer-0.attn")

        self.assertEqual(order, ["boundary", "index"])
        wait_for_layer.assert_called_once_with("layer-0.indexer.k_cache")

    def test_cross_layer_dummy_bootstrap_skips_index_wait(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        context = SimpleNamespace(
            attn_metadata={"layer-0.attn": metadata},
            staged_sfa_graph_dummy_run=True,
        )
        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            patch.object(sfa_v1, "_prepare_sfa_remap_boundary") as prepare_boundary,
            patch.object(sfa_v1, "wait_for_kv_layer_from_connector") as wait_for_layer,
        ):
            impl.bootstrap_cross_layer("layer-0.attn")

        prepare_boundary.assert_called_once_with(
            metadata,
            metadata.req_ids,
            is_dummy_run=True,
            index_topk=impl.index_topk,
            block_size=impl.block_size,
            cached_tokens=None,
        )
        wait_for_layer.assert_not_called()

    def test_remap_boundary_is_resolved_once_per_step(self):
        metadata = self._make_decode_metadata()
        metadata.prompt_lens_cpu_rows = [1000]
        metadata.seq_lens_cpu = torch.tensor([1025])
        original_address = metadata.decode_remap_boundary.data_ptr()

        with (
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=256,
            ),
            patch.object(sfa_v1, "get_lmcache_sparse_cached_tokens") as lookup,
        ):
            first = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
                index_topk=4,
                block_size=256,
                cached_tokens=(900,),
            )
            second = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
                index_topk=4,
                block_size=256,
                cached_tokens=(900,),
            )

        self.assertIs(first, second)
        self.assertEqual(first.data_ptr(), original_address)
        self.assertEqual(first.tolist(), [900])
        lookup.assert_not_called()

    def test_remap_boundary_ignores_dp_padding_rows(self):
        metadata = self._make_decode_metadata(batch_size=2)
        metadata.prompt_lens_cpu_rows = [1000, 0]
        metadata.decode_req_indices_cpu = [0, -1]
        metadata.seq_lens_cpu = torch.tensor([1025, 0])

        with patch.object(
            sfa_v1,
            "_decode_window_save_window_size",
            return_value=256,
        ):
            boundary = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
                index_topk=4,
                block_size=256,
                cached_tokens=(900,),
            )

        self.assertEqual(boundary.tolist(), [900, 0])

    def test_dummy_remap_boundary_ignores_empty_route_frontiers(self):
        metadata = self._make_decode_metadata()

        boundary = sfa_v1._prepare_sfa_remap_boundary(
            metadata,
            ["req-0"],
            is_dummy_run=True,
            index_topk=4,
            block_size=256,
            cached_tokens=(),
        )

        self.assertEqual(boundary.tolist(), [8])

    def test_native_remap_boundary_retains_connector_frontier_lookup(self):
        metadata = self._make_decode_metadata()
        metadata.prompt_lens_cpu_rows = [1000]
        metadata.seq_lens_cpu = torch.tensor([1025])

        with (
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=256,
            ),
            patch.object(
                sfa_v1,
                "get_lmcache_sparse_cached_tokens",
                return_value=[900],
            ) as lookup,
        ):
            boundary = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
                index_topk=4,
                block_size=256,
            )

        self.assertEqual(boundary.tolist(), [900])
        lookup.assert_called_once_with(["req-0"])

    def test_native_mixed_remap_looks_up_decode_requests_only(self):
        metadata = self._make_decode_metadata(batch_size=2)
        metadata.prompt_lens_cpu_rows = [100, 0]
        metadata.decode_req_indices_cpu = [0, -1]
        metadata.seq_lens_cpu = torch.tensor([110, 6400])

        with (
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=0,
            ),
            patch.object(
                sfa_v1,
                "get_lmcache_sparse_cached_tokens",
                return_value=[90],
            ) as lookup,
        ):
            boundary = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["decode-req", "prefill-req"],
                is_dummy_run=False,
                index_topk=4,
                block_size=256,
            )

        self.assertEqual(boundary.tolist(), [90, 0])
        lookup.assert_called_once_with(["decode-req"])

    def test_native_frontiers_are_aligned_after_filtering_prefill(self):
        metadata = self._make_decode_metadata(batch_size=2)
        metadata.decode_req_indices_cpu = [0, -1]

        with patch.object(
            sfa_v1,
            "get_lmcache_sparse_cached_tokens",
            return_value=[90],
        ) as lookup:
            cached_tokens = sfa_v1._resolve_sparse_cached_tokens_by_request(
                metadata,
                ["decode-req", "prefill-req"],
            )

        self.assertEqual(cached_tokens, [90, 0])
        lookup.assert_called_once_with(["decode-req"])

    def test_remap_boundary_uses_unique_request_ids_for_mtp_rows(self):
        metadata = self._make_decode_metadata()
        metadata.prompt_lens_cpu_rows = [100, 100, 200, 200]
        metadata.decode_req_indices_cpu = [0, 0, 1, 1]
        metadata.seq_lens_cpu = torch.tensor([110, 210])
        metadata.decode_scratch_base_cpu = [0, 4, 0, 4]
        metadata.decode_scratch_capacity = 8
        metadata.decode_remap_boundary = torch.empty(4, dtype=torch.int32)
        metadata.decode_remap_boundary_ready = False

        with patch.object(
            sfa_v1,
            "_decode_window_save_window_size",
            return_value=0,
        ):
            boundary = sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0", "req-1"],
                is_dummy_run=False,
                index_topk=4,
                block_size=256,
                cached_tokens=(90, 180),
            )

        self.assertEqual(boundary.tolist(), [90, 90, 180, 180])

    def test_remap_boundary_rejects_scratch_live_alias(self):
        metadata = self._make_decode_metadata()
        metadata.prompt_lens_cpu_rows = [100, 100]
        metadata.decode_req_indices_cpu = [0, 0]
        metadata.seq_lens_cpu = torch.tensor([110])
        metadata.decode_scratch_base_cpu = [0, 4]
        metadata.decode_scratch_capacity = 8
        metadata.decode_remap_boundary = torch.empty(2, dtype=torch.int32)
        metadata.decode_remap_boundary_ready = False

        with (
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=0,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "would alias live KV positions",
            ),
        ):
            sfa_v1._prepare_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
                index_topk=4,
                block_size=256,
                cached_tokens=(7,),
            )

        self.assertFalse(metadata.decode_remap_boundary_ready)

    def test_sparse_lmcache_payload_preserves_duplicate_request_rows(self):
        metadata = self._make_decode_metadata()
        selected = torch.arange(12, dtype=torch.int32).view(3, 4)
        targets = torch.arange(12, dtype=torch.int64).view(3, 4) + 32
        metadata.decode_request_ids_compact = ["req-0", "req-0", "req-1"]
        metadata.decode_valid_row_indices = torch.arange(3, dtype=torch.int32)
        metadata.decode_scratch_base_compact = torch.tensor([0, 4, 0])
        metadata.decode_target_slot_mapping = targets

        payload = sfa_v1._prepare_dsa_sparse_lmcache_payload(
            metadata,
            selected,
            index_topk=4,
        )

        self.assertIs(payload[0], selected)
        self.assertEqual(payload[1], ["req-0", "req-0", "req-1"])
        self.assertIs(payload[1], metadata.decode_request_ids_compact)
        self.assertIs(payload[2], targets)

    def test_staged_sparse_payload_validates_once_per_shared_metadata(self):
        metadata = self._make_decode_metadata()
        metadata.staged_sfa_payload_validated = False
        selected = torch.arange(4, dtype=torch.int32).view(1, 4)

        sfa_v1._prepare_dsa_sparse_lmcache_payload(
            metadata,
            selected,
            index_topk=4,
            validate_once=True,
        )
        metadata.decode_valid_row_indices = None
        payload = sfa_v1._prepare_dsa_sparse_lmcache_payload(
            metadata,
            selected,
            index_topk=4,
            validate_once=True,
        )

        self.assertTrue(metadata.staged_sfa_payload_validated)
        self.assertIs(payload[0], selected)

    def test_eligibility_accepts_single_native_piecewise_decode(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.staged_sfa_graph_key = STAGED_SFA_SINGLETON_GRAPH_KEY
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    reason = impl._cross_layer_ineligible_reason(
                        torch.empty(1, 4, dtype=dtype),
                        self._make_eligible_kv_cache(dtype=dtype),
                        metadata,
                    )
                    self.assertIsNone(reason)
            metadata.need_sparse_lmcache_payload = False
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(1, 4, dtype=torch.bfloat16),
                self._make_eligible_kv_cache(dtype=torch.bfloat16),
                metadata,
            )
            self.assertEqual(
                reason,
                "the v1 sparse LMCache payload path is unavailable",
            )
            metadata.need_sparse_lmcache_payload = True
            with patch.object(
                sfa_v1,
                "staged_sfa_connector_supports_sparse_load",
                return_value=False,
            ):
                reason = impl._cross_layer_ineligible_reason(
                    torch.empty(1, 4, dtype=torch.bfloat16),
                    self._make_eligible_kv_cache(dtype=torch.bfloat16),
                    metadata,
                )
            self.assertEqual(
                reason,
                "the active connector does not support staged sparse "
                "selective loads",
            )

    def test_eligibility_accepts_exact_multi_request_q1_batch(self):
        batch_size = 4
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata(batch_size)
        graph_key = StagedSFAGraphKey.exact_q1(batch_size)
        forward_context = MagicMock(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            staged_sfa_graph_dummy_run=False,
            staged_sfa_graph_key=graph_key,
            batch_descriptor=graph_key.to_legacy_batch_descriptor(),
            dsa_offload_manager=None,
            dsa_adapter_cache=None,
        )

        with (
            patch.object(sfa_v1, "get_forward_context", return_value=forward_context),
            patch.object(sfa_v1, "get_weight_prefetch_method", return_value=None),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(batch_size, 4, dtype=torch.bfloat16),
                self._make_eligible_kv_cache(
                    dtype=torch.bfloat16,
                    num_blocks=batch_size,
                ),
                metadata,
            )

        self.assertIsNone(reason)

    def test_eligibility_accepts_fixed_width_mtp_batch(self):
        impl = self._make_eligible_impl()
        impl.decode_threshold = 2
        impl.vllm_config.speculative_config = SimpleNamespace(
            num_speculative_tokens=1
        )
        metadata = self._make_decode_metadata(4)
        metadata.attn_state = AscendAttentionState.SpecDecoding
        metadata.cum_query_lens = torch.tensor([2, 4])
        metadata.seq_lens = torch.tensor([9, 9])
        metadata.seq_lens_cpu = torch.tensor([9, 9])
        metadata.block_table = torch.arange(2).view(2, 1)
        metadata.indexer_block_table = torch.arange(2).view(2, 1)
        metadata.decode_req_indices = torch.tensor(
            [0, 0, 1, 1], dtype=torch.int32
        )
        metadata.decode_req_indices_cpu = [0, 0, 1, 1]
        metadata.decode_request_ids_compact = ["req-0", "req-1"]
        metadata.req_ids = ["req-0", "req-1"]
        metadata.decode_selected_tokens = torch.empty(
            2, 8, dtype=torch.int32
        )
        metadata.decode_selected_counts = torch.empty(
            2, 16, dtype=torch.int32
        )
        metadata.decode_target_slot_mapping = torch.empty(
            2, 8, dtype=torch.long
        )
        graph_key = StagedSFAGraphKey.fixed_spec(2, 2)
        context = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            staged_sfa_graph_dummy_run=False,
            staged_sfa_graph_key=graph_key,
            batch_descriptor=graph_key.to_legacy_batch_descriptor(),
            dsa_offload_manager=None,
            dsa_adapter_cache=None,
        )
        with (
            patch.object(sfa_v1, "get_forward_context", return_value=context),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(4, 4, dtype=torch.bfloat16),
                self._make_eligible_kv_cache(
                    dtype=torch.bfloat16,
                    num_blocks=2,
                ),
                metadata,
            )
        self.assertIsNone(reason)

    def test_eligibility_rejects_invalid_cache_contract(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.staged_sfa_graph_key = STAGED_SFA_SINGLETON_GRAPH_KEY
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None
        valid = self._make_eligible_kv_cache()
        invalid_contracts = (
            (
                "rank",
                (
                    torch.empty(2, 128, 2, dtype=torch.bfloat16),
                    valid[1],
                    valid[2],
                ),
                "rank-4 PA_BSND",
            ),
            (
                "head_axis",
                (
                    torch.empty(2, 128, 2, 2, dtype=torch.bfloat16),
                    valid[1],
                    valid[2],
                ),
                "one KV head",
            ),
            (
                "hidden_dim",
                (
                    torch.empty(2, 128, 1, 3, dtype=torch.bfloat16),
                    valid[1],
                    valid[2],
                ),
                "hidden dimensions",
            ),
            (
                "different_block_sizes",
                (
                    valid[0],
                    valid[1],
                    torch.empty(2, 64, 1, 2, dtype=torch.bfloat16),
                ),
                "block sizes do not agree",
            ),
            (
                "wrong_configured_block_size",
                self._make_eligible_kv_cache(block_size=64),
                "configured block size",
            ),
            (
                "different_devices",
                (
                    valid[0],
                    valid[1],
                    torch.empty(
                        2,
                        128,
                        1,
                        2,
                        dtype=torch.bfloat16,
                        device="meta",
                    ),
                ),
                "different devices",
            ),
            (
                "different_dtypes",
                (
                    valid[0],
                    valid[1],
                    torch.empty(2, 128, 1, 2, dtype=torch.float16),
                ),
                "share one dtype",
            ),
            (
                "unsupported_dtype",
                self._make_eligible_kv_cache(dtype=torch.float32),
                "must be float16 or bfloat16",
            ),
        )

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            for name, kv_cache, expected_reason in invalid_contracts:
                with self.subTest(name=name):
                    reason = impl._cross_layer_ineligible_reason(
                        torch.empty(1, 4, dtype=torch.bfloat16),
                        kv_cache,
                        metadata,
                    )
                    self.assertIn(expected_reason, reason)

    def test_eligibility_rejects_weight_prefetch(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.staged_sfa_graph_key = STAGED_SFA_SINGLETON_GRAPH_KEY
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None
        weight_prefetch_method = MagicMock()
        weight_prefetch_method.mla_sfa_prefetch_enable = True

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=weight_prefetch_method,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(1, 4),
                self._make_eligible_kv_cache(),
                metadata,
            )

        self.assertEqual(reason, "weight prefetch is enabled")

    def test_eligibility_accepts_explicit_eager_dummy_warmup(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        metadata.need_sparse_lmcache_payload = False
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = True
        forward_context.staged_sfa_graph_key = STAGED_SFA_SINGLETON_GRAPH_KEY
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
            patch.object(
                sfa_v1,
                "staged_sfa_connector_supports_sparse_load",
                return_value=False,
            ) as connector_support,
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(1, 4),
                self._make_eligible_kv_cache(),
                metadata,
            )

        self.assertIsNone(reason)
        connector_support.assert_not_called()

    def test_eligibility_accepts_real_rows_within_graph_capacity(self):
        capacity = 4
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata(capacity)
        metadata.num_actual_tokens = metadata.num_decode_tokens = 1
        metadata.seq_lens[1:] = 0
        metadata.seq_lens_cpu[1:] = 0
        metadata.prompt_lens_cpu_rows = [8, 0, 0, 0]
        metadata.decode_req_indices[1:] = -1
        metadata.decode_req_indices_cpu = [0, -1, -1, -1]
        metadata.decode_valid_rows_all = False
        metadata.decode_valid_row_indices = torch.tensor(
            [0],
            dtype=torch.int32,
        )
        metadata.decode_request_ids_compact = ["req-0"]
        metadata.req_ids = ["req-0"]
        graph_key = StagedSFAGraphKey.exact_q1(capacity)
        forward_context = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            staged_sfa_graph_dummy_run=False,
            staged_sfa_graph_key=graph_key,
            batch_descriptor=graph_key.to_legacy_batch_descriptor(),
            dsa_offload_manager=None,
            dsa_adapter_cache=None,
        )

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._cross_layer_ineligible_reason(
                torch.empty(capacity, 4, dtype=torch.bfloat16),
                self._make_eligible_kv_cache(
                    dtype=torch.bfloat16,
                    num_blocks=capacity,
                ),
                metadata,
            )

        self.assertIsNone(reason)

class TestAscendSFABackend(TestBase):
    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_builder_cls(self):
        with patch.object(sfa_v1, "enable_cp", return_value=False):
            self.assertEqual(AscendSFABackend.get_builder_cls(), AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        with patch.object(sfa_v1, "enable_cp", return_value=False):
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


class TestAscendSFAMetadataBuilder(TestBase):
    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
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

        self.patcher = patch("vllm.config.get_current_vllm_config", return_value=self.mock_cfg)
        self.patcher.start()

        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(
            self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen
        ):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size, vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__", mock_parent_init
        )
        self.parent_init_patcher.start()

        if hasattr(enable_dsa_cp, "cache_clear"):
            enable_dsa_cp.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    @patch("vllm_ascend.attention.sfa_v1.is_v1_kv_transfer_group")
    @patch("vllm_ascend.attention.sfa_v1.has_kv_transfer_group")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    def test_dsa_sparse_metadata_reuses_builder_storage(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_has_kv_transfer_group,
        mock_is_v1_kv_transfer_group,
    ):
        mock_enable_dsa_cp.return_value = False
        mock_has_kv_transfer_group.return_value = True
        mock_is_v1_kv_transfer_group.return_value = True
        mock_get_cos_and_sin_mla.side_effect = lambda positions, _: (
            torch.zeros_like(positions),
            torch.zeros_like(positions),
        )

        kv_cache_spec = MagicMock()
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        vllm_config.model_config.hf_text_config.topk_tokens = 16
        vllm_config.speculative_config.num_speculative_tokens = 1
        vllm_config.scheduler_config.max_num_seqs = 4
        vllm_config.scheduler_config.max_num_batched_tokens = 8

        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_DSA_UNBUNDLE": "1",
                "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
            },
        ):
            builder = AscendSFAMetadataBuilder(
                kv_cache_spec=kv_cache_spec,
                layer_names=["layer1", "layer2"],
                vllm_config=vllm_config,
                device=torch.device("cpu"),
            )
        builder.attn_mask_builder.get_attention_mask = MagicMock(return_value=None)

        def common_metadata(
            query_start_loc,
            computed,
            prompt_lens,
            request_ids,
        ):
            num_reqs = len(request_ids)
            num_tokens = int(query_start_loc[-1])
            return SimpleNamespace(
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                num_input_tokens=num_tokens,
                block_table_tensor=torch.zeros((num_reqs, 4), dtype=torch.int32),
                slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
                positions=torch.arange(num_tokens, dtype=torch.int64),
                indexer_block_table_tensor=None,
                indexer_slot_mapping=None,
                prompt_lens_cpu=prompt_lens,
                query_start_loc_cpu=torch.tensor(query_start_loc, dtype=torch.int32),
                num_computed_tokens_cpu=torch.tensor(computed, dtype=torch.int32),
                query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
                seq_lens=torch.tensor(computed, dtype=torch.int32),
                seq_lens_cpu=torch.tensor(computed, dtype=torch.int32),
                request_ids=request_ids,
                attn_state=AscendAttentionState.DecodeOnly,
                dsa_scratch_state_indices=torch.arange(
                    num_reqs,
                    dtype=torch.int32,
                ),
                dsa_scratch_request_generations=torch.arange(
                    1,
                    num_reqs + 1,
                    dtype=torch.int64,
                ),
            )

        first = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 2, 4],
                [10, 20],
                [9, 19],
                ["r0", "r1"],
            ),
        )
        addresses = (
            first.split_boundary.data_ptr(),
            first.decode_req_indices.data_ptr(),
            first.decode_row_offsets.data_ptr(),
            first.decode_selected_tokens.data_ptr(),
            first.decode_selected_counts.data_ptr(),
            first.decode_target_slot_mapping.data_ptr(),
        )
        assert first.decode_req_indices.tolist() == [0, 0, 1, 1]
        assert first.decode_row_offsets.tolist() == [0, 1, 0, 1]
        assert first.num_decode_tokens == 4

        second = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 1],
                [5],
                [4],
                ["r2"],
            ),
        )
        second_addresses = (
            second.split_boundary.data_ptr(),
            second.decode_req_indices.data_ptr(),
            second.decode_row_offsets.data_ptr(),
            second.decode_selected_tokens.data_ptr(),
            second.decode_selected_counts.data_ptr(),
            second.decode_target_slot_mapping.data_ptr(),
        )

        assert second_addresses == addresses
        assert second.split_boundary.tolist() == [4]
        assert second.decode_req_indices.tolist() == [0]
        assert second.decode_row_offsets.tolist() == [0]
        assert second.num_decode_tokens == 1

        third = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 2, 4],
                [12, 22],
                [11, 21],
                ["r3", "r4"],
            ),
        )
        assert third.decode_req_indices.tolist() == [0, 0, 1, 1]
        assert third.decode_row_offsets.tolist() == [0, 1, 0, 1]
        assert third.split_boundary.tolist() == [11, 11, 21, 21]

        with self.assertRaisesRegex(RuntimeError, "max_num_batched_tokens=8"):
            builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_metadata(
                    [0, 9],
                    [9],
                    [8],
                    ["too-large"],
                ),
            )

        with self.assertRaisesRegex(RuntimeError, "max_num_seqs=4"):
            builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_metadata(
                    [0, 1, 2, 3, 4, 5],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    ["r0", "r1", "r2", "r3", "r4"],
                ),
            )

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

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

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

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

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
        self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config
    ):
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

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly
