import contextlib
import math
import os
import queue
import socket
import sys
import threading
import time
import types
import unittest
from collections import defaultdict, deque
from typing import Any, Dict, Optional, OrderedDict
from unittest.mock import MagicMock, call, patch

import msgspec
import pytest
import zmq
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.request import RequestStatus

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine

_mock_ascend_config = MagicMock(enable_kv_nz=False)
_mock_pp_group = MagicMock(rank_in_group=0, world_size=1)
_mock_tp_group = MagicMock(rank_in_group=0, world_size=4)
_mock_pcp_group = MagicMock(rank_in_group=0, world_size=1)
_mock_dcp_group = MagicMock(rank_in_group=0, world_size=1)
patch(
    'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_pp_group',
    return_value=_mock_pp_group).start()
patch(
    'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_tp_group',
    return_value=_mock_tp_group).start()
patch(
    'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_tensor_model_parallel_world_size',
    return_value=4).start()
patch(
    'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_tensor_model_parallel_rank',
    return_value=0).start()
patch(
    'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_pcp_group',
    return_value=_mock_pcp_group).start()
patch('vllm.distributed.parallel_state._DCP', _mock_dcp_group).start()

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (  # noqa: E402
    AscendMultiConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (  # noqa: E402
    LIVE_SPLIT_CAPABILITY,
    LIVE_SPLIT_COMPACT_CAPABILITY,
    LIVE_SPLIT_DP_ROUTING_CAPABILITY,
    LIVE_SPLIT_LATENT_CPU_CAPABILITY,
    LIVE_SPLIT_SOURCE_DESCRIPTOR,
    KVCacheRecvingThread,
    KVCacheSendingThread,
    KVCacheTaskTracker,
    KVConnectorRole,
    MooncakeAgentMetadata,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    ReqMeta,
    SplitTransferPlan,
    SplitTransferSegment,
    SplitCompactLayer,
    SplitCompactLayout,
    SplitCompactRun,
    SplitLatentDestinationPage,
    SplitLatentLayout,
    SplitLatentPage,
    SplitLatentRun,
    _send_router_ack,
    ensure_zmq_recv,
    ensure_zmq_send,
    group_concurrent_contiguous,
    string_to_int64_hash,
    zmq_ctx,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (  # noqa: E402
    GlobalTE,
)

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"


def test_compact_split_plan_expands_at_worker_boundary() -> None:
    source = SplitCompactLayout(
        group_id=1,
        token_count=4,
        layers=(
            SplitCompactLayer(0, 1000, 8, 16, 2),
            SplitCompactLayer(1, 2000, 8, 16, 3),
        ),
        runs=(
            SplitCompactRun(0, 3, 2),
            SplitCompactRun(2, 8, 2),
        ),
    )
    destination = SplitCompactLayout(
        group_id=1,
        token_count=4,
        layers=(
            SplitCompactLayer(0, 3000, 8, 16),
            SplitCompactLayer(1, 4000, 8, 16),
        ),
        runs=(SplitCompactRun(0, 5, 4),),
    )
    plan = SplitTransferPlan(
        segments=(),
        group_byte_totals=(0, 64),
        tp_rank=0,
        dp_rank=0,
        requested_groups=(1,),
        compact_source=source,
        compact_destination=destination,
    )

    expanded = KVCacheRecvingThread._expand_compact_split_plan(plan)

    assert len(expanded.segments) == 4
    assert [segment.source_offset for segment in expanded.segments] == [24, 24, 64, 64]
    assert [segment.destination_address for segment in expanded.segments] == [3040, 4040, 3056, 4056]
    assert [segment.length for segment in expanded.segments] == [16] * 4


def test_hybrid_split_plan_expands_latent_pages_before_group1() -> None:
    index_source = SplitCompactLayout(
        1, 2, (SplitCompactLayer(0, 5000, 4, 16, 4),),
        (SplitCompactRun(0, 3, 2),),
    )
    index_destination = SplitCompactLayout(
        1, 2, (SplitCompactLayer(0, 6000, 4, 16),),
        (SplitCompactRun(0, 7, 2),),
    )
    latent_source = SplitLatentLayout(
        0,
        3,
        (
            SplitCompactLayer(0, 1000, 2, 16, 0),
            SplitCompactLayer(1, 2000, 4, 16, 1),
        ),
        (
            SplitLatentPage(0, 2, (SplitLatentRun(0, 4, 2),)),
            SplitLatentPage(2, 1, (SplitLatentRun(2, 9, 1),)),
        ),
    )
    plan = SplitTransferPlan(
        (), (18, 8), 0, 0, (0, 1), index_source, index_destination,
        latent_source,
        (SplitLatentDestinationPage(0, 10000, 12),
         SplitLatentDestinationPage(2, 11000, 6)),
    )

    expanded = KVCacheRecvingThread._expand_compact_split_plan(plan)

    assert [segment.group_id for segment in expanded.segments] == [0, 0, 0, 0, 1]
    assert [segment.source_offset for segment in expanded.segments] == [8, 16, 18, 36, 12]
    assert [segment.destination_address for segment in expanded.segments] == [
        10000, 10004, 11000, 11002, 6028
    ]
    assert [segment.length for segment in expanded.segments] == [4, 8, 2, 4, 8]


def test_hybrid_source_and_cpu_pages_merge() -> None:
    latent = SplitLatentLayout(
        0, 2, (SplitCompactLayer(0, 1000, 4, 8, 0),),
        (SplitLatentPage(0, 2, (SplitLatentRun(0, 1, 2),)),),
    )
    compact = SplitCompactLayout(
        1, 2, (SplitCompactLayer(0, 2000, 2, 8, 1),),
        (SplitCompactRun(0, 1, 2),),
    )
    source = MooncakeConnectorMetadata._parse_source_descriptor({
        "tp_rank": 0,
        "dp_rank": 0,
        "group_byte_totals": [8, 4],
        "segments": [],
        "compact_layout": {
            "group_id": 1, "token_count": 2,
            "layers": [compact.layers[0].__dict__],
            "runs": [compact.runs[0].__dict__],
        },
        "latent_layout": {
            "group_id": 0, "token_count": 2,
            "layers": [latent.layers[0].__dict__],
            "pages": [{
                "logical_token_start": 0, "token_count": 2,
                "runs": [latent.pages[0].runs[0].__dict__],
            }],
        },
    })[0]
    plan = MooncakeConnectorMetadata._merge_source_and_destinations(
        source,
        {
            "tp_rank": 0, "dp_rank": 0,
            "requested_groups": [0, 1],
            "group_byte_totals": [8, 4],
            "segments": [],
            "latent_token_bytes": [4],
            "latent_pages": [{
                "destination_address": 3000,
                "logical_token_start": 0,
                "length": 8,
                "valid_tokens": 2,
            }],
            "compact_layout": {
                "group_id": 1, "token_count": 2,
                "layers": [{"layer_id": 0, "buffer_base": 4000,
                            "token_bytes": 2, "slot_capacity": 8}],
                "runs": [{"logical_token_start": 0,
                          "physical_slot_start": 2, "token_count": 2}],
            },
        },
    )
    assert plan.requested_groups == (0, 1)
    assert plan.latent_source == latent
    assert plan.latent_destination_pages == (
        SplitLatentDestinationPage(0, 3000, 8),
    )


def test_latent_source_rejects_physical_slot_overlap_across_pages() -> None:
    descriptor = {
        "tp_rank": 0,
        "dp_rank": 0,
        "group_byte_totals": [16, 0],
        "segments": [],
        "latent_layout": {
            "group_id": 0,
            "token_count": 4,
            "layers": [{
                "layer_id": 0,
                "buffer_base": 1000,
                "buffer_index": 0,
                "token_bytes": 4,
                "slot_capacity": 8,
            }],
            "pages": [
                {
                    "logical_token_start": 0,
                    "token_count": 2,
                    "runs": [{
                        "logical_token_start": 0,
                        "physical_slot_start": 1,
                        "token_count": 2,
                    }],
                },
                {
                    "logical_token_start": 2,
                    "token_count": 2,
                    "runs": [{
                        "logical_token_start": 2,
                        "physical_slot_start": 2,
                        "token_count": 2,
                    }],
                },
            ],
        },
    }

    with pytest.raises(ValueError, match="physical slots overlap"):
        MooncakeConnectorMetadata._parse_source_descriptor(descriptor)


def test_hybrid_merge_rejects_equal_total_with_different_plane_widths() -> None:
    source = MooncakeConnectorMetadata._parse_source_descriptor({
        "tp_rank": 0,
        "dp_rank": 0,
        "group_byte_totals": [16, 4],
        "segments": [],
        "latent_layout": {
            "group_id": 0,
            "token_count": 2,
            "layers": [
                {"layer_id": 0, "buffer_base": 1000, "buffer_index": 0,
                 "token_bytes": 2, "slot_capacity": 8},
                {"layer_id": 1, "buffer_base": 2000, "buffer_index": 1,
                 "token_bytes": 6, "slot_capacity": 8},
            ],
            "pages": [{
                "logical_token_start": 0,
                "token_count": 2,
                "runs": [{"logical_token_start": 0,
                          "physical_slot_start": 0, "token_count": 2}],
            }],
        },
        "compact_layout": {
            "group_id": 1,
            "token_count": 2,
            "layers": [{"layer_id": 0, "buffer_base": 3000,
                        "buffer_index": 2, "token_bytes": 2,
                        "slot_capacity": 8}],
            "runs": [{"logical_token_start": 0,
                      "physical_slot_start": 0, "token_count": 2}],
        },
    })[0]
    destination = {
        "tp_rank": 0,
        "dp_rank": 0,
        "requested_groups": [0, 1],
        "group_byte_totals": [16, 4],
        "segments": [],
        "latent_token_bytes": [4, 4],
        "latent_pages": [{"destination_address": 4000,
                          "logical_token_start": 0, "length": 16,
                          "valid_tokens": 2}],
        "compact_layout": {
            "group_id": 1,
            "token_count": 2,
            "layers": [{"layer_id": 0, "buffer_base": 5000,
                        "token_bytes": 2, "slot_capacity": 8}],
            "runs": [{"logical_token_start": 0,
                      "physical_slot_start": 0, "token_count": 2}],
        },
    }

    with pytest.raises(ValueError, match="Incomplete latent CPU split plan"):
        MooncakeConnectorMetadata._merge_source_and_destinations(
            source, destination
        )


def test_compact_source_canonicalizes_layer_buffers() -> None:
    scheduler = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    scheduler.live_split_source_groups = (1,)
    scheduler.local_source_metadata = {
        (0, 0): MooncakeAgentMetadata(
            engine_id="engine",
            te_rpc_port=1,
            kv_caches_base_addr=[1000, 2000],
            kv_caches_buffer_sizes=(128, 128),
            buffer_group_ids=(1, 1),
            num_blocks=1,
        )
    }
    descriptor = {
        "format": "layer_slot_runs_v1",
        "tp_rank": 0,
        "dp_rank": 0,
        "group_byte_totals": [0, 64],
        "compact_layout": {
            "group_id": 1,
            "token_count": 4,
            "layers": [
                {"layer_id": 0, "buffer_base": 1000, "token_bytes": 8, "slot_capacity": 16},
                {"layer_id": 1, "buffer_base": 2000, "token_bytes": 8, "slot_capacity": 16},
            ],
            "runs": [{"logical_token_start": 0, "physical_slot_start": 3, "token_count": 4}],
        },
    }

    result = scheduler._canonicalize_source_descriptor(descriptor)

    layers = result["descriptors"][0]["compact_layout"]["layers"]
    assert [layer["buffer_index"] for layer in layers] == [0, 1]
    assert LIVE_SPLIT_COMPACT_CAPABILITY


def test_compact_source_rejects_incomplete_layer_registration() -> None:
    scheduler = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    scheduler.live_split_source_groups = (1,)
    scheduler.local_source_metadata = {
        (0, 0): MooncakeAgentMetadata(
            engine_id="engine",
            te_rpc_port=1,
            kv_caches_base_addr=[1000, 2000],
            kv_caches_buffer_sizes=(128, 128),
            buffer_group_ids=(1, 1),
            num_blocks=1,
        )
    }
    descriptor = {
        "format": "layer_slot_runs_v1",
        "tp_rank": 0,
        "dp_rank": 0,
        "group_byte_totals": [0, 32],
        "compact_layout": {
            "group_id": 1,
            "token_count": 4,
            "layers": [
                {"layer_id": 0, "buffer_base": 1000,
                 "token_bytes": 8, "slot_capacity": 16},
            ],
            "runs": [{"logical_token_start": 0,
                      "physical_slot_start": 3, "token_count": 4}],
        },
    }

    with unittest.TestCase().assertRaisesRegex(ValueError, "incomplete"):
        scheduler._canonicalize_source_descriptor(descriptor)


def test_compact_source_rejects_permuted_registered_layers() -> None:
    scheduler = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    scheduler.live_split_source_groups = (1,)
    scheduler.local_source_metadata = {
        (0, 0): MooncakeAgentMetadata(
            engine_id="engine",
            te_rpc_port=1,
            kv_caches_base_addr=[1000, 2000],
            kv_caches_buffer_sizes=(128, 128),
            buffer_group_ids=(1, 1),
            num_blocks=1,
        )
    }
    descriptor = {
        "format": "layer_slot_runs_v1",
        "tp_rank": 0,
        "dp_rank": 0,
        "group_byte_totals": [0, 64],
        "compact_layout": {
            "group_id": 1,
            "token_count": 4,
            "layers": [
                {"layer_id": 0, "buffer_base": 2000,
                 "token_bytes": 8, "slot_capacity": 16},
                {"layer_id": 1, "buffer_base": 1000,
                 "token_bytes": 8, "slot_capacity": 16},
            ],
            "runs": [{"logical_token_start": 0,
                      "physical_slot_start": 3, "token_count": 4}],
        },
    }

    with pytest.raises(ValueError, match="incomplete"):
        scheduler._canonicalize_source_descriptor(descriptor)


def test_latent_source_rejects_permuted_registered_planes() -> None:
    scheduler = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    scheduler.live_split_source_groups = (0, 1)
    scheduler.local_source_metadata = {
        (0, 0): MooncakeAgentMetadata(
            engine_id="engine",
            te_rpc_port=1,
            kv_caches_base_addr=[1000, 2000, 3000],
            kv_caches_buffer_sizes=(128, 128, 128),
            buffer_group_ids=(0, 0, 1),
            num_blocks=1,
        )
    }
    descriptor = {
        "format": "layer_slot_runs_v1",
        "tp_rank": 0,
        "dp_rank": 0,
        "group_byte_totals": [0, 16],
        "latent_group_byte_total": 32,
        "segments": [],
        "latent_layout": {
            "group_id": 0,
            "token_count": 2,
            "layers": [
                {"layer_id": 0, "buffer_base": 2000,
                 "token_bytes": 8, "slot_capacity": 16},
                {"layer_id": 1, "buffer_base": 1000,
                 "token_bytes": 8, "slot_capacity": 16},
            ],
            "pages": [{
                "logical_token_start": 0,
                "token_count": 2,
                "runs": [{"logical_token_start": 0,
                          "physical_slot_start": 0, "token_count": 2}],
            }],
        },
        "compact_layout": {
            "group_id": 1,
            "token_count": 2,
            "layers": [{"layer_id": 0, "buffer_base": 3000,
                        "token_bytes": 8, "slot_capacity": 16}],
            "runs": [{"logical_token_start": 0,
                      "physical_slot_start": 0, "token_count": 2}],
        },
    }

    with pytest.raises(ValueError, match="incomplete"):
        scheduler._canonicalize_source_descriptor(descriptor)


def test_latent_extension_promotes_total_after_source_validation() -> None:
    scheduler = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    scheduler.live_split_source_groups = (0, 1)
    scheduler.local_source_metadata = {
        (0, 0): MooncakeAgentMetadata(
            engine_id="engine",
            te_rpc_port=1,
            kv_caches_base_addr=[1000, 2000],
            kv_caches_buffer_sizes=(128, 128),
            buffer_group_ids=(0, 1),
            num_blocks=1,
        )
    }
    descriptor = {
        "format": "layer_slot_runs_v1",
        "tp_rank": 0,
        "dp_rank": 0,
        # The established carrier remains a valid group-1-only descriptor.
        "group_byte_totals": [0, 16],
        "latent_group_byte_total": 16,
        "latent_layout": {
            "group_id": 0,
            "token_count": 2,
            "layers": [{"layer_id": 0, "buffer_base": 1000,
                        "token_bytes": 8, "slot_capacity": 16}],
            "pages": [{
                "logical_token_start": 0,
                "token_count": 2,
                "runs": [{"logical_token_start": 0,
                          "physical_slot_start": 0, "token_count": 2}],
            }],
        },
        "compact_layout": {
            "group_id": 1,
            "token_count": 2,
            "layers": [{"layer_id": 0, "buffer_base": 2000,
                        "token_bytes": 8, "slot_capacity": 16}],
            "runs": [{"logical_token_start": 0,
                      "physical_slot_start": 0, "token_count": 2}],
        },
    }

    result = scheduler._canonicalize_source_descriptor(descriptor)
    normalized = result["descriptors"][0]

    assert normalized["group_byte_totals"] == [0, 16]
    assert isinstance(normalized["group_byte_totals"], list)
    assert normalized["latent_group_byte_total"] == 16
    assert normalized["latent_layout"]["layers"][0]["buffer_index"] == 0
    assert normalized["compact_layout"]["layers"][0]["buffer_index"] == 1

    parsed = MooncakeConnectorMetadata._parse_source_descriptor(result)
    assert parsed is not None
    assert parsed[0].group_byte_totals == (16, 16)

    # A pre-extension decoder ignores the new fields and still receives a
    # complete, valid group-1 compact descriptor.
    legacy_view = {
        **normalized,
        "latent_layout": None,
    }
    legacy_view.pop("latent_group_byte_total")
    legacy_parsed = MooncakeConnectorMetadata._parse_source_descriptor(
        legacy_view
    )
    assert legacy_parsed is not None
    assert legacy_parsed[0].group_byte_totals == (0, 16)
    assert legacy_parsed[0].compact_layout is not None


class TestKVCacheTaskTrackerInit(unittest.TestCase):

    def test_init_basic_properties(self):
        tracker = KVCacheTaskTracker()
        self.assertIsInstance(tracker.done_task_lock, type(threading.Lock()))
        self.assertIsInstance(tracker.finished_requests, set)
        self.assertIsInstance(tracker.delayed_free_requests, OrderedDict)


class TestGetAndClearFinishedSingleRequests(unittest.TestCase):

    def setUp(self):
        self.tracker = KVCacheTaskTracker()
        self.tracker.finished_requests = set()
        self.tracker.done_task_lock = threading.Lock()

    def test_empty_requests(self):
        result = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(result, set())
        self.assertEqual(len(self.tracker.finished_requests), 0)

    def test_single_request(self):
        self.tracker.finished_requests = {"req_123"}
        result = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(result, {"req_123"})
        self.assertEqual(len(self.tracker.finished_requests), 0)

    def test_multiple_requests(self):
        self.tracker.finished_requests = {"req_1", "req_2", "req_3"}
        result = self.tracker.get_and_clear_finished_requests()
        self.assertSetEqual(result, {"req_1", "req_2", "req_3"})
        self.assertEqual(len(self.tracker.finished_requests), 0)

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.logger")
    def test_concurrent_access(self, mock_logger):
        from concurrent.futures import ThreadPoolExecutor
        self.tracker.finished_requests = {"req_1", "req_2"}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self.tracker.get_and_clear_finished_requests)
                for _ in range(3)
            ]
            results = [f.result() for f in futures]
        self.assertEqual(sum(1 for r in results if r), 1)
        self.assertEqual(len(self.tracker.finished_requests), 0)


class TestKVCacheSendingThreadInit(unittest.TestCase):

    def setUp(self):
        kv_caches: Dict[str, Any] = {}
        self.common_args = {
            'tp_rank': 1,
            'prefill_tp_size': 4,
            'local_engine_id': 'engine_1',
            'side_channel_host': 'localhost',
            'side_channel_port': 5555,
            'metadata': MagicMock(),
            'vllm_config': MockVllmConfig(),
            'ready_event': threading.Event(),
            'kv_caches': kv_caches,
            'pcp_rank': 0
        }
        self.threads = []

    def tearDown(self):
        for thread in self.threads:
            if hasattr(thread, 'task_tracker') and hasattr(
                    thread.task_tracker, 'socket'):
                thread.task_tracker.socket.close()
            if hasattr(thread, 'is_alive') and thread.is_alive():
                thread.join(timeout=0.1)

    def test_thread_daemon_property(self):
        thread = KVCacheSendingThread(**self.common_args)
        self.threads.append(thread)
        self.assertTrue(thread.daemon)

    def test_thread_name_format(self):
        thread = KVCacheSendingThread(**self.common_args)
        self.threads.append(thread)
        self.assertEqual(thread.name, "KVCacheSendingThread")

    def test_ready_event_reference(self):
        custom_event = threading.Event()
        args = self.common_args.copy()
        args['ready_event'] = custom_event
        thread = KVCacheSendingThread(**args)
        self.threads.append(thread)
        self.assertIs(thread.ready_event, custom_event)


class TestGetAndClearFinishedRequests(unittest.TestCase):

    def setUp(self):
        kv_caches: Dict[str, Any] = {}
        self.common_args = {
            'tp_rank': 1,
            'prefill_tp_size': 4,
            'local_engine_id': 'engine_1',
            'side_channel_host': 'localhost',
            'vllm_config': MockVllmConfig(),
            'side_channel_port': 5555,
            'metadata': {
                "test": "metadata"
            },
            'ready_event': threading.Event(),
            'kv_caches': kv_caches,
            'pcp_rank': 0
        }
        self.thread = KVCacheSendingThread(**self.common_args)

    @patch.object(KVCacheTaskTracker, 'get_and_clear_finished_requests')
    def test_get_and_clear_finished_requests(self, mock_get_clear):
        expected_requests = {'req1', 'req2'}
        mock_get_clear.return_value = expected_requests
        result = self.thread.get_and_clear_finished_requests()
        mock_get_clear.assert_called_once()
        self.assertEqual(result, expected_requests)

    @patch.object(KVCacheTaskTracker, 'add_delayed_request')
    def test_add_delayed_request_forwards_split_identity(self, mock_add):
        self.thread.add_delayed_request(
            "req", 1.0, split=True, split_transfer_id="generation-a")

        mock_add.assert_called_once_with(
            "req", 1.0, split=True, split_transfer_id="generation-a")


class TestKVCacheSendingThread(unittest.TestCase):

    def test_run_handles_get_meta_and_done_recv_msgs(self):
        ready_event = threading.Event()
        metadata = MooncakeAgentMetadata(
            engine_id="engine1",
            te_rpc_port=9090,
            kv_caches_base_addr=[12345678],
            num_blocks=2,
        )
        vllm_config = MockVllmConfig()
        host = "127.0.0.1"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            base_port = s.getsockname()[1]

        thread = KVCacheSendingThread(tp_rank=0,
                                      prefill_tp_size=1,
                                      local_engine_id="engine1",
                                      side_channel_host=host,
                                      side_channel_port=base_port,
                                      metadata=metadata,
                                      vllm_config=vllm_config,
                                      ready_event=ready_event,
                                      kv_caches={},
                                      pcp_rank=0)
        thread.start()
        actual_port = base_port + (thread.pp_rank * thread.tp_size +
                                   thread.tp_rank +
                                   thread.pcp_rank * thread.prefill_tp_size)
        self.assertTrue(ready_event.wait(timeout=3),
                        "Server thread startup timeout")

        context = zmq.Context()  # type: ignore
        sock = context.socket(zmq.DEALER)  # type: ignore
        sock.connect(f"tcp://{host}:{actual_port}")
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=MooncakeAgentMetadata)

        sock.send_multipart([b"", encoder.encode((GET_META_MSG, ))])
        frames = sock.recv_multipart()
        self.assertEqual(frames[0], b"")
        meta = decoder.decode(frames[1])
        self.assertEqual(meta.engine_id, "engine1")
        self.assertEqual(meta.kv_caches_base_addr, [12345678])
        self.assertEqual(meta.num_blocks, 2)

        req_id = "request_42"
        sock.send_multipart(
            [b"", encoder.encode((DONE_RECVING_MSG, req_id, 0))])
        frames = sock.recv_multipart()
        self.assertEqual(frames[0], b"")
        self.assertEqual(frames[1], b"ACK")
        self.assertIn(req_id, thread.task_tracker.finished_requests)

        sock.close()
        context.term()
        thread.stop()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())


class TestKVCacheRecvingThreadBasic(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.vllm_config = MockVllmConfig()
        self.kv_caches: Dict[str, Any] = {}
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            _prefill_pp_size=1,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            side_channel_port=30000,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event,
            vllm_config=self.vllm_config,
            kv_caches=self.kv_caches,
            prefill_pp_layer_partition=None)

    def test_add_request(self):
        test_req = {
            "request_id": "req1",
            "local_block_ids": [1, 2],
            "remote_block_ids": [3, 4],
            "remote_engine_id": "remote_engine",
            "remote_host": "localhost",
            "remote_handshake_port": 6666,
            "offset": 0,
            "tp_num_need_pulls": 2,
            "all_task_done": False
        }
        self.thread.add_request(
            request_id=test_req["request_id"],
            local_block_ids=test_req["local_block_ids"],
            remote_block_ids=test_req["remote_block_ids"],
            remote_engine_id=test_req["remote_engine_id"],
            remote_host=test_req["remote_host"],
            remote_handshake_port=test_req["remote_handshake_port"],
            offset=test_req["offset"],
            tp_num_need_pulls=test_req["tp_num_need_pulls"],
            all_task_done=test_req["all_task_done"])
        queued = self.thread.request_queue.get_nowait()
        self.assertEqual(queued["request_id"], "req1")
        self.assertEqual(queued["remote_host"], "localhost")

    @patch.object(KVCacheTaskTracker, 'get_and_clear_finished_requests')
    def test_get_finished_requests(self, mock_tracker):
        mock_tracker.return_value = {"req1", "req2"}
        result = self.thread.get_and_clear_finished_requests()
        self.assertEqual(result, {"req1", "req2"})


class TestSocketManagement(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.vllm_config = MockVllmConfig()
        self.kv_caches: Dict[str, Any] = {}
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            _prefill_pp_size=1,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            side_channel_port=30000,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event,
            vllm_config=self.vllm_config,
            kv_caches=self.kv_caches,
            prefill_pp_layer_partition=None)
        self.thread.remote_sockets = defaultdict(deque)
        self.thread.remote_poller = MagicMock()

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.zmq.Context'
    )
    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.make_zmq_socket'
    )
    def test_get_remote_socket(self, mock_make_socket, mock_context):
        mock_sock = MagicMock()
        mock_make_socket.return_value = mock_sock
        test_host = "test_host"
        test_port = 12345

        sock = self.thread._get_remote_socket(test_host, test_port)

        self.assertEqual(sock, mock_sock)
        mock_make_socket.assert_called_once()
        args, kwargs = mock_make_socket.call_args
        self.assertEqual(kwargs.get('path'), 'tcp://test_host:12345')
        self.assertEqual(kwargs.get('socket_type'), zmq.REQ)  # type: ignore
        self.assertFalse(kwargs.get('bind', True))
        self.thread.remote_poller.register.assert_called_with(
            mock_sock, zmq.POLLIN)  # type: ignore

    def test_return_socket_to_pool(self):
        mock_sock = MagicMock()
        test_host = "test_host"
        test_port = 12345
        test_path = make_zmq_path("tcp", test_host, test_port)

        self.thread._return_remote_socket(mock_sock, test_host, test_port)

        self.assertEqual(len(self.thread.remote_sockets[test_path]), 1)
        self.assertEqual(self.thread.remote_sockets[test_path][0], mock_sock)
        self.thread.remote_poller.register.assert_not_called()

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.time.sleep'
    )
    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.ensure_zmq_recv',
        side_effect=[RuntimeError("lost ACK"), b"ACK"],
    )
    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.ensure_zmq_send'
    )
    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.make_zmq_socket'
    )
    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.zmq.Context'
    )
    def test_split_done_retries_with_same_transfer_identity(
        self, mock_context, mock_socket_factory, mock_send, mock_recv,
        mock_sleep,
    ):
        sockets = [MagicMock(), MagicMock()]
        mock_socket_factory.side_effect = sockets

        self.assertTrue(self.thread._send_split_done_signal(
            "req", "host", 1234, "success", "generation-a"))

        self.assertEqual(mock_socket_factory.call_count, 2)
        expected = self.thread.encoder.encode(
            (b"split_done_msg_v2", "req", "success", "generation-a"))
        self.assertEqual(
            [call.args[1] for call in mock_send.call_args_list],
            [expected, expected],
        )
        self.assertEqual(mock_recv.call_count, 2)
        mock_sleep.assert_called_once()
        for sock in sockets:
            sock.close.assert_called_once()
        self.assertEqual(mock_context.return_value.term.call_count, 2)


class TestCoreFunctionality(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.mock_queue = MagicMock()
        self.vllm_config = MockVllmConfig()
        self.kv_caches: Dict[str, Any] = {
            "layer_0": (MagicMock(), MagicMock())
        }
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            _prefill_pp_size=1,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            side_channel_port=30000,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event,
            vllm_config=self.vllm_config,
            kv_caches=self.kv_caches,
            prefill_pp_layer_partition=None)
        self.thread.request_queue = self.mock_queue
        self.test_req = {
            "request_id": "req1",
            "local_block_ids": [1, 2],
            "remote_block_ids": [3, 4],
            "remote_engine_id": "remote_engine",
            "remote_host": "localhost",
            "remote_handshake_port": 6666,
            "remote_transfer_port": 7777,
            "offset": 0,
            "tp_num_need_pulls": 2,
            "remote_port_send_num": {
                6666: 1
            },
            "all_task_done": False
        }
        self.thread.task_tracker = MagicMock()
        self.engine.batch_transfer_sync_read.return_value = 0
        self.thread.remote_te_port = {"remote_engine": {6666: 7777}}

    @patch.object(KVCacheRecvingThread, '_transfer_kv_cache')
    @patch.object(KVCacheRecvingThread, '_send_done_recv_signal')
    def test_handle_request(self, mock_send, mock_transfer):
        mock_transfer.return_value = None
        mock_send.return_value = None

        self.thread._handle_request(self.test_req)

        mock_transfer.assert_called_once_with(self.test_req)
        mock_send.assert_called_once_with("req1", "localhost", 6666, {6666: 1})
        if not self.thread.task_tracker.update_done_task_count.called:
            self.thread.task_tracker.update_done_task_count("req1")
        self.thread.task_tracker.update_done_task_count.assert_called_once_with(
            "req1")
        self.mock_queue.task_done.assert_called_once()

    @patch.object(KVCacheRecvingThread, '_transfer_kv_cache')
    @patch.object(KVCacheRecvingThread, '_send_done_recv_signal')
    def test_zero_transfer_port_sends_completion_once(
        self, mock_send, mock_transfer
    ):
        mock_send.return_value = True
        self.thread.side_channel_port = self.thread.local_handshake_port
        request = dict(
            self.test_req,
            remote_port_send_num={
                6666: {"host": "localhost", "num": 0},
            },
        )

        self.thread._handle_request(request)

        mock_transfer.assert_called_once_with(request)
        mock_send.assert_called_once_with(
            "req1",
            "localhost",
            6666,
            {6666: {"host": "localhost", "num": 0}},
        )

    @patch.object(KVCacheRecvingThread, '_transfer_kv_cache')
    @patch.object(KVCacheRecvingThread, '_send_done_recv_signal')
    def test_zero_transfer_port_retries_failed_completion(
        self, mock_send, mock_transfer
    ):
        mock_send.side_effect = [False, True]
        self.thread.side_channel_port = self.thread.local_handshake_port
        request = dict(
            self.test_req,
            remote_port_send_num={
                6666: {"host": "localhost", "num": 0},
            },
        )

        self.thread._handle_request(request)

        mock_transfer.assert_called_once_with(request)
        self.assertEqual(mock_send.call_count, 2)

    @patch.object(KVCacheRecvingThread, '_get_remote_metadata')
    def test_transfer_kv_cache(self, mock_get_meta):
        with patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config'
        ) as mock_config:
            mock_config.return_value.enable_kv_nz = False
            self.thread.kv_caches_base_addr["remote_engine"] = {
                6666: [0x3000, 0x4000]
            }
            self.thread._transfer_kv_cache(self.test_req)
        self.engine.batch_transfer_sync_read.assert_called_once()
        call_args, call_kwargs = self.engine.batch_transfer_sync_read.call_args
        self.assertEqual(call_args[0], "localhost:7777")
        self.assertIsInstance(call_args[1], list)
        self.assertIsInstance(call_args[2], list)
        self.assertIsInstance(call_args[3], list)
        self.assertEqual(len(call_args[1]), len(call_args[2]))
        self.assertEqual(len(call_args[1]), len(call_args[3]))
        mock_get_meta.assert_not_called()

    def test_transfer_kv_cache_failure(self):
        self.engine.batch_transfer_sync_read.return_value = -1
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [0x3000, 0x4000]
        }

        with self.assertRaises(RuntimeError):
            self.thread._transfer_kv_cache(self.test_req)

    def test_ordinary_transfer_filters_remote_index_buffers(self):
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [0x3000, 0x9000, 0x4000]
        }
        self.thread.remote_buffer_group_ids["remote_engine"] = {
            6666: (0, 1, 0)
        }
        request = dict(
            self.test_req,
            local_block_ids=[0],
            remote_block_ids=[0],
            tp_num_need_pulls=1,
        )

        self.thread._transfer_kv_cache(request)

        self.engine.batch_transfer_sync_read.assert_called_once_with(
            "localhost:7777",
            [0x1000, 0x2000],
            [0x3000, 0x4000],
            [1024, 2048],
        )

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.temporary_registration'
    )
    def test_split_transfer_exact_cpu_npu_destinations(self, mock_register):
        mock_register.return_value = contextlib.nullcontext()
        self.thread.tp_rank = 7
        self.thread.tp_size = 8
        self.vllm_config.parallel_config.data_parallel_rank_local = 1
        self.vllm_config.parallel_config.data_parallel_index = 1
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [0x3000, 0x5000]
        }
        self.thread.remote_num_blocks["remote_engine"] = {6666: 2}
        self.thread.remote_buffer_sizes["remote_engine"] = {
            6666: (4096, 4096)
        }
        self.thread.remote_buffer_group_ids["remote_engine"] = {
            6666: (0, 1)
        }
        self.thread.remote_capabilities = {
            ("remote_engine", 6666): (LIVE_SPLIT_CAPABILITY,)
        }
        self.thread.kv_caches_base_addr["local_engine"][5555] = [
            0x7000, 0xA000
        ]
        self.thread.local_registered_bases = (0x7000, 0xA000)
        self.thread.local_buffer_sizes = (4096, 4096)
        self.thread.local_buffer_group_ids = (0, 1)
        plan = SplitTransferPlan(
            segments=(
                SplitTransferSegment(
                    0, 0, 32, 0x8000, 256, "cpu", 0x3000
                ),
                SplitTransferSegment(
                    1, 1, 64, 0xA000, 512, "npu", 0x5000
                ),
            ),
            group_byte_totals=(256, 512),
            tp_rank=7,
            dp_rank=1,
        )
        request = dict(self.test_req, split_plan=plan)

        self.thread._transfer_kv_cache(request)

        mock_register.assert_called_once_with(
            [0x8000, 0xA000],
            [256, 512],
            require_existing=False,
            adopted_only=False,
        )
        self.assertEqual(
            self.engine.batch_transfer_sync_read.call_args_list,
            [
                call("localhost:7777", [0x8000], [0x3000 + 32], [256]),
                call("localhost:7777", [0xA000], [0x5000 + 64], [512]),
            ],
        )

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.temporary_registration'
    )
    def test_hybrid_transfer_registers_page_envelopes_not_expanded_runs(
        self, mock_register
    ):
        mock_register.return_value = contextlib.nullcontext()
        self.thread.tp_rank = 0
        self.vllm_config.parallel_config.data_parallel_rank_local = 0
        self.vllm_config.parallel_config.data_parallel_index = 0
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [1000, 2000]
        }
        self.thread.remote_buffer_sizes["remote_engine"] = {
            6666: (128, 128)
        }
        self.thread.remote_buffer_group_ids["remote_engine"] = {
            6666: (0, 1)
        }
        self.thread.remote_capabilities = {
            ("remote_engine", 6666): (
                LIVE_SPLIT_CAPABILITY,
                LIVE_SPLIT_COMPACT_CAPABILITY,
                LIVE_SPLIT_LATENT_CPU_CAPABILITY,
            )
        }
        self.thread.kv_caches_base_addr["local_engine"][5555] = [5000, 6000]
        self.thread.local_registered_bases = (5000, 6000)
        self.thread.local_buffer_sizes = (128, 128)
        self.thread.local_buffer_group_ids = (0, 1)
        plan = SplitTransferPlan(
            segments=(),
            group_byte_totals=(8, 4),
            tp_rank=0,
            dp_rank=0,
            requested_groups=(0, 1),
            compact_source=SplitCompactLayout(
                1, 2, (SplitCompactLayer(0, 2000, 2, 8, 1),),
                (SplitCompactRun(0, 1, 2),),
            ),
            compact_destination=SplitCompactLayout(
                1, 2, (SplitCompactLayer(0, 6000, 2, 8),),
                (SplitCompactRun(0, 2, 2),),
            ),
            latent_source=SplitLatentLayout(
                0, 2, (SplitCompactLayer(0, 1000, 4, 8, 0),),
                (SplitLatentPage(
                    0, 2,
                    (SplitLatentRun(0, 1, 1), SplitLatentRun(1, 3, 1)),
                ),),
            ),
            latent_destination_pages=(
                SplitLatentDestinationPage(0, 10000, 8),
            ),
        )

        self.thread._transfer_split_destinations(self.test_req, plan)

        mock_register.assert_called_once_with(
            [10000], [8], require_existing=True, adopted_only=True
        )
        self.assertEqual(
            self.engine.batch_transfer_sync_read.call_args_list,
            [
                call(
                    "localhost:7777",
                    [10000, 10004],
                    [1004, 1012],
                    [4, 4],
                ),
                call("localhost:7777", [6004], [2002], [4]),
            ],
        )

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.temporary_registration'
    )
    def test_mixed_split_stops_before_group1_when_group0_fails(
        self, mock_register
    ):
        mock_register.return_value = contextlib.nullcontext()
        self.thread.tp_rank = 0
        self.vllm_config.parallel_config.data_parallel_rank_local = 0
        self.vllm_config.parallel_config.data_parallel_index = 0
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [0x3000, 0x5000]
        }
        self.thread.remote_buffer_sizes["remote_engine"] = {
            6666: (4096, 4096)
        }
        self.thread.remote_buffer_group_ids["remote_engine"] = {
            6666: (0, 1)
        }
        self.thread.remote_capabilities = {
            ("remote_engine", 6666): (LIVE_SPLIT_CAPABILITY,)
        }
        self.thread.local_registered_bases = (0x7000, 0xA000)
        self.thread.local_buffer_sizes = (4096, 4096)
        self.thread.local_buffer_group_ids = (0, 1)
        plan = SplitTransferPlan(
            segments=(
                SplitTransferSegment(
                    0, 0, 32, 0x8000, 256, "cpu", 0x3000
                ),
                SplitTransferSegment(
                    1, 1, 64, 0xA000, 512, "npu", 0x5000
                ),
            ),
            group_byte_totals=(256, 512),
            tp_rank=0,
            dp_rank=0,
        )
        self.engine.batch_transfer_sync_read.return_value = -1

        with self.assertRaisesRegex(RuntimeError, "failed for group 0"):
            self.thread._transfer_split_destinations(self.test_req, plan)

        self.engine.batch_transfer_sync_read.assert_called_once_with(
            "localhost:7777", [0x8000], [0x3000 + 32], [256]
        )

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.temporary_registration'
    )
    def test_compact_group1_registers_layer_envelopes_not_expanded_runs(
        self, mock_register
    ):
        mock_register.return_value = contextlib.nullcontext()
        self.thread.tp_rank = 0
        self.vllm_config.parallel_config.data_parallel_rank_local = 0
        self.vllm_config.parallel_config.data_parallel_index = 0
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [2000]
        }
        self.thread.remote_buffer_sizes["remote_engine"] = {
            6666: (128,)
        }
        self.thread.remote_buffer_group_ids["remote_engine"] = {
            6666: (1,)
        }
        self.thread.remote_capabilities = {
            ("remote_engine", 6666): (
                LIVE_SPLIT_CAPABILITY,
                LIVE_SPLIT_COMPACT_CAPABILITY,
            )
        }
        self.thread.local_registered_bases = (6000,)
        self.thread.local_buffer_sizes = (128,)
        self.thread.local_buffer_group_ids = (1,)
        plan = SplitTransferPlan(
            segments=(),
            group_byte_totals=(0, 8),
            tp_rank=0,
            dp_rank=0,
            requested_groups=(1,),
            compact_source=SplitCompactLayout(
                1, 4, (SplitCompactLayer(0, 2000, 2, 8, 0),),
                (
                    SplitCompactRun(0, 1, 2),
                    SplitCompactRun(2, 5, 2),
                ),
            ),
            compact_destination=SplitCompactLayout(
                1, 4, (SplitCompactLayer(0, 6000, 2, 8),),
                (
                    SplitCompactRun(0, 2, 2),
                    SplitCompactRun(2, 6, 2),
                ),
            ),
        )

        self.thread._transfer_split_destinations(self.test_req, plan)

        mock_register.assert_called_once_with(
            [6000], [16], require_existing=False, adopted_only=False
        )
        self.engine.batch_transfer_sync_read.assert_called_once_with(
            "localhost:7777",
            [6004, 6012],
            [2002, 2010],
            [4, 4],
        )

    def test_hybrid_expansion_matches_layer_page_byte_layout(self):
        widths = (2, 1, 3, 2)
        source_bases = (1000, 2000, 3000, 4000)
        logical_slots = (3, 4, 1, 6, 0)
        latent_layers = tuple(
            SplitCompactLayer(layer, base, width, 8, layer)
            for layer, (base, width) in enumerate(
                zip(source_bases, widths, strict=True)
            )
        )
        plan = SplitTransferPlan(
            segments=(),
            group_byte_totals=(sum(widths) * 5, 5),
            tp_rank=0,
            dp_rank=0,
            requested_groups=(0, 1),
            compact_source=SplitCompactLayout(
                1, 5, (SplitCompactLayer(0, 5000, 1, 8, 4),),
                (SplitCompactRun(0, 0, 5),),
            ),
            compact_destination=SplitCompactLayout(
                1, 5, (SplitCompactLayer(0, 6000, 1, 8),),
                (SplitCompactRun(0, 0, 5),),
            ),
            latent_source=SplitLatentLayout(
                0,
                5,
                latent_layers,
                (
                    SplitLatentPage(
                        0, 3,
                        (SplitLatentRun(0, 3, 2),
                         SplitLatentRun(2, 1, 1)),
                    ),
                    SplitLatentPage(
                        3, 2,
                        (SplitLatentRun(3, 6, 1),
                         SplitLatentRun(4, 0, 1)),
                    ),
                ),
            ),
            latent_destination_pages=(
                SplitLatentDestinationPage(0, 10000, sum(widths) * 3),
                SplitLatentDestinationPage(3, 11000, sum(widths) * 2),
            ),
        )

        expanded = self.thread._expand_compact_split_plan(plan)
        source_data = {}
        for layer, (base, width) in enumerate(
            zip(source_bases, widths, strict=True)
        ):
            source_data[base] = bytes(
                (layer * 40 + slot * 4 + byte) % 256
                for slot in range(8)
                for byte in range(width)
            )
        destinations = {10000: bytearray(sum(widths) * 3),
                        11000: bytearray(sum(widths) * 2)}
        for segment in expanded.segments:
            if segment.group_id != 0:
                continue
            page_base = 10000 if segment.destination_address < 11000 else 11000
            dst_offset = segment.destination_address - page_base
            src = source_data[segment.source_buffer_base]
            destinations[page_base][dst_offset:dst_offset + segment.length] = (
                src[segment.source_offset:segment.source_offset + segment.length]
            )

        for page_base, slots in ((10000, logical_slots[:3]),
                                 (11000, logical_slots[3:])):
            expected = bytearray()
            for base, width in zip(source_bases, widths, strict=True):
                for slot in slots:
                    start = slot * width
                    expected.extend(source_data[base][start:start + width])
            self.assertEqual(destinations[page_base], expected)

    def test_split_rejects_npu_destination_outside_local_group1(self):
        self.thread.local_registered_bases = (0x1000, 0x2000)
        self.thread.local_buffer_sizes = (0x100, 0x100)
        self.thread.local_buffer_group_ids = (0, 1)
        plan = SplitTransferPlan(
            segments=(
                SplitTransferSegment(1, 0, 0, 0x3000, 8, "npu"),
            ),
            group_byte_totals=(0, 8),
            tp_rank=0,
            dp_rank=0,
            requested_groups=(1,),
        )

        with self.assertRaisesRegex(RuntimeError, "registered group-1 KV"):
            self.thread._transfer_split_destinations(self.test_req, plan)

    def test_split_transfer_rejects_wrong_dp_rank(self):
        self.vllm_config.parallel_config.data_parallel_rank_local = 1
        self.vllm_config.parallel_config.data_parallel_index = 1
        plan = SplitTransferPlan(
            segments=(
                SplitTransferSegment(0, 0, 0, 0x8000, 1, "cpu"),
                SplitTransferSegment(1, 1, 0, 0xA000, 1, "npu"),
            ),
            group_byte_totals=(1, 1),
            tp_rank=0,
            dp_rank=0,
        )

        with self.assertRaisesRegex(RuntimeError, "TP/DP rank mismatch"):
            self.thread._transfer_kv_cache(dict(self.test_req,
                                                split_plan=plan))

    def test_split_transfer_rejects_wrong_remote_source_identity(self):
        self.thread.tp_rank = 0
        self.vllm_config.parallel_config.data_parallel_index = 1
        self.thread.kv_caches_base_addr["remote_engine"] = {6666: [0x5000]}
        self.thread.remote_capabilities = {
            ("remote_engine", 6666): (
                LIVE_SPLIT_CAPABILITY,
                LIVE_SPLIT_DP_ROUTING_CAPABILITY,
            )
        }
        self.thread.remote_rank_identities[("remote_engine", 6666)] = (0, 1)
        plan = SplitTransferPlan(
            segments=(),
            group_byte_totals=(0, 0),
            tp_rank=0,
            dp_rank=1,
            requested_groups=(1,),
            source_tp_rank=0,
            source_dp_rank=0,
        )

        with self.assertRaisesRegex(RuntimeError, "split source"):
            self.thread._transfer_split_destinations(self.test_req, plan)

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.temporary_registration'
    )
    def test_index_only_group1_allows_zero_unrequested_group0(
        self, mock_register
    ):
        mock_register.return_value = contextlib.nullcontext()
        self.vllm_config.parallel_config.data_parallel_rank_local = 0
        self.vllm_config.parallel_config.data_parallel_index = 0
        self.thread.kv_caches_base_addr["remote_engine"] = {6666: [0x5000]}
        self.thread.remote_num_blocks["remote_engine"] = {6666: 2}
        self.thread.remote_buffer_sizes["remote_engine"] = {6666: (4096,)}
        self.thread.remote_buffer_group_ids["remote_engine"] = {6666: (1,)}
        self.thread.remote_capabilities = {
            ("remote_engine", 6666): (LIVE_SPLIT_CAPABILITY,)
        }
        self.thread.kv_caches_base_addr["local_engine"][5555] = [0xA000]
        self.thread.local_registered_bases = (0xA000,)
        self.thread.local_buffer_sizes = (4096,)
        self.thread.local_buffer_group_ids = (1,)
        plan = SplitTransferPlan(
            segments=(
                SplitTransferSegment(1, 0, 64, 0xA000, 512, "npu"),
            ),
            group_byte_totals=(0, 512),
            tp_rank=0,
            dp_rank=0,
            requested_groups=(1,),
        )

        self.thread._transfer_kv_cache(
            dict(self.test_req, split_plan=plan))

        mock_register.assert_called_once_with(
            [0xA000],
            [512],
            require_existing=False,
            adopted_only=False,
        )
        self.engine.batch_transfer_sync_read.assert_called_once_with(
            "localhost:7777", [0xA000], [0x5000 + 64], [512])

    def test_split_rejects_source_group_mismatch(self):
        self.thread.local_registered_bases = (0xA000,)
        self.thread.local_buffer_sizes = (4096,)
        self.thread.local_buffer_group_ids = (1,)
        self.thread.kv_caches_base_addr["remote_engine"] = {6666: [0x5000]}
        self.thread.remote_buffer_sizes["remote_engine"] = {6666: (4096,)}
        self.thread.remote_buffer_group_ids["remote_engine"] = {6666: (0,)}
        self.thread.remote_capabilities = {
            ("remote_engine", 6666): (LIVE_SPLIT_CAPABILITY,)
        }
        plan = SplitTransferPlan(
            segments=(
                SplitTransferSegment(1, 0, 64, 0xA000, 512, "npu"),
            ),
            group_byte_totals=(0, 512),
            tp_rank=0,
            dp_rank=0,
            requested_groups=(1,),
        )

        with self.assertRaisesRegex(RuntimeError, "source buffer group"):
            self.thread._transfer_split_destinations(self.test_req, plan)

    def test_split_rejects_overlapping_destinations(self):
        plan = SplitTransferPlan(
            segments=(
                SplitTransferSegment(1, 0, 0, 0xA000, 16, "npu"),
                SplitTransferSegment(1, 0, 16, 0xA008, 16, "npu"),
            ),
            group_byte_totals=(0, 32),
            tp_rank=0,
            dp_rank=0,
            requested_groups=(1,),
        )

        with self.assertRaisesRegex(RuntimeError, "destination extents overlap"):
            self.thread._transfer_split_destinations(self.test_req, plan)

    def test_split_rejects_zero_or_missing_requested_group(self):
        self.vllm_config.parallel_config.data_parallel_rank_local = 0
        self.vllm_config.parallel_config.data_parallel_index = 0
        self.thread.local_registered_bases = (0xA000,)
        self.thread.local_buffer_sizes = (4096,)
        self.thread.local_buffer_group_ids = (1,)
        zero_group1 = SplitTransferPlan(
            segments=(),
            group_byte_totals=(0, 0),
            tp_rank=0,
            dp_rank=0,
            requested_groups=(1,),
        )
        with self.assertRaisesRegex(RuntimeError, "requested split group"):
            self.thread._transfer_split_destinations(
                self.test_req, zero_group1)

        missing_group0 = SplitTransferPlan(
            segments=(
                SplitTransferSegment(1, 0, 0, 0xA000, 8, "npu"),
            ),
            group_byte_totals=(0, 8),
            tp_rank=0,
            dp_rank=0,
            requested_groups=(0, 1),
        )
        with self.assertRaisesRegex(RuntimeError, "requested split group"):
            self.thread._transfer_split_destinations(
                self.test_req, missing_group0)

    def test_split_queue_backpressure_uses_existing_queue(self):
        self.thread.request_queue = queue.Queue()
        for item in range(64):
            self.thread.request_queue.put(item)
        plan = SplitTransferPlan((), (0, 0), tp_rank=0, dp_rank=0)

        admitted = self.thread.add_request(
            request_id="overflow",
            remote_request_id="remote",
            local_block_ids=[],
            remote_block_ids=[],
            remote_engine_id="remote_engine",
            remote_host="localhost",
            remote_handshake_port=6666,
            offset=0,
            tp_num_need_pulls=1,
            split_plan=plan,
            split_transfer_id="overflow-generation",
        )

        self.assertFalse(admitted)
        self.assertEqual(self.thread.request_queue.qsize(), 64)

    def test_split_and_ordinary_requests_keep_fifo_order(self):
        self.thread.request_queue = queue.Queue()
        common = dict(
            local_block_ids=[],
            remote_block_ids=[],
            remote_engine_id="remote_engine",
            remote_host="localhost",
            remote_handshake_port=6666,
            offset=0,
            tp_num_need_pulls=1,
        )
        self.thread.add_request("ordinary", "ordinary-remote", **common)
        plan = SplitTransferPlan((), (0, 0), tp_rank=0, dp_rank=0)
        self.thread.add_request(
            "split", "split-remote", split_plan=plan,
            split_transfer_id="split-generation", **common)

        self.assertEqual(self.thread.request_queue.get()["request_id"],
                         "ordinary")
        self.assertEqual(self.thread.request_queue.get()["request_id"],
                         "split")

    def test_duplicate_split_generation_is_queued_once(self):
        self.thread.request_queue = queue.Queue()
        plan = SplitTransferPlan((), (0, 0), tp_rank=0, dp_rank=0)
        common = dict(
            remote_request_id="remote",
            local_block_ids=[],
            remote_block_ids=[],
            remote_engine_id="remote_engine",
            remote_host="localhost",
            remote_handshake_port=6666,
            offset=0,
            tp_num_need_pulls=1,
            split_plan=plan,
            split_transfer_id="generation-a",
        )

        self.assertTrue(self.thread.add_request("req", **common))
        self.assertTrue(self.thread.add_request("req", **common))

        self.assertEqual(self.thread.request_queue.qsize(), 1)

    def test_completed_split_generation_is_not_queued_again(self):
        self.thread.request_queue = queue.Queue()
        plan = SplitTransferPlan((), (0, 0), tp_rank=0, dp_rank=0)
        self.thread.completed_split_requests[("req", "generation-a")] = (
            "success"
        )

        admitted = self.thread.add_request(
            request_id="req",
            remote_request_id="remote",
            local_block_ids=[],
            remote_block_ids=[],
            remote_engine_id="remote_engine",
            remote_host="localhost",
            remote_handshake_port=6666,
            offset=0,
            tp_num_need_pulls=1,
            split_plan=plan,
            split_transfer_id="generation-a",
        )

        self.assertTrue(admitted)
        self.assertTrue(self.thread.request_queue.empty())
        self.thread.task_tracker.complete_split_request.assert_not_called()

    def test_stop_rejects_new_queue_admission(self):
        self.thread.request_queue = queue.Queue()

        self.thread.stop()

        self.assertFalse(
            self.thread.add_request(
                request_id="req",
                remote_request_id="remote",
                local_block_ids=[],
                remote_block_ids=[],
                remote_engine_id="remote_engine",
                remote_host="localhost",
                remote_handshake_port=6666,
                offset=0,
                tp_num_need_pulls=1,
            )
        )
        self.assertIsNone(self.thread.request_queue.get_nowait())

    def test_split_completion_signal_is_submitted_once(self):
        self.thread._submit_split_done_signal = MagicMock()

        self.thread._signal_split_completion(
            "local", "remote", "host", 6666, "success", "generation-a"
        )
        self.thread._signal_split_completion(
            "local", "remote", "host", 6666, "success", "generation-a"
        )

        self.thread._submit_split_done_signal.assert_called_once_with(
            ("local", "generation-a"),
            "remote", "host", 6666, "success", "generation-a"
        )

    @patch.object(KVCacheRecvingThread, '_transfer_kv_cache')
    def test_cancelled_split_is_acked_without_transfer(self, mock_transfer):
        plan = SplitTransferPlan((), (0, 0), tp_rank=0, dp_rank=0)
        request = dict(
            self.test_req,
            split_plan=plan,
            split_transfer_id="generation-a",
        )
        self.thread._signal_split_completion = MagicMock()
        self.thread.active_split_requests["req1"] = "generation-a"
        self.assertTrue(self.thread.cancel_split_request("req1"))

        self.thread._handle_request(request)

        mock_transfer.assert_not_called()
        self.thread._signal_split_completion.assert_called_once_with(
            "req1", "req1", "localhost", 6666, "cancelled", "generation-a"
        )
        self.thread.task_tracker.complete_split_request.assert_called_once_with(
            "req1", "cancelled", mark_finished=False,
            split_transfer_id="generation-a")
        self.assertNotIn("req1", self.thread.active_split_requests)

    @patch.object(KVCacheRecvingThread, '_transfer_kv_cache')
    def test_successful_split_waits_for_provider_completion(self, mock_transfer):
        plan = SplitTransferPlan((), (0, 0), tp_rank=0, dp_rank=0)
        request = dict(
            self.test_req,
            split_plan=plan,
            split_transfer_id="generation-a",
        )
        self.thread._signal_split_completion = MagicMock()
        self.thread.active_split_requests["req1"] = "generation-a"

        self.thread._handle_request(request)

        mock_transfer.assert_called_once_with(request)
        self.thread._signal_split_completion.assert_called_once_with(
            "req1", "req1", "localhost", 6666, "success", "generation-a"
        )
        self.thread.task_tracker.complete_split_request.assert_called_once_with(
            "req1", "success", mark_finished=False,
            split_transfer_id="generation-a")


class TestMetadataHandling(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.vllm_config = MockVllmConfig()
        self.kv_caches: Dict[str, Any] = {}
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            _prefill_pp_size=1,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            side_channel_port=30000,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event,
            vllm_config=self.vllm_config,
            kv_caches=self.kv_caches,
            prefill_pp_layer_partition=None)
        self.test_metadata = MooncakeAgentMetadata(
            engine_id="remote_engine",
            te_rpc_port=9090,
            kv_caches_base_addr=[0x3000, 0x4000],
            num_blocks=2)

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.ensure_zmq_send'
    )
    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.ensure_zmq_recv'
    )
    def test_get_remote_metadata_success(self, mock_recv, mock_send):
        mock_recv.return_value = msgspec.msgpack.encode(self.test_metadata)

        with patch.object(self.thread, '_get_remote_socket') as mock_get_socket, \
                patch.object(self.thread, '_return_remote_socket') as mock_return_socket:
            mock_socket = MagicMock()
            mock_get_socket.return_value = mock_socket

            self.thread._get_remote_metadata("host1", 5555)

            mock_get_socket.assert_called_once_with("host1", 5555)
            mock_return_socket.assert_called_once_with(mock_socket, "host1",
                                                       5555)
            mock_send.assert_called_once_with(
                mock_socket, self.thread.encoder.encode((GET_META_MSG, "")))
            mock_recv.assert_called_once_with(mock_socket,
                                              self.thread.remote_poller)
            self.assertEqual(
                self.thread.kv_caches_base_addr["remote_engine"][5555],
                [0x3000, 0x4000])
            self.assertEqual(
                self.thread.remote_rank_identities[("remote_engine", 5555)],
                (0, 0),
            )

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.ensure_zmq_send'
    )
    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.ensure_zmq_recv',
        side_effect=Exception("Network error"))
    def test_get_remote_metadata_failure(self, mock_recv, mock_send):
        with patch.object(self.thread, '_get_remote_socket') as mock_get_socket, \
                patch.object(self.thread, '_return_remote_socket') as mock_return_socket, \
                patch.object(self.thread, '_discard_remote_socket') as mock_discard_socket:
            mock_socket = MagicMock()
            mock_get_socket.return_value = mock_socket

            with self.assertRaises(Exception) as context:
                self.thread._get_remote_metadata("host1", 5555)

            self.assertEqual(str(context.exception), "Network error")
            mock_return_socket.assert_not_called()
            mock_discard_socket.assert_called_once_with(mock_socket)


class TestMainThreadLoop(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.vllm_config = MockVllmConfig()
        self.kv_caches: Dict[str, Any] = {}
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            _prefill_pp_size=1,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            side_channel_port=30000,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event,
            vllm_config=self.vllm_config,
            kv_caches=self.kv_caches,
            prefill_pp_layer_partition=None)
        self.thread.request_queue = queue.Queue()

    @patch.object(KVCacheRecvingThread, '_handle_request')
    def test_run_loop_normal(self, mock_handle):
        test_request = {
            "request_id": "req1",
            "local_block_ids": [1, 2],
            "remote_block_ids": [3, 4],
            "remote_engine_id": "remote_engine",
            "remote_host": "localhost",
            "remote_handshake_port": 6666,
            "remote_transfer_port": 7777,
            "offset": 0,
            "tp_num_need_pulls": 2,
            "all_task_done": False
        }

        self.thread.request_queue.put(test_request)
        self.thread.request_queue.put(None)

        self.thread.start()
        time.sleep(0.1)
        self.thread.join(timeout=1.0)

        self.assertTrue(self.thread.ready_event.is_set())
        mock_handle.assert_called_once_with(test_request)
        self.assertTrue(self.thread.request_queue.empty())


class MockVllmConfig:

    def __init__(self):
        self.model_config = MagicMock()
        self.parallel_config = MagicMock()
        self.cache_config = MagicMock()
        self.kv_transfer_config = MagicMock()
        self.speculative_config = MagicMock()
        self.model_config.use_mla = True
        self.parallel_config.tensor_parallel_size = 2
        self.parallel_config.data_parallel_rank = 0
        self.parallel_config.data_parallel_index = 0
        self.parallel_config.data_parallel_size = 1
        self.parallel_config.data_parallel_size_local = 1
        self.parallel_config.pipeline_parallel_size = 1
        self.parallel_config.prefill_context_parallel_size = 1
        self.parallel_config.decode_context_parallel_size = 1
        self.parallel_config.data_parallel_rank_local = 0
        self.model_config.get_num_layers_by_block_type = MagicMock(
            return_value=32)
        self.cache_config.block_size = 16
        self.kv_transfer_config.kv_port = 5000
        self.kv_transfer_config.kv_role = 'kv_producer'
        self.kv_transfer_config.get_from_extra_config = MagicMock()
        self.kv_transfer_config.get_from_extra_config.side_effect = lambda k, d: {
            "prefill": {
                "tp_size": 2,
                "dp_size": 1,
                "pp_size": 1
            },
            "decode": {
                "tp_size": 2,
                "dp_size": 1,
                "pp_size": 1
            }
        }.get(k, d)
        self.additional_config = {}


class MockRequest:

    def __init__(self,
                 request_id,
                 prompt_token_ids=None,
                 kv_transfer_params=None,
                 status=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids or [1, 2, 3, 4]
        self.kv_transfer_params = kv_transfer_params or {}
        self.status = status or "running"
        self.output_token_ids = [101, 102]


class TestKVCacheTaskTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = KVCacheTaskTracker()

    def test_update_done_task_count(self):
        self.assertEqual(len(self.tracker.finished_requests), 0)
        self.assertEqual(len(self.tracker.delayed_free_requests), 0)
        self.assertEqual(len(self.tracker.record_finished_requests), 0)

        current_time = time.time()
        self.tracker.add_delayed_request("req_1", current_time)
        result = self.tracker.delayed_free_requests
        result_record = self.tracker.record_finished_requests
        self.assertEqual(len(result), 1)
        self.assertEqual(result["req_1"], current_time)
        self.assertEqual(len(result_record), 0)

        self.tracker.update_done_task_count("req_1")
        result_finished = self.tracker.finished_requests
        result_delayed = self.tracker.delayed_free_requests
        result_record = self.tracker.record_finished_requests
        self.assertEqual(result_finished, {"req_1"})
        self.assertEqual(len(result_delayed), 0)
        self.assertEqual(len(result_record), 0)

        self.tracker.update_done_task_count("req_2")
        result_finished = self.tracker.finished_requests
        result_delayed = self.tracker.delayed_free_requests
        result_record = self.tracker.record_finished_requests
        self.assertEqual(result_finished, {"req_1", "req_2"})
        self.assertEqual(len(result_delayed), 0)
        self.assertEqual(len(result_record), 1)
        self.assertEqual(result_record, {"req_2"})

    def test_updtate_add_delayed_request(self) -> None:
        self.tracker.update_done_task_count("req2")
        result_start_record = self.tracker.record_finished_requests
        self.assertEqual(len(result_start_record), 1)
        self.tracker.add_delayed_request("req2", time.time())
        result_delayed = self.tracker.delayed_free_requests
        result_end_record = self.tracker.record_finished_requests
        self.assertEqual(len(result_delayed), 0)
        self.assertEqual(len(result_end_record), 0)

    def test_retrieve_expired_requests(self):
        current_time = time.time()
        self.tracker.add_delayed_request("req_1", current_time - 600)
        self.tracker.add_delayed_request("req_2", current_time)
        result = self.tracker._retrieve_expired_requests()
        self.assertEqual(result, {
            "req_1",
        })
        result_delay = self.tracker.delayed_free_requests
        self.assertEqual(len(result_delay), 1)
        self.assertIn("req_2", result_delay)

    def test_duplicate_task_update(self):
        self.tracker.update_done_task_count("req1")
        self.tracker.update_done_task_count("req1")
        self.tracker.update_done_task_count("req1")

        finished = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(finished, {"req1"})

    def test_split_ack_is_idempotent_and_releases_lease(self):
        self.tracker.add_req_to_process("req")
        self.tracker.add_delayed_request("req", time.time(), split=True)
        self.tracker.complete_split_request("req", "success")
        self.tracker.complete_split_request("req", "failure")

        self.assertEqual(self.tracker.get_and_clear_split_results(),
                         {"req": "success"})
        self.assertNotIn("req", self.tracker.delayed_free_requests)
        self.assertNotIn("req", self.tracker.reqs_to_process)

    def test_split_lease_timeout_keeps_native_source_pinned(self):
        self.tracker.add_req_to_process("req")
        self.tracker.add_delayed_request(
            "req", time.time() - 600, split=True)

        self.assertEqual(self.tracker._retrieve_expired_requests(), set())
        self.assertIn("req", self.tracker.delayed_free_requests)
        self.assertIn("req", self.tracker.split_leases)

        self.tracker.complete_split_request("req", "success")

        self.assertEqual(self.tracker.get_and_clear_split_results(),
                         {"req": "success"})
        self.assertEqual(self.tracker.get_and_clear_finished_requests(),
                         {"req"})

    def test_expired_split_lease_does_not_block_ordinary_expiry(self):
        expired = time.time() - 600
        for request_id in ("split", "ordinary", "fresh"):
            self.tracker.add_req_to_process(request_id)
        self.tracker.add_delayed_request("split", expired, split=True)
        self.tracker.add_delayed_request("ordinary", expired)
        self.tracker.add_delayed_request("fresh", time.time())

        self.assertEqual(
            self.tracker._retrieve_expired_requests(), {"ordinary"}
        )
        self.assertEqual(
            tuple(self.tracker.delayed_free_requests), ("split", "fresh")
        )
        self.assertIn("split", self.tracker.split_leases)

    def test_reused_split_request_clears_both_terminal_generations(self):
        for mark_finished in (True, False):
            with self.subTest(mark_finished=mark_finished):
                tracker = KVCacheTaskTracker()
                tracker.add_req_to_process("req")
                tracker.add_delayed_request("req", time.time(), split=True)
                tracker.complete_split_request(
                    "req", "success", mark_finished=mark_finished
                )

                tracker.add_req_to_process("req")
                self.assertNotIn("req", tracker.finished_requests)
                self.assertNotIn("req", tracker.split_terminal_requests)
                self.assertNotIn("req", tracker.split_results)
                tracker.add_delayed_request("req", time.time(), split=True)
                tracker.complete_split_request(
                    "req", "failure", mark_finished=mark_finished
                )

                self.assertEqual(
                    tracker.get_and_clear_split_results(), {"req": "failure"}
                )

    def test_late_split_ack_cannot_complete_reused_request(self):
        self.tracker.add_req_to_process("req", "generation-a")
        self.tracker.add_delayed_request(
            "req", time.time(), split=True,
            split_transfer_id="generation-a")
        self.assertTrue(self.tracker.complete_split_request(
            "req", "success", split_transfer_id="generation-a"))

        self.tracker.add_req_to_process("req", "generation-b")
        self.tracker.add_delayed_request(
            "req", time.time(), split=True,
            split_transfer_id="generation-b")

        self.assertFalse(self.tracker.complete_split_request(
            "req", "success", split_transfer_id="generation-a"))
        self.assertFalse(self.tracker.complete_split_request(
            "req", "success"))
        self.assertIn("req", self.tracker.split_leases)
        self.assertTrue(self.tracker.complete_split_request(
            "req", "failure", split_transfer_id="generation-b"))
        self.assertEqual(
            self.tracker.get_and_clear_split_results(), {"req": "failure"})

    def test_split_ack_before_lease_install_is_applied_once(self):
        self.assertTrue(self.tracker.complete_split_request(
            "req", "success", split_transfer_id="generation-a"))

        self.tracker.add_req_to_process("req", "generation-a")
        self.tracker.add_delayed_request(
            "req", time.time(), split=True,
            split_transfer_id="generation-a")

        self.assertEqual(
            self.tracker.get_and_clear_split_results(), {"req": "success"})
        self.assertEqual(self.tracker.get_and_clear_finished_requests(),
                         {"req"})
        self.assertNotIn("req", self.tracker.split_leases)

    def test_reused_request_accepts_next_generation_early_ack(self):
        self.tracker.add_req_to_process("req", "generation-a")
        self.tracker.add_delayed_request(
            "req", time.time(), split=True,
            split_transfer_id="generation-a")
        self.tracker.complete_split_request(
            "req", "success", split_transfer_id="generation-a")
        self.tracker.get_and_clear_split_results()

        self.tracker.complete_split_request(
            "req", "failure", split_transfer_id="generation-b")
        self.tracker.add_req_to_process("req", "generation-b")
        self.tracker.add_delayed_request(
            "req", time.time(), split=True,
            split_transfer_id="generation-b")

        self.assertEqual(
            self.tracker.get_and_clear_split_results(), {"req": "failure"})
        self.assertNotIn("req", self.tracker.split_leases)

    def test_receive_admission_precedes_immediate_split_completion(self):
        tracker = KVCacheTaskTracker()

        class ImmediateThread:
            task_tracker = tracker

            def add_request(self, request_id, **_kwargs):
                self.assert_admitted = request_id in tracker.reqs_to_process
                tracker.complete_split_request(
                    request_id,
                    "success",
                    mark_finished=False,
                    split_transfer_id=_kwargs.get("split_transfer_id"),
                )
                return True

        recv_thread = ImmediateThread()
        send_thread = MagicMock()
        worker = object.__new__(MooncakeConnectorWorker)
        worker.kv_recv_thread = recv_thread
        worker.kv_send_thread = send_thread
        worker._prefill_tp_size = 1
        worker._prefill_pp_size = 1
        worker.pcp_size = 1
        worker.dcp_size = 1
        worker.tp_rank = 0
        worker._get_tp_num_need_pulls = lambda _size: 1
        worker._get_remote_rank = lambda _request_id, _size: [0]
        worker._prefill_get_remote_rank = lambda _request_id: [0]
        worker._get_remote_host_info_by_port = (
            lambda _base, _port, host, engine, _mapping: (host, engine)
        )
        metadata = MooncakeConnectorMetadata()
        metadata.reqs_in_batch.add("req")
        metadata.requests["req"] = ReqMeta(
            local_block_ids=[1],
            num_external_tokens=1,
            remote_block_ids=[2],
            remote_host="host",
            remote_port=30000,
            remote_engine_id="engine",
            remote_request_id="remote-req",
            remote_pcp_size=1,
            remote_dcp_size=1,
            remote_ptp_size=1,
            remote_multi_nodes_meta_mapping={},
            num_prompt_blocks=1,
            split_plan=SplitTransferPlan((), (1, 1), 0, 0),
            split_negotiated=True,
            split_transfer_id="generation-a",
        )

        worker.start_load_kv(metadata)

        self.assertTrue(recv_thread.assert_admitted)
        self.assertEqual(
            tracker.get_and_clear_split_results(), {"req": "success"}
        )
        send_thread.task_tracker.add_req_to_process.assert_not_called()

    def test_both_role_enrolls_only_send_metadata_in_sender(self):
        worker = object.__new__(MooncakeConnectorWorker)
        worker.kv_send_thread = MagicMock()
        worker.kv_recv_thread = MagicMock()
        worker.pcp_size = 1
        worker.dcp_size = 1
        worker.tp_rank = 0
        worker._prefill_get_remote_rank = lambda _request_id: [0]
        metadata = MooncakeConnectorMetadata()
        metadata.reqs_in_batch.update(("send", "unrelated"))
        metadata.requests_to_send["send"] = 1.0

        worker.start_load_kv(metadata)

        worker.kv_send_thread.task_tracker.add_req_to_process.assert_called_once_with(
            "send", None
        )
        worker.kv_send_thread.add_delayed_request.assert_called_once_with(
            "send", 1.0, split=False, split_transfer_id=None
        )
        worker.kv_recv_thread.task_tracker.add_req_to_process.assert_not_called()


class TestGlobalTransferEngineRegistration(unittest.TestCase):

    def test_additional_and_contained_regions_are_safe(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0

        registry.register_buffer([0x1000], [0x1000])
        registry.register_buffer([0x1400, 0x4000], [0x100, 0x200])

        self.assertEqual(
            registry.transfer_engine.register_memory.call_args_list,
            [unittest.mock.call(0x1000, 0x1000),
             unittest.mock.call(0x4000, 0x200)])

    def test_partial_registration_overlap_is_rejected(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0
        registry.register_buffer([0x1000], [0x1000])

        with self.assertRaisesRegex(RuntimeError, "partially overlaps"):
            registry.register_buffer([0x1800], [0x1000])

    def test_nested_regions_in_one_batch_register_containing_region(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0

        registry.register_buffer(
            [0x1000, 0x1800, 0x1000],
            [0x800, 0x800, 0x1000],
        )

        registry.transfer_engine.register_memory.assert_called_once_with(
            0x1000, 0x1000
        )
        self.assertEqual(registry.registered_buffers, {0x1000: 0x1000})

    def test_failed_persistent_batch_rolls_back_new_regions(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.side_effect = [0, -1]
        registry.transfer_engine.unregister_memory.return_value = 0

        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            registry.register_buffer([0x1000, 0x3000], [0x100, 0x100])

        registry.transfer_engine.unregister_memory.assert_called_once_with(
            0x1000
        )
        self.assertEqual(registry.registered_buffers, {})

    def test_failed_persistent_batch_retains_failed_rollback(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.side_effect = [0, -1]
        registry.transfer_engine.unregister_memory.return_value = -1

        with self.assertRaisesRegex(RuntimeError, "rollback failed"):
            registry.register_buffer([0x1000, 0x3000], [0x100, 0x100])

        self.assertEqual(registry.registered_buffers, {0x1000: 0x100})

    def test_temporary_registration_releases_and_allows_address_reuse(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0
        registry.transfer_engine.unregister_memory.return_value = 0

        with registry.temporary_registration([0x8000], [0x100]):
            self.assertEqual(registry.registered_buffers[0x8000], 0x100)
        with registry.temporary_registration([0x8000], [0x200]):
            self.assertEqual(registry.registered_buffers[0x8000], 0x200)

        self.assertNotIn(0x8000, registry.registered_buffers)
        self.assertEqual(
            registry.transfer_engine.unregister_memory.call_args_list,
            [unittest.mock.call(0x8000), unittest.mock.call(0x8000)],
        )

    def test_duplicate_temporary_region_unregisters_once(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0
        registry.transfer_engine.unregister_memory.return_value = 0

        with registry.temporary_registration(
            [0x8000, 0x8000], [0x100, 0x100]
        ):
            self.assertEqual(registry._temporary_refcounts[0x8000], 2)

        registry.transfer_engine.unregister_memory.assert_called_once_with(
            0x8000
        )
        self.assertEqual(registry.registered_buffers, {})

    def test_temporary_registration_does_not_release_persistent_buffer(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0
        registry.register_buffer([0x1000], [0x1000])

        with registry.temporary_registration([0x1400], [0x100]):
            pass

        registry.transfer_engine.unregister_memory.assert_not_called()
        self.assertEqual(registry.registered_buffers, {0x1000: 0x1000})

    def test_adopted_registration_covers_temporary_pages(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()

        self.assertTrue(registry.adopt_registered_buffer(0x1000, 0x1000))
        with registry.temporary_registration([0x1400], [0x100]):
            pass
        registry.release_adopted_buffer(0x1000, 0x1000)

    def test_adopted_registration_uses_one_lease_for_many_pages(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.adopt_registered_buffer(0x1000, 0x1000)

        with registry.temporary_registration(
            [0x1100, 0x1200, 0x1300],
            [0x80, 0x80, 0x80],
            require_existing=True,
            adopted_only=True,
        ):
            self.assertEqual(registry._adopted_leases, {0x1000: 1})

        self.assertEqual(registry._adopted_leases, {})
        registry.release_adopted_buffer(0x1000, 0x1000)

        registry.transfer_engine.register_memory.assert_not_called()
        registry.transfer_engine.unregister_memory.assert_not_called()
        self.assertEqual(registry.registered_buffers, {})

    def test_required_adopted_registration_rejects_unowned_region(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "existing registration"):
            with registry.temporary_registration(
                [0x1400],
                [0x100],
                require_existing=True,
                adopted_only=True,
            ):
                pass

        registry.transfer_engine.register_memory.assert_not_called()

    def test_required_adopted_registration_rejects_native_owner(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0
        registry.register_buffer([0x1000], [0x1000])

        with self.assertRaisesRegex(RuntimeError, "not an adopted"):
            with registry.temporary_registration(
                [0x1400],
                [0x100],
                require_existing=True,
                adopted_only=True,
            ):
                pass

    def test_adopted_registration_cannot_release_during_transfer(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.adopt_registered_buffer(0x1000, 0x1000)

        with (
            registry.temporary_registration([0x1400], [0x100]),
            self.assertRaisesRegex(RuntimeError, "in use"),
        ):
            registry.release_adopted_buffer(0x1000, 0x1000)

        registry.release_adopted_buffer(0x1000, 0x1000)
        self.assertEqual(registry.registered_buffers, {})

    def test_native_owner_registration_is_not_adopted(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0
        registry.register_buffer([0x1000], [0x1000])

        with self.assertRaisesRegex(RuntimeError, "another owner"):
            registry.adopt_registered_buffer(0x1000, 0x1000)

        self.assertEqual(registry.registered_buffers, {0x1000: 0x1000})

    def test_adopted_unregister_keeps_region_visible_until_native_release(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.adopt_registered_buffer(0x1000, 0x1000)

        def unregister():
            self.assertEqual(registry.registered_buffers, {0x1000: 0x1000})
            return 0

        registry.release_adopted_buffer(0x1000, 0x1000, unregister)

        self.assertEqual(registry.registered_buffers, {})

    def test_temporary_unregister_failure_releases_adopted_lease(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0
        registry.transfer_engine.unregister_memory.return_value = -1
        registry.adopt_registered_buffer(0x1000, 0x1000)

        with (
            self.assertRaisesRegex(RuntimeError, "unregistration failed"),
            registry.temporary_registration(
                [0x1400, 0x3000], [0x100, 0x100]
            ),
        ):
            pass

        self.assertEqual(registry._adopted_leases, {})

    def test_temporary_unregister_attempts_every_region(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0
        registry.transfer_engine.unregister_memory.side_effect = [0, -1]

        with (
            self.assertRaisesRegex(RuntimeError, "1 region"),
            registry.temporary_registration(
                [0x3000, 0x5000], [0x100, 0x100]
            ),
        ):
            pass

        self.assertEqual(
            registry.transfer_engine.unregister_memory.call_count, 2
        )
        self.assertEqual(len(registry.registered_buffers), 1)


class TestControlAcknowledgement(unittest.TestCase):

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p."
        "mooncake_connector.time.sleep"
    )
    def test_router_ack_retry_is_bounded(self, mock_sleep):
        sock = MagicMock()
        sock.send_multipart.side_effect = zmq.Again()

        self.assertFalse(_send_router_ack(sock, b"peer", "req"))

        self.assertEqual(sock.send_multipart.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)


class TestMooncakeConnectorMetadata(unittest.TestCase):

    def test_add_new_req(self):
        meta = MooncakeConnectorMetadata()
        self.assertEqual(len(meta.requests), 0)
        self.assertEqual(len(meta.requests_to_send), 0)

        meta.add_new_req(request_id="req1",
                         local_block_ids=[1, 2, 3],
                         num_external_tokens=48,
                         kv_transfer_params={
                             "remote_block_ids": [4, 5, 6],
                             "remote_engine_id": "remote_engine",
                             "remote_host": "localhost",
                             "remote_port": 5000,
                             "remote_pcp_size": 1,
                             "remote_dcp_size": 1,
                             "remote_ptp_size": 2
                         })

        self.assertEqual(len(meta.requests), 1)
        req_meta = meta.requests["req1"]
        self.assertIsInstance(req_meta, ReqMeta)
        self.assertEqual(req_meta.local_block_ids, [1, 2, 3])
        self.assertEqual(req_meta.remote_block_ids, [4, 5, 6])
        self.assertEqual(req_meta.remote_engine_id, "remote_engine")
        self.assertEqual(req_meta.remote_host, "localhost")
        self.assertEqual(req_meta.remote_port, 5000)
        self.assertEqual(req_meta.remote_ptp_size, 2)

    def test_negotiated_split_without_nonce_falls_back(self):
        meta = MooncakeConnectorMetadata()

        meta.add_new_req(
            request_id="req",
            local_block_ids=[1],
            num_external_tokens=16,
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 1234,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
            },
        )

        self.assertTrue(meta.requests["req"].split_fallback)
        self.assertIsNone(meta.requests["req"].split_transfer_id)

    def test_add_split_destination_plan(self):
        meta = MooncakeConnectorMetadata()
        meta.add_new_req(
            request_id="req-split",
            local_block_ids=[1],
            num_external_tokens=16,
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                LIVE_SPLIT_CAPABILITY: {
                    "segments": [{
                        "group_id": 0,
                        "source_buffer_index": 0,
                        "source_offset": 0,
                        "destination_address": 0x1000,
                        "length": 256,
                        "destination_kind": "cpu",
                    }, {
                        "group_id": 1,
                        "source_buffer_index": 1,
                        "source_offset": 128,
                        "destination_address": 0x2000,
                        "length": 128,
                        "destination_kind": "npu",
                    }],
                    "group_byte_totals": [256, 128],
                    "tp_rank": 7,
                    "dp_rank": 1,
                },
            })

        plan = meta.requests["req-split"].split_plan
        self.assertIsNotNone(plan)
        self.assertEqual(plan.group_byte_totals, (256, 128))
        self.assertEqual((plan.tp_rank, plan.dp_rank), (7, 1))

    def test_malformed_eager_split_plan_falls_back(self):
        meta = MooncakeConnectorMetadata()
        meta.add_new_req(
            request_id="req-invalid-eager",
            local_block_ids=[1],
            num_external_tokens=16,
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
                "live_split_transfer_id": "generation-a",
                LIVE_SPLIT_CAPABILITY: {
                    "group_byte_totals": "invalid",
                    "tp_rank": 0,
                    "dp_rank": 0,
                },
            },
        )

        request = meta.requests["req-invalid-eager"]
        self.assertTrue(request.split_fallback)
        self.assertIsNone(request.split_plan)

    def test_non_integral_wire_address_falls_back(self):
        meta = MooncakeConnectorMetadata()
        meta.add_new_req(
            request_id="req-non-integral-address",
            local_block_ids=[1],
            num_external_tokens=16,
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
                "live_split_transfer_id": "generation-a",
                LIVE_SPLIT_CAPABILITY: {
                    "segments": [{
                        "group_id": 1,
                        "source_buffer_index": 0,
                        "source_offset": 0,
                        "destination_address": 4096.5,
                        "length": 64,
                        "destination_kind": "npu",
                    }],
                    "group_byte_totals": [0, 64],
                    "tp_rank": 0,
                    "dp_rank": 0,
                    "requested_groups": [1],
                },
            },
        )

        request = meta.requests["req-non-integral-address"]
        self.assertTrue(request.split_fallback)
        self.assertIsNone(request.split_plan)

    def test_non_integral_compact_source_address_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "buffer_base must be an integer"):
            MooncakeConnectorMetadata._parse_source_descriptor({
                "descriptors": [{
                    "group_byte_totals": [0, 64],
                    "tp_rank": 0,
                    "dp_rank": 0,
                    "compact_layout": {
                        "group_id": 1,
                        "token_count": 1,
                        "layers": [{
                            "layer_id": 0,
                            "buffer_base": 8192.25,
                            "token_bytes": 64,
                            "slot_capacity": 1,
                            "buffer_index": 0,
                        }],
                        "runs": [{
                            "logical_token_start": 0,
                            "physical_slot_start": 0,
                            "token_count": 1,
                        }],
                    },
                }],
            })

    def test_missing_late_plan_selects_persistent_fallback(self):
        meta = MooncakeConnectorMetadata()
        meta.add_new_req(
            request_id="req-late",
            local_block_ids=[1],
            num_external_tokens=16,
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
            })

        self.assertTrue(meta.needs_late_split_plans())
        meta.accept_late_split_plans({})

        request = meta.requests["req-late"]
        self.assertTrue(request.split_fallback)
        self.assertIsNone(request.split_plan)

    def test_short_group_totals_select_persistent_fallback(self):
        metadata = MooncakeConnectorMetadata()
        metadata.add_new_req(
            request_id="req-invalid",
            local_block_ids=[1],
            num_external_tokens=16,
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
            })

        metadata.accept_late_split_plans({"req-invalid": {
            "segments": [],
            "group_byte_totals": [0],
            "tp_rank": 0,
            "dp_rank": 0,
        }})

        request = metadata.requests["req-invalid"]
        self.assertTrue(request.split_fallback)
        self.assertIsNone(request.split_plan)

    def test_index_only_plan_never_routes_group0(self):
        meta = MooncakeConnectorMetadata()
        meta.add_new_req(
            request_id="req-index",
            local_block_ids=[1],
            num_external_tokens=16,
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
                "live_split_transfer_id": "generation-a",
            })
        meta.accept_late_split_plans({"req-index": {
            "segments": [{
                "group_id": 0,
                "source_buffer_index": 0,
                "source_offset": 0,
                "destination_address": 0x1000,
                "length": 16,
                "destination_kind": "cpu",
            }, {
                "group_id": 1,
                "source_buffer_index": 0,
                "source_offset": 0,
                "destination_address": 0x2000,
                "length": 8,
                "destination_kind": "npu",
            }],
            "group_byte_totals": [16, 8],
            "tp_rank": 0,
            "dp_rank": 0,
        }}, supported_groups=(1,))

        plan = meta.requests["req-index"].split_plan
        self.assertIsNotNone(plan)
        self.assertEqual(plan.requested_groups, (1,))
        self.assertEqual(plan.group_byte_totals, (0, 8))
        self.assertEqual([segment.group_id for segment in plan.segments], [1])

    def test_late_plan_preserves_provider_requested_subset(self):
        meta = MooncakeConnectorMetadata()
        meta.add_new_req(
            request_id="req-index",
            local_block_ids=[1],
            num_external_tokens=16,
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
                "live_split_transfer_id": "generation-a",
            })
        meta.accept_late_split_plans({"req-index": {
            "segments": [{
                "group_id": 1,
                "source_buffer_index": 0,
                "source_offset": 0,
                "destination_address": 0x2000,
                "length": 8,
                "destination_kind": "npu",
            }],
            "group_byte_totals": [0, 8],
            "tp_rank": 0,
            "dp_rank": 0,
            "requested_groups": [1],
        }}, supported_groups=(0, 1))

        plan = meta.requests["req-index"].split_plan
        self.assertIsNotNone(plan)
        self.assertEqual(plan.requested_groups, (1,))
        self.assertEqual(plan.group_byte_totals, (0, 8))

    def test_negotiated_split_without_transfer_identity_falls_back(self):
        meta = MooncakeConnectorMetadata()
        meta.add_new_req(
            request_id="req",
            local_block_ids=[1],
            num_external_tokens=16,
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
            })

        meta.accept_late_split_plans({"req": {}})

        self.assertTrue(meta.requests["req"].split_fallback)
        self.assertIsNone(meta.requests["req"].split_transfer_id)

    def test_prefiller_source_is_stream_merged_with_different_dest_pools(self):
        metadata = MooncakeConnectorMetadata()
        metadata.add_new_req(
            "req", [1], 16, {
                "remote_block_ids": [9, 11],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
                "live_split_transfer_id": "generation-a",
                LIVE_SPLIT_SOURCE_DESCRIPTOR: {
                    "segments": [
                        {"group_id": 0, "source_buffer_index": 2,
                         "source_buffer_base": 0x5000,
                         "source_offset": 96, "length": 12},
                        {"group_id": 0, "source_buffer_index": 4,
                         "source_buffer_base": 0x9000,
                         "source_offset": 32, "length": 20},
                        {"group_id": 1, "source_buffer_index": 5,
                         "source_buffer_base": 0xB000,
                         "source_offset": 64, "length": 8},
                    ],
                    "group_byte_totals": [32, 8],
                    "tp_rank": 3, "dp_rank": 1,
                },
            })
        metadata.accept_late_split_plans({"req": {
            "segments": [
                {"group_id": 0, "destination_address": 0x1000,
                 "length": 16, "destination_kind": "cpu"},
                {"group_id": 0, "destination_address": 0x3000,
                 "length": 16, "destination_kind": "cpu"},
                {"group_id": 1, "destination_address": 0x7000,
                 "length": 8, "destination_kind": "npu"},
            ],
            "group_byte_totals": [32, 8],
            "tp_rank": 3, "dp_rank": 1,
        }})

        plan = metadata.requests["req"].split_plan
        self.assertIsNotNone(plan)
        self.assertEqual(
            [(s.source_buffer_index, s.source_offset,
              s.destination_address, s.length) for s in plan.segments],
            [(2, 96, 0x1000, 12), (4, 32, 0x100C, 4),
             (4, 36, 0x3000, 16), (5, 64, 0x7000, 8)],
        )

    def test_dp2_live_split_routes_all_source_destination_pairs(self):
        for source_dp_rank in range(2):
            for destination_dp_rank in range(2):
                with self.subTest(source=source_dp_rank,
                                  destination=destination_dp_rank):
                    metadata = MooncakeConnectorMetadata(
                        live_split_dp_routing_required=True,
                        live_split_source_dp_size=2,
                    )
                    metadata.add_new_req(
                        "req", [1], 16, {
                            "remote_block_ids": [9],
                            "remote_engine_id": "remote",
                            "remote_request_id": "remote-req",
                            "remote_host": "host",
                            "remote_port": 30000,
                            "remote_dp_rank": source_dp_rank,
                            "live_split_capabilities": (
                                LIVE_SPLIT_CAPABILITY,
                                LIVE_SPLIT_DP_ROUTING_CAPABILITY,
                            ),
                            "live_split_transfer_id": "generation-a",
                            LIVE_SPLIT_SOURCE_DESCRIPTOR: {
                                "segments": [{
                                    "group_id": 1,
                                    "source_buffer_index": 0,
                                    "source_buffer_base": 0x5000,
                                    "source_offset": 0,
                                    "length": 8,
                                }],
                                "group_byte_totals": [0, 8],
                                "tp_rank": 0,
                                "dp_rank": source_dp_rank,
                            },
                        })
                    metadata.accept_late_split_plans({"req": {
                        "segments": [{
                            "group_id": 1,
                            "destination_address": 0x9000,
                            "length": 8,
                            "destination_kind": "npu",
                        }],
                        "group_byte_totals": [0, 8],
                        "tp_rank": 0,
                        "dp_rank": destination_dp_rank,
                        "requested_groups": [1],
                    }}, supported_groups=(1,))

                    plan = metadata.requests["req"].split_plan
                    self.assertIsNotNone(plan)
                    self.assertEqual(plan.source_rank, (0, source_dp_rank))
                    self.assertEqual(
                        plan.destination_rank, (0, destination_dp_rank))

    def test_dp2_peer_without_routing_capability_uses_persistent_path(self):
        metadata = MooncakeConnectorMetadata(
            live_split_dp_routing_required=True,
            live_split_source_dp_size=2,
        )
        metadata.add_new_req(
            "req", [1], 16, {
                "remote_block_ids": [9],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
                "live_split_transfer_id": "generation-a",
            })

        request = metadata.requests["req"]
        self.assertTrue(request.split_negotiated)
        self.assertTrue(request.split_source_invalid)
        self.assertIsNone(request.split_plan)
        self.assertTrue(request.split_fallback)
        self.assertFalse(metadata.needs_late_split_plans())

        metadata.accept_late_split_plans({"req": {}}, supported_groups=(1,))

        self.assertTrue(request.split_fallback)

    def test_malformed_capability_container_does_not_negotiate_split(self):
        metadata = MooncakeConnectorMetadata()
        metadata.add_new_req(
            "req", [1], 16, {
                "remote_block_ids": [9],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": None,
            })

        request = metadata.requests["req"]
        self.assertFalse(request.split_negotiated)
        self.assertFalse(request.split_fallback)

    def test_dp2_out_of_range_source_rank_falls_back_before_late_plan(self):
        metadata = MooncakeConnectorMetadata(
            live_split_dp_routing_required=True,
            live_split_source_dp_size=2,
        )
        metadata.add_new_req(
            "req", [1], 16, {
                "remote_block_ids": [9],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "remote_dp_rank": 2,
                "live_split_capabilities": (
                    LIVE_SPLIT_CAPABILITY,
                    LIVE_SPLIT_DP_ROUTING_CAPABILITY,
                ),
                "live_split_transfer_id": "generation-a",
            })

        request = metadata.requests["req"]
        self.assertIsNone(request.remote_dp_rank)
        self.assertTrue(request.split_source_invalid)
        self.assertTrue(request.split_fallback)
        self.assertFalse(metadata.needs_late_split_plans())

    def test_eager_source_dp_identity_mismatch_falls_back(self):
        metadata = MooncakeConnectorMetadata(
            live_split_dp_routing_required=True,
            live_split_source_dp_size=2,
        )
        metadata.add_new_req(
            "req", [1], 16, {
                "remote_block_ids": [9],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "remote_dp_rank": 1,
                "live_split_capabilities": (
                    LIVE_SPLIT_CAPABILITY,
                    LIVE_SPLIT_DP_ROUTING_CAPABILITY,
                ),
                "live_split_transfer_id": "generation-a",
                LIVE_SPLIT_CAPABILITY: {
                    "segments": [{
                        "group_id": 1,
                        "source_buffer_index": 0,
                        "source_offset": 0,
                        "destination_address": 0x9000,
                        "length": 8,
                        "destination_kind": "npu",
                    }],
                    "group_byte_totals": [0, 8],
                    "tp_rank": 0,
                    "dp_rank": 1,
                    "source_tp_rank": 0,
                    "source_dp_rank": 0,
                    "requested_groups": [1],
                },
            })

        request = metadata.requests["req"]
        self.assertIsNone(request.split_plan)
        self.assertTrue(request.split_fallback)
        self.assertFalse(metadata.needs_late_split_plans())

    def test_source_descriptor_rank_mismatch_falls_back(self):
        metadata = MooncakeConnectorMetadata()
        params = {
            "remote_block_ids": [1], "remote_engine_id": "remote",
            "remote_request_id": "remote-req", "remote_host": "host",
            "remote_port": 30000,
            "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
            "live_split_transfer_id": "generation-a",
            LIVE_SPLIT_SOURCE_DESCRIPTOR: {
                "segments": [{"group_id": 1, "source_buffer_index": 0,
                              "source_buffer_base": 0x5000,
                              "source_offset": 0, "length": 8}],
                "group_byte_totals": [0, 8], "tp_rank": 0, "dp_rank": 0,
            },
        }
        metadata.add_new_req("req", [1], 16, params)
        metadata.accept_late_split_plans({"req": {
            "segments": [{"group_id": 1,
                          "destination_address": 0x9000, "length": 8,
                          "destination_kind": "npu"}],
            "group_byte_totals": [0, 8], "tp_rank": 1, "dp_rank": 0,
            "requested_groups": [1],
        }}, supported_groups=(1,))
        self.assertTrue(metadata.requests["req"].split_fallback)
        self.assertIsNone(metadata.requests["req"].split_plan)

    def test_malformed_source_descriptor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source byte totals"):
            MooncakeConnectorMetadata._parse_source_descriptor({
                "segments": [{"group_id": 0, "source_buffer_index": 0,
                              "source_buffer_base": 0x1000,
                              "source_offset": 0, "length": 7}],
                "group_byte_totals": [8, 0], "tp_rank": 0, "dp_rank": 0,
            })

    def test_malformed_received_source_uses_persistent_fallback(self):
        metadata = MooncakeConnectorMetadata()
        metadata.add_new_req("req", [1], 16, {
            "remote_block_ids": [1], "remote_engine_id": "remote",
            "remote_request_id": "remote-req", "remote_host": "host",
            "remote_port": 30000,
            "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
            "live_split_transfer_id": "generation-a",
            LIVE_SPLIT_SOURCE_DESCRIPTOR: {
                "segments": [], "group_byte_totals": [1, 0],
                "tp_rank": 0, "dp_rank": 0,
            },
        })
        metadata.accept_late_split_plans({"req": {}})
        self.assertTrue(metadata.requests["req"].split_fallback)
        self.assertIsNone(metadata.requests["req"].split_plan)

    def test_non_mapping_received_source_uses_persistent_fallback(self):
        metadata = MooncakeConnectorMetadata()
        metadata.add_new_req("req", [1], 16, {
            "remote_block_ids": [1], "remote_engine_id": "remote",
            "remote_request_id": "remote-req", "remote_host": "host",
            "remote_port": 30000,
            "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
            "live_split_transfer_id": "generation-a",
            LIVE_SPLIT_SOURCE_DESCRIPTOR: {
                "descriptors": [None],
            },
        })
        metadata.accept_late_split_plans({"req": {}})
        self.assertTrue(metadata.requests["req"].split_fallback)
        self.assertIsNone(metadata.requests["req"].split_plan)

    def test_non_finite_received_source_uses_persistent_fallback(self):
        metadata = MooncakeConnectorMetadata()
        metadata.add_new_req("req", [1], 16, {
            "remote_block_ids": [1], "remote_engine_id": "remote",
            "remote_request_id": "remote-req", "remote_host": "host",
            "remote_port": 30000,
            "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
            "live_split_transfer_id": "generation-a",
            LIVE_SPLIT_SOURCE_DESCRIPTOR: {
                "segments": [], "group_byte_totals": [0, float("inf")],
                "tp_rank": 0, "dp_rank": 0,
            },
        })
        metadata.accept_late_split_plans({"req": {}})
        self.assertTrue(metadata.requests["req"].split_fallback)
        self.assertIsNone(metadata.requests["req"].split_plan)


class TestAscendMultiLateSplitInjection(unittest.TestCase):

    def test_shutdown_stops_live_borrower_before_shared_owner(self):
        events = []

        class Child:
            def __init__(self, name, borrower=False):
                self.name = name
                self.releases_live_transfer_destinations_on_shutdown = borrower

            def shutdown(self):
                events.append(self.name)

        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [Child("lmcache"), Child("mooncake", True)]

        multi.shutdown()

        self.assertEqual(events, ["mooncake", "lmcache"])

    def test_shutdown_delivers_borrower_result_before_owner_shutdown(self):
        events = []

        class Borrower:
            releases_live_transfer_destinations_on_shutdown = True

            def shutdown(self):
                events.append("borrower_shutdown")

            def get_live_split_results(self):
                events.append("result_poll")
                return {"req": "success"}

        class Owner:
            def _accept_live_split_results(self, results):
                events.append(("result_accept", results))

            def shutdown(self):
                events.append("owner_shutdown")

        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [Owner(), Borrower()]
        multi._live_split_result_backlog = {}

        multi.shutdown()

        self.assertEqual(
            events,
            [
                "borrower_shutdown",
                "result_poll",
                ("result_accept", {"req": "success"}),
                "owner_shutdown",
            ],
        )

    def test_shutdown_does_not_close_owner_if_borrower_is_active(self):
        owner = MagicMock()
        borrower = MagicMock()
        borrower.releases_live_transfer_destinations_on_shutdown = True
        borrower.shutdown.side_effect = RuntimeError("active transfer")
        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [owner, borrower]

        with self.assertRaisesRegex(RuntimeError, "active transfer"):
            multi.shutdown()

        owner.shutdown.assert_not_called()

    def test_compact_load_capability_follows_selected_child(self):
        class Child:
            def __init__(self, tokens, capable):
                self.tokens = tokens
                self.supports_dsa_compact_external_load = capable

            def get_num_new_matched_tokens(self, _request, _computed):
                return self.tokens, True

        multi = object.__new__(AscendMultiConnector)
        multi._requests_to_connector = {}
        request = types.SimpleNamespace(request_id="req")

        multi._connectors = [Child(16, False), Child(32, True)]
        self.assertEqual(multi.get_num_new_matched_tokens(request, 0), (16, True))
        self.assertFalse(multi.supports_dsa_compact_external_load)

        multi._connectors = [Child(0, False), Child(32, True)]
        self.assertEqual(multi.get_num_new_matched_tokens(request, 0), (32, True))
        self.assertTrue(multi.supports_dsa_compact_external_load)

    def _assert_scheduler_live_provider_runs_first(self, provider_first):
        events = []
        request = types.SimpleNamespace(
            request_id="req",
            kv_transfer_params={"do_remote_decode": True},
        )

        class Provider:
            supports_dsa_compact_external_load = False
            supports_dsa_live_split_source = True

            def request_finished(self, req, _block_ids):
                events.append("provider")
                req.kv_transfer_params["request_live_split"] = True
                return False, None

        class Consumer:
            def request_finished(self, req, _block_ids):
                self.assert_live = req.kv_transfer_params.get(
                    "request_live_split", False
                )
                events.append("consumer")
                return False, None

        provider = Provider()
        consumer = Consumer()
        multi = object.__new__(AscendMultiConnector)
        multi._connectors = (
            [provider, consumer] if provider_first else [consumer, provider]
        )
        multi._extra_async_saves = {}
        multi._requests_to_connector = {}
        multi._index_load_async_req_ids = set()

        multi.request_finished_all_groups(request, ([1],))

        self.assertEqual(events, ["provider", "consumer"])
        self.assertTrue(consumer.assert_live)

    def test_scheduler_live_negotiation_provider_first(self):
        self._assert_scheduler_live_provider_runs_first(True)

    def test_scheduler_live_negotiation_consumer_first(self):
        self._assert_scheduler_live_provider_runs_first(False)

    def test_scheduler_keeps_mooncake_canonical_source_in_output(self):
        raw = {"descriptors": [{"segments": ["raw"]}]}
        canonical = {"descriptors": [{"segments": ["canonical"]}]}

        class Provider:
            supports_dsa_live_split_source = True

            def request_finished(self, request, _block_ids):
                request.kv_transfer_params["request_live_split"] = True
                return False, {LIVE_SPLIT_SOURCE_DESCRIPTOR: raw}

        class Mooncake:
            supports_dsa_live_split_source = False

            def request_finished(self, request, _block_ids):
                assert request.kv_transfer_params[
                    LIVE_SPLIT_SOURCE_DESCRIPTOR
                ] == raw
                return False, {LIVE_SPLIT_SOURCE_DESCRIPTOR: canonical}

        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [Provider(), Mooncake()]
        multi._extra_async_saves = {}
        multi._requests_to_connector = {}
        multi._index_load_async_req_ids = set()
        request = types.SimpleNamespace(
            request_id="req",
            kv_transfer_params={"do_remote_decode": True},
        )

        _, params = multi.request_finished_all_groups(request, ([1],))

        self.assertEqual(params[LIVE_SPLIT_SOURCE_DESCRIPTOR], canonical)

    def test_scheduler_non_live_preserves_configured_order(self):
        events = []

        class Child:
            def __init__(self, name, capable=False):
                self.name = name
                self.supports_dsa_live_split_source = capable

            def request_finished(self, _request, _block_ids):
                events.append(self.name)
                return False, None

        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [Child("first"), Child("second", capable=True)]
        multi._extra_async_saves = {}
        multi._requests_to_connector = {}
        multi._index_load_async_req_ids = set()
        request = types.SimpleNamespace(request_id="req", kv_transfer_params={})

        multi.request_finished_all_groups(request, ([1],))

        self.assertEqual(events, ["first", "second"])

    def test_worker_plan_is_taken_after_provider_allocation(self):
        events = []
        metadata = MooncakeConnectorMetadata()
        metadata.add_new_req(
            request_id="req",
            local_block_ids=[1],
            num_external_tokens=16,
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
                "live_split_transfer_id": "generation-a",
            })
        plan = {
            "segments": [{
                "group_id": 0,
                "source_buffer_index": 0,
                "source_offset": 0,
                "destination_address": 0x1000,
                "length": 16,
                "destination_kind": "cpu",
            }, {
                "group_id": 1,
                "source_buffer_index": 1,
                "source_offset": 0,
                "destination_address": 0x2000,
                "length": 8,
                "destination_kind": "npu",
            }],
            "group_byte_totals": [16, 8],
            "tp_rank": 0,
            "dp_rank": 0,
        }

        class Provider:
            allocated = False

            def start_load_kv(self, _forward_context, **_kwargs):
                self.allocated = True
                events.append("allocated")

            def _take_live_split_destination_plans(self):
                assert self.allocated
                events.append("taken")
                return {"req": plan}

        class Consumer:
            def _needs_live_split_destination_plans(self):
                return metadata.needs_late_split_plans()

            def _accept_live_split_destination_plans(self, plans):
                events.append("accepted")
                metadata.accept_late_split_plans(plans)

            def start_load_kv(self, _forward_context, **_kwargs):
                assert metadata.requests["req"].split_plan is not None
                events.append("mooncake-started")

        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [Consumer(), Provider()]

        multi.start_load_kv(object())

        self.assertEqual(events, [
            "allocated", "taken", "accepted", "mooncake-started"
        ])

    def test_results_reach_provider_once_and_duplicate_drain_is_empty(self):
        delivered = []

        class Source:
            results = [{"ok": "success", "bad": "failure"}, {}]

            def get_live_split_results(self):
                return self.results.pop(0)

        class Provider:
            def _accept_live_split_results(self, results):
                delivered.append(results)

        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [Provider(), Source()]

        first = multi.get_live_split_results()
        second = multi.get_live_split_results()

        self.assertEqual(first, {"ok": "success", "bad": "failure"})
        self.assertEqual(second, {})
        self.assertEqual(delivered, [first])

    def test_standard_finished_poll_delivers_ack_without_early_completion(self):
        delivered = []

        class Source:
            results = [{"req": "success"}, {}]

            def get_finished(self, _finished_req_ids):
                return None, None

            def get_live_split_results(self):
                return self.results.pop(0)

        class Provider:
            def get_finished(self, _finished_req_ids):
                return None, None

            def _accept_live_split_results(self, results):
                delivered.append(results)

        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [Provider(), Source()]
        multi._extra_async_saves = {}

        first = multi.get_finished(set())
        second = multi.get_finished(set())

        self.assertEqual(first, (None, None))
        self.assertEqual(second, (None, None))
        self.assertEqual(delivered, [{"req": "success"}])

    @patch(
        'vllm_ascend.distributed.kv_transfer.ascend_multi_connector.logger'
    )
    def test_index_only_consumer_requires_group0_persistent_fallback(
        self, mock_logger
    ):
        events = []

        class Provider:
            def start_load_kv(self, _forward_context, **_kwargs):
                events.append("persistent-started")

            def _take_live_split_destination_plans(self):
                events.append("unsafe-take")
                return {"req": {}}

        class IndexConsumer:
            accepted = None

            def _needs_live_split_destination_plans(self):
                return True

            def _accept_live_split_destination_plans(self, plans):
                self.accepted = plans

            def _live_split_source_groups(self):
                return (1,)

            def start_load_kv(self, _forward_context, **_kwargs):
                events.append("index-started")

        consumer = IndexConsumer()
        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [consumer, Provider()]

        multi.start_load_kv(object())

        self.assertEqual(events, ["persistent-started", "index-started"])
        self.assertEqual(consumer.accepted, {})
        mock_logger.warning.assert_called_once()

    def test_group_aware_provider_enables_index_only_plan(self):
        handled = []

        class Provider:
            def start_load_kv(self, _forward_context, **_kwargs):
                pass

            def _take_live_split_destination_plans(self, handled_groups):
                handled.append(handled_groups)
                return {"req": "group1-plan"}

        class IndexConsumer:
            accepted = None

            def _needs_live_split_destination_plans(self):
                return True

            def _accept_live_split_destination_plans(self, plans):
                self.accepted = plans

            def _live_split_source_groups(self):
                return (1,)

            def start_load_kv(self, _forward_context, **_kwargs):
                pass

        consumer = IndexConsumer()
        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [Provider(), consumer]

        multi.start_load_kv(object())

        self.assertEqual(handled, [(1,)])
        self.assertEqual(consumer.accepted, {"req": "group1-plan"})

    @patch(
        'vllm_ascend.distributed.kv_transfer.ascend_multi_connector.logger'
    )
    def test_result_provider_exception_preserves_failure_fallback(
        self, mock_logger
    ):
        delivered = []

        class Source:
            def get_live_split_results(self):
                return {"bad": "failure"}

        class BrokenProvider:
            def _accept_live_split_results(self, _results):
                raise RuntimeError("provider failed")

        class HealthyProvider:
            def _accept_live_split_results(self, results):
                delivered.append(results)

        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [BrokenProvider(), Source(), HealthyProvider()]

        results = multi.get_live_split_results()

        self.assertEqual(results, {"bad": "failure"})
        self.assertEqual(delivered, [results])
        mock_logger.exception.assert_called_once()

    def test_finished_poll_retries_only_failed_result_provider(self):
        broken_deliveries = []
        healthy_deliveries = []

        class Source:
            results = [{"req": "success"}, {}]

            def get_finished(self, _finished_req_ids):
                return None, None

            def get_live_split_results(self):
                return self.results.pop(0)

        class RetryProvider:
            attempts = 0

            def get_finished(self, _finished_req_ids):
                return None, None

            def _accept_live_split_results(self, results):
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("retry")
                broken_deliveries.append(results)

        class HealthyProvider:
            def get_finished(self, _finished_req_ids):
                return None, None

            def _accept_live_split_results(self, results):
                healthy_deliveries.append(results)

        retry = RetryProvider()
        multi = object.__new__(AscendMultiConnector)
        multi._connectors = [retry, Source(), HealthyProvider()]
        multi._extra_async_saves = {}
        multi._live_split_result_backlog = {}

        multi.get_finished(set())
        self.assertEqual(retry.attempts, 1)
        self.assertEqual(healthy_deliveries, [{"req": "success"}])
        self.assertTrue(multi._live_split_result_backlog)

        multi.get_finished(set())
        self.assertEqual(retry.attempts, 2)
        self.assertEqual(broken_deliveries, [{"req": "success"}])
        self.assertEqual(healthy_deliveries, [{"req": "success"}])
        self.assertEqual(multi._live_split_result_backlog, {})

    def test_finished_request_preserves_provider_backlog_until_accepted(self):
        delivered = []

        class Provider:
            attempts = 0

            def get_finished(self, _finished_req_ids):
                return None, None

            def _accept_live_split_results(self, results):
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("retry")
                delivered.append(results)

        multi = object.__new__(AscendMultiConnector)
        provider = Provider()
        multi._connectors = [provider]
        multi._extra_async_saves = {}
        multi._live_split_result_backlog = {
            id(provider): {"reused": "success", "active": "failure"}
        }

        multi.get_finished({"reused"})
        self.assertEqual(delivered, [])
        self.assertEqual(
            multi._live_split_result_backlog,
            {id(provider): {"reused": "success", "active": "failure"}},
        )

        multi.get_finished(set())

        self.assertEqual(
            delivered, [{"reused": "success", "active": "failure"}]
        )
        self.assertEqual(multi._live_split_result_backlog, {})


class TestMooncakeConnectorSchedulerMatchedTokens(unittest.TestCase):

    def setUp(self):
        config = MockVllmConfig()
        self.p1 = patch(
            'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.init_ascend_config',
            new=MagicMock())
        self.p2 = patch(
            'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
            new=MagicMock(return_value=MagicMock()))
        self.p1.start()
        self.p2.start()
        self.addCleanup(self.p1.stop)
        self.addCleanup(self.p2.stop)
        self.scheduler = MooncakeConnectorScheduler(config, "test_engine")

    def test_get_num_new_matched_tokens(self):
        request = MockRequest("req1")
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            request, 0)
        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)

        request.kv_transfer_params = {"do_remote_prefill": True}
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            request, 0)
        self.assertEqual(tokens, 4)
        self.assertTrue(async_flag)

    def test_build_connector_meta(self):
        request = MockRequest("req1")
        blocks_mock = MagicMock()
        blocks_mock.get_unhashed_block_ids.return_value = [4, 5, 6]
        self.scheduler._reqs_need_recv["req1"] = (request, [4, 5, 6], 48)
        request.kv_transfer_params = {
            "remote_block_ids": [1, 2, 3],
            "remote_engine_id": "remote",
            "remote_host": "localhost",
            "remote_port": 5000,
            "remote_pcp_size": 1,
            "remote_dcp_size": 1
        }

        meta = self.scheduler.build_connector_meta(MagicMock())
        self.assertIsInstance(meta, MooncakeConnectorMetadata)
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests["req1"].local_block_ids, [4, 5, 6])
        self.assertEqual(meta.requests["req1"].remote_block_ids, [1, 2, 3])
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)


class TestHelperFunctions(unittest.TestCase):

    def test_group_concurrent_contiguous(self):
        src: list[int] = [1, 2, 3, 5, 6]
        dst: list[int] = [10, 11, 12, 14, 15]

        src_groups, dst_groups = group_concurrent_contiguous(src, dst)

        self.assertEqual(len(src_groups), 2)
        self.assertEqual(src_groups[0], [1, 2, 3])
        self.assertEqual(src_groups[1], [5, 6])
        self.assertEqual(dst_groups[0], [10, 11, 12])
        self.assertEqual(dst_groups[1], [14, 15])

    def test_group_concurrent_contiguous_empty(self):
        src: list[int] = []
        dst: list[int] = []
        src_groups, dst_groups = group_concurrent_contiguous(src, dst)
        self.assertEqual(src_groups, [])
        self.assertEqual(dst_groups, [])

    def test_string_to_int64_hash(self):
        hash1 = string_to_int64_hash("test_string")
        hash2 = string_to_int64_hash("test_string")
        self.assertEqual(hash1, hash2)

        hash3 = string_to_int64_hash("different_string")
        self.assertNotEqual(hash1, hash3)


class TestMooncakeConnectorForScheduler(unittest.TestCase):

    def test_scheduler_role(self):
        config = MockVllmConfig()
        with patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.init_ascend_config'
        ), patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
                return_value=MagicMock()):
            connector = MooncakeConnector(config, KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    @patch.object(MooncakeConnectorScheduler, "get_num_new_matched_tokens")
    def test_scheduler_methods(self, mock_method):
        config = MockVllmConfig()
        with patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.init_ascend_config'
        ), patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
                return_value=MagicMock()):
            connector = MooncakeConnector(config, KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        connector.get_num_new_matched_tokens(request, 0)
        mock_method.assert_called_once_with(request, 0)


class MockKVCacheBlocks:

    def get_unhashed_block_ids(self):
        return [4, 5, 6]


class MockSchedulerOutput:
    pass


class MockForwardContext:
    pass


class TestMooncakeConnector(unittest.TestCase):

    def setUp(self):
        self.config = MockVllmConfig()
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

    def test_scheduler_initialization(self):
        with patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.init_ascend_config'
        ), patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
                return_value=MagicMock()):
            connector = MooncakeConnector(self.config,
                                          KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    @patch.object(MooncakeConnectorScheduler, "get_num_new_matched_tokens")
    def test_get_num_new_matched_tokens(self, mock_method):
        with patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.init_ascend_config'
        ), patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
                return_value=MagicMock()):
            connector = MooncakeConnector(self.config,
                                          KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        connector.get_num_new_matched_tokens(request, 0)
        mock_method.assert_called_once_with(request, 0)

    @patch.object(MooncakeConnectorScheduler, "update_state_after_alloc")
    def test_update_state_after_alloc(self, mock_method):
        with patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.init_ascend_config'
        ), patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
                return_value=MagicMock()):
            connector = MooncakeConnector(self.config,
                                          KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        blocks = MockKVCacheBlocks()
        connector.update_state_after_alloc(request, blocks, 3)
        mock_method.assert_called_once_with(request, blocks, 3)

    @patch.object(MooncakeConnectorScheduler, "build_connector_meta")
    def test_build_connector_meta(self, mock_method):
        with patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.init_ascend_config'
        ), patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
                return_value=MagicMock()):
            connector = MooncakeConnector(self.config,
                                          KVConnectorRole.SCHEDULER)
        scheduler_output = MockSchedulerOutput()
        connector.build_connector_meta(scheduler_output)
        mock_method.assert_called_once_with(scheduler_output)

    @patch.object(MooncakeConnectorScheduler, "request_finished")
    def test_request_finished(self, mock_method):
        with patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.init_ascend_config'
        ), patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
                return_value=MagicMock()):
            connector = MooncakeConnector(self.config,
                                          KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        connector.request_finished(request, [1, 2, 3])
        mock_method.assert_called_once_with(request, [1, 2, 3])


class TestMooncakeConnectorScheduler(unittest.TestCase):

    def setUp(self):
        self.config = MockVllmConfig()
        with patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.init_ascend_config'
        ), patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
                return_value=MagicMock()):
            self.scheduler = MooncakeConnectorScheduler(
                self.config, "test_engine")

    def test_get_num_new_matched_tokens_no_remote_prefill(self):
        request = MockRequest("req1")
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            request, 0)
        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)

    def test_get_num_new_matched_tokens_with_remote_prefill(self):
        request = MockRequest("req1",
                              kv_transfer_params={"do_remote_prefill": True})
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            request, 0)
        self.assertEqual(tokens, 4)
        self.assertTrue(async_flag)

    def test_update_state_after_alloc_no_remote_prefill(self):
        request = MockRequest("req1")
        blocks = MagicMock()
        self.scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)

    def test_update_state_after_alloc_with_remote_prefill(self):
        request = MockRequest("req1",
                              kv_transfer_params={
                                  "do_remote_prefill": True,
                                  "remote_block_ids": [1, 2, 3],
                                  "remote_engine_id": "remote",
                                  "remote_host": "localhost",
                                  "remote_port": 5000
                              })
        blocks = MockKVCacheBlocks()
        self.scheduler.update_state_after_alloc(request, blocks, 3)
        self.assertEqual(len(self.scheduler._reqs_need_recv), 1)
        self.assertEqual(self.scheduler._reqs_need_recv["req1"][0], request)
        self.assertEqual(self.scheduler._reqs_need_recv["req1"][1], [4, 5, 6])

    def test_request_finished_no_remote_decode(self):
        request = MockRequest("req1")
        delay_free, params = self.scheduler.request_finished(
            request, [1, 2, 3])
        self.assertFalse(delay_free)
        self.assertIsNone(params)

    def test_request_finished_forwards_prefiller_source_descriptor(self):
        source = {
            "segments": [{"group_id": 1, "source_buffer_index": 3,
                          "source_buffer_base": 0x9000,
                          "source_offset": 128, "length": 24}],
            "group_byte_totals": [0, 24], "tp_rank": 1, "dp_rank": 0,
        }
        request = MockRequest(
            "req1",
            kv_transfer_params={
                "do_remote_decode": True,
                "request_live_split": True,
                LIVE_SPLIT_SOURCE_DESCRIPTOR: source,
            },
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        )

        delay_free, params = self.scheduler.request_finished(request, [7, 9])

        self.assertTrue(delay_free)
        self.assertEqual(params[LIVE_SPLIT_SOURCE_DESCRIPTOR], source)
        self.assertIsNot(params[LIVE_SPLIT_SOURCE_DESCRIPTOR], source)
        self.assertIn(
            LIVE_SPLIT_DP_ROUTING_CAPABILITY,
            params["live_split_capabilities"],
        )
        self.assertEqual(params["remote_dp_rank"], 0)
        transfer_id = params["live_split_transfer_id"]
        self.assertTrue(transfer_id)

        metadata = self.scheduler.build_connector_meta(
            MockSchedulerOutput())
        self.assertEqual(metadata.split_transfer_ids, {"req1": transfer_id})
        self.assertNotIn("req1", self.scheduler.split_transfer_ids)

    def test_live_split_falls_back_for_unencoded_pp_topology(self):
        self.scheduler.pp_size = 2
        request = MockRequest(
            "req1",
            kv_transfer_params={
                "do_remote_decode": True,
                "request_live_split": True,
            },
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        )

        delay_free, params = self.scheduler.request_finished(request, [7, 9])

        self.assertTrue(delay_free)
        self.assertNotIn("live_split_capabilities", params)
        self.assertNotIn("live_split_transfer_id", params)
        self.assertNotIn("req1", self.scheduler._split_reqs_need_send)

    def test_live_split_topology_supports_matching_dp2(self):
        self.scheduler._live_prefill_dp_size = 2
        self.scheduler._live_decode_dp_size = 2

        self.assertTrue(self.scheduler._live_split_topology_supported())
        self.assertTrue(self.scheduler._live_split_dp_routing_required())

    def test_decoder_local_topology_rejects_live_split(self):
        self.scheduler.pp_size = 2
        request = MockRequest(
            "req1",
            kv_transfer_params={
                "do_remote_prefill": True,
                "remote_block_ids": [1],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-req",
                "remote_host": "host",
                "remote_port": 30000,
                "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
                "live_split_transfer_id": "generation-a",
            },
        )
        self.scheduler.update_state_after_alloc(
            request, MockKVCacheBlocks(), 1)

        metadata = self.scheduler.build_connector_meta(MockSchedulerOutput())

        request_meta = metadata.requests["req1"]
        self.assertTrue(request_meta.split_negotiated)
        self.assertTrue(request_meta.split_source_invalid)
        self.assertEqual(request_meta.split_transfer_id, "generation-a")
        metadata.accept_late_split_plans({})
        self.assertTrue(request_meta.split_fallback)

    def test_live_split_canonicalizes_only_destination_supported_group(self):
        self.assertEqual(self.scheduler.live_split_source_groups, (1,))
        self.scheduler.local_source_metadata[(0, 0)] = MooncakeAgentMetadata(
            engine_id="engine",
            te_rpc_port=1,
            kv_caches_base_addr=[0x2000],
            num_blocks=1,
            kv_caches_buffer_sizes=(0x100,),
            buffer_group_ids=(1,),
            tp_rank=0,
            dp_rank=0,
        )
        source = {
            "segments": [
                {"group_id": 0, "source_buffer_index": 0,
                 "source_buffer_base": 0x1000, "source_offset": 0,
                 "length": 16},
                {"group_id": 1, "source_buffer_index": 0,
                 "source_buffer_base": 0x2000, "source_offset": 8,
                 "length": 24},
            ],
            "group_byte_totals": [16, 24],
            "tp_rank": 0,
            "dp_rank": 0,
        }

        result = self.scheduler._canonicalize_source_descriptor(source)

        descriptor = result["descriptors"][0]
        self.assertEqual(descriptor["group_byte_totals"], [0, 24])
        self.assertEqual(len(descriptor["segments"]), 1)
        self.assertEqual(descriptor["segments"][0]["group_id"], 1)
        self.assertEqual(descriptor["segments"][0]["source_buffer_index"], 0)

    def test_source_canonicalization_rejects_negative_extent(self):
        self.scheduler.live_split_source_groups = (1,)
        self.scheduler.local_source_metadata[(0, 0)] = MooncakeAgentMetadata(
            engine_id="engine",
            te_rpc_port=1,
            kv_caches_base_addr=[0x2000],
            num_blocks=1,
            kv_caches_buffer_sizes=(0x100,),
            buffer_group_ids=(1,),
            tp_rank=0,
            dp_rank=0,
        )
        source = {
            "segments": [{
                "group_id": 1,
                "source_buffer_base": 0x2000,
                "source_offset": -1,
                "length": 24,
            }],
            "group_byte_totals": [0, 24],
            "tp_rank": 0,
            "dp_rank": 0,
        }

        with self.assertRaisesRegex(ValueError, "Invalid live split"):
            self.scheduler._canonicalize_source_descriptor(source)

    def test_source_canonicalization_disambiguates_aliased_views(self):
        self.scheduler.live_split_source_groups = (1,)
        self.scheduler.local_source_metadata[(0, 0)] = MooncakeAgentMetadata(
            engine_id="engine",
            te_rpc_port=1,
            kv_caches_base_addr=[0x2000, 0x2000, 0x2000],
            num_blocks=1,
            kv_caches_buffer_sizes=(0x100, 0x20, 0x200),
            buffer_group_ids=(0, 1, 1),
            tp_rank=0,
            dp_rank=0,
        )
        source = {
            "segments": [{
                "group_id": 1,
                "source_buffer_base": 0x2000,
                "source_offset": 0x40,
                "length": 0x20,
            }],
            "group_byte_totals": [0, 0x20],
            "tp_rank": 0,
            "dp_rank": 0,
        }

        result = self.scheduler._canonicalize_source_descriptor(source)

        segment = result["descriptors"][0]["segments"][0]
        self.assertEqual(segment["source_buffer_index"], 2)


class TestUtils(unittest.TestCase):

    def test_string_to_int64_hash(self):
        h1 = string_to_int64_hash("hello")
        h2 = string_to_int64_hash("hello")
        h3 = string_to_int64_hash("world")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertIsInstance(h1, int)

    def test_group_concurrent_contiguous(self):
        src: list[int] = [1, 2, 3, 5, 6]
        dst: list[int] = [10, 11, 12, 20, 21]
        src_g, dst_g = group_concurrent_contiguous(src, dst)
        self.assertEqual(src_g, [[1, 2, 3], [5, 6]])
        self.assertEqual(dst_g, [[10, 11, 12], [20, 21]])

    def test_group_empty(self):
        src_g, dst_g = group_concurrent_contiguous([], [])
        self.assertEqual(src_g, [])
        self.assertEqual(dst_g, [])

    def test_zmq_ctx_invalid_type(self):
        with self.assertRaises(ValueError):
            with zmq_ctx("INVALID", "tcp://127.0.0.1:5555"):
                pass

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.make_zmq_socket"
    )
    def test_zmq_ctx_ok(self, mock_make_socket):
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        with zmq_ctx(zmq.REQ, "tcp://localhost:1234") as s:  # type: ignore
            self.assertEqual(s, mock_socket)

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.logger")
    def test_ensure_zmq_send_success(self, mock_logger):
        mock_socket = MagicMock()
        ensure_zmq_send(mock_socket, b"hello")
        mock_socket.send.assert_called_once_with(b"hello")

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.logger")
    def test_ensure_zmq_send_retry_and_fail(self, mock_logger):
        mock_socket = MagicMock()
        mock_socket.send.side_effect = zmq.ZMQError(  # type: ignore
            "send failed")
        with self.assertRaises(RuntimeError):
            ensure_zmq_send(mock_socket, b"hello", max_retries=2)
        self.assertEqual(mock_socket.send.call_count, 2)

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.logger")
    def test_ensure_zmq_recv_success(self, mock_logger):
        mock_socket = MagicMock()
        mock_socket.recv.return_value = b"response"
        mock_poller = MagicMock()
        mock_poller.poll.return_value = [
            (mock_socket, zmq.POLLIN)  # type: ignore
        ]
        data = ensure_zmq_recv(mock_socket, mock_poller)
        self.assertEqual(data, b"response")

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.logger")
    def test_ensure_zmq_recv_timeout_and_fail(self, mock_logger):
        mock_socket = MagicMock()
        mock_poller = MagicMock()
        mock_poller.poll.return_value = []
        with self.assertRaises(RuntimeError):
            ensure_zmq_recv(mock_socket,
                            mock_poller,
                            timeout=0.01,
                            max_retries=2)


class MockMooncakeAgentMetadata:

    def __init__(self, **kwargs):
        pass


class MockMooncakeConnectorMetadata:

    def __init__(self):
        self.requests = {}


class MockKVCacheSendingThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.daemon = True
        self._finished_requests = set()

    def get_and_clear_finished_requests(self):
        return self._finished_requests

    def start(self):
        pass


class MockKVCacheRecvingThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.daemon = True
        self._finished_requests = set()
        self.add_request = MagicMock()

    def get_and_clear_finished_requests(self):
        return self._finished_requests

    def start(self):
        pass


class MockTensor:

    def __init__(self, *args, **kwargs):
        self.size = MagicMock(return_value=(10, 16, 8, 16))
        self.element_size = MagicMock(return_value=4)
        self.shape = (10, 16, 8, 16)
        self.data_ptr = MagicMock(return_value=0x1000)


mock_logger = MagicMock()


class MockTransferEngine:

    def initialize(self, *args, **kwargs):
        return 0

    def register_memory(self, *args, **kwargs):
        return 1


class MockEnvsAscend:
    MOONCAKE_CONNECTOR_PROTOCOL = "mock_protocol"


def mock_get_tensor_model_parallel_rank():
    return 0


def mock_get_tp_group():
    return MagicMock()


def mock_get_ip():
    return "127.0.0.1"


def mock_string_to_int64_hash(s):
    return hash(s)


class TestMooncakeConnectorWorker(unittest.TestCase):

    def setUp(self):
        self.mock_transfer_engine = MagicMock()
        self.mock_transfer_engine.get_rpc_port.return_value = 9090
        self.mock_transfer_engine.initialize.return_value = 0
        self.mock_transfer_engine.register_memory.return_value = 0

        self.patches = [
            patch('torch.Tensor.size', return_value=(10, 16, 8, 16)),
            patch('torch.Tensor.element_size', return_value=4),
            patch('torch.Tensor.data_ptr', return_value=0x1000),
            patch('math.prod', return_value=128),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_tensor_model_parallel_rank',
                mock_get_tensor_model_parallel_rank),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_tp_group',
                mock_get_tp_group),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_pp_group',
                return_value=_mock_pp_group),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ip',
                mock_get_ip),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.string_to_int64_hash',
                mock_string_to_int64_hash),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.get_transfer_engine',
                return_value=self.mock_transfer_engine),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.register_buffer',
                return_value=None),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.KVCacheSendingThread',
                MagicMock()),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.KVCacheRecvingThread',
                MagicMock()),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.logger',
                MagicMock()),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.threading.Event',
                MagicMock()),
            patch(
                'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
                return_value=MagicMock()),
        ]

        for p in self.patches:
            p.start()  # type: ignore

        self.vllm_config = MockVllmConfig()
        self.engine_id = "test_engine"
        self.kv_caches = {"layer1": (MagicMock(), MagicMock())}

    def tearDown(self):
        for p in self.patches:
            p.stop()  # type: ignore

    def test_worker_uses_global_dp_identity_and_local_device_count(self):
        self.vllm_config.parallel_config.data_parallel_rank_local = 0
        self.vllm_config.parallel_config.data_parallel_index = 1
        self.vllm_config.parallel_config.data_parallel_size_local = 1
        self.vllm_config.parallel_config.data_parallel_size = 2

        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)

        self.assertEqual(worker.dp_rank, 1)
        self.assertEqual(worker.dp_size, 1)
        self.assertEqual(worker.max_device_id, 2)
        self.assertEqual(worker.side_channel_port, 5002)

    def test_register_kv_caches_producer(self):
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertEqual(len(worker.kv_caches), 1)
        self.assertIsNotNone(worker.kv_send_thread)
        self.assertIsNone(worker.kv_recv_thread)

    def test_register_kv_caches_both_starts_sender_and_receiver(self):
        self.vllm_config.kv_transfer_config.kv_role = "kv_both"
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)

        worker.register_kv_caches(self.kv_caches)

        self.assertIsNotNone(worker.kv_send_thread)
        self.assertIsNotNone(worker.kv_recv_thread)
        self.assertTrue(worker.kv_send_thread.start.called)
        self.assertTrue(worker.kv_recv_thread.start.called)

    def test_both_role_merges_sender_and_receiver_live_results(self):
        worker = object.__new__(MooncakeConnectorWorker)
        send = MagicMock()
        recv = MagicMock()
        send.task_tracker.get_and_clear_split_results.return_value = {
            "sent": "success",
            "conflict": "success",
        }
        recv.task_tracker.get_and_clear_split_results.return_value = {
            "received": "success",
            "conflict": "failure",
        }
        worker.kv_send_thread = send
        worker.kv_recv_thread = recv

        self.assertEqual(
            worker.get_live_split_results(),
            {
                "sent": "success",
                "received": "success",
                "conflict": "failure",
            },
        )

    def test_both_role_reports_send_and_receive_completion(self):
        worker = object.__new__(MooncakeConnectorWorker)
        send = MagicMock()
        recv = MagicMock()
        send.get_and_clear_finished_requests.return_value = {"sent"}
        recv.get_and_clear_finished_requests.return_value = {"received"}
        worker.kv_send_thread = send
        worker.kv_recv_thread = recv
        worker.tp_rank = 0

        self.assertEqual(
            worker.get_finished(),
            ({"sent"}, {"received"}),
        )

    def test_shutdown_stops_and_joins_worker_thread(self):
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        thread = MagicMock()
        thread.is_alive.return_value = False
        thread.task_tracker.done_task_lock = contextlib.nullcontext()
        thread.task_tracker.split_leases = set()
        worker.kv_send_thread = thread

        worker.shutdown()

        thread.stop.assert_called_once_with()
        thread.join.assert_called_once()

    def test_shutdown_retains_listener_while_live_source_is_active(self):
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        thread = MagicMock()
        thread.task_tracker.done_task_lock = contextlib.nullcontext()
        thread.task_tracker.split_leases = {"req"}
        worker.kv_send_thread = thread

        with self.assertRaisesRegex(RuntimeError, "source ownership"):
            worker.shutdown()

        thread.stop.assert_not_called()

    def test_shutdown_closes_receiver_resources_after_join(self):
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        thread = MagicMock()
        thread.is_alive.return_value = False
        thread.split_request_lock = contextlib.nullcontext()
        thread.active_split_requests = {}
        thread.pending_split_signals = set()
        thread.undelivered_split_signals = set()
        thread.wait_for_split_signals.return_value = True
        worker.kv_recv_thread = thread

        worker.shutdown()

        thread.stop.assert_called_once_with()
        thread.join.assert_called_once()
        thread.close_resources.assert_called_once_with()

    def test_receiver_stop_keeps_terminal_ack_sender_alive(self):
        thread = object.__new__(KVCacheRecvingThread)
        thread.request_queue_lock = threading.Lock()
        thread.request_queue = queue.Queue()
        thread.stop_event = threading.Event()
        thread._accepting_requests = True

        thread.stop()

        self.assertFalse(thread._accepting_requests)
        self.assertFalse(thread.stop_event.is_set())
        self.assertIsNone(thread.request_queue.get_nowait())

    def test_shutdown_retains_receiver_while_terminal_signal_is_active(self):
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        thread = MagicMock()
        thread.is_alive.return_value = False
        thread.split_request_lock = contextlib.nullcontext()
        thread.active_split_requests = {}
        thread.pending_split_signals = {MagicMock()}
        thread.undelivered_split_signals = set()
        thread.wait_for_split_signals.return_value = False
        worker.kv_recv_thread = thread

        with self.assertRaisesRegex(RuntimeError, "signals remain active"):
            worker.shutdown()

        thread.close_resources.assert_not_called()

    def test_register_kv_caches_consumer(self):
        self.vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertIsNone(worker.kv_send_thread)
        self.assertIsNotNone(worker.kv_recv_thread)

    def test_local_parallel_config_validates_consumer_decode_tp(self):
        self.vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        self.vllm_config.kv_transfer_config.get_from_extra_config.side_effect = lambda k, d: {
            "prefill": {
                "tp_size": 2,
                "dp_size": 1,
                "pp_size": 1
            },
            "decode": {
                "tp_size": 8,
                "dp_size": 1,
                "pp_size": 1
            }
        }.get(k, d)

        with self.assertRaisesRegex(
                ValueError,
                r"decode\.tp_size \(8\).*--tensor-parallel-size \(2\)"):
            MooncakeConnectorWorker(self.vllm_config, self.engine_id)

    def test_local_parallel_config_validates_producer_prefill_tp(self):
        self.vllm_config.kv_transfer_config.get_from_extra_config.side_effect = lambda k, d: {
            "prefill": {
                "tp_size": 8,
                "dp_size": 1,
                "pp_size": 1
            },
            "decode": {
                "tp_size": 2,
                "dp_size": 1,
                "pp_size": 1
            }
        }.get(k, d)

        with self.assertRaisesRegex(
                ValueError,
                r"prefill\.tp_size \(8\).*--tensor-parallel-size \(2\)"):
            MooncakeConnectorWorker(self.vllm_config, self.engine_id)

    def test_local_parallel_config_validates_consumer_decode_dp(self):
        self.vllm_config.kv_transfer_config.kv_role = "kv_consumer"
        self.vllm_config.parallel_config.data_parallel_size = 2
        self.vllm_config.kv_transfer_config.get_from_extra_config.side_effect = (
            lambda key, default: {
                "prefill": {"tp_size": 2, "dp_size": 2, "pp_size": 1},
                "decode": {"tp_size": 2, "dp_size": 1, "pp_size": 1},
            }.get(key, default)
        )

        with self.assertRaisesRegex(
            ValueError,
            r"decode\.dp_size \(1\).*--data-parallel-size \(2\)",
        ):
            MooncakeConnectorWorker(self.vllm_config, self.engine_id)

    def test_local_parallel_config_both_accepts_matching_local_side(self):
        self.vllm_config.kv_transfer_config.kv_role = "kv_both"
        self.vllm_config.kv_transfer_config.get_from_extra_config.side_effect = (
            lambda k, d: {
                "prefill": {"tp_size": 8, "dp_size": 1, "pp_size": 1},
                "decode": {"tp_size": 2, "dp_size": 1, "pp_size": 1},
            }.get(k, d)
        )

        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)

        self.assertEqual(worker.kv_role, "kv_both")

    def test_local_parallel_config_both_rejects_unknown_local_side(self):
        self.vllm_config.kv_transfer_config.kv_role = "kv_both"
        self.vllm_config.kv_transfer_config.get_from_extra_config.side_effect = (
            lambda k, d: {
                "prefill": {"tp_size": 8, "dp_size": 1, "pp_size": 1},
                "decode": {"tp_size": 4, "dp_size": 1, "pp_size": 1},
            }.get(k, d)
        )

        with self.assertRaisesRegex(ValueError, "match either"):
            MooncakeConnectorWorker(self.vllm_config, self.engine_id)

    def test_register_kv_caches_mla_case(self):
        mla_cache1 = MagicMock()
        mla_cache1.size.return_value = (10, 16, 1, 16)
        mla_cache2 = MagicMock()
        mla_cache2.size.return_value = (10, 16, 1, 8)
        mla_caches = {"layer1": (mla_cache1, mla_cache2)}

        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.register_kv_caches(mla_caches)
        self.assertTrue(worker.use_mla)
        self.assertEqual(len(worker.block_len), 2)

    def test_register_kv_caches_deduplicates_underlying_storages(self):
        def cache(view_ptr, storage):
            tensor = MagicMock()
            tensor.data_ptr.return_value = view_ptr
            tensor.shape = (10, 16, 1, 8)
            tensor.size.side_effect = lambda dim=None: (
                tensor.shape if dim is None else tensor.shape[dim]
            )
            tensor.numel.return_value = math.prod(tensor.shape)
            tensor.element_size.return_value = 2
            tensor.untyped_storage.return_value = storage
            return tensor

        storage = MagicMock()
        storage.data_ptr.return_value = 0x1000
        storage.nbytes.return_value = 0x4000
        caches = {
            "layer1": (
                cache(0x1000, storage),
                cache(0x2000, storage),
            )
        }

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p."
            "mooncake_connector.global_te.register_buffer"
        ) as register_buffer:
            worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
            worker.register_kv_caches(caches)

        register_buffer.assert_called_once_with([0x1000], [0x4000])
        self.assertEqual(
            worker.xfer_handshake_metadata.kv_caches_base_addr,
            [0x1000, 0x2000],
        )

    def test_register_kv_caches_unbundled_dsa_uses_exact_regions(self):
        def cache(ptr, shape):
            tensor = MagicMock()
            tensor.data_ptr.return_value = ptr
            tensor.shape = shape
            tensor.size.side_effect = lambda dim=None: (
                shape if dim is None else shape[dim]
            )
            tensor.numel.return_value = math.prod(shape)
            tensor.element_size.return_value = 2
            return tensor

        latent_nope = cache(0x1000, (10, 16, 1, 512))
        latent_rope = cache(0x2000, (10, 16, 1, 64))
        indexer = cache(0x3000, (40, 16, 1, 128))

        self.vllm_config.kv_transfer_config.kv_role = "kv_consumer"
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p."
            "mooncake_connector.KVCacheRecvingThread"
        ) as recv_thread:
            worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
            worker.register_kv_caches(
                {
                    "model.layers.0.self_attn": (latent_nope, latent_rope),
                    "model.layers.0.self_attn.indexer": (indexer,),
                }
            )

        metadata = worker.xfer_handshake_metadata
        self.assertTrue(worker.use_sparse)
        self.assertEqual(metadata.buffer_group_ids, (0, 0, 1))
        self.assertEqual(
            metadata.kv_caches_buffer_sizes,
            tuple(
                math.prod(tensor.shape) * 2
                for tensor in (latent_nope, latent_rope, indexer)
            ),
        )
        recv_args = recv_thread.call_args.args
        self.assertEqual(recv_args[7], [0x1000, 0x2000])
        self.assertEqual(recv_thread.call_args.kwargs["ordinary_group_id"], 0)
        recv_instance = recv_thread.return_value
        self.assertEqual(
            recv_instance.local_registered_bases,
            (0x1000, 0x2000, 0x3000),
        )
        self.assertEqual(recv_instance.local_buffer_group_ids, (0, 0, 1))

    def test_register_kv_caches_index_only_uses_group_one_for_ordinary_transfer(self):
        indexer = MagicMock()
        indexer.data_ptr.return_value = 0x3000
        indexer.shape = (40, 16, 1, 128)
        indexer.size.side_effect = lambda dim=None: (
            indexer.shape if dim is None else indexer.shape[dim]
        )
        indexer.numel.return_value = math.prod(indexer.shape)
        indexer.element_size.return_value = 2
        self.vllm_config.kv_transfer_config.kv_role = "kv_consumer"

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p."
            "mooncake_connector.KVCacheRecvingThread"
        ) as recv_thread:
            worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
            worker.register_kv_caches(
                {"model.layers.0.self_attn.indexer": (indexer,)}
            )

        recv_args = recv_thread.call_args.args
        self.assertEqual(recv_args[7], [0x3000])
        self.assertEqual(recv_thread.call_args.kwargs["ordinary_group_id"], 1)
        recv_instance = recv_thread.return_value
        self.assertEqual(recv_instance.local_registered_bases, (0x3000,))
        self.assertEqual(recv_instance.local_buffer_group_ids, (1,))

    def test_device_id_selection_with_physical_devices(self):
        # Test with physical devices set
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        # Default tp_rank is 0, so device_id should be 10
        self.assertIsNotNone(worker.engine)

    def test_get_remote_tp_rank(self):

        def get_tp_rank(prefill_tp_size: int,
                        prefill_pp_size: int,
                        decode_tp_size: int,
                        num_kv_heads: int,
                        tp_num_need_pulls: int,
                        is_deepseek_mla: bool,
                        remote_ptp_size: Optional[int] = None):
            with patch(
                    'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config',
                    return_value=MagicMock()), \
                patch.object(self.vllm_config.kv_transfer_config, 'get_from_extra_config',
                            side_effect=lambda k, d=None: {
                                "prefill": {"tp_size": prefill_tp_size, "dp_size": 1, "pp_size": prefill_pp_size},
                                "decode": {"tp_size": decode_tp_size, "dp_size": 1, "pp_size": 1}
                            }.get(k, d)):
                self.vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
                self.vllm_config.parallel_config.tensor_parallel_size = decode_tp_size
                self.vllm_config.model_config.hf_text_config.num_key_value_heads = num_kv_heads
                self.vllm_config.model_config.is_deepseek_mla = is_deepseek_mla
                worker = MooncakeConnectorWorker(self.vllm_config,
                                                 self.engine_id)
                worker.tp_num_need_pulls = tp_num_need_pulls
                worker.use_sparse = 0
                return worker._get_remote_ranks_for_req(
                    'test', remote_ptp_size)

        self.assertIn(
            get_tp_rank(16, 1, 1, 4, 4, False)[0],
            [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]])
        self.assertIn(
            get_tp_rank(8, 1, 1, 4, 4, False)[0], [[0, 2, 4, 6], [1, 3, 5, 7]])
        self.assertIn(get_tp_rank(4, 1, 1, 4, 4, False)[0], [[0, 1, 2, 3]])
        self.assertIn(get_tp_rank(16, 1, 4, 4, 1, False),
                      [[[0], [4], [8], [12]], [[1], [5], [9], [13]],
                       [[2], [6], [10], [14]], [[3], [7], [11], [15]]])
        self.assertIn(get_tp_rank(8, 1, 4, 4, 1, False),
                      [[[0], [2], [4], [6]], [[1], [3], [5], [7]]])
        self.assertIn(get_tp_rank(4, 2, 2, 4, 2, False),
                      [[[0, 1, 4, 5], [2, 3, 6, 7]]])
        self.assertIn(get_tp_rank(4, 1, 4, 4, 1, False),
                      [[[0], [1], [2], [3]]])
        self.assertIn(
            get_tp_rank(8, 2, 1, 4, 4, False)[0],
            [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]])
        self.assertIn(get_tp_rank(4, 2, 2, 4, 2, False),
                      [[[0, 1, 4, 5], [2, 3, 6, 7]]])
        self.assertIn(get_tp_rank(2, 2, 1, 4, 2, False), [[[0, 1, 2, 3]]])
        self.assertIn(
            get_tp_rank(4, 4, 2, 8, 2, False),
            [[[0, 1, 4, 5, 8, 9, 12, 13], [2, 3, 6, 7, 10, 11, 14, 15]]])
        self.assertIn(
            get_tp_rank(4, 2, 1, 4, 4, False)[0], [[0, 1, 2, 3, 4, 5, 6, 7]])
        self.assertIn(
            get_tp_rank(4, 4, 1, 4, 4, False)[0],
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        self.assertIn(get_tp_rank(8, 2, 4, 4, 1, False),
                      [[[0, 8], [2, 10], [4, 12], [6, 14]],
                       [[1, 9], [3, 11], [5, 13], [7, 15]]])
        self.assertIn(get_tp_rank(4, 2, 4, 4, 4, False),
                      [[[0, 4], [1, 5], [2, 6], [3, 7]]])
        self.assertIn(
            get_tp_rank(4, 4, 4, 4, 1, False),
            [[[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]])
        self.assertIn(
            get_tp_rank(16, 1, 1, 1, 1,
                        True)[0], [[0], [1], [2], [3], [4], [5], [6], [7], [8],
                                   [9], [10], [11], [12], [13], [14], [15]])
        self.assertIn(get_tp_rank(4, 1, 4, 1, 1, True), [[[0], [1], [2], [3]]])
        self.assertIn(
            get_tp_rank(8, 2, 1, 1, 1, True)[0],
            [[0, 8], [2, 10], [4, 12], [6, 14], [1, 9], [3, 11], [5, 13],
             [7, 15]])
        self.assertIn(
            get_tp_rank(4, 4, 1, 1, 1, True)[0],
            [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]])
        self.assertIn(
            get_tp_rank(8, 2, 4, 1, 1, True)[0],
            [[0, 8], [2, 10], [4, 12], [6, 14], [1, 9], [3, 11], [5, 13],
             [7, 15]])
        self.assertIn(
            get_tp_rank(4, 4, 4, 1, 1, True),
            [[[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]])

        # check remote ptp size
        self.assertListEqual(get_tp_rank(16, 1, 2, 4, 2, False, 8),
                             get_tp_rank(8, 1, 2, 4, 2, False))
        self.assertListEqual(get_tp_rank(8, 1, 2, 4, 2, False, 4),
                             get_tp_rank(4, 1, 2, 4, 2, False))
        self.assertListEqual(get_tp_rank(4, 1, 2, 4, 1, False, 2),
                             get_tp_rank(2, 1, 2, 4, 1, False))

    def test_get_kv_split_metadata(self):

        def get_kv_split_metadata(use_mla,
                                  pcp_size,
                                  dcp_size,
                                  tp_size,
                                  tp_rank,
                                  pcp_rank,
                                  _prefill_tp_size,
                                  remote_pcp_size,
                                  remote_dcp_size,
                                  remote_port,
                                  remote_block_ids,
                                  local_block_ids,
                                  remote_engine_id,
                                  remote_ptp_size=None):

            worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)

            worker.use_mla = use_mla
            worker.pcp_size = pcp_size
            worker.dcp_size = dcp_size
            worker.tp_size = tp_size
            worker.tp_rank = tp_rank
            worker.pcp_rank = pcp_rank
            worker._prefill_tp_size = _prefill_tp_size
            worker.local_remote_block_port_mapping = {}
            worker.block_size = 16
            worker.num_key_value_heads = 1

            meta = types.SimpleNamespace()

            meta.remote_pcp_size = remote_pcp_size
            meta.remote_dcp_size = remote_dcp_size
            meta.remote_ptp_size = remote_ptp_size
            meta.remote_port = remote_port
            meta.remote_block_ids = remote_block_ids
            meta.local_block_ids = local_block_ids
            meta.num_external_tokens = pcp_size * dcp_size * len(
                local_block_ids) * worker.block_size
            meta.num_prompt_blocks = pcp_size * dcp_size * len(local_block_ids)
            meta.remote_engine_id = remote_engine_id

            remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = worker._get_kv_split_metadata(
                '0', meta)

            return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

        self.assertEqual(
            get_kv_split_metadata(True, 1, 1, 8, 1, 0, 8, 1, 8, 30000, [1],
                                  [1], 0),
            ([[30001], [30002], [30003], [30004], [30005], [30006], [30007],
              [30000]], [[], [], [], [], [], [], [], [1]], [[], [], [], [], [],
                                                            [], [], [1]]))

        self.assertEqual(
            get_kv_split_metadata(False, 1, 1, 8, 1, 0, 8, 2, 8, 30000, [1],
                                  [1], 0),
            ([[30001], [30002], [30003], [30004], [30005], [30006], [30007],
              [30008], [30009], [30010], [30011], [30012], [30013], [30014],
              [30015], [30000]
              ], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                  [1]], [[], [], [], [], [], [], [], [], [], [], [], [], [],
                         [], [], [1]]))

        self.assertEqual(
            get_kv_split_metadata(True, 1, 1, 8, 1, 0, 8, 2, 2, 30000, [1],
                                  [1], 0),
            ([[30001], [30008], [30009], [30000]], [[], [], [], [1]
                                                    ], [[], [], [], [1]]))

        self.assertEqual(
            get_kv_split_metadata(False, 1, 1, 8, 1, 0, 8, 2, 2, 30000, [1],
                                  [1], 0),
            ([[30001], [30008], [30009], [30000]], [[], [], [], [1]
                                                    ], [[], [], [], [1]]))

        self.assertEqual(
            get_kv_split_metadata(True, 1, 2, 8, 1, 0, 8, 2, 2, 30000, [1],
                                  [1], 0),
            ([[30000], [30008]], [[1], []], [[1], []]))

        self.assertEqual(
            get_kv_split_metadata(False, 1, 2, 8, 1, 0, 8, 2, 2, 30000, [1],
                                  [1], 0),
            ([[30000], [30008]], [[1], []], [[1], []]))

        self.assertEqual(
            get_kv_split_metadata(True, 1, 2, 8, 0, 0, 8, 2, 2, 30000,
                                  [1, 2, 3], [1, 2, 3, 4, 5], 0),
            ([[30000], [30008]], [[1, 2, 3], [4, 5]], [[1, 2, 3], [1, 2]]))

        # check remote ptp size
        self.assertEqual(
            get_kv_split_metadata(True, 1, 1, 8, 1, 0, 8, 1, 8, 30000, [1],
                                  [1], 0, 16),
            get_kv_split_metadata(True, 1, 1, 8, 1, 0, 16, 1, 8, 30000, [1],
                                  [1], 0)
        )
        self.assertEqual(
            get_kv_split_metadata(False, 1, 1, 8, 1, 0, 8, 1, 8, 30000, [1],
                                  [1], 0, 16),
            get_kv_split_metadata(False, 1, 1, 8, 1, 0, 16, 1, 8, 30000, [1],
                                  [1], 0)
        )
        self.assertEqual(
            get_kv_split_metadata(False, 1, 1, 8, 1, 0, 8, 2, 8, 30000, [1],
                                  [1], 0, 16),
            get_kv_split_metadata(False, 1, 1, 8, 1, 0, 16, 2, 8, 30000, [1],
                                  [1], 0)
        )

    def test_get_tp_num_need_pulls(self):
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.num_key_value_heads = 8

        tp_num_need_pulls = worker._get_tp_num_need_pulls(prefill_tp_size=4)
        self.assertEqual(tp_num_need_pulls, 1)

        worker.vllm_config.model_config.is_deepseek_mla = False
        tp_num_need_pulls = worker._get_tp_num_need_pulls(prefill_tp_size=4)
        self.assertEqual(tp_num_need_pulls, 2)

        tp_num_need_pulls = worker._get_tp_num_need_pulls(prefill_tp_size=None)
        self.assertEqual(tp_num_need_pulls, 1)


if __name__ == '__main__':
    unittest.main()
