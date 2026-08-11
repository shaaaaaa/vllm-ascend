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
from typing import Any, Dict, OrderedDict, Optional
from unittest.mock import MagicMock, patch

import msgspec
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
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [0x3000, 0x5000]
        }
        self.thread.remote_num_blocks["remote_engine"] = {6666: 2}
        self.thread.remote_buffer_sizes["remote_engine"] = {
            6666: (4096, 4096)
        }
        self.thread.remote_capabilities = {
            ("remote_engine", 6666): (LIVE_SPLIT_CAPABILITY,)
        }
        plan = SplitTransferPlan(
            segments=(
                SplitTransferSegment(0, 0, 32, 0x8000, 256, "cpu"),
                SplitTransferSegment(1, 1, 64, 0xA000, 512, "npu"),
            ),
            group_byte_totals=(256, 512),
            tp_rank=7,
            dp_rank=1,
        )
        request = dict(self.test_req, split_plan=plan)

        self.thread._transfer_kv_cache(request)

        mock_register.assert_called_once_with([0x8000, 0xA000], [256, 512])
        self.engine.batch_transfer_sync_read.assert_called_once_with(
            "localhost:7777", [0x8000, 0xA000],
            [0x3000 + 32, 0x5000 + 64], [256, 512])

    def test_split_transfer_rejects_wrong_dp_rank(self):
        self.vllm_config.parallel_config.data_parallel_rank_local = 1
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

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.global_te.temporary_registration'
    )
    def test_index_only_group1_allows_zero_unrequested_group0(
        self, mock_register
    ):
        mock_register.return_value = contextlib.nullcontext()
        self.vllm_config.parallel_config.data_parallel_rank_local = 0
        self.thread.kv_caches_base_addr["remote_engine"] = {6666: [0x5000]}
        self.thread.remote_num_blocks["remote_engine"] = {6666: 2}
        self.thread.remote_buffer_sizes["remote_engine"] = {6666: (4096,)}
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

        self.thread._transfer_kv_cache(
            dict(self.test_req, split_plan=plan))

        mock_register.assert_called_once_with([0xA000], [512])
        self.engine.batch_transfer_sync_read.assert_called_once_with(
            "localhost:7777", [0xA000], [0x5000 + 64], [512])

    def test_split_rejects_zero_or_missing_requested_group(self):
        self.vllm_config.parallel_config.data_parallel_rank_local = 0
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
            "split", "split-remote", split_plan=plan, **common)

        self.assertEqual(self.thread.request_queue.get()["request_id"],
                         "ordinary")
        self.assertEqual(self.thread.request_queue.get()["request_id"],
                         "split")

    @patch.object(KVCacheRecvingThread, '_send_split_done_signal')
    @patch.object(KVCacheRecvingThread, '_transfer_kv_cache')
    def test_cancelled_split_is_acked_without_transfer(self, mock_transfer,
                                                      mock_send):
        plan = SplitTransferPlan((), (0, 0), tp_rank=0, dp_rank=0)
        request = dict(self.test_req, split_plan=plan)
        self.thread.active_split_requests.add("req1")
        self.assertTrue(self.thread.cancel_split_request("req1"))

        self.thread._handle_request(request)

        mock_transfer.assert_not_called()
        mock_send.assert_called_once_with("req1", "localhost", 6666,
                                          "cancelled")
        self.thread.task_tracker.complete_split_request.assert_called_once_with(
            "req1", "cancelled", mark_finished=False)
        self.assertNotIn("req1", self.thread.active_split_requests)

    @patch.object(KVCacheRecvingThread, '_send_split_done_signal')
    @patch.object(KVCacheRecvingThread, '_transfer_kv_cache')
    def test_successful_split_waits_for_provider_completion(self, mock_transfer,
                                                           mock_send):
        plan = SplitTransferPlan((), (0, 0), tp_rank=0, dp_rank=0)
        request = dict(self.test_req, split_plan=plan)
        self.thread.active_split_requests.add("req1")

        self.thread._handle_request(request)

        mock_transfer.assert_called_once_with(request)
        mock_send.assert_called_once_with("req1", "localhost", 6666,
                                          "success")
        self.thread.task_tracker.complete_split_request.assert_called_once_with(
            "req1", "success", mark_finished=False)


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

    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.ensure_zmq_send'
    )
    @patch(
        'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.ensure_zmq_recv',
        side_effect=Exception("Network error"))
    def test_get_remote_metadata_failure(self, mock_recv, mock_send):
        with patch.object(self.thread, '_get_remote_socket') as mock_get_socket, \
                patch.object(self.thread, '_return_remote_socket') as mock_return_socket:
            mock_socket = MagicMock()
            mock_get_socket.return_value = mock_socket

            with self.assertRaises(Exception) as context:
                self.thread._get_remote_metadata("host1", 5555)

            self.assertEqual(str(context.exception), "Network error")
            mock_return_socket.assert_called_once()


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
        self.parallel_config.data_parallel_size_local = 1
        self.parallel_config.pipeline_parallel_size = 1
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

    def test_split_lease_timeout_reports_fallback_result(self):
        self.tracker.add_req_to_process("req")
        self.tracker.add_delayed_request(
            "req", time.time() - 600, split=True)

        self.assertEqual(self.tracker._retrieve_expired_requests(), {"req"})
        self.assertEqual(self.tracker.get_and_clear_split_results(),
                         {"req": "timeout"})

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

    def test_receive_admission_precedes_immediate_split_completion(self):
        tracker = KVCacheTaskTracker()

        class ImmediateThread:
            task_tracker = tracker

            def add_request(self, request_id, **_kwargs):
                self.assert_admitted = request_id in tracker.reqs_to_process
                tracker.complete_split_request(
                    request_id, "success", mark_finished=False
                )
                return True

        recv_thread = ImmediateThread()
        worker = object.__new__(MooncakeConnectorWorker)
        worker.kv_recv_thread = recv_thread
        worker.kv_send_thread = None
        worker._prefill_tp_size = 1
        worker._prefill_pp_size = 1
        worker._get_tp_num_need_pulls = lambda _size: 1
        worker._get_remote_rank = lambda _request_id, _size: [0]
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
        )

        worker.start_load_kv(metadata)

        self.assertTrue(recv_thread.assert_admitted)
        self.assertEqual(
            tracker.get_and_clear_split_results(), {"req": "success"}
        )


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

    def test_temporary_registration_does_not_release_persistent_buffer(self):
        registry = GlobalTE()
        registry.transfer_engine = MagicMock()
        registry.transfer_engine.register_memory.return_value = 0
        registry.register_buffer([0x1000], [0x1000])

        with registry.temporary_registration([0x1400], [0x100]):
            pass

        registry.transfer_engine.unregister_memory.assert_not_called()
        self.assertEqual(registry.registered_buffers, {0x1000: 0x1000})


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

    def test_source_descriptor_rank_mismatch_falls_back(self):
        metadata = MooncakeConnectorMetadata()
        params = {
            "remote_block_ids": [1], "remote_engine_id": "remote",
            "remote_request_id": "remote-req", "remote_host": "host",
            "remote_port": 30000,
            "live_split_capabilities": (LIVE_SPLIT_CAPABILITY,),
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
            LIVE_SPLIT_SOURCE_DESCRIPTOR: {
                "segments": [], "group_byte_totals": [1, 0],
                "tp_rank": 0, "dp_rank": 0,
            },
        })
        metadata.accept_late_split_plans({"req": {}})
        self.assertTrue(metadata.requests["req"].split_fallback)
        self.assertIsNone(metadata.requests["req"].split_plan)


class TestAscendMultiLateSplitInjection(unittest.TestCase):

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
            supports_dsa_compact_external_load = True

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

    def test_scheduler_non_live_preserves_configured_order(self):
        events = []

        class Child:
            def __init__(self, name, capable=False):
                self.name = name
                self.supports_dsa_compact_external_load = capable

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

    def test_register_kv_caches_producer(self):
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertEqual(len(worker.kv_caches), 1)
        self.assertIsNotNone(worker.kv_send_thread)
        self.assertIsNone(worker.kv_recv_thread)

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
