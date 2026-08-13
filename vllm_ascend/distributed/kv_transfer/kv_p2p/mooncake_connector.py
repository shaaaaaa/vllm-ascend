# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import hashlib
import json
import math
import os
import queue
import random
import struct
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
import numpy as np
import numpy.typing as npt
import torch
import torch_npu
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

from vllm_ascend import envs as ascend_envs
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import get_transfer_timeout_value
from vllm_ascend.utils import enable_custom_op, is_vl_model

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
SPLIT_DONE_MSG = b"split_done_msg_v2"
LIVE_SPLIT_CAPABILITY = "ascend_live_split_v2"
LIVE_SPLIT_SOURCE_DESCRIPTOR = "ascend_live_split_source_v1"
MAX_PENDING_SPLIT_REQUESTS = 64
MAX_TASK_HISTORY_SIZE = 16000
MAX_SPLIT_TRANSFER_ID_LENGTH = 64
SPLIT_DONE_MAX_ATTEMPTS = 3
CONTROL_ACK_MAX_ATTEMPTS = 3
THREAD_SHUTDOWN_TIMEOUT = 30.0


def _cold_live_log(event: str, **fields: Any) -> None:
    if os.environ.get("LMCACHE_COLD_START_PERF", "0").lower() in (
        "", "0", "false", "no", "off"
    ):
        return
    logger.info(
        "[LMCACHE_COLD_PERF] %s",
        json.dumps(
            {
                "schema": 1,
                "event": event,
                "pid": os.getpid(),
                "monotonic_ms": round(time.perf_counter() * 1000, 3),
                **fields,
            },
            default=str,
            separators=(",", ":"),
        ),
    )


class RemotePortInfo(TypedDict):
    num: int
    host: str


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    te_rpc_port: int
    kv_caches_base_addr: list[int]
    num_blocks: int
    local_ip: str = ""
    capabilities: tuple[str, ...] = ()
    kv_caches_buffer_sizes: tuple[int, ...] = ()
    buffer_group_ids: tuple[int, ...] = ()
    tp_rank: int = 0
    dp_rank: int = 0


@dataclass(frozen=True)
class SplitTransferSegment:
    """One registered local destination and its remote KV source extent."""

    group_id: int
    source_buffer_index: int
    source_offset: int
    destination_address: int
    length: int
    destination_kind: str
    source_buffer_base: int | None = None


@dataclass(frozen=True)
class SplitSourceSegment:
    """Opaque prefiller-owned extent in a registered source buffer."""

    group_id: int
    source_buffer_index: int
    source_buffer_base: int
    source_offset: int
    length: int


@dataclass(frozen=True)
class SplitSourceDescriptor:
    segments: tuple[SplitSourceSegment, ...]
    group_byte_totals: tuple[int, int]
    tp_rank: int
    dp_rank: int


@dataclass(frozen=True)
class SplitTransferPlan:
    segments: tuple[SplitTransferSegment, ...]
    group_byte_totals: tuple[int, int]
    tp_rank: int
    dp_rank: int
    requested_groups: tuple[int, ...] = (0, 1)


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    num_external_tokens: int
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    remote_pcp_size: int
    remote_dcp_size: int
    remote_ptp_size: int | None
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]]
    num_prompt_blocks: int
    split_plan: SplitTransferPlan | None = None
    split_negotiated: bool = False
    split_fallback: bool = False
    split_source: tuple[SplitSourceDescriptor, ...] | None = None
    split_source_invalid: bool = False
    split_transfer_id: str | None = None


@dataclass
class SizedDict(OrderedDict):
    def __init__(self, max_size=16000, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            value: dict[int, list[int]] = {}
            self[key] = value
            return value


class KVCacheTaskTracker:
    def __init__(self):
        super().__init__()

        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()
        # Only used in prefill node. Tracks requests whose kv blocks freeing is
        # intentionally delayed. Each entry is a tuple of (request_id,
        # timestamp). If a request remains in this queue for too long, it will
        # be force-freed.
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.reqs_to_process: set[str] = set()
        self.split_results: dict[str, str] = {}
        self.split_leases: set[str] = set()
        self.split_terminal_requests: OrderedDict[str, None] = OrderedDict()
        self.split_transfer_ids: dict[str, str] = {}
        self.early_split_results: OrderedDict[tuple[str, str], str] = (
            OrderedDict()
        )

    def add_req_to_process(
        self, request_id: str, split_transfer_id: str | None = None
    ):
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                return
            self.finished_requests.discard(request_id)
            self.delayed_free_requests.pop(request_id, None)
            self.split_results.pop(request_id, None)
            self.split_leases.discard(request_id)
            self.split_terminal_requests.pop(request_id, None)
            if split_transfer_id is None:
                self.split_transfer_ids.pop(request_id, None)
            else:
                for key in tuple(self.early_split_results):
                    if key[0] == request_id and key[1] != split_transfer_id:
                        self.early_split_results.pop(key, None)
                self.split_transfer_ids[request_id] = split_transfer_id
            self.reqs_to_process.add(request_id)

    def add_not_transfer_request(self, request_id: str):
        with self.done_task_lock:
            self.finished_requests.add(request_id)
            self.reqs_to_process.discard(request_id)

    def update_done_task_count(self, request_id: str):
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.finished_requests.add(request_id)
                self.reqs_to_process.discard(request_id)
                self.delayed_free_requests.pop(request_id, None)
            else:
                logger.error(
                    "MooncakeConnector finish req not in reqs to process."
                    "If it is a P node, this request may have been force freed."
                )

    def complete_split_request(
        self,
        request_id: str,
        status: str,
        mark_finished: bool = True,
        split_transfer_id: str | None = None,
    ):
        """Record one terminal ACK; duplicate ACKs are harmless."""
        with self.done_task_lock:
            current = self.split_transfer_ids.get(request_id)
            if current is not None and split_transfer_id is None:
                return False
            if split_transfer_id is not None:
                if current is None:
                    self.early_split_results[
                        (request_id, split_transfer_id)
                    ] = status
                    if len(self.early_split_results) > MAX_TASK_HISTORY_SIZE:
                        self.early_split_results.popitem(last=False)
                    return True
                if current != split_transfer_id:
                    return False
            if request_id in self.split_terminal_requests:
                return True
            self.split_terminal_requests[request_id] = None
            if len(self.split_terminal_requests) > MAX_TASK_HISTORY_SIZE:
                self.split_terminal_requests.popitem(last=False)
            self.split_results[request_id] = status
            if mark_finished:
                self.finished_requests.add(request_id)
            self.reqs_to_process.discard(request_id)
            self.delayed_free_requests.pop(request_id, None)
            self.split_leases.discard(request_id)
            self.split_transfer_ids.pop(request_id, None)
            return True

    def get_and_clear_split_results(self) -> dict[str, str]:
        with self.done_task_lock:
            results = dict(self.split_results)
            self.split_results.clear()
            return results

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            expired_requests = self._retrieve_expired_requests()
            finished_requests.update(expired_requests)
            self.finished_requests.clear()
        return finished_requests

    def add_delayed_request(
        self,
        request_id: str,
        delay_start_time: float,
        split: bool = False,
        split_transfer_id: str | None = None,
    ):
        """Add a delayed free request."""
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.delayed_free_requests[request_id] = delay_start_time
                if split:
                    if split_transfer_id is not None:
                        self.split_transfer_ids[request_id] = split_transfer_id
                    self.split_leases.add(request_id)
                    early_status = self.early_split_results.pop(
                        (request_id, split_transfer_id), None
                    )
                    if early_status is not None:
                        self.split_terminal_requests[request_id] = None
                        self.split_results[request_id] = early_status
                        self.finished_requests.add(request_id)
                        self.reqs_to_process.discard(request_id)
                        self.delayed_free_requests.pop(request_id, None)
                        self.split_leases.discard(request_id)
                        self.split_transfer_ids.pop(request_id, None)

    def _retrieve_expired_requests(self):
        """Retrieve all expired delayed requests."""
        expired_requests: set[str] = set()
        # Free delayed requests if they exceed the timeout
        current_time = time.time()
        for request_id, delay_start_time in tuple(
            self.delayed_free_requests.items()
        ):
            if (
                current_time - delay_start_time
                <= envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT
            ):
                break
            if request_id in self.split_leases:
                # A wall-clock timeout does not fence a remote native read.
                # Keep live-source blocks pinned until its terminal ACK.
                continue
            self.delayed_free_requests.pop(request_id, None)
            self.reqs_to_process.discard(request_id)
            expired_requests.add(request_id)
            logger.info("Force freed request: %s", request_id)
        return expired_requests


class KVCacheSendingThread(threading.Thread):
    def __init__(
        self,
        vllm_config: VllmConfig,
        tp_rank: int,
        prefill_tp_size: int,
        local_engine_id: str,
        side_channel_host: str,
        side_channel_port: int,
        metadata: MooncakeAgentMetadata,
        ready_event: threading.Event,
        kv_caches: dict[str, Any],
        pcp_rank: int,
    ):
        super().__init__(daemon=True, name="KVCacheSendingThread")
        self.tp_rank = tp_rank
        self.prefill_tp_size = prefill_tp_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.local_engine_id = local_engine_id
        self.side_channel_host = side_channel_host
        self.side_channel_port = side_channel_port
        self.metadata = metadata
        self.ready_event = ready_event
        self.kv_caches = kv_caches
        self.pcp_rank = pcp_rank
        self.port_send_num: dict[str, int] = {}
        self.stop_event = threading.Event()

        self.task_tracker = KVCacheTaskTracker()

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def add_not_transfer_request(self, request_id: str):
        self.task_tracker.add_not_transfer_request(request_id)

    def add_delayed_request(
        self,
        request_id: str,
        delay_start_time: float,
        split: bool = False,
        split_transfer_id: str | None = None,
    ):
        return self.task_tracker.add_delayed_request(
            request_id,
            delay_start_time,
            split=split,
            split_transfer_id=split_transfer_id,
        )

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            # Listen for new requests for metadata. NOTE(rob): we need each rank
            # to have a unique port. This hack to keeps us moving. We will
            # switch when moving to etcd or where we have a single ZMQ socket in
            # the scheduler.
            device_index = self.pp_rank * self.tp_size + self.tp_rank + self.pcp_rank * self.prefill_tp_size
            handshake_port = self.side_channel_port + device_index
            path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
            logger.info("Starting listening on path: %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                sock.setsockopt(zmq.RCVTIMEO, 100)
                self.ready_event.set()
                self.run_busy_loop(sock)
        except Exception as e:
            logger.error("Mooncake KVCacheSendingThread exception: %s", e, exc_info=True)

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
        while not self.stop_event.is_set():
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error("Invalid message format: %s", frames)
                    continue

                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error("Invalid message format: %s", frames)
                    continue

                msg = decoder.decode(payload[0])
                if msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", encoded_data))
                elif msg[0] == DONE_RECVING_MSG:
                    logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
                    request_id = msg[1]
                    remote_port_send_num = msg[2]
                    if remote_port_send_num:
                        if request_id not in self.port_send_num:
                            self.port_send_num[request_id] = 0
                        self.port_send_num[request_id] += 1
                        device_index = self.pp_rank * self.tp_size + self.tp_rank + self.pcp_rank * self.prefill_tp_size
                        handshake_port = self.side_channel_port + device_index
                        if self.port_send_num[request_id] >= remote_port_send_num[handshake_port]["num"]:
                            self.task_tracker.update_done_task_count(request_id)
                            del self.port_send_num[request_id]
                    else:
                        self.task_tracker.update_done_task_count(request_id)
                    _send_router_ack(sock, identity, request_id)
                elif msg[0] == SPLIT_DONE_MSG:
                    if len(msg) not in (3, 4):
                        raise ValueError("Invalid split completion message")
                    request_id, status = msg[1:3]
                    split_transfer_id = msg[3] if len(msg) == 4 else None
                    if (
                        split_transfer_id is not None
                        and not isinstance(split_transfer_id, str)
                    ):
                        raise ValueError("Invalid split transfer identity")
                    if status not in ("success", "failure", "cancelled", "fallback"):
                        status = "failure"
                    self.task_tracker.complete_split_request(
                        request_id,
                        status,
                        split_transfer_id=split_transfer_id,
                    )
                    _send_router_ack(sock, identity, request_id)
                else:
                    logger.error("Connection listener got unexpected message %s", msg)
            except zmq.Again:
                continue
            except Exception as e:
                logger.error("Connection listener got exception %s: %s", type(e), e)

    def stop(self) -> None:
        self.stop_event.set()


class KVCacheRecvingThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        _prefill_pp_size: int,
        engine: TransferEngine,
        local_engine_id: str,
        local_handshake_port: int,
        side_channel_port: int,
        local_kv_caches_base_addr: list[int],
        block_len: list[int],
        ready_event: threading.Event,
        vllm_config: VllmConfig,
        kv_caches: dict[str, Any],
        prefill_pp_layer_partition: str | None = None,
        ordinary_group_id: int = 0,
    ):
        super().__init__(daemon=True, name="KVCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._prefill_pp_size = _prefill_pp_size
        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.side_channel_port = side_channel_port
        self.engine = engine
        self.ready_event = ready_event

        self.kv_caches = kv_caches
        self.kv_caches_base_addr: dict[str, dict[int, list[int]]] = SizedDict()
        self.kv_caches_base_addr[local_engine_id][local_handshake_port] = local_kv_caches_base_addr
        self.ordinary_group_id = ordinary_group_id
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_num_blocks: dict[str, dict[int, int]] = SizedDict()
        self.remote_buffer_sizes: dict[str, dict[int, tuple[int, ...]]] = (
            SizedDict()
        )
        self.remote_buffer_group_ids: dict[str, dict[int, tuple[int, ...]]] = (
            SizedDict()
        )
        self.block_len = block_len
        # TODO(jianzs): find a better way to detect MLA.
        self.use_mla = len(block_len) == 2

        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.request_queue_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.split_request_lock = threading.Lock()
        self.active_split_requests: dict[str, str | None] = {}
        self.completed_split_requests: OrderedDict[
            tuple[str, str | None], str
        ] = OrderedDict()
        self.cancelled_split_requests: set[tuple[str, str | None]] = set()
        self.pending_split_signals: set[Future[Any]] = set()
        self.undelivered_split_signals: set[tuple[str, str | None]] = set()
        self.stop_event = threading.Event()

        self.task_tracker = KVCacheTaskTracker()

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds

        self.vllm_config = vllm_config
        self.model_config = self.vllm_config.model_config
        self.block_size = self.vllm_config.cache_config.block_size
        self.num_layers = self.model_config.hf_text_config.num_hidden_layers
        self.pp_layer_indices = {
            rank: get_prefill_pp_indices(self.num_layers, rank, self._prefill_pp_size, prefill_pp_layer_partition)
            for rank in range(self._prefill_pp_size)
        }
        if not is_vl_model(vllm_config):
            if self.use_mla:
                self.k_head_dim = self.model_config.hf_text_config.kv_lora_rank
                self.v_head_dim = self.model_config.hf_text_config.qk_rope_head_dim
                self.num_kv_heads = 1
            else:
                self.k_head_dim = self.model_config.hf_text_config.head_dim
                self.v_head_dim = self.model_config.hf_text_config.head_dim
                self.num_kv_heads = max(self.model_config.hf_text_config.num_key_value_heads // self.tp_size, 1)
        self.proc_not_transfer_request: dict[str, bool] = {}

    def add_request(
        self,
        request_id: str,
        remote_request_id: str,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
        offset: int,
        tp_num_need_pulls: int,
        remote_port_send_num: dict[int, RemotePortInfo] | None = None,
        all_task_done: bool = False,
        split_plan: SplitTransferPlan | None = None,
        split_transfer_id: str | None = None,
    ):
        """Add a new request to the queue for processing."""
        if remote_port_send_num is None:
            remote_port_send_num = {}
        logger.debug(f"Adding request {request_id} to the queue.")
        request_data = {
                "request_id": request_id,
                "local_block_ids": local_block_ids,
                "remote_block_ids": remote_block_ids,
                "remote_engine_id": remote_engine_id,
                "remote_request_id": remote_request_id,
                "remote_host": remote_host,
                "remote_handshake_port": remote_handshake_port,
                "offset": offset,
                "tp_num_need_pulls": tp_num_need_pulls,
                "remote_port_send_num": remote_port_send_num,
                "all_task_done": all_task_done,
                "split_plan": split_plan,
                "split_transfer_id": split_transfer_id,
        }
        with self.request_queue_lock:
            if self.stop_event.is_set():
                return False
            if split_plan is not None:
                generation = (request_id, split_transfer_id)
                with self.split_request_lock:
                    completed = self.completed_split_requests.get(generation)
                    if completed is not None:
                        return True
                    if (
                        len(self.undelivered_split_signals)
                        >= MAX_PENDING_SPLIT_REQUESTS
                    ):
                        return False
                    active = self.active_split_requests.get(request_id)
                    if request_id in self.active_split_requests:
                        return active == split_transfer_id
                    if self.request_queue.qsize() >= MAX_PENDING_SPLIT_REQUESTS:
                        return False
                    self.active_split_requests[request_id] = split_transfer_id
                self.request_queue.put_nowait(request_data)
            else:
                self.request_queue.put(request_data)
        return True

    def cancel_split_request(self, request_id: str) -> bool:
        with self.split_request_lock:
            if request_id not in self.active_split_requests:
                return False
            split_transfer_id = self.active_split_requests[request_id]
            self.cancelled_split_requests.add((request_id, split_transfer_id))
            return True

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    self.request_queue.task_done()
                    break
                self._handle_request(request_data)
            except Exception as e:
                logger.error(f"Error in KVCacheTransferThread: {e}")

    def stop(self) -> None:
        with self.request_queue_lock:
            if self.stop_event.is_set():
                return
            self.stop_event.set()
            self.request_queue.put(None)

    def close_resources(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)
        with self.remote_sockets_lock:
            sockets = [
                socket
                for pooled in self.remote_sockets.values()
                for socket in pooled
            ]
            self.remote_sockets.clear()
        for socket in sockets:
            self._discard_remote_socket(socket)

    def _submit_split_done_signal(
        self, generation: tuple[str, str | None], *args: Any
    ) -> None:
        future = self.executor.submit(self._send_split_done_signal, *args)
        with self.split_request_lock:
            self.pending_split_signals.add(future)
        future.add_done_callback(
            lambda done: self._split_done_signal_finished(generation, done)
        )

    def _signal_split_completion(
        self,
        request_id: str,
        remote_request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        status: str,
        split_transfer_id: str | None,
    ) -> None:
        generation = (request_id, split_transfer_id)
        with self.split_request_lock:
            if generation in self.completed_split_requests:
                return
            self.completed_split_requests[generation] = status
            self.undelivered_split_signals.add(generation)
            if len(self.completed_split_requests) > MAX_TASK_HISTORY_SIZE:
                for old_generation in tuple(self.completed_split_requests):
                    if old_generation not in self.undelivered_split_signals:
                        self.completed_split_requests.pop(old_generation)
                        break
        self._submit_split_done_signal(
            generation,
            remote_request_id,
            remote_host,
            remote_handshake_port,
            status,
            split_transfer_id,
        )

    def _split_done_signal_finished(
        self, generation: tuple[str, str | None], future: Future[Any]
    ) -> None:
        with self.split_request_lock:
            self.pending_split_signals.discard(future)
        try:
            delivered = future.result()
        except Exception:
            logger.exception("Mooncake split completion signal task failed")
            return
        if delivered:
            with self.split_request_lock:
                self.undelivered_split_signals.discard(generation)

    def _handle_request(self, req_meta: dict[str, Any]):
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        remote_port_send_num = req_meta["remote_port_send_num"]
        all_task_done = req_meta["all_task_done"]

        split_plan = req_meta.get("split_plan")
        split_transfer_id = req_meta.get("split_transfer_id")
        split_status = "success"
        try:
            if split_plan is not None:
                with self.split_request_lock:
                    if (request_id, split_transfer_id) in self.cancelled_split_requests:
                        split_status = "cancelled"
            if split_status == "cancelled":
                raise InterruptedError("Split transfer was cancelled")
            logger.debug(f"Starting to transfer KV cache for request {remote_request_id}.")
            self._transfer_kv_cache(req_meta)
            if split_plan is not None:
                with self.split_request_lock:
                    if (request_id, split_transfer_id) in self.cancelled_split_requests:
                        split_status = "cancelled"
            logger.debug(f"Finished transferring KV cache for request {remote_request_id}.")
        except InterruptedError:
            logger.info("Cancelled split transfer for request %s", remote_request_id)
        except Exception as e:
            split_status = "failure"
            logger.error(f"Failed to transfer KV cache for request {remote_request_id}: {e}", exc_info=True)
        finally:
            if split_plan is None:
                self._send_done_signal_to_free_remote_port(
                    remote_request_id, remote_host, remote_port_send_num)
                if all_task_done:
                    if len(req_meta["local_block_ids"]) > 0:
                        self.task_tracker.update_done_task_count(request_id)
                    if request_id in self.proc_not_transfer_request:
                        del self.proc_not_transfer_request[request_id]
                self.request_queue.task_done()
                # Ordinary transfers retain their original completion message.
                self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port, remote_port_send_num)
            else:
                self.request_queue.task_done()
                self.task_tracker.complete_split_request(
                    request_id, split_status,
                    mark_finished=False,
                    split_transfer_id=split_transfer_id)
                with self.split_request_lock:
                    if self.active_split_requests.get(request_id) == split_transfer_id:
                        self.active_split_requests.pop(request_id, None)
                    self.cancelled_split_requests.discard(
                        (request_id, split_transfer_id))
                self._signal_split_completion(
                    request_id,
                    remote_request_id,
                    remote_host,
                    remote_handshake_port,
                    split_status,
                    split_transfer_id,
                )

    def _send_done_signal_to_free_remote_port(
        self, request_id: str, remote_host: str, remote_port_send_num: dict[int, RemotePortInfo]
    ):
        if self.side_channel_port != self.local_handshake_port or not remote_port_send_num:
            return
        if request_id not in self.proc_not_transfer_request:
            self.proc_not_transfer_request[request_id] = True
        if self.proc_not_transfer_request[request_id]:
            for remote_port in remote_port_send_num:
                if remote_port_send_num[remote_port]["num"] == 0:
                    remote_host_ = remote_port_send_num[remote_port]["host"]
                    self._send_done_recv_signal(request_id, remote_host_, remote_port, remote_port_send_num)
            self.proc_not_transfer_request[request_id] = False

    def _transfer_kv_cache(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        remote_request_id = req_meta["remote_request_id"]
        remote_block_ids = req_meta["remote_block_ids"]
        local_block_ids = req_meta["local_block_ids"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        offset = req_meta["offset"]
        tp_num_need_pulls = req_meta["tp_num_need_pulls"]

        split_plan = req_meta.get("split_plan")
        _cold_live_log(
            "live_source_transfer_dispatch_entry",
            req_id=req_meta.get("request_id"),
            remote_req_id=remote_request_id,
            split_plan_present=split_plan is not None,
        )
        if split_plan is not None:
            self._transfer_split_destinations(req_meta, split_plan)
            return

        # Full prefix cache hit: do not need to read remote blocks, just notify
        # P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            return

        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

        # Check if we have the remote metadata cached.
        if (
            remote_engine_id not in self.kv_caches_base_addr
            or remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]
        ):
            self._get_remote_metadata(remote_host, remote_handshake_port)

        if tp_num_need_pulls == 1:
            grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                remote_block_ids, local_block_ids
            )
        else:
            remote_block_ids = list(map(lambda x: [x], remote_block_ids))
            local_block_ids = list(map(lambda x: [x], local_block_ids))
            grouped_remote_block_ids, grouped_local_block_ids = remote_block_ids, local_block_ids
        num_transfer_groups = len(grouped_remote_block_ids)
        # tp_num_need_pulls: number of KV caches each Decode node needs to pull from each PP stage
        # Due to GQA, different KV heads are distributed across different ranks, so there are offsets
        # indicating which KV head to pull
        global_offset = offset  # Global offset of request across all ranks
        prefill_pp_rank = offset // tp_num_need_pulls  # PP rank where current request resides
        inner_offset = offset % tp_num_need_pulls  # Offset within each PP stage

        remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][
            remote_handshake_port
        ]
        remote_groups = self.remote_buffer_group_ids[remote_engine_id].get(
            remote_handshake_port, ()
        )
        if remote_groups:
            remote_kv_caches_base_addrs = [
                base
                for base, group in zip(
                    remote_kv_caches_base_addrs, remote_groups, strict=True
                )
                if group == self.ordinary_group_id
            ]
        first_layer_index, end_layer_index = self.pp_layer_indices[prefill_pp_rank]
        # support MTP layer kv transfer
        if self.vllm_config.speculative_config is not None:
            # all MTP layer use the same kv cache layer, so only need to transfer once
            if prefill_pp_rank == self._prefill_pp_size - 1:
                end_layer_index = end_layer_index + 1
        num_cache_per_layer = len(list(self.kv_caches.values())[0])  # Number of KV caches per layer
        local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port][
            first_layer_index * num_cache_per_layer : end_layer_index * num_cache_per_layer
        ]
        logger.debug(f"transfer kv cache first_layer_index:{first_layer_index} , end_layer_index:{end_layer_index}")
        remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
        num_blocks = len(local_block_ids)
        session_id = f"{remote_host}:{remote_transfer_port}"

        req_start_time = time.perf_counter()
        src_list, dst_list, length_list = [], [], []
        block_length = len(self.block_len)
        for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
            zip(local_kv_caches_base_addrs, remote_kv_caches_base_addrs)
        ):
            block_len = self.block_len[k % block_length]
            inner_block_len = block_len // tp_num_need_pulls
            for remote_block_id, local_block_id in zip(grouped_remote_block_ids, grouped_local_block_ids):
                src = src_layer_base_addr + local_block_id[0] * block_len + inner_offset * inner_block_len
                dst = dst_layer_base_addr + remote_block_id[0] * inner_block_len
                length = inner_block_len * len(local_block_id)
                src_list.append(src)
                dst_list.append(dst)
                length_list.append(length)

        ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
        if ret < 0:
            logger.error("Mooncake transfer failed for request %s", req_meta["remote_request_id"])
            raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        req_end_time = time.perf_counter()
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info(
            "KV cache transfer for request %s took %.2f ms (%d groups,"
            " %d blocks). local_ip %s local_device_id %s remote_session_id %s",
            remote_request_id,
            req_transfer_elapsed,
            num_transfer_groups,
            num_blocks,
            get_ip(),
            self.tp_rank,
            session_id,
        )

        # Determine if the current position is the offset position at the end of
        # the KV transmission.
        is_kv_transfer_end = global_offset == tp_num_need_pulls * self._prefill_pp_size - 1
        need_cat_cache = tp_num_need_pulls > 1 and is_kv_transfer_end
        need_nz_cache = get_ascend_config().enable_kv_nz and is_kv_transfer_end
        use_fused_op = ascend_envs.VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
        if need_nz_cache or need_cat_cache:
            # use fused op to reformat kv cache, we keep original implementation to provide ability to disable it.
            if use_fused_op and enable_custom_op():
                if need_cat_cache:
                    # the fused op only support cat GQA/MHA kv cache by head
                    self.reformat_kv_cache_with_fused_op(grouped_local_block_ids, tp_num_need_pulls)
                if need_nz_cache:
                    # maybe use fused op to reformat kv nz too in the future.
                    self.reformat_kv_cache(grouped_local_block_ids, tp_num_need_pulls, False, need_nz_cache)
            else:
                self.reformat_kv_cache(grouped_local_block_ids, tp_num_need_pulls, need_cat_cache, need_nz_cache)

    def _transfer_split_destinations(
        self, req_meta: dict[str, Any], plan: SplitTransferPlan
    ) -> None:
        _cold_live_log(
            "live_source_native_transfer_entry",
            req_id=req_meta.get("request_id"),
            transfer_id=req_meta.get("split_transfer_id"),
            segment_count=len(plan.segments),
            requested_groups=plan.requested_groups,
            group_byte_totals=plan.group_byte_totals,
        )
        if plan.tp_rank != self.tp_rank or plan.dp_rank != self.vllm_config.parallel_config.data_parallel_rank_local:
            raise RuntimeError("Split destination TP/DP rank mismatch")
        requested_groups = set(plan.requested_groups)
        if not requested_groups or not requested_groups.issubset({0, 1}):
            raise RuntimeError("Split destination requested groups are invalid")
        totals = [0, 0]
        destinations: dict[str, list[tuple[int, int]]] = {
            "cpu": [],
            "npu": [],
        }
        for segment in plan.segments:
            if segment.group_id not in requested_groups:
                raise RuntimeError("Split segment targets an unrequested group")
            expected_kind = "cpu" if segment.group_id == 0 else "npu"
            if segment.destination_kind != expected_kind:
                raise RuntimeError(
                    f"Split group {segment.group_id} requires {expected_kind} destinations"
                )
            if segment.length <= 0 or segment.source_offset < 0 or segment.destination_address <= 0:
                raise RuntimeError("Invalid split destination extent")
            totals[segment.group_id] += segment.length
            destinations[segment.destination_kind].append(
                (
                    segment.destination_address,
                    segment.destination_address + segment.length,
                )
            )
        for ranges in destinations.values():
            ranges.sort()
            if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
                raise RuntimeError("Split destination extents overlap")
        if tuple(totals) != plan.group_byte_totals:
            raise RuntimeError(
                f"Split destination byte totals mismatch: {tuple(totals)} != {plan.group_byte_totals}"
            )
        for group_id, total in enumerate(totals):
            if group_id in requested_groups and total <= 0:
                raise RuntimeError("Every requested split group requires bytes")
            if group_id not in requested_groups and total != 0:
                raise RuntimeError("Unrequested split groups must have zero bytes")

        remote_host = req_meta["remote_host"]
        remote_port = req_meta["remote_handshake_port"]
        remote_engine_id = req_meta["remote_engine_id"]
        if (
            remote_engine_id not in self.kv_caches_base_addr
            or remote_port not in self.kv_caches_base_addr[remote_engine_id]
        ):
            self._get_remote_metadata(remote_host, remote_port)
        capabilities = getattr(self, "remote_capabilities", {}).get(
            (remote_engine_id, remote_port), ())
        if LIVE_SPLIT_CAPABILITY not in capabilities:
            raise RuntimeError("Remote Mooncake peer lacks live split capability")
        remote_bases = self.kv_caches_base_addr[remote_engine_id][remote_port]
        remote_sizes = self.remote_buffer_sizes[remote_engine_id][remote_port]
        remote_groups = self.remote_buffer_group_ids[remote_engine_id][remote_port]
        if len(remote_sizes) != len(remote_bases):
            raise RuntimeError(
                "Remote peer omitted registered source buffer sizes"
            )
        if len(remote_groups) != len(remote_bases):
            raise RuntimeError(
                "Remote peer omitted registered source buffer groups"
            )
        for segment in plan.segments:
            if not 0 <= segment.source_buffer_index < len(remote_bases):
                raise RuntimeError("Split source buffer index is out of range")
            if segment.source_buffer_base != remote_bases[
                segment.source_buffer_index
            ]:
                raise RuntimeError("Split source buffer identity mismatch")
            if remote_groups[segment.source_buffer_index] != segment.group_id:
                raise RuntimeError("Split source buffer group mismatch")
            source_size = remote_sizes[segment.source_buffer_index]
            if segment.source_offset + segment.length > source_size:
                raise RuntimeError("Split source extent exceeds registered KV buffer")
        session_id = f"{remote_host}:{self.remote_te_port[remote_engine_id][remote_port]}"
        local_destinations = [segment.destination_address for segment in plan.segments]
        remote_sources = [
            remote_bases[segment.source_buffer_index] + segment.source_offset
            for segment in plan.segments
        ]
        lengths = [segment.length for segment in plan.segments]
        with global_te.temporary_registration(local_destinations, lengths):
            ret = self.engine.batch_transfer_sync_read(
                session_id, local_destinations, remote_sources, lengths)
        if ret < 0:
            raise RuntimeError(f"Mooncake split transfer failed, ret: {ret}")

    def reformat_kv_cache_with_fused_op(self, block_ids: list[list[int]], tp_num_need_pulls: int):
        # Get necessary parameters
        k_cache = list(self.kv_caches.values())[0][0]
        device = k_cache.device
        head_dim = self.model_config.hf_text_config.head_dim
        block_size = self.vllm_config.cache_config.block_size
        num_kv_head = max(self.model_config.hf_text_config.num_key_value_heads // self.tp_size, 1)
        layers = self.model_config.hf_text_config.num_hidden_layers
        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int64, device=device)

        k_caches = []
        v_caches = []
        for _, (k_cache_layer, v_cache_layer) in self.kv_caches.items():
            k_caches.append(k_cache_layer)
            v_caches.append(v_cache_layer)

        torch.ops._C_ascend.transpose_kv_cache_by_block(
            k_caches, v_caches, block_ids_tensor, block_size, num_kv_head, head_dim, tp_num_need_pulls, layers
        )

    def reformat_kv_cache(
        self,
        block_ids: list[list[int]],
        tp_num_need_pulls: int,
        need_cat_cache: bool = False,
        need_nz_cache: bool = False,
    ):
        # Get necessary parameters
        k_cache = list(self.kv_caches.values())[0][0]
        dtype = k_cache.dtype
        device = k_cache.device

        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
        num_blocks = len(flat_block_ids)
        num_tokens = num_blocks * self.block_size

        # Create device tensors for copy operations
        block_table = block_ids_tensor.view(1, -1)
        block_len_tensor = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        seq_start_tensor = torch.tensor([0], dtype=torch.int32, device=device)

        # Initialize buffers
        k_buffer = torch.empty((num_tokens, self.num_kv_heads, self.k_head_dim), dtype=dtype, device=device)
        v_buffer = torch.empty((num_tokens, self.num_kv_heads, self.v_head_dim), dtype=dtype, device=device)

        # Create slot mapping for reshape operations
        block_offsets = torch.arange(0, self.block_size, dtype=torch.int32, device=device)
        slot_mapping = (
            block_offsets.reshape((1, self.block_size)) + block_ids_tensor.reshape((num_blocks, 1)) * self.block_size
        ).flatten()

        # FIXME: Right now, if we skip synchronization at this point, the system
        # will crash in GQA scenarios. However, we still haven't identified the
        # root cause.
        torch.npu.synchronize()

        # Process each layer in the KV cache
        for _, (k_cache_layer, v_cache_layer) in self.kv_caches.items():
            # Load cache data into buffers
            torch_npu.atb.npu_paged_cache_load(
                k_cache_layer,
                v_cache_layer,
                block_table,
                block_len_tensor,
                seq_starts=seq_start_tensor,
                key=k_buffer,
                value=v_buffer,
            )
            if need_cat_cache:
                self._cat_kv_cache(
                    k_cache_layer,
                    v_cache_layer,
                    k_buffer,
                    v_buffer,
                    tp_num_need_pulls,
                    num_blocks,
                    num_tokens,
                    slot_mapping,
                )
            if need_nz_cache:
                self._nz_kv_cache(k_cache_layer, v_cache_layer, k_buffer, v_buffer, slot_mapping)
        # Clean up buffers
        del k_buffer, v_buffer

    def _cat_kv_cache(
        self, k_cache_layer, v_cache_layer, k_buffer, v_buffer, tp_num_need_pulls, num_blocks, num_tokens, slot_mapping
    ):
        def _transpose_kv_cache_between_head(buffer: torch.Tensor) -> torch.Tensor:
            buffer = buffer.view(num_blocks, tp_num_need_pulls, self.block_size, -1)
            buffer.transpose_(1, 2)
            return buffer.contiguous().view(num_tokens, self.num_kv_heads, -1)

        # Transpose KV cache
        k_buffer = _transpose_kv_cache_between_head(k_buffer)
        v_buffer = _transpose_kv_cache_between_head(v_buffer)

        # Reshape and cache the processed buffers
        torch_npu._npu_reshape_and_cache(
            key=k_buffer, value=v_buffer, key_cache=k_cache_layer, value_cache=v_cache_layer, slot_indices=slot_mapping
        )

    def _nz_kv_cache(self, k_cache_layer, v_cache_layer, k_buffer, v_buffer, slot_mapping):
        nz_fmt_last_dim = 16
        k_cache_layer = k_cache_layer.view(
            -1, self.k_head_dim * self.num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        v_cache_layer = v_cache_layer.view(
            -1, self.v_head_dim * self.num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        torch_npu.npu_scatter_pa_kv_cache(k_buffer, v_buffer, k_cache_layer, v_cache_layer, slot_mapping)

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        """Get the metadata from the remote host."""
        sock: zmq.Socket | None = None  # type: ignore
        healthy = False
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local engine id {self.local_engine_id}."
            )
            self.kv_caches_base_addr[engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
            self.remote_te_port[engine_id][remote_handshake_port] = agent_meta.te_rpc_port
            self.remote_num_blocks[engine_id][remote_handshake_port] = agent_meta.num_blocks
            self.remote_buffer_sizes[engine_id][remote_handshake_port] = tuple(
                agent_meta.kv_caches_buffer_sizes
            )
            self.remote_buffer_group_ids[engine_id][remote_handshake_port] = tuple(
                agent_meta.buffer_group_ids
            )
            if not hasattr(self, "remote_capabilities"):
                self.remote_capabilities = {}
            self.remote_capabilities[(engine_id, remote_handshake_port)] = agent_meta.capabilities
            healthy = True
        finally:
            if sock is not None:
                if healthy:
                    self._return_remote_socket(
                        sock, remote_host, remote_handshake_port
                    )
                    logger.debug(
                        "Returned socket to pool for %s:%d",
                        remote_host,
                        remote_handshake_port,
                    )
                else:
                    self._discard_remote_socket(sock)

    def _send_done_recv_signal(
        self,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo],
    ):
        logger.debug(
            "Sending done recving signal for request %s to %s:%d", request_id, remote_host, remote_handshake_port
        )
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id, remote_port_send_num))
            ensure_zmq_send(sock, data_bytes, f"{remote_host}:{remote_handshake_port}")
            resp = ensure_zmq_recv(
                sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}", timeout=self.timeout
            )
            logger.debug(f"Received response for request {request_id}: {resp.decode('utf-8')}")
            if resp != b"ACK":
                logger.error(
                    "Failed to receive ACK for request %s from %s:%d", request_id, remote_host, remote_handshake_port
                )
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except (RuntimeError, zmq.ZMQError) as e:  # type: ignore
            if isinstance(sock, zmq.Socket):  # type: ignore
                self._discard_remote_socket(sock)
                sock = None
                logger.warning(f"Unexpected error occurred in socket, {e}, closing the original channel")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host, remote_handshake_port)

    def _send_split_done_signal(
        self, request_id: str, remote_host: str,
        remote_handshake_port: int, status: str,
        split_transfer_id: str | None,
    ) -> bool:
        while not self.stop_event.is_set():
            for attempt in range(SPLIT_DONE_MAX_ATTEMPTS):
                context = zmq.Context()  # type: ignore
                sock = None
                try:
                    sock = make_zmq_socket(
                        ctx=context,
                        path=make_zmq_path(
                            "tcp", remote_host, remote_handshake_port
                        ),
                        socket_type=zmq.REQ,  # type: ignore
                        bind=False,
                        linger=0,
                    )
                    sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
                    poller = zmq.Poller()  # type: ignore
                    poller.register(sock, zmq.POLLIN)  # type: ignore
                    ensure_zmq_send(
                        sock,
                        self.encoder.encode(
                            (SPLIT_DONE_MSG, request_id, status)
                            if split_transfer_id is None
                            else (
                                SPLIT_DONE_MSG,
                                request_id,
                                status,
                                split_transfer_id,
                            )
                        ),
                        f"{remote_host}:{remote_handshake_port}")
                    response = ensure_zmq_recv(
                        sock, poller,
                        f"{remote_host}:{remote_handshake_port}",
                        timeout=self.timeout,
                    )
                    if response != b"ACK":
                        raise RuntimeError(
                            "Split completion was not acknowledged"
                        )
                    return True
                except (RuntimeError, zmq.ZMQError):  # type: ignore
                    if attempt + 1 < SPLIT_DONE_MAX_ATTEMPTS:
                        time.sleep(0.05 * (attempt + 1))
                finally:
                    if sock is not None:
                        sock.close()
                    context.term()
            logger.warning(
                "Split completion ACK failed for request %s; retrying",
                request_id,
            )
            self.stop_event.wait(0.25)
        return False

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False,
            )
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def _return_remote_socket(
        self,
        sock: zmq.Socket,  # type: ignore
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Return the remote socket to the pool."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)

    def _discard_remote_socket(self, sock: zmq.Socket) -> None:  # type: ignore
        with contextlib.suppress(KeyError):
            self.remote_poller.unregister(sock)
        context = sock.context
        sock.close(linger=0)
        context.term()


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self, live_split_topology_supported: bool = True):
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()
        self.split_requests_to_send: set[str] = set()
        self.split_transfer_ids: dict[str, str] = {}
        self.live_split_topology_supported = live_split_topology_supported

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
    ):
        split_plan_data = kv_transfer_params.get(LIVE_SPLIT_CAPABILITY)
        split_plan = self._parse_split_plan(split_plan_data)
        split_source_invalid = False
        try:
            split_source = self._parse_source_descriptor(
                kv_transfer_params.get(LIVE_SPLIT_SOURCE_DESCRIPTOR)
            )
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Invalid prefiller live-split source descriptor for request "
                "%s; using persistent fallback",
                request_id,
            )
            split_source = None
            split_source_invalid = True
        capabilities = kv_transfer_params.get("live_split_capabilities", ())
        split_negotiated = (
            LIVE_SPLIT_CAPABILITY in capabilities or split_plan is not None
        )
        split_transfer_id = kv_transfer_params.get("live_split_transfer_id")
        _cold_live_log(
            "live_source_decoder_ingest",
            req_id=request_id,
            source_present=LIVE_SPLIT_SOURCE_DESCRIPTOR in kv_transfer_params,
            source_valid=split_source is not None,
            split_negotiated=split_negotiated,
            transfer_id_present=isinstance(split_transfer_id, str),
            transfer_param_keys=sorted(kv_transfer_params),
        )
        if split_negotiated and not self.live_split_topology_supported:
            split_plan = None
            split_source_invalid = True
        if split_negotiated and (
            not isinstance(split_transfer_id, str)
            or not split_transfer_id
            or len(split_transfer_id) > MAX_SPLIT_TRANSFER_ID_LENGTH
        ):
            split_source_invalid = True
            split_transfer_id = None
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            num_external_tokens=num_external_tokens,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_request_id=kv_transfer_params["remote_request_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size", 1),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size", 1),
            remote_ptp_size=kv_transfer_params.get("remote_ptp_size"),
            remote_multi_nodes_meta_mapping=kv_transfer_params.get("remote_multi_nodes_meta_mapping", {}),
            num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
            split_plan=split_plan,
            split_negotiated=split_negotiated,
            split_source=split_source,
            split_source_invalid=split_source_invalid,
            split_fallback=split_negotiated and split_transfer_id is None,
            split_transfer_id=split_transfer_id,
        )

    @staticmethod
    def _parse_source_descriptor(
        descriptor: Any,
    ) -> tuple[SplitSourceDescriptor, ...] | None:
        if descriptor is None:
            return None
        if isinstance(descriptor, SplitSourceDescriptor):
            descriptors = (descriptor,)
        elif not isinstance(descriptor, dict):
            raise ValueError("Live split source descriptor must be a mapping")
        else:
            raw_descriptors = descriptor.get("descriptors", (descriptor,))
            descriptors = tuple(
                SplitSourceDescriptor(
                    segments=tuple(
                        SplitSourceSegment(**segment)
                        for segment in raw["segments"]
                    ),
                    group_byte_totals=tuple(raw["group_byte_totals"]),
                    tp_rank=int(raw["tp_rank"]),
                    dp_rank=int(raw["dp_rank"]),
                )
                for raw in raw_descriptors
            )
        identities: set[tuple[int, int]] = set()
        for parsed in descriptors:
            if len(parsed.group_byte_totals) != 2:
                raise ValueError("Live split source totals require two groups")
            if (parsed.tp_rank, parsed.dp_rank) in identities:
                raise ValueError("Duplicate live split source rank")
            identities.add((parsed.tp_rank, parsed.dp_rank))
            totals = [0, 0]
            for segment in parsed.segments:
                if (
                    segment.group_id not in (0, 1)
                    or segment.source_buffer_index < 0
                    or segment.source_buffer_base <= 0
                    or segment.source_offset < 0
                    or segment.length <= 0
                ):
                    raise ValueError("Invalid live split source extent")
                totals[segment.group_id] += segment.length
            if tuple(totals) != parsed.group_byte_totals:
                raise ValueError("Live split source byte totals mismatch")
        return descriptors

    @classmethod
    def _merge_source_and_destinations(
        cls,
        source: SplitSourceDescriptor,
        plan: Any,
        supported_groups: tuple[int, ...] = (0, 1),
    ) -> SplitTransferPlan:
        if not isinstance(plan, dict):
            raise ValueError("Live split destination plan must be a mapping")
        dest_totals = tuple(plan["group_byte_totals"])
        requested_groups = tuple(
            group for group in plan.get("requested_groups", (0, 1))
            if group in supported_groups
        )
        if (
            len(dest_totals) != 2
            or not requested_groups
            or not set(requested_groups).issubset({0, 1})
        ):
            raise ValueError(
                "invalid destination totals/groups: "
                f"totals={dest_totals}, groups={requested_groups}"
            )
        mismatched_totals = {
            group: (source.group_byte_totals[group], dest_totals[group])
            for group in requested_groups
            if dest_totals[group] != source.group_byte_totals[group]
        }
        if mismatched_totals:
            raise ValueError(
                "source/destination byte totals differ "
                f"(source, destination)={mismatched_totals}"
            )
        tp_rank, dp_rank = int(plan["tp_rank"]), int(plan["dp_rank"])
        if (tp_rank, dp_rank) != (source.tp_rank, source.dp_rank):
            raise ValueError(
                "source/destination ranks differ: "
                f"source={(source.tp_rank, source.dp_rank)}, "
                f"destination={(tp_rank, dp_rank)}"
            )
        destinations = plan["segments"]
        merged: list[SplitTransferSegment] = []
        for group_id in requested_groups:
            sources = [s for s in source.segments if s.group_id == group_id]
            dests = [d for d in destinations if int(d["group_id"]) == group_id]
            source_pos = dest_pos = source_offset = dest_offset = 0
            while source_pos < len(sources) and dest_pos < len(dests):
                src, dst = sources[source_pos], dests[dest_pos]
                dst_length = int(dst["length"])
                length = min(src.length - source_offset,
                             dst_length - dest_offset)
                if length <= 0:
                    raise ValueError("Invalid live split destination extent")
                merged.append(SplitTransferSegment(
                    group_id=group_id,
                    source_buffer_index=src.source_buffer_index,
                    source_offset=src.source_offset + source_offset,
                    destination_address=int(dst["destination_address"])
                    + dest_offset,
                    length=length,
                    destination_kind=str(dst["destination_kind"]),
                    source_buffer_base=src.source_buffer_base,
                ))
                source_offset += length
                dest_offset += length
                if source_offset == src.length:
                    source_pos += 1
                    source_offset = 0
                if dest_offset == dst_length:
                    dest_pos += 1
                    dest_offset = 0
            if source_pos != len(sources) or dest_pos != len(dests):
                raise ValueError(
                    "source/destination extents do not cover the same bytes: "
                    f"group={group_id}, source_extents={len(sources)}, "
                    f"destination_extents={len(dests)}"
                )
        return SplitTransferPlan(
            segments=tuple(merged),
            group_byte_totals=tuple(
                source.group_byte_totals[group]
                if group in requested_groups else 0
                for group in range(2)
            ),
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            requested_groups=requested_groups,
        )

    @staticmethod
    def _parse_split_plan(plan: Any) -> SplitTransferPlan | None:
        if plan is None:
            return plan
        if isinstance(plan, SplitTransferPlan):
            parsed = plan
        elif not isinstance(plan, dict):
            raise ValueError("Live split destination plan must be a mapping")
        else:
            segments = tuple(
                SplitTransferSegment(**segment) for segment in plan["segments"]
            )
            parsed = SplitTransferPlan(
                segments=segments,
                group_byte_totals=tuple(plan["group_byte_totals"]),
                tp_rank=int(plan["tp_rank"]),
                dp_rank=int(plan["dp_rank"]),
                requested_groups=tuple(plan.get("requested_groups", (0, 1))),
            )
        if len(parsed.group_byte_totals) != 2:
            raise ValueError("Live split byte totals require exactly two groups")
        return parsed

    def needs_late_split_plans(self) -> bool:
        return any(
            meta.split_negotiated and meta.split_plan is None
            for meta in self.requests.values()
        )

    def accept_late_split_plans(
        self, plans: dict[str, Any], supported_groups: tuple[int, ...] = (0, 1)
    ) -> None:
        for request_id, meta in self.requests.items():
            if not meta.split_negotiated or meta.split_plan is not None:
                continue
            if meta.split_source_invalid:
                meta.split_fallback = True
                continue
            _cold_live_log(
                "live_source_late_plan_entry",
                req_id=request_id,
                source_present=meta.split_source is not None,
                destination_present=request_id in plans,
                supported_groups=supported_groups,
            )
            try:
                raw_plan = plans.get(request_id)
                if isinstance(raw_plan, dict) and meta.split_source is None:
                    raw_segments = raw_plan.get("segments", ())
                    if (
                        not raw_segments
                        or "source_buffer_index" not in raw_segments[0]
                    ):
                        raise ValueError(
                            "prefiller source descriptor is missing; "
                            "received only a decoder destination plan"
                        )
                source = None
                if meta.split_source is not None and raw_plan is not None:
                    identity = (int(raw_plan["tp_rank"]), int(raw_plan["dp_rank"]))
                    source = next(
                        (
                            item
                            for item in meta.split_source
                            if (item.tp_rank, item.dp_rank) == identity
                        ),
                        None,
                    )
                    if source is None:
                        raise ValueError("Missing live split source rank")
                plan = (
                    self._merge_source_and_destinations(
                        source, raw_plan, supported_groups
                    )
                    if source is not None
                    else self._parse_split_plan(raw_plan)
                )
                if plan is not None:
                    requested_groups = tuple(
                        group for group in plan.requested_groups
                        if group in supported_groups
                    )
                    if not requested_groups:
                        raise ValueError("No requested live-split group is supported")
                    segments = tuple(
                        segment
                        for segment in plan.segments
                        if segment.group_id in requested_groups
                    )
                    totals = tuple(
                        plan.group_byte_totals[group_id]
                        if group_id in requested_groups
                        else 0
                        for group_id in range(2)
                    )
                    plan = SplitTransferPlan(
                        segments=segments,
                        group_byte_totals=totals,
                        tp_rank=plan.tp_rank,
                        dp_rank=plan.dp_rank,
                        requested_groups=requested_groups,
                    )
                meta.split_plan = plan
            except (KeyError, TypeError, ValueError) as error:
                logger.warning(
                    "Invalid late live-split destination plan for request %s; "
                    "using persistent fallback: %s",
                    request_id,
                    error,
                )
            meta.split_fallback = meta.split_plan is None


class MooncakeConnector(KVConnectorBase_V1):
    # Live split registers the unbundled latent and index buffers and routes
    # them by explicit group descriptors. Ordinary transfers use this
    # connector's selected group (latent by default, index for index-only).
    requires_full_dsa_kv_caches = True
    releases_live_transfer_destinations_on_shutdown = True

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = MooncakeConnectorScheduler(
                vllm_config, str(self.engine_id), self._live_split_source_groups()
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, str(self.engine_id))

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        self.connector_worker.cancel_live_split(finished_req_ids)
        return self.connector_worker.get_finished()

    def get_live_split_results(self) -> dict[str, str]:
        assert self.connector_worker is not None
        return self.connector_worker.get_live_split_results()

    def shutdown(self) -> None:
        if self.connector_worker is not None:
            self.connector_worker.shutdown()

    def _needs_live_split_destination_plans(self) -> bool:
        metadata = self._connector_metadata
        return (
            isinstance(metadata, MooncakeConnectorMetadata)
            and metadata.needs_late_split_plans()
        )

    def _accept_live_split_destination_plans(
        self, plans: dict[str, Any]
    ) -> None:
        metadata = self._connector_metadata
        if isinstance(metadata, MooncakeConnectorMetadata):
            metadata.accept_late_split_plans(
                plans, self._live_split_source_groups())

    def _live_split_source_groups(self) -> tuple[int, ...]:
        index_group_id = getattr(self, "index_group_id", None)
        if index_group_id is not None:
            return (int(index_group_id),)
        return (0, 1)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """MooncakeConnector does not save explicitly."""
        pass

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata

    def set_xfer_handshake_metadata(self, metadata: dict[int, KVConnectorHandshakeMetadata]) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        live_split_source_groups: tuple[int, ...] = (0, 1),
    ):
        self.vllm_config = vllm_config
        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.live_split_source_groups = live_split_source_groups
        self.local_ip = get_ip()
        logger.info("Initializing Mooncake Scheduler %s", engine_id)

        self.side_channel_host = get_ip()
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        prefill = vllm_config.kv_transfer_config.get_from_extra_config(
            "prefill", {}
        )
        decode = vllm_config.kv_transfer_config.get_from_extra_config(
            "decode", {}
        )
        self._live_prefill_tp_size = int(prefill.get("tp_size", self.tp_size))
        self._live_decode_tp_size = int(decode.get("tp_size", self.tp_size))
        self._live_prefill_dp_size = int(prefill.get("dp_size", 1))
        self._live_decode_dp_size = int(decode.get("dp_size", 1))
        self.max_device_id = (
            vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.data_parallel_size
            * self.pcp_size
            * vllm_config.parallel_config.pipeline_parallel_size
        )

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int], int]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()
        self._split_reqs_need_send: set[str] = set()
        self.split_transfer_ids: dict[str, str] = {}

        # master-slave meta information for cross-nodes
        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        self.local_source_metadata: dict[
            tuple[int, int], MooncakeAgentMetadata
        ] = {}

    def _live_split_topology_supported(self) -> bool:
        return (
            self.pp_size == 1
            and self.pcp_size == 1
            and self.dcp_size == 1
            and self._live_prefill_tp_size == self._live_decode_tp_size
            and self._live_prefill_dp_size == self._live_decode_dp_size == 1
        )

    def _canonicalize_source_descriptor(self, descriptor: Any) -> dict[str, Any]:
        if not isinstance(descriptor, dict):
            raise ValueError("Live split source descriptor must be a mapping")
        raw_descriptors = descriptor.get("descriptors", (descriptor,))
        normalized = []
        for raw in raw_descriptors:
            identity = (int(raw["tp_rank"]), int(raw["dp_rank"]))
            metadata = self.local_source_metadata.get(identity)
            if metadata is None:
                raise ValueError("Live split source rank has no local handshake")
            bases = metadata.kv_caches_base_addr
            sizes = metadata.kv_caches_buffer_sizes
            groups = metadata.buffer_group_ids
            if not (len(bases) == len(sizes) == len(groups)):
                raise ValueError("Live split source handshake is incomplete")
            indices: dict[int, list[int]] = defaultdict(list)
            for index, base in enumerate(bases):
                indices[base].append(index)
            segments = []
            for segment in raw["segments"]:
                segment = dict(segment)
                group_id = int(segment["group_id"])
                if group_id not in self.live_split_source_groups:
                    continue
                base = int(segment["source_buffer_base"])
                offset = int(segment["source_offset"])
                length = int(segment["length"])
                if base <= 0 or offset < 0 or length <= 0:
                    raise ValueError("Invalid live split source extent")
                base_indices = indices.get(base)
                if not base_indices:
                    raise ValueError("Live split source base is not registered")
                group_matched = False
                index = None
                for candidate in base_indices:
                    if group_id != groups[candidate]:
                        continue
                    group_matched = True
                    if offset + length <= sizes[candidate]:
                        index = candidate
                        break
                if not group_matched:
                    raise ValueError("Live split source group does not match buffer")
                if index is None:
                    raise ValueError("Live split source extent exceeds buffer")
                segment["group_id"] = group_id
                segment["source_buffer_base"] = base
                segment["source_offset"] = offset
                segment["length"] = length
                segment["source_buffer_index"] = index
                segments.append(segment)
            if not segments:
                raise ValueError("Live split source has no supported groups")
            totals = [0, 0]
            for segment in segments:
                totals[int(segment["group_id"])] += int(segment["length"])
            normalized.append(
                {**raw, "segments": segments, "group_byte_totals": totals}
            )
        return {"descriptors": normalized}

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert num_computed_tokens % self.block_size == 0
            # Note: We use the full token count as transmit data here.
            count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
            return count, count > 0

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)
        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port", "remote_request_id")):
                    local_block_ids = blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (request, local_block_ids, num_external_tokens)
                else:
                    logger.warning("Got invalid KVTransferParams: %s. This request will not utilize KVTransfer", params)
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata(
            self._live_split_topology_supported()
        )

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids, num_external_tokens) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            # For the case where there are no remote blocks to pull
            # (block_ids is empty), we don't need to schedule
            # an async read on the worker side.
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        meta.requests_to_send = self._reqs_need_send
        self._reqs_need_send = {}
        meta.split_requests_to_send = self._split_reqs_need_send
        self._split_reqs_need_send = set()
        meta.split_transfer_ids = {
            request_id: self.split_transfer_ids.pop(request_id)
            for request_id in meta.split_requests_to_send
            if request_id in self.split_transfer_ids
        }
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, request_status=%s, kv_transfer_params=%s", request.status, params
        )

        if (
            params is None
            or not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        computed_block_ids = block_ids
        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s", len(computed_block_ids), request.request_id)
            self._reqs_need_send[request.request_id] = time.time()

        num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)

        transfer_params = dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_pcp_size=self.pcp_size,
            remote_dcp_size=self.dcp_size,
            remote_ptp_size=self.tp_size,
            last_token_id=request.output_token_ids[-1],
            remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
            num_prompt_blocks=num_prompt_blocks,
        )
        if (
            params.get("request_live_split", False)
            and delay_free_blocks
            and self._live_split_topology_supported()
        ):
            split_transfer_id = uuid.uuid4().hex
            self.split_transfer_ids[request.request_id] = split_transfer_id
            self._split_reqs_need_send.add(request.request_id)
            transfer_params["live_split_capabilities"] = (LIVE_SPLIT_CAPABILITY,)
            transfer_params["live_split_transfer_id"] = split_transfer_id
            source_descriptor = params.get(LIVE_SPLIT_SOURCE_DESCRIPTOR)
            _cold_live_log(
                "live_source_mooncake_input",
                req_id=request.request_id,
                request_live_split=True,
                source_present=source_descriptor is not None,
                topology_supported=True,
                request_param_keys=sorted(params),
            )
            if source_descriptor is not None:
                # This descriptor is created by the prefiller-side compact
                # provider, which owns the registered source layout.  The
                # decoder must not synthesize source offsets from its pool.
                try:
                    transfer_params[LIVE_SPLIT_SOURCE_DESCRIPTOR] = (
                        self._canonicalize_source_descriptor(source_descriptor)
                    )
                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "Invalid prefiller source registration for %s; "
                        "using persistent fallback",
                        request.request_id,
                        exc_info=True,
                    )
            _cold_live_log(
                "live_source_mooncake_emit",
                req_id=request.request_id,
                source_present=(
                    LIVE_SPLIT_SOURCE_DESCRIPTOR in transfer_params
                ),
                transfer_param_keys=sorted(transfer_params),
            )
        return delay_free_blocks, transfer_params

    def set_xfer_handshake_metadata(self, metadata: dict[int, KVConnectorHandshakeMetadata]) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        for local_rank, rank_metadata in metadata.items():
            self.multi_nodes_meta_mapping[str(local_rank)] = {
                "host": rank_metadata.local_ip,
                "engine_id": rank_metadata.engine_id,
            }
            if isinstance(rank_metadata, MooncakeAgentMetadata):
                self.local_source_metadata[
                    (rank_metadata.tp_rank, rank_metadata.dp_rank)
                ] = rank_metadata


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self._get_prefill_decode_size(vllm_config)
        self._validate_local_parallel_config(vllm_config)
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        if self._prefill_tp_size < self._decode_tp_size:
            raise ValueError(
                f"prefill_tp_size: {self._prefill_tp_size} must be greater than"
                f" or equal to the decode_tp_size: {self._decode_tp_size}"
            )

        # Metadata.
        self.vllm_config = vllm_config
        self.ascend_config = get_ascend_config()
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.pp_rank = get_pp_group().rank_in_group
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.pcp_size = get_pcp_group().world_size
        # Assert that pp_size and pcp_size cannot both be greater than 1
        assert not (self.pp_size > 1 and self.pcp_size > 1), "pp and pcp cannot open in same time"
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.max_device_id = self.tp_size * self.dp_size * self.pcp_size * self.pp_size
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_key_value_heads = self.vllm_config.model_config.hf_text_config.num_key_value_heads

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        device_index = (self.pp_rank + self.pcp_rank) * self.tp_size + self.tp_rank
        self.handshake_port = self.side_channel_port + device_index
        self.sockets: dict = {}
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_send_thread: KVCacheSendingThread | None = None
        self.kv_recv_thread: KVCacheRecvingThread | None = None

        # Handshake metadata of this worker
        self.xfer_handshake_metadata: MooncakeAgentMetadata | None = None

        # kv_transfer variables
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        if self.vllm_config.model_config.is_deepseek_mla:
            self.tp_num_need_pulls = 1
        else:
            num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
            num_p_block_heads = max(1, self.num_key_value_heads // self._prefill_tp_size)
            self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads
        self.local_remote_block_port_mapping: dict[str, list[list[int]] | None] = {}
        self.remote_port_send_num: dict[str, dict[int, RemotePortInfo]] = {}

    def get_live_split_results(self) -> dict[str, str]:
        thread = self.kv_send_thread if self.kv_role == "kv_producer" else self.kv_recv_thread
        if thread is None:
            return {}
        return thread.task_tracker.get_and_clear_split_results()

    def cancel_live_split(self, request_ids: set[str]) -> None:
        if self.kv_role != "kv_consumer" or self.kv_recv_thread is None:
            return
        for request_id in request_ids:
            self.kv_recv_thread.cancel_split_request(request_id)

    def shutdown(self) -> None:
        """Fence transfer threads before shared registrations are released."""
        if self.kv_send_thread is not None:
            tracker = self.kv_send_thread.task_tracker
            with tracker.done_task_lock:
                if tracker.split_leases:
                    raise RuntimeError(
                        "Mooncake live source ownership remains active"
                    )
        threads = tuple(
            thread
            for thread in (self.kv_send_thread, self.kv_recv_thread)
            if thread is not None
        )
        for thread in threads:
            thread.stop()
        for thread in threads:
            thread.join(THREAD_SHUTDOWN_TIMEOUT)
            if thread.is_alive():
                if isinstance(thread, KVCacheSendingThread):
                    thread.stop_event.clear()
                raise RuntimeError(
                    f"{thread.name} did not stop before Mooncake shutdown"
                )
        if self.kv_recv_thread is not None:
            with self.kv_recv_thread.split_request_lock:
                if self.kv_recv_thread.active_split_requests:
                    raise RuntimeError(
                        "Mooncake split transfer ownership remains active"
                    )
                if self.kv_recv_thread.pending_split_signals:
                    raise RuntimeError(
                        "Mooncake split completion signals remain active"
                    )
                if self.kv_recv_thread.undelivered_split_signals:
                    raise RuntimeError(
                        "Mooncake split completion signals remain unacknowledged"
                    )
        if self.kv_recv_thread is not None:
            self.kv_recv_thread.close_resources()

    def _get_prefill_decode_size(self, vllm_config: VllmConfig):
        # get prefill tp and dp size from extra config
        prefill_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("prefill", {})

        assert "tp_size" in prefill_parallel_config
        self._prefill_tp_size = prefill_parallel_config["tp_size"]

        assert "dp_size" in prefill_parallel_config
        self._prefill_dp_size = prefill_parallel_config["dp_size"]
        # get prefill pp size from extra config
        self._prefill_pp_size = prefill_parallel_config.get("pp_size", 1)
        # get decode tp and dp size from extra config
        decode_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("decode", {})
        assert "tp_size" in decode_parallel_config
        self._decode_tp_size = decode_parallel_config["tp_size"]
        assert "dp_size" in decode_parallel_config
        self._decode_dp_size = decode_parallel_config["dp_size"]
        # get prefill pp size from extra config
        self._decode_pp_size = decode_parallel_config.get("pp_size", 1)
        assert self._decode_pp_size == 1, "decode pp size must be 1"
        self._prefill_pp_layer_partition = prefill_parallel_config.get("pp_layer_partition")

    def _validate_local_parallel_config(self, vllm_config: VllmConfig) -> None:
        actual_tp_size = vllm_config.parallel_config.tensor_parallel_size
        actual_pp_size = vllm_config.parallel_config.pipeline_parallel_size
        kv_role = vllm_config.kv_transfer_config.kv_role

        sides: list[tuple[str, int, int]] = []
        if kv_role in ("kv_producer", "kv_both"):
            sides.append(("prefill", self._prefill_tp_size, self._prefill_pp_size))
        if kv_role in ("kv_consumer", "kv_both"):
            sides.append(("decode", self._decode_tp_size, self._decode_pp_size))

        for side, configured_tp_size, configured_pp_size in sides:
            if configured_tp_size != actual_tp_size:
                raise ValueError(
                    "MooncakeConnector kv_connector_extra_config."
                    f"{side}.tp_size ({configured_tp_size}) must match the "
                    f"actual --tensor-parallel-size ({actual_tp_size}) for "
                    f"kv_role={kv_role}. Update either --tensor-parallel-size "
                    f"or kv_connector_extra_config.{side}.tp_size."
                )
            if configured_pp_size != actual_pp_size:
                raise ValueError(
                    "MooncakeConnector kv_connector_extra_config."
                    f"{side}.pp_size ({configured_pp_size}) must match the "
                    f"actual --pipeline-parallel-size ({actual_pp_size}) for "
                    f"kv_role={kv_role}. Update either --pipeline-parallel-size "
                    f"or kv_connector_extra_config.{side}.pp_size."
                )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""

        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        # TODO(tms): Find a more robust way to detect and handle MLA
        self.use_mla = (
            len(first_kv_cache_tuple) == 2
            and first_kv_cache_tuple[0].size(-1) != first_kv_cache_tuple[1].size(-1)
        )
        self.use_sparse = len(first_kv_cache_tuple) >= 3 or any(
            "indexer" in name for name in kv_caches
        )

        self.num_blocks = first_kv_cache.shape[0]
        logger.info("num_blocks: %s", self.num_blocks)
        self.block_len = []
        if self.use_mla or self.use_sparse:
            block_rank = 3  # [block_size, latent_dim]
            for cache in first_kv_cache_tuple:
                block_shape = cache.shape[-block_rank:]
                logger.info("block_shape: %s", block_shape)
                self.block_len.append(
                    cache.element_size() * math.prod(block_shape)
                )
        else:
            # eager:[num_block, block_size, num_head, hidden_dim]
            block_rank = (
                len(first_kv_cache.shape) - 1
            )  # [block_size, kv_heads, head_dim] or [block_size, kv_heads*head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            logger.info("block_shape: %s", block_shape)
            self.block_len = [first_kv_cache.element_size() * math.prod(block_shape)]

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        self.kv_caches = kv_caches
        kv_caches_base_addr = []
        ordinary_kv_caches_base_addr = []
        storage_regions: dict[int, int] = {}
        lengths = []
        buffer_group_ids = []
        configured_group = self.vllm_config.kv_transfer_config.get_from_extra_config(
            "index_group_id", None
        )
        ordinary_group_id = (
            int(configured_group)
            if configured_group is not None
            else 1
            if all("indexer" in name for name in kv_caches)
            else 0
        )
        for layer_name, cache_or_caches in kv_caches.items():
            # Normalize to always be a list of caches
            for i, cache in enumerate(cache_or_caches, 0):
                base_addr = cache.data_ptr()
                region_len = cache.numel() * cache.element_size()
                storage = cache.untyped_storage()
                storage_regions[int(storage.data_ptr())] = int(storage.nbytes())
                kv_caches_base_addr.append(base_addr)
                lengths.append(region_len)
                group_id = (
                    int(configured_group)
                    if configured_group is not None
                    else 1
                    if "indexer" in layer_name
                    or (self.use_sparse and i >= 2)
                    else 0
                )
                buffer_group_ids.append(group_id)
                if group_id == ordinary_group_id:
                    ordinary_kv_caches_base_addr.append(base_addr)
        global_te.register_buffer(
            list(storage_regions), list(storage_regions.values())
        )
        # After KV Caches registered, start the sending or receiving thread.
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            kv_caches_base_addr=kv_caches_base_addr,
            num_blocks=self.num_blocks,
            local_ip=get_ip(),
            capabilities=(LIVE_SPLIT_CAPABILITY,),
            kv_caches_buffer_sizes=tuple(lengths),
            buffer_group_ids=tuple(buffer_group_ids),
            tp_rank=self.tp_rank,
            dp_rank=int(
                getattr(
                    self.vllm_config.parallel_config,
                    "data_parallel_rank_local",
                    0,
                )
                or 0
            ),
        )
        self.xfer_handshake_metadata = metadata

        ready_event = threading.Event()
        if self.kv_role == "kv_producer":
            self.kv_send_thread = KVCacheSendingThread(
                self.vllm_config,
                self.tp_rank,
                self._prefill_tp_size,
                self.engine_id,
                self.side_channel_host,
                self.side_channel_port,
                metadata,
                ready_event,
                self.kv_caches,
                self.pcp_rank,
            )
            self.kv_send_thread.start()
        else:
            self.kv_recv_thread = KVCacheRecvingThread(
                self.tp_rank,
                self.tp_size,
                self._prefill_pp_size,
                self.engine,
                self.engine_id,
                self.handshake_port,
                self.side_channel_port,
                ordinary_kv_caches_base_addr,
                self.block_len,
                ready_event,
                self.vllm_config,
                self.kv_caches,
                self._prefill_pp_layer_partition,
                ordinary_group_id=ordinary_group_id,
            )
            self.kv_recv_thread.start()

        start_wait_time = time.time()
        thread = self.kv_send_thread if self.kv_role == "kv_producer" else self.kv_recv_thread
        assert thread is not None
        while not ready_event.is_set():
            if not thread.is_alive():
                raise RuntimeError("KV Cache sending/receiving thread failed to start.")
            if time.time() - start_wait_time > 5 * 60:
                raise RuntimeError("Timeout waiting for KV Cache thread to be ready.")
            time.sleep(3)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.kv_role == "kv_producer"
            else set()
        )
        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.kv_role == "kv_consumer"
            else set()
        )
        if self.tp_rank == 0:
            logger.debug(
                "Number of completed KV cache send requests: %d, receive requests: %d",
                len(done_sending),
                len(done_recving),
            )
        return done_sending, done_recving

    def _get_kv_split_metadata(
        self,
        req_id: str,
        meta: ReqMeta,
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        """
        In cp/dcp scenario, kv_cache may be split, so we need to pull multiple blocks from multiple remote P node.
        Use this function to calculate remote port and remote block number of each remote P node that we need to pull.
        """
        prefill_tp_size = meta.remote_ptp_size if getattr(meta, "remote_ptp_size", None) else self._prefill_tp_size
        if meta.remote_pcp_size * meta.remote_dcp_size * self.pcp_size * self.dcp_size == 1:
            chosen_rank_list = self._get_remote_rank(req_id, prefill_tp_size)
            remote_handshake_port_list = [[x + meta.remote_port for x in chosen_rank_list]]
            local_block_ids_list, remote_block_ids_list = [meta.local_block_ids], [meta.remote_block_ids]
            return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

        def context_parallel_parameters_check():
            assert (meta.remote_pcp_size * meta.remote_dcp_size) % (self.pcp_size * self.dcp_size) == 0
            if not (self.use_mla or self.use_sparse):
                p_node_heads_per_rank = math.ceil(self.num_key_value_heads / prefill_tp_size)
                d_node_heads_per_rank = math.ceil(self.num_key_value_heads / self.tp_size)
                assert d_node_heads_per_rank % p_node_heads_per_rank == 0

        def get_kv_head_groups(tp_size):
            if self.use_mla or self.use_sparse:
                kv_head_groups = []
                kv_head_ids = [0]
                kv_head_groups.append(tuple(kv_head_ids))
                return kv_head_groups
            if self.num_key_value_heads // tp_size >= 1:
                kv_head_groups = []
                for tp_rank in range(tp_size):
                    kv_head_ids = [
                        head_idx + tp_rank * (self.num_key_value_heads // tp_size)
                        for head_idx in range(self.num_key_value_heads // tp_size)
                    ]
                    kv_head_groups.append(tuple(kv_head_ids))
                return kv_head_groups
            if tp_size // self.num_key_value_heads > 1:
                kv_head_groups = []
                for kv_head_ids_ in range(self.num_key_value_heads):
                    kv_head_groups.append(tuple([kv_head_ids_]))
                return kv_head_groups

        def get_cp_group_meta(tp_size, pcp_size, dcp_size, port_base):
            # key is kv_head_group, value is cp_groups and which cp_groups to select
            cp_group_meta: dict = {}
            kv_head_groups = get_kv_head_groups(tp_size)
            dcp_repeat_num = tp_size // len(kv_head_groups) // dcp_size

            for kv_head_group_idx, kv_head_group in enumerate(kv_head_groups):
                if kv_head_group not in cp_group_meta:
                    cp_group_meta[kv_head_group] = {}
                    cp_group_meta[kv_head_group]["cp_groups"] = []
                    cp_group_meta[kv_head_group]["select_cp_groups_id"] = 0
                kv_head_group_offset = tp_size // len(kv_head_groups) * kv_head_group_idx
                for dcp_repeat_idx in range(dcp_repeat_num):
                    # len(cp_group) == pcp_size * dcp_size
                    cp_group = []
                    dcp_repeat_offset = dcp_size * dcp_repeat_idx
                    for pcp_rank in range(pcp_size):
                        pcp_rank_offset = tp_size * pcp_rank
                        for dcp_rank in range(dcp_size):
                            cp_group.append(
                                dcp_rank + port_base + pcp_rank_offset + dcp_repeat_offset + kv_head_group_offset
                            )
                    cp_group_meta[kv_head_group]["cp_groups"].append(cp_group)

            return cp_group_meta

        def get_local_remote_block_port_mappings():
            context_parallel_parameters_check()
            p_node_cp_group_meta = get_cp_group_meta(
                prefill_tp_size, meta.remote_pcp_size, meta.remote_dcp_size, meta.remote_port
            )
            d_node_cp_group_meta = get_cp_group_meta(self.tp_size, self.pcp_size, self.dcp_size, self.side_channel_port)
            local_remote_block_port_mappings: dict[int, list[list[int]]] = {}
            for d_node_head_key in d_node_cp_group_meta:
                for p_node_head_key in p_node_cp_group_meta:
                    if not set(p_node_head_key).issubset(set(d_node_head_key)):
                        continue
                    d_node_head_group = d_node_cp_group_meta[d_node_head_key]
                    p_node_head_group = p_node_cp_group_meta[p_node_head_key]
                    for d_cp_group in d_node_head_group["cp_groups"]:
                        select_cp_groups_id = p_node_head_group["select_cp_groups_id"]
                        p_cp_groups = p_node_head_group["cp_groups"]
                        p_cp_group = p_cp_groups[select_cp_groups_id]
                        p_node_head_group["select_cp_groups_id"] = (
                            select_cp_groups_id + 1 if select_cp_groups_id + 1 < len(p_cp_groups) else 0
                        )
                        for d_idx, d_port in enumerate(d_cp_group):
                            if d_port not in local_remote_block_port_mappings:
                                local_remote_block_port_mappings[d_port] = []
                            p_port_remote_list = []
                            for p_idx, p_port in enumerate(p_cp_group):
                                if p_idx % len(d_cp_group) == d_idx:
                                    p_port_remote_list.append(p_port)
                            local_remote_block_port_mappings[d_port].append(p_port_remote_list)

            logger.info(
                "p_node_cp_group_meta is:: %s. d_node_cp_group_meta is:: %s. "
                "local_remote_block_port_mappings is:: %s. ",
                p_node_cp_group_meta,
                d_node_cp_group_meta,
                local_remote_block_port_mappings,
            )

            return local_remote_block_port_mappings

        def get_remote_port_send_num(
            local_remote_block_port_mappings: dict[int, list[list[int]]],
        ) -> dict[int, RemotePortInfo]:
            remote_port_send_num: dict[int, RemotePortInfo] = {}
            for port in range(prefill_tp_size * meta.remote_pcp_size):
                remote_host_info = meta.remote_multi_nodes_meta_mapping.get(str(port), None)
                if remote_host_info is None:
                    remote_host = meta.remote_host
                else:
                    remote_host = remote_host_info["host"]
                remote_port_send_num[meta.remote_port + port] = {"num": 0, "host": remote_host}

            for remote_port_head_list in local_remote_block_port_mappings.values():
                for remote_port_list in remote_port_head_list:
                    for remote_port in remote_port_list:
                        remote_port_send_num[remote_port]["num"] += 1
            return remote_port_send_num

        if meta.remote_engine_id not in self.local_remote_block_port_mapping:
            self.local_remote_block_port_mapping[meta.remote_engine_id] = None

        if self.local_remote_block_port_mapping[meta.remote_engine_id] is None:
            local_remote_block_port_mappings = get_local_remote_block_port_mappings()
            self.local_remote_block_port_mapping[meta.remote_engine_id] = local_remote_block_port_mappings[
                self.handshake_port
            ]
            self.remote_port_send_num[meta.remote_engine_id] = get_remote_port_send_num(
                local_remote_block_port_mappings
            )

        local_remote_block_port_mapping = copy.deepcopy(self.local_remote_block_port_mapping[meta.remote_engine_id])

        num_external_blocks = math.ceil(meta.num_external_tokens / self.block_size)

        assert math.ceil(num_external_blocks / (self.pcp_size * self.dcp_size)) == len(meta.local_block_ids), (
            f"num_external_blocks({num_external_blocks}), cp_size({self.pcp_size * self.dcp_size}), "
            f"local_block_ids_len ({len(meta.local_block_ids)})"
        )
        assert meta.num_prompt_blocks >= num_external_blocks, (
            f"meta.num_prompt_blocks({meta.num_prompt_blocks}), num_external_blocks({num_external_blocks})"
        )

        remote_cp_size = meta.remote_pcp_size * meta.remote_dcp_size
        remote_block_nums_all = [meta.num_prompt_blocks // remote_cp_size] * remote_cp_size
        num_remain_blocks = meta.num_prompt_blocks % remote_cp_size
        for i in range(num_remain_blocks):
            remote_block_nums_all[i] += 1
        last_block_location = (num_remain_blocks + remote_cp_size - 1) % remote_cp_size

        # Considering prefix cache, the remote_block_nums_all should be revised
        num_prefix_cached_blocks = meta.num_prompt_blocks - num_external_blocks
        remote_block_nums_all = [num - num_prefix_cached_blocks // remote_cp_size for num in remote_block_nums_all]
        num_remain_blocks = num_prefix_cached_blocks % remote_cp_size
        for i in range(num_remain_blocks):
            remote_block_nums_all[i] -= 1

        # make sure the last block (which may be unfull) of P nodes is put to the last block of D node
        remote_block_nums: list[int] = []
        final_block_idx: int | None = None
        local_cp_rank = self.dcp_rank + self.pcp_rank * self.dcp_size
        local_cp_size = self.dcp_size * self.pcp_size
        for cp_rank, block_num in enumerate(remote_block_nums_all):
            if cp_rank % local_cp_size == local_cp_rank:
                if last_block_location == cp_rank:
                    final_block_idx = len(remote_block_nums)
                remote_block_nums.append(block_num)

        assert local_remote_block_port_mapping is not None
        if final_block_idx is not None:
            final_block_num = remote_block_nums.pop(final_block_idx)
            remote_block_nums.append(final_block_num)
            for mapping in local_remote_block_port_mapping:
                final_block_port = mapping.pop(final_block_idx)
                mapping.append(final_block_port)

        remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = [], [], []
        for idx in range(len(local_remote_block_port_mapping[0])):
            mapping_list = []
            for mapping in local_remote_block_port_mapping:
                mapping_list.append(mapping[idx])
            remote_handshake_port_list.append(mapping_list)

        # the local_block_ids_list and remote_block_ids_list are related with remote_handshake_port_list
        # such as: local_block_ids_list[[1],[2],[5],[6]], remote_block_ids_list[[1],[1],[1],[1]],
        # remote_handshake_port_list[[30000],[30001],[30004],[30005]]
        # D rank will get remote block 1 in port 30004 and save it in local block 5
        local_block_offset = 0
        for remote_kv_id in range(len(remote_handshake_port_list)):
            num_blocks_to_pull = remote_block_nums[remote_kv_id]
            remote_block_ids_list.append(meta.remote_block_ids[:num_blocks_to_pull])
            local_block_ids_list.append(
                meta.local_block_ids[local_block_offset : local_block_offset + num_blocks_to_pull]
            )
            local_block_offset += num_blocks_to_pull

        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        assert tp_num_need_pulls == len(remote_handshake_port_list[0]), (
            f"tp_num_need_pulls: {tp_num_need_pulls}, remote_handshake_port_list: {remote_handshake_port_list}"
        )

        return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        """Start loading KV blocks from remote engine."""
        if metadata.requests:
            _cold_live_log(
                "live_source_worker_load_entry",
                request_ids=sorted(metadata.requests),
                split_requests=[
                    req_id
                    for req_id, meta in metadata.requests.items()
                    if meta.split_negotiated
                ],
            )
        if self.kv_recv_thread is not None:
            for req_id in metadata.reqs_in_batch:
                self.kv_recv_thread.task_tracker.add_req_to_process(
                    req_id,
                    metadata.requests.get(req_id, None).split_transfer_id
                    if req_id in metadata.requests else None,
                )

        for req_id, meta in metadata.requests.items():
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                meta.remote_engine_id,
                len(meta.local_block_ids),
                len(meta.remote_block_ids),
            )

            prefill_tp_size = meta.remote_ptp_size if getattr(meta, "remote_ptp_size", None) else self._prefill_tp_size
            tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
            remote_req_id = meta.remote_request_id

            if meta.split_fallback:
                assert self.kv_recv_thread is not None
                remote_rank = self._get_remote_rank(
                    remote_req_id, prefill_tp_size)[0]
                remote_port = meta.remote_port + remote_rank
                remote_host, _ = self._get_remote_host_info_by_port(
                    meta.remote_port,
                    remote_port,
                    meta.remote_host,
                    meta.remote_engine_id,
                    meta.remote_multi_nodes_meta_mapping,
                )
                self.kv_recv_thread.task_tracker.complete_split_request(
                    req_id, "fallback", mark_finished=False,
                    split_transfer_id=meta.split_transfer_id)
                self.kv_recv_thread._signal_split_completion(
                    req_id,
                    remote_req_id, remote_host, remote_port, "fallback",
                    meta.split_transfer_id)
                continue

            if meta.remote_pcp_size * meta.remote_dcp_size > 1:
                remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = self._get_kv_split_metadata(
                    req_id, meta
                )

                for pcp_dcp_rank in range(len(remote_handshake_port_list)):
                    for i in range(tp_num_need_pulls):
                        assert self.kv_recv_thread is not None
                        remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                            meta.remote_port,
                            remote_handshake_port_list[pcp_dcp_rank][i],
                            meta.remote_host,
                            meta.remote_engine_id,
                            meta.remote_multi_nodes_meta_mapping,
                        )
                        self.kv_recv_thread.add_request(
                            request_id=req_id,
                            remote_request_id=remote_req_id,
                            local_block_ids=local_block_ids_list[pcp_dcp_rank],
                            remote_block_ids=remote_block_ids_list[pcp_dcp_rank],
                            remote_engine_id=remote_engine_id,
                            remote_host=remote_host,
                            remote_handshake_port=remote_handshake_port_list[pcp_dcp_rank][i],
                            offset=i,
                            tp_num_need_pulls=tp_num_need_pulls,
                            remote_port_send_num=self.remote_port_send_num[meta.remote_engine_id],
                            all_task_done=(
                                pcp_dcp_rank == len(remote_handshake_port_list) - 1 and i == tp_num_need_pulls - 1
                            ),
                        )
            else:  # TODO: support prefill context parallel and pipeline parallel open at the same time
                if meta.split_plan is not None:
                    if self._prefill_pp_size != 1:
                        raise RuntimeError(
                            "Live split does not encode prefiller PP rank"
                        )
                    remote_ranks = (meta.split_plan.tp_rank,)
                else:
                    remote_ranks = tuple(
                        self._get_remote_rank(remote_req_id, prefill_tp_size)
                    )
                for i, remote_rank in enumerate(remote_ranks):
                    assert self.kv_recv_thread is not None
                    remote_handshake_port = meta.remote_port + remote_rank
                    remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                        meta.remote_port,
                        remote_handshake_port,
                        meta.remote_host,
                        meta.remote_engine_id,
                        meta.remote_multi_nodes_meta_mapping,
                    )
                    admitted = self.kv_recv_thread.add_request(
                        request_id=req_id,
                        remote_request_id=remote_req_id,
                        local_block_ids=meta.local_block_ids,
                        remote_block_ids=meta.remote_block_ids,
                        remote_engine_id=remote_engine_id,
                        remote_host=remote_host,
                        remote_handshake_port=remote_handshake_port,
                        offset=i,
                        tp_num_need_pulls=tp_num_need_pulls,
                        all_task_done=(i == len(remote_ranks) - 1),
                        split_plan=meta.split_plan,
                        split_transfer_id=meta.split_transfer_id,
                    )
                    if not admitted:
                        self.kv_recv_thread.task_tracker.complete_split_request(
                            req_id, "fallback", mark_finished=False,
                            split_transfer_id=meta.split_transfer_id)
                        self.kv_recv_thread._signal_split_completion(
                            req_id,
                            remote_req_id, remote_host,
                            remote_handshake_port, "fallback",
                            meta.split_transfer_id)
                        break

        for req_id in metadata.reqs_in_batch:
            if self.kv_send_thread is not None:
                self.kv_send_thread.task_tracker.add_req_to_process(
                    req_id, metadata.split_transfer_ids.get(req_id))

        if self.kv_send_thread is not None and self.pcp_size * self.dcp_size == 1:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                if self.tp_rank in self._prefill_get_remote_rank(req_id):
                    self.kv_send_thread.add_delayed_request(
                        req_id, delay_start_time,
                        split=req_id in metadata.split_requests_to_send,
                        split_transfer_id=metadata.split_transfer_ids.get(req_id))
                else:
                    self.kv_send_thread.add_not_transfer_request(req_id)

        if self.kv_send_thread is not None and self.pcp_size * self.dcp_size > 1:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                self.kv_send_thread.add_delayed_request(
                    req_id, delay_start_time,
                    split=req_id in metadata.split_requests_to_send,
                    split_transfer_id=metadata.split_transfer_ids.get(req_id))

    def _get_tp_num_need_pulls(self, prefill_tp_size: int) -> int:
        if prefill_tp_size is None:
            prefill_tp_size = self._prefill_tp_size

        if prefill_tp_size == self._prefill_tp_size:
            return self.tp_num_need_pulls

        if self.vllm_config.model_config.is_deepseek_mla:
            tp_num_need_pulls = 1
        else:
            num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
            num_p_block_heads = max(1, self.num_key_value_heads // prefill_tp_size)
            tp_num_need_pulls = num_d_block_heads // num_p_block_heads
        return tp_num_need_pulls

    def _get_remote_host_info_by_port(
        self,
        base_port: int,
        remote_handshake_port: int,
        remote_host: str,
        remote_engine_id: str,
        remote_multi_nodes_meta_mapping: dict,
    ):
        rank = str(remote_handshake_port - base_port)
        if remote_multi_nodes_meta_mapping is None or remote_multi_nodes_meta_mapping.get(rank) is None:
            return remote_host, remote_engine_id
        info = remote_multi_nodes_meta_mapping[rank]
        return info.get("host", remote_host), info.get("engine_id", remote_engine_id)

    def _prefill_get_remote_rank(self, req_id: str) -> list[int]:
        return sum(self._get_remote_ranks_for_req(req_id), [])

    def _get_remote_rank(self, req_id: str, prefill_tp_size: int | None = None) -> list[int]:
        return self._get_remote_ranks_for_req(req_id, prefill_tp_size)[self.tp_rank]

    def _get_remote_tp_ranks(
        self, tp_ori_data: np.ndarray, rand_group_index: list[int], num_groups: int, prefill_tp_size: int
    ) -> list[list[int]]:
        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        # random split prefill tp list
        tp_sampled_nums = []
        if (
            prefill_tp_size > self.num_key_value_heads
            or self.vllm_config.model_config.is_deepseek_mla
            or self.use_sparse
        ):
            tp_ori_data = tp_ori_data.reshape(-1, num_groups)
            chosen_group = tp_ori_data[:, [rand_group_index]]
            flattened = chosen_group.reshape(-1).tolist()
            tp_sampled_nums = [
                flattened[i : i + tp_num_need_pulls] for i in range(0, len(flattened), tp_num_need_pulls)
            ]
        # non-random split
        else:
            group_size = prefill_tp_size // self._decode_tp_size
            for i in range(self._decode_tp_size):
                slice = tp_ori_data[i * group_size : (i + 1) * group_size]
                tp_sampled_nums.append(slice.tolist())
        return tp_sampled_nums

    def _get_remote_ranks_for_req(self, req_id: str, prefill_tp_size: int | None = None) -> list[list[int]]:
        if prefill_tp_size is None:
            prefill_tp_size = self._prefill_tp_size

        # Divide the ports according to the TP within the PP
        sampled_nums = []
        if prefill_tp_size == self._decode_tp_size:
            sampled_nums = list(
                map(
                    lambda tp: [tp + pp * prefill_tp_size for pp in range(self._prefill_pp_size)],
                    range(prefill_tp_size),
                )
            )
            return sampled_nums
        # use deepseek mla, num_key_value_heads == 128, but consider as 1
        if self.vllm_config.model_config.is_deepseek_mla or self.use_sparse:
            num_kv_head = 1
        else:
            num_kv_head = self.num_key_value_heads
        ori_data = np.arange(prefill_tp_size * self._prefill_pp_size)
        seed = string_to_int64_hash(req_id)
        rand = random.Random(seed)
        # random split prefill tp list
        ori_data = ori_data.reshape(self._prefill_pp_size, -1)
        num_groups = max(
            1, len(ori_data[0]) // num_kv_head
        )  # The number of redundant copies for each KV head within the PP stage
        rand_group_index = rand.sample(
            range(num_groups), (max(self._decode_tp_size // num_kv_head, 1))
        )  # random choose a group
        all_results = [
            self._get_remote_tp_ranks(ori_data[pp_index], rand_group_index, num_groups, prefill_tp_size)
            for pp_index in range(self._prefill_pp_size)
        ]
        for group_index in range(len(all_results[0])):
            group = []
            for pp_index in range(self._prefill_pp_size):
                group.extend(all_results[pp_index][group_index])
            sampled_nums.append(group)
        return sampled_nums


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:  # type: ignore
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def group_concurrent_contiguous(
    src: list[int], dst: list[int]
) -> tuple[list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
    dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(
    socket: zmq.Socket,  # type: ignore
    data: bytes,
    path: str,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning(f"Send failed: {e}, retrying... ({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Send failed after all retries: {e}")
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def _send_router_ack(
    socket: zmq.Socket, identity: bytes, request_id: str  # type: ignore
) -> bool:
    """Send a bounded ROUTER acknowledgement without stalling the listener."""
    for attempt in range(CONTROL_ACK_MAX_ATTEMPTS):
        try:
            socket.send_multipart(
                (identity, b"", b"ACK"), flags=zmq.NOBLOCK
            )
            return True
        except zmq.Again:
            if attempt + 1 < CONTROL_ACK_MAX_ATTEMPTS:
                time.sleep(0.01)
        except zmq.ZMQError:
            break
    logger.warning("Failed to acknowledge control message for %s", request_id)
    return False


def ensure_zmq_recv(
    socket: zmq.Socket,  # type: ignore
    poller: zmq.Poller,  # type: ignore
    path: str,
    timeout: float = 1.0,
    max_retries: int = 3,
) -> bytes:
    retries_left = max_retries
    while True:
        try:
            if dict(poller.poll(int(timeout * 1000))):  # milliseconds
                data = socket.recv()
                return data
            else:
                raise zmq.ZMQError("Receive timeout")  # type: ignore
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning(f"Receive failed: {e}, retrying... ({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Receive failed from {path} after all retries: {e}")
                raise RuntimeError(f"Failed to receive data after {max_retries} retries: {e}")


# decode node should know pp_partition_layer in prefill node,
# it is configured in kv_transfer_config by partition_list_str,
# default using vllm layer split algorithm.
def get_prefill_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int, partition_list_str: str | None = None
) -> tuple[int, int]:
    if partition_list_str is None:
        return get_pp_indices(num_hidden_layers, pp_rank, pp_size)
    else:
        try:
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError("Invalid partition string: {}".format(partition_list_str)) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
        start_layer = sum(partitions[:pp_rank])
        end_layer = start_layer + partitions[pp_rank]
        return (start_layer, end_layer)
