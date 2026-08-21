# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 9000 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Step 5: Add or Remove Prefiller or Decoder Instances (Optional)
# ---------------------------------------------------------------
# You can add or remove prefiller or decoder instances after the proxy is started.
# For example, add 2 prefiller instances:
#
#   curl -X POST http://localhost:9000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["127.0.0.1:8102", "127.0.0.1:8103"]
#         }'
#
# or remove 1 decoder instance:
#
#   curl -X POST http://localhost:9000/instances/remove \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "decode",
#           "instances": "127.0.0.1:8201"
#         }'
#
# This will return a JSON object with the adding or removing info
# and the current prefiller and decoder instances.
#
# When adding instances, if the instances are not started,
# the proxy will wait and try until the instances to be started
# or exceeding the number of attempts
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
# - Automatic decoder-preferred Mooncake placement requires decoder servers to
#   expose vLLM's collective RPC with VLLM_SERVER_DEV_MODE=1.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import functools
import heapq
import ipaddress
import json
import os
import socket
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


_COLD_PERF_FALSE_VALUES = ("", "0", "false", "no", "off")


def _clock_domain() -> tuple[str, str]:
    host = socket.gethostname()
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = str(round(time.time() - time.monotonic()))
    return host, f"{host}:{boot}"


_HOST, _CLOCK_DOMAIN = _clock_domain()


def _service_auth_headers() -> dict[str, str]:
    """Return an authorization header only when a key is configured."""

    api_key = os.environ.get("OPENAI_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _encode_json_payload(payload: Any) -> bytes:
    """Encode a service payload once so retries can reuse the same bytes."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _log_proxy_cold_perf_event(
    event: str,
    request_id: str,
    *,
    endpoint: str,
    **fields: Any,
) -> None:
    if (
        os.environ.get("LMCACHE_COLD_START_PERF", "0").lower()
        in _COLD_PERF_FALSE_VALUES
    ):
        return
    if request_id.startswith(("cmpl-", "chatcmpl-")):
        req_id = request_id
    else:
        prefix = "chatcmpl" if "chat/" in endpoint else "cmpl"
        req_id = f"{prefix}-{request_id}"
    payload = {
        "schema": 1,
        "event": event,
        "pid": os.getpid(),
        "monotonic_ms": round(time.perf_counter() * 1000, 3),
        "wall_time_ns": time.time_ns(),
        "host": _HOST,
        "clock_domain": _CLOCK_DOMAIN,
        "req_id": req_id,
        "proxy_request_id": request_id,
        "endpoint": endpoint,
        **fields,
    }
    print(
        "[LMCACHE_COLD_PERF] "
        + json.dumps(payload, default=str, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


# Add uvloop for faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"


TAINT_PRIORITY = 1e15


class ServerState:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        try:
            ip = ipaddress.ip_address(self.host)
            if isinstance(ip, ipaddress.IPv6Address):
                self.url = f"http://[{host}]:{port}/v1"
        except Exception:
            pass
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0  # Number of active requests
        self.aborted_requests = set()  # Track aborted requests
        self.decoder_mooncake_segments: dict[int, str | None] | None = None
        self.static_decoder_mooncake_segments: dict[int, str] | None = None
        self.decoder_remote_fill: dict[int, dict[str, Any]] = {}
        self.decoder_remote_fill_discovered = False
        self.decoder_placement_discovered_at = 0.0
        self.decoder_placement_last_attempt_at = 0.0
        self.decoder_rank_active_tokens: dict[int, float] = {}
        self.decoder_placement_task: asyncio.Task[dict[int, str | None]] | None = None
        # Removed individual server lock - will use global locks instead

    def __eq__(self, other):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        other_host = other.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return self_host == other_host and str(self.port) == str(other.port)

    def __hash__(self):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return hash((self_host, str(self.port)))

    def __repr__(self):
        return f"{self.host}:{self.port}"


@dataclass
class DecoderReservation:
    """Track one decoder endpoint and optional DP-rank load reservation."""

    server: ServerState
    decoder_idx: int
    decoder_score: float
    dp_rank: int | None = None
    preferred_segment: str | None = None
    remote_fill: dict[str, Any] | None = None


class ProxyState:
    def __init__(
        self,
        prefiller_instances: list[tuple[str, int]],
        decoder_instances: list[tuple[str, int]],
        decoder_mooncake_segments: list[dict[int, str]] | None = None,
        enable_remote_lmcache_store: bool = False,
        decoder_placement_discovery_timeout_seconds: float = 2.0,
        decoder_placement_positive_ttl_seconds: float = 30.0,
        decoder_placement_negative_ttl_seconds: float = 3.0,
    ) -> None:
        self.request_num = 0
        self.tainted_prefillers: list[ServerState] = []
        self.tainted_decoders: list[ServerState] = []
        self.prefillers: list[ServerState] = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.decoder_mooncake_segments = decoder_mooncake_segments
        self.enable_remote_lmcache_store = bool(enable_remote_lmcache_store)
        if decoder_placement_discovery_timeout_seconds <= 0:
            raise ValueError("Decoder placement discovery timeout must be positive")
        self.decoder_placement_discovery_timeout_seconds = float(
            decoder_placement_discovery_timeout_seconds
        )
        if decoder_placement_positive_ttl_seconds <= 0:
            raise ValueError("Positive decoder placement TTL must be positive")
        if decoder_placement_negative_ttl_seconds <= 0:
            raise ValueError("Negative decoder placement TTL must be positive")
        self.decoder_placement_positive_ttl_seconds = float(
            decoder_placement_positive_ttl_seconds
        )
        self.decoder_placement_negative_ttl_seconds = float(
            decoder_placement_negative_ttl_seconds
        )
        if decoder_mooncake_segments is not None:
            if len(decoder_mooncake_segments) != len(self.decoders):
                raise ValueError(
                    "Decoder Mooncake mappings must match decoder endpoints"
                )
            for server, rank_segments in zip(self.decoders, decoder_mooncake_segments):
                server.decoder_mooncake_segments = dict(rank_segments)
                server.static_decoder_mooncake_segments = dict(rank_segments)
                server.decoder_rank_active_tokens = {
                    dp_rank: 0.0 for dp_rank in rank_segments
                }
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        # Removed selection locks - no longer needed for synchronous methods

        # Initialize priority queues for efficient server selection
        # Each entry is (priority_score, server_index, server_reference)
        # Lower priority score = higher priority (less loaded)
        self.prefiller_heap = [(0.0, i, server) for i, server in enumerate(self.prefillers)]
        self.decoder_heap = [(0.0, i, server) for i, server in enumerate(self.decoders)]
        heapq.heapify(self.prefiller_heap)
        heapq.heapify(self.decoder_heap)
        # Dynamic topology mutates these heaps from a background thread. Keep
        # the prototype's paired P+D+proxy topology static while RemoteFill is
        # enabled; endpoint replacement then occurs through a full restart.
        self.node_listener = (
            None if self.enable_remote_lmcache_store else NodeListener(self)
        )

    def _update_prefiller_priority(self, server_idx: int):
        """Update the priority of a prefiller server in the heap."""
        server = self.prefillers[server_idx]
        # Priority based on active_tokens and active_kv_cache
        priority = server.active_tokens + server.active_kv_cache * 0.3
        # Remove old entry and add new one
        self.prefiller_heap = [(p, i, s) for p, i, s in self.prefiller_heap if i != server_idx]
        heapq.heappush(self.prefiller_heap, (priority, server_idx, server))

    def _update_decoder_priority(self, server_idx: int):
        """Update the priority of a decoder server in the heap."""
        server = self.decoders[server_idx]
        priority = server.active_tokens
        # Remove old entry and add new one
        self.decoder_heap = [(p, i, s) for p, i, s in self.decoder_heap if i != server_idx]
        heapq.heappush(self.decoder_heap, (priority, server_idx, server))

    def abort_prefiller_request(self, server_idx: int, request_id):  # Changed to synchronous
        """
        Mark a request as aborted. This will helps to release kv cache in
        prefiller node.
        """
        # No lock needed - atomic operation
        if server_idx >= len(self.prefillers):
            return
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int):  # Changed to synchronous
        """
        Get the set of aborted requests and clear it.
        This is used to release kv cache in prefiller node.
        """
        # No lock needed - atomic operation
        if server_idx >= len(self.prefillers):
            return set()
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_prefiller(self, token_count):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.prefiller_heap:
            raise RuntimeError("No prefiller servers available")

        priority, chosen, server = heapq.heappop(self.prefiller_heap)

        # Update the chosen server atomically
        self.prefillers[chosen].active_tokens += token_count
        self.prefillers[chosen].active_kv_cache += token_count

        # Update priority and re-add to heap
        self._update_prefiller_priority(chosen)

        return chosen

    def release_prefiller(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.prefillers):
            return
        self.prefillers[idx].active_tokens -= token_count
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.prefillers):
            return
        if self.prefillers[idx].active_kv_cache > 0:
            self.prefillers[idx].active_kv_cache -= token_count
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def select_decoder(self, token_count):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.decoder_heap:
            raise RuntimeError("No decoder servers available")

        priority, chosen, server = heapq.heappop(self.decoder_heap)

        # Update the chosen server atomically
        self.decoders[chosen].active_tokens += token_count

        # Update priority and re-add to heap
        self._update_decoder_priority(chosen)

        return chosen

    def release_decoder(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.decoders):
            return
        self.decoders[idx].active_tokens -= token_count
        # Update priority queue after releasing
        self._update_decoder_priority(idx)

    def select_decoder_reservation(
        self, token_count: float
    ) -> DecoderReservation:
        """Reserve an endpoint and its least-loaded configured DP rank.

        Args:
            token_count: Load score charged to the endpoint and selected rank.

        Returns:
            The endpoint/rank reservation that must be released exactly once.
        """
        decoder_idx = self.select_decoder(token_count)
        reservation = DecoderReservation(
            self.decoders[decoder_idx], decoder_idx, token_count
        )
        return self.assign_decoder_rank(reservation)

    def assign_decoder_rank(
        self, reservation: DecoderReservation
    ) -> DecoderReservation:
        """Assign the least-loaded discovered DP rank to an endpoint reservation."""
        server = reservation.server
        rank_segments = server.decoder_mooncake_segments
        if not rank_segments:
            return reservation
        dp_rank = min(
            rank_segments,
            key=lambda rank: (
                server.decoder_rank_active_tokens[rank],
                rank,
            ),
        )
        server.decoder_rank_active_tokens[dp_rank] += reservation.decoder_score
        reservation.dp_rank = dp_rank
        reservation.preferred_segment = rank_segments[dp_rank]
        remote_fill = (
            getattr(server, "decoder_remote_fill", {}).get(dp_rank)
            if getattr(self, "enable_remote_lmcache_store", False)
            else None
        )
        reservation.remote_fill = dict(remote_fill) if remote_fill is not None else None
        return reservation

    @staticmethod
    def reset_decoder_placement(server: ServerState) -> None:
        """Invalidate request-independent placement after a lifecycle change."""

        task = server.decoder_placement_task
        if task is not None and not task.done():
            task.cancel()
        server.decoder_placement_task = None
        server.decoder_remote_fill = {}
        server.decoder_remote_fill_discovered = False
        server.decoder_placement_discovered_at = 0.0
        server.decoder_placement_last_attempt_at = 0.0
        static_segments = getattr(server, "static_decoder_mooncake_segments", None)
        server.decoder_mooncake_segments = (
            dict(static_segments) if static_segments is not None else None
        )
        server.decoder_rank_active_tokens = {
            dp_rank: 0.0 for dp_rank in (static_segments or {})
        }

    def _decoder_placement_is_fresh(self, server: ServerState, now: float) -> bool:
        if not self.enable_remote_lmcache_store:
            return server.decoder_mooncake_segments is not None
        if server.decoder_remote_fill_discovered:
            ttl = (
                getattr(self, "decoder_placement_positive_ttl_seconds", 30.0)
                if server.decoder_remote_fill
                else getattr(
                    self, "decoder_placement_negative_ttl_seconds", 3.0
                )
            )
            return now - getattr(server, "decoder_placement_discovered_at", 0.0) < ttl
        # A failed discovery may be negatively cached only when a static
        # persistent placement remains available for this request.
        return bool(
            server.decoder_mooncake_segments is not None
            and getattr(server, "decoder_placement_last_attempt_at", 0.0)
            and now - getattr(server, "decoder_placement_last_attempt_at", 0.0)
            < getattr(self, "decoder_placement_negative_ttl_seconds", 3.0)
        )

    async def ensure_decoder_mooncake_segments(
        self,
        server: ServerState,
        request_id: str,
        endpoint: str,
    ) -> None:
        """Discover and cache one decoder endpoint's TP0 Mooncake segments."""
        discover_remote_fill = getattr(self, "enable_remote_lmcache_store", False)
        now = time.monotonic()
        if self._decoder_placement_is_fresh(server, now):
            return
        task = server.decoder_placement_task
        if task is None:
            server.decoder_placement_last_attempt_at = now
            task = asyncio.create_task(
                _discover_decoder_mooncake_segments(
                    server,
                    enable_remote_fill=discover_remote_fill,
                    timeout_seconds=getattr(
                        self,
                        "decoder_placement_discovery_timeout_seconds",
                        2.0,
                    ),
                )
            )
            server.decoder_placement_task = task
        try:
            rank_segments = await asyncio.shield(task)
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            if server.decoder_placement_task is task:
                server.decoder_placement_task = None
                server.decoder_remote_fill = {}
                # A transient control-plane failure must not permanently
                # disable discovery. Static Mooncake mappings, when present,
                # remain independently usable for this request.
                server.decoder_remote_fill_discovered = False
            logger.warning(
                "Mooncake placement discovery failed for decoder %s: %s",
                server.url,
                exc,
            )
            _log_proxy_cold_perf_event(
                "proxy_decoder_placement_discovery_failed",
                request_id,
                endpoint=endpoint,
                decoder_url=server.url,
                error=str(exc),
            )
            return

        static_ranks = set(
            getattr(server, "static_decoder_mooncake_segments", None) or {}
        )
        remote_fill_ranks = set(server.decoder_remote_fill)
        if static_ranks and remote_fill_ranks and static_ranks != remote_fill_ranks:
            logger.error(
                "Disabling remote fill for decoder %s because static Mooncake "
                "ranks %s differ from discovered RemoteFill ranks %s",
                server.url,
                sorted(static_ranks),
                sorted(remote_fill_ranks),
            )
            server.decoder_remote_fill = {}
        if server.decoder_mooncake_segments is None:
            server.decoder_mooncake_segments = rank_segments
            server.decoder_rank_active_tokens = {
                dp_rank: 0.0 for dp_rank in rank_segments
            }
        server.decoder_remote_fill_discovered = discover_remote_fill
        server.decoder_placement_discovered_at = time.monotonic()
        server.decoder_placement_task = None
        _log_proxy_cold_perf_event(
            "proxy_decoder_placement_discovered",
            request_id,
            endpoint=endpoint,
            decoder_url=server.url,
            rank_segments=rank_segments,
        )

    def release_decoder_reservation(
        self, reservation: DecoderReservation
    ) -> None:
        """Release a decoder reservation without mutating replacement servers."""
        server = reservation.server
        if reservation.dp_rank is not None:
            rank_tokens = server.decoder_rank_active_tokens.get(
                reservation.dp_rank, 0.0
            )
            server.decoder_rank_active_tokens[reservation.dp_rank] = max(
                0.0, rank_tokens - reservation.decoder_score
            )
        decoder_idx = next(
            (
                index
                for index, decoder in enumerate(self.decoders)
                if decoder is server
            ),
            None,
        )
        if decoder_idx is not None:
            server.active_tokens = max(
                0.0, server.active_tokens - reservation.decoder_score
            )
            self._update_decoder_priority(decoder_idx)
        else:
            server.active_tokens = max(
                0.0, server.active_tokens - reservation.decoder_score
            )

    # Omni_infer's calculate_input_scores function
    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        input_score = length_score * 0.0345 + 120.0745
        return input_score

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length

    async def add_instances(self, instance_type: str, instances: list[ServerState]) -> tuple[list[str], list[str]]:
        if self.enable_remote_lmcache_store:
            raise RuntimeError(
                "Dynamic topology is disabled while remote LMCache store is enabled; "
                "restart the paired proxy, prefiller, and decoder deployment"
            )
        assert self.node_listener is not None
        added_nodes, waiting_nodes = [], []
        for server in instances:
            is_valid = await self.node_listener.check_instance_status(server.client)
            if is_valid and instance_type == InstanceType.PREFILL:
                self.add_prefillers([server])
                added_nodes.append(str(server))
            elif is_valid and instance_type == InstanceType.DECODE:
                self.add_decoders([server])
                added_nodes.append(str(server))
            else:
                node = str(server)
                self.node_listener.waiting_nodes[node] = (instance_type, server, 0)
                waiting_nodes.append(node)
        return added_nodes, waiting_nodes

    def add_prefillers(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_prefillers:
                self.tainted_prefillers.remove(server)
                self.prefiller_heap = [
                    (0, idx, server) if srv == server else (priority, idx, srv)
                    for priority, idx, srv in self.prefiller_heap
                ]
                heapq.heapify(self.prefiller_heap)
            elif server not in self.prefillers:
                self.prefillers.append(server)
                # prefiller_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]
                heapq.heappush(self.prefiller_heap, (0, len(self.prefillers) - 1, server))
        self.print_status(f"Add prefiller instances: {instances}.")

    def add_decoders(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_decoders:
                self.tainted_decoders.remove(server)
                self.decoder_heap = [
                    (0, idx, server) if srv == server else (priority, idx, srv)
                    for priority, idx, srv in self.decoder_heap
                ]
                heapq.heapify(self.decoder_heap)
            elif server not in self.decoders:
                self.decoders.append(server)
                # decoder_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]
                heapq.heappush(self.decoder_heap, (0, len(self.decoders) - 1, server))
        self.print_status(f"Add decoder instances: {instances}.")

    def remove_prefillers(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False
        if self.enable_remote_lmcache_store:
            raise RuntimeError(
                "Dynamic topology is disabled while remote LMCache store is enabled"
            )

        if self.request_num > 0:
            logger.warning(f"Start to taint prefill instances {instances}.")
            self._taint_prefillers(instances)
            return True

        instances_to_remove = set(instances)
        self.prefillers = [server for server in self.prefillers if server not in instances_to_remove]
        prefiller_heap_copy = self.prefiller_heap.copy()
        prefiller_heap_copy.sort(key=lambda x: x[1])  # sorted by key: prefiller_idx
        prefiller_heap = []
        idx = 0
        for priority, _, server in prefiller_heap_copy:
            if server not in instances_to_remove:
                prefiller_heap.append((priority, idx, server))
                idx += 1

        # prefiller_heap: [(priority_0, 0, server_0), (priority_1, 1, server_1)] -> [(priority_1, 0, server_1)]
        self.prefiller_heap = prefiller_heap
        heapq.heapify(self.prefiller_heap)
        self.print_status(f"Remove prefiller instances: {instances}.")
        return False

    def remove_decoders(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False
        if self.enable_remote_lmcache_store:
            raise RuntimeError(
                "Dynamic topology is disabled while remote LMCache store is enabled"
            )

        if self.request_num > 0:
            logger.warning(f"Start to taint decode instances {instances}.")
            self._taint_decoders(instances)
            return True

        instances_to_remove = set(instances)
        self.decoders = [server for server in self.decoders if server not in instances_to_remove]
        decoder_heap_copy = self.decoder_heap.copy()
        decoder_heap_copy.sort(key=lambda x: x[1])  # sorted by key: decoder_idx
        decoder_heap = []
        idx = 0
        for priority, _, server in decoder_heap_copy:
            if server not in instances_to_remove:
                decoder_heap.append((priority, idx, server))
                idx += 1

        # decoder_heap: [(priority_0, 0, server_0), (priority_1, 1, server_1)] -> [(priority_1, 0, server_1)]
        self.decoder_heap = decoder_heap
        heapq.heapify(self.decoder_heap)
        self.print_status(f"Remove decoder instances: {instances}.")
        return False

    def _taint_prefillers(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.prefillers:
            if server in instances_to_taint and server not in self.tainted_prefillers:
                self.tainted_prefillers.append(server)

        self.prefiller_heap = [
            (TAINT_PRIORITY, idx, srv) if srv in instances_to_taint else (priority, idx, srv)
            for priority, idx, srv in self.prefiller_heap
        ]
        heapq.heapify(self.prefiller_heap)

    def _taint_decoders(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.decoders:
            if server in instances_to_taint and server not in self.tainted_decoders:
                self.tainted_decoders.append(server)

        self.decoder_heap = [
            (TAINT_PRIORITY, idx, srv) if srv in instances_to_taint else (priority, idx, srv)
            for priority, idx, srv in self.decoder_heap
        ]
        heapq.heapify(self.decoder_heap)

    def print_status(self, msg: str) -> None:
        status = {
            "prefill_instances": [str(server) for server in self.prefillers],
            "decode_instances": [str(server) for server in self.decoders],
        }
        print(f"{msg} Status: {status}")


proxy_state = None


class NodeListener:
    def __init__(self, proxy):
        self.proxy_state = proxy
        self.waiting_nodes: dict[str, tuple[str, Any, int]] = {}
        self.listening_thread = threading.Thread(target=self._node_listener, daemon=True)
        self.listening_thread.start()

    def _node_listener(self) -> None:
        while True:
            for node, (instance_type, server, check_times) in list(self.waiting_nodes.items()):
                is_valid = asyncio.run(self.check_instance_status(server.client))
                print(f"Checking instance {node}...")
                check_times += 1
                if is_valid:
                    if instance_type == InstanceType.PREFILL:
                        self.proxy_state.add_prefillers([server])
                    else:
                        self.proxy_state.add_decoders([server])
                    self.waiting_nodes.pop(node)
                elif check_times == global_args.max_waiting_retries:
                    print(f"Instance {node} was not added to the proxy.")
                    self.waiting_nodes.pop(node)
                else:
                    self.waiting_nodes[node] = (instance_type, server, check_times)

            if self.proxy_state.tainted_prefillers and not self.proxy_state.request_num:
                need_waiting = self.proxy_state.remove_prefillers(self.proxy_state.tainted_prefillers)
                if not need_waiting:
                    self.proxy_state.tainted_prefillers.clear()

            if self.proxy_state.tainted_decoders and not self.proxy_state.request_num:
                need_waiting = self.proxy_state.remove_decoders(self.proxy_state.tainted_decoders)
                if not need_waiting:
                    self.proxy_state.tainted_decoders.clear()
            time.sleep(global_args.waiting_retry_interval)

    @staticmethod
    async def check_instance_status(client: httpx.AsyncClient) -> bool:
        endpoint = "/models"
        try:
            response = await client.get(endpoint, headers=_service_auth_headers())
            response.raise_for_status()
            return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def _parse_decoder_remote_fill_response(payload: Any) -> dict[int, dict[str, Any]]:
    """Validate bounded remote-fill discovery metadata from decoder TP0s."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Decoder collective RPC returned an invalid response")
    placements: dict[int, dict[str, Any]] = {}
    for result in payload["results"]:
        if result is None:
            continue
        if not isinstance(result, dict):
            raise ValueError("Decoder placement result must be a dictionary")
        remote_fill = result.get("remote_fill")
        if remote_fill is None:
            continue
        if not isinstance(remote_fill, dict):
            raise ValueError("Decoder remote-fill placement must be a dictionary")
        if remote_fill.get("enabled") is not True:
            continue
        dp_rank = result.get("dp_rank")
        if isinstance(dp_rank, bool) or not isinstance(dp_rank, int) or dp_rank < 0:
            raise ValueError(f"Invalid decoder data-parallel rank: {dp_rank!r}")
        advertised_dp_rank = remote_fill.get("dp_rank")
        advertised_tp_rank = remote_fill.get("tp_rank")
        if (
            isinstance(advertised_dp_rank, bool)
            or not isinstance(advertised_dp_rank, int)
            or advertised_dp_rank != dp_rank
            or isinstance(advertised_tp_rank, bool)
            or not isinstance(advertised_tp_rank, int)
            or advertised_tp_rank != 0
        ):
            raise ValueError(
                "Decoder remote-fill placement is not bound to its TP0/DP rank"
            )
        required_strings = (
            "destination_engine_id",
            "control_endpoint",
            "token_hash_algorithm",
        )
        if any(
            not isinstance(remote_fill.get(name), str)
            or not remote_fill[name].strip()
            for name in required_strings
        ):
            raise ValueError("Decoder remote-fill string identity is invalid")
        epoch = remote_fill.get("destination_engine_epoch")
        generation = remote_fill.get("shared_cache_generation")
        tp_size = remote_fill.get("destination_tp_size")
        dp_size = remote_fill.get("destination_dp_size")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("Decoder remote-fill engine epoch is invalid")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("Decoder remote-fill shared-cache generation is invalid")
        if (
            isinstance(tp_size, bool)
            or not isinstance(tp_size, int)
            or tp_size <= 0
            or isinstance(dp_size, bool)
            or not isinstance(dp_size, int)
            or dp_size <= 0
            or dp_rank >= dp_size
        ):
            raise ValueError("Decoder remote-fill parallel topology is invalid")
        global_te_push = remote_fill.get("global_te_push")
        if not isinstance(global_te_push, bool):
            raise ValueError("Decoder remote-fill native capability is invalid")
        if not global_te_push:
            continue
        python_hash_seed = remote_fill.get("python_hash_seed", "")
        if not isinstance(python_hash_seed, str) or (
            remote_fill["token_hash_algorithm"] == "builtin"
            and not python_hash_seed
        ):
            raise ValueError("Decoder remote-fill hash identity is invalid")
        placement = {
            "destination_engine_id": remote_fill["destination_engine_id"].strip(),
            "destination_engine_epoch": epoch,
            "control_endpoint": remote_fill["control_endpoint"].strip(),
            "destination_dp_rank": dp_rank,
            "shared_cache_generation": generation,
            "destination_tp_size": tp_size,
            "destination_dp_size": dp_size,
            "global_te_push": global_te_push,
            "token_hash_algorithm": remote_fill["token_hash_algorithm"].strip(),
            "python_hash_seed": python_hash_seed,
        }
        existing = placements.get(dp_rank)
        if existing is not None and existing != placement:
            raise ValueError(
                f"Decoder DP rank {dp_rank} reported conflicting remote-fill metadata"
            )
        placements[dp_rank] = placement
    return placements


def _parse_decoder_placement_response(
    payload: Any,
    *,
    allow_remote_fill_only: bool = False,
) -> dict[int, str | None]:
    """Validate JSON returned by vLLM's collective RPC endpoint."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Decoder collective RPC returned an invalid response")
    remote_fill_ranks = (
        set(_parse_decoder_remote_fill_response(payload))
        if allow_remote_fill_only
        else set()
    )
    rank_segments: dict[int, str | None] = {}
    for result in payload["results"]:
        if result is None:
            continue
        if not isinstance(result, dict):
            raise ValueError("Decoder placement result must be a dictionary")
        dp_rank = result.get("dp_rank")
        segment = result.get("segment")
        if isinstance(dp_rank, bool) or not isinstance(dp_rank, int) or dp_rank < 0:
            raise ValueError(f"Invalid decoder data-parallel rank: {dp_rank!r}")
        if segment is None and dp_rank in remote_fill_ranks:
            normalized_segment = None
        elif not isinstance(segment, str) or not segment.strip():
            raise ValueError(f"Invalid Mooncake segment for DP rank {dp_rank}")
        else:
            normalized_segment = segment.strip()
        existing = rank_segments.get(dp_rank)
        if dp_rank in rank_segments and existing != normalized_segment:
            raise ValueError(
                f"Decoder DP rank {dp_rank} reported conflicting Mooncake segments"
            )
        rank_segments[dp_rank] = normalized_segment
    if not rank_segments:
        raise ValueError("Decoder did not report any TP0 Mooncake segments")
    return rank_segments


async def _discover_decoder_mooncake_segments(
    server: ServerState,
    *,
    enable_remote_fill: bool = False,
    timeout_seconds: float = 2.0,
) -> dict[int, str | None]:
    """Fetch dynamic TP0 Mooncake addresses through vLLM's existing RPC."""
    collective_rpc_url = server.url.removesuffix("/v1") + "/collective_rpc"
    response = await server.client.post(
        collective_rpc_url,
        json={"method": "get_mooncake_placement_info"},
        headers=_service_auth_headers(),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    server.decoder_remote_fill = (
        _parse_decoder_remote_fill_response(payload) if enable_remote_fill else {}
    )
    return _parse_decoder_placement_response(
        payload,
        allow_remote_fill_only=enable_remote_fill,
    )


def _parse_decoder_mooncake_segments(
    raw_mappings: list[str] | None, decoder_count: int
) -> list[dict[int, str]] | None:
    """Parse endpoint-aligned routable-DP-rank Mooncake segment mappings."""
    if raw_mappings is None:
        return None
    if len(raw_mappings) != decoder_count:
        raise ValueError(
            "Number of --decoder-mooncake-segments arguments must match "
            "the number of decoder endpoints"
        )

    parsed_mappings = []
    for endpoint_idx, raw_mapping in enumerate(raw_mappings):
        if not raw_mapping or not raw_mapping.strip():
            raise ValueError(
                f"Decoder endpoint {endpoint_idx} has an empty Mooncake segment mapping"
            )
        rank_segments = {}
        seen_ranks = set()
        for raw_entry in raw_mapping.split(","):
            entry = raw_entry.strip()
            if not entry or "=" not in entry:
                raise ValueError(
                    "Each decoder Mooncake mapping must use global_rank=segment"
                )
            raw_rank, raw_segment = entry.split("=", 1)
            try:
                dp_rank = int(raw_rank.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid decoder data-parallel rank: {raw_rank!r}"
                ) from exc
            if dp_rank < 0:
                raise ValueError("Decoder data-parallel ranks must be non-negative")
            segment = raw_segment.strip()
            if not segment:
                raise ValueError(
                    f"Decoder data-parallel rank {dp_rank} has an empty Mooncake segment"
                )
            if dp_rank in seen_ranks:
                raise ValueError(
                    f"Decoder data-parallel rank {dp_rank} is mapped more than once"
                )
            seen_ranks.add(dp_rank)
            rank_segments[dp_rank] = segment
        parsed_mappings.append(rank_segments)
    return parsed_mappings


def parse_args() -> argparse.Namespace:
    """Parse proxy endpoints and optional placement-discovery overrides."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument(
        "--decoder-mooncake-segments",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional override for automatically discovered per-decoder request "
            "routing rank to Mooncake segment mappings, "
            'for example "0=decoder-a:12345,1=decoder-b:12345"'
        ),
    )
    parser.add_argument(
        "--enable-remote-lmcache-store",
        action="store_true",
        help=(
            "Opt in to decoder RemoteFill discovery and direct remote "
            "LMCache storage"
        ),
    )
    parser.add_argument(
        "--decoder-placement-discovery-timeout-seconds",
        type=float,
        default=2.0,
        help="Timeout for the decoder placement collective RPC",
    )
    parser.add_argument(
        "--decoder-placement-positive-ttl-seconds",
        type=float,
        default=30.0,
        help="Refresh interval for a usable decoder RemoteFill placement",
    )
    parser.add_argument(
        "--decoder-placement-negative-ttl-seconds",
        type=float,
        default=3.0,
        help="Retry interval after discovery returns no RemoteFill placement",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    parser.add_argument(
        "--max-waiting-retries", type=int, default=3, help="Maximum number of retries for waiting nodes to be started"
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting nodes to be started",
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    if args.decoder_placement_discovery_timeout_seconds <= 0:
        raise ValueError("Decoder placement discovery timeout must be positive")
    if args.decoder_placement_positive_ttl_seconds <= 0:
        raise ValueError("Positive decoder placement TTL must be positive")
    if args.decoder_placement_negative_ttl_seconds <= 0:
        raise ValueError("Negative decoder placement TTL must be positive")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    args.decoder_mooncake_segments = _parse_decoder_mooncake_segments(
        args.decoder_mooncake_segments,
        len(args.decoder_instances),
    )
    return args


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create proxy clients for the application lifetime and close them."""

    global proxy_state
    proxy_state = ProxyState(
        global_args.prefiller_instances,
        global_args.decoder_instances,
        getattr(global_args, "decoder_mooncake_segments", None),
        getattr(global_args, "enable_remote_lmcache_store", False),
        getattr(
            global_args,
            "decoder_placement_discovery_timeout_seconds",
            2.0,
        ),
        getattr(global_args, "decoder_placement_positive_ttl_seconds", 30.0),
        getattr(global_args, "decoder_placement_negative_ttl_seconds", 3.0),
    )
    print(f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients.")
    if proxy_state.enable_remote_lmcache_store:
        # Warm discovery during application startup so the first user request
        # does not pay the collective-RPC timeout.
        await asyncio.gather(
            *(
                proxy_state.ensure_decoder_mooncake_segments(
                    decoder,
                    f"proxy-startup-{index}",
                    "startup",
                )
                for index, decoder in enumerate(proxy_state.decoders)
            )
        )
    yield
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()


async def listen_for_disconnect(request: Request) -> None:
    """Return if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_id: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
    preferred_mooncake_segment: str | None = None,
    remote_fill_handoff: dict[str, Any] | None = None,
) -> httpx.Response:
    if remote_fill_handoff is not None and preferred_mooncake_segment is not None:
        raise ValueError(
            "Remote fill and decoder-directed Mooncake placement are mutually exclusive"
        )
    aborted_requests = proxy_state.acquire_aborted_prefiller_requests(prefiller_id)
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": list(aborted_requests),
    }
    if preferred_mooncake_segment is not None:
        req_data["kv_transfer_params"]["lmcache.mooncake_preferred_segment"] = (
            preferred_mooncake_segment
        )
    if remote_fill_handoff is not None:
        req_data["kv_transfer_params"]["lmcache.remote_fill"] = dict(
            remote_fill_handoff
        )
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    req_data["min_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    encode_started = time.perf_counter()
    request_content = _encode_json_payload(req_data)
    _log_proxy_cold_perf_event(
        "proxy_prefill_body_encode_complete",
        request_id,
        endpoint=endpoint,
        encode_ms=round((time.perf_counter() - encode_started) * 1000, 3),
        body_bytes=len(request_content),
    )
    headers = {
        **_service_auth_headers(),
        "X-Request-Id": request_id,
        "Content-Type": "application/json",
    }
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(
                endpoint,
                content=request_content,
                headers=headers,
            )
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
                raise last_exc


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    request_content: bytes,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
    decoder_dp_rank: int | None = None,
) -> AsyncIterator[bytes]:
    headers = {
        **_service_auth_headers(),
        "X-Request-Id": request_id,
        "Content-Type": "application/json",
    }
    if decoder_dp_rank is not None:
        headers["X-data-parallel-rank"] = str(decoder_dp_rank)
    for attempt in range(1, max_retries + 1):
        first_chunk_sent = False
        try:
            _log_proxy_cold_perf_event(
                "proxy_decoder_send_start",
                request_id,
                endpoint=endpoint,
                attempt=attempt,
                decoder_url=str(getattr(client, "base_url", "")),
                body_bytes=len(request_content),
            )
            async with client.stream(
                "POST",
                endpoint,
                content=request_content,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if not first_chunk_sent:
                        _log_proxy_cold_perf_event(
                            "proxy_decoder_first_byte_received",
                            request_id,
                            endpoint=endpoint,
                            attempt=attempt,
                            response_bytes=len(chunk),
                        )
                    first_chunk_sent = True
                    yield chunk
                return  # Success, exit after streaming
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if first_chunk_sent:
                logger.error(
                    "Streaming to client interrupted after response started: %s",
                    str(e),
                )
                return
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise e
        except Exception as e:
            # If any chunk has been sent, do not retry, just log and drop
            if first_chunk_sent:
                logger.error(f"Streaming to client interrupted after response started: {str(e)}")
                return
            else:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                else:
                    logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                    raise e


async def _handle_select_instance(
    api: str,
    req_data: Any,
    request_length: int,
    *,
    request_id: str | None = None,
    log_request_received: bool = True,
) -> "InstanceInfo":
    prefiller_score = proxy_state.calculate_prefill_scores(request_length)
    logger.debug(f"Request length: {request_length}, Prefiller score: {prefiller_score}")
    request_id = request_id or await proxy_state.next_req_id()
    if log_request_received:
        _log_proxy_cold_perf_event(
            "proxy_request_received",
            request_id,
            endpoint=api,
            request_bytes=request_length,
        )
    decoder_score = proxy_state.calculate_decode_scores(request_length)
    logger.debug("Decoder score: %f", decoder_score)
    prefiller_idx = None
    prefiller_active_released = False
    decoder = None
    reservation = None
    try:
        decoder_idx = proxy_state.select_decoder(decoder_score)
        decoder = proxy_state.decoders[decoder_idx]
        reservation = DecoderReservation(
            decoder, decoder_idx, decoder_score
        )
        # Discovery is a first-use operation. Keep the stable request path
        # synchronous once this endpoint has a mapping (or a cached fallback).
        if (
            getattr(decoder, "decoder_mooncake_segments", None) is None
            or getattr(proxy_state, "enable_remote_lmcache_store", False)
        ):
            await proxy_state.ensure_decoder_mooncake_segments(
                decoder, request_id, api
            )
        proxy_state.assign_decoder_rank(reservation)
        if reservation.preferred_segment is not None:
            _log_proxy_cold_perf_event(
                "proxy_decoder_placement_reserved",
                request_id,
                endpoint=api,
                decoder_url=decoder.url,
                decoder_idx=reservation.decoder_idx,
                dp_rank=reservation.dp_rank,
                preferred_segment=reservation.preferred_segment,
                request_bytes=request_length,
            )

        prefiller_idx = proxy_state.select_prefiller(prefiller_score)
        prefiller = proxy_state.prefillers[prefiller_idx]
        remote_fill_handoff = None
        if (
            getattr(proxy_state, "enable_remote_lmcache_store", False)
            and reservation.remote_fill is not None
        ):
            remote_fill_handoff = {
                **reservation.remote_fill,
                "transfer_id": uuid.uuid4().hex,
                "request_attempt": 1,
                "source_engine_id": str(prefiller),
            }
        _log_proxy_cold_perf_event(
            "proxy_prefiller_dispatch",
            request_id,
            endpoint=api,
            prefiller_url=str(getattr(prefiller, "url", "")),
            request_bytes=request_length,
        )
        response = await send_request_to_service(
            prefiller.client,
            prefiller_idx,
            api,
            req_data,
            request_id,
            # Retrying the same complete prefill HTTP body after a lost
            # response is not yet proven idempotent. RemoteFill protocol
            # operation IDs cannot deduplicate a second model execution.
            max_retries=(1 if remote_fill_handoff is not None else global_args.max_retries),
            base_delay=global_args.retry_delay,
            preferred_mooncake_segment=(
                reservation.preferred_segment
                if reservation and reservation.remote_fill is None
                else None
            ),
            remote_fill_handoff=remote_fill_handoff,
        )
        _log_proxy_cold_perf_event(
            "proxy_prefill_response_received",
            request_id,
            endpoint=api,
            prefiller_url=prefiller.url,
            response_bytes=len(response.content),
            request_bytes=request_length,
        )
        proxy_state.release_prefiller(prefiller_idx, prefiller_score)
        prefiller_active_released = True
        response_json = response.json()
        returned_kv_transfer_params = response_json.get(
            "kv_transfer_params", {}
        )
        if not isinstance(returned_kv_transfer_params, dict):
            raise TypeError(
                "Prefiller kv_transfer_params must be a dictionary"
            )
        # The proxy owns transport metadata. Never inherit caller-provided
        # values, and never send the producer-only placement hint to decode.
        kv_transfer_params = dict(returned_kv_transfer_params)
        returned_remote_fill = kv_transfer_params.pop("lmcache.remote_fill", None)
        kv_transfer_params.pop("lmcache.remote_fill_result", None)
        if remote_fill_handoff is not None:
            if not isinstance(returned_remote_fill, dict):
                raise RuntimeError("Prefiller omitted remote-fill terminal result")
            terminal = returned_remote_fill.get("terminal")
            if not isinstance(terminal, dict):
                raise RuntimeError("Prefiller returned invalid remote-fill terminal result")
            outcome = terminal.get("outcome")
            persistent_end = terminal.get("persistent_common_end")
            required_end = terminal.get("required_store_end")
            terminal_transfer_id = terminal.get("transfer_id")
            valid_ints = (
                not isinstance(persistent_end, bool)
                and isinstance(persistent_end, int)
                and not isinstance(required_end, bool)
                and isinstance(required_end, int)
            )
            if (
                outcome not in {"LOCAL_FULL", "PERSISTENT_ONLY"}
                or not valid_ints
                or persistent_end < 0
                or required_end < 0
                or persistent_end < required_end
                or terminal_transfer_id != remote_fill_handoff["transfer_id"]
            ):
                raise RuntimeError(
                    "Prefiller remote-fill result is not safe for decoder forwarding"
                )
            if outcome == "LOCAL_FULL":
                # Bounded diagnostic hint only. Decoder LMCache still performs
                # ordinary lookup-and-pin and may observe eviction.
                kv_transfer_params["lmcache.remote_fill_result"] = {
                    "outcome": "LOCAL_FULL",
                    "required_store_end": required_end,
                    "destination_engine_epoch": remote_fill_handoff[
                        "destination_engine_epoch"
                    ],
                }
        kv_transfer_params.pop(
            "lmcache.mooncake_preferred_segment", None
        )
        req_data["kv_transfer_params"] = kv_transfer_params

        encode_started = time.perf_counter()
        decoder_body = _encode_json_payload(req_data)
        _log_proxy_cold_perf_event(
            "proxy_decoder_body_encode_complete",
            request_id,
            endpoint=api,
            encode_ms=round((time.perf_counter() - encode_started) * 1000, 3),
            body_bytes=len(decoder_body),
        )
        _log_proxy_cold_perf_event(
            "proxy_decoder_dispatch_ready",
            request_id,
            endpoint=api,
            decoder_url=decoder.url,
            request_bytes=request_length,
            kv_transfer_param_keys=sorted(str(key) for key in kv_transfer_params),
            remote_fill_transfer_id=(
                remote_fill_handoff["transfer_id"]
                if remote_fill_handoff is not None
                else None
            ),
        )
        logger.debug("Using %s %s", prefiller.url, decoder.url)
        return InstanceInfo(
            request_id=request_id,
            prefiller_idx=prefiller_idx,
            prefiller_score=prefiller_score,
            prefiller=prefiller,
            decoder=decoder,
            decoder_body=decoder_body,
            decoder_idx=reservation.decoder_idx,
            decoder_score=decoder_score,
            reservation=reservation,
        )
    except BaseException:
        if reservation is not None:
            proxy_state.release_decoder_reservation(reservation)
        if prefiller_idx is not None:
            if not prefiller_active_released:
                proxy_state.release_prefiller(prefiller_idx, prefiller_score)
            proxy_state.abort_prefiller_request(prefiller_idx, request_id)
            proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        raise


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_idx: int
    prefiller_score: float
    prefiller: ServerState
    decoder_idx: int
    decoder_score: float
    decoder: ServerState
    decoder_body: bytes
    reservation: DecoderReservation | None = None

    def __post_init__(self) -> None:
        if self.reservation is None:
            self.reservation = DecoderReservation(
                self.decoder,
                self.decoder_idx,
                self.decoder_score,
            )


def _release_decoder_reservation(instance_info: InstanceInfo) -> None:
    reservation = instance_info.reservation
    if reservation is None:
        return
    instance_info.reservation = None
    proxy_state.release_decoder_reservation(reservation)


class _CleanupStreamingResponse(StreamingResponse):
    """Run request-owned cleanup even if body iteration never starts."""

    def __init__(
        self,
        *args: Any,
        cleanup: Callable[[], None],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._cleanup()


async def _handle_completions(
    api: str, request: Request
) -> StreamingResponse:
    instance_info = None
    response_owns_cleanup = False
    request_count_released = False

    def release_request_count() -> None:
        nonlocal request_count_released
        if request_count_released:
            return
        proxy_state.request_num -= 1
        request_count_released = True

    try:
        proxy_state.request_num += 1
        request_id = str(uuid.uuid4())
        headers = getattr(request, "headers", {})
        content_length = headers.get("content-length") if headers else None
        _log_proxy_cold_perf_event(
            "proxy_request_received",
            request_id,
            endpoint=api,
            request_bytes=(
                int(content_length)
                if isinstance(content_length, str) and content_length.isdigit()
                else None
            ),
        )
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        instance_info = await _handle_select_instance(
            api,
            req_data,
            request_length,
            request_id=request_id,
            log_request_received=False,
        )
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        # refer to vLLM sampling_params: max_token default value
        origin_max_tokens = req_data.get("max_tokens", 16)
        released_kv = False

        def cleanup_current_request() -> None:
            nonlocal released_kv
            try:
                if not released_kv:
                    proxy_state.abort_prefiller_request(
                        instance_info.prefiller_idx,
                        instance_info.request_id,
                    )
                    proxy_state.release_prefiller_kv(
                        instance_info.prefiller_idx,
                        instance_info.prefiller_score,
                    )
                    released_kv = True
            finally:
                try:
                    _release_decoder_reservation(instance_info)
                finally:
                    # Keep dynamically removed endpoints tainted until the
                    # response and every request-owned reservation are done.
                    release_request_count()

        async def generate_stream() -> AsyncIterator[bytes]:
            nonlocal instance_info, released_kv
            _log_proxy_cold_perf_event(
                "proxy_decoder_generator_entry",
                instance_info.request_id,
                endpoint=api,
                decoder_url=instance_info.decoder.url,
                request_bytes=request_length,
            )
            generated_token = ""
            retry_count = 0
            retry = True
            completion_tokens = 0
            # Only one await per chunk, minimal logic in loop
            try:
                while retry:
                    retry = False
                    async for chunk in stream_service_response_with_retry(
                        instance_info.decoder.client,
                        api,
                        instance_info.decoder_body,
                        request_id=instance_info.request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                        decoder_dp_rank=(
                            instance_info.reservation.dp_rank
                            if instance_info.reservation
                            else None
                        ),
                    ):
                        if not released_kv and chunk:
                            proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                            released_kv = True
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            logger.debug(f"Skipping chunk: {chunk}")
                            yield chunk
                            continue
                        if not chunk_str:
                            continue
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: ") :]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            # if chunk is [done], skip it.
                            logger.debug(f"Skipping chunk: {chunk_str}")
                            yield chunk
                            continue
                        choices = chunk_json.get("choices", [])
                        if not choices:
                            yield chunk
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        content = delta.get("content") or message.get("content") or choice.get("text") or ""
                        generated_token += content

                        stop_reason = choice.get("stop_reason")
                        usage = chunk_json.get("usage", {})
                        completion_tokens = (
                            (completion_tokens + 1)
                            if stream_flag
                            else (completion_tokens + usage.get("completion_tokens"))
                        )
                        if stop_reason == "recomputed":
                            retry = True
                            retry_count += 1
                            if chat_flag:
                                messages[0]["content"] = origin_prompt + generated_token
                            else:
                                req_data["prompt"] = origin_prompt + generated_token
                            req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                            tmp_request_length = len(json.dumps(req_data).encode("utf-8"))
                            _release_decoder_reservation(instance_info)
                            instance_info = await _handle_select_instance(api, req_data, tmp_request_length)
                            released_kv = False
                            break
                        if retry_count > 0 and not stream_flag:
                            if chat_flag:
                                choice["message"]["content"] = generated_token
                            else:
                                choice["text"] = generated_token
                            chunk = json.dumps(chunk_json).encode("utf-8")
                        yield chunk
            except Exception as e:
                logger.error(
                    f"Error during streaming from decoder {instance_info.decoder.url}: {str(e)} "
                    f"the aborted request {instance_info.request_id} will be routing to the target "
                    "prefiller when new request is ready to dispatch to it"
                )
            finally:
                cleanup_current_request()

        # Determine the correct media type based on stream flag
        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        response = _CleanupStreamingResponse(
            generate_stream(),
            media_type=media_type,
            headers={"X-Request-Id": instance_info.request_id},
            cleanup=cleanup_current_request,
        )
        response_owns_cleanup = True
        return response
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        if not response_owns_cleanup:
            try:
                if instance_info is not None:
                    proxy_state.abort_prefiller_request(
                        instance_info.prefiller_idx,
                        instance_info.request_id,
                    )
                    proxy_state.release_prefiller_kv(
                        instance_info.prefiller_idx,
                        instance_info.prefiller_score,
                    )
                    _release_decoder_reservation(instance_info)
            finally:
                release_request_count()


async def _handle_adjust_instances(adjust_mode: str, request: Request):
    try:
        req_data = await request.json()
        instance_type = req_data.get("type", "")
        instances = req_data.get("instances", [])
        if isinstance(instances, str):
            instances = [instances]
        instances = trans_instances(instances)
        all_msg = f"{adjust_mode} {instance_type} instances: {[str(server) for server in instances]}."

        if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
            return {
                "error": f"Instance type {instance_type} is not supported. "
                f"Only support '{InstanceType.PREFILL}' and '{InstanceType.DECODE}'."
            }

        if adjust_mode == "add":
            added_nodes, waiting_nodes = await proxy_state.add_instances(instance_type, instances)
            if waiting_nodes:
                all_msg = (
                    f"{adjust_mode} {instance_type} instances: {added_nodes}. "
                    f"Instances {waiting_nodes} are waiting to be added."
                )
        elif adjust_mode == "remove":
            if instance_type == InstanceType.PREFILL:
                need_waiting = proxy_state.remove_prefillers(instances)
            else:
                need_waiting = proxy_state.remove_decoders(instances)

            if need_waiting:
                all_msg = f"Instances {instances} are isolated and waiting to be removed."
        return {
            "message": all_msg,
            "current_prefill_instances": [str(prefiller) for prefiller in proxy_state.prefillers],
            "current_decode_instances": [str(decoder) for decoder in proxy_state.decoders],
        }
    except Exception as e:
        logger.error(f"Failed to {adjust_mode} instances: {e}")
        raise e


def trans_instances(instances: list[str]) -> list[ServerState]:
    server_list = []
    for instance in instances:
        h, p = instance.split(":")
        server_list.append(ServerState(h, int(p)))
    return server_list


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders),
    }


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await _handle_adjust_instances("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await _handle_adjust_instances("remove", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
