# SPDX-License-Identifier: Apache-2.0
"""Attach actual P/D routes to non-streaming profiling responses on request."""

import asyncio
import json
import math
import os
import time
from urllib.parse import urlsplit, urlunsplit


def profile_route(instance_info, profiling_request=None):
    reservation = instance_info.reservation
    route = {
        "request_id": instance_info.request_id,
        "prefiller_url": instance_info.prefiller.url,
        "decoder_url": instance_info.decoder.url,
        "decoder_dp_rank": reservation.dp_rank if reservation is not None else None,
    }
    if profiling_request is not None and profiling_request.profile_start is not None:
        route["profile_start"] = profiling_request.profile_start
    return route


def profile_response(body, routes):
    response = json.loads(body)
    if isinstance(response, dict):
        response["vllm_profile_routes"] = routes
    return json.dumps(response, ensure_ascii=False).encode("utf-8")


async def profile_response_chunks(chunks, routes):
    body = b"".join([chunk async for chunk in chunks])
    try:
        yield profile_response(body, routes)
    except (ValueError, UnicodeError):
        yield body


def normalize_decoder_url(value):
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Profile target must be an HTTP(S) decoder URL")
    if parsed.query or parsed.fragment:
        raise ValueError("Profile target URL must contain only the decoder API address")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), path, "", ""))


def profile_capabilities(state):
    return {
        "version": 3,
        "target_decoder": True,
        "prefill_batch_barrier": True,
        "deferred_profile_start": True,
        "decoder_urls": [normalize_decoder_url(decoder.url) for decoder in state.decoders],
    }


class ProfileBatch:
    """Keep one client batch together until every P request has completed."""

    def __init__(self, registry, key, size, target, timeout, profile_decoders=(), authorization=None):
        self.registry = registry
        self.key = key
        self.size = size
        self.target = target
        self.timeout = timeout
        self.profile_decoders = profile_decoders
        self.authorization = authorization
        self.profile_start = None
        self.start_task = None
        self.joined = set()
        self.active = set()
        self.ready = set()
        self.event = asyncio.Event()
        self.error = None
        self.expiry = asyncio.get_running_loop().call_later(timeout, self.expire)

    def fail(self, message):
        if not self.event.is_set():
            self.error = message
            if self.start_task is not None and self.start_task is not asyncio.current_task():
                self.start_task.cancel()
            self.event.set()

    async def start_profilers(self):
        attempted = []
        headers = {"Authorization": self.authorization} if self.authorization else {}
        started_at = time.time_ns()
        try:
            for url, decoder in self.profile_decoders:
                attempted.append((url, decoder))
                response = await decoder.client.post(
                    url + "/start_profile", json={}, headers=headers, timeout=self.timeout,
                )
                response.raise_for_status()
            self.profile_start = {
                "requested_at_ns": started_at,
                "ready_at_ns": time.time_ns(),
                "decoder_urls": [url for url, _ in self.profile_decoders],
            }
            self.expiry.cancel()
            self.event.set()
        except BaseException as error:
            self.fail(f"Starting D profiling after prefill failed: {error}")
            for url, decoder in reversed(attempted):
                try:
                    response = await decoder.client.post(
                        url + "/stop_profile", json={}, headers=headers, timeout=self.timeout,
                    )
                    response.raise_for_status()
                except Exception as stop_error:
                    self.error += f". Cleanup for {url} failed: {stop_error}"

    def expire(self):
        self.fail("Timed out waiting for every prefill in the profiling batch")
        if self.registry.get(self.key) is self:
            self.registry.pop(self.key)

    def close(self, member):
        self.active.discard(member)
        self.fail("A profiling request ended before every prefill was ready")
        if len(self.joined) == self.size and not self.active:
            self.expiry.cancel()
            if self.registry.get(self.key) is self:
                self.registry.pop(self.key)


class ProfileRequest:
    """Resolve an allowed decoder and own a request's optional batch membership."""

    def __init__(self, request, state):
        headers = getattr(request, "headers", {})
        target = headers.get("x-vllm-profile-target")
        self.target_decoder_idx = None
        self.batch = None
        self.member = None
        if target:
            target = normalize_decoder_url(target)
            matches = [
                index for index, decoder in enumerate(state.decoders) if normalize_decoder_url(decoder.url) == target
            ]
            if len(matches) != 1:
                raise ValueError(f"Profile target {target!r} must match exactly one configured D instance")
            self.target_decoder_idx = matches[0]
        profile_decoders = ()
        start_header = headers.get("x-vllm-profile-start")
        if start_header:
            requested = json.loads(start_header)
            if not isinstance(requested, list) or not requested or any(not isinstance(url, str) for url in requested):
                raise ValueError("Profile start requires a nonempty list of decoder URLs")
            requested = tuple(normalize_decoder_url(url) for url in requested)
            available = {normalize_decoder_url(decoder.url): decoder for decoder in state.decoders}
            if len(set(requested)) != len(requested) or any(url not in available for url in requested):
                raise ValueError("Profile start URLs must be distinct configured D instances")
            if target is not None and target not in requested:
                raise ValueError("Profile start must include the target D instance")
            profile_decoders = tuple((url, available[url]) for url in requested)
        key = headers.get("x-vllm-profile-batch")
        if not key:
            if profile_decoders:
                raise ValueError("Deferred profiling requires a profiling batch ID")
            return
        size = int(headers.get("x-vllm-profile-batch-size", "0"))
        member = int(headers.get("x-vllm-profile-member", "-1"))
        timeout = float(headers.get("x-vllm-profile-batch-timeout", "600"))
        if size < 1 or not 0 <= member < size or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid profiling batch size, member index or timeout")
        registry = getattr(state, "_profile_batches", None)
        if registry is None:
            registry = state._profile_batches = {}
        batch = registry.get(key)
        authorization = headers.get("authorization")
        if not authorization and os.environ.get("OPENAI_API_KEY"):
            authorization = "Bearer " + os.environ["OPENAI_API_KEY"]
        if batch is None:
            batch = registry[key] = ProfileBatch(
                registry, key, size, target, timeout, profile_decoders, authorization,
            )
        if (batch.size, batch.target, batch.timeout, batch.profile_decoders, batch.authorization) != (
            size, target, timeout, profile_decoders, authorization,
        ):
            batch.fail("Profiling batch members disagree on size, target or timeout")
            raise ValueError("Profiling batch members must use the same size, target, timeout and start configuration")
        if batch.error:
            raise RuntimeError(batch.error)
        if member in batch.joined:
            batch.fail("Duplicate profiling batch member")
            raise ValueError("Duplicate profiling batch member")
        batch.joined.add(member)
        batch.active.add(member)
        self.member = member
        self.batch = batch

    async def prefill_ready(self):
        if self.batch is None:
            return
        batch = self.batch
        batch.ready.add(self.member)
        if len(batch.ready) == batch.size:
            if batch.profile_decoders:
                if batch.start_task is None:
                    batch.start_task = asyncio.create_task(batch.start_profilers())
            else:
                batch.expiry.cancel()
                batch.event.set()
        await batch.event.wait()
        if batch.error:
            raise RuntimeError(batch.error)

    @property
    def profile_start(self):
        return self.batch.profile_start if self.batch is not None else None

    def close(self):
        if self.batch is not None:
            self.batch.close(self.member)
            self.batch = None
