# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for targeted P/D routing, decode overlap and failure cleanup."""

import asyncio
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_enhanced as enhanced
from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as normal
from examples.disaggregated_prefill_v1.profile_routes import ProfileRequest, profile_capabilities


def batch_request(index, size, target="http://D1:7921", timeout=3, batch="same-batch"):
    return NS(
        headers={
            "x-vllm-profile-target": target,
            "x-vllm-profile-batch": batch,
            "x-vllm-profile-batch-size": str(size),
            "x-vllm-profile-member": str(index),
            "x-vllm-profile-batch-timeout": str(timeout),
            "x-vllm-profile-routes": "1",
        },
        json=AsyncMock(return_value={"prompt": [index, 10], "max_tokens": 128, "stream": False}),
        body=AsyncMock(return_value=b"{}"),
    )


def test_barrier_releases_only_after_all_prefills():
    async def run():
        state = NS(decoders=[NS(url="http://D1:7921/v1")])
        requests = [ProfileRequest(batch_request(index, 16), state) for index in range(16)]
        waiting = [asyncio.create_task(request.prefill_ready()) for request in requests[:-1]]
        await asyncio.sleep(0)
        assert all(not task.done() for task in waiting)
        await requests[-1].prefill_ready()
        await asyncio.gather(*waiting)
        for request in requests:
            request.close()
        assert state._profile_batches == {}

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["close", "timeout", "cancel"])
def test_barrier_failure_wakes_remaining_requests(failure):
    async def run():
        state = NS(decoders=[NS(url="http://D1:7921/v1")])
        first = ProfileRequest(batch_request(0, 2, timeout=0.05), state)
        second = ProfileRequest(batch_request(1, 2, timeout=0.05), state)
        waiting = asyncio.create_task(first.prefill_ready())
        await asyncio.sleep(0)
        if failure == "close":
            second.close()
        elif failure == "cancel":
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            first.close()
            waiting = asyncio.create_task(second.prefill_ready())
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(waiting, 1)
        first.close()
        second.close()
        await asyncio.sleep(0.06)
        assert state._profile_batches == {}

    asyncio.run(run())


def test_invalid_target_leaves_batch_registry_unmodified():
    state = NS(decoders=[NS(url="http://D1:7921/v1")])
    with pytest.raises(ValueError, match="exactly one"):
        ProfileRequest(batch_request(0, 1, target="http://unexpected:7921"), state)
    assert not hasattr(state, "_profile_batches")


@pytest.mark.parametrize("mismatch", ["duplicate", "size"])
def test_inconsistent_batch_wakes_waiting_member(mismatch):
    async def run():
        state = NS(decoders=[NS(url="http://D1:7921/v1")])
        first = ProfileRequest(batch_request(0, 2, timeout=0.05), state)
        waiting = asyncio.create_task(first.prefill_ready())
        await asyncio.sleep(0)
        incoming = batch_request(0 if mismatch == "duplicate" else 1, 3 if mismatch == "size" else 2, timeout=0.05)
        with pytest.raises(ValueError):
            ProfileRequest(incoming, state)
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(waiting, 1)
        first.close()
        await asyncio.sleep(0.06)
        assert state._profile_batches == {}

    asyncio.run(run())


@pytest.mark.parametrize("proxy", [normal, enhanced], ids=["normal", "enhanced"])
@pytest.mark.parametrize("fail_prefill", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_sixteen_prefills_target_one_decoder_and_release_together(monkeypatch, proxy, fail_prefill, deferred):
    async def run():
        monkeypatch.setattr(normal, "NodeListener", lambda _: None)
        state = proxy.ProxyState(
            [("P0", 7910), ("P1", 7910)],
            [("D0", 7920), ("D0", 7921), ("D1", 7920), ("D1", 7921)],
        )
        monkeypatch.setattr(proxy, "proxy_state", state)
        monkeypatch.setattr(
            proxy,
            "global_args",
            NS(max_retries=1, retry_delay=0, backend_request_timeout=5, use_original_lb=True),
            raising=False,
        )
        prefills = []
        decodes = []
        profile_starts = []
        started = 0
        everyone_at_decode = asyncio.Event()

        async def prefill(client, index, api, payload, request_id, **kwargs):
            await asyncio.sleep(0)
            if fail_prefill and payload["prompt"][0] == 5:
                raise RuntimeError("prefill failed")
            prefills.append(index)
            return httpx.Response(
                200,
                json={"kv_transfer_params": {"remote_host": f"P{index}", "source": "normal"}},
                request=httpx.Request("POST", "http://P/prefill"),
            )

        async def decode(index, payload):
            nonlocal started
            assert len(prefills) == 16
            if deferred:
                assert profile_starts == [3]
            assert index == 3
            assert payload["kv_transfer_params"]["source"] == "normal"
            decodes.append(index)
            started += 1
            if started == 16:
                everyone_at_decode.set()
            await asyncio.wait_for(everyone_at_decode.wait(), 2)
            return {"choices": [{"text": "answer", "finish_reason": "length"}], "usage": {"completion_tokens": 128}}

        async def stream(client, api, body, **kwargs):
            index = next(index for index, decoder in enumerate(state.decoders) if decoder.client is client)
            yield json.dumps(await decode(index, json.loads(body))).encode()

        def make_post(index):
            async def post(api, json, **kwargs):
                if api.endswith("/start_profile"):
                    assert len(prefills) == 16
                    profile_starts.append(index)
                    return httpx.Response(200, request=httpx.Request("POST", api))
                return httpx.Response(
                    200,
                    json=await decode(index, json),
                    request=httpx.Request("POST", "http://D/decode"),
                )

            return post

        monkeypatch.setattr(proxy, "send_request_to_service", prefill)
        for index, decoder in enumerate(state.decoders):
            decoder.decoder_mooncake_segments = {}
            monkeypatch.setattr(decoder.client, "post", AsyncMock(side_effect=make_post(index)))
        if proxy is normal:
            monkeypatch.setattr(proxy, "stream_service_response_with_retry", stream)

        async def invoke(index):
            request = batch_request(index, 16)
            if deferred:
                request.headers["x-vllm-profile-start"] = json.dumps(["http://D1:7921"])
            try:
                response = await proxy._handle_completions("/completions", request)
            except RuntimeError:
                if fail_prefill:
                    return None
                raise
            if proxy is normal:
                body = b"".join([chunk async for chunk in response.body_iterator])
                response._cleanup()
            else:
                body = response.body
            return json.loads(body)

        try:
            assert profile_capabilities(state)["decoder_urls"][3] == "http://d1:7921"
            results = await asyncio.wait_for(asyncio.gather(*(invoke(index) for index in range(16))), 5)
            if fail_prefill:
                assert decodes == []
                assert profile_starts == []
                assert all(result is None or "error" in result for result in results)
            else:
                assert decodes == [3] * 16
                assert set(prefills) == {0, 1}
                assert all(result["vllm_profile_routes"][0]["decoder_url"] == "http://D1:7921/v1" for result in results)
                assert profile_starts == ([3] if deferred else [])
            assert state.request_num == 0
            assert all(decoder.active_tokens == 0 for decoder in state.decoders)
            assert all(prefiller.active_tokens == pytest.approx(0) for prefiller in state.prefillers)
            assert all(prefiller.active_kv_cache == pytest.approx(0) for prefiller in state.prefillers)
            assert state._profile_batches == {}
        finally:
            await asyncio.gather(*(node.client.aclose() for node in state.prefillers + state.decoders))

    asyncio.run(run())


@pytest.mark.parametrize("proxy", [normal, enhanced], ids=["normal", "enhanced"])
@pytest.mark.parametrize("remote_fill", [False, True])
def test_pinning_uses_real_selection_and_original_kv_handoff(monkeypatch, proxy, remote_fill):
    async def run():
        monkeypatch.setattr(normal, "NodeListener", lambda _: None)
        state = proxy.ProxyState(
            [("P0", 7910), ("P1", 7910)],
            [("D0", 7920), ("D0", 7921), ("D1", 7920), ("D1", 7921)],
            enable_remote_lmcache_store=remote_fill,
        )
        monkeypatch.setattr(proxy, "proxy_state", state)
        monkeypatch.setattr(proxy, "global_args", NS(max_retries=1, retry_delay=0, use_original_lb=True), raising=False)
        handoff = {
            "destination_engine_id": "target-engine",
            "destination_engine_epoch": 7,
            "destination_dp_rank": 0,
            "api_dp_rank": 0,
            "mooncake_preferred_segment": "D1:12345",
        }
        for decoder in state.decoders:
            decoder.decoder_mooncake_segments = {0: "D1:12345"} if remote_fill else {}
            decoder.decoder_remote_fill = {0: handoff}
            decoder.decoder_rank_active_tokens = {0: 0}
        monkeypatch.setattr(state, "ensure_decoder_placements", AsyncMock(), raising=False)
        monkeypatch.setattr(state, "decoder_placements_ready", lambda: True, raising=False)
        monkeypatch.setattr(state, "ensure_decoder_mooncake_segments", AsyncMock(), raising=False)
        if proxy is enhanced:
            # A profiling target stays fixed even outside the normal count-slack window.
            state.decoders[3].active_requests = 8
            monkeypatch.setattr(
                state, "resolve_prefix_affinity", MagicMock(side_effect=AssertionError("target is fixed"))
            )

        captured = []

        async def prefill(client, index, api, payload, request_id, **kwargs):
            captured.append(copy.deepcopy(kwargs.get("remote_fill_handoff")))
            params = {"remote_host": "P0", "source": "original-kv"}
            if remote_fill:
                params["lmcache.remote_fill"] = {
                    "terminal": {
                        "outcome": "LOCAL_FULL",
                        "persistent_common_end": 50000,
                        "required_store_end": 50000,
                        "transfer_id": captured[-1]["transfer_id"],
                    },
                }
            return httpx.Response(
                200,
                json={"kv_transfer_params": params},
                request=httpx.Request("POST", "http://P/prefill"),
            )

        monkeypatch.setattr(proxy, "send_request_to_service", prefill)
        body = {"prompt": [1, 2], "max_tokens": 128}
        try:
            info = await proxy._handle_select_instance("/completions", body, 50000, target_decoder_idx=3)
            assert info.decoder is state.decoders[3]
            if proxy is enhanced:
                assert info.decoder.active_requests == 9
            assert body["kv_transfer_params"]["source"] == "original-kv"
            if remote_fill:
                assert captured[0]["destination_engine_id"] == "target-engine"
                assert info.reservation.dp_rank == 0
            proxy._release_decoder_reservation(info)
            state.release_prefiller_kv(info.prefiller_idx, info.prefiller_score)
            if proxy is enhanced:
                assert info.decoder.active_requests == 8
            assert all(decoder.active_tokens == 0 for decoder in state.decoders)
        finally:
            await asyncio.gather(*(node.client.aclose() for node in state.prefillers + state.decoders))

    asyncio.run(run())
