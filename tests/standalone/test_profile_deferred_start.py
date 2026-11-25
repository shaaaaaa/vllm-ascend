# SPDX-License-Identifier: Apache-2.0
"""Start D profiling only after every P member reaches the handoff."""

import asyncio
import json
from types import SimpleNamespace as NS

import httpx
import pytest

from examples.disaggregated_prefill_v1.profile_routes import ProfileRequest, profile_capabilities


def make_request(index, count, urls, timeout=2):
    return NS(headers={
        "x-vllm-profile-target": urls[-1],
        "x-vllm-profile-start": json.dumps(urls),
        "x-vllm-profile-batch": "deferred-start",
        "x-vllm-profile-batch-size": str(count),
        "x-vllm-profile-member": str(index),
        "x-vllm-profile-batch-timeout": str(timeout),
        "authorization": "Bearer test-key",
    })


def response(url, status=200):
    return httpx.Response(status, request=httpx.Request("POST", url))


@pytest.mark.parametrize("members", [1, 16])
def test_all_prefills_finish_and_all_profilers_start_before_handoff(members):
    async def run():
        calls = []
        start_entered = asyncio.Event()
        allow_start = asyncio.Event()
        urls = [f"http://d{index}:7920" for index in range(4)]

        async def post(url, **kwargs):
            assert kwargs["headers"]["Authorization"] == "Bearer test-key"
            calls.append(url)
            if url == urls[0] + "/start_profile":
                start_entered.set()
                await allow_start.wait()
            return response(url)

        state = NS(decoders=[NS(url=url + "/v1", client=NS(post=post)) for url in urls])
        requests = [ProfileRequest(make_request(index, members, urls), state) for index in range(members)]
        waiting = [asyncio.create_task(item.prefill_ready()) for item in requests[:-1]]
        await asyncio.sleep(0)
        assert calls == []
        waiting.append(asyncio.create_task(requests[-1].prefill_ready()))
        await asyncio.wait_for(start_entered.wait(), 1)
        assert all(not item.done() for item in waiting)
        assert calls == [urls[0] + "/start_profile"]
        allow_start.set()
        await asyncio.wait_for(asyncio.gather(*waiting), 1)
        assert calls == [url + "/start_profile" for url in urls]
        assert all(item.profile_start["decoder_urls"] == urls for item in requests)
        assert requests[0].profile_start["ready_at_ns"] >= requests[0].profile_start["requested_at_ns"]
        for item in requests:
            item.close()
        assert state._profile_batches == {}
        assert profile_capabilities(state)["deferred_profile_start"] is True

    asyncio.run(run())


def test_start_failure_cleans_attempted_profilers_and_wakes_all_members():
    async def run():
        calls = []
        urls = [f"http://d{index}:7920" for index in range(3)]

        async def post(url, **kwargs):
            calls.append(url)
            return response(url, 500 if url == urls[1] + "/start_profile" else 200)

        state = NS(decoders=[NS(url=url, client=NS(post=post)) for url in urls])
        requests = [ProfileRequest(make_request(i, 2, urls), state) for i in range(2)]
        batch = requests[0].batch
        results = await asyncio.gather(*(item.prefill_ready() for item in requests), return_exceptions=True)
        await batch.start_task
        assert all(isinstance(result, RuntimeError) for result in results)
        assert calls == [
            urls[0] + "/start_profile", urls[1] + "/start_profile",
            urls[1] + "/stop_profile", urls[0] + "/stop_profile",
        ]
        for item in requests:
            item.close()
        assert state._profile_batches == {}

    asyncio.run(run())


@pytest.mark.parametrize("abort", ["cancel", "timeout"])
def test_aborting_during_start_does_not_release_decode(abort):
    async def run():
        calls = []
        entered = asyncio.Event()
        urls = ["http://d0:7920"]

        async def post(url, **kwargs):
            calls.append(url)
            if url.endswith("/start_profile"):
                entered.set()
                await asyncio.Event().wait()
            return response(url)

        state = NS(decoders=[NS(url=urls[0], client=NS(post=post))])
        requests = [ProfileRequest(make_request(i, 2, urls, timeout=.1), state) for i in range(2)]
        batch = requests[0].batch
        waiting = [asyncio.create_task(item.prefill_ready()) for item in requests]
        await asyncio.wait_for(entered.wait(), 1)
        if abort == "cancel":
            waiting[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting[0]
            requests[0].close()
        results = await asyncio.gather(*waiting, return_exceptions=True)
        await batch.start_task
        assert all(isinstance(result, BaseException) for result in results)
        assert calls == [urls[0] + "/start_profile", urls[0] + "/stop_profile"]
        for item in requests:
            item.close()
        assert state._profile_batches == {}

    asyncio.run(run())


@pytest.mark.parametrize("urls", [["http://outside:7920"], ["http://d0:7920", "http://d0:7920"]])
def test_start_targets_are_configured_unique_decoders(urls):
    state = NS(decoders=[NS(url="http://d0:7920")])
    req = make_request(0, 1, urls)
    req.headers.pop("x-vllm-profile-target")
    with pytest.raises(ValueError, match="distinct configured"):
        ProfileRequest(req, state)
    assert not hasattr(state, "_profile_batches")
