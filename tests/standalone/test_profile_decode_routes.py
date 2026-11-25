# SPDX-License-Identifier: Apache-2.0
"""Exercise proxy-owned routing metadata through four decoder instances on CPU."""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_enhanced as enhanced
from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as normal
from examples.disaggregated_prefill_v1.profile_routes import profile_response_chunks


@pytest.mark.parametrize("proxy", [normal, enhanced], ids=["normal", "enhanced"])
@pytest.mark.parametrize("decoder_index", range(4))
@pytest.mark.parametrize("report_route", [False, True])
def test_four_instance_route_comes_from_selected_decoder(monkeypatch, proxy, decoder_index, report_route):
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
        info_options = {"decoder_body": b"{}"} if proxy is normal else {}
        info = proxy.InstanceInfo(
            request_id="request-four-decoders",
            prefiller_idx=1,
            prefiller_score=1,
            prefiller=state.prefillers[1],
            decoder_idx=decoder_index,
            decoder_score=1,
            decoder=state.decoders[decoder_index],
            **info_options,
        )
        info.reservation.dp_rank = decoder_index
        body = {"id": "completion", "choices": [{"text": "answer", "finish_reason": "length"}]}
        response = httpx.Response(200, json=body, request=httpx.Request("POST", info.decoder.url + "/completions"))
        monkeypatch.setattr(proxy, "_handle_select_instance", AsyncMock(return_value=info))
        monkeypatch.setattr(info.decoder.client, "post", AsyncMock(return_value=response))

        async def chunks(*args, **kwargs):
            yield response.content[:12]
            yield response.content[12:]

        if proxy is normal:
            monkeypatch.setattr(proxy, "stream_service_response_with_retry", chunks)
        request = NS(
            json=AsyncMock(return_value={"prompt": [1, 2], "max_tokens": 128, "stream": False}),
            body=AsyncMock(return_value=b"{}"),
            headers={"x-vllm-profile-routes": "1"} if report_route else {},
        )
        try:
            result = await proxy._handle_completions("/completions", request)
            if proxy is normal:
                content = b"".join([chunk async for chunk in result.body_iterator])
                result._cleanup()
            else:
                content = result.body
            value = json.loads(content)
            assert value["choices"] == body["choices"]
            if report_route:
                assert value["vllm_profile_routes"] == [
                    {
                        "request_id": "request-four-decoders",
                        "prefiller_url": "http://P1:7910/v1",
                        "decoder_url": state.decoders[decoder_index].url,
                        "decoder_dp_rank": decoder_index,
                    }
                ]
            else:
                assert value == body
            assert state.request_num == 0
        finally:
            await asyncio.gather(*(node.client.aclose() for node in state.prefillers + state.decoders))

    asyncio.run(run())


def test_recompute_routes_are_included_after_response_generator_finishes():
    routes = [{"decoder_url": "http://D0:7920/v1"}]

    async def chunks():
        yield b'{"id":"request",'
        routes.append({"decoder_url": "http://D1:7921/v1"})
        yield b'"choices":[]}'

    async def run():
        return b"".join([chunk async for chunk in profile_response_chunks(chunks(), routes)])

    assert json.loads(asyncio.run(run()))["vllm_profile_routes"] == routes


def test_malformed_response_is_preserved_for_client_error_reporting():
    async def chunks():
        yield b"incomplete response"

    async def run():
        return b"".join([chunk async for chunk in profile_response_chunks(chunks(), [])])

    assert asyncio.run(run()) == b"incomplete response"
