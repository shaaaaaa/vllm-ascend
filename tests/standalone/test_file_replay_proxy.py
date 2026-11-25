# SPDX-License-Identifier: Apache-2.0
"""Exercise the real proxy dispatch functions for all two-P/two-D routes."""

import asyncio
import base64
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from examples.disaggregated_prefill_v1 import load_balance_proxy_server_enhanced as enhanced
from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as normal
from examples.disaggregated_prefill_v1.vllm_token_counter import VLLMTokenCounter


@pytest.mark.parametrize("proxy", [normal, enhanced], ids=["normal", "enhanced"])
@pytest.mark.parametrize("p_rank,d_rank", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("file_mode", [False, True])
@pytest.mark.parametrize("prompt", ["original prompt", [151331, 151333, 42, 43]])
@pytest.mark.parametrize("chat", [False, True])
def test_source_survives_every_proxy_route(monkeypatch, proxy, p_rank, d_rank, file_mode, prompt, chat):
    tag = base64.urlsafe_b64encode(json.dumps(dict(url="http://7.150.3.60:18080/random", role="PD")).encode()).decode()
    transfer = {"remote_host": "caller-must-not-override-transport"}
    if file_mode:
        transfer["lmcache.tag.file_replay"] = tag
    request = dict(
        prompt=prompt,
        add_special_tokens=False,
        max_tokens=128,
        temperature=0,
        cache_salt="same-salt-for-p-and-d",
        kv_transfer_params=transfer,
    )
    messages = [
        dict(role="user", content="original prompt"),
        dict(role="assistant", content="", reasoning="unfinished thought"),
        dict(role="user", content="next question"),
    ]
    if chat:
        request.pop("prompt")
        request.pop("add_special_tokens")
        request.update(messages=messages, chat_template_kwargs=dict(clear_thinking=False))
    sent = []

    async def post(endpoint, **kwargs):
        sent.append(kwargs.get("json") or json.loads(kwargs["content"]))
        # P need not echo the file tag: the proxy must retain the caller's source.
        return NS(
            raise_for_status=lambda: None,
            json=lambda: {"kv_transfer_params": {"remote_host": "actual-p", "source": "ordinary-lmcache"}},
        )

    p_nodes = [NS(url=f"http://p{i}:7910/v1", client=NS(post=AsyncMock(side_effect=post))) for i in range(2)]
    d_nodes = [NS(url=f"http://d{i}:7920/v1", decoder_mooncake_segments={}) for i in range(2)]
    state = NS(
        enable_remote_lmcache_store=False,
        prefillers=p_nodes,
        decoders=d_nodes,
        acquire_aborted_prefiller_requests=lambda _: set(),
        calculate_prefill_scores=lambda _: 1.0,
        calculate_decode_scores=lambda _: 1.0,
        next_req_id=AsyncMock(return_value="test-request"),
        select_prefiller=lambda _: p_rank,
        select_decoder=lambda _: d_rank,
        assign_decoder_rank=lambda _: None,
        release_prefiller=MagicMock(),
    )
    monkeypatch.setattr(proxy, "proxy_state", state)
    monkeypatch.setattr(
        proxy, "global_args", NS(max_retries=1, retry_delay=0, backend_request_timeout=5), raising=False
    )
    api = "/chat/completions" if chat else "/completions"
    result = asyncio.run(proxy._handle_select_instance(api, request, 100))
    assert (result.prefiller_idx, result.decoder_idx) == (p_rank, d_rank)
    assert len(sent) == 1
    for body in (sent[0], request):
        if chat:
            assert body["messages"] == messages
            assert body["chat_template_kwargs"] == dict(clear_thinking=False)
            assert "prompt" not in body
        else:
            assert body["prompt"] == prompt
            assert body["add_special_tokens"] is False
        assert body["cache_salt"] == "same-salt-for-p-and-d"
        assert body["kv_transfer_params"].get("lmcache.tag.file_replay") == (tag if file_mode else None)
        assert body["kv_transfer_params"].get("remote_host") != "caller-must-not-override-transport"
    assert sent[0]["max_tokens"] == 1 and request["max_tokens"] == 128


def test_native_prompt_counter_uses_ids_without_reencoding():
    for cls, method in (
        (VLLMTokenCounter, "count_prompt_tokens"),
        (enhanced.TokenizerAnalyzer, "_count_prompt_tokens"),
    ):
        counter = cls.__new__(cls)
        counter.tokenizer = NS(encode=MagicMock(side_effect=AssertionError("must not re-encode token IDs")))
        assert getattr(counter, method)([151331, 151333, 42, 43]) == 4


def test_native_prompt_placement_counts_ids_with_analysis_disabled(monkeypatch):
    async def run():
        state = enhanced.ProxyState(
            [("prefiller", 8001)],
            [("decoder", 8002)],
            enable_remote_lmcache_store=True,
            enable_prefix_affinity_routing=True,
        )
        monkeypatch.setattr(enhanced, "proxy_state", state)
        monkeypatch.setattr(enhanced, "global_args", NS(use_original_lb=False))
        assert state.vllm_token_counter is None and state.tokenizer_analyzer is None
        prompt = [42] * 500_000
        payload = dict(prompt=prompt, stream=True, add_special_tokens=False)
        request = NS(json=AsyncMock(return_value=payload), body=AsyncMock(return_value=json.dumps(payload).encode()))
        select = AsyncMock(side_effect=RuntimeError("stop after observing placement"))
        monkeypatch.setattr(enhanced, "_handle_select_instance", select)
        try:
            await enhanced._handle_completions("/completions", request)
            assert select.await_args.args[2] == 500_000
            assert select.await_args.args[1]["prompt"] == prompt
            anchors = select.await_args.kwargs["prefix_anchors"]
            assert anchors and all(anchor.token_end > 1 for anchor in anchors)
        finally:
            await asyncio.gather(*(node.client.aclose() for node in state.prefillers + state.decoders))

    asyncio.run(run())
