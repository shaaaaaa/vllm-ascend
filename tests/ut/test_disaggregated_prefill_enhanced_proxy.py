import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from examples.disaggregated_prefill_v1 import (
    load_balance_proxy_server_enhanced as proxy,
)
from examples.disaggregated_prefill_v1 import (
    load_balance_proxy_server_example as example_proxy,
)


def _remote_fill(dp_rank: int = 0) -> dict:
    return {
        "enabled": True,
        "destination_engine_id": "decoder-engine",
        "destination_engine_epoch": 7,
        "control_endpoint": "tcp://decoder:19001",
        "shared_cache_generation": 0,
        "destination_tp_size": 8,
        "destination_dp_size": 2,
        "global_te_push": True,
        "token_hash_algorithm": "builtin",
        "python_hash_seed": "0",
        "descriptor_verification_capability": "ab" * 32,
        "tp_rank": 0,
        "dp_rank": dp_rank,
    }


def _prime_remote_fill(state, dp_rank: int = 0):
    decoder = state.decoders[0]
    decoder.decoder_remote_fill = {
        dp_rank: {
            key: value
            for key, value in _remote_fill(dp_rank).items()
            if key not in {"enabled", "tp_rank", "dp_rank"}
        }
        | {"destination_dp_rank": dp_rank}
    }
    decoder.decoder_rank_active_tokens = {dp_rank: 0.0}
    decoder.decoder_placement_discovered_at = time.monotonic()
    return decoder


class TestEnhancedRemoteFillProxy(unittest.TestCase):
    def setUp(self) -> None:
        proxy.global_args = SimpleNamespace(
            max_retries=3,
            retry_delay=0.001,
            use_original_lb=True,
        )

    def tearDown(self) -> None:
        state = proxy.proxy_state
        if state is not None:
            async def close_clients() -> None:
                await asyncio.gather(
                    *(server.client.aclose() for server in state.prefillers + state.decoders)
                )

            asyncio.run(close_clients())
        proxy.proxy_state = None

    def test_remote_fill_only_placement_is_valid(self):
        placement = _remote_fill()
        payload = {
            "results": [{"dp_rank": 0, "segment": None, "remote_fill": placement}]
        }
        self.assertEqual(
            proxy._parse_decoder_remote_fill_response(payload)[0][
                "destination_engine_epoch"
            ],
            7,
        )

    def test_remote_fill_parser_matches_qualified_example(self):
        remote_fill = _remote_fill(1)
        payload = {
            "results": [
                None,
                {"dp_rank": 1, "segment": None, "remote_fill": remote_fill},
            ]
        }
        self.assertEqual(
            proxy._parse_decoder_remote_fill_response(payload),
            example_proxy._parse_decoder_remote_fill_response(payload),
        )

    def test_discovery_refresh_preserves_active_rank_load(self):
        state = proxy.ProxyState(
            [("prefiller", 8001)],
            [("decoder", 8002)],
            enable_remote_lmcache_store=True,
        )
        proxy.proxy_state = state
        decoder = _prime_remote_fill(state)
        decoder.decoder_rank_active_tokens[0] = 10.0
        decoder.decoder_placement_discovered_at = 0.0
        with patch.object(
            proxy,
            "_discover_decoder_remote_fill",
            AsyncMock(return_value=decoder.decoder_remote_fill),
        ):
            asyncio.run(state.ensure_decoder_remote_fill(decoder))
        self.assertEqual(decoder.decoder_rank_active_tokens, {0: 10.0})

    def test_remote_fill_selects_decoder_first_and_scrubs_capability(self):
        state = proxy.ProxyState(
            [("prefiller", 8001)],
            [("decoder", 8002)],
            enable_remote_lmcache_store=True,
        )
        proxy.proxy_state = state
        _prime_remote_fill(state)
        order = []
        original_select_decoder = state.select_decoder
        original_select_prefiller = state.select_prefiller

        def select_decoder(score):
            order.append("decoder")
            return original_select_decoder(score)

        def select_prefiller(score):
            order.append("prefiller")
            return original_select_prefiller(score)

        state.select_decoder = select_decoder
        state.select_prefiller = select_prefiller

        async def send(*args, **kwargs):
            order.append("send")
            handoff = kwargs["remote_fill_handoff"]
            return SimpleNamespace(
                content=b"{}",
                json=lambda: {
                    "kv_transfer_params": {
                        "remote_engine_id": "persistent",
                        "lmcache.remote_fill": {
                            "terminal": {
                                "outcome": "LOCAL_FULL",
                                "persistent_common_end": 1024,
                                "required_store_end": 1024,
                                "transfer_id": handoff["transfer_id"],
                            }
                        },
                    }
                }
            )

        req_data = {"messages": [], "stream": True}
        with patch.object(proxy, "send_request_to_service", side_effect=send):
            info = asyncio.run(
                proxy._handle_select_instance(
                    "/chat/completions",
                    req_data,
                    1024,
                )
            )

        self.assertEqual(order, ["decoder", "prefiller", "send"])
        self.assertEqual(info.reservation.dp_rank, 0)
        params = req_data["kv_transfer_params"]
        self.assertNotIn("lmcache.remote_fill", params)
        self.assertEqual(params["lmcache.remote_fill_result"]["outcome"], "LOCAL_FULL")
        proxy._release_decoder_reservation(info)
        state.release_prefiller_kv(info.prefiller_idx, info.prefiller_score)

        async def send_mismatched(*args, **kwargs):
            return SimpleNamespace(
                content=b"{}",
                json=lambda: {
                    "kv_transfer_params": {
                        "lmcache.remote_fill": {
                            "terminal": {
                                "outcome": "LOCAL_FULL",
                                "persistent_common_end": 1024,
                                "required_store_end": 1024,
                                "transfer_id": "wrong",
                            }
                        }
                    }
                },
            )

        with (
            patch.object(proxy, "send_request_to_service", side_effect=send_mismatched),
            self.assertRaises(RuntimeError),
        ):
            asyncio.run(
                proxy._handle_select_instance("/completions", {"prompt": "x"}, 1024)
            )
        self.assertEqual(state.prefillers[0].active_kv_cache, 0)
        self.assertEqual(state.decoders[0].active_tokens, 0)

    def test_disabled_mode_keeps_prefiller_then_decoder_order(self):
        state = proxy.ProxyState([("prefiller", 8001)], [("decoder", 8002)])
        proxy.proxy_state = state
        order = []
        original_select_decoder = state.select_decoder
        original_select_prefiller = state.select_prefiller

        def select_decoder(score):
            order.append("decoder")
            return original_select_decoder(score)

        def select_prefiller(score):
            order.append("prefiller")
            return original_select_prefiller(score)

        async def send(*args, **kwargs):
            order.append("send")
            self.assertIsNone(kwargs["remote_fill_handoff"])
            return SimpleNamespace(
                content=b"{}",
                json=lambda: {"kv_transfer_params": {"remote_engine_id": "persistent"}},
            )

        state.select_decoder = select_decoder
        state.select_prefiller = select_prefiller
        req_data = {"prompt": "x", "kv_transfer_params": {"forged": True}}
        with patch.object(proxy, "send_request_to_service", side_effect=send):
            info = asyncio.run(
                proxy._handle_select_instance(
                    "/completions",
                    req_data,
                    100,
                )
            )
        self.assertEqual(order, ["prefiller", "send", "decoder"])
        self.assertEqual(
            req_data["kv_transfer_params"],
            {"remote_engine_id": "persistent"},
        )
        proxy._release_decoder_reservation(info)
        state.release_prefiller_kv(info.prefiller_idx, info.prefiller_score)

    def test_cleanup_without_body_iteration_is_idempotent(self):
        state = proxy.ProxyState([("prefiller", 8001)], [("decoder", 8002)])
        proxy.proxy_state = state
        prefiller_score = state.calculate_prefill_scores(100)
        decoder_score = state.calculate_decode_scores(100)
        prefiller_idx = state.select_prefiller(prefiller_score)
        state.release_prefiller(prefiller_idx, prefiller_score)
        decoder_idx = state.select_decoder(decoder_score)
        info = proxy.InstanceInfo(
            "request",
            prefiller_idx,
            prefiller_score,
            state.prefillers[prefiller_idx],
            decoder_idx,
            decoder_score,
            state.decoders[decoder_idx],
        )
        request = SimpleNamespace(
            json=AsyncMock(return_value={"prompt": "hello", "stream": True}),
            body=AsyncMock(return_value=b"hello"),
        )
        with patch.object(proxy, "_handle_select_instance", AsyncMock(return_value=info)):
            response = asyncio.run(proxy._handle_completions("/completions", request))

        response._cleanup()
        response._cleanup()
        self.assertEqual(state.request_num, 0)
        self.assertEqual(state.prefillers[0].active_kv_cache, 0)
        self.assertEqual(state.decoders[0].active_tokens, 0)

    def test_selection_cancellation_releases_placement_reservation(self):
        state = proxy.ProxyState(
            [("prefiller", 8001)],
            [("decoder", 8002)],
            enable_remote_lmcache_store=True,
        )
        proxy.proxy_state = state
        _prime_remote_fill(state)
        entered = asyncio.Event()

        async def send(*args, **kwargs):
            entered.set()
            await asyncio.Future()

        async def cancel_selection() -> None:
            with patch.object(proxy, "send_request_to_service", side_effect=send):
                task = asyncio.create_task(
                    proxy._handle_select_instance("/completions", {"prompt": "x"}, 100)
                )
                await entered.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(cancel_selection())
        self.assertEqual(state.prefillers[0].active_tokens, 0)
        self.assertEqual(state.prefillers[0].active_kv_cache, 0)
        self.assertEqual(state.decoders[0].active_tokens, 0)
        self.assertEqual(state.decoders[0].decoder_rank_active_tokens[0], 0)

    def test_disconnect_waits_for_request_cleanup(self):
        cleaned = None

        async def run() -> None:
            nonlocal cleaned
            cleaned = asyncio.Event()
            entered = asyncio.Event()

            async def handler(*, request):
                try:
                    entered.set()
                    await asyncio.Future()
                finally:
                    cleaned.set()

            async def receive():
                await entered.wait()
                return {"type": "http.disconnect"}

            wrapped = proxy.with_cancellation(handler)
            await wrapped(request=SimpleNamespace(receive=receive))
            self.assertTrue(cleaned.is_set())

        asyncio.run(run())

    def test_route_cancellation_cancels_request_handler(self):
        async def run() -> None:
            entered = asyncio.Event()
            cleaned = asyncio.Event()

            async def handler(*, request):
                try:
                    entered.set()
                    await asyncio.Future()
                finally:
                    cleaned.set()

            async def receive():
                await asyncio.Future()

            wrapped = proxy.with_cancellation(handler)
            task = asyncio.create_task(
                wrapped(request=SimpleNamespace(receive=receive))
            )
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(cleaned.is_set())

        asyncio.run(run())

    def test_adjust_rejects_before_constructing_clients(self):
        proxy.proxy_state = proxy.ProxyState(
            [("prefiller", 8001)],
            [("decoder", 8002)],
            enable_remote_lmcache_store=True,
        )
        request = SimpleNamespace(
            json=AsyncMock(return_value={"type": "decode", "instances": ["new:9000"]})
        )
        with patch.object(proxy, "trans_instances") as transform:
            result = asyncio.run(proxy._handle_adjust_instances("add", request))
        transform.assert_not_called()
        self.assertIn("disabled", result["error"])

    def test_decoder_transport_sends_reserved_dp_rank(self):
        captured = {}

        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                yield b"data: {}\n\n"

        class Client:
            base_url = "http://decoder:8002/v1"

            def stream(self, method, endpoint, **kwargs):
                captured.update(method=method, endpoint=endpoint, **kwargs)
                return Response()

        async def collect() -> list[bytes]:
            return [
                chunk
                async for chunk in proxy.stream_service_response_with_retry(
                    Client(),
                    "/completions",
                    {},
                    "request",
                    decoder_dp_rank=3,
                )
            ]

        self.assertEqual(asyncio.run(collect()), [b"data: {}\n\n"])
        self.assertEqual(captured["headers"]["X-data-parallel-rank"], "3")
        self.assertEqual(captured["json"], {})

    def test_decoder_transport_does_not_retry_after_streaming_started(self):
        class Response:
            def __init__(self, attempt: int):
                self.attempt = attempt

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                yield f"data: {self.attempt}\n\n".encode()
                if self.attempt == 1:
                    raise httpx.ReadError(
                        "stream failed",
                        request=httpx.Request("POST", "http://decoder"),
                    )

        class Client:
            calls = 0

            def stream(self, *args, **kwargs):
                self.calls += 1
                return Response(self.calls)

        client = Client()

        async def collect() -> list[bytes]:
            return [
                chunk
                async for chunk in proxy.stream_service_response_with_retry(
                    client,
                    "/completions",
                    {},
                    "request",
                    base_delay=0,
                )
            ]

        self.assertEqual(asyncio.run(collect()), [b"data: 1\n\n"])
        self.assertEqual(client.calls, 1)

    def test_failed_recompute_does_not_release_old_attempt_twice(self):
        state = proxy.ProxyState([("prefiller", 8001)], [("decoder", 8002)])
        proxy.proxy_state = state
        prefiller_score = state.calculate_prefill_scores(100)
        decoder_score = state.calculate_decode_scores(100)
        prefiller_idx = state.select_prefiller(prefiller_score)
        state.release_prefiller(prefiller_idx, prefiller_score)
        decoder_idx = state.select_decoder(decoder_score)
        info = proxy.InstanceInfo(
            "old",
            prefiller_idx,
            prefiller_score,
            state.prefillers[prefiller_idx],
            decoder_idx,
            decoder_score,
            state.decoders[decoder_idx],
        )
        request = SimpleNamespace(
            json=AsyncMock(return_value={"prompt": "hello", "stream": True}),
            body=AsyncMock(return_value=b"hello"),
        )

        async def stream(*args, **kwargs):
            yield (
                b'data: {"choices":[{"delta":{},'
                b'"finish_reason":"recomputed"}]}\n\n'
            )

        select = AsyncMock(side_effect=[info, RuntimeError("replacement failed")])
        release_kv = MagicMock(wraps=state.release_prefiller_kv)
        release_decoder = MagicMock(wraps=state.release_decoder)
        state.release_prefiller_kv = release_kv
        state.release_decoder = release_decoder
        with (
            patch.object(proxy, "_handle_select_instance", select),
            patch.object(proxy, "stream_service_response_with_retry", stream),
        ):
            response = asyncio.run(proxy._handle_completions("/completions", request))

            async def consume() -> None:
                async for _ in response.body_iterator:
                    pass

            asyncio.run(consume())

        self.assertEqual(release_kv.call_count, 1)
        self.assertEqual(release_decoder.call_count, 1)
        self.assertEqual(state.request_num, 0)


if __name__ == "__main__":
    unittest.main()
