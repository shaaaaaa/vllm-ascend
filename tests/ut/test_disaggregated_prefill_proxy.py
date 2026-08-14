import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from examples.disaggregated_prefill_v1 import (
    load_balance_proxy_server_example as proxy,
)


class TestProxyColdPerfLogging(unittest.TestCase):
    def test_log_event_uses_correlatable_completion_request_id(self):
        with (
            patch.dict(os.environ, {"LMCACHE_COLD_START_PERF": "1"}),
            patch.object(proxy.time, "perf_counter", return_value=12.345),
            patch.object(proxy.logger, "info") as log_info,
        ):
            proxy._log_proxy_cold_perf_event(
                "proxy_decoder_send_start",
                "request-uuid",
                endpoint="/completions",
                attempt=1,
            )

        args = log_info.call_args.args
        self.assertEqual(args[0], "[LMCACHE_COLD_PERF] %s")
        payload = json.loads(args[1])
        self.assertEqual(payload["event"], "proxy_decoder_send_start")
        self.assertEqual(payload["monotonic_ms"], 12345.0)
        self.assertEqual(payload["req_id"], "cmpl-request-uuid")
        self.assertEqual(payload["proxy_request_id"], "request-uuid")
        self.assertEqual(payload["attempt"], 1)

    def test_log_event_is_disabled_by_default(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(proxy.logger, "info") as log_info,
        ):
            proxy._log_proxy_cold_perf_event(
                "proxy_decoder_send_start",
                "request-uuid",
                endpoint="/completions",
            )

        log_info.assert_not_called()

    def test_instance_selection_logs_handoff_boundaries(self):
        request_id = "request-uuid"
        prefiller = SimpleNamespace(
            client=object(),
            url="http://prefiller:8001/v1",
        )
        decoder = SimpleNamespace(
            client=object(),
            url="http://decoder:8002/v1",
        )
        state = SimpleNamespace(
            calculate_prefill_scores=MagicMock(return_value=100.0),
            next_req_id=AsyncMock(return_value=request_id),
            select_prefiller=MagicMock(return_value=0),
            prefillers=[prefiller],
            release_prefiller=MagicMock(),
            calculate_decode_scores=MagicMock(return_value=10.0),
            select_decoder=MagicMock(return_value=0),
            decoders=[decoder],
        )
        response = SimpleNamespace(
            content=b'{"kv_transfer_params":{"source":"live"}}',
            json=MagicMock(
                return_value={
                    "kv_transfer_params": {"source": "live"},
                }
            ),
        )
        req_data = {}

        with (
            patch.object(proxy, "proxy_state", state),
            patch.object(
                proxy,
                "global_args",
                SimpleNamespace(max_retries=3, retry_delay=0.001),
                create=True,
            ),
            patch.object(
                proxy,
                "send_request_to_service",
                AsyncMock(return_value=response),
            ),
            patch.object(proxy, "_log_proxy_cold_perf_event") as log_event,
        ):
            result = asyncio.run(
                proxy._handle_select_instance(
                    "/completions",
                    req_data,
                    request_length=4096,
                )
            )

        self.assertIs(result.prefiller, prefiller)
        self.assertIs(result.decoder, decoder)
        self.assertEqual(
            log_event.call_args_list,
            [
                call(
                    "proxy_prefill_response_received",
                    request_id,
                    endpoint="/completions",
                    prefiller_url=prefiller.url,
                    response_bytes=len(response.content),
                    request_bytes=4096,
                ),
                call(
                    "proxy_decoder_dispatch_ready",
                    request_id,
                    endpoint="/completions",
                    decoder_url=decoder.url,
                    request_bytes=4096,
                    kv_transfer_param_keys=["source"],
                ),
            ],
        )
        self.assertEqual(
            req_data["kv_transfer_params"],
            {"source": "live"},
        )

    def test_decoder_stream_logs_actual_send_start(self):
        class Response:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            async def aiter_bytes():
                yield b"chunk"

        class StreamContext:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        client = SimpleNamespace(
            base_url="http://decoder:8002/v1/",
            stream=MagicMock(return_value=StreamContext()),
        )

        async def collect_chunks():
            return [
                chunk
                async for chunk in proxy.stream_service_response_with_retry(
                    client,
                    "/completions",
                    {},
                    "request-uuid",
                )
            ]

        with patch.object(
            proxy, "_log_proxy_cold_perf_event"
        ) as log_event:
            chunks = asyncio.run(collect_chunks())

        self.assertEqual(chunks, [b"chunk"])
        log_event.assert_called_once_with(
            "proxy_decoder_send_start",
            "request-uuid",
            endpoint="/completions",
            attempt=1,
            decoder_url="http://decoder:8002/v1/",
        )

    def test_streaming_response_logs_lazy_generator_entry(self):
        request = SimpleNamespace(
            json=AsyncMock(return_value={"prompt": "hello", "stream": True}),
            body=AsyncMock(return_value=b'{"prompt":"hello"}'),
        )
        prefiller = SimpleNamespace(url="http://prefiller:8001/v1")
        decoder = SimpleNamespace(
            client=object(),
            url="http://decoder:8002/v1",
        )
        instance_info = proxy.InstanceInfo(
            request_id="request-uuid",
            prefiller_idx=0,
            prefiller_score=100.0,
            prefiller=prefiller,
            decoder_idx=0,
            decoder_score=10.0,
            decoder=decoder,
        )
        state = SimpleNamespace(
            request_num=0,
            release_prefiller_kv=MagicMock(),
            release_decoder=MagicMock(),
        )

        async def decoder_chunks(*args, **kwargs):
            yield b'data: {"choices":[{"text":"x"}]}\n\n'

        async def consume_response():
            response = await proxy._handle_completions(
                "/completions",
                request,
            )
            return [chunk async for chunk in response.body_iterator]

        with (
            patch.object(proxy, "proxy_state", state),
            patch.object(
                proxy,
                "global_args",
                SimpleNamespace(max_retries=3, retry_delay=0.001),
                create=True,
            ),
            patch.object(
                proxy,
                "_handle_select_instance",
                AsyncMock(return_value=instance_info),
            ),
            patch.object(
                proxy,
                "stream_service_response_with_retry",
                decoder_chunks,
            ),
            patch.object(proxy, "_log_proxy_cold_perf_event") as log_event,
        ):
            chunks = asyncio.run(consume_response())

        self.assertEqual(
            chunks,
            [b'data: {"choices":[{"text":"x"}]}\n\n'],
        )
        log_event.assert_called_once_with(
            "proxy_decoder_generator_entry",
            "request-uuid",
            endpoint="/completions",
            decoder_url=decoder.url,
            request_bytes=len(b'{"prompt":"hello"}'),
        )


if __name__ == "__main__":
    unittest.main()
