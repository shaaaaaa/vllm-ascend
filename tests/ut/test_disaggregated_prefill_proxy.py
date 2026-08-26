import asyncio
import json
import os
import unittest
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from examples.disaggregated_prefill_v1 import (
    load_balance_proxy_server_example as proxy,
)


class TestProxyColdPerfLogging(unittest.TestCase):
    @staticmethod
    def _remote_fill_placement() -> dict:
        return {
            "enabled": True,
            "destination_engine_id": "decoder-engine-0",
            "destination_engine_epoch": 7,
            "control_endpoint": "tcp://decoder:19001",
            "shared_cache_generation": 3,
            "destination_tp_size": 8,
            "destination_dp_size": 2,
            "global_te_push": True,
            "token_hash_algorithm": "builtin",
            "python_hash_seed": "0",
            "descriptor_verification_capability": "ab" * 32,
            "tp_rank": 0,
            "dp_rank": 1,
        }

    def test_decoder_placement_response_uses_tp0_results(self):
        self.assertEqual(
            proxy._parse_decoder_placement_response(
                {
                    "results": [
                        {"dp_rank": 1, "segment": "decoder-b:2001"},
                        None,
                        {"dp_rank": 0, "segment": "decoder-a:2000"},
                    ]
                }
            ),
            {0: "decoder-a:2000", 1: "decoder-b:2001"},
        )

        invalid_payloads = (
            {},
            {"results": []},
            {"results": [{"dp_rank": -1, "segment": "host:1"}]},
            {"results": [{"dp_rank": 0, "segment": ""}]},
            {"results": ["not-a-dictionary"]},
            {
                "results": [
                    {"dp_rank": 0, "segment": "host-a:1"},
                    {"dp_rank": 0, "segment": "host-b:1"},
                ]
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                proxy._parse_decoder_placement_response(payload)

    def test_decoder_placement_discovery_calls_existing_collective_rpc(self):
        response = SimpleNamespace(
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={
                    "results": [
                        None,
                        {"dp_rank": 3, "segment": "decoder-a:12345"},
                    ]
                }
            ),
        )
        server = SimpleNamespace(
            client=SimpleNamespace(post=AsyncMock(return_value=response)),
            url="http://decoder:8002/v1",
        )

        result = asyncio.run(
            proxy._discover_decoder_mooncake_segments(
                server,
                timeout_seconds=0.75,
            )
        )

        self.assertEqual(result, {3: "decoder-a:12345"})
        server.client.post.assert_awaited_once_with(
            "http://decoder:8002/collective_rpc",
            json={"method": "get_mooncake_placement_info"},
            headers=proxy._service_auth_headers(),
            timeout=0.75,
        )
        response.raise_for_status.assert_called_once_with()

    def test_decoder_remote_fill_placement_can_replace_segment_hint(self):
        remote_fill = self._remote_fill_placement()
        payload = {
            "results": [
                {
                    "dp_rank": 1,
                    "segment": None,
                    "remote_fill": remote_fill,
                }
            ]
        }

        self.assertEqual(
            proxy._parse_decoder_placement_response(
                payload,
                allow_remote_fill_only=True,
            ),
            {1: None},
        )
        self.assertEqual(
            proxy._parse_decoder_remote_fill_response(payload),
            {
                1: {
                    key: value
                    for key, value in remote_fill.items()
                    if key not in {"enabled", "tp_rank", "dp_rank"}
                }
                | {"destination_dp_rank": 1}
            },
        )

        for field, invalid in (
            ("destination_tp_size", 0),
            ("destination_dp_size", True),
            ("destination_dp_size", 1),
            ("global_te_push", "yes"),
            ("descriptor_verification_capability", "bad"),
            ("tp_rank", 1),
            ("dp_rank", 0),
        ):
            broken = dict(remote_fill)
            broken[field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                proxy._parse_decoder_remote_fill_response(
                    {
                        "results": [
                            {
                                "dp_rank": 1,
                                "segment": None,
                                "remote_fill": broken,
                            }
                        ]
                    }
                )

    def test_h0_off_does_not_advertise_remote_fill(self):
        remote_fill = self._remote_fill_placement() | {"global_te_push": False}
        payload = {
            "results": [
                {"dp_rank": 1, "segment": "decoder:1234", "remote_fill": remote_fill}
            ]
        }

        self.assertEqual(proxy._parse_decoder_remote_fill_response(payload), {})
        self.assertEqual(
            proxy._parse_decoder_placement_response(payload),
            {1: "decoder:1234"},
        )

    def test_static_segment_mapping_is_kept_when_remote_fill_is_discovered(self):
        decoder = SimpleNamespace(
            client=object(),
            url="http://decoder:8002/v1",
            decoder_mooncake_segments={1: "static-decoder:1234"},
            decoder_remote_fill={},
            decoder_remote_fill_discovered=False,
            decoder_rank_active_tokens={1: 0.0},
            decoder_placement_task=None,
            static_decoder_mooncake_segments={1: "static-decoder:1234"},
        )
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.decoders = [decoder]
        state.enable_remote_lmcache_store = True
        state.decoder_placement_discovery_timeout_seconds = 0.5
        discovered_remote_fill = {
            key: value
            for key, value in self._remote_fill_placement().items()
            if key not in {"enabled", "tp_rank", "dp_rank"}
        } | {"destination_dp_rank": 1}

        async def discover(server, **_kwargs):
            server.decoder_remote_fill = {1: discovered_remote_fill}
            return {1: None}

        with (
            patch.object(
                proxy,
                "_discover_decoder_mooncake_segments",
                AsyncMock(side_effect=discover),
            ) as discovery,
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            asyncio.run(
                state.ensure_decoder_mooncake_segments(
                    decoder, "request-1", "/completions"
                )
            )

        discovery.assert_awaited_once_with(
            decoder,
            enable_remote_fill=True,
            timeout_seconds=0.5,
        )
        self.assertEqual(
            decoder.decoder_mooncake_segments,
            {1: "static-decoder:1234"},
        )
        self.assertEqual(decoder.decoder_remote_fill, {1: discovered_remote_fill})
        self.assertTrue(decoder.decoder_remote_fill_discovered)

    def test_static_rank_mismatch_disables_remote_fill(self):
        decoder = SimpleNamespace(
            client=object(),
            url="http://decoder:8002/v1",
            decoder_mooncake_segments={0: "static-decoder:1234"},
            static_decoder_mooncake_segments={0: "static-decoder:1234"},
            decoder_remote_fill={},
            decoder_remote_fill_discovered=False,
            decoder_rank_active_tokens={0: 0.0},
            decoder_placement_task=None,
        )
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.enable_remote_lmcache_store = True
        state.decoder_placement_discovery_timeout_seconds = 0.5

        async def discover(server, **_kwargs):
            server.decoder_remote_fill = {1: {"control_endpoint": "tcp://d:1"}}
            return {1: None}

        with (
            patch.object(
                proxy,
                "_discover_decoder_mooncake_segments",
                AsyncMock(side_effect=discover),
            ),
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            asyncio.run(
                state.ensure_decoder_mooncake_segments(
                    decoder, "request-1", "/completions"
                )
            )

        self.assertEqual(decoder.decoder_remote_fill, {})
        self.assertEqual(
            decoder.decoder_mooncake_segments,
            {0: "static-decoder:1234"},
        )

    def test_successful_negative_discovery_is_retried_after_short_ttl(self):
        decoder = SimpleNamespace(
            url="http://decoder:8002/v1",
            decoder_mooncake_segments={0: "static-decoder:1234"},
            static_decoder_mooncake_segments={0: "static-decoder:1234"},
            decoder_remote_fill={},
            decoder_remote_fill_discovered=False,
            decoder_rank_active_tokens={0: 0.0},
            decoder_placement_task=None,
        )
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.enable_remote_lmcache_store = True
        state.decoder_placement_discovery_timeout_seconds = 0.5
        state.decoder_placement_positive_ttl_seconds = 30.0
        state.decoder_placement_negative_ttl_seconds = 3.0
        remote_fill = {"control_endpoint": "tcp://decoder:19000"}

        async def discover(server, **_kwargs):
            if discover.call_count == 2:
                server.decoder_remote_fill = {0: remote_fill}
            return {0: "decoder:1234"}

        discover = AsyncMock(side_effect=discover)
        with (
            patch.object(proxy, "_discover_decoder_mooncake_segments", discover),
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            asyncio.run(
                state.ensure_decoder_mooncake_segments(
                    decoder, "request-1", "/completions"
                )
            )
            asyncio.run(
                state.ensure_decoder_mooncake_segments(
                    decoder, "request-2", "/completions"
                )
            )
            self.assertEqual(discover.await_count, 1)
            decoder.decoder_placement_discovered_at -= 4.0
            asyncio.run(
                state.ensure_decoder_mooncake_segments(
                    decoder, "request-3", "/completions"
                )
            )

        self.assertEqual(discover.await_count, 2)
        self.assertEqual(decoder.decoder_remote_fill, {0: remote_fill})

    def test_reset_decoder_placement_preserves_only_static_mapping(self):
        task = MagicMock()
        task.done.return_value = False
        decoder = SimpleNamespace(
            decoder_placement_task=task,
            decoder_remote_fill={0: {"destination_engine_epoch": 7}},
            decoder_remote_fill_discovered=True,
            decoder_placement_discovered_at=10.0,
            decoder_placement_last_attempt_at=9.0,
            decoder_mooncake_segments={0: "dynamic:1"},
            static_decoder_mooncake_segments={1: "static:2"},
            decoder_rank_active_tokens={0: 123.0},
        )

        proxy.ProxyState.reset_decoder_placement(decoder)

        task.cancel.assert_called_once_with()
        self.assertIsNone(decoder.decoder_placement_task)
        self.assertEqual(decoder.decoder_remote_fill, {})
        self.assertFalse(decoder.decoder_remote_fill_discovered)
        self.assertEqual(decoder.decoder_mooncake_segments, {1: "static:2"})
        self.assertEqual(decoder.decoder_rank_active_tokens, {1: 0.0})

    def test_decoder_placement_discovery_is_cached(self):
        decoder = SimpleNamespace(
            url="http://decoder:8002/v1",
            decoder_mooncake_segments=None,
            decoder_remote_fill={},
            decoder_remote_fill_discovered=False,
            decoder_rank_active_tokens={},
            decoder_placement_task=None,
        )
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.decoders = [decoder]
        state.enable_remote_lmcache_store = True
        state.decoder_placement_discovery_timeout_seconds = 0.5

        with (
            patch.object(
                proxy,
                "_discover_decoder_mooncake_segments",
                AsyncMock(return_value={3: "decoder-a:12345"}),
            ) as discover,
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            asyncio.run(
                state.ensure_decoder_mooncake_segments(
                    decoder, "request-1", "/completions"
                )
            )
            asyncio.run(
                state.ensure_decoder_mooncake_segments(
                    decoder, "request-2", "/completions"
                )
            )

        discover.assert_awaited_once_with(
            decoder,
            enable_remote_fill=True,
            timeout_seconds=0.5,
        )
        self.assertEqual(
            decoder.decoder_mooncake_segments,
            {3: "decoder-a:12345"},
        )
        self.assertEqual(decoder.decoder_rank_active_tokens, {3: 0.0})

    def test_decoder_placement_discovery_failure_retries(self):
        decoder = SimpleNamespace(
            url="http://decoder:8002/v1",
            decoder_mooncake_segments=None,
            decoder_remote_fill={},
            decoder_remote_fill_discovered=False,
            decoder_rank_active_tokens={},
            decoder_placement_task=None,
        )
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.decoders = [decoder]
        state.enable_remote_lmcache_store = True
        state.decoder_placement_discovery_timeout_seconds = 0.5

        with (
            patch.object(
                proxy,
                "_discover_decoder_mooncake_segments",
                AsyncMock(
                    side_effect=[
                        proxy.httpx.RequestError("unavailable"),
                        {3: "decoder-a:12345"},
                    ]
                ),
            ) as discover,
            patch.object(proxy, "_log_proxy_cold_perf_event") as log_event,
        ):
            asyncio.run(
                state.ensure_decoder_mooncake_segments(
                    decoder, "request-1", "/completions"
                )
            )
            asyncio.run(
                state.ensure_decoder_mooncake_segments(
                    decoder, "request-2", "/completions"
                )
            )

        self.assertEqual(discover.await_count, 2)
        self.assertEqual(decoder.decoder_mooncake_segments, {3: "decoder-a:12345"})
        self.assertTrue(decoder.decoder_remote_fill_discovered)
        self.assertIn(
            call(
                "proxy_decoder_placement_discovery_failed",
                "request-1",
                endpoint="/completions",
                decoder_url=decoder.url,
                error="unavailable",
            ),
            log_event.call_args_list,
        )

    def test_static_mapping_skips_discovery_when_remote_fill_is_disabled(self):
        decoder = SimpleNamespace(
            url="http://decoder:8002/v1",
            decoder_mooncake_segments={0: "static-decoder:1234"},
            decoder_remote_fill={},
            decoder_remote_fill_discovered=False,
            decoder_rank_active_tokens={0: 0.0},
            decoder_placement_task=None,
        )
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.enable_remote_lmcache_store = False
        state.decoder_placement_discovery_timeout_seconds = 0.5
        discovery = AsyncMock()

        with patch.object(
            proxy,
            "_discover_decoder_mooncake_segments",
            discovery,
        ):
            asyncio.run(
                state.ensure_decoder_mooncake_segments(
                    decoder,
                    "request-1",
                    "/completions",
                )
            )

        discovery.assert_not_awaited()

    def test_cached_remote_fill_is_not_selected_without_explicit_opt_in(self):
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.enable_remote_lmcache_store = False
        server = SimpleNamespace(
            decoder_mooncake_segments={0: "static-decoder:1234"},
            decoder_rank_active_tokens={0: 0.0},
            decoder_remote_fill={0: {"control_endpoint": "tcp://decoder:19001"}},
        )
        reservation = proxy.DecoderReservation(server, 0, 1.0)

        state.assign_decoder_rank(reservation)

        self.assertEqual(reservation.preferred_segment, "static-decoder:1234")
        self.assertIsNone(reservation.remote_fill)

    def test_decoder_mooncake_segment_mapping_validation(self):
        self.assertEqual(
            proxy._parse_decoder_mooncake_segments(
                ["0=decoder-a:12345, 1=decoder-b:12345", "2=decoder-c:12345"],
                2,
            ),
            [
                {0: "decoder-a:12345", 1: "decoder-b:12345"},
                {2: "decoder-c:12345"},
            ],
        )
        self.assertIsNone(proxy._parse_decoder_mooncake_segments(None, 2))
        self.assertEqual(
            proxy._parse_decoder_mooncake_segments(
                ["0=decoder-a:12345", "0=decoder-b:12345"], 2
            ),
            [{0: "decoder-a:12345"}, {0: "decoder-b:12345"}],
        )

        invalid_mappings = (
            (["0=a:1"], 2),
            (["0=a:1,0=b:1"], 1),
            (["0="], 1),
            ([""], 1),
            (["not-a-mapping"], 1),
            (["-1=a:1"], 1),
        )
        for mappings, decoder_count in invalid_mappings:
            with self.subTest(mappings=mappings), self.assertRaises(ValueError):
                proxy._parse_decoder_mooncake_segments(
                    mappings, decoder_count
                )

    def test_log_event_uses_correlatable_completion_request_id(self):
        with (
            patch.dict(os.environ, {"LMCACHE_COLD_START_PERF": "1"}),
            patch.object(proxy.time, "perf_counter", return_value=12.345),
            patch("builtins.print") as print_line,
        ):
            proxy._log_proxy_cold_perf_event(
                "proxy_decoder_send_start",
                "request-uuid",
                endpoint="/completions",
                attempt=1,
            )

        print_line.assert_called_once()
        args, kwargs = print_line.call_args
        self.assertTrue(args[0].startswith("[LMCACHE_COLD_PERF] "))
        self.assertIs(kwargs["file"], proxy.sys.stderr)
        self.assertIs(kwargs["flush"], True)
        payload = json.loads(args[0].split(" ", 1)[1])
        self.assertEqual(payload["event"], "proxy_decoder_send_start")
        self.assertEqual(payload["monotonic_ms"], 12345.0)
        self.assertEqual(payload["req_id"], "cmpl-request-uuid")
        self.assertEqual(payload["proxy_request_id"], "request-uuid")
        self.assertEqual(payload["attempt"], 1)
        self.assertGreater(payload["wall_time_ns"], 0)
        self.assertEqual(payload["host"], proxy._HOST)
        self.assertEqual(payload["clock_domain"], proxy._CLOCK_DOMAIN)

    def test_log_event_is_disabled_by_default(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("builtins.print") as print_line,
        ):
            proxy._log_proxy_cold_perf_event(
                "proxy_decoder_send_start",
                "request-uuid",
                endpoint="/completions",
            )

        print_line.assert_not_called()

    def test_service_auth_header_is_omitted_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(proxy._service_auth_headers(), {})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=True):
            self.assertEqual(
                proxy._service_auth_headers(),
                {"Authorization": "Bearer secret"},
            )

    def test_dynamic_topology_is_rejected_for_remote_fill_prototype(self):
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.enable_remote_lmcache_store = True

        with self.assertRaisesRegex(RuntimeError, "Dynamic topology"):
            asyncio.run(state.add_instances(proxy.InstanceType.DECODE, []))
        with self.assertRaisesRegex(RuntimeError, "Dynamic topology"):
            state.remove_decoders([object()])

    def test_prefiller_payload_is_encoded_once_and_reused_for_retry(self):
        response = SimpleNamespace(
            raise_for_status=MagicMock(),
        )
        client = SimpleNamespace(
            post=AsyncMock(
                side_effect=[
                    proxy.httpx.RequestError("retry"),
                    response,
                ]
            )
        )
        state = SimpleNamespace(
            acquire_aborted_prefiller_requests=MagicMock(
                return_value={"aborted-request"}
            )
        )
        request_data = {
            "prompt": "hello",
            "stream": True,
            "max_tokens": 16,
            "stream_options": {"include_usage": True},
        }

        with (
            patch.object(proxy, "proxy_state", state),
            patch.object(
                proxy,
                "_encode_json_payload",
                wraps=proxy._encode_json_payload,
            ) as encode_payload,
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            result = asyncio.run(
                proxy.send_request_to_service(
                    client,
                    0,
                    "/completions",
                    request_data,
                    "request-uuid",
                    max_retries=2,
                    base_delay=0,
                )
            )

        self.assertIs(result, response)
        encode_payload.assert_called_once()
        first_content = client.post.call_args_list[0].kwargs["content"]
        second_content = client.post.call_args_list[1].kwargs["content"]
        self.assertIs(first_content, second_content)
        encoded = json.loads(first_content)
        self.assertFalse(encoded["stream"])
        self.assertEqual(encoded["max_tokens"], 1)
        self.assertEqual(encoded["min_tokens"], 1)
        self.assertNotIn("stream_options", encoded)
        self.assertEqual(
            encoded["kv_transfer_params"]["aborted_request"],
            ["aborted-request"],
        )
        self.assertTrue(request_data["stream"])
        self.assertEqual(request_data["max_tokens"], 16)

    def test_prefiller_payload_gets_affinity_hint_without_mutating_request(self):
        response = SimpleNamespace(raise_for_status=MagicMock())
        client = SimpleNamespace(post=AsyncMock(return_value=response))
        state = SimpleNamespace(
            acquire_aborted_prefiller_requests=MagicMock(return_value=set())
        )
        request_data = {
            "prompt": "hello",
            "kv_transfer_params": {"caller": "value"},
        }

        with (
            patch.object(proxy, "proxy_state", state),
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            asyncio.run(
                proxy.send_request_to_service(
                    client,
                    0,
                    "/completions",
                    request_data,
                    "request-uuid",
                    preferred_mooncake_segment="decoder-a:12345",
                )
            )

        payload = json.loads(client.post.call_args.kwargs["content"])
        self.assertEqual(
            payload["kv_transfer_params"][
                "lmcache.mooncake_preferred_segment"
            ],
            "decoder-a:12345",
        )
        self.assertEqual(
            request_data["kv_transfer_params"], {"caller": "value"}
        )

    def test_prefiller_payload_gets_remote_fill_handoff(self):
        response = SimpleNamespace(raise_for_status=MagicMock())
        client = SimpleNamespace(post=AsyncMock(return_value=response))
        state = SimpleNamespace(
            acquire_aborted_prefiller_requests=MagicMock(return_value=set())
        )
        request_data = {"prompt": "hello"}
        handoff = {
            **{
                key: value
                for key, value in self._remote_fill_placement().items()
                if key not in {"enabled", "tp_rank", "dp_rank"}
            },
            "destination_dp_rank": 1,
            "transfer_id": "transfer-1",
            "request_attempt": 1,
            "source_engine_id": "prefiller-0",
        }

        with (
            patch.object(proxy, "proxy_state", state),
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            asyncio.run(
                proxy.send_request_to_service(
                    client,
                    0,
                    "/completions",
                    request_data,
                    "request-uuid",
                    remote_fill_handoff=handoff,
                )
            )

        payload = json.loads(client.post.call_args.kwargs["content"])
        self.assertEqual(
            payload["kv_transfer_params"]["lmcache.remote_fill"], handoff
        )
        self.assertEqual(
            payload["kv_transfer_params"]["lmcache.remote_fill"][
                "descriptor_verification_capability"
            ],
            "ab" * 32,
        )
        self.assertNotIn("kv_transfer_params", request_data)

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
            ensure_decoder_mooncake_segments=AsyncMock(),
            assign_decoder_rank=MagicMock(side_effect=lambda value: value),
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
            patch.object(
                proxy.time,
                "perf_counter",
                side_effect=[10.0, 10.002],
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
                    "proxy_request_received",
                    request_id,
                    endpoint="/completions",
                    request_bytes=4096,
                ),
                call(
                    "proxy_prefiller_dispatch",
                    request_id,
                    endpoint="/completions",
                    prefiller_url=prefiller.url,
                    request_bytes=4096,
                ),
                call(
                    "proxy_prefill_response_received",
                    request_id,
                    endpoint="/completions",
                    prefiller_url=prefiller.url,
                    response_bytes=len(response.content),
                    request_bytes=4096,
                ),
                call(
                    "proxy_decoder_body_encode_complete",
                    request_id,
                    endpoint="/completions",
                    encode_ms=2.0,
                    body_bytes=len(
                        b'{"kv_transfer_params":{"source":"live"}}'
                    ),
                ),
                call(
                    "proxy_decoder_dispatch_ready",
                    request_id,
                    endpoint="/completions",
                    decoder_url=decoder.url,
                    request_bytes=4096,
                    kv_transfer_param_keys=["source"],
                    remote_fill_transfer_id=None,
                ),
            ],
        )
        self.assertEqual(
            req_data["kv_transfer_params"],
            {"source": "live"},
        )
        self.assertEqual(
            result.decoder_body,
            b'{"kv_transfer_params":{"source":"live"}}',
        )

    def test_remote_fill_terminal_is_validated_and_scrubbed_before_decode(self):
        request_id = "request-uuid"
        prefiller = proxy.ServerState.__new__(proxy.ServerState)
        prefiller.host = "prefiller"
        prefiller.port = 8001
        prefiller.url = "http://prefiller:8001/v1"
        prefiller.client = object()
        decoder = SimpleNamespace(
            client=object(),
            url="http://decoder:8002/v1",
            decoder_mooncake_segments={1: None},
        )
        remote_fill = {
            key: value
            for key, value in self._remote_fill_placement().items()
            if key not in {"enabled", "tp_rank", "dp_rank"}
        } | {"destination_dp_rank": 1}

        def assign_rank(reservation):
            reservation.dp_rank = 1
            reservation.remote_fill = dict(remote_fill)
            return reservation

        state = SimpleNamespace(
            enable_remote_lmcache_store=True,
            calculate_prefill_scores=MagicMock(return_value=100.0),
            next_req_id=AsyncMock(return_value=request_id),
            select_prefiller=MagicMock(return_value=0),
            prefillers=[prefiller],
            release_prefiller=MagicMock(),
            calculate_decode_scores=MagicMock(return_value=10.0),
            select_decoder=MagicMock(return_value=0),
            decoders=[decoder],
            ensure_decoder_mooncake_segments=AsyncMock(),
            assign_decoder_rank=MagicMock(side_effect=assign_rank),
        )

        async def send_prefill(*_args, **kwargs):
            handoff = kwargs["remote_fill_handoff"]
            self.assertIsNone(kwargs["preferred_mooncake_segment"])
            self.assertEqual(kwargs["max_retries"], 1)
            self.assertEqual(handoff["source_engine_id"], "prefiller:8001")
            terminal = {
                "outcome": "LOCAL_FULL",
                "persistent_common_end": 4096,
                "required_store_end": 4096,
                "transfer_id": handoff["transfer_id"],
            }
            response_json = {
                "kv_transfer_params": {
                    "source": "ordinary-lmcache",
                    "lmcache.remote_fill_result": {
                        "outcome": "untrusted-prefiller-value"
                    },
                    "lmcache.remote_fill": {
                        "terminal": terminal,
                        "must_not_reach_decoder": "secret-control-state",
                        "descriptor_verification_capability": "ab" * 32,
                    },
                }
            }
            return SimpleNamespace(
                content=json.dumps(response_json).encode(),
                json=MagicMock(return_value=response_json),
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
                proxy, "send_request_to_service", side_effect=send_prefill
            ),
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            result = asyncio.run(
                proxy._handle_select_instance(
                    "/completions", req_data, request_length=4096
                )
            )

        self.assertEqual(
            json.loads(result.decoder_body)["kv_transfer_params"],
            {
                "source": "ordinary-lmcache",
                "lmcache.remote_fill_result": {
                    "outcome": "LOCAL_FULL",
                    "required_store_end": 4096,
                    "destination_engine_epoch": 7,
                },
            },
        )
        self.assertNotIn("lmcache.remote_fill", req_data["kv_transfer_params"])
        self.assertNotIn(
            "must_not_reach_decoder",
            result.decoder_body.decode(),
        )
        self.assertNotIn(
            "descriptor_verification_capability",
            result.decoder_body.decode(),
        )

    def test_prefiller_metadata_replaces_caller_metadata_when_missing(self):
        prefiller = SimpleNamespace(
            client=object(), url="http://prefiller:8001/v1"
        )
        decoder = SimpleNamespace(
            client=object(), url="http://decoder:8002/v1"
        )
        state = SimpleNamespace(
            calculate_prefill_scores=MagicMock(return_value=100.0),
            next_req_id=AsyncMock(return_value="request-uuid"),
            select_prefiller=MagicMock(return_value=0),
            prefillers=[prefiller],
            release_prefiller=MagicMock(),
            calculate_decode_scores=MagicMock(return_value=10.0),
            select_decoder=MagicMock(return_value=0),
            decoders=[decoder],
            ensure_decoder_mooncake_segments=AsyncMock(),
            assign_decoder_rank=MagicMock(side_effect=lambda value: value),
        )
        response = SimpleNamespace(
            content=b"{}",
            json=MagicMock(
                return_value={
                    "kv_transfer_params": {
                        "lmcache.remote_fill_result": {
                            "outcome": "caller-or-prefiller-forged"
                        }
                    }
                }
            ),
        )
        req_data = {
            "kv_transfer_params": {
                "caller": "must-not-reach-decoder",
                "lmcache.mooncake_preferred_segment": "untrusted:1",
            }
        }

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
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            result = asyncio.run(
                proxy._handle_select_instance(
                    "/completions", req_data, request_length=4096
                )
            )

        self.assertEqual(req_data["kv_transfer_params"], {})
        self.assertEqual(
            json.loads(result.decoder_body)["kv_transfer_params"], {}
        )

    def test_non_dictionary_prefiller_metadata_fails_closed(self):
        prefiller = SimpleNamespace(
            client=object(), url="http://prefiller:8001/v1"
        )
        decoder = SimpleNamespace(url="http://decoder:8002/v1")
        reservations = []

        def assign_rank(reservation):
            reservation.dp_rank = 3
            reservation.preferred_segment = "decoder-a:12345"
            reservations.append(reservation)
            return reservation

        state = SimpleNamespace(
            decoder_mooncake_segments=[{3: "decoder-a:12345"}],
            calculate_prefill_scores=MagicMock(return_value=100.0),
            calculate_decode_scores=MagicMock(return_value=10.0),
            next_req_id=AsyncMock(return_value="request-uuid"),
            select_decoder=MagicMock(return_value=0),
            decoders=[decoder],
            ensure_decoder_mooncake_segments=AsyncMock(),
            assign_decoder_rank=MagicMock(side_effect=assign_rank),
            select_prefiller=MagicMock(return_value=0),
            prefillers=[prefiller],
            release_prefiller=MagicMock(),
            release_decoder_reservation=MagicMock(),
            abort_prefiller_request=MagicMock(),
            release_prefiller_kv=MagicMock(),
        )
        response = SimpleNamespace(
            content=b"bad",
            json=MagicMock(return_value={"kv_transfer_params": ["bad"]}),
        )

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
            patch.object(proxy, "_log_proxy_cold_perf_event"),
            self.assertRaises(TypeError),
        ):
            asyncio.run(
                proxy._handle_select_instance(
                    "/completions", {}, request_length=4096
                )
            )

        state.release_decoder_reservation.assert_called_once_with(
            reservations[0]
        )
        state.abort_prefiller_request.assert_called_once_with(
            0, "request-uuid"
        )
        state.release_prefiller_kv.assert_called_once_with(0, 100.0)

    def test_affinity_reserves_decoder_before_prefill_and_strips_hint(self):
        request_id = "request-uuid"
        order = []
        prefiller = SimpleNamespace(
            client=object(), url="http://prefiller:8001/v1"
        )
        decoder = SimpleNamespace(
            client=object(), url="http://decoder:8002/v1"
        )
        def reserve_decoder(_score):
            order.append("reserve_decoder")
            return 0

        async def discover_decoder(*_args):
            order.append("discover_decoder")

        def assign_rank(reservation):
            order.append("assign_rank")
            reservation.dp_rank = 3
            reservation.preferred_segment = "decoder-a:12345"
            return reservation

        def select_prefiller(_score):
            order.append("select_prefiller")
            return 0

        state = SimpleNamespace(
            decoder_mooncake_segments=[{3: "decoder-a:12345"}],
            calculate_prefill_scores=MagicMock(return_value=100.0),
            calculate_decode_scores=MagicMock(return_value=10.0),
            next_req_id=AsyncMock(return_value=request_id),
            select_decoder=MagicMock(side_effect=reserve_decoder),
            decoders=[decoder],
            ensure_decoder_mooncake_segments=AsyncMock(
                side_effect=discover_decoder
            ),
            assign_decoder_rank=MagicMock(side_effect=assign_rank),
            select_prefiller=MagicMock(side_effect=select_prefiller),
            prefillers=[prefiller],
            release_prefiller=MagicMock(),
        )
        response = SimpleNamespace(
            content=b"response",
            json=MagicMock(
                return_value={
                    "kv_transfer_params": {
                        "source": "live",
                        "lmcache.mooncake_preferred_segment": (
                            "decoder-a:12345"
                        ),
                    }
                }
            ),
        )

        async def send_prefill(*args, **kwargs):
            order.append("send_prefill")
            self.assertEqual(
                kwargs["preferred_mooncake_segment"],
                "decoder-a:12345",
            )
            return response

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
                proxy, "send_request_to_service", side_effect=send_prefill
            ),
            patch.object(proxy, "_log_proxy_cold_perf_event") as log_event,
        ):
            result = asyncio.run(
                proxy._handle_select_instance(
                    "/completions", req_data, request_length=4096
                )
            )

        self.assertEqual(
            order,
            [
                "reserve_decoder",
                "discover_decoder",
                "assign_rank",
                "select_prefiller",
                "send_prefill",
            ],
        )
        self.assertEqual(
            result.reservation.dp_rank, 3
        )
        self.assertEqual(
            result.reservation.preferred_segment, "decoder-a:12345"
        )
        self.assertEqual(
            json.loads(result.decoder_body)["kv_transfer_params"],
            {"source": "live"},
        )
        self.assertEqual(
            log_event.call_args_list[0],
            call(
                "proxy_request_received",
                request_id,
                endpoint="/completions",
                request_bytes=4096,
            ),
        )
        placement_event = next(
            item
            for item in log_event.call_args_list
            if item.args[0] == "proxy_decoder_placement_reserved"
        )
        self.assertEqual(
            placement_event.args[:2],
            ("proxy_decoder_placement_reserved", request_id),
        )
        self.assertEqual(placement_event.kwargs["dp_rank"], 3)
        self.assertEqual(
            placement_event.kwargs["preferred_segment"],
            "decoder-a:12345",
        )

    def test_affinity_prefill_cancellation_releases_all_reservations(self):
        prefiller = SimpleNamespace(client=object())
        decoder = SimpleNamespace(url="http://decoder:8002/v1")
        reservations = []

        def assign_rank(reservation):
            reservation.dp_rank = 3
            reservation.preferred_segment = "decoder-a:12345"
            reservations.append(reservation)
            return reservation

        state = SimpleNamespace(
            decoder_mooncake_segments=[{3: "decoder-a:12345"}],
            calculate_prefill_scores=MagicMock(return_value=100.0),
            calculate_decode_scores=MagicMock(return_value=10.0),
            next_req_id=AsyncMock(return_value="request-uuid"),
            select_decoder=MagicMock(return_value=0),
            decoders=[decoder],
            ensure_decoder_mooncake_segments=AsyncMock(),
            assign_decoder_rank=MagicMock(side_effect=assign_rank),
            select_prefiller=MagicMock(return_value=0),
            prefillers=[prefiller],
            release_decoder_reservation=MagicMock(),
            release_prefiller=MagicMock(),
            release_prefiller_kv=MagicMock(),
            abort_prefiller_request=MagicMock(),
        )

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
                AsyncMock(side_effect=asyncio.CancelledError),
            ),
            patch.object(proxy, "_log_proxy_cold_perf_event"),
            self.assertRaises(asyncio.CancelledError),
        ):
            asyncio.run(
                proxy._handle_select_instance(
                    "/completions", {}, request_length=4096
                )
            )

        state.release_decoder_reservation.assert_called_once_with(
            reservations[0]
        )
        state.release_prefiller.assert_called_once_with(0, 100.0)
        state.abort_prefiller_request.assert_called_once_with(
            0, "request-uuid"
        )
        state.release_prefiller_kv.assert_called_once_with(0, 100.0)

    def test_decoder_reservation_selects_endpoint_before_dp_rank(self):
        decoder_zero = SimpleNamespace(
            decoder_mooncake_segments={0: "zero:1"},
            decoder_rank_active_tokens={0: 0.0},
        )
        decoder_one = SimpleNamespace(
            decoder_mooncake_segments={1: "one-a:1", 2: "one-b:1"},
            decoder_rank_active_tokens={1: 20.0, 2: 5.0},
        )
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.decoders = [decoder_zero, decoder_one]
        state.select_decoder = MagicMock(return_value=1)

        reservation = state.select_decoder_reservation(10.0)

        state.select_decoder.assert_called_once_with(10.0)
        self.assertIs(reservation.server, decoder_one)
        self.assertEqual(reservation.decoder_idx, 1)
        self.assertEqual(reservation.dp_rank, 2)
        self.assertEqual(reservation.preferred_segment, "one-b:1")
        self.assertEqual(decoder_one.decoder_rank_active_tokens[2], 15.0)
        self.assertEqual(decoder_zero.decoder_rank_active_tokens[0], 0.0)

    def test_unmapped_dynamic_decoder_uses_normal_endpoint_reservation(self):
        decoder = SimpleNamespace(
            decoder_mooncake_segments=None,
            decoder_rank_active_tokens={},
        )
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.decoders = [decoder]
        state.select_decoder = MagicMock(return_value=0)

        reservation = state.select_decoder_reservation(10.0)

        state.select_decoder.assert_called_once_with(10.0)
        self.assertIs(reservation.server, decoder)
        self.assertIsNone(reservation.dp_rank)
        self.assertIsNone(reservation.preferred_segment)

    def test_reservation_release_does_not_mutate_equal_replacement_server(self):
        class Endpoint(SimpleNamespace):
            def __eq__(self, _other):
                return True

        original = Endpoint(
            active_tokens=10.0,
            decoder_rank_active_tokens={3: 10.0},
        )
        replacement = Endpoint(
            active_tokens=50.0,
            decoder_rank_active_tokens={3: 50.0},
        )
        state = proxy.ProxyState.__new__(proxy.ProxyState)
        state.decoders = [replacement]
        state._update_decoder_priority = MagicMock()
        reservation = proxy.DecoderReservation(
            original, 0, 10.0, 3, "decoder-a:12345"
        )

        state.release_decoder_reservation(reservation)

        self.assertEqual(original.active_tokens, 0.0)
        self.assertEqual(original.decoder_rank_active_tokens[3], 0.0)
        self.assertEqual(replacement.active_tokens, 50.0)
        self.assertEqual(replacement.decoder_rank_active_tokens[3], 50.0)
        state._update_decoder_priority.assert_not_called()

    def test_instance_reservation_release_is_idempotent(self):
        decoder = SimpleNamespace()
        instance_info = proxy.InstanceInfo(
            request_id="request-uuid",
            prefiller_idx=0,
            prefiller_score=100.0,
            prefiller=SimpleNamespace(),
            decoder_idx=0,
            decoder_score=10.0,
            decoder=decoder,
            decoder_body=b"{}",
        )
        reservation = instance_info.reservation
        state = SimpleNamespace(release_decoder_reservation=MagicMock())

        with patch.object(proxy, "proxy_state", state):
            proxy._release_decoder_reservation(instance_info)
            proxy._release_decoder_reservation(instance_info)

        state.release_decoder_reservation.assert_called_once_with(reservation)
        self.assertIsNone(instance_info.reservation)

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
                    b"{}",
                    "request-uuid",
                )
            ]

        with patch.object(
            proxy, "_log_proxy_cold_perf_event"
        ) as log_event:
            chunks = asyncio.run(collect_chunks())

        self.assertEqual(chunks, [b"chunk"])
        self.assertEqual(
            log_event.call_args_list,
            [
                call(
                    "proxy_decoder_send_start",
                    "request-uuid",
                    endpoint="/completions",
                    attempt=1,
                    decoder_url="http://decoder:8002/v1/",
                    body_bytes=2,
                ),
                call(
                    "proxy_decoder_first_byte_received",
                    "request-uuid",
                    endpoint="/completions",
                    attempt=1,
                    response_bytes=5,
                ),
            ],
        )
        stream_kwargs = client.stream.call_args.kwargs
        self.assertEqual(stream_kwargs["content"], b"{}")
        self.assertEqual(
            stream_kwargs["headers"]["Content-Type"],
            "application/json",
        )
        self.assertNotIn("X-data-parallel-rank", stream_kwargs["headers"])

    def test_decoder_dp_rank_header_is_stable_across_transport_retry(self):
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
            stream=MagicMock(
                side_effect=[
                    proxy.httpx.RequestError("retry"),
                    StreamContext(),
                ]
            ),
        )

        async def collect_chunks():
            return [
                chunk
                async for chunk in proxy.stream_service_response_with_retry(
                    client,
                    "/completions",
                    b"{}",
                    "request-uuid",
                    max_retries=2,
                    base_delay=0,
                    decoder_dp_rank=3,
                )
            ]

        with patch.object(proxy, "_log_proxy_cold_perf_event"):
            chunks = asyncio.run(collect_chunks())

        self.assertEqual(chunks, [b"chunk"])
        self.assertEqual(client.stream.call_count, 2)
        for stream_call in client.stream.call_args_list:
            self.assertEqual(
                stream_call.kwargs["headers"]["X-data-parallel-rank"],
                "3",
            )

    def test_decoder_transport_error_after_first_chunk_is_not_retried(self):
        class Response:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            async def aiter_bytes():
                yield b"first"
                raise proxy.httpx.ReadError("connection lost")

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
                    b"{}",
                    "request-uuid",
                    max_retries=2,
                    base_delay=0,
                    decoder_dp_rank=3,
                )
            ]

        with patch.object(proxy, "_log_proxy_cold_perf_event"):
            chunks = asyncio.run(collect_chunks())

        self.assertEqual(chunks, [b"first"])
        self.assertEqual(client.stream.call_count, 1)
        self.assertEqual(
            client.stream.call_args.kwargs["headers"][
                "X-data-parallel-rank"
            ],
            "3",
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
            decoder_body=b"{}",
        )
        state = SimpleNamespace(
            request_num=0,
            release_prefiller_kv=MagicMock(),
            release_decoder_reservation=MagicMock(),
        )

        async def decoder_chunks(*args, **kwargs):
            yield b'data: {"choices":[{"text":"x"}]}\n\n'

        async def consume_response():
            response = await proxy._handle_completions(
                "/completions",
                request,
            )
            self.assertEqual(response.headers["x-request-id"], "request-uuid")
            self.assertEqual(state.request_num, 1)
            chunks = [chunk async for chunk in response.body_iterator]
            self.assertEqual(state.request_num, 0)
            response._cleanup()
            self.assertEqual(state.request_num, 0)
            return chunks

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
        self.assertEqual(
            log_event.call_args_list[0].args[0],
            "proxy_request_received",
        )
        self.assertEqual(
            log_event.call_args_list[-1],
            call(
                "proxy_decoder_generator_entry",
                "request-uuid",
                endpoint="/completions",
                decoder_url=decoder.url,
                request_bytes=len(b'{"prompt":"hello"}'),
            ),
        )
        state.release_decoder_reservation.assert_called_once()

    def test_recompute_releases_old_reservation_before_replacement(self):
        request = SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "prompt": "hello",
                    "stream": True,
                    "max_tokens": 4,
                }
            ),
            body=AsyncMock(return_value=b'{"prompt":"hello"}'),
        )
        prefiller_old = SimpleNamespace(url="http://prefiller-old:8001/v1")
        prefiller_new = SimpleNamespace(url="http://prefiller-new:8001/v1")
        decoder_old = SimpleNamespace(
            client=object(), url="http://decoder-old:8002/v1"
        )
        decoder_new = SimpleNamespace(
            client=object(), url="http://decoder-new:8002/v1"
        )
        old_info = proxy.InstanceInfo(
            request_id="old-request",
            prefiller_idx=0,
            prefiller_score=100.0,
            prefiller=prefiller_old,
            decoder_idx=0,
            decoder_score=10.0,
            decoder=decoder_old,
            decoder_body=b"{}",
            reservation=proxy.DecoderReservation(
                decoder_old, 0, 10.0, 1, "old-segment:1"
            ),
        )
        new_info = proxy.InstanceInfo(
            request_id="new-request",
            prefiller_idx=1,
            prefiller_score=110.0,
            prefiller=prefiller_new,
            decoder_idx=1,
            decoder_score=11.0,
            decoder=decoder_new,
            decoder_body=b"{}",
            reservation=proxy.DecoderReservation(
                decoder_new, 1, 11.0, 2, "new-segment:1"
            ),
        )
        old_reservation = old_info.reservation
        order = []

        async def select_instance(*args, **kwargs):
            selected = old_info if not order else new_info
            order.append(f"select:{selected.request_id}")
            return selected

        async def decoder_chunks(*args, request_id, **kwargs):
            if request_id == "old-request":
                yield (
                    b'data: {"choices":[{"text":"a",'
                    b'"stop_reason":"recomputed"}]}\n\n'
                )
            else:
                yield b'data: {"choices":[{"text":"b"}]}\n\n'

        def release_reservation(reservation):
            order.append(
                "release:old-request"
                if reservation is old_reservation
                else "release:new-request"
            )

        state = SimpleNamespace(
            request_num=0,
            release_prefiller_kv=MagicMock(),
            abort_prefiller_request=MagicMock(),
            release_decoder_reservation=MagicMock(
                side_effect=release_reservation
            ),
        )

        async def consume_response():
            response = await proxy._handle_completions(
                "/completions", request
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
                side_effect=select_instance,
            ),
            patch.object(
                proxy,
                "stream_service_response_with_retry",
                decoder_chunks,
            ),
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            chunks = asyncio.run(consume_response())

        self.assertEqual(
            chunks, [b'data: {"choices":[{"text":"b"}]}\n\n']
        )
        self.assertEqual(
            order,
            [
                "select:old-request",
                "release:old-request",
                "select:new-request",
                "release:new-request",
            ],
        )
        self.assertEqual(
            state.release_prefiller_kv.call_args_list,
            [call(0, 100.0), call(1, 110.0)],
        )
        state.abort_prefiller_request.assert_not_called()

    def test_stream_cancellation_releases_reservation_and_prefiller_kv(self):
        request = SimpleNamespace(
            json=AsyncMock(return_value={"prompt": "hello", "stream": True}),
            body=AsyncMock(return_value=b'{"prompt":"hello"}'),
        )
        prefiller = SimpleNamespace(url="http://prefiller:8001/v1")
        decoder = SimpleNamespace(
            client=object(), url="http://decoder:8002/v1"
        )
        instance_info = proxy.InstanceInfo(
            request_id="request-uuid",
            prefiller_idx=0,
            prefiller_score=100.0,
            prefiller=prefiller,
            decoder_idx=0,
            decoder_score=10.0,
            decoder=decoder,
            decoder_body=b"{}",
            reservation=proxy.DecoderReservation(
                decoder, 0, 10.0, 3, "decoder-a:12345"
            ),
        )
        reservation = instance_info.reservation
        started = asyncio.Event()

        async def decoder_chunks(*args, **kwargs):
            started.set()
            await asyncio.Future()
            yield b"unreachable"

        state = SimpleNamespace(
            request_num=0,
            release_prefiller_kv=MagicMock(),
            abort_prefiller_request=MagicMock(),
            release_decoder_reservation=MagicMock(),
        )

        async def cancel_response():
            response = await proxy._handle_completions(
                "/completions", request
            )
            self.assertEqual(state.request_num, 1)
            next_chunk = asyncio.create_task(
                response.body_iterator.__anext__()
            )
            await started.wait()
            next_chunk.cancel()
            with suppress(asyncio.CancelledError):
                await next_chunk
            self.assertEqual(state.request_num, 0)

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
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            asyncio.run(cancel_response())

        state.abort_prefiller_request.assert_called_once_with(
            0, "request-uuid"
        )
        state.release_prefiller_kv.assert_called_once_with(0, 100.0)
        state.release_decoder_reservation.assert_called_once_with(reservation)

    def test_selection_cancellation_releases_active_request_count(self):
        request = SimpleNamespace(
            json=AsyncMock(return_value={"prompt": "hello"}),
            body=AsyncMock(return_value=b'{"prompt":"hello"}'),
        )
        state = SimpleNamespace(request_num=0)

        with (
            patch.object(proxy, "proxy_state", state),
            patch.object(
                proxy,
                "_handle_select_instance",
                AsyncMock(side_effect=asyncio.CancelledError),
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            asyncio.run(proxy._handle_completions("/completions", request))

        self.assertEqual(state.request_num, 0)

    def test_response_cancellation_before_iteration_releases_reservation(self):
        request = SimpleNamespace(
            json=AsyncMock(return_value={"prompt": "hello", "stream": True}),
            body=AsyncMock(return_value=b'{"prompt":"hello"}'),
        )
        prefiller = SimpleNamespace(url="http://prefiller:8001/v1")
        decoder = SimpleNamespace(
            client=object(), url="http://decoder:8002/v1"
        )
        instance_info = proxy.InstanceInfo(
            request_id="request-uuid",
            prefiller_idx=0,
            prefiller_score=100.0,
            prefiller=prefiller,
            decoder_idx=0,
            decoder_score=10.0,
            decoder=decoder,
            decoder_body=b"{}",
            reservation=proxy.DecoderReservation(
                decoder, 0, 10.0, 3, "decoder-a:12345"
            ),
        )
        reservation = instance_info.reservation
        state = SimpleNamespace(
            request_num=0,
            release_prefiller_kv=MagicMock(),
            abort_prefiller_request=MagicMock(),
            release_decoder_reservation=MagicMock(),
        )

        async def never_started(*args, **kwargs):
            raise AssertionError("decoder stream must not start")
            yield b"unreachable"

        async def cancel_before_iteration():
            response = await proxy._handle_completions(
                "/completions", request
            )
            self.assertEqual(state.request_num, 1)

            async def receive():
                return {"type": "http.disconnect"}

            async def send(_message):
                raise asyncio.CancelledError

            with self.assertRaises(asyncio.CancelledError):
                await response(
                    {
                        "type": "http",
                        "asgi": {"spec_version": "2.4"},
                    },
                    receive,
                    send,
                )
            self.assertEqual(state.request_num, 0)

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
                never_started,
            ),
            patch.object(proxy, "_log_proxy_cold_perf_event"),
        ):
            asyncio.run(cancel_before_iteration())

        state.abort_prefiller_request.assert_called_once_with(
            0, "request-uuid"
        )
        state.release_prefiller_kv.assert_called_once_with(0, 100.0)
        state.release_decoder_reservation.assert_called_once_with(reservation)


if __name__ == "__main__":
    unittest.main()
