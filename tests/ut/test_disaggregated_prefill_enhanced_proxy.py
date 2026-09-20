import asyncio
import json
import time
import unittest
from collections import Counter
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
        "destination_remote_session": "decoder:12345",
        "descriptor_verification_capability": "ab" * 32,
        "tp_rank": 0,
        "dp_rank": dp_rank,
    }


def _prime_remote_fill(
    state,
    dp_rank: int = 0,
    api_dp_rank: int | None = None,
    decoder_idx: int = 0,
    segment: str = "decoder:12345",
):
    decoder = state.decoders[decoder_idx]
    decoder.decoder_remote_fill = {
        dp_rank: {
            key: value
            for key, value in _remote_fill(dp_rank).items()
            if key not in {"enabled", "tp_rank", "dp_rank"}
        }
        | {
            "api_dp_rank": dp_rank if api_dp_rank is None else api_dp_rank,
            "destination_dp_rank": dp_rank,
            "mooncake_preferred_segment": segment,
        }
    }
    decoder.decoder_rank_active_tokens = {dp_rank: 0.0}
    decoder.decoder_placement_discovered_at = time.monotonic()
    return decoder


def _reserve_instance(state, request_id: str = "request", tokens: int = 100):
    prefiller_score = state.calculate_prefill_scores(tokens)
    decoder_score = state.calculate_decode_scores(tokens)
    prefiller_idx = state.select_prefiller(prefiller_score)
    state.release_prefiller(prefiller_idx, prefiller_score)
    decoder_idx = state.select_decoder(decoder_score)
    return proxy.InstanceInfo(
        request_id,
        prefiller_idx,
        prefiller_score,
        state.prefillers[prefiller_idx],
        decoder_idx,
        decoder_score,
        state.decoders[decoder_idx],
    )


def _prefill_reply(handoff, end=100):
    params = {"remote_engine_id": "persistent"}
    if handoff is not None:
        params["lmcache.remote_fill"] = {"terminal": {
            "outcome": "LOCAL_FULL", "persistent_common_end": end,
            "required_store_end": end, "transfer_id": handoff["transfer_id"],
        }}
    return SimpleNamespace(json=lambda: {"kv_transfer_params": params})


class TestEnhancedRemoteFillProxy(unittest.TestCase):
    def setUp(self) -> None:
        proxy.global_args = SimpleNamespace(
            max_retries=3,
            retry_delay=0.001,
            use_original_lb=True,
            backend_request_timeout=600.0,
            decoder_read_timeout=120.0,
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

    def test_burst_dispatch_stays_interleaved_with_cold_fresh_and_expired_placement(self):
        async def run(mode):
            state = proxy.ProxyState(
                [(f"p{i}", 7910) for i in range(4)],
                [(f"d{i}", 7920) for i in range(4)], enable_remote_lmcache_store=True,
            )
            proxy.proxy_state = state
            placements = []
            for i in range(4):
                d = _prime_remote_fill(state, dp_rank=i, api_dp_rank=0, decoder_idx=i)
                d.decoder_remote_fill[i]["destination_engine_id"] = f"engine-{i}"
                d.decoder_remote_fill[i]["destination_dp_size"] = 4
                placements.append(d.decoder_remote_fill)
                if mode == "cold":
                    d.decoder_remote_fill = {}
                    d.decoder_placement_discovered_at = 0
                elif mode == "expired":
                    d.decoder_placement_discovered_at -= 31
            started = [asyncio.Event() for _ in range(4)]
            release = [asyncio.Event() for _ in range(4)]
            sent, all_sent, finish = [], asyncio.Event(), asyncio.Event()

            async def discover(server, **kwargs):
                i = state.decoders.index(server)
                started[i].set()
                await release[i].wait()
                return placements[i]

            async def send(client, p, api, data, request_id, **kwargs):
                handoff = kwargs["remote_fill_handoff"]
                d = handoff["destination_dp_rank"]
                self.assertEqual(handoff["destination_engine_id"], f"engine-{d}")
                sent.append((p, d))
                if len(sent) == 32:
                    all_sent.set()
                await finish.wait()
                return SimpleNamespace(json=lambda: {"kv_transfer_params": {
                    "remote_engine_id": "persistent",
                    "lmcache.remote_fill": {"terminal": {
                        "outcome": "LOCAL_FULL", "persistent_common_end": 100,
                        "required_store_end": 100, "transfer_id": handoff["transfer_id"],
                    }},
                }})

            try:
                with patch.object(proxy, "_discover_decoder_remote_fill", side_effect=discover) as rpc, \
                     patch.object(proxy, "send_request_to_service", side_effect=send):
                    tasks = [asyncio.create_task(proxy._handle_select_instance(
                        "/completions", {"prompt": "same"}, 100,
                    )) for _ in range(32)]
                    if mode != "fresh":
                        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
                        for event in release[:3]:
                            event.set()
                        for _ in range(10):
                            await asyncio.sleep(0)
                        self.assertEqual(sent, [])
                        self.assertTrue(all(d.active_tokens == 0 for d in state.decoders))
                        release[3].set()
                    await asyncio.wait_for(all_sent.wait(), 2)
                    self.assertEqual(rpc.call_count, 0 if mode == "fresh" else 4)
                    self.assertEqual(Counter(d for _, d in sent), dict.fromkeys(range(4), 8))
                    self.assertEqual(Counter(sent), {(p, d): 2 for p in range(4) for d in range(4)})
                    for offset in range(0, 32, 4):
                        self.assertEqual({d for _, d in sent[offset:offset + 4]}, set(range(4)))
                    finish.set()
                    infos = await asyncio.gather(*tasks)
                    for info in infos:
                        self.assertEqual(info.reservation.api_dp_rank, 0)
                        self.assertEqual(proxy._decoder_headers(info.request_id, 0)["X-data-parallel-rank"], "0")
                        proxy._release_decoder_reservation(info)
                        state.release_prefiller_kv(info.prefiller_idx, info.prefiller_score)
                    self.assertTrue(all(d.active_tokens == 0 for d in state.decoders))
                    self.assertTrue(all(p.active_tokens == p.active_kv_cache == 0 for p in state.prefillers))
            finally:
                finish.set()
                for event in release:
                    event.set()
                await asyncio.gather(*(s.client.aclose() for s in state.prefillers + state.decoders))
                proxy.proxy_state = None

        for mode in ("cold", "fresh", "expired"):
            with self.subTest(mode=mode):
                asyncio.run(run(mode))

    def test_pair_tie_breaking_never_overrides_lower_load_or_preference(self):
        state = proxy.ProxyState([("p", 7910)], [("d0", 7920), ("d1", 7920)])
        proxy.proxy_state = state
        state._pair_last_used[state.prefillers[0], state.decoders[0]] = 10000
        with self.assertRaises(IndexError):
            state.select_decoder(100, preferred_idx=0, prefiller_idx=99)
        self.assertTrue(all(d.active_tokens == 0 for d in state.decoders))
        self.assertEqual(state.select_decoder(100, preferred_idx=1, prefiller_idx=0), 1)
        self.assertEqual(state.select_decoder(100, prefiller_idx=0), 0)
        state.release_decoder(0, 100)
        state.release_decoder(1, 100)
        removed = state.decoders[1]
        state.remove_decoders([removed])
        asyncio.run(removed.client.aclose())
        self.assertTrue(all(pair[1] in state.decoders for pair in state._pair_last_used))

    def test_cancelled_request_does_not_cancel_shared_discovery_or_reserve_load(self):
        state = proxy.ProxyState([("p", 7910)], [("d0", 7920), ("d1", 7920)],
                                 enable_remote_lmcache_store=True)
        proxy.proxy_state = state

        async def run():
            started = [asyncio.Event(), asyncio.Event()]
            release = asyncio.Event()

            async def discover(server, **kwargs):
                i = state.decoders.index(server)
                started[i].set()
                await release.wait()
                return {0: {"api_dp_rank": 0, "destination_dp_rank": 0,
                            "destination_engine_epoch": 7}}

            async def send(*args, **kwargs):
                return _prefill_reply(kwargs["remote_fill_handoff"])

            with patch.object(proxy, "_discover_decoder_remote_fill", side_effect=discover) as rpc, \
                 patch.object(proxy, "send_request_to_service", side_effect=send):
                tasks = [asyncio.create_task(proxy._handle_select_instance(
                    "/completions", {"prompt": "x"}, 100,
                )) for _ in range(2)]
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
                tasks[0].cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await tasks[0]
                self.assertFalse(state._placement_barrier.cancelled())
                self.assertTrue(all(d.active_tokens == 0 for d in state.decoders))
                self.assertEqual(state.prefillers[0].active_tokens, 0)
                release.set()
                info = await asyncio.wait_for(tasks[1], 2)
                self.assertEqual(rpc.call_count, 2)
                proxy._release_decoder_reservation(info)
                state.release_prefiller_kv(info.prefiller_idx, info.prefiller_score)
                self.assertTrue(all(d.active_tokens == 0 for d in state.decoders))
        asyncio.run(run())

    def test_discovery_failure_uses_persistent_fallback_without_repeated_rpc(self):
        state = proxy.ProxyState([("p", 7910)], [("d0", 7920), ("d1", 7920)],
                                 enable_remote_lmcache_store=True)
        proxy.proxy_state = state
        healthy = _prime_remote_fill(state, decoder_idx=1)

        async def run():
            async def discover(server, **kwargs):
                self.assertIs(server, state.decoders[0])
                raise httpx.ReadTimeout("discovery timed out")

            async def send(*args, **kwargs):
                return _prefill_reply(kwargs["remote_fill_handoff"])

            with patch.object(proxy, "_discover_decoder_remote_fill", side_effect=discover) as rpc, \
                 patch.object(proxy, "send_request_to_service", side_effect=send):
                infos = [await proxy._handle_select_instance(
                    "/completions", {"prompt": "x"}, 100,
                ) for _ in range(2)]
                self.assertIsNone(infos[0].reservation.remote_fill)
                self.assertIs(infos[1].reservation.server, healthy)
                self.assertIsNotNone(infos[1].reservation.remote_fill)
                self.assertEqual(rpc.call_count, 1)
                for info in infos:
                    proxy._release_decoder_reservation(info)
                    state.release_prefiller_kv(info.prefiller_idx, info.prefiller_score)
                self.assertTrue(all(d.active_tokens == 0 for d in state.decoders))
                self.assertEqual(state.prefillers[0].active_kv_cache, 0)
        asyncio.run(run())

    def test_discovery_finishes_after_sole_waiter_cancels(self):
        state = proxy.ProxyState([("p", 7910)], [("d", 7920)],
                                 enable_remote_lmcache_store=True)
        proxy.proxy_state = state

        async def run():
            started, release = asyncio.Event(), asyncio.Event()

            async def discover(*args, **kwargs):
                started.set()
                await release.wait()
                return {}

            with patch.object(proxy, "_discover_decoder_remote_fill", side_effect=discover) as rpc:
                waiter = asyncio.create_task(state.ensure_decoder_placements())
                await asyncio.wait_for(started.wait(), 2)
                barrier = state._placement_barrier
                waiter.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiter
                release.set()
                await asyncio.wait_for(asyncio.shield(barrier), 2)
                await state.ensure_decoder_placements()
                self.assertEqual(rpc.call_count, 1)
                self.assertTrue(state.decoder_placements_ready())
                self.assertEqual(state.decoders[0].active_tokens, 0)
        asyncio.run(run())

    def test_decoder_added_during_refresh_is_discovered_before_readiness(self):
        state = proxy.ProxyState([("p", 7910)], [("d0", 7920)], enable_remote_lmcache_store=True)
        proxy.proxy_state = state

        async def run():
            started = [asyncio.Event(), asyncio.Event()]
            release = [asyncio.Event(), asyncio.Event()]

            async def discover(server, **kwargs):
                i = state.decoders.index(server)
                started[i].set()
                await release[i].wait()
                return {}

            with patch.object(proxy, "_discover_decoder_remote_fill", side_effect=discover) as rpc:
                ready = asyncio.create_task(state.ensure_decoder_placements())
                await asyncio.wait_for(started[0].wait(), 2)
                state.add_decoders([proxy.ServerState("d1", 7920)])
                release[0].set()
                await asyncio.wait_for(started[1].wait(), 2)
                self.assertFalse(ready.done())
                release[1].set()
                await asyncio.wait_for(ready, 2)
                self.assertEqual(rpc.call_count, 2)
                with state._state_lock:
                    self.assertTrue(state.decoder_placements_ready())
        asyncio.run(run())

    def test_affinity_is_resolved_after_refresh_using_current_epoch_and_load(self):
        async def run(mode):
            state = proxy.ProxyState([("p0", 7910), ("p1", 7910)],
                                     [("d0", 7920), ("d1", 7920)],
                                     enable_remote_lmcache_store=True, enable_prefix_affinity_routing=True)
            proxy.proxy_state = state
            for i in range(2):
                _prime_remote_fill(state, decoder_idx=i)
                if mode != "valid":
                    state.decoders[i].decoder_placement_discovered_at = 0
            anchor = proxy.PrefixAffinityAnchor("prefix", 20000)
            previous = proxy.DecoderReservation(state.decoders[1], 1, 2000,
                                               dp_rank=0, preferred_segment="decoder:12345",
                                               remote_fill={"destination_engine_epoch": 7})
            state.record_prefix_affinity((anchor,), 20000, 1, previous, 20000)

            async def discover(server, **kwargs):
                result = {rank: dict(value) for rank, value in server.decoder_remote_fill.items()}
                if mode == "epoch":
                    result[0]["destination_engine_epoch"] = 8
                elif server is state.decoders[1]:
                    server.active_tokens = 1e9
                    state._update_decoder_priority(1)
                return result

            async def send(*args, **kwargs):
                return _prefill_reply(kwargs["remote_fill_handoff"], 20000)

            try:
                with patch.object(proxy, "_discover_decoder_remote_fill", side_effect=discover), \
                     patch.object(proxy, "send_request_to_service", side_effect=send):
                    info = await proxy._handle_select_instance(
                        "/completions", {"prompt": "x"}, 20000, prefix_anchors=(anchor,),
                    )
                    self.assertEqual((info.prefiller_idx, info.decoder_idx),
                                     (1, 1) if mode == "valid" else (0, 0))
                    self.assertEqual(info.reservation.remote_fill["destination_engine_epoch"],
                                     8 if mode == "epoch" else 7)
                    proxy._release_decoder_reservation(info)
                    state.release_prefiller_kv(info.prefiller_idx, info.prefiller_score)
            finally:
                await asyncio.gather(*(s.client.aclose() for s in state.prefillers + state.decoders))
                proxy.proxy_state = None
        for mode in ("valid", "epoch", "load"):
            with self.subTest(mode=mode):
                asyncio.run(run(mode))

    def test_disable_tokenizer_analysis_skips_both_exact_analyzers(self):
        proxy.global_args = SimpleNamespace(
            model_name="model",
            max_model_len=131072,
            chat_template=None,
            trust_remote_code=False,
            disable_tokenizer_analysis=True,
            default_max_tokens=None,
            override_max_tokens=None,
            context_length_margin=5,
            disable_metrics=True,
            disable_metrics_polling=True,
            metrics_poll_interval=5.0,
            prefiller_instances=[("prefiller", 8001)],
            decoder_instances=[("decoder", 8002)],
            enable_remote_lmcache_store=True,
            use_original_lb=False,
        )

        async def run() -> None:
            with (
                patch.object(proxy, "VLLM_TOKEN_COUNTER_AVAILABLE", True),
                patch.object(proxy, "VLLMTokenCounter") as counter,
                patch.object(proxy, "TokenizerAnalyzer") as analyzer,
            ):
                async with proxy.lifespan(None):
                    counter.assert_not_called()
                    analyzer.assert_not_called()
                    self.assertIsNone(proxy.proxy_state.vllm_token_counter)
                    self.assertIsNone(proxy.proxy_state.tokenizer_analyzer)
                    self.assertTrue(proxy.proxy_state.enable_remote_lmcache_store)

        asyncio.run(run())
        proxy.proxy_state = None

    def test_no_analyzer_uses_existing_character_estimate(self):
        proxy.global_args.use_original_lb = False
        state = proxy.ProxyState(
            [("prefiller", 8001)],
            [("decoder", 8002)],
            enable_remote_lmcache_store=True,
            enable_prefix_affinity_routing=True,
        )
        proxy.proxy_state = state
        info = _reserve_instance(state)
        request = SimpleNamespace(
            json=AsyncMock(return_value={"prompt": "abcdefgh", "stream": True}),
            body=AsyncMock(return_value=b"unused"),
        )
        select = AsyncMock(return_value=info)

        with patch.object(proxy, "_handle_select_instance", select):
            response = asyncio.run(proxy._handle_completions("/completions", request))

        self.assertEqual(select.await_args.args[2:], (2, None))
        anchors = select.await_args.kwargs["prefix_anchors"]
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].token_end, 2)
        self.assertTrue(anchors[0].key.startswith("auto-"))
        response._cleanup()

    def test_automatic_affinity_uses_progressive_bounded_prefixes(self):
        common = b"x" * (70 << 10)
        first = proxy._automatic_prefix_affinity_anchors(common + b"a", 18000)
        second = proxy._automatic_prefix_affinity_anchors(common + b"b", 18000)

        self.assertLessEqual(len(first), 4)
        self.assertNotEqual(first[0].key, second[0].key)
        self.assertEqual(first[-2:], second[-2:])

    def test_prefix_affinity_header_is_bounded_and_sorted(self):
        anchors = proxy._parse_prefix_affinity_header(
            "1800=system-v1,42000=conversation-v7,12000=project-v2"
        )
        self.assertEqual(
            [(item.key, item.token_end) for item in anchors],
            [
                ("conversation-v7", 42000),
                ("project-v2", 12000),
                ("system-v1", 1800),
            ],
        )
        self.assertEqual(proxy._parse_prefix_affinity_header("bad"), ())
        self.assertEqual(
            proxy._parse_prefix_affinity_header(
                "1=a,2=b,3=c,4=d,5=e"
            ),
            (),
        )

    def test_prefix_affinity_ignores_common_prefix_and_load_skew(self):
        state = proxy.ProxyState(
            [("prefiller-0", 8001), ("prefiller-1", 8001)],
            [("decoder-0", 8002), ("decoder-1", 8002)],
            enable_remote_lmcache_store=True,
            enable_prefix_affinity_routing=True,
        )
        proxy.proxy_state = state
        _prime_remote_fill(state, decoder_idx=1, segment="decoder-1:12345")
        reservation = proxy.DecoderReservation(
            state.decoders[1],
            1,
            100.0,
            dp_rank=0,
            preferred_segment="decoder-1:12345",
            remote_fill={"destination_engine_epoch": 7},
        )
        full = proxy.PrefixAffinityAnchor("conversation", 20000)
        state.record_prefix_affinity((full,), 20000, 1, reservation, 20000)

        anchor, record, selected = state.resolve_prefix_affinity(
            (full,), 20000, 800.0, 2000.0
        )
        self.assertEqual((anchor, record.prefiller_idx, selected), (full, 1, True))

        state.prefillers[1].active_tokens = 10000
        state._update_prefiller_priority(1)
        self.assertFalse(
            state.resolve_prefix_affinity(
                (full,), 20000, 800.0, 2000.0
            )[2]
        )

        system = proxy.PrefixAffinityAnchor("system", 9000)
        state.record_prefix_affinity((system,), 9000, 1, reservation, 9000)
        self.assertFalse(
            state.resolve_prefix_affinity(
                (system,), 60000, 800.0, 2000.0
            )[2]
        )

        state.decoders[1].decoder_remote_fill[0]["destination_engine_epoch"] = 8
        self.assertEqual(
            state.resolve_prefix_affinity((full,), 20000, 800.0, 2000.0),
            (None, None, False),
        )
        self.assertNotIn(full.key, state.prefix_affinity)

    def test_terminal_persistence_learns_and_reuses_exact_placement(self):
        state = proxy.ProxyState(
            [("prefiller-0", 8001), ("prefiller-1", 8001)],
            [("decoder-0", 8002), ("decoder-1", 8002)],
            enable_remote_lmcache_store=True,
            enable_prefix_affinity_routing=True,
        )
        proxy.proxy_state = state
        _prime_remote_fill(state, decoder_idx=0, segment="decoder-0:12345")
        _prime_remote_fill(state, decoder_idx=1, segment="decoder-1:12345")
        anchors = (
            proxy.PrefixAffinityAnchor("full-prompt", 10000),
            proxy.PrefixAffinityAnchor("partial-prompt", 8192),
        )
        calls = []

        async def send(*args, **kwargs):
            calls.append((args[1], kwargs["preferred_mooncake_segment"]))
            handoff = kwargs["remote_fill_handoff"]
            return SimpleNamespace(
                json=lambda: {
                    "kv_transfer_params": {
                        "lmcache.remote_fill": {
                            "terminal": {
                                "outcome": "PERSISTENT_ONLY",
                                "persistent_common_end": 10000,
                                "required_store_end": 10000,
                                "transfer_id": handoff["transfer_id"],
                            }
                        }
                    }
                }
            )

        with patch.object(proxy, "send_request_to_service", side_effect=send):
            first = asyncio.run(
                proxy._handle_select_instance(
                    "/completions",
                    {"prompt": "x"},
                    10000,
                    prefix_anchors=anchors,
                )
            )
            self.assertEqual((first.prefiller_idx, first.decoder_idx), (0, 0))
            self.assertIn("partial-prompt", state.prefix_affinity)
            proxy._release_decoder_reservation(first)
            state.release_prefiller_kv(first.prefiller_idx, first.prefiller_score)

            second = asyncio.run(
                proxy._handle_select_instance(
                    "/completions",
                    {"prompt": "x"},
                    10000,
                    prefix_anchors=anchors,
                )
            )

        self.assertEqual((second.prefiller_idx, second.decoder_idx), (0, 0))
        self.assertEqual(calls, [(0, "decoder-0:12345")] * 2)
        proxy._release_decoder_reservation(second)
        state.release_prefiller_kv(second.prefiller_idx, second.prefiller_score)

    def test_remote_fill_only_placement_is_valid(self):
        placement = _remote_fill()
        placement.pop("destination_remote_session")
        payload = {
            "results": [
                {
                    "dp_rank": 0,
                    "api_dp_rank": 0,
                    "segment": None,
                    "remote_fill": placement,
                }
            ]
        }
        parsed = proxy._parse_decoder_remote_fill_response(payload)[0]
        self.assertEqual(parsed["destination_engine_epoch"], 7)
        self.assertNotIn("mooncake_preferred_segment", parsed)

    def test_remote_fill_placement_carries_decoder_mooncake_segment(self):
        payload = {
            "results": [
                {
                    "dp_rank": 0,
                    "api_dp_rank": 0,
                    "segment": "decoder:12345",
                    "remote_fill": _remote_fill(),
                }
            ]
        }

        placement = proxy._parse_decoder_remote_fill_response(payload)[0]

        self.assertEqual(
            placement["mooncake_preferred_segment"], "decoder:12345"
        )

    def test_remote_fill_placement_uses_advertised_native_session(self):
        payload = {
            "results": [
                {
                    "dp_rank": 0,
                    "api_dp_rank": 0,
                    "segment": None,
                    "remote_fill": _remote_fill(),
                }
            ]
        }

        placement = proxy._parse_decoder_remote_fill_response(payload)[0]

        self.assertEqual(
            placement["mooncake_preferred_segment"], "decoder:12345"
        )

    def test_remote_fill_parser_matches_qualified_example(self):
        remote_fill = _remote_fill(1)
        payload = {
            "results": [
                None,
                {
                    "dp_rank": 1,
                    "api_dp_rank": 0,
                    "segment": None,
                    "remote_fill": remote_fill,
                },
            ]
        }
        parsed = proxy._parse_decoder_remote_fill_response(payload)
        self.assertEqual(parsed[1].pop("api_dp_rank"), 0)
        self.assertEqual(
            parsed[1].pop("mooncake_preferred_segment"), "decoder:12345"
        )
        self.assertEqual(parsed, example_proxy._parse_decoder_remote_fill_response(payload))

    def test_remote_fill_parser_rejects_conflicting_mooncake_segment(self):
        payload = {
            "results": [
                {
                    "dp_rank": 0,
                    "api_dp_rank": 0,
                    "segment": "other-decoder:12345",
                    "remote_fill": _remote_fill(),
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "placement identities disagree"):
            proxy._parse_decoder_remote_fill_response(payload)

    def test_remote_fill_parser_requires_api_dp_rank(self):
        payload = {
            "results": [
                {"dp_rank": 1, "segment": None, "remote_fill": _remote_fill(1)}
            ]
        }
        with self.assertRaisesRegex(ValueError, "TP0/DP rank"):
            proxy._parse_decoder_remote_fill_response(payload)

    def test_external_dp_reservation_separates_api_and_remote_fill_ranks(self):
        state = proxy.ProxyState([("prefiller", 8001)], [("decoder", 8002)])
        proxy.proxy_state = state
        _prime_remote_fill(state, dp_rank=1, api_dp_rank=0)
        info = _reserve_instance(state)

        state.assign_decoder_rank(info.reservation)

        self.assertEqual(info.reservation.dp_rank, 1)
        self.assertEqual(info.reservation.api_dp_rank, 0)
        self.assertEqual(info.reservation.preferred_segment, "decoder:12345")
        self.assertEqual(info.reservation.remote_fill["destination_dp_rank"], 1)
        self.assertNotIn("api_dp_rank", info.reservation.remote_fill)
        proxy._release_decoder_reservation(info)

    def test_discovery_refresh_preserves_active_rank_load(self):
        state = proxy.ProxyState(
            [("prefiller", 8001)],
            [("decoder", 8002)],
            enable_remote_lmcache_store=True,
        )
        proxy.proxy_state = state
        decoder = _prime_remote_fill(state)
        decoder.decoder_remote_fill[1] = dict(decoder.decoder_remote_fill[0]) | {
            "destination_dp_rank": 1
        }
        decoder.decoder_rank_active_tokens = {0: 10.0, 1: 0.0}
        decoder.decoder_placement_discovered_at = 0.0
        with patch.object(
            proxy,
            "_discover_decoder_remote_fill",
            AsyncMock(return_value={1: decoder.decoder_remote_fill[1]}),
        ):
            asyncio.run(
                state.ensure_decoder_remote_fill(decoder, wait_for_result=True)
            )
        self.assertEqual(decoder.decoder_rank_active_tokens, {0: 10.0, 1: 0.0})
        state.release_decoder(0, 10.0, 0)
        self.assertEqual(decoder.decoder_rank_active_tokens, {1: 0.0})

    def test_stale_placement_refresh_does_not_block_selection(self):
        state = proxy.ProxyState(
            [("prefiller", 8001)],
            [("decoder", 8002)],
            enable_remote_lmcache_store=True,
        )
        proxy.proxy_state = state
        decoder = _prime_remote_fill(state)
        decoder.decoder_placement_discovered_at = 0.0
        decoder.decoder_placement_last_attempt_at = 0.0

        async def run() -> None:
            started = asyncio.Event()
            release = asyncio.Event()

            async def discover(*args, **kwargs):
                started.set()
                await release.wait()
                return decoder.decoder_remote_fill

            with patch.object(
                proxy,
                "_discover_decoder_remote_fill",
                side_effect=discover,
            ):
                await state.ensure_decoder_remote_fill(decoder)
                await started.wait()
                self.assertIsNotNone(decoder.decoder_placement_task)
                self.assertIn(0, decoder.decoder_remote_fill)
                release.set()
                await decoder.decoder_placement_task

        asyncio.run(run())

    def test_concurrent_requests_wait_for_the_same_initial_discovery(self):
        state = proxy.ProxyState(
            [("prefiller", 8001)], [("decoder", 8002)],
            enable_remote_lmcache_store=True,
        )
        proxy.proxy_state = state
        decoder = state.decoders[0]
        placement = _prime_remote_fill(state, dp_rank=1, api_dp_rank=0)
        result = dict(placement.decoder_remote_fill)
        decoder.decoder_remote_fill = {}
        decoder.decoder_placement_discovered_at = 0.0

        async def run():
            started, release = asyncio.Event(), asyncio.Event()

            async def discover(*args, **kwargs):
                started.set()
                await release.wait()
                return result

            async def reserve():
                await state.ensure_decoder_remote_fill(decoder, wait_for_result=True)
                reservation = proxy.DecoderReservation(decoder, 0, 1.0)
                return state.assign_decoder_rank(reservation)

            with patch.object(proxy, "_discover_decoder_remote_fill", side_effect=discover) as rpc:
                first = asyncio.create_task(reserve())
                await started.wait()
                followers = [asyncio.create_task(reserve()) for _ in range(15)]
                await asyncio.sleep(0)
                returned_early = sum(task.done() for task in followers)
                release.set()
                reservations = await asyncio.gather(first, *followers)
                self.assertEqual(returned_early, 0)
                self.assertEqual(rpc.await_count, 1)
                for reservation in reservations:
                    self.assertEqual(reservation.remote_fill["destination_dp_rank"], 1)
                    self.assertEqual(reservation.api_dp_rank, 0)
                    state.release_decoder(0, 1.0, reservation.dp_rank)
                await state.ensure_decoder_remote_fill(decoder, wait_for_result=True)
                self.assertEqual(rpc.await_count, 1)

        asyncio.run(run())

    def test_cancelling_discovery_waiter_keeps_shared_discovery_alive(self):
        state = proxy.ProxyState(
            [("prefiller", 8001)], [("decoder", 8002)],
            enable_remote_lmcache_store=True,
        )
        proxy.proxy_state = state
        decoder = state.decoders[0]

        async def run():
            started, release = asyncio.Event(), asyncio.Event()

            async def discover(*args, **kwargs):
                started.set()
                await release.wait()
                return {}  # Negative discovery retains its existing fallback/TTL.

            with patch.object(proxy, "_discover_decoder_remote_fill", side_effect=discover) as rpc:
                first = asyncio.create_task(state.ensure_decoder_remote_fill(decoder, wait_for_result=True))
                await started.wait()
                second = asyncio.create_task(state.ensure_decoder_remote_fill(decoder, wait_for_result=True))
                await asyncio.sleep(0)
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                task = decoder.decoder_placement_task
                waiting = not second.done()
                release.set()
                await asyncio.gather(second, task)
                self.assertTrue(waiting)
                self.assertFalse(task.cancelled())
                await state.ensure_decoder_remote_fill(decoder, wait_for_result=True)
                self.assertEqual(rpc.await_count, 1)

        asyncio.run(run())

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

        def select_decoder(score, preferred_idx=None, **kwargs):
            order.append("decoder")
            return original_select_decoder(score, preferred_idx, **kwargs)

        def select_prefiller(score, preferred_idx=None):
            order.append("prefiller")
            return original_select_prefiller(score, preferred_idx)

        state.select_decoder = select_decoder
        state.select_prefiller = select_prefiller

        async def send(*args, **kwargs):
            order.append("send")
            handoff = kwargs["remote_fill_handoff"]
            self.assertEqual(
                kwargs["preferred_mooncake_segment"], "decoder:12345"
            )
            return SimpleNamespace(
                content=b"{}",
                json=lambda: {
                    "kv_transfer_params": {
                        "remote_engine_id": "persistent",
                        "remote_block_ids": None,
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

    def test_first_request_lazily_discovers_remote_fill(self):
        state = proxy.ProxyState(
            [("prefiller", 8001)],
            [("decoder", 8002)],
            enable_remote_lmcache_store=True,
        )
        proxy.proxy_state = state
        placement = {
            key: value
            for key, value in _remote_fill().items()
            if key not in {"enabled", "tp_rank", "dp_rank"}
        } | {"api_dp_rank": 0, "destination_dp_rank": 0}

        async def send(*args, **kwargs):
            handoff = kwargs["remote_fill_handoff"]
            self.assertIsNotNone(handoff)
            return SimpleNamespace(
                json=lambda: {
                    "kv_transfer_params": {
                        "lmcache.remote_fill": {
                            "terminal": {
                                "outcome": "LOCAL_FULL",
                                "persistent_common_end": 1024,
                                "required_store_end": 1024,
                                "transfer_id": handoff["transfer_id"],
                            }
                        }
                    }
                }
            )

        req_data = {"prompt": "x"}
        with (
            patch.object(
                proxy,
                "_discover_decoder_remote_fill",
                AsyncMock(return_value={0: placement}),
            ) as discover,
            patch.object(proxy, "send_request_to_service", side_effect=send),
        ):
            info = asyncio.run(
                proxy._handle_select_instance("/completions", req_data, 1024)
            )

        discover.assert_awaited_once()
        self.assertEqual(info.reservation.dp_rank, 0)
        proxy._release_decoder_reservation(info)
        state.release_prefiller_kv(info.prefiller_idx, info.prefiller_score)

    def test_unavailable_remote_fill_accepts_null_prefiller_params(self):
        state = proxy.ProxyState(
            [("prefiller", 8001)],
            [("decoder", 8002)],
            enable_remote_lmcache_store=True,
        )
        proxy.proxy_state = state
        req_data = {"prompt": "x"}
        with (
            patch.object(
                proxy,
                "_discover_decoder_remote_fill",
                AsyncMock(return_value={}),
            ),
            patch.object(
                proxy,
                "send_request_to_service",
                AsyncMock(
                    return_value=SimpleNamespace(
                        json=lambda: {"kv_transfer_params": None}
                    )
                ),
            ),
        ):
            info = asyncio.run(
                proxy._handle_select_instance("/completions", req_data, 100)
            )

        self.assertEqual(req_data["kv_transfer_params"], {})
        self.assertIsNone(info.reservation.remote_fill)
        proxy._release_decoder_reservation(info)
        state.release_prefiller_kv(info.prefiller_idx, info.prefiller_score)

    def test_disabled_mode_keeps_prefiller_then_decoder_order(self):
        state = proxy.ProxyState([("prefiller", 8001)], [("decoder", 8002)])
        proxy.proxy_state = state
        order = []
        original_select_decoder = state.select_decoder
        original_select_prefiller = state.select_prefiller

        def select_decoder(score, preferred_idx=None, **kwargs):
            order.append("decoder")
            return original_select_decoder(score, preferred_idx, **kwargs)

        def select_prefiller(score, preferred_idx=None):
            order.append("prefiller")
            return original_select_prefiller(score, preferred_idx)

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

    def test_decoder_transport_sends_api_dp_rank(self):
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
                yield b"data: [DONE]\n\n"

        class Client:
            def stream(self, method, endpoint, **kwargs):
                captured.update(method=method, endpoint=endpoint, **kwargs)
                return Response()

        async def collect() -> list[bytes]:
            return [
                chunk
                async for chunk in proxy.stream_decoder_response(
                    Client(),
                    "/completions",
                    {},
                    "request",
                    decoder_api_dp_rank=3,
                )
            ]

        self.assertEqual(
            asyncio.run(collect()),
            [b"data: {}\n\n", b"data: [DONE]\n\n"],
        )
        self.assertEqual(captured["headers"]["X-data-parallel-rank"], "3")
        self.assertEqual(captured["json"], {})

    def test_decoder_transport_rejects_clean_eof_without_done(self):
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
            def stream(self, *args, **kwargs):
                return Response()

        async def collect() -> None:
            async for _ in proxy.stream_decoder_response(
                Client(), "/completions", {}, "request"
            ):
                pass

        with self.assertRaisesRegex(RuntimeError, "without \\[DONE\\]"):
            asyncio.run(collect())

    def test_decoder_transport_does_not_retry_after_streaming_started(self):
        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                yield b"data: 1\n\n"
                raise httpx.ReadError(
                    "stream failed",
                    request=httpx.Request("POST", "http://decoder"),
                )

        class Client:
            calls = 0

            def stream(self, *args, **kwargs):
                self.calls += 1
                return Response()

        client = Client()

        chunks = []

        async def collect() -> None:
            async for chunk in proxy.stream_decoder_response(
                    client,
                    "/completions",
                    {},
                    "request",
                ):
                chunks.append(chunk)

        with self.assertRaises(httpx.ReadError):
            asyncio.run(collect())
        self.assertEqual(chunks, [b"data: 1\n\n"])
        self.assertEqual(client.calls, 1)

    def test_prefiller_timeout_restores_unsent_abort_ids(self):
        state = proxy.ProxyState([("prefiller", 8001)], [("decoder", 8002)])
        proxy.proxy_state = state
        state.prefillers[0].aborted_requests.add("old-request")
        proxy.global_args.backend_request_timeout = 0.01

        class Client:
            calls = 0

            async def post(self, *args, **kwargs):
                self.calls += 1
                await asyncio.Future()

        client = Client()
        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(
                proxy.send_request_to_service(
                    client,
                    0,
                    "/completions",
                    {},
                    "request",
                )
            )
        self.assertEqual(client.calls, 1)
        self.assertEqual(
            state.prefillers[0].aborted_requests,
            {"old-request"},
        )

    def test_nonstream_decoder_response_is_returned_eagerly(self):
        state = proxy.ProxyState([("prefiller", 8001)], [("decoder", 8002)])
        proxy.proxy_state = state
        info = _reserve_instance(state)
        info.reservation.dp_rank = 1
        info.reservation.api_dp_rank = 0
        request = SimpleNamespace(
            json=AsyncMock(return_value={"prompt": "hello", "stream": False}),
            body=AsyncMock(return_value=b"hello"),
        )
        backend_response = httpx.Response(
            200,
            json={"choices": [{"text": "ok", "finish_reason": "stop"}]},
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "http://decoder"),
        )
        post = AsyncMock(return_value=backend_response)
        with (
            patch.object(
                proxy,
                "_handle_select_instance",
                AsyncMock(return_value=info),
            ),
            patch.object(
                info.decoder.client,
                "post",
                post,
            ),
        ):
            response = asyncio.run(
                proxy._handle_completions("/completions", request)
            )

        self.assertNotIsInstance(response, proxy.StreamingResponse)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["choices"][0]["text"], "ok")
        self.assertEqual(post.await_args.kwargs["headers"]["X-data-parallel-rank"], "0")
        self.assertEqual(state.request_num, 0)
        self.assertEqual(state.prefillers[0].active_kv_cache, 0)
        self.assertEqual(state.decoders[0].active_tokens, 0)

    def test_stream_failure_emits_error_and_done(self):
        state = proxy.ProxyState([("prefiller", 8001)], [("decoder", 8002)])
        proxy.proxy_state = state
        info = _reserve_instance(state)
        info.reservation.dp_rank = 1
        info.reservation.api_dp_rank = 0
        request = SimpleNamespace(
            json=AsyncMock(return_value={"prompt": "hello", "stream": True}),
            body=AsyncMock(return_value=b"hello"),
        )

        async def stream(*args, **kwargs):
            self.assertEqual(kwargs["decoder_api_dp_rank"], 0)
            yield b'data: {"choices":[{"text":"x"}]}\n\n'
            raise httpx.ReadError(
                "stream failed",
                request=httpx.Request("POST", "http://decoder"),
            )

        with (
            patch.object(
                proxy,
                "_handle_select_instance",
                AsyncMock(return_value=info),
            ),
            patch.object(proxy, "stream_decoder_response", stream),
        ):
            response = asyncio.run(
                proxy._handle_completions("/completions", request)
            )

            async def consume() -> list[bytes]:
                return [chunk async for chunk in response.body_iterator]

            chunks = asyncio.run(consume())

        self.assertIn(b'"code":"decoder_backend_error"', chunks[-2])
        self.assertEqual(chunks[-1], b"data: [DONE]\n\n")
        self.assertEqual(state.request_num, 0)
        self.assertEqual(state.prefillers[0].active_kv_cache, 0)
        self.assertEqual(state.decoders[0].active_tokens, 0)

    def test_recompute_after_forwarded_event_fails_without_replacement(self):
        state = proxy.ProxyState([("prefiller", 8001)], [("decoder", 8002)])
        proxy.proxy_state = state
        info = _reserve_instance(state)
        request = SimpleNamespace(
            json=AsyncMock(return_value={"prompt": "hello", "stream": True}),
            body=AsyncMock(return_value=b"hello"),
        )

        async def stream(*args, **kwargs):
            yield b'data: {"choices":[{"text":"visible"}]}\n\n'
            yield (
                b'data: {"choices":[{"delta":{},'
                b'"finish_reason":"recomputed"}]}\n\n'
            )

        select = AsyncMock(return_value=info)
        with (
            patch.object(proxy, "_handle_select_instance", select),
            patch.object(proxy, "stream_decoder_response", stream),
        ):
            response = asyncio.run(
                proxy._handle_completions("/completions", request)
            )

            async def consume() -> list[bytes]:
                return [chunk async for chunk in response.body_iterator]

            chunks = asyncio.run(consume())

        self.assertEqual(select.await_count, 1)
        self.assertEqual(len(chunks), 3)
        self.assertIn(b'"code":"decoder_recompute_error"', chunks[-2])
        self.assertEqual(chunks[-1], b"data: [DONE]\n\n")
        self.assertEqual(state.request_num, 0)
        self.assertEqual(state.prefillers[0].active_kv_cache, 0)
        self.assertEqual(state.decoders[0].active_tokens, 0)

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
            patch.object(proxy, "stream_decoder_response", stream),
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
