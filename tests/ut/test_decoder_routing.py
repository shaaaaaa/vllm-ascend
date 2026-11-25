# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from itertools import permutations
from pathlib import Path

from examples.disaggregated_prefill_v1.decoder_routing import GAUGES, DecoderRouter, parse_gauges, routing_key
from examples.disaggregated_prefill_v1.summarize_decoder_routes import benchmark, summarize


ENDPOINTS = [("a", 1), ("a", 2), ("b", 1), ("b", 2)]


def make_router(policy="dual-home", topology=ENDPOINTS):
    return DecoderRouter(topology, policy, 16, 2, 6, 0.95)


class TestDecoderRouting(unittest.TestCase):
    def setUp(self):
        self.r = make_router()
        self.key, self.source = routing_key({"x-falconkv-session-id": "S"}, {}, (), "m")
        self.home, self.backup = self.r.pair(self.key)
        self.observe(self.home, 10)
        for i in range(4):
            if i != self.home:
                self.observe(i, 6)

    def observe(self, i, run, wait=0, kv=0.5, t=1):
        self.r.observe(i, (run, wait, kv), t, self.r.handoffs[i])

    def choose(self, now=2, key=True):
        return self.r.choose(self.key if key else None, list(range(4)), self.home, [7] * 4, [0] * 4, [0] * 4, now)

    def test_pair_stability_across_order_turns_and_sources(self):
        endpoints = {ENDPOINTS[i] for i in self.r.pair(self.key)}
        for perm in permutations(ENDPOINTS):
            self.assertEqual({perm[i] for i in make_router(topology=perm).pair(self.key)}, endpoints)
        self.assertNotEqual(ENDPOINTS[self.home][0], ENDPOINTS[self.backup][0])
        chat = {"messages": [{"role": "system", "content": "common"}, {"role": "user", "content": "first"}]}
        key, source = routing_key({}, chat, (), "m")
        chat["messages"].append({"role": "user", "content": "followup"})
        self.assertEqual((key, source), routing_key({}, chat, (), "m"))
        self.assertEqual(source, "first_user")
        self.assertNotEqual(self.key, routing_key({"x-falconkv-session-id": "S"}, chat, (), "other")[0])

    def test_affinity_then_queue_switch_and_recovery(self):
        self.assertEqual(self.choose().chosen, self.home)
        self.observe(self.home, 10, wait=1, t=2)
        self.assertEqual(self.choose().chosen, self.home)
        self.observe(self.home, 10, wait=1, t=3)
        self.assertEqual((self.choose(3).chosen, self.choose(3).reason), (self.backup, "queue"))
        self.observe(self.home, 10, t=4)
        self.assertEqual(self.choose(4).chosen, self.backup)
        self.observe(self.home, 10, t=5)
        self.assertEqual(self.choose(5).chosen, self.home)

    def test_pending_capacity_handoff_and_escape(self):
        self.r.pending[self.home] = 6
        self.assertEqual(self.choose().reason, "reserved")
        self.r.pending[self.home] -= 1
        self.r.handoffs[self.home] += 1
        self.assertEqual(self.choose().loads[self.home]["free"], 0)
        self.observe(self.home, 11, t=2)
        self.assertEqual(self.choose().loads[self.home]["free"], 0)  # now included in engine sample
        self.observe(self.backup, 8, kv=0.97)
        result = self.choose()
        self.assertEqual(result.reason, "escape")
        self.assertNotIn(result.chosen, result.pair)

    def test_fallback_and_weak_history(self):
        self.assertEqual(self.choose(key=False).reason, "missing_key")
        self.assertEqual(self.choose(10).reason, "stale")
        self.r.remember(self.key, self.home, 7, 1)

        def pick(epoch):
            return self.r.choose(self.key, list(range(4)), None, [epoch] * 4, [0] * 4, [0] * 4, 2)

        self.assertEqual(pick(7).chosen, self.home)
        self.assertEqual(pick(8).chosen, self.backup)
        self.r.observe(self.backup, None, 2, 0)
        self.assertIsNone(self.choose().chosen)

    def test_real_queue_skew_and_missing_duration(self):
        for i in (0, 1):
            self.observe(i, 16, wait=4)
        self.r.tick(1)
        self.r.observe(2, None, 3, 0)
        self.r.tick(3)
        self.r.tick(5)
        self.assertEqual((self.r.counts["valid_s"], self.r.counts["skew_s"], self.r.counts["missing_s"]), (2, 2, 2))

    def test_global_vs_dual_home_use_same_signal_but_different_pool(self):
        outsider = next(i for i in range(4) if i not in (self.home, self.backup))
        forecasts = []
        for policy in ("dual-home", "global"):
            router = make_router(policy)
            for i in range(4):
                router.observe(i, (0 if i == outsider else 9, 0, 0.2), 1, 0)
            forecasts.append(router.choose(self.key, list(range(4)), None, [7] * 4, [0] * 4, [0] * 4, 2))
        self.assertIn(forecasts[0].chosen, (self.home, self.backup))
        self.assertEqual(forecasts[1].chosen, outsider)

    def test_client_latency_from_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench.log"
            path.write_text(
                "id ✓ in=   13 out=264 TTFT=15662.4ms TPOT=   32.9ms ITL= -ms\n"
                "id ✓ in=  111 out=1   TTFT=   10.2ms TPOT=      -ms ITL= -ms\n",
                encoding="utf-8",
            )
            result = benchmark(str(path))
        self.assertEqual(result["ttft"]["p99_ms"], 15662.4)
        self.assertEqual(result["tpot"]["n"], 1)

    def test_strict_metrics(self):
        raw = "\n".join(f'{name}{{model_name="a",engine="2"}} {val}' for name, val in zip(GAUGES, (8, 4, 0.92)))
        self.assertEqual(parse_gauges(raw), (8, 4, 0.92))
        for bad in (raw.split("\n", 1)[1], raw + "\n" + raw, raw.replace("0.92", "NaN")):
            with self.assertRaises(ValueError):
                parse_gauges(bad)

    def test_summary_keeps_drain_separate_and_rejects_partial_windows(self):
        def log(kind, **data):
            return f"... ROUTE_{kind} {json.dumps(data)}"

        lines = [
            log("COMPLETE", arrived=2, ts=3),
            log("COMPLETE", arrived=2, ts=130),
            log("COMPLETE", arrived=-10, ts=2),
            log("WINDOW", start=1, end=61, counts={"valid_s": 45, "missing_s": 15, "skew_s": 9}),
            log("WINDOW", start=61, end=121, counts={}),
        ]
        report = summarize(lines, 1, 120)
        self.assertEqual(report["stats_coverage_pct"], 75)
        self.assertEqual(report["host_skew_pct_of_valid"], 20)
        self.assertEqual((report["proxy_success_in_window"], report["proxy_success_during_drain"]), (1, 1))
        self.assertEqual(report["excluded_edge_windows"], 1)
