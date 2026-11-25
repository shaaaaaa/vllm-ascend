# SPDX-License-Identifier: Apache-2.0
"""Fixed-topology decoder routing. No KV lookups or network I/O on the decision path."""

import hashlib
import json
import math
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass

SESSION_HEADER = "x-falconkv-session-id"
MAX_KEY_BYTES = 256
FIRST_USER_CHARS = 1024
MAX_HISTORY = 65536
HISTORY_TTL = 1800
QUEUE_SAMPLES = 2
WINDOW_SECONDS = 60
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc")


def digest(*parts):
    # JSON array framing prevents namespace/key/endpoint concatenation collisions.
    return hashlib.blake2s(
        json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode(), digest_size=16
    ).digest()


def routing_key(headers, body, explicit_anchors, default_model):
    """Session identity is a routing hint, never a claimed number of cached tokens."""
    model = body.get("model") or default_model or ""
    sid = headers.get(SESSION_HEADER)
    if isinstance(sid, str) and 0 < len(sid.encode()) <= MAX_KEY_BYTES and sid.strip():
        return digest(model, "session", sid), "session"
    if explicit_anchors:
        anchor = min(explicit_anchors, key=lambda a: (a.token_end, a.key))
        return digest(model, "anchor", anchor.key), "anchor"
    # For clients without headers, avoid hashing the whole changing JSON body
    # or just the shared system/tool preamble. Bounded, approximate, not a session ID.
    for msg in body.get("messages") or ():
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = msg.get("content")
            if isinstance(text, list):
                text = next((p.get("text") for p in text if isinstance(p, dict) and p.get("text")), None)
            if isinstance(text, str) and text.strip():
                return digest(model, "first_user", text[:FIRST_USER_CHARS]), "first_user"
            break
    return None, "missing"


def parse_gauges(text):
    """One engine per endpoint. Missing, duplicated, and invalid series are unknown, never zero."""
    values = {}
    for line in text.splitlines():
        if not line.startswith(GAUGES):
            continue
        match = re.fullmatch(r"(vllm:\w+)(?:\{.*\})?\s+(\S+)(?:\s+\d+)?", line)
        if not match or match[1] not in GAUGES:
            continue
        name, raw = match.groups()
        val = float(raw)
        if name in values or not math.isfinite(val) or val < 0:
            raise ValueError("ambiguous or invalid routing gauge")
        values[name] = val
    if set(values) != set(GAUGES):
        raise ValueError("incomplete decoder routing gauges")
    r, w, kv = (values[g] for g in GAUGES)
    if not (r.is_integer() and w.is_integer() and kv <= 1):
        raise ValueError("invalid routing gauge units")
    return int(r), int(w), kv


@dataclass
class Sample:
    running: int
    waiting: int
    kv: float
    timestamp: float
    handoffs: int  # monotonic send count at scrape start


@dataclass
class Decision:
    chosen: int | None  # None: use the unmodified legacy policy
    reason: str
    pair: tuple
    warm: int | None
    loads: list


class DecoderRouter:
    """Owned by the proxy; all mutations under ProxyState's reservation lock or on its event loop."""

    def __init__(self, endpoints, mode, capacity, interval, max_age, kv_limit):
        self.endpoints = tuple((h, int(p)) for h, p in endpoints)
        self.hosts = sorted({h for h, _ in endpoints})
        self.groups = [[i for i, (h, _) in enumerate(endpoints) if h == host] for host in self.hosts]
        if mode not in {"legacy", "global", "dual-home"} or len(set(self.endpoints)) != len(endpoints):
            raise ValueError("invalid decoder policy or duplicate decoder endpoints")
        if mode == "dual-home" and len(self.hosts) != 2:
            raise ValueError("dual-home requires exactly two physical decoder hosts (one address per host)")
        if capacity < 1 or not (0 < interval < max_age < math.inf) or not (0 < kv_limit <= 1):
            raise ValueError("invalid route capacity/interval/max-age/KV limit")
        self.mode, self.capacity = mode, capacity
        self.interval, self.max_age, self.kv_limit = interval, max_age, kv_limit
        self.samples = [None] * len(endpoints)
        self.pending = [0] * len(endpoints)  # reserved but not sent to D
        self.handoffs = [0] * len(endpoints)
        self.streak = [0] * len(endpoints)
        self.congested = [False] * len(endpoints)
        self.history = OrderedDict()
        self.counts = Counter()
        self.started = time.time()
        self.last_tick = None
        self.last_coverage = (False, False)

    def pair(self, key):
        if key is None or len(self.groups) != 2:
            return ()
        return tuple(max(group, key=lambda i: digest(key.hex(), *self.endpoints[i])) for group in self.groups)

    def observe(self, i, values, started, handoffs):
        prev = self.samples[i]
        if prev is None or started - prev.timestamp > self.max_age or values is None:
            self.streak[i], self.congested[i] = 0, False
        if values is None:
            self.samples[i] = None
            return
        self.samples[i] = Sample(*values, started, handoffs)
        if values[1]:
            self.streak[i] = max(0, self.streak[i]) + 1
            if self.streak[i] >= QUEUE_SAMPLES:
                self.congested[i] = True
        else:
            self.streak[i] = min(0, self.streak[i]) - 1
            if -self.streak[i] >= QUEUE_SAMPLES:
                self.congested[i] = False

    def remember(self, key, i, epoch, now):
        # Weak, bounded evidence of a *successful* D attempt, not page residency.
        if key is not None and epoch is not None:
            slot = (key, i)
            self.history[slot] = (epoch, now)
            self.history.move_to_end(slot)
            if len(self.history) > MAX_HISTORY:
                self.history.popitem(last=False)

    def choose(self, key, healthy, affinity, epochs, active, scores, now):
        pair = self.pair(key)
        loads = []
        for i, s in enumerate(self.samples):
            if s is None or now - s.timestamp > self.max_age:
                loads.append(None)
                continue
            # Upper bound at handoff boundary: don't add all active to backend counts.
            incoming = self.pending[i] + self.handoffs[i] - s.handoffs
            used = max(active[i], s.running + s.waiting + incoming)
            loads.append(
                dict(
                    r=s.running,
                    w=s.waiting,
                    kv=round(s.kv, 4),
                    p=self.pending[i],
                    active=active[i],
                    incoming=incoming,
                    free=self.capacity - used,
                    age=round(now - s.timestamp, 3),
                )
            )
        if self.mode == "legacy":
            return Decision(None, "legacy", pair, affinity, loads)
        if not healthy or any(loads[i] is None for i in healthy):
            return Decision(None, "stale", pair, affinity, loads)
        if self.mode == "dual-home" and key is None:
            return Decision(None, "missing_key", pair, affinity, loads)
        pool = [i for i in (pair if self.mode == "dual-home" else healthy) if i in healthy]
        # Higher quality prefix evidence first, then recently served session, subject to epoch/TTL.
        recent = [
            (t, i)
            for i in pool
            if (entry := self.history.get((key, i))) is not None
            for ep, t in (entry,)
            if ep == epochs[i] and now - t <= HISTORY_TTL
        ]
        warm = affinity if affinity in pool else (max(recent)[1] if recent else None)

        def room(i):
            return loads[i]["free"] > 0 and self.samples[i].kv < self.kv_limit and not self.congested[i]

        def cost(i):
            s = self.samples[i]
            return (
                s.kv >= self.kv_limit,
                self.congested[i],
                -loads[i]["free"],
                s.waiting,
                scores[i],
                digest((key or b"").hex(), *self.endpoints[i]),
            )

        available = [i for i in pool if room(i)]
        elsewhere = [i for i in healthy if i not in pool and room(i)]
        if warm is not None and warm in available:
            return Decision(warm, "warm", pair, warm, loads)
        if not available and elsewhere:
            return Decision(min(elsewhere, key=cost), "escape", pair, warm, loads)
        chosen = min(available or pool or healthy, key=cost)
        reason = (
            "load"
            if warm is None or warm == chosen
            else ("queue" if self.congested[warm] else "kv" if self.samples[warm].kv >= self.kv_limit else "reserved")
        )
        return Decision(chosen, reason, pair, warm, loads)

    def count_decision(self, decision, actual, source):
        self.counts["decisions"] += 1
        self.counts[f"reason:{decision.reason}"] += 1
        self.counts[f"key:{source}"] += 1
        self.counts[f"dp:{actual}"] += 1
        if decision.warm is not None:
            self.counts["warm_available"] += 1
            self.counts["warm_chosen"] += int(actual == decision.warm)
        if decision.pair:
            self.counts["pair:" + ",".join(map(str, decision.pair))] += 1

    def tick(self, now):
        """Time-weight previous complete observation, capping long gaps; never impute missing seconds."""
        if self.last_tick is not None:
            dt = now - self.last_tick
            valid, skew = self.last_coverage
            observed = min(dt, self.interval * 1.5) if valid else 0
            self.counts["valid_s"] += observed
            self.counts["missing_s"] += dt - observed
            self.counts["skew_s"] += observed if skew else 0
        self.last_tick = now
        valid = all(s and now - s.timestamp <= self.max_age for s in self.samples)
        skew = bool(
            valid
            and len(self.groups) == 2
            and any(
                all(self.samples[i].running >= self.capacity and self.samples[i].waiting > 0 for i in g)
                and any(
                    self.samples[i].running < self.capacity
                    and self.samples[i].waiting == 0
                    and self.samples[i].kv < self.kv_limit
                    for i in self.groups[1 - n]
                )
                for n, g in enumerate(self.groups)
            )
        )
        self.last_coverage = valid, skew

    def flush(self, now, *, force=False):
        if not force and now - self.started < WINDOW_SECONDS:
            return None
        record = dict(start=self.started, end=now, counts=dict(self.counts))
        self.started, self.counts = now, Counter()
        return record
