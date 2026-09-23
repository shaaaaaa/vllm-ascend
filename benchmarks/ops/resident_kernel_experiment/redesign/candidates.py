# SPDX-License-Identifier: Apache-2.0
"""Executable, correctness-first lookup candidates. No serving dispatch changes.

The Torch implementations are algorithm references, not sub-10us kernels.
Inputs and snapshot must stay immutable until all consumers finish. Token IDs
are request-local positions, not vocabulary IDs. Versions identify the KV value
at that position (in particular after a speculative rollback).
"""
from dataclasses import dataclass, replace
from typing import Sequence

import torch

MISS, INVALID, TAIL, DENSE = -1, -2, -3, -4
VARIANTS = (
    "reload", "fixed_position", "bounded_position", "hash_snapshot",
    "direct_directory", "bitmap_directory", "sorted_snapshot", "cube_join",
    "full_hbm",
)


@dataclass(frozen=True)
class Query:
    tokens: torch.Tensor          # int32 [B,Q,K], Q=2 means one speculative token
    versions: torch.Tensor        # int32 [B,Q,K]
    active: torch.Tensor          # bool [B,Q]
    boundary: torch.Tensor        # int32 [B,Q], first non-offloaded position
    lengths: torch.Tensor         # int32 [B,Q], causal upper bound (exclusive)
    epochs: torch.Tensor          # int64 [B], request-slot incarnation
    universe: int                 # backing source size per request

    def validate(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError("tokens must be [requests, query_rows, topk]")
        b, q, k = self.tokens.shape
        if b < 1 or q not in (1, 2) or k < 1 or not 0 < self.universe < 2**31:
            raise ValueError("invalid request/query/top-k/universe geometry")
        specs = ((self.tokens, (b, q, k), torch.int32),
                 (self.versions, (b, q, k), torch.int32),
                 (self.active, (b, q), torch.bool),
                 (self.boundary, (b, q), torch.int32),
                 (self.lengths, (b, q), torch.int32),
                 (self.epochs, (b,), torch.int64))
        for x, shape, dtype in specs:
            if tuple(x.shape) != shape or x.dtype != dtype or x.device != self.tokens.device:
                raise ValueError("query shape/dtype/device mismatch")

    def masks(self):
        good_meta = ((self.boundary >= 0) & (self.boundary <= self.lengths)
                     & (self.lengths <= self.universe))
        valid = (self.active & good_meta)[..., None] & (self.tokens >= 0)
        valid = valid & (self.tokens < self.lengths[..., None])
        offload = valid & (self.tokens < self.boundary[..., None])
        return valid.flatten(1), offload.flatten(1)

    def to(self, device):
        return Query(*(x.to(device) for x in (
            self.tokens, self.versions, self.active, self.boundary,
            self.lengths, self.epochs)), self.universe)


@dataclass(frozen=True)
class Snapshot:
    tokens: torch.Tensor          # int32 [B,N], occurrence-ordered bank
    versions: torch.Tensor        # int32 [B,N]
    ready: torch.Tensor           # bool [B,N], publish only after complete copy
    epochs: torch.Tensor          # int64 [B]
    kv: torch.Tensor              # [B,N,D], normalized interleaved row payload

    def validate(self, query: Query) -> None:
        b, q, k = query.tokens.shape
        n = q * k
        for x, dtype in ((self.tokens, torch.int32), (self.versions, torch.int32),
                         (self.ready, torch.bool)):
            if x.shape != (b, n) or x.dtype != dtype or x.device != query.tokens.device:
                raise ValueError("snapshot must have one slot per query occurrence")
        if self.epochs.shape != (b,) or self.epochs.dtype != torch.int64:
            raise ValueError("invalid snapshot epochs")
        if self.kv.ndim != 3 or self.kv.shape[:2] != (b, n):
            raise ValueError("invalid snapshot payload geometry")
        if self.kv.device != query.tokens.device or self.epochs.device != query.tokens.device:
            raise ValueError("snapshot device mismatch")

    def to(self, device):
        return Snapshot(*(x.to(device) for x in (
            self.tokens, self.versions, self.ready, self.epochs, self.kv)))


@dataclass(frozen=True)
class Plan:
    source: torch.Tensor          # [B,N] old slot or MISS/INVALID/TAIL/DENSE
    variant: str


def _verified(query, snap, candidates):
    """Verify every candidate, including table collisions and stale slots."""
    b, n = snap.tokens.shape
    c = candidates.long()
    safe = c.clamp(0, n - 1)
    tags = snap.tokens.gather(1, safe)
    versions = snap.versions.gather(1, safe)
    ready = snap.ready.gather(1, safe)
    ok = ((c >= 0) & (c < n) & ready & (tags == query.tokens.flatten(1))
          & (versions == query.versions.flatten(1))
          & (snap.epochs == query.epochs)[:, None])
    return torch.where(ok, c, MISS)


def _finish(query, source, variant):
    valid, offload = query.masks()
    source = torch.where(offload, source, TAIL)
    return Plan(torch.where(valid, source, INVALID).to(torch.int32), variant)


def _table(query, snap, size, *, direct):
    # Integer last-writer selection. No races between duplicate token entries:
    # scatter_reduce chooses the greatest source slot deterministically.
    b, n = snap.tokens.shape
    ids = torch.arange(n, device=snap.tokens.device, dtype=torch.int64).expand(b, n)
    eligible = snap.ready & (snap.tokens >= 0) & (snap.tokens < query.universe)
    bucket = snap.tokens.long().clamp_min(0)
    bucket = bucket.clamp_max(size - 1) if direct else bucket.remainder(size)
    table = torch.full((b, size), MISS, dtype=torch.int64, device=ids.device)
    table.scatter_reduce_(1, bucket, torch.where(eligible, ids, MISS),
                          reduce="amax", include_self=True)
    return table


def signed_bits(tokens, versions):
    """64 small exact +/-1 terms encode the complete two-int32 identity."""
    shifts = torch.arange(32, device=tokens.device, dtype=torch.int64)
    def encode(x):
        return (((x.long()[..., None] >> shifts) & 1) * 2 - 1).to(torch.float16)
    return torch.cat((encode(tokens), encode(versions)), dim=-1)


def lookup(query: Query, snap: Snapshot, variant: str, *, radius=2,
           buckets=8192, tile=128, validate=True) -> Plan:
    """Return an immutable-snapshot plan; all preprocessing is included here.

    No data-dependent host reads. Shape validation may be disabled after setup.
    The floating-point matrix candidate is exact because all 64 summed products
    are +/-1, with every intermediate integer representable in FP16.
    """
    if validate:
        query.validate()
        snap.validate(query)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    if radius < 0 or radius > 32 or buckets < 1 or tile < 1:
        raise ValueError("invalid radius/buckets/tile")
    b, q, k = query.tokens.shape
    n = q * k
    tokens, versions = query.tokens.flatten(1), query.versions.flatten(1)
    device = tokens.device
    empty = torch.full((b, n), MISS, dtype=torch.int64, device=device)
    if variant == "full_hbm":
        valid, _ = query.masks()
        return Plan(torch.where(valid, torch.full_like(empty, DENSE), INVALID).int(), variant)
    if variant == "reload":
        return _finish(query, empty, variant)
    if variant in ("fixed_position", "bounded_position"):
        pos = torch.arange(n, device=device).expand(b, n)
        source = _verified(query, snap, pos)
        if variant == "bounded_position":
            # No wrapping at query-row ends. Other query rows are probed too.
            rank, row = pos.remainder(k), pos.div(k, rounding_mode="floor")
            deltas = [0] + [d for i in range(1, radius + 1) for d in (i, -i)]
            for other in range(q):
                for delta in deltas:
                    candidate_rank = rank + delta
                    candidate = (row + other).remainder(q) * k + candidate_rank
                    candidate = torch.where((candidate_rank >= 0) & (candidate_rank < k), candidate, MISS)
                    match = _verified(query, snap, candidate)
                    source = torch.where(source >= 0, source, match)
        return _finish(query, source, variant)
    if variant in ("hash_snapshot", "direct_directory", "bitmap_directory"):
        direct = variant != "hash_snapshot"
        size = query.universe if direct else buckets
        table = _table(query, snap, size, direct=direct)
        index = tokens.long().clamp(0, query.universe - 1)
        if not direct:
            index = index.remainder(size)
        candidate = table.gather(1, index)
        if variant == "bitmap_directory":
            # Explicit one-bit-per-position bitmap, built from the directory.
            # No non-atomic bitset scatter from independent token writers.
            pad = (-size) % 32
            occupied = torch.nn.functional.pad(table >= 0, (0, pad))
            powers = 1 << torch.arange(32, device=device, dtype=torch.int64)
            words = (occupied.reshape(b, -1, 32).long() * powers).sum(-1)
            word = words.gather(1, index.div(32, rounding_mode="floor"))
            present = ((word >> index.remainder(32)) & 1).bool()
            candidate = torch.where(present, candidate, MISS)
        return _finish(query, _verified(query, snap, candidate), variant)
    if variant == "sorted_snapshot":
        # Token/version pair sorted as two unsigned halves without FP casting.
        # Sorting each step is intentionally timed, not hidden as free setup.
        eligible = snap.ready & (snap.tokens >= 0) & (snap.tokens < query.universe)
        # Non-negative token ID <2**31 makes this signed int64 key overflow-safe.
        keys = (snap.tokens.long().clamp_min(0) << 32) | (snap.versions.long() & 0xffffffff)
        keys = torch.where(eligible, keys, torch.iinfo(torch.int64).max)
        sorted_keys, order = keys.sort(dim=1, stable=True)
        wanted = (tokens.long().clamp_min(0) << 32) | (versions.long() & 0xffffffff)
        positions = torch.searchsorted(sorted_keys.contiguous(), wanted.contiguous()).clamp_max(n - 1)
        candidate = order.gather(1, positions)
        return _finish(query, _verified(query, snap, candidate), variant)
    # Streaming matrix equality join: never materialize the full N x N result.
    previous = signed_bits(snap.tokens, snap.versions).transpose(1, 2)
    encoded = signed_bits(tokens, versions)
    sources = []
    for start in range(0, n, tile):
        score = torch.matmul(encoded[:, start:start + tile], previous)
        equal = (score == 64) & snap.ready[:, None, :]
        first = equal.to(torch.int32).argmax(dim=-1)
        sources.append(torch.where(equal.any(dim=-1), first, MISS))
    return _finish(query, _verified(query, snap, torch.cat(sources, dim=1)), variant)


def oracle_sources(query: Query, snap: Snapshot) -> torch.Tensor:
    """Independent CPU dictionary oracle; not used by any lookup candidate."""
    query, snap = query.to("cpu"), snap.to("cpu")
    query.validate()
    snap.validate(query)
    b, q, k = query.tokens.shape
    out = torch.full((b, q * k), INVALID, dtype=torch.int32)
    for request in range(b):
        previous = {}
        if int(query.epochs[request]) == int(snap.epochs[request]):
            for i, (t, v, ready) in enumerate(zip(
                    snap.tokens[request].tolist(), snap.versions[request].tolist(),
                    snap.ready[request].tolist(), strict=True)):
                if ready and 0 <= t < query.universe:
                    previous[t, v] = i
        for row in range(q):
            boundary, length = int(query.boundary[request, row]), int(query.lengths[request, row])
            if not query.active[request, row] or not 0 <= boundary <= length <= query.universe:
                continue
            for j, (t, v) in enumerate(zip(query.tokens[request, row].tolist(),
                                           query.versions[request, row].tolist(), strict=True)):
                if 0 <= t < length:
                    out[request, row * k + j] = (TAIL if t >= boundary else previous.get((t, v), MISS))
    return out


def assert_safe(query: Query, snap: Snapshot, plan: Plan) -> None:
    """Accept false misses, never false hits. Does not require old miss order."""
    expected = oracle_sources(query, snap)
    got = plan.source.detach().cpu()
    if got.shape != expected.shape or got.dtype != torch.int32:
        raise AssertionError("invalid plan shape/dtype")
    valid, offload = query.to("cpu").masks()
    if plan.variant == "full_hbm":
        torch.testing.assert_close(got, torch.where(valid, DENSE, INVALID).int())
        return
    torch.testing.assert_close(got == INVALID, expected == INVALID)
    torch.testing.assert_close(got == TAIL, expected == TAIL)
    hits = got >= 0
    if bool((got >= snap.tokens.shape[1]).any()) or bool(((got < TAIL) & (got != INVALID)).any()):
        raise AssertionError("out-of-range source")
    verified = _verified(query.to("cpu"), snap.to("cpu"), got)
    if bool((hits & (~offload | (verified < 0))).any()):
        raise AssertionError("false hit: token/version/readiness/epoch mismatch")
    if bool(((got == MISS) & ~offload).any()):
        raise AssertionError("non-offloaded entry reported as an offload miss")


def gather_dense(query: Query, dense: torch.Tensor) -> torch.Tensor:
    b, q, k = query.tokens.shape
    if dense.ndim != 3 or dense.shape[:2] != (b, query.universe):
        raise ValueError("dense source must be [B,universe,D]")
    if dense.device != query.tokens.device:
        raise ValueError("source and query devices differ")
    index = query.tokens.flatten(1).long().clamp(0, query.universe - 1)
    values = dense.gather(1, index[..., None].expand(-1, -1, dense.shape[-1]))
    valid, _ = query.masks()
    return torch.where(valid[..., None], values, torch.zeros_like(values))


def materialize(query: Query, snap: Snapshot, plan: Plan, dense: torch.Tensor) -> torch.Tensor:
    """Functional two-bank copy. Native resolve_copy fuses lookup + this copy.

    `dense` is the complete current KV truth, INCLUDING the live tail. This
    normalized HBM fixture is NOT an LMCache CPU-source transfer integration.
    """
    fresh = gather_dense(query, dense)
    if fresh.dtype != snap.kv.dtype or fresh.shape != snap.kv.shape:
        raise ValueError("snapshot/current KV layout mismatch")
    index = plan.source.long().clamp_min(0)
    old = snap.kv.gather(1, index[..., None].expand_as(snap.kv))
    return torch.where((plan.source >= 0)[..., None], old, fresh)


class Transaction:
    """CPU/reference transaction with explicit completion before publication.

    A real asynchronous caller must wait/record the device completion dependency
    before complete(). There is deliberately no 'commit on Python launch' path.
    """
    def __init__(self, query: Query, payload: torch.Tensor):
        query.validate()
        if payload.device.type != "cpu":
            raise ValueError("reference transaction is CPU-only; native publication needs a real device completion dependency")
        b, q, k = query.tokens.shape
        if payload.ndim != 3 or payload.shape[:2] != (b, q * k):
            raise ValueError("payload geometry mismatch")
        valid, _ = query.masks()
        self.pending = Snapshot(query.tokens.flatten(1).clone(), query.versions.flatten(1).clone(),
                                valid.clone(), query.epochs.clone(), payload)
        self.state = "pending"

    def complete(self, success: bool) -> None:
        if self.state != "pending":
            raise RuntimeError("transaction already resolved")
        self.state = "complete" if success else "failed"

    def publish(self) -> Snapshot:
        if self.state != "complete":
            raise RuntimeError("only a successfully completed transfer may publish")
        self.state = "published"
        return self.pending


def shared_plan(query: Query, snapshots: Sequence[Snapshot], variant="hash_snapshot", **kwargs) -> Plan:
    """Only reuse a slot if EVERY layer agrees on its identity and readiness.

    Layer payloads need not match. Each layer retains its own KV bank. This
    permits safe conservative misses even if independently evolved banks differ.
    """
    if not snapshots:
        raise ValueError("empty shared-indexer layer group")
    first = snapshots[0]
    ready = first.ready.clone()
    for snap in snapshots:
        snap.validate(query)
        ready &= (snap.ready & (snap.tokens == first.tokens) & (snap.versions == first.versions)
                  & (snap.epochs == first.epochs)[:, None])
    return lookup(query, replace(first, ready=ready), variant, **kwargs)


def attention(query_vectors, selected_kv, valid):
    """Small independent attention oracle: payload is [key | value]."""
    d = query_vectors.shape[-1]
    keys, values = selected_kv[..., :d], selected_kv[..., d:]
    scores = torch.einsum("bqhd,bqkd->bqhk", query_vectors, keys) / d**0.5
    mask = valid[:, :, None, :]
    scores = torch.where(mask, scores, -torch.inf)
    maximum = scores.amax(-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = torch.where(mask, (scores - maximum).exp(), 0)
    denom = weights.sum(-1, keepdim=True)
    return torch.einsum("bqhk,bqkv->bqhv", weights / denom.clamp_min(torch.finfo(weights.dtype).tiny), values)


def split_attention(query_vectors, selected_kv, valid, hits):
    """Functional hit/miss overlap algebra; not a native asynchronous SFA kernel."""
    d = query_vectors.shape[-1]
    keys, values = selected_kv[..., :d], selected_kv[..., d:]
    scores = torch.einsum("bqhd,bqkd->bqhk", query_vectors, keys) / d**0.5
    parts = []
    for subset in (valid & hits, valid & ~hits):
        mask = subset[:, :, None, :]
        local = torch.where(mask, scores, -torch.inf)
        maximum = local.amax(-1, keepdim=True)
        safe_max = torch.where(torch.isfinite(maximum), maximum, 0)
        weights = torch.where(mask, (local - safe_max).exp(), 0)
        parts.append((maximum, weights.sum(-1, keepdim=True),
                      torch.einsum("bqhk,bqkv->bqhv", weights, values)))
    maximum = torch.maximum(parts[0][0], parts[1][0])
    maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    scales = [torch.where(torch.isfinite(m), (m - maximum).exp(), 0) for m, _, _ in parts]
    denominator = sum(s * part[1] for s, part in zip(scales, parts, strict=True))
    numerator = sum(s * part[2] for s, part in zip(scales, parts, strict=True))
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)
