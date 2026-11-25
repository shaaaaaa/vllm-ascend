# SPDX-License-Identifier: Apache-2.0
"""Startup-owned, one-layer-ahead transfers inside the target ACL graph.

No request scheduling or storage policy lives here. Each edge joins before the
consumer touches its KV, so the root completion event still protects all sources.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu


def validate_destinations(caches: dict[str, tuple], destinations: set[str]) -> None:
    """Check byte ranges, not storage identity (disjoint views are legal)."""
    ranges = []
    for name, planes in caches.items():
        for plane in planes:
            if not isinstance(plane, torch.Tensor) or not plane.numel():
                continue
            if not plane.is_contiguous():
                raise ValueError(f"overlap requires contiguous cache planes: {name}")
            begin = plane.data_ptr()
            ranges.append((name, plane.device, begin, begin + plane.numel() * plane.element_size()))
    for i, (name, device, begin, end) in enumerate(ranges):
        for other, other_device, first, last in ranges[i + 1:]:
            if (name in destinations or other in destinations) and device == other_device:
                if begin < last and first < end:
                    raise ValueError(f"prefetch cache ranges overlap: {name}, {other}")


@dataclass
class RetrievalEdge:
    stream: Any
    transfer: Any
    destinations: tuple
    max_aiv_cores: int = 0
    pieces: int = 1
    capture_priority: Any = None
    release_at_consumer: bool = False

    def __post_init__(self) -> None:
        self.payload = None
        self.pending_join = False
        self.start = torch.npu.Event(enable_timing=False)
        self.done = torch.npu.Event(enable_timing=False)
        # Initialize event resources outside capture; also join initialization.
        main = torch.npu.current_stream()
        self.start.record(main)
        self.stream.wait_event(self.start)
        self.done.record(self.stream)
        main.wait_event(self.done)

    def launch(self, selected, counts, slots, destinations) -> None:
        """Retain the producer's graph inputs and fork here only for early prefetch."""
        if self.pending_join:
            raise RuntimeError("prefetch launched twice without a consumer join")
        # The native transfer owns fixed destinations. Reject substitutions,
        # including functionalized copies, rather than writing hidden storage.
        if len(destinations) != len(self.destinations) or any(
            actual.data_ptr() != expected.data_ptr()
            or actual.shape != expected.shape or actual.stride() != expected.stride()
            or actual.dtype != expected.dtype or actual.device != expected.device
            for actual, expected in zip(destinations, self.destinations)
        ):
            raise RuntimeError("prefetch destination changed after startup")
        payload = (selected, counts, slots)
        if self.payload is not None and any(
            (actual is None) != (previous is None) or (
                actual is not None and (actual.data_ptr() != previous.data_ptr()
                                        or actual.shape != previous.shape or actual.stride() != previous.stride()
                                        or actual.dtype != previous.dtype or actual.device != previous.device)
            ) for actual, previous in zip(payload, self.payload)
        ):
            raise RuntimeError("prefetch payload storage changed after capture")
        # Hold the ACTUAL custom-op arguments, including compiler-produced views,
        # until graph cleanup. No temporary can be recycled while the side stream
        # reads it. These Python statements run at capture, never at replay.
        if self.payload is None:
            self.payload = payload
        if not self.release_at_consumer:
            self._submit_copy(payload)
        self.pending_join = True

    def _submit_copy(self, payload) -> None:
        selected, counts, slots = payload
        main = torch.npu.current_stream()
        self.start.record(main)
        tracker = self.capture_priority
        before = tracker.before_first_fork(main) if tracker is not None and tracker.side is None else None
        with torch.npu.stream(self.stream):
            self.stream.wait_event(self.start)
            pieces = min(self.pieces, selected.numel()) if self.release_at_consumer else self.pieces
            if pieces != 1:
                self.transfer.load(selected, counts, slots, max_aiv_cores=self.max_aiv_cores, pieces=pieces)
            elif self.max_aiv_cores:
                self.transfer.load(selected, counts, slots, max_aiv_cores=self.max_aiv_cores)
            else:
                self.transfer.load(selected, counts, slots)
            self.done.record(self.stream)
            if before is not None:
                tracker.after_first_fork(self.stream, before)

    def join(self) -> None:
        # Python executes only at capture/warmup, not graph replay. Priming an
        # event outside capture is NOT proof of a record inside this graph.
        if not self.pending_join:
            raise RuntimeError("consumer prefetch wait has no producer launch in this capture")
        if self.release_at_consumer:
            # Record the release AFTER the consumer input norm and all joined
            # foreground streams. Only this fork/join interval admits the copy.
            self._submit_copy(self.payload)
        torch.npu.current_stream().wait_event(self.done)
        self.pending_join = False


@dataclass
class ProbeCopy:
    destination: torch.Tensor

    def load(self, selected, counts, slots) -> None:
        self.destination.copy_(selected)


class CaptureProbe:
    """Fail startup if this runtime cannot replay the required fork/join.

    Retained by the owner on failure, including after uncertain submission.
    This is a dependency probe, not a retrieval performance benchmark.
    """

    def __init__(self, stream, device, *, release_at_consumer=False) -> None:
        self.source = torch.zeros(64, device=device)
        self.dest = torch.empty_like(self.source)
        self.observed = torch.empty_like(self.source)
        self.before_release = torch.empty_like(self.source) if release_at_consumer else None
        self.matrix = torch.ones((16, 16), device=device, dtype=torch.float16)
        self.product = torch.empty_like(self.matrix)
        self.edge = RetrievalEdge(stream, ProbeCopy(self.dest), (self.dest,), release_at_consumer=release_at_consumer)
        self.graph = torch.npu.NPUGraph()

    def verify(self) -> None:
        torch.mm(self.matrix, self.matrix, out=self.product)
        torch.npu.synchronize()
        with torch.npu.graph(self.graph):
            self.edge.launch(self.source, None, None, (self.dest,))
            torch.mm(self.matrix, self.matrix, out=self.product)
            if self.before_release is not None:
                self.before_release.copy_(self.dest)
            self.edge.join()
            self.observed.copy_(self.dest)
        for value in (1, 7, 2):
            self.source.fill_(value)
            if self.before_release is not None:
                self.dest.fill_(-value)
            self.graph.replay()
            if not torch.equal(self.observed.cpu(), torch.full((64,), float(value))):
                raise RuntimeError("shared retrieval multi-stream capture probe failed")
            if self.before_release is not None and not torch.equal(
                self.before_release.cpu(), torch.full((64,), float(-value))
            ):
                raise RuntimeError("prefetch wrote the destination before the consumer release")
        torch.npu.synchronize()


class SharedRetrievalOverlap:
    """One stream per worker, immutable per-key edges after startup setup."""

    def __init__(self, pieces: int = 1, low_priority: bool = False, *, main_gate: bool = False) -> None:
        self.pieces = pieces
        self.low_priority = low_priority
        self.main_gate = main_gate
        if self.pieces < 1:
            raise ValueError("VLLM_ASCEND_SFA_PREFETCH_PIECES must be positive")
        if main_gate and low_priority:
            raise ValueError("Set VLLM_ASCEND_SFA_PREFETCH_LOW_PRIORITY=0 when using the main-path gate")
        self.stream = None
        self.probe = None
        self.pairs: dict[str, tuple[str, tuple]] = {}
        self.incoming: dict[Any, dict[str, RetrievalEdge]] = {}
        self.outgoing: dict[Any, dict[str, RetrievalEdge]] = {}
        self.failed = False

    def configure(self, impls, groups, kv_caches) -> None:
        if self.stream is not None or self.incoming:
            raise RuntimeError("cannot replace overlap layout after capture setup")
        by_name = dict(impls)
        pairs = {}
        for group in groups:
            for current, following in zip(group.members, group.members[1:]):
                producer, consumer = by_name[current], by_name[following]
                if not consumer.skip_topk or producer.shared_resident_plan is not consumer.shared_resident_plan:
                    raise ValueError("invalid shared retrieval topology")
                if kv_caches[current][0].dtype not in (torch.float16, torch.bfloat16):
                    continue
                if not hasattr(torch_npu, "npu_transpose_batchmatmul"):
                    continue
                pairs[current] = (following, tuple(kv_caches[following][:2]))
        if pairs:
            validate_destinations(kv_caches, {name for name, _ in pairs.values()})
        self.pairs = pairs
        for name, impl in impls:
            impl._retrieval_overlap = self

    def prepare(self, key, impls, groups, max_cube_tokens: int) -> None:
        if self.failed:
            raise RuntimeError("shared retrieval setup failed; restart the worker")
        if key in self.incoming:
            return
        by_name = dict(impls)
        if any(not group.active for group in groups):
            raise RuntimeError("shared retrieval requires active resident plans")
        incoming, outgoing = {}, {}
        self.incoming[key], self.outgoing[key] = incoming, outgoing
        # Oversized keys keep serial retrieval; this is a startup shape decision.
        if key.token_capacity <= max_cube_tokens and self.pairs:
            try:
                for current, (following, destinations) in self.pairs.items():
                    before = by_name[current]._full_graph_transfers.get(key.request_capacity)
                    transfer = by_name[following]._full_graph_transfers.get(key.request_capacity)
                    if before is None or transfer is None:
                        raise RuntimeError("shared retrieval transfer was not prepared at startup")
                    # SFA output follows the actual warmed-up query dtype; cache
                    # dtype alone does not prove that _v_up_proj takes its AIC path.
                    query = by_name[current]._staged_sfa_bridge_buffers[0]
                    if query.dtype not in (torch.float16, torch.bfloat16):
                        continue
                    actual = by_name[following]._staged_sfa_capture_state.runtime[1][:2]
                    if len(actual) != len(destinations) or any(
                        a.data_ptr() != b.data_ptr() for a, b in zip(actual, destinations)
                    ):
                        raise RuntimeError("prefetch destination does not match prepared transfer")
                    if self.stream is None:
                        device = destinations[0].device
                        self.stream = torch.npu.Stream(device=device)
                        self.probe = CaptureProbe(self.stream, device, release_at_consumer=self.main_gate)
                        self.probe.verify()
                        self.probe = None
                    edge = RetrievalEdge(
                        self.stream, transfer, destinations, max_aiv_cores=0 if self.main_gate else 12,
                        pieces=self.pieces, release_at_consumer=self.main_gate,
                    )
                    incoming[following] = outgoing[current] = edge
            except BaseException:
                self.failed = True
                raise

    def finish_gate_capture(self, key) -> None:
        edges = self.outgoing[key].values()
        if any(edge.payload is None or edge.pending_join for edge in edges):
            raise RuntimeError("target graph is missing a gated producer/consumer pair")
        logging.getLogger("vllm.sfa_retrieval_overlap").info(
            "SFA main-path gate captured for %s: %d consumer release/join windows, up to %d pieces per transfer",
            key, len(edges), self.pieces,
        )

    def begin_priority_capture(self, key):
        edges = self.outgoing.get(key, {})
        if not edges:
            return None
        from vllm_ascend.compilation.sfa_copy_priority import CapturedCopyPriority

        tracker = CapturedCopyPriority(torch.npu.current_stream())
        for edge in edges.values():
            edge.capture_priority = tracker
        return tracker

    def finish_priority_capture(self, key, tracker):
        edges = self.outgoing[key]
        tracker.finish(len(edges), self.pieces)
        for edge in edges.values():
            edge.capture_priority = None

    def edges(self, key):
        if self.failed or key not in self.incoming:
            raise RuntimeError("shared retrieval key is not prepared")
        return self.incoming[key], self.outgoing[key]

    def clear(self) -> None:
        # Called only AFTER the root owner synchronizes, including on failure.
        self.incoming.clear()
        self.outgoing.clear()
        self.probe = None
        self.stream = None
        self.failed = False
        # Layer cache references remain valid across graph-only recapture. KV
        # reinitialization calls configure() again before compiling the model.

    def release_layout(self, impls) -> None:
        """Drop temporary profiling KV, after synchronized graph cleanup."""
        if self.stream is not None or self.incoming or self.outgoing:
            raise RuntimeError("clear overlap graphs before releasing cache layout")
        for _, impl in impls:
            if getattr(impl, "_retrieval_overlap", None) is self:
                impl._retrieval_overlap = None
        self.pairs.clear()
