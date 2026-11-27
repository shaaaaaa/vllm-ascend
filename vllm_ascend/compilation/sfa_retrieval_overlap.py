# SPDX-License-Identifier: Apache-2.0
"""Startup-owned shared-indexer KV transfers inside the target ACL graph.

No request scheduling or storage policy lives here. Each edge joins before the
consumer touches its KV, so the root completion event still protects all sources.
"""

import logging
import os
import platform
import re
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch_npu


def _cann_version():
    query = getattr(torch.npu, "get_cann_version", None)
    if query is not None:
        for args in ((), ("RUNTIME",)):
            try:
                version = str(query(*args) or "").strip()
            except (RuntimeError, TypeError):
                continue
            if re.match(r"^\d+\.\d+", version):
                return version
    toolkit = Path(os.environ.get("ASCEND_HOME_PATH") or "/usr/local/Ascend/ascend-toolkit/latest")
    info = toolkit / (platform.machine() + "-linux") / "ascend_toolkit_install.info"
    try:
        match = re.search(r"(?mi)^\s*version\s*=\s*(\S+)", info.read_text(encoding="utf-8"))
    except OSError:
        return ""
    return match.group(1) if match else ""


class CopyGate:
    """One device admission word and one consumable Notify for the graph's copy FIFO."""

    def __init__(self, device) -> None:
        try:
            from lmcache_ascend.c_ops import SparsePrefetchGate
        except ImportError as error:
            raise RuntimeError("Rebuild LMCache-Ascend on dev-lmy to use the SFA copy windows") from error
        self.native = SparsePrefetchGate(torch.zeros(1, dtype=torch.int64, device=device))
        self.pending = 0  # Capture-time number of consumers still using this grant.
        self.main_stream = None
        self.open = False
        self.free_aiv = 0
        self.local_tp4 = False
        self.counts: Counter[str] = Counter()
        self.soc = self.cann = ""
        if torch.device(device).type == "npu":
            self.soc = torch.npu.get_device_name(device)
            self.cann = _cann_version()

    @property
    def supports_decoder_windows(self):
        return self.soc == "Ascend910B3" and re.match(r"^8\.5\.1(?:\D|$)", self.cann) is not None

    @contextmanager
    def window(self, *kernels):
        # Closing reacquires the FIFO token before the next protected kernel.
        if self.open:
            raise RuntimeError("nested SFA copy admission on the main stream")
        self.native.set_open(True)
        self.open = True
        self.counts.update(kernels)
        try:
            yield
        finally:
            self.native.set_open(False)
            self.open = False


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
    gate: CopyGate | None = None

    def __post_init__(self) -> None:
        self.payload = None
        self.pending_join = False
        self.from_indexer = False
        self.windows_used: set[str] = set()
        self.window_open = False
        self.start = torch.npu.Event(enable_timing=False)
        self.done = torch.npu.Event(enable_timing=False)
        # Initialize event resources outside capture; also join initialization.
        main = torch.npu.current_stream()
        self.start.record(main)
        self.stream.wait_event(self.start)
        self.done.record(self.stream)
        main.wait_event(self.done)

    def launch(self, selected, counts, slots, destinations) -> None:
        """Capture the entire copy train once, after SFA has produced its inputs."""
        self.bind_payload((selected, counts, slots), destinations)
        _submit_transfers((self,))

    def bind_payload(self, payload, destinations) -> None:
        if self.pending_join or (self.gate is not None and self.gate.pending):
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
        self.windows_used.clear()

    def enqueue_on_side(self, copy_plan=None) -> None:
        selected, counts, slots = self.payload
        if self.gate is not None:
            self.transfer.load(selected, counts, slots, max_aiv_cores=self.max_aiv_cores,
                               pieces=min(self.pieces, selected.numel()), admission=self.gate.native,
                               **({"copy_plan": copy_plan} if copy_plan is not None else {}))
        elif self.pieces != 1:
            self.transfer.load(selected, counts, slots, max_aiv_cores=self.max_aiv_cores, pieces=self.pieces)
        elif self.max_aiv_cores:
            self.transfer.load(selected, counts, slots, max_aiv_cores=self.max_aiv_cores)
        else:
            self.transfer.load(selected, counts, slots)
        self.done.record(self.stream)
        self.pending_join = True

    @contextmanager
    def cube_window(self, name: str):
        if (self.gate is None or not self.pending_join or self.window_open or name not in {"vup", "router"}
                or name in self.windows_used or (name == "vup" and "router" in self.windows_used)):
            raise RuntimeError(f"unexpected SFA copy window {name!r} in this capture")
        self.windows_used.add(name)
        self.window_open = True
        try:
            with self.gate.window(name):
                yield
        finally:
            self.window_open = False

    def join(self, qkv_projection=None, *projection_args):
        """Wait only for this layer. A grouped QKV projection keeps the entire FIFO admitted."""
        # Python executes only at capture/warmup, not graph replay. Priming an
        # event outside capture is NOT proof of a record inside this graph.
        if not self.pending_join or self.window_open:
            raise RuntimeError("consumer prefetch wait has no producer launch in this capture")
        self.window_open = True
        try:
            with self.gate.window("drain") if self.gate is not None else nullcontext():
                torch.npu.current_stream().wait_event(self.done)
                if qkv_projection is not None:
                    if self.gate is not None:
                        self.windows_used.add("qkv")
                        self.gate.counts["qkv"] += 1
                    return qkv_projection(*projection_args)
        finally:
            if self.gate is not None:
                self.gate.pending -= 1
            self.window_open = False
            self.pending_join = False


def _submit_transfers(edges, copy_plan=None) -> None:
    """One graph fork, with a distinct side-stream completion record after each layer."""
    first = edges[0]
    main = torch.npu.current_stream()
    if first.gate is not None:
        first.gate.pending = len(edges)
        first.gate.main_stream = main
        # Zero selects the native automatic grid, which may use all 40 AIVs.
        first.gate.free_aiv = max(0, 40 - max(edge.max_aiv_cores or 40 for edge in edges))
    first.start.record(main)
    tracker = first.capture_priority
    before = tracker.before_first_fork(main) if tracker is not None and tracker.side is None else None
    with torch.npu.stream(first.stream):
        first.stream.wait_event(first.start)
        for edge in edges:
            edge.enqueue_on_side(copy_plan)
        if before is not None:
            tracker.after_first_fork(first.stream, before)


class SharedPrefetchGroup:
    """One producer payload, one background FIFO, independently joined consumers."""

    def __init__(self, edges: tuple[RetrievalEdge, ...], packed_width: int | None = None) -> None:
        if not edges:
            raise ValueError("shared prefetch group requires at least one consumer")
        self.edges = edges
        self.capacity = edges[0].transfer.request_capacity
        if any(edge.stream is not edges[0].stream or edge.transfer.request_capacity != self.capacity
               for edge in edges):
            raise ValueError("shared prefetch requires one stream and one request capacity")
        if any(edge.gate is not edges[0].gate for edge in edges):
            raise ValueError("all consumers of an indexer must share the copy FIFO's admission token")
        self.copy_plan = None
        self.packed_width = packed_width
        first = edges[0]
        self.pieces = first.pieces
        if packed_width is not None and first.gate is not None and all(
            (edge.pieces, edge.max_aiv_cores) == (first.pieces, first.max_aiv_cores) for edge in edges
        ):
            self.pieces = min(first.pieces, self.capacity * packed_width)
            if self.pieces > 1:
                self.copy_plan = first.transfer.allocate_copy_plan(self.pieces)
        for edge in edges:
            edge.from_indexer = True

    def launch(self, selected, counts, slots, *, producer_counts=None) -> None:
        payload = (selected[:self.capacity], counts[:self.capacity], slots[:self.capacity])
        for edge in self.edges:
            edge.bind_payload(payload, edge.destinations)
        plan = None
        if producer_counts is not None:
            if producer_counts.shape != payload[1].shape:
                raise ValueError("indexer miss counts and the capture capacity differ")
            plan = self.copy_plan
            if plan is None:
                payload[1].copy_(producer_counts)
            else:
                if selected.shape[1] != self.packed_width:
                    raise ValueError("indexer packed width changed after the shared plan was allocated")
                first = self.edges[0]
                # Replaces the bridge's ordinary counts copy. The fork orders both
                # dense counts and the slice table before any of the side consumers.
                first.transfer.publish_copy_plan(producer_counts, payload[1], plan, self.packed_width,
                                                 self.pieces, first.max_aiv_cores)
        _submit_transfers(self.edges, plan)


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

    def __init__(self, stream, device, *, main_gate=False) -> None:
        shape = (1, 64, 1, 512) if main_gate else (64,)
        self.source = torch.zeros(shape, device=device, dtype=torch.float16)
        self.pe_source = self.source.new_zeros((1, 64, 1, 64)) if main_gate else None
        self.dest = torch.empty_like(self.source)
        self.observed = torch.empty_like(self.source)
        self.before_release = torch.empty_like(self.source) if main_gate else None
        self.after_close = torch.empty_like(self.source) if main_gate else None
        self.before_reopen = torch.empty_like(self.source) if main_gate else None
        self.matrix = torch.ones((16, 16), device=device, dtype=torch.float16)
        self.product = torch.empty_like(self.matrix)
        self.payload = (self.source, None, None)
        if self.pe_source is not None:
            from lmcache_ascend.v1.npu_connector.sparse_graph import SparseGraphTransfer

            slots = torch.arange(64, device=device, dtype=torch.int32).view(1, 64)
            self.payload = (slots, torch.full((1,), 64, device=device, dtype=torch.int32), slots)
            planes = (self.dest, torch.empty_like(self.pe_source))
            transfer = SparseGraphTransfer(planes, slots, 64, 64)
            # The native kernel uses each plane's byte stride and stores only raw source addresses.
            transfer.ptrs[0].fill_(self.source.data_ptr())
            transfer.ptrs[1].fill_(self.pe_source.data_ptr())
            transfer.valid_tokens.fill_(64)
            self.edge = RetrievalEdge(stream, transfer, planes, pieces=8, gate=CopyGate(device))
        else:
            self.edge = RetrievalEdge(stream, ProbeCopy(self.dest), (self.dest,))
        self.graph = torch.npu.NPUGraph()

    def verify(self) -> None:
        torch.mm(self.matrix, self.matrix, out=self.product)
        torch.npu.synchronize()
        with torch.npu.graph(self.graph):
            self.edge.launch(*self.payload, self.edge.destinations)
            if self.before_release is not None:
                self.before_release.copy_(self.dest)
                with self.edge.cube_window("vup"):
                    torch.mm(self.matrix, self.matrix, out=self.product)
                self.after_close.copy_(self.dest)
                torch.mm(self.matrix, self.matrix, out=self.product)
                self.before_reopen.copy_(self.dest)
                with self.edge.cube_window("router"):
                    torch.mm(self.matrix, self.matrix, out=self.product)
            else:
                torch.mm(self.matrix, self.matrix, out=self.product)
            self.edge.join()
            self.observed.copy_(self.dest)
        for value in (1, 7, 2):
            self.source.fill_(value)
            if self.pe_source is not None:
                self.dest.fill_(-value)
                self.pe_source.fill_(-value)
                self.edge.destinations[1].zero_()
            self.graph.replay()
            if not torch.equal(self.observed.cpu(), torch.full(self.source.shape, float(value))):
                raise RuntimeError("shared retrieval multi-stream capture probe failed")
            if self.before_release is not None:
                if not torch.all(self.edge.destinations[1].cpu() == -value):
                    raise RuntimeError("shared retrieval multi-stream capture probe copied incorrect PE data")
                if not torch.equal(self.before_release.cpu(), torch.full(self.source.shape, float(-value))):
                    raise RuntimeError("prefetch wrote the destination before its first window")
                if not torch.equal(self.after_close.cpu(), self.before_reopen.cpu()):
                    raise RuntimeError("prefetch wrote the destination inside a protected interval")
        torch.npu.synchronize()


class SharedRetrievalOverlap:
    """One stream per worker, immutable per-key edges after startup setup."""

    def __init__(self, pieces: int = 1, low_priority: bool = False, *, main_gate: bool = False,
                 multi_layer: bool = False, local_tp4: bool = False) -> None:
        self.pieces = pieces
        self.low_priority = low_priority
        self.main_gate = main_gate
        self.multi_layer = multi_layer
        self.local_tp4 = local_tp4
        if self.pieces < 1:
            raise ValueError("VLLM_ASCEND_SFA_PREFETCH_PIECES must be positive")
        if main_gate and low_priority:
            raise ValueError("Set VLLM_ASCEND_SFA_PREFETCH_LOW_PRIORITY=0 when using the main-path gate")
        self.stream = None
        self.probe = None
        self.pairs: dict[str, tuple[str, tuple]] = {}
        self.incoming: dict[Any, dict[str, RetrievalEdge]] = {}
        self.outgoing: dict[Any, dict[str, RetrievalEdge]] = {}
        self.indexer_groups: dict[Any, dict[str, SharedPrefetchGroup]] = {}
        self.failed = False

    def configure(self, impls, groups, kv_caches) -> None:
        if self.stream is not None or self.incoming:
            raise RuntimeError("cannot replace overlap layout after capture setup")
        by_name = dict(impls)
        pairs = {}
        for group in groups:
            if by_name[group.members[0]].skip_topk:
                raise ValueError("shared retrieval group must start with an indexer producer")
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
        gate = None
        indexer_groups = {}
        self.incoming[key], self.outgoing[key] = incoming, outgoing
        self.indexer_groups[key] = indexer_groups
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
                        self.probe = CaptureProbe(self.stream, device, main_gate=self.main_gate)
                        self.probe.verify()
                        self.probe = None
                    if self.main_gate and gate is None:
                        gate = CopyGate(destinations[0].device)
                        gate.local_tp4 = self.local_tp4
                    edge = RetrievalEdge(
                        self.stream, transfer, destinations, max_aiv_cores=by_name[following].sfa_copy_max_aiv_cores,
                        pieces=self.pieces, gate=gate,
                    )
                    incoming[following] = outgoing[current] = edge
                if self.multi_layer:
                    for group in groups:
                        consumers = group.members[1:]
                        if consumers and all(name in incoming for name in consumers):
                            edges = tuple(incoming[name] for name in consumers)
                            # This is startup dummy preparation. Move any outstanding
                            # KV zero-fill ahead of capture and all background writes.
                            for name, edge in zip(consumers, edges):
                                state = by_name[name]._staged_sfa_capture_state
                                if state.initialized_cache_capacity < key.request_capacity:
                                    for plane in edge.destinations:
                                        plane[state.initialized_cache_capacity:key.request_capacity].zero_()
                                    state.initialized_cache_capacity = key.request_capacity
                            producer = by_name[group.members[0]]
                            width = producer._staged_sfa_bridge_buffers[3].shape[1] if gate is not None else None
                            indexer_groups[group.members[0]] = SharedPrefetchGroup(edges, width)
            except BaseException:
                self.failed = True
                raise

    def finish_capture(self, key) -> None:
        edges = self.outgoing[key].values()
        if any(edge.payload is None or edge.pending_join for edge in edges):
            raise RuntimeError("target graph is missing a prefetch launch or consumer wait")
        groups = self.indexer_groups[key].values()
        gate = next((e.gate for e in edges if e.gate is not None), None)
        logging.getLogger("vllm.sfa_retrieval_overlap").info(
            "SFA %s captured for %s: edges=%d vup=%d router=%d pieces_per_transfer=%d "
            "indexer_groups=%d max_consumers_per_indexer=%d qkv_windows=%d "
            "copy_aiv_limit=%s soc=%s cann=%s windows=%s",
            "main-path gate" if self.main_gate else "shared-group prefetch", key, len(edges),
            sum("vup" in e.windows_used for e in edges), sum("router" in e.windows_used for e in edges),
            self.pieces, len(groups), max((len(g.edges) for g in groups), default=0),
            gate.counts["qkv"] if gate else 0, next((e.max_aiv_cores for e in edges), None),
            gate.soc if gate else "", gate.cann if gate else "", dict(gate.counts) if gate else {},
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
        self.indexer_groups.clear()
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
