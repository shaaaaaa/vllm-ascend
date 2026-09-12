# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-bounded diagnostics, never installed in clean performance runs.

No profiler, tensor inspection, or per-layer/device completion waits. Host
scopes are nested so exclusive times do not double-count children. Optional
NPU events describe current-stream spans (including waits/host submission
gaps), NOT a sum of kernel execution times.
"""

import math
import threading
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any
from unittest.mock import patch

MAX_DEVICE_INTERVALS = 4096
DETAIL_EVENT_STRIDE = 32


@dataclass
class Moments:
    count: int = 0
    total: float = 0.0
    mean: float = 0.0
    m2: float = 0.0
    maximum: float = 0.0

    def add(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid timing interval: {value}")
        self.count += 1
        self.total += value
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.maximum = max(self.maximum, value)

    def report(self) -> dict:
        return {
            "count": self.count,
            "total_ms": self.total,
            "mean_ms": self.mean,
            "std_ms": math.sqrt(max(0.0, self.m2 / self.count)) if self.count else 0.0,
            "max_ms": self.maximum,
        }


class DecodeTiming:
    def __init__(self, event_factory=None):
        self.active = False
        self.thread_id = threading.get_ident()
        self.event_factory = event_factory
        self.scopes: dict[str, dict[str, Moments]] = {}
        self.stack: list[dict[str, Any]] = []
        self.pending: list[tuple[str, Any, Any]] = []
        self.device_intervals = 0
        self.device_intervals_dropped = 0
        self.patches = ExitStack()
        self.decode_steps = 0
        self.prefill_steps = 0
        self.query_tokens: Counter = Counter()
        self.sampled_tokens: Counter = Counter()
        self.last_target_events = None

    @property
    def recording(self) -> bool:
        return self.active and threading.get_ident() == self.thread_id

    def add(self, stage: str, metric: str, value: float) -> None:
        self.scopes.setdefault(stage, {}).setdefault(metric, Moments()).add(value)

    @contextmanager
    def scope(self, stage: str, *, device: bool = False):
        if not self.recording:
            yield
            return
        start_event = end_event = None
        if device and self.event_factory is not None:
            if self.device_intervals < MAX_DEVICE_INTERVALS:
                start_event, end_event = self.event_factory(), self.event_factory()
                start_event.record()
                self.device_intervals += 1
            else:
                self.device_intervals_dropped += 1
        frame = {"stage": stage, "child_wall": 0.0, "child_cpu": 0.0, "start_event": start_event}
        if stage == "root.replay_submit" and start_event is not None:
            for parent in reversed(self.stack):
                if parent["stage"] == "target.forward" and parent["start_event"] is not None:
                    self.pending.append(("target.before_replay", parent["start_event"], start_event))
                    break
        self.stack.append(frame)
        wall_start, cpu_start = time.perf_counter_ns(), time.thread_time_ns()
        try:
            yield
        finally:
            wall = (time.perf_counter_ns() - wall_start) / 1e6
            cpu = (time.thread_time_ns() - cpu_start) / 1e6
            self.stack.pop()
            self.add(stage, "wall", wall)
            self.add(stage, "self_wall", max(0.0, wall - frame["child_wall"]))
            self.add(stage, "self_cpu", max(0.0, cpu - frame["child_cpu"]))
            if self.stack:
                self.stack[-1]["child_wall"] += wall
                self.stack[-1]["child_cpu"] += cpu
            if end_event is not None:
                end_event.record()
                self.pending.append((stage, start_event, end_event))
                if stage == "target.forward":
                    self.last_target_events = (start_event, end_event)

    @property
    def detail_device_sample(self) -> bool:
        # Bound diagnostic perturbation/event resources. Host substage timings
        # cover ALL decode steps; fine-grained stream spans sample deterministically.
        return self.decode_steps <= 4 or self.decode_steps % DETAIL_EVENT_STRIDE == 0

    def observe(self, owner, name: str, stage, *, device: bool = False, sample_device: bool = False) -> None:
        original = getattr(owner, name)

        @wraps(original)
        def call(*args, **kwargs):
            label = stage() if callable(stage) else stage
            with self.scope(label, device=device and (not sample_device or self.detail_device_sample)):
                return original(*args, **kwargs)

        self.patches.enter_context(patch.object(owner, name, call))

    def kv_wait_stage(self) -> str:
        if any(frame["stage"] == "mtp.propose" for frame in self.stack):
            return "kv.wait.mtp"
        if any(frame["stage"] == "source.prepare" for frame in self.stack):
            return "kv.wait.source_prepare"
        return "kv.wait.target"

    def begin_step(self, scheduler, prompt_tokens: int) -> None:
        """Use CPU scheduler state, not query length: a last prefill may be Q1."""
        self.active = False
        if not scheduler.num_scheduled_tokens:
            return
        if len(scheduler.num_scheduled_tokens) != 1:
            raise RuntimeError("Decode diagnostics require exactly one scheduled request")
        request_id = next(iter(scheduler.num_scheduled_tokens))
        cached = scheduler.scheduled_cached_reqs
        if request_id in cached.req_ids:
            index = cached.req_ids.index(request_id)
            self.active = cached.num_computed_tokens[index] >= prompt_tokens and cached.num_output_tokens[index] > 0
        if self.active:
            self.decode_steps += 1
            self.query_tokens[scheduler.num_scheduled_tokens[request_id]] += 1
        else:
            self.prefill_steps += 1

    def close(self) -> None:
        self.active = False
        self.patches.close()

    def report(self) -> dict:
        """Caller must synchronize ONCE after the request, outside all scopes."""
        if self.stack:
            raise RuntimeError("Cannot report inside a timing scope")
        for stage, start, end in self.pending:
            self.add(stage, "stream_span", float(start.elapsed_time(end)))
        self.pending.clear()
        return {
            "decode_steps": self.decode_steps,
            "prefill_steps_excluded": self.prefill_steps,
            "query_tokens_histogram": dict(self.query_tokens),
            "sampled_tokens_histogram": dict(self.sampled_tokens),
            "device_intervals_dropped": self.device_intervals_dropped,
            "detail_event_stride": DETAIL_EVENT_STRIDE,
            "stages": {
                name: {key: value.report() for key, value in metrics.items()} for name, metrics in self.scopes.items()
            },
        }


class ReplayProbe:
    """Replace only the entry's Python handle, never the captured graph itself."""

    def __init__(self, graph, timing: DecodeTiming):
        self.graph = graph
        self.timing = timing

    def replay(self):
        with self.timing.scope("root.replay_submit", device=True):
            return self.graph.replay()

    def __getattr__(self, name):
        return getattr(self.graph, name)


def install_sampling_timing(runner, timing: DecodeTiming) -> None:
    from vllm_ascend.sample.rejection_diagnostics import reset_stage_recorder, set_stage_recorder

    original = runner._sample

    def record(name, operation, args, kwargs):
        with timing.scope(f"sampling.{name}", device=timing.detail_device_sample):
            return operation(*args, **kwargs)

    @wraps(original)
    def sample(*args, **kwargs):
        if not timing.recording:
            return original(*args, **kwargs)
        token = set_stage_recorder(record)
        try:
            return original(*args, **kwargs)
        finally:
            reset_stage_recorder(token)

    timing.patches.enter_context(patch.object(runner, "_sample", sample))
    # Keep the original sampler implementations and tensor values. Do not
    # duplicate sampling, force readiness, inspect logits, or call .cpu().
    sampler = runner.sampler
    for method in ("apply_logits_processors", "sample", "gather_logprobs"):
        original_method = getattr(sampler, method)

        def observed(*args, _original=original_method, _name=method, **kwargs):
            with timing.scope(f"sampling.sampler.{_name}", device=timing.detail_device_sample):
                return _original(*args, **kwargs)

        timing.patches.enter_context(patch.object(sampler, method, observed))


def install_decode_timing(worker, connector, *, prompt_tokens: int, event_factory=None) -> DecodeTiming:
    """Install for the diagnostic request; restore everything on stop/error."""
    timing = DecodeTiming(event_factory)
    runner = worker.model_runner
    original_execute, original_sample = worker.execute_model, worker.sample_tokens

    def execute(scheduler_output, *args, **kwargs):
        timing.begin_step(scheduler_output, prompt_tokens)
        with timing.scope("worker.execute"):
            return original_execute(scheduler_output, *args, **kwargs)

    def sample(*args, **kwargs):
        with timing.scope("worker.sample"):
            result = original_sample(*args, **kwargs)
        if timing.recording and result is not None:
            # Async scheduling is disabled by the benchmark. Never read a tensor
            # or trigger device transfers just to count output tokens.
            rows = result.sampled_token_ids
            if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
                raise RuntimeError("Expected synchronous CPU sampled token lists")
            timing.sampled_tokens[sum(sum(token >= 0 for token in row) for row in rows)] += 1
        return result

    try:
        timing.patches.enter_context(patch.object(worker, "execute_model", execute))
        timing.patches.enter_context(patch.object(worker, "sample_tokens", sample))
        for method, stage, device in (
            ("_model_forward", "target.forward", True),
            ("_prepare_inputs", "inputs.prepare", False),
            ("_build_attention_metadata", "attention.metadata", False),
            ("propose_draft_token_ids", "mtp.propose", True),
            ("_sample", "sampling", False),
            ("_bookkeeping_sync", "bookkeeping", False),
            ("_copy_draft_token_ids_to_cpu", "mtp.readback", False),
            ("finalize_kv_connector", "kv.finalize", False),
        ):
            timing.observe(runner, method, stage, device=device)
        if hasattr(runner, "rejection_sampler"):
            install_sampling_timing(runner, timing)
        if hasattr(runner, "model"):
            timing.observe(runner.model, "compute_logits", "target.compute_logits", device=True)
        for method, stage in (
            ("prepare_sparse_graph_step", "source.prepare"),
            ("start_load_kv", "kv.start_load"),
            ("wait_for_layer_load", timing.kv_wait_stage),
            ("wait_for_save", "kv.wait_save"),
        ):
            timing.observe(connector, method, stage)
        graph = runner._sfa_full_graph
        timing.observe(graph, "bind_sources", "source.bind")
        timing.observe(graph, "validate_inputs", "signature.validate")
        timing.observe(graph, "run", "root.run")
        for entry in graph.entries.values():
            timing.patches.enter_context(patch.object(entry, "graph", ReplayProbe(entry.graph, timing)))
        impls = runner._staged_sfa_impls or runner._collect_staged_sfa_impls()
        if len(impls) != 8:
            raise RuntimeError(f"Expected eight benchmark target layers, got {len(impls)}")
        for index, (_, impl) in enumerate(impls):
            timing.observe(impl, "prepare_full_graph_layer", f"metadata.L{index}")
            timing.observe(impl, "_cross_layer_metadata_ineligible_reason", "metadata.shared_check")
            timing.observe(impl, "cross_layer_lmcache_retrieve", f"retrieve.L{index}", device=True, sample_device=True)
    except BaseException:
        timing.close()
        raise
    return timing
