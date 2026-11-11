# SPDX-License-Identifier: Apache-2.0
"""Exceptional SFA route diagnostics, independent of PD_SERVING_PERF.

Only the native/rejected route calls the snapshot builder. Healthy graph
selection checks an existing episode flag, without clocks or metadata scans.
Nothing here revalidates a route, reads device data or changes eligibility.
"""

import json
import math
import os
import time

import numpy as np
from vllm.logger import logger

from vllm_ascend.attention.utils import unwrap_staged_sfa_connector_metadata
from vllm_ascend.serving_perf import _non_json_field, cold_perf_clock_fields


def _name(value):
    return getattr(value, "name", getattr(value, "value", value))


def _host_scalar(value):
    """Malformed scalar metadata must not serialize containers or device data."""
    if isinstance(value, (np.integer, np.floating, np.bool_, np.str_)):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple, dict, set)):
        return f"<non-scalar {type(value).__name__}>"
    return _non_json_field(value)


def _cpu_row_int(values, index):
    # The route can reject a malformed CPU shape. Diagnostics must preserve
    # that rejection, not fail trying to turn a non-scalar row into an int.
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            return None
    elif not isinstance(values, (list, tuple)):
        return None
    if index < len(values) and isinstance(values[index], (int, np.integer)):
        return int(values[index])
    return None


def _emit(action, reason, **fields):
    logger.info(
        "[SFA_ROUTE] action=%s reason=%s %s",
        action,
        reason,
        json.dumps(
            dict(schema=1, action=action, reason=reason, pid=os.getpid(), **cold_perf_clock_fields(), **fields),
            separators=(",", ":"),
            default=_non_json_field,
        ),
    )


class SFARouteDiagnostics:
    """One bounded episode per worker; retain only scalars and request IDs."""

    def __init__(self):
        self.active = False
        self.episode = 0

    def prepare(self, signature):
        now = time.monotonic()
        if not self.active:
            self.active = True
            self.episode += 1
            self.started = now
            self.steps = 0
            self.reported_steps = 0
            self.signature = None
        self.steps += 1
        changed = signature != self.signature
        due = changed or now - self.last_report >= 5.0
        if due:
            self.signature = signature
            self.last_report = now
        return due, now

    def record(self, action, reason, now, details, *, error=None):
        self.action, self.reason = action, reason
        if details is None:
            return
        self.request_ids = details["request_ids"]
        self.dp_rank = details["dp_rank"]
        _emit(
            action,
            reason,
            event="route_rejected" if error else "native_route_selected",
            episode=self.episode,
            non_staged_route_steps=self.steps,
            suppressed_steps=self.steps - self.reported_steps - 1,
            elapsed_ms=round((now - self.started) * 1000, 3),
            error=error,
            **details,
        )
        self.reported_steps = self.steps

    def restored(self):
        # Selection is observed here, not completion of an NPU graph replay.
        _emit(
            "staged",
            "eligible",
            event="graph_route_restored",
            episode=self.episode,
            previous_action=self.action,
            previous_reason=self.reason,
            previous_request_ids=self.request_ids,
            dp_rank=self.dp_rank,
            non_staged_route_steps=self.steps,
            suppressed_steps=self.steps - self.reported_steps,
            elapsed_ms=round((time.monotonic() - self.started) * 1000, 3),
        )
        self.active = False
        self.signature = None
        self.request_ids = ()


def _snapshot(runner, route, local_route, request_ids, scheduled, scheduler_output, **context):
    batch = getattr(runner, "input_batch", None)
    metadata = unwrap_staged_sfa_connector_metadata(getattr(scheduler_output, "kv_connector_metadata", None))
    main = {}
    metadata_requests = getattr(metadata, "requests", ())
    # A malformed metadata container can itself be the route's fatal reason.
    if isinstance(metadata_requests, (list, tuple)):
        for item in metadata_requests:
            req_id = getattr(item, "req_id", None)
            if req_id in request_ids and not getattr(item, "is_decode_window_save", False):
                main.setdefault(req_id, []).append(item)
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    resumed_ids = getattr(cached, "resumed_req_ids", ())
    new_ids = {item.req_id for item in getattr(scheduler_output, "scheduled_new_reqs", ())}
    drafts = getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
    rows = []
    for i, req_id in enumerate(request_ids):
        row = dict(
            row=i,
            request_id=req_id,
            scheduled_tokens=scheduled[i] if i < len(scheduled) else None,
            scheduled_draft_tokens=len(drafts.get(req_id, ())),
            new_request=req_id in new_ids,
            resumed_request=req_id in resumed_ids,
        )
        for field, attr in (
            ("computed_tokens", "num_computed_tokens_cpu"),
            ("prompt_tokens", "num_prompt_tokens"),
            ("history_tokens", "num_tokens_no_spec"),
        ):
            values = getattr(batch, attr, ())
            row[field] = _cpu_row_int(values, i)
        marker = i < len(local_route.cold_compact_resumes) and local_route.cold_compact_resumes[i]
        # A retained connector marker is not a proof that the runner's layout
        # checks passed (e.g. frontier_computed[i] can still reject this row).
        row["retained_cold_resume_marker"] = bool(marker)
        row["retained_route_frontier"] = local_route.frontiers[i] if i < len(local_route.frontiers) else None
        computed_ends = getattr(local_route.cold_compact_resumes, "computed_ends", local_route.frontiers)
        row["retained_resume_computed_end"] = computed_ends[i] if marker and i < len(computed_ends) else None
        row["computed_before_prompt_without_retained_marker"] = (
            row["computed_tokens"] is not None
            and row["prompt_tokens"] is not None
            and row["computed_tokens"] < row["prompt_tokens"]
            and not marker
        )
        row["connector_main_count"] = len(main.get(req_id, ()))
        row["connector"] = []
        for item in main.get(req_id, ()):
            load = getattr(item, "load_spec", None)
            row["connector"].append(
                {
                    **{
                        key: _host_scalar(getattr(item, key, None))
                        for key in (
                            "is_sparse_decode",
                            "resumed_from_preemption",
                            "dsa_current_released_frontier",
                            "dsa_nonresident_frontier",
                        )
                    },
                    "load_spec_present": load is not None,
                    **{
                        key: _host_scalar(getattr(load, key, None))
                        for key in (
                            "can_load",
                            "lmcache_cached_tokens",
                            "vllm_cached_tokens",
                            "dsa_committed_end",
                            "dsa_remap_frontier",
                            "dsa_scratch_capacity",
                            "dsa_cold_compact_resume",
                            "dsa_cold_resident_load",
                            "dsa_cold_resume_computed_end",
                        )
                    },
                }
            )
        rows.append(row)
    parallel = runner.parallel_config
    return dict(
        **context,
        request_ids=request_ids,
        scheduled_query_lengths=scheduled,
        rows=rows,
        dp_rank=getattr(runner, "dp_rank", None),
        dp_size=parallel.data_parallel_size,
        tp_size=getattr(parallel, "tensor_parallel_size", None),
        local_action=local_route.action.value,
        local_reason=local_route.reason.value,
        local_failed_checks=local_route.failed_checks,
        failed_checks=route.failed_checks,
        attention_state=_name(runner.attn_state),
        query_width=1 + int(getattr(runner.speculative_config, "num_speculative_tokens", 0)),
        decode_threshold=runner.decode_threshold,
        index_topk=getattr(runner, "dsa_index_topk", None),
        local_uniform_query_len=local_route.uniform_query_len,
        dp_bounded_decode=getattr(runner, "_staged_sfa_dp_bounded_decode", None),
        dp_peer_votes=getattr(runner, "_staged_sfa_diagnostic_peer_votes", None),
        capture_token_sizes=runner._staged_sfa_graph_capture_sizes,
        calculate_kv_scales=getattr(runner, "calculate_kv_scales", False),
        async_acceptance_pending=getattr(runner, "_async_pending", None) is not None,
        independent_dp_active=getattr(runner, "_sfa_independent", None) is not None,
        connector_metadata_present=metadata is not None,
        connector_requests_container=type(metadata_requests).__name__,
    )


def apply_sfa_route_with_diagnostics(
    runner,
    route,
    *,
    local_route=None,
    dp_route_action=None,
    scheduler_output=None,
    num_reqs=0,
    scheduled=(),
    **context,
):
    """Snapshot before async reconciliation, then report selection or rejection."""
    diagnostics = getattr(runner, "_sfa_route_diagnostics", None)
    if diagnostics is None:
        diagnostics = runner._sfa_route_diagnostics = SFARouteDiagnostics()
    local_route = local_route or route
    request_ids = tuple(runner.input_batch.req_ids[:num_reqs]) if num_reqs else ()
    scheduled = tuple(int(value) for value in scheduled)
    context["num_reqs"] = num_reqs
    dp_action = _name(dp_route_action)
    signature = (
        route.action,
        route.reason,
        route.failed_checks,
        local_route.action,
        local_route.reason,
        local_route.failed_checks,
        request_ids,
        scheduled,
        dp_action,
        getattr(runner, "_staged_sfa_diagnostic_peer_votes", None),
        _name(runner.attn_state),
        tuple(context.items()),
    )
    due, now = diagnostics.prepare(signature)
    details = (
        _snapshot(runner, route, local_route, request_ids, scheduled, scheduler_output, dp_action=dp_action, **context)
        if due
        else None
    )
    try:
        key = runner._apply_staged_sfa_route(route)
    except Exception as exc:
        if details is None:
            details = _snapshot(
                runner, route, local_route, request_ids, scheduled, scheduler_output, dp_action=dp_action, **context
            )
        diagnostics.record(
            route.action.value,
            route.reason.value,
            now,
            details,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    diagnostics.record(route.action.value, route.reason.value, now, details)
    return key
