# SPDX-License-Identifier: Apache-2.0
"""One success-path agreement for already prepared, unchanged SFA forwards.

All live target participants use the same EP vote. Fallible rollback has its own
transition-only agreement before ordinary preparation, whose vote is retained.
No device collective is skipped and no previous step authorizes current replay.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, override_forward_context, track_batchsize
from vllm.logger import logger
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import StagedSFAQueryProfile, set_ascend_forward_context
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import StagedSFARouteAction, StagedSFARouteDecision, StagedSFARouteReason

PROFILES = tuple(StagedSFAQueryProfile)
ACTIONS = tuple(StagedSFARouteAction)


@dataclass
class PreparedStep:
    context: object
    call: object
    metadata: object = None
    connector: object = None

    def close(self):
        if self.connector is not None:
            self.connector.close()


def initialize_single_agreement(runner, ep):
    """Replace the existing EP startup barrier with a mode/capability check."""
    requested = bool(envs.VLLM_ASCEND_SFA_SINGLE_AGREEMENT)
    supported = (
        getattr(runner, "_sfa_preparation_groups", None) == (("sfa_full_graph::prepare_agreement_ep", ep.cpu_group),)
        and hasattr(runner, "_async_snapshot")
        and not track_batchsize
    )
    flags = torch.tensor([int(requested), int(supported)], dtype=torch.int32)
    dist.all_reduce(flags, group=ep.cpu_group)
    if flags[0].item() not in (0, ep.world_size):
        raise ValueError("SFA single-agreement configuration differs across EP ranks")
    if flags.tolist() == [ep.world_size, ep.world_size]:
        runner._sfa_single_agreement = SingleAgreement(ep, runner.parallel_config.tensor_parallel_size)
    elif requested:
        logger.warning("SFA single agreement unavailable for this runner/topology; retaining existing agreement")


class SingleAgreement:
    def __init__(self, ep, tp_size):
        self.group, self.rank, self.tp_size = ep.cpu_group, ep.rank_in_group, tp_size
        # Proposal, predicted key, error, sequence, and explicit prepared status.
        self.wire = torch.zeros(10, ep.world_size, dtype=torch.int64)
        self.counts = torch.empty(ep.world_size // tp_size, dtype=torch.int32)
        self.expected = torch.empty(3, 1, dtype=torch.int64)
        self.rollback_failed = torch.empty(1, dtype=torch.int32)
        self.failed = False
        self.active = False
        self.sequence = 0
        self.previous = None
        self.context_kwargs = None
        self.prepared = None
        self.complete = False
        self.native_second = False
        self.fast_steps = self.slow_steps = self.rollback_steps = 0

    def activate(self, keys):
        """All ranks finish startup capture before changing the live protocol."""
        signatures = [None] * self.wire.shape[1]
        signature = sorted(
            (key.token_capacity, key.request_capacity, key.query_profile.value, key.max_query_len) for key in keys
        )
        dist.all_gather_object(signatures, signature, group=self.group)
        if not signature or any(value != signature for value in signatures):
            raise ValueError("SFA capture keys differ across the single-agreement group")
        self.active = True

    def discard(self):
        if self.prepared is not None:
            self.prepared.close()
        self.prepared = None

    def remember(self, context, kwargs):
        self.previous = (context.staged_sfa_graph_key, context.staged_sfa_graph_dummy_run, kwargs)

    def can_prepare(self, runner, idle):
        if self.previous is None or self.previous[1] != idle or not self.context_kwargs:
            return False
        key, _, kwargs = self.previous
        if (
            self.context_kwargs.get("aclgraph_runtime_mode") != CUDAGraphMode.PIECEWISE
            or self.context_kwargs.get("staged_sfa_graph_key") != key
            or bool(self.context_kwargs.get("staged_sfa_graph_dummy_run", False)) != idle
        ):
            return False
        if set(kwargs) != {"input_ids", "positions", "intermediate_tensors", "inputs_embeds"}:
            return False
        if kwargs["inputs_embeds"] is not None or kwargs["intermediate_tensors"] is not None:
            return False
        # Cached model views must still address the buffers this step updates.
        for name, buffer in (("input_ids", runner.input_ids.gpu), ("positions", runner.positions.gpu)):
            view = kwargs[name]
            if (
                view is None
                or view.data_ptr() != buffer.data_ptr()
                or view.shape != (key.token_capacity,)
                or view.stride() != buffer.stride()
                or view.dtype != buffer.dtype
                or view.device != buffer.device
            ):
                return False
        return True

    def forward_context(self, kwargs):
        if self.prepared is not None:
            return override_forward_context(self.prepared.context)
        self.context_kwargs = kwargs
        return set_ascend_forward_context(**kwargs)

    def reject_failed_execution(self, *args, **kwargs):
        """Reject queued model/idle RPCs before they can mutate worker state."""
        raise RuntimeError("SFA rollback failed; restart the worker group before further execution")

    def vote(self, runner, proposal, *, idle):
        if self.failed:
            self.reject_failed_execution()
        self.discard()
        self.complete = self.native_second = False
        self.sequence += 1
        error = None
        self.wire.zero_()
        column = self.wire[:, self.rank]
        column[:4].copy_(proposal)
        try:
            with record_function_or_nullcontext("sfa_single_agreement::prepare"):
                self.prepared = runner._prepare_single_agreement(idle)
            key = (
                self.prepared.context.staged_sfa_graph_key
                if self.prepared is not None
                else runner._async_snapshot.key
                if getattr(runner, "_async_pending", None) is not None
                else None
            )
            if key is not None:
                column[4], column[5], column[6] = (
                    key.token_capacity,
                    PROFILES.index(key.query_profile),
                    key.max_query_len,
                )
        except Exception as exc:
            error = exc
        column[7], column[8] = int(error is not None), self.sequence
        column[9] = int(self.prepared is not None)
        with record_function_or_nullcontext("sfa_single_agreement::vote"):
            dist.all_reduce(self.wire, group=self.group)
        if torch.any(self.wire[8] != self.sequence) or torch.any(self.wire[7]):
            self.discard()
            self.previous = None
            raise RuntimeError("SFA combined agreement failed on this or a peer worker") from error
        # Scheduler proposals must agree within TP; readiness may differ.
        proposals = self.wire[:4].view(4, -1, self.tp_size)
        if not torch.equal(proposals, proposals[:, :, :1].expand_as(proposals)):
            self.discard()
            self.previous = None
            raise RuntimeError("SFA batch proposal differs within a TP group")
        batch = self.wire[:4, :: self.tp_size]
        graph = int(batch[1].min()) == CUDAGraphMode.PIECEWISE.value and int(batch[2].max()) == ACTIONS.index(
            StagedSFARouteAction.STAGED
        )
        key = runner._staged_sfa_capture_keys(int(batch[0].max()))[int(bool(batch[3].max()))] if graph else None
        expected = None
        if key is not None:
            expected = self.expected
            expected[0], expected[1], expected[2] = (
                key.token_capacity,
                PROFILES.index(key.query_profile),
                key.max_query_len,
            )
        ready = self.wire[9].bool()
        self.complete = expected is not None and bool(torch.all(ready)) and bool(torch.all(self.wire[4:7] == expected))
        if self.complete:
            self.fast_steps += 1
            return batch
        matching = torch.all(self.wire[4:7] == expected, dim=0) if expected is not None else torch.zeros_like(ready)
        tp_ready = (ready & matching).view(-1, self.tp_size).all(dim=1).repeat_interleave(self.tp_size)
        rollback = (self.wire[4] > 0) & (~matching | (ready & ~tp_ready))
        needs_rollback = bool(torch.any(rollback))
        self.native_second = not graph and needs_rollback
        if needs_rollback:
            error = None
            try:
                if rollback[self.rank]:
                    self.discard()
                    runner._cancel_single_preparation()
            except Exception as exc:
                error = exc
            # This common decision includes unprepared ranks with pending async
            # reconciliation. No peer may enter TP source setup before this gate.
            self.rollback_failed.fill_(int(error is not None))
            self.rollback_steps += 1
            with record_function_or_nullcontext("sfa_single_agreement::rollback_vote"):
                dist.all_reduce(self.rollback_failed, group=self.group)
            if self.rollback_failed.item():
                self.failed = True
                runner.execute_model = runner._dummy_run = self.reject_failed_execution
                self.previous = None
                self.discard()
                raise RuntimeError("SFA rollback failed on this or a peer worker; restart the worker group") from error
        self.slow_steps += 1
        return batch

    def prepare(self, runner, metadata, connector, frontiers, *, idle):
        """Prepare only with an unchanged retained source binding and graph key."""
        key, _, kwargs = self.previous
        if not self.can_prepare(runner, idle):
            return None
        sources = () if idle else connector.sources
        ids = () if idle else tuple(runner.input_batch.req_ids)
        if not runner._sfa_full_graph.sources_match(key.request_capacity, ids, sources):
            return None
        context_kwargs = dict(
            self.context_kwargs,
            attn_metadata=metadata,
            staged_sfa_route=StagedSFARouteDecision(
                StagedSFARouteAction.STAGED,
                StagedSFARouteReason.ELIGIBLE,
                key,
                frontiers=frontiers,
                uniform_query_len=0 if idle else runner.decode_threshold,
            ),
        )
        with set_ascend_forward_context(**context_kwargs):
            context = get_forward_context()
            context.lmcache_warm_graph_step = connector
            update_cos_sin(kwargs["positions"])
            call = runner._prepare_sfa_full_graph_call(kwargs, key.token_capacity, sources=sources, agree=False)
        return PreparedStep(context, call, connector=connector)
