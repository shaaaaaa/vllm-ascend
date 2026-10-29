# SPDX-License-Identifier: Apache-2.0
"""CPU protocol, actual runner hooks, and real subprocess/Gloo coverage."""

import multiprocessing
import socket
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from sfa_test_support import ROOT, definitions
from test_sfa_async_mtp import setup as async_setup
from test_sfa_full_graph_routing import routing


class RuntimeMode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def policy():
    runner, ns, *_ = routing.__wrapped__()
    ns.update(
        dataclass=dataclass,
        torch=torch,
        dist=dist,
        logger=NS(warning=Mock()),
        envs=NS(VLLM_ASCEND_SFA_SINGLE_AGREEMENT=True),
        track_batchsize=False,
        CUDAGraphMode=RuntimeMode,
        PROFILES=tuple(ns["StagedSFAQueryProfile"]),
        ACTIONS=tuple(ns["StagedSFARouteAction"]),
    )
    definitions(
        ROOT / "vllm_ascend/worker/sfa_agreement.py",
        {"PreparedStep", "SingleAgreement", "initialize_single_agreement"},
        ns,
    )
    return runner, ns


def packet(ns, *, sizes=(8, 8), bounded=False, prepared=(True,) * 4, pending=(False,) * 4, failed=-1, native=False):
    key = ns["StagedSFAGraphKey"].bounded_decode(4, 2) if bounded else ns["StagedSFAGraphKey"].fixed_spec(4, 2)
    wire = torch.zeros(10, 4, dtype=torch.int64)
    for rank in range(4):
        wire[0, rank] = sizes[rank // 2]
        wire[1, rank] = 1
        action = ns["StagedSFARouteAction"].SAFE_NATIVE if native and rank >= 2 else ns["StagedSFARouteAction"].STAGED
        wire[2, rank] = ns["ACTIONS"].index(action)
        wire[3, rank] = bounded
        if prepared[rank] or pending[rank]:
            wire[4:7, rank] = torch.tensor((8, ns["PROFILES"].index(key.query_profile), 2))
        wire[7, rank] = rank == failed
        wire[8, rank] = 1
        wire[9, rank] = prepared[rank]
    return wire, key


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("case", ["ready", "idle", "peer_slow", "grow", "native", "failed", "sequence", "tp"])
def test_all_ranks_choose_the_same_completion_protocol(rank, case):
    runner, ns = policy()
    wire, key = packet(
        ns,
        sizes=(8, 16) if case == "grow" else (8, 8),
        bounded=case == "idle",
        prepared=(True, True, False, True) if case == "peer_slow" else (True,) * 4,
        failed=2 if case == "failed" else -1,
        native=case == "native",
    )
    if case == "sequence":
        wire[8, 2] = 99
    if case == "tp":
        wire[0, 3] = 16
    collective = Mock(side_effect=lambda value, **kw: value.copy_(wire) if value.shape == wire.shape else value.zero_())
    ns["dist"] = NS(all_reduce=collective)
    single = ns["SingleAgreement"](NS(cpu_group="ep", rank_in_group=rank, world_size=4), 2)
    closed = Mock()
    prepared = ns["PreparedStep"](NS(staged_sfa_graph_key=key), object(), connector=NS(close=closed))
    runner._cancel_single_preparation = Mock()
    runner._prepare_single_agreement = lambda idle: None if case == "peer_slow" and rank == 2 else prepared
    if case in ("failed", "sequence", "tp"):
        with pytest.raises(RuntimeError):
            single.vote(runner, wire[:4, rank], idle=case == "idle")
        assert single.prepared is None
    else:
        result = single.vote(runner, wire[:4, rank], idle=case == "idle")
        assert result.shape == (4, 2)
        assert single.complete == (case in ("ready", "idle"))
        assert single.native_second == (case == "native")
        assert (single.prepared is not None) == (
            case not in ("grow", "native") and not (case == "peer_slow" and rank >= 2)
        )
    assert collective.call_count == (2 if case in ("grow", "peer_slow", "native") else 1)
    single.discard()
    single.discard()
    assert closed.call_count == int(not (case == "peer_slow" and rank == 2))


@pytest.mark.parametrize(
    "flags,enabled,error",
    [([0, 4], False, False), ([4, 4], True, False), ([4, 0], False, False), ([2, 4], False, True)],
)
def test_startup_mode_agrees_before_live_protocol(flags, enabled, error):
    runner, ns = policy()
    runner._sfa_single_agreement = None
    runner._sfa_preparation_groups = (("sfa_full_graph::prepare_agreement_ep", "ep"),)
    runner._async_snapshot = None
    runner.parallel_config.tensor_parallel_size = 2
    ns["dist"] = NS(all_reduce=lambda value, **kw: value.copy_(torch.tensor(flags)))
    ep = NS(cpu_group="ep", rank_in_group=0, world_size=4)
    if error:
        with pytest.raises(ValueError, match="configuration differs"):
            ns["initialize_single_agreement"](runner, ep)
    else:
        ns["initialize_single_agreement"](runner, ep)
        assert (runner._sfa_single_agreement is not None) == enabled
        if enabled:
            assert not runner._sfa_single_agreement.active


def test_capture_signature_activation_and_mismatch():
    _, ns = policy()
    single = ns["SingleAgreement"](NS(cpu_group="ep", rank_in_group=0, world_size=4), 2)
    key = ns["StagedSFAGraphKey"].fixed_spec(4, 2)
    ns["dist"] = NS(all_gather_object=lambda rows, value, **kw: rows.__setitem__(slice(None), [value] * 4))
    single.activate([key])
    assert single.active
    single.active = False
    ns["dist"] = NS(
        all_gather_object=lambda rows, value, **kw: rows.__setitem__(slice(None), [value, value, [], value])
    )
    with pytest.raises(ValueError, match="capture keys"):
        single.activate([key])
    assert not single.active


def test_batchsize_timing_keeps_its_original_forward_context_scope():
    runner, ns = policy()
    runner._sfa_single_agreement = None
    runner._sfa_preparation_groups = (("sfa_full_graph::prepare_agreement_ep", "ep"),)
    runner._async_snapshot = None
    runner.parallel_config.tensor_parallel_size = 2
    ns.update(track_batchsize=True, dist=NS(all_reduce=lambda value, **kw: value.mul_(4)))
    ns["initialize_single_agreement"](runner, NS(cpu_group="ep", world_size=4, rank_in_group=0))
    assert runner._sfa_single_agreement is None


def test_prepared_metadata_is_reused_without_second_async_kernel():
    runner, scheduled, shape, events, _ = async_setup.__wrapped__()
    runner._update_states(scheduled)
    runner._prepare_inputs(scheduled, torch.tensor([2, 2, 2]).numpy())
    result = runner._build_attention_metadata(**shape)
    runner._sfa_single_agreement = NS(prepared=NS(metadata=result))
    bases = runner._async_bases.clone()
    assert runner._build_attention_metadata(**shape) is result
    assert events.count("kernel") == 1 and torch.equal(bases, runner._async_bases)


@pytest.mark.parametrize("change", ["none", "eager", "key", "idle", "buffer", "model_kwargs"])
def test_preview_requires_the_current_captured_mode_and_input_storage(change):
    runner, ns = policy()
    key = ns["StagedSFAGraphKey"].fixed_spec(4, 2)
    runner.input_ids, runner.positions = NS(gpu=torch.zeros(8, dtype=torch.int32)), NS(gpu=torch.zeros(8))
    single = ns["SingleAgreement"](NS(cpu_group="ep", rank_in_group=0, world_size=4), 2)
    kwargs = dict(
        input_ids=runner.input_ids.gpu[:8],
        positions=runner.positions.gpu[:8],
        intermediate_tensors=None,
        inputs_embeds=None,
    )
    single.previous = key, False, kwargs
    single.context_kwargs = dict(
        aclgraph_runtime_mode=ns["CUDAGraphMode"].PIECEWISE, staged_sfa_graph_key=key, staged_sfa_graph_dummy_run=False
    )
    if change == "eager":
        single.context_kwargs["aclgraph_runtime_mode"] = ns["CUDAGraphMode"].NONE
    elif change == "key":
        single.context_kwargs["staged_sfa_graph_key"] = None
    elif change == "idle":
        single.context_kwargs["staged_sfa_graph_dummy_run"] = True
    elif change == "buffer":
        runner.positions.gpu = torch.zeros(8)
    elif change == "model_kwargs":
        kwargs["extra"] = object()
    assert single.can_prepare(runner, False) == (change == "none")


def test_actual_async_preview_reuses_one_prepared_context_and_metadata(monkeypatch):
    runner, scheduled, shape, events, _ = async_setup.__wrapped__()
    runner._update_states(scheduled)
    runner._prepare_inputs(scheduled, torch.tensor([2, 2, 2]).numpy())
    _, ns = policy()
    snapshot = runner._async_snapshot
    key = snapshot.key
    runner.decode_threshold = 2
    runner._staged_sfa_layer_names = (snapshot.target_name,)
    single = ns["SingleAgreement"](NS(cpu_group="ep", rank_in_group=0, world_size=4), 2)
    single.previous = (
        key,
        False,
        dict(
            input_ids=runner.input_ids.gpu[: key.token_capacity],
            positions=runner.positions.gpu[: key.token_capacity],
            inputs_embeds=None,
            intermediate_tensors=None,
        ),
    )
    single.context_kwargs = dict(
        aclgraph_runtime_mode=ns["CUDAGraphMode"].PIECEWISE,
        staged_sfa_graph_key=key,
        staged_sfa_graph_dummy_run=False,
        attn_metadata=snapshot.metadata,
    )
    runner._sfa_single_agreement = single
    runner._sfa_full_graph = NS(sources_match=lambda *args: True)
    step = NS(sources=(object(),) * 3, close=Mock())
    facade = ModuleType("vllm.distributed.kv_transfer")
    facade.get_kv_transfer_group = lambda: NS(prepare_warm_sparse_graph_step=lambda *a: step)
    monkeypatch.setitem(sys.modules, facade.__name__, facade)
    current = [None]

    @contextmanager
    def context(**kwargs):
        previous, current[0] = current[0], NS(**kwargs)
        try:
            yield
        finally:
            current[0] = previous

    @contextmanager
    def override(value):
        previous, current[0] = current[0], value
        try:
            yield
        finally:
            current[0] = previous

    ns.update(
        set_ascend_forward_context=context,
        override_forward_context=override,
        get_forward_context=lambda: current[0],
        update_cos_sin=lambda positions: events.append("cos"),
    )
    calls = []

    def prepare(kwargs, tokens, *, sources, agree):
        assert not agree and sources is step.sources and tokens == key.token_capacity
        calls.append(current[0])
        return NS(context=current[0])

    runner._prepare_sfa_full_graph_call = prepare
    prepared = runner._prepare_single_agreement(False)
    single.prepared = prepared
    assert current[0] is None and calls == [prepared.context]
    assert runner._build_attention_metadata(**shape) is prepared.metadata
    assert events.count("kernel") == events.count("cos") == 1
    with single.forward_context(dict(single.context_kwargs, attn_metadata=prepared.metadata[0])):
        assert current[0] is prepared.call.context
    single.discard()
    step.close.assert_called_once()


def test_same_key_tp_fallback_reconciles_after_speculative_metadata_once():
    runner, scheduled, shape, events, _ = async_setup.__wrapped__()
    runner._update_states(scheduled)
    runner._prepare_inputs(scheduled, torch.tensor([2, 2, 2]).numpy())
    runner._build_attention_metadata(**shape)
    runner._cancel_single_preparation()
    runner._build_attention_metadata(**shape)
    assert events.count("kernel") == 1 and events.count("state_update") == 1
    assert runner._async_pending is None and "normal_build" in events


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("case", ["grow", "peer_slow", "native", "unprepared"])
@pytest.mark.parametrize("fault", ["staging", "allocation", "rebuild"])
def test_actual_rollback_failure_is_agreed_before_continuation(monkeypatch, rank, case, fault):
    runner, scheduled, shape, events, _ = async_setup.__wrapped__()
    runner._update_states(scheduled)
    runner._prepare_inputs(scheduled, torch.tensor([2, 2, 2]).numpy())
    if case != "unprepared" and not (case == "peer_slow" and rank == 2):
        runner._build_attention_metadata(**shape)
    routes, ns = policy()
    wire, key = packet(
        ns,
        sizes=(8, 16) if case in ("grow", "unprepared") else (8, 8),
        prepared=(False,) * 4
        if case == "unprepared"
        else (True, True, False, True)
        if case == "peer_slow"
        else (True,) * 4,
        pending=(True,) * 4,
        native=case == "native",
    )
    runner._async_snapshot.key = key
    runner._staged_sfa_capture_keys = routes._staged_sfa_capture_keys
    runner._prepare_single_agreement = lambda idle: (
        None
        if case == "unprepared" or (case == "peer_slow" and rank == 2)
        else ns["PreparedStep"](NS(staged_sfa_graph_key=key), None)
    )
    single = ns["SingleAgreement"](NS(cpu_group="ep", rank_in_group=rank, world_size=4), 2)
    calls = []

    def reduce(value, **kwargs):
        calls.append(tuple(value.shape))
        if value.shape == wire.shape:
            value.copy_(wire)
        else:
            assert value.tolist() == [int(rank == 3)]
            value.fill_(1)  # Rank 3's rollback failed; every participant sees it.

    ns["dist"] = NS(all_reduce=reduce)
    namespace = runner._cancel_single_preparation.__func__.__globals__
    original_np = namespace["np"]

    def allocate(*args, **kwargs):
        if rank == 3 and fault == "allocation":
            raise MemoryError("rollback allocation failed")
        return original_np.full(*args, **kwargs)

    monkeypatch.setitem(namespace, "np", NS(full=allocate, int32=original_np.int32))
    if rank == 3 and fault != "allocation":
        method = "_ensure_host_staging_ready" if fault == "staging" else "_prepare_inputs"
        original = getattr(runner, method)

        def fail(*args, **kwargs):
            if fault == "rebuild":
                original(*args, **kwargs)  # Earlier rebuilding work may already be submitted.
            raise RuntimeError("rollback staging/rebuild failed")

        monkeypatch.setattr(runner, method, fail)
    with pytest.raises(RuntimeError, match="rollback") as failure:
        single.vote(runner, wire[:4, rank], idle=False)
    assert len(calls) == 2 and single.prepared is None
    assert (failure.value.__cause__ is not None) == (rank == 3)
    if rank == 3:
        assert isinstance(failure.value.__cause__, MemoryError if fault == "allocation" else RuntimeError)
    # A queued RPC must not re-enter the wire protocol with partially rebuilt state.
    with pytest.raises(RuntimeError, match="restart"):
        single.vote(runner, wire[:4, rank], idle=False)
    with pytest.raises(RuntimeError, match="restart"):
        runner.execute_model(scheduled)
    with pytest.raises(RuntimeError, match="restart"):
        runner._dummy_run(8, uniform_decode=True)
    assert len(calls) == 2


@pytest.mark.parametrize("valid", [False, True])
def test_idle_preview_validates_then_resets_cached_write_slots(valid):
    runner, ns, _, modes, _ = routing.__wrapped__()
    fields = (
        "slot_mapping",
        "indexer_slot_mapping",
        "indexer_c8_slot_mapping",
        "block_table",
        "indexer_block_table",
        "indexer_c8_block_table",
        "seq_lens",
    )
    metadata = NS(**{name: torch.full((4,), 77) for name in fields}, decode_remap_boundary=object())
    transfer = object()
    layer = NS(
        _staged_sfa_capture_state=NS(runtime=object()),
        _full_graph_transfers={4: transfer},
        full_graph_metadata_inputs=lambda md: {},
        prepare_full_graph_metadata=Mock(),
    )
    runner._staged_sfa_impls = (("layer", layer),)
    context = NS(
        staged_sfa_graph_dummy_run=True,
        staged_sfa_graph_key=NS(request_capacity=4),
        cudagraph_runtime_mode=modes.PIECEWISE,
        attn_metadata={"layer": metadata},
    )
    ns["get_forward_context"] = lambda: context

    def validate(*args):
        assert all(torch.all(getattr(metadata, name) == 77) for name in fields)
        if not valid:
            raise ValueError("replaced idle storage")

    runner._sfa_full_graph = NS(
        sealed=True,
        get_transfers=lambda cap: (transfer,),
        validate_idle_metadata=validate,
        bind_sources=Mock(),
        prepare_run=Mock(),
    )
    if valid:
        runner._prepare_sfa_full_graph_call({}, 8, sources=(), agree=False)
        for name in fields:
            assert torch.all(getattr(metadata, name) == (-1 if "slot_mapping" in name else 0))
    else:
        with pytest.raises(ValueError, match="idle storage"):
            runner._prepare_sfa_full_graph_call({}, 8, sources=(), agree=False)
        assert all(torch.all(getattr(metadata, name) == 77) for name in fields)


COHORT_VOTES = {
    "ready": 1,
    "peer_slow": 3,
    "peer_dp_slow": 2,
    "grow": 3,
    "native": 3,
    "unprepared_grow": 3,
    "unprepared_same": 2,
    "failed": 1,
    "rollback_grow": 2,
    "rollback_peer": 2,
    "rollback_unprepared": 2,
}


def _cohort_worker(rank, port, results, connection=None):
    torch.set_num_threads(1)
    if connection is None:
        store = dist.TCPStore("127.0.0.1", port, 4, rank == 0, timedelta(seconds=35), use_libuv=False)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=4, timeout=timedelta(seconds=35))
    group = dist.group.WORLD if connection is None else "ep"
    try:
        runner, ns = policy()
        runner.vllm_config.parallel_config = NS(data_parallel_size=2)
        runner._sfa_preparation_groups = (("prepare_ep", group),)
        for case, votes in COHORT_VOTES.items():
            calls = []

            def reduce(tensor, calls=calls, **kwargs):
                calls.append(tuple(tensor.shape))
                if connection is not None:
                    connection.send((kwargs.get("op") == dist.ReduceOp.MAX, tensor.tolist()))
                    assert connection.poll(35), "Missing collective peer"
                    tensor.copy_(torch.tensor(connection.recv()))
                    return
                return dist.all_reduce(tensor, **kwargs)

            ns.update(
                dist=NS(all_reduce=reduce, ReduceOp=dist.ReduceOp),
                uses_local_sfa_fail_stop=lambda cfg: False,
                record_function_or_nullcontext=lambda label: nullcontext(),
            )
            unprepared = case in ("unprepared_grow", "unprepared_same", "rollback_unprepared")
            peer_slow = case in ("peer_slow", "rollback_peer")
            wire, key = packet(
                ns,
                sizes=(8, 16)
                if case in ("grow", "rollback_grow", "unprepared_grow", "rollback_unprepared")
                else (8, 8),
                prepared=(False,) * 4 if unprepared else (True, True, False, True) if peer_slow else (True,) * 4,
                native=case == "native",
            )
            single = ns["SingleAgreement"](NS(cpu_group=group, rank_in_group=rank, world_size=4), 2)

            # Real async rollback and an injected allocation failure, rather than
            # a mocked cancellation that cannot exercise the failure window.
            actual = case.startswith("rollback_") or unprepared
            current = runner
            if actual:
                current, scheduled, shape, events, _ = async_setup.__wrapped__()
                current._update_states(scheduled)
                current._prepare_inputs(scheduled, torch.tensor([2, 2, 2]).numpy())
                if not unprepared and not (peer_slow and rank == 2):
                    current._build_attention_metadata(**shape)
                current._async_snapshot.key = key
                current._staged_sfa_capture_keys = runner._staged_sfa_capture_keys
                namespace = current._cancel_single_preparation.__func__.__globals__
                original_np = namespace["np"]

                def allocate(*args, case=case, original_np=original_np, **kwargs):
                    if rank == 3 and case.startswith("rollback_"):
                        time.sleep(0.01)
                        raise MemoryError("injected rollback allocation failure")
                    return original_np.full(*args, **kwargs)

                namespace["np"] = NS(full=allocate, int32=original_np.int32)

            def prepare(idle, case=case, key=key, peer_slow=peer_slow, unprepared=unprepared):
                if rank == 2:
                    time.sleep(0.01)
                    if case == "failed":
                        raise ValueError("injected source failure")
                    if peer_slow or case == "peer_dp_slow":
                        return None
                if unprepared or (case == "peer_dp_slow" and rank == 3):
                    return None
                return ns["PreparedStep"](NS(staged_sfa_graph_key=key), None)

            runner._cancel_single_preparation = Mock()
            current._prepare_single_agreement = prepare
            try:
                single.vote(current, wire[:4, rank], idle=False)
                assert case != "failed" and not case.startswith("rollback_")
                if unprepared:
                    unchanged = case == "unprepared_same"
                    assert (current._async_pending is not None) == unchanged
                    assert events.count("state_update") == int(not unchanged)
                    assert "kernel" not in events
                    final_key = runner._staged_sfa_capture_keys(8 if unchanged else 16)[0]
                    current._apply_staged_sfa_route(final_key)
                    assert events.count("state_update") == int(not unchanged)
                if not single.complete:
                    # Ordinary TP/source preparation may begin only after vote
                    # returns; the final preparation agreement remains intact.
                    runner._coordinate_sfa_full_graph_preparation(None, False)
            except RuntimeError as exc:
                assert case == "failed" or case.startswith("rollback_")
                assert ("combined agreement failed" if case == "failed" else "rollback failed") in str(exc)
                if case.startswith("rollback_"):
                    assert single.failed and single.prepared is None
                    with pytest.raises(RuntimeError, match="restart"):
                        single.vote(current, wire[:4, rank], idle=False)
            finally:
                if actual:
                    namespace["np"] = original_np
            assert len(calls) == votes
            results.put((rank, case, len(calls)))
    finally:
        if connection is None:
            dist.destroy_process_group()
        else:
            connection.close()


@pytest.mark.parametrize(
    "backend",
    [
        "ipc",
        pytest.param(
            "gloo",
            marks=pytest.mark.skipif(
                sys.platform == "win32" or not dist.is_available() or not dist.is_gloo_available(),
                reason="Desktop Windows Gloo reports unsupported device; run this case on Linux",
            ),
        ),
    ],
)
def test_real_subprocess_tp2_dp2_protocol(backend):
    ctx = multiprocessing.get_context("spawn")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    results = ctx.Queue()
    pipes = [ctx.Pipe() for _ in range(4)] if backend == "ipc" else []
    children = [
        ctx.Process(target=_cohort_worker, args=(rank, port, results, pipes[rank][1] if pipes else None))
        for rank in range(4)
    ]
    try:
        for child in children:
            child.start()
        for _, child_pipe in pipes:
            child_pipe.close()
        # Real subprocess rendezvous; only Gloo's reduction is emulated here.
        # Different call counts or ordering fail instead of silently matching.
        for _ in range(sum(COHORT_VOTES.values())) if pipes else ():
            messages = []
            for parent_pipe, _ in pipes:
                assert parent_pipe.poll(35), "A participant missed the next agreement"
                messages.append(parent_pipe.recv())
            assert len({value[0] for value in messages}) == 1
            values = torch.stack([torch.tensor(value[1]) for value in messages])
            combined = values.max(dim=0).values if messages[0][0] else values.sum(dim=0)
            for parent_pipe, _ in pipes:
                parent_pipe.send(combined.tolist())
        deadline = time.monotonic() + 55
        for child in children:
            child.join(max(0, deadline - time.monotonic()))
            assert child.exitcode == 0
        assert len({results.get(timeout=3) for _ in range(4 * len(COHORT_VOTES))}) == 4 * len(COHORT_VOTES)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(5)
        results.close()
        for parent_pipe, _ in pipes:
            parent_pipe.close()
