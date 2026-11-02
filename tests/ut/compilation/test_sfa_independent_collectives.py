# SPDX-License-Identifier: Apache-2.0
"""Routing must not share collective sequencing with background LMCache imports."""

import gc
import multiprocessing
import socket
import sys
import threading
import time
from collections import defaultdict
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from sfa_test_support import extract
from test_sfa_independent import ROOT, independent


def _import_vote(tp):
    for name in ("LMCache-NPU", "LMCache"):
        source = ROOT.parent / name / "lmcache/integration/vllm/vllm_service_factory.py"
        if source.is_file():
            return extract(source, "collective_all_true", dict(torch=torch, dist=dist, tpg=tp))
    pytest.skip("Requires the matching sibling LMCache checkout")


def test_cold_import_and_routing_allow_opposite_tp_thread_order(monkeypatch):
    # Blocking CPU rendezvous models process-group sequence slots. It does not
    # emulate Gloo transport or NPU execution. Both callers execute real methods.
    width = 4
    current = threading.local()
    condition = threading.Condition()
    sequence, slots, results = defaultdict(int), defaultdict(dict), {}
    entered = [threading.Event() for _ in range(width)]
    failures, ready = [], []
    create = Mock(return_value="routing-tp")
    monkeypatch.setattr(independent.dist, "new_group", create)

    def all_reduce(tensor, *, group, op=None):
        rank = current.rank
        with condition:
            key = group, sequence[rank, group]
            sequence[rank, group] += 1
            values = slots[key]
            values[rank] = tensor.clone(), op
            entered[rank].set()
            if len(values) == width:
                signatures = {(tuple(value.shape), value.dtype, operation) for value, operation in values.values()}
                if len(signatures) != 1:
                    results[key] = RuntimeError(f"collective slot mismatch: {signatures}")
                else:
                    batch = torch.stack([value for value, _ in values.values()])
                    results[key] = batch.min(0).values if op == torch.distributed.ReduceOp.MIN else batch.sum(0)
                condition.notify_all()
            assert condition.wait_for(lambda: key in results, timeout=5), "collective did not complete"
            result = results[key]
            if isinstance(result, Exception):
                raise result
            tensor.copy_(result)

    monkeypatch.setattr(independent.dist, "all_reduce", all_reduce)
    policies, imports = [], []
    for rank in range(width):
        tp = NS(cpu_group="cache-tp", ranks=list(range(width)), rank_in_group=rank, world_size=width)
        policies.append(independent.IndependentDP(tp, 4, 24))
        imports.append(_import_vote(tp))

    def run(rank, routing):
        current.rank = rank
        try:
            if routing:
                assert policies[rank].agree(8, 1, 1, False, requests=4).tolist() == [8] * 4
            else:
                ready.append(imports[rank](rank != 2))
        except BaseException as error:
            failures.append(error)

    # TP0 reaches the foreground vote first; peers reach import readiness first.
    first = [threading.Thread(target=run, args=(rank, rank == 0), daemon=True) for rank in range(width)]
    second = [threading.Thread(target=run, args=(rank, rank != 0), daemon=True) for rank in range(width)]
    for thread in first:
        thread.start()
    try:
        assert all(event.wait(5) for event in entered)
    finally:
        for thread in second:
            thread.start()
        for thread in first + second:
            thread.join(6)
    assert not any(thread.is_alive() for thread in first + second)
    assert not failures, [str(error) for error in failures]
    assert ready == [False] * width  # Failed import is not mistaken for a routing proposal.
    assert {group for group, _ in slots} == {"cache-tp", "routing-tp"}
    assert create.call_count == width
    for call in create.call_args_list:
        assert call.kwargs["ranks"] == list(range(width))
        assert call.kwargs["backend"] == "gloo" and call.kwargs["use_local_synchronization"]


def test_single_rank_has_no_new_process_group_or_collective(monkeypatch):
    create, reduce = Mock(), Mock()
    monkeypatch.setattr(dist, "new_group", create)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    policy = independent.IndependentDP(NS(rank_in_group=0, world_size=1), 4, 24)
    assert policy.agree(8, 1, 1, False, requests=4).tolist() == [8] * 4
    create.assert_not_called()
    reduce.assert_not_called()


def test_group_creation_failure_does_not_install_an_active_policy(monkeypatch):
    tp = NS(cpu_group="cache-tp", ranks=[0, 1], rank_in_group=0, world_size=2)
    monkeypatch.setitem(sys.modules, "vllm.distributed", NS(get_tp_group=lambda: tp))
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.sfa_independent", independent)
    monkeypatch.setattr(dist, "new_group", Mock(side_effect=RuntimeError("group setup failed")))
    activate = extract(ROOT / "vllm_ascend/worker/worker.py", "activate_independent_dp", {})
    worker = NS(
        model_runner=NS(_sfa_independent=None), dp_guardian=object(),
        parallel_config=NS(data_parallel_size=2), _independent_dp_contract=dict(digest="a" * 64, capacity=24),
    )
    with pytest.raises(RuntimeError, match="group setup failed"):
        activate(worker, True)
    assert worker.model_runner._sfa_independent is None


def _gloo_overlap_worker(rank, port):
    gc.disable()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4,
                            timeout=timedelta(seconds=15))
    try:
        ranks = list(range(rank // 2 * 2, rank // 2 * 2 + 2))
        cache = dist.new_group(ranks=ranks, backend="gloo", use_local_synchronization=True,
                               timeout=timedelta(seconds=15))
        tp = NS(cpu_group=cache, ranks=ranks, rank_in_group=rank % 2, world_size=2)
        policy = independent.IndependentDP(tp, 2, 24)
        assert policy.group is not cache
        vote = _import_vote(tp)
        failures = []
        entered = threading.Event()

        def run(routing):
            try:
                entered.set()
                for step in range(8):
                    if routing:
                        tokens = 2 * (rank // 2 + 1)
                        assert policy.agree(tokens, 1, 1, False, draft=bool(step % 2)).tolist() == [tokens] * 2
                    else:
                        payload = [dict(request=step) if rank % 2 == 0 else None]
                        dist.broadcast_object_list(payload, src=ranks[0], group=cache)
                        assert payload == [dict(request=step)]
                        assert vote(step % 3 != 0 or rank % 2 == 0) == (step % 3 != 0)
            except BaseException as error:
                failures.append(error)

        first = threading.Thread(target=run, args=(rank % 2 == 0,), daemon=True)
        second = threading.Thread(target=run, args=(rank % 2 != 0,), daemon=True)
        first.start()
        assert entered.wait(5)
        time.sleep(0.05)
        second.start()
        first.join(20)
        second.join(20)
        assert not first.is_alive() and not second.is_alive()
        assert not failures, [str(error) for error in failures]
    finally:
        # Same registered-group teardown used by WorkerProc after connector drain.
        dist.destroy_process_group()
    assert not dist.is_initialized()


@pytest.mark.skipif(
    sys.platform != "linux" or not dist.is_available() or not dist.is_gloo_available(),
    reason="Requires real Linux Gloo; Windows Gloo reports unsupported device",
)
def test_real_gloo_disjoint_tp_groups_with_concurrent_cache_and_route_traffic(monkeypatch):
    _import_vote(NS(cpu_group=None))  # Check the source dependency before starting children.
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    context = multiprocessing.get_context("spawn")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    children = [context.Process(target=_gloo_overlap_worker, args=(rank, port)) for rank in range(4)]
    try:
        for child in children:
            child.start()
        deadline = time.monotonic() + 60
        for child in children:
            child.join(max(0, deadline - time.monotonic()))
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(5)
