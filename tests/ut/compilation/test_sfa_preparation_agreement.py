# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup topology proof and failure propagation for the EP preparation vote."""

import ast
import importlib.util
import multiprocessing
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from sfa_test_support import ROOT, extract, load_module


@pytest.fixture
def policy(monkeypatch):
    return load_module(ROOT / "vllm_ascend/compilation/sfa_fail_stop.py", "preparation_policy", monkeypatch)


def topology(rank=0, *, width=4, replicas=4, offset=0):
    config = NS(enable_expert_parallel=True, distributed_executor_backend="mp", data_parallel_size=replicas,
                tensor_parallel_size=width, pipeline_parallel_size=1, prefill_context_parallel_size=1,
                decode_context_parallel_size=1, enable_elastic_ep=False, enable_dbo=False)
    ranks = list(range(offset, offset + width * replicas))
    tp_ranks = ranks[rank // width * width:(rank // width + 1) * width]
    dp_ranks = ranks[rank % width::width]
    tp = NS(world_size=width, ranks=tp_ranks, rank=offset + rank, cpu_group="tp")
    dp = NS(world_size=replicas, ranks=dp_ranks, rank=offset + rank, cpu_group="dp")
    ep = NS(world_size=len(ranks), ranks=ranks, rank_in_group=rank, cpu_group="ep")
    return config, tp, dp, ep


@pytest.mark.parametrize("offset", [0, 16])
def test_all_tp4_dp4_ranks_select_the_same_ep_protocol(policy, offset):
    for rank in range(16):
        config, tp, dp, ep = topology(rank, offset=offset)
        assert policy.preparation_error_groups(config, is_moe=True, tp=tp, dp=dp, get_ep=lambda ep=ep: ep) == (
            ("sfa_full_graph::prepare_agreement_ep", "ep"),
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_selects_groups_once_at_initialization(policy, enabled):
    config, tp, dp, ep = topology()
    runner = NS(parallel_config=config, vllm_config=NS(parallel_config=config))
    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text(encoding="utf8"))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Attribute) and t.attr == "_sfa_preparation_groups" for t in n.targets))
    select = Mock(wraps=policy.preparation_error_groups)
    getters = [Mock(return_value=group) for group in (tp, dp, ep)]
    scope = dict(self=runner, vllm_config=runner.vllm_config, preparation_error_groups=select,
                 sfa_full_graph_enabled=lambda cfg: enabled, is_moe_model=lambda cfg: True,
                 get_tp_group=getters[0], get_dp_group=getters[1], get_ep_group=getters[2])
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source), "exec"), scope)
    assert select.call_count == int(enabled)
    if enabled:
        scope.update(torch=torch, uses_local_sfa_fail_stop=policy.uses_local_sfa_fail_stop,
                     record_function_or_nullcontext=lambda name: nullcontext(),
                     dist=NS(all_reduce=Mock(), ReduceOp=NS(MAX="max")))
        coordinate = extract(source, "_coordinate_sfa_full_graph_preparation", scope)
        for _ in range(3):
            coordinate(runner, None, dummy_run=False)
        assert scope["dist"].all_reduce.call_count == 3
        assert all(call.kwargs["group"] == "ep" for call in scope["dist"].all_reduce.call_args_list)
    else:
        assert runner._sfa_preparation_groups is None
    assert all(getter.call_count == int(enabled) for getter in getters)


@pytest.mark.parametrize("field,value", [
    ("enable_expert_parallel", False), ("distributed_executor_backend", "ray"),
    ("pipeline_parallel_size", 2), ("prefill_context_parallel_size", 2),
    ("decode_context_parallel_size", 2), ("enable_elastic_ep", True), ("enable_dbo", True),
])
def test_other_topologies_keep_tp_then_dp(policy, field, value):
    config, tp, dp, _ = topology()
    setattr(config, field, value)
    get_ep = Mock(side_effect=AssertionError("unsupported topology must not access EP"))
    groups = policy.preparation_error_groups(config, is_moe=True, tp=tp, dp=dp, get_ep=get_ep)
    assert groups is None
    get_ep.assert_not_called()


@pytest.mark.parametrize("width,replicas,is_moe", [(1, 4, True), (4, 1, True), (1, 1, True), (4, 4, False)])
def test_singleton_groups_and_dense_models_preserve_old_protocol(policy, width, replicas, is_moe):
    config, tp, dp, _ = topology(width=width, replicas=replicas)
    groups = policy.preparation_error_groups(config, is_moe=is_moe, tp=tp, dp=dp,
                                           get_ep=Mock(side_effect=AssertionError("unneeded EP lookup")))
    assert groups is None


@pytest.mark.parametrize("corrupt", ["ep_size", "duplicate", "tp_members", "dp_members", "local_rank", "index"])
def test_inconsistent_membership_fails_at_startup_instead_of_local_fallback(policy, corrupt):
    config, tp, dp, ep = topology()
    if corrupt == "ep_size":
        ep.world_size = 8
    elif corrupt == "duplicate":
        ep.ranks[-1] = ep.ranks[0]
    elif corrupt == "tp_members":
        tp.ranks[-1] = 7
    elif corrupt == "dp_members":
        dp.ranks[-1] = 13
    elif corrupt == "local_rank":
        dp.rank = 1
    else:
        ep.rank_in_group = 16
    with pytest.raises(ValueError, match="EP membership"):
        policy.preparation_error_groups(config, is_moe=True, tp=tp, dp=dp, get_ep=lambda: ep)


@pytest.mark.parametrize("failing_rank", [None, *range(16)])
def test_one_ep_vote_propagates_any_workers_error_before_replay(policy, failing_rank):
    # Execute the actual runner method for every rank, including idle peers.
    # The reduction stub supplies the collective MAX, not an inference result.
    namespace = dict(torch=torch, uses_local_sfa_fail_stop=policy.uses_local_sfa_fail_stop)
    coordinate = extract(ROOT / "vllm_ascend/worker/model_runner_v1.py",
                         "_coordinate_sfa_full_graph_preparation", namespace)
    for rank in range(16):
        config, tp, dp, ep = topology(rank)
        scopes, calls = [], []

        @contextmanager
        def scope(name, scopes=scopes):
            scopes.append(name)
            yield

        def reduce(tensor, *, op, group, rank=rank, calls=calls):
            assert group == "ep" and tensor.dtype == torch.int32 and tensor.numel() == 1
            assert tensor.item() == int(rank == failing_rank)
            assert op == "max"
            calls.append(group)
            tensor.fill_(int(failing_rank is not None))

        namespace.update(record_function_or_nullcontext=scope,
                         dist=NS(all_reduce=reduce, ReduceOp=NS(MAX="max")))
        runner = NS(vllm_config=NS(parallel_config=config), _sfa_preparation_groups=policy.preparation_error_groups(
            config, is_moe=True, tp=tp, dp=dp, get_ep=lambda ep=ep: ep,
        ))
        error = RuntimeError("injected preparation failure") if rank == failing_rank else None
        if failing_rank is None:
            coordinate(runner, None, dummy_run=rank >= 4)
        else:
            with pytest.raises(RuntimeError, match="preparation failed") as result:
                coordinate(runner, error, dummy_run=rank >= 4)
            assert result.value.__cause__ is error
        assert calls == ["ep"] and scopes == ["sfa_full_graph::prepare_agreement_ep"]


def test_fallback_resolves_current_groups_after_reconfiguration(policy):
    config, tp, dp, _ = topology()
    config.enable_elastic_ep = True
    calls = []
    coordinate = extract(ROOT / "vllm_ascend/worker/model_runner_v1.py",
                         "_coordinate_sfa_full_graph_preparation", dict(
                             torch=torch, uses_local_sfa_fail_stop=policy.uses_local_sfa_fail_stop,
                             record_function_or_nullcontext=lambda name: nullcontext(),
                             get_tp_group=lambda: tp, get_dp_group=lambda: dp,
                             dist=NS(all_reduce=lambda tensor, **kw: calls.append(kw["group"]),
                                     ReduceOp=NS(MAX="max")),
                         ))
    runner = NS(vllm_config=NS(parallel_config=config), _sfa_preparation_groups=None)
    coordinate(runner, None, False)
    tp.cpu_group, dp.cpu_group = "new_tp", "new_dp"
    coordinate(runner, None, True)
    assert calls == ["tp", "dp", "new_tp", "new_dp"]


def _gloo_worker(rank, rendezvous, results):
    path = ROOT / "vllm_ascend/compilation/sfa_fail_stop.py"
    spec = importlib.util.spec_from_file_location("gloo_preparation_policy", path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        config, tp, dp, ep = topology(rank, width=2, replicas=2)
        for group in ([0, 1], [2, 3], [0, 2], [1, 3]):
            pg = dist.new_group(group, backend="gloo")
            if group == tp.ranks:
                tp.cpu_group = pg
            if group == dp.ranks:
                dp.cpu_group = pg
        ep.cpu_group = dist.group.WORLD
        calls = []

        def reduce(tensor, **kwargs):
            calls.append(kwargs["group"])
            dist.all_reduce(tensor, **kwargs)

        coordinate = extract(ROOT / "vllm_ascend/worker/model_runner_v1.py",
                             "_coordinate_sfa_full_graph_preparation", dict(
                                 torch=torch, uses_local_sfa_fail_stop=policy.uses_local_sfa_fail_stop,
                                 record_function_or_nullcontext=lambda name: nullcontext(),
                                 get_tp_group=lambda: tp, get_dp_group=lambda: dp,
                                 dist=NS(all_reduce=reduce, ReduceOp=dist.ReduceOp),
                             ))
        for use_ep in (False, True):
            groups = policy.preparation_error_groups(config, is_moe=True, tp=tp, dp=dp,
                                                      get_ep=lambda: ep) if use_ep else None
            runner = NS(vllm_config=NS(parallel_config=config), _sfa_preparation_groups=groups)
            for bad_rank in (None, 0, 1, 2, 3):
                calls.clear()
                error = ValueError("injected") if rank == bad_rank else None
                try:
                    coordinate(runner, error, dummy_run=rank >= 2)
                    failed = False
                except RuntimeError as exc:
                    assert "preparation failed" in str(exc) and exc.__cause__ is error
                    failed = True
                assert failed == (bad_rank is not None)
                assert len(calls) == (1 if use_ep else 2)
            results.put((rank, use_ep))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "win32" or not dist.is_available() or not dist.is_gloo_available(),
                    reason="Requires Linux CPU/Gloo; no NPU is needed")
def test_real_gloo_tp2_dp2_and_ep_agree_for_every_failing_rank(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    rendezvous = (tmp_path / "store").as_uri()
    children = [ctx.Process(target=_gloo_worker, args=(rank, rendezvous, results)) for rank in range(4)]
    try:
        for child in children:
            child.start()
        deadline = time.monotonic() + 90
        for child in children:
            child.join(max(0, deadline - time.monotonic()))
            assert child.exitcode == 0
        assert {results.get(timeout=5) for _ in range(8)} == {(rank, ep) for rank in range(4) for ep in (False, True)}
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(5)
        results.close()
