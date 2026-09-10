# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU/Gloo multiprocess diagnostic agreement; not an HCCL/NPU test."""

import importlib.util
import multiprocessing
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist


def check_rank(rank, world_size, rendezvous, results):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_parity.py"
    spec = importlib.util.spec_from_file_location("gloo_parity", path)
    parity = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parity
    spec.loader.exec_module(parity)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        group = SimpleNamespace(world_size=world_size, rank_in_group=rank, cpu_group=dist.group.WORLD)
        assert parity.coordinated_check(lambda: rank, group=group, phase="before target") == rank

        def observe():
            if rank == world_size - 1:
                raise parity.ParityError("layer=4 kv mismatch")

        try:
            parity.coordinated_check(observe, group=group, phase="after target")
        except parity.ParityError as exc:
            results.put((rank, str(exc)))
        else:
            raise AssertionError("A peer's error did not propagate")
        # Both ranks reached the same error boundary; neither raced ahead or
        # left the peer stranded before another collective.
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="PyTorch Gloo is unavailable")
@pytest.mark.parametrize("world_size", [2, 8])
def test_actual_gloo_workers_agree_on_nonzero_rank_failure(tmp_path, world_size):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    rendezvous = (tmp_path / "rendezvous").as_uri()
    processes = [
        context.Process(target=check_rank, args=(rank, world_size, rendezvous, results)) for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert all(process.exitcode == 0 for process in processes), [p.exitcode for p in processes]
        messages = dict(results.get(timeout=2) for _ in processes)
        assert set(messages) == set(range(world_size))
        assert len(set(messages.values())) == 1
        assert f"rank={world_size - 1} ParityError: layer=4 kv mismatch" in messages[0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)
        results.close()
        results.join_thread()
