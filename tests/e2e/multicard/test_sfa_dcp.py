# SPDX-License-Identifier: Apache-2.0
"""Qualify counted SFA and the head exchange under a real HCCL replay."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

torch_npu = pytest.importorskip("torch_npu")


def _verify_dcp_probe(probe):
    error = None
    try:
        probe.verify()
    except RuntimeError as caught:
        if not str(caught).startswith("shared retrieval multi-stream capture probe failed"):
            raise
        error = caught
    # Like server startup, agree after every probe replay and before entering
    # another captured collective. A rank-local parity failure must not leave
    # successful peers waiting in the attention test's next all-gather.
    failed = torch.tensor([int(error is not None)], dtype=torch.int32, device="npu")
    dist.all_reduce(failed)
    if failed.item():
        if error is not None:
            raise error
        raise RuntimeError("DCP retrieval capture probe failed on a peer")


def _retain_dcp_stages(monkeypatch, module):
    """Observe the real path without adding operations to the captured graph.

    Retaining tensors changes allocator reuse, so this is a separate test case
    from normal execution. A pass here cannot substitute for the untraced test.
    """
    stages = {}
    merge = module.merge_sfa_dcp
    exchange = dist.all_to_all_single

    def record_merge(partial, maximum, total, group):
        stages.update(partial=partial, maximum=maximum, total=total)
        return merge(partial, maximum, total, group)

    def record_exchange(output, input, *args, **kwargs):
        stages.update(send=input, recv=output)
        return exchange(output, input, *args, **kwargs)

    monkeypatch.setattr(module, "merge_sfa_dcp", record_merge)
    monkeypatch.setattr(dist, "all_to_all_single", record_exchange)
    return stages


def _empty_dcp_stage_comparisons(stages, label):
    # Called only after replay, never during graph capture. All-empty selections
    # have exact, rank-independent expected values at every stage, so no extra
    # collective or cross-rank failure branch is required to localize a mismatch.
    comparisons = []
    for name in ("partial", "maximum", "total", "send", "recv"):
        actual = stages[name].cpu()
        expected = torch.zeros_like(actual)
        if name in ("send", "recv"):
            # Production packet: 512 values, four bytes of FP32 max, four bytes
            # of FP32 sum, eight padding elements. Empty max is -inf (ff800000).
            expected[..., 512:516] = torch.tensor([0, 0, 128, 255], dtype=actual.dtype)
        comparisons.append((label + f" kind={name}", actual, expected))
    return comparisons


def _assert_dcp_comparisons(comparisons):
    failures = []
    for label, actual, expected in comparisons:
        try:
            # Attention has BF16 arithmetic error; copies, zero outputs and
            # encoded statistic bytes must match exactly.
            rtol, atol = (3e-2, 2e-2) if label.endswith("kind=attention") else (0, 0)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except AssertionError as error:
            failures.append(f"DCP parity {label}\n{error}")
    # Report later replay/prefetch failures too, after all collective work ends.
    if failures:
        raise AssertionError("\n\n".join(failures))


@pytest.fixture(scope="module")
def dcp_world():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (4, 8):
        pytest.skip("launch with torchrun and 4 or 8 local NPU ranks")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        # Reinitializing HCCL between parametrized cases is not reliable on the
        # reported runtime. Keep the same group until all cases have finished.
        yield world
    finally:
        dist.destroy_process_group()


@torch.inference_mode()
@pytest.mark.parametrize("overlap,trace_stages,verify_probe", [
    pytest.param(False, False, False, id="False"),
    pytest.param(True, False, True, id="True"),
    pytest.param(True, True, True, id="True-trace"),
    pytest.param(False, True, True, id="serial-after-probe"),
    pytest.param(True, True, False, id="overlap-without-probe"),
])
def test_counted_sfa_dcp_collectives_replay(overlap, trace_stages, verify_probe, monkeypatch, dcp_world):
    world = dcp_world
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm_ascend.ops.triton import sfa_dcp

    comparisons = []
    graph = None
    try:
        rank = dist.get_rank()
        if trace_stages and rank == 0:
            print(
                f"DCP diagnostic runtime: torch={torch.__version__} "
                f"torch_npu={torch_npu.__version__} "
                f"git={getattr(getattr(torch_npu, 'version', None), 'git_version', 'unknown')}",
                flush=True,
            )
        rows, heads = 2, 64
        local_heads = heads // world
        owned = slice(rank * local_heads, (rank + 1) * local_heads)
        rng = torch.Generator().manual_seed(174)
        all_q = torch.randn((rows, heads, 576), generator=rng, dtype=torch.bfloat16)
        cache = torch.randn((world, 128, 576), generator=rng, dtype=torch.bfloat16)
        nope, rope = all_q[:, owned].npu().split((512, 64), dim=-1)
        k = cache[rank, :, :512].contiguous().view(1, 128, 1, 512).npu()
        pe = cache[rank, :, 512:].contiguous().view(1, 128, 1, 64).npu()
        table = torch.tensor([[0]], dtype=torch.int32, device="npu")
        q_lens = torch.tensor([2], dtype=torch.int32, device="npu")
        kv_lens = torch.tensor([128], dtype=torch.int32, device="npu")
        indices = torch.full((rows, 1, 2048), -1, dtype=torch.int32, device="npu")
        counts = torch.zeros((rows, 16), dtype=torch.int32, device="npu")
        scale = 576**-0.5

        def all_gather(packet, *, dim):
            if packet.ndim == 1:
                gathered = packet.new_empty(packet.numel()*world)
                dist.all_gather_into_tensor(gathered, packet)
                return gathered
            head_major = packet.movedim(dim, 0).contiguous()
            gathered = head_major.new_empty((heads, rows, 576))
            dist.all_gather_into_tensor(gathered, head_major)
            return gathered.movedim(0, dim)

        group = SimpleNamespace(all_gather=all_gather, device_group=dist.group.WORLD,
                                rank_in_group=rank, world_size=world)
        edge = None
        if overlap or verify_probe:
            from vllm_ascend.compilation.sfa_retrieval_overlap import CaptureProbe, ProbeCopy, RetrievalEdge

            stream = torch.npu.Stream()
        if verify_probe:
            _verify_dcp_probe(CaptureProbe(stream, "npu", group))
        if overlap:
            source = torch.zeros(64, device="npu")
            destination = torch.empty_like(source)
            observed = torch.empty_like(source)
            edge = RetrievalEdge(stream, ProbeCopy(destination), (destination,))

        stages = _retain_dcp_stages(monkeypatch, sfa_dcp) if trace_stages else {}

        def forward():
            output = sfa_dcp.execute_sfa_dcp(
                nope, rope, (k, pe), indices, counts, table, q_lens, kv_lens, scale, group,
                prefetch=(edge, source, counts, None, (destination,)) if edge else None,
            )
            if edge:
                edge.join()
                observed.copy_(destination)
            return output

        forward()
        torch.npu.synchronize()
        stages.clear()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = forward()
        for step, sizes in enumerate((
            [(0 if peer == 0 else 19 + peer, 11 * peer) for peer in range(world)],
            [(0, 0) for _ in range(world)],
            [(128 - peer, 128) for peer in range(world)],
        )):
            indices.fill_(-1)
            counts[:, 0].copy_(torch.tensor(sizes[rank], dtype=torch.int32))
            for row, count in enumerate(sizes[rank]):
                indices[row, 0, :count] = torch.arange(count, dtype=torch.int32, device="npu")
            if edge:
                source.fill_(sizes[rank][0])
            graph.replay()
            reference = torch.zeros((rows, local_heads, 512), dtype=torch.float32)
            for row in range(rows):
                keys = torch.cat([cache[peer, :sizes[peer][row]] for peer in range(world)]).float()
                if keys.numel():
                    weights = (all_q[row, owned].float() @ keys.T * scale).softmax(dim=-1)
                    reference[row] = weights @ keys[:, :512]
            label = (
                f"rank={rank} replay={step} local_counts={sizes[rank]} traced={trace_stages} "
                f"overlap={overlap} probe={verify_probe}"
            )
            comparisons.append((label + " kind=attention", output.cpu(), reference.bfloat16()))
            if edge:
                comparisons.append((label + " kind=prefetch", observed.cpu(),
                                    torch.full((64,), float(sizes[rank][0]))))
            if trace_stages and all(count == 0 for pair in sizes for count in pair):
                comparisons.append((label + " kind=counts", counts[:, 0].cpu(),
                                    torch.tensor(sizes[rank], dtype=torch.int32)))
                comparisons.extend(_empty_dcp_stage_comparisons(stages, label))
    finally:
        if graph is not None:
            torch.npu.synchronize()
            graph.reset()
    # A rank-local parity failure must never strand peers in a later replay.
    _assert_dcp_comparisons(comparisons)
