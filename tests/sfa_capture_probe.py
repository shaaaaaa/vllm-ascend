# SPDX-License-Identifier: Apache-2.0
"""Synthetic capture diagnostics, isolated from the production startup path."""

from dataclasses import dataclass

import torch

from vllm_ascend.compilation.sfa_retrieval_overlap import RetrievalEdge


@dataclass
class ProbeCopy:
    destination: torch.Tensor

    def load(self, selected, counts, slots) -> None:
        self.destination.copy_(selected)


class CaptureProbe:
    """Synthetic fork/join and DCP qualification; never run by serving startup.

    Tests own this graph and its operands through completion and teardown.
    """

    def __init__(self, stream, device, dcp_group=None) -> None:
        self.dcp_group = dcp_group
        self.source = torch.zeros(64, device=device)
        self.dest = torch.empty_like(self.source)
        self.observed = torch.empty_like(self.source)
        self.matrix = torch.ones((16, 16), device=device, dtype=torch.float16)
        self.product = torch.empty_like(self.matrix)
        self.edge = RetrievalEdge(stream, ProbeCopy(self.dest), (self.dest,))
        self.graph = torch.npu.NPUGraph()
        if dcp_group is not None:
            self.selected = torch.empty_like(self.source)
            self.mask = torch.arange(64, device=device) < dcp_group.rank_in_group + 1
            self.exchange = torch.empty(64 * dcp_group.world_size, device=device)

    def verify(self) -> None:
        torch.mm(self.matrix, self.matrix, out=self.product)
        torch.npu.synchronize()
        with torch.npu.graph(self.graph):
            selected = self.source
            if self.dcp_group is not None:
                # Model-stream gather -> local compute -> side-stream prefetch
                # -> model-stream exchange -> consumer join, just as DCP SFA.
                gathered = self.dcp_group.all_gather(self.source, dim=0)
                self.selected.copy_(gathered.view(-1, 64).sum(0) * self.mask)
                selected = self.selected
            torch.mm(self.matrix, self.matrix, out=self.product)
            self.edge.launch(selected, None, None, (self.dest,))
            if self.dcp_group is not None:
                torch.distributed.all_to_all_single(
                    self.exchange, gathered, group=self.dcp_group.device_group,
                )
            torch.mm(self.matrix, self.matrix, out=self.product)
            self.edge.join()
            self.observed.copy_(self.dest)
        failures = []
        rank = 0 if self.dcp_group is None else self.dcp_group.rank_in_group
        world = 1 if self.dcp_group is None else self.dcp_group.world_size
        for value in (1, 7, 2):
            self.source.fill_(value + rank)
            self.graph.replay()
            expected = torch.full((64,), float(value), dtype=self.source.dtype, device="cpu")
            if self.dcp_group is not None:
                expected.fill_(world * value + world * (world-1) / 2)
                expected[rank+1:] = 0
            observed = self.observed.cpu()
            mismatch = not torch.equal(observed, expected)
            if self.dcp_group is not None:
                # Prefetch correctness does not prove that the following HCCL
                # exchange replayed correctly. Check its result independently;
                # these small readbacks run only in the startup probe.
                exchange_observed = self.exchange.cpu()
                exchange_expected = torch.full_like(exchange_observed, float(value + rank))
                mismatch = mismatch or not torch.equal(exchange_observed, exchange_expected)
            if mismatch:
                # Only a failed startup probe pays for these extra readbacks.
                # Snapshot before the next replay overwrites the evidence. These
                # are observations, not proof that one particular stream caused
                # the mismatch: a broken join can still have work in flight.
                snapshots = [
                    ("observed", observed, expected),
                    ("source", self.source, torch.full_like(expected, float(value + rank))),
                    ("destination", self.dest, expected),
                ]
                if self.dcp_group is not None:
                    gathered_expected = (
                        torch.arange(world, dtype=expected.dtype, device="cpu") + value
                    ).repeat_interleave(64)
                    snapshots.extend((
                        ("gathered", gathered, gathered_expected),
                        ("selected", self.selected, expected),
                        ("exchange", exchange_observed, exchange_expected),
                    ))
                details = []
                for name, actual, wanted in snapshots:
                    try:
                        actual, wanted = actual.cpu().flatten(), wanted.flatten()
                        bad = (actual != wanted).nonzero().flatten()
                        sample = bad[:8]
                        details.append(
                            f"{name}: mismatches={bad.numel()} indices={sample.tolist()} "
                            f"actual={actual[sample].tolist()} expected={wanted[sample].tolist()}"
                        )
                    except Exception as error:
                        # Optional diagnostics must not abandon peer replays.
                        details.append(f"{name}: diagnostic readback failed: {type(error).__name__}: {error}")
                failures.append(f"replay={value}; " + "; ".join(details))
                if self.dcp_group is None:
                    raise RuntimeError("shared retrieval multi-stream capture probe failed: " + failures[-1])
                # A local parity mismatch must not skip collectives that peers
                # will replay. The enclosing preparation agreement propagates
                # this failure after every participant finishes the probe.
        torch.npu.synchronize()
        if failures:
            raise RuntimeError(
                f"shared retrieval multi-stream capture probe failed: rank={rank} world_size={world}; "
                + " | ".join(failures)
            )
