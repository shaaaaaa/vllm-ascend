"""Check replay-test snapshot ownership independently of device execution."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from resident_experiment import Case, reference
from test_kernels import test_graph_replay_changes_generation_and_padding_without_host_fences as replay_test


@pytest.mark.parametrize("mtp,shards", [(1, 4), (2, 1), (2, 4)])
def test_replay_harness_accepts_correct_outputs(monkeypatch, mtp, shards):
    # Execute the actual replay test with an exact CPU oracle in place of the
    # device. This tests the harness, not graph capture or NPU correctness.
    active = []

    def execute(case):
        output, _ = reference(case)
        case.reset_from(output)

    class Graph:
        def replay(self):
            execute(self.case)

    @contextmanager
    def capture(graph):
        active.append(graph)
        try:
            yield
        finally:
            active.pop()

    def run(case, optimized, stage="full"):
        assert stage == "full"
        if active:
            active[-1].case = case
        else:
            execute(case)

    monkeypatch.setattr(Case, "run", run)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        NPUGraph=Graph, graph=capture, synchronize=lambda: None,
    ), raising=False)
    replay_test(torch.device("cpu"), False, mtp, shards)
