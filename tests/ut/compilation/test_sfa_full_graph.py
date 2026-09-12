# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; runnable with --confcutdir=tests/ut/compilation."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def graph_module(monkeypatch):
    context = SimpleNamespace(
        staged_sfa_graph_key="q2",
        cudagraph_runtime_mode="piecewise",
        staged_sfa_graph_dummy_run=True,
        capturing=False,
    )
    counter = SimpleNamespace(num_cudagraph_captured=0)
    validator = Mock()
    modules = {
        "vllm.compilation.counter": {"compilation_counter": counter},
        "vllm.compilation.monitor": {"validate_cudagraph_capturing_enabled": validator},
        "vllm.config": {"CUDAGraphMode": SimpleNamespace(NONE="none")},
        "vllm.forward_context": {"get_forward_context": lambda: context},
        "vllm.platforms": {"current_platform": SimpleNamespace(get_global_graph_pool=lambda: "pool")},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    captures = []
    stream = SimpleNamespace(synchronize=Mock())

    @contextmanager
    def capture(graph, pool):
        assert context.sfa_full_graph_active and context.capturing
        captures.append((graph, pool))
        yield

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            NPUGraph=lambda: SimpleNamespace(replay=Mock()),
            graph=capture,
            current_stream=lambda: stream,
        ),
        raising=False,
    )
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/compilation/sfa_full_graph.py"
    spec = importlib.util.spec_from_file_location("tested_sfa_full_graph", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, context, captures, stream


def test_one_replay_without_reentering_target_python(graph_module):
    module, context, captures, stream = graph_module
    wrapper = module.SFAFullGraph()
    x = torch.ones(2)
    target = Mock(side_effect=lambda **kw: kw["input_ids"] + 2)
    out = wrapper.run(target, input_ids=x, positions={"nested": x})
    assert wrapper.seal(("q2",)) == 1
    context.staged_sfa_graph_dummy_run = False
    x.add_(3)
    for _ in range(3):
        assert wrapper.run(target, input_ids=x, positions={"nested": x}) is out
    assert target.call_count == 1
    assert len(captures) == 1
    assert captures[0][0].replay.call_count == 3
    assert stream.synchronize.call_count == 3
    assert wrapper.replay_count == 3


def test_no_live_capture_or_changed_keyword_addresses(graph_module):
    module, context, _, _ = graph_module
    wrapper = module.SFAFullGraph()
    x = torch.ones(2)
    wrapper.run(lambda **kw: x, positions=x)
    wrapper.seal(("q2",))
    context.staged_sfa_graph_dummy_run = False
    with pytest.raises(RuntimeError, match="address or layout"):
        wrapper.run(Mock(), positions=x.clone())
    context.staged_sfa_graph_key = "q1"
    with pytest.raises(RuntimeError, match="live capture is prohibited"):
        wrapper.run(Mock(), positions=x)


def test_capture_failure_restores_context_and_profile_reset(graph_module):
    module, context, _, _ = graph_module
    wrapper = module.SFAFullGraph()
    with pytest.raises(ValueError, match="capture failed"):
        wrapper.run(Mock(side_effect=ValueError("capture failed")))
    assert not context.capturing and not context.sfa_full_graph_active
    assert not wrapper.entries
    wrapper.run(lambda: torch.ones(1))
    with pytest.raises(RuntimeError, match="Incomplete"):
        wrapper.seal(("q1", "q2"))
    wrapper.clear()
    assert not wrapper.entries and not wrapper.sealed


def test_eager_warmup_never_captures(graph_module):
    module, context, captures, _ = graph_module
    context.cudagraph_runtime_mode = "none"
    wrapper = module.SFAFullGraph()
    runnable = Mock(return_value="warmup")
    assert wrapper.run(runnable) == "warmup"
    assert not captures


def test_metadata_storage_cannot_be_replaced_between_replays(graph_module):
    module, context, _, _ = graph_module
    wrapper = module.SFAFullGraph()
    metadata = torch.zeros(1, dtype=torch.int32)
    wrapper.run(lambda: metadata, graph_inputs={"seq_lens": metadata})
    wrapper.seal(("q2",))
    context.staged_sfa_graph_dummy_run = False
    metadata.fill_(512)
    wrapper.run(Mock(), graph_inputs={"seq_lens": metadata})
    with pytest.raises(RuntimeError, match="address or layout"):
        wrapper.run(Mock(), graph_inputs={"seq_lens": metadata.clone()})


def test_preflight_checks_do_not_capture_or_replay(graph_module):
    module, context, captures, _ = graph_module
    wrapper = module.SFAFullGraph()
    x = torch.zeros(1)
    wrapper.validate_inputs(input_ids=x)
    assert not captures
    wrapper.run(lambda **kw: x, input_ids=x)
    wrapper.seal(("q2",))
    context.staged_sfa_graph_dummy_run = False
    wrapper.validate_inputs(input_ids=x)
    with pytest.raises(RuntimeError, match="address or layout"):
        wrapper.validate_inputs(input_ids=x.clone())
    context.staged_sfa_graph_key = "missing"
    with pytest.raises(RuntimeError, match="live capture is prohibited"):
        wrapper.validate_inputs(input_ids=x)
    captures[0][0].replay.assert_not_called()


@pytest.mark.parametrize("layers", [1, 8, 80])
def test_unchanged_sources_do_not_even_enumerate_layers(graph_module, layers):
    module, context, _, _ = graph_module
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=2)
    wrapper = module.SFAFullGraph()

    # Equality is intentionally illegal, as for dataclasses containing tensors.
    class Source:
        def __eq__(self, other):
            raise AssertionError("Must compare snapshot identity, not tensor values")

    sources = (Source(), Source())
    transfers = tuple(SimpleNamespace(bind_batch=Mock()) for _ in range(layers))
    enumerate_layers = Mock(return_value=transfers)
    assert wrapper.bind_sources(sources, ("a", "b"), enumerate_layers)
    for step in range(300):
        # Fresh containers, identical immutable snapshots; simulated sequence
        # lengths/top-k changes must not force CPU source-table updates.
        assert not wrapper.bind_sources(tuple(list(sources)), ("a", "b"), enumerate_layers)
    enumerate_layers.assert_called_once()
    for index, transfer in enumerate(transfers):
        transfer.bind_batch.assert_called_once_with(sources, index)
    assert wrapper.source_binding_count == 1


@pytest.mark.parametrize("change", ["new_source", "window", "restore", "reorder", "request", "empty", "remove"])
def test_source_changes_always_rebind_all_layers(graph_module, change):
    module, context, _, _ = graph_module
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=2)
    wrapper = module.SFAFullGraph()
    source = SimpleNamespace(total_tokens=4096)
    other = SimpleNamespace(total_tokens=4096)
    transfers = tuple(SimpleNamespace(bind_batch=Mock()) for _ in range(8))
    sources, requests = (source, other), ("a", "b")
    wrapper.bind_sources(sources, requests, lambda: transfers)
    if change in ("new_source", "restore"):
        sources = (SimpleNamespace(total_tokens=4096), other)
    elif change == "window":
        sources = (SimpleNamespace(total_tokens=4352), other)
    elif change == "reorder":
        sources, requests = (other, source), ("b", "a")
    elif change == "request":
        requests = ("new-a", "b")  # Even if a shared source object is reused.
    elif change == "empty":
        sources, requests = (), ()
    else:
        sources, requests = (source, None), ("a", "b")
    assert wrapper.bind_sources(sources, requests, lambda: transfers)
    for index, transfer in enumerate(transfers):
        assert transfer.bind_batch.call_count == 2
        transfer.bind_batch.assert_called_with(sources, index)
    assert not wrapper.bind_sources(sources, requests, Mock(side_effect=AssertionError("layer loop")))


def test_bindings_follow_shared_capacity_not_graph_key(graph_module):
    module, context, _, _ = graph_module
    wrapper = module.SFAFullGraph()
    source_a, source_b = object(), object()
    transfer = SimpleNamespace(bind_batch=Mock())
    lazy = Mock(return_value=(transfer,))
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1, name="q1")
    assert wrapper.bind_sources((source_a,), ("a",), lazy)
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1, name="q2")
    assert not wrapper.bind_sources((source_a,), ("a",), lazy)
    assert wrapper.bind_sources((source_b,), ("b",), lazy)
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1, name="q1")
    assert wrapper.bind_sources((source_a,), ("a",), lazy)
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=2, name="r2")
    assert wrapper.bind_sources((source_a,), ("a",), lazy)
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1, name="q1")
    assert not wrapper.bind_sources((source_a,), ("a",), lazy)
    assert lazy.call_count == 4


def test_failed_partial_binding_invalidates_previous_binding(graph_module):
    module, context, _, _ = graph_module
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1)
    wrapper = module.SFAFullGraph()
    transfers = tuple(SimpleNamespace(bind_batch=Mock()) for _ in range(3))
    source_a, source_b = object(), object()
    wrapper.bind_sources((source_a,), ("a",), lambda: transfers)
    transfers[1].bind_batch.side_effect = RuntimeError("invalid source layer")
    with pytest.raises(RuntimeError, match="invalid source layer"):
        wrapper.bind_sources((source_b,), ("b",), lambda: transfers)
    assert not wrapper.source_bindings
    transfers[1].bind_batch.side_effect = None
    assert wrapper.bind_sources((source_a,), ("a",), lambda: transfers)
    assert transfers[0].bind_batch.call_count == 3
    assert transfers[2].bind_batch.call_count == 2


def test_capture_reset_forces_rebinding_even_for_identical_sources(graph_module):
    module, context, _, _ = graph_module
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1)
    wrapper = module.SFAFullGraph()
    source = object()
    lazy = Mock(return_value=(SimpleNamespace(bind_batch=Mock()),))
    wrapper.bind_sources((source,), ("a",), lazy)
    wrapper.clear()
    assert not wrapper.source_bindings and wrapper.source_binding_count == 0
    assert wrapper.bind_sources((source,), ("a",), lazy)
    assert lazy.call_count == 2


@pytest.mark.parametrize("sources,requests", [((None, None), ("a", "b")), ((None,), ()), ((), ("a",))])
def test_invalid_source_lanes_fail_before_binding(graph_module, sources, requests):
    module, context, _, _ = graph_module
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1)
    lazy = Mock()
    with pytest.raises(ValueError, match="source lanes"):
        module.SFAFullGraph().bind_sources(sources, requests, lazy)
    lazy.assert_not_called()
