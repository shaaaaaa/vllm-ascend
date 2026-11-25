# SPDX-License-Identifier: Apache-2.0
"""Execute the actual SFA methods without importing the NPU model stack.

Device operations are modeled; native stream ordering has a separate NPU test.
Unlike a permissive event mock, waits require a record for the current payload.
"""

import ast
from collections import namedtuple
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

SOURCE = (Path(__file__).parents[3] / "vllm_ascend/attention/sfa_v1.py").read_text(encoding="utf-8")


@pytest.fixture
def handoff():
    operations = []
    device = SimpleNamespace(generation=0, capturing=False)

    class Event:
        def __init__(self):
            assert not device.capturing
            self.generation = -1

        def record(self, stream=None):
            assert not device.capturing, "handoff record must stay outside capture"
            assert stream is None or stream is device.stream
            self.generation = device.generation
            operations.append("record")

        def wait(self):
            assert self.generation == device.generation, "missing current producer record"
            operations.append("wait")

    device.stream = SimpleNamespace(wait_event=lambda event: event.wait())
    npu = SimpleNamespace(Event=Mock(side_effect=Event), current_stream=lambda: device.stream)
    namespace = dict(
        torch=SimpleNamespace(npu=npu),
        _staged_sfa_profile_scope=lambda _: nullcontext(),
        CUDAGraphMode=SimpleNamespace(NONE=0, PIECEWISE=1),
        StagedSFARouteAction=SimpleNamespace(STAGED="staged"),
        _prepare_sfa_remap_boundary=Mock(),
        _LMCACHE_SPARSE_WAIT_SYNC_ONCE=False,
    )
    cls = next(
        node for node in ast.parse(SOURCE).body if isinstance(node, ast.ClassDef) and node.name == "AscendSFAImpl"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"cross_layer_graph_pre", "cross_layer_lmcache_retrieve"}
    ]
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(methods)
    exec(compile(tree, "sfa_event_handoff", "exec"), namespace)
    key = namedtuple("GraphKey", "request_capacity token_capacity")(2, 2)
    context = SimpleNamespace(
        staged_sfa_graph_key=key,
        staged_sfa_graph_dummy_run=False,
        staged_sfa_route=SimpleNamespace(action="staged", graph_key=key, frontiers=(8, 8)),
        cudagraph_runtime_mode=0,
    )
    namespace["get_forward_context"] = lambda: context
    bridge = (
        torch.zeros(2, 1, 2),
        torch.zeros(2, 1, 2),
        torch.zeros(2, 1, 4, dtype=torch.int32),
        torch.zeros(2, 4, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, 4, dtype=torch.int64),
    )
    metadata = MagicMock(
        decode_request_ids_compact=["a", "b"],
        num_actual_tokens=2,
        num_decode_tokens=2,
        indexer_slot_mapping=torch.zeros(2, dtype=torch.int64),
    )
    state = SimpleNamespace(
        producer_event=Event(),
        runtime=(None, (torch.zeros(2, 4, 1, 2),) * 3, None, False),
        remap_boundary=None,
        initialized_cache_capacity=2,
        register=Mock(),
    )
    impl = SimpleNamespace(
        _staged_sfa_capture_state=state,
        _staged_sfa_bridge_buffers=bridge,
        _cross_layer_kv_cache=Mock(return_value=(state.runtime[1], None, False)),
        _cross_layer_ineligible_reason=Mock(return_value=None),
        _cross_layer_pre_compute=Mock(return_value=bridge),
        _copy_to_staged_sfa_bridge=lambda *args: bridge,
        index_topk=4,
        skip_topk=False,
        has_indexer=True,
        use_sparse_c8_indexer=False,
        index_cache_enabled=False,
    )

    def connector(_, **kwargs):
        if "payload_event" not in kwargs:
            operations.append("indexer")
            return
        assert metadata.reshape_cache_event is kwargs["payload_event"]
        assert kwargs["payload_event"] is state.producer_event
        device.stream.wait_event(kwargs["payload_event"])
        operations.append("connector")

    namespace["wait_for_kv_layer_from_connector"] = Mock(side_effect=connector)

    def retrieve(next_layer=""):
        namespace["cross_layer_lmcache_retrieve"](impl, "layer", next_layer, *bridge[3:], metadata, context)

    def pre():
        return namespace["cross_layer_graph_pre"](impl, "layer", bridge[0], (), metadata, False, bridge[0])

    return SimpleNamespace(**locals())


@pytest.mark.parametrize("capacities", [(2, 2, 2, 2), (2, 4, 2, 4)])
def test_every_staged_replay_records_before_connector_wait(handoff, capacities):
    h = handoff
    event = h.state.producer_event
    for generation, capacity in enumerate(capacities, start=1):
        h.device.generation = generation  # Replay does not call Python graph_pre.
        h.context.staged_sfa_graph_key = type(h.key)(capacity, capacity)
        h.retrieve()
        assert h.state.producer_event is event
    assert h.operations == ["record", "wait", "connector"] * len(capacities)
    h.npu.Event.assert_not_called()


@pytest.mark.parametrize("failure", ["missing", "record"])
def test_failed_handoff_does_not_consume_payload(handoff, failure):
    h = handoff
    if failure == "missing":
        h.state.producer_event = None
    else:
        h.state.producer_event.record = Mock(side_effect=RuntimeError("record failed"))
    with pytest.raises(RuntimeError, match="warmup|record failed"):
        h.retrieve()
    h.namespace["wait_for_kv_layer_from_connector"].assert_not_called()
    assert h.operations == []


def test_dummy_does_not_record_or_load(handoff):
    h = handoff
    h.context.staged_sfa_graph_dummy_run = True
    h.device.capturing = True
    h.retrieve()
    assert h.operations == []
    h.namespace["wait_for_kv_layer_from_connector"].assert_not_called()


def test_unconfigured_retrieve_does_not_record_or_load(handoff):
    h = handoff
    h.context.staged_sfa_graph_key = None
    h.retrieve()
    assert h.operations == []
    h.namespace["wait_for_kv_layer_from_connector"].assert_not_called()


def test_failed_initial_record_does_not_publish_event(handoff):
    h = handoff
    h.state.producer_event = None
    h.npu.Event.side_effect = None
    h.npu.Event.return_value = SimpleNamespace(record=Mock(side_effect=RuntimeError("init failed")))
    with pytest.raises(RuntimeError, match="init failed"):
        h.pre()
    assert h.state.producer_event is None
    h.impl._cross_layer_pre_compute.assert_not_called()


def test_eager_warmup_materializes_once_capture_never_records(handoff):
    h = handoff
    h.state.producer_event = None
    h.context.staged_sfa_graph_dummy_run = True
    h.pre()
    event = h.state.producer_event
    h.npu.Event.assert_called_once_with()
    assert h.operations == ["record"]
    h.state.remap_boundary = h.metadata.decode_remap_boundary
    h.context.cudagraph_runtime_mode = 1
    h.device.capturing = True
    h.pre()
    assert h.state.producer_event is event
    assert h.operations == ["record"]
    h.state.register.assert_called_once()


def test_capture_cannot_initialize_missing_event(handoff):
    h = handoff
    h.state.producer_event = None
    h.context.cudagraph_runtime_mode = 1
    h.device.capturing = True
    with pytest.raises(RuntimeError, match="not created by eager warmup"):
        h.pre()
    h.npu.Event.assert_not_called()


@pytest.mark.parametrize("requests", [["a"], ["b", "a"], ["new", "resumed"]])
def test_handoff_preserves_payload_and_request_order(handoff, requests):
    h = handoff
    h.metadata.decode_request_ids_compact = requests
    for tensor in h.bridge[3:]:
        tensor.copy_(torch.arange(tensor.numel()).reshape(tensor.shape))
    before = [tensor.clone() for tensor in h.bridge[3:]]
    h.retrieve()
    kwargs = h.namespace["wait_for_kv_layer_from_connector"].call_args.kwargs
    assert kwargs["request_ids"] is requests
    assert kwargs["token_start_index"] is None
    for name, tensor, snapshot in zip(
        ("selected_tokens", "selected_token_counts", "target_slot_mapping"), h.bridge[3:], before
    ):
        assert torch.equal(tensor, snapshot)
        assert kwargs[name].data_ptr() == tensor.data_ptr()
        assert torch.equal(kwargs[name], snapshot[: len(requests)])


@pytest.mark.parametrize("has_next_indexer", [False, True])
def test_shared_indexer_prefetch_order_is_preserved(handoff, has_next_indexer):
    h = handoff
    h.state.runtime = (None, (), None, True)
    h.impl.index_cache_enabled = True
    h.impl._layer_has_indexer_by_name = lambda *_: has_next_indexer
    h.context.attn_metadata = {"next": h.metadata}
    h.namespace["_dsa_indexer_layer_name"] = lambda name: name + ".indexer"
    h.retrieve("next")
    expected = ["record", "wait", "connector"] + (["indexer"] if has_next_indexer else [])
    assert h.operations == expected
    h.namespace["_prepare_sfa_remap_boundary"].assert_called_once()


@pytest.mark.parametrize("full,consumer,enabled", [(True, False, True), (False, False, True),
                                                    (True, True, True), (True, False, False)])
def test_indexer_launches_from_its_fixed_bridge_before_own_retrieval(handoff, full, consumer, enabled):
    h = handoff
    h.context.sfa_full_graph_active = full
    h.impl.skip_topk = consumer
    h.impl._shared_retrieval_edges = lambda: ({}, {})
    h.impl._cross_layer_pre_compute.side_effect = lambda *a, **kw: h.operations.append("index plan") or h.bridge
    h.impl._copy_to_staged_sfa_bridge = lambda *a: h.operations.append("publish bridge") or h.bridge
    group = SimpleNamespace(launch=Mock(side_effect=lambda *a: h.operations.append("queue consumers")))
    h.impl._retrieval_overlap = SimpleNamespace(
        multi_layer=enabled, indexer_groups={h.context.staged_sfa_graph_key: {"layer": group}}
    )
    h.impl.sfa_copy_max_aiv_cores = 12
    h.impl._full_graph_transfer = SimpleNamespace(load=Mock(side_effect=lambda *a, **kw: h.operations.append("own KV")))
    h.pre()
    if full and not consumer and enabled:
        group.launch.assert_called_once_with(*h.bridge[3:])
        h.retrieve()
        assert h.impl._full_graph_transfer.load.call_args.kwargs == {"max_aiv_cores": 12}
        assert h.operations == ["index plan", "publish bridge", "queue consumers", "own KV"]
    else:
        group.launch.assert_not_called()


def test_native_fallback_does_not_touch_staged_event(handoff):
    h = handoff
    h.context.staged_sfa_graph_key = None
    h.impl.forward = Mock()
    h.impl._cross_layer_empty_outputs = Mock()
    h.pre()
    h.impl.forward.assert_called_once()
    assert h.operations == []
    h.npu.Event.assert_not_called()


def test_group_consumer_carries_its_own_event_into_the_fused_qkv(handoff):
    h = handoff
    edge = SimpleNamespace(from_indexer=True, join=Mock())
    h.context.sfa_full_graph_active = True
    h.impl.skip_topk = True
    h.impl._retrieval_overlap = SimpleNamespace(multi_layer=True)
    h.impl._shared_retrieval_edges = lambda: ({"layer": edge}, {})
    h.pre()
    edge.join.assert_not_called()
    assert h.impl._cross_layer_pre_compute.call_args.kwargs["qkv_prefetch"] is edge
    h.impl._full_graph_transfer = SimpleNamespace(load=Mock())
    h.retrieve()
    h.impl._full_graph_transfer.load.assert_not_called()


def test_connector_failure_does_not_advance_to_next_layer(handoff):
    h = handoff
    h.namespace["wait_for_kv_layer_from_connector"].side_effect = RuntimeError("load failed")
    with pytest.raises(RuntimeError, match="load failed"):
        h.retrieve("next")
    h.namespace["_prepare_sfa_remap_boundary"].assert_not_called()
    assert h.operations == ["record"]
