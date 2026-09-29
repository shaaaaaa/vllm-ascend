# SPDX-License-Identifier: Apache-2.0
"""Check actual SFA event boundaries without loading NPU kernels."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend/attention"


def compile_nodes(nodes, namespace):
    module = ast.parse("from __future__ import annotations")
    module.body.extend(nodes)
    exec(compile(ast.fix_missing_locations(module), "<bank-event>", "exec"), namespace)


def marker_runtime():
    tree = ast.parse((ROOT / "utils.py").read_text(encoding="utf-8"))
    method = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "record_layerwise_prefill_bank_use"
    )
    operations, events = [], []

    class Event:
        def record(self, stream):
            assert stream is operations
            self.predecessors = tuple(operations)

    connector = NS(record_layerwise_prefill_bank_use=lambda name, event: events.append((name, event)))
    namespace = dict(
        torch=NS(npu=NS(Event=Event, current_stream=lambda: operations)),
        get_kv_transfer_group=lambda: connector,
    )
    compile_nodes([method], namespace)
    return namespace, operations, events, connector


@pytest.mark.parametrize("p_node", [True, False])
@pytest.mark.parametrize("skip_topk,has_indexer", [(False, True), (True, True), (True, False)])
def test_indexer_handoff_precedes_topk_publication_and_latent_attention(p_node, skip_topk, has_indexer):
    namespace, operations, events, _ = marker_runtime()
    tree = ast.parse((ROOT / "sfa_v1.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
    forward = next(n for n in cls.body if getattr(n, "name", None) == "forward")
    indexer = next(
        n
        for n in forward.body
        if isinstance(n, ast.With) and ast.unparse(n.items[0].context_expr) == "_dsa_prof.section('indexer')"
    )
    operations.append("scatter index KV" if has_indexer else "shared topk")
    namespace.update(
        self=NS(
            _layerwise_prefill_p_node=p_node,
            skip_topk=skip_topk,
            has_indexer=has_indexer,
            index_cache_enabled=True,
            _get_indexcache_topk_indices=lambda n: operations.append("reuse topk"),
            indexer_select_post_process=lambda **kw: operations.append("read index KV"),
            _update_indexcache_topk_indices=lambda value: operations.append("publish topk"),
        ),
        hidden_states=NS(shape=(8,)),
        q_c=None,
        kv_cache=None,
        attn_metadata=None,
        cos=None,
        sin=None,
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        index_lmcache_enabled=has_indexer,
        index_layer_name="indexer",
        content_diagnostics_enabled=False,
        _dsa_prof=NS(section=lambda name: nullcontext()),
    )
    compile_nodes([indexer], namespace)
    operations.append("latent SFA")
    assert len(events) == int(p_node and has_indexer)
    if events:
        name, event = events[0]
        assert name == "indexer"
        assert "scatter index KV" in event.predecessors
        assert "publish topk" not in event.predecessors
        assert "latent SFA" not in event.predecessors
        if not skip_topk:
            assert event.predecessors[-1] == "read index KV"


@pytest.mark.parametrize("p_node", [True, False])
@pytest.mark.parametrize("layer_name", ["layer0", "last_target", "mtp"])
def test_latent_handoff_follows_sfa_before_projection(p_node, layer_name):
    namespace, operations, events, _ = marker_runtime()
    tree = ast.parse((ROOT / "sfa_v1.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
    forward = next(n for n in cls.body if getattr(n, "name", None) == "forward")
    index = next(
        i for i, n in enumerate(forward.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "attn_output is None"
    )
    namespace.update(
        self=NS(
            _layerwise_prefill_p_node=p_node,
            _execute_sparse_flash_attention_process=lambda *a, **kw: operations.append("read latent KV") or 1,
        ),
        attn_output=None,
        ql_nope=None,
        q_pe=None,
        kv_cache=None,
        topk_indices=None,
        attn_metadata=None,
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        layer_name=layer_name,
        _sparse_indices_padding_zeroed=False,
        _dsa_prof=NS(section=lambda name: nullcontext(), step=lambda: None),
    )
    compile_nodes(forward.body[index : index + 2], namespace)
    operations.extend(["v_up/o_proj", "FFN", "N+1 norm"])
    assert len(events) == int(p_node)
    if events:
        assert events[0][0] == layer_name
        assert events[0][1].predecessors == ("read latent KV",)


def test_missing_connector_handoff_api_fails_before_event_recording():
    namespace, _, events, connector = marker_runtime()
    del connector.record_layerwise_prefill_bank_use
    with pytest.raises(RuntimeError, match="update LMCache"):
        namespace["record_layerwise_prefill_bank_use"]("layer")
    assert events == []
