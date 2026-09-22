# SPDX-License-Identifier: Apache-2.0
"""Host tensor checks for shared producer indices and consumer remapping."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import torch


def consumer(shared, shrink):
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    names = {"_get_indexcache_topk_indices", "_update_indexcache_topk_indices"}
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), ns)
    obj = type("Consumer", (), {name: ns[name] for name in names})()
    obj.topk_indices_buffer = shared
    obj._indexcache_topk_staging = torch.empty_like(shared) if shrink else None
    return obj


def test_shrink_consumers_never_remap_each_others_raw_indices():
    shared = torch.arange(24).view(6, 4)
    original = shared.clone()
    first, second = consumer(shared, True), consumer(shared, True)
    first._get_indexcache_topk_indices(3).add_(100)
    assert torch.equal(shared, original)
    assert torch.equal(second._get_indexcache_topk_indices(3).squeeze(1), original[:3])
    first._update_indexcache_topk_indices(torch.full((3, 1, 4), 7))
    assert torch.equal(second._get_indexcache_topk_indices(3), torch.full((3, 1, 4), 7))


def test_read_only_consumer_does_not_add_a_copy():
    shared = torch.arange(24).view(6, 4)
    obj = consumer(shared, False)
    assert obj._get_indexcache_topk_indices(3).data_ptr() == shared.data_ptr()


def test_prefiller_shared_consumer_records_its_latent_write_event():
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    creation = next(n for n in forward.body if isinstance(n, ast.If)
                    and "torch.npu.Event()" in ast.unparse(n))
    code = compile(ast.fix_missing_locations(ast.Module(body=[creation], type_ignores=[])), str(path), "exec")
    for has_indexer in (False, True):
        recorded = []
        event = NS(record=lambda recorded=recorded: recorded.append(True))
        metadata = NS()
        exec(code, dict(self=NS(is_kv_producer=True, has_indexer=has_indexer),
                        kv_cache=object(), attn_metadata=metadata,
                        torch=NS(npu=NS(Event=lambda event=event: event))))
        assert metadata.reshape_cache_event is event
        # A full producer records later, after its indexer scatter as before.
        assert recorded == ([] if has_indexer else [True])


def test_staged_consumer_prepares_the_next_physical_indexer():
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    names = {"_cross_layer_kv_cache", "_layer_has_indexer_by_name", "cross_layer_lmcache_retrieve"}
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    waits = []
    stream = object()
    records = []
    event = NS(record=lambda actual: records.append(actual))
    ns = dict(
        torch=NS(npu=NS(current_stream=lambda: stream)), _staged_sfa_profile_scope=lambda _: nullcontext(),
        _dsa_index_lmcache_enabled=lambda: True,
        _dsa_indexer_layer_name=lambda name: name.rsplit(".attn", 1)[0] + ".indexer.k_cache",
        wait_for_kv_layer_from_connector=lambda name, **kw: waits.append(name),
        _prepare_sfa_remap_boundary=lambda *a, **kw: None,
        _LMCACHE_SPARSE_WAIT_SYNC_ONCE=False,
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"), ns)
    obj = type("Consumer", (), {name: ns[name] for name in names})()
    obj.has_indexer, obj.dsa_offload_unbundle = False, True
    obj.index_cache_enabled = True
    current, following = "model.layers.5.self_attn.attn", "model.layers.6.self_attn.attn"
    cache, index_name, enabled = obj._cross_layer_kv_cache(current, (object(), object()))
    assert index_name is None and enabled
    obj._staged_sfa_capture_state = NS(runtime=(current, cache, index_name, enabled), producer_event=event)
    obj._staged_sfa_bridge_buffers = None
    obj._staged_graph_content_diagnostic_enabled = lambda _: False
    obj._target_sfa_diag_pre_retrieve = lambda *a: None
    obj._target_sfa_diag_post_retrieve = lambda *a: None
    obj.index_topk = 4
    next_impl = NS(has_indexer=True)
    metadata = NS(decode_request_ids_compact=["r"], num_actual_tokens=1)
    context = NS(staged_sfa_graph_key=NS(token_capacity=1), staged_sfa_route=NS(frontiers=(4,)),
                 attn_metadata={following: NS(req_ids=["r"])},
                 no_compile_layers={following.rsplit(".attn", 1)[0]: NS(mla_attn=NS(impl=next_impl))})
    payload = torch.zeros(1, 4)
    obj.cross_layer_lmcache_retrieve(current, following, payload, payload, payload, metadata, context)
    assert waits == [current, "model.layers.6.self_attn.indexer.k_cache"]
    assert records == [stream]
    assert metadata.reshape_cache_event is event
    waits.clear()
    next_impl.has_indexer = False
    obj.cross_layer_lmcache_retrieve(current, following, payload, payload, payload, metadata, context)
    assert waits == [current]

    # A full-indexer model already knows that every following layer owns one;
    # do not add name parsing / wrapper lookup to its decode callback.
    obj.has_indexer = next_impl.has_indexer = True
    obj.index_cache_enabled = False
    def unexpected_lookup(*args):
        raise AssertionError("legacy decode performed a shared-indexer lookup")

    obj._layer_has_indexer_by_name = unexpected_lookup
    waits.clear()
    obj.cross_layer_lmcache_retrieve(current, following, payload, payload, payload, metadata, context)
    assert waits == [current, "model.layers.6.self_attn.indexer.k_cache"]
    assert records == [stream] * 3
