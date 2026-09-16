# SPDX-License-Identifier: Apache-2.0
"""Shared-indexer ownership must precede bounded full-graph lane remapping."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("shared_model", [False, True])
def test_producer_publishes_raw_topk_before_pack_and_consumer_has_no_indexer(bounded, shared_model):
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and any(isinstance(c, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "bounded_decode"
                                                           for t in c.targets) for c in n.body)
                  and "npu_scatter_nd_update_" in ast.unparse(n))
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "self.has_indexer" and "npu_scatter_nd_update_" in ast.unparse(n))
    stop = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "staged_mtp" for t in n.targets))
    code = compile(ast.Module(body=method.body[start:stop], type_ignores=[]), str(path), "exec")
    names = {"_get_indexcache_topk_indices", "_update_indexcache_topk_indices"}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    api = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), api)
    shared = torch.zeros(2, 4, dtype=torch.int32)
    raw = torch.arange(8, dtype=torch.int32).view(2, 1, 4)
    events = []

    def select(*, x, q_c, **kwargs):
        assert x is hidden and q_c is query
        events.append("select")
        return raw.clone()

    def pack(topk, boundary, rows, lengths, width):
        events.append("pack")
        topk.add_(100)  # Deliberately destructive: shared raw indices must survive.
        return topk, boundary, rows, torch.arange(2)

    owner = NS(has_indexer=True, skip_topk=False, index_cache_enabled=shared_model,
               topk_indices_buffer=shared, _indexcache_topk_staging=torch.empty_like(shared),
               _mask_staged_index_scatter_padding=lambda slots, keys, *a: (slots, keys),
               indexer_select_post_process=select)

    def publish(value):
        events.append("publish")
        api["_update_indexcache_topk_indices"](owner, value)

    def reuse(count):
        events.append("reuse")
        return api["_get_indexcache_topk_indices"](owner, count)

    owner._update_indexcache_topk_indices = publish
    owner._get_indexcache_topk_indices = reuse
    hidden, query = torch.zeros(2, 8), torch.zeros(2, 8)
    key = NS(query_profile="bounded" if bounded else "fixed", request_capacity=2, max_query_len=2)
    ns = dict(self=owner, torch_npu=NS(npu_scatter_nd_update_=lambda *a: events.append("scatter")),
              indexer_slot_mapping=torch.arange(2), k_li=torch.zeros(2, 4), indexer_cache=torch.zeros(2, 4),
              row_req_indices=torch.arange(2), hidden_states=hidden, q_c=query, kv_cache=None,
              cos=None, sin=None, actual_seq_lengths_query=torch.tensor([1, 3]), actual_seq_lengths_key=None,
              indexer_block_table=None, get_forward_context=lambda: NS(staged_sfa_graph_key=key),
              StagedSFAQueryProfile=NS(DECODE_BOUNDED="bounded"), pack_decode_lanes=pack,
              request_block_table=torch.zeros(2, 3), remap_boundary=torch.zeros(2))
    exec(code, ns)
    assert events == ["scatter", "select"] + (["publish"] if shared_model else []) + (["pack"] if bounded else [])
    if shared_model:
        assert torch.equal(shared, raw.squeeze(1))
        owner.has_indexer, owner.skip_topk = False, True
        ns["indexer_cache"] = ns["k_li"] = None
        events.clear()
        exec(code, ns)
        assert events == ["reuse"] + (["pack"] if bounded else [])
        assert torch.equal(shared, raw.squeeze(1))
        assert torch.equal(ns["topk_indices"], raw + (100 if bounded else 0))
