# SPDX-License-Identifier: Apache-2.0
"""Target-only mask elision; native/draft callers retain the mask contract."""

import ast
import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from sfa_test_support import ROOT
from test_sfa_dcp_prepare import ops as preparation_ops


@pytest.mark.parametrize("bounded,shared", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("source", ["c8", "local_bf16", "external", "consumer", "legacy_consumer", "cp", "draft"])
def test_target_caller_elides_only_proven_masks(monkeypatch, bounded, shared, source):
    prepare, _, calls = preparation_ops.__wrapped__()
    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    module.prepare_sfa_dcp_topk = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = ROOT / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_cross_layer_pre_compute")
    eligibility = next(n for n in method.body
                       if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "causal_topk")
    branch = next(n for n in method.body if isinstance(n, ast.If)
                  and any(isinstance(c, ast.ImportFrom) and c.names[0].name == "prepare_sfa_dcp_topk" for c in n.body))
    qualified = source in ("c8", "local_bf16")
    raw = torch.tensor([[[0, 127, 128, -1]], [[0, 127, 128, -1]]], dtype=torch.int32)
    positions = torch.tensor([128, 128 if qualified else -1])
    original = raw.clone()
    dcp = NS(
        topk=torch.full_like(raw, -77),
        counts=torch.full((2, 16), -77, dtype=torch.int32),
        positions=positions,
        layout=NS(world_size=8, rank=0),
        mask_indexer_writes=source == "draft",
    )
    ns = dict(
        self=NS(
            has_indexer=source != "consumer",
            skip_topk=source in ("consumer", "legacy_consumer"),
            dsa_cp_decoder=source == "cp",
            block_size=128,
            use_sparse_c8_indexer=source == "c8",
            use_torch_npu_lightning_indexer=source == "external",
        ),
        dcp=dcp,
        topk_indices=raw,
        bounded_decode=bounded,
        group=NS(active=True) if shared else None,
        owner_rows=None,
    )
    exec(compile(ast.Module(body=[eligibility, branch], type_ignores=[]), str(path), "exec"), ns)
    assert torch.equal(raw, original) and dcp.counts.eq(-77).all()
    if qualified:
        assert calls == []
        if bounded or shared:
            assert ns["topk_indices"] is raw and dcp.topk.eq(-77).all()
        else:
            assert ns["topk_indices"] is dcp.topk and torch.equal(dcp.topk, raw)
            dcp.topk.fill_(123)
            assert torch.equal(raw, original)
    else:
        assert calls[0][0] == "_mask_topk"
        assert ns["topk_indices"] is dcp.topk and dcp.topk[1].eq(-1).all()


@pytest.mark.parametrize("local,mixed", [(True, False), (False, True), (True, True)])
def test_causal_hint_never_skips_rank_compaction(local, mixed):
    prepare, _, calls = preparation_ops.__wrapped__()
    raw = torch.tensor([[[0, 128, 129, -1]]], dtype=torch.int32)
    out, counts = torch.empty_like(raw), torch.full((1, 16), -77, dtype=torch.int32)
    prepare(
        raw,
        torch.tensor([128]),
        out,
        counts,
        1,
        8,
        1,
        128,
        local=local,
        row_requests=torch.tensor([-1]) if mixed else None,
        _causal_global_ids=True,
    )
    assert calls == [("_prepare_topk", (1,))]
    assert out.tolist() == [[[0, -1, -1, -1]]] and counts[0, 0] == 1
