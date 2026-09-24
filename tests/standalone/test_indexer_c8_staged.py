# SPDX-License-Identifier: Apache-2.0
"""Run staged key/scale publication on CPU; NPU operator behavior is separate."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("idle", [False, True])
def test_staged_scatter_keeps_key_scale_pair_and_padding(c8, idle):
    source = Path(__file__).parents[2] / "vllm_ascend/attention/sfa_v1.py"
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl"
    )
    names = {"_cross_layer_pre_compute", "_mask_staged_index_scatter_padding"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(ast.ClassDef(name="Impl", bases=[], keywords=[], body=methods, decorator_list=[]))
    calls = []

    def scatter(cache, indices, updates):
        calls.append(cache.dtype)
        for row, slot in enumerate(indices.flatten().tolist()):
            assert 0 <= slot < len(cache)
            cache[slot].copy_(updates[row])

    ns = dict(
        torch=torch,
        torch_npu=SimpleNamespace(npu_scatter_nd_update_=scatter),
        get_forward_context=lambda: SimpleNamespace(staged_sfa_graph_key=None),
        get_weight_prefetch_method=lambda: SimpleNamespace(maybe_prefetch_mla_or_sla_weight_in_current_stream=Mock()),
    )
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    impl = ns["Impl"]()
    impl.use_sparse_c8_indexer = c8
    impl.has_indexer = True
    impl.skip_topk = False
    impl.q_lora_rank = impl.kv_lora_rank = impl.qk_rope_head_dim = 1
    impl.q_a_layernorm = lambda x: x
    impl.fused_qkv_a_proj = Mock(return_value=(torch.ones(4, 3),))
    impl.exec_kv = Mock()
    impl._q_proj_and_k_up_proj = lambda x: (x, x)
    impl.rope_single = lambda x, *args: x
    keys = torch.full((1, 8, 1, 128), 7, dtype=torch.int8 if c8 else torch.bfloat16)
    scales = torch.full((1, 8, 1, 1), 3, dtype=torch.float16) if c8 else None
    new_keys = torch.arange(4, dtype=keys.dtype)[:, None].expand(4, 128).clone()
    new_scales = torch.arange(4, dtype=torch.float16)[:, None] + 1 if c8 else None
    impl.indexer_select_pre_process = lambda **kw: (new_keys, new_scales)

    class ReachedIndexer(Exception):
        pass

    def indexer(**kwargs):
        # Both planes must be published before the quantized indexer reads them.
        cache = kwargs["kv_cache"]
        assert len(cache) == (4 if c8 else 3)
        assert cache[2] is keys
        if c8:
            assert cache[3] is scales
        raise ReachedIndexer

    impl.indexer_select_post_process = indexer
    fn = impl._cross_layer_pre_compute
    args = {name: None for name in fn.__func__.__code__.co_varnames[1 : fn.__func__.__code__.co_argcount]}
    args.update(
        hidden_states=torch.ones(4, 1),
        indexer_cache=keys,
        indexer_scale_cache=scales,
        indexer_slot_mapping=torch.tensor([2, 4, -1, 999]),
        row_req_indices=torch.tensor([-1, -1, -1, -1] if idle else [0, 0, -1, -1]),
    )
    with pytest.raises(ReachedIndexer):
        fn(**args)
    expected_keys = torch.full_like(keys, 7)
    if not idle:
        expected_keys.view(8, 128)[2] = new_keys[0]
        expected_keys.view(8, 128)[4] = new_keys[1]
    assert torch.equal(keys, expected_keys)
    assert calls == ([torch.float16, torch.int8] if c8 else [torch.bfloat16])
    if c8:
        expected_scales = torch.full_like(scales, 3)
        if not idle:
            expected_scales.view(8, 1)[2] = new_scales[0]
            expected_scales.view(8, 1)[4] = new_scales[1]
        assert torch.equal(scales, expected_scales)


@pytest.mark.parametrize("c8,producer", [(False, True), (True, True), (True, False)])
def test_unbundled_staged_binding_retains_scales(c8, producer):
    source = Path(__file__).parents[2] / "vllm_ascend/attention/sfa_v1.py"
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_cross_layer_kv_cache")
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(method)
    keys, scales = torch.empty(1), torch.empty(1)
    cache = (keys, scales) if c8 else (keys,)
    registry = {"index": SimpleNamespace(kv_cache=[cache])} if producer else {}
    get_context = Mock(return_value=SimpleNamespace(no_compile_layers=registry, virtual_engine=0))
    ns = dict(
        _dsa_indexer_layer_name=lambda _: "index",
        get_forward_context=get_context,
        _dsa_index_lmcache_enabled=lambda: True,
    )
    exec(compile(tree, str(source), "exec"), ns)
    impl = SimpleNamespace(dsa_offload_unbundle=True, has_indexer=producer, use_sparse_c8_indexer=c8 and producer)
    latent = (torch.empty(1), torch.empty(1))
    for _ in range(2):
        actual, name, enabled = ns[method.name](impl, "attn", latent)
        expected = latent + cache if producer else latent
        assert len(actual) == len(expected)
        assert all(a is b for a, b in zip(actual, expected))
        assert name == ("index" if producer else None)
        assert enabled
    assert get_context.call_count == int(producer)


@pytest.mark.parametrize("c8", [False, True])
def test_first_consume_diagnostic_uses_physical_indexer_table(c8):
    source = Path(__file__).parents[2] / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "queue_group1_first_consume"
    )
    expression = next(k.value for k in call.keywords if k.arg == "indexer_block_table")
    logical = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    physical = logical * 2
    ns = dict(
        self=SimpleNamespace(use_sparse_c8_indexer=c8),
        attn_metadata=SimpleNamespace(indexer_block_table=logical, indexer_c8_block_table=physical),
    )
    actual = eval(compile(ast.Expression(body=expression), str(source), "eval"), ns)
    assert actual is (physical if c8 else logical)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_c8_hadamard_matches_activation_dtype_and_survives_later_model(dtype):
    source = Path(__file__).parents[2] / "vllm_ascend/attention/sfa_v1.py"
    cls = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process_weights_after_loading")
    nodes = [
        n
        for n in method.body
        if isinstance(n, ast.If)
        and any(isinstance(x, ast.Attribute) and x.attr in ("q_hadamard", "k_hadamard") for x in ast.walk(n))
    ]
    # Include any startup bindings immediately following the shared matrices.
    nodes += [
        n
        for n in method.body
        if isinstance(n, ast.Assign)
        and any(isinstance(x, ast.Attribute) and x.attr in ("q_hadamard", "k_hadamard") for x in ast.walk(n))
    ]
    state = type("State", (), {"q_hadamard": None, "k_hadamard": None})

    def host_tensor(data, **kwargs):
        kwargs["device"] = "cpu"
        return torch.tensor(data, **kwargs)

    proxy = SimpleNamespace(tensor=host_tensor, bfloat16=torch.bfloat16)
    # A signed Hadamard matrix generated without a SciPy test dependency.
    basis = torch.ones(1, 1)
    while basis.shape[0] < 128:
        basis = torch.cat((torch.cat((basis, basis), 1), torch.cat((basis, -basis), 1)), 0)
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec")
    first = state()
    first.use_sparse_c8_indexer = True
    namespace = dict(
        self=first,
        AscendSFAImpl=state,
        act_dtype=dtype,
        torch=proxy,
        scipy=SimpleNamespace(linalg=SimpleNamespace(hadamard=lambda _: basis.numpy())),
    )
    exec(code, namespace)
    values = torch.arange(128, dtype=dtype).view(1, 128)
    assert first.q_hadamard.dtype == first.k_hadamard.dtype == dtype
    expected = values @ (basis.to(dtype) / (128**0.5))
    assert torch.equal(values @ first.q_hadamard, expected)
    second = state()
    second.use_sparse_c8_indexer = True
    other = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    exec(code, dict(namespace, self=second, act_dtype=other))
    assert second.q_hadamard.dtype == other
    assert first.q_hadamard.dtype == dtype
    assert torch.equal(values @ first.k_hadamard, expected)
    for name in ("indexer_select_pre_process", "indexer_select_post_process"):
        compute = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        product = next(
            n
            for n in ast.walk(compute)
            if isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.MatMult)
            and isinstance(n.right, ast.Attribute)
            and n.right.attr in ("q_hadamard", "k_hadamard")
        )
        result = eval(
            compile(ast.Expression(body=product), str(source), "eval"),
            dict(self=first, AscendSFAImpl=state, k_li=values, q_li=values),
        )
        assert torch.equal(result, expected)
