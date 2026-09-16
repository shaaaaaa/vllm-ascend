# SPDX-License-Identifier: Apache-2.0
"""Execute the real builder and runner bank loop on CPU (no NPU imports)."""

import ast
from copy import copy
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def source_class(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def execute(nodes, namespace):
    # Postpone production type annotations without importing accelerator modules.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, "<real-prefill-metadata>", "exec"), namespace)


@pytest.fixture
def builder_class():
    state = source_class("vllm_ascend/attention/attention_v1.py", "AscendAttentionState")
    cls = source_class("vllm_ascend/attention/sfa_v1.py", "AscendSFAMetadataBuilder")
    names = {"build", "rebind_layerwise_prefill_metadata"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls = ast.ClassDef(name="Builder", bases=[], keywords=[], body=methods, decorator_list=[])
    rope = Mock(side_effect=lambda positions, _: (positions.float()[:, None], -positions.float()[:, None]))
    ns = dict(copy=copy, Enum=Enum, torch=torch, get_cos_and_sin_mla=rope)
    execute([state, cls], ns)
    return ns["Builder"], ns["AscendAttentionState"], rope


def new_builder(cls):
    builder = cls()
    builder.dsa_shrink_latent = False
    builder.enable_dsa_cp = False
    builder.scratch_capacity = None
    builder.decode_remap_boundary = torch.zeros(16, dtype=torch.int32)
    builder.model_config = SimpleNamespace(get_head_size=lambda: 576)
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda _: None)
    # num_decode_tokens is explicitly returned by real build; other dataclass
    # defaults used by the rebind path are included in the CPU constructor.
    builder.metadata_cls = lambda **kw: SimpleNamespace(**kw)
    return builder


def common(state, step=0):
    return SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=7,
        num_input_tokens=8,
        positions=torch.arange(8) + step * 8,
        query_start_loc=torch.tensor([0, 3, 7]),
        query_start_loc_cpu=torch.tensor([0, 3, 7]),
        seq_lens=torch.tensor([3, 4]) + step * 8,
        seq_lens_cpu=torch.tensor([3, 4]) + step * 8,
        attn_state=state,
        block_table_tensor=torch.zeros(3, 5, dtype=torch.int32),
        slot_mapping=torch.zeros(10, dtype=torch.long),
        indexer_block_table_tensor=None,
        indexer_slot_mapping=None,
    )


def assert_same(actual, expected):
    assert vars(actual).keys() == vars(expected).keys()
    for field, value in vars(expected).items():
        other = getattr(actual, field)
        if isinstance(value, torch.Tensor):
            assert torch.equal(other, value), field
        else:
            assert other == value, field


@pytest.mark.parametrize("state_name", ["PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill"])
@pytest.mark.parametrize("layers", [8, 79, 80])
def test_real_runner_reuses_build_but_preserves_all_layer_banks(builder_class, state_name, layers):
    cls, states, rope = builder_class
    builder = new_builder(cls)
    builder.build = Mock(wraps=builder.build)
    oracle = new_builder(cls)
    runner_cls = source_class("vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner")
    fn = next(n for n in runner_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_attention_metadata")
    helper = next(n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == "_build_attn_group_metadata")
    # Execute the actual layer loop, not a test reimplementation of its reuse.
    group_loop = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For)
        and ast.unparse(n.target) == "attn_gid"
        and "shared_prefill_metadata" in ast.unparse(n)
    )
    bank_helper = next(
        n for n in runner_cls.body if getattr(n, "name", None) == "_layerwise_prefill_common_attn_metadata"
    )
    sibling = next(n for n in runner_cls.body if getattr(n, "name", None) == "_layerwise_prefill_indexer_sibling")
    runner_stub = ast.ClassDef(name="Runner", bases=[], keywords=[], body=[bank_helper, sibling], decorator_list=[])
    names = [f"model.layers.{i}.self_attn.attn" for i in range(layers)]
    full = [i for i in range(layers) if i < 3 or (i >= 6 and (i - 6) % 4 == 0) or i == 79]
    refs = {name: SimpleNamespace(kv_group=0, bank=i % 2) for i, name in enumerate(names)}
    for ordinal, layer in enumerate(full):
        refs[names[layer].rsplit(".", 1)[0] + ".indexer.k_cache"] = SimpleNamespace(kv_group=1, bank=ordinal % 2)
    group = SimpleNamespace(layer_names=names, get_metadata_builder=lambda _: builder)
    ns = dict(
        copy=copy,
        GDNAttentionMetadataBuilder=type("GDN", (), {}),
        AscendEagleProposer=type("Eagle", (), {}),
        AscendDraftModelProposer=type("Draft", (), {}),
    )
    execute([runner_stub, helper], ns)
    runner = ns["Runner"]()
    runner.layerwise_prefill_p_node = True
    runner.attn_groups = [[group]]
    runner._layerwise_prefill_refs = lambda: refs
    runner.speculative_config = True
    runner.drafter = ns["AscendEagleProposer"]()
    runner.drafter.attn_layer_names = [names[-1]]
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False))
    )
    previous = None
    for step in range(2):
        cm = common(getattr(states, state_name), step)
        tables = {
            (gid, bank): (
                torch.full((3, 5), 100 * step + 10 * gid + bank, dtype=torch.int32),
                torch.arange(10) + 100 * step + 10 * gid + bank,
            )
            for gid in (0, 1)
            for bank in (0, 1)
        }
        getter = lambda gid, bank, tables=tables: tables[gid, bank]
        output = {}
        ns.update(
            self=runner,
            kv_cache_gid=0,
            cm=cm,
            attn_metadata=output,
            cascade_attn_prefix_lens=None,
            use_spec_decode=False,
            for_cudagraph_capture=False,
            spec_decode_common_attn_metadata=None,
            _get_block_table_and_slot_mapping=getter,
        )
        builder.build.reset_mock()
        rope.reset_mock()
        execute([group_loop], ns)
        assert builder.build.call_count == 1
        assert rope.call_count == 1
        assert len({id(m) for m in output.values()}) == layers
        first = output[names[0]]
        assert ns["spec_decode_common_attn_metadata"].block_table_tensor is tables[0, (layers - 1) % 2][0]
        for name in names:
            ref = refs[name]
            index = refs.get(runner._layerwise_prefill_indexer_sibling(name))
            layer_cm = runner._layerwise_prefill_common_attn_metadata(
                cm, 0, ref.bank, index.bank if index else None, getter
            )
            expected = oracle.build(0, layer_cm)
            assert_same(output[name], expected)
            assert output[name].cos is first.cos
            assert output[name].sin is first.sin
        if previous is not None:
            assert not torch.equal(previous.cos, first.cos)
            assert not torch.equal(previous.block_table, first.block_table)
        previous = first


@pytest.mark.parametrize(
    "cp,state_name,num_decode",
    [(True, "ChunkedPrefill", 0), (False, "DecodeOnly", 0), (False, "SpecDecoding", 0), (False, "ChunkedPrefill", 2)],
)
def test_unsupported_reuse_returns_none(builder_class, cp, state_name, num_decode):
    cls, states, _ = builder_class
    builder = new_builder(cls)
    builder.enable_dsa_cp = cp
    template = SimpleNamespace(attn_state=getattr(states, state_name), num_decode_tokens=num_decode)
    assert builder.rebind_layerwise_prefill_metadata(template, None) is None


@pytest.mark.parametrize("p_node,routed", [(True, False), (False, False), (True, True)])
def test_bank_table_padding_prepared_once_per_forward(p_node, routed):
    cls = source_class("vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner")
    fn = next(n for n in cls.body if getattr(n, "name", None) == "_build_attention_metadata")
    init = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "prefill_bank_views")
    getter = next(n for n in fn.body if getattr(n, "name", None) == "_get_block_table_and_slot_mapping")
    runner = SimpleNamespace(
        layerwise_prefill_p_node=p_node,
        use_cp=False,
        pcp_size=1,
        model_config=SimpleNamespace(enable_return_routed_experts=routed),
    )
    encoder_cls = type("EncoderOnlyAttentionSpec", (), {})
    for step in range(2):
        tables = [
            [
                SimpleNamespace(
                    get_device_tensor=Mock(return_value=torch.full((3, 5), 100 * step + 10 * group + bank)),
                    slot_mapping=SimpleNamespace(gpu=torch.arange(16) + 100 * step + 10 * group + bank),
                )
                for group in range(2)
            ]
            for bank in range(2)
        ]
        runner.input_batch = SimpleNamespace(layerwise_prefill_block_tables=tables, block_table=tables[0])
        ns = dict(
            self=runner,
            torch=torch,
            num_reqs=2,
            num_reqs_padded=3,
            num_tokens=7,
            num_tokens_padded=8,
            EncoderOnlyAttentionSpec=encoder_cls,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=object()) for _ in range(2)],
        )
        execute([init, getter], ns)
        for _ in range(5):
            for group in range(2):
                for bank in range(2):
                    blocks, slots = ns["_get_block_table_and_slot_mapping"](group, bank)
                    assert (blocks[:2] == 100 * step + 10 * group + bank).all()
                    assert (blocks[2:] == 0).all()
                    assert slots[7] == -1
                    assert slots.shape == (8,)
        for bank_tables in tables:
            for table in bank_tables:
                assert table.get_device_tensor.call_count == (1 if p_node and not routed else 5)


@pytest.mark.parametrize("capture,has_rebind", [(False, False), (False, True), (True, False), (True, True)])
def test_capture_and_unsupported_builders_keep_original_build_path(capture, has_rebind):
    cls = source_class("vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner")
    fn = next(n for n in cls.body if getattr(n, "name", None) == "_build_attention_metadata")
    helper = next(n for n in fn.body if getattr(n, "name", None) == "_build_attn_group_metadata")
    builder = SimpleNamespace(
        build=Mock(side_effect=lambda **_: object()), build_for_cudagraph_capture=Mock(side_effect=lambda _: object())
    )
    if has_rebind:
        builder.rebind_layerwise_prefill_metadata = Mock(return_value=None)
    group = SimpleNamespace(layer_names=["a", "b"], get_metadata_builder=lambda _: builder)
    runner = SimpleNamespace(
        attn_groups=[[group]],
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False))
        ),
    )
    ns = dict(
        self=runner,
        attn_metadata={},
        cascade_attn_prefix_lens=None,
        use_spec_decode=False,
        for_cudagraph_capture=capture,
        GDNAttentionMetadataBuilder=type("GDN", (), {}),
    )
    execute([helper], ns)
    template = None
    for _ in range(2):
        result = ns["_build_attn_group_metadata"](0, 0, object(), shared_prefill_metadata=template)
        assert result is not template
        assert ns["attn_metadata"]["a"] is ns["attn_metadata"]["b"] is result
        template = result
    assert builder.build.call_count == (0 if capture else 2)
    assert builder.build_for_cudagraph_capture.call_count == (2 if capture else 0)
    if has_rebind:
        assert builder.rebind_layerwise_prefill_metadata.call_count == (0 if capture else 1)
