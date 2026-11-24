# SPDX-License-Identifier: Apache-2.0
"""Validate startup signatures against real kernel definitions, without an NPU."""

import ast
import inspect
import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from sfa_test_support import ROOT, extract, load_module

OPS = ROOT / "vllm_ascend/ops/triton"


class Kernel:
    def __init__(self, path, node, calls):
        self.name, self.calls = node.name, calls
        self.signature = inspect.signature(extract(path, node.name, {}))

    def warmup(self, *args, grid, **kwargs):
        assert grid == (1,)
        assert not any(isinstance(value, torch.dtype) for value in kwargs.values())
        kwargs.pop("multibuffer", None)
        if self.name == "_rope_query_packet":
            assert kwargs.pop("enable_fp_fusion") is False
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        # Compile-only startup must never receive live tensor pointers.
        assert not any(isinstance(value, torch.Tensor) for value in bound.arguments.values())
        self.calls.append((self.name, dict(bound.arguments)))


@pytest.fixture
def warmup(monkeypatch):
    calls = []
    triton = NS(next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", NS(HAS_TRITON=True, triton=triton))
    for name in (
        "sfa_dcp",
        "sfa_query",
        "rope",
        "bincount",
        "penalty",
        "reject_sample",
        "spec_decode.utils",
        "spec_decode.indexer_c8_metadata",
    ):
        path = OPS / (name.replace(".", "/") + ".py")
        module = ModuleType("vllm_ascend.ops.triton." + name)
        for node in ast.parse(path.read_text(encoding="utf8")).body:
            if isinstance(node, ast.FunctionDef) and any("triton.jit" in ast.unparse(d) for d in node.decorator_list):
                setattr(module, node.name, Kernel(path, node, calls))
        if name == "reject_sample":
            module.cal_grid_and_block_size = extract(
                path,
                "cal_grid_and_block_size",
                {
                    "get_vectorcore_num": lambda: 4,
                    "triton": NS(next_power_of_2=triton.next_power_of_2, cdiv=lambda n, d: (n + d - 1) // d),
                },
            )
        if name == "spec_decode.utils":
            module.PREPARE_INPUTS_BLOCK_SIZE = 4
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.rejection_sampler", NS(MAX_SPEC_LEN=128))
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ops.triton.triton_utils", NS(init_device_properties_triton=lambda: None)
    )
    module = load_module(OPS / "decode_warmup.py", "decode_warmup_test", monkeypatch)
    return module, calls


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dcp_warms_dense_mixed_exchange_and_draft_variants(warmup, world, dtype):
    module, calls = warmup
    module.warmup_dcp(2048, 64 // world, dtype, world, world - 1, 128, draft=True)
    prepared = [args for name, args in calls if name == "_prepare_topk"]
    assert {(a["positions"], a["row_requests"], a["local"]) for a in prepared} == {
        (position, requests, local)
        for position in (torch.int32, torch.int64)
        for local, requests in ((True, None), (False, torch.int32))
    }
    assert all(a["world"] == world and a["rank"] == world - 1 for a in prepared)
    drafts = [a for name, a in calls if name == "prepare_dcp_mtp_draft_kernel"]
    assert len(drafts) == 8 and {a["rejected"] for a in drafts} == {None, torch.int32}
    assert {name for name, _ in calls} == {
        "_mask_topk",
        "_prepare_topk",
        "_unpack_queries",
        "_pack",
        "_combine",
        "prepare_dcp_mtp_draft_kernel",
    }


def test_query_and_mapping_cover_real_layouts(warmup):
    module, calls = warmup
    module.warmup_query(8, 512, 64, 192)
    assert len(calls) == 8
    assert {a["qs0"] for _, a in calls} == {512, 8 * 512}
    assert {(a["qo1"], a["ro1"]) for _, a in calls} == {(512, 64), (576, 576)}
    module.warmup_indexer_metadata(18, 36)
    mappings = [a for name, a in calls if name == "_map_indexer_metadata"]
    assert {(a["TABLE_STRIDE"], a["slots"]) for a in mappings} == {
        (stride, dtype) for stride in (18, 36) for dtype in (torch.int32, torch.int64)
    }
    module.warmup_indexer_rope(64, 128, 64, False, torch.bfloat16)
    assert calls[-1][1]["USE_COS_SIN"] is False


@pytest.mark.parametrize("spec_tokens", [0, 1, 3])
def test_sampling_warms_small_and_large_tiles_and_optional_inputs(warmup, spec_tokens):
    module, calls = warmup
    module.warmup_sampling(257, 17, spec_tokens)
    names = {name for name, _ in calls}
    assert {"apply_all_penalties_kernel", "token_bin_counts_and_mask_kernel"} <= names
    if not spec_tokens:
        assert len(calls) == 3
        return
    greedy = [a for name, a in calls if name == "rejection_greedy_sample_triton"]
    assert {(a["BLOCK_SIZE"], a["is_greedy_ptr"]) for a in greedy} == {
        (tile, ptr) for tile in (1, 2, 4, 8) for ptr in (None, torch.bool)
    }
    assert "sample_recovered_tokens_kernel" in names
    module.warmup_spec_decode(257, spec_tokens == 1)
    assert calls[-1][0] == ("prepare_next_mtp_tokens_kernel" if spec_tokens == 1 else "prepare_inputs_padded_kernel")


@pytest.mark.parametrize(
    "dcp,sparse,pooling,spec",
    [
        (False, False, False, 0),
        (False, True, False, 1),
        (True, True, False, 1),
        (True, True, False, 0),
        (False, False, True, 0),
    ],
)
@pytest.mark.parametrize("packet_mode", ["direct", "fused", "fused_contiguous"])
def test_common_startup_selects_used_paths_without_optimization_flags(
    warmup, monkeypatch, dcp, sparse, pooling, spec, packet_mode
):
    module, calls = warmup

    class Impl:
        local_num_heads, kv_lora_rank, qk_rope_head_dim, qk_head_dim = 16, 512, 64, 192
        has_indexer, n_head, head_dim, is_rope_neox_style = True, 64, 128, False
        _dcp_packet_io = packet_mode

    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.sfa_v1", NS(AscendSFAImpl=Impl))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.attention_layer_base", NS(AttentionLayerBase=object))
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        NS(
            get_layers_from_vllm_config=lambda *args: {
                "target": NS(impl=Impl()),
                "draft": NS(impl=Impl()),
            }
        ),
    )
    table = torch.empty((2, 18), dtype=torch.int32)
    runner = NS(
        speculative_config=NS(num_speculative_tokens=spec),
        is_pooling_model=pooling,
        use_sparse=sparse,
        input_batch=NS(vocab_size=257, block_table=[None, NS(get_device_tensor=lambda: table)]),
        max_num_reqs=5,
        drafter=NS(method="mtp", prepare_inputs_padded=lambda: None),
        vllm_config=object(),
        ascend_config=NS(indexer_hbm_block_map=(0,)),
        dtype=torch.bfloat16,
        dsa_decoder_dcp=dcp,
        dcp_size=4,
        dcp_rank=3,
        block_size=128,
        dsa_index_topk=2048,
        model_config=NS(get_num_attention_heads=lambda config: 16),
        parallel_config=object(),
    )
    module.warmup_decode_kernels(runner)
    names = [name for name, _ in calls]
    assert ("_pack" in names) == dcp
    writers = [args["CONTIGUOUS"] for name, args in calls if name == "_rope_query_packet"]
    assert writers == ([packet_mode == "fused_contiguous"] if dcp and sparse and packet_mode != "direct" else [])
    assert ("_copy_query_bridge" in names) == sparse
    assert ("apply_all_penalties_kernel" in names) != pooling
    assert ("prepare_next_mtp_tokens_kernel" in names) == bool(spec)
    assert ("prepare_dcp_mtp_draft_kernel" in names) == (dcp and bool(spec))
    if sparse:
        assert names.count("_copy_query_bridge") == 8  # Identical target/draft geometry is compiled once.
        assert names.count("_triton_rope_siso") == 2
    module.HAS_TRITON = False
    count = len(calls)
    module.warmup_decode_kernels(runner)
    assert len(calls) == count


def test_batch_and_history_extents_are_not_specialized():
    parameters = {
        "sfa_dcp.py": {
            "_prepare_topk": {"rows", "actual_rows"},
            "_unpack_queries": {"rows"},
            "_pack": {"rows"},
            "_combine": {"local_rows"},
        },
        "sfa_query.py": {"_copy_query_bridge": {"items", "qs1"}},
        "spec_decode/indexer_c8_metadata.py": {"_map_indexer_metadata": {"TABLE_SIZE", "SLOT_SIZE"}},
        "reject_sample.py": {
            "rejection_greedy_sample_spec_len_1_triton": {"vec_len"},
            "rejection_random_sample_kernel": {"vec_len"},
        },
        "bincount.py": {"token_bin_counts_and_mask_kernel": {"batch_size", "seq_len", "tokens_batch_stride"}},
        "penalty.py": {"apply_all_penalties_kernel": {"num_seqs"}},
        "rope.py": {"_triton_rope_siso": {"num_tokens"}},
    }
    for path, functions in parameters.items():
        tree = ast.parse((OPS / path).read_text(encoding="utf8"))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name not in functions:
                continue
            excluded = ast.literal_eval(
                next(k.value for k in node.decorator_list[0].keywords if k.arg == "do_not_specialize")
            )
            assert functions[node.name] <= set(excluded)
            assert all(a.annotation is None for a in node.args.args if a.arg in functions[node.name])
