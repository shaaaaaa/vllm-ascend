# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the production MTP metadata unpadding path.

Only the metadata constructor is replaced to avoid importing NPU dependencies.
The unpadding method and the runner's decision to call it execute unchanged.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def source_class(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def execute(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, "<real-common-metadata>", "exec"), namespace)


@pytest.fixture
def metadata_class():
    cls = source_class("vllm_ascend/attention/utils.py", "AscendCommonAttentionMetadata")
    method = next(node for node in cls.body if getattr(node, "name", None) == "unpadded")

    class Metadata(SimpleNamespace):
        # The production dataclass defaults for the fields under test.
        indexer_block_table_tensor = None
        indexer_slot_mapping = None
        prompt_lens_cpu = None

    namespace = {"AscendCommonAttentionMetadata": Metadata}
    execute([method], namespace)
    Metadata.unpadded = namespace["unpadded"]
    return Metadata


def make_metadata(cls, tokens, padded_tokens, num_reqs=1, padded_reqs=1):
    # Distinct physical addresses represent the MTP latent/indexer banks.
    latent_slots = torch.arange(padded_tokens) + 1024
    indexer_slots = torch.arange(padded_tokens) + 4096
    latent_slots[tokens:] = -1
    indexer_slots[tokens:] = -1
    query_lengths = [tokens // num_reqs + (i < tokens % num_reqs) for i in range(num_reqs)]
    query_lengths += [0] * (padded_reqs - num_reqs)
    query_start_loc = torch.tensor([0, *np.cumsum(query_lengths)])
    return cls(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor([131614] * num_reqs + [0] * (padded_reqs - num_reqs)),
        seq_lens_cpu=torch.tensor([131614] * num_reqs + [0] * (padded_reqs - num_reqs)),
        num_computed_tokens_cpu=torch.tensor([131614 - tokens] * padded_reqs),
        num_reqs=padded_reqs,
        num_actual_tokens=tokens,
        max_query_len=tokens,
        max_seq_len=131614,
        decode_token_per_req=2,
        block_table_tensor=torch.full((padded_reqs, 8), 8, dtype=torch.int32),
        slot_mapping=latent_slots,
        indexer_block_table_tensor=torch.full((padded_reqs, 8), 32, dtype=torch.int32),
        indexer_slot_mapping=indexer_slots,
        prompt_lens_cpu=np.array([131614] * num_reqs + [0] * (padded_reqs - num_reqs)),
        causal=True,
        actual_seq_lengths_q=query_lengths,
        positions=torch.arange(padded_tokens) + 131614 - tokens,
        attn_state="ChunkedPrefill",
        graph_pad_size=padded_tokens,
        num_input_tokens=padded_tokens,
        prefill_context_parallel_metadata=None,
        request_ids=[f"request-{i}" for i in range(padded_reqs)],
        cold_compact_resumes=(False,) * padded_reqs,
        resident_state_indices=None,
        resident_state_generations=None,
        resident_state_indices_cpu=None,
        resident_state_generations_cpu=None,
    )


@pytest.mark.parametrize(
    "tokens,padded_tokens,padded_reqs",
    [(542, 544, 1), (7, 8, 1), (16, 16, 4), (7, 8, 4), (1216, 1216, 1)],
)
def test_runner_unpadding_keeps_mtp_indexer_addresses(metadata_class, tokens, padded_tokens, padded_reqs):
    original = make_metadata(metadata_class, tokens, padded_tokens, padded_reqs=padded_reqs)
    runner = source_class("vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner")
    build = next(node for node in runner.body if getattr(node, "name", None) == "_build_attention_metadata")
    unpadding = next(
        node
        for node in build.body
        if isinstance(node, ast.If) and "spec_decode_common_attn_metadata.unpadded(" in ast.unparse(node)
    )
    namespace = dict(
        spec_decode_common_attn_metadata=original,
        num_tokens=tokens,
        num_tokens_padded=padded_tokens,
        num_reqs=1,
        num_reqs_padded=padded_reqs,
    )
    execute([unpadding], namespace)
    result = namespace["spec_decode_common_attn_metadata"]

    # Preserve the original physical tensor views, including padding, just as
    # the latent path does. The attention builder owns subsequent slicing.
    assert result.indexer_block_table_tensor is original.indexer_block_table_tensor
    assert result.indexer_slot_mapping is original.indexer_slot_mapping
    assert result.block_table_tensor is original.block_table_tensor
    assert result.slot_mapping is original.slot_mapping
    assert torch.all(result.indexer_slot_mapping[:tokens] != result.slot_mapping[:tokens])
    assert result.num_input_tokens == padded_tokens
    assert result.num_reqs == 1
    assert result.num_actual_tokens == tokens
    assert original.num_reqs == padded_reqs
    assert original.graph_pad_size == padded_tokens
    assert result.prompt_lens_cpu.tolist() == [131614]
    assert np.shares_memory(result.prompt_lens_cpu, original.prompt_lens_cpu)


@pytest.mark.parametrize("prompt_type", [list, np.array, torch.tensor, None])
def test_unpadding_slices_prompt_boundaries_by_request(metadata_class, prompt_type):
    original = make_metadata(metadata_class, 7, 8, num_reqs=2, padded_reqs=3)
    original.prompt_lens_cpu = None if prompt_type is None else prompt_type([131614, 8192, 0])
    result = original.unpadded(7, 2)

    if prompt_type is None:
        assert result.prompt_lens_cpu is None
    else:
        assert result.prompt_lens_cpu is not None
        assert list(result.prompt_lens_cpu) == [131614, 8192]
        assert len(original.prompt_lens_cpu) == 3
        if prompt_type is torch.tensor:
            assert result.prompt_lens_cpu.data_ptr() == original.prompt_lens_cpu.data_ptr()
        elif prompt_type is np.array:
            assert np.shares_memory(result.prompt_lens_cpu, original.prompt_lens_cpu)


def test_unpadding_without_separate_indexer_keeps_none(metadata_class):
    original = make_metadata(metadata_class, 7, 8)
    original.indexer_block_table_tensor = None
    original.indexer_slot_mapping = None
    original.prompt_lens_cpu = None
    result = original.unpadded(7, 1)

    assert result.indexer_block_table_tensor is None
    assert result.indexer_slot_mapping is None
    assert result.prompt_lens_cpu is None
    assert result.slot_mapping is original.slot_mapping
