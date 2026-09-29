# SPDX-License-Identifier: Apache-2.0
"""CPU regression using the actual vLLM parser and Ascend bookkeeping methods.

Load source ASTs to avoid importing NPU kernels. No device sampling is mocked
here: these tests start with sampled tokens and verify scheduler-facing output.
"""

import __future__

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_class(path, name, namespace, methods=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    if methods is not None:
        cls.bases = []
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        assert {node.name for node in cls.body} == set(methods)
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace[name]


@pytest.fixture(scope="module")
def runtime():
    spec = importlib.util.find_spec("vllm")
    vllm_path = Path(spec.origin).parent if spec and spec.origin else ROOT.parent / "vllm" / "vllm"
    assert (vllm_path / "v1/outputs.py").is_file(), "Install vLLM or use a sibling vllm checkout"
    namespace = dict(np=np, torch=torch, NamedTuple=NamedTuple, PLACEHOLDER_TOKEN_ID=-1)
    for name in ("LogprobsLists", "LogprobsTensors"):
        load_class(vllm_path / "v1/outputs.py", name, namespace)
    namespace["AscendRejectionSampler"] = load_class(
        vllm_path / "v1/sample/rejection_sampler.py", "RejectionSampler", namespace, ("parse_output",)
    )
    runner = load_class(
        ROOT / "vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", namespace, ("_bookkeeping_sync",)
    )
    return runner._bookkeeping_sync, namespace["LogprobsTensors"]


def run_bookkeeping(runtime, tokens, *, with_logprobs, discard=(), asynchronous=False):
    bookkeeping, logprobs_class = runtime
    sampled = torch.tensor(tokens, dtype=torch.int32)
    requests, width = sampled.shape
    ids = [f"r{i}" for i in range(requests)]
    batch = NS(
        req_ids=ids,
        req_id_to_index={name: i for i, name in enumerate(ids)},
        vocab_size=100,
        generators={},
        num_tokens_no_spec=np.full(requests, 2),
        num_tokens=np.full(requests, 2),
        token_ids_cpu=np.zeros((requests, 16), dtype=np.int32),
        is_token_ids=np.zeros((requests, 16), dtype=bool),
    )
    state = NS(
        input_batch=batch,
        requests={name: NS(output_token_ids=[]) for name in ids},
        discard_request_indices=NS(np=np.array(discard, dtype=np.int64)),
        num_discarded_requests=len(discard),
        use_async_scheduling=asynchronous,
        num_spec_tokens=width - 1,
        max_model_len=16,
        _to_list=lambda value: value.tolist(),
        _get_prompt_logprobs_dict=lambda *args: {},
    )
    rows = requests * width
    tensors = logprobs_class(
        torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2),
        -torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2),
        torch.arange(rows, dtype=torch.int32),
    )
    # Async bookkeeping must leave device logprobs for AsyncGPUModelRunnerOutput.
    payload = object() if asynchronous and with_logprobs else tensors
    output = bookkeeping(
        state,
        NS(num_scheduled_tokens={name: 1 for name in ids}),
        NS(sampled_token_ids=sampled, logprobs_tensors=payload if with_logprobs else None),
        None,
        torch.zeros(requests, 1),
        requests,
        None,
    )
    return output, state, tensors


@pytest.mark.parametrize("with_logprobs", [False, True])
@pytest.mark.parametrize("discard", [(), (2,)])
def test_mtp_logprobs_keep_valid_rows_and_scalar_request_offsets(runtime, with_logprobs, discard):
    output, state, tensors = run_bookkeeping(
        runtime,
        [[10, 11, 12], [20, -1, -1], [30, 31, -1], [40, 41, 100]],
        with_logprobs=with_logprobs,
        discard=discard,
    )
    logprobs, token_ids = output[:2]
    expected_tokens = [[10, 11, 12], [20], [] if discard else [30, 31], [40, 41]]
    assert token_ids == expected_tokens
    for index, tokens in enumerate(expected_tokens):
        assert state.requests[f"r{index}"].output_token_ids == tokens
    if not with_logprobs:
        assert logprobs is None
        return
    expected_rows = [[0, 1, 2], [3], [6, 7], [9, 10]]
    for index, tokens in enumerate(expected_tokens):
        # The scheduler can truncate accepted tokens at max_tokens/stop.
        for count in {0, 1, len(tokens)} if tokens else {0}:
            sliced = logprobs.slice_request(index, count)
            rows = expected_rows[index][:count]
            np.testing.assert_array_equal(sliced.logprob_token_ids, tensors.logprob_token_ids[rows].numpy())
            np.testing.assert_array_equal(sliced.logprobs, tensors.logprobs[rows].numpy())
            np.testing.assert_array_equal(sliced.sampled_token_ranks, tensors.selected_token_ranks[rows].numpy())
    assert logprobs.cu_num_generated_tokens == [0, 3, 4, 6, 8]


@pytest.mark.parametrize("with_logprobs", [False, True])
def test_ordinary_decode_keeps_one_logprob_row_per_request(runtime, with_logprobs):
    output, _, tensors = run_bookkeeping(runtime, [[10], [20], [30]], with_logprobs=with_logprobs, discard=(1,))
    logprobs, tokens = output[:2]
    assert tokens == [[10], [], [30]]
    if with_logprobs:
        np.testing.assert_array_equal(logprobs.slice_request(2, 1).logprobs, tensors.logprobs[2:3].numpy())
        assert logprobs.cu_num_generated_tokens is None
    else:
        assert logprobs is None


@pytest.mark.parametrize("tokens", [[[10], [20]], [[10, 11], [20, -1]]])
def test_async_bookkeeping_does_not_read_logprob_tensors(runtime, tokens):
    output, state, _ = run_bookkeeping(runtime, tokens, with_logprobs=True, discard=(1,), asynchronous=True)
    assert output[0] is None
    assert output[1] == []
    assert output[-1] == [1]
    assert state.input_batch.prev_req_id_to_index == {"r0": 0}
