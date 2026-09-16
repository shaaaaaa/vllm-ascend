# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token placement, ownership and padding contracts for fused async MTP."""

import pytest
import torch
from sfa_test_support import ROOT, AsyncMTPTokenKernel, extract, load_module
from test_sfa_async_mtp import setup  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode


@pytest.mark.parametrize("n", [1, 3, 16, 31, 32, 33, 65])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fused_tokens_match_copies_and_gather(n, dtype):
    kernel = AsyncMTPTokenKernel()
    # Offset views exercise pointer base handling; inactive lanes retain padding.
    sampled = torch.arange(n + 1, dtype=torch.int32)[1:].view(n, 1)
    draft = (torch.arange(n + 1, dtype=dtype) - 3)[1:].view(n, 1)
    if dtype == torch.int64:
        draft[0] = 2**32 + 17  # Preserve the old int32 copy conversion.
    inputs = torch.full((2 * n + 7,), -77, dtype=torch.int32)
    held = []
    for _ in range(3):
        output = torch.empty(n, dtype=torch.int32)
        kernel[((n + 31) // 32,)](sampled, draft, inputs, output, n, BLOCK=32)
        expected = torch.stack((sampled[:, 0], draft[:, 0].int()), dim=1).flatten()
        torch.testing.assert_close(inputs[: 2 * n], expected)
        assert torch.all(inputs[2 * n :] == -77)
        held.append((output, draft[:, 0].int().clone()))
        sampled.add_(5)
        draft.add_(7)
    for output, expected in held:
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("field", ["sampled", "draft"])
def test_strided_tokens_use_existing_fallback(request, field):
    runner, scheduled, _, events, _ = request.getfixturevalue("setup")
    tokens = torch.arange(6, dtype=torch.int32)[::2].view(3, 1)
    if field == "sampled":
        runner.input_batch.prev_sampled_token_ids = tokens
    else:
        runner._draft_token_ids = tokens
    assert not runner._eligible(scheduled)
    runner._update_states(scheduled)
    runner._prepare_inputs(scheduled, None)
    assert events == ["sync", "state_update", "normal_prepare"]


def test_npu_fused_tokens_preserve_queued_consumers(monkeypatch):
    pytest.importorskip("torch_npu")
    pytest.importorskip("triton")
    if not torch.npu.is_available():
        pytest.skip("Requires NPU")
    module = load_module(ROOT / "vllm_ascend/ops/triton/spec_decode/async_mtp.py", "mtp_tokens_npu", monkeypatch)
    held = []
    for n in (1, 3, 32, 33, 65):
        for dtype in (torch.int32, torch.int64):
            sampled = torch.arange(n, dtype=torch.int32, device="npu").view(n, 1)
            draft = torch.arange(n, dtype=dtype, device="npu").view(n, 1)
            inputs = torch.full((2 * n + 7,), -77, dtype=torch.int32, device="npu")
            for _ in range(3):
                output = torch.empty(n, dtype=torch.int32, device="npu")
                expected = torch.stack((sampled[:, 0], draft[:, 0].int()), dim=1).flatten()
                module.prepare_async_mtp_tokens_kernel[((n + 31) // 32,)](sampled, draft, inputs, output, n, BLOCK=32)
                held.append((inputs.clone(), output, expected))
                sampled.add_(5)
                draft.add_(7)
    torch.npu.synchronize()
    for inputs, output, expected in held:
        torch.testing.assert_close(inputs[: expected.numel()], expected)
        assert torch.all(inputs[expected.numel() :] == -77)
        torch.testing.assert_close(output, expected[1::2])


def test_padded_logits_cache_tracks_layout_and_retains_old_outputs(request, monkeypatch):
    runner, _, _, _, ns = request.getfixturevalue("setup")
    ns["lmhead_tp_enable"] = lambda: True
    runner.max_num_reqs, runner.uniform_decode_query_len = 4, 2
    runner._async_pending = object()
    pad = torch.nn.functional.pad
    calls = []

    def recorded(*args, **kwargs):
        calls.append(1)
        return pad(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "pad", recorded)
    held = []
    for n in (3, 3, 1, 1, 3):
        runner.input_batch.num_reqs = n
        runner.input_batch.prev_sampled_token_ids = torch.ones(n, 1, dtype=torch.int32)
        runner._draft_token_ids = torch.ones(n, 1, dtype=torch.int32)
        logits, _, _ = runner._prepare_inputs(None, None)
        held.append((logits, list(range(2 * n)) + [0] * (8 - 2 * n)))
    assert len(calls) == 3
    assert held[0][0] is held[1][0] and held[2][0] is held[3][0]
    for logits, expected in held:
        assert logits.tolist() == expected


@pytest.mark.parametrize("lmhead_tp", [False, True])
def test_warm_preparation_launches_once_without_tensor_copies_or_padding(request, lmhead_tp):
    runner, _, _, _, ns = request.getfixturevalue("setup")
    ns["lmhead_tp_enable"] = lambda: lmhead_tp
    runner.max_num_reqs, runner.uniform_decode_query_len = 4, 2
    runner._async_pending = object()
    launches = []

    class LaunchOnly:
        # Kernel math/lifetimes are tested separately. Count dispatch outside it.
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append(grid)

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operations.append(str(func))
            return func(*args, **(kwargs or {}))

    ns["prepare_async_mtp_tokens_kernel"] = LaunchOnly()
    runner._prepare_inputs(None, None)  # Populate the optional index cache.
    launches.clear()
    operations = []
    with Trace():
        runner._prepare_inputs(None, None)
    assert launches == [(1,)]
    assert operations.count("aten.new_empty.default") == 1  # Fresh output ownership.
    assert not set(operations).intersection(
        {"aten.copy_.default", "aten.index.Tensor", "aten.constant_pad_nd.default", "aten._local_scalar_dense.default"}
    )


def test_fused_draft_output_survives_route_fallback(request):
    runner, scheduled, _, events, ns = request.getfixturevalue("setup")
    # Use the real constructor helper so the returned object holds fused output.
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    runner._fixed_mtp_metadata = extract(path, "_fixed_mtp_metadata_arrays", {"torch": torch})(3, "cpu")
    metadata = extract(
        path,
        "_fixed_spec_decode_metadata",
        {"np": ns["np"], "SpecDecodeMetadata": lambda **kwargs: type("Metadata", (), kwargs)()},
    )
    runner._fixed_spec_decode_metadata = lambda *args: metadata(runner, *args)
    runner.input_batch.prev_sampled_token_ids.fill_(13)
    runner._draft_token_ids.fill_(27)
    runner._update_states(scheduled)
    _, spec, _ = runner._prepare_inputs(scheduled, None)
    assert events == ["token_kernel"]
    assert spec.draft_token_ids.tolist() == [27, 27, 27]
    runner._apply_staged_sfa_route(None)
    assert events == ["token_kernel", "sync", "state_update", "normal_prepare"]
    runner.input_ids.gpu.fill_(99)
    runner._draft_token_ids.fill_(101)
    assert spec.draft_token_ids.tolist() == [27, 27, 27]
