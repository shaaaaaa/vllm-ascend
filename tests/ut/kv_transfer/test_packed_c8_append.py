# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.packed_c8_sfa import append_packed_c8_main_kv


@pytest.fixture
def case(monkeypatch):
    cache = torch.full((34, 128, 1, 656), 97, dtype=torch.int8)
    projected = torch.arange(2 * 576).reshape(2, 1, 1, 576).to(torch.bfloat16)
    gamma = torch.ones(512, dtype=torch.bfloat16)
    cos = torch.ones(2, 1, 1, 64, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    slots = torch.tensor([32 * 128 + 127, 33 * 128], dtype=torch.int32)
    normalized = torch.full((2, 1, 1, 512), 3, dtype=torch.bfloat16)
    rotated = (torch.arange(128).reshape(2, 1, 1, 64) - 50).to(torch.bfloat16)
    quantized = (torch.arange(1024).reshape(2, 1, 512) % 211 - 105).to(torch.int8)
    scales = torch.tensor([0.25, 0.5, 2, 4, 8, 16, 32, 64], dtype=torch.float32).view(2, 1, 4)

    def scatter(destination, indices, updates):
        destination.index_copy_(0, indices.flatten().long(), updates)

    backend = SimpleNamespace(
        npu_rms_norm=Mock(return_value=(normalized, None)),
        npu_interleave_rope=Mock(return_value=rotated),
        npu_dynamic_block_quant=Mock(return_value=(quantized, scales)),
        npu_scatter_nd_update_=Mock(side_effect=scatter),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    return SimpleNamespace(
        backend=backend,
        inputs=dict(packed_main=cache, projected_kv=projected, gamma=gamma, cos=cos, sin=sin, slots=slots),
        normalized=normalized,
        rotated=rotated,
        quantized=quantized,
        scales=scales,
    )


@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16], ids=["fp32-scale", "bf16-scale"])
@pytest.mark.parametrize("strided_slots", [False, True], ids=["int32-slots", "strided-int64-slots"])
def test_append_packs_bytes_and_updates_only_the_live_physical_slots(case, scale_dtype, strided_slots):
    expected_slots = [32 * 128 + 127, 33 * 128]
    slot_dtype = torch.int32
    if strided_slots:
        expected_slots = [33 * 128 + 3, 32 * 128 + 127]
        slot_dtype = torch.int64
        case.inputs["slots"] = torch.tensor([expected_slots[0], -1, expected_slots[1], -1], dtype=slot_dtype)[::2]
        assert not case.inputs["slots"].is_contiguous()
    case.backend.npu_dynamic_block_quant.return_value = (case.quantized, case.scales.to(scale_dtype))

    epsilon = 1e-5
    assert append_packed_c8_main_kv(**case.inputs, epsilon=epsilon) is None
    backend = case.backend
    rms_input, weight = backend.npu_rms_norm.call_args.args
    assert backend.npu_rms_norm.call_args.kwargs == {"epsilon": epsilon}
    torch.testing.assert_close(rms_input, case.inputs["projected_kv"][..., :512], rtol=0, atol=0)
    assert weight is case.inputs["gamma"]
    rope_input, cos, sin = backend.npu_interleave_rope.call_args.args
    torch.testing.assert_close(rope_input, case.inputs["projected_kv"][..., 512:], rtol=0, atol=0)
    assert cos is case.inputs["cos"] and sin is case.inputs["sin"]
    (quant_input,) = backend.npu_dynamic_block_quant.call_args.args
    torch.testing.assert_close(quant_input, case.normalized.view(2, 1, 512), rtol=0, atol=0)
    assert backend.npu_dynamic_block_quant.call_args.kwargs == {
        "dst_type": torch.int8,
        "row_block_size": 1,
        "col_block_size": 128,
    }

    cache = case.inputs["packed_main"]
    rows, indices, packed = backend.npu_scatter_nd_update_.call_args.args
    assert rows.data_ptr() == cache.data_ptr()
    assert indices.dtype == slot_dtype
    assert indices.is_contiguous()
    assert indices.tolist() == [[slot] for slot in expected_slots]
    assert packed.shape == (2, 656) and packed.dtype == torch.int8
    physical_slots = torch.tensor(expected_slots)
    written = rows.index_select(0, physical_slots)
    torch.testing.assert_close(written[:, :512], case.quantized.view(2, 512), rtol=0, atol=0)
    torch.testing.assert_close(
        written[:, 512:640].contiguous().view(torch.bfloat16), case.rotated.view(2, 64), rtol=0, atol=0
    )
    torch.testing.assert_close(
        written[:, 640:].contiguous().view(torch.float32), case.scales.view(2, 4), rtol=0, atol=0, check_dtype=True
    )
    untouched = torch.ones(rows.shape[0], dtype=torch.bool)
    untouched[physical_slots] = False
    assert torch.all(rows[untouched] == 97)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("slots", torch.tensor([4096, -1]), "without padding"),
        ("slots", torch.tensor([4096, 34 * 128]), "without padding"),
        ("slots", torch.tensor([4096, 4096]), "unique and in range"),
        ("slots", torch.tensor([4096, 4097], dtype=torch.float32), "integer slot map"),
        ("slots", torch.tensor([4096]), "per-token RoPE tables"),
        ("slots", torch.empty(2, device="meta", dtype=torch.int32), "same device"),
        ("packed_main", torch.empty(34, 128, 1, 655, dtype=torch.int8), "contiguous INT8"),
        ("sin", torch.empty(2, 1, 1, 64, dtype=torch.float16), "BF16"),
    ],
)
def test_invalid_input_fails_before_backend_ops(case, field, invalid, message):
    inputs = {**case.inputs, field: invalid}
    with pytest.raises(ValueError, match=message):
        append_packed_c8_main_kv(**inputs, epsilon=1e-5)
    for op in vars(case.backend).values():
        op.assert_not_called()
    assert torch.all(case.inputs["packed_main"] == 97)


def test_empty_append_performs_no_backend_work(case):
    inputs = dict(case.inputs)
    for field in ("projected_kv", "cos", "sin", "slots"):
        inputs[field] = inputs[field][:0]
    append_packed_c8_main_kv(**inputs, epsilon=1e-5)
    for op in vars(case.backend).values():
        op.assert_not_called()


def test_incompatible_quantizer_result_does_not_scatter(case):
    case.backend.npu_dynamic_block_quant.return_value = (case.quantized, case.scales[..., :3])
    with pytest.raises(ValueError, match="incompatible RoPE or quantizer output"):
        append_packed_c8_main_kv(**case.inputs, epsilon=1e-5)
    case.backend.npu_scatter_nd_update_.assert_not_called()
    assert torch.all(case.inputs["packed_main"] == 97)
