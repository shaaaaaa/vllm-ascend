#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend import utils
from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear


def test_forward_keeps_router_compute_and_logits_fp32() -> None:
    gate = AscendGateLinear(
        input_size=16,
        output_size=4,
        bias=False,
        prefix="test.gate",
    )
    gate.set_out_dtype(torch.bfloat16)
    hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
    compute_input_dtypes = []
    replicated_forward = ReplicatedLinear.forward

    def record_compute_input(
        layer: ReplicatedLinear, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.nn.Parameter | None]:
        compute_input_dtypes.append(x.dtype)
        return replicated_forward(layer, x)

    with mock.patch.object(ReplicatedLinear, "forward", new=record_compute_input):
        output, output_bias = gate(hidden_states)

    assert hidden_states.dtype == torch.bfloat16
    assert compute_input_dtypes == [torch.float32]
    assert gate.weight.dtype == torch.float32
    assert output.dtype == torch.float32
    assert output_bias is None


def test_forward_has_no_logging_side_effects() -> None:
    gate = AscendGateLinear(input_size=16, output_size=4, bias=False)
    hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)

    with mock.patch("vllm_ascend.ops.fused_moe.gate_linear.logger") as logger:
        gate(hidden_states)

    logger.info_once.assert_not_called()


def test_weight_loader_keeps_gate_weight_fp32() -> None:
    gate = AscendGateLinear(
        input_size=16,
        output_size=4,
        bias=False,
        prefix="test.gate",
    )
    loaded_weight = torch.randn_like(gate.weight, dtype=torch.bfloat16)

    gate.weight_loader(gate.weight, loaded_weight)

    assert gate.weight.dtype == torch.float32
    assert torch.equal(gate.weight, loaded_weight.float())


def test_gate_linear_oot_registration_instantiates_ascend_gate() -> None:
    with mock.patch.dict(op_registry_oot, {}, clear=True):
        CustomOp.register_oot(
            _decorated_op_cls=AscendGateLinear,
            name="GateLinear",
        )

        gate = GateLinear(input_size=16, output_size=4, bias=False)

    assert type(gate) is AscendGateLinear


@pytest.mark.parametrize(
    ("model_type", "moe_router_dtype", "expected_registered"),
    [
        ("glm_moe_dsa", None, True),
        ("glm_moe_dsa", "bfloat16", True),
        ("other_moe", "float32", True),
        ("other_moe", None, False),
        ("deepseek_v3", None, False),
    ],
)
def test_gate_linear_registration_is_model_specific(
    model_type: str,
    moe_router_dtype: str | None,
    expected_registered: bool,
) -> None:
    hf_text_config = SimpleNamespace(model_type=model_type)
    if moe_router_dtype is not None:
        hf_text_config.moe_router_dtype = moe_router_dtype
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=hf_text_config))

    previous_registered = utils._ASCEND_CUSTOMOP_IS_REIGISTERED
    previous_ops = utils.REGISTERED_ASCEND_OPS
    try:
        utils._ASCEND_CUSTOMOP_IS_REIGISTERED = False
        with (
            mock.patch("vllm.model_executor.custom_op.CustomOp.register_oot"),
            mock.patch("vllm_ascend.utils.is_310p", return_value=False),
        ):
            utils.register_ascend_customop(vllm_config)

        registered_gate = utils.REGISTERED_ASCEND_OPS.get("GateLinear")
        if expected_registered:
            assert registered_gate is AscendGateLinear
        else:
            assert registered_gate is None
    finally:
        utils._ASCEND_CUSTOMOP_IS_REIGISTERED = previous_registered
        utils.REGISTERED_ASCEND_OPS = previous_ops
