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

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear


class AscendGateLinear(GateLinear):
    """Run the MoE router linear operation entirely in FP32 on Ascend."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            out_dtype=out_dtype,
            params_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=prefix,
        )
        logger.info_once(
            "[FP32_ROUTER_CHECK] impl=%s weight=%s configured_out=%s",
            type(self).__name__,
            self.weight.dtype,
            self.out_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """Compute FP32 router logits regardless of the configured output dtype."""
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return ReplicatedLinear.forward(self, x)
