#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
import torch
import vllm.model_executor.layers.utils
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.utils.torch_utils import direct_register_custom_op


def unquantized_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def unquantized_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    output_shape = (x.shape[0], weight.shape[0])
    return torch.empty(output_shape, dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="unquantized_gemm",
    op_func=unquantized_gemm,
    fake_impl=unquantized_gemm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def sfa_router_gemm(x: torch.Tensor, weight: torch.Tensor, attention_name: str) -> torch.Tensor:
    # Resolve per-key edges inside the opaque GEMM, at ACL capture. Eager,
    # profiling warmup, draft graphs and layers without an outgoing edge
    # use exactly the same FP32 linear calculation.
    if is_forward_context_available():
        edges = getattr(get_forward_context(), "sfa_router_copy_edges", None)
        edge = edges.get(attention_name) if edges is not None else None
        if edge is not None and x.is_contiguous() and weight.is_contiguous():
            with edge.cube_window("router"):
                return unquantized_gemm(x, weight)
    return unquantized_gemm(x, weight)


def sfa_router_gemm_fake(x: torch.Tensor, weight: torch.Tensor, attention_name: str) -> torch.Tensor:
    return unquantized_gemm_fake(x, weight)


direct_register_custom_op(
    op_name="sfa_router_gemm",
    op_func=sfa_router_gemm,
    fake_impl=sfa_router_gemm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.device.type == "npu":
        attention = getattr(layer, "sfa_router_attention_name", None)
        if attention and bias is None and x.ndim == 2 and x.dtype == weight.dtype == torch.float32:
            return torch.ops.vllm.sfa_router_gemm(x, weight, attention)
        return torch.ops.vllm.unquantized_gemm(x, weight, bias)
    else:
        return torch.nn.functional.linear(x, weight, bias)


vllm.model_executor.layers.utils.default_unquantized_gemm = default_unquantized_gemm
