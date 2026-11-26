# SPDX-License-Identifier: Apache-2.0
"""Capture-time admission for the CANN 8.5.1 / 910B3 decode kernels."""

from contextlib import ExitStack, contextmanager, nullcontext

import torch
import torch_npu
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context, is_forward_context_available


def current_gate():
    if not is_forward_context_available():
        return None
    gate = getattr(get_forward_context(), "sfa_copy_gate", None)
    if (
        gate is not None
        and gate.pending
        and gate.supports_decoder_windows
        and not gate.open
        and torch.npu.current_stream() == gate.main_stream
    ):
        return gate
    return None


def tp_fits(gate, byte_count):
    # The intra-server TP4 HCCL mesh uses at most eight AIVs below 1 MiB
    # for AllReduce (including deterministic mode) and at most five for AllGather.
    return gate is not None and gate.local_tp4 and gate.free_aiv >= 8 and 0 < byte_count < 1024 * 1024


def projection_name(prefix):
    for suffix, name in (
        (".self_attn.fused_qkv_a_proj", "qkv"),
        (".self_attn.q_b_proj", "q_b"),
        (".self_attn.o_proj", "o_proj"),
        (".mlp.shared_experts.gate_up_proj", "shared_up"),
        (".mlp.shared_experts.down_proj", "shared_down"),
        (".mlp.gate_up_proj", "dense_up"),
        (".mlp.down_proj", "dense_down"),
    ):
        if prefix.endswith(suffix):
            return name
    return None


def tagged_projection(layer, x, bias):
    return (
        getattr(layer, "sfa_copy_projection", None) is not None
        and bias is None
        and x.dtype == torch.bfloat16
        and x.ndim == 2
    )


def w8a8_projection(layer, x, *, reduce=False):
    token_scale = None
    if layer.sfa_copy_projection == "dense_up" and torch.compiler.is_compiling():
        # Keep DynamicQuant visible to the compiler's AddRmsNormDynamicQuant fusion.
        x, token_scale = torch_npu.npu_dynamic_quant(x)
    return torch.ops.vllm.sfa_decode_w8a8(
        x, layer.weight, layer.weight_scale, token_scale, layer.sfa_copy_projection, reduce
    )


@contextmanager
def consecutive_windows(gate):
    """Keep one device grant across adjacent eligible kernels in this call."""
    with ExitStack() as stack:

        def select(name, eligible):
            if not eligible:
                stack.close()
            elif gate.open:
                gate.counts[name] += 1
            else:
                stack.enter_context(gate.window(name))

        yield select


def small_w8a8_shape(rows, width, columns):
    # Net-decode ND x NZ has baseM >= 16 and baseN >= 64. Bound the
    # number of base tiles below the 682-tile 1:2 switch for small K.
    return (
        0 < rows <= 64
        and width > 0
        and width % 128 == 0
        and columns > 0
        and columns % 64 == 0
        and (width > 1152 or ((rows + 15) // 16) * ((columns + 63) // 64) < 682)
    )


def qkv_grant_fits(gate, x, weight):
    return (
        gate is not None
        and gate.pending > 1
        and gate.free_aiv >= 20
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and torch_npu.get_npu_format(x) == 2
        and x.size(0) <= gate.free_aiv
        and x.size(1) <= 8192
        and weight.ndim == 2
        and weight.dtype == torch.int8
        and torch_npu.get_npu_format(weight) == 29
        and small_w8a8_shape(x.size(0), x.size(1), weight.size(1))
    )


def join_through_qkv(layer, x):
    # The replicated, bias-free GLM QKV-A is exactly DQ followed by GEMM.
    # If both fit, avoid reacquiring the FIFO token between this layer's
    # completion event and QKV-A while the next layers still have work.
    gate = current_gate()
    return (
        gate is not None
        and type(layer).__name__ in {"DeepSeekV2FusedQkvAProjLinear", "AscendMergedColumnParallelLinear"}
        and layer.tp_size == 1
        and not layer.gather_output
        and getattr(layer, "custom_op", None) is None
        and layer.bias is None
        and not layer._forward_hooks
        and not layer._forward_pre_hooks
        and qkv_grant_fits(gate, x, layer.weight)
    )


def swiglu_fits(rows, columns, free_aiv):
    # BF16 CANN SwiGlu key 27. Aligned column tiles step by 256 elements.
    # Both buffering variants fit at least 8192 elements per UB tile on B3
    # (192 KiB, at most 20 bytes/element). Bound every candidate tiling.
    if not (0 < rows <= 64 and 256 <= columns <= 8192 and columns % 256 == 0):
        return False
    for width in range(256, columns + 1, 256):
        height = 8192 // width
        tiles = ((rows + height - 1) // height) * ((columns + width - 1) // width)
        if tiles > free_aiv:
            return False
    return True


def sfa_decode_swiglu(x: torch.Tensor) -> torch.Tensor:
    gate = current_gate()
    eligible = (
        gate is not None
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.size(1) % 2 == 0
        and torch_npu.get_npu_format(x) == 2
        and swiglu_fits(x.size(0), x.size(1) // 2, gate.free_aiv)
    )
    with gate.window("swiglu") if eligible else nullcontext():
        return torch_npu.npu_swiglu(x)


def sfa_decode_swiglu_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.size(-1) // 2))


def sfa_decode_w8a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    token_scale: torch.Tensor | None,
    name: str,
    reduce: bool,
) -> torch.Tensor:
    return w8a8_with_gate(current_gate(), x, weight, scale, token_scale, name, reduce)


def w8a8_with_gate(gate, x, weight, scale, token_scale, name, reduce):
    rows, width = x.shape
    columns = weight.size(1)
    eligible = gate is not None and x.is_contiguous() and torch_npu.get_npu_format(x) == 2
    with consecutive_windows(gate) as select:
        if token_scale is None:
            # BF16 DynamicQuant key 2 uses min(rows, 40) AIVs on aligned rows.
            select(
                "dynamic_quant",
                eligible
                and x.dtype == torch.bfloat16
                and width % 128 == 0
                and width <= 8192
                and 0 < rows <= gate.free_aiv,
            )
            x, token_scale = torch_npu.npu_dynamic_quant(x)
        select(
            name,
            eligible
            and gate.free_aiv >= 20
            and small_w8a8_shape(rows, width, columns)
            and x.dtype == weight.dtype == torch.int8
            and weight.ndim == 2
            and weight.size(0) == width
            and scale.dtype == torch.bfloat16
            and scale.shape == (columns,)
            and scale.is_contiguous()
            and token_scale.dtype == torch.float32
            and token_scale.shape == (rows,)
            and token_scale.is_contiguous()
            and x.is_contiguous()
            and torch_npu.get_npu_format(x) == 2
            and torch_npu.get_npu_format(weight) == 29,
        )
        output = torch_npu.npu_quant_matmul(
            x, weight, scale, pertoken_scale=token_scale, bias=None, output_dtype=torch.bfloat16
        )
    # Finish the admitted copy piece before launching the dependent HCCL task.
    if reduce:
        output = tensor_model_parallel_all_reduce(output)
        return output


def sfa_decode_w8a8_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    token_scale: torch.Tensor | None,
    name: str,
    reduce: bool,
) -> torch.Tensor:
    return x.new_empty((x.size(0), weight.size(1)), dtype=torch.bfloat16)
