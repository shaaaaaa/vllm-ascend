# SPDX-License-Identifier: Apache-2.0
"""Capture and compiler contracts for the qualified GLM decode windows."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from sfa_test_support import definitions, extract, load_module
from test_sfa_retrieval_overlap import capture_runtime  # noqa: F401

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def decoder(capture_runtime, monkeypatch):  # noqa: F811
    overlap, calls = capture_runtime
    gate = overlap.CopyGate("cpu")
    gate.soc, gate.cann = "Ascend910B3", "8.5.1"
    gate.pending, gate.free_aiv, gate.local_tp4 = 3, 20, True
    gate.main_stream = torch.npu.current_stream()
    context = NS(sfa_copy_gate=gate)
    native = NS(get_npu_format=lambda tensor: 29 if tensor.dtype == torch.int8 and tensor.shape == (128, 64) else 2)

    def quantize(x):
        calls.append(("quant", gate.open))
        return x.round().to(torch.int8), torch.ones(x.shape[0])

    def matmul(x, weight, scale, *, pertoken_scale, bias, output_dtype):
        calls.append(("matmul", gate.open))
        assert bias is None and output_dtype == torch.bfloat16
        return ((x.float() @ weight.float()) * scale * pertoken_scale[:, None]).to(output_dtype)

    def reduce(x):
        calls.append(("reduce", gate.open))
        return x * 4

    native.npu_dynamic_quant, native.npu_quant_matmul = quantize, matmul
    monkeypatch.setitem(sys.modules, "torch_npu", native)
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        NS(is_forward_context_available=lambda: True, get_forward_context=lambda: context),
    )
    monkeypatch.setitem(sys.modules, "vllm.distributed", NS(tensor_model_parallel_all_reduce=reduce))
    windows = load_module(ROOT / "vllm_ascend/compilation/sfa_copy_windows.py", "decoder_windows_test", monkeypatch)
    calls.clear()
    return windows, gate, calls, native


@pytest.mark.parametrize(
    ("rows", "cap", "local", "events"),
    [
        (16, 20, True, [True, ("quant", True), ("matmul", True), False, ("reduce", False)]),
        (20, 20, True, [True, ("quant", True), ("matmul", True), False, ("reduce", False)]),
        (21, 20, True, [("quant", False), True, ("matmul", True), False, ("reduce", False)]),
        (32, 20, False, [("quant", False), True, ("matmul", True), False, ("reduce", False)]),
        (16, 24, True, [True, ("quant", True), False, ("matmul", False), ("reduce", False)]),
        (28, 40, True, [("quant", False), ("matmul", False), ("reduce", False)]),
    ],
)
@pytest.mark.parametrize("projection", ["o_proj", "dense_down", "shared_down"])
def test_projection_closes_copy_window_before_allreduce(decoder, rows, cap, local, events, projection):
    w, gate, calls, _ = decoder
    gate.free_aiv, gate.local_tp4 = 40 - cap, local
    x = torch.arange(rows * 128).reshape(rows, 128).remainder(5).bfloat16()
    weight = torch.arange(128 * 64).reshape(128, 64).remainder(7).to(torch.int8)
    scale = torch.ones(64, dtype=torch.bfloat16)
    value = w.sfa_decode_w8a8(x, weight, scale, None, projection, True)
    assert [entry[1] if entry[0] == "gate" else entry for entry in calls] == events
    torch.testing.assert_close(value, (x.float() @ weight.float()).bfloat16() * 4)
    assert not gate.open


def test_capture_selection_and_exception_close(decoder):
    w, gate, _, _ = decoder
    assert w.current_gate() is gate
    with torch.npu.stream(torch.npu.Stream()):
        assert w.current_gate() is None
    with pytest.raises(ValueError, match="failed"), w.consecutive_windows(gate) as select:
        select("q_b", True)
        assert w.current_gate() is None
        raise ValueError("failed")
    assert not gate.open
    gate.pending = 0
    assert w.current_gate() is None
    gate.pending = 1
    gate.soc = "Ascend910B4"
    assert w.current_gate() is None
    gate.soc, gate.cann = "Ascend910B3", "9.1.0"
    assert w.current_gate() is None


def test_cann_detection_uses_installed_toolkit_when_runtime_has_no_version(capture_runtime, monkeypatch, tmp_path):  # noqa: F811
    module, _ = capture_runtime
    toolkit = tmp_path / "toolkit"
    info = toolkit / (module.platform.machine() + "-linux") / "ascend_toolkit_install.info"
    info.parent.mkdir(parents=True)
    info.write_text("arch=host\nversion=8.5.1.21\n", encoding="utf-8")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(toolkit))
    monkeypatch.setattr(torch.npu, "get_cann_version", Mock(side_effect=RuntimeError("unavailable")), raising=False)
    assert module._cann_version() == "8.5.1.21"
    monkeypatch.setattr(torch.npu, "get_cann_version", lambda *args: "unknown" if not args else "9.1.0")
    assert module._cann_version() == "9.1.0"


def test_qkv_wait_spans_only_supported_projection_and_padded_rows(decoder):
    w, gate, _, _ = decoder
    layer = type("DeepSeekV2FusedQkvAProjLinear", (torch.nn.Module,), {})()
    layer.tp_size, layer.gather_output, layer.bias = 1, False, None
    layer.weight = torch.ones((128, 64), dtype=torch.int8)
    x = torch.ones((21, 128), dtype=torch.bfloat16)
    assert w.join_through_qkv(layer, x[:20])
    assert not w.join_through_qkv(layer, x)
    hook = layer.register_forward_hook(lambda *_: None)
    assert not w.join_through_qkv(layer, x[:20])
    hook.remove()
    gate.pending = 1
    assert not w.join_through_qkv(layer, x[:20])


@pytest.mark.parametrize(
    ("rows", "expected_rms", "expected_residual"), [(10, True, True), (11, True, False), (28, False, False)]
)
def test_norm_and_residual_respect_actual_padded_shape(decoder, rows, expected_rms, expected_residual):
    w, gate, _, native = decoder
    observed = []

    def rms(x, weight, eps):
        observed.append(("rms", gate.open, x.is_contiguous()))
        return (x * weight).clone(), None

    class Kernel:
        def __getitem__(self, grid):
            def launch(x, y, out, scale, n_elements, num_blocks, **kwargs):
                observed.append(("muls_add", gate.open, grid[0]))
                out.copy_(x * scale + y)

            return launch

    native.npu_rms_norm = rms
    prefetch = NS(maybe_prefetch_mlp_weight_postprocess=lambda x: observed.append(("prefetch", gate.open)))
    scope = dict(
        torch=torch,
        current_gate=w.current_gate,
        nullcontext=nullcontext,
        get_weight_prefetch_method=lambda: prefetch,
        get_vectorcore_num=lambda: 40,
        muls_add_kernel=Kernel(),
    )
    definitions(ROOT / "vllm_ascend/ops/layernorm.py", {"forward_oot"}, scope, class_name="AscendRMSNorm")
    muls_add = extract(ROOT / "vllm_ascend/ops/triton/muls_add.py", "muls_add_triton", scope)
    raw = torch.ones((rows, 12288), dtype=torch.bfloat16)[:, ::2]
    norm = NS(weight=torch.full((6144,), 2, dtype=raw.dtype), variance_epsilon=1e-5, bias_loaded=False)
    x = scope["forward_oot"](norm, raw)
    torch.testing.assert_close(muls_add(x, x, scale=2.0), raw * 6)
    assert observed == [
        ("rms", expected_rms, expected_rms),
        ("prefetch", False),
        ("muls_add", expected_residual, min(2 * rows, 40)),
    ]


@pytest.mark.parametrize(("expert_gate", "combined"), [(False, True), (True, False)])
def test_shared_experts_apply_one_tp_reduction_after_the_optional_gate(decoder, expert_gate, combined):
    w, gate, calls, _ = decoder

    class Down(torch.nn.Module):
        def forward(self, x):
            calls.append(("down", gate.open))
            return x * 2, None

    down = type("AscendRowParallelLinear", (Down,), {})()
    down.tp_size, down.custom_op, down.input_is_parallel = 4, None, True
    down.reduce_results, down.bias, down.sfa_copy_projection = False, None, "shared_down"
    hidden = torch.ones((28, 128), dtype=torch.bfloat16)
    experts = NS(down_proj=down, act_fn=lambda x: x, expert_gate=(lambda x: (x * 0, None)) if expert_gate else None)
    ctx = NS(_shared_experts=experts, multistream_overlap_shared_expert=False, _shared_experts_part1=lambda x: x)

    def projection(layer, x, *, reduce):
        assert reduce and layer is down
        calls.append(("combined", gate.open))
        return x * 8

    scope = dict(
        torch=torch,
        F=torch.nn.functional,
        _EXTRA_CTX=NS(moe_comm_type="mc2"),
        MoECommType=NS(ALLTOALL="alltoall", MC2="mc2", FUSED_MC2="fused"),
        shared_expert_dp_enabled=lambda: False,
        npu_stream_switch=lambda *args, **kwargs: nullcontext(),
        shared_experts_calculation_stream=lambda: None,
        tagged_projection=w.tagged_projection,
        tp_fits=w.tp_fits,
        current_gate=w.current_gate,
        w8a8_projection=projection,
        tensor_model_parallel_all_reduce=w.tensor_model_parallel_all_reduce,
    )
    definitions(
        ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py",
        {"_shared_experts_part2", "_forward_shared_experts"},
        scope,
        class_name="AscendSharedFusedMoE",
    )
    ctx._shared_experts_part2 = lambda *args, **kwargs: scope["_shared_experts_part2"](ctx, *args, **kwargs)
    evts = NS(before_routed_experts=NS(recorded=True), before_dispatch=None, before_combine=None)
    out = scope["_forward_shared_experts"](ctx, hidden, evts)
    torch.testing.assert_close(out, hidden * (8 if combined else 4))
    assert ("combined", False) in calls if combined else ("reduce", False) in calls
    assert sum(op[0] in {"combined", "reduce"} for op in calls) == 1


def test_glm_projection_shapes_and_collective_boundary(decoder):
    w, gate, _, _ = decoder
    for rows in (1, 16, 28, 32, 64):
        for k, n in ((6144, 2112), (1536, 12288), (2048, 6144), (6144, 4096), (512, 6144)):
            assert w.small_w8a8_shape(rows, k, n)
    assert not w.small_w8a8_shape(32, 512, 32768)
    assert not w.small_w8a8_shape(65, 1536, 4096)
    assert not w.small_w8a8_shape(16, 129, 4096)
    assert w.tp_fits(gate, 336 * 1024)
    assert not w.tp_fits(gate, 1024 * 1024)
    assert w.projection_name("model.layers.4.mlp.shared_experts.down_proj") == "shared_down"
    assert w.projection_name("model.layers.4.mlp.experts.1.down_proj") is None


def test_compiler_keeps_custom_op_until_capture(decoder):
    from torch.fx.experimental.proxy_tensor import make_fx

    w, gate, calls, _ = decoder
    op = torch.library.custom_op("test_sfa_windows::w8a8_with_tp", mutates_args=())(w.sfa_decode_w8a8)
    op.register_fake(w.sfa_decode_w8a8_fake)
    x = torch.ones((28, 128), dtype=torch.bfloat16)
    weight = torch.ones((128, 64), dtype=torch.int8)
    scale = torch.ones(64, dtype=torch.bfloat16)
    model = make_fx(lambda a, b, c: op(a, b, c, None, "o_proj", True), tracing_mode="fake")(x, weight, scale)
    assert not calls
    expected = torch.full((28, 64), 512.0, dtype=torch.bfloat16)
    for pending in (0, 3, 0):
        gate.pending = pending
        calls.clear()
        torch.testing.assert_close(model(x, weight, scale), expected)
        assert ("matmul", bool(pending)) in calls and ("reduce", False) in calls


def test_fifo_grant_survives_first_consumer_join(decoder, capture_runtime):  # noqa: F811
    w, gate, _, native = decoder
    module, _ = capture_runtime
    gate.pending = 0
    side = torch.npu.Stream()
    data = torch.ones(1, 8)
    edges = tuple(
        module.RetrievalEdge(
            side, NS(request_capacity=1, load=Mock()), (data.clone(),), max_aiv_cores=20, pieces=4, gate=gate
        )
        for _ in range(3)
    )
    group = module.SharedPrefetchGroup(edges)
    group.launch(data, data, data)
    assert [e.transfer.load.call_count for e in edges] == [1] * 3
    assert gate.free_aiv == 20
    activation = torch.ones((1, 512), dtype=torch.bfloat16)
    observed = []

    def activate(x):
        observed.append(gate.open)
        return x.chunk(2, dim=-1)[0]

    native.npu_swiglu = activate
    for index, edge in enumerate(edges):
        edge.join()
        assert gate.pending == 2 - index
        assert (w.current_gate() is gate) == (index < 2)
        w.sfa_decode_swiglu(activation)
        assert not gate.open
    assert observed == [True, True, False]


@pytest.mark.parametrize("is_ep_comm", [False, True])
def test_maybe_pad_and_reduce_executes_once_with_the_copy_window_closed(decoder, is_ep_comm):
    w, _, calls, _ = decoder
    reduce_op = extract(
        ROOT / "vllm_ascend/ops/register_custom_ops.py",
        "_maybe_pad_and_reduce_impl",
        {
            "get_forward_context": lambda: NS(flash_comm_v1_enabled=False),
            "enable_sp_by_pass": lambda: False,
            "tensor_model_parallel_all_reduce": w.tensor_model_parallel_all_reduce,
        },
    )
    x = torch.ones((16, 128), dtype=torch.bfloat16)
    torch.testing.assert_close(reduce_op(x, is_ep_comm), x * 4)
    assert calls == [("reduce", False)]


@pytest.mark.parametrize(
    ("rows", "width", "free", "admitted"),
    [(28, 1024, 20, True), (32, 4096, 20, True), (64, 16384, 20, False), (16, 513, 20, False), (32, 4096, 0, False)],
)
def test_swiglu_tiling_limit(decoder, rows, width, free, admitted):
    w, gate, _, native = decoder
    seen = []

    def activate(x):
        seen.append(gate.open)
        a, b = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(a) * b

    native.npu_swiglu = activate
    gate.free_aiv = free
    x = torch.ones((rows, width), dtype=torch.bfloat16)
    if width % 2:
        # Real SwiGlu rejects this shape. Verify it reaches the original operator with the gate closed.
        with pytest.raises(RuntimeError):
            w.sfa_decode_swiglu(x)
    else:
        torch.testing.assert_close(w.sfa_decode_swiglu(x), torch.nn.functional.silu(x[:, : width // 2]))
    assert seen == [admitted]


def test_k_up_materializes_layout_before_opening(decoder):
    w, gate, calls, native = decoder
    scope = dict(torch=torch, torch_npu=native, current_gate=w.current_gate, nullcontext=nullcontext)
    k_up = extract(ROOT / "vllm_ascend/attention/sfa_v1.py", "_q_proj_and_k_up_proj", scope)
    hidden = torch.arange(4 * 2 * 48).reshape(4, -1).remainder(5).half()
    impl = NS(
        local_num_heads=2,
        qk_head_dim=48,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        W_UK_T=torch.ones((2, 32, 512), dtype=torch.float16),
        q_proj=lambda x: (x,),
    )
    reference = k_up(impl, hidden)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result = k_up(impl, hidden, prefetch_pending=True)
    assert any(e.key == "aten::contiguous" for e in profile.key_averages())
    assert calls == [("gate", True), ("gate", False)]
    torch.testing.assert_close(result, reference)
    assert not gate.open


@pytest.mark.parametrize(("local", "aux", "flat"), [(True, False, True), (False, False, False), (True, True, False)])
def test_post_combine_gathers_exact_tp_shards(decoder, local, aux, flat):
    w, gate, _, _ = decoder
    gate.local_tp4 = local
    seen = []

    def gather(output, x, group):
        assert group == "tp4"
        seen.append(("tensor" if isinstance(output, torch.Tensor) else "list", gate.open))
        shards = output.chunk(4) if isinstance(output, torch.Tensor) else output
        for rank, chunk in enumerate(shards):
            chunk.copy_(x + rank)

    ns = dict(
        torch=torch,
        current_gate=w.current_gate,
        tp_fits=w.tp_fits,
        dist=NS(all_gather=gather, all_gather_into_tensor=gather),
        get_ascend_config=lambda: NS(multistream_overlap_shared_expert=aux, multistream_overlap_gate=False),
    )
    definitions(
        ROOT / "vllm_ascend/ops/fused_moe/prepare_finalize.py",
        {"finalize"},
        ns,
        class_name="PrepareAndFinalizeWithAll2All",
    )
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    ctx = NS(
        tp_size=4,
        num_tokens=7,
        enable_shared_expert_dp=False,
        replace_allreduce=False,
        moe_config=NS(tp_group=NS(device_group="tp4")),
    )
    actual = ns["finalize"](ctx, x, True, (8, 2))
    torch.testing.assert_close(actual, torch.cat([x + i for i in range(4)])[:7])
    assert seen == [("tensor" if flat else "list", flat)]
