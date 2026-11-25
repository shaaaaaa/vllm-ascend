# SPDX-License-Identifier: Apache-2.0
"""Exercise the live packet dispatch, fused RoPE arithmetic, and native DMA addresses on a host."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from sfa_test_support import ROOT, Pointer, extract
from test_sfa_dcp_merge import DcpTL
from test_sfa_dcp_tuning import async_ops, gather_queries  # noqa: F401


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("active,mode", [(True, "output"), (True, "direct"), (True, "fused"), (False, "fused")])
def test_full_operator_preserves_gather_placement_and_packet_lifetime(monkeypatch, early, active, mode):
    events = []
    q, r = (torch.zeros(2, 8, n) for n in (512, 64))
    packet = torch.cat((q, r), -1)
    key = object()
    ctx = NS(sfa_full_graph_active=active, staged_sfa_graph_key=key)
    impl = NS(
        shared_resident_plan=NS(active=True),
        skip_topk=False,
        _dcp_packet_io=mode,
        _dcp_early_gather=early,
        _shared_retrieval_edges=lambda: ({}, {}),
    )

    def fused(q_in, r_in, cos, sin):
        assert q_in is q and r_in is r
        events.append("rope_packet")
        return packet

    def submit(q_in, r_in, group, *, packet_mode, packet=None):
        assert packet_mode == mode and q_in is q
        if mode == "fused":
            assert r_in.data_ptr() == packet[..., 512:].data_ptr()
        else:
            assert packet is None
        events.append("gather")
        return NS(wait=lambda: events.append("join"))

    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.sfa_query", NS(rope_sfa_dcp_query_packet=fused))
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.triton.sfa_dcp", NS(start_sfa_dcp_query_gather=submit))

    def pre(*args, query_ready=None, packet_rope=None, **kwargs):
        rr = packet_rope(q, r, None, None) if packet_rope else r
        if query_ready:
            query_ready(q, rr)
        events.append("plan")
        return q, rr, r, r, r, r

    def post(*args, query_gather=None, **kwargs):
        assert (query_gather is not None) == active
        events.append("attention")

    impl.cross_layer_graph_pre, impl.cross_layer_graph_post = pre, post
    impl.cross_layer_lmcache_retrieve = lambda *args: events.append("retrieve")
    run = extract(
        ROOT / "vllm_ascend/ops/mla.py",
        "sfa_forward_full",
        {
            "get_forward_context": lambda: ctx,
            "_mla_runtime_state": lambda _: (impl, "layer", (), NS(sfa_dcp=1)),
            "get_dcp_group": lambda: "dcp",
        },
    )
    run(q, False, q, "layer", "", [], [])
    expected = ["rope_packet"] if active and mode == "fused" else []
    if active and early:
        expected.append("gather")
    assert events == expected + ["plan", "retrieve"] + (["gather"] if active and not early else []) + ["attention"] + (
        ["join"] if active else []
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,heads", [(1, 1), (2, 8), (7, 16)])
def test_fused_packet_preserves_latent_bits_and_interleave_rope(dtype, rows, heads):
    rng = torch.Generator().manual_seed(9)
    bits = torch.randint(-32768, 32768, (heads, rows, 512), generator=rng, dtype=torch.int16).transpose(0, 1)
    r = torch.randn((rows, heads, 192), generator=rng, dtype=dtype)[..., 128:]
    c, s = (torch.randn(rows, 128, generator=rng, dtype=dtype)[:, :64] for _ in range(2))
    r[0, 0, :4].zero_()
    c[0, :2].fill_(-1)
    s[0, :2] = s.new_tensor([1, -1])
    out = torch.empty(rows, heads, 576, dtype=dtype)
    tl = DcpTL()
    kernel = extract(ROOT / "vllm_ascend/ops/triton/sfa_query.py", "_rope_query_packet", {"tl": tl})
    tl.programs = 3
    for pid in range(tl.programs):
        tl.pid = pid
        kernel(
            *(Pointer(t) for t in (bits, r, c, s, out.view(torch.int16), out)),
            rows * heads,
            heads,
            *bits.stride()[:2],
            *r.stride()[:2],
            c.stride(0),
            s.stride(0),
        )
    even, odd = r.float()[..., ::2], r.float()[..., 1::2]
    ref = torch.cat((even, odd), -1) * c.float()[:, None] + torch.cat((-odd, even), -1) * s.float()[:, None]
    assert torch.equal(out[..., :512].view(torch.int16), bits)
    assert torch.equal(out[..., 512:].view(torch.int16), ref.to(dtype).view(torch.int16))


@pytest.mark.parametrize("mode", ["output", "direct", "fused"])
@pytest.mark.parametrize("after_pack", [False, True])
def test_candidate_consumes_one_gather_and_one_send(request, monkeypatch, mode, after_pack):
    events = []
    ns = async_ops(request.getfixturevalue("gather_queries"), events)
    q, rope = (torch.randn(3, 8, n, dtype=torch.bfloat16) for n in (512, 64))
    group = NS(device_group="dcp", world_size=2)
    packet = torch.cat((q, rope), -1)
    gather = ns["start_sfa_dcp_query_gather"](
        q, rope, group, packet=packet if mode == "fused" else None, packet_mode=mode
    )
    if mode != "output":
        ns["_unpack_sfa_dcp_queries"] = lambda *a: pytest.fail("unexpected query unpack")
    send = torch.empty(2, 3, 8, 528)
    topk = torch.full((3, 1, 2048), -1, dtype=torch.int32)
    counts = torch.zeros(3, 16, dtype=torch.int32)

    def native(inp, r, key, key_rope, indices, selected_counts, *args):
        events.append("native")
        assert args[-1] == 2
        assert indices is topk and selected_counts is counts
        if mode == "output":
            assert torch.equal(inp, q.repeat(1, 2, 1)) and torch.equal(r, rope.repeat(1, 2, 1))
        else:
            assert r is None and torch.equal(inp, packet.expand(2, -1, -1, -1))
        return send

    def exchange(tensor, comm, prefetch):
        assert tensor is send and comm == "dcp"
        events.append("exchange")
        if prefetch:
            prefetch[0].launch(*prefetch[1:])
        return send

    ns["_exchange_sfa_dcp_packet"] = exchange
    ns["merge_sfa_dcp"] = lambda *a, **kw: pytest.fail("unexpected output pack")
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_attention_dcp_packet", native, raising=False)
    meta = torch.ones(1, dtype=torch.int32)
    edge = NS(launch=lambda *a: events.append("copy"))
    assert (
        ns["execute_sfa_dcp"](
            q,
            rope,
            (None, None),
            topk,
            counts,
            meta,
            meta,
            meta,
            1.0,
            group,
            query_gather=gather,
            prefetch=(edge, None, None, None, None),
            prefetch_after_pack=after_pack,
        )
        is send
    )
    gather.wait()
    assert events == ["submit", "wait", "native"] + (["exchange", "copy"] if after_pack else ["copy", "exchange"])


@pytest.mark.parametrize(
    "reference,stage",
    [("off", "output"), ("off", "direct"), ("direct", "fused"), ("fused", "fused")],
)
@pytest.mark.parametrize("ref_cap,cap", [(0, 0), (0, 12), (12, 0), (12, 12)])
def test_benchmark_selects_reference_and_candidate(monkeypatch, reference, stage, ref_cap, cap):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/ops"))
    import benchmark_sfa_dcp_packet_io as bench

    args = NS(stage=stage, reference_stage=reference, unpack_placement="producer",
              reference_serial_aiv_cores=ref_cap, serial_aiv_cores=cap)
    monkeypatch.setattr(bench, "UnpackPipeline", lambda a, i, s, g, overlap: NS(overlap=overlap, args=a))
    for candidate, mode in ((False, reference), (True, stage)):
        arm = bench.packet_pipeline(args, None, None, None, candidate)
        assert arm.packet_mode == mode and arm.overlap == (mode in ("off", "output"))
        assert arm.args.serial_aiv_cores == (cap if candidate else ref_cap)
        assert args.serial_aiv_cores == cap


def test_native_dma_and_packet_writers(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(compiler[0]):
        pytest.skip("requires a host C++ compiler")
    kernel = ROOT / "csrc/sparse_flash_attention/op_kernel"

    def function(file, signature):
        source = (kernel / f"sparse_flash_attention_{file}.h").read_text(encoding="utf8")
        start = source.index(signature)
        pos = source.index("{", start)
        depth = 1
        for end in range(pos + 1, len(source)):
            depth += (source[end] == "{") - (source[end] == "}")
            if not depth:
                return source[pos : end + 1]
        raise AssertionError(signature)

    functions = {
        "gm_to_l1": ("service_cube_mla", "::CopyGmToL1("),
        "read": ("service_cube_mla", "void CopyPacketToL1("),
        "partial": ("service_vector_mla", "::Bmm2DataCopyOutTrans("),
        "stats": ("service_vector_mla", "::WritePacketStats("),
        "zero": ("service_vector_mla", "::ZeroPacketRow("),
    }
    for name, (file, sig) in functions.items():
        (tmp_path / f"{name}.inc").write_text(function(file, sig), encoding="utf8")
    exe = tmp_path / ("packet.exe" if os.name == "nt" else "packet")
    src = Path(__file__).with_name("sfa_dcp_packet_io_host.cpp")
    subprocess.run(
        [*compiler, "-std=c++17", "-I", str(tmp_path), "-I", str(kernel), str(src), "-o", str(exe)],
        check=True,
        timeout=90,
    )
    subprocess.run([str(exe)], check=True, timeout=30)
