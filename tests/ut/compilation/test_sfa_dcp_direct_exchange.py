# SPDX-License-Identifier: Apache-2.0
"""Direct exchange routing, raw statistics and full benchmark scope on CPU."""

from types import SimpleNamespace

import pytest
import torch
from sfa_test_support import ROOT, Pointer, extract, load_module
from test_sfa_dcp_merge import DcpTL

OPS = ROOT / "vllm_ascend/ops/triton/sfa_dcp.py"


class Combine:
    def __init__(self):
        self.tl = DcpTL()
        self.body = extract(OPS, "_combine", {"tl": self.tl, "get_element": lambda value, index: value[index]})

    def __getitem__(self, grid):
        def launch(*args):
            self.tl.programs = grid[0]
            for self.tl.pid in range(grid[0]):
                self.body(*(Pointer(arg) if isinstance(arg, torch.Tensor) else arg for arg in args))

        return launch


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 2, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_direct_exchange_routes_native_views_and_preserves_fp32_mass(world, rows, dtype):
    heads, dim = world * 2, 512
    generator = torch.Generator().manual_seed(81)
    partial = torch.randn((world, rows, heads, dim), generator=generator, dtype=dtype)
    maxima = torch.randn((world, rows, heads), generator=generator) * 12
    sums = torch.rand((world, rows, heads), generator=generator) * 2048 + 1
    # Empty ranks/heads and an all-empty output, with arbitrary empty maxima.
    sums[0, 0, :2] = 0
    sums[:, 0, 2] = 0
    maxima[sums == 0] = float("nan")
    if rows > 1:
        # Computing max+log(sum) would lose this 1:2 mass ratio at FP32 max.
        maxima[:, 1] = 2**24
        sums[:, 1] = 0
        sums[0, 1], sums[1, 1] = 1, 2
        partial[0, 1], partial[1, 1] = 1, 7
    if rows > 2:
        maxima[:, 2] = 0
        partial[:, 2] = 1e36 if dtype == torch.bfloat16 else 60000
    partial[sums == 0] = 0
    safe_max = torch.where(sums > 0, maxima.double(), -float("inf"))
    best = safe_max.amax(0)
    best[best == -float("inf")] = 0
    weights = sums.double() * (safe_max - best).exp()
    weights /= weights.sum(0).clamp_min(1e-300)
    reference = (weights[..., None] * partial.double()).sum(0).to(dtype)

    for rank in range(world):
        calls = []
        sources = partial[rank], maxima[rank], sums[rank]

        def exchange(dest, source, *, group, calls=calls, sources=sources, rank=rank):
            row, plane = divmod(len(calls), 3)
            assert group == "dcp"
            native = sources[plane][row]
            assert source.is_contiguous() and dest.is_contiguous()
            assert source.data_ptr() == native.data_ptr()  # No hidden packing/copy.
            assert source.dtype == native.dtype
            all_values = (partial, maxima, sums)[plane]
            # Rank r receives its two heads from every KV owner in source order.
            dest.copy_(all_values[:, row, rank * 2 : rank * 2 + 2].flatten(0, 1))
            calls.append((row, plane))

        merge = extract(
            OPS,
            "merge_sfa_dcp",
            {
                "torch": torch,
                "dist": SimpleNamespace(get_world_size=lambda _: world, all_to_all_single=exchange),
                "init_device_properties_triton": lambda: None,
                "get_vectorcore_num": lambda: 3,
                "_combine": Combine(),
            },
        )
        output = merge(*sources, "dcp", _direct=True)
        assert len(calls) == rows * 3
        torch.testing.assert_close(output, reference[:, rank * 2 : rank * 2 + 2], rtol=1e-2, atol=1e-3)
        if rows > 1:
            assert torch.all(output[1] == 5)


@pytest.mark.parametrize("failure", ["heads", "value_dim", "dtype", "stat_dtype", "strides"])
def test_direct_exchange_rejects_invalid_native_output_before_communication(failure):
    partial = torch.empty((2, 8, 512), dtype=torch.bfloat16)
    maximum = torch.empty((2, 8), dtype=torch.float32)
    if failure == "heads":
        partial = torch.empty((2, 7, 512), dtype=partial.dtype)
    elif failure == "value_dim":
        partial = torch.empty((2, 8, 256), dtype=partial.dtype)
    elif failure == "dtype":
        partial = partial.float()
    elif failure == "stat_dtype":
        maximum = maximum.half()
    elif failure == "strides":
        partial = torch.empty((8, 2, 512), dtype=partial.dtype).transpose(0, 1)
    merge = extract(OPS, "merge_sfa_dcp", {"torch": torch, "dist": SimpleNamespace(get_world_size=lambda _: 8)})
    with pytest.raises(ValueError):
        merge(partial, maximum, maximum, None, _direct=True)


@pytest.fixture
def bench(monkeypatch):
    return load_module(ROOT / "benchmarks/ops/benchmark_sfa_dcp_exchange.py", "dcp_exchange_bench", monkeypatch)


@pytest.mark.parametrize("layout", ["balanced", "skewed", "empty"])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_benchmark_counts_cover_global_topk_and_padding(bench, layout, world):
    for step in range(3):
        all_counts = torch.stack([bench.make_counts(4, 2, 3, world, rank, step, layout, 2048) for rank in range(world)])
        assert not all_counts[:, 8:].any()
        expected = torch.zeros(8, dtype=torch.int32) if layout == "empty" else 2048 - (step + torch.arange(8)) % 17
        assert torch.equal(all_counts[:, :8].sum(0), expected)
        if layout == "skewed":
            assert torch.all((all_counts[:, :8] > 0).sum(0) == 1)


def test_benchmark_times_complete_attention_for_each_layer_and_step(bench, monkeypatch):
    import sys
    from types import ModuleType

    calls = []
    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    module.execute_sfa_dcp = lambda *args, **kwargs: calls.append((args, kwargs)) or len(calls)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = SimpleNamespace(
        requests=2,
        query_rows=2,
        padding=3,
        heads=64,
        world=8,
        rank=1,
        seed=8,
        dtype="bfloat16",
        local_kv_tokens=128,
        steps=3,
        layers=2,
        layout="skewed",
    )
    case = bench.AttentionCase(args, torch.device("cpu"))
    assert not case.frames[0][0].is_contiguous()
    assert case.metadata[1].tolist() == [2, 4, 7]
    assert case.metadata[2].tolist() == [128, 128, 0]
    for direct in (False, True):
        calls.clear()
        result = case.forward("group", direct)
        assert len(result) == args.steps * args.layers == len(calls)
        for i, (positional, keyword) in enumerate(calls):
            assert positional[0] is case.frames[i // args.layers][0]
            assert positional[2] is case.cache and positional[-1] == "group"
            assert keyword == {"_direct_exchange": direct}


def test_benchmark_reports_slowest_rank_per_sample(bench):
    result = bench.summarize(
        [
            {"packed": [1, 4, 2], "direct": [3, 2, 1]},
            {"packed": [3, 1, 5], "direct": [2, 1, 4]},
        ]
    )
    assert result["packed"]["samples_us"] == [3, 4, 5]
    assert result["direct"]["samples_us"] == [3, 2, 4]
    assert result["packed"]["median_us"] == 4
    assert result["direct"]["median_us"] == 3


@pytest.mark.parametrize("direct", [False, True])
def test_direct_selection_preserves_query_attention_prefetch_order(monkeypatch, direct):
    events = []

    def gather(q, rope, group):
        events.append("gather")
        return q, rope

    def native(*args):
        events.append("attention")
        return "partial", "max", "sum"

    def merge(*args, **kwargs):
        events.append("merge")
        assert args == ("partial", "max", "sum", "group")
        assert kwargs == ({"_direct": True} if direct else {})
        return "result"

    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_attention_dcp", native, raising=False)
    execute = extract(
        OPS,
        "execute_sfa_dcp",
        {
            "torch": torch,
            "_gather_sfa_dcp_queries": gather,
            "merge_sfa_dcp": merge,
        },
    )
    edge = SimpleNamespace(launch=lambda *args: events.append("prefetch"))
    metadata = torch.zeros(2, dtype=torch.int32)
    result = execute(
        None,
        None,
        (None, None),
        None,
        None,
        metadata,
        metadata,
        metadata,
        1.0,
        SimpleNamespace(device_group="group"),
        prefetch=(edge, None, None, None, None),
        _direct_exchange=direct,
    )
    assert result == "result"
    assert events == ["gather", "attention", "prefetch", "merge"]


def test_benchmark_agrees_on_parity_failure_and_reports_mismatch(bench, monkeypatch):
    def agree(errors, local, *, group):
        assert group == "control" and "Mismatched elements" in local
        errors[:] = [local, None]

    monkeypatch.setattr(bench.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(bench.dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(bench.dist, "all_gather_object", agree)
    with pytest.raises(AssertionError, match="direct replay=1"):
        bench.check_outputs([torch.ones(2)], [torch.zeros(2)], "control", "direct replay=1")
