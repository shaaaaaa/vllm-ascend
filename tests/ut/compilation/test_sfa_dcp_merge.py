# SPDX-License-Identifier: Apache-2.0
"""Check the production pack and combine indexing with a CPU Triton shim."""

import pytest
import torch
from sfa_test_support import ROOT, HostTL, Pointer, extract


class DcpTL(HostTL):
    # Signed int32 has the same byte lanes after the packing mask.
    uint32 = torch.int32
    static_range = staticmethod(range)
    exp = staticmethod(torch.exp)
    zeros = staticmethod(lambda shape, dtype: torch.zeros(shape, dtype=dtype))
    max = staticmethod(torch.amax)
    sum = staticmethod(torch.sum)

    @staticmethod
    def load(ptr):
        return HostTL.load(ptr, True, 0)

    @staticmethod
    def store(ptr, value):
        HostTL.store(ptr, value, True)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dcp_head_exchange_matches_full_precision_merge(world, dtype):
    tl = DcpTL()
    ns = {"tl": tl, "get_element": lambda tensor, index: tensor[index]}
    path = ROOT / "vllm_ascend/ops/triton/sfa_dcp.py"
    for name in ("_pack", "_unpack_stat", "_combine"):
        extract(path, name, ns)
    rng = torch.Generator().manual_seed(91)
    rows, local_heads, dim = 5, 2, 512
    heads = world * local_heads
    partial = torch.randn((world, rows, heads, dim), generator=rng, dtype=dtype)
    maximum = 12 * torch.randn((world, rows, heads), generator=rng)
    total = 1 + 512 * torch.rand((world, rows, heads), generator=rng)
    total[0, 1].zero_()
    total[:, 2].zero_()
    # Two nonempty shards with the same finite logits must be weighted by
    # their selected counts, even when log(2) is below one FP32 ulp of max.
    maximum[:, 3].fill_(2**24)
    total[:, 3].zero_()
    total[0, 3].fill_(1)
    total[1, 3].fill_(2)
    partial[0, 3].fill_(1)
    partial[1, 3].fill_(7)
    # A valid partial must not overflow when multiplied by an unnormalized sum.
    maximum[:, 4].zero_()
    total[:, 4].zero_()
    total[0, 4].fill_(1024)
    total[1, 4].fill_(1)
    partial[0, 4].fill_(1e36 if dtype == torch.bfloat16 else 60000)
    partial[1, 4].zero_()
    partial[total == 0] = 0
    send = torch.empty((world, world, rows, local_heads, dim + 16), dtype=dtype)
    tl.programs = 3
    for rank in range(world):
        for pid in range(tl.programs):
            tl.pid = pid
            ns["_pack"](
                Pointer(partial[rank]), Pointer(maximum[rank]), Pointer(total[rank]),
                Pointer(send[rank]), rows, heads, local_heads, dim,
            )

    lse = maximum.double() + total.double().log()
    lse[:, 2].zero_()
    reference = (lse.softmax(dim=0)[..., None] * partial.double()).sum(dim=0).to(dtype)
    for rank in range(world):
        output = torch.empty((rows, local_heads, dim), dtype=dtype)
        recv = send[:, rank].contiguous()
        for pid in range(tl.programs):
            tl.pid = pid
            ns["_combine"](Pointer(recv), Pointer(output), rows * local_heads, world, dim)
        start = rank * local_heads
        torch.testing.assert_close(output, reference[:, start : start + local_heads], rtol=1e-2, atol=1e-3)
        assert not output[2].any()
