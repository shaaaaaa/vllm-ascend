# SPDX-License-Identifier: Apache-2.0

import torch
import vllm.v1.sample.ops.logprobs as logprobs
import vllm.v1.sample.sampler as sampler

from vllm_ascend.patch.worker.patch_logprobs import batched_count_greater_than


def test_logprobs_count_uses_eager_ascend_helper():
    assert logprobs.batched_count_greater_than is batched_count_greater_than
    assert sampler.batched_count_greater_than is batched_count_greater_than
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 4.0, 0.0]])
    values = torch.tensor([[2.0], [3.0]])
    assert torch.equal(batched_count_greater_than(x, values), torch.tensor([1, 2]))
