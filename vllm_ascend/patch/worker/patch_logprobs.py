# SPDX-License-Identifier: Apache-2.0

import torch
import vllm.v1.sample.ops.logprobs as logprobs
import vllm.v1.sample.sampler as sampler


def batched_count_greater_than(
    x: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    return (x >= values).sum(-1)


logprobs.batched_count_greater_than = batched_count_greater_than
sampler.batched_count_greater_than = batched_count_greater_than
