# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs"])
def prepare_next_mtp_tokens_kernel(
    sampled, backup, next_tokens, valid_counts,
    num_reqs, vocab_size, row_stride, token_stride,
    WIDTH: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Fuse the existing Ascend next-token/count rule for one-draft MTP."""
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK_SIZE
    for start in tl.range(pid * BLOCK_SIZE, num_reqs, step):
        rows = start + tl.arange(0, BLOCK_SIZE)
        mask = rows < num_reqs
        first = tl.load(sampled + rows * row_stride, mask, other=-1)
        second = tl.full((BLOCK_SIZE,), -1, tl.int32)
        if WIDTH == 2:
            second = tl.load(sampled + rows * row_stride + token_stride, mask, other=-1)
        count = ((first != -1) & (first < vocab_size)).to(tl.int32)
        count += ((second != -1) & (second < vocab_size)).to(tl.int32)
        fallback = tl.load(backup + rows, mask & (count == 0), other=0)
        # Match gather(count - 1), including irregular validity patterns;
        # selecting the last valid physical index would change the old rule.
        token = tl.where(count == 2, second, first)
        tl.store(next_tokens + rows, tl.where(count > 0, token, fallback), mask)
        tl.store(valid_counts + rows, count, mask)


@triton.jit(do_not_specialize=["num_reqs"])
def prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,
    num_reqs,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # Grid-Stride Loop:
    block_start_step = num_programs * BLOCK_SIZE

    for block_start in tl.range(pid * BLOCK_SIZE, num_reqs, block_start_step):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_reqs

        # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
        # cumulative sum (first entry is the first value, not zero).
        cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + offsets, mask=mask)

        prev_indices = offsets - 1
        has_prev = offsets > 0
        cu_draft_prev = tl.load(
            cu_num_draft_tokens_ptr + prev_indices,
            mask=mask & has_prev,
            other=0,
        )

        num_draft_tokens = tl.where(has_prev, cu_draft_curr - cu_draft_prev, cu_draft_curr)

        valid_count = tl.load(valid_sampled_tokens_count_ptr + offsets, mask=mask)
        num_rejected = num_draft_tokens + 1 - valid_count
        num_rejected = tl.where(num_draft_tokens > 0, num_rejected, 0)

        # query_start_loc[req_idx + 1] is the start position of the next request,
        # which is one past the last token of this request.
        q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + offsets + 1, mask=mask) - 1

        index_to_sample = q_last_tok_idx - num_rejected
        tl.store(token_indices_to_sample_ptr + offsets, index_to_sample, mask=mask)
        tl.store(num_rejected_tokens_gpu_ptr + offsets, num_rejected, mask=mask)
