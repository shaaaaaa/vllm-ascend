/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
#define SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor npu_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value, int64_t sparse_block_size,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode)
{
    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }
    // construct the output tensor
    at::Tensor output = at::empty(query.sizes(), query.options().dtype(query.dtype()));
    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(
        aclnnSparseFlashAttention,
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        sparse_block_size,
        layout_query_ptr,
        layout_kv_ptr,
        sparse_mode,
        output);
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_sparse_flash_attention_dcp(
    const at::Tensor &query, const at::Tensor &query_rope,
    const at::Tensor &key, const at::Tensor &key_rope,
    const at::Tensor &sparse_indices, const at::Tensor &selected_counts,
    const at::Tensor &block_table, const at::Tensor &actual_seq_lengths_query,
    const at::Tensor &actual_seq_lengths_kv, double scale_value)
{
    TORCH_CHECK(query.dim() == 3 && query.numel() > 0 && query.size(1) % 8 == 0 &&
                    key.dim() == 4 && key.size(2) == 1,
                "DCP SFA requires TND queries with aligned heads and one paged KV head");
    TORCH_CHECK(sparse_indices.dim() == 3 && sparse_indices.size(0) == query.size(0) &&
                    sparse_indices.size(1) == 1 && selected_counts.dim() == 2 &&
                    selected_counts.size(0) == query.size(0) &&
                    selected_counts.size(1) == 16 &&
                    selected_counts.scalar_type() == at::kInt,
                "DCP SFA requires cacheline-separated int32 counts for packed sparse rows");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) > 0 &&
                    actual_seq_lengths_query.dim() == 1 && actual_seq_lengths_kv.dim() == 1 &&
                    actual_seq_lengths_query.size(0) == block_table.size(0) &&
                    actual_seq_lengths_kv.size(0) == block_table.size(0),
                "DCP SFA requires one query boundary and local KV extent per block table row");
    auto output = at::empty(query.sizes(), query.options());
    auto stats_options = query.options().dtype(at::kFloat);
    auto softmax_max = at::empty({query.size(0), query.size(1)}, stats_options);
    auto softmax_sum = at::empty_like(softmax_max);
    char tnd[] = "TND";
    char paged[] = "PA_BSND";
    char *layout_query = tnd;
    char *layout_kv = paged;
    const int64_t sparse_block_size = 1;
    const int64_t sparse_mode = 0;
    EXEC_NPU_CMD(
        aclnnSparseFlashAttentionDcp, query, key, key, sparse_indices, block_table,
        actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope,
        selected_counts, scale_value, sparse_block_size, layout_query, layout_kv,
        sparse_mode, output, softmax_max, softmax_sum);
    return {output, softmax_max, softmax_sum};
}
}
#endif
