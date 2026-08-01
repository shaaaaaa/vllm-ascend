/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>
#include "kernels/types.h"
#include "torch_npu/csrc/aten/common/from_blob.h"

namespace vllm_ascend {
  extern void get_masked_input_and_mask_impl(
    void* stream,
    void* input,
    void* masked_input,
    void* mask_out,
    const int64_t org_vocab_start_index,
    const int64_t org_vocab_end_index,
    const int64_t num_org_vocab_padding, 
    const int64_t added_vocab_start_index,
    const int64_t added_vocab_end_index,
    const int64_t size,
    const uint32_t loop_cnt,
    const uint32_t aiv_num);

  extern void dsa_prepare_sparse_indices_impl(
    void* stream,
    void* topk_indices,
    void* split_boundary,
    void* row_req_indices,
    void* request_block_table,
    void* selected_packed,
    void* selected_counts,
    void* target_slots,
    uint32_t row_count,
    uint32_t row_width,
    uint32_t request_count,
    uint32_t block_table_width,
    uint32_t scratch_capacity,
    uint32_t selected_count_stride,
    uint32_t bitmap_words,
    uint32_t block_size,
    bool need_packed,
    bool clear_invalid_rows);

  extern void dsa_prepare_sparse_indices_legacy_impl(
    void* stream,
    void* topk_indices,
    void* split_boundary,
    void* valid_rows,
    void* scratch_base,
    void* selected_packed,
    void* row_req_indices,
    uint32_t row_count,
    uint32_t row_width,
    uint32_t valid_row_count,
    uint32_t core_count,
    bool need_packed,
    bool clear_invalid_rows,
    uint32_t packed_key_stride);

  extern void dsa_prepare_sparse_indices_staged_impl(
    void* stream,
    void* topk_indices,
    void* split_boundary,
    void* row_req_indices,
    void* request_block_table,
    void* selected_packed,
    void* selected_count,
    void* target_slots,
    void* local_to_union,
    uint32_t row_count,
    uint32_t row_width,
    uint32_t request_count,
    uint32_t rows_per_request,
    uint32_t scratch_capacity,
    uint32_t block_table_width,
    uint32_t selected_count_stride,
    uint32_t block_size,
    uint32_t core_count,
    bool need_packed,
    bool clear_invalid_rows);

  extern void dsa_prepare_sparse_indices_sharded_impl(
    void* stream,
    void* topk_indices,
    void* split_boundary,
    void* row_req_indices,
    void* request_block_table,
    void* selected_packed,
    void* selected_count,
    void* target_slots,
    void* local_to_union,
    void* shard_packed,
    void* shard_mapping,
    void* shard_counts,
    uint32_t request_count,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t scratch_capacity,
    uint32_t block_table_width,
    uint32_t selected_count_stride,
    uint32_t block_size,
    uint32_t core_count,
    bool need_packed,
    bool clear_invalid_rows);

  extern void dsa_staged_hash_union_impl(
    void* stream,
    void* row_packed,
    void* selected_packed,
    void* local_to_union,
    void* selected_count,
    void* request_block_table,
    void* target_slots,
    uint32_t row_count,
    uint32_t row_width,
    uint32_t max_tokens,
    uint32_t block_table_width,
    uint32_t selected_count_stride,
    uint32_t block_size);

  extern void dsa_staged_sort_union_impl(
    void* stream,
    void* row_packed,
    void* selected_packed,
    void* local_to_union,
    void* selected_count,
    void* request_block_table,
    void* target_slots,
    uint32_t row_count,
    uint32_t row_width,
    uint32_t block_table_width,
    uint32_t selected_count_stride,
    uint32_t block_size);

  extern void dsa_staged_sharded_sort_union_impl(
    void* stream,
    void* topk_indices,
    void* split_boundary,
    void* selected_packed,
    void* local_to_union,
    void* selected_count,
    void* request_block_table,
    void* target_slots,
    void* shard_packed,
    void* shard_mapping,
    void* shard_counts,
    uint32_t request_count,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t shard_count,
    uint32_t block_table_width,
    uint32_t selected_count_stride,
    uint32_t shard_count_stride,
    uint32_t block_size);

  extern void dsa_staged_sharded_vector_union_impl(
    void* stream,
    void* topk_indices,
    void* split_boundary,
    void* selected_packed,
    void* local_to_union,
    void* selected_count,
    void* request_block_table,
    void* target_slots,
    void* shard_packed,
    void* shard_mapping,
    void* shard_counts,
    void* shard_pairs,
    uint32_t request_count,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t shard_count,
    uint32_t block_table_width,
    uint32_t selected_count_stride,
    uint32_t shard_count_stride,
    uint32_t block_size);

  extern void dsa_staged_sharded_vector_dedup_impl(
    void* stream,
    void* topk_indices,
    void* split_boundary,
    void* selected_packed,
    void* local_to_union,
    void* selected_count,
    void* request_block_table,
    void* target_slots,
    void* shard_packed,
    void* shard_mapping,
    void* shard_counts,
    uint32_t request_count,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t shard_count,
    uint32_t block_table_width,
    uint32_t selected_count_stride,
    uint32_t shard_count_stride,
    uint32_t block_size);

  extern void dsa_staged_remap_rows_impl(
    void* stream,
    void* local_indices,
    void* local_to_union,
    uint32_t row_count,
    uint32_t row_width);

  extern void dsa_resident_remap_rows_impl(
    void* stream,
    void* topk_indices,
    void* position_to_union,
    void* union_to_slot,
    uint32_t row_count,
    uint32_t row_width,
    uint32_t rows_per_request,
    uint32_t scratch_capacity);

  extern void dsa_resident_lookup_rows_impl(
    void* stream,
    void* selected_packed,
    void* selected_count,
    void* request_state_indices,
    void* lookup_indices,
    uint32_t request_count,
    uint32_t scratch_capacity,
    uint32_t selected_count_stride,
    uint32_t token_stride,
    uint32_t dummy_state_base);

  extern void dsa_resident_finalize_rows_impl(
    void* stream,
    void* selected_packed,
    void* selected_count,
    void* target_slots,
    void* request_block_table,
    void* request_state_indices,
    void* request_state_generations,
    void* old_slots,
    void* slot_to_token,
    void* state_generations,
    void* union_to_slot,
    void* reverse_indices,
    void* reverse_values,
    uint32_t request_count,
    uint32_t scratch_capacity,
    uint32_t selected_count_stride,
    uint32_t block_table_width,
    uint32_t token_stride,
    uint32_t slot_stride,
    uint32_t generation_stride,
    uint32_t dummy_state_base,
    uint32_t block_size);

  extern void dsa_resident_sharded_union_impl(
    void* stream,
    void* topk_indices,
    void* split_boundary,
    void* row_req_indices,
    void* shard_packed,
    void* shard_mapping,
    void* shard_counts,
    void* request_state_indices,
    void* request_state_generations,
    void* state_tokens,
    void* state_slots,
    void* state_counts,
    void* state_generations,
    void* prior_slots,
    void* shard_miss_tokens,
    void* shard_miss_positions,
    void* shard_evictable_slots,
    uint32_t request_count,
    uint32_t state_row_count,
    uint32_t dummy_state_base,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t shard_count,
    uint32_t shard_capacity,
    uint32_t shard_count_stride,
    uint32_t shard_count_request_stride,
    uint32_t generation_stride);

  extern void dsa_resident_sharded_union_v2_impl(
    void* stream,
    void* topk_indices,
    void* split_boundary,
    void* row_req_indices,
    void* shard_packed,
    void* shard_mapping,
    void* shard_counts,
    void* request_state_indices,
    void* request_state_generations,
    void* state_tokens,
    void* state_slots,
    void* state_counts,
    void* state_generations,
    void* prior_slots,
    void* shard_miss_tokens,
    void* shard_miss_positions,
    void* shard_evictable_slots,
    uint32_t request_count,
    uint32_t state_row_count,
    uint32_t dummy_state_base,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t shard_count,
    uint32_t shard_capacity,
    uint32_t shard_count_stride,
    uint32_t shard_count_request_stride,
    uint32_t generation_stride);

  extern void dsa_resident_sorted_plan_impl(
    void* stream,
    void* topk_indices,
    void* shard_packed,
    void* shard_mapping,
    void* shard_counts,
    void* request_block_table,
    void* request_state_indices,
    void* request_state_generations,
    void* state_tokens,
    void* state_slots,
    void* state_counts,
    void* state_generations,
    void* prior_slots,
    void* shard_miss_tokens,
    void* shard_miss_positions,
    void* shard_evictable_slots,
    void* miss_tokens,
    void* miss_counts,
    void* target_slots,
    uint32_t request_count,
    uint32_t state_row_count,
    uint32_t dummy_state_base,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t shard_count,
    uint32_t capacity,
    uint32_t shard_count_stride,
    uint32_t shard_count_request_stride,
    uint32_t miss_count_stride,
    uint32_t generation_stride,
    uint32_t block_table_width,
    uint32_t block_size);

  extern void dsa_resident_parallel_plan_v2_impl(
    void* stream,
    void* topk_indices,
    void* shard_packed,
    void* shard_mapping,
    void* shard_counts,
    void* request_block_table,
    void* request_state_indices,
    void* request_state_generations,
    void* state_tokens,
    void* state_slots,
    void* state_counts,
    void* state_generations,
    void* prior_slots,
    void* shard_miss_tokens,
    void* shard_miss_positions,
    void* shard_evictable_slots,
    void* miss_tokens,
    void* miss_counts,
    void* target_slots,
    uint32_t request_count,
    uint32_t state_row_count,
    uint32_t dummy_state_base,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t shard_count,
    uint32_t capacity,
    uint32_t shard_count_stride,
    uint32_t shard_count_request_stride,
    uint32_t miss_count_stride,
    uint32_t generation_stride,
    uint32_t block_table_width,
    uint32_t block_size);

  extern void dsa_resident_sorted_plan_no_remap_impl(
    void* stream,
    void* topk_indices,
    void* shard_packed,
    void* shard_mapping,
    void* shard_counts,
    void* request_block_table,
    void* request_state_indices,
    void* request_state_generations,
    void* state_tokens,
    void* state_slots,
    void* state_counts,
    void* state_generations,
    void* prior_slots,
    void* shard_miss_tokens,
    void* shard_miss_positions,
    void* shard_evictable_slots,
    void* miss_tokens,
    void* miss_counts,
    void* target_slots,
    uint32_t request_count,
    uint32_t state_row_count,
    uint32_t dummy_state_base,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t shard_count,
    uint32_t capacity,
    uint32_t shard_count_stride,
    uint32_t shard_count_request_stride,
    uint32_t miss_count_stride,
    uint32_t generation_stride,
    uint32_t block_table_width,
    uint32_t block_size);

  extern void dsa_resident_sorted_update_debug_impl(
    void* stream,
    void* topk_indices,
    void* shard_packed,
    void* shard_mapping,
    void* shard_counts,
    void* prior_slots,
    void* request_state_indices,
    void* request_state_generations,
    void* state_tokens,
    void* state_slots,
    void* state_counts,
    void* state_generations,
    uint32_t request_count,
    uint32_t state_row_count,
    uint32_t dummy_state_base,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t shard_count,
    uint32_t capacity,
    uint32_t shard_count_stride,
    uint32_t shard_count_request_stride,
    uint32_t generation_stride);

  extern void dsa_resident_sorted_remap_impl(
    void* stream,
    void* topk_indices,
    void* shard_mapping,
    void* shard_counts,
    void* prior_slots,
    uint32_t request_count,
    uint32_t rows_per_request,
    uint32_t row_width,
    uint32_t shard_count,
    uint32_t capacity,
    uint32_t shard_count_stride,
    uint32_t shard_count_request_stride);

  extern void dsa_resident_sorted_read_probe_impl(
    void* stream,
    void* shard_counts,
    void* prior_slots,
    void* debug_info,
    void* prior_readback,
    uint32_t request_count,
    uint32_t shard_count,
    uint32_t capacity,
    uint32_t shard_count_stride,
    uint32_t shard_count_request_stride);

  extern void dsa_resident_sorted_finalize_debug_impl(
    void* stream,
    void* shard_packed,
    void* shard_counts,
    void* prior_slots,
    void* shard_miss_tokens,
    void* shard_miss_positions,
    void* shard_evictable_slots,
    void* miss_tokens,
    void* miss_counts,
    void* target_slots,
    void* request_block_table,
    void* debug_info,
    uint32_t request_count,
    uint32_t shard_count,
    uint32_t capacity,
    uint32_t shard_count_stride,
    uint32_t shard_count_request_stride,
    uint32_t miss_count_stride,
    uint32_t block_table_width,
    uint32_t block_size,
    uint32_t debug_stage);

  extern void dsa_staged_unique_finalize_impl(
    void* stream,
    void* unique_keys,
    void* inverse,
    void* row_req_indices,
    void* selected_packed,
    void* local_to_union,
    void* selected_count,
    void* request_block_table,
    void* target_slots,
    uint32_t unique_count,
    uint32_t row_count,
    uint32_t row_width,
    uint32_t request_count,
    uint32_t scratch_capacity,
    uint32_t block_table_width,
    uint32_t selected_count_stride,
    uint32_t block_size,
    uint32_t block_size_shift,
    uint32_t packed_key_stride);

  extern void dsa_staged_copy_rows_impl(
    void* stream,
    void* output,
    void* local_indices,
    uint32_t row_count,
    uint32_t row_width);
    
  torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
    if (!tensor.is_privateuseone()) {
      throw std::runtime_error("Tensor must be on NPU device");
    }
    // Get the raw data pointer
    void* data_ptr = tensor.data_ptr();
    // Get tensor sizes and strides
    std::vector<int64_t> sizes = tensor.sizes().vec();
    std::vector<int64_t> strides = tensor.strides().vec();
    // Get tensor options (dtype, device)
    auto options = tensor.options();
    // Create a new tensor from the raw data pointer
    auto new_tensor = at_npu::native::from_blob(data_ptr, sizes, strides, options);
    return new_tensor;
  }

  extern void bgmv_shrink_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *indices,
        uint32_t indicesSize,
        void *y, 
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        float scale);

    extern void bgmv_expand_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *indices,
        uint32_t indicesSize,
        void *y,
        void *y_out,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t lora_rank,
        uint32_t output_hidden_dim,
        uint32_t slice_offset,
        uint32_t output_full_dim);

    extern void sgmv_shrink_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *loraIndices,
        uint32_t loraIndicesSize,
        void *seqLen,
        uint32_t seqLenSize,
        void *y,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        float scale);

    extern void sgmv_expand_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *loraIndices,
        uint32_t loraIndicesSize,
        void *seqLen,
        uint32_t seqLenSize,
        void *y,
        void *y_out,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t lora_rank,
        uint32_t output_hidden_dim,
        uint32_t slice_offset,
        uint32_t output_full_dim);

    extern void mla_preprocess_impl(
        void* stream,
        void* hidden_state,
        void* quant_scale1,
        void* quant_offset1,
        void* wdqkv,
        void* bias1,
        void* gamma2,
        void* beta2,
        void* quant_scale2,
        void* quant_offset2,
        void* gamma3,
        void* sin1,
        void* cos1,
        void* sin2,
        void* cos2,
        void* keycache,
        void* slot_mapping,
        void* wuq,
        void* bias2,
        void* wuk,
        void* descale1,
        void* descale2,
        void* ctkv_scale,
        void* qnope_scale,
        void* q,
        void* keycache_out,
        void* q2,
        void* keycache_out2,
        void* inner_out,
        void* workspace,
        void* tiling,
        const uint32_t block_dim
    );

    extern void batch_matmul_transpose_impl(
        void* stream,
        void* gm_a,
        void* gm_b,
        void* gm_c,
        void* gm_tiling_data,
        const uint32_t block_dim
    );
}
