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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <torch_npu/csrc/npu/Module.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "ops.h"
#include "utils.h"
#include "aclnn_torch_adapter/op_api_common.h"
#include "add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"
#include "apply_top_k_top_p_custom/apply_top_k_top_p_custom_torch_adpt.h"
#include "batch_matmul_transpose/batch_matmul_transpose_torch_adpt.h"
#include "dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h"
#include "dispatch_gmm_combine_decode/dispatch_gmm_combine_decode_torch_adpt.h"
#include "dispatch_layout/dispatch_layout_torch_adpt.h"
#include "grouped_matmul_swiglu_quant_weight_nz_tensor_list/grouped_matmul_swiglu_quant_torch_adpt.h"
#include "lightning_indexer_vllm/lightning_indexer_vllm_torch_adpt.h"
#include "matmul_allreduce_add_rmsnorm/matmul_allreduce_add_rmsnorm_torch_adpt.h"
#include "mla_preprocess/mla_preprocess_torch_adpt.h"
#include "moe_combine_normal/moe_combine_normal_torch_adpt.h"
#include "moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h"
#include "sparse_flash_attention/sparse_flash_attention_torch_adpt.h"
#include "lightning_indexer_quant/lightning_indexer_quant_torch_adpt.h"
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

namespace vllm_ascend {
void swap_blocks_impl(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping, aclrtStream stream)
{
    torch::Device src_device = src.device();
    torch::Device dst_device = dst.device();
    aclrtMemcpyKind memcpy_type;

    if ((!src_device.is_cpu()) && (!dst_device.is_cpu())) {
        TORCH_CHECK(src_device.index() == dst_device.index(),
                    "src and dst must be on the same npu");
        memcpy_type = ACL_MEMCPY_DEVICE_TO_DEVICE;
    } else if ((!src_device.is_cpu()) && dst_device.is_cpu()) {
        memcpy_type = ACL_MEMCPY_DEVICE_TO_HOST;
    } else if (src_device.is_cpu() && (!dst_device.is_cpu())) {
        memcpy_type = ACL_MEMCPY_HOST_TO_DEVICE;
    } else {
        TORCH_CHECK(false, "Invalid device combination, src tensor device: ", src_device, ", dst tensor device: ", dst_device);
    }

    TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

    char* src_ptr = static_cast<char*>(src.data_ptr());
    char* dst_ptr = static_cast<char*>(dst.data_ptr());

    const int64_t block_size_in_bytes = src.element_size() * src.stride(0);
    
    const int64_t num_blocks = block_mapping.size(0);
    const int64_t max_src_block = src.size(0);
    const int64_t max_dst_block = dst.size(0);
    for (size_t i = 0; i < num_blocks; i++) {
        int64_t src_block_number = block_mapping[i][0].item<int64_t>();
        int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
        TORCH_CHECK(src_block_number >= 0 && src_block_number <= max_src_block,
                    "src block index ", src_block_number, " out of range (max: ", max_src_block, ")");
        TORCH_CHECK(dst_block_number >= 0 && dst_block_number <= max_dst_block,
                    "dst block index ", dst_block_number, " out of range (max: ", max_dst_block, ")");
        
        int64_t src_offset = src_block_number * block_size_in_bytes;
        int64_t dst_offset = dst_block_number * block_size_in_bytes;

        aclrtMemcpyAsync(dst_ptr + dst_offset, block_size_in_bytes,
                         src_ptr + src_offset, block_size_in_bytes,
                         memcpy_type, stream);
    }
}

void swap_blocks(torch::Tensor &x, torch::Tensor &y, const torch::Tensor &z)
{    
  
    const c10_npu::OptionalNPUGuard npuGuard(
        (!x.device().is_cpu()) ? x.device() : y.device()
    );
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();                       
    swap_blocks_impl(x, y, z, stream);           
    return;
}

AscendType get_dtype_from_torch(at::ScalarType scalarType)
{
    if (scalarType == at::ScalarType::Float) {
        return AscendType::FP32;
    } else if (scalarType == at::ScalarType::BFloat16) {
        return AscendType::BF16;
    } else {
        return AscendType::FP16;
    }
}

std::tuple<at::Tensor, at::Tensor> get_masked_input_and_mask(
    at::Tensor &input,
    const int64_t org_vocab_start_index,
    const int64_t org_vocab_end_index,
    const int64_t num_org_vocab_padding,
    const int64_t added_vocab_start_index,
    const int64_t added_vocab_end_index)
    /*
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py#L161-L198
    Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    In this example, we will have the original vocab size = 1010,
    added vocab size = 16 and padding to 64. Therefore, the total
    vocab size with padding will be 1088 (because we first pad 1010 to
    1024, add 16, and then pad to 1088).
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1015 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1000 | ... | 1015 |  -1  | ... |  -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  520 | ... | 543 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 519  | 520 | ... |  543 |
    Parameters:
        org_vocab_start_index //base embeddings start
        org_vocab_end_index //base embeddings end
        num_org_vocab_padding //base embeddings padding
        added_vocab_start_index //LoRA embeddings start
        added_vocab_end_index //LoRA embeddings end
    */
{
    // Input validation
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(org_vocab_start_index >= 0, "org_vocab_start_index must be non-negative");
    TORCH_CHECK(org_vocab_end_index >= org_vocab_start_index, "org_vocab_end_index must be greater than org_vocab_start_index");
    TORCH_CHECK(num_org_vocab_padding >= 0, "num_org_vocab_padding must be non-negative");
    TORCH_CHECK(added_vocab_start_index >= org_vocab_end_index, "added_vocab_start_index must be greater than org_vocab_end_index");
    TORCH_CHECK(added_vocab_end_index >= added_vocab_start_index, "added_vocab_end_index must be greater than added_vocab_start_index");

    // Get total number of elements
    int64_t size = input.numel();

    // Create output tensors
    at::Tensor masked_input = at::empty_like(input);
	at::Tensor mask = at::empty_like(input).to(at::kBool);

    // Get data pointers
    void *input_ptr = input.data_ptr();
    void *masked_input_ptr = masked_input.data_ptr();
    void *mask_ptr = mask.data_ptr();

    // Get current stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    // Get scalar type
    at::ScalarType scalar_type = input.scalar_type();

    // Create and configure OpCommand
    at_npu::native::OpCommand cmd;
    cmd.Name("get_masked_input_and_mask");
    cmd.SetCustomHandler([scalar_type, size, stream,
                         input_ptr, masked_input_ptr, mask_ptr,
                         org_vocab_start_index, org_vocab_end_index,
                         num_org_vocab_padding, added_vocab_start_index,
                         added_vocab_end_index]() -> int {
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        uint32_t loop_cnt = (size + aiv_num - 1) / aiv_num;

        // Call implementation
        get_masked_input_and_mask_impl(
            stream,
            input_ptr,
            masked_input_ptr,
            mask_ptr,
            org_vocab_start_index,
            org_vocab_end_index,
            num_org_vocab_padding,
            added_vocab_start_index,
            added_vocab_end_index,
            size,
            loop_cnt,
            aiv_num);

        return 0;
    });
    cmd.Run();
    return {masked_input, mask};
}

at::Tensor npu_dsa_prepare_sparse_indices_legacy_(
    at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    const at::Tensor &valid_rows,
    const at::Tensor &scratch_base,
    bool need_packed,
    const c10::optional<at::Tensor> &row_req_indices,
    int64_t packed_key_stride)
{
    TORCH_CHECK(topk_indices.is_privateuseone(),
                "topk_indices must be on an NPU device");
    TORCH_CHECK(split_boundary.device() == topk_indices.device() &&
                    valid_rows.device() == topk_indices.device() &&
                    scratch_base.device() == topk_indices.device(),
                "all sparse-index preparation tensors must be on the same NPU device");
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt &&
                    split_boundary.scalar_type() == at::kInt &&
                    valid_rows.scalar_type() == at::kInt &&
                    scratch_base.scalar_type() == at::kInt,
                "sparse-index preparation only supports int32 tensors");
    TORCH_CHECK(topk_indices.is_contiguous() && split_boundary.is_contiguous() &&
                    valid_rows.is_contiguous() && scratch_base.is_contiguous(),
                "sparse-index preparation inputs must be contiguous");
    if (row_req_indices.has_value()) {
        TORCH_CHECK(row_req_indices->device() == topk_indices.device(),
                    "row_req_indices must be on the same NPU device");
        TORCH_CHECK(row_req_indices->scalar_type() == at::kInt,
                    "row_req_indices must be int32");
        TORCH_CHECK(row_req_indices->is_contiguous() &&
                        row_req_indices->dim() == 1,
                    "row_req_indices must be a contiguous 1D tensor");
    }
    TORCH_CHECK(topk_indices.dim() == 2 || topk_indices.dim() == 3,
                "topk_indices must have shape [rows, k] or [rows, 1, k]");
    TORCH_CHECK(topk_indices.dim() != 3 || topk_indices.size(1) == 1,
                "three-dimensional topk_indices must have shape [rows, 1, k]");
    TORCH_CHECK(split_boundary.dim() == 1 && valid_rows.dim() == 1 &&
                    scratch_base.dim() == 1,
                "split_boundary, valid_rows, and scratch_base must be one-dimensional");

    const int64_t row_count = topk_indices.size(0);
    TORCH_CHECK(row_count > 0, "topk_indices must contain at least one row");
    TORCH_CHECK(!row_req_indices.has_value() ||
                    row_req_indices->numel() >= row_count,
                "row_req_indices must cover every top-k row");
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t valid_row_count = valid_rows.numel();
    TORCH_CHECK(split_boundary.numel() >= row_count &&
                    scratch_base.numel() >= row_count,
                "split_boundary and scratch_base must cover every top-k row");
    TORCH_CHECK(valid_row_count <= row_count,
                "valid_rows cannot contain more entries than top-k rows");
    TORCH_CHECK(row_width > 0 && row_width <= 4096,
                "sparse-index preparation supports at most 4096 entries per row");
    TORCH_CHECK(row_width % 64 == 0,
                "sparse-index row width must be a multiple of 64 int32 values");
    TORCH_CHECK(packed_key_stride >= 0 &&
                    packed_key_stride <= std::numeric_limits<int32_t>::max(),
                "packed_key_stride must fit int32");
    TORCH_CHECK(packed_key_stride == 0 ||
                    (need_packed && row_req_indices.has_value()),
                "packed key encoding requires packed output and row requests");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(topk_indices.data_ptr()) % 256 == 0,
        "topk_indices must start at a 256-byte-aligned address so adjacent "
        "rows cannot share a write transaction/cacheline");

    at::Tensor selected_packed = at::empty(
        {need_packed ? valid_row_count : 0, row_width},
        topk_indices.options());
    if (need_packed) {
        TORCH_CHECK(
            reinterpret_cast<std::uintptr_t>(selected_packed.data_ptr()) % 256 == 0,
            "selected_packed must start at a 256-byte-aligned address");
    }
    const bool clear_invalid_rows = row_req_indices.has_value();
    if (valid_row_count == 0 && !clear_invalid_rows) {
        return selected_packed;
    }

    // valid_rows is produced from np.flatnonzero in the SFA metadata builder,
    // so every source row has exactly one owner. Keeping the check on host
    // would require an NPU-to-CPU synchronization; direct callers must retain
    // the same ordered-unique contract.
    const c10_npu::OptionalNPUGuard npu_guard(topk_indices.device());

    static thread_local int32_t cached_device = -1;
    static thread_local int64_t cached_aiv_count = 0;
    const int32_t current_device =
        static_cast<int32_t>(topk_indices.get_device());
    if (current_device != cached_device || cached_aiv_count <= 0) {
        TORCH_CHECK(
            aclGetDeviceCapability(
                current_device,
                ACL_DEVICE_INFO_VECTOR_CORE_NUM,
                &cached_aiv_count) == ACL_SUCCESS,
            "failed to query the NPU vector core count");
        cached_device = current_device;
    }
    TORCH_CHECK(cached_aiv_count > 0,
                "NPU reported no available vector cores");

    // Keep enough work on each core for narrow rows while using one core per
    // row for the common k=2048 path. A row is never split across cores.
    constexpr int64_t target_elements_per_core = 2048;
    const int64_t rows_per_core = std::max<int64_t>(
        1, target_elements_per_core / row_width);
    const int64_t work_row_count =
        clear_invalid_rows ? row_count : valid_row_count;
    const int64_t requested_cores =
        (work_row_count + rows_per_core - 1) / rows_per_core;
    const uint32_t core_count = static_cast<uint32_t>(
        std::max<int64_t>(1, std::min(cached_aiv_count, requested_cores)));

    void *topk_ptr = topk_indices.data_ptr();
    void *split_boundary_ptr = split_boundary.data_ptr();
    void *valid_rows_ptr = valid_rows.data_ptr();
    void *scratch_base_ptr = scratch_base.data_ptr();
    void *selected_packed_ptr = selected_packed.data_ptr();
    void *row_req_indices_ptr = clear_invalid_rows
        ? row_req_indices->data_ptr()
        : split_boundary.data_ptr();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_prepare_sparse_indices_legacy_");
    cmd.SetCustomHandler([
        stream,
        topk_ptr,
        split_boundary_ptr,
        valid_rows_ptr,
        scratch_base_ptr,
        selected_packed_ptr,
        row_req_indices_ptr,
        row_count,
        row_width,
        valid_row_count,
        core_count,
        need_packed,
        clear_invalid_rows,
        packed_key_stride]() -> int {
        dsa_prepare_sparse_indices_legacy_impl(
            stream,
            topk_ptr,
            split_boundary_ptr,
            valid_rows_ptr,
            scratch_base_ptr,
            selected_packed_ptr,
            row_req_indices_ptr,
            static_cast<uint32_t>(row_count),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(valid_row_count),
            core_count,
            need_packed,
            clear_invalid_rows,
            static_cast<uint32_t>(packed_key_stride));
        return 0;
    });
    cmd.Run();
    return selected_packed;
}

at::Tensor npu_dsa_staged_union_(
    const at::Tensor &row_packed,
    at::Tensor &selected_packed,
    at::Tensor &local_to_union,
    at::Tensor &selected_count,
    const at::Tensor &request_block_table,
    at::Tensor &target_slots,
    int64_t block_size,
    int64_t max_tokens,
    bool use_sort)
{
    TORCH_CHECK(row_packed.is_privateuseone(),
                "row_packed must be on an NPU device");
    const auto device = row_packed.device();
    TORCH_CHECK(selected_packed.device() == device &&
                    local_to_union.device() == device &&
                    selected_count.device() == device &&
                    request_block_table.device() == device &&
                    target_slots.device() == device,
                "all staged-union tensors must share one NPU device");
    TORCH_CHECK(row_packed.scalar_type() == at::kInt &&
                    selected_packed.scalar_type() == at::kInt &&
                    local_to_union.scalar_type() == at::kInt &&
                    selected_count.scalar_type() == at::kInt &&
                    request_block_table.scalar_type() == at::kInt &&
                    target_slots.scalar_type() == at::kLong,
                "staged-union indices must be int32 and slots int64");
    TORCH_CHECK(row_packed.is_contiguous() &&
                    selected_packed.is_contiguous() &&
                    local_to_union.is_contiguous() &&
                    selected_count.is_contiguous() &&
                    request_block_table.is_contiguous() &&
                    target_slots.is_contiguous(),
                "all staged-union tensors must be contiguous");
    TORCH_CHECK(row_packed.dim() == 2 &&
                    request_block_table.dim() == 2 &&
                    request_block_table.size(0) > 0,
                "benchmark staged union requires [rows,k] and requests");
    const int64_t row_count = row_packed.size(0);
    const int64_t row_width = row_packed.size(1);
    const int64_t request_count = row_count / 2;
    const int64_t total = row_count * row_width;
    const int64_t selected_count_stride =
        selected_count.numel() / request_count;
    TORCH_CHECK(row_count > 0 && row_count % 2 == 0 &&
                    row_width == 2048 &&
                    request_block_table.size(0) == request_count,
                "benchmark staged union requires two [2048] rows per request");
    TORCH_CHECK(selected_packed.numel() >= total &&
                    local_to_union.numel() >= total &&
                    selected_count.numel() >= request_count &&
                    target_slots.numel() >= total,
                "staged-union output buffers are too small");
    TORCH_CHECK(request_count == 1 || selected_count_stride >= 8,
                "batched staged union requires each selected-count row to "
                "occupy at least one 32-byte transaction");
    TORCH_CHECK(block_size > 0 &&
                    (block_size & (block_size - 1)) == 0 &&
                    request_block_table.size(1) * block_size >=
                        2 * row_width,
                "block_size must be a positive power of two and the request "
                "block table must cover both rows");
    TORCH_CHECK(max_tokens > 0 && max_tokens % 32 == 0,
                "max_tokens must be a positive multiple of 32");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* row_ptr = row_packed.data_ptr();
    void* packed_ptr = selected_packed.data_ptr();
    void* map_ptr = local_to_union.data_ptr();
    void* count_ptr = selected_count.data_ptr();
    void* table_ptr = request_block_table.data_ptr();
    void* slots_ptr = target_slots.data_ptr();
    const int64_t table_width = request_block_table.size(1);
    at_npu::native::OpCommand cmd;
    cmd.Name(use_sort ? "npu_dsa_staged_sort_union_"
                      : "npu_dsa_staged_hash_union_");
    cmd.SetCustomHandler([
        stream, row_ptr, packed_ptr, map_ptr, count_ptr, table_ptr,
        slots_ptr, row_count, row_width, max_tokens, table_width,
        selected_count_stride, block_size, use_sort]() -> int {
        if (use_sort) {
            dsa_staged_sort_union_impl(
                stream, row_ptr, packed_ptr, map_ptr, count_ptr,
                table_ptr, slots_ptr,
                static_cast<uint32_t>(row_count),
                static_cast<uint32_t>(row_width),
                static_cast<uint32_t>(table_width),
                static_cast<uint32_t>(selected_count_stride),
                static_cast<uint32_t>(block_size));
        } else {
            dsa_staged_hash_union_impl(
                stream, row_ptr, packed_ptr, map_ptr, count_ptr,
                table_ptr, slots_ptr,
                static_cast<uint32_t>(row_count),
                static_cast<uint32_t>(row_width),
                static_cast<uint32_t>(max_tokens),
                static_cast<uint32_t>(table_width),
                static_cast<uint32_t>(selected_count_stride),
                static_cast<uint32_t>(block_size));
        }
        return 0;
    });
    cmd.Run();
    return selected_count;
}

at::Tensor npu_dsa_staged_sharded_union_(
    at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    at::Tensor &selected_packed,
    at::Tensor &local_to_union,
    at::Tensor &selected_count,
    const at::Tensor &request_block_table,
    at::Tensor &target_slots,
    at::Tensor &shard_packed,
    at::Tensor &shard_mapping,
    at::Tensor &shard_counts,
    int64_t block_size)
{
    TORCH_CHECK(topk_indices.is_privateuseone(),
                "topk_indices must be on an NPU device");
    const auto device = topk_indices.device();
    TORCH_CHECK(split_boundary.device() == device &&
                    selected_packed.device() == device &&
                    local_to_union.device() == device &&
                    selected_count.device() == device &&
                    request_block_table.device() == device &&
                    target_slots.device() == device &&
                    shard_packed.device() == device &&
                    shard_mapping.device() == device &&
                    shard_counts.device() == device,
                "all sharded-union tensors must share one NPU device");
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt &&
                    split_boundary.scalar_type() == at::kInt &&
                    selected_packed.scalar_type() == at::kInt &&
                    local_to_union.scalar_type() == at::kInt &&
                    selected_count.scalar_type() == at::kInt &&
                    request_block_table.scalar_type() == at::kInt &&
                    target_slots.scalar_type() == at::kLong &&
                    shard_packed.scalar_type() == at::kInt &&
                    shard_mapping.scalar_type() == at::kInt &&
                    shard_counts.scalar_type() == at::kInt,
                "sharded-union indices must be int32 and slots int64");
    TORCH_CHECK(topk_indices.is_contiguous() &&
                    split_boundary.is_contiguous() &&
                    selected_packed.is_contiguous() &&
                    local_to_union.is_contiguous() &&
                    selected_count.is_contiguous() &&
                    request_block_table.is_contiguous() &&
                    target_slots.is_contiguous() &&
                    shard_packed.is_contiguous() &&
                    shard_mapping.is_contiguous() &&
                    shard_counts.is_contiguous(),
                "all sharded-union tensors must be contiguous");
    TORCH_CHECK((topk_indices.dim() == 2 ||
                    (topk_indices.dim() == 3 &&
                     topk_indices.size(1) == 1)) &&
                    split_boundary.dim() == 1 &&
                    selected_packed.dim() == 2 &&
                    local_to_union.dim() == 2 &&
                    selected_count.dim() == 2 &&
                    request_block_table.dim() == 2 &&
                    target_slots.dim() == 2 &&
                    shard_packed.dim() == 3 &&
                    shard_mapping.dim() == 3 &&
                    shard_counts.dim() == 3,
                "invalid sharded-union tensor ranks");
    const int64_t row_count = topk_indices.size(0);
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t request_count = selected_packed.size(0);
    TORCH_CHECK(row_count > 0 && request_count > 0 &&
                    row_count % request_count == 0,
                "sharded union requires a uniform MTP depth");
    const int64_t rows_per_request = row_count / request_count;
    TORCH_CHECK(rows_per_request > 0 && rows_per_request <= 8,
                "benchmark sharded union supports MTP depths from 1 to 8");
    int64_t shard_count = 1;
    while (shard_count < rows_per_request) {
        shard_count <<= 1;
    }
    const int64_t request_width = rows_per_request * row_width;
    const int64_t selected_count_stride =
        selected_count.numel() / request_count;
    const int64_t shard_count_stride =
        shard_counts.numel() / (shard_count * request_count);
    TORCH_CHECK(row_width == 2048 &&
                    split_boundary.numel() >= row_count &&
                    request_block_table.size(0) == request_count,
                "benchmark sharded union requires [2048] MTP rows");
    TORCH_CHECK(selected_packed.numel() >= row_count * row_width &&
                    local_to_union.numel() >= row_count * row_width &&
                    selected_count.numel() >= request_count &&
                    target_slots.numel() >= row_count * row_width,
                "sharded-union output buffers are too small");
    TORCH_CHECK(shard_packed.size(0) == request_count &&
                    shard_packed.size(1) == shard_count &&
                    shard_packed.size(2) == row_width &&
                    shard_mapping.size(0) == request_count &&
                    shard_mapping.size(1) == shard_count &&
                    shard_mapping.size(2) == request_width &&
                    shard_counts.size(0) == request_count &&
                    shard_counts.size(1) == shard_count &&
                    shard_count_stride >= 16,
                "invalid sharded-union scratch buffer shapes");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(topk_indices.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(selected_packed.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(local_to_union.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(target_slots.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(shard_packed.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(shard_mapping.data_ptr()) % 256 == 0,
        "sharded-union vector outputs must start at a 256-byte-aligned "
        "address");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(selected_count.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(shard_counts.data_ptr()) % 64 == 0,
        "sharded-union scalar-count outputs must start at a 64-byte-aligned "
        "address");
    TORCH_CHECK(request_count == 1 || selected_count_stride >= 16,
                "batched sharded union requires one 64-byte cacheline per "
                "selected-count row");
    TORCH_CHECK(block_size > 0 &&
                    (block_size & (block_size - 1)) == 0 &&
                    request_block_table.size(1) * block_size >=
                        request_width,
                "block table must cover the sharded union output");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* boundary_ptr = split_boundary.data_ptr();
    void* packed_ptr = selected_packed.data_ptr();
    void* map_ptr = local_to_union.data_ptr();
    void* count_ptr = selected_count.data_ptr();
    void* table_ptr = request_block_table.data_ptr();
    void* slots_ptr = target_slots.data_ptr();
    void* shard_packed_ptr = shard_packed.data_ptr();
    void* shard_mapping_ptr = shard_mapping.data_ptr();
    void* shard_counts_ptr = shard_counts.data_ptr();
    const int64_t table_width = request_block_table.size(1);
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_staged_sharded_union_");
    cmd.SetCustomHandler([
        stream, topk_ptr, boundary_ptr, packed_ptr, map_ptr, count_ptr, table_ptr,
        slots_ptr, shard_packed_ptr, shard_mapping_ptr, shard_counts_ptr,
        request_count, rows_per_request, row_width, shard_count,
        table_width, selected_count_stride, shard_count_stride,
        block_size]() -> int {
        dsa_staged_sharded_sort_union_impl(
            stream, topk_ptr, boundary_ptr, packed_ptr, map_ptr, count_ptr, table_ptr,
            slots_ptr, shard_packed_ptr, shard_mapping_ptr,
            shard_counts_ptr, static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(rows_per_request),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(shard_count),
            static_cast<uint32_t>(table_width),
            static_cast<uint32_t>(selected_count_stride),
            static_cast<uint32_t>(shard_count_stride),
            static_cast<uint32_t>(block_size));
        return 0;
    });
    cmd.Run();
    return selected_count;
}

at::Tensor npu_dsa_staged_sharded_vector_union_(
    at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    at::Tensor &selected_packed,
    at::Tensor &local_to_union,
    at::Tensor &selected_count,
    const at::Tensor &request_block_table,
    at::Tensor &target_slots,
    at::Tensor &shard_packed,
    at::Tensor &shard_mapping,
    at::Tensor &shard_counts,
    at::Tensor &shard_pairs,
    int64_t block_size)
{
    TORCH_CHECK(topk_indices.is_privateuseone(),
                "topk_indices must be on an NPU device");
    const auto device = topk_indices.device();
    TORCH_CHECK(split_boundary.device() == device &&
                    selected_packed.device() == device &&
                    local_to_union.device() == device &&
                    selected_count.device() == device &&
                    request_block_table.device() == device &&
                    target_slots.device() == device &&
                    shard_packed.device() == device &&
                    shard_mapping.device() == device &&
                    shard_counts.device() == device &&
                    shard_pairs.device() == device,
                "all vector sharded-union tensors must share one NPU device");
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt &&
                    split_boundary.scalar_type() == at::kInt &&
                    selected_packed.scalar_type() == at::kInt &&
                    local_to_union.scalar_type() == at::kInt &&
                    selected_count.scalar_type() == at::kInt &&
                    request_block_table.scalar_type() == at::kInt &&
                    target_slots.scalar_type() == at::kLong &&
                    shard_packed.scalar_type() == at::kInt &&
                    shard_mapping.scalar_type() == at::kInt &&
                    shard_counts.scalar_type() == at::kInt &&
                    shard_pairs.scalar_type() == at::kInt,
                "vector sharded-union indices must be int32 and slots int64");
    TORCH_CHECK(topk_indices.is_contiguous() &&
                    split_boundary.is_contiguous() &&
                    selected_packed.is_contiguous() &&
                    local_to_union.is_contiguous() &&
                    selected_count.is_contiguous() &&
                    request_block_table.is_contiguous() &&
                    target_slots.is_contiguous() &&
                    shard_packed.is_contiguous() &&
                    shard_mapping.is_contiguous() &&
                    shard_counts.is_contiguous() &&
                    shard_pairs.is_contiguous(),
                "all vector sharded-union tensors must be contiguous");
    TORCH_CHECK((topk_indices.dim() == 2 ||
                    (topk_indices.dim() == 3 &&
                     topk_indices.size(1) == 1)) &&
                    split_boundary.dim() == 1 &&
                    selected_packed.dim() == 2 &&
                    local_to_union.dim() == 2 &&
                    selected_count.dim() == 2 &&
                    request_block_table.dim() == 2 &&
                    target_slots.dim() == 2 &&
                    shard_packed.dim() == 3 &&
                    shard_mapping.dim() == 3 &&
                    shard_counts.dim() == 3 &&
                    shard_pairs.dim() == 3,
                "invalid vector sharded-union tensor ranks");
    const int64_t row_count = topk_indices.size(0);
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t request_count = selected_packed.size(0);
    TORCH_CHECK(row_count > 0 && request_count > 0 &&
                    row_count % request_count == 0,
                "vector sharded union requires a uniform MTP depth");
    const int64_t rows_per_request = row_count / request_count;
    TORCH_CHECK(rows_per_request > 0 && rows_per_request <= 8,
                "vector sharded union supports MTP depths from 1 to 8");
    int64_t shard_count = 1;
    while (shard_count < rows_per_request) {
        shard_count <<= 1;
    }
    const int64_t request_width = rows_per_request * row_width;
    const int64_t mapping_parts = shard_count;
    const int64_t mapping_part_width = request_width / mapping_parts;
    const int64_t selected_count_stride =
        selected_count.numel() / request_count;
    const int64_t shard_count_stride =
        shard_counts.numel() / (shard_count * request_count);
    TORCH_CHECK(row_width == 2048 &&
                    split_boundary.numel() >= row_count &&
                    request_block_table.size(0) == request_count,
                "vector sharded union requires [2048] MTP rows");
    TORCH_CHECK(selected_packed.numel() >= row_count * row_width &&
                    local_to_union.numel() >= row_count * row_width &&
                    selected_count.numel() >= request_count &&
                    target_slots.numel() >= row_count * row_width,
                "vector sharded-union output buffers are too small");
    TORCH_CHECK(shard_packed.size(0) == request_count &&
                    shard_packed.size(1) == shard_count &&
                    shard_packed.size(2) == row_width &&
                    shard_mapping.size(0) == request_count &&
                    shard_mapping.size(1) == shard_count &&
                    shard_mapping.size(2) == request_width &&
                    shard_counts.size(0) == request_count &&
                    shard_counts.size(1) == shard_count &&
                    shard_count_stride >= 16 &&
                    shard_pairs.size(0) == request_count &&
                    shard_pairs.size(1) == shard_count &&
                    shard_pairs.size(2) == 2 * row_width,
                "invalid vector sharded-union scratch buffer shapes");
    TORCH_CHECK(mapping_part_width <= row_width &&
                    mapping_part_width % 64 == 0,
                "parallel mapping parts must own complete 256-byte "
                "destination transactions");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(topk_indices.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(selected_packed.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(local_to_union.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(target_slots.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(shard_packed.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(shard_mapping.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(shard_pairs.data_ptr()) % 256 == 0,
        "vector sharded-union vector outputs must start at a 256-byte-aligned "
        "address");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(selected_count.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(shard_counts.data_ptr()) % 64 == 0,
        "vector sharded-union scalar counts must start at a 64-byte-aligned "
        "address");
    TORCH_CHECK(request_count == 1 || selected_count_stride >= 16,
                "batched vector sharded union requires one 64-byte cacheline "
                "per selected-count row");
    TORCH_CHECK(block_size > 0 &&
                    (block_size & (block_size - 1)) == 0 &&
                    request_block_table.size(1) * block_size >=
                        request_width,
                "block table must cover the vector sharded union output");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* boundary_ptr = split_boundary.data_ptr();
    void* packed_ptr = selected_packed.data_ptr();
    void* map_ptr = local_to_union.data_ptr();
    void* count_ptr = selected_count.data_ptr();
    void* table_ptr = request_block_table.data_ptr();
    void* slots_ptr = target_slots.data_ptr();
    void* shard_packed_ptr = shard_packed.data_ptr();
    void* shard_mapping_ptr = shard_mapping.data_ptr();
    void* shard_counts_ptr = shard_counts.data_ptr();
    void* shard_pairs_ptr = shard_pairs.data_ptr();
    const int64_t table_width = request_block_table.size(1);
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_staged_sharded_vector_union_");
    cmd.SetCustomHandler([
        stream, topk_ptr, boundary_ptr, packed_ptr, map_ptr, count_ptr,
        table_ptr, slots_ptr, shard_packed_ptr, shard_mapping_ptr,
        shard_counts_ptr, shard_pairs_ptr, request_count,
        rows_per_request, row_width, shard_count, table_width,
        selected_count_stride, shard_count_stride,
        block_size]() -> int {
        dsa_staged_sharded_vector_union_impl(
            stream, topk_ptr, boundary_ptr, packed_ptr, map_ptr,
            count_ptr, table_ptr, slots_ptr, shard_packed_ptr,
            shard_mapping_ptr, shard_counts_ptr, shard_pairs_ptr,
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(rows_per_request),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(shard_count),
            static_cast<uint32_t>(table_width),
            static_cast<uint32_t>(selected_count_stride),
            static_cast<uint32_t>(shard_count_stride),
            static_cast<uint32_t>(block_size));
        return 0;
    });
    cmd.Run();
    return selected_count;
}

at::Tensor npu_dsa_staged_sharded_vector_dedup_(
    at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    at::Tensor &selected_packed,
    at::Tensor &local_to_union,
    at::Tensor &selected_count,
    const at::Tensor &request_block_table,
    at::Tensor &target_slots,
    at::Tensor &shard_packed,
    at::Tensor &shard_mapping,
    at::Tensor &shard_counts,
    int64_t block_size)
{
    TORCH_CHECK(topk_indices.is_privateuseone(),
                "topk_indices must be on an NPU device");
    const auto device = topk_indices.device();
    TORCH_CHECK(split_boundary.device() == device &&
                    selected_packed.device() == device &&
                    local_to_union.device() == device &&
                    selected_count.device() == device &&
                    request_block_table.device() == device &&
                    target_slots.device() == device &&
                    shard_packed.device() == device &&
                    shard_mapping.device() == device &&
                    shard_counts.device() == device,
                "all vector-dedup tensors must share one NPU device");
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt &&
                    split_boundary.scalar_type() == at::kInt &&
                    selected_packed.scalar_type() == at::kInt &&
                    local_to_union.scalar_type() == at::kInt &&
                    selected_count.scalar_type() == at::kInt &&
                    request_block_table.scalar_type() == at::kInt &&
                    target_slots.scalar_type() == at::kLong &&
                    shard_packed.scalar_type() == at::kInt &&
                    shard_mapping.scalar_type() == at::kInt &&
                    shard_counts.scalar_type() == at::kInt,
                "vector-dedup indices must be int32 and slots int64");
    TORCH_CHECK(topk_indices.is_contiguous() &&
                    split_boundary.is_contiguous() &&
                    selected_packed.is_contiguous() &&
                    local_to_union.is_contiguous() &&
                    selected_count.is_contiguous() &&
                    request_block_table.is_contiguous() &&
                    target_slots.is_contiguous() &&
                    shard_packed.is_contiguous() &&
                    shard_mapping.is_contiguous() &&
                    shard_counts.is_contiguous(),
                "all vector-dedup tensors must be contiguous");
    TORCH_CHECK((topk_indices.dim() == 2 ||
                    (topk_indices.dim() == 3 &&
                     topk_indices.size(1) == 1)) &&
                    split_boundary.dim() == 1 &&
                    selected_packed.dim() == 2 &&
                    local_to_union.dim() == 2 &&
                    selected_count.dim() == 2 &&
                    request_block_table.dim() == 2 &&
                    target_slots.dim() == 2 &&
                    shard_packed.dim() == 3 &&
                    shard_mapping.dim() == 3 &&
                    shard_counts.dim() == 3,
                "invalid vector-dedup tensor ranks");
    const int64_t row_count = topk_indices.size(0);
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t request_count = selected_packed.size(0);
    TORCH_CHECK(row_count > 0 && request_count > 0 &&
                    row_count % request_count == 0,
                "vector dedup requires a uniform MTP depth");
    const int64_t rows_per_request = row_count / request_count;
    TORCH_CHECK(rows_per_request > 0 && rows_per_request <= 8,
                "vector dedup supports MTP depths from 1 to 8");
    int64_t shard_count = 1;
    while (shard_count < rows_per_request) {
        shard_count <<= 1;
    }
    const int64_t request_width = rows_per_request * row_width;
    const int64_t mapping_part_width = request_width / shard_count;
    const int64_t selected_count_stride =
        selected_count.numel() / request_count;
    const int64_t shard_count_stride =
        shard_counts.numel() / (shard_count * request_count);
    TORCH_CHECK(row_width == 2048 &&
                    split_boundary.numel() >= row_count &&
                    request_block_table.size(0) == request_count,
                "vector dedup requires [2048] MTP rows");
    TORCH_CHECK(selected_packed.numel() >= row_count * row_width &&
                    local_to_union.numel() >= row_count * row_width &&
                    selected_count.numel() >= request_count &&
                    target_slots.numel() >= row_count * row_width,
                "vector-dedup output buffers are too small");
    TORCH_CHECK(shard_packed.size(0) == request_count &&
                    shard_packed.size(1) == shard_count &&
                    shard_packed.size(2) == row_width &&
                    shard_mapping.size(0) == request_count &&
                    shard_mapping.size(1) == shard_count &&
                    shard_mapping.size(2) == request_width &&
                    shard_counts.size(0) == request_count &&
                    shard_counts.size(1) == shard_count &&
                    shard_count_stride >= 16,
                "invalid vector-dedup scratch buffer shapes");
    TORCH_CHECK(mapping_part_width <= row_width &&
                    mapping_part_width % 64 == 0,
                "position map parts must own complete 256-byte "
                "destination transactions");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(topk_indices.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(selected_packed.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(local_to_union.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(target_slots.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(shard_packed.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(shard_mapping.data_ptr()) % 256 == 0,
        "vector-dedup vector outputs must start at a 256-byte-aligned "
        "address");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(selected_count.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(shard_counts.data_ptr()) % 64 == 0,
        "vector-dedup scalar outputs must start at a 64-byte-aligned address");
    TORCH_CHECK(selected_count_stride >= 16,
                "vector dedup reserves one count cacheline per request");
    TORCH_CHECK(block_size > 0 &&
                    (block_size & (block_size - 1)) == 0 &&
                    request_block_table.size(1) * block_size >=
                        request_width,
                "block table must cover the vector-dedup output");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* boundary_ptr = split_boundary.data_ptr();
    void* packed_ptr = selected_packed.data_ptr();
    void* map_ptr = local_to_union.data_ptr();
    void* count_ptr = selected_count.data_ptr();
    void* table_ptr = request_block_table.data_ptr();
    void* slots_ptr = target_slots.data_ptr();
    void* shard_packed_ptr = shard_packed.data_ptr();
    void* shard_mapping_ptr = shard_mapping.data_ptr();
    void* shard_counts_ptr = shard_counts.data_ptr();
    const int64_t table_width = request_block_table.size(1);
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_staged_sharded_vector_dedup_");
    cmd.SetCustomHandler([
        stream, topk_ptr, boundary_ptr, packed_ptr, map_ptr, count_ptr,
        table_ptr, slots_ptr, shard_packed_ptr, shard_mapping_ptr,
        shard_counts_ptr, request_count, rows_per_request, row_width,
        shard_count, table_width, selected_count_stride,
        shard_count_stride, block_size]() -> int {
        dsa_staged_sharded_vector_dedup_impl(
            stream, topk_ptr, boundary_ptr, packed_ptr, map_ptr,
            count_ptr, table_ptr, slots_ptr, shard_packed_ptr,
            shard_mapping_ptr, shard_counts_ptr,
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(rows_per_request),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(shard_count),
            static_cast<uint32_t>(table_width),
            static_cast<uint32_t>(selected_count_stride),
            static_cast<uint32_t>(shard_count_stride),
            static_cast<uint32_t>(block_size));
        return 0;
    });
    cmd.Run();
    return selected_count;
}

at::Tensor npu_dsa_staged_remap_rows_(
    at::Tensor &local_indices,
    const at::Tensor &local_to_union)
{
    TORCH_CHECK(local_indices.is_privateuseone() &&
                    local_to_union.device() == local_indices.device(),
                "staged remap tensors must share one NPU device");
    TORCH_CHECK(local_indices.scalar_type() == at::kInt &&
                    local_to_union.scalar_type() == at::kInt &&
                    local_indices.is_contiguous() &&
                    local_to_union.is_contiguous(),
                "staged remap tensors must be contiguous int32");
    TORCH_CHECK(local_indices.dim() == 2 ||
                    (local_indices.dim() == 3 &&
                     local_indices.size(1) == 1),
                "local_indices must be [rows,k] or [rows,1,k]");
    const int64_t rows = local_indices.size(0);
    const int64_t width = local_indices.numel() / rows;
    TORCH_CHECK(local_to_union.numel() >= rows * width,
                "local_to_union is too small");
    const c10_npu::OptionalNPUGuard npu_guard(local_indices.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* indices_ptr = local_indices.data_ptr();
    void* map_ptr = local_to_union.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_staged_remap_rows_");
    cmd.SetCustomHandler([
        stream, indices_ptr, map_ptr, rows, width]() -> int {
        dsa_staged_remap_rows_impl(
            stream, indices_ptr, map_ptr,
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(width));
        return 0;
    });
    cmd.Run();
    return local_indices;
}

at::Tensor npu_dsa_resident_remap_rows_(
    at::Tensor &topk_indices,
    const at::Tensor &position_to_union,
    const at::Tensor &union_to_slot)
{
    TORCH_CHECK(topk_indices.is_privateuseone() &&
                    position_to_union.device() == topk_indices.device() &&
                    union_to_slot.device() == topk_indices.device(),
                "resident remap tensors must share one NPU device");
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt &&
                    position_to_union.scalar_type() == at::kInt &&
                    union_to_slot.scalar_type() == at::kInt &&
                    topk_indices.is_contiguous() &&
                    position_to_union.is_contiguous() &&
                    union_to_slot.is_contiguous(),
                "resident remap tensors must be contiguous int32");
    TORCH_CHECK(topk_indices.dim() == 2 ||
                    (topk_indices.dim() == 3 &&
                     topk_indices.size(1) == 1),
                "topk_indices must be [rows,k] or [rows,1,k]");
    TORCH_CHECK(position_to_union.dim() == 2 &&
                    union_to_slot.dim() == 2,
                "resident remap maps must be rank two");
    const int64_t row_count = topk_indices.size(0);
    const int64_t request_count = union_to_slot.size(0);
    TORCH_CHECK(request_count > 0 && row_count % request_count == 0,
                "resident remap rows must be request-major");
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t rows_per_request = row_count / request_count;
    const int64_t scratch_capacity = union_to_slot.size(1);
    TORCH_CHECK(
        scratch_capacity == rows_per_request * row_width &&
            position_to_union.numel() ==
                request_count * scratch_capacity,
        "resident remap shapes do not match MTP * topk capacity");
    const c10_npu::OptionalNPUGuard npu_guard(topk_indices.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* position_ptr = position_to_union.data_ptr();
    void* slots_ptr = union_to_slot.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_resident_remap_rows_");
    cmd.SetCustomHandler([
        stream, topk_ptr, position_ptr, slots_ptr, row_count,
        row_width, rows_per_request, scratch_capacity]() -> int {
        dsa_resident_remap_rows_impl(
            stream, topk_ptr, position_ptr, slots_ptr,
            static_cast<uint32_t>(row_count),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(rows_per_request),
            static_cast<uint32_t>(scratch_capacity));
        return 0;
    });
    cmd.Run();
    return topk_indices;
}

at::Tensor npu_dsa_resident_lookup_rows_(
    const at::Tensor &selected_packed,
    const at::Tensor &selected_count,
    const at::Tensor &request_state_indices,
    at::Tensor &lookup_indices,
    int64_t token_stride,
    int64_t dummy_state_base)
{
    TORCH_CHECK(
        selected_packed.is_privateuseone() &&
            selected_count.device() == selected_packed.device() &&
            request_state_indices.device() == selected_packed.device() &&
            lookup_indices.device() == selected_packed.device(),
        "resident lookup tensors must share one NPU device");
    TORCH_CHECK(
        selected_packed.scalar_type() == at::kInt &&
            selected_count.scalar_type() == at::kInt &&
            request_state_indices.scalar_type() == at::kInt &&
            lookup_indices.scalar_type() == at::kLong,
        "resident lookup expects int32 inputs and int64 indices");
    TORCH_CHECK(
        selected_packed.is_contiguous() &&
            selected_count.is_contiguous() &&
            request_state_indices.is_contiguous() &&
            lookup_indices.is_contiguous(),
        "resident lookup tensors must be contiguous");
    TORCH_CHECK(
        selected_packed.dim() == 2 &&
            (selected_count.dim() == 1 || selected_count.dim() == 2) &&
            request_state_indices.dim() == 1 &&
            lookup_indices.sizes() == selected_packed.sizes(),
        "resident lookup tensor shapes do not match");
    const int64_t request_count = selected_packed.size(0);
    const int64_t scratch_capacity = selected_packed.size(1);
    const int64_t count_stride =
        selected_count.dim() == 1 ? 1 : selected_count.size(1);
    TORCH_CHECK(
        request_count > 0 && scratch_capacity > 0 &&
            scratch_capacity <= 4096 &&
            scratch_capacity % 256 == 0 &&
            count_stride > 0 &&
            selected_count.size(0) >= request_count &&
            request_state_indices.numel() >= request_count,
        "resident lookup metadata is too small");
    TORCH_CHECK(
        token_stride > scratch_capacity &&
            token_stride % 32 == 0 &&
            dummy_state_base >= request_count,
        "resident lookup stride or dummy-state base is invalid");
    TORCH_CHECK(
        2 * dummy_state_base * token_stride <=
            static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "resident flattened lookup indices exceed int32");

    const c10_npu::OptionalNPUGuard npu_guard(selected_packed.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* selected_ptr = selected_packed.data_ptr();
    void* count_ptr = selected_count.data_ptr();
    void* state_ptr = request_state_indices.data_ptr();
    void* index_ptr = lookup_indices.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_resident_lookup_rows_");
    cmd.SetCustomHandler([
        stream, selected_ptr, count_ptr, state_ptr, index_ptr,
        request_count, scratch_capacity, count_stride, token_stride,
        dummy_state_base]() -> int {
        dsa_resident_lookup_rows_impl(
            stream, selected_ptr, count_ptr, state_ptr, index_ptr,
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(scratch_capacity),
            static_cast<uint32_t>(count_stride),
            static_cast<uint32_t>(token_stride),
            static_cast<uint32_t>(dummy_state_base));
        return 0;
    });
    cmd.Run();
    return lookup_indices;
}

at::Tensor npu_dsa_resident_finalize_rows_(
    at::Tensor &topk_indices,
    const at::Tensor &position_to_union,
    at::Tensor &selected_packed,
    at::Tensor &selected_count,
    at::Tensor &target_slots,
    const at::Tensor &request_block_table,
    const at::Tensor &request_state_indices,
    const at::Tensor &request_state_generations,
    const at::Tensor &old_slots,
    at::Tensor &slot_to_token,
    at::Tensor &state_generations,
    at::Tensor &union_to_slot,
    at::Tensor &reverse_indices,
    at::Tensor &reverse_values,
    int64_t token_stride,
    int64_t block_size)
{
    const auto device = selected_packed.device();
    TORCH_CHECK(
        selected_packed.is_privateuseone() &&
            topk_indices.device() == device &&
            position_to_union.device() == device &&
            selected_count.device() == device &&
            target_slots.device() == device &&
            request_block_table.device() == device &&
            request_state_indices.device() == device &&
            request_state_generations.device() == device &&
            old_slots.device() == device &&
            slot_to_token.device() == device &&
            state_generations.device() == device &&
            union_to_slot.device() == device &&
            reverse_indices.device() == device &&
            reverse_values.device() == device,
        "resident finalize tensors must share one NPU device");
    TORCH_CHECK(
        topk_indices.scalar_type() == at::kInt &&
            position_to_union.scalar_type() == at::kInt &&
            selected_packed.scalar_type() == at::kInt &&
            selected_count.scalar_type() == at::kInt &&
            target_slots.scalar_type() == at::kLong &&
            request_block_table.scalar_type() == at::kInt &&
            request_state_indices.scalar_type() == at::kInt &&
            request_state_generations.scalar_type() == at::kLong &&
            old_slots.scalar_type() == at::kShort &&
            slot_to_token.scalar_type() == at::kInt &&
            state_generations.scalar_type() == at::kLong &&
            union_to_slot.scalar_type() == at::kInt &&
            reverse_indices.scalar_type() == at::kLong &&
            reverse_values.scalar_type() == at::kShort,
        "resident finalize tensor dtypes do not match");
    TORCH_CHECK(
        topk_indices.is_contiguous() &&
            position_to_union.is_contiguous() &&
            selected_packed.is_contiguous() &&
            selected_count.is_contiguous() &&
            target_slots.is_contiguous() &&
            request_block_table.is_contiguous() &&
            request_state_indices.is_contiguous() &&
            request_state_generations.is_contiguous() &&
            old_slots.is_contiguous() &&
            slot_to_token.is_contiguous() &&
            state_generations.is_contiguous() &&
            union_to_slot.is_contiguous() &&
            reverse_indices.is_contiguous() &&
            reverse_values.is_contiguous(),
        "resident finalize tensors must be contiguous");

    TORCH_CHECK(
        (topk_indices.dim() == 2 ||
            (topk_indices.dim() == 3 &&
             topk_indices.size(1) == 1)) &&
            selected_packed.dim() == 2 &&
            (selected_count.dim() == 1 || selected_count.dim() == 2) &&
            target_slots.dim() == 2 &&
            request_block_table.dim() == 2 &&
            request_state_indices.dim() == 1 &&
            request_state_generations.dim() == 1 &&
            old_slots.dim() == 2 &&
            slot_to_token.dim() == 2 &&
            state_generations.dim() == 2 &&
            union_to_slot.dim() == 2 &&
            reverse_indices.dim() == 2 &&
            reverse_values.dim() == 2 &&
            position_to_union.dim() == 2,
        "resident finalize tensors must have request-major shapes");
    const int64_t request_count = selected_packed.size(0);
    const int64_t scratch_capacity = selected_packed.size(1);
    const int64_t row_count = topk_indices.size(0);
    TORCH_CHECK(
        request_count > 0 && row_count % request_count == 0,
        "resident finalize top-k rows are not request-major");
    const int64_t rows_per_request = row_count / request_count;
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t selected_count_stride =
        selected_count.dim() == 1 ? 1 : selected_count.size(1);
    const int64_t block_table_width = request_block_table.size(1);
    const int64_t slot_stride = slot_to_token.size(1);
    const int64_t generation_stride = state_generations.size(1);
    const int64_t dummy_state_base = slot_to_token.size(0) / 2;
    TORCH_CHECK(
        scratch_capacity <= 4096 &&
            scratch_capacity % 256 == 0 &&
            selected_count_stride > 0 &&
            block_table_width > 0 &&
            scratch_capacity == rows_per_request * row_width &&
            position_to_union.numel() ==
                request_count * scratch_capacity &&
            target_slots.sizes() == selected_packed.sizes() &&
            old_slots.sizes() == selected_packed.sizes() &&
            union_to_slot.sizes() == selected_packed.sizes() &&
            reverse_indices.sizes() == selected_packed.sizes() &&
            reverse_values.sizes() == selected_packed.sizes(),
        "resident finalize payload shapes do not match");
    TORCH_CHECK(
        selected_count.size(0) >= request_count &&
            request_block_table.size(0) >= request_count &&
            request_state_indices.numel() >= request_count &&
            request_state_generations.numel() >= request_count &&
            slot_to_token.size(0) == state_generations.size(0) &&
            slot_to_token.size(0) == 2 * dummy_state_base &&
            slot_stride >= scratch_capacity + 1 &&
            token_stride % 32 == 0 &&
            slot_stride % 16 == 0 &&
            generation_stride >= 8 &&
            generation_stride % 8 == 0,
        "resident finalize persistent-state shapes do not match");
    TORCH_CHECK(
        token_stride > scratch_capacity && block_size > 0 &&
            request_block_table.size(1) * block_size >=
                scratch_capacity,
        "resident finalize strides do not cover the scratch cache");
    TORCH_CHECK(
        2 * dummy_state_base * token_stride <=
            static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "resident flattened reverse indices exceed int32");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* position_ptr = position_to_union.data_ptr();
    void* selected_ptr = selected_packed.data_ptr();
    void* count_ptr = selected_count.data_ptr();
    void* target_ptr = target_slots.data_ptr();
    void* block_table_ptr = request_block_table.data_ptr();
    void* state_ptr = request_state_indices.data_ptr();
    void* request_generation_ptr = request_state_generations.data_ptr();
    void* old_slot_ptr = old_slots.data_ptr();
    void* slot_to_token_ptr = slot_to_token.data_ptr();
    void* state_generation_ptr = state_generations.data_ptr();
    void* union_to_slot_ptr = union_to_slot.data_ptr();
    void* reverse_index_ptr = reverse_indices.data_ptr();
    void* reverse_value_ptr = reverse_values.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_resident_finalize_rows_");
    cmd.SetCustomHandler([
        stream, topk_ptr, position_ptr, selected_ptr, count_ptr,
        target_ptr, block_table_ptr, state_ptr,
        request_generation_ptr, old_slot_ptr, slot_to_token_ptr,
        state_generation_ptr, union_to_slot_ptr, reverse_index_ptr,
        reverse_value_ptr, request_count, scratch_capacity,
        selected_count_stride, block_table_width, token_stride,
        slot_stride, generation_stride, dummy_state_base, block_size,
        row_count, row_width, rows_per_request]() -> int {
        dsa_resident_finalize_rows_impl(
            stream, selected_ptr, count_ptr, target_ptr,
            block_table_ptr, state_ptr, request_generation_ptr,
            old_slot_ptr, slot_to_token_ptr, state_generation_ptr,
            union_to_slot_ptr, reverse_index_ptr, reverse_value_ptr,
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(scratch_capacity),
            static_cast<uint32_t>(selected_count_stride),
            static_cast<uint32_t>(block_table_width),
            static_cast<uint32_t>(token_stride),
            static_cast<uint32_t>(slot_stride),
            static_cast<uint32_t>(generation_stride),
            static_cast<uint32_t>(dummy_state_base),
            static_cast<uint32_t>(block_size));
        dsa_resident_remap_rows_impl(
            stream, topk_ptr, position_ptr, union_to_slot_ptr,
            static_cast<uint32_t>(row_count),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(rows_per_request),
            static_cast<uint32_t>(scratch_capacity));
        return 0;
    });
    cmd.Run();
    return selected_count;
}

namespace {

at::Tensor resident_sharded_union_common_(
    const at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    const at::Tensor &row_req_indices,
    at::Tensor &shard_packed,
    at::Tensor &shard_mapping,
    at::Tensor &shard_counts,
    const at::Tensor &request_state_indices,
    const at::Tensor &request_state_generations,
    at::Tensor &state_tokens,
    at::Tensor &state_slots,
    at::Tensor &state_counts,
    const at::Tensor &state_generations,
    at::Tensor &prior_slots,
    at::Tensor &shard_miss_tokens,
    at::Tensor &shard_miss_positions,
    at::Tensor &shard_evictable_slots,
    int64_t mtp,
    int64_t dummy_state_base,
    bool experimental_v2)
{
    const auto device = topk_indices.device();
    TORCH_CHECK(
        topk_indices.is_privateuseone() &&
            split_boundary.device() == device &&
            row_req_indices.device() == device &&
            shard_packed.device() == device &&
            shard_mapping.device() == device &&
            shard_counts.device() == device &&
            request_state_indices.device() == device &&
            request_state_generations.device() == device &&
            state_tokens.device() == device &&
            state_slots.device() == device &&
            state_counts.device() == device &&
            state_generations.device() == device &&
            prior_slots.device() == device &&
            shard_miss_tokens.device() == device &&
            shard_miss_positions.device() == device &&
            shard_evictable_slots.device() == device,
        "resident sharded-sort tensors must share one NPU device");
    TORCH_CHECK(
        topk_indices.scalar_type() == at::kInt &&
            split_boundary.scalar_type() == at::kInt &&
            row_req_indices.scalar_type() == at::kInt &&
            shard_packed.scalar_type() == at::kInt &&
            shard_mapping.scalar_type() == at::kShort &&
            shard_counts.scalar_type() == at::kInt &&
            request_state_indices.scalar_type() == at::kInt &&
            request_state_generations.scalar_type() == at::kLong &&
            state_tokens.scalar_type() == at::kInt &&
            state_slots.scalar_type() == at::kShort &&
            state_counts.scalar_type() == at::kInt &&
            state_generations.scalar_type() == at::kLong &&
            prior_slots.scalar_type() == at::kShort &&
            shard_miss_tokens.scalar_type() == at::kInt &&
            shard_miss_positions.scalar_type() == at::kShort &&
            shard_evictable_slots.scalar_type() == at::kShort,
        "resident sharded-union tensor dtypes are invalid");
    TORCH_CHECK(
        topk_indices.is_contiguous() &&
            split_boundary.is_contiguous() &&
            row_req_indices.is_contiguous() &&
            shard_packed.is_contiguous() &&
            shard_mapping.is_contiguous() &&
            shard_counts.is_contiguous() &&
            request_state_indices.is_contiguous() &&
            request_state_generations.is_contiguous() &&
            state_tokens.is_contiguous() &&
            state_slots.is_contiguous() &&
            state_counts.is_contiguous() &&
            state_generations.is_contiguous() &&
            prior_slots.is_contiguous() &&
            shard_miss_tokens.is_contiguous() &&
            shard_miss_positions.is_contiguous() &&
            shard_evictable_slots.is_contiguous(),
        "resident sharded-sort tensors must be contiguous");
    TORCH_CHECK(
        (topk_indices.dim() == 2 ||
            (topk_indices.dim() == 3 &&
             topk_indices.size(1) == 1)) &&
            split_boundary.dim() == 1 &&
            row_req_indices.dim() == 1 &&
            shard_packed.dim() == 3 &&
            shard_mapping.dim() == 3 &&
            shard_counts.dim() == 3 &&
            request_state_indices.dim() == 1 &&
            request_state_generations.dim() == 1 &&
            state_tokens.dim() == 3 &&
            state_slots.dim() == 3 &&
            state_counts.dim() == 3 &&
            state_generations.dim() == 2 &&
            prior_slots.dim() == 3 &&
            shard_miss_tokens.dim() == 3 &&
            shard_miss_positions.dim() == 3 &&
            shard_evictable_slots.dim() == 3,
        "resident sharded-sort tensor ranks are invalid");

    const int64_t row_count = topk_indices.size(0);
    const int64_t request_count = shard_packed.size(0);
    TORCH_CHECK(
        request_count > 0 && row_count > 0 &&
            row_count % request_count == 0,
        "resident sharded-sort rows must be request-major");
    const int64_t rows_per_request = row_count / request_count;
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t request_width = rows_per_request * row_width;
    const int64_t shard_count = shard_packed.size(1);
    const int64_t shard_capacity = shard_packed.size(2);
    const int64_t shard_count_stride = shard_counts.size(2);
    const int64_t shard_count_request_stride =
        shard_counts.size(1) * shard_count_stride;
    const int64_t state_row_count = state_tokens.size(0);
    const int64_t generation_stride = state_generations.size(1);
    TORCH_CHECK(
        mtp >= 1 && mtp <= 2 && rows_per_request == mtp,
        "resident sharded union currently requires MTP=1 or MTP=2");
    int64_t expected_shard_count = 1;
    while (expected_shard_count <= mtp) {
        expected_shard_count <<= 1;
    }
    TORCH_CHECK(
        row_width == 2048 &&
            shard_count == expected_shard_count &&
            shard_capacity == request_width &&
            request_width % (16 * shard_count) == 0 &&
            split_boundary.numel() >= row_count &&
            row_req_indices.numel() >= row_count &&
            shard_mapping.size(0) == request_count &&
            shard_mapping.size(1) == shard_count &&
            shard_mapping.size(2) == request_width &&
            shard_counts.size(0) == request_count &&
            shard_counts.size(1) == shard_count &&
            shard_count_stride >= 16 &&
            dummy_state_base >= request_count &&
            state_row_count >= dummy_state_base + request_count &&
            state_slots.sizes() == state_tokens.sizes() &&
            state_tokens.size(1) == shard_count &&
            state_tokens.size(2) == shard_capacity &&
            state_counts.size(0) == state_row_count &&
            state_counts.size(1) == shard_count &&
            state_counts.size(2) >= 16 &&
            state_generations.size(0) == state_row_count &&
            generation_stride >= 8 &&
            prior_slots.sizes() == shard_packed.sizes() &&
            shard_miss_tokens.sizes() == shard_packed.sizes() &&
            shard_miss_positions.sizes() == shard_packed.sizes() &&
            shard_evictable_slots.sizes() == shard_packed.sizes() &&
            request_state_indices.numel() >= request_count &&
            request_state_generations.numel() >= request_count,
        "resident sharded-sort workspace shapes are invalid");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(
            shard_packed.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                shard_mapping.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                shard_counts.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                state_tokens.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                state_slots.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                state_counts.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                state_generations.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                prior_slots.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                shard_miss_tokens.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                shard_miss_positions.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                shard_evictable_slots.data_ptr()) % 64 == 0,
        "resident sharded-sort output bases must be 64-byte aligned");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* boundary_ptr = split_boundary.data_ptr();
    void* row_request_ptr = row_req_indices.data_ptr();
    void* packed_ptr = shard_packed.data_ptr();
    void* mapping_ptr = shard_mapping.data_ptr();
    void* count_ptr = shard_counts.data_ptr();
    void* request_state_ptr = request_state_indices.data_ptr();
    void* request_generation_ptr =
        request_state_generations.data_ptr();
    void* state_token_ptr = state_tokens.data_ptr();
    void* state_slot_ptr = state_slots.data_ptr();
    void* state_count_ptr = state_counts.data_ptr();
    void* state_generation_ptr = state_generations.data_ptr();
    void* prior_slot_ptr = prior_slots.data_ptr();
    void* shard_miss_token_ptr = shard_miss_tokens.data_ptr();
    void* shard_miss_position_ptr = shard_miss_positions.data_ptr();
    void* shard_evictable_slot_ptr = shard_evictable_slots.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name(
        experimental_v2
            ? "npu_dsa_resident_sharded_union_v2_"
            : "npu_dsa_resident_sharded_union_");
    cmd.SetCustomHandler([
        stream, topk_ptr, boundary_ptr, row_request_ptr,
        packed_ptr, mapping_ptr, count_ptr, request_state_ptr,
        request_generation_ptr, state_token_ptr, state_slot_ptr,
        state_count_ptr, state_generation_ptr, prior_slot_ptr,
        shard_miss_token_ptr, shard_miss_position_ptr,
        shard_evictable_slot_ptr,
        request_count, state_row_count, dummy_state_base,
        rows_per_request, row_width, shard_count, shard_capacity,
        shard_count_stride, shard_count_request_stride,
        generation_stride, experimental_v2]() -> int {
        auto impl = experimental_v2
            ? dsa_resident_sharded_union_v2_impl
            : dsa_resident_sharded_union_impl;
        impl(
            stream, topk_ptr, boundary_ptr, row_request_ptr,
            packed_ptr, mapping_ptr, count_ptr, request_state_ptr,
            request_generation_ptr, state_token_ptr, state_slot_ptr,
            state_count_ptr, state_generation_ptr, prior_slot_ptr,
            shard_miss_token_ptr, shard_miss_position_ptr,
            shard_evictable_slot_ptr,
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(state_row_count),
            static_cast<uint32_t>(dummy_state_base),
            static_cast<uint32_t>(rows_per_request),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(shard_count),
            static_cast<uint32_t>(shard_capacity),
            static_cast<uint32_t>(shard_count_stride),
            static_cast<uint32_t>(shard_count_request_stride),
            static_cast<uint32_t>(generation_stride));
        return 0;
    });
    cmd.Run();
    return shard_counts;
}

}  // namespace

at::Tensor npu_dsa_resident_sharded_union_(
    const at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    const at::Tensor &row_req_indices,
    at::Tensor &shard_packed,
    at::Tensor &shard_mapping,
    at::Tensor &shard_counts,
    const at::Tensor &request_state_indices,
    const at::Tensor &request_state_generations,
    at::Tensor &state_tokens,
    at::Tensor &state_slots,
    at::Tensor &state_counts,
    const at::Tensor &state_generations,
    at::Tensor &prior_slots,
    at::Tensor &shard_miss_tokens,
    at::Tensor &shard_miss_positions,
    at::Tensor &shard_evictable_slots,
    int64_t mtp,
    int64_t dummy_state_base)
{
    return resident_sharded_union_common_(
        topk_indices, split_boundary, row_req_indices,
        shard_packed, shard_mapping, shard_counts,
        request_state_indices, request_state_generations,
        state_tokens, state_slots, state_counts, state_generations,
        prior_slots, shard_miss_tokens, shard_miss_positions,
        shard_evictable_slots, mtp, dummy_state_base, false);
}

at::Tensor npu_dsa_resident_sharded_union_v2_(
    const at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    const at::Tensor &row_req_indices,
    at::Tensor &shard_packed,
    at::Tensor &shard_mapping,
    at::Tensor &shard_counts,
    const at::Tensor &request_state_indices,
    const at::Tensor &request_state_generations,
    at::Tensor &state_tokens,
    at::Tensor &state_slots,
    at::Tensor &state_counts,
    const at::Tensor &state_generations,
    at::Tensor &prior_slots,
    at::Tensor &shard_miss_tokens,
    at::Tensor &shard_miss_positions,
    at::Tensor &shard_evictable_slots,
    int64_t mtp,
    int64_t dummy_state_base)
{
    return resident_sharded_union_common_(
        topk_indices, split_boundary, row_req_indices,
        shard_packed, shard_mapping, shard_counts,
        request_state_indices, request_state_generations,
        state_tokens, state_slots, state_counts, state_generations,
        prior_slots, shard_miss_tokens, shard_miss_positions,
        shard_evictable_slots, mtp, dummy_state_base, true);
}

static at::Tensor resident_sorted_plan_common_(
    at::Tensor &topk_indices,
    const at::Tensor &shard_packed,
    const at::Tensor &shard_mapping,
    const at::Tensor &shard_counts,
    const at::Tensor &request_block_table,
    const at::Tensor &request_state_indices,
    const at::Tensor &request_state_generations,
    at::Tensor &state_tokens,
    at::Tensor &state_slots,
    at::Tensor &state_counts,
    at::Tensor &state_generations,
    at::Tensor &prior_slots,
    const at::Tensor &shard_miss_tokens,
    const at::Tensor &shard_miss_positions,
    const at::Tensor &shard_evictable_slots,
    at::Tensor &miss_tokens,
    at::Tensor &miss_counts,
    at::Tensor &target_slots,
    int64_t block_size,
    int64_t dummy_state_base,
    bool fused_remap,
    bool experimental_v2)
{
    const auto device = topk_indices.device();
    TORCH_CHECK(
        topk_indices.is_privateuseone() &&
            shard_packed.device() == device &&
            shard_mapping.device() == device &&
            shard_counts.device() == device &&
            request_block_table.device() == device &&
            request_state_indices.device() == device &&
            request_state_generations.device() == device &&
            state_tokens.device() == device &&
            state_slots.device() == device &&
            state_counts.device() == device &&
            state_generations.device() == device &&
            prior_slots.device() == device &&
            shard_miss_tokens.device() == device &&
            shard_miss_positions.device() == device &&
            shard_evictable_slots.device() == device &&
            miss_tokens.device() == device &&
            miss_counts.device() == device &&
            target_slots.device() == device,
        "sorted-resident tensors must share one NPU device");
    TORCH_CHECK(
        topk_indices.scalar_type() == at::kInt &&
            shard_packed.scalar_type() == at::kInt &&
            shard_mapping.scalar_type() == at::kShort &&
            shard_counts.scalar_type() == at::kInt &&
            request_block_table.scalar_type() == at::kInt &&
            request_state_indices.scalar_type() == at::kInt &&
            request_state_generations.scalar_type() == at::kLong &&
            state_tokens.scalar_type() == at::kInt &&
            state_slots.scalar_type() == at::kShort &&
            state_counts.scalar_type() == at::kInt &&
            state_generations.scalar_type() == at::kLong &&
            prior_slots.scalar_type() == at::kShort &&
            shard_miss_tokens.scalar_type() == at::kInt &&
            shard_miss_positions.scalar_type() == at::kShort &&
            shard_evictable_slots.scalar_type() == at::kShort &&
            miss_tokens.scalar_type() == at::kInt &&
            miss_counts.scalar_type() == at::kInt &&
            target_slots.scalar_type() == at::kLong,
        "sorted-resident tensor dtypes are invalid");
    TORCH_CHECK(
        topk_indices.is_contiguous() &&
            shard_packed.is_contiguous() &&
            shard_mapping.is_contiguous() &&
            shard_counts.is_contiguous() &&
            request_block_table.is_contiguous() &&
            request_state_indices.is_contiguous() &&
            request_state_generations.is_contiguous() &&
            state_tokens.is_contiguous() &&
            state_slots.is_contiguous() &&
            state_counts.is_contiguous() &&
            state_generations.is_contiguous() &&
            prior_slots.is_contiguous() &&
            shard_miss_tokens.is_contiguous() &&
            shard_miss_positions.is_contiguous() &&
            shard_evictable_slots.is_contiguous() &&
            miss_tokens.is_contiguous() &&
            miss_counts.is_contiguous() &&
            target_slots.is_contiguous(),
        "sorted-resident tensors must be contiguous");
    TORCH_CHECK(
        (topk_indices.dim() == 2 ||
            (topk_indices.dim() == 3 &&
             topk_indices.size(1) == 1)) &&
            shard_packed.dim() == 3 &&
            shard_mapping.dim() == 3 &&
            shard_counts.dim() == 3 &&
            request_block_table.dim() == 2 &&
            request_state_indices.dim() == 1 &&
            request_state_generations.dim() == 1 &&
            state_tokens.dim() == 3 &&
            state_slots.dim() == 3 &&
            state_counts.dim() == 3 &&
            state_generations.dim() == 2 &&
            prior_slots.dim() == 3 &&
            shard_miss_tokens.dim() == 3 &&
            shard_miss_positions.dim() == 3 &&
            shard_evictable_slots.dim() == 3 &&
            miss_tokens.dim() == 2 &&
            miss_counts.dim() == 2 &&
            target_slots.dim() == 2,
        "sorted-resident tensor ranks are invalid");

    const int64_t request_count = shard_packed.size(0);
    const int64_t shard_count = shard_packed.size(1);
    const int64_t capacity = shard_packed.size(2);
    const int64_t row_count = topk_indices.size(0);
    TORCH_CHECK(
        request_count > 0 && row_count > 0 &&
            row_count % request_count == 0,
        "sorted-resident top-k rows must be request-major");
    const int64_t rows_per_request = row_count / request_count;
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t request_width = rows_per_request * row_width;
    const int64_t state_row_count = state_tokens.size(0);
    const int64_t shard_count_stride = shard_counts.size(2);
    const int64_t shard_count_request_stride =
        shard_counts.size(1) * shard_count_stride;
    const int64_t miss_count_stride = miss_counts.size(1);
    const int64_t generation_stride = state_generations.size(1);
    const int64_t block_table_width = request_block_table.size(1);
    int64_t expected_shard_count = 1;
    while (expected_shard_count <= rows_per_request) {
        expected_shard_count <<= 1;
    }
    TORCH_CHECK(
        rows_per_request >= 1 && rows_per_request <= 2 &&
            row_width == 2048 &&
            capacity == request_width &&
            shard_count == expected_shard_count &&
            request_width % (16 * shard_count) == 0 &&
            block_size > 0 &&
            block_table_width * block_size >= capacity,
        "sorted-resident dimensions are unsupported");
    TORCH_CHECK(
        !experimental_v2 ||
            (block_size & (block_size - 1)) == 0,
        "parallel resident v2 requires a power-of-two block_size");
    TORCH_CHECK(
        dummy_state_base >= request_count &&
            state_row_count >= dummy_state_base + request_count &&
            state_slots.sizes() == state_tokens.sizes() &&
            state_tokens.size(1) == shard_count &&
            state_tokens.size(2) == capacity &&
            state_counts.size(0) == state_row_count &&
            state_counts.size(1) == shard_count &&
            state_counts.size(2) >= 16 &&
            state_generations.size(0) == state_row_count &&
            generation_stride >= 8,
        "sorted-resident persistent-state shapes are invalid");
    TORCH_CHECK(
        shard_mapping.size(0) == request_count &&
            shard_mapping.size(1) == shard_count &&
            shard_mapping.size(2) == request_width &&
            shard_counts.size(0) == request_count &&
            shard_counts.size(1) == shard_count &&
            shard_count_stride >= 16 &&
            prior_slots.sizes() == shard_packed.sizes() &&
            shard_miss_tokens.sizes() == shard_packed.sizes() &&
            shard_miss_positions.sizes() == shard_packed.sizes() &&
            shard_evictable_slots.sizes() == shard_packed.sizes() &&
            miss_tokens.size(0) == request_count &&
            miss_tokens.size(1) == capacity &&
            miss_counts.size(0) == request_count &&
            miss_count_stride >= 16 &&
            target_slots.sizes() == miss_tokens.sizes() &&
            request_block_table.size(0) >= request_count &&
            request_state_indices.numel() >= request_count &&
            request_state_generations.numel() >= request_count,
        "sorted-resident workspace shapes are invalid");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(
            state_tokens.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                state_slots.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                state_counts.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                state_generations.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                prior_slots.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                shard_miss_tokens.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                shard_miss_positions.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                shard_evictable_slots.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                miss_counts.data_ptr()) % 64 == 0,
        "sorted-resident cacheline-owned buffers must be 64-byte aligned");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* packed_ptr = shard_packed.data_ptr();
    void* mapping_ptr = shard_mapping.data_ptr();
    void* shard_count_ptr = shard_counts.data_ptr();
    void* block_table_ptr = request_block_table.data_ptr();
    void* request_state_ptr = request_state_indices.data_ptr();
    void* request_generation_ptr =
        request_state_generations.data_ptr();
    void* state_token_ptr = state_tokens.data_ptr();
    void* state_slot_ptr = state_slots.data_ptr();
    void* state_count_ptr = state_counts.data_ptr();
    void* state_generation_ptr = state_generations.data_ptr();
    void* prior_slot_ptr = prior_slots.data_ptr();
    void* shard_miss_token_ptr = shard_miss_tokens.data_ptr();
    void* shard_miss_position_ptr = shard_miss_positions.data_ptr();
    void* shard_evictable_slot_ptr = shard_evictable_slots.data_ptr();
    void* miss_token_ptr = miss_tokens.data_ptr();
    void* miss_count_ptr = miss_counts.data_ptr();
    void* target_ptr = target_slots.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name(
        experimental_v2
            ? "npu_dsa_resident_parallel_plan_v2_"
            : fused_remap
            ? "npu_dsa_resident_sorted_plan_"
            : "npu_dsa_resident_sorted_plan_no_remap_");
    cmd.SetCustomHandler([
        stream, topk_ptr, packed_ptr, mapping_ptr,
        shard_count_ptr, block_table_ptr, request_state_ptr,
        request_generation_ptr, state_token_ptr, state_slot_ptr,
        state_count_ptr, state_generation_ptr,
        prior_slot_ptr, shard_miss_token_ptr,
        shard_miss_position_ptr, shard_evictable_slot_ptr,
        miss_token_ptr, miss_count_ptr, target_ptr,
        request_count, state_row_count, dummy_state_base,
        rows_per_request, row_width, shard_count, capacity,
        shard_count_stride, shard_count_request_stride,
        miss_count_stride, generation_stride,
        block_table_width, block_size, fused_remap,
        experimental_v2]() -> int {
        auto impl = experimental_v2
            ? dsa_resident_parallel_plan_v2_impl
            : fused_remap
                ? dsa_resident_sorted_plan_impl
                : dsa_resident_sorted_plan_no_remap_impl;
        impl(
            stream, topk_ptr, packed_ptr, mapping_ptr,
            shard_count_ptr, block_table_ptr, request_state_ptr,
            request_generation_ptr, state_token_ptr, state_slot_ptr,
            state_count_ptr, state_generation_ptr,
            prior_slot_ptr, shard_miss_token_ptr,
            shard_miss_position_ptr, shard_evictable_slot_ptr,
            miss_token_ptr, miss_count_ptr,
            target_ptr,
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(state_row_count),
            static_cast<uint32_t>(dummy_state_base),
            static_cast<uint32_t>(rows_per_request),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(shard_count),
            static_cast<uint32_t>(capacity),
            static_cast<uint32_t>(shard_count_stride),
            static_cast<uint32_t>(shard_count_request_stride),
            static_cast<uint32_t>(miss_count_stride),
            static_cast<uint32_t>(generation_stride),
            static_cast<uint32_t>(block_table_width),
            static_cast<uint32_t>(block_size));
        return 0;
    });
    cmd.Run();
    return miss_counts;
}

at::Tensor npu_dsa_resident_sorted_plan_(
    at::Tensor &topk_indices,
    const at::Tensor &shard_packed,
    const at::Tensor &shard_mapping,
    const at::Tensor &shard_counts,
    const at::Tensor &request_block_table,
    const at::Tensor &request_state_indices,
    const at::Tensor &request_state_generations,
    at::Tensor &state_tokens,
    at::Tensor &state_slots,
    at::Tensor &state_counts,
    at::Tensor &state_generations,
    at::Tensor &prior_slots,
    const at::Tensor &shard_miss_tokens,
    const at::Tensor &shard_miss_positions,
    const at::Tensor &shard_evictable_slots,
    at::Tensor &miss_tokens,
    at::Tensor &miss_counts,
    at::Tensor &target_slots,
    int64_t block_size,
    int64_t dummy_state_base)
{
    return resident_sorted_plan_common_(
        topk_indices, shard_packed, shard_mapping, shard_counts,
        request_block_table, request_state_indices,
        request_state_generations, state_tokens, state_slots,
        state_counts, state_generations, prior_slots,
        shard_miss_tokens, shard_miss_positions, shard_evictable_slots,
        miss_tokens, miss_counts, target_slots,
        block_size, dummy_state_base, true, false);
}

at::Tensor npu_dsa_resident_parallel_plan_v2_(
    at::Tensor &topk_indices,
    const at::Tensor &shard_packed,
    const at::Tensor &shard_mapping,
    const at::Tensor &shard_counts,
    const at::Tensor &request_block_table,
    const at::Tensor &request_state_indices,
    const at::Tensor &request_state_generations,
    at::Tensor &state_tokens,
    at::Tensor &state_slots,
    at::Tensor &state_counts,
    at::Tensor &state_generations,
    at::Tensor &prior_slots,
    const at::Tensor &shard_miss_tokens,
    const at::Tensor &shard_miss_positions,
    const at::Tensor &shard_evictable_slots,
    at::Tensor &miss_tokens,
    at::Tensor &miss_counts,
    at::Tensor &target_slots,
    int64_t block_size,
    int64_t dummy_state_base)
{
    return resident_sorted_plan_common_(
        topk_indices, shard_packed, shard_mapping, shard_counts,
        request_block_table, request_state_indices,
        request_state_generations, state_tokens, state_slots,
        state_counts, state_generations, prior_slots,
        shard_miss_tokens, shard_miss_positions, shard_evictable_slots,
        miss_tokens, miss_counts, target_slots,
        block_size, dummy_state_base, true, true);
}

at::Tensor npu_dsa_resident_sorted_plan_no_remap_(
    at::Tensor &topk_indices,
    const at::Tensor &shard_packed,
    const at::Tensor &shard_mapping,
    const at::Tensor &shard_counts,
    const at::Tensor &request_block_table,
    const at::Tensor &request_state_indices,
    const at::Tensor &request_state_generations,
    at::Tensor &state_tokens,
    at::Tensor &state_slots,
    at::Tensor &state_counts,
    at::Tensor &state_generations,
    at::Tensor &prior_slots,
    const at::Tensor &shard_miss_tokens,
    const at::Tensor &shard_miss_positions,
    const at::Tensor &shard_evictable_slots,
    at::Tensor &miss_tokens,
    at::Tensor &miss_counts,
    at::Tensor &target_slots,
    int64_t block_size,
    int64_t dummy_state_base)
{
    return resident_sorted_plan_common_(
        topk_indices, shard_packed, shard_mapping, shard_counts,
        request_block_table, request_state_indices,
        request_state_generations, state_tokens, state_slots,
        state_counts, state_generations, prior_slots,
        shard_miss_tokens, shard_miss_positions, shard_evictable_slots,
        miss_tokens, miss_counts, target_slots,
        block_size, dummy_state_base, false, false);
}

at::Tensor npu_dsa_resident_sorted_update_debug_(
    at::Tensor &topk_indices,
    const at::Tensor &shard_packed,
    const at::Tensor &shard_mapping,
    const at::Tensor &shard_counts,
    const at::Tensor &prior_slots,
    const at::Tensor &request_state_indices,
    const at::Tensor &request_state_generations,
    at::Tensor &state_tokens,
    at::Tensor &state_slots,
    at::Tensor &state_counts,
    at::Tensor &state_generations,
    int64_t dummy_state_base)
{
    const auto device = topk_indices.device();
    TORCH_CHECK(
        topk_indices.is_privateuseone() &&
            shard_packed.device() == device &&
            shard_mapping.device() == device &&
            shard_counts.device() == device &&
            prior_slots.device() == device &&
            request_state_indices.device() == device &&
            request_state_generations.device() == device &&
            state_tokens.device() == device &&
            state_slots.device() == device &&
            state_counts.device() == device &&
            state_generations.device() == device,
        "resident update-debug tensors must share one NPU device");
    TORCH_CHECK(
        topk_indices.scalar_type() == at::kInt &&
            shard_packed.scalar_type() == at::kInt &&
            shard_mapping.scalar_type() == at::kShort &&
            shard_counts.scalar_type() == at::kInt &&
            prior_slots.scalar_type() == at::kShort &&
            request_state_indices.scalar_type() == at::kInt &&
            request_state_generations.scalar_type() == at::kLong &&
            state_tokens.scalar_type() == at::kInt &&
            state_slots.scalar_type() == at::kShort &&
            state_counts.scalar_type() == at::kInt &&
            state_generations.scalar_type() == at::kLong,
        "resident update-debug tensor dtypes are invalid");
    TORCH_CHECK(
        topk_indices.is_contiguous() && shard_packed.is_contiguous() &&
            shard_mapping.is_contiguous() && shard_counts.is_contiguous() &&
            prior_slots.is_contiguous() &&
            request_state_indices.is_contiguous() &&
            request_state_generations.is_contiguous() &&
            state_tokens.is_contiguous() && state_slots.is_contiguous() &&
            state_counts.is_contiguous() && state_generations.is_contiguous(),
        "resident update-debug tensors must be contiguous");
    TORCH_CHECK(
        (topk_indices.dim() == 2 ||
         (topk_indices.dim() == 3 && topk_indices.size(1) == 1)) &&
            shard_packed.dim() == 3 && shard_mapping.dim() == 3 &&
            shard_counts.dim() == 3 && prior_slots.dim() == 3 &&
            request_state_indices.dim() == 1 &&
            request_state_generations.dim() == 1 &&
            state_tokens.dim() == 3 && state_slots.dim() == 3 &&
            state_counts.dim() == 3 && state_generations.dim() == 2,
        "resident update-debug tensor ranks are invalid");

    const int64_t request_count = shard_packed.size(0);
    const int64_t shard_count = shard_packed.size(1);
    const int64_t capacity = shard_packed.size(2);
    const int64_t row_count = topk_indices.size(0);
    TORCH_CHECK(
        request_count > 0 && row_count > 0 &&
            row_count % request_count == 0,
        "resident update-debug top-k rows must be request-major");
    const int64_t rows_per_request = row_count / request_count;
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t request_width = rows_per_request * row_width;
    const int64_t state_row_count = state_tokens.size(0);
    const int64_t shard_count_stride = shard_counts.size(2);
    const int64_t shard_count_request_stride =
        shard_counts.size(1) * shard_count_stride;
    const int64_t generation_stride = state_generations.size(1);
    int64_t expected_shard_count = 1;
    while (expected_shard_count <= rows_per_request) {
        expected_shard_count <<= 1;
    }
    TORCH_CHECK(
        rows_per_request >= 1 && rows_per_request <= 2 &&
            row_width == 2048 && capacity == request_width &&
            shard_count == expected_shard_count &&
            request_width % (16 * shard_count) == 0 &&
            dummy_state_base >= request_count &&
            state_row_count >= dummy_state_base + request_count &&
            state_slots.sizes() == state_tokens.sizes() &&
            state_tokens.size(1) == shard_count &&
            state_tokens.size(2) == capacity &&
            state_counts.size(0) == state_row_count &&
            state_counts.size(1) == shard_count &&
            shard_count_stride >= 16 &&
            state_counts.size(2) >= 16 &&
            state_generations.size(0) == state_row_count &&
            generation_stride >= 8 &&
            shard_mapping.size(0) == request_count &&
            shard_mapping.size(1) == shard_count &&
            shard_mapping.size(2) == request_width &&
            shard_counts.size(0) == request_count &&
            shard_counts.size(1) == shard_count &&
            prior_slots.sizes() == shard_packed.sizes() &&
            request_state_indices.numel() >= request_count &&
            request_state_generations.numel() >= request_count,
        "resident update-debug tensor shapes are invalid");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* packed_ptr = shard_packed.data_ptr();
    void* mapping_ptr = shard_mapping.data_ptr();
    void* count_ptr = shard_counts.data_ptr();
    void* prior_ptr = prior_slots.data_ptr();
    void* request_state_ptr = request_state_indices.data_ptr();
    void* request_generation_ptr = request_state_generations.data_ptr();
    void* state_token_ptr = state_tokens.data_ptr();
    void* state_slot_ptr = state_slots.data_ptr();
    void* state_count_ptr = state_counts.data_ptr();
    void* state_generation_ptr = state_generations.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_resident_sorted_update_debug_");
    cmd.SetCustomHandler([
        stream, topk_ptr, packed_ptr, mapping_ptr, count_ptr,
        prior_ptr, request_state_ptr,
        request_generation_ptr, state_token_ptr, state_slot_ptr,
        state_count_ptr, state_generation_ptr, request_count,
        state_row_count, dummy_state_base, rows_per_request,
        row_width, shard_count, capacity, shard_count_stride,
        shard_count_request_stride, generation_stride]() -> int {
        dsa_resident_sorted_update_debug_impl(
            stream, topk_ptr, packed_ptr, mapping_ptr, count_ptr,
            prior_ptr, request_state_ptr,
            request_generation_ptr, state_token_ptr, state_slot_ptr,
            state_count_ptr, state_generation_ptr,
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(state_row_count),
            static_cast<uint32_t>(dummy_state_base),
            static_cast<uint32_t>(rows_per_request),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(shard_count),
            static_cast<uint32_t>(capacity),
            static_cast<uint32_t>(shard_count_stride),
            static_cast<uint32_t>(shard_count_request_stride),
            static_cast<uint32_t>(generation_stride));
        return 0;
    });
    cmd.Run();
    return topk_indices;
}

at::Tensor npu_dsa_resident_sorted_remap_(
    at::Tensor &topk_indices,
    const at::Tensor &shard_mapping,
    const at::Tensor &shard_counts,
    const at::Tensor &prior_slots)
{
    const auto device = topk_indices.device();
    TORCH_CHECK(
        topk_indices.is_privateuseone() &&
            shard_mapping.device() == device &&
            shard_counts.device() == device &&
            prior_slots.device() == device,
        "sorted-resident remap tensors must share one NPU device");
    TORCH_CHECK(
        topk_indices.scalar_type() == at::kInt &&
            shard_mapping.scalar_type() == at::kShort &&
            shard_counts.scalar_type() == at::kInt &&
            prior_slots.scalar_type() == at::kShort,
        "sorted-resident remap tensor dtypes are invalid");
    TORCH_CHECK(
        topk_indices.is_contiguous() &&
            shard_mapping.is_contiguous() &&
            shard_counts.is_contiguous() &&
            prior_slots.is_contiguous(),
        "sorted-resident remap tensors must be contiguous");
    TORCH_CHECK(
        (topk_indices.dim() == 2 ||
            (topk_indices.dim() == 3 &&
             topk_indices.size(1) == 1)) &&
            shard_mapping.dim() == 3 &&
            shard_counts.dim() == 3 &&
            prior_slots.dim() == 3,
        "sorted-resident remap tensor ranks are invalid");

    const int64_t request_count = shard_mapping.size(0);
    const int64_t row_count = topk_indices.size(0);
    TORCH_CHECK(
        request_count > 0 && row_count > 0 &&
            row_count % request_count == 0,
        "sorted-resident remap rows must be request-major");
    const int64_t rows_per_request = row_count / request_count;
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t request_width = rows_per_request * row_width;
    const int64_t shard_count = shard_mapping.size(1);
    const int64_t capacity = prior_slots.size(2);
    const int64_t shard_count_stride = shard_counts.size(2);
    const int64_t shard_count_request_stride =
        shard_counts.size(1) * shard_count_stride;
    int64_t expected_shard_count = 1;
    while (expected_shard_count <= rows_per_request) {
        expected_shard_count <<= 1;
    }
    TORCH_CHECK(
        rows_per_request >= 1 && rows_per_request <= 2 &&
            row_width == 2048 &&
            request_width == capacity &&
            shard_count == expected_shard_count &&
            request_width % (16 * shard_count) == 0 &&
            shard_mapping.size(2) == request_width &&
            shard_counts.size(0) == request_count &&
            shard_counts.size(1) == shard_count &&
            shard_count_stride >= 16 &&
            prior_slots.size(0) == request_count &&
            prior_slots.size(1) == shard_count,
        "sorted-resident remap workspace shapes are invalid");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(
            topk_indices.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
            shard_mapping.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                shard_counts.data_ptr()) % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(
                prior_slots.data_ptr()) % 64 == 0,
        "sorted-resident remap inputs must be 64-byte aligned");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* mapping_ptr = shard_mapping.data_ptr();
    void* count_ptr = shard_counts.data_ptr();
    void* prior_ptr = prior_slots.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_resident_sorted_remap_");
    cmd.SetCustomHandler([
        stream, topk_ptr, mapping_ptr, count_ptr, prior_ptr,
        request_count, rows_per_request, row_width, shard_count,
        capacity, shard_count_stride,
        shard_count_request_stride]() -> int {
        dsa_resident_sorted_remap_impl(
            stream, topk_ptr, mapping_ptr, count_ptr, prior_ptr,
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(rows_per_request),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(shard_count),
            static_cast<uint32_t>(capacity),
            static_cast<uint32_t>(shard_count_stride),
            static_cast<uint32_t>(shard_count_request_stride));
        return 0;
    });
    cmd.Run();
    return topk_indices;
}

at::Tensor npu_dsa_resident_sorted_read_probe_(
    const at::Tensor &shard_counts,
    const at::Tensor &prior_slots,
    at::Tensor &debug_info,
    at::Tensor &prior_readback)
{
    const auto device = shard_counts.device();
    TORCH_CHECK(
        shard_counts.is_privateuseone() &&
            prior_slots.device() == device &&
            debug_info.device() == device &&
            prior_readback.device() == device,
        "resident read-probe tensors must share one NPU device");
    TORCH_CHECK(
        shard_counts.scalar_type() == at::kInt &&
            prior_slots.scalar_type() == at::kShort &&
            debug_info.scalar_type() == at::kInt &&
            prior_readback.scalar_type() == at::kShort,
        "resident read-probe tensor dtypes are invalid");
    TORCH_CHECK(
        shard_counts.is_contiguous() &&
            prior_slots.is_contiguous() &&
            debug_info.is_contiguous() &&
            prior_readback.is_contiguous(),
        "resident read-probe tensors must be contiguous");
    TORCH_CHECK(
        shard_counts.dim() == 3 &&
            prior_slots.dim() == 3 &&
            debug_info.dim() == 2 &&
            prior_readback.dim() == 3,
        "resident read-probe tensor ranks are invalid");

    const int64_t request_count = prior_slots.size(0);
    const int64_t shard_count = prior_slots.size(1);
    const int64_t capacity = prior_slots.size(2);
    const int64_t shard_count_stride = shard_counts.size(2);
    const int64_t shard_count_request_stride =
        shard_counts.size(1) * shard_count_stride;
    TORCH_CHECK(
        request_count > 0 &&
            (shard_count == 2 || shard_count == 4) &&
            capacity > 0 &&
            shard_counts.size(0) == request_count &&
            shard_counts.size(1) == shard_count &&
            shard_count_stride >= 16 &&
            debug_info.size(0) == request_count &&
            debug_info.size(1) == 32 &&
            prior_readback.sizes() == prior_slots.sizes(),
        "resident read-probe tensor shapes are invalid");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* count_ptr = shard_counts.data_ptr();
    void* prior_ptr = prior_slots.data_ptr();
    void* debug_ptr = debug_info.data_ptr();
    void* readback_ptr = prior_readback.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_resident_sorted_read_probe_");
    cmd.SetCustomHandler([
        stream, count_ptr, prior_ptr, debug_ptr, readback_ptr,
        request_count, shard_count, capacity,
        shard_count_stride,
        shard_count_request_stride]() -> int {
        dsa_resident_sorted_read_probe_impl(
            stream, count_ptr, prior_ptr, debug_ptr, readback_ptr,
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(shard_count),
            static_cast<uint32_t>(capacity),
            static_cast<uint32_t>(shard_count_stride),
            static_cast<uint32_t>(shard_count_request_stride));
        return 0;
    });
    cmd.Run();
    return debug_info;
}

at::Tensor npu_dsa_resident_sorted_finalize_debug_(
    const at::Tensor &shard_packed,
    const at::Tensor &shard_counts,
    at::Tensor &prior_slots,
    const at::Tensor &shard_miss_tokens,
    const at::Tensor &shard_miss_positions,
    const at::Tensor &shard_evictable_slots,
    at::Tensor &miss_tokens,
    at::Tensor &miss_counts,
    at::Tensor &target_slots,
    const at::Tensor &request_block_table,
    at::Tensor &debug_info,
    int64_t block_size,
    int64_t debug_stage)
{
    const auto device = shard_packed.device();
    TORCH_CHECK(
        shard_packed.is_privateuseone() &&
            shard_counts.device() == device &&
            prior_slots.device() == device &&
            shard_miss_tokens.device() == device &&
            shard_miss_positions.device() == device &&
            shard_evictable_slots.device() == device &&
            miss_tokens.device() == device &&
            miss_counts.device() == device &&
            target_slots.device() == device &&
            request_block_table.device() == device &&
            debug_info.device() == device,
        "resident finalize-debug tensors must share one NPU device");
    TORCH_CHECK(
        shard_packed.scalar_type() == at::kInt &&
            shard_counts.scalar_type() == at::kInt &&
            prior_slots.scalar_type() == at::kShort &&
            shard_miss_tokens.scalar_type() == at::kInt &&
            shard_miss_positions.scalar_type() == at::kShort &&
            shard_evictable_slots.scalar_type() == at::kShort &&
            miss_tokens.scalar_type() == at::kInt &&
            miss_counts.scalar_type() == at::kInt &&
            target_slots.scalar_type() == at::kLong &&
            request_block_table.scalar_type() == at::kInt &&
            debug_info.scalar_type() == at::kInt,
        "resident finalize-debug tensor dtypes are invalid");
    TORCH_CHECK(
        shard_packed.is_contiguous() &&
            shard_counts.is_contiguous() &&
            prior_slots.is_contiguous() &&
            shard_miss_tokens.is_contiguous() &&
            shard_miss_positions.is_contiguous() &&
            shard_evictable_slots.is_contiguous() &&
            miss_tokens.is_contiguous() &&
            miss_counts.is_contiguous() &&
            target_slots.is_contiguous() &&
            request_block_table.is_contiguous() &&
            debug_info.is_contiguous(),
        "resident finalize-debug tensors must be contiguous");
    TORCH_CHECK(
        shard_packed.dim() == 3 &&
            shard_counts.dim() == 3 &&
            prior_slots.dim() == 3 &&
            shard_miss_tokens.dim() == 3 &&
            shard_miss_positions.dim() == 3 &&
            shard_evictable_slots.dim() == 3 &&
            miss_tokens.dim() == 2 &&
            miss_counts.dim() == 2 &&
            target_slots.dim() == 2 &&
            request_block_table.dim() == 2 &&
            debug_info.dim() == 2,
        "resident finalize-debug tensor ranks are invalid");

    const int64_t request_count = shard_packed.size(0);
    const int64_t shard_count = shard_packed.size(1);
    const int64_t capacity = shard_packed.size(2);
    const int64_t shard_count_stride = shard_counts.size(2);
    const int64_t shard_count_request_stride =
        shard_counts.size(1) * shard_count_stride;
    const int64_t miss_count_stride = miss_counts.size(1);
    const int64_t block_table_width = request_block_table.size(1);
    TORCH_CHECK(
        request_count > 0 &&
            (shard_count == 2 || shard_count == 4) &&
            capacity > 0 &&
            prior_slots.sizes() == shard_packed.sizes() &&
            shard_miss_tokens.sizes() == shard_packed.sizes() &&
            shard_miss_positions.sizes() == shard_packed.sizes() &&
            shard_evictable_slots.sizes() == shard_packed.sizes() &&
            shard_counts.size(0) == request_count &&
            shard_counts.size(1) == shard_count &&
            shard_count_stride >= 16 &&
            miss_tokens.size(0) == request_count &&
            miss_tokens.size(1) == capacity &&
            target_slots.sizes() == miss_tokens.sizes() &&
            miss_counts.size(0) == request_count &&
            miss_count_stride >= 16 &&
            request_block_table.size(0) >= request_count &&
            debug_info.size(0) == request_count &&
            debug_info.size(1) == 16 &&
            block_size > 0 &&
            block_table_width * block_size >= capacity &&
            debug_stage >= 0 && debug_stage <= 11,
        "resident finalize-debug tensor shapes are invalid");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* packed_ptr = shard_packed.data_ptr();
    void* count_ptr = shard_counts.data_ptr();
    void* prior_ptr = prior_slots.data_ptr();
    void* shard_miss_token_ptr = shard_miss_tokens.data_ptr();
    void* shard_miss_position_ptr = shard_miss_positions.data_ptr();
    void* shard_evictable_slot_ptr = shard_evictable_slots.data_ptr();
    void* miss_token_ptr = miss_tokens.data_ptr();
    void* miss_count_ptr = miss_counts.data_ptr();
    void* target_ptr = target_slots.data_ptr();
    void* block_table_ptr = request_block_table.data_ptr();
    void* debug_ptr = debug_info.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_resident_sorted_finalize_debug_");
    cmd.SetCustomHandler([
        stream, packed_ptr, count_ptr, prior_ptr,
        shard_miss_token_ptr, shard_miss_position_ptr,
        shard_evictable_slot_ptr,
        miss_token_ptr, miss_count_ptr,
        target_ptr, block_table_ptr, debug_ptr,
        request_count, shard_count, capacity,
        shard_count_stride, shard_count_request_stride,
        miss_count_stride, block_table_width,
        block_size, debug_stage]() -> int {
        dsa_resident_sorted_finalize_debug_impl(
            stream, packed_ptr, count_ptr, prior_ptr,
            shard_miss_token_ptr, shard_miss_position_ptr,
            shard_evictable_slot_ptr,
            miss_token_ptr, miss_count_ptr,
            target_ptr, block_table_ptr, debug_ptr,
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(shard_count),
            static_cast<uint32_t>(capacity),
            static_cast<uint32_t>(shard_count_stride),
            static_cast<uint32_t>(shard_count_request_stride),
            static_cast<uint32_t>(miss_count_stride),
            static_cast<uint32_t>(block_table_width),
            static_cast<uint32_t>(block_size),
            static_cast<uint32_t>(debug_stage));
        return 0;
    });
    cmd.Run();
    return miss_counts;
}

at::Tensor npu_dsa_staged_unique_finalize_(
    const at::Tensor &unique_keys,
    const at::Tensor &inverse,
    const at::Tensor &row_req_indices,
    at::Tensor &selected_packed,
    at::Tensor &local_to_union,
    at::Tensor &selected_count,
    const at::Tensor &request_block_table,
    at::Tensor &target_slots,
    int64_t block_size,
    int64_t packed_key_stride)
{
    TORCH_CHECK(unique_keys.is_privateuseone(),
                "unique_keys must be on an NPU device");
    const auto device = unique_keys.device();
    TORCH_CHECK(inverse.device() == device &&
                    row_req_indices.device() == device &&
                    selected_packed.device() == device &&
                    local_to_union.device() == device &&
                    selected_count.device() == device &&
                    request_block_table.device() == device &&
                    target_slots.device() == device,
                "all unique-finalize tensors must share one NPU device");
    TORCH_CHECK(unique_keys.scalar_type() == at::kInt &&
                    inverse.scalar_type() == at::kLong &&
                    row_req_indices.scalar_type() == at::kInt &&
                    selected_packed.scalar_type() == at::kInt &&
                    local_to_union.scalar_type() == at::kInt &&
                    selected_count.scalar_type() == at::kInt &&
                    request_block_table.scalar_type() == at::kInt &&
                    target_slots.scalar_type() == at::kLong,
                "unique finalize requires int32 keys/outputs and int64 inverse/slots");
    TORCH_CHECK(unique_keys.is_contiguous() && inverse.is_contiguous() &&
                    row_req_indices.is_contiguous() &&
                    selected_packed.is_contiguous() &&
                    local_to_union.is_contiguous() &&
                    selected_count.is_contiguous() &&
                    request_block_table.is_contiguous() &&
                    target_slots.is_contiguous(),
                "all unique-finalize tensors must be contiguous");
    TORCH_CHECK(unique_keys.dim() == 1 && inverse.dim() == 1 &&
                    row_req_indices.dim() == 1 &&
                    selected_packed.dim() == 2 &&
                    local_to_union.dim() == 2 &&
                    selected_count.dim() == 2 &&
                    request_block_table.dim() == 2 &&
                    target_slots.dim() == 2,
                "invalid unique-finalize tensor ranks");
    const int64_t row_count = row_req_indices.numel();
    const int64_t request_count = selected_packed.size(0);
    const int64_t row_width = local_to_union.size(1);
    const int64_t scratch_capacity = selected_packed.size(1);
    const int64_t selected_count_stride =
        selected_count.numel() / request_count;
    TORCH_CHECK(row_count > 0 && request_count > 0 &&
                    row_width == 2048 &&
                    inverse.numel() == row_count * row_width &&
                    local_to_union.size(0) == row_count,
                "unique finalize requires flattened inverse for [rows,2048]");
    TORCH_CHECK(request_block_table.size(0) == request_count &&
                    selected_count.size(0) == request_count &&
                    target_slots.sizes() == selected_packed.sizes(),
                "unique-finalize request buffers have inconsistent shapes");
    TORCH_CHECK(scratch_capacity >= 2 * row_width &&
                    block_size > 0 &&
                    (block_size & (block_size - 1)) == 0 &&
                    request_block_table.size(1) * block_size >=
                        scratch_capacity,
                "unique-finalize buffers require power-of-two block size");
    TORCH_CHECK(request_count == 1 || selected_count_stride >= 8,
                "batched unique finalize requires isolated count writes");
    TORCH_CHECK(packed_key_stride > 0 &&
                    request_count * packed_key_stride <=
                        std::numeric_limits<int32_t>::max(),
                "packed unique keys must fit int32");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* keys_ptr = unique_keys.data_ptr();
    void* inverse_ptr = inverse.data_ptr();
    void* requests_ptr = row_req_indices.data_ptr();
    void* packed_ptr = selected_packed.data_ptr();
    void* map_ptr = local_to_union.data_ptr();
    void* count_ptr = selected_count.data_ptr();
    void* table_ptr = request_block_table.data_ptr();
    void* slots_ptr = target_slots.data_ptr();
    const int64_t unique_count = unique_keys.numel();
    const int64_t table_width = request_block_table.size(1);
    const uint32_t block_size_shift = static_cast<uint32_t>(
        __builtin_ctzll(static_cast<unsigned long long>(block_size)));

    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_staged_unique_finalize_");
    cmd.SetCustomHandler([
        stream, keys_ptr, inverse_ptr, requests_ptr, packed_ptr, map_ptr,
        count_ptr, table_ptr, slots_ptr, unique_count, row_count,
        row_width, request_count, scratch_capacity, table_width,
        selected_count_stride, block_size, block_size_shift,
        packed_key_stride]() -> int {
        dsa_staged_unique_finalize_impl(
            stream, keys_ptr, inverse_ptr, requests_ptr, packed_ptr,
            map_ptr, count_ptr, table_ptr, slots_ptr,
            static_cast<uint32_t>(unique_count),
            static_cast<uint32_t>(row_count),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(scratch_capacity),
            static_cast<uint32_t>(table_width),
            static_cast<uint32_t>(selected_count_stride),
            static_cast<uint32_t>(block_size),
            block_size_shift,
            static_cast<uint32_t>(packed_key_stride));
        return 0;
    });
    cmd.Run();
    return selected_count;
}

at::Tensor npu_dsa_staged_copy_rows_(
    at::Tensor &output,
    const at::Tensor &local_indices)
{
    TORCH_CHECK(output.is_privateuseone() &&
                    local_indices.device() == output.device(),
                "staged copy tensors must share one NPU device");
    TORCH_CHECK(output.scalar_type() == at::kInt &&
                    local_indices.scalar_type() == at::kInt &&
                    output.is_contiguous() &&
                    local_indices.is_contiguous(),
                "staged copy tensors must be contiguous int32");
    TORCH_CHECK(output.sizes() == local_indices.sizes(),
                "staged copy tensors must have identical shapes");
    TORCH_CHECK(output.dim() == 2 || output.dim() == 3,
                "staged copy tensors must be [rows,k] or [rows,1,k]");
    const int64_t rows = output.size(0);
    const int64_t width = output.numel() / rows;
    const c10_npu::OptionalNPUGuard npu_guard(output.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* output_ptr = output.data_ptr();
    void* local_indices_ptr = local_indices.data_ptr();
    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_staged_copy_rows_");
    cmd.SetCustomHandler([
        stream, output_ptr, local_indices_ptr, rows, width]() -> int {
        dsa_staged_copy_rows_impl(
            stream, output_ptr, local_indices_ptr,
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(width));
        return 0;
    });
    cmd.Run();
    return output;
}

at::Tensor npu_dsa_prepare_sparse_indices_(
    at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    const at::Tensor &row_req_indices,
    const at::Tensor &request_block_table,
    at::Tensor &selected_packed,
    at::Tensor &selected_counts,
    at::Tensor &target_slots,
    int64_t block_size,
    bool need_packed,
    bool clear_invalid_rows)
{
    TORCH_CHECK(topk_indices.is_privateuseone(),
                "topk_indices must be on an NPU device");
    const auto device = topk_indices.device();
    TORCH_CHECK(split_boundary.device() == device &&
                    row_req_indices.device() == device &&
                    request_block_table.device() == device &&
                    selected_packed.device() == device &&
                    selected_counts.device() == device &&
                    target_slots.device() == device,
                "all sparse-index preparation tensors must share one NPU device");
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt &&
                    split_boundary.scalar_type() == at::kInt &&
                    row_req_indices.scalar_type() == at::kInt &&
                    request_block_table.scalar_type() == at::kInt &&
                    selected_packed.scalar_type() == at::kInt &&
                    selected_counts.scalar_type() == at::kInt &&
                    target_slots.scalar_type() == at::kLong,
                "sparse indices/counts/tables must be int32 and target slots int64");
    TORCH_CHECK(topk_indices.is_contiguous() && split_boundary.is_contiguous() &&
                    row_req_indices.is_contiguous() &&
                    request_block_table.is_contiguous() &&
                    selected_packed.is_contiguous() &&
                    selected_counts.is_contiguous() &&
                    target_slots.is_contiguous(),
                "all sparse-index preparation tensors must be contiguous");
    TORCH_CHECK(topk_indices.dim() == 2 ||
                    (topk_indices.dim() == 3 && topk_indices.size(1) == 1),
                "topk_indices must have shape [rows,k] or [rows,1,k]");
    TORCH_CHECK(request_block_table.dim() == 2 && selected_packed.dim() == 2 &&
                    target_slots.sizes() == selected_packed.sizes(),
                "request tables and preallocated payload buffers must be 2D");
    const int64_t row_count = topk_indices.size(0);
    TORCH_CHECK(row_count > 0, "topk_indices must contain at least one row");
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t request_count = request_block_table.size(0);
    const int64_t block_table_width = request_block_table.size(1);
    TORCH_CHECK(request_count > 0, "request block table must contain a request");
    TORCH_CHECK(split_boundary.numel() >= row_count &&
                    row_req_indices.numel() >= row_count,
                "split_boundary and row_req_indices must cover every row");
    TORCH_CHECK(selected_packed.size(0) == request_count &&
                    selected_counts.dim() == 2 &&
                    selected_counts.size(0) == request_count &&
                    selected_counts.size(1) >= 16 &&
                    selected_counts.size(1) % 16 == 0,
                "preallocated payload rows must match request block-table rows");
    TORCH_CHECK(block_size > 0 && row_width > 0 && row_width <= 4096,
                "block_size and sparse row width must be supported and positive");
    TORCH_CHECK(row_width % 16 == 0,
                "sparse row width must be a multiple of 16 int32 values so "
                "different request cores never update the same 64-byte cacheline");
    const int64_t scratch_capacity = selected_packed.size(1);
    const int64_t selected_count_stride = selected_counts.size(1);
    TORCH_CHECK(scratch_capacity >= row_width,
                "selected payload width must cover one sparse row");
    const int64_t bitmap_words =
        (block_table_width * block_size + 31) / 32;
    TORCH_CHECK(request_block_table.size(1) * block_size >= scratch_capacity,
                "request block table is too short for the fixed scratch prefix");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* boundary_ptr = split_boundary.data_ptr();
    void* row_req_ptr = row_req_indices.data_ptr();
    void* table_ptr = request_block_table.data_ptr();
    void* packed_ptr = selected_packed.data_ptr();
    void* counts_ptr = selected_counts.data_ptr();
    void* slots_ptr = target_slots.data_ptr();

    at_npu::native::OpCommand cmd;
    cmd.Name("npu_dsa_prepare_sparse_indices_");
    cmd.SetCustomHandler([
        stream, topk_ptr, boundary_ptr, row_req_ptr, table_ptr, packed_ptr,
        counts_ptr, slots_ptr, row_count, row_width, request_count,
        block_table_width, scratch_capacity, selected_count_stride,
        bitmap_words, block_size,
        need_packed, clear_invalid_rows]() -> int {
        dsa_prepare_sparse_indices_impl(
            stream, topk_ptr, boundary_ptr, row_req_ptr, table_ptr, packed_ptr,
            counts_ptr, slots_ptr,
            static_cast<uint32_t>(row_count),
            static_cast<uint32_t>(row_width),
            static_cast<uint32_t>(request_count),
            static_cast<uint32_t>(block_table_width),
            static_cast<uint32_t>(scratch_capacity),
            static_cast<uint32_t>(selected_count_stride),
            static_cast<uint32_t>(bitmap_words),
            static_cast<uint32_t>(block_size), need_packed,
            clear_invalid_rows);
        return 0;
    });
    cmd.Run();
    return selected_counts;
}

at::Tensor npu_dsa_prepare_sparse_indices_staged_common_(
    at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    const at::Tensor &row_req_indices,
    const at::Tensor &request_block_table,
    at::Tensor &selected_packed,
    at::Tensor &selected_counts,
    at::Tensor &target_slots,
    at::Tensor &local_to_union_workspace,
    at::Tensor *shard_packed_workspace,
    at::Tensor *shard_mapping_workspace,
    at::Tensor *shard_counts_workspace,
    int64_t block_size,
    int64_t mtp,
    bool need_packed,
    bool clear_invalid_rows,
    bool use_sharded_sort)
{
    TORCH_CHECK(mtp == 1 || mtp == 2,
                "staged sparse-index preparation only supports MTP=1 or "
                "MTP=2; got MTP=", mtp);
    TORCH_CHECK(topk_indices.is_privateuseone(),
                "topk_indices must be on an NPU device");
    const auto device = topk_indices.device();
    TORCH_CHECK(split_boundary.device() == device &&
                    row_req_indices.device() == device &&
                    request_block_table.device() == device &&
                    selected_packed.device() == device &&
                    selected_counts.device() == device &&
                    target_slots.device() == device &&
                    local_to_union_workspace.device() == device,
                "all staged sparse-index tensors must share one NPU device");
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt &&
                    split_boundary.scalar_type() == at::kInt &&
                    row_req_indices.scalar_type() == at::kInt &&
                    request_block_table.scalar_type() == at::kInt &&
                    selected_packed.scalar_type() == at::kInt &&
                    selected_counts.scalar_type() == at::kInt &&
                    local_to_union_workspace.scalar_type() == at::kInt &&
                    target_slots.scalar_type() == at::kLong,
                "staged sparse indices/counts/tables must be int32 and "
                "target slots int64");
    TORCH_CHECK(topk_indices.is_contiguous() &&
                    split_boundary.is_contiguous() &&
                    row_req_indices.is_contiguous() &&
                    request_block_table.is_contiguous() &&
                    selected_packed.is_contiguous() &&
                    selected_counts.is_contiguous() &&
                    target_slots.is_contiguous() &&
                    local_to_union_workspace.is_contiguous(),
                "all staged sparse-index tensors must be contiguous");
    TORCH_CHECK(topk_indices.dim() == 2 ||
                    (topk_indices.dim() == 3 &&
                     topk_indices.size(1) == 1),
                "topk_indices must have shape [rows,k] or [rows,1,k]");
    TORCH_CHECK(request_block_table.dim() == 2 &&
                    selected_packed.dim() == 2 &&
                    local_to_union_workspace.sizes() ==
                        selected_packed.sizes() &&
                    target_slots.sizes() == selected_packed.sizes(),
                "staged payload and workspace buffers must be matching 2D "
                "tensors");

    const int64_t row_count = topk_indices.size(0);
    const int64_t request_count = request_block_table.size(0);
    TORCH_CHECK(row_count > 0 && request_count > 0,
                "staged sparse-index preparation requires non-empty rows "
                "and requests");
    const int64_t row_width = topk_indices.numel() / row_count;
    const int64_t scratch_capacity = selected_packed.size(1);
    const int64_t block_table_width = request_block_table.size(1);
    TORCH_CHECK(row_count == request_count * mtp,
                "fixed staged layout requires exactly MTP rows per request");
    TORCH_CHECK(row_width == 2048,
                "staged sort union currently requires top-k width 2048");
    TORCH_CHECK(split_boundary.numel() >= row_count &&
                    row_req_indices.numel() >= row_count,
                "split_boundary and row_req_indices must cover every row");
    TORCH_CHECK(selected_packed.size(0) == request_count &&
                    selected_counts.dim() == 2 &&
                    selected_counts.size(0) == request_count &&
                    selected_counts.size(1) >= 16 &&
                    selected_counts.size(1) % 16 == 0,
                "staged payload/count rows must match request rows and each "
                "count row must occupy whole 64-byte cachelines");
    TORCH_CHECK(scratch_capacity >= mtp * row_width,
                "staged payload/workspace width must cover all MTP rows");
    if (use_sharded_sort) {
        TORCH_CHECK(shard_packed_workspace != nullptr &&
                        shard_mapping_workspace != nullptr &&
                        shard_counts_workspace != nullptr,
                    "the production sharded-sort path requires caller-owned "
                    "shard workspaces");
        TORCH_CHECK(shard_packed_workspace->device() == device &&
                        shard_mapping_workspace->device() == device &&
                        shard_counts_workspace->device() == device,
                    "all production sharded-sort workspaces must share the "
                    "input NPU device");
        TORCH_CHECK(shard_packed_workspace->scalar_type() == at::kInt &&
                        shard_mapping_workspace->scalar_type() == at::kInt &&
                        shard_counts_workspace->scalar_type() == at::kInt,
                    "all production sharded-sort workspaces must be int32");
        TORCH_CHECK(shard_packed_workspace->is_contiguous() &&
                        shard_mapping_workspace->is_contiguous() &&
                        shard_counts_workspace->is_contiguous(),
                    "all production sharded-sort workspaces must be "
                    "contiguous");
        if (mtp == 2) {
            TORCH_CHECK(scratch_capacity == mtp * row_width,
                        "the production sharded-sort path requires an exact "
                        "request-width payload/workspace");
            TORCH_CHECK(
                shard_packed_workspace->dim() == 3 &&
                    shard_packed_workspace->size(0) == request_count &&
                    shard_packed_workspace->size(1) == mtp &&
                    shard_packed_workspace->size(2) == scratch_capacity &&
                    shard_mapping_workspace->dim() == 3 &&
                    shard_mapping_workspace->sizes() ==
                        shard_packed_workspace->sizes() &&
                    shard_counts_workspace->dim() == 3 &&
                    shard_counts_workspace->size(0) == request_count &&
                    shard_counts_workspace->size(1) == mtp &&
                    shard_counts_workspace->size(2) == 16,
                "invalid production sharded-sort workspace shapes");
        }
        TORCH_CHECK(
            reinterpret_cast<std::uintptr_t>(
                shard_packed_workspace->data_ptr()) % 256 == 0 &&
                reinterpret_cast<std::uintptr_t>(
                    shard_mapping_workspace->data_ptr()) % 256 == 0 &&
                reinterpret_cast<std::uintptr_t>(
                    shard_counts_workspace->data_ptr()) % 64 == 0,
            "production sharded-sort vector workspaces must be 256-byte "
            "aligned and count workspaces must be 64-byte aligned");
    }
    TORCH_CHECK(block_size > 0 &&
                    (block_size & (block_size - 1)) == 0 &&
                    block_table_width * block_size >= scratch_capacity,
                "block_size must be a positive power of two and the request "
                "block table must cover the fixed scratch prefix");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(topk_indices.data_ptr()) % 256 == 0 &&
            reinterpret_cast<std::uintptr_t>(selected_packed.data_ptr()) %
                    256 ==
                0 &&
            reinterpret_cast<std::uintptr_t>(
                local_to_union_workspace.data_ptr()) %
                    256 ==
                0 &&
            reinterpret_cast<std::uintptr_t>(target_slots.data_ptr()) % 256 ==
                0 &&
            reinterpret_cast<std::uintptr_t>(selected_counts.data_ptr()) %
                    64 ==
                0,
        "staged row buffers must be 256-byte aligned and the count buffer "
        "must be 64-byte aligned");

    const c10_npu::OptionalNPUGuard npu_guard(device);
    static thread_local int32_t cached_device = -1;
    static thread_local int64_t cached_aiv_count = 0;
    const int32_t current_device =
        static_cast<int32_t>(topk_indices.get_device());
    if (current_device != cached_device || cached_aiv_count <= 0) {
        TORCH_CHECK(
            aclGetDeviceCapability(
                current_device,
                ACL_DEVICE_INFO_VECTOR_CORE_NUM,
                &cached_aiv_count) == ACL_SUCCESS,
            "failed to query the NPU vector core count");
        cached_device = current_device;
    }
    TORCH_CHECK(cached_aiv_count > 0,
                "NPU reported no available vector cores");
    const uint32_t core_count = static_cast<uint32_t>(
        std::min<int64_t>(cached_aiv_count, row_count));
    const int64_t selected_count_stride = selected_counts.size(1);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    void* topk_ptr = topk_indices.data_ptr();
    void* boundary_ptr = split_boundary.data_ptr();
    void* row_req_ptr = row_req_indices.data_ptr();
    void* table_ptr = request_block_table.data_ptr();
    void* packed_ptr = selected_packed.data_ptr();
    void* counts_ptr = selected_counts.data_ptr();
    void* slots_ptr = target_slots.data_ptr();
    void* map_ptr = local_to_union_workspace.data_ptr();
    void* shard_packed_ptr =
        use_sharded_sort ? shard_packed_workspace->data_ptr() : nullptr;
    void* shard_mapping_ptr =
        use_sharded_sort ? shard_mapping_workspace->data_ptr() : nullptr;
    void* shard_counts_ptr =
        use_sharded_sort ? shard_counts_workspace->data_ptr() : nullptr;

    at_npu::native::OpCommand cmd;
    cmd.Name(
        use_sharded_sort
            ? "npu_dsa_prepare_sparse_indices_sharded_"
            : "npu_dsa_prepare_sparse_indices_staged_");
    cmd.SetCustomHandler([
        stream, topk_ptr, boundary_ptr, row_req_ptr, table_ptr, packed_ptr,
        counts_ptr, slots_ptr, map_ptr, shard_packed_ptr,
        shard_mapping_ptr, shard_counts_ptr, row_count, row_width,
        request_count, mtp, scratch_capacity, block_table_width,
        selected_count_stride, block_size, core_count, need_packed,
        clear_invalid_rows, use_sharded_sort]() -> int {
        if (use_sharded_sort) {
            dsa_prepare_sparse_indices_sharded_impl(
                stream, topk_ptr, boundary_ptr, row_req_ptr, table_ptr,
                packed_ptr, counts_ptr, slots_ptr, map_ptr,
                shard_packed_ptr, shard_mapping_ptr, shard_counts_ptr,
                static_cast<uint32_t>(request_count),
                static_cast<uint32_t>(mtp),
                static_cast<uint32_t>(row_width),
                static_cast<uint32_t>(scratch_capacity),
                static_cast<uint32_t>(block_table_width),
                static_cast<uint32_t>(selected_count_stride),
                static_cast<uint32_t>(block_size), core_count, need_packed,
                clear_invalid_rows);
        } else {
            dsa_prepare_sparse_indices_staged_impl(
                stream, topk_ptr, boundary_ptr, row_req_ptr, table_ptr,
                packed_ptr, counts_ptr, slots_ptr, map_ptr,
                static_cast<uint32_t>(row_count),
                static_cast<uint32_t>(row_width),
                static_cast<uint32_t>(request_count),
                static_cast<uint32_t>(mtp),
                static_cast<uint32_t>(scratch_capacity),
                static_cast<uint32_t>(block_table_width),
                static_cast<uint32_t>(selected_count_stride),
                static_cast<uint32_t>(block_size), core_count, need_packed,
                clear_invalid_rows);
        }
        return 0;
    });
    cmd.Run();
    return selected_counts;
}

at::Tensor npu_dsa_prepare_sparse_indices_staged_(
    at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    const at::Tensor &row_req_indices,
    const at::Tensor &request_block_table,
    at::Tensor &selected_packed,
    at::Tensor &selected_counts,
    at::Tensor &target_slots,
    at::Tensor &local_to_union_workspace,
    int64_t block_size,
    int64_t mtp,
    bool need_packed,
    bool clear_invalid_rows)
{
    return npu_dsa_prepare_sparse_indices_staged_common_(
        topk_indices, split_boundary, row_req_indices, request_block_table,
        selected_packed, selected_counts, target_slots,
        local_to_union_workspace, nullptr, nullptr, nullptr,
        block_size, mtp, need_packed,
        clear_invalid_rows, false);
}

at::Tensor npu_dsa_prepare_sparse_indices_sharded_(
    at::Tensor &topk_indices,
    const at::Tensor &split_boundary,
    const at::Tensor &row_req_indices,
    const at::Tensor &request_block_table,
    at::Tensor &selected_packed,
    at::Tensor &selected_counts,
    at::Tensor &target_slots,
    at::Tensor &local_to_union_workspace,
    at::Tensor &shard_packed_workspace,
    at::Tensor &shard_mapping_workspace,
    at::Tensor &shard_counts_workspace,
    int64_t block_size,
    int64_t mtp,
    bool need_packed,
    bool clear_invalid_rows)
{
    return npu_dsa_prepare_sparse_indices_staged_common_(
        topk_indices, split_boundary, row_req_indices, request_block_table,
        selected_packed, selected_counts, target_slots,
        local_to_union_workspace, &shard_packed_workspace,
        &shard_mapping_workspace, &shard_counts_workspace,
        block_size, mtp, need_packed,
        clear_invalid_rows, true);
}

void bgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0) && x.size(0) == indices.size(0),
                "the first dimension of x, y, indices should be same");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* indices_ptr = indices.data_ptr();
    int indices_size = indices.size(0);
    void* y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size, input_hidden_token,
                          lora_rank, scale_f]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        bgmv_shrink_impl(dtype, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size, num_tokens_per_core,
                         input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor bgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y,
                       int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0) && x.size(0) == indices.size(0),
                "the first dimension of x, y, indices should be same");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* indices_ptr = indices.data_ptr();
    int indices_size = indices.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, y_out_ptr, batch_size, lora_rank,
                          slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        bgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, y_out_ptr, batch_size,
                         num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}

void sgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                 at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size,
                          seq_len_ptr, seq_len_size, y_ptr,
                          batch_size, input_hidden_token, lora_rank, scale_f]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_shrink_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size,
                         y_ptr, batch_size,
                         num_tokens_per_core, input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor sgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                          batch_size, lora_rank, slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                         batch_size, num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> dispatch_prefill(
    const at::Tensor& x, const at::Tensor& topk_idx, const at::Tensor& topk_weights,
    const at::Tensor& num_tokens_per_rank, const at::Tensor& is_token_in_rank, at::Tensor& num_tokens_per_expert,
    int64_t num_worst_tokens, c10::string_view groupEp, int64_t rank, int64_t num_ranks) {
    std::vector<char> group_ep_chrs(groupEp.begin(), groupEp.end());
    group_ep_chrs.push_back('\0');
    char* group_ep_ptr = &group_ep_chrs[0];
    at::Tensor new_x = x;

    // Type checks
    TORCH_BIND_ASSERT(is_token_in_rank.scalar_type() == at::kBool);
    TORCH_BIND_ASSERT(num_tokens_per_expert.scalar_type() == at::kInt);
    TORCH_BIND_ASSERT(num_tokens_per_rank.scalar_type() == at::kInt);

    // Shape and contiguous checks
    TORCH_BIND_ASSERT(new_x.dim() == 2 and new_x.is_contiguous());
    // TORCH_BIND_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    TORCH_BIND_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    TORCH_BIND_ASSERT(is_token_in_rank.size(0) == new_x.size(0) and is_token_in_rank.size(1) == num_ranks);
    TORCH_BIND_ASSERT(num_tokens_per_expert.dim() == 1 and num_tokens_per_expert.is_contiguous());
    TORCH_BIND_ASSERT(num_tokens_per_expert.size(0) % num_ranks == 0);
    TORCH_BIND_ASSERT(num_tokens_per_rank.dim() == 1 and num_tokens_per_rank.is_contiguous());
    TORCH_BIND_ASSERT(num_tokens_per_rank.size(0) == num_ranks);

    auto num_tokens = static_cast<int>(new_x.size(0));
    auto hidden = static_cast<int>(new_x.size(1));
    auto num_experts = static_cast<int64_t>(num_tokens_per_expert.size(0));
    auto num_local_experts = static_cast<int>(num_experts / num_ranks);

    // Top-k checks
    int num_topk = 0;
    num_topk = static_cast<int>(topk_idx.size(1));
    TORCH_BIND_ASSERT(num_experts > 0);
    TORCH_BIND_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    TORCH_BIND_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    TORCH_BIND_ASSERT(num_tokens == topk_idx.size(0));
    TORCH_BIND_ASSERT(num_topk == topk_weights.size(1));
    TORCH_BIND_ASSERT(topk_weights.scalar_type() == at::kFloat);

    int send_per_group = 3;  // (send_to_expert_num, send_to_expert_offset, send_rank_tokens)

    auto send_data = at::empty({num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));
    int64_t send_count = send_per_group * num_local_experts * num_ranks;

    auto send_data_offset = at::empty({num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_data = at::empty({num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));

    int64_t local_rank_size = num_ranks;
    int64_t local_rank_id = rank % local_rank_size;

    EXEC_NPU_CMD(aclnnNotifyDispatch,
        send_data,
        num_tokens_per_expert, 
        send_count,
        num_tokens,
        group_ep_ptr,  // commGroup
        num_ranks,     // rankSize
        rank,          // rankId
        local_rank_size,
        local_rank_id,
        send_data_offset,
        recv_data);

    auto options_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    std::vector<int32_t> local_expert_acc(num_experts, 0);
    auto send_token_idx_cpu = at::empty({num_tokens, num_topk}, options_cpu);
    auto send_token_idx_ptr = send_token_idx_cpu.data_ptr<int>();

    auto topk_idx_cpu = topk_idx.to(at::kCPU);
    auto topk_idx_ptr = topk_idx_cpu.data_ptr<int64_t>();
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < num_topk; ++j) {
            int64_t expert_idx = topk_idx_ptr[i * num_topk + j];
            if (expert_idx >= 0) {
                int32_t cnt = local_expert_acc[expert_idx];
                send_token_idx_ptr[i * num_topk + j] = cnt;
                local_expert_acc[expert_idx]++;
            }
        }
    }

    TORCH_BIND_ASSERT(recv_data.dim() == 1 and recv_data.is_contiguous());
    TORCH_BIND_ASSERT(recv_data.size(0) % num_experts == 0);
    at::Tensor recv_offset_cpu = at::empty({num_experts}, options_cpu);
    at::Tensor recv_count_cpu = at::empty({num_experts}, options_cpu);
    auto recv_data_cpu = recv_data.to(at::kCPU);
    auto recv_data_ptr = recv_data_cpu.data_ptr<int>();
    auto recv_count_ptr = recv_count_cpu.data_ptr<int>();
    auto recv_offset_ptr = recv_offset_cpu.data_ptr<int>();
    int64_t total_recv_tokens = 0;
    int64_t num_max_dispatch_tokens_per_rank = 0;
    std::vector<int64_t> num_recv_tokens_per_expert_list;

    for (int64_t local_e = 0; local_e < num_local_experts; ++local_e) {
        int64_t local_expert_recv_tokens = 0;
        for (int64_t src_rank = 0; src_rank < num_ranks; ++src_rank) {
            int64_t index = local_e * num_ranks + src_rank;
            int64_t pair_idx = send_per_group * (src_rank * num_local_experts + local_e);

            int recv_cnt = recv_data_ptr[pair_idx];                 // count from this src_rank for
                                                                    // this global_expert
            int recv_off = recv_data_ptr[pair_idx + 1];             // offset in that src_rank's window
            int64_t send_num_tokens = recv_data_ptr[pair_idx + 2];  // all bs from rank

            total_recv_tokens += recv_cnt;
            recv_count_ptr[index] = total_recv_tokens;
            recv_offset_ptr[index] = recv_off;
            num_max_dispatch_tokens_per_rank = std::max(num_max_dispatch_tokens_per_rank, send_num_tokens);

            local_expert_recv_tokens += recv_cnt;
        }
        num_recv_tokens_per_expert_list.push_back(local_expert_recv_tokens);
    }
    auto option = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    at::Tensor num_recv_tokens_per_expert = torch::from_blob(
        num_recv_tokens_per_expert_list.data(), {static_cast<int64_t>(num_recv_tokens_per_expert_list.size())}, option)
        .clone();

    at::Tensor expert_ids = topk_idx.to(at::kInt);
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t quant_mode = 0;
    int64_t global_bs = static_cast<int64_t>(
        std::max(num_max_dispatch_tokens_per_rank * num_ranks, static_cast<int64_t>(num_worst_tokens)));

    auto send_token_idx = send_token_idx_cpu.to(x.device());
    auto recv_offset = recv_offset_cpu.to(x.device());
    auto recv_count = recv_count_cpu.to(x.device());

    int total_cnt = total_recv_tokens;
    if (total_cnt == 0) {
        total_cnt = 1;
    }
    auto expandx_out = at::empty({total_cnt, hidden}, x.options());
    auto dynamic_scales_out = at::empty({total_cnt}, at::dtype(at::kFloat).device(x.device()));
    auto expand_idx_out = at::empty({total_cnt * 3}, at::dtype(at::kInt).device(x.device()));

    EXEC_NPU_CMD(aclnnMoeDispatchNormal,
        new_x,
        expert_ids,
        send_data_offset,
        send_token_idx,
        recv_offset,
        recv_count,
        group_ep_ptr,  // commGroup
        num_ranks,     // rankSize
        rank,          // rankId
        group_ep_ptr,
        tp_size,
        tp_rank,
        num_experts,
        quant_mode,
        global_bs,
        expandx_out,
        dynamic_scales_out,
        expand_idx_out);

    // Return values
    return {expandx_out, expand_idx_out, recv_count, num_recv_tokens_per_expert};
}

std::tuple<at::Tensor, at::Tensor> npu_gemma_rms_norm(
    const at::Tensor& x,
    const at::Tensor& gamma,
    double epsilon)
{
    int64_t dim_x = x.dim();
    int64_t dim_gamma = gamma.dim();
    int64_t diff = dim_x - dim_gamma;
    std::vector<int64_t> new_shape;
    at::Tensor rstd;
    if (diff > 0) {
        new_shape.reserve(dim_x);
        auto x_sizes = x.sizes();
        for (int64_t i = 0; i < diff; ++i) {
            new_shape.push_back(x_sizes[i]);
        }
        for (int64_t i = 0; i < dim_gamma; ++i) {
            new_shape.push_back(1);
        }
    } else {
        new_shape.assign(dim_x, 1);
    }
    rstd = at::empty(new_shape, x.options().dtype(at::kFloat));
    at::Tensor y = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnGemmaRmsNorm, x, gamma, epsilon, y, rstd);
    return std::tuple<at::Tensor, at::Tensor>(y, rstd);
}

void transpose_kv_cache_by_block(
    const at::TensorList &kCache,
    const at::TensorList &vCache,
    const at::Tensor &blockIDs,
    int64_t blockSize,
    int64_t headNum,
    int64_t headDim,
    int64_t splitNum,
    int64_t layerNum)
{

    EXEC_NPU_CMD(aclnnTransposeKvCacheByBlock, kCache, vCache, blockIDs,
                 blockSize, headNum, headDim, splitNum, layerNum);

}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_copy_and_expand_eagle_inputs(
    const at::Tensor &target_token_ids,
    const at::Tensor &target_positions,
    const at::Tensor &next_token_ids,
    const at::Tensor &query_start_loc,
    const at::Tensor &query_end_loc,
    int64_t padding_token_id,
    int64_t parallel_drafting_token_id,
    int64_t num_padding_slots_per_request,
    bool shift_input_ids,
    int64_t total_draft_tokens)
{
    int64_t total_input_tokens = target_token_ids.size(0);
    int64_t num_reqs = query_start_loc.size(0) - 1;

    auto device = target_token_ids.device();
    at::Tensor out_input_ids = at::empty({total_draft_tokens}, at::dtype(at::kInt).device(device));
    at::Tensor out_positions = at::empty({total_draft_tokens}, at::dtype(at::kInt).device(device));
    at::Tensor out_is_rejected_token_mask = at::empty({total_draft_tokens}, at::dtype(at::kChar).device(device));
    at::Tensor out_is_masked_token_mask = at::empty({total_draft_tokens}, at::dtype(at::kChar).device(device));
    at::Tensor out_new_token_indices = at::empty({num_reqs * num_padding_slots_per_request}, at::dtype(at::kInt).device(device));
    at::Tensor out_hidden_state_mapping = at::empty({total_input_tokens}, at::dtype(at::kInt).device(device));

    EXEC_NPU_CMD(aclnnCopyAndExpandEagleInputs,
        target_token_ids, target_positions, next_token_ids, query_start_loc, query_end_loc,
        padding_token_id, parallel_drafting_token_id, num_padding_slots_per_request,
        shift_input_ids, total_input_tokens,
        out_input_ids, out_positions, out_is_rejected_token_mask, out_is_masked_token_mask,
        out_new_token_indices, out_hidden_state_mapping);

    return {out_input_ids, out_positions, out_is_rejected_token_mask, out_is_masked_token_mask,
            out_new_token_indices, out_hidden_state_mapping};
}

at::Tensor npu_causal_conv1d_custom(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef query_start_loc_opt,
    at::IntArrayRef cache_indices_opt,
    at::IntArrayRef initial_state_mode_opt,
    at::IntArrayRef num_accepted_tokens_opt,
    int64_t  activation_mode,
    int64_t  pad_slot_id,
    int64_t  run_mode)
{
    at::Tensor output = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnCausalConv1d,
                    x,
                    weight,
                    bias_opt,
                    conv_state,
                    query_start_loc_opt,
                    cache_indices_opt,
                    initial_state_mode_opt,
                    num_accepted_tokens_opt,
                    activation_mode,
                    pad_slot_id,
                    run_mode,
                    output
                );

    return output;
}
  
// It is expected that further improvements will be made after it is incorporated into CANN on June 30th.
std::vector<at::Tensor> moe_grouped_matmul(
    at::Tensor x,
    at::Tensor weight,
    const at::Tensor& group_list,
    int64_t split_item,
    int64_t group_type,
    int64_t group_list_type
)
{
    bool transpose_weight = false;
    bool weight_nz = true;

    at::TensorList x_list = at::TensorList(x);
    at::TensorList weight_list = at::TensorList(weight);
    std::vector<at::Tensor> y;
    c10::TensorOptions options = x_list[0].options().dtype(x[0].scalar_type());
    auto m = x_list[0].sizes()[0];
    auto n = weight_list[0].sizes()[1];
    if (!transpose_weight) {
        n = weight_list[0].sizes()[2];
    }
    at::Tensor y_0 = at::empty(at::IntArrayRef{m, n}, options);
    y.emplace_back(y_0);
    at::TensorList result = at::TensorList(y);

    EXEC_NPU_CMD(aclnnMoeGroupedMatmulWeightNz,
                x_list, weight_list, group_list, transpose_weight, result);

    return y;
}

} // namespace vllm_ascend

TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{

    // vLLM-Ascend custom ops
    // Gemma RmsNorm
    ops.def(
        "npu_gemma_rms_norm(Tensor x, "
                            "Tensor gamma, "
                            "float epsilon=1e-6)"
        "-> (Tensor y ,Tensor rstd)"
        );
    ops.impl("npu_gemma_rms_norm", torch::kPrivateUse1, &vllm_ascend::npu_gemma_rms_norm);
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", torch::kPrivateUse1, &vllm_ascend::weak_ref_tensor);

    ops.def(
        "get_masked_input_and_mask(Tensor input, "
        "                         int org_vocab_start_index, "
        "                         int org_vocab_end_index, "
        "                         int num_org_vocab_padding, "
        "                         int added_vocab_start_index, "
        "                         int added_vocab_end_index) -> (Tensor masked_input, Tensor mask)");
    ops.impl("get_masked_input_and_mask", torch::kPrivateUse1, &vllm_ascend::get_masked_input_and_mask);

    ops.def(
        "npu_dsa_prepare_sparse_indices_(Tensor(a!) topk_indices, Tensor split_boundary, "
        "Tensor row_req_indices, Tensor request_block_table, "
        "Tensor(b!) selected_packed, Tensor(c!) selected_counts, "
        "Tensor(d!) target_slots, "
        "int block_size, bool need_packed, bool clear_invalid_rows) -> Tensor(c!)");
    ops.impl(
        "npu_dsa_prepare_sparse_indices_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_prepare_sparse_indices_);
    ops.def(
        "npu_dsa_prepare_sparse_indices_staged_("
        "Tensor(a!) topk_indices, Tensor split_boundary, "
        "Tensor row_req_indices, Tensor request_block_table, "
        "Tensor(b!) selected_packed, Tensor(c!) selected_counts, "
        "Tensor(d!) target_slots, Tensor(e!) local_to_union_workspace, "
        "int block_size, int mtp, bool need_packed, "
        "bool clear_invalid_rows) -> Tensor(c!)");
    ops.impl(
        "npu_dsa_prepare_sparse_indices_staged_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_prepare_sparse_indices_staged_);
    ops.def(
        "npu_dsa_prepare_sparse_indices_sharded_("
        "Tensor(a!) topk_indices, Tensor split_boundary, "
        "Tensor row_req_indices, Tensor request_block_table, "
        "Tensor(b!) selected_packed, Tensor(c!) selected_counts, "
        "Tensor(d!) target_slots, Tensor(e!) local_to_union_workspace, "
        "Tensor(f!) shard_packed_workspace, "
        "Tensor(g!) shard_mapping_workspace, "
        "Tensor(h!) shard_counts_workspace, "
        "int block_size, int mtp, bool need_packed, "
        "bool clear_invalid_rows) -> Tensor(c!)");
    ops.impl(
        "npu_dsa_prepare_sparse_indices_sharded_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_prepare_sparse_indices_sharded_);
    ops.def(
        "npu_dsa_prepare_sparse_indices_legacy_(Tensor(a!) topk_indices, "
        "Tensor split_boundary, Tensor valid_rows, Tensor scratch_base, "
        "bool need_packed, Tensor? row_req_indices=None, "
        "int packed_key_stride=0) -> Tensor");
    ops.impl(
        "npu_dsa_prepare_sparse_indices_legacy_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_prepare_sparse_indices_legacy_);
    ops.def(
        "npu_dsa_staged_union_(Tensor row_packed, "
        "Tensor(a!) selected_packed, Tensor(b!) local_to_union, "
        "Tensor(c!) selected_count, Tensor request_block_table, "
        "Tensor(d!) target_slots, int block_size, int max_tokens, "
        "bool use_sort) -> Tensor(c!)");
    ops.impl(
        "npu_dsa_staged_union_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_staged_union_);
    ops.def(
        "npu_dsa_staged_sharded_union_(Tensor(a!) topk_indices, "
        "Tensor split_boundary, Tensor(b!) selected_packed, "
        "Tensor(c!) local_to_union, "
        "Tensor(d!) selected_count, Tensor request_block_table, "
        "Tensor(e!) target_slots, Tensor(f!) shard_packed, "
        "Tensor(g!) shard_mapping, Tensor(h!) shard_counts, "
        "int block_size) -> Tensor(d!)");
    ops.impl(
        "npu_dsa_staged_sharded_union_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_staged_sharded_union_);
    ops.def(
        "npu_dsa_staged_sharded_vector_union_(Tensor(a!) topk_indices, "
        "Tensor split_boundary, Tensor(b!) selected_packed, "
        "Tensor(c!) local_to_union, "
        "Tensor(d!) selected_count, Tensor request_block_table, "
        "Tensor(e!) target_slots, Tensor(f!) shard_packed, "
        "Tensor(g!) shard_mapping, Tensor(h!) shard_counts, "
        "Tensor(i!) shard_pairs, int block_size) -> Tensor(d!)");
    ops.impl(
        "npu_dsa_staged_sharded_vector_union_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_staged_sharded_vector_union_);
    ops.def(
        "npu_dsa_staged_sharded_vector_dedup_(Tensor(a!) topk_indices, "
        "Tensor split_boundary, Tensor(b!) selected_packed, "
        "Tensor(c!) local_to_union, "
        "Tensor(d!) selected_count, Tensor request_block_table, "
        "Tensor(e!) target_slots, Tensor(f!) shard_packed, "
        "Tensor(g!) shard_mapping, Tensor(h!) shard_counts, "
        "int block_size) -> Tensor(d!)");
    ops.impl(
        "npu_dsa_staged_sharded_vector_dedup_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_staged_sharded_vector_dedup_);
    ops.def(
        "npu_dsa_staged_remap_rows_(Tensor(a!) local_indices, "
        "Tensor local_to_union) -> Tensor(a!)");
    ops.impl(
        "npu_dsa_staged_remap_rows_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_staged_remap_rows_);
    ops.def(
        "npu_dsa_resident_remap_rows_(Tensor(a!) topk_indices, "
        "Tensor position_to_union, Tensor union_to_slot) -> Tensor(a!)");
    ops.impl(
        "npu_dsa_resident_remap_rows_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_remap_rows_);
    ops.def(
        "npu_dsa_resident_lookup_rows_(Tensor selected_packed, "
        "Tensor selected_count, Tensor request_state_indices, "
        "Tensor(a!) lookup_indices, int token_stride, "
        "int dummy_state_base) -> Tensor(a!)");
    ops.impl(
        "npu_dsa_resident_lookup_rows_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_lookup_rows_);
    ops.def(
        "npu_dsa_resident_finalize_rows_(Tensor(a!) topk_indices, "
        "Tensor position_to_union, Tensor(b!) selected_packed, "
        "Tensor(c!) selected_count, Tensor(d!) target_slots, "
        "Tensor request_block_table, Tensor request_state_indices, "
        "Tensor request_state_generations, Tensor old_slots, "
        "Tensor(e!) slot_to_token, Tensor(f!) state_generations, "
        "Tensor(g!) union_to_slot, Tensor(h!) reverse_indices, "
        "Tensor(i!) reverse_values, int token_stride, "
        "int block_size) -> Tensor(c!)");
    ops.impl(
        "npu_dsa_resident_finalize_rows_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_finalize_rows_);
    ops.def(
        "npu_dsa_resident_sharded_union_(Tensor topk_indices, "
        "Tensor split_boundary, Tensor row_req_indices, "
        "Tensor(a!) shard_packed, Tensor(b!) shard_mapping, "
        "Tensor(c!) shard_counts, Tensor request_state_indices, "
        "Tensor request_state_generations, Tensor state_tokens, "
        "Tensor state_slots, Tensor(d!) state_counts, "
        "Tensor state_generations, Tensor(e!) prior_slots, "
        "Tensor(f!) shard_miss_tokens, "
        "Tensor(g!) shard_miss_positions, "
        "Tensor(h!) shard_evictable_slots, "
        "int mtp, int dummy_state_base) -> Tensor(c!)");
    ops.impl(
        "npu_dsa_resident_sharded_union_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_sharded_union_);
    ops.def(
        "npu_dsa_resident_sharded_union_v2_(Tensor topk_indices, "
        "Tensor split_boundary, Tensor row_req_indices, "
        "Tensor(a!) shard_packed, Tensor(b!) shard_mapping, "
        "Tensor(c!) shard_counts, Tensor request_state_indices, "
        "Tensor request_state_generations, Tensor state_tokens, "
        "Tensor state_slots, Tensor(d!) state_counts, "
        "Tensor state_generations, Tensor(e!) prior_slots, "
        "Tensor(f!) shard_miss_tokens, "
        "Tensor(g!) shard_miss_positions, "
        "Tensor(h!) shard_evictable_slots, "
        "int mtp, int dummy_state_base) -> Tensor(c!)");
    ops.impl(
        "npu_dsa_resident_sharded_union_v2_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_sharded_union_v2_);
    ops.def(
        "npu_dsa_resident_sorted_plan_(Tensor(a!) topk_indices, "
        "Tensor shard_packed, Tensor shard_mapping, "
        "Tensor shard_counts, Tensor request_block_table, "
        "Tensor request_state_indices, "
        "Tensor request_state_generations, "
        "Tensor(b!) state_tokens, Tensor(c!) state_slots, "
        "Tensor(d!) state_counts, Tensor(e!) state_generations, "
        "Tensor(f!) prior_slots, Tensor shard_miss_tokens, "
        "Tensor shard_miss_positions, Tensor shard_evictable_slots, "
        "Tensor(g!) miss_tokens, Tensor(h!) miss_counts, "
        "Tensor(i!) target_slots, int block_size, "
        "int dummy_state_base) -> Tensor(h!)");
    ops.impl(
        "npu_dsa_resident_sorted_plan_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_sorted_plan_);
    ops.def(
        "npu_dsa_resident_parallel_plan_v2_("
        "Tensor(a!) topk_indices, "
        "Tensor shard_packed, Tensor shard_mapping, "
        "Tensor shard_counts, Tensor request_block_table, "
        "Tensor request_state_indices, "
        "Tensor request_state_generations, "
        "Tensor(b!) state_tokens, Tensor(c!) state_slots, "
        "Tensor(d!) state_counts, Tensor(e!) state_generations, "
        "Tensor(f!) prior_slots, Tensor shard_miss_tokens, "
        "Tensor shard_miss_positions, Tensor shard_evictable_slots, "
        "Tensor(g!) miss_tokens, Tensor(h!) miss_counts, "
        "Tensor(i!) target_slots, int block_size, "
        "int dummy_state_base) -> Tensor(h!)");
    ops.impl(
        "npu_dsa_resident_parallel_plan_v2_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_parallel_plan_v2_);
    ops.def(
        "npu_dsa_resident_sorted_plan_no_remap_("
        "Tensor(a!) topk_indices, "
        "Tensor shard_packed, Tensor shard_mapping, "
        "Tensor shard_counts, Tensor request_block_table, "
        "Tensor request_state_indices, "
        "Tensor request_state_generations, "
        "Tensor(b!) state_tokens, Tensor(c!) state_slots, "
        "Tensor(d!) state_counts, Tensor(e!) state_generations, "
        "Tensor(f!) prior_slots, Tensor shard_miss_tokens, "
        "Tensor shard_miss_positions, Tensor shard_evictable_slots, "
        "Tensor(g!) miss_tokens, Tensor(h!) miss_counts, "
        "Tensor(i!) target_slots, int block_size, "
        "int dummy_state_base) -> Tensor(h!)");
    ops.impl(
        "npu_dsa_resident_sorted_plan_no_remap_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_sorted_plan_no_remap_);
    ops.def(
        "npu_dsa_resident_sorted_update_debug_("
        "Tensor(a!) topk_indices, Tensor shard_packed, "
        "Tensor shard_mapping, Tensor shard_counts, "
        "Tensor prior_slots, "
        "Tensor request_state_indices, "
        "Tensor request_state_generations, "
        "Tensor(b!) state_tokens, Tensor(c!) state_slots, "
        "Tensor(d!) state_counts, Tensor(e!) state_generations, "
        "int dummy_state_base) -> Tensor(a!)");
    ops.impl(
        "npu_dsa_resident_sorted_update_debug_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_sorted_update_debug_);
    ops.def(
        "npu_dsa_resident_sorted_remap_("
        "Tensor(a!) topk_indices, Tensor shard_mapping, "
        "Tensor shard_counts, Tensor prior_slots) -> Tensor(a!)");
    ops.impl(
        "npu_dsa_resident_sorted_remap_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_sorted_remap_);
    ops.def(
        "npu_dsa_resident_sorted_read_probe_("
        "Tensor shard_counts, Tensor prior_slots, "
        "Tensor(a!) debug_info, Tensor(b!) prior_readback) "
        "-> Tensor(a!)");
    ops.impl(
        "npu_dsa_resident_sorted_read_probe_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_sorted_read_probe_);
    ops.def(
        "npu_dsa_resident_sorted_finalize_debug_("
        "Tensor shard_packed, Tensor shard_counts, "
        "Tensor(a!) prior_slots, Tensor shard_miss_tokens, "
        "Tensor shard_miss_positions, Tensor shard_evictable_slots, "
        "Tensor(b!) miss_tokens, Tensor(c!) miss_counts, "
        "Tensor(d!) target_slots, Tensor request_block_table, "
        "Tensor(e!) debug_info, int block_size, "
        "int debug_stage) -> Tensor(c!)");
    ops.impl(
        "npu_dsa_resident_sorted_finalize_debug_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_resident_sorted_finalize_debug_);
    ops.def(
        "npu_dsa_staged_unique_finalize_(Tensor unique_keys, "
        "Tensor inverse, Tensor row_req_indices, "
        "Tensor(a!) selected_packed, Tensor(b!) local_to_union, "
        "Tensor(c!) selected_count, Tensor request_block_table, "
        "Tensor(d!) target_slots, int block_size, "
        "int packed_key_stride) -> Tensor(c!)");
    ops.impl(
        "npu_dsa_staged_unique_finalize_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_staged_unique_finalize_);
    ops.def(
        "npu_dsa_staged_copy_rows_(Tensor(a!) output, "
        "Tensor local_indices) -> Tensor(a!)");
    ops.impl(
        "npu_dsa_staged_copy_rows_",
        torch::kPrivateUse1,
        &vllm_ascend::npu_dsa_staged_copy_rows_);

    ops.def("bgmv_shrink(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y, float scale) -> ()");
    ops.impl("bgmv_shrink", torch::kPrivateUse1, &vllm_ascend::bgmv_shrink);

    ops.def(
        "bgmv_expand(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");
    ops.impl("bgmv_expand", torch::kPrivateUse1, &vllm_ascend::bgmv_expand);

    ops.def("sgmv_shrink(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y, float scale) -> ()");
    ops.impl("sgmv_shrink", torch::kPrivateUse1, &vllm_ascend::sgmv_shrink);

    ops.def(
        "sgmv_expand(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");
    ops.impl("sgmv_expand", torch::kPrivateUse1, &vllm_ascend::sgmv_expand);

    ops.def(
        "mla_preprocess(Tensor hiddenState, Tensor wdqkv,"
        "               Tensor? descale0, Tensor gamma1, Tensor? beta1, Tensor wuq, Tensor? descale1,"
        "               Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache,"
        "               Tensor kv_cache_rope, Tensor slotmapping, Tensor? quant_scale0,"
        "               Tensor? quant_offset0, Tensor? bias0, Tensor? quant_scale1, Tensor? quant_offset1,"
        "               Tensor? bias1, Tensor? ctkv_scale, Tensor? q_nope_scale, str? cache_mode,"
        "               str? quant_mode, bool? enable_inner_out, Tensor! q_out0, Tensor! kv_cache_out0, Tensor! q_out1,"
        "               Tensor! kv_cache_out1, Tensor! inner_out) -> (Tensor q_out0, Tensor kv_cache_out0,"
        "                                          Tensor q_out1, Tensor kv_cache_out1, Tensor inner_out)"
    );
    ops.impl("mla_preprocess", torch::kPrivateUse1, &vllm_ascend::mla_preprocess);

    //batch_matmul ops refer to sgl-kernel-npu
    ops.def(
            "batch_matmul_transpose(Tensor tensor_a, Tensor tensor_b, Tensor tensor_c, str? format_mode=None, str? quant_mode=None) -> ()");    
    ops.impl("batch_matmul_transpose", torch::kPrivateUse1, &vllm_ascend::batch_matmul_transpose);

    ops.def("swap_blocks(Tensor! x, Tensor! y, Tensor z) -> ()");    
    ops.impl("swap_blocks", torch::kPrivateUse1, &vllm_ascend::swap_blocks);

    ops.def(
        "grouped_matmul_swiglu_quant(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale,"
        "                            Tensor group_list, *, Tensor? bias=None,"
        "                            Tensor? offset=None) -> (Tensor output, Tensor output_scale, Tensor output_offset)");
    ops.impl("grouped_matmul_swiglu_quant", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant);

    ops.def(
        "dispatch_gmm_combine_decode(Tensor x, Tensor expert_ids, Tensor[] gmm1_permuted_weight,"
        "                            Tensor[] gmm1_permuted_weight_scale,"
        "                            Tensor[] gmm2_weight, Tensor[] gmm2_weight_scale,"
        "                            Tensor expert_scales, Tensor? expert_smooth_scales=None,"
        "                            Tensor? x_active_mask=None,"
        "                            str group_ep='',"
        "                            int ep_rank_size=0, int ep_rank_id=0, int moe_expert_num=0,"
        "                            int shared_expert_num=1, int shared_expert_rank_num=0,"
        "                            int quant_mode=0,"
        "                            int global_bs=0) -> (Tensor output, Tensor expert_token_nums)"
    );
    ops.impl("dispatch_gmm_combine_decode", torch::kPrivateUse1, &vllm_ascend::dispatch_gmm_combine_decode);

    ops.def(
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale,"
        "                                                  Tensor group_list, *,"
        "                                                  Tensor? bias=None, Tensor? offset=None) ->"
        "                                                  (Tensor output, Tensor output_scale, Tensor output_offset)"
    );
    ops.impl("grouped_matmul_swiglu_quant_weight_nz_tensor_list", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant_weight_nz_tensor_list);

    ops.def(
        "npu_lightning_indexer(Tensor query, Tensor key, Tensor weights, *,"
        "                      Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None,"
        "                      Tensor? block_table=None, str layout_query='BSND', str layout_key='BSND',"
        "                      int sparse_count=2048, int sparse_mode=3) -> Tensor"
    );
    ops.impl("npu_lightning_indexer", torch::kPrivateUse1, &vllm_ascend::npu_lightning_indexer);

    ops.def(
        "npu_sparse_flash_attention(Tensor query, Tensor key, Tensor value,"
        "                           Tensor sparse_indices, float scale_value, int sparse_block_size, *,"
        "                           Tensor? block_table=None, Tensor? actual_seq_lengths_query=None,"
        "                           Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None,"
        "                           Tensor? key_rope=None, str layout_query='BSND', str layout_kv='BSND',"
        "                           int sparse_mode=3) -> Tensor"
    );
    ops.impl("npu_sparse_flash_attention", torch::kPrivateUse1, &vllm_ascend::npu_sparse_flash_attention);

    ops.def(
        "dispatch_ffn_combine(Tensor x, Tensor[] weight1, Tensor[] weight2, Tensor expert_idx,"
        "                     Tensor[] scale1, Tensor[] scale2, Tensor probs, str group,"
        "                     int max_output_size, Tensor! out, Tensor! expert_token_nums) -> (Tensor out, Tensor expert_token_nums)"
    );
    ops.impl("dispatch_ffn_combine", torch::kPrivateUse1, &vllm_ascend::dispatch_ffn_combine);

    ops.def("matmul_allreduce_add_rmsnorm(Tensor x1, Tensor x2, Tensor residual, Tensor gamma, \
        str groupTp, int tpRankSize, int tpRankId, float epsilon, bool isTransB, bool isGatherAddOut) -> (Tensor output, Tensor add_out)");
    ops.impl("matmul_allreduce_add_rmsnorm", torch::kPrivateUse1, &vllm_ascend::matmul_allreduce_add_rmsnorm);

    ops.def("get_dispatch_layout(Tensor topk_idx, int num_experts, int "
            "num_ranks) -> (Tensor num_tokens_per_rank, Tensor "
            "num_tokens_per_expert, Tensor is_token_in_rank_bool)");
    ops.impl("get_dispatch_layout", torch::kPrivateUse1,
             &vllm_ascend::get_dispatch_layout);

    ops.def(
        "dispatch_prefill(Tensor x, Tensor topk_idx, Tensor topk_weights, "
        "Tensor num_tokens_per_rank, Tensor is_token_in_rank, Tensor "
        "num_tokens_per_expert, int num_worst_tokens, str groupEp, int rank, "
        "int num_ranks) -> (Tensor expandx_out, Tensor expand_idx_out, Tensor "
        "recv_count, Tensor num_recv_tokens_per_expert)");
    ops.impl("dispatch_prefill", torch::kPrivateUse1,
             &vllm_ascend::dispatch_prefill);

    ops.def("combine_prefill(Tensor x, Tensor topk_idx, Tensor topk_weights, "
            "Tensor src_idx, Tensor send_head, str grouEp, int rank, int "
            "num_ranks) -> Tensor");
    ops.impl("combine_prefill", torch::kPrivateUse1,
             &vllm_ascend::combine_prefill);
    
    ops.def(
        "npu_moe_init_routing_custom(Tensor x, Tensor expert_idx, *, Tensor? scale=None, Tensor? offset=None, int active_num=-1, "
        "                            int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, int expert_tokens_num_type=0, "
        "                            bool expert_tokens_num_flag=False, int quant_mode=0, int[2] active_expert_range=[], "
        "                            int row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)"
    );
    ops.impl("npu_moe_init_routing_custom", torch::kPrivateUse1, &vllm_ascend::npu_moe_init_routing_custom);
    // vLLM-Ascend custom ops
    ops.def(
        "moe_gating_top_k(Tensor x, "
                            "int k, "
                            "int k_group, "
                            "int group_count, "
                            "int group_select_mode, "
                            "int renorm, "
                            "int norm_type, "
                            "bool out_flag, "
                            "float routed_scaling_factor, "
                            "float eps,"
                            "Tensor? bias_opt=None)"
                            
        "-> (Tensor y ,Tensor expert_idx, Tensor out)"
        );
    ops.impl("moe_gating_top_k", torch::kPrivateUse1,&vllm_ascend::moe_gating_top_k);

    ops.def(
        "npu_add_rms_norm_bias(Tensor x1, "
                            "Tensor x2, "
                            "Tensor gamma, "
                            "Tensor? beta=None, "
                            "float epsilon=1e-6)"
        "-> (Tensor y ,Tensor rstd, Tensor x)"
        );
    ops.impl("npu_add_rms_norm_bias", torch::kPrivateUse1, &vllm_ascend::npu_add_rms_norm_bias);

    ops.def("npu_apply_top_k_top_p(Tensor logits, Tensor? p=None, Tensor? k=None) -> Tensor");
    ops.impl("npu_apply_top_k_top_p", torch::kPrivateUse1, &vllm_ascend::npu_apply_top_k_top_p);
    ops.def(
        "transpose_kv_cache_by_block(Tensor[] kCache, Tensor[] vCache, Tensor blockIDs, int blockSize, int headNum, int headDim, int splitNum, int layerNum) -> ()"
    );
    ops.impl("transpose_kv_cache_by_block", torch::kPrivateUse1, &vllm_ascend::transpose_kv_cache_by_block);

    ops.def(
        "npu_copy_and_expand_eagle_inputs(Tensor target_token_ids, Tensor target_positions, "
        "Tensor next_token_ids, Tensor query_start_loc, Tensor query_end_loc, "
        "int padding_token_id, int parallel_drafting_token_id, int num_padding_slots_per_request, "
        "bool shift_input_ids, int total_draft_tokens) -> "
        "(Tensor out_input_ids, Tensor out_positions, Tensor out_is_rejected_token_mask, "
        "Tensor out_is_masked_token_mask, Tensor out_new_token_indices, Tensor out_hidden_state_mapping)"
    );
    ops.impl("npu_copy_and_expand_eagle_inputs", torch::kPrivateUse1, &vllm_ascend::npu_copy_and_expand_eagle_inputs);
    ops.def(
        "npu_causal_conv1d_custom(Tensor x, "
        "                         Tensor weight, "
        "                         Tensor conv_state, "
        "                         Tensor? bias_opt, "
        "                         int[] query_start_loc_opt, "
        "                         int[] cache_indices_opt, "
        "                         int[] initial_state_mode_opt, "
        "                         int[] num_accepted_tokens_opt, "
        "                         int activation_mode, "
        "                         int pad_slot_id, "
        "                         int run_mode"
        ") -> (Tensor output)");
    ops.impl("npu_causal_conv1d_custom", torch::kPrivateUse1, &vllm_ascend::npu_causal_conv1d_custom);
    ops.def(
        "moe_grouped_matmul("
            "Tensor x,"
            "Tensor weight,"
            "Tensor group_list,"
            "int split_item,"
            "int group_type,"
            "int group_list_type)"

        "-> Tensor[]"
    );
    ops.impl("moe_grouped_matmul", torch::kPrivateUse1,&vllm_ascend::moe_grouped_matmul);

    // This operator is planned to be integrated into PTA in the near future.
    // Once that happens, the implementation in csrc will be removed.
    ops.def(
        "npu_lightning_indexer_quant(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, "
        "                            Tensor key_dequant_scale, *, Tensor? actual_seq_lengths_query=None, "
        "                            Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, "
        "                            int query_quant_mode=0, int key_quant_mode=0, "
        "                            str layout_query='BSND', str layout_key='BSND',"
        "                            int sparse_count=2048, int sparse_mode=3) -> Tensor"
    );
    ops.impl("npu_lightning_indexer_quant", torch::kPrivateUse1, &vllm_ascend::npu_lightning_indexer_quant);
}
