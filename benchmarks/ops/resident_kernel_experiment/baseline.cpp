// SPDX-License-Identifier: Apache-2.0
// Build only the resident kernel family, independently of the serving extension.
#ifndef RESIDENT_EXPERIMENT_SKIP_UNCHANGED
#define dsa_resident_sharded_union_kernel dsa_resident_sharded_union_kernel_baseline
#define dsa_resident_sorted_read_probe_kernel dsa_resident_sorted_read_probe_kernel_baseline
#define dsa_resident_sorted_finalize_kernel dsa_resident_sorted_finalize_kernel_baseline
#define dsa_resident_sorted_update_kernel dsa_resident_sorted_update_kernel_baseline
#define dsa_resident_sorted_state_update_kernel dsa_resident_sorted_state_update_kernel_baseline
#define dsa_resident_sorted_remap_kernel dsa_resident_sorted_remap_kernel_baseline
#define dsa_resident_sharded_union_impl dsa_resident_sharded_union_impl_baseline
#define dsa_resident_sorted_plan_impl dsa_resident_sorted_plan_impl_baseline
#define dsa_resident_sorted_update_debug_impl dsa_resident_sorted_update_debug_impl_baseline
#define dsa_resident_sorted_plan_no_remap_impl dsa_resident_sorted_plan_no_remap_impl_baseline
#define dsa_resident_sorted_remap_impl dsa_resident_sorted_remap_impl_baseline
#define dsa_resident_sorted_read_probe_impl dsa_resident_sorted_read_probe_impl_baseline
#define dsa_resident_sorted_finalize_debug_impl dsa_resident_sorted_finalize_debug_impl_baseline
#endif
#include "../../../csrc/kernels/resident_sorted_cache.cpp"
#include "launch.h"

void resident_experiment_run(void* stream, const ResidentLaunch& a, int stage)
{
    const auto* p = a.tensors;
    const uint32_t countStride = 16;
    const uint32_t requestStride = a.shards * countStride;
    const uint32_t generationStride = 8;
    if (stage == 0 || stage == 1) {
        vllm_ascend::dsa_resident_sharded_union_impl(
            stream, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
            p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15],
            a.requests, a.stateRows, a.dummyBase, a.mtp, 2048, a.shards,
            a.capacity, countStride, requestStride, generationStride, a.cores);
    }
    if (stage == 0 || stage == 2) {
        dsa_resident_sorted_finalize_kernel<<<
            vllm_ascend::ResidentPhysicalBlockCount(a.requests, a.cores),
            nullptr, stream>>>(
            static_cast<int32_t*>(p[3]), static_cast<int32_t*>(p[5]),
            static_cast<int16_t*>(p[12]), static_cast<int32_t*>(p[13]),
            static_cast<int16_t*>(p[14]), static_cast<int16_t*>(p[15]),
            static_cast<int32_t*>(p[16]), static_cast<int32_t*>(p[17]),
            static_cast<int64_t*>(p[18]), static_cast<int32_t*>(p[19]),
            static_cast<int32_t*>(p[17]), a.requests, a.shards, a.capacity,
            countStride, requestStride, countStride, a.blockTableWidth,
            a.blockSize, 0);
    }
    if (stage == 0 || stage == 3) {
        dsa_resident_sorted_update_kernel<<<
            vllm_ascend::ResidentPhysicalBlockCount(a.requests * a.shards, a.cores),
            nullptr, stream>>>(
            static_cast<int32_t*>(p[0]), static_cast<int32_t*>(p[3]),
            static_cast<int16_t*>(p[4]), static_cast<int32_t*>(p[5]),
            static_cast<int16_t*>(p[12]), static_cast<int32_t*>(p[6]),
            static_cast<int64_t*>(p[7]), static_cast<int32_t*>(p[8]),
            static_cast<int16_t*>(p[9]), static_cast<int32_t*>(p[10]),
            static_cast<int64_t*>(p[11]), a.requests, a.stateRows, a.dummyBase,
            a.mtp, 2048, a.shards, a.capacity, countStride, requestStride,
            generationStride);
    }
}
