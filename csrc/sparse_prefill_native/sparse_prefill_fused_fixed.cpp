// SPDX-License-Identifier: Apache-2.0
#include "selected_kv_assert.h"
#include "sparse_prefill_fused_services.h"

extern "C" __global__ __aicore__ void sparse_prefill_partition_finish_fixed(
    GM_ADDR lists, GM_ADDR descriptors, GM_ADDR membership,
    GM_ADDR counters, GM_ADDR groups, GM_ADDR bitmap,
    GM_ADDR sources, GM_ADDR lengths,
    uint32_t history, uint32_t query, uint32_t width, uint32_t capacity)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    SparsePrefillPipeline::Run(
        lists, descriptors, membership, counters, groups, bitmap,
        sources, lengths, history, query, width, capacity);
}
