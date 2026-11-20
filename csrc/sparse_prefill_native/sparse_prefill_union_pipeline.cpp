// SPDX-License-Identifier: Apache-2.0
#include "sparse_prefill_services.h"

extern "C" __global__ __aicore__ void sparse_prefill_adaptive_union(
    GM_ADDR topk, GM_ADDR lists, GM_ADDR descriptors, uint32_t history, uint32_t query
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    SparsePrefillLocal::LocalQueryUnion op;
    op.Init(topk, lists, descriptors, history, query, pipe);
    op.Process();
}

extern "C" int32_t launch_sparse_prefill_adaptive_prefix(
    void* stream, void* topk, void* lists, void* descriptors, uint32_t history, uint32_t query
) {
    if (!stream || !topk || !lists || !descriptors || history > 524288U || query == 0U || query > 16384U) {
        return -1;
    }
    sparse_prefill_adaptive_union<<<SparsePrefillPlan::CORES, nullptr, stream>>>(
        static_cast<GM_ADDR>(topk), static_cast<GM_ADDR>(lists), static_cast<GM_ADDR>(descriptors), history, query
    );
    return 0;
}
