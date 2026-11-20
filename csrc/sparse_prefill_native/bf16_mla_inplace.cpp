// SPDX-License-Identifier: Apache-2.0
#include "selected_inplace_common.h"

extern "C" __global__ __aicore__ void sparse_prefill_bf16_mla_inplace(
    GM_ADDR ids, GM_ADDR status, GM_ADDR pageBases, GM_ADDR metadata, GM_ADDR table, GM_ADDR c0, GM_ADDR c1,
    uint32_t reqs, uint32_t cap, uint32_t sp, uint32_t tr, uint32_t tc, uint32_t ts, uint32_t dp, uint32_t gc
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    SparsePrefillCopy::RunSelectedMlaInplace(
        pipe, ids, status, pageBases, metadata, table, c0, c1, reqs, cap, sp, tr, tc, ts, dp, gc
    );
}

extern "C" int32_t launch_sparse_prefill_bf16_mla_inplace(
    void* stream, void* ids, void* status, void* pageBases, void* metadata, void* table, void* c0, void* c1,
    uint32_t reqs, uint32_t cap, uint32_t sp, uint32_t tr, uint32_t tc, uint32_t ts, uint32_t dp
) {
    const int32_t gc = SparsePrefillCopy::ResolveLaunchGroupCores(reqs, cap, sp, tr, tc, ts, dp);
    if (!stream || !ids || !status || !pageBases || !metadata || !table || !c0 || !c1 || gc <= 0) {
        return -1;
    }
    sparse_prefill_bf16_mla_inplace<<<SparsePrefillCopy::CORES, nullptr, stream>>>(
        static_cast<GM_ADDR>(ids), static_cast<GM_ADDR>(status), static_cast<GM_ADDR>(pageBases),
        static_cast<GM_ADDR>(metadata), static_cast<GM_ADDR>(table), static_cast<GM_ADDR>(c0), static_cast<GM_ADDR>(c1),
        reqs, cap, sp, tr, tc, ts, dp, static_cast<uint32_t>(gc)
    );
    return 0;
}
