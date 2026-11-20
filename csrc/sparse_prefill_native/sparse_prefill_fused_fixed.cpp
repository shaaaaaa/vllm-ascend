// SPDX-License-Identifier: Apache-2.0
#include "sparse_prefill_services.h"

extern "C" __global__ __aicore__ void sparse_prefill_partition_finish_fixed(
    GM_ADDR lists, GM_ADDR descriptors, GM_ADDR membership, GM_ADDR counters,
    GM_ADDR groups, GM_ADDR bitmap, GM_ADDR sources, GM_ADDR lengths,
    uint32_t history, uint32_t query, uint32_t width, uint32_t capacity
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    {
        SparsePrefillPlan::PartitionUnion op;
        op.Init(lists, descriptors, membership, counters, history, query, width, pipe);
        op.Process();
    }
    AscendC::SyncAll();
    pipe.Reset();
    {
        SparsePrefillBitmap::BuildMembershipWords op;
        op.Init(membership, counters, groups, bitmap, width, pipe);
        op.Process();
    }
    AscendC::SyncAll();
    pipe.Reset();
    {
        SparsePrefillBitmap::FinishSortedBitmap op;
        op.Init(groups, bitmap, sources, lengths, query, width, capacity, pipe);
        op.Process();
    }
    AscendC::SyncAll();
}
