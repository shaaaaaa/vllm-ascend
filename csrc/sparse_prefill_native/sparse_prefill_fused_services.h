#ifndef SPARSE_PREFILL_FUSED_SERVICES_H
#define SPARSE_PREFILL_FUSED_SERVICES_H
// SPDX-License-Identifier: Apache-2.0
#include "sparse_prefill_local_union_services.h"
#include "sparse_prefill_bitmap_services.h"

namespace SparsePrefillPipeline {
__aicore__ inline void Run(
    GM_ADDR lists, GM_ADDR descriptors, GM_ADDR membership,
    GM_ADDR counters, GM_ADDR groups, GM_ADDR bitmap,
    GM_ADDR sources, GM_ADDR lengths,
    uint32_t history, uint32_t query, uint32_t width, uint32_t capacity)
{
    using namespace AscendC;
    TPipe pipe;
    {
        SparsePrefillPlan::PartitionUnion op;
        op.Init(lists, descriptors, membership, counters,
                history, query, width, pipe, query > SparsePrefillLocal::SHORT_QUERY_ROWS);
        op.Process();
    }
    SyncAll();
    pipe.Reset();
    {
        SparsePrefillBitmap::BuildMembershipWords op;
        op.Init(membership, counters, groups, bitmap, width, pipe);
        op.Process();
    }
    SyncAll();
    pipe.Reset();
    {
        SparsePrefillBitmap::FinishSortedBitmap op;
        op.Init(groups, bitmap, sources, lengths, query, capacity, pipe);
        op.Process();
    }
    SyncAll();
}
} // namespace SparsePrefillPipeline
#endif
