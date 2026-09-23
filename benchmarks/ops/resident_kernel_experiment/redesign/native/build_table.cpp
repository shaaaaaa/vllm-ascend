// SPDX-License-Identifier: Apache-2.0
#include "build_table_core.h"
extern "C" __global__ __aicore__ void resident_snapshot_build_redesign(RedesignLaunch a) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    redesign::BuildTable<> op; op.Init(a); op.Process();
}
void redesign_build(void* stream, const RedesignLaunch& a) {
    uint32_t blocks = a.requests * (a.tableSize / redesign::T);
    RedesignLaunch launch = a;
    resident_snapshot_build_redesign<<<blocks < a.cores ? blocks : a.cores, nullptr, stream>>>(&launch);
}
