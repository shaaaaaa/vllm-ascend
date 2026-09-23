// SPDX-License-Identifier: Apache-2.0
#include "build_table_core.h"
extern "C" __global__ __aicore__ void resident_wide_build_redesign(RedesignLaunch a) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    redesign::BuildTable<2048> op; op.Init(a); op.Process();
}
void redesign_wide_build(void* stream, const RedesignLaunch& a) {
    uint32_t blocks = a.requests * ((a.tableSize + 2047) / 2048);
    RedesignLaunch launch = a;
    resident_wide_build_redesign<<<blocks < a.cores ? blocks : a.cores, nullptr, stream>>>(&launch);
}
