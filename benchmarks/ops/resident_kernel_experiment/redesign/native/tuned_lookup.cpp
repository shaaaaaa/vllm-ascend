// SPDX-License-Identifier: Apache-2.0
#include "lookup_core.h"
extern "C" __global__ __aicore__ void resident_tuned_lookup_redesign(RedesignLaunch a) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    redesign::Lookup<false, false, true> op; op.Init(a); op.Process();
}
void redesign_tuned_lookup(void* stream, const RedesignLaunch& a) {
    uint32_t blocks = a.requests * a.queries * a.topk / redesign::T;
    RedesignLaunch launch = a;
    resident_tuned_lookup_redesign<<<blocks < a.cores ? blocks : a.cores, nullptr, stream>>>(&launch);
}
