// SPDX-License-Identifier: Apache-2.0
#include "lookup_core.h"
extern "C" __global__ __aicore__ void resident_resolve_copy_redesign(RedesignLaunch a) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    redesign::Lookup<true> op; op.Init(a); op.Process();
}
void redesign_resolve_copy(void* stream, const RedesignLaunch& a) {
    uint32_t blocks = a.requests * a.queries * a.topk / redesign::T;
    // CANN's host stub marshals a by-value device struct through a host pointer.
    RedesignLaunch launch = a;
    resident_resolve_copy_redesign<<<blocks < a.cores ? blocks : a.cores, nullptr, stream>>>(&launch);
}
