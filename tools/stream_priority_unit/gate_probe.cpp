// SPDX-License-Identifier: Apache-2.0
#include "ascend_aiv_gate.h"

extern "C" aclError stream_priority_submit(aclrtFuncHandle, uint32_t, aclrtStream, uint64_t*, uint32_t, uint64_t);

// One cache line per block, matching pulse.cpp. The optional event signals when
// the other lane becomes ready, including from inside a protected foreground batch.
struct GatePulse {
    aclrtFuncHandle kernel;
    uint64_t* stamps;
    uint64_t hold_ticks;
    aclrtEvent ready;
    uint32_t blocks;
    uint32_t count;
    uint32_t signal_after;
};

static aclError enqueue_pulses(aclrtStream stream, void* work) {
    const auto& p = *static_cast<const GatePulse*>(work);
    if (!p.ready) {
        return stream_priority_submit(p.kernel, p.blocks, stream, p.stamps, p.count, p.hold_ticks);
    }
    if (p.signal_after > 0) {
        auto rc = stream_priority_submit(p.kernel, p.blocks, stream, p.stamps, p.signal_after, p.hold_ticks);
        if (rc != ACL_SUCCESS) return rc;
    }
    auto rc = aclrtRecordEvent(p.ready, stream);
    if (rc != ACL_SUCCESS || p.signal_after == p.count) return rc;
    auto* rest = p.stamps + static_cast<size_t>(p.signal_after) * p.blocks * 8;
    return stream_priority_submit(p.kernel, p.blocks, stream, rest, p.count - p.signal_after, p.hold_ticks);
}

extern "C" __attribute__((visibility("default"))) aclError aiv_probe_initialize(const aiv_gate* gate,
                                                                                aclrtStream stream) {
    return aiv_gate_initialize(gate, stream);
}

extern "C" __attribute__((visibility("default"))) aclError aiv_probe_pulses(const aiv_gate* gate, aclrtStream stream,
                                                                           int mode, uint32_t charge, GatePulse* p) {
    if (!p || !p->kernel || !p->stamps || !p->blocks || !p->count || p->signal_after > p->count)
        return ACL_ERROR_INVALID_PARAM;
    if (mode == 0) return enqueue_pulses(stream, p);
    if (charge < p->blocks) return ACL_ERROR_INVALID_PARAM;
    if (mode == 1) return aiv_gate_foreground(gate, stream, charge, enqueue_pulses, p);
    if (mode == 2 && p->count == 1) return aiv_gate_background_piece(gate, stream, charge, enqueue_pulses, p);
    return ACL_ERROR_INVALID_PARAM;
}

extern "C" __attribute__((visibility("default"))) uint32_t aiv_probe_kernel_width(aclmdlRITask task,
                                                                                  uint32_t capacity) {
    return aiv_gate_kernel_width(task, capacity);
}
