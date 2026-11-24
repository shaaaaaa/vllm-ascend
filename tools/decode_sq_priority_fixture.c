// SPDX-License-Identifier: Apache-2.0
// Test-process adapter for CANN v9.1.0 / 910B. The real HAL allocates both queues.
// ABI: include/driver/hal_pkg/trs_pkg.h (halSqCqInputInfo), ascend_hal_define.h
// (halSqCqOutputInfo), src/runtime/core/src/stream/stream_sqcq_manage.hpp (rtStreamAllocInfo_t).
#ifdef __linux__
#define _GNU_SOURCE
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#ifdef __linux__
#include <dlfcn.h>
#endif

typedef int (*Allocate)(uint32_t, void *, void *);

struct InputPrefix {
    uint32_t type, ts_id, sqe_size, cqe_size, sqe_depth, cqe_depth, group_id, flag, cq_id, sq_id;
    uint32_t info[5];
};
_Static_assert(offsetof(struct InputPrefix, info) == 40, "CANN 9.1 allocation message");
enum { NO_SQ_MEM = 1u << 16, SOFTWARE_SQ = 1u << 3, LOW_PRIORITY = 7 };

static Allocate backend;
static uint32_t selected_device, first_priority;
static atomic_uint armed, attempts;
static atomic_uint sq_ids[2] = {UINT32_MAX, UINT32_MAX};
static uint32_t accepted_priorities[2];

int halSqCqAllocate(uint32_t device, void *input, void *output);

int sq_test_arm(Allocate real_allocate, uint32_t device, uint32_t first) {
    if (!real_allocate || real_allocate == halSqCqAllocate || (first != 0 && first != LOW_PRIORITY) ||
        atomic_load(&armed)) {
        return -1;
    }
    backend = real_allocate;
    selected_device = device;
    first_priority = first;
    atomic_store(&armed, 1);
    return 0;
}

int halSqCqAllocate(uint32_t device, void *input, void *output) {
    if (!atomic_load(&armed)) {
#ifdef __linux__
        Allocate next = (Allocate)dlsym(RTLD_NEXT, "halSqCqAllocate");
        return next ? next(device, input, output) : -1;
#else
        return -1;
#endif
    }
    if (device != selected_device || !input || !output) {
        return backend(device, input, output);
    }
    struct InputPrefix msg;
    memcpy(&msg, input, sizeof(msg));
    // DeviceSqCqPool::BatchAllocSqCq uses this exact flag and anonymous software-SQ message.
    if (msg.type != 0 || msg.flag != NO_SQ_MEM || msg.info[0] != UINT32_MAX || msg.info[1] != 0 ||
        !(msg.info[2] & SOFTWARE_SQ)) {
        return backend(device, input, output);
    }
    const unsigned slot = atomic_fetch_add(&attempts, 1);
    if (slot >= 2) {
        return backend(device, input, output);
    }
    const uint32_t priority = slot == 0 ? first_priority : LOW_PRIORITY - first_priority;
    unsigned char *field = (unsigned char *)input + offsetof(struct InputPrefix, info) + sizeof(uint32_t);
    memcpy(field, &priority, sizeof(priority));
    const int rc = backend(device, input, output);
    memcpy(field, &msg.info[1], sizeof(msg.info[1]));
    if (rc == 0) {
        uint32_t id;
        memcpy(&id, output, sizeof(id));  // halSqCqOutputInfo.sqId
        accepted_priorities[slot] = priority;
        atomic_store(&sq_ids[slot], id);
    }
    return rc;
}

int sq_test_lookup(uint32_t sq_id, uint32_t *priority) {
    if (sq_id == UINT32_MAX || !priority) {
        return -1;
    }
    for (unsigned i = 0; i < 2; ++i) {
        if (atomic_load(&sq_ids[i]) == sq_id) {
            *priority = accepted_priorities[i];
            return 0;
        }
    }
    return -1;
}

unsigned sq_test_attempts(void) {
    return atomic_load(&attempts);
}
