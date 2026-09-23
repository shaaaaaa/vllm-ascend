// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstddef>
struct RedesignLaunch {
    // Transport addresses as integers: the device compiler cannot reinterpret
    // a generic void* as a __gm__ pointer across address spaces.
    uint64_t p[12];
    uint32_t requests, queries, topk, universe, mode, radius, tableSize, rowBytes, cores;
    uint32_t tuning; // low 7 bits: copy rows; bit 7: pipeline; bit 8: interior lookup
};
static_assert(offsetof(RedesignLaunch, requests) == 96, "launch address layout changed");
static_assert(sizeof(RedesignLaunch) == 136, "host/device launch ABI changed");
void redesign_build(void* stream, const RedesignLaunch& a);
void redesign_lookup(void* stream, const RedesignLaunch& a);
void redesign_resolve_copy(void* stream, const RedesignLaunch& a);
void redesign_batched_copy(void* stream, const RedesignLaunch& a);
void redesign_tuned_copy(void* stream, const RedesignLaunch& a);
void redesign_tuned_lookup(void* stream, const RedesignLaunch& a);
void redesign_wide_build(void* stream, const RedesignLaunch& a);
