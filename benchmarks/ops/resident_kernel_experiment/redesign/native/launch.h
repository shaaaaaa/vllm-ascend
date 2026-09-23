// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
struct RedesignLaunch {
    void* p[12];
    uint32_t requests, queries, topk, universe, mode, radius, tableSize, rowBytes, cores;
};
void redesign_build(void* stream, const RedesignLaunch& a);
void redesign_lookup(void* stream, const RedesignLaunch& a);
void redesign_resolve_copy(void* stream, const RedesignLaunch& a);
