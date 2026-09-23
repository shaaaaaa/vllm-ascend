// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstddef>
struct PackLaunch {
    uint64_t p[8];
    uint32_t requests, entries, cores;
};
static_assert(sizeof(PackLaunch) == 80 && offsetof(PackLaunch, requests) == 64, "pack host/device ABI changed");
void redesign_pack_sources(void* stream, const PackLaunch& a);
