// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

struct ResidentLaunch {
    void* tensors[22];
    uint32_t requests, stateRows, dummyBase, mtp, shards, capacity;
    uint32_t blockSize, blockTableWidth, cores;
    uint32_t dcpBits, dcpRank;
};

constexpr int kExactCombinedVariant = 8;
constexpr int kQueryMembershipVariant = 9;

void resident_experiment_run(void* stream, const ResidentLaunch& args, int stage);
void resident_experiment_run_optimized(
    void* stream, const ResidentLaunch& args, int stage);
void resident_experiment_run_variant(
    void* stream, const ResidentLaunch& args, int variant, int stage);
