// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

struct ResidentLaunch {
    void* tensors[20];
    uint32_t requests, stateRows, dummyBase, mtp, shards, capacity;
    uint32_t blockSize, blockTableWidth, cores;
};

void resident_experiment_run(void* stream, const ResidentLaunch& args, int stage);
void resident_experiment_run_optimized(
    void* stream, const ResidentLaunch& args, int stage);
void resident_experiment_run_variant(
    void* stream, const ResidentLaunch& args, int variant, int stage);
