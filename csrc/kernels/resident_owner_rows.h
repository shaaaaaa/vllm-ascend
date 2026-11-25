// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// Host launch arguments only. The device entry point receives ordinary GM
// pointers and scalars. Replicated kernels have no owner-layout arguments.
struct ResidentOwnerRows {
    void *gathered, *inverse, *lanes, *positions, *published;
    uint32_t gatheredRows, inverseRows, modelRows, actualRows;
    uint32_t positionBytes, publishedStride, reject;
};
