// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include <cstdint>

// One 64-byte cache line per block. The two streams write disjoint addresses.
extern "C" __global__ __aicore__ void stream_priority_pulse_kernel(__gm__ uint64_t* stamps, uint64_t hold_ticks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  const uint64_t start = AscendC::GetSystemCycle();
  uint64_t end;
  do {
    end = AscendC::GetSystemCycle();
  } while (end - start < hold_ticks);

  AscendC::GlobalTensor<uint64_t> row;
  row.SetGlobalBuffer(stamps + AscendC::GetBlockIdx() * 8);
  row.SetValue(0, start);
  row.SetValue(1, end);
  AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
      row);
}

// Each iteration submits one independent kernel. Using one host call for the
// short train keeps the Python interpreter out of its submission interval.
extern "C" __attribute__((visibility("default"))) int32_t stream_priority_submit(uint32_t blocks, void* stream,
                                                                                 uint64_t* stamps, uint32_t tasks,
                                                                                 uint64_t hold_ticks) {
  for (uint32_t task = 0; task < tasks; ++task) {
    uint64_t* row = stamps + static_cast<uint64_t>(task) * blocks * 8;
    const auto rc = stream_priority_pulse_kernel<<<blocks, nullptr, stream>>>(row, hold_ticks);
    if (rc != 0) {
      return static_cast<int32_t>(rc);
    }
  }
  return 0;
}
