#pragma once

#include <cstdint>

namespace vllm_ascend {
// Bitmap plus int32 prefix counts use at most 128 KiB of the 192 KiB A2 UB.
constexpr uint32_t DSA_BITMAP_WINDOW_WORDS = 16384;
constexpr uint32_t DSA_BITMAP_WINDOW_TOKENS = DSA_BITMAP_WINDOW_WORDS * 32;
}  // namespace vllm_ascend
