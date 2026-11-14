// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace SparsePrefillLI {
// Two logical-page contracts are supported:
//   M3v19 native KV:             16384 B data, 256 B scale, 16384 B stride
//   target packed mixed indexer: 32768 B data, 512 B scale, 32768 B stride
// The UB staging buffer and the launch bounds use the larger contract.
constexpr uint32_t MAX_DATA_PAGE_BYTES = 32768U;
constexpr uint32_t MAX_SCALE_PAGE_BYTES = 512U;
constexpr uint32_t PAGE_ALIGN_BYTES = 32U;
}

// Source page IDs index mainPageBases. Destination LI logical-block IDs are a
// separate domain because a packed bundle holds 32 main pages but 82 LI logical
// blocks, so b * 32 + m is not b * 82 + k for the same source entry.
//
// destinationIds has one uint32 entry per source page and must come from the
// owning LI block table; it is not derived from the main page ID. Passing
// nullptr keeps destination ID == source page ID, which is the native whole-page
// lease path. The mapped entry point is the packed-layout entry point.
//
// The old launch_dsa_selected_li_pages ABI is unchanged and always takes the
// unmapped path.
extern "C" int32_t launch_dsa_selected_li_pages(void* stream, void* mainPageBases,
    void* targetK, void* targetScale, uint32_t domain, uint32_t destinationPages,
    uint32_t mainOffset, uint32_t dataOffset, uint32_t scaleOffset,
    uint32_t dataBytes, uint32_t scaleBytes, uint32_t destinationStrideBytes);

extern "C" int32_t launch_dsa_selected_li_pages_mapped(void* stream, void* mainPageBases,
    const uint32_t* destinationIds, void* targetK, void* targetScale,
    uint32_t domain, uint32_t destinationPages, uint32_t mainOffset,
    uint32_t dataOffset, uint32_t scaleOffset, uint32_t dataBytes,
    uint32_t scaleBytes, uint32_t destinationStrideBytes);
