// SPDX-License-Identifier: Apache-2.0
// Full LI pages share the original main-KV page IDs and whole-object lease.
#include "selected_kv_assert.h"
#include "li_page_gather.h"

using namespace AscendC;
using namespace SparsePrefillLI;

// Source pages are indexed by the main page table. The destination is either
// the same source page ID (native lease) or an explicit LI logical-block ID
// read from destinationIds (packed layout, where one bundle holds 32 main pages
// but 82 LI logical blocks). The K destination may space logical pages by
// destinationStrideBytes, while the scale sidecar is a separate tightly packed
// slab: scale block d starts at d * scaleBytes (256 B for M3v19 native, 512 B
// for the target packed layout).
extern "C" __global__ __aicore__ void dsa_selected_li_pages(
    GM_ADDR mainPageBases, GM_ADDR destinationIds, GM_ADDR targetK, GM_ADDR targetScale,
    uint32_t domain, uint32_t destinationPages, uint32_t mainOffset, uint32_t dataOffset,
    uint32_t scaleOffset, uint32_t dataBytes, uint32_t scaleBytes,
    uint32_t destinationStrideBytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<uint64_t> bases;
    GlobalTensor<uint32_t> destinations;
    GlobalTensor<uint8_t> destinationK, destinationScale, source;
    bases.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(mainPageBases));
    if (destinationIds != nullptr) {
        destinations.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(destinationIds));
    }
    destinationK.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(targetK));
    destinationScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(targetScale));
    TPipe pipe;
    TBuf<TPosition::VECCALC> bytes;
    pipe.InitBuffer(bytes, MAX_DATA_PAGE_BYTES);
    auto local = bytes.Get<uint8_t>();
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    for (uint64_t page = GetBlockIdx(); page < domain; page += GetBlockNum()) {
        const uint64_t main = bases.GetValue(page);
        // FULL/shared/partial/append pages are left to the original SDK load.
        if (main == 0) continue;
        ASSERT_MSG(main > mainOffset, "Invalid leased LI object base");
        if (main <= mainOffset) return;
        uint64_t destination = page;
        if (destinationIds != nullptr) {
            destination = destinations.GetValue(page);
            ASSERT_MSG(destination < destinationPages, "Invalid LI destination block");
            if (destination >= destinationPages) return;
        }
        const uint64_t blob = main - mainOffset;
        const bool valid = blob <= UINT64_MAX - dataOffset - dataBytes &&
            blob <= UINT64_MAX - scaleOffset - scaleBytes;
        ASSERT_MSG(valid, "LI component address overflows");
        if (!valid) return;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(blob + dataOffset));
        DataCopyPad(local, source, {1, dataBytes, 0, 0, 0}, pad);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(destinationK[destination * destinationStrideBytes], local,
                    {1, dataBytes, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(blob + scaleOffset));
        DataCopyPad(local, source, {1, scaleBytes, 0, 0, 0}, pad);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(destinationScale[destination * scaleBytes], local,
                    {1, scaleBytes, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
}

namespace {
int32_t Launch(void* stream, void* mainPageBases, const uint32_t* destinationIds,
    void* targetK, void* targetScale, uint32_t domain, uint32_t destinationPages,
    uint32_t mainOffset, uint32_t dataOffset, uint32_t scaleOffset,
    uint32_t dataBytes, uint32_t scaleBytes, uint32_t destinationStrideBytes)
{
    if (!stream || !mainPageBases || !targetK || !targetScale || domain == 0U ||
        destinationPages == 0U || (destinationIds == nullptr && domain > destinationPages) ||
        dataBytes == 0U || dataBytes > MAX_DATA_PAGE_BYTES ||
        dataBytes % PAGE_ALIGN_BYTES != 0U || scaleBytes == 0U ||
        scaleBytes > MAX_SCALE_PAGE_BYTES || scaleBytes % PAGE_ALIGN_BYTES != 0U ||
        destinationStrideBytes < dataBytes ||
        destinationStrideBytes % PAGE_ALIGN_BYTES != 0U ||
        dataOffset > UINT32_MAX - dataBytes || scaleOffset > UINT32_MAX - scaleBytes) return -1;
    dsa_selected_li_pages<<<8, nullptr, stream>>>(
        static_cast<uint8_t*>(mainPageBases),
        reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(destinationIds)),
        static_cast<uint8_t*>(targetK), static_cast<uint8_t*>(targetScale),
        domain, destinationPages, mainOffset, dataOffset, scaleOffset,
        dataBytes, scaleBytes, destinationStrideBytes);
    return 0;
}
} // namespace

extern "C" int32_t launch_dsa_selected_li_pages(void* stream, void* mainPageBases,
    void* targetK, void* targetScale, uint32_t domain, uint32_t destinationPages,
    uint32_t mainOffset, uint32_t dataOffset, uint32_t scaleOffset,
    uint32_t dataBytes, uint32_t scaleBytes, uint32_t destinationStrideBytes)
{
    return Launch(stream, mainPageBases, nullptr, targetK, targetScale, domain,
        destinationPages, mainOffset, dataOffset, scaleOffset, dataBytes, scaleBytes,
        destinationStrideBytes);
}

extern "C" int32_t launch_dsa_selected_li_pages_mapped(void* stream, void* mainPageBases,
    const uint32_t* destinationIds, void* targetK, void* targetScale,
    uint32_t domain, uint32_t destinationPages, uint32_t mainOffset,
    uint32_t dataOffset, uint32_t scaleOffset, uint32_t dataBytes,
    uint32_t scaleBytes, uint32_t destinationStrideBytes)
{
    return Launch(stream, mainPageBases, destinationIds, targetK, targetScale, domain,
        destinationPages, mainOffset, dataOffset, scaleOffset, dataBytes, scaleBytes,
        destinationStrideBytes);
}
