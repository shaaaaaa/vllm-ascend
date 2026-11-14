// SPDX-License-Identifier: Apache-2.0
// Public ACL launch for the counted-union AscendC kernel containing SyncAll.
// Keep the launch ABI separate from the legacy generator's per-file AIV type.
//
// The target fixed kernel source takes eight user pointers and four scalars
// (no topk, no explicit overflow). The CANN-generated entry still prepends the
// ASCENDC_DUMP pointer and appends the hidden overflow pointer, so the raw
// argument block below matches the generated 104-byte ABI. topk belongs to the
// prefix union launch only.
#include <acl/acl.h>
#include <cstdint>
#include <new>

namespace {
constexpr uint32_t KERNEL_CORES = 32U; // SparsePrefillPlan::CORES
constexpr uint32_t MAX_HISTORY = 524288U;
constexpr uint32_t MAX_BATCHED_TOKENS = 8192U;
constexpr uint32_t PAGE_ROWS = 128U;
// A3 reserves 75 core dump slots. The generated host stub defaults
// ONE_CORE_DUMP_SIZE to 1024, so this buffer must stay 75 * 1024 bytes.
constexpr uint32_t ASSERT_DUMP_BYTES = 75U * 1024U;
constexpr const char* KERNEL_NAME = "sparse_prefill_partition_finish_fixed_0";
} // namespace

extern "C" int32_t launch_sparse_prefill_adaptive_prefix(
    void* stream, void* topk, void* lists, void* descriptors,
    uint32_t history, uint32_t query);

struct SparsePrefillFusedHandle {
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
    void* ffts = nullptr;
    void* dump = nullptr;
    uint32_t maxCapacity = 0U;
};

extern "C" int32_t create_sparse_prefill_fixed_handle(const char* path, uint32_t maxCapacity, void** result)
{
    if (!result) return -1;
    *result = nullptr;
    if (!path || maxCapacity == 0U || maxCapacity > MAX_HISTORY ||
        maxCapacity % PAGE_ROWS != 0U) return -1;
    auto* handle = new (std::nothrow) SparsePrefillFusedHandle();
    if (!handle) return -1;
    auto status = aclrtBinaryLoadFromFile(path, nullptr, &handle->binary);
    if (status == ACL_SUCCESS) {
        status = aclrtBinaryGetFunction(handle->binary, KERNEL_NAME, &handle->function);
    }
    if (status == ACL_SUCCESS) status = aclrtGetHardwareSyncAddr(&handle->ffts);
    if (status == ACL_SUCCESS) {
        status = aclrtMalloc(&handle->dump, ASSERT_DUMP_BYTES, ACL_MEM_MALLOC_NORMAL_ONLY);
    }
    if (status != ACL_SUCCESS) {
        if (handle->binary) aclrtBinaryUnLoad(handle->binary);
        delete handle;
        return status;
    }
    handle->maxCapacity = maxCapacity;
    *result = handle;
    return ACL_SUCCESS;
}

extern "C" int32_t destroy_sparse_prefill_fused_handle(void* opaque)
{
    auto* handle = static_cast<SparsePrefillFusedHandle*>(opaque);
    if (!handle) return ACL_SUCCESS;
    // The caller has retired all queued launches. Retain each live resource if
    // its release fails so the caller can retry without a double free.
    if (handle->binary) {
        const auto status = aclrtBinaryUnLoad(handle->binary);
        if (status != ACL_SUCCESS) return status;
        handle->binary = nullptr;
    }
    if (handle->dump) {
        const auto status = aclrtFree(handle->dump);
        if (status != ACL_SUCCESS) return status;
        handle->dump = nullptr;
    }
    delete handle;
    return ACL_SUCCESS;
}

extern "C" int32_t launch_sparse_prefill_fused_plan(
    void* opaque, void* stream, void* topk, void* lists, void* descriptors, void* membership,
    void* counters, void* groups, void* bitmap, void* sources, void* lengths,
    uint32_t history, uint32_t query, uint32_t width, uint32_t capacity)
{
    if (!opaque || !stream || !topk || !lists || !descriptors || !membership || !counters ||
        !groups || !bitmap || !sources || !lengths || history > MAX_HISTORY ||
        query == 0U || query > MAX_BATCHED_TOKENS || capacity == 0U || capacity > MAX_HISTORY ||
        capacity % PAGE_ROWS != 0U || width < 128U || width > 16384U || width % 128U != 0U ||
        width * 32U < history) return -1;
    auto* handle = static_cast<SparsePrefillFusedHandle*>(opaque);
    if (!handle->binary || !handle->dump) return -1;
    if (handle->maxCapacity != 0U && capacity > handle->maxCapacity) return -1;
    struct Arguments {
        // ASCENDC_DUMP adds this hidden argument before the hardware-sync slot.
        void* dump;
        void* ffts;
        void* lists; void* descriptors; void* membership;
        void* counters; void* groups; void* bitmap; void* sources; void* lengths;
        uint32_t history; uint32_t query; uint32_t width; uint32_t capacity;
        void* overflow;
    } arguments{handle->dump, handle->ffts, lists, descriptors, membership, counters, groups,
                bitmap, sources, lengths, history, query, width, capacity, nullptr};
    aclrtLaunchKernelAttr attribute{};
    attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attribute.value.schemMode = 1; // Every core must be admitted before SyncAll.
    aclrtLaunchKernelCfg config{&attribute, 1};
    const auto status = launch_sparse_prefill_adaptive_prefix(stream, topk, lists, descriptors, history, query);
    if (status != ACL_SUCCESS) return status;
    return aclrtLaunchKernelWithHostArgs(handle->function, KERNEL_CORES, stream, &config,
                                         &arguments, sizeof(arguments), nullptr, 0);
}
