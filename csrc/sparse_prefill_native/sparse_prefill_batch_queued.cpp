// SPDX-License-Identifier: Apache-2.0
// One queue entry launches serial request unions on the current compute stream.
// Only owner IDs/status persist; every request reuses the same transient scratch.
// Load the active CANN ACL headers before torch_npu's bundled ACL snapshot.
//
// The target fixed kernel takes no topk and no explicit overflow argument, so
// the raw launcher is called with eleven pointers and four scalars. topk is
// still required because the plan entry runs the prefix union first.
#include <acl/acl.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace {
using TensorList = c10::List<at::Tensor>;
using IntList = c10::List<int64_t>;
using Fused = int32_t (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*,
                         void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);
using Copy = int32_t (*)(void*, void*, void*, void*, void*, void*, void*,
                        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
constexpr int64_t MAX_HISTORY = 524288, MAX_TOKENS = 8192;
constexpr int64_t MAX_REQUESTS = 256, TOPK = 2048, ROW_BYTES = 656, PAGE_ROWS = 128;

std::vector<at::Tensor> KeepIntBuffers(const TensorList& tensors, size_t count, const at::Tensor& reference)
{
    TORCH_CHECK(tensors.size() == count, "Wrong selected union buffer count");
    auto buffers = tensors.vec();
    for (const auto& tensor : buffers) {
        TORCH_CHECK(tensor.device() == reference.device() && tensor.is_contiguous() &&
                    tensor.scalar_type() == at::kInt, "Union buffers must be contiguous int32 on one NPU");
    }
    return buffers;
}

void Plan(const at::Tensor& topk, const TensorList& scratch, const TensorList& outputs,
          const IntList& histories, const IntList& starts, const IntList& queries,
          int64_t capacity, int64_t function, int64_t handle)
{
    TORCH_CHECK(function && handle && topk.device().is_privateuseone() &&
                topk.scalar_type() == at::kInt && topk.is_contiguous(), "Invalid selected TopK");
    TORCH_CHECK(topk.dim() == 3 && topk.size(0) > 0 && topk.size(0) <= MAX_TOKENS &&
                topk.size(1) == 1 && topk.size(2) == TOPK, "Expected original [T,1,2048] TopK");
    const auto h = histories.vec(), start = starts.vec(), q = queries.vec();
    const auto count = static_cast<int64_t>(h.size());
    TORCH_CHECK(count > 0 && count <= MAX_REQUESTS && start.size() == h.size() && q.size() == h.size() &&
                capacity > 0 && capacity <= MAX_HISTORY && capacity % PAGE_ROWS == 0, "Invalid union geometry");
    int64_t previousEnd = 0, largestQuery = 0;
    for (int64_t i = 0; i < count; ++i) {
        TORCH_CHECK(h[i] >= 0 && h[i] <= MAX_HISTORY && q[i] > 0 && q[i] <= topk.size(0) &&
                    start[i] >= previousEnd && start[i] <= topk.size(0) - q[i], "Invalid selected H/Q span");
        previousEnd = start[i] + q[i];
        largestQuery = std::max(largestQuery, q[i]);
    }
    const auto a = KeepIntBuffers(scratch, 6, topk), o = KeepIntBuffers(outputs, 2, topk);
    const int64_t cohorts = (largestQuery + 1) / 2;
    const int64_t needs[] = {cohorts * 4096, cohorts * 8, MAX_HISTORY, 32 * 8, 32 * 8, MAX_HISTORY / 32};
    for (size_t i = 0; i < 6; ++i) TORCH_CHECK(a[i].numel() >= needs[i], "Selected union scratch too small");
    TORCH_CHECK(o[0].dim() == 2 && o[0].size(0) >= count && o[0].size(1) == capacity &&
                o[1].dim() == 2 && o[1].size(0) >= count && o[1].size(1) == 8, "Invalid union output shape");
    c10_npu::NPUGuard guard(topk.device());
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const auto launch = reinterpret_cast<Fused>(uintptr_t(function));
    at_npu::native::OpCommand::RunOpApi("DsaSelectedUnion", [=]() -> int {
        for (int64_t i = 0; i < count; ++i) {
            const auto width = static_cast<uint32_t>(std::max<int64_t>(128, ((h[i] + 4095) / 4096) * 128));
            // Offset the pointer only. The prefix kernel's query-local row is zero.
            const int rc = launch(reinterpret_cast<void*>(uintptr_t(handle)), stream,
                topk.data_ptr<int32_t>() + start[i] * TOPK,
                a[0].data_ptr(), a[1].data_ptr(), a[2].data_ptr(), a[3].data_ptr(), a[4].data_ptr(),
                a[5].data_ptr(), o[0].data_ptr<int32_t>() + i * capacity,
                o[1].data_ptr<int32_t>() + i * 8, h[i], q[i], width, capacity);
            if (rc) return rc;
        }
        return 0;
    });
}

void InplaceCopy(const at::Tensor& ids, const at::Tensor& status, const at::Tensor& pageBases,
                 const at::Tensor& metadata, const at::Tensor& table, const at::Tensor& cache, int64_t function)
{
    TORCH_CHECK(function && ids.device().is_privateuseone(), "Selected copy requires an NPU entry");
    const std::vector<at::Tensor> tensors{ids, status, pageBases, metadata, table, cache};
    for (const auto& tensor : tensors) TORCH_CHECK(tensor.device() == ids.device(), "Selected copy device mismatch");
    for (const auto& tensor : {ids, status, metadata, table}) {
        TORCH_CHECK(tensor.scalar_type() == at::kInt, "Selected IDs/status/metadata/table must be int32");
    }
    for (const auto& tensor : {ids, status, pageBases, metadata, cache}) {
        TORCH_CHECK(tensor.is_contiguous(), "Selected copy buffers must be contiguous");
    }
    TORCH_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(0) <= MAX_REQUESTS &&
                ids.size(1) > 0 && ids.size(1) <= MAX_HISTORY && ids.size(1) % PAGE_ROWS == 0,
                "Invalid selected IDs shape");
    const auto requests = ids.size(0), capacity = ids.size(1);
    TORCH_CHECK(status.dim() == 2 && status.size(0) == requests && status.size(1) == 8 &&
                metadata.dim() == 2 && metadata.size(0) == requests && metadata.size(1) == 4,
                "Invalid selected count or request metadata shape");
    TORCH_CHECK(cache.scalar_type() == at::kChar && cache.dim() == 4 && cache.size(0) > 0 &&
                cache.size(0) <= UINT32_MAX && cache.size(1) == PAGE_ROWS && cache.size(2) == 1 &&
                cache.size(3) == ROW_BYTES, "Selected destination must be native packed C8 pages");
    TORCH_CHECK(pageBases.scalar_type() == at::kUInt64 && pageBases.dim() == 1 && pageBases.numel() > 0 &&
                pageBases.numel() <= cache.size(0), "Invalid selected source page domain");
    TORCH_CHECK(table.dim() == 2 && table.size(0) > 0 && table.size(0) <= MAX_REQUESTS && table.size(1) > 0 &&
                table.size(1) <= UINT32_MAX && table.stride(1) == 1 && table.stride(0) >= table.size(1) &&
                table.stride(0) <= UINT32_MAX, "Original page table needs unit column stride");
    c10_npu::NPUGuard guard(ids.device());
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const auto launch = reinterpret_cast<Copy>(uintptr_t(function));
    at_npu::native::OpCommand::RunOpApi("DsaSelectedInplaceCopy", [=]() -> int {
        return launch(stream, tensors[0].data_ptr(), tensors[1].data_ptr(), tensors[2].data_ptr(),
            tensors[3].data_ptr(), tensors[4].data_ptr(), tensors[5].data_ptr(), requests, capacity,
            pageBases.numel(), table.size(0), table.size(1), table.stride(0), cache.size(0));
    });
}
} // namespace

TORCH_LIBRARY_FRAGMENT(dsa_prefill_queue, m) {
    m.def("batch_plan(Tensor topk, Tensor(a!)[] scratch, Tensor(b!)[] outputs, int[] histories, int[] starts, int[] queries, int capacity, int function, int handle) -> ()");
    m.def("inplace_copy(Tensor ids, Tensor status, Tensor page_bases, Tensor metadata, Tensor table, Tensor(a!) cache, int function) -> ()");
}
TORCH_LIBRARY_IMPL(dsa_prefill_queue, PrivateUse1, m) {
    m.impl("batch_plan", Plan);
    m.impl("inplace_copy", InplaceCopy);
}
