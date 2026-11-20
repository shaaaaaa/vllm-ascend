// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <algorithm>
#include <cstdint>
#include <new>
#include <vector>

extern "C" int32_t launch_sparse_prefill_adaptive_prefix(
    void* stream, void* topk, void* lists, void* descriptors, uint32_t history, uint32_t query
);

namespace {

constexpr uint32_t KERNEL_CORES = 32U, ASSERT_DUMP_BYTES = 75U * 1024U;
constexpr int64_t MAX_HISTORY = 524288, MAX_TOKENS = 16384, MAX_REQUESTS = 256;
constexpr int64_t TOPK = 2048, PAGE_ROWS = 128, NOPE_DIM = 512, PE_DIM = 64;

struct SparsePrefillFusedHandle {
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
    void *ffts = nullptr, *dump = nullptr;
    uint32_t maxCapacity = 0U;
};

struct CopyDims {
    uint32_t reqs, cap, tRows, tCols, tStride;
};

using TensorList = c10::List<at::Tensor>;
using IntList = c10::List<int64_t>;
using Fused = int32_t (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);
using MlaCopy = int32_t (*)(void*, void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

std::vector<at::Tensor> KeepIntBuffers(const TensorList& tensors, size_t count, const at::Tensor& ref) {
    TORCH_CHECK(tensors.size() == count, "Wrong selected union buffer count");
    auto out = tensors.vec();
    for (const auto& t : out) {
        TORCH_CHECK(t.device() == ref.device() && t.is_contiguous() && t.scalar_type() == at::kInt, "Union buffers must be contiguous int32 on one NPU");
    }
    return out;
}

bool OkPlane(const at::Tensor& t, const at::Tensor& ref, int64_t pages, int64_t dim) {
    return t.defined() && t.device() == ref.device() && t.is_contiguous() && t.dim() == 4 &&
           pages > 0 && pages <= UINT32_MAX &&
           t.size(0) == pages && t.size(1) == PAGE_ROWS && t.size(2) == 1 && t.size(3) == dim;
}

CopyDims ValidateCopyCommon(
    const at::Tensor& ids, const at::Tensor& status, const at::Tensor& pageBases,
    const at::Tensor& metadata, const at::Tensor& table, int64_t fn
) {
    auto ok = [&](const at::Tensor& t, bool i32 = true, bool contig = true) {
        return t.defined() && t.device() == ids.device() && (!i32 || t.scalar_type() == at::kInt) && (!contig || t.is_contiguous());
    };
    TORCH_CHECK(
        fn && ids.device().is_privateuseone() && ok(ids) && ok(status) && ok(pageBases, false) &&
        ok(metadata) && ok(table, true, false) && ids.dim() == 2 &&
        ids.size(0) > 0 && ids.size(0) <= MAX_REQUESTS &&
        ids.size(1) > 0 && ids.size(1) <= MAX_HISTORY && ids.size(1) % PAGE_ROWS == 0,
        "Invalid selected copy inputs/IDs"
    );
    TORCH_CHECK(
        status.dim() == 2 && status.size(0) == ids.size(0) && status.size(1) == 8 &&
        metadata.dim() == 2 && metadata.size(0) == ids.size(0) && metadata.size(1) == 4 &&
        table.dim() == 2 && table.size(0) > 0 && table.size(0) <= MAX_REQUESTS &&
        table.size(1) > 0 && table.size(1) <= UINT32_MAX &&
        table.stride(1) == 1 && table.stride(0) >= table.size(1) && table.stride(0) <= UINT32_MAX,
        "Invalid status/metadata/table shape"
    );
    return {
        static_cast<uint32_t>(ids.size(0)), static_cast<uint32_t>(ids.size(1)),
        static_cast<uint32_t>(table.size(0)), static_cast<uint32_t>(table.size(1)), static_cast<uint32_t>(table.stride(0)),
    };
}

void Plan(
    const at::Tensor& topk, const TensorList& scratch, const TensorList& outputs,
    const IntList& histories, const IntList& starts, const IntList& queries, int64_t cap, int64_t fn, int64_t handle
) {
    TORCH_CHECK(
        fn && handle && topk.device().is_privateuseone() && topk.scalar_type() == at::kInt &&
        topk.is_contiguous() && topk.dim() == 3 &&
        topk.size(0) > 0 && topk.size(0) <= MAX_TOKENS && topk.size(1) == 1 && topk.size(2) == TOPK,
        "Invalid selected TopK"
    );
    const auto h = histories.vec(), start = starts.vec(), q = queries.vec();
    const auto count = static_cast<int64_t>(h.size());
    TORCH_CHECK(
        count > 0 && count <= MAX_REQUESTS && start.size() == h.size() && q.size() == h.size() &&
        cap > 0 && cap <= MAX_HISTORY && cap % PAGE_ROWS == 0,
        "Invalid union geometry"
    );

    int64_t previousQueryEnd = 0, maxQ = 0;
    for (int64_t i = 0; i < count; ++i) {
        TORCH_CHECK(
            h[i] >= 0 && h[i] <= MAX_HISTORY && q[i] > 0 && q[i] <= topk.size(0) &&
            start[i] >= previousQueryEnd && start[i] <= topk.size(0) - q[i],
            "Invalid selected H/Q span"
        );
        previousQueryEnd = start[i] + q[i];
        maxQ = std::max(maxQ, q[i]);
    }

    const auto a = KeepIntBuffers(scratch, 6, topk), o = KeepIntBuffers(outputs, 2, topk);
    const int64_t ch = (maxQ + 1) / 2, needs[] = {ch * 4096, ch * 8, MAX_HISTORY, 256, 256, MAX_HISTORY / 32};
    for (size_t i = 0; i < 6; ++i) {
        TORCH_CHECK(a[i].numel() >= needs[i], "Selected union scratch too small");
    }
    TORCH_CHECK(
        o[0].dim() == 2 && o[0].size(0) >= count && o[0].size(1) == cap &&
        o[1].dim() == 2 && o[1].size(0) >= count && o[1].size(1) == 8,
        "Invalid union output shape"
    );

    c10_npu::NPUGuard guard(topk.device());
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const auto launch = reinterpret_cast<Fused>(uintptr_t(fn));
    at_npu::native::OpCommand::RunOpApi("DsaSelectedUnion", [=]() -> int {
        for (int64_t i = 0; i < count; ++i) {
            const uint32_t width = static_cast<uint32_t>(std::max<int64_t>(128, ((h[i] + 4095) / 4096) * 128));
            const int rc = launch(
                reinterpret_cast<void*>(uintptr_t(handle)), stream, topk.data_ptr<int32_t>() + start[i] * TOPK,
                a[0].data_ptr(), a[1].data_ptr(), a[2].data_ptr(), a[3].data_ptr(), a[4].data_ptr(), a[5].data_ptr(),
                o[0].data_ptr<int32_t>() + i * cap, o[1].data_ptr<int32_t>() + i * 8,
                static_cast<uint32_t>(h[i]), static_cast<uint32_t>(q[i]), width, static_cast<uint32_t>(cap)
            );
            if (rc != 0) {
                return rc;
            }
        }
        return 0;
    });
}

void MlaInplaceCopy(
    const at::Tensor& ids, const at::Tensor& status, const at::Tensor& pageBases,
    const at::Tensor& metadata, const at::Tensor& table, const at::Tensor& kNope, const at::Tensor& kPe, int64_t fn
) {
    const auto d = ValidateCopyCommon(ids, status, pageBases, metadata, table, fn);
    const int64_t pages = kNope.defined() && kNope.dim() == 4 ? kNope.size(0) : 0;
    TORCH_CHECK(
        OkPlane(kNope, ids, pages, NOPE_DIM) && OkPlane(kPe, ids, pages, PE_DIM) &&
        (kNope.scalar_type() == at::kBFloat16 || kNope.scalar_type() == at::kHalf) &&
        kPe.scalar_type() == kNope.scalar_type() &&
        pageBases.scalar_type() == at::kUInt64 && pageBases.dim() == 2 &&
        pageBases.size(0) > 0 && pageBases.size(0) <= pages && pageBases.size(1) == 2,
        "Selected k_nope/k_pe or page domain mismatch"
    );
    const std::vector<at::Tensor> t{ids, status, pageBases, metadata, table, kNope, kPe};
    const uint32_t sp = static_cast<uint32_t>(pageBases.size(0)), dp = static_cast<uint32_t>(pages);
    c10_npu::NPUGuard guard(ids.device());
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const auto launch = reinterpret_cast<MlaCopy>(uintptr_t(fn));
    at_npu::native::OpCommand::RunOpApi("DsaSelectedMlaInplaceCopy", [=]() -> int {
        return launch(
            stream, t[0].data_ptr(), t[1].data_ptr(), t[2].data_ptr(), t[3].data_ptr(),
            t[4].data_ptr(), t[5].data_ptr(), t[6].data_ptr(), d.reqs, d.cap, sp, d.tRows, d.tCols, d.tStride, dp
        );
    });
}

} // namespace

extern "C" int32_t create_sparse_prefill_fixed_handle(const char* path, uint32_t maxCapacity, void** result) {
    if (!result) {
        return -1;
    }
    *result = nullptr;
    if (!path || maxCapacity == 0U || maxCapacity > MAX_HISTORY || maxCapacity % PAGE_ROWS != 0U) {
        return -1;
    }
    int32_t devId = 0;
    int64_t aivNum = 0;
    if (const auto ds = aclrtGetDevice(&devId); ds != ACL_SUCCESS) {
        return ds;
    }
    if (const auto cs = aclGetDeviceCapability(devId, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aivNum); cs != ACL_SUCCESS) {
        return cs;
    }
    if (aivNum < static_cast<int64_t>(KERNEL_CORES)) {
        return -2;
    }
    auto* h = new (std::nothrow) SparsePrefillFusedHandle();
    if (!h) {
        return -1;
    }
    auto s = aclrtBinaryLoadFromFile(path, nullptr, &h->binary);
    if (s == ACL_SUCCESS) {
        s = aclrtBinaryGetFunction(h->binary, "sparse_prefill_partition_finish_fixed_0", &h->function);
    }
    if (s == ACL_SUCCESS) {
        s = aclrtGetHardwareSyncAddr(&h->ffts);
    }
    if (s == ACL_SUCCESS) {
        s = aclrtMalloc(&h->dump, ASSERT_DUMP_BYTES, ACL_MEM_MALLOC_NORMAL_ONLY);
    }
    if (s != ACL_SUCCESS) {
        if (h->binary) {
            aclrtBinaryUnLoad(h->binary);
        }
        delete h;
        return s;
    }
    h->maxCapacity = maxCapacity;
    *result = h;
    return ACL_SUCCESS;
}

extern "C" int32_t destroy_sparse_prefill_fused_handle(void* opaque) {
    auto* h = static_cast<SparsePrefillFusedHandle*>(opaque);
    if (!h) {
        return ACL_SUCCESS;
    }
    if (h->binary) {
        if (const auto s = aclrtBinaryUnLoad(h->binary); s != ACL_SUCCESS) {
            return s;
        }
        h->binary = nullptr;
    }
    if (h->dump) {
        if (const auto s = aclrtFree(h->dump); s != ACL_SUCCESS) {
            return s;
        }
        h->dump = nullptr;
    }
    delete h;
    return ACL_SUCCESS;
}

extern "C" int32_t launch_sparse_prefill_fused_plan(
    void* opaque, void* stream, void* topk, void* lists, void* descriptors, void* membership,
    void* counters, void* groups, void* bitmap, void* sources, void* lengths,
    uint32_t history, uint32_t query, uint32_t width, uint32_t cap
) {
    auto* h = static_cast<SparsePrefillFusedHandle*>(opaque);
    if (!h || !stream || !topk || !lists || !descriptors || !membership || !counters ||
        !groups || !bitmap || !sources || !lengths || !h->binary || !h->dump ||
        history > MAX_HISTORY || query == 0U || query > MAX_TOKENS ||
        cap == 0U || cap > MAX_HISTORY || cap % PAGE_ROWS != 0U ||
        (h->maxCapacity != 0U && cap > h->maxCapacity) ||
        width < 128U || width > 16384U || width % 128U != 0U || width * 32U < history) {
        return -1;
    }
    struct Args {
        void *dump, *ffts, *lists, *descriptors, *membership, *counters, *groups, *bitmap, *sources, *lengths;
        uint32_t history, query, width, cap;
        void* overflow;
    } args{h->dump, h->ffts, lists, descriptors, membership, counters, groups, bitmap, sources, lengths, history, query, width, cap, nullptr};

    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr.value.schemMode = 1;
    aclrtLaunchKernelCfg cfg{&attr, 1};
    if (const auto s = launch_sparse_prefill_adaptive_prefix(stream, topk, lists, descriptors, history, query); s != ACL_SUCCESS) {
        return s;
    }
    return aclrtLaunchKernelWithHostArgs(h->function, KERNEL_CORES, stream, &cfg, &args, sizeof(args), nullptr, 0);
}

TORCH_LIBRARY_FRAGMENT(dsa_prefill_queue, m) {
    m.def("batch_plan(Tensor topk, Tensor(a!)[] scratch, Tensor(b!)[] outputs, int[] histories, int[] starts, int[] queries, int capacity, int function, int handle) -> ()");
    m.def("mla_inplace_copy(Tensor ids, Tensor status, Tensor page_bases, Tensor metadata, Tensor table, Tensor(a!) k_nope, Tensor(b!) k_pe, int function) -> ()");
}

TORCH_LIBRARY_IMPL(dsa_prefill_queue, PrivateUse1, m) {
    m.impl("batch_plan", Plan);
    m.impl("mla_inplace_copy", MlaInplaceCopy);
}
