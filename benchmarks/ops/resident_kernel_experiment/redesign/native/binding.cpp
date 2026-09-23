// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <vector>
#include <initializer_list>
#include "launch.h"
namespace {
static_assert(sizeof(uintptr_t) <= sizeof(uint64_t), "device address would be truncated");
void run_impl(at::TensorList t, int64_t universe, int64_t mode, int64_t radius, bool fused, bool batched,
              int64_t tuning = 0, bool wide = false) {
    TORCH_CHECK(tuning == 0 || ((tuning & ~511LL) == 0 && (tuning & 127) >= 1 && (tuning & 127) <= 64),
                "invalid copy/lookup tuning");
    TORCH_CHECK(t.size() == 12, "expected 12 redesign tensors");
    TORCH_CHECK(t[0].is_privateuseone() && t[0].dim() == 3, "requires NPU [B,Q,K] tokens");
    const auto device = t[0].device();
    const auto b = t[0].size(0), q = t[0].size(1), k = t[0].size(2), n = q * k;
    TORCH_CHECK(b > 0 && b <= 1024 && (q == 1 || q == 2) && k > 0 && k <= 2048 && k % 256 == 0,
                "native geometry: B=1..1024, Q=1/2, K a multiple of 256 up to 2048");
    TORCH_CHECK(universe > 0 && universe <= 1048576 && mode >= 0 && mode <= 4 && radius >= 0 && radius <= 32,
                "invalid universe/mode/radius");
    TORCH_CHECK(t[7].dim() == 2 && t[9].dim() == 3, "invalid table/payload rank");
    const auto size = t[7].size(1), width = t[9].size(2);
    TORCH_CHECK(size >= 256 && size % 256 == 0, "table must contain whole 256-cell slices");
    if (mode == 3) {
        TORCH_CHECK(size <= 8192 && (size & (size - 1)) == 0, "hash size must be a power of two <=8192");
    }
    if (mode == 4) {
        TORCH_CHECK(size == ((universe + 255) / 256) * 256, "directory size mismatch");
    }
    if (mode <= 2) {
        TORCH_CHECK(size == 256, "non-table modes require a 256-cell placeholder");
    }
    RedesignLaunch a{};
    for (size_t i = 0; i < t.size(); ++i) {
        TORCH_CHECK(t[i].device() == device && t[i].is_contiguous(), "device/contiguity mismatch at tensor ", i);
        if (i < 9) {
            TORCH_CHECK(t[i].scalar_type() == (i == 6 ? at::kLong : at::kInt), "metadata dtype mismatch");
        } else {
            TORCH_CHECK(t[i].scalar_type() == t[9].scalar_type(), "KV dtypes differ");
        }
        a.p[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(t[i].data_ptr()));
        TORCH_CHECK(a.p[i] % 32 == 0, "unaligned buffer");
        for (size_t j = 0; j < i; ++j) {
            auto p = a.p[i], r = a.p[j];
            TORCH_CHECK(p + t[i].nbytes() <= r || r + t[j].nbytes() <= p, "buffers overlap");
        }
    }
    auto shape = [&](size_t i, std::initializer_list<int64_t> dims) {
        TORCH_CHECK(t[i].sizes().equals(at::IntArrayRef(dims)), "shape mismatch at tensor ", i);
    };
    shape(1, {b,q,k}); shape(2, {b,n}); shape(3, {b,n}); shape(4, {b,n});
    shape(5, {b,q,16}); shape(6, {b,8}); shape(7, {b,size}); shape(8, {b,n});
    shape(9, {b,n,width}); shape(10, {b,universe,width}); shape(11, {b,n,width});
    const auto bytes = width * t[9].element_size();
    TORCH_CHECK((!fused && bytes == 0) || (bytes > 0 && bytes <= 8192 && bytes % 32 == 0),
                "KV row must be 32-byte aligned, <=8192 bytes; zero width is lookup-only");
    const c10_npu::OptionalNPUGuard guard(device);
    int64_t cores = 0;
    TORCH_CHECK(aclGetDeviceCapability(device.index(), ACL_DEVICE_INFO_VECTOR_CORE_NUM, &cores) == ACL_SUCCESS && cores > 0,
                "cannot query AIV count");
    a.requests = b; a.queries = q; a.topk = k; a.universe = universe;
    a.mode = mode; a.radius = radius; a.tableSize = size; a.rowBytes = bytes; a.cores = cores;
    a.tuning = tuning;
    auto stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand command;
    command.Name("resident_redesign");
    command.SetCustomHandler([a, stream, fused, batched, wide, owners = std::vector<at::Tensor>(t.begin(), t.end())]() -> int {
        if (a.mode >= 3) {
            if (wide) redesign_wide_build(stream, a);
            else redesign_build(stream, a);
        }
        if (a.tuning != 0 && fused) redesign_tuned_copy(stream, a);
        else if (a.tuning != 0) redesign_tuned_lookup(stream, a);
        else if (fused && batched) redesign_batched_copy(stream, a);
        else if (fused) redesign_resolve_copy(stream, a);
        else redesign_lookup(stream, a);
        return 0;
    });
    command.Run();
}
void run(at::TensorList t, int64_t universe, int64_t mode, int64_t radius, bool fused) {
    run_impl(t, universe, mode, radius, fused, false);
}
void run_batched(at::TensorList t, int64_t universe, int64_t mode, int64_t radius, bool fused) {
    run_impl(t, universe, mode, radius, fused, true);
}
void run_tuned(at::TensorList t, int64_t universe, int64_t mode, int64_t radius, bool fused,
               bool batched, int64_t tuning, bool wide) {
    run_impl(t, universe, mode, radius, fused, batched, tuning, wide);
}
}
TORCH_LIBRARY(resident_redesign, m) {
    m.def("run_(Tensor(a!)[] tensors, int universe, int mode, int radius, bool fused=False) -> ()");
    m.def("run_batched_(Tensor(a!)[] tensors, int universe, int mode, int radius, bool fused=False) -> ()");
    m.def("run_tuned_(Tensor(a!)[] tensors, int universe, int mode, int radius, bool fused, bool batched, int tuning, bool wide) -> ()");
}
TORCH_LIBRARY_IMPL(resident_redesign, PrivateUse1, m) {
    m.impl("run_", &run);
    m.impl("run_batched_", &run_batched);
    m.impl("run_tuned_", &run_tuned);
}
