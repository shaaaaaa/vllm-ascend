// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <vector>
#include "pack_launch.h"
namespace {
void pack(at::TensorList t) {
    TORCH_CHECK(t.size() == 8 && t[0].is_privateuseone() && t[0].dim() == 3, "expected 8 NPU descriptors");
    int64_t r = t[0].size(0), n = t[0].size(1) * t[0].size(2);
    TORCH_CHECK(r > 0 && r <= 1024 && n > 0 && n <= 4096 && n % 256 == 0, "invalid descriptor geometry");
    PackLaunch a{};
    for (size_t i = 0; i < t.size(); ++i) {
        TORCH_CHECK(t[i].device() == t[0].device() && t[i].is_contiguous(), "device/stride mismatch");
        TORCH_CHECK(t[i].scalar_type() == ((i == 2 || i == 3 || i == 6) ? at::kLong : at::kInt), "dtype mismatch");
        a.p[i] = reinterpret_cast<uintptr_t>(t[i].data_ptr());
        TORCH_CHECK(a.p[i] % 32 == 0, "unaligned descriptor");
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(a.p[i] + t[i].nbytes() <= a.p[j] || a.p[j] + t[j].nbytes() <= a.p[i], "aliased descriptors");
        }
    }
    TORCH_CHECK(t[1].sizes() == t[0].sizes(), "token shape mismatch");
    TORCH_CHECK(t[2].sizes().equals(at::IntArrayRef({r,n})) && t[3].sizes() == t[2].sizes(), "slot shape mismatch");
    TORCH_CHECK(t[4].sizes().equals(at::IntArrayRef({r})), "boundary shape mismatch");
    TORCH_CHECK(t[5].sizes().equals(at::IntArrayRef({3,r,n/256,256})) && t[6].sizes() == t[5].sizes(), "payload shape mismatch");
    TORCH_CHECK(t[7].sizes().equals(at::IntArrayRef({3,r,n/256,16})), "count shape mismatch");
    const c10_npu::OptionalNPUGuard guard(t[0].device());
    int64_t cores = 0;
    TORCH_CHECK(aclGetDeviceCapability(t[0].device().index(), ACL_DEVICE_INFO_VECTOR_CORE_NUM, &cores) == ACL_SUCCESS && cores > 0, "cannot query AIV count");
    a.requests = r; a.entries = n; a.cores = cores;
    auto stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand command;
    command.Name("resident_pack_sources");
    command.SetCustomHandler([a, stream, owners = std::vector<at::Tensor>(t.begin(), t.end())]() -> int {
        redesign_pack_sources(stream, a); return 0;
    });
    command.Run();
}
}
TORCH_LIBRARY_FRAGMENT(resident_redesign, m) { m.def("pack_sources_(Tensor(a!)[] tensors) -> ()"); }
TORCH_LIBRARY_IMPL(resident_redesign, PrivateUse1, m) { m.impl("pack_sources_", &pack); }
