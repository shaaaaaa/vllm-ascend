// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <initializer_list>
#include <vector>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include "launch.h"

namespace {
void run(at::TensorList tensors, int64_t dummyBase, int64_t blockSize,
         bool optimized, int64_t stage)
{
    TORCH_CHECK(tensors.size() == 20, "expected 20 resident tensors");
    const auto device = tensors[0].device();
    TORCH_CHECK(tensors[0].is_privateuseone(), "resident experiment requires NPU tensors");
    TORCH_CHECK(stage >= 0 && stage <= 3, "stage must be 0..3");
    TORCH_CHECK(blockSize > 0 && blockSize <= 4096, "invalid block size");
    TORCH_CHECK(tensors[3].dim() == 3 && tensors[8].dim() == 3 &&
                tensors[19].dim() == 2 && tensors[0].dim() == 3,
                "invalid resident tensor rank");
    const int64_t requests = tensors[3].size(0);
    const int64_t shards = tensors[3].size(1);
    const int64_t capacity = tensors[3].size(2);
    const int64_t mtp = capacity / 2048;
    const int64_t stateRows = tensors[8].size(0);
    const int64_t tableWidth = tensors[19].size(1);
    TORCH_CHECK(requests > 0 && requests <= 4096, "invalid request count");
    TORCH_CHECK(shards > 0 && shards <= 8 && (shards & (shards - 1)) == 0,
                "shards must be a power of two up to 8");
    TORCH_CHECK((mtp == 1 || mtp == 2) && capacity == mtp * 2048,
                "only query widths 1/2 with top-k 2048 are supported");
    TORCH_CHECK(dummyBase > 0 && dummyBase <= 65536 &&
                stateRows >= dummyBase + requests && stateRows <= 131072,
                "persistent/dummy state rows are incomplete");
    TORCH_CHECK(tableWidth >= (capacity + blockSize - 1) / blockSize &&
                tableWidth <= 65536, "block table does not cover scratch capacity");

    constexpr at::ScalarType types[] = {
        at::kInt, at::kInt, at::kInt, at::kInt, at::kShort, at::kInt,
        at::kInt, at::kLong, at::kInt, at::kShort, at::kInt, at::kLong,
        at::kShort, at::kInt, at::kShort, at::kShort, at::kInt, at::kInt,
        at::kLong, at::kInt};
    ResidentLaunch a{};
    for (size_t i = 0; i < tensors.size(); ++i) {
        TORCH_CHECK(tensors[i].device() == device && tensors[i].is_contiguous() &&
                    tensors[i].scalar_type() == types[i],
                    "tensor ", i, " has the wrong device/layout/dtype");
        a.tensors[i] = tensors[i].data_ptr();
        TORCH_CHECK(reinterpret_cast<uintptr_t>(a.tensors[i]) % 64 == 0,
                    "resident buffers must be cacheline aligned");
    }
    auto shape = [&](size_t i, std::initializer_list<int64_t> dims) {
        TORCH_CHECK(tensors[i].sizes().equals(at::IntArrayRef(dims)),
                    "tensor ", i, " has the wrong shape");
    };
    shape(0, {requests * mtp, 1, 2048});
    shape(1, {requests * mtp}); shape(2, {requests * mtp});
    for (size_t i : {3, 4, 12, 13, 14, 15}) shape(i, {requests, shards, capacity});
    shape(5, {requests, shards, 16});
    shape(6, {requests}); shape(7, {requests});
    shape(8, {stateRows, shards, capacity}); shape(9, {stateRows, shards, capacity});
    shape(10, {stateRows, shards, 16}); shape(11, {stateRows, 8});
    shape(16, {requests, capacity}); shape(17, {requests, 16});
    shape(18, {requests, capacity}); shape(19, {requests, tableWidth});
    // The harness requires separate buffers. Reject aliases that could let one
    // stage overwrite another's input or defeat cross-core single-writer rules.
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto begin = reinterpret_cast<uintptr_t>(a.tensors[i]);
        for (size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<uintptr_t>(a.tensors[j]);
            TORCH_CHECK(begin + tensors[i].nbytes() <= other ||
                        other + tensors[j].nbytes() <= begin, "resident buffers overlap");
        }
    }
    const c10_npu::OptionalNPUGuard guard(device);
    int64_t cores = 0;
    TORCH_CHECK(aclGetDeviceCapability(device.index(), ACL_DEVICE_INFO_VECTOR_CORE_NUM,
                                      &cores) == ACL_SUCCESS && cores > 0,
                "cannot query AIV count");
    a.requests = requests; a.stateRows = stateRows; a.dummyBase = dummyBase;
    a.mtp = mtp; a.shards = shards; a.capacity = capacity;
    a.blockSize = blockSize; a.blockTableWidth = tableWidth; a.cores = cores;
    auto stream = c10_npu::getCurrentNPUStream().stream();
    // Use the same task-queue/stream path as production. A raw ctypes launch
    // could overtake queued torch-npu copies even if it uses the same stream.
    at_npu::native::OpCommand command;
    command.Name(optimized ? "resident_experiment_optimized" : "resident_experiment_baseline");
    command.SetCustomHandler([stream, a, optimized, stage,
                              owners = std::vector<at::Tensor>(tensors.begin(), tensors.end())]() -> int {
        if (optimized) resident_experiment_run_optimized(stream, a, stage);
        else resident_experiment_run(stream, a, stage);
        return 0;
    });
    command.Run();
}
}  // namespace

TORCH_LIBRARY(resident_experiment, m) {
    m.def("run_(Tensor(a!)[] tensors, int dummy_base, int block_size, "
          "bool optimized=False, int stage=0) -> ()");
}
TORCH_LIBRARY_IMPL(resident_experiment, PrivateUse1, m) {
    m.impl("run_", &run);
}
