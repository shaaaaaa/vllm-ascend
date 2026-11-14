// SPDX-License-Identifier: Apache-2.0
// Host contract for dsa_prefill_queue::li_page_gather:
//   layout="native": 16384 B K / 256 B scale / 16384 B stride per page.
//     destination_ids is optional; when omitted the destination ID equals the
//     source page ID, which preserves the M3v19 native whole-page lease.
//   layout="packed": 32768 B K / 512 B scale / 32768 B stride per logical LI
//     block. destination_ids is required and maps every source main page to
//     the LI logical block that owns the same data; a packed bundle has 32 main
//     pages but 82 LI logical blocks, so source and destination IDs differ.
// A per-extent C8 view exposes 16384 B K and 256 B scale per extent and is
// therefore rejected by layout="packed". It cannot be silently reinterpreted
// as a logical 32 KiB LI page. Per-extent destinations would need an explicit
// extent contract with separate destination offsets, which this op does not
// provide.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <optional>
#include <string>
#include "li_page_gather.h"

namespace {
using namespace SparsePrefillLI;

void LIPageGather(const at::Tensor& mainPageBases, const at::Tensor& targetK,
                  const at::Tensor& targetScale, int64_t mainOffset,
                  int64_t dataOffset, int64_t scaleOffset, const std::string& layout,
                  const std::optional<at::Tensor>& destinationIds)
{
    const bool native = layout == "native";
    TORCH_CHECK(native || layout == "packed",
                "LI layout must be 'native' or 'packed', got '", layout, "'");
    TORCH_CHECK(mainPageBases.device().is_privateuseone() &&
                mainPageBases.scalar_type() == at::kUInt64 && mainPageBases.dim() == 1 &&
                mainPageBases.numel() > 0 && mainPageBases.numel() <= UINT32_MAX,
                "LI source must be an NPU uint64 page-base table");
    for (const auto& tensor : {mainPageBases, targetK, targetScale}) {
        TORCH_CHECK(tensor.device() == mainPageBases.device() && tensor.is_contiguous(),
                    "LI source and native caches must be contiguous on one NPU");
    }
    TORCH_CHECK(targetK.scalar_type() == at::kChar && targetK.dim() >= 2 &&
                targetScale.scalar_type() == at::kHalf && targetScale.dim() >= 2 &&
                targetK.size(0) > 0 && targetK.size(0) <= UINT32_MAX &&
                targetScale.size(0) == targetK.size(0) &&
                mainPageBases.numel() <= targetK.size(0),
                "LI destination cache geometry is invalid");
    const uint32_t dataBytes = static_cast<uint32_t>(targetK.numel() / targetK.size(0));
    const uint32_t scaleBytes = static_cast<uint32_t>(
        targetScale.numel() / targetScale.size(0) * static_cast<int64_t>(sizeof(int16_t)));
    TORCH_CHECK((!native && dataBytes == 32768U && scaleBytes == 512U) ||
                (native && dataBytes == 16384U && scaleBytes == 256U),
                "LI layout '", layout, "' requires ",
                native ? "16384-byte K and 256-byte scale pages"
                       : "logical 32768-byte K and 512-byte scale pages; a per-extent "
                         "[slots*164,128,1,128] / [slots*164,128,1,1] view is not a logical LI page");
    TORCH_CHECK(mainOffset >= 0 && mainOffset <= UINT32_MAX && dataOffset >= 0 &&
                dataOffset <= UINT32_MAX - dataBytes && scaleOffset >= 0 &&
                scaleOffset <= UINT32_MAX - scaleBytes,
                "LI component offsets exceed the registered object geometry");
    TORCH_CHECK(native || destinationIds.has_value(),
                "LI layout 'packed' requires destination IDs from the owning LI block table");
    std::optional<at::Tensor> destinationMap;
    if (destinationIds.has_value()) {
        destinationMap = destinationIds;
        const auto& ids = *destinationMap;
        TORCH_CHECK(ids.defined() && ids.device() == mainPageBases.device() &&
                    ids.is_contiguous() && ids.scalar_type() == at::kInt &&
                    ids.dim() == 1 && ids.numel() == mainPageBases.numel(),
                    "LI destination IDs must be contiguous int32 on the source device, "
                    "one per source page");
    }
    c10_npu::NPUGuard guard(mainPageBases.device());
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand::RunOpApi("DsaLIPageGather", [=]() -> int {
        return launch_dsa_selected_li_pages_mapped(stream, mainPageBases.data_ptr(),
            destinationMap.has_value()
                ? reinterpret_cast<const uint32_t*>(destinationMap->data_ptr<int32_t>())
                : nullptr,
            targetK.data_ptr(), targetScale.data_ptr(), mainPageBases.numel(),
            targetK.size(0), mainOffset, dataOffset, scaleOffset, dataBytes, scaleBytes,
            dataBytes);
    });
}
} // namespace

TORCH_LIBRARY_FRAGMENT(dsa_prefill_queue, m) {
    m.def("li_page_gather(Tensor main_page_bases, Tensor(a!) target_k, Tensor(b!) target_scale, int main_offset, int data_offset, int scale_offset, str layout, Tensor? destination_ids=None) -> ()");
}
TORCH_LIBRARY_IMPL(dsa_prefill_queue, PrivateUse1, m) {
    m.impl("li_page_gather", LIPageGather);
}
