// SPDX-License-Identifier: Apache-2.0
#include "../../sparse_flash_attention/op_host/sparse_flash_attention_def.h"

namespace ops {
class SparseFlashAttentionDcpPacket : public SparseFlashAttention {
public:
    explicit SparseFlashAttentionDcpPacket(const char *name) : SparseFlashAttention(name, true, true) {}
};
OP_ADD(SparseFlashAttentionDcpPacket);
} // namespace ops
