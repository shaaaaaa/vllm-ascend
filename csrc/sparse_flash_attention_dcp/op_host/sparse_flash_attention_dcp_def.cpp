// SPDX-License-Identifier: Apache-2.0
#include "../../sparse_flash_attention/op_host/sparse_flash_attention_def.h"

namespace ops {
class SparseFlashAttentionDcp : public SparseFlashAttention {
public:
    explicit SparseFlashAttentionDcp(const char *name) : SparseFlashAttention(name, true) {}
};
OP_ADD(SparseFlashAttentionDcp);
} // namespace ops
