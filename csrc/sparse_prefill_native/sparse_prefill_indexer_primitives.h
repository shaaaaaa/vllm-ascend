// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kernel_operator.h"
#include "../lightning_indexer_vllm/op_kernel/lightning_indexer_common.h"
#include "../lightning_indexer_vllm/op_kernel/lightning_indexer_vector.h"

namespace SparsePrefillIndexer {

using LIServiceVec::ExtractIndex;
using LIServiceVec::SortAll;

// M3v19 encodes a logical ID in the raw bits of a finite float sort key.
// Keep its 21-bit domain independent of the indexer's packed payload format.
constexpr uint32_t LONG_SOURCE_BITS = 21U;
constexpr uint32_t LONG_SOURCE_MASK = (1U << LONG_SOURCE_BITS) - 1U;

// SortAll produces interleaved score and uint32 payload words. Pattern 1
// selects the even words as raw bits, paired with ExtractIndex's odd words.
__aicore__ inline void ExtractScoreBits(const AscendC::LocalTensor<uint32_t>& dst,
                                       const AscendC::LocalTensor<uint32_t>& pairs,
                                       uint32_t count)
{
    using namespace AscendC;
    constexpr uint32_t PAIRS_PER_REPEAT =
        LIServiceVec::VEC_REPEAT_BYTES / (2U * sizeof(uint32_t));
    constexpr uint8_t SCORE_PATTERN = 1U;
    AscendC::GatherMaskParams params;
    params.repeatTimes = (count + PAIRS_PER_REPEAT - 1U) / PAIRS_PER_REPEAT;
    params.src0BlockStride = 1;
    params.src0RepeatStride = LIServiceVec::B32_VEC_REPEAT_STRIDE;
    params.src1RepeatStride = 0;
    uint64_t selected = 0;
    AscendC::GatherMask(dst, pairs, SCORE_PATTERN, false, 0U, params, selected);
    AscendC::PipeBarrier<PIPE_V>();
}

} // namespace SparsePrefillIndexer
