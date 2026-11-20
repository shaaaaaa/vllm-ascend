// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifdef NDEBUG
#error "Selected-KV kernels require assertions in Release: compile with -UNDEBUG"
#endif
#if !defined(ASCENDC_DUMP) || ASCENDC_DUMP != 1
#error "Selected-KV kernels require ASCENDC_DUMP=1"
#endif

#include "kernel_operator.h"

#ifndef ASSERT_MSG
#error "Selected-KV kernels require the SDK device ASSERT_MSG API"
#endif

#include "../lightning_indexer_vllm/op_kernel/lightning_indexer_common.h"
#include "../lightning_indexer_vllm/op_kernel/lightning_indexer_vector.h"

namespace SparsePrefillPlan {

using namespace AscendC;
using LIServiceVec::ExtractIndex;
using LIServiceVec::SortAll;

constexpr uint32_t TOPK = 2048U, COHORT_ROWS = 2U, COHORT_SIZE = TOPK * COHORT_ROWS, CORES = 32U;
constexpr uint32_t MAX_PARTITION = 32768U, CHUNK = 4096U, COUNTER_STRIDE = 8U, COHORT_META_GROUPS = 512U, MASK_BYTES = 4096U;
constexpr int32_t SOURCE_KEY_BASE = static_cast<int32_t>(0x40000000U + ((1U << 21U) - 1U));

template <HardEvent E>
__aicore__ inline void Fence() {
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
    SetFlag<E>(id);
    WaitFlag<E>(id);
}

template <class T>
__aicore__ inline void BindGm(GlobalTensor<T>& g, GM_ADDR p) {
    g.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p));
}

template <class T>
__aicore__ inline T LoadGmScalar(GlobalTensor<T>& g, uint64_t idx) {
    DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g[idx]);
    return g.GetValue(idx);
}

template <class D, class A, class B>
__aicore__ inline void BitAnd(D d, A a, B b, uint32_t u16s) {
    And(d.template ReinterpretCast<uint16_t>(), a.template ReinterpretCast<uint16_t>(), b.template ReinterpretCast<uint16_t>(), u16s);
    PipeBarrier<PIPE_V>();
}

template <class D, class A, class B>
__aicore__ inline void BitOr(D d, A a, B b, uint32_t u16s) {
    Or(d.template ReinterpretCast<uint16_t>(), a.template ReinterpretCast<uint16_t>(), b.template ReinterpretCast<uint16_t>(), u16s);
    PipeBarrier<PIPE_V>();
}

template <class D, class S>
__aicore__ inline void Shr32(D d, S s, uint32_t k, uint32_t n) {
    ShiftRight(d.template ReinterpretCast<uint32_t>(), s.template ReinterpretCast<uint32_t>(), k, n);
    PipeBarrier<PIPE_V>();
}

template <class D, class S>
__aicore__ inline void Shl32(D d, S s, uint32_t k, uint32_t n) {
    ShiftLeft(d.template ReinterpretCast<uint32_t>(), s.template ReinterpretCast<uint32_t>(), k, n);
    PipeBarrier<PIPE_V>();
}

template <class T>
__aicore__ inline void VAdd(LocalTensor<T> d, LocalTensor<T> a, LocalTensor<T> b, uint32_t n) {
    Add(d, a, b, n);
    PipeBarrier<PIPE_V>();
}

template <class T>
__aicore__ inline void VSub(LocalTensor<T> d, LocalTensor<T> a, LocalTensor<T> b, uint32_t n) {
    Sub(d, a, b, n);
    PipeBarrier<PIPE_V>();
}

template <class D, class S>
__aicore__ inline void VCast(LocalTensor<D> d, LocalTensor<S> s, RoundMode r, uint32_t n) {
    Cast(d, s, r, n);
    PipeBarrier<PIPE_V>();
}

template <class D, class S>
__aicore__ inline void MaskConst32(D d, S s, LocalTensor<int32_t> c, uint32_t m, uint32_t n) {
    Duplicate(c, static_cast<int32_t>(m), n);
    PipeBarrier<PIPE_V>();
    BitAnd(d, s, c, n * 2U);
}

__aicore__ inline void ExtractScoreBits(const LocalTensor<uint32_t>& dst, const LocalTensor<uint32_t>& pairs, uint32_t count) {
    constexpr uint32_t PAIRS = LIServiceVec::VEC_REPEAT_BYTES / (2U * sizeof(uint32_t));
    GatherMaskParams p;
    p.repeatTimes = (count + PAIRS - 1U) / PAIRS;
    p.src0BlockStride = 1;
    p.src0RepeatStride = LIServiceVec::B32_VEC_REPEAT_STRIDE;
    p.src1RepeatStride = 0;
    uint64_t sel = 0;
    GatherMask(dst, pairs, static_cast<uint8_t>(1U), false, 0U, p, sel);
    PipeBarrier<PIPE_V>();
}

template <class D, class S, class M>
__aicore__ inline uint32_t CompactByMask(D dst, S src, M mask, uint32_t elems) {
    GatherMaskParams p;
    p.repeatTimes = 1;
    p.src0BlockStride = 1;
    p.src0RepeatStride = 8;
    p.src1RepeatStride = 8;
    uint64_t sel = 0U;
    GatherMask(dst.template ReinterpretCast<uint32_t>(), src.template ReinterpretCast<uint32_t>(), mask.template ReinterpretCast<uint32_t>(), true, elems, p, sel);
    Fence<HardEvent::V_S>();
    return static_cast<uint32_t>(sel);
}

__aicore__ inline void SetMeta8(LocalTensor<int32_t> s, int32_t v0 = 0, int32_t v1 = 0, int32_t v2 = 0, int32_t v3 = 0, int32_t v4 = 0) {
    const int32_t v[8] = {v0, v1, v2, v3, v4, 0, 0, 0};
    for (uint32_t i = 0U; i < 8U; ++i) {
        s.SetValue(i, v[i]);
    }
}

__aicore__ inline void StoreMeta8(GlobalTensor<int32_t> gm, LocalTensor<int32_t> s, int32_t v0 = 0, int32_t v1 = 0, int32_t v2 = 0, int32_t v3 = 0, int32_t v4 = 0) {
    SetMeta8(s, v0, v1, v2, v3, v4);
    Fence<HardEvent::S_MTE3>();
    DataCopy(gm, s, 8U);
    Fence<HardEvent::MTE3_S>();
}

struct PartitionUnion {
    __aicore__ inline void Init(
        GM_ADDR lists, GM_ADDR cohorts, GM_ADDR membership, GM_ADDR counters,
        uint32_t, uint32_t query, uint32_t width, TPipe& pipe
    ) {
        BindGm(listsGm, lists);
        BindGm(cohortsGm, cohorts);
        BindGm(membershipGm, membership);
        BindGm(countersGm, counters);
        cohorts_ = (query + COHORT_ROWS - 1U) / COHORT_ROWS;
        width_ = width;
        base_ = GetBlockIdx();
        pipe.InitBuffer(bitmapBuf, MAX_PARTITION * 4U);
        pipe.InitBuffer(inputBuf, CHUNK * 4U);
        pipe.InitBuffer(compactBuf, CHUNK * 4U);
        pipe.InitBuffer(maskBuf, MASK_BYTES * 2U);
        pipe.InitBuffer(counterBuf, COUNTER_STRIDE * 4U);
        pipe.InitBuffer(cohortBuf, COHORT_META_GROUPS * COUNTER_STRIDE * 4U);
    }

    __aicore__ inline void Process() {
        auto bitmap = bitmapBuf.Get<float>(), floats = compactBuf.Get<float>();
        auto input = inputBuf.Get<int32_t>(), compact = compactBuf.Get<int32_t>(), cohort = cohortBuf.Get<int32_t>();
        auto mask = maskBuf.Get<uint8_t>(), other = mask[MASK_BYTES];
        Duplicate(bitmap, 0.0F, width_);
        Fence<HardEvent::V_S>();
        uint32_t error = 0U;

        for (uint32_t base = 0U; base < cohorts_; base += COHORT_META_GROUPS) {
            const uint32_t records = cohorts_ - base < COHORT_META_GROUPS ? cohorts_ - base : COHORT_META_GROUPS;
            if (base != 0U) {
                Fence<HardEvent::S_MTE2>();
            }
            DataCopy(cohort, cohortsGm[base * COUNTER_STRIDE], records * COUNTER_STRIDE);
            Fence<HardEvent::MTE2_S>();

            for (uint32_t local = 0U; local < records; ++local) {
                const uint32_t listCount = static_cast<uint32_t>(cohort.GetValue(local * COUNTER_STRIDE));
                error |= static_cast<uint32_t>(cohort.GetValue(local * COUNTER_STRIDE + 1U));
                if (listCount == 0U) {
                    continue;
                }
                const uint64_t srcBegin = static_cast<uint32_t>(cohort.GetValue(local * COUNTER_STRIDE + 3U));
                for (uint32_t chunk = 0U; chunk < listCount; chunk += CHUNK) {
                    const uint32_t rem = listCount - chunk;
                    const uint32_t count = rem >= CHUNK ? CHUNK : rem;
                    const uint32_t aligned = (count + 63U) / 64U * 64U;
                    DataCopyPad(input, listsGm[srcBegin + chunk], {1, count * 4U, 0, 0, 0}, {false, 0, 0, 0});
                    Fence<HardEvent::MTE2_S>();
                    for (uint32_t i = count; i < aligned; ++i) {
                        input.SetValue(i, -1);
                    }
                    Fence<HardEvent::S_V>();
                    VCast(floats, input, RoundMode::CAST_NONE, aligned);
                    CompareScalar(mask, floats, 0.0F, CMPMODE::GE, aligned);
                    Shr32(compact, input, 5U, aligned);
                    Shl32(compact, compact, 5U, aligned);
                    VSub(compact, input, compact, aligned);
                    VCast(floats, compact, RoundMode::CAST_NONE, aligned);
                    CompareScalar(other, floats, static_cast<float>(static_cast<int32_t>(base_)), CMPMODE::EQ, aligned);
                    PipeBarrier<PIPE_V>();
                    BitAnd(mask, mask, other, aligned / 16U);
                    const uint32_t n = CompactByMask(compact, input, mask, aligned);
                    for (uint32_t i = 0U; i < n; ++i) {
                        bitmap.SetValue(static_cast<uint32_t>(compact.GetValue(i)) / CORES, 1.0F);
                    }
                    Fence<HardEvent::V_MTE2>();
                    Fence<HardEvent::S_MTE2>();
                }
            }
        }

        Fence<HardEvent::S_MTE3>();
        DataCopy(membershipGm[static_cast<uint64_t>(base_) * width_], bitmap.ReinterpretCast<int32_t>(), width_);
        Fence<HardEvent::MTE3_S>();
        StoreMeta8(countersGm[base_ * COUNTER_STRIDE], counterBuf.Get<int32_t>(), 0, 0, static_cast<int32_t>(error));
    }

    GlobalTensor<int32_t> listsGm, cohortsGm, membershipGm, countersGm;
    TBuf<QuePosition::VECCALC> bitmapBuf, inputBuf, compactBuf, maskBuf, counterBuf, cohortBuf;
    uint32_t cohorts_, width_, base_;
};

} // namespace SparsePrefillPlan

namespace SparsePrefillLocal {

using namespace AscendC;
using namespace SparsePrefillPlan;

constexpr uint32_t BUFFER_IDS = 8192U, MERGE_IDS = BUFFER_IDS - 64U, VECTOR = 64U, SHORT_QUERY_ROWS = CORES * COHORT_ROWS * 2U;

__aicore__ inline void MergeSortedRuns(LocalTensor<float> dst, LocalTensor<float> l, LocalTensor<float> r, uint32_t lc, uint32_t rc) {
    constexpr uint32_t MAX_SOURCE_PAIRS = 4095U, SPLIT_PAIRS = 4092U;
    MrgSort4Info p;
    p.elementLengths[0] = lc;
    p.elementLengths[1] = rc;
    p.elementLengths[2] = 0U;
    p.elementLengths[3] = 0U;
    p.ifExhaustedSuspension = false;
    p.validBit = 0b0011;
    p.repeatTimes = 1;
    MrgSortSrcList<float> s{l, r, l, l};
    if (lc > MAX_SOURCE_PAIRS) {
        p.elementLengths[0] = SPLIT_PAIRS;
        p.elementLengths[1] = lc - SPLIT_PAIRS;
        p.elementLengths[2] = rc;
        s.src2 = l[SPLIT_PAIRS * 2U];
        s.src3 = r;
        p.validBit = 0b0111;
    } else if (rc > MAX_SOURCE_PAIRS) {
        p.elementLengths[1] = SPLIT_PAIRS;
        p.elementLengths[2] = rc - SPLIT_PAIRS;
        s.src3 = r[SPLIT_PAIRS * 2U];
        p.validBit = 0b0111;
    }
    MrgSort<float>(dst, s, p);
    PipeBarrier<PIPE_V>();
}

struct LocalQueryUnion {
    __aicore__ inline void Init(GM_ADDR topk, GM_ADDR lists, GM_ADDR descriptors, uint32_t history, uint32_t query, TPipe& pipe) {
        BindGm(topkGm, topk);
        BindGm(listsGm, lists);
        BindGm(descriptorsGm, descriptors);
        history_ = history;
        query_ = query;
        adaptive_ = query > SHORT_QUERY_ROWS;
        pipe.InitBuffer(accBuf, BUFFER_IDS * 8U);
        pipe.InitBuffer(workBuf, BUFFER_IDS * 8U);
        pipe.InitBuffer(scratchBuf, BUFFER_IDS * 4U);
        pipe.InitBuffer(maskBuf, MASK_BYTES * 4U);
        pipe.InitBuffer(stateBuf, COUNTER_STRIDE * 4U);
    }

    __aicore__ inline void Process() {
        const uint32_t cohorts = (query_ + 1U) / 2U, limit = cohorts * (GetBlockIdx() + 1U) / CORES;
        first_ = cohorts * GetBlockIdx() / CORES;
        if (first_ == limit) {
            return;
        }
        auto acc = accBuf.Get<float>(), work = workBuf.Get<float>(), inc = scratchBuf.Get<float>();

        for (uint32_t g = first_; g < limit; ++g) {
            const uint32_t rawCount = ReadPair(g);
            bool direct = false;
            const uint32_t count = Deduplicate(work, inc, work[COHORT_SIZE * 2U], rawCount, adaptive_, &direct);
            if (count == 0U) {
                continue;
            }
            if (direct) {
                if (accCount_ != 0U) {
                    Flush();
                }
                WriteIds(inc.ReinterpretCast<int32_t>(), count);
                continue;
            }
            if (accCount_ != 0U && accCount_ + count > MERGE_IDS) {
                Flush();
            }
            if (accCount_ == 0U) {
                BitOr(acc, inc, inc, count * 4U);
                accCount_ = count;
                continue;
            }
            MergeSortedRuns(work, acc, inc, accCount_, count);
            accCount_ = Deduplicate(work, acc, inc, accCount_ + count);
        }

        if (accCount_ != 0U) {
            Flush();
        }
        Publish(lastCount_, lastOffset_, first_ + (runs_ == 0U ? 0U : runs_ - 1U));

        auto state = stateBuf.Get<int32_t>();
        SetMeta8(state);
        Fence<HardEvent::S_MTE3>();
        for (uint32_t g = first_ + (runs_ == 0U ? 1U : runs_); g < limit; ++g) {
            DataCopy(descriptorsGm[g * COUNTER_STRIDE], state, COUNTER_STRIDE);
        }
        Fence<HardEvent::MTE3_S>();
    }

    __aicore__ inline uint32_t ReadPair(uint32_t group) {
        const uint32_t row = group * 2U, rows = query_ - row < 2U ? query_ - row : 2U, count = rows * TOPK;
        auto input = workBuf.Get<float>(), tmp = input[COHORT_SIZE * 2U];
        auto raw = input[count].ReinterpretCast<int32_t>();
        auto live = maskBuf.Get<uint8_t>(), other = live[MASK_BYTES], bad = live[MASK_BYTES * 2U], extra = live[MASK_BYTES * 3U];

        if (group != first_) {
            Fence<HardEvent::V_MTE2>();
            Fence<HardEvent::S_MTE2>();
        }
        DataCopyPad(raw, topkGm[static_cast<uint64_t>(row) * TOPK], {1, count * 4U, 0, 0, 0}, {false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        VCast(input, raw, RoundMode::CAST_NONE, count);
        CompareScalar(live, input, 0.0F, CMPMODE::GE, count);
        for (uint32_t r = 0U; r < rows; ++r) {
            CompareScalar(other[r * TOPK / 8U], input[r * TOPK], static_cast<float>(static_cast<int32_t>(history_ + row + r)), CMPMODE::LE, TOPK);
        }
        CompareScalar(bad, input, -1.0F, CMPMODE::LT, count);
        CompareScalar(extra, input, static_cast<float>(static_cast<int32_t>(history_ + query_)), CMPMODE::GE, count);
        PipeBarrier<PIPE_V>();
        BitAnd(live, live, other, count / 16U);
        BitOr(bad, bad, extra, count / 16U);
        Fence<HardEvent::V_S>();
        for (uint32_t i = 0U; i < rows * (TOPK / 32U); ++i) {
            error_ |= bad.ReinterpretCast<uint32_t>().GetValue(i);
        }
        Fence<HardEvent::S_V>();
        CompareScalar(other, input, static_cast<float>(static_cast<int32_t>(history_)), CMPMODE::LT, count);
        PipeBarrier<PIPE_V>();
        BitAnd(live, live, other, count / 16U);
        Duplicate(input.ReinterpretCast<int32_t>(), SOURCE_KEY_BASE, count);
        PipeBarrier<PIPE_V>();
        VSub(input.ReinterpretCast<int32_t>(), input.ReinterpretCast<int32_t>(), raw, count);
        Select(input, live, input, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        PipeBarrier<PIPE_V>();
        SortAll(input, tmp, count);
        return count;
    }

    __aicore__ inline void ExpandPairMask() {
        constexpr uint32_t HW = BUFFER_IDS / 16U;
        auto mask = maskBuf.Get<uint8_t>();
        auto flt = mask[MASK_BYTES].ReinterpretCast<float>();
        auto x = mask[MASK_BYTES * 2U].ReinterpretCast<int32_t>(), tmp = mask[MASK_BYTES * 2U + 2048U].ReinterpretCast<int32_t>();
        auto c = mask[MASK_BYTES * 3U].ReinterpretCast<int32_t>();
        VCast(flt, mask.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, HW);
        VCast(x, flt, RoundMode::CAST_RINT, HW);
        Shl32(x, x, 16U, HW);
        Shr32(x, x, 16U, HW);
        const uint32_t shifts[4] = {8U, 4U, 2U, 1U};
        const uint32_t masks[4] = {0x00FF00FFU, 0x0F0F0F0FU, 0x33333333U, 0x55555555U};
        for (uint32_t s = 0U; s < 4U; ++s) {
            Shl32(tmp, x, shifts[s], HW);
            BitOr(x, x, tmp, HW * 2U);
            MaskConst32(x, x, c, masks[s], HW);
        }
        Shl32(tmp, x, 1U, HW);
        BitOr(mask, x, tmp, HW * 2U);
    }

    __aicore__ inline uint32_t Deduplicate(
        LocalTensor<float> src, LocalTensor<float> dst, LocalTensor<float> keys, uint32_t count, bool adaptive = false, bool* direct = nullptr
    ) {
        if (count == 0U) {
            return 0U;
        }
        const uint32_t aligned = (count + VECTOR - 1U) / VECTOR * VECTOR;
        Fence<HardEvent::V_S>();
        for (uint32_t i = count * 2U; i < aligned * 2U; ++i) {
            src.SetValue(i, 0.0F);
        }
        Fence<HardEvent::S_V>();

        ExtractScoreBits(keys.ReinterpretCast<uint32_t>(), src.ReinterpretCast<uint32_t>(), aligned);
        auto off = dst[aligned].ReinterpretCast<int32_t>();
        ArithProgression(off, 0, 1, aligned);
        PipeBarrier<PIPE_V>();
        Adds(off, off, -1, aligned);
        PipeBarrier<PIPE_V>();
        Relu(off, off, aligned);
        PipeBarrier<PIPE_V>();
        Muls(off, off, 4, aligned);
        PipeBarrier<PIPE_V>();
        Gather(dst, keys, off.ReinterpretCast<uint32_t>(), 0U, aligned);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_S>();
        dst.ReinterpretCast<uint32_t>().SetValue(0U, dst.ReinterpretCast<uint32_t>().GetValue(0U) + 1U);
        Fence<HardEvent::S_V>();

        auto mask = maskBuf.Get<uint8_t>();
        Duplicate(mask.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0U), 512U);
        PipeBarrier<PIPE_V>();
        Compare(mask, keys, dst, CMPMODE::NE, aligned);
        CompareScalar(mask[2048U], keys, 0.0F, CMPMODE::GT, aligned);
        PipeBarrier<PIPE_V>();
        BitAnd(mask, mask, mask[2048U], aligned / 16U);

        if (adaptive) {
            auto ids = dst.ReinterpretCast<uint32_t>();
            ExtractIndex(ids, src.ReinterpretCast<uint32_t>(), aligned);
            PipeBarrier<PIPE_V>();
            const uint32_t unique = CompactByMask(ids[aligned], ids, mask, aligned);
            if (unique == 0U || unique * 8ULL > static_cast<uint64_t>(count) * 7ULL) {
                if (unique != 0U) {
                    BitOr(dst, dst[aligned], dst[aligned], unique * 2U);
                    *direct = true;
                }
                return unique;
            }
        }
        ExpandPairMask();
        return CompactByMask(dst, src, mask, aligned * 2U) / 2U;
    }

    __aicore__ inline void Publish(uint32_t count, uint32_t offset, uint32_t desc) {
        StoreMeta8(descriptorsGm[desc * COUNTER_STRIDE], stateBuf.Get<int32_t>(), static_cast<int32_t>(count), error_ != 0U ? 1 : 0, 0, static_cast<int32_t>(offset));
    }

    __aicore__ inline void Flush() {
        auto ids = workBuf.Get<uint32_t>();
        ExtractIndex(ids, accBuf.Get<uint32_t>(), accCount_);
        Fence<HardEvent::V_MTE3>();
        WriteIds(ids.ReinterpretCast<int32_t>(), accCount_);
        accCount_ = 0U;
    }

    __aicore__ inline void WriteIds(LocalTensor<int32_t> ids, uint32_t count) {
        Fence<HardEvent::V_MTE3>();
        lastOffset_ = first_ * COHORT_SIZE + packed_;
        lastCount_ = count;
        DataCopyPad(listsGm[lastOffset_], ids, {1, static_cast<uint16_t>(count * 4U), 0, 0});
        Fence<HardEvent::MTE3_S>();
        Publish(count, lastOffset_, first_ + runs_);
        packed_ += adaptive_ ? (count + COHORT_SIZE - 1U) / COHORT_SIZE * COHORT_SIZE : count;
        ++runs_;
    }

    GlobalTensor<int32_t> topkGm, listsGm, descriptorsGm;
    TBuf<QuePosition::VECCALC> accBuf, workBuf, scratchBuf, maskBuf, stateBuf;
    uint32_t history_, query_, first_ = 0U, accCount_ = 0U, runs_ = 0U, packed_ = 0U, lastCount_ = 0U, lastOffset_ = 0U, error_ = 0U;
    bool adaptive_ = false;
};

} // namespace SparsePrefillLocal

namespace SparsePrefillBitmap {

using namespace AscendC;
using namespace SparsePrefillPlan;

constexpr uint32_t MAX_HISTORY = 524288U, WORDS = MAX_HISTORY / 32U, WORDS_PER_CORE = WORDS / CORES;

struct BuildMembershipWords {
    __aicore__ inline void Init(GM_ADDR membership, GM_ADDR residueCounts, GM_ADDR groupCounts, GM_ADDR bitmap, uint32_t width, TPipe& pipe) {
        BindGm(membershipGm, membership);
        BindGm(residueGm, residueCounts);
        BindGm(groupsGm, groupCounts);
        BindGm(bitmapGm, bitmap);
        width_ = width;
        words_ = ((width + CORES - 1U) / CORES + 63U) / 64U * 64U;
        pipe.InitBuffer(matrixBuf, CORES * words_ * 4U);
        for (uint32_t i = 0U; i < 5U; ++i) {
            pipe.InitBuffer(rows[i], WORDS_PER_CORE * 4U);
        }
        pipe.InitBuffer(bitsBuf, words_ * 4U);
        pipe.InitBuffer(countBuf, COUNTER_STRIDE * 4U);
    }

    __aicore__ inline void Process() {
        const uint32_t firstWord = GetBlockIdx() * words_;
        const uint32_t cols = firstWord >= width_ ? 0U : (width_ - firstWord < words_ ? width_ - firstWord : words_);
        auto matrix = matrixBuf.Get<float>();
        if (cols < words_) {
            Duplicate(matrix, 0.0F, CORES * words_);
            Fence<HardEvent::V_MTE2>();
        }
        if (cols != 0U) {
            for (uint32_t r = 0U; r < CORES; ++r) {
                DataCopyPad(matrix[r * words_], membershipGm[static_cast<uint64_t>(r) * width_ + firstWord], {1, cols * 4U, 0, 0, 0}, {false, 0, 0, 0});
            }
            Fence<HardEvent::MTE2_V>();
        }

        auto offsets = rows[0].Get<int32_t>(), values = rows[1].Get<int32_t>(), tmp = rows[2].Get<int32_t>();
        auto constant = rows[3].Get<int32_t>(), high = rows[4].Get<int32_t>();
        auto bits = bitsBuf.Get<uint32_t>();

        ArithProgression(offsets, 0, 1, WORDS_PER_CORE);
        PipeBarrier<PIPE_V>();
        Shr32(tmp, offsets, 5U, WORDS_PER_CORE);
        Shl32(constant, tmp, 5U, WORDS_PER_CORE);
        VSub(offsets, offsets, constant, WORDS_PER_CORE);
        Muls(offsets, offsets, static_cast<int32_t>(words_), WORDS_PER_CORE);
        PipeBarrier<PIPE_V>();
        VAdd(offsets, offsets, tmp, WORDS_PER_CORE);
        Muls(offsets, offsets, 4, WORDS_PER_CORE);
        PipeBarrier<PIPE_V>();

        for (uint32_t b = 0U; b < words_ / 16U; ++b) {
            Gather(values.ReinterpretCast<float>(), matrix, offsets.ReinterpretCast<uint32_t>(), 0U, WORDS_PER_CORE);
            PipeBarrier<PIPE_V>();
            CompareScalar(bits.ReinterpretCast<uint8_t>()[b * 64U], values.ReinterpretCast<float>(), 0.0F, CMPMODE::GT, WORDS_PER_CORE);
            PipeBarrier<PIPE_V>();
            if (b + 1U < words_ / 16U) {
                Adds(offsets, offsets, 64, WORDS_PER_CORE);
                PipeBarrier<PIPE_V>();
            }
        }

        Shr32(high, bits, 31U, words_);
        MaskConst32(values, bits, constant, 0x7FFFFFFFU, words_);
        Shr32(tmp, values, 1U, words_);
        MaskConst32(tmp, tmp, constant, 0x55555555U, words_);
        VSub(values, values, tmp, words_);
        MaskConst32(tmp, values, constant, 0x33333333U, words_);
        Shr32(values, values, 2U, words_);
        BitAnd(values, values, constant, words_ * 2U);
        VAdd(values, values, tmp, words_);
        Shr32(tmp, values, 4U, words_);
        VAdd(values, values, tmp, words_);
        MaskConst32(values, values, constant, 0x0F0F0F0FU, words_);
        for (uint32_t s = 8U; s <= 16U; s <<= 1U) {
            Shr32(tmp, values, s, words_);
            VAdd(values, values, tmp, words_);
        }
        MaskConst32(values, values, constant, 63U, words_);
        VAdd(values, values, high, words_);
        VCast(high.ReinterpretCast<float>(), values, RoundMode::CAST_NONE, words_);
        ReduceSum(constant.ReinterpretCast<float>(), high.ReinterpretCast<float>(), tmp.ReinterpretCast<float>(), words_);
        Fence<HardEvent::V_S>();

        auto stats = countBuf.Get<int32_t>();
        SetMeta8(stats, static_cast<int32_t>(constant.ReinterpretCast<float>().GetValue(0U)), LoadGmScalar(residueGm, GetBlockIdx() * 8U + 2U), static_cast<int32_t>(words_));
        Fence<HardEvent::S_MTE3>();
        DataCopy(bitmapGm[firstWord], bits, words_);
        DataCopy(groupsGm[GetBlockIdx() * 8U], stats, 8U);
        Fence<HardEvent::MTE3_S>();
    }

    GlobalTensor<float> membershipGm;
    GlobalTensor<int32_t> residueGm, groupsGm;
    GlobalTensor<uint32_t> bitmapGm;
    TBuf<QuePosition::VECCALC> matrixBuf, rows[5], bitsBuf, countBuf;
    uint32_t width_, words_;
};

struct FinishSortedBitmap {
    __aicore__ inline void Init(GM_ADDR groups, GM_ADDR bitmap, GM_ADDR sources, GM_ADDR lengths, uint32_t query, uint32_t width, uint32_t capacity, TPipe& pipe) {
        BindGm(groupsGm, groups);
        BindGm(bitmapGm, bitmap);
        BindGm(sourcesGm, sources);
        BindGm(lengthsGm, lengths);
        query_ = query;
        capacity_ = capacity;
        words_ = ((width + CORES - 1U) / CORES + 63U) / 64U * 64U;
        ASSERT_MSG(words_ >= 64U && words_ <= WORDS_PER_CORE && words_ % 64U == 0U, "Invalid compact bitmap extent");
        pipe.InitBuffer(bitsBuf, words_ * 4U);
        pipe.InitBuffer(inputBuf, 8192U * 4U);
        pipe.InitBuffer(outputBuf, 8192U * 4U);
        pipe.InitBuffer(lengthBuf, 32U);
        pipe.InitBuffer(allGroupBuf, CORES * 32U);
    }

    __aicore__ inline void Process() {
        uint32_t before = 0U, total = 0U, error = 0U;
        auto allGroups = allGroupBuf.Get<int32_t>();
        DataCopy(allGroups, groupsGm, CORES * 8U);
        Fence<HardEvent::MTE2_S>();

        for (uint32_t g = 0U; g < CORES; ++g) {
            const uint32_t c = static_cast<uint32_t>(allGroups.GetValue(g * 8U));
            if (g < GetBlockIdx()) {
                before += c;
            }
            total += c;
            error |= static_cast<uint32_t>(allGroups.GetValue(g * 8U + 1U));
        }

        if (total > capacity_ || error != 0U) {
            if (GetBlockIdx() == 0U) {
                WriteLengths(total, static_cast<int32_t>(error), total > capacity_ ? 1 : 0);
            }
            return;
        }

        auto bits = bitsBuf.Get<uint32_t>();
        const uint32_t firstWord = GetBlockIdx() * words_;
        DataCopy(bits, bitmapGm[firstWord], words_);
        Fence<HardEvent::MTE2_V>();
        if (GetBlockIdx() == 0U) {
            WriteLengths(total, 0, 0);
        }

        auto input = inputBuf.Get<int32_t>(), output = outputBuf.Get<int32_t>();
        Fence<HardEvent::S_V>();
        uint32_t packed = 0U;

        for (uint32_t first = 0U; first < words_ * 32U; first += 8192U) {
            const uint32_t rows = words_ * 32U - first < 8192U ? words_ * 32U - first : 8192U;
            ArithProgression(input, static_cast<int32_t>(firstWord * 32U + first), 1, rows);
            PipeBarrier<PIPE_V>();
            const uint32_t sel = CompactByMask(output, input, bits[first / 32U], rows);
            ASSERT_MSG(before + packed + sel <= capacity_, "Sorted union write exceeds capacity");
            if (before + packed + sel > capacity_) {
                return;
            }
            if (sel != 0U) {
                Fence<HardEvent::S_MTE3>();
                DataCopyPad(sourcesGm[before + packed], output, {1, static_cast<uint16_t>(sel * 4U), 0, 0});
                Fence<HardEvent::MTE3_S>();
            }
            packed += sel;
            Fence<HardEvent::S_V>();
        }
    }

    __aicore__ inline void WriteLengths(uint32_t total, int32_t err, int32_t ovf) {
        StoreMeta8(lengthsGm, lengthBuf.Get<int32_t>(), static_cast<int32_t>(total), static_cast<int32_t>(query_), static_cast<int32_t>(total + query_), err, ovf);
    }

    GlobalTensor<int32_t> groupsGm, sourcesGm, lengthsGm;
    GlobalTensor<uint32_t> bitmapGm;
    TBuf<QuePosition::VECCALC> bitsBuf, inputBuf, outputBuf, lengthBuf, allGroupBuf;
    uint32_t query_, capacity_, words_;
};

} // namespace SparsePrefillBitmap
