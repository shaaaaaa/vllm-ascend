#ifndef SPARSE_PREFILL_LOCAL_UNION_SERVICES_H
#define SPARSE_PREFILL_LOCAL_UNION_SERVICES_H
// SPDX-License-Identifier: Apache-2.0
#include "sparse_prefill_plan_services.h"

namespace SparsePrefillLocal {
using namespace AscendC;
using SparsePrefillPlan::Fence;
using SparsePrefillPlan::CompactParams;
constexpr uint32_t CORES = SparsePrefillPlan::CORES;
constexpr uint32_t STRIDE = SparsePrefillPlan::COUNTER_STRIDE;
constexpr uint32_t BASE_IDS = SparsePrefillPlan::COHORT_SIZE;
constexpr uint32_t TOPK = SparsePrefillPlan::TOPK;
constexpr uint32_t BUFFER_IDS = 8192U;
// Keep the padded interleaved ExtractIndex below its 255-repeat limit.
constexpr uint32_t MERGE_IDS = BUFFER_IDS - 64U;
constexpr uint32_t MASK_BYTES = 4096U;
constexpr uint32_t VECTOR = 64U;
// At most two cohorts per producer need no amortized layout probe. The
// bounded merge already fits in UB; longer inputs use density adaptation.
constexpr uint32_t SHORT_QUERY_ROWS = CORES * SparsePrefillPlan::COHORT_ROWS * 2U;
constexpr uint32_t LOW_OVERLAP_NUMERATOR = 7U;
constexpr uint32_t LOW_OVERLAP_DENOMINATOR = 8U;

// MrgSort accepts at most 4095 pairs per source. Split at 4092 to keep
// the next source 32-byte aligned (each score/index pair occupies 8 bytes).
// The combined MERGE_IDS bound means at most one input needs splitting.
__aicore__ inline void MergeSortedRuns(LocalTensor<float> dst, LocalTensor<float> left,
    LocalTensor<float> right, uint32_t leftCount, uint32_t rightCount)
{
    constexpr uint32_t MAX_SOURCE_PAIRS = 4095U, SPLIT_PAIRS = 4092U;
    MrgSort4Info params;
    params.elementLengths[0] = leftCount;
    params.elementLengths[1] = rightCount;
    params.elementLengths[2] = 0U;
    params.elementLengths[3] = 0U;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b0011; params.repeatTimes = 1;
    MrgSortSrcList<float> sources;
    sources.src1 = left; sources.src2 = right;
    sources.src3 = left; sources.src4 = left;
    if (leftCount > MAX_SOURCE_PAIRS) {
        params.elementLengths[0] = SPLIT_PAIRS;
        params.elementLengths[1] = leftCount - SPLIT_PAIRS;
        params.elementLengths[2] = rightCount;
        sources.src2 = left[SPLIT_PAIRS * 2U]; sources.src3 = right;
        params.validBit = 0b0111;
    } else if (rightCount > MAX_SOURCE_PAIRS) {
        params.elementLengths[1] = SPLIT_PAIRS;
        params.elementLengths[2] = rightCount - SPLIT_PAIRS;
        sources.src3 = right[SPLIT_PAIRS * 2U];
        params.validBit = 0b0111;
    }
    MrgSort<float>(dst, sources, params);
    PipeBarrier<PIPE_V>();
}

class LocalQueryUnion {
public:
    __aicore__ inline void Init(GM_ADDR topk, GM_ADDR lists, GM_ADDR descriptors,
        uint32_t history, uint32_t query, TPipe& pipe)
    {
        topkGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topk));
        listsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lists));
        descriptorsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(descriptors));
        history_ = history; query_ = query;
        adaptive_ = query > SHORT_QUERY_ROWS;
        pipe.InitBuffer(accBuf, BUFFER_IDS * 2U * sizeof(float));
        pipe.InitBuffer(workBuf, BUFFER_IDS * 2U * sizeof(float));
        pipe.InitBuffer(scratchBuf, BUFFER_IDS * sizeof(float));
        pipe.InitBuffer(maskBuf, MASK_BYTES * 4U);
        pipe.InitBuffer(stateBuf, STRIDE * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t cohorts = (query_ + 1U) / 2U;
        first_ = cohorts * GetBlockIdx() / CORES;
        const uint32_t limit = cohorts * (GetBlockIdx() + 1U) / CORES;
        if (first_ == limit) return;
        auto acc = accBuf.Get<float>();
        auto work = workBuf.Get<float>();
        auto incoming = scratchBuf.Get<float>();
        for (uint32_t group = first_; group < limit; ++group) {
            const uint32_t rawCount = ReadPair(group);
            bool direct = false;
            const uint32_t count = Deduplicate(work, incoming, work[BASE_IDS * 2U], rawCount, adaptive_, &direct);
            if (count == 0U) continue;
            if (direct) {
                if (accCount_ != 0U) Flush();
                WriteIds(incoming.ReinterpretCast<int32_t>(), count);
                continue;
            }
            if (accCount_ != 0U && accCount_ + count > MERGE_IDS) Flush();
            if (accCount_ == 0U) {
                BitCopy(acc, incoming, count * 2U);
                accCount_ = count;
                continue;
            }
            mergedLayout_ = true;
            MergeSortedRuns(work, acc, incoming, accCount_, count);
            accCount_ = Deduplicate(work, acc, incoming, accCount_ + count);
        }
        if (accCount_ != 0U) Flush();
        // Empty/future-only inputs and errors after the last nonempty run
        // still publish the cumulative validation flags through MTE3.
        Publish(lastCount_, lastOffset_, first_ + (runs_ == 0U ? 0U : runs_ - 1U));
        // Reset only descriptors that were not produced. A low-overlap
        // fixed layout writes every record exactly once, like legacy MTP.
        auto state = stateBuf.Get<int32_t>();
        for (uint32_t i = 0U; i < STRIDE; ++i) state.SetValue(i, 0);
        Fence<HardEvent::S_MTE3>();
        const uint32_t produced = runs_ == 0U ? 1U : runs_;
        for (uint32_t group = first_ + produced; group < limit; ++group) {
            DataCopy(descriptorsGm[group * STRIDE], state, STRIDE);
        }
        Fence<HardEvent::MTE3_S>();
        if (adaptive_) {
            DataCopy(state, descriptorsGm[first_ * STRIDE], STRIDE);
            Fence<HardEvent::MTE2_S>();
            state.SetValue(4U, mergedLayout_ ? 1 : 0);
            Fence<HardEvent::S_MTE3>();
            DataCopy(descriptorsGm[first_ * STRIDE], state, STRIDE);
            Fence<HardEvent::MTE3_S>();
        }
    }

private:
    __aicore__ inline void BitCopy(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count)
    {
        Or(dst.ReinterpretCast<uint16_t>(), src.ReinterpretCast<uint16_t>(),
           src.ReinterpretCast<uint16_t>(), count * 2U);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint32_t ReadPair(uint32_t group)
    {
        const uint32_t row = group * 2U;
        const uint32_t rows = query_ - row < 2U ? query_ - row : 2U;
        const uint32_t count = rows * TOPK;
        auto input = workBuf.Get<float>();
        auto raw = input[count].ReinterpretCast<int32_t>();
        auto tmp = input[BASE_IDS * 2U];
        auto live = maskBuf.Get<uint8_t>();
        auto other = live[MASK_BYTES];
        auto bad = live[MASK_BYTES * 2U];
        auto extra = live[MASK_BYTES * 3U];
        DataCopyPad(raw, topkGm[static_cast<uint64_t>(row) * TOPK],
                    {1, count * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        Cast(input, raw, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        CompareScalar(live, input, 0.0F, CMPMODE::GE, count);
        for (uint32_t r = 0U; r < rows; ++r) {
            CompareScalar(other[r * TOPK / 8U], input[r * TOPK],
                          static_cast<float>(static_cast<int32_t>(history_ + row + r)), CMPMODE::LE, TOPK);
        }
        CompareScalar(bad, input, -1.0F, CMPMODE::LT, count);
        CompareScalar(extra, input, static_cast<float>(static_cast<int32_t>(history_ + query_)), CMPMODE::GE, count);
        PipeBarrier<PIPE_V>();
        And(live.ReinterpretCast<uint16_t>(), live.ReinterpretCast<uint16_t>(),
            other.ReinterpretCast<uint16_t>(), count / 16U);
        Or(bad.ReinterpretCast<uint16_t>(), bad.ReinterpretCast<uint16_t>(),
           extra.ReinterpretCast<uint16_t>(), count / 16U);
        Fence<HardEvent::V_S>();
        auto badWords = bad.ReinterpretCast<uint32_t>();
        for (uint32_t r = 0U; r < rows; ++r) {
            for (uint32_t w = 0U; w < TOPK / 32U; ++w) {
                const uint32_t index = r * (TOPK / 32U) + w;
                error_ |= badWords.GetValue(index);
            }
        }
        Fence<HardEvent::S_V>();
        CompareScalar(other, input, static_cast<float>(static_cast<int32_t>(history_)), CMPMODE::LT, count);
        PipeBarrier<PIPE_V>();
        And(live.ReinterpretCast<uint16_t>(), live.ReinterpretCast<uint16_t>(),
            other.ReinterpretCast<uint16_t>(), count / 16U);
        Duplicate(input.ReinterpretCast<int32_t>(), SparsePrefillPlan::SOURCE_KEY_BASE, count);
        PipeBarrier<PIPE_V>();
        Sub(input.ReinterpretCast<int32_t>(), input.ReinterpretCast<int32_t>(), raw, count);
        PipeBarrier<PIPE_V>();
        Select(input, live, input, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        PipeBarrier<PIPE_V>();
        SparsePrefillIndexer::SortAll(input, tmp, count);
        return count;
    }

    // Expand one unique-key bit to two pair-element bits using vector SWAR.
    // int16 -> float -> int32 is exact, including a set high mask bit.
    __aicore__ inline void ExpandPairMask()
    {
        constexpr uint32_t HALF_WORDS = BUFFER_IDS / 16U;
        auto mask = maskBuf.Get<uint8_t>();
        auto floating = mask[MASK_BYTES].ReinterpretCast<float>();
        auto x = mask[MASK_BYTES * 2U].ReinterpretCast<uint32_t>();
        auto tmp = mask[MASK_BYTES * 2U + 2048U].ReinterpretCast<uint32_t>();
        auto constant = mask[MASK_BYTES * 3U].ReinterpretCast<uint32_t>();
        Cast(floating, mask.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, HALF_WORDS);
        PipeBarrier<PIPE_V>();
        Cast(x.ReinterpretCast<int32_t>(), floating, RoundMode::CAST_RINT, HALF_WORDS);
        PipeBarrier<PIPE_V>();
        ShiftLeft(x, x, 16U, HALF_WORDS); PipeBarrier<PIPE_V>();
        ShiftRight(x, x, 16U, HALF_WORDS); PipeBarrier<PIPE_V>();
        const uint32_t shifts[4] = {8U, 4U, 2U, 1U};
        const uint32_t masks[4] = {0x00FF00FFU, 0x0F0F0F0FU, 0x33333333U, 0x55555555U};
        for (uint32_t step = 0U; step < 4U; ++step) {
            ShiftLeft(tmp, x, shifts[step], HALF_WORDS);
            PipeBarrier<PIPE_V>();
            Or(x.ReinterpretCast<uint16_t>(), x.ReinterpretCast<uint16_t>(),
               tmp.ReinterpretCast<uint16_t>(), HALF_WORDS * 2U);
            Duplicate(constant, masks[step], HALF_WORDS);
            PipeBarrier<PIPE_V>();
            And(x.ReinterpretCast<uint16_t>(), x.ReinterpretCast<uint16_t>(),
                constant.ReinterpretCast<uint16_t>(), HALF_WORDS * 2U);
            PipeBarrier<PIPE_V>();
        }
        ShiftLeft(tmp, x, 1U, HALF_WORDS);
        PipeBarrier<PIPE_V>();
        Or(mask.ReinterpretCast<uint16_t>(), x.ReinterpretCast<uint16_t>(),
           tmp.ReinterpretCast<uint16_t>(), HALF_WORDS * 2U);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint32_t Deduplicate(LocalTensor<float> src, LocalTensor<float> dst,
                                         LocalTensor<float> keys, uint32_t count,
                                         bool adaptive = false, bool* direct = nullptr)
    {
        if (count == 0U) return 0U;
        const uint32_t aligned = (count + VECTOR - 1U) / VECTOR * VECTOR;
        Fence<HardEvent::V_S>();
        for (uint32_t i = count * 2U; i < aligned * 2U; ++i) src.SetValue(i, 0.0F);
        Fence<HardEvent::S_V>();
        SparsePrefillIndexer::ExtractScoreBits(keys.ReinterpretCast<uint32_t>(),
                                           src.ReinterpretCast<uint32_t>(), aligned);
        auto offsets = dst[aligned].ReinterpretCast<int32_t>();
        ArithProgression(offsets, 0, 1, aligned);
        PipeBarrier<PIPE_V>();
        Adds(offsets, offsets, -1, aligned); PipeBarrier<PIPE_V>();
        Relu(offsets, offsets, aligned); PipeBarrier<PIPE_V>();
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(int32_t)), aligned);
        PipeBarrier<PIPE_V>();
        Gather(dst, keys, offsets.ReinterpretCast<uint32_t>(), 0U, aligned);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_S>();
        dst.ReinterpretCast<uint32_t>().SetValue(0U, dst.ReinterpretCast<uint32_t>().GetValue(0U) + 1U);
        Fence<HardEvent::S_V>();
        auto mask = maskBuf.Get<uint8_t>();
        // Expanded mask reads a fixed 1024-byte region, including padding.
        Duplicate(mask.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0U), 512U);
        PipeBarrier<PIPE_V>();
        Compare(mask, keys, dst, CMPMODE::NE, aligned);
        CompareScalar(mask[2048U], keys, 0.0F, CMPMODE::GT, aligned);
        PipeBarrier<PIPE_V>();
        And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
            mask[2048U].ReinterpretCast<uint16_t>(), aligned / 16U);
        PipeBarrier<PIPE_V>();
        if (adaptive) {
            // Sparse overlap cannot repay another pair merge. Obtain the
            // unique ID count with MTP's cheaper one-element compaction;
            // publish this run directly when fewer than 1/8 IDs are removed.
            // This decision stays in UB and never reaches the host planner.
            auto ids = dst.ReinterpretCast<uint32_t>();
            SparsePrefillIndexer::ExtractIndex(ids, src.ReinterpretCast<uint32_t>(), aligned);
            PipeBarrier<PIPE_V>();
            uint64_t unique = 0U;
            GatherMask(ids[aligned], ids, mask.ReinterpretCast<uint32_t>(),
                       true, aligned, CompactParams(), unique);
            Fence<HardEvent::V_S>();
            if (unique == 0U) return 0U;
            if (unique * LOW_OVERLAP_DENOMINATOR > static_cast<uint64_t>(count) * LOW_OVERLAP_NUMERATOR) {
                BitCopy(dst, dst[aligned], static_cast<uint32_t>(unique));
                *direct = true;
                return static_cast<uint32_t>(unique);
            }
        }
        ExpandPairMask();
        uint64_t selected = 0U;
        GatherMask(dst.ReinterpretCast<uint32_t>(), src.ReinterpretCast<uint32_t>(),
                   mask.ReinterpretCast<uint32_t>(), true, aligned * 2U, CompactParams(), selected);
        Fence<HardEvent::V_S>();
        return static_cast<uint32_t>(selected) / 2U;
    }

    __aicore__ inline void Publish(uint32_t count, uint32_t offset, uint32_t descriptor)
    {
        auto state = stateBuf.Get<int32_t>();
        for (uint32_t i = 0U; i < STRIDE; ++i) state.SetValue(i, 0);
        state.SetValue(0U, static_cast<int32_t>(count));
        state.SetValue(1U, error_ != 0U ? 1 : 0);
        state.SetValue(3U, static_cast<int32_t>(offset));
        Fence<HardEvent::S_MTE3>();
        DataCopy(descriptorsGm[descriptor * STRIDE], state, STRIDE);
        Fence<HardEvent::MTE3_S>();
    }

    __aicore__ inline void Flush()
    {
        auto ids = workBuf.Get<uint32_t>();
        SparsePrefillIndexer::ExtractIndex(ids, accBuf.Get<uint32_t>(), accCount_);
        Fence<HardEvent::V_MTE3>();
        WriteIds(ids.ReinterpretCast<int32_t>(), accCount_);
        accCount_ = 0U;
    }

    __aicore__ inline void WriteIds(LocalTensor<int32_t> ids, uint32_t count)
    {
        Fence<HardEvent::V_MTE3>();
        lastOffset_ = first_ * BASE_IDS + packed_;
        lastCount_ = count;
        DataCopyPad(listsGm[lastOffset_], ids,
                    {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
        Fence<HardEvent::MTE3_S>();
        Publish(count, lastOffset_, first_ + runs_);
        // Direct cohorts retain their original fixed slots when no local
        // merge is useful. A merged run consumes <= its input cohort slots.
        packed_ += adaptive_ ? (count + BASE_IDS - 1U) / BASE_IDS * BASE_IDS : count;
        ++runs_;
    }

    GlobalTensor<int32_t> topkGm, listsGm, descriptorsGm;
    TBuf<QuePosition::VECCALC> accBuf, workBuf, scratchBuf, maskBuf, stateBuf;
    uint32_t history_, query_;
    uint32_t first_ = 0U, accCount_ = 0U, runs_ = 0U, packed_ = 0U;
    uint32_t lastCount_ = 0U, lastOffset_ = 0U, error_ = 0U;
    bool adaptive_ = false;
    bool mergedLayout_ = false;
};
} // namespace SparsePrefillLocal
#endif
