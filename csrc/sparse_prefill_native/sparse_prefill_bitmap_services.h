#ifndef SPARSE_PREFILL_BITMAP_SERVICES_H
#define SPARSE_PREFILL_BITMAP_SERVICES_H
/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Sorted historical union: bounded bitmap membership and count publication.
#include "kernel_operator.h"

namespace SparsePrefillBitmap {
using namespace AscendC;
constexpr uint32_t CORES = 32U;
constexpr uint32_t MAX_HISTORY = 524288U;
constexpr uint32_t WORDS = MAX_HISTORY / 32U;
constexpr uint32_t WORDS_PER_CORE = WORDS / CORES;

template <HardEvent E> __aicore__ inline void Fence()
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
    SetFlag<E>(id);
    WaitFlag<E>(id);
}

// Transpose residue membership into a sorted-ID bitmap. Finish reads the
// bitmap and per-group counts; no per-word remap ranks are needed.
class BuildMembershipWords {
public:
    __aicore__ inline void Init(GM_ADDR membership, GM_ADDR residueCounts, GM_ADDR groupCounts,
                                GM_ADDR bitmap, uint32_t width, TPipe& pipe)
    {
        membershipGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(membership));
        residueGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(residueCounts));
        groupsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupCounts));
        bitmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(bitmap));
        width_ = width;
        words_ = ((width + CORES - 1U) / CORES + 63U) / 64U * 64U;
        // Gather executes complete 64-lane groups; sanitize every offset lane.
        pipe.InitBuffer(matrixBuf, CORES * words_ * sizeof(float));
        for (uint32_t i = 0U; i < 5U; ++i) pipe.InitBuffer(rows[i], WORDS_PER_CORE * sizeof(int32_t));
        pipe.InitBuffer(bitsBuf, words_ * sizeof(uint32_t));
        pipe.InitBuffer(countBuf, 8U * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t firstWord = GetBlockIdx() * words_;
        const uint32_t cols = firstWord >= width_ ? 0U :
            (width_ - firstWord < words_ ? width_ - firstWord : words_);
        auto matrix = matrixBuf.Get<float>();
        if (cols < words_) {
            Duplicate(matrix, 0.0F, CORES * words_);
            Fence<HardEvent::V_MTE2>();
        }
        if (cols != 0U) {
            for (uint32_t residue = 0U; residue < CORES; ++residue) {
                DataCopyPad(matrix[residue * words_],
                            membershipGm[static_cast<uint64_t>(residue) * width_ + firstWord],
                            {1, static_cast<uint32_t>(cols * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            }
            Fence<HardEvent::MTE2_V>();
        }
        auto offsets = rows[0].Get<int32_t>();
        auto values = rows[1].Get<int32_t>();
        auto tmp = rows[2].Get<int32_t>();
        auto constant = rows[3].Get<int32_t>();
        auto high = rows[4].Get<int32_t>();
        auto bits = bitsBuf.Get<uint32_t>();
        // i maps to residue i%32 and word i/32. Build once, then advance
        // sixteen words per batch; avoid scalar rebuilding in the hot loop.
        ArithProgression(offsets, 0, 1, WORDS_PER_CORE);
        PipeBarrier<PIPE_V>();
        ShiftRight(tmp.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), 5U, WORDS_PER_CORE);
        PipeBarrier<PIPE_V>();
        ShiftLeft(constant.ReinterpretCast<uint32_t>(), tmp.ReinterpretCast<uint32_t>(), 5U, WORDS_PER_CORE);
        PipeBarrier<PIPE_V>();
        Sub(offsets, offsets, constant, WORDS_PER_CORE);
        PipeBarrier<PIPE_V>();
        Muls(offsets, offsets, static_cast<int32_t>(words_), WORDS_PER_CORE);
        PipeBarrier<PIPE_V>();
        Add(offsets, offsets, tmp, WORDS_PER_CORE);
        PipeBarrier<PIPE_V>();
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(float)), WORDS_PER_CORE);
        PipeBarrier<PIPE_V>();
        for (uint32_t batch = 0U; batch < words_ / 16U; ++batch) {
            Gather(values.ReinterpretCast<float>(), matrix, offsets.ReinterpretCast<uint32_t>(), 0U, WORDS_PER_CORE);
            PipeBarrier<PIPE_V>();
            CompareScalar(bits.ReinterpretCast<uint8_t>()[batch * 64U], values.ReinterpretCast<float>(),
                          0.0F, CMPMODE::GT, WORDS_PER_CORE);
            PipeBarrier<PIPE_V>();
            if (batch + 1U < words_ / 16U) {
                Adds(offsets, offsets, 16 * static_cast<int32_t>(sizeof(float)), WORDS_PER_CORE);
                PipeBarrier<PIPE_V>();
            }
        }
        ShiftRight(high.ReinterpretCast<uint32_t>(), bits, 31U, words_);
        Duplicate(constant, INT32_MAX, words_);
        PipeBarrier<PIPE_V>();
        And32(values, bits.ReinterpretCast<int32_t>(), constant);
        ShiftRight(tmp.ReinterpretCast<uint32_t>(), values.ReinterpretCast<uint32_t>(), 1U, words_);
        Duplicate(constant, static_cast<int32_t>(0x55555555U), words_);
        PipeBarrier<PIPE_V>();
        And32(tmp, tmp, constant);
        Sub(values, values, tmp, words_);
        Duplicate(constant, static_cast<int32_t>(0x33333333U), words_);
        PipeBarrier<PIPE_V>();
        And32(tmp, values, constant);
        ShiftRight(values.ReinterpretCast<uint32_t>(), values.ReinterpretCast<uint32_t>(), 2U, words_);
        PipeBarrier<PIPE_V>();
        And32(values, values, constant);
        Add(values, values, tmp, words_);
        PipeBarrier<PIPE_V>();
        ShiftRight(tmp.ReinterpretCast<uint32_t>(), values.ReinterpretCast<uint32_t>(), 4U, words_);
        PipeBarrier<PIPE_V>();
        Add(values, values, tmp, words_);
        Duplicate(constant, static_cast<int32_t>(0x0F0F0F0FU), words_);
        PipeBarrier<PIPE_V>();
        And32(values, values, constant);
        ShiftRight(tmp.ReinterpretCast<uint32_t>(), values.ReinterpretCast<uint32_t>(), 8U, words_);
        PipeBarrier<PIPE_V>();
        Add(values, values, tmp, words_);
        PipeBarrier<PIPE_V>();
        ShiftRight(tmp.ReinterpretCast<uint32_t>(), values.ReinterpretCast<uint32_t>(), 16U, words_);
        PipeBarrier<PIPE_V>();
        Add(values, values, tmp, words_);
        Duplicate(constant, 63, words_);
        PipeBarrier<PIPE_V>();
        And32(values, values, constant);
        Add(values, values, high, words_);
        PipeBarrier<PIPE_V>();
        // Each popcount is <= 32 and the total is <= 16384. Numeric float
        // conversion and every partial sum are exact within that bound.
        Cast(high.ReinterpretCast<float>(), values, RoundMode::CAST_NONE, words_);
        PipeBarrier<PIPE_V>();
        ReduceSum(constant.ReinterpretCast<float>(), high.ReinterpretCast<float>(),
                  tmp.ReinterpretCast<float>(), words_);
        Fence<HardEvent::V_S>();
        const uint32_t count = static_cast<uint32_t>(
            static_cast<int32_t>(constant.ReinterpretCast<float>().GetValue(0U)));
        auto stats = countBuf.Get<int32_t>();
        for (uint32_t i = 0U; i < 8U; ++i) stats.SetValue(i, 0);
        stats.SetValue(0U, static_cast<int32_t>(count));
        stats.SetValue(1U, residueGm.GetValue(GetBlockIdx() * 8U + 2U));
        stats.SetValue(2U, static_cast<int32_t>(words_));
        Fence<HardEvent::S_MTE3>();
        DataCopy(bitmapGm[firstWord], bits, words_);
        DataCopy(groupsGm[GetBlockIdx() * 8U], stats, 8U);
        Fence<HardEvent::MTE3_S>();
    }

private:
    __aicore__ inline void And32(LocalTensor<int32_t> dst, LocalTensor<int32_t> left, LocalTensor<int32_t> right)
    {
        And(dst.ReinterpretCast<uint16_t>(), left.ReinterpretCast<uint16_t>(),
            right.ReinterpretCast<uint16_t>(), words_ * 2U);
        PipeBarrier<PIPE_V>();
    }
    GlobalTensor<float> membershipGm;
    GlobalTensor<int32_t> residueGm, groupsGm;
    GlobalTensor<uint32_t> bitmapGm;
    TBuf<QuePosition::VECCALC> matrixBuf, rows[5], bitsBuf, countBuf;
    uint32_t width_, words_;
};

class FinishSortedBitmap {
public:
    __aicore__ inline void Init(GM_ADDR groups, GM_ADDR bitmap, GM_ADDR sources, GM_ADDR lengths,
                                uint32_t query, uint32_t capacity, TPipe& pipe)
    {
        groupsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groups));
        bitmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(bitmap));
        sourcesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sources));
        lengthsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lengths));
        query_ = query; capacity_ = capacity;
        words_ = static_cast<uint32_t>(groupsGm.GetValue(2U));
        ASSERT_MSG(words_ >= 64U && words_ <= WORDS_PER_CORE && words_ % 64U == 0U, "Invalid compact bitmap extent");
        pipe.InitBuffer(bitsBuf, words_ * sizeof(uint32_t));
        pipe.InitBuffer(inputBuf, 8192U * sizeof(int32_t));
        pipe.InitBuffer(outputBuf, 8192U * sizeof(int32_t));
        pipe.InitBuffer(lengthBuf, 8U * sizeof(int32_t));
        pipe.InitBuffer(allGroupBuf, CORES * 8U * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t before = 0U, total = 0U, error = 0U;
        auto allGroups = allGroupBuf.Get<int32_t>();
        DataCopy(allGroups, groupsGm, CORES * 8U);
        Fence<HardEvent::MTE2_S>();
        for (uint32_t group = 0U; group < CORES; ++group) {
            const uint32_t count = static_cast<uint32_t>(allGroups.GetValue(group * 8U));
            if (group < GetBlockIdx()) before += count;
            total += count;
            error |= static_cast<uint32_t>(allGroups.GetValue(group * 8U + 1U));
        }
        if (total > capacity_ || error != 0U) {
            // Publish failure without writing a truncated ID list. The device
            // copy consumer rejects this status; counts do not return to CPU.
            if (GetBlockIdx() == 0U) {
                auto lengths = lengthBuf.Get<int32_t>();
                for (uint32_t i = 0U; i < 8U; ++i) lengths.SetValue(i, 0);
                lengths.SetValue(0U, static_cast<int32_t>(total));
                lengths.SetValue(1U, static_cast<int32_t>(query_));
                lengths.SetValue(2U, static_cast<int32_t>(total + query_));
                lengths.SetValue(3U, static_cast<int32_t>(error));
                lengths.SetValue(4U, total > capacity_ ? 1 : 0);
                Fence<HardEvent::S_MTE3>();
                DataCopy(lengthsGm, lengths, 8U);
                Fence<HardEvent::MTE3_S>();
            }
            return;
        }
        auto bits = bitsBuf.Get<uint32_t>();
        const uint32_t firstWord = GetBlockIdx() * words_;
        DataCopy(bits, bitmapGm[firstWord], words_);
        Fence<HardEvent::MTE2_V>();
        if (GetBlockIdx() == 0U) {
            auto lengths = lengthBuf.Get<int32_t>();
            for (uint32_t i = 0U; i < 8U; ++i) lengths.SetValue(i, 0);
            lengths.SetValue(0U, static_cast<int32_t>(total));
            lengths.SetValue(1U, static_cast<int32_t>(query_));
            lengths.SetValue(2U, static_cast<int32_t>(total + query_));
            Fence<HardEvent::S_MTE3>();
            DataCopy(lengthsGm, lengths, 8U);
            Fence<HardEvent::MTE3_S>();
        }
        auto input = inputBuf.Get<int32_t>();
        auto output = outputBuf.Get<int32_t>();
        Fence<HardEvent::S_V>();
        uint32_t packed = 0U;
        for (uint32_t first = 0U; first < words_ * 32U; first += 8192U) {
            const uint32_t rows = words_ * 32U - first < 8192U ? words_ * 32U - first : 8192U;
            ArithProgression(input, static_cast<int32_t>(firstWord * 32U + first), 1, rows);
            PipeBarrier<PIPE_V>();
            GatherMaskParams compact;
            compact.repeatTimes = 1;
            compact.src0BlockStride = 1;
            compact.src0RepeatStride = 8;
            compact.src1RepeatStride = 8;
            uint64_t count = 0U;
            GatherMask(output.ReinterpretCast<uint32_t>(), input.ReinterpretCast<uint32_t>(),
                       bits[first / 32U], true, rows, compact, count);
            Fence<HardEvent::V_S>();
            const uint32_t selected = static_cast<uint32_t>(count);
            ASSERT_MSG(before + packed + selected <= capacity_, "Sorted union write exceeds capacity");
            if (before + packed + selected > capacity_) return;
            if (selected != 0U) {
                Fence<HardEvent::S_MTE3>();
                DataCopyPad(sourcesGm[before + packed], output,
                            {1, static_cast<uint16_t>(selected * sizeof(int32_t)), 0, 0});
                Fence<HardEvent::MTE3_S>();
            }
            packed += selected;
            Fence<HardEvent::S_V>();
        }
    }

private:
    GlobalTensor<int32_t> groupsGm, sourcesGm, lengthsGm;
    GlobalTensor<uint32_t> bitmapGm;
    TBuf<QuePosition::VECCALC> bitsBuf, inputBuf, outputBuf, lengthBuf;
    uint32_t query_, capacity_, words_;
    TBuf<QuePosition::VECCALC> allGroupBuf;
};
} // namespace SparsePrefillBitmap


#endif
