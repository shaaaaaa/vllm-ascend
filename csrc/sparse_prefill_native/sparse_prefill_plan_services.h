#ifndef SPARSE_PREFILL_PLAN_SERVICES_H
#define SPARSE_PREFILL_PLAN_SERVICES_H
/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Prefill-specific composition of indexer sort/extract and vector-dedup primitives.
// Each logical-ID partition has one writer; repeated TopK IDs never contend
// on a cross-core GM atomic or scatter destination.
#include "kernel_operator.h"
#include "sparse_prefill_indexer_primitives.h"

namespace SparsePrefillPlan {
using namespace AscendC;
constexpr uint32_t TOPK = 2048U;
constexpr uint32_t COHORT_ROWS = 2U;
constexpr uint32_t COHORT_SIZE = TOPK * COHORT_ROWS;
constexpr uint32_t CORES = 32U;
constexpr uint32_t MAX_PARTITION = 32768U;
constexpr uint32_t CHUNK = 4096U;
constexpr uint32_t COUNTER_STRIDE = 8U;
constexpr uint32_t COHORT_META_GROUPS = 512U;
constexpr uint32_t MASK_BYTES = 4096U;
constexpr uint32_t SOURCE_MASK = SparsePrefillIndexer::LONG_SOURCE_MASK;
constexpr int32_t SOURCE_KEY_BASE = static_cast<int32_t>(0x40000000U + SOURCE_MASK);

template <HardEvent E> __aicore__ inline void Fence()
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
    SetFlag<E>(id);
    WaitFlag<E>(id);
}

__aicore__ inline GatherMaskParams CompactParams()
{
    GatherMaskParams params;
    params.repeatTimes = 1;
    params.src0BlockStride = 1;
    params.src0RepeatStride = 8;
    params.src1RepeatStride = 8;
    return params;
}

class PartitionUnion {
public:
    __aicore__ inline void Init(GM_ADDR lists, GM_ADDR cohorts, GM_ADDR membership, GM_ADDR counters,
                                uint32_t /* history */, uint32_t query, uint32_t width,
                                TPipe& pipe, bool adaptiveLayout)
    {
        listsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lists));
        cohortsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cohorts));
        membershipGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(membership));
        countersGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counters));
        cohorts_ = (query + COHORT_ROWS - 1U) / COHORT_ROWS;
        width_ = width; base_ = GetBlockIdx();
        adaptiveLayout_ = adaptiveLayout;
        // Separate launch/TPipe: 128 KiB bitmap + 16 KiB input + 16 KiB
        // compact/float scratch + 8 KiB masks + 16 KiB cohort metadata
        // + 32 B counters = 184 KiB.
        // Cohort SortAll buffers never coexist with this partition layout.
        pipe.InitBuffer(bitmapBuf, MAX_PARTITION * sizeof(int32_t));
        pipe.InitBuffer(inputBuf, CHUNK * sizeof(int32_t));
        pipe.InitBuffer(compactBuf, CHUNK * sizeof(int32_t));
        pipe.InitBuffer(maskBuf, MASK_BYTES * 2U);
        pipe.InitBuffer(counterBuf, COUNTER_STRIDE * sizeof(int32_t));
        pipe.InitBuffer(cohortBuf, COHORT_META_GROUPS * COUNTER_STRIDE * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        bool runs = true;
        if (adaptiveLayout_) {
            runs = false;
            // One immutable format bit per producer core. Unlike per-run
            // offsets, this check is independent of Q and never races with
            // the partition output counters.
            if (cohorts_ >= CORES && cohorts_ % CORES <= 1U) {
                // One strided DMA replaces 32 scalar GM probes. Reuse the
                // existing metadata UB, then retire its scalar readers before
                // loading the full cohort directory in ProcessLayout.
                auto modes = cohortBuf.Get<int32_t>();
                DataCopy(modes, cohortsGm,
                    {static_cast<uint16_t>(CORES), 1U, static_cast<uint16_t>(cohorts_ / CORES - 1U), 0U});
                Fence<HardEvent::MTE2_S>();
                for (uint32_t core = 0U; core < CORES; ++core) {
                    if (modes.GetValue(core * COUNTER_STRIDE + 4U) != 0) { runs = true; break; }
                }
                Fence<HardEvent::S_MTE2>();
            } else {
                for (uint32_t core = 0U; core < CORES; ++core) {
                    const uint32_t first = cohorts_ * core / CORES;
                    const uint32_t limit = cohorts_ * (core + 1U) / CORES;
                    if (first != limit && cohortsGm.GetValue(first * COUNTER_STRIDE + 4U) != 0) {
                        runs = true;
                        break;
                    }
                }
            }
        }
        if (runs) ProcessLayout<true>();
        else ProcessLayout<false>();
    }

private:
    template <bool UseRuns> __aicore__ inline void ProcessLayout()
    {
        LocalTensor<float> bitmap = bitmapBuf.Get<float>();
        LocalTensor<int32_t> input = inputBuf.Get<int32_t>();
        LocalTensor<int32_t> compact = compactBuf.Get<int32_t>();
        LocalTensor<float> floats = compact.ReinterpretCast<float>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<uint8_t> other = mask[MASK_BYTES];
        Duplicate(bitmap, 0.0F, width_);
        Fence<HardEvent::V_S>();
        uint32_t error = 0U;
        LocalTensor<int32_t> cohort = cohortBuf.Get<int32_t>();
        // Tile the cohort directory in 16 KiB UB, independently of Q.
        for (uint32_t base = 0U; base < cohorts_; base += COHORT_META_GROUPS) {
            const uint32_t records = cohorts_ - base < COHORT_META_GROUPS ? cohorts_ - base : COHORT_META_GROUPS;
            if (base != 0U) Fence<HardEvent::S_MTE2>();
            DataCopy(cohort, cohortsGm[base * COUNTER_STRIDE], records * COUNTER_STRIDE);
            Fence<HardEvent::MTE2_S>();
            for (uint32_t local = 0U; local < records; ++local) {
                const uint32_t group = base + local;
                const uint32_t listCount = static_cast<uint32_t>(cohort.GetValue(local * COUNTER_STRIDE));
                error |= static_cast<uint32_t>(cohort.GetValue(local * COUNTER_STRIDE + 1U));
                if (listCount == 0U) continue;
                const uint64_t sourceBegin = UseRuns ?
                    static_cast<uint32_t>(cohort.GetValue(local * COUNTER_STRIDE + 3U)) :
                    static_cast<uint64_t>(group) * COHORT_SIZE;
                for (uint32_t chunk = 0U; chunk < listCount; chunk += CHUNK) {
                    const uint32_t count = UseRuns ? (listCount - chunk < CHUNK ? listCount - chunk : CHUNK) : listCount;
                    DataCopyPad(input, listsGm[sourceBegin + chunk],
                                {1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
                    Fence<HardEvent::MTE2_S>();
                    const uint32_t aligned = (count + 63U) / 64U * 64U;
                    for (uint32_t i = count; i < aligned; ++i) input.SetValue(i, -1);
                    Fence<HardEvent::S_V>();
                    Cast(floats, input, RoundMode::CAST_NONE, aligned);
                    PipeBarrier<PIPE_V>();
                    CompareScalar(mask, floats, 0.0F, CMPMODE::GE, aligned);
                    ShiftRight(compact.ReinterpretCast<uint32_t>(), input.ReinterpretCast<uint32_t>(), 5U, aligned);
                    PipeBarrier<PIPE_V>();
                    ShiftLeft(compact.ReinterpretCast<uint32_t>(), compact.ReinterpretCast<uint32_t>(), 5U, aligned);
                    PipeBarrier<PIPE_V>();
                    Sub(compact, input, compact, aligned);
                    PipeBarrier<PIPE_V>();
                    Cast(floats, compact, RoundMode::CAST_NONE, aligned);
                    PipeBarrier<PIPE_V>();
                    CompareScalar(other, floats, static_cast<float>(static_cast<int32_t>(base_)), CMPMODE::EQ, aligned);
                    PipeBarrier<PIPE_V>();
                    And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
                        other.ReinterpretCast<uint16_t>(), aligned / 16U);
                    PipeBarrier<PIPE_V>();
                    uint64_t countLocal = 0U;
                    GatherMask(compact.ReinterpretCast<uint32_t>(), input.ReinterpretCast<uint32_t>(),
                               mask.ReinterpretCast<uint32_t>(), true, aligned, CompactParams(), countLocal);
                    Fence<HardEvent::V_S>();
                    for (uint32_t i = 0U; i < static_cast<uint32_t>(countLocal); ++i) {
                        bitmap.SetValue(static_cast<uint32_t>(compact.GetValue(i)) / CORES, 1.0F);
                    }
                    if constexpr (!UseRuns) break;
                }
            }
        }
        LocalTensor<int32_t> state = counterBuf.Get<int32_t>();
        Fence<HardEvent::S_MTE3>();
        DataCopy(membershipGm[static_cast<uint64_t>(GetBlockIdx()) * width_],
                 bitmap.ReinterpretCast<int32_t>(), width_);
        Fence<HardEvent::MTE3_S>();
        for (uint32_t i = 0U; i < COUNTER_STRIDE; ++i) state.SetValue(i, 0);
        state.SetValue(2, static_cast<int32_t>(error));
        Fence<HardEvent::S_MTE3>();
        DataCopy(countersGm[GetBlockIdx() * COUNTER_STRIDE], state, COUNTER_STRIDE);
        Fence<HardEvent::MTE3_S>();
    }

private:
    GlobalTensor<int32_t> listsGm, cohortsGm, membershipGm, countersGm;
    TBuf<QuePosition::VECCALC> bitmapBuf, inputBuf, compactBuf, maskBuf, counterBuf, cohortBuf;
    uint32_t cohorts_, width_, base_;
    bool adaptiveLayout_;
};

}  // namespace SparsePrefillPlan

#endif
