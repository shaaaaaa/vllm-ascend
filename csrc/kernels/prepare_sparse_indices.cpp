/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "kernel_operator.h"

namespace {

template <AscendC::HardEvent event>
__aicore__ inline void SyncPipeline()
{
    const event_t eventId = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

class DSAPrepareSparseIndicesKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* requestBlockTable,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* selectedCounts,
        __gm__ int64_t* targetSlots,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t requestCount,
        uint32_t blockTableWidth,
        uint32_t scratchCapacity,
        uint32_t selectedCountStride,
        uint32_t bitmapWords,
        uint32_t blockSize,
        uint32_t needPacked,
        uint32_t clearInvalidRows)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        requestCount_ = requestCount;
        blockTableWidth_ = blockTableWidth;
        scratchCapacity_ = scratchCapacity;
        selectedCountStride_ = selectedCountStride;
        bitmapWords_ = bitmapWords;
        bufferWords_ = ((bitmapWords + 7) / 8) * 8;
        blockSize_ = blockSize;
        needPacked_ = needPacked != 0;
        clearInvalidRows_ = clearInvalidRows != 0;
        topkIndices_.SetGlobalBuffer(topkIndices);
        splitBoundary_.SetGlobalBuffer(splitBoundary, rowCount);
        rowReqIndices_.SetGlobalBuffer(rowReqIndices, rowCount);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable,
            static_cast<uint64_t>(requestCount) * blockTableWidth);
        selectedPacked_.SetGlobalBuffer(
            selectedPacked,
            static_cast<uint64_t>(requestCount) * scratchCapacity);
        selectedCounts_.SetGlobalBuffer(
            selectedCounts,
            static_cast<uint64_t>(requestCount) * selectedCountStride);
        targetSlots_.SetGlobalBuffer(
            targetSlots,
            static_cast<uint64_t>(requestCount) * scratchCapacity);
        pipe_.InitBuffer(bitmapBuffer_, bufferWords_ * sizeof(int32_t));
        pipe_.InitBuffer(prefixBuffer_, bufferWords_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t req = AscendC::GetBlockIdx();
        if (req >= requestCount_) {
            return;
        }
        bool hasPositiveBoundary = false;
        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) == static_cast<int32_t>(req)
                && splitBoundary_.GetValue(row) > 0) {
                hasPositiveBoundary = true;
                break;
            }
        }
        if (!hasPositiveBoundary) {
            selectedCounts_.SetValue(
                static_cast<uint64_t>(req) * selectedCountStride_, 0);
            if (req == 0 && clearInvalidRows_) {
                ClearInvalidRows();
            }
            return;
        }
        AscendC::LocalTensor<int32_t> bitmap =
            bitmapBuffer_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> prefix =
            prefixBuffer_.Get<int32_t>();
        AscendC::Duplicate(bitmap, static_cast<int32_t>(0), bufferWords_);
        // Duplicate runs on the vector pipeline, while the bitmap below is
        // updated through scalar GetValue/SetValue accesses.  Wait for the
        // clear to finish before the scalar pipeline starts modifying it;
        // otherwise the delayed clear can erase request-union bits.
        AscendC::PipeBarrier<PIPE_V>();
        SyncPipeline<AscendC::HardEvent::V_S>();
        const uint64_t packedOffset =
            static_cast<uint64_t>(req) * scratchCapacity_;

        // One AIV owns one request, so setting bitmap words does not need
        // atomics even when several MTP rows select the same position.
        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) != static_cast<int32_t>(req)) {
                continue;
            }
            const int32_t boundary = splitBoundary_.GetValue(row);
            const uint64_t rowOffset = static_cast<uint64_t>(row) * rowWidth_;
            for (uint32_t col = 0; col < rowWidth_; ++col) {
                const uint64_t indexOffset = rowOffset + col;
                const int32_t token = topkIndices_.GetValue(indexOffset);
                if (token < 0 || token >= boundary) {
                    continue;
                }
                const uint32_t word = static_cast<uint32_t>(token) >> 5;
                const uint32_t bit = static_cast<uint32_t>(token) & 31;
                const uint32_t value =
                    static_cast<uint32_t>(bitmap.GetValue(word));
                bitmap.SetValue(
                    word, static_cast<int32_t>(value | (1U << bit)));
            }
        }

        // Prefix popcount gives every selected position a deterministic rank
        // in ascending token-position order.
        uint32_t uniqueCount = 0;
        for (uint32_t word = 0; word < bitmapWords_; ++word) {
            prefix.SetValue(word, static_cast<int32_t>(uniqueCount));
            uniqueCount += Popcount32(
                static_cast<uint32_t>(bitmap.GetValue(word)));
        }

        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) != static_cast<int32_t>(req)) {
                continue;
            }
            const int32_t boundary = splitBoundary_.GetValue(row);
            const uint64_t rowOffset = static_cast<uint64_t>(row) * rowWidth_;
            for (uint32_t col = 0; col < rowWidth_; ++col) {
                const uint64_t indexOffset = rowOffset + col;
                const int32_t token = topkIndices_.GetValue(indexOffset);
                if (token < 0 || token >= boundary) {
                    continue;
                }
                const uint32_t word = static_cast<uint32_t>(token) >> 5;
                const uint32_t bit = static_cast<uint32_t>(token) & 31;
                const uint32_t bitmapValue =
                    static_cast<uint32_t>(bitmap.GetValue(word));
                const uint32_t lowerMask = bit == 0 ? 0 : ((1U << bit) - 1);
                const uint32_t scratchSlot =
                    static_cast<uint32_t>(prefix.GetValue(word))
                    + Popcount32(bitmapValue & lowerMask);
                topkIndices_.SetValue(
                    indexOffset, static_cast<int32_t>(scratchSlot));
                if (needPacked_) {
                    selectedPacked_.SetValue(
                        packedOffset + scratchSlot, token);
                    const uint32_t logicalBlock = scratchSlot / blockSize_;
                    const uint32_t blockOffset = scratchSlot % blockSize_;
                    const int32_t physicalBlock =
                        requestBlockTable_.GetValue(
                            static_cast<uint64_t>(req) * blockTableWidth_
                            + logicalBlock);
                    if (physicalBlock <= 0) {
                        AscendC::Trap();
                        return;
                    }
                    targetSlots_.SetValue(
                        packedOffset + scratchSlot,
                        static_cast<int64_t>(physicalBlock) * blockSize_
                            + blockOffset);
                }
            }
        }
        selectedCounts_.SetValue(
            static_cast<uint64_t>(req) * selectedCountStride_,
            needPacked_ ? static_cast<int32_t>(uniqueCount) : 0);

        if (req == 0 && clearInvalidRows_) {
            ClearInvalidRows();
        }
    }

private:
    __aicore__ inline void ClearInvalidRows()
    {
        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) >= 0) {
                continue;
            }
            const uint64_t rowOffset = static_cast<uint64_t>(row) * rowWidth_;
            for (uint32_t col = 0; col < rowWidth_; ++col) {
                topkIndices_.SetValue(rowOffset + col, 0);
            }
        }
    }

    __aicore__ inline uint32_t Popcount32(uint32_t value) const
    {
        value = value - ((value >> 1) & 0x55555555U);
        value = (value & 0x33333333U) + ((value >> 2) & 0x33333333U);
        value = (value + (value >> 4)) & 0x0F0F0F0FU;
        value = value + (value >> 8);
        value = value + (value >> 16);
        return value & 0x3FU;
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> selectedCounts_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bitmapBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prefixBuffer_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestCount_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t bitmapWords_ = 0;
    uint32_t bufferWords_ = 0;
    uint32_t blockSize_ = 0;
    bool needPacked_ = false;
    bool clearInvalidRows_ = false;
};

class DSAPrepareSparseIndicesReuseKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* requestBlockTable,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* selectedCounts,
        __gm__ int64_t* targetSlots,
        __gm__ int32_t* requestStateIndices,
        __gm__ int64_t* requestGenerations,
        __gm__ int32_t* residentTokenIds,
        __gm__ int64_t* residentGenerations,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t requestCount,
        uint32_t blockTableWidth,
        uint32_t scratchCapacity,
        uint32_t selectedCountStride,
        uint32_t bitmapWords,
        uint32_t blockSize,
        uint32_t stateRowCount,
        uint32_t residentTokenStride,
        uint32_t residentGenerationStride,
        uint32_t clearInvalidRows)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        requestCount_ = requestCount;
        blockTableWidth_ = blockTableWidth;
        tokenCapacity_ =
            static_cast<uint64_t>(blockTableWidth) * blockSize;
        scratchCapacity_ = scratchCapacity;
        selectedCountStride_ = selectedCountStride;
        bitmapWords_ = bitmapWords;
        bitmapBufferWords_ = ((bitmapWords + 7) / 8) * 8;
        rankBufferWords_ = ((scratchCapacity + 7) / 8) * 8;
        blockSize_ = blockSize;
        stateRowCount_ = stateRowCount;
        residentTokenStride_ = residentTokenStride;
        residentGenerationStride_ = residentGenerationStride;
        clearInvalidRows_ = clearInvalidRows != 0;

        topkIndices_.SetGlobalBuffer(topkIndices);
        splitBoundary_.SetGlobalBuffer(splitBoundary, rowCount);
        rowReqIndices_.SetGlobalBuffer(rowReqIndices, rowCount);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable,
            static_cast<uint64_t>(requestCount) * blockTableWidth);
        selectedPacked_.SetGlobalBuffer(
            selectedPacked,
            static_cast<uint64_t>(requestCount) * scratchCapacity);
        selectedCounts_.SetGlobalBuffer(
            selectedCounts,
            static_cast<uint64_t>(requestCount) * selectedCountStride);
        targetSlots_.SetGlobalBuffer(
            targetSlots,
            static_cast<uint64_t>(requestCount) * scratchCapacity);
        requestStateIndices_.SetGlobalBuffer(
            requestStateIndices, requestCount);
        requestGenerations_.SetGlobalBuffer(
            requestGenerations, requestCount);
        residentTokenIds_.SetGlobalBuffer(
            residentTokenIds,
            static_cast<uint64_t>(stateRowCount) * residentTokenStride);
        residentGenerations_.SetGlobalBuffer(
            residentGenerations,
            static_cast<uint64_t>(stateRowCount) * residentGenerationStride);

        pipe_.InitBuffer(
            bitmapBuffer_, bitmapBufferWords_ * sizeof(int32_t));
        pipe_.InitBuffer(
            prefixBuffer_, bitmapBufferWords_ * sizeof(int32_t));
        pipe_.InitBuffer(
            rankToSlotBuffer_, rankBufferWords_ * sizeof(int32_t));
        pipe_.InitBuffer(
            residentBuffer_, scratchCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(
            residentGenerationBuffer_,
            residentGenerationBufferWords_ * sizeof(int64_t));
    }

    __aicore__ inline void Process()
    {
        // Exactly one AIV owns every compact request. Callers must map compact
        // requests to distinct stable state rows.
        const uint32_t req = AscendC::GetBlockIdx();
        if (req >= requestCount_) {
            return;
        }
        const int64_t requestGeneration = requestGenerations_.GetValue(req);
        if (requestGeneration <= 0) {
            // Graph padding has no request lifetime and must never claim or
            // invalidate a stable resident-state row.
            SetCount(req, 0);
            if (req == 0 && clearInvalidRows_) {
                ClearInvalidRows();
            }
            return;
        }
        bool hasPositiveBoundary = false;
        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) == static_cast<int32_t>(req)
                && splitBoundary_.GetValue(row) > 0) {
                hasPositiveBoundary = true;
                break;
            }
        }
        if (!hasPositiveBoundary) {
            // No row of this request reads LMCache-side scratch this step.
            // Leave persistent state entirely untouched and validate/reset its
            // generation lazily on the first positive-boundary step.
            SetCount(req, 0);
            if (req == 0 && clearInvalidRows_) {
                ClearInvalidRows();
            }
            return;
        }
        const int32_t stateValue = requestStateIndices_.GetValue(req);
        if (stateValue < 0
            || static_cast<uint32_t>(stateValue) >= stateRowCount_) {
            // This indicates a broken host-side lifetime binding. Do not
            // publish an error sentinel through selectedCounts_: downstream
            // packing would interpret a negative count as an empty payload
            // and continue with stale scratch data.
            AscendC::Trap();
            return;
        }
        const uint32_t stateRow = static_cast<uint32_t>(stateValue);
        const uint64_t residentOffset =
            static_cast<uint64_t>(stateRow) * residentTokenStride_;
        const uint64_t generationOffset =
            static_cast<uint64_t>(stateRow) * residentGenerationStride_;
        AscendC::LocalTensor<int32_t> resident =
            residentBuffer_.Get<int32_t>();
        AscendC::LocalTensor<int64_t> residentGeneration =
            residentGenerationBuffer_.Get<int64_t>();
        // Persistent state may be owned by a different physical AIV on the
        // next compact-batch launch. GlobalTensor scalar SetValue/GetValue
        // accesses the per-core DCache, so using scalar GM access here would
        // require explicit cache maintenance across launches. DMA the state
        // through UB instead; MTE2/MTE3 accesses GM directly and avoids that
        // cross-core DCache dependency.
        AscendC::DataCopy(
            resident, residentTokenIds_[residentOffset], scratchCapacity_);
        AscendC::DataCopy(
            residentGeneration,
            residentGenerations_[generationOffset],
            residentGenerationBufferWords_);
        SyncPipeline<AscendC::HardEvent::MTE2_S>();

        bool stateDirty = false;
        if (residentGeneration.GetValue(0) != requestGeneration) {
            // Clearing the entire row is necessary: after the generation is
            // published, an untouched stale slot must never become a hit.
            // The scalar generation comparison controls whether this vector
            // write is issued, so explicitly order S -> V.
            SyncPipeline<AscendC::HardEvent::S_V>();
            AscendC::Duplicate(
                resident, static_cast<int32_t>(-1), scratchCapacity_);
            AscendC::PipeBarrier<PIPE_V>();
            SyncPipeline<AscendC::HardEvent::V_S>();
            residentGeneration.SetValue(0, requestGeneration);
            stateDirty = true;
        }

        AscendC::LocalTensor<int32_t> bitmap =
            bitmapBuffer_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> prefix =
            prefixBuffer_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> rankToSlot =
            rankToSlotBuffer_.Get<int32_t>();
        AscendC::Duplicate(
            bitmap, static_cast<int32_t>(0), bitmapBufferWords_);
        AscendC::Duplicate(
            rankToSlot, static_cast<int32_t>(-1), rankBufferWords_);
        // The union and rank map below use scalar accesses to local tensors.
        // Fence both vector clears before the scalar pipeline reads/writes.
        AscendC::PipeBarrier<PIPE_V>();
        SyncPipeline<AscendC::HardEvent::V_S>();

        // Build the LMCache-side union across all MTP rows of this request.
        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) != static_cast<int32_t>(req)) {
                continue;
            }
            const int32_t boundary = splitBoundary_.GetValue(row);
            const uint64_t rowOffset = static_cast<uint64_t>(row) * rowWidth_;
            for (uint32_t col = 0; col < rowWidth_; ++col) {
                const int32_t token =
                    topkIndices_.GetValue(rowOffset + col);
                if (token < 0 || token >= boundary) {
                    continue;
                }
                if (static_cast<uint64_t>(token) >= tokenCapacity_) {
                    // A selected LMCache token must be addressable by the
                    // request block table. Trap instead of turning this into
                    // a zero-length retrieve.
                    AscendC::Trap();
                    return;
                }
                const uint32_t word = static_cast<uint32_t>(token) >> 5;
                if (word >= bitmapWords_) {
                    AscendC::Trap();
                    return;
                }
                const uint32_t bit = static_cast<uint32_t>(token) & 31;
                const uint32_t value =
                    static_cast<uint32_t>(bitmap.GetValue(word));
                bitmap.SetValue(
                    word, static_cast<int32_t>(value | (1U << bit)));
            }
        }

        uint32_t uniqueCount = 0;
        for (uint32_t word = 0; word < bitmapWords_; ++word) {
            prefix.SetValue(word, static_cast<int32_t>(uniqueCount));
            uniqueCount += Popcount32(
                static_cast<uint32_t>(bitmap.GetValue(word)));
        }
        if (uniqueCount > scratchCapacity_) {
            AscendC::Trap();
            return;
        }

        // Preserve every resident hit at its current physical scratch slot.
        // rankToSlot is indexed by the union's ascending token rank.
        for (uint32_t slot = 0; slot < scratchCapacity_; ++slot) {
            const int32_t token = resident.GetValue(slot);
            if (!IsDesiredToken(token, bitmap)) {
                continue;
            }
            const uint32_t rank = TokenRank(token, bitmap, prefix);
            rankToSlot.SetValue(rank, static_cast<int32_t>(slot));
        }

        const uint64_t packedOffset =
            static_cast<uint64_t>(req) * scratchCapacity_;
        uint32_t missCount = 0;
        uint32_t nextFreeSlot = 0;
        // Visit the request's rows again and assign each first-seen miss.
        // LMCache preserves selected-token/target-slot pairing and does not
        // require token order. This avoids scanning 32 bits for every nonempty
        // bitmap word when max_model_len is much larger than top-k.
        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) != static_cast<int32_t>(req)) {
                continue;
            }
            const int32_t boundary = splitBoundary_.GetValue(row);
            const uint64_t rowOffset = static_cast<uint64_t>(row) * rowWidth_;
            for (uint32_t col = 0; col < rowWidth_; ++col) {
                const int32_t token =
                    topkIndices_.GetValue(rowOffset + col);
                if (token < 0 || token >= boundary) {
                    continue;
                }
                const uint32_t rank = TokenRank(token, bitmap, prefix);
                if (rankToSlot.GetValue(rank) >= 0) {
                    continue;
                }
                while (nextFreeSlot < scratchCapacity_
                       && IsRetainedSlot(
                           nextFreeSlot, resident, bitmap, prefix,
                           rankToSlot)) {
                    ++nextFreeSlot;
                }
                if (nextFreeSlot >= scratchCapacity_) {
                    AscendC::Trap();
                    return;
                }
                rankToSlot.SetValue(
                    rank, static_cast<int32_t>(nextFreeSlot));
                resident.SetValue(nextFreeSlot, token);
                stateDirty = true;
                selectedPacked_.SetValue(
                    packedOffset + missCount, token);

                const uint32_t logicalBlock = nextFreeSlot / blockSize_;
                const uint32_t blockOffset = nextFreeSlot % blockSize_;
                const int32_t physicalBlock =
                    requestBlockTable_.GetValue(
                        static_cast<uint64_t>(req) * blockTableWidth_
                        + logicalBlock);
                if (physicalBlock <= 0) {
                    AscendC::Trap();
                    return;
                }
                targetSlots_.SetValue(
                    packedOffset + missCount,
                    static_cast<int64_t>(physicalBlock) * blockSize_
                        + blockOffset);
                ++missCount;
                ++nextFreeSlot;
            }
        }

        // Every LMCache-side occurrence, including duplicate MTP selections,
        // resolves to the physical slot retained or assigned above.
        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) != static_cast<int32_t>(req)) {
                continue;
            }
            const int32_t boundary = splitBoundary_.GetValue(row);
            const uint64_t rowOffset = static_cast<uint64_t>(row) * rowWidth_;
            for (uint32_t col = 0; col < rowWidth_; ++col) {
                const uint64_t indexOffset = rowOffset + col;
                const int32_t token = topkIndices_.GetValue(indexOffset);
                if (token < 0 || token >= boundary) {
                    continue;
                }
                const uint32_t rank = TokenRank(token, bitmap, prefix);
                topkIndices_.SetValue(
                    indexOffset, rankToSlot.GetValue(rank));
            }
        }
        if (stateDirty) {
            StoreResidentState(
                residentOffset, generationOffset, resident,
                residentGeneration);
        }
        SetCount(req, static_cast<int32_t>(missCount));

        if (req == 0 && clearInvalidRows_) {
            ClearInvalidRows();
        }
    }

private:
    __aicore__ inline uint32_t LowerMask(uint32_t bit) const
    {
        return bit == 0 ? 0 : ((1U << bit) - 1);
    }

    __aicore__ inline uint32_t Popcount32(uint32_t value) const
    {
        value = value - ((value >> 1) & 0x55555555U);
        value = (value & 0x33333333U) + ((value >> 2) & 0x33333333U);
        value = (value + (value >> 4)) & 0x0F0F0F0FU;
        value = value + (value >> 8);
        value = value + (value >> 16);
        return value & 0x3FU;
    }

    __aicore__ inline bool IsDesiredToken(
        int32_t token, AscendC::LocalTensor<int32_t> &bitmap) const
    {
        if (token < 0) {
            return false;
        }
        const uint32_t word = static_cast<uint32_t>(token) >> 5;
        if (word >= bitmapWords_) {
            return false;
        }
        const uint32_t bit = static_cast<uint32_t>(token) & 31;
        return (
            static_cast<uint32_t>(bitmap.GetValue(word)) & (1U << bit)) != 0;
    }

    __aicore__ inline uint32_t TokenRank(
        int32_t token,
        AscendC::LocalTensor<int32_t> &bitmap,
        AscendC::LocalTensor<int32_t> &prefix) const
    {
        const uint32_t word = static_cast<uint32_t>(token) >> 5;
        const uint32_t bit = static_cast<uint32_t>(token) & 31;
        return static_cast<uint32_t>(prefix.GetValue(word))
            + Popcount32(
                static_cast<uint32_t>(bitmap.GetValue(word))
                & LowerMask(bit));
    }

    __aicore__ inline bool IsRetainedSlot(
        uint32_t slot,
        AscendC::LocalTensor<int32_t> &resident,
        AscendC::LocalTensor<int32_t> &bitmap,
        AscendC::LocalTensor<int32_t> &prefix,
        AscendC::LocalTensor<int32_t> &rankToSlot) const
    {
        const int32_t token = resident.GetValue(slot);
        if (!IsDesiredToken(token, bitmap)) {
            return false;
        }
        const uint32_t rank = TokenRank(token, bitmap, prefix);
        // If corrupt state contains the same token twice, only the slot chosen
        // in rankToSlot is protected; the duplicate remains evictable.
        return rankToSlot.GetValue(rank) == static_cast<int32_t>(slot);
    }

    __aicore__ inline void StoreResidentState(
        uint64_t residentOffset,
        uint64_t generationOffset,
        AscendC::LocalTensor<int32_t> &resident,
        AscendC::LocalTensor<int64_t> &residentGeneration)
    {
        // All resident mutations are scalar by this point. A generation reset
        // was fenced V->S before reaching here, so S->MTE3 orders both local
        // buffers before DMA publishes them to GM.
        SyncPipeline<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(
            residentTokenIds_[residentOffset], resident, scratchCapacity_);
        AscendC::DataCopy(
            residentGenerations_[generationOffset],
            residentGeneration,
            residentGenerationBufferWords_);
    }

    __aicore__ inline void SetCount(uint32_t req, int32_t count)
    {
        selectedCounts_.SetValue(
            static_cast<uint64_t>(req) * selectedCountStride_, count);
    }

    __aicore__ inline void ClearInvalidRows()
    {
        for (uint32_t row = 0; row < rowCount_; ++row) {
            if (rowReqIndices_.GetValue(row) >= 0) {
                continue;
            }
            const uint64_t rowOffset = static_cast<uint64_t>(row) * rowWidth_;
            for (uint32_t col = 0; col < rowWidth_; ++col) {
                topkIndices_.SetValue(rowOffset + col, 0);
            }
        }
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> selectedCounts_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::GlobalTensor<int32_t> requestStateIndices_;
    AscendC::GlobalTensor<int64_t> requestGenerations_;
    AscendC::GlobalTensor<int32_t> residentTokenIds_;
    AscendC::GlobalTensor<int64_t> residentGenerations_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bitmapBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prefixBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rankToSlotBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> residentBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC>
        residentGenerationBuffer_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestCount_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint64_t tokenCapacity_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t bitmapWords_ = 0;
    uint32_t bitmapBufferWords_ = 0;
    uint32_t rankBufferWords_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t stateRowCount_ = 0;
    uint32_t residentTokenStride_ = 0;
    uint32_t residentGenerationStride_ = 0;
    static constexpr uint32_t residentGenerationBufferWords_ = 8;
    bool clearInvalidRows_ = false;
};

}  // namespace

extern "C" __global__ __aicore__ void dsa_prepare_sparse_indices_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* requestBlockTable,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* selectedCounts,
    __gm__ int64_t* targetSlots,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t requestCount,
    uint32_t blockTableWidth,
    uint32_t scratchCapacity,
    uint32_t selectedCountStride,
    uint32_t bitmapWords,
    uint32_t blockSize,
    uint32_t needPacked,
    uint32_t clearInvalidRows)
{
    // This is a vector-only kernel. Mixed AIC/AIV launches execute the entry
    // on both core types; letting AIC run the same in-place updates races with
    // AIV and can overwrite a request's remapped indices.
    if ASCEND_IS_AIV {
        DSAPrepareSparseIndicesKernel op;
        op.Init(topkIndices, splitBoundary, rowReqIndices, requestBlockTable,
                selectedPacked, selectedCounts, targetSlots,
                rowCount, rowWidth, requestCount, blockTableWidth,
                scratchCapacity, selectedCountStride, bitmapWords, blockSize, needPacked,
                clearInvalidRows);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_prepare_sparse_indices_reuse_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* requestBlockTable,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* selectedCounts,
    __gm__ int64_t* targetSlots,
    __gm__ int32_t* requestStateIndices,
    __gm__ int64_t* requestGenerations,
    __gm__ int32_t* residentTokenIds,
    __gm__ int64_t* residentGenerations,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t requestCount,
    uint32_t blockTableWidth,
    uint32_t scratchCapacity,
    uint32_t selectedCountStride,
    uint32_t bitmapWords,
    uint32_t blockSize,
    uint32_t stateRowCount,
    uint32_t residentTokenStride,
    uint32_t residentGenerationStride,
    uint32_t clearInvalidRows)
{
    if ASCEND_IS_AIV {
        DSAPrepareSparseIndicesReuseKernel op;
        op.Init(
            topkIndices, splitBoundary, rowReqIndices, requestBlockTable,
            selectedPacked, selectedCounts, targetSlots, requestStateIndices,
            requestGenerations, residentTokenIds, residentGenerations,
            rowCount, rowWidth, requestCount, blockTableWidth,
            scratchCapacity, selectedCountStride, bitmapWords, blockSize,
            stateRowCount, residentTokenStride, residentGenerationStride,
            clearInvalidRows);
        op.Process();
    }
}

namespace vllm_ascend {

void dsa_prepare_sparse_indices_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* rowReqIndices, void* requestBlockTable, void* selectedPacked,
    void* selectedCounts, void* targetSlots,
    uint32_t rowCount, uint32_t rowWidth, uint32_t requestCount,
    uint32_t blockTableWidth, uint32_t scratchCapacity,
    uint32_t selectedCountStride,
    uint32_t bitmapWords, uint32_t blockSize, bool needPacked,
    bool clearInvalidRows)
{
    dsa_prepare_sparse_indices_kernel<<<requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(selectedCounts),
        static_cast<int64_t*>(targetSlots), rowCount, rowWidth,
        requestCount, blockTableWidth, scratchCapacity, selectedCountStride,
        bitmapWords,
        blockSize, needPacked, clearInvalidRows);
}

void dsa_prepare_sparse_indices_reuse_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* rowReqIndices, void* requestBlockTable, void* selectedPacked,
    void* selectedCounts, void* targetSlots, void* requestStateIndices,
    void* requestGenerations, void* residentTokenIds,
    void* residentGenerations, uint32_t rowCount, uint32_t rowWidth,
    uint32_t requestCount, uint32_t blockTableWidth,
    uint32_t scratchCapacity, uint32_t selectedCountStride,
    uint32_t bitmapWords, uint32_t blockSize, uint32_t stateRowCount,
    uint32_t residentTokenStride, uint32_t residentGenerationStride,
    bool clearInvalidRows)
{
    dsa_prepare_sparse_indices_reuse_kernel
        <<<requestCount, nullptr, stream>>>(
            static_cast<int32_t*>(topkIndices),
            static_cast<int32_t*>(splitBoundary),
            static_cast<int32_t*>(rowReqIndices),
            static_cast<int32_t*>(requestBlockTable),
            static_cast<int32_t*>(selectedPacked),
            static_cast<int32_t*>(selectedCounts),
            static_cast<int64_t*>(targetSlots),
            static_cast<int32_t*>(requestStateIndices),
            static_cast<int64_t*>(requestGenerations),
            static_cast<int32_t*>(residentTokenIds),
            static_cast<int64_t*>(residentGenerations),
            rowCount, rowWidth, requestCount, blockTableWidth,
            scratchCapacity, selectedCountStride, bitmapWords, blockSize,
            stateRowCount, residentTokenStride, residentGenerationStride,
            clearInvalidRows);
}

}  // namespace vllm_ascend
