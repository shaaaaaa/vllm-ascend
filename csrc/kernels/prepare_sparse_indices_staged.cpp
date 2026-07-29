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

constexpr uint32_t kSortGroup = 32;
constexpr uint32_t kPairWidth = 2;
constexpr uint32_t kMergeWays = 4;
constexpr uint32_t kCumSumTileWidth = 512;
constexpr uint32_t kCumSumTransposeRows = 16;
constexpr uint32_t kCumSumWorkspaceBytes =
    2 * kCumSumTransposeRows * kCumSumTileWidth * sizeof(float);
constexpr uint32_t kDataBlockBytes = 32;
constexpr uint32_t kInt32PerDataBlock =
    kDataBlockBytes / sizeof(int32_t);
constexpr uint32_t kInt32PerCacheline = 16;
constexpr AscendC::CumSumConfig kCumSumConfig{true, false, false};

template <AscendC::HardEvent event>
__aicore__ inline void Sync()
{
    const int32_t id =
        static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(id);
    AscendC::WaitFlag<event>(id);
}

template <typename T>
__aicore__ inline void CopyLocalToGlobalExact(
    AscendC::GlobalTensor<T> dst,
    AscendC::LocalTensor<T> src,
    uint32_t count)
{
    if (count == 0) {
        return;
    }
    const uint32_t bytes = count * sizeof(T);
    if ((bytes & (kDataBlockBytes - 1)) == 0) {
        AscendC::DataCopy(dst, src, count);
        return;
    }
    // The count-based DataCopy rounds an unaligned byte length down.
    const AscendC::DataCopyParams params{
        1,
        static_cast<uint16_t>(bytes),
        0,
        0};
    AscendC::DataCopyPad(dst, src, params);
}

__aicore__ inline void CopyLocalToGlobalExact(
    AscendC::GlobalTensor<int64_t> dst,
    AscendC::LocalTensor<int64_t> src,
    uint32_t count)
{
    if (count == 0) {
        return;
    }
    AscendC::GlobalTensor<int32_t> dstWords;
    dstWords.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(
            const_cast<__gm__ int64_t*>(dst.GetPhyAddr())),
        2 * count);
    CopyLocalToGlobalExact(
        dstWords, src.ReinterpretCast<int32_t>(), 2 * count);
}

template <typename T>
__aicore__ inline void CopyGlobalToLocalExact(
    AscendC::LocalTensor<T> dst,
    AscendC::GlobalTensor<T> src,
    uint32_t count)
{
    if (count == 0) {
        return;
    }
    const uint32_t bytes = count * sizeof(T);
    if ((bytes & (kDataBlockBytes - 1)) == 0) {
        AscendC::DataCopy(dst, src, count);
        return;
    }
    // DataCopyPad keeps the dynamic tail and pads only the local destination.
    const AscendC::DataCopyParams params{
        1,
        static_cast<uint16_t>(bytes),
        0,
        0};
    AscendC::DataCopyPad(dst, src, params, {});
}

// Production pre-union stage for fixed-width pure-decode batches. Each AIV
// owns one complete top-k row, so all GM writes are naturally cacheline
// disjoint. Selected tokens are compacted into the caller-owned output buffer;
// the same row is remapped in place to row-local ranks. Unselected positions
// remain absolute for the later boundary-aware remap.
class DSAStagedCompactRowsKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* rowPacked,
        __gm__ int32_t* rowCounts,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t requestCount,
        uint32_t rowsPerRequest,
        uint32_t scratchCapacity,
        uint32_t coreCount,
        bool clearInvalidRows)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        requestCount_ = requestCount;
        rowsPerRequest_ = rowsPerRequest;
        scratchCapacity_ = scratchCapacity;
        coreCount_ = coreCount;
        clearInvalidRows_ = clearInvalidRows;
        topkIndices_.SetGlobalBuffer(
            topkIndices,
            static_cast<uint64_t>(rowCount_) * rowWidth_);
        splitBoundary_.SetGlobalBuffer(splitBoundary, rowCount_);
        rowReqIndices_.SetGlobalBuffer(rowReqIndices, rowCount_);
        rowPacked_.SetGlobalBuffer(
            rowPacked,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        rowCounts_.SetGlobalBuffer(
            rowCounts,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);

        const uint32_t rowBytes = rowWidth_ * sizeof(int32_t);
        const uint32_t maskBytes = rowWidth_ / 8;
        pipe_.InitBuffer(inputBuf_, rowBytes);
        pipe_.InitBuffer(clampedBuf_, rowBytes);
        pipe_.InitBuffer(packedBuf_, rowBytes);
        pipe_.InitBuffer(flagsBuf_, rowBytes);
        pipe_.InitBuffer(prefixBuf_, rowBytes);
        pipe_.InitBuffer(cumSumWorkspaceBuf_, kCumSumWorkspaceBytes);
        pipe_.InitBuffer(nonNegativeMaskBuf_, maskBytes);
        pipe_.InitBuffer(beforeBoundaryMaskBuf_, maskBytes);
        pipe_.InitBuffer(selectedMaskBuf_, maskBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        for (uint32_t row = core; row < rowCount_; row += coreCount_) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const uint32_t request = row / rowsPerRequest_;
        const uint32_t requestRow = row % rowsPerRequest_;
        const uint64_t inputOffset =
            static_cast<uint64_t>(row) * rowWidth_;
        const uint64_t packedOffset =
            static_cast<uint64_t>(request) * scratchCapacity_
            + static_cast<uint64_t>(requestRow) * rowWidth_;
        const int32_t rowRequest = rowReqIndices_.GetValue(row);

        auto input = inputBuf_.Get<int32_t>();
        auto clamped = clampedBuf_.Get<int32_t>();
        auto packed = packedBuf_.Get<int32_t>();
        auto flags = flagsBuf_.Get<float>();
        auto prefix = prefixBuf_.Get<float>();
        auto workspace = cumSumWorkspaceBuf_.Get<uint8_t>();
        auto nonNegativeMask = nonNegativeMaskBuf_.Get<uint8_t>();
        auto beforeBoundaryMask = beforeBoundaryMaskBuf_.Get<uint8_t>();
        auto selectedMask = selectedMaskBuf_.Get<uint8_t>();

        AscendC::DataCopy(
            input, topkIndices_[inputOffset], rowWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();

        if (rowRequest < 0) {
            AscendC::Duplicate(
                packed, static_cast<int32_t>(0x7FFFFFFF), rowWidth_);
            if (clearInvalidRows_) {
                AscendC::Duplicate(
                    input, static_cast<int32_t>(0), rowWidth_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(
                rowPacked_[packedOffset], packed, rowWidth_);
            if (clearInvalidRows_) {
                AscendC::DataCopy(
                    topkIndices_[inputOffset], input, rowWidth_);
            }
            rowCounts_.SetValue(packedOffset, 0);
            return;
        }

        const int32_t boundary = splitBoundary_.GetValue(row);
        if (boundary <= 0) {
            AscendC::Duplicate(
                packed, static_cast<int32_t>(0x7FFFFFFF), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(
                rowPacked_[packedOffset], packed, rowWidth_);
            rowCounts_.SetValue(packedOffset, 0);
            return;
        }
        AscendC::Maxs(
            clamped, input, static_cast<int32_t>(0), rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            nonNegativeMask,
            clamped,
            input,
            AscendC::CMPMODE::EQ,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(
            clamped, input, boundary - 1, rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            beforeBoundaryMask,
            clamped,
            input,
            AscendC::CMPMODE::EQ,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(
            selectedMask.ReinterpretCast<uint16_t>(),
            nonNegativeMask.ReinterpretCast<uint16_t>(),
            beforeBoundaryMask.ReinterpretCast<uint16_t>(),
            rowWidth_ / 16);
        AscendC::PipeBarrier<PIPE_V>();

        // INT32_MAX converts to the smallest sort key after negation, keeping
        // every padded element behind all valid non-negative token positions.
        AscendC::Duplicate(
            packed, static_cast<int32_t>(0x7FFFFFFF), rowWidth_);
        AscendC::Duplicate(prefix, 1.0F, rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            flags,
            selectedMask,
            prefix,
            0.0F,
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::GatherMaskParams gatherParams;
        gatherParams.repeatTimes = 1;
        gatherParams.src0BlockStride = 1;
        gatherParams.src0RepeatStride = 8;
        gatherParams.src1RepeatStride = 8;
        uint64_t selectedCount = 0;
        AscendC::GatherMask(
            packed,
            input,
            selectedMask.ReinterpretCast<uint32_t>(),
            true,
            rowWidth_,
            gatherParams,
            selectedCount);
        AscendC::PipeBarrier<PIPE_V>();

        float carry = 0.0F;
        auto lastRow = clamped.ReinterpretCast<float>();
        for (uint32_t tileOffset = 0; tileOffset < rowWidth_;
             tileOffset += kCumSumTileWidth) {
            const AscendC::CumSumInfo info{1, kCumSumTileWidth};
            auto tilePrefix = prefix[tileOffset];
            auto tileFlags = flags[tileOffset];
            AscendC::CumSum<float, kCumSumConfig>(
                tilePrefix,
                lastRow,
                tileFlags,
                workspace,
                info);
            AscendC::PipeBarrier<PIPE_V>();
            if (tileOffset != 0) {
                Sync<AscendC::HardEvent::S_V>();
                AscendC::Adds(
                    tilePrefix, tilePrefix, carry, kCumSumTileWidth);
                AscendC::PipeBarrier<PIPE_V>();
            }
            Sync<AscendC::HardEvent::V_S>();
            carry = prefix.GetValue(
                tileOffset + kCumSumTileWidth - 1);
        }

        AscendC::Cast(
            clamped,
            prefix,
            AscendC::RoundMode::CAST_ROUND,
            rowWidth_);
        AscendC::Adds(clamped, clamped, -1, rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            input.ReinterpretCast<float>(),
            selectedMask,
            clamped.ReinterpretCast<float>(),
            input.ReinterpretCast<float>(),
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            topkIndices_[inputOffset], input, rowWidth_);
        AscendC::DataCopy(
            rowPacked_[packedOffset], packed, rowWidth_);
        rowCounts_.SetValue(
            packedOffset, static_cast<int32_t>(selectedCount));
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> rowPacked_;
    AscendC::GlobalTensor<int32_t> rowCounts_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> clampedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> flagsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prefixBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cumSumWorkspaceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> nonNegativeMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> beforeBoundaryMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestCount_ = 0;
    uint32_t rowsPerRequest_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t coreCount_ = 0;
    bool clearInvalidRows_ = false;
};

class DSAStagedHashUnionKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* rowPacked,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* localToUnion,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int64_t* targetSlots,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t maxTokens,
        uint32_t blockTableWidth,
        uint32_t selectedCountStride,
        uint32_t blockSize)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        requestCount_ = rowCount / 2;
        requestWidth_ = 2 * rowWidth;
        maxTokens_ = maxTokens;
        hashCapacity_ = 2 * requestWidth_;
        uint32_t capacityBits = 0;
        uint32_t capacity = hashCapacity_;
        while (capacity > 1) {
            capacity >>= 1;
            ++capacityBits;
        }
        hashShift_ = 32 - capacityBits;
        blockTableWidth_ = blockTableWidth;
        selectedCountStride_ = selectedCountStride;
        blockSize_ = blockSize;
        uint32_t shiftedBlockSize = blockSize_;
        while (shiftedBlockSize > 1) {
            shiftedBlockSize >>= 1;
            ++blockSizeShift_;
        }
        rowPacked_.SetGlobalBuffer(rowPacked, rowCount * rowWidth);
        selectedPacked_.SetGlobalBuffer(
            selectedPacked, requestCount_ * requestWidth_);
        localToUnion_.SetGlobalBuffer(localToUnion, rowCount * rowWidth);
        selectedCount_.SetGlobalBuffer(
            selectedCount, requestCount_ * selectedCountStride_);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable, requestCount_ * blockTableWidth);
        targetSlots_.SetGlobalBuffer(
            targetSlots, requestCount_ * requestWidth_);
        pipe_.InitBuffer(
            inputBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            hashBuf_, hashCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(
            unionBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            mappingBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            targetBuf_, requestWidth_ * sizeof(int64_t));
        pipe_.InitBuffer(
            blockTableBuf_, blockTableWidth_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        const uint64_t rowOffset =
            static_cast<uint64_t>(request) * requestWidth_;
        const uint64_t outputOffset =
            static_cast<uint64_t>(request) * requestWidth_;
        auto input = inputBuf_.Get<int32_t>();
        auto hash = hashBuf_.Get<int32_t>();
        auto unionLocal = unionBuf_.Get<int32_t>();
        auto mapping = mappingBuf_.Get<int32_t>();
        auto targets = targetBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();
        AscendC::DataCopy(
            input, rowPacked_[rowOffset], requestWidth_);
        CopyGlobalToLocalExact(
            blockTable,
            requestBlockTable_[
                static_cast<uint64_t>(request) * blockTableWidth_],
            blockTableWidth_);
        AscendC::Duplicate(hash, static_cast<int32_t>(-1), hashCapacity_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::MTE2_S>();
        Sync<AscendC::HardEvent::V_S>();

        uint32_t count = 0;
        for (uint32_t i = 0; i < requestWidth_; ++i) {
            const int32_t token = input.GetValue(i);
            if (token < 0 || token >= static_cast<int32_t>(maxTokens_)) {
                mapping.SetValue(i, static_cast<int32_t>(-1));
                continue;
            }
            uint32_t bucket =
                (static_cast<uint32_t>(token) * 2654435761U) >> hashShift_;
            for (uint32_t probe = 0; probe < hashCapacity_; ++probe) {
                const uint32_t slot =
                    (bucket + probe) & (hashCapacity_ - 1);
                const int32_t existing = hash.GetValue(slot);
                if (existing < 0) {
                    hash.SetValue(slot, static_cast<int32_t>(count));
                    unionLocal.SetValue(count, token);
                    mapping.SetValue(i, static_cast<int32_t>(count));
                    ++count;
                    break;
                }
                if (unionLocal.GetValue(
                        static_cast<uint32_t>(existing)) == token) {
                    mapping.SetValue(i, existing);
                    break;
                }
            }
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        CopyLocalToGlobalExact(
            selectedPacked_[outputOffset], unionLocal, count);
        AscendC::DataCopy(
            localToUnion_[rowOffset], mapping, requestWidth_);
        Sync<AscendC::HardEvent::MTE3_V>();
        Sync<AscendC::HardEvent::S_V>();

        auto ranks = hash;
        auto logicalBlocks = hash[requestWidth_];
        auto physicalBlocks = input;
        auto blockTableOffsets = mapping;
        AscendC::CreateVecIndex(
            ranks, static_cast<int32_t>(0), count);
        AscendC::ShiftRight(
            logicalBlocks,
            ranks,
            static_cast<int32_t>(blockSizeShift_),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            blockTableOffsets,
            logicalBlocks,
            static_cast<int32_t>(sizeof(int32_t)),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(
            physicalBlocks,
            blockTable,
            blockTableOffsets.ReinterpretCast<uint32_t>(),
            static_cast<uint32_t>(0),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            physicalBlocks,
            physicalBlocks,
            static_cast<int32_t>(blockSize_),
            count);
        AscendC::Muls(
            logicalBlocks,
            logicalBlocks,
            static_cast<int32_t>(blockSize_),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(ranks, ranks, logicalBlocks, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(
            physicalBlocks, physicalBlocks, ranks, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            targets,
            physicalBlocks,
            AscendC::RoundMode::CAST_NONE,
            count);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            targetSlots_[outputOffset], targets, count);
        selectedCount_.SetValue(
            request * selectedCountStride_, static_cast<int32_t>(count));
    }

private:
    AscendC::GlobalTensor<int32_t> rowPacked_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hashBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> unionBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mappingBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockTableBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestCount_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t maxTokens_ = 0;
    uint32_t hashCapacity_ = 0;
    uint32_t hashShift_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t blockSizeShift_ = 0;
};

class DSAStagedSortUnionKernel {
public:
    __aicore__ inline void InitProduction(
        __gm__ int32_t* rowPacked,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* localToUnion,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int64_t* targetSlots,
        __gm__ int32_t* rowCounts,
        uint32_t requestCount,
        uint32_t rowWidth,
        uint32_t scratchCapacity,
        uint32_t blockTableWidth,
        uint32_t selectedCountStride,
        uint32_t blockSize,
        bool needPacked)
    {
        Init(
            rowPacked,
            selectedPacked,
            localToUnion,
            selectedCount,
            requestBlockTable,
            targetSlots,
            2 * requestCount,
            rowWidth,
            blockTableWidth,
            selectedCountStride,
            blockSize);
        scratchCapacity_ = scratchCapacity;
        boundedRows_ = true;
        needPacked_ = needPacked;
        rowPacked_.SetGlobalBuffer(
            rowPacked,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        selectedPacked_.SetGlobalBuffer(
            selectedPacked,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        localToUnion_.SetGlobalBuffer(
            localToUnion,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        targetSlots_.SetGlobalBuffer(
            targetSlots,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        rowCounts_.SetGlobalBuffer(
            rowCounts,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
    }

    __aicore__ inline void Init(
        __gm__ int32_t* rowPacked,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* localToUnion,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int64_t* targetSlots,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t blockTableWidth,
        uint32_t selectedCountStride,
        uint32_t blockSize)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        requestCount_ = rowCount / 2;
        requestWidth_ = 2 * rowWidth;
        scratchCapacity_ = requestWidth_;
        blockTableWidth_ = blockTableWidth;
        selectedCountStride_ = selectedCountStride;
        blockSize_ = blockSize;
        uint32_t shiftedBlockSize = blockSize_;
        while (shiftedBlockSize > 1) {
            shiftedBlockSize >>= 1;
            ++blockSizeShift_;
        }
        rowPacked_.SetGlobalBuffer(rowPacked, rowCount * rowWidth);
        selectedPacked_.SetGlobalBuffer(
            selectedPacked, requestCount_ * requestWidth_);
        localToUnion_.SetGlobalBuffer(localToUnion, rowCount * rowWidth);
        selectedCount_.SetGlobalBuffer(
            selectedCount, requestCount_ * selectedCountStride_);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable, requestCount_ * blockTableWidth);
        targetSlots_.SetGlobalBuffer(
            targetSlots, requestCount_ * requestWidth_);
        pipe_.InitBuffer(
            sortSrcBuf_, requestWidth_ * kPairWidth * sizeof(float));
        pipe_.InitBuffer(
            sortTmpBuf_, requestWidth_ * kPairWidth * sizeof(float));
        pipe_.InitBuffer(
            sortInputBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            unionBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            mappingBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            targetBuf_, requestWidth_ * sizeof(int64_t));
        pipe_.InitBuffer(
            blockTableBuf_, blockTableWidth_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        const uint64_t rowOffset =
            static_cast<uint64_t>(request) * scratchCapacity_;
        const uint64_t outputOffset =
            static_cast<uint64_t>(request) * scratchCapacity_;
        auto src = sortSrcBuf_.Get<float>();
        auto tmp = sortTmpBuf_.Get<float>();
        auto input = sortInputBuf_.Get<int32_t>();
        auto unionLocal = unionBuf_.Get<int32_t>();
        auto mapping = mappingBuf_.Get<int32_t>();
        auto targets = targetBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();
        auto srcInt = src.ReinterpretCast<int32_t>();
        uint32_t validElements = requestWidth_;
        if (boundedRows_) {
            validElements = static_cast<uint32_t>(
                rowCounts_.GetValue(rowOffset))
                + static_cast<uint32_t>(
                    rowCounts_.GetValue(rowOffset + rowWidth_));
            if (validElements == 0) {
                selectedCount_.SetValue(
                    request * selectedCountStride_, 0);
                return;
            }
            AscendC::DataCopy(
                input, rowPacked_[rowOffset], rowWidth_);
            AscendC::DataCopy(
                input[rowWidth_],
                rowPacked_[rowOffset + rowWidth_],
                rowWidth_);
        } else {
            AscendC::DataCopy(
                input, rowPacked_[rowOffset], requestWidth_);
        }
        if (needPacked_) {
            CopyGlobalToLocalExact(
                blockTable,
                requestBlockTable_[
                    static_cast<uint64_t>(request) * blockTableWidth_],
                blockTableWidth_);
        }
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(
            src, input, AscendC::RoundMode::CAST_NONE, requestWidth_);
        AscendC::Muls(src, src, -1.0F, requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CreateVecIndex(
            srcInt[requestWidth_], static_cast<int32_t>(0), requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        SortAll(src, tmp);
        Sync<AscendC::HardEvent::V_S>();

        auto sortedInt = src.ReinterpretCast<int32_t>();
        int32_t previous = -1;
        uint32_t rank = 0;
        for (uint32_t i = 0; i < validElements; ++i) {
            const int32_t token =
                -static_cast<int32_t>(src.GetValue(kPairWidth * i));
            const uint32_t original = static_cast<uint32_t>(
                sortedInt.GetValue(kPairWidth * i + 1));
            if (i == 0 || token != previous) {
                unionLocal.SetValue(rank, token);
                previous = token;
                ++rank;
            }
            mapping.SetValue(original, static_cast<int32_t>(rank - 1));
        }

        if (needPacked_ && rank != 0) {
            Sync<AscendC::HardEvent::S_V>();
            auto ranks = src.ReinterpretCast<int32_t>();
            auto logicalBlocks = ranks[requestWidth_];
            auto physicalBlocks = tmp.ReinterpretCast<int32_t>();
            auto blockTableOffsets = physicalBlocks[requestWidth_];
            AscendC::CreateVecIndex(
                ranks, static_cast<int32_t>(0), rank);
            AscendC::ShiftRight(
                logicalBlocks,
                ranks,
                static_cast<int32_t>(blockSizeShift_),
                rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                blockTableOffsets,
                logicalBlocks,
                static_cast<int32_t>(sizeof(int32_t)),
                rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(
                physicalBlocks,
                blockTable,
                blockTableOffsets.ReinterpretCast<uint32_t>(),
                static_cast<uint32_t>(0),
                rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                physicalBlocks,
                physicalBlocks,
                static_cast<int32_t>(blockSize_),
                rank);
            AscendC::Muls(
                logicalBlocks,
                logicalBlocks,
                static_cast<int32_t>(blockSize_),
                rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(ranks, ranks, logicalBlocks, rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(
                physicalBlocks, physicalBlocks, ranks, rank);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                targets,
                physicalBlocks,
                AscendC::RoundMode::CAST_NONE,
                rank);
            AscendC::PipeBarrier<PIPE_V>();
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            localToUnion_[rowOffset], mapping, requestWidth_);
        if (needPacked_) {
            CopyLocalToGlobalExact(
                selectedPacked_[outputOffset], unionLocal, rank);
            CopyLocalToGlobalExact(
                targetSlots_[outputOffset], targets, rank);
        }
        selectedCount_.SetValue(
            request * selectedCountStride_,
            needPacked_ ? static_cast<int32_t>(rank) : 0);
    }

private:
    __aicore__ inline void SortAll(
        AscendC::LocalTensor<float>& src,
        AscendC::LocalTensor<float>& tmp)
    {
        const uint32_t repeats = requestWidth_ / kSortGroup;
        AscendC::Sort32(
            tmp,
            src,
            src[requestWidth_].ReinterpretCast<uint32_t>(),
            repeats);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t groups = repeats;
        uint32_t elements = kSortGroup;
        uint32_t pass = 0;
        while (groups > 1) {
            auto input = pass % 2 == 0 ? tmp : src;
            auto output = pass % 2 == 0 ? src : tmp;
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = elements;
            params.elementLengths[1] = elements;
            params.elementLengths[2] = elements;
            params.elementLengths[3] = elements;
            params.ifExhaustedSuspension = false;
            params.validBit = 0b1111;
            if (groups <= kMergeWays) {
                params.repeatTimes = 1;
                params.validBit =
                    groups == 2 ? 0b0011
                    : groups == 3 ? 0b0111
                                  : 0b1111;
            } else {
                params.repeatTimes = groups / kMergeWays;
            }
            AscendC::MrgSortSrcList<float> list;
            list.src1 = input;
            list.src2 = input[kPairWidth * elements];
            list.src3 = input[2 * kPairWidth * elements];
            list.src4 = input[3 * kPairWidth * elements];
            AscendC::MrgSort<float>(output, list, params);
            AscendC::PipeBarrier<PIPE_V>();
            groups = groups <= kMergeWays ? 1 : groups / kMergeWays;
            elements *= kMergeWays;
            ++pass;
        }
        if (pass % 2 == 0) {
            AscendC::DataCopy(
                src, tmp, requestWidth_ * kPairWidth);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::GlobalTensor<int32_t> rowPacked_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<int32_t> rowCounts_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortSrcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortInputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> unionBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mappingBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockTableBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestCount_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t blockSizeShift_ = 0;
    bool boundedRows_ = false;
    bool needPacked_ = true;
};

class DSAStagedSingleRowFinalizeKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* rowCounts,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int64_t* targetSlots,
        uint32_t requestCount,
        uint32_t rowWidth,
        uint32_t scratchCapacity,
        uint32_t blockTableWidth,
        uint32_t selectedCountStride,
        uint32_t blockSize,
        bool needPacked)
    {
        requestCount_ = requestCount;
        rowWidth_ = rowWidth;
        scratchCapacity_ = scratchCapacity;
        blockTableWidth_ = blockTableWidth;
        selectedCountStride_ = selectedCountStride;
        blockSize_ = blockSize;
        needPacked_ = needPacked;
        uint32_t shifted = blockSize_;
        while (shifted > 1) {
            shifted >>= 1;
            ++blockSizeShift_;
        }
        rowCounts_.SetGlobalBuffer(
            rowCounts,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        selectedCount_.SetGlobalBuffer(
            selectedCount,
            static_cast<uint64_t>(requestCount_) * selectedCountStride_);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable,
            static_cast<uint64_t>(requestCount_) * blockTableWidth_);
        targetSlots_.SetGlobalBuffer(
            targetSlots,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        pipe_.InitBuffer(rankBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(logicalBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(physicalBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(offsetBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(targetBuf_, rowWidth_ * sizeof(int64_t));
        pipe_.InitBuffer(
            blockTableBuf_, blockTableWidth_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        if (!needPacked_) {
            selectedCount_.SetValue(
                static_cast<uint64_t>(request) * selectedCountStride_, 0);
            return;
        }
        const uint64_t outputOffset =
            static_cast<uint64_t>(request) * scratchCapacity_;
        const uint32_t count = static_cast<uint32_t>(
            rowCounts_.GetValue(outputOffset));
        if (count == 0) {
            selectedCount_.SetValue(
                static_cast<uint64_t>(request) * selectedCountStride_, 0);
            return;
        }

        auto ranks = rankBuf_.Get<int32_t>();
        auto logical = logicalBuf_.Get<int32_t>();
        auto physical = physicalBuf_.Get<int32_t>();
        auto offsets = offsetBuf_.Get<int32_t>();
        auto targets = targetBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();
        if (needPacked_) {
            CopyGlobalToLocalExact(
                blockTable,
                requestBlockTable_[
                    static_cast<uint64_t>(request) * blockTableWidth_],
                blockTableWidth_);
        }
        Sync<AscendC::HardEvent::MTE2_V>();

        AscendC::CreateVecIndex(
            ranks, static_cast<int32_t>(0), count);
        AscendC::ShiftRight(
            logical,
            ranks,
            static_cast<int32_t>(blockSizeShift_),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            offsets,
            logical,
            static_cast<int32_t>(sizeof(int32_t)),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(
            physical,
            blockTable,
            offsets.ReinterpretCast<uint32_t>(),
            static_cast<uint32_t>(0),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            physical,
            physical,
            static_cast<int32_t>(blockSize_),
            count);
        AscendC::Muls(
            logical,
            logical,
            static_cast<int32_t>(blockSize_),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(ranks, ranks, logical, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(physical, physical, ranks, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            targets,
            physical,
            AscendC::RoundMode::CAST_NONE,
            count);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            targetSlots_[outputOffset], targets, count);
        selectedCount_.SetValue(
            static_cast<uint64_t>(request) * selectedCountStride_,
            static_cast<int32_t>(count));
    }

private:
    AscendC::GlobalTensor<int32_t> rowCounts_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rankBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> logicalBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> physicalBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockTableBuf_;
    uint32_t requestCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t blockSizeShift_ = 0;
    bool needPacked_ = true;
};

class DSAStagedBoundaryRemapKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* localToUnion,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t rowsPerRequest,
        uint32_t scratchCapacity,
        uint32_t coreCount)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        rowsPerRequest_ = rowsPerRequest;
        scratchCapacity_ = scratchCapacity;
        coreCount_ = coreCount;
        topkIndices_.SetGlobalBuffer(
            topkIndices,
            static_cast<uint64_t>(rowCount_) * rowWidth_);
        splitBoundary_.SetGlobalBuffer(splitBoundary, rowCount_);
        rowReqIndices_.SetGlobalBuffer(rowReqIndices, rowCount_);
        const uint32_t requestCount = rowCount_ / rowsPerRequest_;
        localToUnion_.SetGlobalBuffer(
            localToUnion,
            static_cast<uint64_t>(requestCount) * scratchCapacity_);
        const uint32_t rowBytes = rowWidth_ * sizeof(int32_t);
        const uint32_t maskBytes = rowWidth_ / 8;
        pipe_.InitBuffer(inputBuf_, rowBytes);
        pipe_.InitBuffer(clampedBuf_, rowBytes);
        pipe_.InitBuffer(mappingBuf_, rowBytes);
        pipe_.InitBuffer(outputBuf_, rowBytes);
        pipe_.InitBuffer(offsetBuf_, rowBytes);
        pipe_.InitBuffer(nonNegativeMaskBuf_, maskBytes);
        pipe_.InitBuffer(beforeBoundaryMaskBuf_, maskBytes);
        pipe_.InitBuffer(selectedMaskBuf_, maskBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        for (uint32_t row = core; row < rowCount_; row += coreCount_) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        if (rowReqIndices_.GetValue(row) < 0) {
            // The compact stage already zeroed graph padding rows.
            return;
        }
        const uint32_t request = row / rowsPerRequest_;
        const uint32_t requestRow = row % rowsPerRequest_;
        const uint64_t inputOffset =
            static_cast<uint64_t>(row) * rowWidth_;
        const uint64_t mappingOffset =
            static_cast<uint64_t>(request) * scratchCapacity_
            + static_cast<uint64_t>(requestRow) * rowWidth_;
        const int32_t boundary = splitBoundary_.GetValue(row);
        if (boundary <= 0) {
            return;
        }

        auto input = inputBuf_.Get<int32_t>();
        auto clamped = clampedBuf_.Get<int32_t>();
        auto mapping = mappingBuf_.Get<int32_t>();
        auto output = outputBuf_.Get<int32_t>();
        auto offsets = offsetBuf_.Get<uint32_t>();
        auto nonNegativeMask = nonNegativeMaskBuf_.Get<uint8_t>();
        auto beforeBoundaryMask = beforeBoundaryMaskBuf_.Get<uint8_t>();
        auto selectedMask = selectedMaskBuf_.Get<uint8_t>();
        AscendC::DataCopy(
            input, topkIndices_[inputOffset], rowWidth_);
        AscendC::DataCopy(
            mapping, localToUnion_[mappingOffset], rowWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();

        // After compaction selected positions contain a local rank. Because a
        // row is unique, every local rank is below its split boundary, while
        // ignored absolute positions are at or above that boundary.
        AscendC::Maxs(
            clamped, input, static_cast<int32_t>(0), rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            nonNegativeMask,
            clamped,
            input,
            AscendC::CMPMODE::EQ,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(
            clamped, input, boundary - 1, rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            beforeBoundaryMask,
            clamped,
            input,
            AscendC::CMPMODE::EQ,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(
            selectedMask.ReinterpretCast<uint16_t>(),
            nonNegativeMask.ReinterpretCast<uint16_t>(),
            beforeBoundaryMask.ReinterpretCast<uint16_t>(),
            rowWidth_ / 16);
        AscendC::PipeBarrier<PIPE_V>();

        // Clamp every lane before Gather; ignored absolute positions may be
        // much larger than the row-local map, even though Select discards
        // their gathered values.
        AscendC::Maxs(
            clamped, input, static_cast<int32_t>(0), rowWidth_);
        AscendC::Mins(
            clamped,
            clamped,
            static_cast<int32_t>(rowWidth_ - 1),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            offsets.ReinterpretCast<int32_t>(),
            clamped,
            static_cast<int32_t>(sizeof(int32_t)),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(
            output,
            mapping,
            offsets,
            static_cast<uint32_t>(0),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            output.ReinterpretCast<float>(),
            selectedMask,
            output.ReinterpretCast<float>(),
            input.ReinterpretCast<float>(),
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            topkIndices_[inputOffset], output, rowWidth_);
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> clampedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mappingBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> nonNegativeMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> beforeBoundaryMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t rowsPerRequest_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t coreCount_ = 0;
};

// Experimental MTP-aware sharded sort union. Tokens are partitioned by value,
// not by source position, across next_power_of_two(MTP) disjoint shards.
// Their concatenation is already a union, but only each individual shard is
// sorted. No code may compare ranks or infer token order across shards. The
// first launch produces shard-local ranks; the second launch chooses the
// concatenation order and converts them to request-local ranks.
class DSAStagedShardedSortKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* shardPacked,
        __gm__ int32_t* shardMapping,
        __gm__ int32_t* shardCounts,
        uint32_t requestCount,
        uint32_t rowsPerRequest,
        uint32_t rowWidth,
        uint32_t shardCapacity,
        uint32_t shardCount,
        uint32_t shardCountStride,
        uint32_t shardCountRequestStride,
        bool useRowReqIndices)
    {
        requestCount_ = requestCount;
        rowsPerRequest_ = rowsPerRequest;
        rowWidth_ = rowWidth;
        requestWidth_ = rowsPerRequest * rowWidth;
        shardWidth_ = shardCapacity;
        shardCount_ = shardCount;
        shardCountStride_ = shardCountStride;
        shardCountRequestStride_ = shardCountRequestStride;
        useRowReqIndices_ = useRowReqIndices;
        uint32_t shifted = shardCount_;
        while (shifted > 1) {
            shifted >>= 1;
            ++shardBits_;
        }
        topkIndices_.SetGlobalBuffer(
            topkIndices, requestCount_ * requestWidth_);
        splitBoundary_.SetGlobalBuffer(
            splitBoundary, requestCount_ * rowsPerRequest_);
        rowReqIndices_.SetGlobalBuffer(
            rowReqIndices, requestCount_ * rowsPerRequest_);
        shardPacked_.SetGlobalBuffer(
            shardPacked, requestCount_ * shardCount_ * shardWidth_);
        shardMapping_.SetGlobalBuffer(
            shardMapping, requestCount_ * shardCount_ * requestWidth_);
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            requestCount_ * shardCountRequestStride_);

        // Scan one MTP row at a time.  This keeps UB use independent of the
        // MTP depth while all rows belonging to the request feed one shard.
        pipe_.InitBuffer(inputBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(workBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(clampedBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(indexBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(compactTokenBuf_, shardWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(compactIndexBuf_, shardWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            sortSrcBuf_, shardWidth_ * kPairWidth * sizeof(float));
        pipe_.InitBuffer(
            sortTmpBuf_, shardWidth_ * kPairWidth * sizeof(float));
        pipe_.InitBuffer(unionBuf_, shardWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(mappingBuf_, requestWidth_ * sizeof(int32_t));
        const uint32_t maskBytes = rowWidth_ / 8;
        pipe_.InitBuffer(shardMaskBuf_, maskBytes);
        pipe_.InitBuffer(nonNegativeMaskBuf_, maskBytes);
        pipe_.InitBuffer(beforeBoundaryMaskBuf_, maskBytes);
        pipe_.InitBuffer(selectedMaskBuf_, maskBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t request = block / shardCount_;
        const uint32_t shard = block % shardCount_;
        if (request >= requestCount_) {
            return;
        }

        const uint64_t shardOffset =
            (static_cast<uint64_t>(request) * shardCount_ + shard)
            * shardWidth_;
        const uint64_t mapOffset =
            (static_cast<uint64_t>(request) * shardCount_ + shard)
            * requestWidth_;
        const uint64_t countOffset =
            static_cast<uint64_t>(request) * shardCountRequestStride_
            + shard * shardCountStride_;

        auto input = inputBuf_.Get<int32_t>();
        auto work = workBuf_.Get<int32_t>();
        auto clamped = clampedBuf_.Get<int32_t>();
        auto indices = indexBuf_.Get<int32_t>();
        auto compactTokens = compactTokenBuf_.Get<int32_t>();
        auto compactIndices = compactIndexBuf_.Get<int32_t>();
        auto src = sortSrcBuf_.Get<float>();
        auto tmp = sortTmpBuf_.Get<float>();
        auto unionLocal = unionBuf_.Get<int32_t>();
        auto mapping = mappingBuf_.Get<int32_t>();
        auto shardMask = shardMaskBuf_.Get<uint8_t>();
        auto nonNegativeMask = nonNegativeMaskBuf_.Get<uint8_t>();
        auto beforeBoundaryMask =
            beforeBoundaryMaskBuf_.Get<uint8_t>();
        auto selectedMask = selectedMaskBuf_.Get<uint8_t>();
        auto srcInt = src.ReinterpretCast<int32_t>();

        AscendC::Duplicate(
            compactTokens,
            static_cast<int32_t>(0x7FFFFFFF),
            shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::GatherMaskParams gatherParams;
        gatherParams.repeatTimes = 1;
        gatherParams.src0BlockStride = 1;
        gatherParams.src0RepeatStride = 8;
        gatherParams.src1RepeatStride = 8;
        uint32_t shardElements = 0;
        uint32_t compactEnd = 0;
        for (uint32_t mtpRow = 0; mtpRow < rowsPerRequest_; ++mtpRow) {
            const uint32_t row = request * rowsPerRequest_ + mtpRow;
            if (useRowReqIndices_ &&
                rowReqIndices_.GetValue(row) != static_cast<int32_t>(request)) {
                continue;
            }
            const uint64_t inputOffset =
                static_cast<uint64_t>(request) * requestWidth_
                + static_cast<uint64_t>(mtpRow) * rowWidth_;
            const int32_t boundary = splitBoundary_.GetValue(
                request * rowsPerRequest_ + mtpRow);
            AscendC::DataCopy(
                input, topkIndices_[inputOffset], rowWidth_);
            Sync<AscendC::HardEvent::MTE2_V>();

            // shard = token % next_power_of_two(MTP), expressed using
            // vector shifts/multiply/subtract instead of a scalar loop.
            AscendC::ShiftRight(
                work, input, static_cast<int32_t>(shardBits_), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                work, work, static_cast<int32_t>(shardCount_), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(work, input, work, rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(
                indices, static_cast<int32_t>(shard), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                shardMask,
                work,
                indices,
                AscendC::CMPMODE::EQ,
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(
                clamped, input, static_cast<int32_t>(0), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                nonNegativeMask,
                clamped,
                input,
                AscendC::CMPMODE::EQ,
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(
                clamped, input, boundary - 1, rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                beforeBoundaryMask,
                clamped,
                input,
                AscendC::CMPMODE::EQ,
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(
                selectedMask.ReinterpretCast<uint16_t>(),
                shardMask.ReinterpretCast<uint16_t>(),
                nonNegativeMask.ReinterpretCast<uint16_t>(),
                rowWidth_ / 16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(
                selectedMask.ReinterpretCast<uint16_t>(),
                selectedMask.ReinterpretCast<uint16_t>(),
                beforeBoundaryMask.ReinterpretCast<uint16_t>(),
                rowWidth_ / 16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CreateVecIndex(
                indices,
                static_cast<int32_t>(mtpRow * rowWidth_),
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();

            uint64_t tokenElements = 0;
            uint64_t indexElements = 0;
            // GatherMask requires a 32-byte-aligned local destination. Sort
            // sentinels already occupy any gap before this row's payload.
            const uint32_t compactOffset =
                (compactEnd + kInt32PerDataBlock - 1)
                & ~(kInt32PerDataBlock - 1);
            AscendC::GatherMask(
                compactTokens[compactOffset],
                input,
                selectedMask.ReinterpretCast<uint32_t>(),
                true,
                rowWidth_,
                gatherParams,
                tokenElements);
            AscendC::GatherMask(
                compactIndices[compactOffset],
                indices,
                selectedMask.ReinterpretCast<uint32_t>(),
                true,
                rowWidth_,
                gatherParams,
                indexElements);
            AscendC::PipeBarrier<PIPE_V>();
            Sync<AscendC::HardEvent::V_S>();
            shardElements += static_cast<uint32_t>(tokenElements);
            compactEnd =
                compactOffset + static_cast<uint32_t>(tokenElements);
        }

        AscendC::Duplicate(
            mapping, -static_cast<int32_t>(requestWidth_ + 1),
            requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        // A single MTP row is already unique by contract. Preserve its top-k
        // order and build only the compact mapping; do not sort or dedup.
        if (rowsPerRequest_ == 1) {
            for (uint32_t i = 0; i < shardElements; ++i) {
                const uint32_t original = static_cast<uint32_t>(
                    compactIndices.GetValue(i));
                mapping.SetValue(original, static_cast<int32_t>(i));
            }
            Sync<AscendC::HardEvent::S_MTE3>();
            Sync<AscendC::HardEvent::V_MTE3>();
            CopyLocalToGlobalExact(
                shardPacked_[shardOffset],
                compactTokens,
                shardElements);
            AscendC::DataCopy(
                shardMapping_[mapOffset], mapping, requestWidth_);
            // One 64-byte-aligned count cacheline is reserved per shard.
            shardCounts_.SetValue(
                countOffset, static_cast<int32_t>(shardElements));
            return;
        }

        const uint32_t sortElements = SortElementCount(compactEnd);
        AscendC::Cast(
            src,
            compactTokens,
            AscendC::RoundMode::CAST_NONE,
            sortElements);
        AscendC::Muls(src, src, -1.0F, sortElements);
        AscendC::DataCopy(
            srcInt[sortElements], compactIndices, sortElements);
        AscendC::PipeBarrier<PIPE_V>();
        SortAll(src, tmp, sortElements);
        // Finalize adds each shard's request-local prefix before taking the
        // maximum across dense maps.  A plain -1 sentinel would therefore
        // become a non-negative false rank for every shard with a non-zero
        // prefix and overwrite mappings owned by earlier shards.
        AscendC::Duplicate(
            mapping,
            -static_cast<int32_t>(requestWidth_ + 1),
            requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_S>();

        // The sort result contains only this value shard. No duplicate can
        // cross to a sibling shard. "rank" below is shard-local: sibling
        // shards have no relative token ordering, so only the finalizer may
        // add a prefix chosen from the actual shard concatenation order.
        auto sortedInt = src.ReinterpretCast<int32_t>();
        int32_t previous = -1;
        uint32_t rank = 0;
        const uint32_t count = shardElements;
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t token =
                -static_cast<int32_t>(src.GetValue(kPairWidth * i));
            const uint32_t original = static_cast<uint32_t>(
                sortedInt.GetValue(kPairWidth * i + 1));
            if (i == 0 || token != previous) {
                unionLocal.SetValue(rank, token);
                previous = token;
                ++rank;
            }
            mapping.SetValue(original, static_cast<int32_t>(rank - 1));
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            shardPacked_[shardOffset], unionLocal, rank);
        AscendC::DataCopy(
            shardMapping_[mapOffset], mapping, requestWidth_);
        // shardCountStride is at least 16 int32 values. Consequently every
        // stage AIV owns a distinct 64-byte GM cacheline for this scalar
        // count write.
        shardCounts_.SetValue(countOffset, static_cast<int32_t>(rank));
    }

private:
    __aicore__ inline uint32_t SortElementCount(uint32_t count)
    {
        uint32_t groups =
            (count + kSortGroup - 1) / kSortGroup;
        groups = groups == 0 ? 1 : groups;
        uint32_t scale = 1;
        while (groups > kMergeWays) {
            groups = (groups + kMergeWays - 1) / kMergeWays;
            scale *= kMergeWays;
        }
        return groups * scale * kSortGroup;
    }

    __aicore__ inline void SortAll(
        AscendC::LocalTensor<float>& src,
        AscendC::LocalTensor<float>& tmp,
        uint32_t sortElements)
    {
        const uint32_t repeats = sortElements / kSortGroup;
        AscendC::Sort32(
            tmp,
            src,
            src[sortElements].ReinterpretCast<uint32_t>(),
            repeats);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t groups = repeats;
        uint32_t elements = kSortGroup;
        uint32_t pass = 0;
        while (groups > 1) {
            auto input = pass % 2 == 0 ? tmp : src;
            auto output = pass % 2 == 0 ? src : tmp;
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = elements;
            params.elementLengths[1] = elements;
            params.elementLengths[2] = elements;
            params.elementLengths[3] = elements;
            params.ifExhaustedSuspension = false;
            params.validBit = 0b1111;
            if (groups <= kMergeWays) {
                params.repeatTimes = 1;
                params.validBit =
                    groups == 2 ? 0b0011
                    : groups == 3 ? 0b0111
                                  : 0b1111;
            } else {
                params.repeatTimes = groups / kMergeWays;
            }
            AscendC::MrgSortSrcList<float> list;
            list.src1 = input;
            list.src2 = input[kPairWidth * elements];
            list.src3 = input[2 * kPairWidth * elements];
            list.src4 = input[3 * kPairWidth * elements];
            AscendC::MrgSort<float>(output, list, params);
            AscendC::PipeBarrier<PIPE_V>();
            groups = groups <= kMergeWays ? 1 : groups / kMergeWays;
            elements *= kMergeWays;
            ++pass;
        }
        if (pass % 2 == 0) {
            AscendC::DataCopy(src, tmp, sortElements * kPairWidth);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> shardPacked_;
    AscendC::GlobalTensor<int32_t> shardMapping_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> clampedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> compactTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> compactIndexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortSrcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> unionBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mappingBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shardMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> nonNegativeMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> beforeBoundaryMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;
    uint32_t requestCount_ = 0;
    uint32_t rowsPerRequest_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t shardWidth_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t shardCountRequestStride_ = 0;
    uint32_t shardBits_ = 0;
    bool useRowReqIndices_ = false;
};

// Experimental follow-up to DSAStagedShardedSortKernel. Deduplication and
// local-rank generation stay on the vector pipeline. The original experiment
// emits pairs for a second position sort; the dense-map mode instead performs
// only the unavoidable branchless local scatter and lets position-owned AIVs
// merge shard-local ranks after their prefixes are known.
class DSAStagedShardedVectorStageKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* shardPacked,
        __gm__ int32_t* shardMapping,
        __gm__ int32_t* shardCounts,
        __gm__ int32_t* shardPairs,
        uint32_t requestCount,
        uint32_t rowsPerRequest,
        uint32_t rowWidth,
        uint32_t shardCount,
        uint32_t shardCountStride,
        bool writeDenseLocalMap)
    {
        requestCount_ = requestCount;
        rowsPerRequest_ = rowsPerRequest;
        rowWidth_ = rowWidth;
        requestWidth_ = rowsPerRequest * rowWidth;
        shardWidth_ = rowWidth;
        shardCount_ = shardCount;
        shardCountStride_ = shardCountStride;
        writeDenseLocalMap_ = writeDenseLocalMap;
        uint32_t shifted = shardCount_;
        while (shifted > 1) {
            shifted >>= 1;
            ++shardBits_;
        }
        topkIndices_.SetGlobalBuffer(
            topkIndices, requestCount_ * requestWidth_);
        splitBoundary_.SetGlobalBuffer(
            splitBoundary, requestCount_ * rowsPerRequest_);
        shardPacked_.SetGlobalBuffer(
            shardPacked, requestCount_ * shardCount_ * shardWidth_);
        shardMapping_.SetGlobalBuffer(
            shardMapping, requestCount_ * shardCount_ * requestWidth_);
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            requestCount_ * shardCount_ * shardCountStride_);
        if (!writeDenseLocalMap_) {
            shardPairs_.SetGlobalBuffer(
                shardPairs,
                requestCount_ * shardCount_ * shardWidth_ * kPairWidth);
        }

        pipe_.InitBuffer(inputBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(workBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(clampedBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(indexBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(compactTokenBuf_, shardWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(compactIndexBuf_, shardWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            cumSumWorkspaceBuf_, kCumSumWorkspaceBytes);
        if (rowsPerRequest_ == 1) {
            pipe_.InitBuffer(
                mappingBuf_, requestWidth_ * sizeof(int32_t));
        } else {
            pipe_.InitBuffer(
                sortSrcBuf_, shardWidth_ * kPairWidth * sizeof(float));
            pipe_.InitBuffer(
                sortTmpBuf_, shardWidth_ * kPairWidth * sizeof(float));
            pipe_.InitBuffer(
                unionBuf_, shardWidth_ * sizeof(int32_t));
        }
        const uint32_t maskBytes = rowWidth_ / 8;
        pipe_.InitBuffer(shardMaskBuf_, maskBytes);
        pipe_.InitBuffer(nonNegativeMaskBuf_, maskBytes);
        pipe_.InitBuffer(beforeBoundaryMaskBuf_, maskBytes);
        pipe_.InitBuffer(selectedMaskBuf_, maskBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t request = block / shardCount_;
        const uint32_t shard = block % shardCount_;
        if (request >= requestCount_) {
            return;
        }

        const uint64_t shardOffset =
            (static_cast<uint64_t>(request) * shardCount_ + shard)
            * shardWidth_;
        const uint64_t mapOffset =
            (static_cast<uint64_t>(request) * shardCount_ + shard)
            * requestWidth_;
        const uint64_t countOffset =
            (static_cast<uint64_t>(request) * shardCount_ + shard)
            * shardCountStride_;
        const uint64_t pairOffset =
            (static_cast<uint64_t>(request) * shardCount_ + shard)
            * shardWidth_ * kPairWidth;

        auto input = inputBuf_.Get<int32_t>();
        auto work = workBuf_.Get<int32_t>();
        auto clamped = clampedBuf_.Get<int32_t>();
        auto indices = indexBuf_.Get<int32_t>();
        auto compactTokens = compactTokenBuf_.Get<int32_t>();
        auto compactIndices = compactIndexBuf_.Get<int32_t>();
        auto shardMask = shardMaskBuf_.Get<uint8_t>();
        auto nonNegativeMask = nonNegativeMaskBuf_.Get<uint8_t>();
        auto beforeBoundaryMask =
            beforeBoundaryMaskBuf_.Get<uint8_t>();
        auto selectedMask = selectedMaskBuf_.Get<uint8_t>();

        AscendC::Duplicate(
            compactTokens,
            static_cast<int32_t>(0x7FFFFFFF),
            shardWidth_);
        AscendC::Duplicate(
            compactIndices,
            static_cast<int32_t>(requestWidth_ + 1),
            shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::GatherMaskParams gatherParams;
        gatherParams.repeatTimes = 1;
        gatherParams.src0BlockStride = 1;
        gatherParams.src0RepeatStride = 8;
        gatherParams.src1RepeatStride = 8;
        uint32_t shardElements = 0;
        uint32_t compactEnd = 0;
        for (uint32_t mtpRow = 0; mtpRow < rowsPerRequest_; ++mtpRow) {
            const uint64_t inputOffset =
                static_cast<uint64_t>(request) * requestWidth_
                + static_cast<uint64_t>(mtpRow) * rowWidth_;
            const int32_t boundary = splitBoundary_.GetValue(
                request * rowsPerRequest_ + mtpRow);
            AscendC::DataCopy(
                input, topkIndices_[inputOffset], rowWidth_);
            Sync<AscendC::HardEvent::MTE2_V>();

            AscendC::ShiftRight(
                work, input, static_cast<int32_t>(shardBits_), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                work, work, static_cast<int32_t>(shardCount_), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(work, input, work, rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(
                indices, static_cast<int32_t>(shard), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                shardMask,
                work,
                indices,
                AscendC::CMPMODE::EQ,
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(
                clamped, input, static_cast<int32_t>(0), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                nonNegativeMask,
                clamped,
                input,
                AscendC::CMPMODE::EQ,
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(
                clamped, input, boundary - 1, rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                beforeBoundaryMask,
                clamped,
                input,
                AscendC::CMPMODE::EQ,
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(
                selectedMask.ReinterpretCast<uint16_t>(),
                shardMask.ReinterpretCast<uint16_t>(),
                nonNegativeMask.ReinterpretCast<uint16_t>(),
                rowWidth_ / 16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(
                selectedMask.ReinterpretCast<uint16_t>(),
                selectedMask.ReinterpretCast<uint16_t>(),
                beforeBoundaryMask.ReinterpretCast<uint16_t>(),
                rowWidth_ / 16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CreateVecIndex(
                indices,
                static_cast<int32_t>(mtpRow * rowWidth_),
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();

            uint64_t tokenElements = 0;
            uint64_t indexElements = 0;
            // Sibling shards may finish in any order, but every GatherMask
            // destination within one shard must remain 32-byte aligned.
            const uint32_t compactOffset =
                (compactEnd + kInt32PerDataBlock - 1)
                & ~(kInt32PerDataBlock - 1);
            AscendC::GatherMask(
                compactTokens[compactOffset],
                input,
                selectedMask.ReinterpretCast<uint32_t>(),
                true,
                rowWidth_,
                gatherParams,
                tokenElements);
            AscendC::GatherMask(
                compactIndices[compactOffset],
                indices,
                selectedMask.ReinterpretCast<uint32_t>(),
                true,
                rowWidth_,
                gatherParams,
                indexElements);
            AscendC::PipeBarrier<PIPE_V>();
            Sync<AscendC::HardEvent::V_S>();
            shardElements += static_cast<uint32_t>(tokenElements);
            compactEnd =
                compactOffset + static_cast<uint32_t>(tokenElements);
        }

        if (rowsPerRequest_ == 1) {
            ProcessSingleRow(
                compactTokens, input, work, clamped, selectedMask,
                shardElements, shardOffset, mapOffset, countOffset);
            return;
        }

        if (shardElements == 0) {
            if (writeDenseLocalMap_) {
                auto mapping = cumSumWorkspaceBuf_.Get<int32_t>();
                AscendC::Duplicate(
                    mapping,
                    -static_cast<int32_t>(requestWidth_ + 1),
                    requestWidth_);
                AscendC::PipeBarrier<PIPE_V>();
                Sync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopy(
                    shardMapping_[mapOffset], mapping, requestWidth_);
            }
            shardCounts_.SetValue(countOffset, 0);
            shardCounts_.SetValue(countOffset + 1, 0);
            Sync<AscendC::HardEvent::S_MTE3>();
            return;
        }

        auto src = sortSrcBuf_.Get<float>();
        auto tmp = sortTmpBuf_.Get<float>();
        auto unionLocal = unionBuf_.Get<int32_t>();
        auto srcInt = src.ReinterpretCast<int32_t>();
        AscendC::Cast(
            src,
            compactTokens,
            AscendC::RoundMode::CAST_NONE,
            shardWidth_);
        AscendC::Muls(src, src, -1.0F, shardWidth_);
        AscendC::DataCopy(
            srcInt[shardWidth_], compactIndices, shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        SortAll(src, tmp);
        AscendC::PipeBarrier<PIPE_V>();

        // SortAll emits interleaved (key, payload) pairs. Fixed GatherMask
        // patterns extract the even token keys and odd original positions.
        auto sortedTokens = input.ReinterpretCast<float>();
        auto previousTokens = work.ReinterpretCast<float>();
        AscendC::GatherMaskParams fixedGatherParams;
        fixedGatherParams.repeatTimes = static_cast<uint8_t>(
            shardWidth_ * kPairWidth * sizeof(float) / 256);
        fixedGatherParams.src0BlockStride = 1;
        fixedGatherParams.src0RepeatStride = 8;
        fixedGatherParams.src1RepeatStride = 0;
        uint64_t extractedTokens = 0;
        uint64_t extractedIndices = 0;
        AscendC::GatherMask(
            sortedTokens,
            src,
            static_cast<uint8_t>(1),
            false,
            static_cast<uint32_t>(0),
            fixedGatherParams,
            extractedTokens);
        AscendC::GatherMask(
            indices,
            srcInt,
            static_cast<uint8_t>(2),
            false,
            static_cast<uint32_t>(0),
            fixedGatherParams,
            extractedIndices);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            sortedTokens, sortedTokens, -1.0F, shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        // Build previousTokens[i] = sortedTokens[max(i - 1, 0)] using Gather.
        // The first predecessor is overwritten with -1, which cannot equal a
        // selected token because split filtering already rejected negatives.
        AscendC::CreateVecIndex(compactTokens, 0, shardWidth_);
        AscendC::Adds(
            clamped, compactTokens, static_cast<int32_t>(-1), shardWidth_);
        AscendC::Maxs(
            clamped, clamped, static_cast<int32_t>(0), shardWidth_);
        AscendC::Muls(
            clamped,
            clamped,
            static_cast<int32_t>(sizeof(float)),
            shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(
            previousTokens,
            sortedTokens,
            clamped.ReinterpretCast<uint32_t>(),
            static_cast<uint32_t>(0),
            shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_S>();
        previousTokens.SetValue(0, -1.0F);
        Sync<AscendC::HardEvent::S_V>();

        // The validity mask prevents the first padded sort sentinel from
        // becoming a false unique head.
        AscendC::CreateVecIndex(compactTokens, 0, shardWidth_);
        AscendC::Mins(
            clamped,
            compactTokens,
            static_cast<int32_t>(shardElements - 1),
            shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            nonNegativeMask,
            clamped,
            compactTokens,
            AscendC::CMPMODE::EQ,
            shardWidth_);
        AscendC::Compare(
            selectedMask,
            sortedTokens,
            previousTokens,
            AscendC::CMPMODE::NE,
            shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(
            selectedMask.ReinterpretCast<uint16_t>(),
            selectedMask.ReinterpretCast<uint16_t>(),
            nonNegativeMask.ReinterpretCast<uint16_t>(),
            shardWidth_ / 16);
        AscendC::PipeBarrier<PIPE_V>();

        auto flags = previousTokens;
        auto prefix = clamped.ReinterpretCast<float>();
        AscendC::Duplicate(prefix, 1.0F, shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            flags,
            selectedMask,
            prefix,
            0.0F,
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        // The AICore compiler rejects a direct float-to-unsigned conversion.
        // PrefixSum is bounded by shardWidth_, so convert through int32_t.
        const int32_t signedUniqueCount = static_cast<int32_t>(
            PrefixSum(flags, prefix, compactIndices.ReinterpretCast<float>()));
        const uint32_t uniqueCount =
            static_cast<uint32_t>(signedUniqueCount);

        // Every occurrence receives its shard-local rank. GatherMask emits
        // only the head token for the shard union.
        AscendC::Adds(prefix, prefix, -1.0F, shardWidth_);
        AscendC::Cast(
            compactTokens,
            prefix,
            AscendC::RoundMode::CAST_ROUND,
            shardWidth_);
        AscendC::Cast(
            clamped,
            sortedTokens,
            AscendC::RoundMode::CAST_ROUND,
            shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t gatheredUnique = 0;
        AscendC::GatherMask(
            unionLocal,
            clamped,
            selectedMask.ReinterpretCast<uint32_t>(),
            true,
            shardWidth_,
            gatherParams,
            gatheredUnique);
        AscendC::PipeBarrier<PIPE_V>();

        if (writeDenseLocalMap_) {
            WriteDenseLocalMap(
                indices,
                compactTokens,
                unionLocal,
                shardElements,
                uniqueCount,
                shardOffset,
                mapOffset,
                countOffset);
            return;
        }

        // Sort only this shard's mapping payload by original position.
        // local_rank remains the payload; no cross-shard ordering is inferred.
        AscendC::Cast(
            src,
            indices,
            AscendC::RoundMode::CAST_NONE,
            shardWidth_);
        AscendC::Muls(src, src, -1.0F, shardWidth_);
        AscendC::DataCopy(
            srcInt[shardWidth_], compactTokens, shardWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        SortAll(src, tmp);

        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            shardPacked_[shardOffset], unionLocal, uniqueCount);
        AscendC::DataCopy(
            shardPairs_[pairOffset],
            src.ReinterpretCast<int32_t>(),
            shardWidth_ * kPairWidth);
        shardCounts_.SetValue(
            countOffset, static_cast<int32_t>(uniqueCount));
        shardCounts_.SetValue(
            countOffset + 1, static_cast<int32_t>(shardElements));
        Sync<AscendC::HardEvent::S_MTE3>();
    }

private:
    __aicore__ inline void WriteDenseLocalMap(
        AscendC::LocalTensor<int32_t>& sortedOriginal,
        AscendC::LocalTensor<int32_t>& sortedLocalRank,
        AscendC::LocalTensor<int32_t>& unionLocal,
        uint32_t shardElements,
        uint32_t uniqueCount,
        uint64_t shardOffset,
        uint64_t mapOffset,
        uint64_t countOffset)
    {
        // Publish the union before reusing the CumSum workspace for the dense
        // scatter.  Keeping this copy adjacent to the GatherMask producer
        // avoids extending unionLocal's vector-pipeline lifetime across the
        // following scalar scatter.
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            shardPacked_[shardOffset], unionLocal, uniqueCount);

        // CumSum has completed, so its 64-KiB workspace can be reused as the
        // dense request mapping without increasing peak UB use. Every stage
        // AIV writes only its own shard scratch segment.
        auto mapping = cumSumWorkspaceBuf_.Get<int32_t>();
        AscendC::Duplicate(
            mapping,
            -static_cast<int32_t>(requestWidth_ + 1),
            requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_S>();
        for (uint32_t i = 0; i < shardElements; ++i) {
            const uint32_t original = static_cast<uint32_t>(
                sortedOriginal.GetValue(i));
            const int32_t localRank = sortedLocalRank.GetValue(i);
            mapping.SetValue(original, localRank);
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            shardMapping_[mapOffset], mapping, requestWidth_);
        shardCounts_.SetValue(
            countOffset, static_cast<int32_t>(uniqueCount));
        shardCounts_.SetValue(
            countOffset + 1, static_cast<int32_t>(shardElements));
        Sync<AscendC::HardEvent::S_MTE3>();
    }

    __aicore__ inline void ProcessSingleRow(
        AscendC::LocalTensor<int32_t>& compactTokens,
        AscendC::LocalTensor<int32_t>& input,
        AscendC::LocalTensor<int32_t>& work,
        AscendC::LocalTensor<int32_t>& clamped,
        AscendC::LocalTensor<uint8_t>& selectedMask,
        uint32_t shardElements,
        uint64_t shardOffset,
        uint64_t mapOffset,
        uint64_t countOffset)
    {
        auto mapping = mappingBuf_.Get<int32_t>();
        AscendC::Duplicate(
            mapping,
            -static_cast<int32_t>(requestWidth_ + 1),
            requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        if (shardElements != 0) {
            auto flags = work.ReinterpretCast<float>();
            auto prefix = clamped.ReinterpretCast<float>();
            AscendC::Duplicate(prefix, 1.0F, rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(
                flags,
                selectedMask,
                prefix,
                0.0F,
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            PrefixSum(
                flags, prefix, compactIndexBuf_.Get<float>());
            AscendC::Cast(
                input,
                prefix,
                AscendC::RoundMode::CAST_ROUND,
                rowWidth_);
            AscendC::Adds(input, input, -1, rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(
                mapping.ReinterpretCast<float>(),
                selectedMask,
                input.ReinterpretCast<float>(),
                mapping.ReinterpretCast<float>(),
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Sync<AscendC::HardEvent::V_MTE3>();
        if (shardElements != 0) {
            CopyLocalToGlobalExact(
                shardPacked_[shardOffset],
                compactTokens,
                shardElements);
        }
        AscendC::DataCopy(
            shardMapping_[mapOffset], mapping, requestWidth_);
        shardCounts_.SetValue(
            countOffset, static_cast<int32_t>(shardElements));
        shardCounts_.SetValue(
            countOffset + 1, static_cast<int32_t>(shardElements));
        Sync<AscendC::HardEvent::S_MTE3>();
    }

    __aicore__ inline float PrefixSum(
        AscendC::LocalTensor<float>& flags,
        AscendC::LocalTensor<float>& prefix,
        AscendC::LocalTensor<float> lastRow)
    {
        auto workspace = cumSumWorkspaceBuf_.Get<uint8_t>();
        float carry = 0.0F;
        for (uint32_t tileOffset = 0; tileOffset < shardWidth_;
             tileOffset += kCumSumTileWidth) {
            AscendC::LocalTensor<float> tilePrefix =
                prefix[tileOffset];
            AscendC::LocalTensor<float> tileFlags =
                flags[tileOffset];
            const AscendC::CumSumInfo info{1, kCumSumTileWidth};
            AscendC::CumSum<float, kCumSumConfig>(
                tilePrefix,
                lastRow,
                tileFlags,
                workspace,
                info);
            AscendC::PipeBarrier<PIPE_V>();
            if (tileOffset != 0) {
                Sync<AscendC::HardEvent::S_V>();
                AscendC::Adds(
                    tilePrefix,
                    tilePrefix,
                    carry,
                    kCumSumTileWidth);
                AscendC::PipeBarrier<PIPE_V>();
            }
            Sync<AscendC::HardEvent::V_S>();
            carry = prefix.GetValue(
                tileOffset + kCumSumTileWidth - 1);
        }
        return carry;
    }

    __aicore__ inline void SortAll(
        AscendC::LocalTensor<float>& src,
        AscendC::LocalTensor<float>& tmp)
    {
        const uint32_t repeats = shardWidth_ / kSortGroup;
        AscendC::Sort32(
            tmp,
            src,
            src[shardWidth_].ReinterpretCast<uint32_t>(),
            repeats);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t groups = repeats;
        uint32_t elements = kSortGroup;
        uint32_t pass = 0;
        while (groups > 1) {
            auto input = pass % 2 == 0 ? tmp : src;
            auto output = pass % 2 == 0 ? src : tmp;
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = elements;
            params.elementLengths[1] = elements;
            params.elementLengths[2] = elements;
            params.elementLengths[3] = elements;
            params.ifExhaustedSuspension = false;
            params.validBit = 0b1111;
            if (groups <= kMergeWays) {
                params.repeatTimes = 1;
                params.validBit =
                    groups == 2 ? 0b0011
                    : groups == 3 ? 0b0111
                                  : 0b1111;
            } else {
                params.repeatTimes = groups / kMergeWays;
            }
            AscendC::MrgSortSrcList<float> list;
            list.src1 = input;
            list.src2 = input[kPairWidth * elements];
            list.src3 = input[2 * kPairWidth * elements];
            list.src4 = input[3 * kPairWidth * elements];
            AscendC::MrgSort<float>(output, list, params);
            AscendC::PipeBarrier<PIPE_V>();
            groups = groups <= kMergeWays ? 1 : groups / kMergeWays;
            elements *= kMergeWays;
            ++pass;
        }
        if (pass % 2 == 0) {
            AscendC::DataCopy(
                src, tmp, shardWidth_ * kPairWidth);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> shardPacked_;
    AscendC::GlobalTensor<int32_t> shardMapping_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::GlobalTensor<int32_t> shardPairs_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> clampedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> compactTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> compactIndexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortSrcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> unionBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mappingBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cumSumWorkspaceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shardMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> nonNegativeMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> beforeBoundaryMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;
    uint32_t requestCount_ = 0;
    uint32_t rowsPerRequest_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t shardWidth_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t shardBits_ = 0;
    bool writeDenseLocalMap_ = false;
};

class DSAStagedShardedParallelMapKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* shardPairs,
        __gm__ int32_t* shardMapping,
        __gm__ int32_t* shardCounts,
        uint32_t requestCount,
        uint32_t rowsPerRequest,
        uint32_t rowWidth,
        uint32_t shardCount,
        uint32_t shardCountStride,
        uint32_t mappingParts)
    {
        requestCount_ = requestCount;
        requestWidth_ = rowsPerRequest * rowWidth;
        rowWidth_ = rowWidth;
        shardCount_ = shardCount;
        shardCountStride_ = shardCountStride;
        mappingParts_ = mappingParts;
        partWidth_ = requestWidth_ / mappingParts_;
        const uint64_t pairElements =
            static_cast<uint64_t>(requestCount_) * shardCount_
            * rowWidth_ * kPairWidth;
        shardPairsFloat_.SetGlobalBuffer(
            (__gm__ float*)shardPairs, pairElements);
        shardPairsInt_.SetGlobalBuffer(shardPairs, pairElements);
        shardMapping_.SetGlobalBuffer(
            shardMapping,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * requestWidth_);
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * shardCountStride_);
        pipe_.InitBuffer(mappingBuf_, rowWidth_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t blocksPerRequest =
            shardCount_ * mappingParts_;
        const uint32_t request = block / blocksPerRequest;
        if (request >= requestCount_) {
            return;
        }
        const uint32_t requestBlock = block % blocksPerRequest;
        const uint32_t shard = requestBlock / mappingParts_;
        const uint32_t part = requestBlock % mappingParts_;
        const uint32_t destinationBegin = part * partWidth_;
        const uint32_t destinationEnd =
            destinationBegin + partWidth_;
        const uint64_t shardLinear =
            static_cast<uint64_t>(request) * shardCount_ + shard;
        const uint64_t pairOffset =
            shardLinear * rowWidth_ * kPairWidth;
        const uint64_t mapOffset =
            shardLinear * requestWidth_;
        const uint64_t countOffset =
            shardLinear * shardCountStride_;

        // Every mapping AIV owns a complete 256-byte-aligned interval.
        // Initialization and following scalar stores therefore cannot touch a
        // sibling AIV's 64-byte cacheline or vector transaction.
        auto mapping = mappingBuf_.Get<int32_t>();
        AscendC::Duplicate(
            mapping,
            -static_cast<int32_t>(requestWidth_ + 1),
            partWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            shardMapping_[mapOffset + destinationBegin],
            mapping,
            partWidth_);
        Sync<AscendC::HardEvent::MTE3_S>();

        const uint32_t occurrenceCount = static_cast<uint32_t>(
            shardCounts_.GetValue(countOffset + 1));
        const uint32_t begin = LowerBound(
            pairOffset, occurrenceCount, destinationBegin);
        const uint32_t end = LowerBound(
            pairOffset, occurrenceCount, destinationEnd);
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t original = static_cast<uint32_t>(
                -static_cast<int32_t>(
                    shardPairsFloat_.GetValue(
                        pairOffset + kPairWidth * i)));
            const int32_t localRank =
                shardPairsInt_.GetValue(
                    pairOffset + kPairWidth * i + 1);
            shardMapping_.SetValue(
                mapOffset + original, localRank);
        }
        Sync<AscendC::HardEvent::S_MTE3>();
    }

private:
    __aicore__ inline uint32_t LowerBound(
        uint64_t pairOffset,
        uint32_t count,
        uint32_t target) const
    {
        uint32_t first = 0;
        uint32_t last = count;
        while (first < last) {
            const uint32_t middle = first + (last - first) / 2;
            const uint32_t original = static_cast<uint32_t>(
                -static_cast<int32_t>(
                    shardPairsFloat_.GetValue(
                        pairOffset + kPairWidth * middle)));
            if (original < target) {
                first = middle + 1;
            } else {
                last = middle;
            }
        }
        return first;
    }

    AscendC::GlobalTensor<float> shardPairsFloat_;
    AscendC::GlobalTensor<int32_t> shardPairsInt_;
    AscendC::GlobalTensor<int32_t> shardMapping_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mappingBuf_;
    uint32_t requestCount_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t mappingParts_ = 0;
    uint32_t partWidth_ = 0;
};

// Runs after every value-shard stage AIV has completed. It chooses the final
// shard concatenation order, publishes the corresponding prefix offsets, and
// assembles the outputs that do not depend on the per-position mapping.
class DSAStagedShardedPrefixKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int64_t* targetSlots,
        __gm__ int32_t* shardPacked,
        __gm__ int32_t* shardCounts,
        uint32_t requestCount,
        uint32_t rowsPerRequest,
        uint32_t rowWidth,
        uint32_t shardCount,
        uint32_t blockTableWidth,
        uint32_t selectedCountStride,
        uint32_t shardCountStride,
        uint32_t blockSize)
    {
        requestCount_ = requestCount;
        requestWidth_ = rowsPerRequest * rowWidth;
        rowWidth_ = rowWidth;
        shardCount_ = shardCount;
        blockTableWidth_ = blockTableWidth;
        selectedCountStride_ = selectedCountStride;
        shardCountStride_ = shardCountStride;
        blockSize_ = blockSize;
        uint32_t shifted = blockSize_;
        while (shifted > 1) {
            shifted >>= 1;
            ++blockSizeShift_;
        }
        selectedPacked_.SetGlobalBuffer(
            selectedPacked,
            static_cast<uint64_t>(requestCount_) * requestWidth_);
        selectedCount_.SetGlobalBuffer(
            selectedCount,
            static_cast<uint64_t>(requestCount_) * selectedCountStride_);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable,
            static_cast<uint64_t>(requestCount_) * blockTableWidth_);
        targetSlots_.SetGlobalBuffer(
            targetSlots,
            static_cast<uint64_t>(requestCount_) * requestWidth_);
        shardPacked_.SetGlobalBuffer(
            shardPacked,
            static_cast<uint64_t>(requestCount_) * shardCount_ * rowWidth_);
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * shardCountStride_);

        pipe_.InitBuffer(packedBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(rankBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(logicalBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(physicalBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(targetBuf_, rowWidth_ * sizeof(int64_t));
        pipe_.InitBuffer(
            blockTableBuf_, blockTableWidth_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        const uint64_t outputBase =
            static_cast<uint64_t>(request) * requestWidth_;
        const uint64_t shardBase =
            static_cast<uint64_t>(request) * shardCount_ * rowWidth_;
        const uint64_t countBase =
            static_cast<uint64_t>(request) * selectedCountStride_;

        auto packed = packedBuf_.Get<int32_t>();
        auto ranks = rankBuf_.Get<int32_t>();
        auto logical = logicalBuf_.Get<int32_t>();
        auto physical = physicalBuf_.Get<int32_t>();
        auto targets = targetBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();

        CopyGlobalToLocalExact(
            blockTable,
            requestBlockTable_[
                static_cast<uint64_t>(request) * blockTableWidth_],
            blockTableWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();

        uint32_t count = 0;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            // Slots [1, 1 + shardCount) are scratch space for the following
            // position-owned map launch. Slot zero remains the public count.
            selectedCount_.SetValue(
                countBase + 1 + shard, static_cast<int32_t>(count));
            const uint32_t shardUnique = static_cast<uint32_t>(
                shardCounts_.GetValue(
                    (static_cast<uint64_t>(request) * shardCount_ + shard)
                    * shardCountStride_));
            if (shardUnique != 0) {
                CopyGlobalToLocalExact(
                    packed,
                    shardPacked_[shardBase + shard * rowWidth_],
                    shardUnique);
                Sync<AscendC::HardEvent::MTE2_MTE3>();
                CopyLocalToGlobalExact(
                    selectedPacked_[outputBase + count],
                    packed,
                    shardUnique);
                Sync<AscendC::HardEvent::MTE3_MTE2>();
            }
            count += shardUnique;
        }

        if (count != 0) {
            // The loop orders each selected copy against the following MTE2
            // load.  After the final shard, however, packed is reused directly
            // by vector target generation.  Complete that last MTE3 read
            // before allowing the V pipe to overwrite the same UB buffer.
            Sync<AscendC::HardEvent::MTE3_V>();
        }

        for (uint32_t offset = 0; offset < count; offset += rowWidth_) {
            const uint32_t remaining = count - offset;
            const uint32_t tile =
                remaining < rowWidth_ ? remaining : rowWidth_;
            AscendC::CreateVecIndex(
                ranks, static_cast<int32_t>(offset), tile);
            AscendC::ShiftRight(
                logical,
                ranks,
                static_cast<int32_t>(blockSizeShift_),
                tile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                packed,
                logical,
                static_cast<int32_t>(sizeof(int32_t)),
                tile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(
                physical,
                blockTable,
                packed.ReinterpretCast<uint32_t>(),
                static_cast<uint32_t>(0),
                tile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                physical,
                physical,
                static_cast<int32_t>(blockSize_),
                tile);
            AscendC::Muls(
                logical,
                logical,
                static_cast<int32_t>(blockSize_),
                tile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(ranks, ranks, logical, tile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(physical, physical, ranks, tile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                targets,
                physical,
                AscendC::RoundMode::CAST_NONE,
                tile);
            AscendC::PipeBarrier<PIPE_V>();
            Sync<AscendC::HardEvent::V_MTE3>();
            CopyLocalToGlobalExact(
                targetSlots_[outputBase + offset], targets, tile);
            Sync<AscendC::HardEvent::MTE3_V>();
        }
        selectedCount_.SetValue(
            countBase, static_cast<int32_t>(count));
        Sync<AscendC::HardEvent::S_MTE3>();
    }

private:
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::GlobalTensor<int32_t> shardPacked_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rankBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> logicalBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> physicalBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockTableBuf_;
    uint32_t requestCount_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t blockSizeShift_ = 0;
};

// Mapping ownership follows contiguous original-position ranges rather than
// value shards. Each AIV reads every shard's local ranks for its range, applies
// the prefix selected by DSAStagedShardedPrefixKernel, and exclusively writes
// complete cachelines of the final mapping and remapped top-k indices.
class DSAStagedShardedPositionMapKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* localToUnion,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* shardMapping,
        uint32_t requestCount,
        uint32_t rowsPerRequest,
        uint32_t rowWidth,
        uint32_t shardCount,
        uint32_t selectedCountStride)
    {
        requestCount_ = requestCount;
        requestWidth_ = rowsPerRequest * rowWidth;
        shardCount_ = shardCount;
        selectedCountStride_ = selectedCountStride;
        partWidth_ = requestWidth_ / shardCount_;
        topkIndices_.SetGlobalBuffer(
            topkIndices,
            static_cast<uint64_t>(requestCount_) * requestWidth_);
        localToUnion_.SetGlobalBuffer(
            localToUnion,
            static_cast<uint64_t>(requestCount_) * requestWidth_);
        selectedCount_.SetGlobalBuffer(
            selectedCount,
            static_cast<uint64_t>(requestCount_) * selectedCountStride_);
        shardMapping_.SetGlobalBuffer(
            shardMapping,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * requestWidth_);

        pipe_.InitBuffer(accumMapBuf_, partWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(shardMapBuf_, partWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(inputBuf_, partWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(outputBuf_, partWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(remapMaskBuf_, partWidth_ / 8);
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t request = block / shardCount_;
        const uint32_t part = block % shardCount_;
        if (request >= requestCount_) {
            return;
        }
        const uint32_t partBegin = part * partWidth_;
        const uint64_t requestBase =
            static_cast<uint64_t>(request) * requestWidth_;
        const uint64_t shardBase =
            static_cast<uint64_t>(request) * shardCount_ * requestWidth_;
        const uint64_t countBase =
            static_cast<uint64_t>(request) * selectedCountStride_;

        auto accumMap = accumMapBuf_.Get<int32_t>();
        auto shardMap = shardMapBuf_.Get<int32_t>();
        auto input = inputBuf_.Get<int32_t>();
        auto output = outputBuf_.Get<int32_t>();
        auto remapMask = remapMaskBuf_.Get<uint8_t>();

        AscendC::Duplicate(
            accumMap,
            -static_cast<int32_t>(requestWidth_ + 1),
            partWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const int32_t shardOffset =
                selectedCount_.GetValue(countBase + 1 + shard);
            AscendC::DataCopy(
                shardMap,
                shardMapping_[
                    shardBase
                    + static_cast<uint64_t>(shard) * requestWidth_
                    + partBegin],
                partWidth_);
            Sync<AscendC::HardEvent::MTE2_V>();
            AscendC::Adds(
                shardMap, shardMap, shardOffset, partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(
                accumMap, accumMap, shardMap, partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            // The next iteration reuses shardMap as an MTE2 destination.
            // PipeBarrier orders only vector instructions; without this
            // cross-pipe dependency the following DMA may overwrite ranks
            // that Max is still consuming.
            Sync<AscendC::HardEvent::V_MTE2>();
        }

        AscendC::DataCopy(
            input,
            topkIndices_[requestBase + partBegin],
            partWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Maxs(
            output, accumMap, static_cast<int32_t>(0), partWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            remapMask,
            output,
            accumMap,
            AscendC::CMPMODE::EQ,
            partWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            output.ReinterpretCast<float>(),
            remapMask,
            accumMap.ReinterpretCast<float>(),
            input.ReinterpretCast<float>(),
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            partWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            localToUnion_[requestBase + partBegin],
            accumMap,
            partWidth_);
        AscendC::DataCopy(
            topkIndices_[requestBase + partBegin],
            output,
            partWidth_);
    }

private:
    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> shardMapping_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumMapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shardMapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> remapMaskBuf_;
    uint32_t requestCount_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t partWidth_ = 0;
};

class DSAStagedShardedFinalizeKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* localToUnion,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int64_t* targetSlots,
        __gm__ int32_t* shardPacked,
        __gm__ int32_t* shardMapping,
        __gm__ int32_t* shardCounts,
        uint32_t requestCount,
        uint32_t rowsPerRequest,
        uint32_t rowWidth,
        uint32_t shardCapacity,
        uint32_t shardCount,
        uint32_t blockTableWidth,
        uint32_t selectedCountStride,
        uint32_t shardCountStride,
        uint32_t shardCountRequestStride,
        uint32_t blockSize,
        bool useRowReqIndices,
        bool clearInvalidRows,
        bool needPacked)
    {
        requestCount_ = requestCount;
        rowsPerRequest_ = rowsPerRequest;
        rowWidth_ = rowWidth;
        requestWidth_ = rowsPerRequest * rowWidth;
        shardCapacity_ = shardCapacity;
        shardCount_ = shardCount;
        blockTableWidth_ = blockTableWidth;
        selectedCountStride_ = selectedCountStride;
        shardCountStride_ = shardCountStride;
        shardCountRequestStride_ = shardCountRequestStride;
        blockSize_ = blockSize;
        useRowReqIndices_ = useRowReqIndices;
        clearInvalidRows_ = clearInvalidRows;
        needPacked_ = needPacked;
        uint32_t shifted = blockSize_;
        while (shifted > 1) {
            shifted >>= 1;
            ++blockSizeShift_;
        }
        topkIndices_.SetGlobalBuffer(
            topkIndices, requestCount_ * requestWidth_);
        rowReqIndices_.SetGlobalBuffer(
            rowReqIndices, requestCount_ * rowsPerRequest_);
        selectedPacked_.SetGlobalBuffer(
            selectedPacked, requestCount_ * requestWidth_);
        localToUnion_.SetGlobalBuffer(
            localToUnion, requestCount_ * requestWidth_);
        selectedCount_.SetGlobalBuffer(
            selectedCount, requestCount_ * selectedCountStride_);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable, requestCount_ * blockTableWidth_);
        targetSlots_.SetGlobalBuffer(
            targetSlots, requestCount_ * requestWidth_);
        shardPacked_.SetGlobalBuffer(
            shardPacked, requestCount_ * shardCount_ * shardCapacity_);
        shardMapping_.SetGlobalBuffer(
            shardMapping, requestCount_ * shardCount_ * requestWidth_);
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            requestCount_ * shardCountRequestStride_);
        pipe_.InitBuffer(packedBuf_, shardCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(accumMapBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(shardMapBuf_, requestWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(rankBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(logicalBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(physicalBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(targetBuf_, rowWidth_ * sizeof(int64_t));
        const uint32_t maskBytes = rowWidth_ / 8;
        pipe_.InitBuffer(remapMaskBuf_, maskBytes);
        pipe_.InitBuffer(
            blockTableBuf_, blockTableWidth_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        const uint64_t outputOffset =
            static_cast<uint64_t>(request) * requestWidth_;
        const uint64_t shardBase =
            static_cast<uint64_t>(request) * shardCount_ * shardCapacity_;
        const uint64_t mapBase =
            static_cast<uint64_t>(request) * shardCount_ * requestWidth_;

        auto packed = packedBuf_.Get<int32_t>();
        auto accumMap = accumMapBuf_.Get<int32_t>();
        auto shardMap = shardMapBuf_.Get<int32_t>();
        auto ranks = rankBuf_.Get<int32_t>();
        auto logical = logicalBuf_.Get<int32_t>();
        auto physical = physicalBuf_.Get<int32_t>();
        auto targets = targetBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();
        auto remapMask = remapMaskBuf_.Get<uint8_t>();
        CopyGlobalToLocalExact(
            blockTable,
            requestBlockTable_[
                static_cast<uint64_t>(request) * blockTableWidth_],
            blockTableWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();

        AscendC::Duplicate(
            accumMap,
            -static_cast<int32_t>(requestWidth_ + 1),
            requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t count = 0;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t shardUnique = static_cast<uint32_t>(
                shardCounts_.GetValue(
                    static_cast<uint64_t>(request)
                        * shardCountRequestStride_
                    + shard * shardCountStride_));
            CopyGlobalToLocalExact(
                packed,
                shardPacked_[shardBase + shard * shardCapacity_],
                shardUnique);
            AscendC::DataCopy(
                shardMap,
                shardMapping_[mapBase + shard * requestWidth_],
                requestWidth_);
            Sync<AscendC::HardEvent::MTE2_V>();
            // count is the prefix in the finalizer's chosen concatenation
            // order, not a token-order relation between sibling shards.
            AscendC::Adds(
                shardMap, shardMap, static_cast<int32_t>(count),
                requestWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(
                accumMap, accumMap, shardMap, requestWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            Sync<AscendC::HardEvent::V_MTE2>();
            if (needPacked_) {
                Sync<AscendC::HardEvent::MTE2_MTE3>();
                CopyLocalToGlobalExact(
                    selectedPacked_[outputOffset + count],
                    packed,
                    shardUnique);
                Sync<AscendC::HardEvent::MTE3_MTE2>();
            }
            count += shardUnique;
        }

        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            localToUnion_[outputOffset], accumMap, requestWidth_);

        // Boundary-selected positions have a non-negative union rank.
        // Preserve every ignored/live-cache position as its original absolute
        // token index while remapping selected positions in place.
        for (uint32_t mtpRow = 0; mtpRow < rowsPerRequest_; ++mtpRow) {
            const uint32_t rowOffset = mtpRow * rowWidth_;
            const uint32_t row = request * rowsPerRequest_ + mtpRow;
            if (useRowReqIndices_ &&
                rowReqIndices_.GetValue(row) != static_cast<int32_t>(request)) {
                if (clearInvalidRows_) {
                    AscendC::Duplicate(
                        physical, static_cast<int32_t>(0), rowWidth_);
                    AscendC::PipeBarrier<PIPE_V>();
                    Sync<AscendC::HardEvent::V_MTE3>();
                    AscendC::DataCopy(
                        topkIndices_[outputOffset + rowOffset],
                        physical,
                        rowWidth_);
                    Sync<AscendC::HardEvent::MTE3_MTE2>();
                }
                continue;
            }
            auto mapping = accumMap[rowOffset];
            AscendC::DataCopy(
                packed,
                topkIndices_[outputOffset + rowOffset],
                rowWidth_);
            Sync<AscendC::HardEvent::MTE2_V>();
            AscendC::Maxs(
                logical, mapping, static_cast<int32_t>(0), rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                remapMask,
                logical,
                mapping,
                AscendC::CMPMODE::EQ,
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(
                physical.ReinterpretCast<float>(),
                remapMask,
                mapping.ReinterpretCast<float>(),
                packed.ReinterpretCast<float>(),
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(
                topkIndices_[outputOffset + rowOffset],
                physical,
                rowWidth_);
            // The next row immediately reuses packed as an MTE2 destination.
            // Wait on that consumer pipe explicitly instead of relying on a
            // V-side dependency through a different local tensor.
            Sync<AscendC::HardEvent::MTE3_MTE2>();
        }

        if (needPacked_) {
            for (uint32_t offset = 0; offset < count; offset += rowWidth_) {
                const uint32_t remaining = count - offset;
                const uint32_t tile =
                    remaining < rowWidth_ ? remaining : rowWidth_;
                AscendC::CreateVecIndex(
                    ranks, static_cast<int32_t>(offset), tile);
                AscendC::ShiftRight(
                    logical,
                    ranks,
                    static_cast<int32_t>(blockSizeShift_),
                    tile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(
                    shardMap,
                    logical,
                    static_cast<int32_t>(sizeof(int32_t)),
                    tile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(
                    physical,
                    blockTable,
                    shardMap.ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0),
                    tile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(
                    physical, physical,
                    static_cast<int32_t>(blockSize_), tile);
                AscendC::Muls(
                    logical, logical,
                    static_cast<int32_t>(blockSize_), tile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(ranks, ranks, logical, tile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(physical, physical, ranks, tile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(
                    targets,
                    physical,
                    AscendC::RoundMode::CAST_NONE,
                    tile);
                AscendC::PipeBarrier<PIPE_V>();
                Sync<AscendC::HardEvent::V_MTE3>();
                CopyLocalToGlobalExact(
                    targetSlots_[outputOffset + offset], targets, tile);
                Sync<AscendC::HardEvent::MTE3_V>();
            }
        }
        selectedCount_.SetValue(
            static_cast<uint64_t>(request) * selectedCountStride_,
            needPacked_ ? static_cast<int32_t>(count) : 0);
    }

private:
    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::GlobalTensor<int32_t> shardPacked_;
    AscendC::GlobalTensor<int32_t> shardMapping_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumMapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shardMapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rankBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> logicalBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> physicalBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> remapMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockTableBuf_;
    uint32_t requestCount_ = 0;
    uint32_t rowsPerRequest_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t shardCapacity_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t shardCountRequestStride_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t blockSizeShift_ = 0;
    bool useRowReqIndices_ = false;
    bool clearInvalidRows_ = false;
    bool needPacked_ = true;
};

class DSAStagedRemapRowsKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* localIndices,
        __gm__ int32_t* localToUnion,
        uint32_t rowCount,
        uint32_t rowWidth)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        localIndices_.SetGlobalBuffer(
            localIndices, rowCount * rowWidth);
        localToUnion_.SetGlobalBuffer(
            localToUnion, rowCount * rowWidth);
        pipe_.InitBuffer(inputBuf_, rowWidth * sizeof(int32_t));
        pipe_.InitBuffer(mapBuf_, rowWidth * sizeof(int32_t));
        pipe_.InitBuffer(outputBuf_, rowWidth * sizeof(int32_t));
        pipe_.InitBuffer(offsetBuf_, rowWidth * sizeof(uint32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t row = AscendC::GetBlockIdx();
        if (row >= rowCount_) {
            return;
        }
        const uint32_t offset = row * rowWidth_;
        auto input = inputBuf_.Get<int32_t>();
        auto mapping = mapBuf_.Get<int32_t>();
        auto output = outputBuf_.Get<int32_t>();
        auto offsets = offsetBuf_.Get<uint32_t>();
        AscendC::DataCopy(input, localIndices_[offset], rowWidth_);
        AscendC::DataCopy(mapping, localToUnion_[offset], rowWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Muls(
            offsets.ReinterpretCast<int32_t>(),
            input,
            static_cast<int32_t>(sizeof(int32_t)),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(
            output, mapping, offsets, static_cast<uint32_t>(0), rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(localIndices_[offset], output, rowWidth_);
    }

private:
    AscendC::GlobalTensor<int32_t> localIndices_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
};

// Resident mapping is position-sharded: one AIV owns one complete source
// top-k row. Each output row is cacheline aligned and has a single writer.
class DSAResidentRemapRowsKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* positionToUnion,
        __gm__ int32_t* unionToSlot,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t rowsPerRequest,
        uint32_t scratchCapacity)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        rowsPerRequest_ = rowsPerRequest;
        scratchCapacity_ = scratchCapacity;
        topkIndices_.SetGlobalBuffer(
            topkIndices,
            static_cast<uint64_t>(rowCount_) * rowWidth_);
        positionToUnion_.SetGlobalBuffer(
            positionToUnion,
            static_cast<uint64_t>(rowCount_) * rowWidth_);
        const uint32_t requestCount = rowCount_ / rowsPerRequest_;
        unionToSlot_.SetGlobalBuffer(
            unionToSlot,
            static_cast<uint64_t>(requestCount) * scratchCapacity_);
        pipe_.InitBuffer(inputBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(rankBuf_, rowWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(clampedBuf_, rowWidth_ * sizeof(int32_t));
        // ranks becomes the Gather destination after Compare, while clamped
        // becomes the final Select destination after serving as byte offsets.
        // Their lifetimes do not overlap, so this reduces peak UB usage by two
        // row buffers without changing the remap algorithm.
        pipe_.InitBuffer(
            unionBuf_, scratchCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(selectedMaskBuf_, rowWidth_ / 8);
    }

    __aicore__ inline void Process()
    {
        const uint32_t row = AscendC::GetBlockIdx();
        if (row >= rowCount_) {
            return;
        }
        const uint32_t request = row / rowsPerRequest_;
        const uint64_t rowOffset =
            static_cast<uint64_t>(row) * rowWidth_;
        const uint64_t unionOffset =
            static_cast<uint64_t>(request) * scratchCapacity_;
        auto input = inputBuf_.Get<int32_t>();
        auto ranks = rankBuf_.Get<int32_t>();
        auto clamped = clampedBuf_.Get<int32_t>();
        auto unionSlots = unionBuf_.Get<int32_t>();
        auto selectedMask = selectedMaskBuf_.Get<uint8_t>();
        AscendC::DataCopy(
            input, topkIndices_[rowOffset], rowWidth_);
        AscendC::DataCopy(
            ranks, positionToUnion_[rowOffset], rowWidth_);
        AscendC::DataCopy(
            unionSlots, unionToSlot_[unionOffset], scratchCapacity_);
        Sync<AscendC::HardEvent::MTE2_V>();

        AscendC::Maxs(
            clamped, ranks, static_cast<int32_t>(0), rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            selectedMask,
            clamped,
            ranks,
            AscendC::CMPMODE::EQ,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(
            clamped,
            clamped,
            static_cast<int32_t>(scratchCapacity_ - 1),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            clamped,
            clamped,
            static_cast<int32_t>(sizeof(int32_t)),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(
            ranks,
            unionSlots,
            clamped.ReinterpretCast<uint32_t>(),
            static_cast<uint32_t>(0),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            clamped.ReinterpretCast<float>(),
            selectedMask,
            ranks.ReinterpretCast<float>(),
            input.ReinterpretCast<float>(),
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(
            topkIndices_[rowOffset], clamped, rowWidth_);
    }

private:
    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> positionToUnion_;
    AscendC::GlobalTensor<int32_t> unionToSlot_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rankBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> clampedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> unionBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t rowsPerRequest_ = 0;
    uint32_t scratchCapacity_ = 0;
};

// Build the flattened int64 indices consumed by the native NPU gather. One
// AIV owns one complete request row; padding is redirected to that request's
// private sentinel token, so no two requests touch the same cacheline.
class DSAResidentLookupRowsKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestStateIndices,
        __gm__ int32_t* lookupIndexWords,
        uint32_t requestCount,
        uint32_t scratchCapacity,
        uint32_t selectedCountStride,
        uint32_t tokenStride,
        uint32_t dummyStateBase)
    {
        requestCount_ = requestCount;
        scratchCapacity_ = scratchCapacity;
        selectedCountStride_ = selectedCountStride;
        tokenStride_ = tokenStride;
        dummyStateBase_ = dummyStateBase;
        selectedPacked_.SetGlobalBuffer(
            selectedPacked,
            static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        selectedCount_.SetGlobalBuffer(
            selectedCount,
            static_cast<uint64_t>(requestCount_) * selectedCountStride_);
        requestStateIndices_.SetGlobalBuffer(
            requestStateIndices, requestCount_);
        lookupIndexWords_.SetGlobalBuffer(
            lookupIndexWords,
            2 * static_cast<uint64_t>(requestCount_) * scratchCapacity_);
        pipe_.InitBuffer(tokenBuf_, scratchCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(positionBuf_, scratchCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(indexBuf_, scratchCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(index64Buf_, scratchCapacity_ * sizeof(int64_t));
        pipe_.InitBuffer(validMaskBuf_, scratchCapacity_ / 8);
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        const uint64_t offset =
            static_cast<uint64_t>(request) * scratchCapacity_;
        auto tokens = tokenBuf_.Get<int32_t>();
        auto positions = positionBuf_.Get<int32_t>();
        auto indices = indexBuf_.Get<int32_t>();
        auto indices64 = index64Buf_.Get<int64_t>();
        auto validMask = validMaskBuf_.Get<uint8_t>();
        AscendC::DataCopy(
            tokens, selectedPacked_[offset], scratchCapacity_);
        Sync<AscendC::HardEvent::MTE2_V>();

        int32_t count = selectedCount_.GetValue(
            static_cast<uint64_t>(request) * selectedCountStride_);
        if (count < 0) {
            count = 0;
        } else if (count > static_cast<int32_t>(scratchCapacity_)) {
            count = static_cast<int32_t>(scratchCapacity_);
        }
        const int32_t state = requestStateIndices_.GetValue(request);
        const bool validState =
            state >= 0 &&
            state < static_cast<int32_t>(dummyStateBase_);
        const uint32_t safeState = validState
            ? static_cast<uint32_t>(state)
            : dummyStateBase_ + request;
        const int32_t base =
            static_cast<int32_t>(safeState * tokenStride_);
        const int32_t sentinel =
            base + static_cast<int32_t>(tokenStride_ - 1);

        AscendC::CreateVecIndex(
            positions, static_cast<int32_t>(0), scratchCapacity_);
        AscendC::Duplicate(indices, count, scratchCapacity_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            validMask,
            positions,
            indices,
            AscendC::CMPMODE::LT,
            scratchCapacity_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(indices, tokens, base, scratchCapacity_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(
            positions, sentinel, scratchCapacity_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            indices.ReinterpretCast<float>(),
            validMask,
            indices.ReinterpretCast<float>(),
            positions.ReinterpretCast<float>(),
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            scratchCapacity_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            indices64,
            indices,
            AscendC::RoundMode::CAST_NONE,
            scratchCapacity_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            lookupIndexWords_[2 * offset],
            indices64.ReinterpretCast<int32_t>(),
            2 * scratchCapacity_);
    }

private:
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestStateIndices_;
    AscendC::GlobalTensor<int32_t> lookupIndexWords_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> positionBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> index64Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> validMaskBuf_;
    uint32_t requestCount_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t tokenStride_ = 0;
    uint32_t dummyStateBase_ = 0;
};

// Fuse row-local residency validation, free-slot assignment, miss compaction,
// forward-map publication, and LMCache target construction. The remaining
// loops are three linear passes over request-local UB tensors; the 130K
// reverse lookup remains a native gather/scatter in Python. One AIV owns
// every output row of one request.
class DSAResidentFinalizeRowsKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* targetSlotWords,
        __gm__ int32_t* requestBlockTable,
        __gm__ int32_t* requestStateIndices,
        __gm__ int64_t* requestStateGenerations,
        __gm__ int16_t* oldSlots,
        __gm__ int32_t* slotToToken,
        __gm__ int64_t* stateGenerations,
        __gm__ int32_t* unionToSlot,
        __gm__ int32_t* reverseIndexWords,
        __gm__ int16_t* reverseValues,
        uint32_t requestCount,
        uint32_t scratchCapacity,
        uint32_t selectedCountStride,
        uint32_t blockTableWidth,
        uint32_t tokenStride,
        uint32_t slotStride,
        uint32_t generationStride,
        uint32_t dummyStateBase,
        uint32_t blockSize)
    {
        requestCount_ = requestCount;
        scratchCapacity_ = scratchCapacity;
        selectedCountStride_ = selectedCountStride;
        blockTableWidth_ = blockTableWidth;
        tokenStride_ = tokenStride;
        slotStride_ = slotStride;
        generationStride_ = generationStride;
        dummyStateBase_ = dummyStateBase;
        blockSize_ = blockSize;
        const uint64_t requestElements =
            static_cast<uint64_t>(requestCount_) * scratchCapacity_;
        selectedPacked_.SetGlobalBuffer(selectedPacked, requestElements);
        selectedCount_.SetGlobalBuffer(
            selectedCount,
            static_cast<uint64_t>(requestCount_) * selectedCountStride_);
        targetSlotWords_.SetGlobalBuffer(
            targetSlotWords, 2 * requestElements);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable,
            static_cast<uint64_t>(requestCount_) * blockTableWidth_);
        requestStateIndices_.SetGlobalBuffer(
            requestStateIndices, requestCount_);
        requestStateGenerations_.SetGlobalBuffer(
            requestStateGenerations, requestCount_);
        oldSlots_.SetGlobalBuffer(oldSlots, requestElements);
        slotToToken_.SetGlobalBuffer(
            slotToToken,
            static_cast<uint64_t>(2 * dummyStateBase_) * slotStride_);
        stateGenerations_.SetGlobalBuffer(
            stateGenerations,
            static_cast<uint64_t>(2 * dummyStateBase_) * generationStride_);
        unionToSlot_.SetGlobalBuffer(unionToSlot, requestElements);
        reverseIndexWords_.SetGlobalBuffer(
            reverseIndexWords, 2 * requestElements);
        reverseValues_.SetGlobalBuffer(reverseValues, requestElements);

        pipe_.InitBuffer(tokenBuf_, scratchCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(forwardBuf_, scratchCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(mapBuf_, scratchCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(freeBuf_, scratchCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(oldSlotBuf_, scratchCapacity_ * sizeof(int16_t));
        pipe_.InitBuffer(targetBuf_, scratchCapacity_ * sizeof(int64_t));
        pipe_.InitBuffer(indexBuf_, scratchCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(
            blockTableBuf_, blockTableWidth_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        const uint64_t offset =
            static_cast<uint64_t>(request) * scratchCapacity_;
        auto tokens = tokenBuf_.Get<int32_t>();
        auto forward = forwardBuf_.Get<int32_t>();
        auto mapping = mapBuf_.Get<int32_t>();
        auto freeSlots = freeBuf_.Get<int32_t>();
        auto oldSlots = oldSlotBuf_.Get<int16_t>();
        auto targets = targetBuf_.Get<int64_t>();
        auto reverseIndices = indexBuf_.Get<int32_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();

        int32_t count = selectedCount_.GetValue(
            static_cast<uint64_t>(request) * selectedCountStride_);
        if (count < 0) {
            count = 0;
        } else if (count > static_cast<int32_t>(scratchCapacity_)) {
            count = static_cast<int32_t>(scratchCapacity_);
        }
        const int32_t state = requestStateIndices_.GetValue(request);
        const bool validState =
            state >= 0 &&
            state < static_cast<int32_t>(dummyStateBase_);
        const uint32_t safeState = validState
            ? static_cast<uint32_t>(state)
            : dummyStateBase_ + request;
        const int64_t requestedGeneration =
            requestStateGenerations_.GetValue(request);
        const uint64_t generationOffset =
            static_cast<uint64_t>(safeState) * generationStride_;
        const bool generationMatches =
            validState &&
            stateGenerations_.GetValue(generationOffset) ==
                requestedGeneration;
        const uint64_t forwardOffset =
            static_cast<uint64_t>(safeState) * slotStride_;
        const int32_t tokenBase =
            static_cast<int32_t>(safeState * tokenStride_);
        const int32_t reverseSentinel =
            tokenBase + static_cast<int32_t>(tokenStride_ - 1);

        AscendC::DataCopy(
            tokens, selectedPacked_[offset], scratchCapacity_);
        AscendC::DataCopy(
            oldSlots, oldSlots_[offset], scratchCapacity_);
        CopyGlobalToLocalExact(
            blockTable,
            requestBlockTable_[
                static_cast<uint64_t>(request) * blockTableWidth_],
            blockTableWidth_);
        if (generationMatches) {
            AscendC::DataCopy(
                forward, slotToToken_[forwardOffset], scratchCapacity_);
        } else {
            AscendC::Duplicate(
                forward, static_cast<int32_t>(-1), scratchCapacity_);
        }
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Duplicate(
            mapping, static_cast<int32_t>(-1), scratchCapacity_);
        AscendC::Duplicate(
            reverseIndices, reverseSentinel, scratchCapacity_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::MTE2_S>();
        Sync<AscendC::HardEvent::V_S>();

        // Mark protected slots. LocalTensor does not expose an arbitrary
        // scatter primitive in the supported AscendC API, so the fused
        // implementation retains a request-local scalar phase: mark hits,
        // scan free slots, then assign/compact misses.
        for (int32_t i = 0; i < count; ++i) {
            const int32_t candidate =
                static_cast<int32_t>(oldSlots.GetValue(i));
            const bool hit =
                candidate >= 0 &&
                candidate < static_cast<int32_t>(scratchCapacity_) &&
                forward.GetValue(candidate) == tokens.GetValue(i);
            if (hit) {
                mapping.SetValue(candidate, 1);
            }
        }

        uint32_t freeCount = 0;
        for (uint32_t slot = 0; slot < scratchCapacity_; ++slot) {
            if (mapping.GetValue(slot) < 0) {
                freeSlots.SetValue(freeCount++, static_cast<int32_t>(slot));
            }
        }

        uint32_t missCount = 0;
        for (uint32_t i = 0; i < scratchCapacity_; ++i) {
            const int32_t candidate =
                static_cast<int32_t>(oldSlots.GetValue(i));
            oldSlots.SetValue(i, static_cast<int16_t>(-1));
            if (i >= static_cast<uint32_t>(count)) {
                mapping.SetValue(i, static_cast<int32_t>(-1));
                continue;
            }
            const int32_t token = tokens.GetValue(i);
            const bool hit =
                candidate >= 0 &&
                candidate < static_cast<int32_t>(scratchCapacity_) &&
                forward.GetValue(candidate) == token;
            if (hit) {
                mapping.SetValue(i, candidate);
                continue;
            }
            const int32_t slot = freeSlots.GetValue(missCount);
            mapping.SetValue(i, slot);
            freeSlots.SetValue(missCount, token);
            forward.SetValue(slot, token);
            reverseIndices.SetValue(i, tokenBase + token);
            oldSlots.SetValue(i, static_cast<int16_t>(slot));
            const int32_t logicalBlock =
                slot / static_cast<int32_t>(blockSize_);
            const int32_t physicalBlock =
                blockTable.GetValue(logicalBlock);
            targets.SetValue(
                missCount,
                static_cast<int64_t>(physicalBlock) * blockSize_ +
                    slot % static_cast<int32_t>(blockSize_));
            ++missCount;
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        CopyLocalToGlobalExact(
            selectedPacked_[offset], freeSlots, missCount);
        CopyLocalToGlobalExact(
            targetSlotWords_[2 * offset],
            targets.ReinterpretCast<int32_t>(),
            2 * missCount);
        AscendC::DataCopy(
            slotToToken_[forwardOffset], forward, scratchCapacity_);
        AscendC::DataCopy(
            unionToSlot_[offset], mapping, scratchCapacity_);
        AscendC::DataCopy(
            reverseValues_[offset], oldSlots, scratchCapacity_);
        Sync<AscendC::HardEvent::MTE3_V>();
        Sync<AscendC::HardEvent::S_V>();
        AscendC::Cast(
            targets,
            reverseIndices,
            AscendC::RoundMode::CAST_NONE,
            scratchCapacity_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            reverseIndexWords_[2 * offset],
            targets.ReinterpretCast<int32_t>(),
            2 * scratchCapacity_);
        selectedCount_.SetValue(
            static_cast<uint64_t>(request) * selectedCountStride_,
            static_cast<int32_t>(missCount));
        stateGenerations_.SetValue(
            generationOffset, requestedGeneration);
    }

private:
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> targetSlotWords_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int32_t> requestStateIndices_;
    AscendC::GlobalTensor<int64_t> requestStateGenerations_;
    AscendC::GlobalTensor<int16_t> oldSlots_;
    AscendC::GlobalTensor<int32_t> slotToToken_;
    AscendC::GlobalTensor<int64_t> stateGenerations_;
    AscendC::GlobalTensor<int32_t> unionToSlot_;
    AscendC::GlobalTensor<int32_t> reverseIndexWords_;
    AscendC::GlobalTensor<int16_t> reverseValues_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> forwardBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mapBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> freeBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oldSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockTableBuf_;
    uint32_t requestCount_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t tokenStride_ = 0;
    uint32_t slotStride_ = 0;
    uint32_t generationStride_ = 0;
    uint32_t dummyStateBase_ = 0;
    uint32_t blockSize_ = 0;
};

class DSAStagedUniqueFinalizeKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* uniqueKeys,
        __gm__ int32_t* inverseWords,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* localToUnion,
        __gm__ int32_t* selectedCount,
        __gm__ int32_t* requestBlockTable,
        __gm__ int32_t* targetSlotWords,
        uint32_t uniqueCount,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t requestCount,
        uint32_t scratchCapacity,
        uint32_t blockTableWidth,
        uint32_t selectedCountStride,
        uint32_t blockSize,
        uint32_t blockSizeShift,
        uint32_t packedKeyStride)
    {
        uniqueCount_ = uniqueCount;
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        requestCount_ = requestCount;
        scratchCapacity_ = scratchCapacity;
        blockTableWidth_ = blockTableWidth;
        selectedCountStride_ = selectedCountStride;
        blockSize_ = blockSize;
        blockSizeShift_ = blockSizeShift;
        packedKeyStride_ = packedKeyStride;
        uniqueKeys_.SetGlobalBuffer(uniqueKeys, uniqueCount);
        inverseWords_.SetGlobalBuffer(
            inverseWords,
            2 * static_cast<uint64_t>(rowCount) * rowWidth);
        rowReqIndices_.SetGlobalBuffer(rowReqIndices, rowCount);
        selectedPacked_.SetGlobalBuffer(
            selectedPacked,
            static_cast<uint64_t>(requestCount) * scratchCapacity);
        localToUnion_.SetGlobalBuffer(
            localToUnion,
            static_cast<uint64_t>(rowCount) * rowWidth);
        selectedCount_.SetGlobalBuffer(
            selectedCount,
            static_cast<uint64_t>(requestCount) * selectedCountStride);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable,
            static_cast<uint64_t>(requestCount) * blockTableWidth);
        targetSlotWords_.SetGlobalBuffer(
            targetSlotWords,
            2 * static_cast<uint64_t>(requestCount) * scratchCapacity);
        pipe_.InitBuffer(keyBuf_, scratchCapacity * sizeof(int32_t));
        pipe_.InitBuffer(work0Buf_, scratchCapacity * sizeof(int32_t));
        pipe_.InitBuffer(work1Buf_, scratchCapacity * sizeof(int32_t));
        pipe_.InitBuffer(
            physicalBlockBuf_, scratchCapacity * sizeof(int32_t));
        pipe_.InitBuffer(
            targetBuf_, scratchCapacity * sizeof(int64_t));
        pipe_.InitBuffer(
            blockTableBuf_, blockTableWidth * sizeof(int32_t));
        pipe_.InitBuffer(inverseBuf_, rowWidth * sizeof(int64_t));
        pipe_.InitBuffer(mapBuf_, rowWidth * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        if (block < requestCount_) {
            FinalizeRequest(block);
            return;
        }
        const uint32_t row = block - requestCount_;
        if (row < rowCount_) {
            FinalizeRow(row);
        }
    }

private:
    __aicore__ inline uint32_t LowerBound(int32_t key) const
    {
        uint32_t first = 0;
        uint32_t last = uniqueCount_;
        while (first < last) {
            const uint32_t middle = first + (last - first) / 2;
            if (uniqueKeys_.GetValue(middle) < key) {
                first = middle + 1;
            } else {
                last = middle;
            }
        }
        return first;
    }

    __aicore__ inline void FinalizeRequest(uint32_t request)
    {
        const int32_t keyBase =
            static_cast<int32_t>(request * packedKeyStride_);
        const uint32_t begin = LowerBound(keyBase);
        const uint32_t end = LowerBound(
            keyBase + static_cast<int32_t>(packedKeyStride_));
        const uint32_t count = end - begin;
        const uint64_t outputOffset =
            static_cast<uint64_t>(request) * scratchCapacity_;

        auto keys = keyBuf_.Get<int32_t>();
        auto ranks = work0Buf_.Get<int32_t>();
        auto logicalBlocks = work1Buf_.Get<int32_t>();
        auto physicalBlocks = physicalBlockBuf_.Get<int32_t>();
        auto targets = targetBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();
        CopyGlobalToLocalExact(keys, uniqueKeys_[begin], count);
        CopyGlobalToLocalExact(
            blockTable,
            requestBlockTable_[
                static_cast<uint64_t>(request) * blockTableWidth_],
            blockTableWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Adds(keys, keys, -keyBase, count);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            selectedPacked_[outputOffset], keys, count);

        AscendC::CreateVecIndex(
            ranks, static_cast<int32_t>(0), count);
        AscendC::ShiftRight(
            logicalBlocks,
            ranks,
            static_cast<int32_t>(blockSizeShift_),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::MTE3_V>();
        AscendC::Muls(
            keys,
            logicalBlocks,
            static_cast<int32_t>(sizeof(int32_t)),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(
            physicalBlocks,
            blockTable,
            keys.ReinterpretCast<uint32_t>(),
            static_cast<uint32_t>(0),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            physicalBlocks,
            physicalBlocks,
            static_cast<int32_t>(blockSize_),
            count);
        AscendC::Muls(
            logicalBlocks,
            logicalBlocks,
            static_cast<int32_t>(blockSize_),
            count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(ranks, ranks, logicalBlocks, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(
            physicalBlocks, physicalBlocks, ranks, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            targets,
            physicalBlocks,
            AscendC::RoundMode::CAST_NONE,
            count);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            targetSlotWords_[2 * outputOffset],
            targets.ReinterpretCast<int32_t>(),
            2 * count);
        selectedCount_.SetValue(
            static_cast<uint64_t>(request) * selectedCountStride_,
            static_cast<int32_t>(count));
    }

    __aicore__ inline void FinalizeRow(uint32_t row)
    {
        const int32_t request = rowReqIndices_.GetValue(row);
        const uint32_t begin = LowerBound(
            request * static_cast<int32_t>(packedKeyStride_));
        const uint64_t rowOffset = static_cast<uint64_t>(row) * rowWidth_;
        auto inverseWords = inverseBuf_.Get<int32_t>();
        auto mapping = mapBuf_.Get<int32_t>();
        AscendC::DataCopy(
            inverseWords,
            inverseWords_[2 * rowOffset],
            2 * rowWidth_);
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(
            mapping,
            inverseWords.ReinterpretCast<int64_t>(),
            AscendC::RoundMode::CAST_NONE,
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(
            mapping,
            mapping,
            -static_cast<int32_t>(begin),
            rowWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(localToUnion_[rowOffset], mapping, rowWidth_);
    }

    AscendC::GlobalTensor<int32_t> uniqueKeys_;
    AscendC::GlobalTensor<int32_t> inverseWords_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> selectedPacked_;
    AscendC::GlobalTensor<int32_t> localToUnion_;
    AscendC::GlobalTensor<int32_t> selectedCount_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int32_t> targetSlotWords_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> keyBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> physicalBlockBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockTableBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inverseBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mapBuf_;
    uint32_t uniqueCount_ = 0;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestCount_ = 0;
    uint32_t scratchCapacity_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t selectedCountStride_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t blockSizeShift_ = 0;
    uint32_t packedKeyStride_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void dsa_staged_compact_rows_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* rowPacked,
    __gm__ int32_t* rowCounts,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t scratchCapacity,
    uint32_t coreCount,
    bool clearInvalidRows)
{
    if ASCEND_IS_AIV {
        DSAStagedCompactRowsKernel op;
        op.Init(
            topkIndices, splitBoundary, rowReqIndices, rowPacked, rowCounts,
            rowCount, rowWidth, requestCount, rowsPerRequest,
            scratchCapacity, coreCount, clearInvalidRows);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_production_sort_union_kernel(
    __gm__ int32_t* rowPacked,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* localToUnion,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    __gm__ int32_t* rowCounts,
    uint32_t requestCount,
    uint32_t rowWidth,
    uint32_t scratchCapacity,
    uint32_t blockTableWidth,
    uint32_t selectedCountStride,
    uint32_t blockSize,
    bool needPacked)
{
    if ASCEND_IS_AIV {
        DSAStagedSortUnionKernel op;
        op.InitProduction(
            rowPacked, selectedPacked, localToUnion, selectedCount,
            requestBlockTable, targetSlots, rowCounts, requestCount,
            rowWidth, scratchCapacity, blockTableWidth,
            selectedCountStride, blockSize, needPacked);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_single_row_finalize_kernel(
    __gm__ int32_t* rowCounts,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    uint32_t requestCount,
    uint32_t rowWidth,
    uint32_t scratchCapacity,
    uint32_t blockTableWidth,
    uint32_t selectedCountStride,
    uint32_t blockSize,
    bool needPacked)
{
    if ASCEND_IS_AIV {
        DSAStagedSingleRowFinalizeKernel op;
        op.Init(
            rowCounts, selectedCount, requestBlockTable, targetSlots,
            requestCount, rowWidth, scratchCapacity, blockTableWidth,
            selectedCountStride, blockSize, needPacked);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_boundary_remap_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* localToUnion,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t rowsPerRequest,
    uint32_t scratchCapacity,
    uint32_t coreCount)
{
    if ASCEND_IS_AIV {
        DSAStagedBoundaryRemapKernel op;
        op.Init(
            topkIndices, splitBoundary, rowReqIndices, localToUnion,
            rowCount, rowWidth, rowsPerRequest, scratchCapacity,
            coreCount);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_hash_union_kernel(
    __gm__ int32_t* rowPacked,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* localToUnion,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t maxTokens,
    uint32_t blockTableWidth,
    uint32_t selectedCountStride,
    uint32_t blockSize)
{
    if ASCEND_IS_AIV {
        DSAStagedHashUnionKernel op;
        op.Init(rowPacked, selectedPacked, localToUnion, selectedCount,
                requestBlockTable, targetSlots, rowCount, rowWidth,
                maxTokens, blockTableWidth, selectedCountStride, blockSize);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_sort_union_kernel(
    __gm__ int32_t* rowPacked,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* localToUnion,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t blockTableWidth,
    uint32_t selectedCountStride,
    uint32_t blockSize)
{
    if ASCEND_IS_AIV {
        DSAStagedSortUnionKernel op;
        op.Init(rowPacked, selectedPacked, localToUnion, selectedCount,
                requestBlockTable, targetSlots, rowCount, rowWidth,
                blockTableWidth, selectedCountStride, blockSize);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_sharded_sort_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* shardPacked,
    __gm__ int32_t* shardMapping,
    __gm__ int32_t* shardCounts,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t shardCountStride)
{
    if ASCEND_IS_AIV {
        DSAStagedShardedSortKernel op;
        op.Init(
            topkIndices, splitBoundary, topkIndices, shardPacked,
            shardMapping, shardCounts, requestCount, rowsPerRequest,
            rowWidth, rowWidth, shardCount, shardCountStride,
            shardCount * shardCountStride, false);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_production_sharded_sort_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* shardPacked,
    __gm__ int32_t* shardMapping,
    __gm__ int32_t* shardCounts,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride)
{
    if ASCEND_IS_AIV {
        DSAStagedShardedSortKernel op;
        op.Init(
            topkIndices, splitBoundary, rowReqIndices, shardPacked,
            shardMapping, shardCounts, requestCount, rowsPerRequest,
            rowWidth, rowsPerRequest * rowWidth, shardCount, shardCountStride,
            shardCountRequestStride, true);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_sharded_vector_stage_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* shardPacked,
    __gm__ int32_t* shardMapping,
    __gm__ int32_t* shardCounts,
    __gm__ int32_t* shardPairs,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t shardCountStride)
{
    if ASCEND_IS_AIV {
        DSAStagedShardedVectorStageKernel op;
        op.Init(
            topkIndices, splitBoundary, shardPacked, shardMapping,
            shardCounts, shardPairs, requestCount, rowsPerRequest,
            rowWidth, shardCount, shardCountStride, false);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_sharded_vector_dedup_stage_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* shardPacked,
    __gm__ int32_t* shardMapping,
    __gm__ int32_t* shardCounts,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t shardCountStride)
{
    if ASCEND_IS_AIV {
        DSAStagedShardedVectorStageKernel op;
        op.Init(
            topkIndices, splitBoundary, shardPacked, shardMapping,
            shardCounts, shardMapping, requestCount, rowsPerRequest,
            rowWidth, shardCount, shardCountStride, true);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_sharded_parallel_map_kernel(
    __gm__ int32_t* shardPairs,
    __gm__ int32_t* shardMapping,
    __gm__ int32_t* shardCounts,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t shardCountStride,
    uint32_t mappingParts)
{
    if ASCEND_IS_AIV {
        DSAStagedShardedParallelMapKernel op;
        op.Init(
            shardPairs, shardMapping, shardCounts, requestCount,
            rowsPerRequest, rowWidth, shardCount, shardCountStride,
            mappingParts);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_sharded_prefix_kernel(
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    __gm__ int32_t* shardPacked,
    __gm__ int32_t* shardCounts,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t blockTableWidth,
    uint32_t selectedCountStride,
    uint32_t shardCountStride,
    uint32_t blockSize)
{
    if ASCEND_IS_AIV {
        DSAStagedShardedPrefixKernel op;
        op.Init(
            selectedPacked, selectedCount, requestBlockTable, targetSlots,
            shardPacked, shardCounts, requestCount, rowsPerRequest,
            rowWidth, shardCount, blockTableWidth, selectedCountStride,
            shardCountStride, blockSize);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_sharded_position_map_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* localToUnion,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* shardMapping,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t selectedCountStride)
{
    if ASCEND_IS_AIV {
        DSAStagedShardedPositionMapKernel op;
        op.Init(
            topkIndices, localToUnion, selectedCount, shardMapping,
            requestCount, rowsPerRequest, rowWidth, shardCount,
            selectedCountStride);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_sharded_finalize_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* localToUnion,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    __gm__ int32_t* shardPacked,
    __gm__ int32_t* shardMapping,
    __gm__ int32_t* shardCounts,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t blockTableWidth,
    uint32_t selectedCountStride,
    uint32_t shardCountStride,
    uint32_t blockSize)
{
    if ASCEND_IS_AIV {
        DSAStagedShardedFinalizeKernel op;
        op.Init(
            topkIndices, topkIndices, selectedPacked, localToUnion,
            selectedCount, requestBlockTable, targetSlots, shardPacked,
            shardMapping, shardCounts, requestCount, rowsPerRequest,
            rowWidth, rowWidth, shardCount, blockTableWidth,
            selectedCountStride,
            shardCountStride, shardCount * shardCountStride, blockSize,
            false, false, true);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void
dsa_staged_production_sharded_finalize_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* localToUnion,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int64_t* targetSlots,
    __gm__ int32_t* shardPacked,
    __gm__ int32_t* shardMapping,
    __gm__ int32_t* shardCounts,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t blockTableWidth,
    uint32_t selectedCountStride,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t blockSize,
    bool needPacked,
    bool clearInvalidRows)
{
    if ASCEND_IS_AIV {
        DSAStagedShardedFinalizeKernel op;
        op.Init(
            topkIndices, rowReqIndices, selectedPacked, localToUnion,
            selectedCount, requestBlockTable, targetSlots, shardPacked,
            shardMapping, shardCounts, requestCount, rowsPerRequest,
            rowWidth, rowsPerRequest * rowWidth, shardCount,
            blockTableWidth, selectedCountStride,
            shardCountStride, shardCountRequestStride, blockSize, true,
            clearInvalidRows, needPacked);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_remap_rows_kernel(
    __gm__ int32_t* localIndices,
    __gm__ int32_t* localToUnion,
    uint32_t rowCount,
    uint32_t rowWidth)
{
    if ASCEND_IS_AIV {
        DSAStagedRemapRowsKernel op;
        op.Init(localIndices, localToUnion, rowCount, rowWidth);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_resident_remap_rows_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* positionToUnion,
    __gm__ int32_t* unionToSlot,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t rowsPerRequest,
    uint32_t scratchCapacity)
{
    if ASCEND_IS_AIV {
        DSAResidentRemapRowsKernel op;
        op.Init(
            topkIndices, positionToUnion, unionToSlot, rowCount,
            rowWidth, rowsPerRequest, scratchCapacity);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_resident_lookup_rows_kernel(
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestStateIndices,
    __gm__ int32_t* lookupIndexWords,
    uint32_t requestCount,
    uint32_t scratchCapacity,
    uint32_t selectedCountStride,
    uint32_t tokenStride,
    uint32_t dummyStateBase)
{
    if ASCEND_IS_AIV {
        DSAResidentLookupRowsKernel op;
        op.Init(
            selectedPacked, selectedCount, requestStateIndices,
            lookupIndexWords, requestCount, scratchCapacity,
            selectedCountStride, tokenStride, dummyStateBase);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_resident_finalize_rows_kernel(
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* targetSlotWords,
    __gm__ int32_t* requestBlockTable,
    __gm__ int32_t* requestStateIndices,
    __gm__ int64_t* requestStateGenerations,
    __gm__ int16_t* oldSlots,
    __gm__ int32_t* slotToToken,
    __gm__ int64_t* stateGenerations,
    __gm__ int32_t* unionToSlot,
    __gm__ int32_t* reverseIndexWords,
    __gm__ int16_t* reverseValues,
    uint32_t requestCount,
    uint32_t scratchCapacity,
    uint32_t selectedCountStride,
    uint32_t blockTableWidth,
    uint32_t tokenStride,
    uint32_t slotStride,
    uint32_t generationStride,
    uint32_t dummyStateBase,
    uint32_t blockSize)
{
    if ASCEND_IS_AIV {
        DSAResidentFinalizeRowsKernel op;
        op.Init(
            selectedPacked, selectedCount, targetSlotWords,
            requestBlockTable, requestStateIndices,
            requestStateGenerations, oldSlots, slotToToken,
            stateGenerations, unionToSlot, reverseIndexWords,
            reverseValues, requestCount, scratchCapacity,
            selectedCountStride, blockTableWidth, tokenStride,
            slotStride, generationStride, dummyStateBase, blockSize);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_unique_finalize_kernel(
    __gm__ int32_t* uniqueKeys,
    __gm__ int32_t* inverseWords,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* localToUnion,
    __gm__ int32_t* selectedCount,
    __gm__ int32_t* requestBlockTable,
    __gm__ int32_t* targetSlotWords,
    uint32_t uniqueCount,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t requestCount,
    uint32_t scratchCapacity,
    uint32_t blockTableWidth,
    uint32_t selectedCountStride,
    uint32_t blockSize,
    uint32_t blockSizeShift,
    uint32_t packedKeyStride)
{
    if ASCEND_IS_AIV {
        DSAStagedUniqueFinalizeKernel op;
        op.Init(
            uniqueKeys, inverseWords, rowReqIndices, selectedPacked,
            localToUnion, selectedCount, requestBlockTable, targetSlotWords,
            uniqueCount, rowCount, rowWidth, requestCount,
            scratchCapacity, blockTableWidth, selectedCountStride,
            blockSize, blockSizeShift, packedKeyStride);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dsa_staged_copy_rows_kernel(
    __gm__ int32_t* output,
    __gm__ int32_t* localIndices,
    uint32_t rowCount,
    uint32_t rowWidth)
{
    if ASCEND_IS_AIV {
        const uint32_t row = AscendC::GetBlockIdx();
        if (row >= rowCount) {
            return;
        }
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> rowBuf;
        pipe.InitBuffer(rowBuf, rowWidth * sizeof(int32_t));
        auto rowLocal = rowBuf.Get<int32_t>();
        AscendC::GlobalTensor<int32_t> outputGm;
        AscendC::GlobalTensor<int32_t> localIndicesGm;
        const uint64_t total =
            static_cast<uint64_t>(rowCount) * rowWidth;
        outputGm.SetGlobalBuffer(output, total);
        localIndicesGm.SetGlobalBuffer(localIndices, total);
        const uint64_t offset = static_cast<uint64_t>(row) * rowWidth;
        AscendC::DataCopy(rowLocal, localIndicesGm[offset], rowWidth);
        Sync<AscendC::HardEvent::MTE2_MTE3>();
        AscendC::DataCopy(outputGm[offset], rowLocal, rowWidth);
    }
}

namespace vllm_ascend {

static void dsa_prepare_sparse_indices_single_row_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* rowReqIndices, void* requestBlockTable,
    void* selectedPacked, void* selectedCount, void* targetSlots,
    void* localToUnion, uint32_t requestCount, uint32_t rowWidth,
    uint32_t scratchCapacity, uint32_t blockTableWidth,
    uint32_t selectedCountStride, uint32_t blockSize,
    uint32_t coreCount, bool needPacked, bool clearInvalidRows)
{
    dsa_staged_compact_rows_kernel<<<coreCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        requestCount, rowWidth, requestCount, 1,
        scratchCapacity, coreCount, clearInvalidRows);
    dsa_staged_single_row_finalize_kernel<<<
        requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        requestCount, rowWidth, scratchCapacity,
        blockTableWidth, selectedCountStride, blockSize,
        needPacked);
}

void dsa_prepare_sparse_indices_staged_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* rowReqIndices, void* requestBlockTable,
    void* selectedPacked, void* selectedCount, void* targetSlots,
    void* localToUnion, uint32_t rowCount, uint32_t rowWidth,
    uint32_t requestCount, uint32_t rowsPerRequest,
    uint32_t scratchCapacity, uint32_t blockTableWidth,
    uint32_t selectedCountStride, uint32_t blockSize,
    uint32_t coreCount, bool needPacked, bool clearInvalidRows)
{
    if (rowsPerRequest == 1) {
        dsa_prepare_sparse_indices_single_row_impl(
            stream, topkIndices, splitBoundary, rowReqIndices,
            requestBlockTable, selectedPacked, selectedCount, targetSlots,
            localToUnion, requestCount, rowWidth, scratchCapacity,
            blockTableWidth, selectedCountStride, blockSize, coreCount,
            needPacked, clearInvalidRows);
        return;
    }

    dsa_staged_compact_rows_kernel<<<coreCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        rowCount, rowWidth, requestCount, rowsPerRequest,
        scratchCapacity, coreCount, clearInvalidRows);

    dsa_staged_production_sort_union_kernel<<<
        requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(localToUnion),
        requestCount, rowWidth, scratchCapacity,
        blockTableWidth, selectedCountStride, blockSize,
        needPacked);
    dsa_staged_boundary_remap_kernel<<<coreCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(localToUnion),
        rowCount, rowWidth, rowsPerRequest, scratchCapacity,
        coreCount);
}

void dsa_prepare_sparse_indices_sharded_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* rowReqIndices, void* requestBlockTable,
    void* selectedPacked, void* selectedCount, void* targetSlots,
    void* localToUnion, void* shardPacked, void* shardMapping,
    void* shardCounts, uint32_t requestCount,
    uint32_t rowsPerRequest, uint32_t rowWidth,
    uint32_t scratchCapacity, uint32_t blockTableWidth,
    uint32_t selectedCountStride, uint32_t blockSize,
    uint32_t coreCount, bool needPacked, bool clearInvalidRows)
{
    if (rowsPerRequest == 1) {
        // MTP=1 top-k is unique by contract. Keep the public production
        // operator uniform, but bypass all shard partition/sort/union work.
        dsa_prepare_sparse_indices_single_row_impl(
            stream, topkIndices, splitBoundary, rowReqIndices,
            requestBlockTable, selectedPacked, selectedCount, targetSlots,
            localToUnion, requestCount, rowWidth, scratchCapacity,
            blockTableWidth, selectedCountStride, blockSize, coreCount,
            needPacked, clearInvalidRows);
        return;
    }

    (void)scratchCapacity;
    const uint32_t shardCount = rowsPerRequest;
    dsa_staged_production_sharded_sort_kernel<<<
        shardCount * requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(shardPacked),
        static_cast<int32_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        requestCount, rowsPerRequest, rowWidth, shardCount,
        kInt32PerCacheline, shardCount * kInt32PerCacheline);
    dsa_staged_production_sharded_finalize_kernel<<<
        requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(shardPacked),
        static_cast<int32_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        requestCount, rowsPerRequest, rowWidth, shardCount,
        blockTableWidth, selectedCountStride, kInt32PerCacheline,
        shardCount * kInt32PerCacheline, blockSize, needPacked,
        clearInvalidRows);
}

void dsa_staged_hash_union_impl(
    void* stream, void* rowPacked, void* selectedPacked,
    void* localToUnion, void* selectedCount, void* requestBlockTable,
    void* targetSlots, uint32_t rowCount, uint32_t rowWidth,
    uint32_t maxTokens, uint32_t blockTableWidth,
    uint32_t selectedCountStride, uint32_t blockSize)
{
    dsa_staged_hash_union_kernel<<<rowCount / 2, nullptr, stream>>>(
        static_cast<int32_t*>(rowPacked),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        rowCount, rowWidth, maxTokens, blockTableWidth,
        selectedCountStride, blockSize);
}

void dsa_staged_sort_union_impl(
    void* stream, void* rowPacked, void* selectedPacked,
    void* localToUnion, void* selectedCount, void* requestBlockTable,
    void* targetSlots, uint32_t rowCount, uint32_t rowWidth,
    uint32_t blockTableWidth, uint32_t selectedCountStride,
    uint32_t blockSize)
{
    dsa_staged_sort_union_kernel<<<rowCount / 2, nullptr, stream>>>(
        static_cast<int32_t*>(rowPacked),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        rowCount, rowWidth, blockTableWidth, selectedCountStride,
        blockSize);
}

void dsa_staged_sharded_sort_union_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* selectedPacked,
    void* localToUnion, void* selectedCount, void* requestBlockTable,
    void* targetSlots, void* shardPacked, void* shardMapping,
    void* shardCounts, uint32_t requestCount, uint32_t rowsPerRequest,
    uint32_t rowWidth, uint32_t shardCount,
    uint32_t blockTableWidth, uint32_t selectedCountStride,
    uint32_t shardCountStride, uint32_t blockSize)
{
    dsa_staged_sharded_sort_kernel<<<shardCount * requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(shardPacked),
        static_cast<int32_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        requestCount, rowsPerRequest, rowWidth, shardCount,
        shardCountStride);
    dsa_staged_sharded_finalize_kernel<<<requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(shardPacked),
        static_cast<int32_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        requestCount, rowsPerRequest, rowWidth, shardCount,
        blockTableWidth, selectedCountStride, shardCountStride,
        blockSize);
}

void dsa_staged_sharded_vector_union_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* selectedPacked,
    void* localToUnion, void* selectedCount, void* requestBlockTable,
    void* targetSlots, void* shardPacked, void* shardMapping,
    void* shardCounts, void* shardPairs,
    uint32_t requestCount, uint32_t rowsPerRequest,
    uint32_t rowWidth, uint32_t shardCount,
    uint32_t blockTableWidth, uint32_t selectedCountStride,
    uint32_t shardCountStride, uint32_t blockSize)
{
    dsa_staged_sharded_vector_stage_kernel<<<
        shardCount * requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(shardPacked),
        static_cast<int32_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        static_cast<int32_t*>(shardPairs),
        requestCount, rowsPerRequest, rowWidth, shardCount,
        shardCountStride);
    if (rowsPerRequest > 1) {
        const uint32_t mappingParts = shardCount;
        dsa_staged_sharded_parallel_map_kernel<<<
            requestCount * shardCount * mappingParts, nullptr, stream>>>(
            static_cast<int32_t*>(shardPairs),
            static_cast<int32_t*>(shardMapping),
            static_cast<int32_t*>(shardCounts),
            requestCount, rowsPerRequest, rowWidth, shardCount,
            shardCountStride, mappingParts);
    }
    dsa_staged_sharded_finalize_kernel<<<requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(shardPacked),
        static_cast<int32_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        requestCount, rowsPerRequest, rowWidth, shardCount,
        blockTableWidth, selectedCountStride, shardCountStride,
        blockSize);
}

void dsa_staged_sharded_vector_dedup_impl(
    void* stream, void* topkIndices, void* splitBoundary,
    void* selectedPacked,
    void* localToUnion, void* selectedCount, void* requestBlockTable,
    void* targetSlots, void* shardPacked, void* shardMapping,
    void* shardCounts, uint32_t requestCount, uint32_t rowsPerRequest,
    uint32_t rowWidth, uint32_t shardCount,
    uint32_t blockTableWidth, uint32_t selectedCountStride,
    uint32_t shardCountStride, uint32_t blockSize)
{
    dsa_staged_sharded_vector_dedup_stage_kernel<<<
        shardCount * requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(shardPacked),
        static_cast<int32_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        requestCount, rowsPerRequest, rowWidth, shardCount,
        shardCountStride);
    if (rowsPerRequest == 1) {
        // A single top-k row is unique by contract and its only shard has a
        // zero prefix. Reuse the two-launch finalizer fast path.
        dsa_staged_sharded_finalize_kernel<<<requestCount, nullptr, stream>>>(
            static_cast<int32_t*>(topkIndices),
            static_cast<int32_t*>(selectedPacked),
            static_cast<int32_t*>(localToUnion),
            static_cast<int32_t*>(selectedCount),
            static_cast<int32_t*>(requestBlockTable),
            static_cast<int64_t*>(targetSlots),
            static_cast<int32_t*>(shardPacked),
            static_cast<int32_t*>(shardMapping),
            static_cast<int32_t*>(shardCounts),
            requestCount, rowsPerRequest, rowWidth, shardCount,
            blockTableWidth, selectedCountStride, shardCountStride,
            blockSize);
        return;
    }

    dsa_staged_sharded_prefix_kernel<<<requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(shardPacked),
        static_cast<int32_t*>(shardCounts),
        requestCount, rowsPerRequest, rowWidth, shardCount,
        blockTableWidth, selectedCountStride, shardCountStride,
        blockSize);
    dsa_staged_sharded_position_map_kernel<<<
        requestCount * shardCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(shardMapping),
        requestCount, rowsPerRequest, rowWidth, shardCount,
        selectedCountStride);
}

void dsa_staged_remap_rows_impl(
    void* stream, void* localIndices, void* localToUnion,
    uint32_t rowCount, uint32_t rowWidth)
{
    dsa_staged_remap_rows_kernel<<<rowCount, nullptr, stream>>>(
        static_cast<int32_t*>(localIndices),
        static_cast<int32_t*>(localToUnion), rowCount, rowWidth);
}

void dsa_resident_remap_rows_impl(
    void* stream, void* topkIndices, void* positionToUnion,
    void* unionToSlot, uint32_t rowCount, uint32_t rowWidth,
    uint32_t rowsPerRequest, uint32_t scratchCapacity)
{
    dsa_resident_remap_rows_kernel<<<rowCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(positionToUnion),
        static_cast<int32_t*>(unionToSlot),
        rowCount, rowWidth, rowsPerRequest, scratchCapacity);
}

void dsa_resident_lookup_rows_impl(
    void* stream, void* selectedPacked, void* selectedCount,
    void* requestStateIndices, void* lookupIndices,
    uint32_t requestCount, uint32_t scratchCapacity,
    uint32_t selectedCountStride, uint32_t tokenStride,
    uint32_t dummyStateBase)
{
    dsa_resident_lookup_rows_kernel<<<requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestStateIndices),
        static_cast<int32_t*>(lookupIndices),
        requestCount, scratchCapacity, selectedCountStride,
        tokenStride, dummyStateBase);
}

void dsa_resident_finalize_rows_impl(
    void* stream, void* selectedPacked, void* selectedCount,
    void* targetSlots, void* requestBlockTable,
    void* requestStateIndices, void* requestStateGenerations,
    void* oldSlots, void* slotToToken, void* stateGenerations,
    void* unionToSlot, void* reverseIndices, void* reverseValues,
    uint32_t requestCount, uint32_t scratchCapacity,
    uint32_t selectedCountStride, uint32_t blockTableWidth,
    uint32_t tokenStride, uint32_t slotStride,
    uint32_t generationStride, uint32_t dummyStateBase,
    uint32_t blockSize)
{
    dsa_resident_finalize_rows_kernel<<<requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(targetSlots),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int32_t*>(requestStateIndices),
        static_cast<int64_t*>(requestStateGenerations),
        static_cast<int16_t*>(oldSlots),
        static_cast<int32_t*>(slotToToken),
        static_cast<int64_t*>(stateGenerations),
        static_cast<int32_t*>(unionToSlot),
        static_cast<int32_t*>(reverseIndices),
        static_cast<int16_t*>(reverseValues),
        requestCount, scratchCapacity, selectedCountStride,
        blockTableWidth, tokenStride, slotStride, generationStride,
        dummyStateBase, blockSize);
}

void dsa_staged_unique_finalize_impl(
    void* stream, void* uniqueKeys, void* inverse, void* rowReqIndices,
    void* selectedPacked, void* localToUnion, void* selectedCount,
    void* requestBlockTable, void* targetSlots, uint32_t uniqueCount,
    uint32_t rowCount, uint32_t rowWidth, uint32_t requestCount,
    uint32_t scratchCapacity, uint32_t blockTableWidth,
    uint32_t selectedCountStride, uint32_t blockSize,
    uint32_t blockSizeShift,
    uint32_t packedKeyStride)
{
    dsa_staged_unique_finalize_kernel<<<requestCount + rowCount, nullptr, stream>>>(
        static_cast<int32_t*>(uniqueKeys),
        static_cast<int32_t*>(inverse),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(localToUnion),
        static_cast<int32_t*>(selectedCount),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int32_t*>(targetSlots),
        uniqueCount, rowCount, rowWidth, requestCount, scratchCapacity,
        blockTableWidth, selectedCountStride, blockSize, blockSizeShift,
        packedKeyStride);
}

void dsa_staged_copy_rows_impl(
    void* stream, void* output, void* localIndices,
    uint32_t rowCount, uint32_t rowWidth)
{
    dsa_staged_copy_rows_kernel<<<rowCount, nullptr, stream>>>(
        static_cast<int32_t*>(output),
        static_cast<int32_t*>(localIndices), rowCount, rowWidth);
}

}  // namespace vllm_ascend
