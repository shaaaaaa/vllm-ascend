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

constexpr uint32_t kUbAlignBytes = 32;
constexpr uint32_t kCumSumTileWidth = 512;
constexpr uint32_t kCumSumTransposeRows = 16;
constexpr uint32_t kCumSumWorkspaceBytes =
    2 * kCumSumTransposeRows * kCumSumTileWidth * sizeof(float);
constexpr AscendC::CumSumConfig kCumSumConfig{true, false, false};

// A complete source row is the ownership unit. validRows must be unique, and
// every row is at least one 256-byte transaction, so two AIVs never write the
// same transaction while top-k indices are updated in place.
class DSAPrepareSparseIndicesKernel {
public:
    __aicore__ inline DSAPrepareSparseIndicesKernel() = default;

    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* validRows,
        __gm__ int32_t* scratchBase,
        __gm__ int32_t* selectedPacked,
        __gm__ int32_t* rowReqIndices,
        uint32_t rowCount,
        uint32_t rowWidth,
        uint32_t validRowCount,
        uint32_t coreCount,
        uint32_t needPacked,
        uint32_t clearInvalidRows)
    {
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        validRowCount_ = validRowCount;
        coreCount_ = coreCount;
        needPacked_ = needPacked != 0;
        clearInvalidRows_ = clearInvalidRows != 0;

        topkIndicesGm_.SetGlobalBuffer(topkIndices);
        splitBoundaryGm_.SetGlobalBuffer(splitBoundary);
        validRowsGm_.SetGlobalBuffer(validRows, validRowCount);
        scratchBaseGm_.SetGlobalBuffer(scratchBase);
        if (needPacked_) {
            selectedPackedGm_.SetGlobalBuffer(
                selectedPacked,
                static_cast<uint64_t>(validRowCount) * rowWidth);
        }
        if (clearInvalidRows_) {
            rowReqIndicesGm_.SetGlobalBuffer(rowReqIndices, rowCount);
        }

        const uint32_t rowBytes = rowWidth * sizeof(int32_t);
        const uint32_t maskBytes = rowWidth / 8;
        const uint32_t maskBufferBytes =
            (maskBytes + kUbAlignBytes - 1) / kUbAlignBytes * kUbAlignBytes;
        pipe_.InitBuffer(inputBuf_, rowBytes);
        pipe_.InitBuffer(clampedInputBuf_, rowBytes);
        if (needPacked_) {
            pipe_.InitBuffer(packedBuf_, rowBytes);
        }
        pipe_.InitBuffer(selectionFlagsBuf_, rowBytes);
        pipe_.InitBuffer(prefixRanksBuf_, rowBytes);
        pipe_.InitBuffer(cumSumWorkspaceBuf_, kCumSumWorkspaceBytes);
        pipe_.InitBuffer(nonNegativeMaskBuf_, maskBufferBytes);
        pipe_.InitBuffer(beforeBoundaryMaskBuf_, maskBufferBytes);
        pipe_.InitBuffer(selectedMaskBuf_, maskBufferBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        for (uint32_t packedRow = coreIdx; packedRow < validRowCount_;
             packedRow += coreCount_) {
            ProcessRow(packedRow);
        }
        if (clearInvalidRows_) {
            for (uint32_t sourceRow = coreIdx; sourceRow < rowCount_;
                 sourceRow += coreCount_) {
                if (rowReqIndicesGm_.GetValue(sourceRow) < 0) {
                    ZeroPaddingRow(sourceRow);
                }
            }
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t packedRow)
    {
        using namespace AscendC;

        const int32_t sourceRow = validRowsGm_.GetValue(packedRow);
        const int32_t splitBoundary = splitBoundaryGm_.GetValue(sourceRow);
        const int32_t base = scratchBaseGm_.GetValue(sourceRow);
        const uint64_t sourceOffset =
            static_cast<uint64_t>(sourceRow) * rowWidth_;
        const uint64_t packedOffset =
            static_cast<uint64_t>(packedRow) * rowWidth_;

        LocalTensor<int32_t> input = inputBuf_.Get<int32_t>();
        LocalTensor<int32_t> clampedInput = clampedInputBuf_.Get<int32_t>();
        LocalTensor<float> selectionFlags = selectionFlagsBuf_.Get<float>();
        LocalTensor<float> prefixRanks = prefixRanksBuf_.Get<float>();
        LocalTensor<uint8_t> cumSumWorkspace =
            cumSumWorkspaceBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> nonNegativeMask =
            nonNegativeMaskBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> beforeBoundaryMask =
            beforeBoundaryMaskBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> selectedMask = selectedMaskBuf_.Get<uint8_t>();

        DataCopy(input, topkIndicesGm_[sourceOffset], rowWidth_);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        // A2/A3 only support EQ for int32 comparisons. Clamp first and then
        // compare for equality so token positions remain exact above 2^24.
        Maxs(clampedInput, input, static_cast<int32_t>(0), rowWidth_);
        PipeBarrier<PIPE_V>();
        Compare(
            nonNegativeMask,
            clampedInput,
            input,
            CMPMODE::EQ,
            rowWidth_);
        PipeBarrier<PIPE_V>();
        Mins(
            clampedInput,
            input,
            splitBoundary - 1,
            rowWidth_);
        PipeBarrier<PIPE_V>();
        Compare(
            beforeBoundaryMask,
            clampedInput,
            input,
            CMPMODE::EQ,
            rowWidth_);
        PipeBarrier<PIPE_V>();

        // Compare masks are bit-packed. rowWidth is a multiple of 64, so the
        // source rows contain complete 256-byte vector repeats and the mask
        // contains complete uint16_t words.
        And(
            selectedMask.ReinterpretCast<uint16_t>(),
            nonNegativeMask.ReinterpretCast<uint16_t>(),
            beforeBoundaryMask.ReinterpretCast<uint16_t>(),
            rowWidth_ / 16);
        PipeBarrier<PIPE_V>();

        if (needPacked_) {
            LocalTensor<int32_t> packed = packedBuf_.Get<int32_t>();
            Duplicate(packed, static_cast<int32_t>(0), rowWidth_);
        }
        PipeBarrier<PIPE_V>();

        GatherMaskParams gatherParams;
        gatherParams.repeatTimes = 1;
        gatherParams.src0BlockStride = 1;
        gatherParams.src0RepeatStride = 8;
        gatherParams.src1RepeatStride = 8;

        if (needPacked_) {
            LocalTensor<int32_t> packed = packedBuf_.Get<int32_t>();
            uint64_t selectedCount = 0;
            GatherMask(
                packed,
                input,
                selectedMask.ReinterpretCast<uint32_t>(),
                true,
                rowWidth_,
                gatherParams,
                selectedCount);
            PipeBarrier<PIPE_V>();
        }

        // CumSum supports float on A2/A3. Its input is only a 0/1 selection
        // flag, and rowWidth <= 4096, so every accumulated rank is exact.
        Duplicate(prefixRanks, 1.0F, rowWidth_);
        PipeBarrier<PIPE_V>();
        Select(
            selectionFlags,
            selectedMask,
            prefixRanks,
            0.0F,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            rowWidth_);
        PipeBarrier<PIPE_V>();

        // CumSum along [1, K] needs 2 * 16 * K float bytes of temporary
        // storage. A whole K=2048/4096 row does not fit in UB, so scan fixed
        // tiles and carry the final rank into the next tile. This keeps the
        // per-element work on the vector pipeline while bounding workspace at
        // 64 KiB for every supported row width.
        LocalTensor<float> lastRow = clampedInput.ReinterpretCast<float>();
        float carry = 0.0F;
        for (uint32_t tileOffset = 0; tileOffset < rowWidth_;
             tileOffset += kCumSumTileWidth) {
            const uint32_t remaining = rowWidth_ - tileOffset;
            const uint32_t tileWidth =
                remaining < kCumSumTileWidth ? remaining : kCumSumTileWidth;
            const CumSumInfo cumSumInfo{1, tileWidth};
            LocalTensor<float> tilePrefix = prefixRanks[tileOffset];
            LocalTensor<float> tileFlags = selectionFlags[tileOffset];
            CumSum<float, kCumSumConfig>(
                tilePrefix,
                lastRow,
                tileFlags,
                cumSumWorkspace,
                cumSumInfo);
            PipeBarrier<PIPE_V>();

            if (tileOffset != 0) {
                SetFlag<HardEvent::S_V>(0);
                WaitFlag<HardEvent::S_V>(0);
                Adds(
                    tilePrefix,
                    tilePrefix,
                    carry,
                    tileWidth);
                PipeBarrier<PIPE_V>();
            }

            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            carry = prefixRanks.GetValue(tileOffset + tileWidth - 1);
        }

        // Convert only the small exact ranks to int32. The final float Select
        // is a bitwise 32-bit mux over reinterpreted int32 tensors; token
        // positions (including values above 2^24 and negative padding) are
        // never numerically converted to float.
        Cast(clampedInput, prefixRanks, RoundMode::CAST_ROUND, rowWidth_);
        PipeBarrier<PIPE_V>();
        Adds(clampedInput, clampedInput, base - 1, rowWidth_);
        PipeBarrier<PIPE_V>();
        Select(
            input.ReinterpretCast<float>(),
            selectedMask,
            clampedInput.ReinterpretCast<float>(),
            input.ReinterpretCast<float>(),
            SELMODE::VSEL_TENSOR_TENSOR_MODE,
            rowWidth_);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopy(topkIndicesGm_[sourceOffset], input, rowWidth_);
        if (needPacked_) {
            LocalTensor<int32_t> packed = packedBuf_.Get<int32_t>();
            DataCopy(selectedPackedGm_[packedOffset], packed, rowWidth_);
        }
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void ZeroPaddingRow(uint32_t sourceRow)
    {
        using namespace AscendC;

        // Wait for this core's preceding row write before reusing inputBuf_.
        // Complete rows remain the ownership unit, so no other AIV writes the
        // same 256-byte transaction.
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);

        LocalTensor<int32_t> input = inputBuf_.Get<int32_t>();
        Duplicate(input, static_cast<int32_t>(0), rowWidth_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(
            topkIndicesGm_[static_cast<uint64_t>(sourceRow) * rowWidth_],
            input,
            rowWidth_);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> clampedInputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectionFlagsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prefixRanksBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cumSumWorkspaceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> nonNegativeMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> beforeBoundaryMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;

    AscendC::GlobalTensor<int32_t> topkIndicesGm_;
    AscendC::GlobalTensor<int32_t> splitBoundaryGm_;
    AscendC::GlobalTensor<int32_t> validRowsGm_;
    AscendC::GlobalTensor<int32_t> scratchBaseGm_;
    AscendC::GlobalTensor<int32_t> selectedPackedGm_;
    AscendC::GlobalTensor<int32_t> rowReqIndicesGm_;

    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t validRowCount_ = 0;
    uint32_t coreCount_ = 0;
    bool needPacked_ = false;
    bool clearInvalidRows_ = false;
};

}  // namespace

extern "C" __global__ __aicore__ void dsa_prepare_sparse_indices_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* validRows,
    __gm__ int32_t* scratchBase,
    __gm__ int32_t* selectedPacked,
    __gm__ int32_t* rowReqIndices,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t validRowCount,
    uint32_t coreCount,
    uint32_t needPacked,
    uint32_t clearInvalidRows)
{
    DSAPrepareSparseIndicesKernel op;
    op.Init(
        topkIndices,
        splitBoundary,
        validRows,
        scratchBase,
        selectedPacked,
        rowReqIndices,
        rowCount,
        rowWidth,
        validRowCount,
        coreCount,
        needPacked,
        clearInvalidRows);
    op.Process();
}

namespace vllm_ascend {

void dsa_prepare_sparse_indices_impl(
    void* stream,
    void* topkIndices,
    void* splitBoundary,
    void* validRows,
    void* scratchBase,
    void* selectedPacked,
    void* rowReqIndices,
    uint32_t rowCount,
    uint32_t rowWidth,
    uint32_t validRowCount,
    uint32_t coreCount,
    bool needPacked,
    bool clearInvalidRows)
{
    dsa_prepare_sparse_indices_kernel<<<coreCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(validRows),
        static_cast<int32_t*>(scratchBase),
        static_cast<int32_t*>(selectedPacked),
        static_cast<int32_t*>(rowReqIndices),
        rowCount,
        rowWidth,
        validRowCount,
        coreCount,
        needPacked ? 1U : 0U,
        clearInvalidRows ? 1U : 0U);
}

}  // namespace vllm_ascend
