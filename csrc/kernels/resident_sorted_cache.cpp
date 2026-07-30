/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"

namespace {

constexpr uint32_t kDataBlockBytes = 32;
constexpr uint32_t kInt32PerDataBlock =
    kDataBlockBytes / sizeof(int32_t);
constexpr uint32_t kSortGroup = 32;
constexpr uint32_t kPairWidth = 2;
constexpr uint32_t kMergeWays = 4;

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
    const AscendC::DataCopyParams params{
        1,
        static_cast<uint16_t>(bytes),
        0,
        0};
    AscendC::DataCopyPad(dst, src, params);
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
    const AscendC::DataCopyParams params{
        1,
        static_cast<uint16_t>(bytes),
        0,
        0};
    AscendC::DataCopyPad(dst, src, params, {});
}

// This kernel is intentionally independent of the production staged-union
// kernels. One AIV owns one (request, token-value shard), so every output row
// and scalar count cacheline has exactly one writer.
//
// deduplicate=false is the MTP=1 path: the input row is unique by contract,
// but it must still be sorted for the resident intersection.
// deduplicate=true is the MTP=2 path: equal values from the two rows collapse
// to one shard-local union entry.
class DSAResidentShardedUnionKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* splitBoundary,
        __gm__ int32_t* rowReqIndices,
        __gm__ int32_t* shardPacked,
        __gm__ int16_t* shardMapping,
        __gm__ int32_t* shardCounts,
        __gm__ int32_t* requestStateIndices,
        __gm__ int64_t* requestStateGenerations,
        __gm__ int32_t* stateTokens,
        __gm__ int16_t* stateSlots,
        __gm__ int32_t* stateCounts,
        __gm__ int64_t* stateGenerations,
        __gm__ int16_t* priorSlots,
        uint32_t requestCount,
        uint32_t stateRowCount,
        uint32_t dummyStateBase,
        uint32_t rowsPerRequest,
        uint32_t rowWidth,
        uint32_t shardCount,
        uint32_t shardCapacity,
        uint32_t shardCountStride,
        uint32_t shardCountRequestStride,
        uint32_t generationStride)
    {
        requestCount_ = requestCount;
        stateRowCount_ = stateRowCount;
        dummyStateBase_ = dummyStateBase;
        rowsPerRequest_ = rowsPerRequest;
        rowWidth_ = rowWidth;
        requestWidth_ = rowsPerRequest_ * rowWidth_;
        shardCount_ = shardCount;
        shardCapacity_ = shardCapacity;
        shardCountStride_ = shardCountStride;
        shardCountRequestStride_ = shardCountRequestStride;
        generationStride_ = generationStride;
        deduplicate_ = rowsPerRequest_ > 1;
        uint32_t shifted = shardCount_;
        while (shifted > 1) {
            ++shardBits_;
            shifted >>= 1;
        }

        topkIndices_.SetGlobalBuffer(
            topkIndices,
            static_cast<uint64_t>(requestCount_) * requestWidth_);
        splitBoundary_.SetGlobalBuffer(
            splitBoundary,
            static_cast<uint64_t>(requestCount_) * rowsPerRequest_);
        rowReqIndices_.SetGlobalBuffer(
            rowReqIndices,
            static_cast<uint64_t>(requestCount_) * rowsPerRequest_);
        shardPacked_.SetGlobalBuffer(
            shardPacked,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * shardCapacity_);
        shardMapping_.SetGlobalBuffer(
            shardMapping,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * requestWidth_);
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            static_cast<uint64_t>(requestCount_)
                * shardCountRequestStride_);
        requestStateIndices_.SetGlobalBuffer(
            requestStateIndices, requestCount_);
        requestStateGenerations_.SetGlobalBuffer(
            requestStateGenerations, requestCount_);
        const uint64_t stateShardElements =
            static_cast<uint64_t>(stateRowCount_) * shardCount_
            * shardCapacity_;
        stateTokens_.SetGlobalBuffer(
            stateTokens, stateShardElements);
        stateSlots_.SetGlobalBuffer(
            stateSlots, stateShardElements);
        stateCounts_.SetGlobalBuffer(
            stateCounts,
            static_cast<uint64_t>(stateRowCount_)
                * shardCountRequestStride_);
        stateGenerations_.SetGlobalBuffer(
            stateGenerations,
            static_cast<uint64_t>(stateRowCount_)
                * generationStride_);
        priorSlots_.SetGlobalBuffer(
            priorSlots,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * shardCapacity_);

        const uint32_t rowBytes = rowWidth_ * sizeof(int32_t);
        const uint32_t requestBytes =
            requestWidth_ * sizeof(int32_t);
        const uint32_t pairBytes =
            shardCapacity_ * kPairWidth * sizeof(float);
        const uint32_t maskBytes = rowWidth_ / 8;
        pipe_.InitBuffer(inputBuf_, rowBytes);
        pipe_.InitBuffer(workBuf_, rowBytes);
        pipe_.InitBuffer(clampedBuf_, rowBytes);
        pipe_.InitBuffer(indexBuf_, rowBytes);
        pipe_.InitBuffer(compactTokenBuf_, requestBytes);
        pipe_.InitBuffer(compactIndexBuf_, requestBytes);
        pipe_.InitBuffer(sortSrcBuf_, pairBytes);
        pipe_.InitBuffer(sortTmpBuf_, pairBytes);
        pipe_.InitBuffer(sortedTokenBuf_, requestBytes);
        pipe_.InitBuffer(
            mappingBuf_, requestWidth_ * sizeof(int16_t));
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
            * shardCapacity_;
        const uint64_t mappingOffset =
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
        auto sortedTokens = sortedTokenBuf_.Get<int32_t>();
        auto mapping = mappingBuf_.Get<int16_t>();
        auto shardMask = shardMaskBuf_.Get<uint8_t>();
        auto nonNegativeMask = nonNegativeMaskBuf_.Get<uint8_t>();
        auto beforeBoundaryMask =
            beforeBoundaryMaskBuf_.Get<uint8_t>();
        auto selectedMask = selectedMaskBuf_.Get<uint8_t>();
        auto srcInt = src.ReinterpretCast<int32_t>();

        AscendC::Duplicate(
            compactTokens,
            static_cast<int32_t>(0x7FFFFFFF),
            requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::GatherMaskParams gatherParams;
        gatherParams.repeatTimes = 1;
        gatherParams.src0BlockStride = 1;
        gatherParams.src0RepeatStride = 8;
        gatherParams.src1RepeatStride = 8;
        uint32_t compactEnd = 0;
        uint32_t selectedElements = 0;
        for (uint32_t mtpRow = 0; mtpRow < rowsPerRequest_; ++mtpRow) {
            const uint32_t row = request * rowsPerRequest_ + mtpRow;
            if (rowReqIndices_.GetValue(row) !=
                static_cast<int32_t>(request)) {
                continue;
            }
            const uint64_t inputOffset =
                static_cast<uint64_t>(request) * requestWidth_
                + static_cast<uint64_t>(mtpRow) * rowWidth_;
            const int32_t boundary =
                splitBoundary_.GetValue(row);
            AscendC::DataCopy(
                input, topkIndices_[inputOffset], rowWidth_);
            Sync<AscendC::HardEvent::MTE2_V>();

            // shard = token % shardCount. shardCount is a host-validated
            // power of two, but vector shift/multiply keeps this path valid
            // for signed int32 input before the non-negative mask is applied.
            AscendC::ShiftRight(
                work, input, static_cast<int32_t>(shardBits_),
                rowWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                work, work, static_cast<int32_t>(shardCount_),
                rowWidth_);
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
            selectedElements +=
                static_cast<uint32_t>(tokenElements);
            compactEnd =
                compactOffset + static_cast<uint32_t>(tokenElements);
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
        AscendC::Duplicate(
            mapping,
            static_cast<int16_t>(-1),
            requestWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_S>();

        auto sortedInt = src.ReinterpretCast<int32_t>();
        uint32_t rank = 0;
        if (!deduplicate_) {
            // MTP=1 is unique by contract. Keep this as a separate loop so
            // the launch-time MTP choice does not add a branch per token.
            for (uint32_t i = 0; i < selectedElements; ++i) {
                const int32_t token =
                    -static_cast<int32_t>(
                        src.GetValue(kPairWidth * i));
                const uint32_t original = static_cast<uint32_t>(
                    sortedInt.GetValue(kPairWidth * i + 1));
                sortedTokens.SetValue(rank, token);
                mapping.SetValue(
                    original, static_cast<int16_t>(rank));
                ++rank;
            }
        } else {
            int32_t previous = -1;
            for (uint32_t i = 0; i < selectedElements; ++i) {
                const int32_t token =
                    -static_cast<int32_t>(
                        src.GetValue(kPairWidth * i));
                const uint32_t original = static_cast<uint32_t>(
                    sortedInt.GetValue(kPairWidth * i + 1));
                if (i == 0 || token != previous) {
                    sortedTokens.SetValue(rank, token);
                    previous = token;
                    ++rank;
                }
                mapping.SetValue(
                    original, static_cast<int16_t>(rank - 1));
            }
        }

        // The compact-index storage is dead after sorting. Reuse its two
        // int16 halves for old resident slots and intersection output; reuse
        // compact-token storage for old resident tokens. This fuses the old
        // intersection kernel without increasing peak UB.
        auto oldTokens = compactTokens;
        auto reusedInt16 = compactIndices.ReinterpretCast<int16_t>();
        auto oldSlots = reusedInt16;
        auto priorSlots = reusedInt16[shardCapacity_];
        const int32_t state =
            requestStateIndices_.GetValue(request);
        const bool realState =
            state >= 0 &&
            state < static_cast<int32_t>(dummyStateBase_);
        const uint32_t safeState = realState
            ? static_cast<uint32_t>(state)
            : dummyStateBase_ + request;
        const int64_t requestedGeneration =
            requestStateGenerations_.GetValue(request);
        const int64_t storedGeneration =
            stateGenerations_.GetValue(
                static_cast<uint64_t>(safeState)
                * generationStride_);
        const bool generationMatches =
            realState && storedGeneration == requestedGeneration;
        const uint64_t stateCountOffset =
            static_cast<uint64_t>(safeState)
                * shardCountRequestStride_
            + shard * shardCountStride_;
        const uint32_t oldCount = generationMatches
            ? static_cast<uint32_t>(
                  stateCounts_.GetValue(stateCountOffset))
            : 0U;
        const uint64_t stateShardOffset =
            (static_cast<uint64_t>(safeState) * shardCount_ + shard)
            * shardCapacity_;
        if (oldCount > 0) {
            CopyGlobalToLocalExact(
                oldTokens,
                stateTokens_[stateShardOffset],
                oldCount);
            CopyGlobalToLocalExact(
                oldSlots,
                stateSlots_[stateShardOffset],
                oldCount);
        }
        Sync<AscendC::HardEvent::MTE2_S>();

        uint32_t oldIndex = 0;
        for (uint32_t currentIndex = 0;
             currentIndex < rank;
             ++currentIndex) {
            const int32_t token =
                sortedTokens.GetValue(currentIndex);
            while (
                oldIndex < oldCount &&
                oldTokens.GetValue(oldIndex) < token) {
                ++oldIndex;
            }
            const int16_t slot =
                oldIndex < oldCount &&
                    oldTokens.GetValue(oldIndex) == token
                ? oldSlots.GetValue(oldIndex)
                : static_cast<int16_t>(-1);
            priorSlots.SetValue(currentIndex, slot);
        }
        if (!generationMatches) {
            stateCounts_.SetValue(stateCountOffset, 0);
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            shardPacked_[shardOffset], sortedTokens, rank);
        AscendC::DataCopy(
            shardMapping_[mappingOffset], mapping, requestWidth_);
        CopyLocalToGlobalExact(
            priorSlots_[shardOffset], priorSlots, rank);
        // Host validation reserves one full 64-byte int32 cacheline per
        // (request, shard), so sibling AIVs never share this write line.
        shardCounts_.SetValue(
            countOffset, static_cast<int32_t>(rank));
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
            groups =
                groups <= kMergeWays ? 1 : groups / kMergeWays;
            elements *= kMergeWays;
            ++pass;
        }
        if (pass % 2 == 0) {
            AscendC::DataCopy(
                src, tmp, sortElements * kPairWidth);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> splitBoundary_;
    AscendC::GlobalTensor<int32_t> rowReqIndices_;
    AscendC::GlobalTensor<int32_t> shardPacked_;
    AscendC::GlobalTensor<int16_t> shardMapping_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::GlobalTensor<int32_t> requestStateIndices_;
    AscendC::GlobalTensor<int64_t> requestStateGenerations_;
    AscendC::GlobalTensor<int32_t> stateTokens_;
    AscendC::GlobalTensor<int16_t> stateSlots_;
    AscendC::GlobalTensor<int32_t> stateCounts_;
    AscendC::GlobalTensor<int64_t> stateGenerations_;
    AscendC::GlobalTensor<int16_t> priorSlots_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> clampedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> compactTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> compactIndexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortSrcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortedTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mappingBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shardMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC>
        nonNegativeMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC>
        beforeBoundaryMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;
    uint32_t requestCount_ = 0;
    uint32_t stateRowCount_ = 0;
    uint32_t dummyStateBase_ = 0;
    uint32_t rowsPerRequest_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t shardCapacity_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t shardCountRequestStride_ = 0;
    uint32_t generationStride_ = 0;
    uint32_t shardBits_ = 0;
    bool deduplicate_ = false;
};

class DSAResidentSortedFinalizeKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* shardPacked,
        __gm__ int32_t* shardCounts,
        __gm__ int16_t* priorSlots,
        __gm__ uint8_t* overwrittenSlots,
        __gm__ int32_t* missTokens,
        __gm__ int32_t* missCounts,
        __gm__ int64_t* targetSlots,
        __gm__ int32_t* requestBlockTable,
        uint32_t requestCount,
        uint32_t shardCount,
        uint32_t capacity,
        uint32_t shardCountStride,
        uint32_t shardCountRequestStride,
        uint32_t missCountStride,
        uint32_t blockTableWidth,
        uint32_t blockSize)
    {
        requestCount_ = requestCount;
        shardCount_ = shardCount;
        capacity_ = capacity;
        shardCountStride_ = shardCountStride;
        shardCountRequestStride_ = shardCountRequestStride;
        missCountStride_ = missCountStride;
        blockTableWidth_ = blockTableWidth;
        blockSize_ = blockSize;
        const uint64_t requestShardElements =
            static_cast<uint64_t>(requestCount_) * shardCount_
            * capacity_;
        const uint64_t requestElements =
            static_cast<uint64_t>(requestCount_) * capacity_;
        shardPacked_.SetGlobalBuffer(
            shardPacked, requestShardElements);
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            static_cast<uint64_t>(requestCount_)
                * shardCountRequestStride_);
        priorSlots_.SetGlobalBuffer(
            priorSlots, requestShardElements);
        overwrittenSlots_.SetGlobalBuffer(
            overwrittenSlots, requestElements);
        missTokens_.SetGlobalBuffer(
            missTokens, requestElements);
        missCounts_.SetGlobalBuffer(
            missCounts,
            static_cast<uint64_t>(requestCount_)
                * missCountStride_);
        targetSlots_.SetGlobalBuffer(
            targetSlots, requestElements);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable,
            static_cast<uint64_t>(requestCount_)
                * blockTableWidth_);

        pipe_.InitBuffer(
            protectedBuf_, capacity_ * sizeof(int16_t));
        pipe_.InitBuffer(
            freeSlotBuf_, capacity_ * sizeof(int16_t));
        pipe_.InitBuffer(
            overwrittenBuf_, capacity_ * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        const uint64_t requestOffset =
            static_cast<uint64_t>(request) * capacity_;
        const uint64_t requestShardBase =
            static_cast<uint64_t>(request) * shardCount_ * capacity_;
        auto protectedSlots = protectedBuf_.Get<int16_t>();
        auto freeSlots = freeSlotBuf_.Get<int16_t>();
        auto overwritten = overwrittenBuf_.Get<uint8_t>();
        AscendC::Duplicate(
            protectedSlots, static_cast<int16_t>(0), capacity_);
        AscendC::Duplicate(
            overwritten.ReinterpretCast<int16_t>(),
            static_cast<int16_t>(0),
            capacity_ / sizeof(int16_t));
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_S>();

        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t count = static_cast<uint32_t>(
                shardCounts_.GetValue(
                    static_cast<uint64_t>(request)
                        * shardCountRequestStride_
                    + shard * shardCountStride_));
            const uint64_t shardOffset =
                requestShardBase
                + static_cast<uint64_t>(shard) * capacity_;
            for (uint32_t index = 0; index < count; ++index) {
                const int16_t slot =
                    priorSlots_.GetValue(shardOffset + index);
                if (slot >= 0) {
                    protectedSlots.SetValue(
                        static_cast<uint32_t>(slot),
                        static_cast<int16_t>(1));
                }
            }
        }

        uint32_t freeCount = 0;
        for (uint32_t slot = 0; slot < capacity_; ++slot) {
            if (protectedSlots.GetValue(slot) == 0) {
                freeSlots.SetValue(
                    freeCount++, static_cast<int16_t>(slot));
            }
        }

        uint32_t missCount = 0;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t count = static_cast<uint32_t>(
                shardCounts_.GetValue(
                    static_cast<uint64_t>(request)
                        * shardCountRequestStride_
                    + shard * shardCountStride_));
            const uint64_t shardOffset =
                requestShardBase
                + static_cast<uint64_t>(shard) * capacity_;
            for (uint32_t index = 0; index < count; ++index) {
                int16_t slot =
                    priorSlots_.GetValue(shardOffset + index);
                if (slot < 0) {
                    slot = freeSlots.GetValue(missCount);
                    const int32_t token =
                        shardPacked_.GetValue(shardOffset + index);
                    missTokens_.SetValue(
                        requestOffset + missCount, token);
                    const uint32_t logicalSlot =
                        static_cast<uint32_t>(slot);
                    const uint32_t logicalBlock =
                        logicalSlot / blockSize_;
                    const uint32_t blockOffset =
                        logicalSlot % blockSize_;
                    const int32_t physicalBlock =
                        requestBlockTable_.GetValue(
                            static_cast<uint64_t>(request)
                                * blockTableWidth_
                            + logicalBlock);
                    targetSlots_.SetValue(
                        requestOffset + missCount,
                        static_cast<int64_t>(physicalBlock)
                                * blockSize_
                            + blockOffset);
                    overwritten.SetValue(
                        logicalSlot, static_cast<uint8_t>(1));
                    ++missCount;
                }
                priorSlots_.SetValue(
                    shardOffset + index, slot);
            }
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        CopyLocalToGlobalExact(
            overwrittenSlots_[requestOffset],
            overwritten,
            capacity_);
        // One cacheline per request prevents count false sharing.
        missCounts_.SetValue(
            static_cast<uint64_t>(request) * missCountStride_,
            static_cast<int32_t>(missCount));
    }

private:
    AscendC::GlobalTensor<int32_t> shardPacked_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::GlobalTensor<int16_t> priorSlots_;
    AscendC::GlobalTensor<uint8_t> overwrittenSlots_;
    AscendC::GlobalTensor<int32_t> missTokens_;
    AscendC::GlobalTensor<int32_t> missCounts_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> protectedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> freeSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> overwrittenBuf_;
    uint32_t requestCount_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t capacity_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t shardCountRequestStride_ = 0;
    uint32_t missCountStride_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t blockSize_ = 0;
};

class DSAResidentSortedUpdateKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* shardPacked,
        __gm__ int16_t* shardMapping,
        __gm__ int32_t* shardCounts,
        __gm__ int16_t* priorSlots,
        __gm__ uint8_t* overwrittenSlots,
        __gm__ int32_t* requestStateIndices,
        __gm__ int64_t* requestStateGenerations,
        __gm__ int32_t* stateTokens,
        __gm__ int16_t* stateSlots,
        __gm__ int32_t* stateCounts,
        __gm__ int64_t* stateGenerations,
        uint32_t requestCount,
        uint32_t stateRowCount,
        uint32_t dummyStateBase,
        uint32_t rowsPerRequest,
        uint32_t rowWidth,
        uint32_t shardCount,
        uint32_t capacity,
        uint32_t shardCountStride,
        uint32_t shardCountRequestStride,
        uint32_t generationStride)
    {
        requestCount_ = requestCount;
        stateRowCount_ = stateRowCount;
        dummyStateBase_ = dummyStateBase;
        rowsPerRequest_ = rowsPerRequest;
        rowWidth_ = rowWidth;
        requestWidth_ = rowsPerRequest_ * rowWidth_;
        shardCount_ = shardCount;
        capacity_ = capacity;
        shardCountStride_ = shardCountStride;
        shardCountRequestStride_ = shardCountRequestStride;
        generationStride_ = generationStride;
        const uint64_t requestElements =
            static_cast<uint64_t>(requestCount_) * requestWidth_;
        const uint64_t requestShardElements =
            static_cast<uint64_t>(requestCount_) * shardCount_
            * capacity_;
        const uint64_t stateShardElements =
            static_cast<uint64_t>(stateRowCount_)
            * shardCount_ * capacity_;
        topkIndices_.SetGlobalBuffer(topkIndices, requestElements);
        shardPacked_.SetGlobalBuffer(
            shardPacked, requestShardElements);
        shardMapping_.SetGlobalBuffer(
            shardMapping,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * requestWidth_);
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            static_cast<uint64_t>(requestCount_)
                * shardCountRequestStride_);
        priorSlots_.SetGlobalBuffer(
            priorSlots, requestShardElements);
        overwrittenSlots_.SetGlobalBuffer(
            overwrittenSlots,
            static_cast<uint64_t>(requestCount_) * capacity_);
        requestStateIndices_.SetGlobalBuffer(
            requestStateIndices, requestCount_);
        requestStateGenerations_.SetGlobalBuffer(
            requestStateGenerations, requestCount_);
        stateTokens_.SetGlobalBuffer(
            stateTokens, stateShardElements);
        stateSlots_.SetGlobalBuffer(
            stateSlots, stateShardElements);
        stateCounts_.SetGlobalBuffer(
            stateCounts,
            static_cast<uint64_t>(stateRowCount_)
                * shardCountRequestStride_);
        stateGenerations_.SetGlobalBuffer(
            stateGenerations,
            static_cast<uint64_t>(stateRowCount_)
                * generationStride_);

        pipe_.InitBuffer(
            oldTokenBuf_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(
            oldSlotBuf_, capacity_ * sizeof(int16_t));
        pipe_.InitBuffer(
            currentTokenBuf_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(
            priorSlotBuf_, capacity_ * sizeof(int16_t));
        pipe_.InitBuffer(
            overwrittenBuf_, capacity_ * sizeof(uint8_t));
        pipe_.InitBuffer(
            survivorTokenBuf_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(
            survivorSlotBuf_, capacity_ * sizeof(int16_t));
        pipe_.InitBuffer(
            mergedTokenBuf_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(
            mergedSlotBuf_, capacity_ * sizeof(int16_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t request = block / shardCount_;
        const uint32_t shard = block % shardCount_;
        if (request >= requestCount_) {
            return;
        }
        const int32_t state =
            requestStateIndices_.GetValue(request);
        const bool realState =
            state >= 0 &&
            state < static_cast<int32_t>(dummyStateBase_);
        const uint32_t safeState = realState
            ? static_cast<uint32_t>(state)
            : dummyStateBase_ + request;
        const uint64_t requestCountOffset =
            static_cast<uint64_t>(request)
                * shardCountRequestStride_
            + shard * shardCountStride_;
        const int64_t requestedGeneration =
            requestStateGenerations_.GetValue(request);
        const uint64_t oldCountOffset =
            static_cast<uint64_t>(safeState)
                * shardCountRequestStride_
            + shard * shardCountStride_;
        const uint32_t oldCount = static_cast<uint32_t>(
            stateCounts_.GetValue(oldCountOffset));
        const uint32_t currentCount = static_cast<uint32_t>(
            shardCounts_.GetValue(requestCountOffset));
        const uint64_t requestShardBase =
            static_cast<uint64_t>(request) * shardCount_ * capacity_;
        const uint64_t requestShardOffset =
            requestShardBase
            + static_cast<uint64_t>(shard) * capacity_;
        const uint64_t oldStateOffset =
            (static_cast<uint64_t>(safeState) * shardCount_ + shard)
            * capacity_;

        auto oldTokens = oldTokenBuf_.Get<int32_t>();
        auto oldSlots = oldSlotBuf_.Get<int16_t>();
        auto currentTokens = currentTokenBuf_.Get<int32_t>();
        auto priorSlots = priorSlotBuf_.Get<int16_t>();
        auto overwritten = overwrittenBuf_.Get<uint8_t>();
        auto survivorTokens = survivorTokenBuf_.Get<int32_t>();
        auto survivorSlots = survivorSlotBuf_.Get<int16_t>();
        auto mergedTokens = mergedTokenBuf_.Get<int32_t>();
        auto mergedSlots = mergedSlotBuf_.Get<int16_t>();

        if (oldCount > 0) {
            CopyGlobalToLocalExact(
                oldTokens, stateTokens_[oldStateOffset], oldCount);
            CopyGlobalToLocalExact(
                oldSlots, stateSlots_[oldStateOffset], oldCount);
        }
        CopyGlobalToLocalExact(
            currentTokens,
            shardPacked_[requestShardOffset],
            currentCount);
        CopyGlobalToLocalExact(
            priorSlots,
            priorSlots_[requestShardOffset],
            currentCount);
        CopyGlobalToLocalExact(
            overwritten,
            overwrittenSlots_[
                static_cast<uint64_t>(request) * capacity_],
            capacity_);
        Sync<AscendC::HardEvent::MTE2_S>();

        uint32_t survivorCount = 0;
        for (uint32_t index = 0; index < oldCount; ++index) {
            const int16_t slot = oldSlots.GetValue(index);
            if (slot >= 0 &&
                overwritten.GetValue(
                    static_cast<uint32_t>(slot)) == 0) {
                survivorTokens.SetValue(
                    survivorCount, oldTokens.GetValue(index));
                survivorSlots.SetValue(survivorCount, slot);
                ++survivorCount;
            }
        }

        // Merge two already-sorted runs: surviving old residents and the
        // current misses. Hits are already present in the survivor run and
        // must not be inserted a second time.
        uint32_t oldIndex = 0;
        uint32_t currentIndex = 0;
        uint32_t mergedCount = 0;
        while (oldIndex < survivorCount ||
               currentIndex < currentCount) {
            while (currentIndex < currentCount) {
                const int16_t slot =
                    priorSlots.GetValue(currentIndex);
                if (slot >= 0 &&
                    overwritten.GetValue(
                        static_cast<uint32_t>(slot)) != 0) {
                    break;
                }
                ++currentIndex;
            }
            const bool haveOld = oldIndex < survivorCount;
            const bool haveMiss = currentIndex < currentCount;
            if (!haveOld && !haveMiss) {
                break;
            }
            const int32_t oldToken = haveOld
                ? survivorTokens.GetValue(oldIndex)
                : static_cast<int32_t>(0x7FFFFFFF);
            const int32_t missToken = haveMiss
                ? currentTokens.GetValue(currentIndex)
                : static_cast<int32_t>(0x7FFFFFFF);
            if (haveOld && (!haveMiss || oldToken < missToken)) {
                mergedTokens.SetValue(mergedCount, oldToken);
                mergedSlots.SetValue(
                    mergedCount,
                    survivorSlots.GetValue(oldIndex));
                ++oldIndex;
            } else {
                mergedTokens.SetValue(mergedCount, missToken);
                mergedSlots.SetValue(
                    mergedCount,
                    priorSlots.GetValue(currentIndex));
                ++currentIndex;
            }
            ++mergedCount;
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        CopyLocalToGlobalExact(
            stateTokens_[oldStateOffset],
            mergedTokens,
            mergedCount);
        CopyLocalToGlobalExact(
            stateSlots_[oldStateOffset],
            mergedSlots,
            mergedCount);
        const uint64_t newCountOffset =
            static_cast<uint64_t>(safeState)
                * shardCountRequestStride_
            + shard * shardCountStride_;
        stateCounts_.SetValue(
            newCountOffset, static_cast<int32_t>(mergedCount));

        RemapPositionPartition(
            request, shard, requestShardBase);

        // Every shard copied its complete old list to private UB before
        // overwriting its own disjoint GM range. Generation mismatches were
        // materialized as zero counts by the fused union kernel, so
        // sibling blocks do not re-read this generation publication.
        if (shard == 0) {
            stateGenerations_.SetValue(
                static_cast<uint64_t>(safeState)
                    * generationStride_,
                requestedGeneration);
        }
    }

private:
    __aicore__ inline void RemapPositionPartition(
        uint32_t request,
        uint32_t part,
        uint64_t requestShardBase)
    {
        const uint32_t partWidth = requestWidth_ / shardCount_;
        const uint32_t begin = part * partWidth;
        const uint32_t end = begin + partWidth;
        const uint64_t requestOffset =
            static_cast<uint64_t>(request) * requestWidth_;
        const uint64_t mappingBase =
            static_cast<uint64_t>(request) * shardCount_
            * requestWidth_;
        for (uint32_t position = begin; position < end; ++position) {
            const int32_t token =
                topkIndices_.GetValue(requestOffset + position);
            if (token < 0) {
                continue;
            }
            const uint32_t tokenShard =
                static_cast<uint32_t>(token)
                & (shardCount_ - 1U);
            const int16_t rank =
                shardMapping_.GetValue(
                    mappingBase
                    + static_cast<uint64_t>(tokenShard)
                        * requestWidth_
                    + position);
            if (rank >= 0) {
                const int16_t slot =
                    priorSlots_.GetValue(
                        requestShardBase
                        + static_cast<uint64_t>(tokenShard)
                            * capacity_
                        + static_cast<uint32_t>(rank));
                topkIndices_.SetValue(
                    requestOffset + position,
                    static_cast<int32_t>(slot));
            }
        }
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> shardPacked_;
    AscendC::GlobalTensor<int16_t> shardMapping_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::GlobalTensor<int16_t> priorSlots_;
    AscendC::GlobalTensor<uint8_t> overwrittenSlots_;
    AscendC::GlobalTensor<int32_t> requestStateIndices_;
    AscendC::GlobalTensor<int64_t> requestStateGenerations_;
    AscendC::GlobalTensor<int32_t> stateTokens_;
    AscendC::GlobalTensor<int16_t> stateSlots_;
    AscendC::GlobalTensor<int32_t> stateCounts_;
    AscendC::GlobalTensor<int64_t> stateGenerations_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oldTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oldSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> currentTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> priorSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> overwrittenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> survivorTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> survivorSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mergedTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mergedSlotBuf_;
    uint32_t requestCount_ = 0;
    uint32_t stateRowCount_ = 0;
    uint32_t dummyStateBase_ = 0;
    uint32_t rowsPerRequest_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t capacity_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t shardCountRequestStride_ = 0;
    uint32_t generationStride_ = 0;
};

extern "C" __global__ __aicore__ void
dsa_resident_sharded_union_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* splitBoundary,
    __gm__ int32_t* rowReqIndices,
    __gm__ int32_t* shardPacked,
    __gm__ int16_t* shardMapping,
    __gm__ int32_t* shardCounts,
    __gm__ int32_t* requestStateIndices,
    __gm__ int64_t* requestStateGenerations,
    __gm__ int32_t* stateTokens,
    __gm__ int16_t* stateSlots,
    __gm__ int32_t* stateCounts,
    __gm__ int64_t* stateGenerations,
    __gm__ int16_t* priorSlots,
    uint32_t requestCount,
    uint32_t stateRowCount,
    uint32_t dummyStateBase,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t shardCapacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t generationStride)
{
    DSAResidentShardedUnionKernel op;
    op.Init(
        topkIndices, splitBoundary, rowReqIndices,
        shardPacked, shardMapping, shardCounts,
        requestStateIndices, requestStateGenerations,
        stateTokens, stateSlots, stateCounts, stateGenerations,
        priorSlots, requestCount, stateRowCount, dummyStateBase,
        rowsPerRequest, rowWidth, shardCount, shardCapacity,
        shardCountStride, shardCountRequestStride, generationStride);
    op.Process();
}

extern "C" __global__ __aicore__ void
dsa_resident_sorted_finalize_kernel(
    __gm__ int32_t* shardPacked,
    __gm__ int32_t* shardCounts,
    __gm__ int16_t* priorSlots,
    __gm__ uint8_t* overwrittenSlots,
    __gm__ int32_t* missTokens,
    __gm__ int32_t* missCounts,
    __gm__ int64_t* targetSlots,
    __gm__ int32_t* requestBlockTable,
    uint32_t requestCount,
    uint32_t shardCount,
    uint32_t capacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t missCountStride,
    uint32_t blockTableWidth,
    uint32_t blockSize)
{
    DSAResidentSortedFinalizeKernel op;
    op.Init(
        shardPacked, shardCounts, priorSlots, overwrittenSlots,
        missTokens, missCounts, targetSlots,
        requestBlockTable, requestCount, shardCount, capacity,
        shardCountStride, shardCountRequestStride, missCountStride,
        blockTableWidth, blockSize);
    op.Process();
}

extern "C" __global__ __aicore__ void
dsa_resident_sorted_update_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* shardPacked,
    __gm__ int16_t* shardMapping,
    __gm__ int32_t* shardCounts,
    __gm__ int16_t* priorSlots,
    __gm__ uint8_t* overwrittenSlots,
    __gm__ int32_t* requestStateIndices,
    __gm__ int64_t* requestStateGenerations,
    __gm__ int32_t* stateTokens,
    __gm__ int16_t* stateSlots,
    __gm__ int32_t* stateCounts,
    __gm__ int64_t* stateGenerations,
    uint32_t requestCount,
    uint32_t stateRowCount,
    uint32_t dummyStateBase,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t capacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t generationStride)
{
    DSAResidentSortedUpdateKernel op;
    op.Init(
        topkIndices, shardPacked, shardMapping, shardCounts,
        priorSlots, overwrittenSlots,
        requestStateIndices, requestStateGenerations,
        stateTokens, stateSlots, stateCounts, stateGenerations,
        requestCount, stateRowCount,
        dummyStateBase, rowsPerRequest, rowWidth, shardCount,
        capacity, shardCountStride, shardCountRequestStride,
        generationStride);
    op.Process();
}

}  // namespace

namespace vllm_ascend {

void dsa_resident_sharded_union_impl(
    void* stream,
    void* topkIndices,
    void* splitBoundary,
    void* rowReqIndices,
    void* shardPacked,
    void* shardMapping,
    void* shardCounts,
    void* requestStateIndices,
    void* requestStateGenerations,
    void* stateTokens,
    void* stateSlots,
    void* stateCounts,
    void* stateGenerations,
    void* priorSlots,
    uint32_t requestCount,
    uint32_t stateRowCount,
    uint32_t dummyStateBase,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t shardCapacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t generationStride)
{
    dsa_resident_sharded_union_kernel<<<
        requestCount * shardCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(splitBoundary),
        static_cast<int32_t*>(rowReqIndices),
        static_cast<int32_t*>(shardPacked),
        static_cast<int16_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        static_cast<int32_t*>(requestStateIndices),
        static_cast<int64_t*>(requestStateGenerations),
        static_cast<int32_t*>(stateTokens),
        static_cast<int16_t*>(stateSlots),
        static_cast<int32_t*>(stateCounts),
        static_cast<int64_t*>(stateGenerations),
        static_cast<int16_t*>(priorSlots),
        requestCount, stateRowCount, dummyStateBase, rowsPerRequest,
        rowWidth, shardCount, shardCapacity, shardCountStride,
        shardCountRequestStride, generationStride);
}

void dsa_resident_sorted_plan_impl(
    void* stream,
    void* topkIndices,
    void* shardPacked,
    void* shardMapping,
    void* shardCounts,
    void* requestBlockTable,
    void* requestStateIndices,
    void* requestStateGenerations,
    void* stateTokens,
    void* stateSlots,
    void* stateCounts,
    void* stateGenerations,
    void* priorSlots,
    void* overwrittenSlots,
    void* missTokens,
    void* missCounts,
    void* targetSlots,
    uint32_t requestCount,
    uint32_t stateRowCount,
    uint32_t dummyStateBase,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t capacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t missCountStride,
    uint32_t generationStride,
    uint32_t blockTableWidth,
    uint32_t blockSize)
{
    dsa_resident_sorted_finalize_kernel<<<
        requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(shardPacked),
        static_cast<int32_t*>(shardCounts),
        static_cast<int16_t*>(priorSlots),
        static_cast<uint8_t*>(overwrittenSlots),
        static_cast<int32_t*>(missTokens),
        static_cast<int32_t*>(missCounts),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(requestBlockTable),
        requestCount, shardCount, capacity, shardCountStride,
        shardCountRequestStride, missCountStride,
        blockTableWidth, blockSize);
    dsa_resident_sorted_update_kernel<<<
        requestCount * shardCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(shardPacked),
        static_cast<int16_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        static_cast<int16_t*>(priorSlots),
        static_cast<uint8_t*>(overwrittenSlots),
        static_cast<int32_t*>(requestStateIndices),
        static_cast<int64_t*>(requestStateGenerations),
        static_cast<int32_t*>(stateTokens),
        static_cast<int16_t*>(stateSlots),
        static_cast<int32_t*>(stateCounts),
        static_cast<int64_t*>(stateGenerations),
        requestCount, stateRowCount, dummyStateBase,
        rowsPerRequest, rowWidth, shardCount, capacity,
        shardCountStride, shardCountRequestStride,
        generationStride);
}

}  // namespace vllm_ascend
