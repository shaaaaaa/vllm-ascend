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
constexpr uint32_t kInt16PerDataBlock =
    kDataBlockBytes / sizeof(int16_t);
constexpr uint32_t kSortGroup = 32;
constexpr uint32_t kPairWidth = 2;
constexpr uint32_t kMergeWays = 4;
constexpr uint32_t kMaxResidentShards = 8;
constexpr uint32_t kResidentProbeDebugInts = 32;
constexpr uint32_t kResidentFinalizeDebugInts = 16;
constexpr uint32_t kShardCurrentCount = 0;
constexpr uint32_t kShardMissCount = 1;
constexpr uint32_t kShardEvictableCount = 2;
constexpr uint32_t kShardOldCount = 3;
constexpr uint32_t kShardSelectedEvictCount = 4;

template <AscendC::HardEvent event>
__aicore__ inline void Sync()
{
    const int32_t id =
        static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(id);
    AscendC::WaitFlag<event>(id);
}

template <typename T>
__aicore__ inline T ReadGlobalScalarFresh(
    AscendC::GlobalTensor<T>& tensor,
    uint64_t offset)
{
    AscendC::DataCacheCleanAndInvalid<
        T,
        AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(tensor[offset]);
    return tensor.GetValue(offset);
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
        __gm__ int32_t* shardMissTokens,
        __gm__ int16_t* shardMissPositions,
        __gm__ int16_t* shardEvictableSlots,
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
        shardMissTokens_.SetGlobalBuffer(
            shardMissTokens,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * shardCapacity_);
        shardMissPositions_.SetGlobalBuffer(
            shardMissPositions,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * shardCapacity_);
        shardEvictableSlots_.SetGlobalBuffer(
            shardEvictableSlots,
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
        if (selectedElements > 0 && !deduplicate_) {
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
        } else if (selectedElements > 0) {
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
        // Sort workspaces are dead after shard-local sort/dedup. Reuse them
        // for the compact intersection products instead of increasing UB.
        auto shardMissTokens = src.ReinterpretCast<int32_t>();
        auto tmpInt16 = tmp.ReinterpretCast<int16_t>();
        auto shardMissPositions = tmpInt16;
        auto shardEvictableSlots = tmpInt16[shardCapacity_];
        const int32_t state = ReadGlobalScalarFresh(
            requestStateIndices_, request);
        const bool realState =
            state >= 0 &&
            state < static_cast<int32_t>(dummyStateBase_);
        const uint32_t safeState = realState
            ? static_cast<uint32_t>(state)
            : dummyStateBase_ + request;
        const int64_t requestedGeneration = ReadGlobalScalarFresh(
            requestStateGenerations_, request);
        const int64_t storedGeneration = ReadGlobalScalarFresh(
            stateGenerations_,
            static_cast<uint64_t>(safeState) * generationStride_);
        const bool generationMatches =
            realState && storedGeneration == requestedGeneration;
        const uint64_t stateCountOffset =
            static_cast<uint64_t>(safeState)
                * shardCountRequestStride_
            + shard * shardCountStride_;
        const uint32_t oldCount = generationMatches
            ? static_cast<uint32_t>(
                  ReadGlobalScalarFresh(
                      stateCounts_, stateCountOffset))
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
        uint32_t missCount = 0;
        uint32_t evictableCount = 0;
        if (rank > 0) {
            for (uint32_t currentIndex = 0;
                 currentIndex < rank;
                 ++currentIndex) {
                const int32_t token =
                    sortedTokens.GetValue(currentIndex);
                while (
                    oldIndex < oldCount &&
                    oldTokens.GetValue(oldIndex) < token) {
                    shardEvictableSlots.SetValue(
                        evictableCount++, oldSlots.GetValue(oldIndex));
                    ++oldIndex;
                }
                int16_t slot = static_cast<int16_t>(-1);
                if (oldIndex < oldCount &&
                    oldTokens.GetValue(oldIndex) == token) {
                    slot = oldSlots.GetValue(oldIndex);
                    ++oldIndex;
                } else {
                    shardMissTokens.SetValue(missCount, token);
                    shardMissPositions.SetValue(
                        missCount, static_cast<int16_t>(currentIndex));
                    ++missCount;
                }
                priorSlots.SetValue(currentIndex, slot);
            }
        }
        while (oldIndex < oldCount) {
            shardEvictableSlots.SetValue(
                evictableCount++, oldSlots.GetValue(oldIndex));
            ++oldIndex;
        }
        if (!generationMatches) {
            stateCounts_.SetValue(stateCountOffset, 0);
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        Sync<AscendC::HardEvent::V_MTE3>();
        if (rank > 0) {
            CopyLocalToGlobalExact(
                shardPacked_[shardOffset], sortedTokens, rank);
        }
        AscendC::DataCopy(
            shardMapping_[mappingOffset], mapping, requestWidth_);
        if (rank > 0) {
            CopyLocalToGlobalExact(
                priorSlots_[shardOffset], priorSlots, rank);
        }
        if (missCount > 0) {
            CopyLocalToGlobalExact(
                shardMissTokens_[shardOffset],
                shardMissTokens,
                missCount);
            CopyLocalToGlobalExact(
                shardMissPositions_[shardOffset],
                shardMissPositions,
                missCount);
        }
        if (evictableCount > 0) {
            CopyLocalToGlobalExact(
                shardEvictableSlots_[shardOffset],
                shardEvictableSlots,
                evictableCount);
        }
        // Host validation reserves one full 64-byte int32 cacheline per
        // (request, shard), so sibling AIVs never share this write line.
        shardCounts_.SetValue(
            countOffset + kShardCurrentCount,
            static_cast<int32_t>(rank));
        shardCounts_.SetValue(
            countOffset + kShardMissCount,
            static_cast<int32_t>(missCount));
        shardCounts_.SetValue(
            countOffset + kShardEvictableCount,
            static_cast<int32_t>(evictableCount));
        shardCounts_.SetValue(
            countOffset + kShardOldCount,
            static_cast<int32_t>(oldCount));
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
    AscendC::GlobalTensor<int32_t> shardMissTokens_;
    AscendC::GlobalTensor<int16_t> shardMissPositions_;
    AscendC::GlobalTensor<int16_t> shardEvictableSlots_;
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

// Test-only read probe. It never mutates resident state. For every shard it
// records the count observed by raw scalar GM access, DCCI-protected scalar
// access, and MTE2 bulk access. It also publishes the complete active
// prior-slot payload after the same GM-to-UB copy used by the production
// finalize kernel, so tests can compare every element on the host.
class DSAResidentSortedReadProbeKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* shardCounts,
        __gm__ int16_t* priorSlots,
        __gm__ int32_t* debugInfo,
        __gm__ int16_t* priorReadback,
        uint32_t requestCount,
        uint32_t shardCount,
        uint32_t capacity,
        uint32_t shardCountStride,
        uint32_t shardCountRequestStride)
    {
        requestCount_ = requestCount;
        shardCount_ = shardCount;
        capacity_ = capacity;
        shardCountStride_ = shardCountStride;
        shardCountRequestStride_ = shardCountRequestStride;
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            static_cast<uint64_t>(requestCount_)
                * shardCountRequestStride_);
        priorSlots_.SetGlobalBuffer(
            priorSlots,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * capacity_);
        debugInfo_.SetGlobalBuffer(
            debugInfo,
            static_cast<uint64_t>(requestCount_)
                * kResidentProbeDebugInts);
        priorReadback_.SetGlobalBuffer(
            priorReadback,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * capacity_);
        pipe_.InitBuffer(countBuf_, kDataBlockBytes);
        pipe_.InitBuffer(
            priorBuf_, capacity_ * sizeof(int16_t));
        pipe_.InitBuffer(
            debugBuf_,
            kResidentProbeDebugInts * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        auto countLocal = countBuf_.Get<int32_t>();
        auto priorLocal = priorBuf_.Get<int16_t>();
        auto debug = debugBuf_.Get<int32_t>();
        AscendC::Duplicate(
            debug,
            static_cast<int32_t>(0),
            kResidentProbeDebugInts);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_S>();
        debug.SetValue(0, static_cast<int32_t>(0x52535031));
        debug.SetValue(1, static_cast<int32_t>(shardCount_));
        debug.SetValue(2, static_cast<int32_t>(capacity_));

        const uint64_t requestShardBase =
            static_cast<uint64_t>(request) * shardCount_ * capacity_;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint64_t countOffset =
                static_cast<uint64_t>(request)
                    * shardCountRequestStride_
                + shard * shardCountStride_;
            const int32_t rawCount =
                shardCounts_.GetValue(countOffset);
            const int32_t freshCount =
                ReadGlobalScalarFresh(shardCounts_, countOffset);
            AscendC::DataCopy(
                countLocal,
                shardCounts_[countOffset],
                kInt32PerDataBlock);
            Sync<AscendC::HardEvent::MTE2_S>();
            const int32_t bulkCount = countLocal.GetValue(0);
            const uint32_t safeCount =
                bulkCount > 0 &&
                    static_cast<uint32_t>(bulkCount) <= capacity_
                ? static_cast<uint32_t>(bulkCount)
                : 0U;
            const uint64_t priorOffset =
                requestShardBase
                + static_cast<uint64_t>(shard) * capacity_;
            if (safeCount > 0) {
                CopyGlobalToLocalExact(
                    priorLocal,
                    priorSlots_[priorOffset],
                    safeCount);
            }
            Sync<AscendC::HardEvent::MTE2_S>();
            uint32_t bulkNegative = 0;
            uint32_t bulkNonnegative = 0;
            if (safeCount > 0) {
                for (uint32_t index = 0; index < safeCount; ++index) {
                    if (priorLocal.GetValue(index) < 0) {
                        ++bulkNegative;
                    } else {
                        ++bulkNonnegative;
                    }
                }
            }
            if (safeCount > 0) {
                Sync<AscendC::HardEvent::S_MTE3>();
                CopyLocalToGlobalExact(
                    priorReadback_[priorOffset],
                    priorLocal,
                    safeCount);
            }
            const uint32_t base = 4 + shard * 7;
            debug.SetValue(base, rawCount);
            debug.SetValue(base + 1, freshCount);
            debug.SetValue(base + 2, bulkCount);
            debug.SetValue(base + 3, safeCount > 0
                ? static_cast<int32_t>(priorLocal.GetValue(0))
                : static_cast<int32_t>(0x7FFF));
            debug.SetValue(base + 4, safeCount > 0
                ? static_cast<int32_t>(
                    priorLocal.GetValue(safeCount - 1))
                : static_cast<int32_t>(0x7FFF));
            debug.SetValue(
                base + 5, static_cast<int32_t>(bulkNegative));
            debug.SetValue(
                base + 6, static_cast<int32_t>(bulkNonnegative));
            if (safeCount > 0) {
                Sync<AscendC::HardEvent::MTE3_MTE2>();
            } else {
                Sync<AscendC::HardEvent::S_MTE2>();
            }
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(
            debugInfo_[
                static_cast<uint64_t>(request)
                    * kResidentProbeDebugInts],
            debug,
            kResidentProbeDebugInts);
    }

private:
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::GlobalTensor<int16_t> priorSlots_;
    AscendC::GlobalTensor<int32_t> debugInfo_;
    AscendC::GlobalTensor<int16_t> priorReadback_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> countBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> priorBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> debugBuf_;
    uint32_t requestCount_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t capacity_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t shardCountRequestStride_ = 0;
};

class DSAResidentSortedFinalizeKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* shardPacked,
        __gm__ int32_t* shardCounts,
        __gm__ int16_t* priorSlots,
        __gm__ int32_t* shardMissTokens,
        __gm__ int16_t* shardMissPositions,
        __gm__ int16_t* shardEvictableSlots,
        __gm__ int32_t* missTokens,
        __gm__ int32_t* missCounts,
        __gm__ int64_t* targetSlots,
        __gm__ int32_t* requestBlockTable,
        __gm__ int32_t* debugInfo,
        uint32_t requestCount,
        uint32_t shardCount,
        uint32_t capacity,
        uint32_t shardCountStride,
        uint32_t shardCountRequestStride,
        uint32_t missCountStride,
        uint32_t blockTableWidth,
        uint32_t blockSize,
        uint32_t debugStage)
    {
        requestCount_ = requestCount;
        shardCount_ = shardCount;
        capacity_ = capacity;
        shardCountStride_ = shardCountStride;
        shardCountRequestStride_ = shardCountRequestStride;
        missCountStride_ = missCountStride;
        blockTableWidth_ = blockTableWidth;
        blockSize_ = blockSize;
        debugStage_ = debugStage;
        blockTableEntries_ =
            (capacity_ + blockSize_ - 1) / blockSize_;
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
        shardMissTokens_.SetGlobalBuffer(
            shardMissTokens, requestShardElements);
        shardMissPositions_.SetGlobalBuffer(
            shardMissPositions, requestShardElements);
        shardEvictableSlots_.SetGlobalBuffer(
            shardEvictableSlots, requestShardElements);
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
        debugInfo_.SetGlobalBuffer(
            debugInfo,
            static_cast<uint64_t>(requestCount_)
                * kResidentFinalizeDebugInts);

        const uint32_t packedElements =
            capacity_ + shardCount_ * kInt16PerDataBlock;
        pipe_.InitBuffer(
            protectedBuf_, packedElements * sizeof(int16_t));
        pipe_.InitBuffer(
            freeSlotBuf_, packedElements * sizeof(int16_t));
        pipe_.InitBuffer(
            packedTokenBuf_, packedElements * sizeof(int32_t));
        pipe_.InitBuffer(
            packedPriorSlotBuf_, packedElements * sizeof(int16_t));
        pipe_.InitBuffer(
            missTokenBuf_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(
            targetSlotBuf_, capacity_ * sizeof(int64_t));
        const uint32_t blockTableBytes =
            (blockTableEntries_ * sizeof(int32_t)
                + kDataBlockBytes - 1)
            & ~(kDataBlockBytes - 1);
        pipe_.InitBuffer(blockTableBuf_, blockTableBytes);
        pipe_.InitBuffer(
            shardMetadataBuf_,
            shardCount_ * shardCountStride_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t request = AscendC::GetBlockIdx();
        if (request >= requestCount_) {
            return;
        }
        if (debugStage_ == 0) {
            ProcessCompact(request);
            return;
        }
        const uint64_t requestOffset =
            static_cast<uint64_t>(request) * capacity_;
        const uint64_t requestShardBase =
            static_cast<uint64_t>(request) * shardCount_ * capacity_;
        auto protectedSlots = protectedBuf_.Get<int16_t>();
        auto freeSlots = freeSlotBuf_.Get<int16_t>();
        auto packedTokens = packedTokenBuf_.Get<int32_t>();
        auto packedPriorSlots =
            packedPriorSlotBuf_.Get<int16_t>();
        auto missTokens = missTokenBuf_.Get<int32_t>();
        auto targetSlots = targetSlotBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();
        auto shardMetadata = shardMetadataBuf_.Get<int32_t>();
        AscendC::Duplicate(
            protectedSlots, static_cast<int16_t>(0), capacity_);
        AscendC::PipeBarrier<PIPE_V>();

        const AscendC::DataCopyParams blockTableCopy{
            1,
            static_cast<uint16_t>(
                blockTableEntries_ * sizeof(int32_t)),
            0,
            0};
        AscendC::DataCopyPad(
            blockTable,
            requestBlockTable_[
                static_cast<uint64_t>(request) * blockTableWidth_],
            blockTableCopy,
            {});
        AscendC::DataCopy(
            shardMetadata,
            shardCounts_[
                static_cast<uint64_t>(request)
                    * shardCountRequestStride_],
            shardCount_ * shardCountStride_);
        Sync<AscendC::HardEvent::MTE2_S>();
        uint32_t shardCounts[kMaxResidentShards] = {};
        uint32_t shardOffsets[kMaxResidentShards] = {};
        uint32_t packedEnd = 0;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t count = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardCurrentCount));
            const uint32_t localOffset =
                (packedEnd + kInt16PerDataBlock - 1)
                & ~(kInt16PerDataBlock - 1);
            const uint64_t shardOffset =
                requestShardBase
                + static_cast<uint64_t>(shard) * capacity_;
            shardCounts[shard] = count;
            shardOffsets[shard] = localOffset;
            if (count > 0) {
                CopyGlobalToLocalExact(
                    packedTokens[localOffset],
                    shardPacked_[shardOffset],
                    count);
                CopyGlobalToLocalExact(
                    packedPriorSlots[localOffset],
                    priorSlots_[shardOffset],
                    count);
            }
            packedEnd = localOffset + count;
        }
        Sync<AscendC::HardEvent::MTE2_S>();
        Sync<AscendC::HardEvent::V_S>();
        if (debugStage_ == 1) {
            PublishDebug(
                blockTable, request, packedEnd,
                0, 0, 0, 0x7FFF, 0x7FFF,
                0x7FFFFFFF, 0x7FFFFFFF, -1, -1);
            return;
        }

        uint32_t protectedCount = 0;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t count = shardCounts[shard];
            const uint32_t localOffset = shardOffsets[shard];
            if (count > 0) {
                for (uint32_t index = 0; index < count; ++index) {
                    const int16_t slot =
                        packedPriorSlots.GetValue(
                            localOffset + index);
                    if (slot >= 0) {
                        protectedSlots.SetValue(
                            static_cast<uint32_t>(slot),
                            static_cast<int16_t>(1));
                        ++protectedCount;
                    }
                }
            }
        }
        if (debugStage_ == 2) {
            PublishDebug(
                blockTable, request, packedEnd,
                protectedCount, 0, 0, 0x7FFF, 0x7FFF,
                0x7FFFFFFF, 0x7FFFFFFF, -1, -1);
            return;
        }
        if (debugStage_ == 3) {
            Sync<AscendC::HardEvent::S_MTE3>();
            CopyLocalToGlobalExact(
                priorSlots_[requestShardBase],
                protectedSlots,
                capacity_);
            PublishDebug(
                blockTable, request, packedEnd,
                protectedCount, 0, 0, 0x7FFF, 0x7FFF,
                0x7FFFFFFF, 0x7FFFFFFF, -1, -1);
            return;
        }

        uint32_t freeCount = 0;
        for (uint32_t slot = 0; slot < capacity_; ++slot) {
            if (protectedSlots.GetValue(slot) == 0) {
                freeSlots.SetValue(
                    freeCount++, static_cast<int16_t>(slot));
            }
        }
        if (debugStage_ == 4) {
            PublishDebug(
                blockTable, request, packedEnd,
                protectedCount, freeCount, 0, 0x7FFF, 0x7FFF,
                0x7FFFFFFF, 0x7FFFFFFF, -1, -1);
            return;
        }
        if (debugStage_ == 5) {
            Sync<AscendC::HardEvent::S_MTE3>();
            if (freeCount > 0) {
                CopyLocalToGlobalExact(
                    priorSlots_[requestShardBase],
                    freeSlots,
                    freeCount);
            }
            PublishDebug(
                blockTable, request, packedEnd,
                protectedCount, freeCount, 0, 0x7FFF, 0x7FFF,
                0x7FFFFFFF, 0x7FFFFFFF, -1, -1);
            return;
        }

        uint32_t missCount = 0;
        int32_t firstAssignedSlot = 0x7FFF;
        int32_t lastAssignedSlot = 0x7FFF;
        int32_t firstMissToken = 0x7FFFFFFF;
        int32_t lastMissToken = 0x7FFFFFFF;
        int32_t firstTarget = -1;
        int32_t lastTarget = -1;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t count = shardCounts[shard];
            const uint32_t localOffset = shardOffsets[shard];
            if (count > 0) {
                for (uint32_t index = 0; index < count; ++index) {
                    int16_t slot =
                        packedPriorSlots.GetValue(
                            localOffset + index);
                    if (slot < 0) {
                        slot = freeSlots.GetValue(missCount);
                        const int32_t token =
                            packedTokens.GetValue(localOffset + index);
                        missTokens.SetValue(missCount, token);
                        const uint32_t logicalSlot =
                            static_cast<uint32_t>(slot);
                        const uint32_t logicalBlock =
                            logicalSlot / blockSize_;
                        const uint32_t blockOffset =
                            logicalSlot % blockSize_;
                        const int32_t physicalBlock =
                            blockTable.GetValue(logicalBlock);
                        targetSlots.SetValue(
                            missCount,
                            static_cast<int64_t>(physicalBlock)
                                    * blockSize_
                                + blockOffset);
                        const int32_t target =
                            physicalBlock
                                * static_cast<int32_t>(blockSize_)
                            + static_cast<int32_t>(blockOffset);
                        if (missCount == 0) {
                            firstAssignedSlot =
                                static_cast<int32_t>(slot);
                            firstMissToken = token;
                            firstTarget = target;
                        }
                        lastAssignedSlot =
                            static_cast<int32_t>(slot);
                        lastMissToken = token;
                        lastTarget = target;
                        ++missCount;
                    }
                    packedPriorSlots.SetValue(
                        localOffset + index, slot);
                }
            }
        }
        if (debugStage_ == 6) {
            PublishDebug(
                blockTable, request, packedEnd,
                protectedCount, freeCount, missCount,
                firstAssignedSlot, lastAssignedSlot,
                firstMissToken, lastMissToken,
                firstTarget, lastTarget);
            return;
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t count = shardCounts[shard];
            if (count == 0) {
                continue;
            }
            const uint64_t shardOffset =
                requestShardBase
                + static_cast<uint64_t>(shard) * capacity_;
            CopyLocalToGlobalExact(
                priorSlots_[shardOffset],
                packedPriorSlots[shardOffsets[shard]],
                count);
        }
        if (debugStage_ == 7) {
            PublishDebug(
                blockTable, request, packedEnd,
                protectedCount, freeCount, missCount,
                firstAssignedSlot, lastAssignedSlot,
                firstMissToken, lastMissToken,
                firstTarget, lastTarget);
            return;
        }
        if (debugStage_ == 8) {
            PublishDebug(
                blockTable, request, packedEnd,
                protectedCount, freeCount, missCount,
                firstAssignedSlot, lastAssignedSlot,
                firstMissToken, lastMissToken,
                firstTarget, lastTarget);
            return;
        }
        if (missCount > 0) {
            CopyLocalToGlobalExact(
                missTokens_[requestOffset],
                missTokens,
                missCount);
        }
        if (debugStage_ == 9) {
            PublishDebug(
                blockTable, request, packedEnd,
                protectedCount, freeCount, missCount,
                firstAssignedSlot, lastAssignedSlot,
                firstMissToken, lastMissToken,
                firstTarget, lastTarget);
            return;
        }
        if (missCount > 0) {
            CopyLocalToGlobalExact(
                targetSlots_[requestOffset],
                targetSlots,
                missCount);
        }
        if (debugStage_ == 10) {
            PublishDebug(
                blockTable, request, packedEnd,
                protectedCount, freeCount, missCount,
                firstAssignedSlot, lastAssignedSlot,
                firstMissToken, lastMissToken,
                firstTarget, lastTarget);
            return;
        }
        // One cacheline per request prevents count false sharing.
        missCounts_.SetValue(
            static_cast<uint64_t>(request) * missCountStride_,
            static_cast<int32_t>(missCount));
        if (debugStage_ == 11) {
            PublishDebug(
                blockTable, request, packedEnd,
                protectedCount, freeCount, missCount,
                firstAssignedSlot, lastAssignedSlot,
                firstMissToken, lastMissToken,
                firstTarget, lastTarget);
        }
    }

private:
    __aicore__ inline void ProcessCompact(uint32_t request)
    {
        const uint64_t requestOffset =
            static_cast<uint64_t>(request) * capacity_;
        const uint64_t requestShardBase =
            static_cast<uint64_t>(request) * shardCount_ * capacity_;
        auto missPositions = protectedBuf_.Get<int16_t>();
        auto evictableSlots = freeSlotBuf_.Get<int16_t>();
        auto inputMissTokens = packedTokenBuf_.Get<int32_t>();
        auto packedPriorSlots = packedPriorSlotBuf_.Get<int16_t>();
        auto outputMissTokens = missTokenBuf_.Get<int32_t>();
        auto targetSlots = targetSlotBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();
        auto shardMetadata = shardMetadataBuf_.Get<int32_t>();

        const AscendC::DataCopyParams blockTableCopy{
            1,
            static_cast<uint16_t>(
                blockTableEntries_ * sizeof(int32_t)),
            0,
            0};
        AscendC::DataCopyPad(
            blockTable,
            requestBlockTable_[
                static_cast<uint64_t>(request) * blockTableWidth_],
            blockTableCopy,
            {});
        AscendC::DataCopy(
            shardMetadata,
            shardCounts_[
                static_cast<uint64_t>(request)
                    * shardCountRequestStride_],
            shardCount_ * shardCountStride_);
        Sync<AscendC::HardEvent::MTE2_S>();

        uint32_t currentCounts[kMaxResidentShards] = {};
        uint32_t missCounts[kMaxResidentShards] = {};
        uint32_t evictableCounts[kMaxResidentShards] = {};
        uint32_t selectedEvictCounts[kMaxResidentShards] = {};
        uint32_t priorOffsets[kMaxResidentShards] = {};
        uint32_t missOffsets[kMaxResidentShards] = {};
        uint32_t evictableOffsets[kMaxResidentShards] = {};
        uint32_t priorEnd = 0;
        uint32_t missEnd = 0;
        uint32_t evictableEnd = 0;
        uint32_t totalMissCount = 0;
        uint32_t totalEvictableCount = 0;
        uint32_t totalOldCount = 0;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t currentCount = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardCurrentCount));
            const uint32_t shardMissCount = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardMissCount));
            const uint32_t evictableCount = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardEvictableCount));
            const uint32_t oldCount = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardOldCount));
            const uint32_t priorOffset =
                (priorEnd + kInt16PerDataBlock - 1)
                & ~(kInt16PerDataBlock - 1);
            const uint32_t missOffset =
                (missEnd + kInt16PerDataBlock - 1)
                & ~(kInt16PerDataBlock - 1);
            const uint64_t shardOffset =
                requestShardBase
                + static_cast<uint64_t>(shard) * capacity_;
            currentCounts[shard] = currentCount;
            missCounts[shard] = shardMissCount;
            evictableCounts[shard] = evictableCount;
            priorOffsets[shard] = priorOffset;
            missOffsets[shard] = missOffset;
            if (currentCount > 0) {
                CopyGlobalToLocalExact(
                    packedPriorSlots[priorOffset],
                    priorSlots_[shardOffset],
                    currentCount);
            }
            if (shardMissCount > 0) {
                CopyGlobalToLocalExact(
                    inputMissTokens[missOffset],
                    shardMissTokens_[shardOffset],
                    shardMissCount);
                CopyGlobalToLocalExact(
                    missPositions[missOffset],
                    shardMissPositions_[shardOffset],
                    shardMissCount);
            }
            priorEnd = priorOffset + currentCount;
            missEnd = missOffset + shardMissCount;
            totalMissCount += shardMissCount;
            totalEvictableCount += evictableCount;
            totalOldCount += oldCount;
        }

        // Finalize consumes evictable slots shard-major. Only copy the
        // prefix that can actually be paired with a miss. Each UB segment
        // remains 32-byte aligned for DataCopy/DataCopyPad; the small gaps
        // are covered by packedElements' per-shard padding allowance.
        uint32_t remainingEvictions = totalMissCount < totalEvictableCount
            ? totalMissCount
            : totalEvictableCount;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t selectedCount =
                evictableCounts[shard] < remainingEvictions
                ? evictableCounts[shard]
                : remainingEvictions;
            const uint32_t evictableOffset =
                (evictableEnd + kInt16PerDataBlock - 1)
                & ~(kInt16PerDataBlock - 1);
            const uint64_t shardOffset =
                requestShardBase
                + static_cast<uint64_t>(shard) * capacity_;
            const uint64_t countOffset =
                static_cast<uint64_t>(request)
                    * shardCountRequestStride_
                + shard * shardCountStride_;
            selectedEvictCounts[shard] = selectedCount;
            evictableOffsets[shard] = evictableOffset;
            if (selectedCount > 0) {
                CopyGlobalToLocalExact(
                    evictableSlots[evictableOffset],
                    shardEvictableSlots_[shardOffset],
                    selectedCount);
            }
            shardCounts_.SetValue(
                countOffset + kShardSelectedEvictCount,
                static_cast<int32_t>(selectedCount));
            evictableEnd = evictableOffset + selectedCount;
            remainingEvictions -= selectedCount;
        }
        const uint32_t totalSelectedEvictCount =
            totalMissCount < totalEvictableCount
            ? totalMissCount
            : totalEvictableCount;
        Sync<AscendC::HardEvent::MTE2_S>();
        Sync<AscendC::HardEvent::V_S>();

        uint32_t globalMiss = 0;
        uint32_t evictableShard = 0;
        uint32_t evictableIndex = 0;
        if (totalMissCount > 0) {
            for (uint32_t shard = 0; shard < shardCount_; ++shard) {
                const uint32_t shardMissCount = missCounts[shard];
                if (shardMissCount == 0) {
                    continue;
                }
                for (uint32_t localMiss = 0;
                     localMiss < shardMissCount;
                     ++localMiss) {
                    int16_t slot;
                    if (globalMiss < totalSelectedEvictCount) {
                        while (
                            evictableShard < shardCount_ &&
                            evictableIndex >=
                                selectedEvictCounts[evictableShard]) {
                            ++evictableShard;
                            evictableIndex = 0;
                        }
                        slot = evictableSlots.GetValue(
                            evictableOffsets[evictableShard]
                                + evictableIndex);
                        ++evictableIndex;
                    } else {
                        // Resident allocation preserves a dense occupied
                        // prefix. Every old entry not present in the current
                        // union is emitted above as an evictable slot; after
                        // pooling all shards, any remaining miss grows the
                        // prefix from totalOldCount. Therefore the global
                        // candidate count is always capacity - hitCount.
                        slot = static_cast<int16_t>(
                            totalOldCount
                            + globalMiss
                            - totalSelectedEvictCount);
                    }
                    const uint32_t position = static_cast<uint32_t>(
                        missPositions.GetValue(
                            missOffsets[shard] + localMiss));
                    packedPriorSlots.SetValue(
                        priorOffsets[shard] + position, slot);
                    const int32_t token = inputMissTokens.GetValue(
                        missOffsets[shard] + localMiss);
                    outputMissTokens.SetValue(globalMiss, token);
                    const uint32_t logicalSlot =
                        static_cast<uint32_t>(slot);
                    const uint32_t logicalBlock =
                        logicalSlot / blockSize_;
                    const uint32_t blockOffset =
                        logicalSlot % blockSize_;
                    const int32_t physicalBlock =
                        blockTable.GetValue(logicalBlock);
                    targetSlots.SetValue(
                        globalMiss,
                        static_cast<int64_t>(physicalBlock)
                                * blockSize_
                            + blockOffset);
                    ++globalMiss;
                }
            }
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t currentCount = currentCounts[shard];
            if (currentCount == 0) {
                continue;
            }
            const uint64_t shardOffset =
                requestShardBase
                + static_cast<uint64_t>(shard) * capacity_;
            CopyLocalToGlobalExact(
                priorSlots_[shardOffset],
                packedPriorSlots[priorOffsets[shard]],
                currentCount);
        }
        if (totalMissCount > 0) {
            CopyLocalToGlobalExact(
                missTokens_[requestOffset],
                outputMissTokens,
                totalMissCount);
            CopyLocalToGlobalExact(
                targetSlots_[requestOffset],
                targetSlots,
                totalMissCount);
        }
        missCounts_.SetValue(
            static_cast<uint64_t>(request) * missCountStride_,
            static_cast<int32_t>(totalMissCount));
    }

    __aicore__ inline void PublishDebug(
        AscendC::LocalTensor<int32_t> debug,
        uint32_t request,
        uint32_t packedEnd,
        uint32_t protectedCount,
        uint32_t freeCount,
        uint32_t missCount,
        int32_t firstAssignedSlot,
        int32_t lastAssignedSlot,
        int32_t firstMissToken,
        int32_t lastMissToken,
        int32_t firstTarget,
        int32_t lastTarget)
    {
        debug.SetValue(0, static_cast<int32_t>(0x52534631));
        debug.SetValue(1, static_cast<int32_t>(debugStage_));
        debug.SetValue(2, static_cast<int32_t>(shardCount_));
        debug.SetValue(3, static_cast<int32_t>(capacity_));
        debug.SetValue(4, static_cast<int32_t>(packedEnd));
        debug.SetValue(5, static_cast<int32_t>(protectedCount));
        debug.SetValue(6, static_cast<int32_t>(freeCount));
        debug.SetValue(7, static_cast<int32_t>(missCount));
        debug.SetValue(8, firstAssignedSlot);
        debug.SetValue(9, lastAssignedSlot);
        debug.SetValue(10, firstMissToken);
        debug.SetValue(11, lastMissToken);
        debug.SetValue(12, firstTarget);
        debug.SetValue(13, lastTarget);
        debug.SetValue(14, static_cast<int32_t>(blockTableEntries_));
        debug.SetValue(15, static_cast<int32_t>(blockSize_));
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(
            debugInfo_[
                static_cast<uint64_t>(request)
                    * kResidentFinalizeDebugInts],
            debug,
            kResidentFinalizeDebugInts);
    }

    AscendC::GlobalTensor<int32_t> shardPacked_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::GlobalTensor<int16_t> priorSlots_;
    AscendC::GlobalTensor<int32_t> shardMissTokens_;
    AscendC::GlobalTensor<int16_t> shardMissPositions_;
    AscendC::GlobalTensor<int16_t> shardEvictableSlots_;
    AscendC::GlobalTensor<int32_t> missTokens_;
    AscendC::GlobalTensor<int32_t> missCounts_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::GlobalTensor<int32_t> debugInfo_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> protectedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> freeSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC>
        packedPriorSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> missTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> targetSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockTableBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shardMetadataBuf_;
    uint32_t requestCount_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t capacity_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t shardCountRequestStride_ = 0;
    uint32_t missCountStride_ = 0;
    uint32_t blockTableWidth_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t blockTableEntries_ = 0;
    uint32_t debugStage_ = 0;
};

class DSAResidentSortedUpdateKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int32_t* shardPacked,
        __gm__ int16_t* shardMapping,
        __gm__ int32_t* shardCounts,
        __gm__ int16_t* priorSlots,
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
            selectedMaskBuf_, capacity_ * sizeof(uint8_t));
        pipe_.InitBuffer(
            survivorTokenBuf_, capacity_ * sizeof(int32_t));
        // Remap reinterprets both slot buffers as int32/float. For MTP=1
        // with one shard, remapPartWidth equals capacity, so the old int16
        // allocation exposed only half the required elements and corrupted
        // the second half of the remapped row.
        const uint32_t remapPartWidth = requestWidth_ / shardCount_;
        const uint32_t remapInt32Bytes =
            remapPartWidth * sizeof(int32_t);
        const uint32_t compactInt16Bytes =
            capacity_ * sizeof(int16_t);
        const uint32_t remapReuseBytes =
            remapInt32Bytes > compactInt16Bytes
                ? remapInt32Bytes
                : compactInt16Bytes;
        pipe_.InitBuffer(survivorSlotBuf_, remapReuseBytes);
        pipe_.InitBuffer(
            mergedTokenBuf_, capacity_ * sizeof(int32_t));
        pipe_.InitBuffer(mergedSlotBuf_, remapReuseBytes);
        pipe_.InitBuffer(
            requestMetadataBuf_,
            shardCount_ * shardCountStride_ * sizeof(int32_t));
        pipe_.InitBuffer(
            stateMetadataBuf_,
            shardCountStride_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t request = block / shardCount_;
        const uint32_t shard = block % shardCount_;
        if (request >= requestCount_) {
            return;
        }
        const int32_t state = ReadGlobalScalarFresh(
            requestStateIndices_, request);
        const bool realState =
            state >= 0 &&
            state < static_cast<int32_t>(dummyStateBase_);
        const uint32_t safeState = realState
            ? static_cast<uint32_t>(state)
            : dummyStateBase_ + request;
        const int64_t requestedGeneration = ReadGlobalScalarFresh(
            requestStateGenerations_, request);
        const uint64_t oldCountOffset =
            static_cast<uint64_t>(safeState)
                * shardCountRequestStride_
            + shard * shardCountStride_;
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
        auto mergedTokens = mergedTokenBuf_.Get<int32_t>();
        auto mergedSlots = mergedSlotBuf_.Get<int16_t>();
        auto requestMetadata = requestMetadataBuf_.Get<int32_t>();
        auto stateMetadata = stateMetadataBuf_.Get<int32_t>();

        AscendC::DataCopy(
            requestMetadata,
            shardCounts_[
                static_cast<uint64_t>(request)
                    * shardCountRequestStride_],
            shardCount_ * shardCountStride_);
        AscendC::DataCopy(
            stateMetadata,
            stateCounts_[oldCountOffset],
            shardCountStride_);
        Sync<AscendC::HardEvent::MTE2_S>();
        const uint32_t oldCount = static_cast<uint32_t>(
            stateMetadata.GetValue(kShardCurrentCount));
        const uint32_t currentCount = static_cast<uint32_t>(
            requestMetadata.GetValue(
                shard * shardCountStride_
                + kShardCurrentCount));
        const uint32_t selectedEvictCount = static_cast<uint32_t>(
            requestMetadata.GetValue(
                shard * shardCountStride_
                + kShardSelectedEvictCount));

        if (oldCount > 0) {
            CopyGlobalToLocalExact(
                oldTokens, stateTokens_[oldStateOffset], oldCount);
            CopyGlobalToLocalExact(
                oldSlots, stateSlots_[oldStateOffset], oldCount);
        }
        if (currentCount > 0) {
            CopyGlobalToLocalExact(
                currentTokens,
                shardPacked_[requestShardOffset],
                currentCount);
            CopyGlobalToLocalExact(
                priorSlots,
                priorSlots_[requestShardOffset],
                currentCount);
        }
        Sync<AscendC::HardEvent::MTE2_S>();

        // Rebuild the sorted resident state in one merge. Finalize consumes
        // each shard's evictable list from the beginning and publishes that
        // prefix length above. Therefore the first selectedEvictCount old
        // tokens absent from current are removed; later absent old tokens
        // remain resident, hits retain their old slot, and misses already
        // carry their newly assigned slot in priorSlots.
        uint32_t oldIndex = 0;
        uint32_t currentIndex = 0;
        uint32_t evictableIndex = 0;
        uint32_t mergedCount = 0;
        if (oldCount > 0 || currentCount > 0) {
            while (oldIndex < oldCount ||
                   currentIndex < currentCount) {
                const bool haveOld = oldIndex < oldCount;
                const bool haveCurrent = currentIndex < currentCount;
                const int32_t oldToken = haveOld
                    ? oldTokens.GetValue(oldIndex)
                    : static_cast<int32_t>(0x7FFFFFFF);
                const int32_t currentToken = haveCurrent
                    ? currentTokens.GetValue(currentIndex)
                    : static_cast<int32_t>(0x7FFFFFFF);
                if (haveOld &&
                    (!haveCurrent || oldToken < currentToken)) {
                    if (evictableIndex >= selectedEvictCount) {
                        mergedTokens.SetValue(mergedCount, oldToken);
                        mergedSlots.SetValue(
                            mergedCount, oldSlots.GetValue(oldIndex));
                        ++mergedCount;
                    }
                    ++evictableIndex;
                    ++oldIndex;
                } else if (haveCurrent &&
                           (!haveOld || currentToken < oldToken)) {
                    mergedTokens.SetValue(mergedCount, currentToken);
                    mergedSlots.SetValue(
                        mergedCount,
                        priorSlots.GetValue(currentIndex));
                    ++currentIndex;
                    ++mergedCount;
                } else {
                    // Equal token: preserve the resident slot and consume
                    // both sorted inputs exactly once.
                    mergedTokens.SetValue(mergedCount, currentToken);
                    mergedSlots.SetValue(
                        mergedCount, oldSlots.GetValue(oldIndex));
                    ++oldIndex;
                    ++currentIndex;
                    ++mergedCount;
                }
            }
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        if (mergedCount > 0) {
            CopyLocalToGlobalExact(
                stateTokens_[oldStateOffset],
                mergedTokens,
                mergedCount);
            CopyLocalToGlobalExact(
                stateSlots_[oldStateOffset],
                mergedSlots,
                mergedCount);
        }
        const uint64_t newCountOffset =
            static_cast<uint64_t>(safeState)
                * shardCountRequestStride_
            + shard * shardCountStride_;
        stateCounts_.SetValue(
            newCountOffset, static_cast<int32_t>(mergedCount));

        RemapPositionPartition(
            request, shard, requestShardBase, requestMetadata);

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

    // Split-path state update. Keep Process() above unchanged so the original
    // fused update+remap kernel remains available as an exact fallback.
    __aicore__ inline void ProcessStateOnly()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t request = block / shardCount_;
        const uint32_t shard = block % shardCount_;
        if (request >= requestCount_) {
            return;
        }
        const int32_t state = ReadGlobalScalarFresh(
            requestStateIndices_, request);
        const bool realState =
            state >= 0 &&
            state < static_cast<int32_t>(dummyStateBase_);
        const uint32_t safeState = realState
            ? static_cast<uint32_t>(state)
            : dummyStateBase_ + request;
        const int64_t requestedGeneration = ReadGlobalScalarFresh(
            requestStateGenerations_, request);
        const uint64_t oldCountOffset =
            static_cast<uint64_t>(safeState)
                * shardCountRequestStride_
            + shard * shardCountStride_;
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
        auto mergedTokens = mergedTokenBuf_.Get<int32_t>();
        auto mergedSlots = mergedSlotBuf_.Get<int16_t>();
        auto requestMetadata = requestMetadataBuf_.Get<int32_t>();
        auto stateMetadata = stateMetadataBuf_.Get<int32_t>();

        AscendC::DataCopy(
            requestMetadata,
            shardCounts_[
                static_cast<uint64_t>(request)
                    * shardCountRequestStride_],
            shardCount_ * shardCountStride_);
        AscendC::DataCopy(
            stateMetadata,
            stateCounts_[oldCountOffset],
            shardCountStride_);
        Sync<AscendC::HardEvent::MTE2_S>();
        const uint32_t oldCount = static_cast<uint32_t>(
            stateMetadata.GetValue(kShardCurrentCount));
        const uint32_t currentCount = static_cast<uint32_t>(
            requestMetadata.GetValue(
                shard * shardCountStride_
                + kShardCurrentCount));
        const uint32_t selectedEvictCount = static_cast<uint32_t>(
            requestMetadata.GetValue(
                shard * shardCountStride_
                + kShardSelectedEvictCount));

        if (oldCount > 0) {
            CopyGlobalToLocalExact(
                oldTokens, stateTokens_[oldStateOffset], oldCount);
            CopyGlobalToLocalExact(
                oldSlots, stateSlots_[oldStateOffset], oldCount);
        }
        if (currentCount > 0) {
            CopyGlobalToLocalExact(
                currentTokens,
                shardPacked_[requestShardOffset],
                currentCount);
            CopyGlobalToLocalExact(
                priorSlots,
                priorSlots_[requestShardOffset],
                currentCount);
        }
        Sync<AscendC::HardEvent::MTE2_S>();

        // Rebuild the sorted resident state in one merge. Finalize consumes
        // each shard's evictable list from the beginning and publishes that
        // prefix length above. Therefore the first selectedEvictCount old
        // tokens absent from current are removed; later absent old tokens
        // remain resident, hits retain their old slot, and misses already
        // carry their newly assigned slot in priorSlots.
        uint32_t oldIndex = 0;
        uint32_t currentIndex = 0;
        uint32_t evictableIndex = 0;
        uint32_t mergedCount = 0;
        if (oldCount > 0 || currentCount > 0) {
            while (oldIndex < oldCount ||
                   currentIndex < currentCount) {
                const bool haveOld = oldIndex < oldCount;
                const bool haveCurrent = currentIndex < currentCount;
                const int32_t oldToken = haveOld
                    ? oldTokens.GetValue(oldIndex)
                    : static_cast<int32_t>(0x7FFFFFFF);
                const int32_t currentToken = haveCurrent
                    ? currentTokens.GetValue(currentIndex)
                    : static_cast<int32_t>(0x7FFFFFFF);
                if (haveOld &&
                    (!haveCurrent || oldToken < currentToken)) {
                    if (evictableIndex >= selectedEvictCount) {
                        mergedTokens.SetValue(mergedCount, oldToken);
                        mergedSlots.SetValue(
                            mergedCount, oldSlots.GetValue(oldIndex));
                        ++mergedCount;
                    }
                    ++evictableIndex;
                    ++oldIndex;
                } else if (haveCurrent &&
                           (!haveOld || currentToken < oldToken)) {
                    mergedTokens.SetValue(mergedCount, currentToken);
                    mergedSlots.SetValue(
                        mergedCount,
                        priorSlots.GetValue(currentIndex));
                    ++currentIndex;
                    ++mergedCount;
                } else {
                    // Equal token: preserve the resident slot and consume
                    // both sorted inputs exactly once.
                    mergedTokens.SetValue(mergedCount, currentToken);
                    mergedSlots.SetValue(
                        mergedCount, oldSlots.GetValue(oldIndex));
                    ++oldIndex;
                    ++currentIndex;
                    ++mergedCount;
                }
            }
        }

        Sync<AscendC::HardEvent::S_MTE3>();
        if (mergedCount > 0) {
            CopyLocalToGlobalExact(
                stateTokens_[oldStateOffset],
                mergedTokens,
                mergedCount);
            CopyLocalToGlobalExact(
                stateSlots_[oldStateOffset],
                mergedSlots,
                mergedCount);
        }
        const uint64_t newCountOffset =
            static_cast<uint64_t>(safeState)
                * shardCountRequestStride_
            + shard * shardCountStride_;
        stateCounts_.SetValue(
            newCountOffset, static_cast<int32_t>(mergedCount));

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
        uint64_t requestShardBase,
        AscendC::LocalTensor<int32_t> shardMetadata)
    {
        const uint32_t partWidth = requestWidth_ / shardCount_;
        const uint32_t begin = part * partWidth;
        const uint64_t requestOffset =
            static_cast<uint64_t>(request) * requestWidth_;
        const uint64_t mappingBase =
            static_cast<uint64_t>(request) * shardCount_
            * requestWidth_;

        // The state merge has finished consuming these buffers. Reuse them
        // for the same vector Gather+Select algorithm as the established
        // resident row remapper, adapted to shard-local int16 ranks/slots.
        auto input = oldTokenBuf_.Get<int32_t>();
        auto mappingOrGathered = oldSlotBuf_.Get<int16_t>();
        auto rankFloatOrCandidate = currentTokenBuf_.Get<float>();
        auto shardSlots = priorSlotBuf_.Get<int16_t>();
        auto ranksOrOutput = survivorTokenBuf_.Get<int32_t>();
        auto accumulatedSlots = survivorSlotBuf_.Get<int32_t>();
        auto clampedOffsets = mergedTokenBuf_.Get<int32_t>();
        auto gatheredFloat = mergedSlotBuf_.Get<float>();
        auto selectedMask = selectedMaskBuf_.Get<uint8_t>();

        // old/current/survivor buffers were consumed by the scalar pipeline,
        // while merged buffers are still MTE3 sources. Complete those
        // dependencies before MTE2/vector reuse of the same UB allocations.
        Sync<AscendC::HardEvent::S_MTE2>();
        Sync<AscendC::HardEvent::S_V>();
        Sync<AscendC::HardEvent::MTE3_V>();

        CopyGlobalToLocalExact(
            input,
            topkIndices_[requestOffset + begin],
            partWidth);
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Duplicate(
            accumulatedSlots, static_cast<int32_t>(-1), partWidth);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t count = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardCurrentCount));
            if (count == 0) {
                continue;
            }
            CopyGlobalToLocalExact(
                mappingOrGathered,
                shardMapping_[
                    mappingBase
                    + static_cast<uint64_t>(shard) * requestWidth_
                    + begin],
                partWidth);
            CopyGlobalToLocalExact(
                shardSlots,
                priorSlots_[
                    requestShardBase
                    + static_cast<uint64_t>(shard) * capacity_],
                count);
            Sync<AscendC::HardEvent::MTE2_V>();
            Sync<AscendC::HardEvent::MTE2_S>();

            // Atlas A2 has no direct int16 -> int32 Cast. Convert through
            // float, which exactly represents every int16 rank and slot.
            AscendC::Cast(
                rankFloatOrCandidate,
                mappingOrGathered,
                AscendC::RoundMode::CAST_NONE,
                partWidth);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                ranksOrOutput,
                rankFloatOrCandidate,
                AscendC::RoundMode::CAST_ROUND,
                partWidth);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Maxs(
                clampedOffsets,
                ranksOrOutput,
                static_cast<int32_t>(0),
                partWidth);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                selectedMask,
                clampedOffsets,
                ranksOrOutput,
                AscendC::CMPMODE::EQ,
                partWidth);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(
                clampedOffsets,
                clampedOffsets,
                static_cast<int32_t>(count - 1),
                partWidth);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                clampedOffsets,
                clampedOffsets,
                static_cast<int32_t>(sizeof(int16_t)),
                partWidth);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Gather(
                mappingOrGathered,
                shardSlots,
                clampedOffsets.ReinterpretCast<uint32_t>(),
                static_cast<uint32_t>(0),
                partWidth);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                gatheredFloat,
                mappingOrGathered,
                AscendC::RoundMode::CAST_NONE,
                partWidth);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                rankFloatOrCandidate.ReinterpretCast<int32_t>(),
                gatheredFloat,
                AscendC::RoundMode::CAST_ROUND,
                partWidth);
            AscendC::PipeBarrier<PIPE_V>();

            // Exactly one value shard owns each selected original position.
            // Invalid (-1) mappings leave the accumulated slot unchanged.
            AscendC::Select(
                accumulatedSlots.ReinterpretCast<float>(),
                selectedMask,
                rankFloatOrCandidate,
                accumulatedSlots.ReinterpretCast<float>(),
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                partWidth);
            AscendC::PipeBarrier<PIPE_V>();
            // The next iteration reuses both int16 MTE2 destinations after
            // the vector pipeline has consumed them.
            Sync<AscendC::HardEvent::V_MTE2>();
        }

        // Preserve unselected/split-boundary positions exactly as the old
        // resident remapper does; selected positions receive their slot.
        AscendC::Maxs(
            clampedOffsets,
            accumulatedSlots,
            static_cast<int32_t>(0),
            partWidth);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            selectedMask,
            clampedOffsets,
            accumulatedSlots,
            AscendC::CMPMODE::EQ,
            partWidth);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            ranksOrOutput.ReinterpretCast<float>(),
            selectedMask,
            accumulatedSlots.ReinterpretCast<float>(),
            input.ReinterpretCast<float>(),
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            partWidth);
        AscendC::PipeBarrier<PIPE_V>();

        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            topkIndices_[requestOffset + begin],
            ranksOrOutput,
            partWidth);
    }

    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int32_t> shardPacked_;
    AscendC::GlobalTensor<int16_t> shardMapping_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::GlobalTensor<int16_t> priorSlots_;
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> survivorTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> survivorSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mergedTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mergedSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> requestMetadataBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateMetadataBuf_;
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

// Split-path remap. One AIV owns one contiguous 1024-position partition,
// so output cachelines have a single writer. Unlike the fused fallback, this
// kernel uses dedicated UB and has no dependency on state-update UB reuse.
class DSAResidentSortedRemapKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* topkIndices,
        __gm__ int16_t* shardMapping,
        __gm__ int32_t* shardCounts,
        __gm__ int16_t* priorSlots,
        uint32_t requestCount,
        uint32_t rowsPerRequest,
        uint32_t rowWidth,
        uint32_t shardCount,
        uint32_t capacity,
        uint32_t shardCountStride,
        uint32_t shardCountRequestStride)
    {
        requestCount_ = requestCount;
        requestWidth_ = rowsPerRequest * rowWidth;
        shardCount_ = shardCount;
        capacity_ = capacity;
        partWidth_ = requestWidth_ / shardCount_;
        shardCountStride_ = shardCountStride;
        shardCountRequestStride_ = shardCountRequestStride;
        topkIndices_.SetGlobalBuffer(
            topkIndices,
            static_cast<uint64_t>(requestCount_) * requestWidth_);
        shardMapping_.SetGlobalBuffer(
            shardMapping,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * requestWidth_);
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            static_cast<uint64_t>(requestCount_)
                * shardCountRequestStride_);
        priorSlots_.SetGlobalBuffer(
            priorSlots,
            static_cast<uint64_t>(requestCount_) * shardCount_
                * capacity_);

        pipe_.InitBuffer(
            inputBuf_, partWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            mappingBuf_, partWidth_ * sizeof(int16_t));
        pipe_.InitBuffer(
            rankFloatBuf_, partWidth_ * sizeof(float));
        pipe_.InitBuffer(
            shardSlotBuf_, capacity_ * sizeof(int16_t));
        pipe_.InitBuffer(
            rankBuf_, partWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            accumulatedBuf_, partWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            offsetBuf_, partWidth_ * sizeof(int32_t));
        pipe_.InitBuffer(
            gatheredFloatBuf_, partWidth_ * sizeof(float));
        pipe_.InitBuffer(selectedMaskBuf_, partWidth_ / 8);
        pipe_.InitBuffer(
            shardMetadataBuf_,
            shardCount_ * shardCountStride_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t request = block / shardCount_;
        const uint32_t part = block % shardCount_;
        if (request >= requestCount_) {
            return;
        }
        const uint32_t begin = part * partWidth_;
        const uint64_t requestOffset =
            static_cast<uint64_t>(request) * requestWidth_;
        const uint64_t mappingBase =
            static_cast<uint64_t>(request) * shardCount_
            * requestWidth_;
        const uint64_t requestShardBase =
            static_cast<uint64_t>(request) * shardCount_ * capacity_;

        auto input = inputBuf_.Get<int32_t>();
        auto mappingOrGathered = mappingBuf_.Get<int16_t>();
        auto rankFloatOrCandidate = rankFloatBuf_.Get<float>();
        auto shardSlots = shardSlotBuf_.Get<int16_t>();
        auto ranksOrOutput = rankBuf_.Get<int32_t>();
        auto accumulatedSlots = accumulatedBuf_.Get<int32_t>();
        auto clampedOffsets = offsetBuf_.Get<int32_t>();
        auto gatheredFloat = gatheredFloatBuf_.Get<float>();
        auto selectedMask = selectedMaskBuf_.Get<uint8_t>();
        auto shardMetadata = shardMetadataBuf_.Get<int32_t>();

        CopyGlobalToLocalExact(
            input,
            topkIndices_[requestOffset + begin],
            partWidth_);
        AscendC::DataCopy(
            shardMetadata,
            shardCounts_[
                static_cast<uint64_t>(request)
                    * shardCountRequestStride_],
            shardCount_ * shardCountStride_);
        Sync<AscendC::HardEvent::MTE2_V>();
        Sync<AscendC::HardEvent::MTE2_S>();
        AscendC::Duplicate(
            accumulatedSlots,
            static_cast<int32_t>(-1),
            partWidth_);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t count = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardCurrentCount));
            if (count == 0) {
                continue;
            }

            CopyGlobalToLocalExact(
                mappingOrGathered,
                shardMapping_[
                    mappingBase
                    + static_cast<uint64_t>(shard) * requestWidth_
                    + begin],
                partWidth_);
            CopyGlobalToLocalExact(
                shardSlots,
                priorSlots_[
                    requestShardBase
                    + static_cast<uint64_t>(shard) * capacity_],
                count);
            Sync<AscendC::HardEvent::MTE2_V>();

            AscendC::Cast(
                rankFloatOrCandidate,
                mappingOrGathered,
                AscendC::RoundMode::CAST_NONE,
                partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                ranksOrOutput,
                rankFloatOrCandidate,
                AscendC::RoundMode::CAST_ROUND,
                partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(
                clampedOffsets,
                ranksOrOutput,
                static_cast<int32_t>(0),
                partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                selectedMask,
                clampedOffsets,
                ranksOrOutput,
                AscendC::CMPMODE::EQ,
                partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(
                clampedOffsets,
                clampedOffsets,
                static_cast<int32_t>(count - 1),
                partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(
                clampedOffsets,
                clampedOffsets,
                static_cast<int32_t>(sizeof(int16_t)),
                partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(
                mappingOrGathered,
                shardSlots,
                clampedOffsets.ReinterpretCast<uint32_t>(),
                static_cast<uint32_t>(0),
                partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                gatheredFloat,
                mappingOrGathered,
                AscendC::RoundMode::CAST_NONE,
                partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                rankFloatOrCandidate.ReinterpretCast<int32_t>(),
                gatheredFloat,
                AscendC::RoundMode::CAST_ROUND,
                partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(
                accumulatedSlots.ReinterpretCast<float>(),
                selectedMask,
                rankFloatOrCandidate,
                accumulatedSlots.ReinterpretCast<float>(),
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                partWidth_);
            AscendC::PipeBarrier<PIPE_V>();
            Sync<AscendC::HardEvent::V_MTE2>();
        }

        AscendC::Maxs(
            clampedOffsets,
            accumulatedSlots,
            static_cast<int32_t>(0),
            partWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            selectedMask,
            clampedOffsets,
            accumulatedSlots,
            AscendC::CMPMODE::EQ,
            partWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            ranksOrOutput.ReinterpretCast<float>(),
            selectedMask,
            accumulatedSlots.ReinterpretCast<float>(),
            input.ReinterpretCast<float>(),
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            partWidth_);
        AscendC::PipeBarrier<PIPE_V>();
        Sync<AscendC::HardEvent::V_MTE3>();
        CopyLocalToGlobalExact(
            topkIndices_[requestOffset + begin],
            ranksOrOutput,
            partWidth_);
    }

private:
    AscendC::GlobalTensor<int32_t> topkIndices_;
    AscendC::GlobalTensor<int16_t> shardMapping_;
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::GlobalTensor<int16_t> priorSlots_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mappingBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rankFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shardSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rankBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulatedBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatheredFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shardMetadataBuf_;
    uint32_t requestCount_ = 0;
    uint32_t requestWidth_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t capacity_ = 0;
    uint32_t partWidth_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t shardCountRequestStride_ = 0;
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
    __gm__ int32_t* shardMissTokens,
    __gm__ int16_t* shardMissPositions,
    __gm__ int16_t* shardEvictableSlots,
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
        priorSlots, shardMissTokens, shardMissPositions,
        shardEvictableSlots,
        requestCount, stateRowCount, dummyStateBase,
        rowsPerRequest, rowWidth, shardCount, shardCapacity,
        shardCountStride, shardCountRequestStride, generationStride);
    op.Process();
}

extern "C" __global__ __aicore__ void
dsa_resident_sorted_read_probe_kernel(
    __gm__ int32_t* shardCounts,
    __gm__ int16_t* priorSlots,
    __gm__ int32_t* debugInfo,
    __gm__ int16_t* priorReadback,
    uint32_t requestCount,
    uint32_t shardCount,
    uint32_t capacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride)
{
    DSAResidentSortedReadProbeKernel op;
    op.Init(
        shardCounts, priorSlots, debugInfo, priorReadback,
        requestCount, shardCount, capacity,
        shardCountStride, shardCountRequestStride);
    op.Process();
}

extern "C" __global__ __aicore__ void
dsa_resident_sorted_finalize_kernel(
    __gm__ int32_t* shardPacked,
    __gm__ int32_t* shardCounts,
    __gm__ int16_t* priorSlots,
    __gm__ int32_t* shardMissTokens,
    __gm__ int16_t* shardMissPositions,
    __gm__ int16_t* shardEvictableSlots,
    __gm__ int32_t* missTokens,
    __gm__ int32_t* missCounts,
    __gm__ int64_t* targetSlots,
    __gm__ int32_t* requestBlockTable,
    __gm__ int32_t* debugInfo,
    uint32_t requestCount,
    uint32_t shardCount,
    uint32_t capacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t missCountStride,
    uint32_t blockTableWidth,
    uint32_t blockSize,
    uint32_t debugStage)
{
    DSAResidentSortedFinalizeKernel op;
    op.Init(
        shardPacked, shardCounts, priorSlots,
        shardMissTokens, shardMissPositions, shardEvictableSlots,
        missTokens, missCounts, targetSlots,
        requestBlockTable, debugInfo,
        requestCount, shardCount, capacity,
        shardCountStride, shardCountRequestStride, missCountStride,
        blockTableWidth, blockSize, debugStage);
    op.Process();
}

extern "C" __global__ __aicore__ void
dsa_resident_sorted_update_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* shardPacked,
    __gm__ int16_t* shardMapping,
    __gm__ int32_t* shardCounts,
    __gm__ int16_t* priorSlots,
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
        priorSlots,
        requestStateIndices, requestStateGenerations,
        stateTokens, stateSlots, stateCounts, stateGenerations,
        requestCount, stateRowCount,
        dummyStateBase, rowsPerRequest, rowWidth, shardCount,
        capacity, shardCountStride, shardCountRequestStride,
        generationStride);
    op.Process();
}

extern "C" __global__ __aicore__ void
dsa_resident_sorted_state_update_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int32_t* shardPacked,
    __gm__ int16_t* shardMapping,
    __gm__ int32_t* shardCounts,
    __gm__ int16_t* priorSlots,
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
        priorSlots,
        requestStateIndices, requestStateGenerations,
        stateTokens, stateSlots, stateCounts, stateGenerations,
        requestCount, stateRowCount,
        dummyStateBase, rowsPerRequest, rowWidth, shardCount,
        capacity, shardCountStride, shardCountRequestStride,
        generationStride);
    op.ProcessStateOnly();
}

extern "C" __global__ __aicore__ void
dsa_resident_sorted_remap_kernel(
    __gm__ int32_t* topkIndices,
    __gm__ int16_t* shardMapping,
    __gm__ int32_t* shardCounts,
    __gm__ int16_t* priorSlots,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t capacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride)
{
    DSAResidentSortedRemapKernel op;
    op.Init(
        topkIndices, shardMapping, shardCounts, priorSlots,
        requestCount, rowsPerRequest, rowWidth, shardCount,
        capacity, shardCountStride, shardCountRequestStride);
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
    void* shardMissTokens,
    void* shardMissPositions,
    void* shardEvictableSlots,
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
        static_cast<int32_t*>(shardMissTokens),
        static_cast<int16_t*>(shardMissPositions),
        static_cast<int16_t*>(shardEvictableSlots),
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
    void* shardMissTokens,
    void* shardMissPositions,
    void* shardEvictableSlots,
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
        static_cast<int32_t*>(shardMissTokens),
        static_cast<int16_t*>(shardMissPositions),
        static_cast<int16_t*>(shardEvictableSlots),
        static_cast<int32_t*>(missTokens),
        static_cast<int32_t*>(missCounts),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int32_t*>(missCounts),
        requestCount, shardCount, capacity, shardCountStride,
        shardCountRequestStride, missCountStride,
        blockTableWidth, blockSize, 0);
    dsa_resident_sorted_update_kernel<<<
        requestCount * shardCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(shardPacked),
        static_cast<int16_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        static_cast<int16_t*>(priorSlots),
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

void dsa_resident_sorted_update_debug_impl(
    void* stream,
    void* topkIndices,
    void* shardPacked,
    void* shardMapping,
    void* shardCounts,
    void* priorSlots,
    void* requestStateIndices,
    void* requestStateGenerations,
    void* stateTokens,
    void* stateSlots,
    void* stateCounts,
    void* stateGenerations,
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
    dsa_resident_sorted_update_kernel<<<
        requestCount * shardCount, nullptr, stream>>>(
        static_cast<int32_t*>(topkIndices),
        static_cast<int32_t*>(shardPacked),
        static_cast<int16_t*>(shardMapping),
        static_cast<int32_t*>(shardCounts),
        static_cast<int16_t*>(priorSlots),
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

void dsa_resident_sorted_plan_no_remap_impl(
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
    void* shardMissTokens,
    void* shardMissPositions,
    void* shardEvictableSlots,
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
            static_cast<int32_t*>(shardMissTokens),
            static_cast<int16_t*>(shardMissPositions),
            static_cast<int16_t*>(shardEvictableSlots),
            static_cast<int32_t*>(missTokens),
            static_cast<int32_t*>(missCounts),
            static_cast<int64_t*>(targetSlots),
            static_cast<int32_t*>(requestBlockTable),
            static_cast<int32_t*>(missCounts),
            requestCount, shardCount, capacity, shardCountStride,
            shardCountRequestStride, missCountStride,
            blockTableWidth, blockSize, 0);
    dsa_resident_sorted_state_update_kernel<<<
        requestCount * shardCount, nullptr, stream>>>(
            static_cast<int32_t*>(topkIndices),
            static_cast<int32_t*>(shardPacked),
            static_cast<int16_t*>(shardMapping),
            static_cast<int32_t*>(shardCounts),
            static_cast<int16_t*>(priorSlots),
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

void dsa_resident_sorted_remap_impl(
    void* stream,
    void* topkIndices,
    void* shardMapping,
    void* shardCounts,
    void* priorSlots,
    uint32_t requestCount,
    uint32_t rowsPerRequest,
    uint32_t rowWidth,
    uint32_t shardCount,
    uint32_t capacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride)
{
    dsa_resident_sorted_remap_kernel<<<
        requestCount * shardCount, nullptr, stream>>>(
            static_cast<int32_t*>(topkIndices),
            static_cast<int16_t*>(shardMapping),
            static_cast<int32_t*>(shardCounts),
            static_cast<int16_t*>(priorSlots),
            requestCount, rowsPerRequest, rowWidth, shardCount,
            capacity, shardCountStride, shardCountRequestStride);
}

void dsa_resident_sorted_read_probe_impl(
    void* stream,
    void* shardCounts,
    void* priorSlots,
    void* debugInfo,
    void* priorReadback,
    uint32_t requestCount,
    uint32_t shardCount,
    uint32_t capacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride)
{
    dsa_resident_sorted_read_probe_kernel<<<
        requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(shardCounts),
        static_cast<int16_t*>(priorSlots),
        static_cast<int32_t*>(debugInfo),
        static_cast<int16_t*>(priorReadback),
        requestCount, shardCount, capacity,
        shardCountStride, shardCountRequestStride);
}

void dsa_resident_sorted_finalize_debug_impl(
    void* stream,
    void* shardPacked,
    void* shardCounts,
    void* priorSlots,
    void* shardMissTokens,
    void* shardMissPositions,
    void* shardEvictableSlots,
    void* missTokens,
    void* missCounts,
    void* targetSlots,
    void* requestBlockTable,
    void* debugInfo,
    uint32_t requestCount,
    uint32_t shardCount,
    uint32_t capacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t missCountStride,
    uint32_t blockTableWidth,
    uint32_t blockSize,
    uint32_t debugStage)
{
    // Test-only boundary isolation: launch the exact production finalize
    // kernel without launching the following state-update/remap kernel.
    dsa_resident_sorted_finalize_kernel<<<
        requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(shardPacked),
        static_cast<int32_t*>(shardCounts),
        static_cast<int16_t*>(priorSlots),
        static_cast<int32_t*>(shardMissTokens),
        static_cast<int16_t*>(shardMissPositions),
        static_cast<int16_t*>(shardEvictableSlots),
        static_cast<int32_t*>(missTokens),
        static_cast<int32_t*>(missCounts),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(requestBlockTable),
        static_cast<int32_t*>(debugInfo),
        requestCount, shardCount, capacity, shardCountStride,
        shardCountRequestStride, missCountStride,
        blockTableWidth, blockSize, debugStage);
}

}  // namespace vllm_ascend
