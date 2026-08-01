/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
constexpr uint32_t kCacheLineBytes = 64;
constexpr uint32_t kInt32PerDataBlock =
    kDataBlockBytes / sizeof(int32_t);
constexpr uint32_t kInt16PerDataBlock =
    kDataBlockBytes / sizeof(int16_t);
constexpr uint32_t kInt32PerCacheLine =
    kCacheLineBytes / sizeof(int32_t);
constexpr uint32_t kMaxResidentShards = 8;

// Existing union/finalize metadata. Keep the production fields unchanged and
// use only the spare entries in each 16-int, cacheline-private shard record.
constexpr uint32_t kShardCurrentCount = 0;
constexpr uint32_t kShardMissCount = 1;
constexpr uint32_t kShardEvictableCount = 2;
constexpr uint32_t kShardOldCount = 3;
constexpr uint32_t kShardSelectedEvictCount = 4;
constexpr uint32_t kShardMissPrefix = 5;
constexpr uint32_t kShardSelectedEvictPrefix = 6;
constexpr uint32_t kShardTotalSelectedEvictCount = 7;
constexpr uint32_t kShardTotalOldCount = 8;
constexpr uint32_t kShardTotalMissCount = 9;

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

// The only request-global stage. It touches O(shard_count) scalar metadata
// and never moves a token/slot payload.
class DSAResidentFinalizeCoordinatorKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* shardCounts,
        __gm__ int32_t* missCounts,
        uint32_t requestCount,
        uint32_t shardCount,
        uint32_t shardCountStride,
        uint32_t shardCountRequestStride,
        uint32_t missCountStride)
    {
        requestCount_ = requestCount;
        shardCount_ = shardCount;
        shardCountStride_ = shardCountStride;
        shardCountRequestStride_ = shardCountRequestStride;
        missCountStride_ = missCountStride;
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            static_cast<uint64_t>(requestCount_)
                * shardCountRequestStride_);
        missCounts_.SetGlobalBuffer(
            missCounts,
            static_cast<uint64_t>(requestCount_) * missCountStride_);
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
        auto shardMetadata = shardMetadataBuf_.Get<int32_t>();
        AscendC::DataCopy(
            shardMetadata,
            shardCounts_[
                static_cast<uint64_t>(request)
                    * shardCountRequestStride_],
            shardCount_ * shardCountStride_);
        Sync<AscendC::HardEvent::MTE2_S>();

        uint32_t missCounts[kMaxResidentShards] = {};
        uint32_t evictableCounts[kMaxResidentShards] = {};
        uint32_t totalMissCount = 0;
        uint32_t totalEvictableCount = 0;
        uint32_t totalOldCount = 0;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            missCounts[shard] = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardMissCount));
            evictableCounts[shard] = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardEvictableCount));
            totalMissCount += missCounts[shard];
            totalEvictableCount += evictableCounts[shard];
            totalOldCount += static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardOldCount));
        }

        const uint32_t totalSelectedEvictCount =
            totalMissCount < totalEvictableCount
            ? totalMissCount
            : totalEvictableCount;
        uint32_t missPrefix = 0;
        uint32_t selectedEvictPrefix = 0;
        uint32_t remainingEvictions = totalSelectedEvictCount;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t selectedCount =
                evictableCounts[shard] < remainingEvictions
                ? evictableCounts[shard]
                : remainingEvictions;
            const uint64_t countOffset =
                static_cast<uint64_t>(request)
                    * shardCountRequestStride_
                + shard * shardCountStride_;
            shardCounts_.SetValue(
                countOffset + kShardSelectedEvictCount,
                static_cast<int32_t>(selectedCount));
            shardCounts_.SetValue(
                countOffset + kShardMissPrefix,
                static_cast<int32_t>(missPrefix));
            shardCounts_.SetValue(
                countOffset + kShardSelectedEvictPrefix,
                static_cast<int32_t>(selectedEvictPrefix));
            shardCounts_.SetValue(
                countOffset + kShardTotalSelectedEvictCount,
                static_cast<int32_t>(totalSelectedEvictCount));
            shardCounts_.SetValue(
                countOffset + kShardTotalOldCount,
                static_cast<int32_t>(totalOldCount));
            shardCounts_.SetValue(
                countOffset + kShardTotalMissCount,
                static_cast<int32_t>(totalMissCount));
            missPrefix += missCounts[shard];
            selectedEvictPrefix += selectedCount;
            remainingEvictions -= selectedCount;
        }
        missCounts_.SetValue(
            static_cast<uint64_t>(request) * missCountStride_,
            static_cast<int32_t>(totalMissCount));
    }

private:
    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::GlobalTensor<int32_t> missCounts_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shardMetadataBuf_;
    uint32_t requestCount_ = 0;
    uint32_t shardCount_ = 0;
    uint32_t shardCountStride_ = 0;
    uint32_t shardCountRequestStride_ = 0;
    uint32_t missCountStride_ = 0;
};

// A copied and reduced finalize worker. Every request/shard AICore repeats the
// tiny O(shard_count) prefix calculation, then writes only its own 64-byte
// metadata record. One AICore owns one request/shard prior-slot slice and a
// cacheline-aligned partition of the compact LMCache payload, so sibling
// writers never share a cacheline even when shard miss counts are not aligned.
class DSAResidentShardedFinalizeWorkerKernel {
public:
    __aicore__ inline void Init(
        __gm__ int32_t* shardCounts,
        __gm__ int16_t* priorSlots,
        __gm__ int32_t* shardMissTokens,
        __gm__ int16_t* shardMissPositions,
        __gm__ int16_t* shardEvictableSlots,
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
        blockTableEntries_ =
            (capacity_ + blockSize_ - 1) / blockSize_;
        const uint64_t requestShardElements =
            static_cast<uint64_t>(requestCount_) * shardCount_
            * capacity_;
        const uint64_t requestElements =
            static_cast<uint64_t>(requestCount_) * capacity_;
        shardCounts_.SetGlobalBuffer(
            shardCounts,
            static_cast<uint64_t>(requestCount_)
                * shardCountRequestStride_);
        priorSlots_.SetGlobalBuffer(priorSlots, requestShardElements);
        shardMissTokens_.SetGlobalBuffer(
            shardMissTokens, requestShardElements);
        shardMissPositions_.SetGlobalBuffer(
            shardMissPositions, requestShardElements);
        shardEvictableSlots_.SetGlobalBuffer(
            shardEvictableSlots, requestShardElements);
        missTokens_.SetGlobalBuffer(missTokens, requestElements);
        missCounts_.SetGlobalBuffer(
            missCounts,
            static_cast<uint64_t>(requestCount_) * missCountStride_);
        targetSlots_.SetGlobalBuffer(targetSlots, requestElements);
        requestBlockTable_.SetGlobalBuffer(
            requestBlockTable,
            static_cast<uint64_t>(requestCount_)
                * blockTableWidth_);

        const uint32_t packedInt16Elements =
            capacity_ + shardCount_ * kInt16PerDataBlock;
        const uint32_t packedInt32Elements =
            capacity_ + shardCount_ * kInt32PerDataBlock;
        const uint32_t maxOutputElements =
            (capacity_ + shardCount_ - 1) / shardCount_
            + kInt32PerCacheLine;
        pipe_.InitBuffer(
            priorSlotBuf_, capacity_ * sizeof(int16_t));
        pipe_.InitBuffer(
            missPositionBuf_, capacity_ * sizeof(int16_t));
        pipe_.InitBuffer(
            packedMissTokenBuf_,
            packedInt32Elements * sizeof(int32_t));
        pipe_.InitBuffer(
            packedEvictSlotBuf_,
            packedInt16Elements * sizeof(int16_t));
        pipe_.InitBuffer(
            outputMissTokenBuf_,
            maxOutputElements * sizeof(int32_t));
        pipe_.InitBuffer(
            outputTargetSlotBuf_,
            maxOutputElements * sizeof(int64_t));
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
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t request = block / shardCount_;
        const uint32_t ownerShard = block % shardCount_;
        if (request >= requestCount_) {
            return;
        }
        const uint64_t ownerCountOffset =
            static_cast<uint64_t>(request)
                * shardCountRequestStride_
            + ownerShard * shardCountStride_;
        auto shardMetadata = shardMetadataBuf_.Get<int32_t>();
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
        uint32_t missPrefixes[kMaxResidentShards] = {};
        uint32_t missOffsets[kMaxResidentShards] = {};
        uint32_t totalMissCount = 0;
        uint32_t totalEvictableCount = 0;
        uint32_t totalOldCount = 0;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            currentCounts[shard] = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardCurrentCount));
            missCounts[shard] = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardMissCount));
            evictableCounts[shard] = static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardEvictableCount));
            missPrefixes[shard] = totalMissCount;
            totalMissCount += missCounts[shard];
            totalEvictableCount += evictableCounts[shard];
            totalOldCount += static_cast<uint32_t>(
                shardMetadata.GetValue(
                    shard * shardCountStride_
                    + kShardOldCount));
        }
        const uint32_t totalSelectedEvictCount =
            totalMissCount < totalEvictableCount
            ? totalMissCount
            : totalEvictableCount;
        uint32_t selectedEvictPrefix = 0;
        uint32_t remainingEvictions = totalSelectedEvictCount;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            selectedEvictPrefixes_[shard] = selectedEvictPrefix;
            selectedEvictCounts_[shard] =
                evictableCounts[shard] < remainingEvictions
                ? evictableCounts[shard]
                : remainingEvictions;
            selectedEvictPrefix += selectedEvictCounts_[shard];
            remainingEvictions -= selectedEvictCounts_[shard];
        }

        shardCounts_.SetValue(
            ownerCountOffset + kShardSelectedEvictCount,
            static_cast<int32_t>(selectedEvictCounts_[ownerShard]));
        shardCounts_.SetValue(
            ownerCountOffset + kShardMissPrefix,
            static_cast<int32_t>(missPrefixes[ownerShard]));
        shardCounts_.SetValue(
            ownerCountOffset + kShardSelectedEvictPrefix,
            static_cast<int32_t>(selectedEvictPrefixes_[ownerShard]));
        shardCounts_.SetValue(
            ownerCountOffset + kShardTotalSelectedEvictCount,
            static_cast<int32_t>(totalSelectedEvictCount));
        shardCounts_.SetValue(
            ownerCountOffset + kShardTotalOldCount,
            static_cast<int32_t>(totalOldCount));
        shardCounts_.SetValue(
            ownerCountOffset + kShardTotalMissCount,
            static_cast<int32_t>(totalMissCount));
        if (ownerShard == 0) {
            missCounts_.SetValue(
                static_cast<uint64_t>(request) * missCountStride_,
                static_cast<int32_t>(totalMissCount));
        }
        if (totalMissCount == 0) {
            return;
        }

        const uint64_t requestShardBase =
            static_cast<uint64_t>(request) * shardCount_ * capacity_;
        const uint64_t ownerShardOffset =
            requestShardBase
            + static_cast<uint64_t>(ownerShard) * capacity_;
        auto priorSlots = priorSlotBuf_.Get<int16_t>();
        auto missPositions = missPositionBuf_.Get<int16_t>();
        auto packedMissTokens = packedMissTokenBuf_.Get<int32_t>();
        auto packedEvictSlots = packedEvictSlotBuf_.Get<int16_t>();
        auto outputMissTokens = outputMissTokenBuf_.Get<int32_t>();
        auto outputTargetSlots = outputTargetSlotBuf_.Get<int64_t>();
        auto blockTable = blockTableBuf_.Get<int32_t>();

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

        const uint32_t ownerCurrentCount = currentCounts[ownerShard];
        const uint32_t ownerMissCount = missCounts[ownerShard];
        if (ownerCurrentCount > 0) {
            CopyGlobalToLocalExact(
                priorSlots,
                priorSlots_[ownerShardOffset],
                ownerCurrentCount);
        }
        if (ownerMissCount > 0) {
            CopyGlobalToLocalExact(
                missPositions,
                shardMissPositions_[ownerShardOffset],
                ownerMissCount);
        }

        uint32_t missEnd = 0;
        uint32_t evictEnd = 0;
        for (uint32_t shard = 0; shard < shardCount_; ++shard) {
            const uint32_t missOffset =
                (missEnd + kInt32PerDataBlock - 1)
                & ~(kInt32PerDataBlock - 1);
            const uint32_t evictOffset =
                (evictEnd + kInt16PerDataBlock - 1)
                & ~(kInt16PerDataBlock - 1);
            const uint64_t shardOffset =
                requestShardBase
                + static_cast<uint64_t>(shard) * capacity_;
            missOffsets[shard] = missOffset;
            evictOffsets_[shard] = evictOffset;
            if (missCounts[shard] > 0) {
                CopyGlobalToLocalExact(
                    packedMissTokens[missOffset],
                    shardMissTokens_[shardOffset],
                    missCounts[shard]);
            }
            if (selectedEvictCounts_[shard] > 0) {
                CopyGlobalToLocalExact(
                    packedEvictSlots[evictOffset],
                    shardEvictableSlots_[shardOffset],
                    selectedEvictCounts_[shard]);
            }
            missEnd = missOffset + missCounts[shard];
            evictEnd = evictOffset + selectedEvictCounts_[shard];
        }
        Sync<AscendC::HardEvent::MTE2_S>();

        for (uint32_t localMiss = 0;
             localMiss < ownerMissCount;
             ++localMiss) {
            const uint32_t globalMiss =
                missPrefixes[ownerShard] + localMiss;
            const int16_t slot = ResolveSlot(
                globalMiss,
                totalSelectedEvictCount,
                totalOldCount,
                packedEvictSlots);
            const uint32_t position = static_cast<uint32_t>(
                missPositions.GetValue(localMiss));
            priorSlots.SetValue(position, slot);
        }

        const uint32_t totalCacheLines =
            (totalMissCount + kInt32PerCacheLine - 1)
            / kInt32PerCacheLine;
        const uint32_t baseLines = totalCacheLines / shardCount_;
        const uint32_t extraLines = totalCacheLines % shardCount_;
        const uint32_t startLine =
            ownerShard * baseLines
            + (ownerShard < extraLines ? ownerShard : extraLines);
        const uint32_t ownedLines =
            baseLines + (ownerShard < extraLines ? 1 : 0);
        const uint32_t outputBegin =
            startLine * kInt32PerCacheLine;
        uint32_t outputEnd = outputBegin;
        if (outputBegin < totalMissCount) {
            outputEnd =
                outputBegin + ownedLines * kInt32PerCacheLine;
            if (outputEnd > totalMissCount) {
                outputEnd = totalMissCount;
            }
        }
        const uint32_t outputCount = outputEnd - outputBegin;
        for (uint32_t outputIndex = 0;
             outputIndex < outputCount;
             ++outputIndex) {
            const uint32_t globalMiss = outputBegin + outputIndex;
            uint32_t sourceShard = 0;
            while (
                sourceShard + 1 < shardCount_ &&
                globalMiss >= missPrefixes[sourceShard + 1]) {
                ++sourceShard;
            }
            const uint32_t sourceIndex =
                globalMiss - missPrefixes[sourceShard];
            outputMissTokens.SetValue(
                outputIndex,
                packedMissTokens.GetValue(
                    missOffsets[sourceShard] + sourceIndex));
            const int16_t slot = ResolveSlot(
                globalMiss,
                totalSelectedEvictCount,
                totalOldCount,
                packedEvictSlots);
            const uint32_t logicalSlot = static_cast<uint32_t>(slot);
            const uint32_t logicalBlock = logicalSlot / blockSize_;
            const uint32_t blockOffset = logicalSlot % blockSize_;
            const int32_t physicalBlock =
                blockTable.GetValue(logicalBlock);
            outputTargetSlots.SetValue(
                outputIndex,
                static_cast<int64_t>(physicalBlock) * blockSize_
                    + blockOffset);
        }

        if (ownerMissCount == 0 && outputCount == 0) {
            return;
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        if (ownerMissCount > 0) {
            CopyLocalToGlobalExact(
                priorSlots_[ownerShardOffset],
                priorSlots,
                ownerCurrentCount);
        }
        if (outputCount > 0) {
            const uint64_t requestOffset =
                static_cast<uint64_t>(request) * capacity_;
            CopyLocalToGlobalExact(
                missTokens_[requestOffset + outputBegin],
                outputMissTokens,
                outputCount);
            CopyLocalToGlobalExact(
                targetSlots_[requestOffset + outputBegin],
                outputTargetSlots,
                outputCount);
        }
    }

private:
    __aicore__ inline int16_t ResolveSlot(
        uint32_t globalMiss,
        uint32_t totalSelectedEvictCount,
        uint32_t totalOldCount,
        AscendC::LocalTensor<int16_t> packedEvictSlots)
    {
        if (globalMiss >= totalSelectedEvictCount) {
            return static_cast<int16_t>(
                totalOldCount + globalMiss
                - totalSelectedEvictCount);
        }
        uint32_t evictShard = 0;
        while (
            evictShard + 1 < shardCount_ &&
            globalMiss >=
                selectedEvictPrefixes_[evictShard]
                    + selectedEvictCounts_[evictShard]) {
            ++evictShard;
        }
        return packedEvictSlots.GetValue(
            evictOffsets_[evictShard]
                + globalMiss
                - selectedEvictPrefixes_[evictShard]);
    }

    AscendC::GlobalTensor<int32_t> shardCounts_;
    AscendC::GlobalTensor<int16_t> priorSlots_;
    AscendC::GlobalTensor<int32_t> shardMissTokens_;
    AscendC::GlobalTensor<int16_t> shardMissPositions_;
    AscendC::GlobalTensor<int16_t> shardEvictableSlots_;
    AscendC::GlobalTensor<int32_t> missTokens_;
    AscendC::GlobalTensor<int32_t> missCounts_;
    AscendC::GlobalTensor<int64_t> targetSlots_;
    AscendC::GlobalTensor<int32_t> requestBlockTable_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> priorSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> missPositionBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedMissTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> packedEvictSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputMissTokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputTargetSlotBuf_;
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
    uint32_t selectedEvictCounts_[kMaxResidentShards] = {};
    uint32_t selectedEvictPrefixes_[kMaxResidentShards] = {};
    uint32_t evictOffsets_[kMaxResidentShards] = {};
};

extern "C" __global__ __aicore__ void
dsa_resident_finalize_coordinator_kernel(
    __gm__ int32_t* shardCounts,
    __gm__ int32_t* missCounts,
    uint32_t requestCount,
    uint32_t shardCount,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t missCountStride)
{
    DSAResidentFinalizeCoordinatorKernel op;
    op.Init(
        shardCounts, missCounts, requestCount, shardCount,
        shardCountStride, shardCountRequestStride, missCountStride);
    op.Process();
}

extern "C" __global__ __aicore__ void
dsa_resident_sharded_finalize_worker_kernel(
    __gm__ int32_t* shardCounts,
    __gm__ int16_t* priorSlots,
    __gm__ int32_t* shardMissTokens,
    __gm__ int16_t* shardMissPositions,
    __gm__ int16_t* shardEvictableSlots,
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
    DSAResidentShardedFinalizeWorkerKernel op;
    op.Init(
        shardCounts, priorSlots,
        shardMissTokens, shardMissPositions, shardEvictableSlots,
        missTokens, missCounts, targetSlots, requestBlockTable,
        requestCount, shardCount, capacity,
        shardCountStride, shardCountRequestStride,
        missCountStride,
        blockTableWidth, blockSize);
    op.Process();
}

}  // namespace

namespace vllm_ascend {

void dsa_resident_finalize_coordinator_impl(
    void* stream,
    void* shardCounts,
    void* missCounts,
    uint32_t requestCount,
    uint32_t shardCount,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t missCountStride)
{
    dsa_resident_finalize_coordinator_kernel<<<
        requestCount, nullptr, stream>>>(
        static_cast<int32_t*>(shardCounts),
        static_cast<int32_t*>(missCounts),
        requestCount, shardCount, shardCountStride,
        shardCountRequestStride, missCountStride);
}

void dsa_resident_sharded_finalize_worker_impl(
    void* stream,
    void* shardCounts,
    void* priorSlots,
    void* shardMissTokens,
    void* shardMissPositions,
    void* shardEvictableSlots,
    void* missTokens,
    void* missCounts,
    void* targetSlots,
    void* requestBlockTable,
    uint32_t requestCount,
    uint32_t shardCount,
    uint32_t capacity,
    uint32_t shardCountStride,
    uint32_t shardCountRequestStride,
    uint32_t missCountStride,
    uint32_t blockTableWidth,
    uint32_t blockSize)
{
    dsa_resident_sharded_finalize_worker_kernel<<<
        requestCount * shardCount, nullptr, stream>>>(
        static_cast<int32_t*>(shardCounts),
        static_cast<int16_t*>(priorSlots),
        static_cast<int32_t*>(shardMissTokens),
        static_cast<int16_t*>(shardMissPositions),
        static_cast<int16_t*>(shardEvictableSlots),
        static_cast<int32_t*>(missTokens),
        static_cast<int32_t*>(missCounts),
        static_cast<int64_t*>(targetSlots),
        static_cast<int32_t*>(requestBlockTable),
        requestCount, shardCount, capacity,
        shardCountStride, shardCountRequestStride,
        missCountStride,
        blockTableWidth, blockSize);
}

}  // namespace vllm_ascend
