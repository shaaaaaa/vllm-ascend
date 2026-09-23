// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common.h"
namespace redesign {
template<bool CopyKV, bool BatchedCopy = false> class Lookup {
    RedesignLaunch a;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateBuf, tableBuf, workBuf, kvBuf;
    AscendC::GlobalTensor<int32_t> current, revision, tags, versions, ready, meta, table, output;
    AscendC::GlobalTensor<int64_t> epochs;
    AscendC::GlobalTensor<uint8_t> oldKV, denseKV, newKV;
    AscendC::LocalTensor<int32_t> oldTags, oldVersions, oldReady, localTable;
    AscendC::LocalTensor<int32_t> token, version, candidate, offset, tmp, gathered, indices, result;
    AscendC::LocalTensor<float> chosen, candidateFloat;
    AscendC::LocalTensor<uint8_t> mask, check;
    uint32_t n;
    __aicore__ inline void Verify() {
        Range(mask, candidate, tmp, 0, n, T);
        AscendC::Maxs(offset, candidate, static_cast<int32_t>(0), T); V();
        AscendC::Mins(offset, offset, static_cast<int32_t>(n - 1), T); V();
        AscendC::Muls(offset, offset, static_cast<int32_t>(4), T); V();
        AscendC::Gather(gathered, oldTags, offset.ReinterpretCast<uint32_t>(), 0, T); V();
        AscendC::Compare(check, gathered, token, AscendC::CMPMODE::EQ, T); V(); And(mask, check, T);
        AscendC::Gather(gathered, oldVersions, offset.ReinterpretCast<uint32_t>(), 0, T); V();
        AscendC::Compare(check, gathered, version, AscendC::CMPMODE::EQ, T); V(); And(mask, check, T);
        AscendC::Gather(gathered, oldReady, offset.ReinterpretCast<uint32_t>(), 0, T); V();
        AscendC::Duplicate(tmp, static_cast<int32_t>(1), T); V();
        AscendC::Compare(check, gathered, tmp, AscendC::CMPMODE::EQ, T); V(); And(mask, check, T);
    }
    __aicore__ inline void SelectCandidate() {
        AscendC::Cast(candidateFloat, candidate, AscendC::RoundMode::CAST_NONE, T); V();
        AscendC::Select(chosen, mask, candidateFloat, chosen,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, T); V();
    }
public:
    __aicore__ inline void Init(RedesignLaunch args) {
        a = args; n = a.queries * a.topk;
        current.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[0]));
        revision.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[1]));
        tags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[2]));
        versions.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[3]));
        ready.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[4]));
        meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[5]));
        epochs.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(a.p[6]));
        table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[7]));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[8]));
        oldKV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a.p[9]));
        denseKV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a.p[10]));
        newKV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a.p[11]));
        pipe.InitBuffer(stateBuf, 3 * n * 4);
        pipe.InitBuffer(tableBuf, W * 4);
        pipe.InitBuffer(workBuf, 10 * T * 4 + 2 * (T / 8));
        if constexpr (CopyKV) pipe.InitBuffer(kvBuf, BatchedCopy ? 16384 : a.rowBytes);
        oldTags = stateBuf.Get<int32_t>(); oldVersions = oldTags[n]; oldReady = oldVersions[n];
        localTable = tableBuf.Get<int32_t>(); token = workBuf.Get<int32_t>();
        version = token[T]; candidate = version[T]; offset = candidate[T]; tmp = offset[T];
        gathered = tmp[T]; indices = gathered[T]; result = indices[T];
        chosen = result[T].ReinterpretCast<float>(); candidateFloat = chosen[T];
        mask = candidateFloat[T].ReinterpretCast<uint8_t>(); check = mask[T / 8];
    }
    __aicore__ inline void Process() {
        uint32_t tiles = n / T;
        for (uint32_t block = AscendC::GetBlockIdx(); block < a.requests * tiles;
             block += AscendC::GetBlockNum()) {
            uint32_t request = block / tiles, start = (block % tiles) * T;
            uint32_t row = start / a.topk, rank = start % a.topk;
            uint64_t base = static_cast<uint64_t>(request) * n + start;
            uint64_t m = (static_cast<uint64_t>(request) * a.queries + row) * 16;
            int32_t active = Fresh(meta, m), boundary = meta.GetValue(m + 1), length = meta.GetValue(m + 2);
            int64_t epoch = Fresh(epochs, static_cast<uint64_t>(request) * 8);
            bool sameEpoch = epoch == epochs.GetValue(static_cast<uint64_t>(request) * 8 + 1);
            bool good = active == 1 && boundary >= 0 && boundary <= length && length <= a.universe;
            AscendC::DataCopy(token, current[base], T);
            if (a.mode != 0) AscendC::DataCopy(version, revision[base], T);
            if (good && sameEpoch && a.mode != 0) {
                uint64_t oldBase = a.mode == 1 ? base : static_cast<uint64_t>(request) * n;
                uint32_t oldCount = a.mode == 1 ? T : n;
                AscendC::DataCopy(oldTags, tags[oldBase], oldCount);
                AscendC::DataCopy(oldVersions, versions[oldBase], oldCount);
                AscendC::DataCopy(oldReady, ready[oldBase], oldCount);
            }
            Fence<AscendC::HardEvent::MTE2_V>();
            AscendC::Duplicate(chosen, -1.0F, T); V();
            if (good && sameEpoch && a.mode != 0) {
                if (a.mode == 1) {
                    // Fully contiguous fast path: no full-snapshot loads,
                    // gathers, directory build, sorting, or compaction.
                    AscendC::Compare(mask, oldTags, token, AscendC::CMPMODE::EQ, T); V();
                    AscendC::Compare(check, oldVersions, version, AscendC::CMPMODE::EQ, T); V(); And(mask, check, T);
                    AscendC::Duplicate(tmp, static_cast<int32_t>(1), T); V();
                    AscendC::Compare(check, oldReady, tmp, AscendC::CMPMODE::EQ, T); V(); And(mask, check, T);
                    AscendC::CreateVecIndex(candidate, static_cast<int32_t>(start), T); V();
                    SelectCandidate();
                } else if (a.mode == 2) {
                    // Bounded rank-displacement search across query rows.
                    uint32_t rows = a.mode == 1 ? 1 : a.queries;
                    uint32_t radius = a.mode == 1 ? 0 : a.radius;
                    AscendC::CreateVecIndex(indices, static_cast<int32_t>(rank), T); V();
                    for (uint32_t other = 0; other < rows; ++other) {
                        uint32_t previousRow = (row + other) % a.queries;
                        for (int32_t delta = -static_cast<int32_t>(radius);
                             delta <= static_cast<int32_t>(radius); ++delta) {
                            AscendC::Adds(candidate, indices, static_cast<int32_t>(previousRow * a.topk) + delta, T); V();
                            Verify();
                            // Do not wrap a window across either query-row edge.
                            AscendC::Adds(gathered, indices, delta, T); V();
                            Range(check, gathered, tmp, 0, a.topk, T); And(mask, check, T);
                            SelectCandidate();
                        }
                    }
                } else {
                    // Large directories stream through UB windows. Hash tables
                    // fit in a single window. Table construction is a timed task.
                    uint32_t window = a.tableSize < W ? a.tableSize : W;
                    for (uint32_t low = 0; low < a.tableSize; low += window) {
                        uint32_t count = a.tableSize - low < window ? a.tableSize - low : window;
                        AscendC::DataCopy(localTable, table[static_cast<uint64_t>(request) * a.tableSize + low], count);
                        Fence<AscendC::HardEvent::MTE2_V>();
                        if (a.mode == 3) Mod(offset, token, tmp, a.tableSize, T);
                        else { AscendC::Adds(offset, token, -static_cast<int32_t>(low), T); V(); }
                        AscendC::Maxs(offset, offset, static_cast<int32_t>(0), T); V();
                        AscendC::Mins(offset, offset, static_cast<int32_t>(count - 1), T); V();
                        AscendC::Muls(offset, offset, static_cast<int32_t>(4), T); V();
                        AscendC::Gather(candidate, localTable, offset.ReinterpretCast<uint32_t>(), 0, T); V();
                        Verify();
                        if (a.mode == 4) { Range(check, token, tmp, low, low + count, T); And(mask, check, T); }
                        SelectCandidate();
                        Fence<AscendC::HardEvent::V_MTE2>();
                    }
                }
            }
            if (good) {
                Range(mask, token, tmp, 0, boundary, T);
                AscendC::Select(chosen, mask, chosen, -3.0F,
                               AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, T); V();
                Range(mask, token, tmp, 0, length, T);
                AscendC::Select(chosen, mask, chosen, -2.0F,
                               AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, T); V();
            } else { AscendC::Duplicate(chosen, -2.0F, T); V(); }
            AscendC::Cast(result, chosen, AscendC::RoundMode::CAST_RINT, T); V();
            Fence<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(output[base], result, T);
            if constexpr (CopyKV && BatchedCopy) {
                Fence<AscendC::HardEvent::V_S>();
                auto payload = kvBuf.Get<uint8_t>();
                uint32_t rows = 16384 / a.rowBytes;
                if (rows > 16) rows = 16;
                for (uint32_t first = 0; first < T; first += rows) {
                    uint32_t count = T - first < rows ? T - first : rows;
                    // Zero invalid rows before disjoint valid-row DMA writes.
                    AscendC::Duplicate(payload.ReinterpretCast<uint32_t>(),
                                       static_cast<uint32_t>(0), count * a.rowBytes / 4); V();
                    Fence<AscendC::HardEvent::V_MTE2>();
                    Fence<AscendC::HardEvent::V_MTE3>();
                    for (uint32_t j = 0; j < count; ++j) {
                        int32_t source = result.GetValue(first + j);
                        if (source == -2) continue;
                        uint64_t src = source >= 0
                            ? (static_cast<uint64_t>(request) * n + source) * a.rowBytes
                            : (static_cast<uint64_t>(request) * a.universe + token.GetValue(first + j)) * a.rowBytes;
                        if (source >= 0) AscendC::DataCopy(payload[j * a.rowBytes], oldKV[src], a.rowBytes);
                        else AscendC::DataCopy(payload[j * a.rowBytes], denseKV[src], a.rowBytes);
                    }
                    // One completion fence and contiguous output per group.
                    Fence<AscendC::HardEvent::MTE2_MTE3>();
                    AscendC::DataCopy(newKV[(base + first) * a.rowBytes], payload, count * a.rowBytes);
                    Fence<AscendC::HardEvent::MTE3_MTE2>();
                    Fence<AscendC::HardEvent::MTE3_V>();
                }
            } else if constexpr (CopyKV) {
                Fence<AscendC::HardEvent::V_S>();
                auto payload = kvBuf.Get<uint8_t>();
                // Read-only old bank and disjoint output bank; no hit can be
                // evicted by an earlier miss. Original occurrence order survives.
                for (uint32_t j = 0; j < T; ++j) {
                    int32_t source = result.GetValue(j);
                    if (source == -2) {
                        AscendC::Duplicate(payload.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), a.rowBytes / 4); V();
                        Fence<AscendC::HardEvent::V_MTE3>();
                    } else {
                        uint64_t src = source >= 0
                            ? (static_cast<uint64_t>(request) * n + source) * a.rowBytes
                            : (static_cast<uint64_t>(request) * a.universe + token.GetValue(j)) * a.rowBytes;
                        if (source >= 0) AscendC::DataCopy(payload, oldKV[src], a.rowBytes);
                        else AscendC::DataCopy(payload, denseKV[src], a.rowBytes);
                        Fence<AscendC::HardEvent::MTE2_MTE3>();
                    }
                    AscendC::DataCopy(newKV[(base + j) * a.rowBytes], payload, a.rowBytes);
                    Fence<AscendC::HardEvent::MTE3_MTE2>();
                    Fence<AscendC::HardEvent::MTE3_V>();
                }
            }
            // The next grid-stride tile must not reuse UB while writes or
            // vector reads of the preceding tile remain outstanding.
            Fence<AscendC::HardEvent::MTE3_V>();
            Fence<AscendC::HardEvent::V_MTE2>();
        }
    }
};
} // namespace redesign
