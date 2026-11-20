// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifdef NDEBUG
#error "Selected-KV kernels require assertions in Release: compile with -UNDEBUG"
#endif
#if !defined(ASCENDC_DUMP) || ASCENDC_DUMP != 1
#error "Selected-KV kernels require ASCENDC_DUMP=1"
#endif

#include "kernel_operator.h"
#include <cstdint>

#ifndef ASSERT_MSG
#error "Selected-KV kernels require the SDK device ASSERT_MSG API"
#endif

namespace SparsePrefillCopy {

using namespace AscendC;
using CopyQueue = TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2>;

constexpr uint32_t PAGE_ROWS = 128U, CORES = 32U, MAX_LOCAL_IDS = 16384U, MAX_COLUMNS = 4096U, MAX_RUN = 32U;
constexpr uint32_t NOPE_ROW_BYTES = 1024U, PE_ROW_BYTES = 128U;
constexpr int32_t MAX_HISTORY = 524288, MAX_BATCHED_TOKENS = 16384;

enum class RowState : uint8_t { Invalid, Full, Selected };
enum class SliceState : uint8_t { Abort, Empty, Ready };

struct DualRowAddress {
    uint64_t nopeSource = 0, peSource = 0, nopeDestination = 0, peDestination = 0;
};

template <HardEvent E>
__aicore__ inline void Fence() {
    const auto ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
    SetFlag<E>(ev);
    WaitFlag<E>(ev);
}

template <class T>
__aicore__ inline void BindGm(GlobalTensor<T>& g, GM_ADDR p) {
    g.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p));
}

template <class T>
__aicore__ inline T LoadGmScalar(GlobalTensor<T>& g, uint64_t idx) {
    DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g[idx]);
    return g.GetValue(idx);
}

__aicore__ inline uint32_t HistoryPages(uint32_t history) {
    return (history + PAGE_ROWS - 1U) / PAGE_ROWS;
}

__aicore__ inline bool ValidDomain(int32_t id, uint32_t history, int32_t page, uint32_t sp, uint32_t dp) {
    return id >= 0 && static_cast<uint32_t>(id) < history && page >= 0 &&
           static_cast<uint32_t>(page) < sp && static_cast<uint32_t>(page) < dp;
}

__aicore__ inline bool AddOffset(uint64_t base, uint32_t row, uint32_t bytes, uint64_t& out) {
    const uint64_t off = static_cast<uint64_t>(row) * bytes;
    if (base == 0U || base > UINT64_MAX - off - (bytes - 1U)) {
        return false;
    }
    out = base + off;
    return true;
}

__aicore__ inline bool AdjacentPlane(uint64_t s0, uint64_t d0, uint64_t s1, uint64_t d1, uint64_t bytes) {
    return s1 >= s0 && s1 - s0 == bytes && d1 >= d0 && d1 - d0 == bytes;
}

__aicore__ inline RowState DecodeMlaRow(
    int32_t id, uint32_t history, int32_t page, uint32_t sp, uint32_t dp, uint64_t nb, uint64_t pb, DualRowAddress& out
) {
    if (!ValidDomain(id, history, page, sp, dp)) {
        return RowState::Invalid;
    }
    const uint32_t row = static_cast<uint32_t>(id) % PAGE_ROWS;
    const uint64_t slot = static_cast<uint64_t>(page) * PAGE_ROWS + row;
    out = {0U, 0U, slot * NOPE_ROW_BYTES, slot * PE_ROW_BYTES};
    if (nb == 0U && pb == 0U) {
        return RowState::Full;
    }
    uint64_t ns = 0U, ps = 0U;
    if (!AddOffset(nb, row, NOPE_ROW_BYTES, ns) || !AddOffset(pb, row, PE_ROW_BYTES, ps)) {
        return RowState::Invalid;
    }
    out.nopeSource = ns;
    out.peSource = ps;
    return RowState::Selected;
}

__aicore__ inline bool AdjacentMla(const DualRowAddress& cur, const DualRowAddress& next, uint32_t run) {
    return AdjacentPlane(cur.nopeSource, cur.nopeDestination, next.nopeSource, next.nopeDestination, static_cast<uint64_t>(run) * NOPE_ROW_BYTES) &&
           AdjacentPlane(cur.peSource, cur.peDestination, next.peSource, next.peDestination, static_cast<uint64_t>(run) * PE_ROW_BYTES);
}

__aicore__ inline void CopyIn(CopyQueue& q, GlobalTensor<uint8_t>& src, uint32_t bytes) {
    auto t = q.AllocTensor<uint8_t>();
    DataCopyPad(t, src, {1, bytes, 0, 0, 0}, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
    q.EnQue(t);
}

__aicore__ inline void CopyOut(CopyQueue& q, GlobalTensor<uint8_t>& dst, uint64_t off, uint32_t bytes) {
    auto t = q.DeQue<uint8_t>();
    DataCopyPad(dst[off], t, {1, bytes, 0, 0, 0});
    q.FreeTensor(t);
}

__aicore__ inline void DiscardPending(CopyQueue& q) {
    auto t = q.DeQue<uint8_t>();
    q.FreeTensor(t);
}

__aicore__ inline void AbortCopy(CopyQueue& q0, CopyQueue& q1, uint32_t pending) {
    if (pending != 0U) {
        DiscardPending(q0);
        DiscardPending(q1);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void FlushCopy(
    CopyQueue& q0, CopyQueue& q1, GlobalTensor<uint8_t>& d0, GlobalTensor<uint8_t>& d1, const DualRowAddress& addr, uint32_t rows
) {
    CopyOut(q0, d0, addr.nopeDestination, rows * NOPE_ROW_BYTES);
    CopyOut(q1, d1, addr.peDestination, rows * PE_ROW_BYTES);
}

__aicore__ inline RowState ResolveCopyRow(
    int32_t id, int32_t h, LocalTensor<int32_t>& table, GlobalTensor<uint64_t>& pages,
    uint32_t sp, uint32_t dp, uint32_t& col, int32_t& page, uint64_t& b0, uint64_t& b1, DualRowAddress& out
) {
    ASSERT_MSG(id >= 0 && id < h, "Selected copy token outside history");
    if (id < 0 || id >= h) {
        return RowState::Invalid;
    }
    const uint32_t c = static_cast<uint32_t>(id) / PAGE_ROWS;
    if (c != col) {
        const int32_t p = table.GetValue(c);
        const bool ok = p >= 0 && static_cast<uint32_t>(p) < sp && static_cast<uint32_t>(p) < dp;
        ASSERT_MSG(ok, "Selected copy page outside source or destination domain");
        if (!ok) {
            return RowState::Invalid;
        }
        col = c;
        page = p;
        const uint64_t idx = static_cast<uint64_t>(static_cast<uint32_t>(p)) * 2U;
        b0 = LoadGmScalar(pages, idx);
        b1 = LoadGmScalar(pages, idx + 1U);
    }
    const auto st = DecodeMlaRow(id, static_cast<uint32_t>(h), page, sp, dp, b0, b1, out);
    ASSERT_MSG(st != RowState::Invalid, "Invalid selected row address");
    return st;
}

__aicore__ inline SliceState LoadRequestSlice(
    uint32_t r, uint32_t lane, uint32_t gc, uint32_t cap, uint32_t tRows, uint32_t tCols, uint32_t tStride,
    GlobalTensor<int32_t>& meta, GlobalTensor<int32_t>& state, GlobalTensor<int32_t>& blockTable,
    GlobalTensor<int32_t>& selection, LocalTensor<int32_t>& localTable, LocalTensor<int32_t>& localIds,
    int32_t& hOut, uint32_t& beginOut, uint32_t& endOut
) {
    const int32_t row = LoadGmScalar(meta, r * 4U), h = LoadGmScalar(meta, r * 4U + 1U);
    const int32_t qStart = LoadGmScalar(meta, r * 4U + 2U), query = LoadGmScalar(meta, r * 4U + 3U);
    const int32_t cnt = LoadGmScalar(state, r * 8U);
    const bool valid =
        row >= 0 && static_cast<uint32_t>(row) < tRows &&
        h > 0 && h <= MAX_HISTORY && HistoryPages(static_cast<uint32_t>(h)) <= tCols &&
        query > 0 && query <= MAX_BATCHED_TOKENS && qStart >= 0 && qStart <= MAX_BATCHED_TOKENS - query &&
        cnt >= 0 && static_cast<uint32_t>(cnt) <= cap && cnt <= h &&
        LoadGmScalar(state, r * 8U + 1U) == query &&
        LoadGmScalar(state, r * 8U + 3U) == 0 &&
        LoadGmScalar(state, r * 8U + 4U) == 0;
    ASSERT_MSG(valid, "Invalid selected copy metadata or union status");
    if (!valid) {
        return SliceState::Abort;
    }

    const uint32_t count = static_cast<uint32_t>(cnt);
    const uint32_t perLane = (count + gc - 1U) / gc;
    ASSERT_MSG(perLane <= MAX_LOCAL_IDS, "Selected copy local ID capacity exceeded");
    if (perLane > MAX_LOCAL_IDS) {
        return SliceState::Abort;
    }

    const uint32_t begin = lane * perLane;
    const uint32_t end = begin + perLane < count ? begin + perLane : count;
    if (begin >= end) {
        return SliceState::Empty;
    }

    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(localTable, blockTable[static_cast<uint64_t>(row) * tStride], {1, HistoryPages(static_cast<uint32_t>(h)) * 4U, 0, 0, 0}, pad);
    DataCopyPad(localIds, selection[static_cast<uint64_t>(r) * cap + begin], {1, (end - begin) * 4U, 0, 0, 0}, pad);
    Fence<HardEvent::MTE2_S>();
    hOut = h;
    beginOut = begin;
    endOut = end;
    return SliceState::Ready;
}

__aicore__ inline void RunSelectedMlaInplace(
    TPipe& pipe, GM_ADDR ids, GM_ADDR status, GM_ADDR pageBases, GM_ADDR metadata, GM_ADDR table,
    GM_ADDR c0, GM_ADDR c1, uint32_t requests, uint32_t cap, uint32_t sp, uint32_t tRows, uint32_t tCols,
    uint32_t tStride, uint32_t dp, uint32_t gc
) {
    GlobalTensor<int32_t> selection, state, meta, blockTable;
    GlobalTensor<uint64_t> pages;
    GlobalTensor<uint8_t> dst0, dst1, src0, src1;
    BindGm(selection, ids);
    BindGm(state, status);
    BindGm(meta, metadata);
    BindGm(blockTable, table);
    BindGm(pages, pageBases);
    BindGm(dst0, c0);
    BindGm(dst1, c1);

    CopyQueue q0, q1;
    TBuf<TPosition::VECCALC> idsBuf, tableBuf;
    pipe.InitBuffer(q0, 2, MAX_RUN * NOPE_ROW_BYTES);
    pipe.InitBuffer(q1, 2, MAX_RUN * PE_ROW_BYTES);
    pipe.InitBuffer(idsBuf, MAX_LOCAL_IDS * sizeof(int32_t));
    pipe.InitBuffer(tableBuf, MAX_COLUMNS * sizeof(int32_t));
    auto localIds = idsBuf.Get<int32_t>();
    auto localTable = tableBuf.Get<int32_t>();

    const uint32_t lane = GetBlockIdx() % gc, groups = GetBlockNum() / gc;
    for (uint32_t r = GetBlockIdx() / gc; r < requests; r += groups) {
        int32_t h = 0, page = -1;
        uint32_t begin = 0U, end = 0U, col = UINT32_MAX, pending = 0U;
        uint64_t b0 = 0U, b1 = 0U;

        const auto slice = LoadRequestSlice(
            r, lane, gc, cap, tRows, tCols, tStride, meta, state, blockTable, selection, localTable, localIds, h, begin, end
        );
        if (slice == SliceState::Abort) {
            PipeBarrier<PIPE_ALL>();
            return;
        }
        if (slice == SliceState::Empty) {
            continue;
        }

        DualRowAddress next{}, pend{};
        RowState nextState = RowState::Invalid;
        bool resolved = false;
        for (uint32_t cursor = begin; cursor < end;) {
            if (!resolved) {
                nextState = ResolveCopyRow(localIds.GetValue(cursor - begin), h, localTable, pages, sp, dp, col, page, b0, b1, next);
            }
            resolved = false;
            if (nextState == RowState::Invalid) {
                AbortCopy(q0, q1, pending);
                return;
            }
            if (nextState == RowState::Full) {
                ++cursor;
                continue;
            }

            const DualRowAddress cur = next;
            uint32_t run = 1U;
            while (run < MAX_RUN && cursor + run < end) {
                nextState = ResolveCopyRow(localIds.GetValue(cursor + run - begin), h, localTable, pages, sp, dp, col, page, b0, b1, next);
                if (nextState == RowState::Invalid) {
                    AbortCopy(q0, q1, pending);
                    return;
                }
                if (nextState == RowState::Full || !AdjacentMla(cur, next, run)) {
                    resolved = true;
                    break;
                }
                ++run;
            }

            src0.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cur.nopeSource));
            src1.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cur.peSource));
            CopyIn(q0, src0, run * NOPE_ROW_BYTES);
            CopyIn(q1, src1, run * PE_ROW_BYTES);
            if (pending != 0U) {
                FlushCopy(q0, q1, dst0, dst1, pend, pending);
            }
            pend = cur;
            pending = run;
            cursor += run;
        }

        if (pending != 0U) {
            FlushCopy(q0, q1, dst0, dst1, pend, pending);
        }
        Fence<HardEvent::S_MTE2>();
    }
    PipeBarrier<PIPE_ALL>();
}

inline int32_t ResolveLaunchGroupCores(
    uint32_t requests, uint32_t cap, uint32_t sp, uint32_t tRows, uint32_t tCols, uint32_t tStride, uint32_t dp
) {
    if (requests == 0U || requests > 256U || cap == 0U || cap > 524288U || cap % PAGE_ROWS != 0U ||
        sp == 0U || sp > dp || tRows == 0U || tRows > 256U || tCols == 0U || tStride < tCols) {
        return -1;
    }
    uint32_t gc = 8U;
    while (gc * MAX_LOCAL_IDS < cap) {
        gc *= 2U;
    }
    return gc <= CORES ? static_cast<int32_t>(gc) : -1;
}

} // namespace SparsePrefillCopy
