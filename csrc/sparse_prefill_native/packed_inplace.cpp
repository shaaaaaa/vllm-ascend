// SPDX-License-Identifier: Apache-2.0
#include "selected_kv_assert.h"
#include "packed_inplace_address.h"

using namespace AscendC;
using namespace SparsePrefillInplace;
namespace {
constexpr uint32_t MAX_LOCAL_IDS = 16384U, MAX_COLUMNS = 4096U;
constexpr uint32_t MAX_COPY_BYTES = 32768U, CORES = 32U;
constexpr int32_t MAX_BATCHED_TOKENS = 8192;
using CopyQueue = TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2>;

struct RowAddress {
    uint64_t source = 0, destination = 0;
};

struct PageAddress {
    uint32_t column = UINT32_MAX;
    int32_t page = -1;
    uint64_t source = 0;
};

__aicore__ inline RowState ResolveRow(int32_t id, int32_t history, LocalTensor<int32_t>& table,
    GlobalTensor<uint64_t>& bases, uint32_t sourcePages, uint32_t destinationPages,
    PageAddress& cached, RowAddress& address)
{
    ASSERT_MSG(id >= 0 && id < history, "Selected copy token outside history");
    if (id < 0 || id >= history) return RowState::Invalid;
    const uint32_t column = static_cast<uint32_t>(id) / PAGE_ROWS;
    if (column != cached.column) {
        const int32_t page = table.GetValue(column);
        const bool valid = page >= 0 && static_cast<uint32_t>(page) < sourcePages &&
            static_cast<uint32_t>(page) < destinationPages;
        ASSERT_MSG(valid, "Selected copy page outside source or destination domain");
        if (!valid) return RowState::Invalid;
        cached.column = column;
        cached.page = page;
        cached.source = bases.GetValue(page);
    }
    const auto state = DecodeRow(id, history, cached.page, sourcePages, destinationPages,
                                 cached.source, address.source, address.destination);
    ASSERT_MSG(state != RowState::Invalid, "Invalid selected row address");
    return state;
}

__aicore__ inline void CopyIn(CopyQueue& queue, GlobalTensor<uint8_t>& src, uint32_t bytes)
{
    auto local = queue.AllocTensor<uint8_t>();
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    DataCopyPad(local, src, {1, bytes, 0, 0, 0}, pad);
    queue.EnQue(local);
}

__aicore__ inline void CopyOut(CopyQueue& queue, GlobalTensor<uint8_t>& dst,
    uint64_t destination, uint32_t bytes)
{
    auto local = queue.DeQue<uint8_t>();
    DataCopyPad(dst[destination], local, {1, bytes, 0, 0, 0});
    queue.FreeTensor(local);
}
}

extern "C" __global__ __aicore__ void sparse_prefill_packed_inplace(
    GM_ADDR ids, GM_ADDR status, GM_ADDR pageBases, GM_ADDR metadata,
    GM_ADDR table, GM_ADDR cache, uint32_t requests, uint32_t capacity,
    uint32_t sourcePages, uint32_t tableRows, uint32_t tableColumns,
    uint32_t tableStride, uint32_t destinationPages, uint32_t groupCores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int32_t> selection, state, meta, blockTable;
    GlobalTensor<uint64_t> pages;
    GlobalTensor<uint8_t> dst, src;
    selection.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids));
    state.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status));
    meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(metadata));
    blockTable.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
    pages.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(pageBases));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cache));
    TPipe pipe;
    CopyQueue copyQueue;
    TBuf<TPosition::VECCALC> idsBuf, tableBuf;
    pipe.InitBuffer(copyQueue, 2, MAX_COPY_BYTES);
    pipe.InitBuffer(idsBuf, MAX_LOCAL_IDS * sizeof(int32_t));
    pipe.InitBuffer(tableBuf, MAX_COLUMNS * sizeof(int32_t));
    auto localIds = idsBuf.Get<int32_t>();
    auto localTable = tableBuf.Get<int32_t>();
    const uint32_t group = GetBlockIdx() / groupCores, lane = GetBlockIdx() % groupCores;
    const uint32_t groups = GetBlockNum() / groupCores, maxRun = MAX_COPY_BYTES / ROW_BYTES;
    for (uint32_t r = group; r < requests; r += groups) {
        const int32_t row = meta.GetValue(r * 4U), h = meta.GetValue(r * 4U + 1U);
        const int32_t qStart = meta.GetValue(r * 4U + 2U), query = meta.GetValue(r * 4U + 3U);
        const int32_t countValue = state.GetValue(r * 8U);
        const bool valid = row >= 0 && static_cast<uint32_t>(row) < tableRows &&
            h > 0 && h <= 524288 && HistoryPages(static_cast<uint32_t>(h)) <= tableColumns &&
            query > 0 && query <= MAX_BATCHED_TOKENS && qStart >= 0 && qStart <= MAX_BATCHED_TOKENS - query &&
            countValue >= 0 && static_cast<uint32_t>(countValue) <= capacity && countValue <= h &&
            state.GetValue(r * 8U + 1U) == query && state.GetValue(r * 8U + 3U) == 0 &&
            state.GetValue(r * 8U + 4U) == 0;
        ASSERT_MSG(valid, "Invalid original-page copy metadata or union status");
        if (!valid) {
            PipeBarrier<PIPE_ALL>();
            return;
        }
        const uint32_t count = static_cast<uint32_t>(countValue);
        const uint32_t perLane = (count + groupCores - 1U) / groupCores;
        const uint32_t begin = lane * perLane;
        const uint32_t end = begin + perLane < count ? begin + perLane : count;
        ASSERT_MSG(perLane <= MAX_LOCAL_IDS, "Selected copy local ID capacity exceeded");
        if (perLane > MAX_LOCAL_IDS) {
            PipeBarrier<PIPE_ALL>();
            return;
        }
        if (begin >= end) continue;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(localTable, blockTable[static_cast<uint64_t>(row) * tableStride],
                    {1, HistoryPages(static_cast<uint32_t>(h)) * 4U, 0, 0, 0}, pad);
        DataCopyPad(localIds, selection[static_cast<uint64_t>(r) * capacity + begin],
                    {1, (end - begin) * 4U, 0, 0, 0}, pad);
        const auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event);
        WaitFlag<HardEvent::MTE2_S>(event);
        uint32_t cursor = begin;
        PageAddress cached;
        RowAddress next;
        RowState nextState = RowState::Invalid;
        bool resolved = false;
        uint64_t pendingDestination = 0;
        uint32_t pendingBytes = 0;
        while (cursor < end) {
            if (!resolved) {
                nextState = ResolveRow(localIds.GetValue(cursor - begin), h, localTable, pages,
                                       sourcePages, destinationPages, cached, next);
            }
            resolved = false;
            if (nextState == RowState::Invalid) {
                PipeBarrier<PIPE_ALL>();
                return;
            }
            if (nextState == RowState::Full) { ++cursor; continue; }
            const RowAddress current = next;
            uint32_t run = 1U;
            while (run < maxRun && cursor + run < end) {
                nextState = ResolveRow(localIds.GetValue(cursor + run - begin), h, localTable, pages,
                                       sourcePages, destinationPages, cached, next);
                if (nextState == RowState::Invalid) {
                    PipeBarrier<PIPE_ALL>();
                    return;
                }
                if (nextState == RowState::Full ||
                    !Adjacent(current.source, current.destination, next.source, next.destination, run)) {
                    resolved = true;
                    break;
                }
                ++run;
            }
            src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(current.source));
            // Prefetch the next run before draining the previous UB slot.
            CopyIn(copyQueue, src, run * ROW_BYTES);
            if (pendingBytes != 0) CopyOut(copyQueue, dst, pendingDestination, pendingBytes);
            pendingDestination = current.destination;
            pendingBytes = run * ROW_BYTES;
            cursor += run;
        }
        if (pendingBytes != 0) CopyOut(copyQueue, dst, pendingDestination, pendingBytes);
    }
    PipeBarrier<PIPE_ALL>();
}

extern "C" int32_t launch_sparse_prefill_packed_inplace(void* stream, void* ids, void* status,
    void* pageBases, void* metadata, void* table, void* cache, uint32_t requests,
    uint32_t capacity, uint32_t sourcePages, uint32_t tableRows, uint32_t tableColumns,
    uint32_t tableStride, uint32_t destinationPages)
{
    if (!stream || !ids || !status || !pageBases || !metadata || !table || !cache ||
        requests == 0U || requests > 256U || capacity == 0U || capacity > 524288U || capacity % PAGE_ROWS ||
        sourcePages == 0U || sourcePages > destinationPages || tableRows == 0U || tableRows > 256U ||
        tableColumns == 0U || tableStride < tableColumns) return -1;
    uint32_t groupCores = 8U;
    while (groupCores * MAX_LOCAL_IDS < capacity) groupCores *= 2U;
    sparse_prefill_packed_inplace<<<CORES, nullptr, stream>>>(
        static_cast<uint8_t*>(ids), static_cast<uint8_t*>(status), static_cast<uint8_t*>(pageBases),
        static_cast<uint8_t*>(metadata), static_cast<uint8_t*>(table), static_cast<uint8_t*>(cache),
        requests, capacity, sourcePages, tableRows, tableColumns, tableStride, destinationPages, groupCores);
    return 0;
}
