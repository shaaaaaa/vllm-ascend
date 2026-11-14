// SPDX-License-Identifier: Apache-2.0
#ifndef SPARSE_PREFILL_PACKED_INPLACE_ADDRESS_H
#define SPARSE_PREFILL_PACKED_INPLACE_ADDRESS_H
#include <cstdint>

namespace SparsePrefillInplace {
constexpr uint32_t PAGE_ROWS = 128U;
constexpr uint32_t ROW_BYTES = 656U;
enum class RowState : uint8_t { Invalid, Full, Selected };

__aicore__ inline uint32_t HistoryPages(uint32_t history)
{
    return (history + PAGE_ROWS - 1U) / PAGE_ROWS;
}

// The source vector is indexed by native physical page ID, but its values are
// independently leased DDR addresses. Zero is authorized FULL coverage only.
__aicore__ inline RowState DecodeRow(int32_t id, uint32_t history, int32_t page,
    uint32_t sourcePages, uint32_t destinationPages, uint64_t base,
    uint64_t& source, uint64_t& destination)
{
    if (id < 0 || static_cast<uint32_t>(id) >= history || page < 0 ||
        static_cast<uint32_t>(page) >= sourcePages || static_cast<uint32_t>(page) >= destinationPages) {
        return RowState::Invalid;
    }
    const uint64_t offset = static_cast<uint32_t>(id) % PAGE_ROWS * ROW_BYTES;
    destination = (static_cast<uint64_t>(page) * PAGE_ROWS) * ROW_BYTES + offset;
    if (base == 0U) return RowState::Full;
    if (base > UINT64_MAX - offset - (ROW_BYTES - 1U)) return RowState::Invalid;
    source = base + offset;
    return RowState::Selected;
}

__aicore__ inline bool Adjacent(uint64_t source, uint64_t destination,
    uint64_t nextSource, uint64_t nextDestination, uint32_t run)
{
    const uint64_t bytes = static_cast<uint64_t>(run) * ROW_BYTES;
    return nextSource >= source && nextSource - source == bytes &&
           nextDestination >= destination && nextDestination - destination == bytes;
}
} // namespace SparsePrefillInplace
#endif
