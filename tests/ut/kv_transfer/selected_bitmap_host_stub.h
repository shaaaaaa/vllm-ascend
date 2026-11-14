// SPDX-License-Identifier: Apache-2.0
// Scalar host models of the AscendC operations used by the bitmap header.
// This does not model device alignment, streams, barriers, or SDK codegen.
#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#define ASSERT_MSG(condition, message) do { if (!(condition)) throw std::runtime_error(message); } while (0)
using GM_ADDR = uint8_t*;

namespace AscendC {
using event_t = int;
enum class HardEvent { V_MTE2, MTE2_V, V_S, S_MTE3, MTE3_S, MTE2_S, V_MTE3, S_V };
enum class QuePosition { VECCALC, VECIN, VECOUT };
enum class CMPMODE { GT, GE };
enum class SELMODE { VSEL_TENSOR_SCALAR_MODE };
enum class RoundMode { CAST_NONE };
constexpr int PIPE_V = 0;
inline uint32_t blockIndex = 0;
inline uint32_t GetBlockIdx() { return blockIndex; }
struct Region { uintptr_t first, last; };
inline std::vector<Region> regions;
inline Region watched{};
inline size_t watchedWrites = 0;
inline void (*observeGlobalAccess)(const void*, size_t, bool) = nullptr;
inline void Register(void* data, size_t bytes) {
    auto first = reinterpret_cast<uintptr_t>(data);
    regions.push_back({first, first + bytes});
}
inline void CheckGlobal(const void* data, size_t bytes, bool write) {
    auto first = reinterpret_cast<uintptr_t>(data);
    const bool valid = std::any_of(regions.begin(), regions.end(), [&](Region range) {
        return first >= range.first && first <= range.last && bytes <= range.last - first;
    });
    if (!valid) throw std::out_of_range("GM access outside a registered tensor");
    if (write && first < watched.last && first + bytes > watched.first) ++watchedWrites;
    if (observeGlobalAccess) observeGlobalAccess(data, bytes, write);
}

template <typename T, bool Global = false> class Tensor {
public:
    Tensor() = default;
    Tensor(uint8_t* data, size_t bytes, std::shared_ptr<std::vector<uint64_t>> owner = {})
        : data_(data), bytes_(bytes), owner_(std::move(owner)) {}
    void SetGlobalBuffer(T* data) { data_ = reinterpret_cast<uint8_t*>(data); }
    uint8_t* Access(size_t bytes, bool write) const {
        if constexpr (Global) CheckGlobal(data_, bytes, write);
        else if (bytes > bytes_) throw std::out_of_range("UB access outside allocated TBuf");
        return data_;
    }
    T GetValue(size_t index) const {
        T value;
        std::memcpy(&value, (*this)[index].Access(sizeof(T), false), sizeof(T));
        return value;
    }
    void SetValue(size_t index, T value) const {
        std::memcpy((*this)[index].Access(sizeof(T), true), &value, sizeof(T));
    }
    Tensor operator[](size_t index) const {
        const size_t offset = index * sizeof(T);
        if constexpr (!Global) {
            if (offset > bytes_) throw std::out_of_range("UB offset outside allocated TBuf");
        }
        return {data_ + offset, Global ? 0 : bytes_ - offset, owner_};
    }
    template <typename U> Tensor<U, Global> ReinterpretCast() const { return {data_, bytes_, owner_}; }
private:
    uint8_t* data_ = nullptr;
    size_t bytes_ = 0;
    std::shared_ptr<std::vector<uint64_t>> owner_;
};
template <typename T> using LocalTensor = Tensor<T>;
template <typename T> using GlobalTensor = Tensor<T, true>;
template <QuePosition> struct TBuf {
    std::shared_ptr<std::vector<uint64_t>> data;
    size_t bytes = 0;
    template <typename T> LocalTensor<T> Get() const {
        return {reinterpret_cast<uint8_t*>(data->data()), bytes, data};
    }
};
struct TPipe {
    template <QuePosition Position> void InitBuffer(TBuf<Position>& buffer, size_t bytes) {
        buffer.bytes = bytes;
        buffer.data = std::make_shared<std::vector<uint64_t>>((bytes + 7) / 8, UINT64_MAX);
    }
    template <typename Queue> void InitBuffer(Queue& queue, size_t count, size_t bytes) {
        queue.Init(count, bytes);
    }
    int FetchEventID(HardEvent) const { return 0; }
};
inline TPipe* GetTPipePtr() { static TPipe pipe; return &pipe; }
template <HardEvent> void SetFlag(event_t) {}
template <HardEvent> void WaitFlag(event_t) {}
template <int> void PipeBarrier() {}
struct DataCopyParams { uint32_t blocks, bytes, sourceStride, destinationStride, reserved = 0; };
struct PaddingParams { bool enabled; uint32_t left, right; float value; };
template <typename T, bool D, bool S>
void DataCopy(Tensor<T, D> dst, Tensor<T, S> src, size_t count) {
    std::memcpy(dst.Access(count * sizeof(T), true), src.Access(count * sizeof(T), false), count * sizeof(T));
}
template <typename T, bool D, bool S>
void DataCopyPad(Tensor<T, D> dst, Tensor<T, S> src, DataCopyParams params) {
    if (params.blocks != 1 || params.sourceStride || params.destinationStride)
        throw std::runtime_error("The host stub only supports contiguous one-block copies");
    if (params.bytes % sizeof(T)) throw std::runtime_error("Unaligned copy size");
    DataCopy(dst, src, params.bytes / sizeof(T));
}
template <typename T, bool D, bool S>
void DataCopyPad(Tensor<T, D> dst, Tensor<T, S> src, DataCopyParams params, PaddingParams padding) {
    if (padding.enabled) throw std::runtime_error("Padding not implemented by the host stub");
    DataCopyPad(dst, src, params);
}
template <typename T> void Duplicate(LocalTensor<T> dst, T value, size_t count) {
    for (size_t i = 0; i < count; ++i) dst.SetValue(i, value);
}
template <typename T> void ArithProgression(LocalTensor<T> dst, T first, T step, size_t count) {
    for (size_t i = 0; i < count; ++i) dst.SetValue(i, first + static_cast<T>(i) * step);
}
#define HOST_BINARY(name, operation) \
    template <typename T> void name(LocalTensor<T> dst, LocalTensor<T> left, LocalTensor<T> right, size_t count) { \
        for (size_t i = 0; i < count; ++i) dst.SetValue(i, left.GetValue(i) operation right.GetValue(i)); \
    }
HOST_BINARY(Add, +)
HOST_BINARY(Sub, -)
HOST_BINARY(And, &)
HOST_BINARY(Or, |)
#undef HOST_BINARY
#define HOST_SCALAR(name, operation) \
    template <typename T, typename S> void name(LocalTensor<T> dst, LocalTensor<T> src, S value, size_t count) { \
        for (size_t i = 0; i < count; ++i) dst.SetValue(i, src.GetValue(i) operation value); \
    }
HOST_SCALAR(Adds, +)
HOST_SCALAR(Muls, *)
HOST_SCALAR(ShiftLeft, <<)
HOST_SCALAR(ShiftRight, >>)
#undef HOST_SCALAR
inline void Gather(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<uint32_t> offsets,
                   uint32_t base, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const uint32_t offset = offsets.GetValue(i) + base;
        if (offset % sizeof(float)) throw std::runtime_error("Unaligned gather offset");
        dst.SetValue(i, src.GetValue(offset / sizeof(float)));
    }
}
inline void CompareScalar(LocalTensor<uint8_t> dst, LocalTensor<float> src, float value,
                          CMPMODE mode, size_t count) {
    Duplicate(dst, uint8_t{0}, (count + 7) / 8);
    for (size_t i = 0; i < count; ++i) {
        const bool match = mode == CMPMODE::GT ? src.GetValue(i) > value : src.GetValue(i) >= value;
        if (match) dst.SetValue(i / 8, dst.GetValue(i / 8) | (1U << (i % 8)));
    }
}
inline void Select(LocalTensor<float> dst, LocalTensor<uint8_t> mask, LocalTensor<float> src,
                   float otherwise, SELMODE, size_t count) {
    for (size_t i = 0; i < count; ++i)
        dst.SetValue(i, (mask.GetValue(i / 8) & (1U << (i % 8))) ? src.GetValue(i) : otherwise);
}
template <typename D, typename S>
void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode, size_t count) {
    for (size_t i = 0; i < count; ++i) dst.SetValue(i, static_cast<D>(src.GetValue(i)));
}
inline void ReduceSum(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float>, size_t count) {
    float result = 0;
    for (size_t i = 0; i < count; ++i) result += src.GetValue(i);
    dst.SetValue(0, result);
}
struct GatherMaskParams { uint32_t repeatTimes, src0BlockStride, src0RepeatStride, src1RepeatStride; };
inline void GatherMask(LocalTensor<uint32_t> dst, LocalTensor<uint32_t> src, LocalTensor<uint32_t> bits,
                       bool, uint32_t count, GatherMaskParams, uint64_t& selected) {
    selected = 0;
    for (size_t i = 0; i < count; ++i)
        if (bits.GetValue(i / 32) & (1U << (i % 32))) dst.SetValue(selected++, src.GetValue(i));
}
} // namespace AscendC
