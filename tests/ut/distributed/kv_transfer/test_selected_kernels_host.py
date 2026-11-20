# SPDX-License-Identifier: Apache-2.0
"""Host C++17 compilation and semantic tests for ondemand-selected MLA kernels."""

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
KDIR = ROOT / "csrc" / "sparse_prefill_native"

HOST_STUB = r"""#pragma once
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#define __aicore__
#define __global__
#define __gm__
#define KERNEL_TASK_TYPE_DEFAULT(...)
#define KERNEL_TYPE_AIV_ONLY 0
#define KERNEL_TYPE_MIX_AIV_1_0 1
#define PIPE_ALL 0
#define PIPE_V 1
#define ASSERT_MSG(cond, msg) do { if (!(cond) && assertOn) throw std::runtime_error(msg); } while (0)
using GM_ADDR = uint8_t*;
using event_t = int32_t;

namespace AscendC {
using U8 = uint8_t;
using U16 = uint16_t;
using U32 = uint32_t;
using U64 = uint64_t;
using Z = size_t;

struct Region { uintptr_t b, e; };
inline std::vector<Region> regions;
inline thread_local U32 blockIndex = 0, blockCount = 32, queueDepth = 0;
inline thread_local int activePipes = 0;
inline thread_local bool assertOn = true;
inline thread_local std::unordered_set<uintptr_t> cleanCacheLines;

struct SyncBarrier {
    std::mutex mu;
    std::condition_variable cv;
    U32 count = 0, gen = 0;
    bool aborted = false;
    void Reset() {
        std::lock_guard<std::mutex> lk(mu);
        count = 0;
        aborted = false;
    }
    void Abort() {
        std::lock_guard<std::mutex> lk(mu);
        aborted = true;
        cv.notify_all();
    }
    void Wait(U32 total) {
        std::unique_lock<std::mutex> lk(mu);
        if (aborted) {
            throw std::runtime_error("barrier_abort");
        }
        U32 g = gen;
        if (++count == total) {
            count = 0;
            ++gen;
            cv.notify_all();
        } else {
            cv.wait(lk, [&] { return aborted || gen != g; });
            if (aborted) {
                throw std::runtime_error("barrier_abort");
            }
        }
    }
};
inline SyncBarrier g_sync;

inline void Reg(const void* p, Z n) {
    auto a = reinterpret_cast<uintptr_t>(p);
    regions.push_back({a, a + n});
}
inline void Chk(const void* p, Z n) {
    auto a = reinterpret_cast<uintptr_t>(p);
    for (const auto& r : regions) {
        if (a >= r.b && a <= r.e && n <= r.e - a) {
            return;
        }
    }
    throw std::runtime_error("OOB");
}

inline U32 GetBlockIdx() { return blockIndex; }
inline U32 GetBlockNum() { return blockCount; }

enum class HardEvent : int { MTE2_S, MTE2_V, S_V, S_MTE2, S_MTE3, MTE3_S, V_S, V_MTE2, V_MTE3 };
enum class TPosition : int { VECCALC };
enum class QuePosition : int { VECIN, VECOUT, VECCALC = int(TPosition::VECCALC) };
enum class CMPMODE : int { LT, LE, GT, GE, EQ, NE };
enum class SELMODE : int { VSEL_TENSOR_SCALAR_MODE };
enum class RoundMode : int { CAST_NONE, CAST_RINT };
enum class CacheLine : int { SINGLE_CACHE_LINE };
enum class DcciDst : int { CACHELINE_OUT };
struct DataCopyParams { U16 blockCount = 1, blockLen = 0, srcStride = 0, dstStride = 0; };
struct DataCopyExtParams { U16 blockCount = 1; U32 blockLen = 0, srcStride = 0, dstStride = 0, rsv = 0; };
template <class T> struct DataCopyPadExtParams {
    bool isPad = false;
    U8 leftPadding = 0, rightPadding = 0;
    T paddingValue{};
};
template <int> inline void PipeBarrier() {}
template <HardEvent> inline void SetFlag(event_t) {}
template <HardEvent> inline void WaitFlag(event_t) {}

template <class T, bool G = false> struct Tensor {
    U8* d = nullptr;
    Z n = 0, k = 0;
    U8* Access(Z bytes) const {
        if (G) {
            Chk(d, bytes);
        } else if (bytes > n) {
            throw std::out_of_range("UB");
        }
        return d;
    }
    T GetValue(Z i) const {
        U8* p = (*this)[i].Access(sizeof(T));
        if constexpr (G) {
            if (cleanCacheLines.find(reinterpret_cast<uintptr_t>(p) & ~uintptr_t(63)) == cleanCacheLines.end()) {
                throw std::runtime_error("stale_dcache");
            }
        }
        T v{};
        std::memcpy(&v, p, sizeof(T));
        return v;
    }
    void SetValue(Z i, T v) const {
        std::memcpy((*this)[i].Access(sizeof(T)), &v, sizeof(T));
    }
    Tensor operator[](Z i) const {
        Z f = i * sizeof(T);
        if constexpr (!G) {
            if (f > n) {
                throw std::out_of_range("UB");
            }
        }
        return {d + f, G ? 0 : n - f, 0};
    }
    void SetGlobalBuffer(__gm__ T* p) const { const_cast<Tensor*>(this)->d = reinterpret_cast<U8*>(p); }
    template <class U> Tensor<U, G> ReinterpretCast() const { return {d, n, k}; }
};

template <class T> using GlobalTensor = Tensor<T, true>;
template <class T> using LocalTensor = Tensor<T, false>;

inline void InvalidateCacheRange(const void* p, Z bytes) {
    if (bytes == 0U || cleanCacheLines.empty()) {
        return;
    }
    uintptr_t a = reinterpret_cast<uintptr_t>(p);
    uintptr_t first = a & ~uintptr_t(63);
    uintptr_t last = (a + bytes - 1U) & ~uintptr_t(63);
    for (uintptr_t line = first;; line += 64U) {
        cleanCacheLines.erase(line);
        if (line == last) {
            break;
        }
    }
}

template <class T, CacheLine, DcciDst> inline void DataCacheCleanAndInvalid(GlobalTensor<T> t) {
    U8* p = t.Access(sizeof(T));
    cleanCacheLines.insert(reinterpret_cast<uintptr_t>(p) & ~uintptr_t(63));
}

struct Buf {
    std::vector<U8> v;
    template <class T> LocalTensor<T> Get() { return {v.data(), v.size(), 0}; }
};
template <auto> struct TBuf : Buf {};

template <QuePosition, QuePosition, int D> struct TQueBind {
    std::vector<std::vector<U8>> q;
    int cap = 0, live = 0;
    Z qBytes = 0;
    void Init(int num, Z bytes) {
        if (num <= 0 || num > D) {
            throw std::logic_error("qcap");
        }
        cap = num;
        qBytes = bytes;
    }
    ~TQueBind() noexcept(false) {
        bool leaked = std::uncaught_exceptions() == 0 && (!q.empty() || live != 0);
        q.clear();
        if (leaked) {
            throw std::logic_error("lk");
        }
    }
    template <class T> LocalTensor<T> AllocTensor() {
        if (cap <= 0 || live >= cap) {
            throw std::logic_error("qfull");
        }
        ++live;
        ++queueDepth;
        return {new U8[qBytes], qBytes, 1};
    }
    template <class T> void EnQue(LocalTensor<T> t) {
        if (!t.k || static_cast<int>(q.size()) >= cap) {
            throw std::logic_error("qovf");
        }
        q.emplace_back(t.d, t.d + t.n);
        delete[] t.d;
    }
    template <class T> LocalTensor<T> DeQue() {
        if (q.empty()) {
            throw std::runtime_error("emptyQ");
        }
        Z sz = q.front().size();
        auto* p = new U8[sz];
        std::memcpy(p, q.front().data(), sz);
        q.erase(q.begin());
        return {p, sz, 1};
    }
    template <class T> void FreeTensor(LocalTensor<T> t) {
        if (!t.k || live <= 0) {
            throw std::logic_error("badfree");
        }
        delete[] t.d;
        --live;
        --queueDepth;
    }
};

struct TPipe {
    bool tracked_;
    explicit TPipe(bool track = true) : tracked_(track) {
        if (tracked_) {
            if (activePipes != 0) {
                throw std::logic_error("dup_pipe");
            }
            activePipes = 1;
            cleanCacheLines.clear();
        }
    }
    ~TPipe() {
        if (tracked_) {
            activePipes = 0;
            cleanCacheLines.clear();
        }
    }
    void Reset() {}
    event_t FetchEventID(HardEvent) { return 0; }
    template <auto P> void InitBuffer(TBuf<P>& b, Z n) { b.v.assign(n, 0); }
    template <QuePosition A, QuePosition B, int D> void InitBuffer(TQueBind<A, B, D>& q, int num, Z n) {
        q.Init(num, n);
    }
};
inline thread_local TPipe g_evPipe(false);
inline TPipe* GetTPipePtr() {
    if (activePipes != 1) {
        throw std::logic_error("no_pipe");
    }
    return &g_evPipe;
}
inline void SyncAll() {
    if (activePipes != 1) {
        throw std::logic_error("sync_pipe");
    }
    g_sync.Wait(blockCount);
    cleanCacheLines.clear();
}

template <class T> void DataCopy(LocalTensor<T> d, GlobalTensor<T> s, U32 c) {
    std::memcpy(d.Access(c * sizeof(T)), s.Access(c * sizeof(T)), c * sizeof(T));
}
template <class T> void DataCopy(GlobalTensor<T> d, LocalTensor<T> s, U32 c) {
    Z bytes = Z(c) * sizeof(T);
    U8* dp = d.Access(bytes);
    std::memcpy(dp, s.Access(bytes), bytes);
    InvalidateCacheRange(dp, bytes);
}
template <class T> void DataCopy(LocalTensor<T> d, GlobalTensor<T> s, DataCopyParams p) {
    Z blk = 32;
    Z ss = Z(p.srcStride) * blk + blk;
    Z ds = Z(p.dstStride) * blk + blk;
    for (U16 i = 0; i < p.blockCount; ++i) {
        std::memcpy(
            d.Access(i * ds + p.blockLen * blk) + i * ds,
            s.Access(i * ss + p.blockLen * blk) + i * ss,
            p.blockLen * blk
        );
    }
}
template <class T> void DataCopyPad(LocalTensor<T> d, GlobalTensor<T> s, DataCopyExtParams p, DataCopyPadExtParams<T>) {
    std::memcpy(d.Access(p.blockLen), s.Access(p.blockLen), p.blockLen);
}
template <class T> void DataCopyPad(GlobalTensor<T> d, LocalTensor<T> s, DataCopyExtParams p) {
    U8* dp = d.Access(p.blockLen);
    std::memcpy(dp, s.Access(p.blockLen), p.blockLen);
    InvalidateCacheRange(dp, p.blockLen);
}
template <class T> void Duplicate(LocalTensor<T> d, T v, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, v);
    }
}
template <class T> void ArithProgression(LocalTensor<T> d, T f, T step, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, static_cast<T>(f + static_cast<T>(i) * step));
    }
}
template <class T> void Relu(LocalTensor<T> d, LocalTensor<T> s, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, s.GetValue(i) > 0 ? s.GetValue(i) : T{0});
    }
}
template <class T> void Add(LocalTensor<T> d, LocalTensor<T> a, LocalTensor<T> b, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, a.GetValue(i) + b.GetValue(i));
    }
}
template <class T> void Sub(LocalTensor<T> d, LocalTensor<T> a, LocalTensor<T> b, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, a.GetValue(i) - b.GetValue(i));
    }
}
template <class T> void And(LocalTensor<T> d, LocalTensor<T> a, LocalTensor<T> b, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, a.GetValue(i) & b.GetValue(i));
    }
}
template <class T> void Or(LocalTensor<T> d, LocalTensor<T> a, LocalTensor<T> b, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, a.GetValue(i) | b.GetValue(i));
    }
}
template <class T> void Adds(LocalTensor<T> d, LocalTensor<T> s, T v, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, s.GetValue(i) + v);
    }
}
template <class T> void Muls(LocalTensor<T> d, LocalTensor<T> s, T v, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, s.GetValue(i) * v);
    }
}
template <class T> void ShiftLeft(LocalTensor<T> d, LocalTensor<T> s, U32 k, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, s.GetValue(i) << k);
    }
}
template <class T> void ShiftRight(LocalTensor<T> d, LocalTensor<T> s, U32 k, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, s.GetValue(i) >> k);
    }
}
template <class T> void Gather(LocalTensor<T> d, LocalTensor<T> s, LocalTensor<U32> o, U32 b, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, s.GetValue((o.GetValue(i) + b) / sizeof(T)));
    }
}
inline void CompareScalar(LocalTensor<U8> d, LocalTensor<float> s, float v, CMPMODE m, Z c) {
    Duplicate(d, U8{0}, (c + 7) / 8);
    for (Z i = 0; i < c; ++i) {
        float x = s.GetValue(i);
        bool hit = m == CMPMODE::GT ? x > v : (m == CMPMODE::GE ? x >= v : (m == CMPMODE::LT ? x < v : false));
        hit = hit || (m == CMPMODE::LE ? x <= v : (m == CMPMODE::EQ ? x == v : (m == CMPMODE::NE ? x != v : false)));
        if (hit) {
            d.SetValue(i / 8, d.GetValue(i / 8) | (1U << (i % 8)));
        }
    }
}
template <class T> void Compare(LocalTensor<U8> d, LocalTensor<T> l, LocalTensor<T> r, CMPMODE, Z c) {
    Duplicate(d, U8{0}, (c + 7) / 8);
    for (Z i = 0; i < c; ++i) {
        if (l.GetValue(i) != r.GetValue(i)) {
            d.SetValue(i / 8, d.GetValue(i / 8) | (1U << (i % 8)));
        }
    }
}
inline void Select(LocalTensor<float> d, LocalTensor<U8> m, LocalTensor<float> s, float o, SELMODE, Z c) {
    for (Z i = 0; i < c; ++i) {
        d.SetValue(i, (m.GetValue(i / 8) & (1U << (i % 8))) ? s.GetValue(i) : o);
    }
}
template <class D, class S> void Cast(LocalTensor<D> d, LocalTensor<S> s, RoundMode r, Z c) {
    for (Z i = 0; i < c; ++i) {
        if constexpr (std::is_integral_v<D> && std::is_floating_point_v<S>) {
            d.SetValue(i, static_cast<D>(r == RoundMode::CAST_RINT ? std::nearbyint(s.GetValue(i)) : s.GetValue(i)));
        } else {
            (void)r;
            d.SetValue(i, static_cast<D>(s.GetValue(i)));
        }
    }
}
inline void ReduceSum(LocalTensor<float> d, LocalTensor<float> s, LocalTensor<float>, Z c) {
    float a = 0;
    for (Z i = 0; i < c; ++i) {
        a += s.GetValue(i);
    }
    d.SetValue(0, a);
}
struct GatherMaskParams {
    GatherMaskParams() = default;
    GatherMaskParams(U8 b, U16 r, U16 s0, U8 s1)
        : src0BlockStride(b), repeatTimes(r), src0RepeatStride(s0), src1RepeatStride(s1) {}
    U8 src0BlockStride = 1;
    U16 repeatTimes = 0;
    U16 src0RepeatStride = 8;
    U8 src1RepeatStride = 8;
};
inline void GatherMask(
    LocalTensor<U32> d, LocalTensor<U32> s, LocalTensor<U32> b, bool, U32 c, GatherMaskParams, U64& sel
) {
    sel = 0;
    for (U32 i = 0; i < c; ++i) {
        if (b.GetValue(i / 32) & (1U << (i % 32))) {
            d.SetValue(sel++, s.GetValue(i));
        }
    }
}
inline void GatherMask(LocalTensor<U32> d, LocalTensor<U32> s, U8, bool, U32, GatherMaskParams p, U64& sel) {
    sel = U64(p.repeatTimes) * 32U;
    for (U64 i = 0; i < sel; ++i) {
        d.SetValue(i, s.GetValue(i * 2U));
    }
}
struct MrgSort4Info {
    MrgSort4Info() = default;
    MrgSort4Info(const U16 len[4], bool susp, U16 vb, U16 r)
        : ifExhaustedSuspension(susp), validBit(vb), repeatTimes(static_cast<U8>(r)) {
        for (int i = 0; i < 4; ++i) {
            elementLengths[i] = len[i];
        }
    }
    U16 elementLengths[4]{};
    bool ifExhaustedSuspension = false;
    U16 validBit = 0;
    U8 repeatTimes = 1;
};
template <class T> struct MrgSortSrcList { LocalTensor<T> src1, src2, src3, src4; };
template <class T> void MrgSort(LocalTensor<T> d, MrgSortSrcList<T> s, MrgSort4Info p) {
    const LocalTensor<T> src[] = {s.src1, s.src2, s.src3, s.src4};
    std::vector<std::pair<float, U32>> v;
    for (int k = 0; k < 4; ++k) {
        if (p.validBit & (1U << k)) {
            // AscendC MrgSort permits at most 4095 pairs per valid source.
            // Enforce this API bound independently of MergeSortedRuns splitting.
            if (p.elementLengths[k] > 4095U) {
                throw std::runtime_error("mrgsort_source_length");
            }
            for (U32 i = 0; i < p.elementLengths[k]; ++i) {
                v.emplace_back(src[k].GetValue(i * 2U), src[k].template ReinterpretCast<U32>().GetValue(i * 2U + 1U));
            }
        }
    }
    std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (Z i = 0; i < v.size(); ++i) {
        d.SetValue(i * 2U, v[i].first);
        d.template ReinterpretCast<U32>().SetValue(i * 2U + 1U, v[i].second);
    }
}
} // namespace AscendC
"""

VEC_H = r"""#pragma once
#include "kernel_operator.h"
namespace LIServiceVec {
using namespace AscendC;
constexpr int64_t VEC_REPEAT_BYTES = 256, B32_VEC_REPEAT_STRIDE = 8;
inline void SortAll(LocalTensor<float>& s, LocalTensor<float>&, int64_t c) {
    std::vector<std::pair<float, U32>> v;
    for (int64_t i = 0; i < c; ++i) {
        v.emplace_back(s.GetValue(i), s.template ReinterpretCast<U32>().GetValue(c + i));
    }
    std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int64_t i = 0; i < c; ++i) {
        s.SetValue(i * 2, v[i].first);
        s.template ReinterpretCast<U32>().SetValue(i * 2 + 1, v[i].second);
    }
}
inline void ExtractIndex(const LocalTensor<U32>& d, const LocalTensor<U32>& s, int64_t c) {
    for (int64_t i = 0; i < (c + 31) / 32 * 32; ++i) {
        d.SetValue(i, s.GetValue(i * 2 + 1));
    }
}
} // namespace LIServiceVec
"""

CPP_TESTS = r"""#include "selected_inplace_common.h"
#include "sparse_prefill_services.h"
#include <thread>

using namespace AscendC;
using namespace SparsePrefillCopy;
using namespace SparsePrefillPlan;
using namespace SparsePrefillLocal;
using namespace SparsePrefillBitmap;

static std::string g_err;
template <class F> int Guard(bool aon, F&& f) {
    assertOn = aon;
    queueDepth = 0;
    cleanCacheLines.clear();
    g_err.clear();
    try {
        f();
        queueDepth = 0;
        cleanCacheLines.clear();
        return 0;
    } catch (const std::exception& e) {
        queueDepth = 0;
        cleanCacheLines.clear();
        g_err = e.what();
        return 1;
    }
}
template <class T> U8* R8(T* p) { return reinterpret_cast<U8*>(p); }
template <class V> U8* P8(V& v) { return R8(v.data()); }

static void FillPlane(std::vector<U8>& buf, U8 tag, U32 pageBytes, U32 rowBytes) {
    for (Z i = 0; i < buf.size(); ++i) {
        U32 pg = static_cast<U32>(i / pageBytes), rw = static_cast<U32>((i % pageBytes) / rowBytes);
        U8 v = static_cast<U8>((tag + pg * 37U + rw * 13U + (i % rowBytes) * 7U) & 0xFFU);
        if (v == 0xA5U || v == 0x5AU) {
            v ^= 0x11U;
        }
        buf[i] = v;
    }
}

static void CopyRowRef(
    std::vector<U8>& cn, std::vector<U8>& cpe, const std::vector<U8>& sn, const std::vector<U8>& spe,
    U32 dstPg, U32 srcPg, U32 row, U32 nr, U32 er
) {
    std::memcpy(&cn[dstPg * nr + row * NOPE_ROW_BYTES], &sn[srcPg * nr + row * NOPE_ROW_BYTES], NOPE_ROW_BYTES);
    std::memcpy(&cpe[dstPg * er + row * PE_ROW_BYTES], &spe[srcPg * er + row * PE_ROW_BYTES], PE_ROW_BYTES);
}

extern "C" {
const char* host_last_error() { return g_err.c_str(); }
void sparse_prefill_bf16_mla_inplace(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t
);
void sparse_prefill_adaptive_union(GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t);
void sparse_prefill_partition_finish_fixed(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t, uint32_t, uint32_t
);

int test_mla_decode_and_copy(int gc_in, int mode, int aon) {
    return Guard(aon != 0, [&] {
        DualRowAddress m{};
        auto dm = [&](int i, U32 m0, int p, U64 a, U64 b) { return DecodeMlaRow(i, 256, m0, 4, p, a, b, m); };
        auto rl = [&](U32 r, U32 h, U32 q) { return ResolveLaunchGroupCores(r, h, 16, q, 16, 16, 16); };
        if (dm(129, 2, 4, 0x20000ULL, 0x40000ULL) != RowState::Selected ||
            m.nopeSource != 0x20000ULL + NOPE_ROW_BYTES || m.peSource != 0x40000ULL + PE_ROW_BYTES ||
            m.nopeDestination != (2U * PAGE_ROWS + 1U) * NOPE_ROW_BYTES ||
            m.peDestination != (2U * PAGE_ROWS + 1U) * PE_ROW_BYTES ||
            dm(5, 1, 4, 0, 0) != RowState::Full || dm(5, 1, 4, 4096, 0) != RowState::Invalid ||
            dm(5, 1, 4, 4096, UINT64_MAX - 10) != RowState::Invalid ||
            !AdjacentMla(
                {1000, 2000, 3000, 4000},
                {1000 + NOPE_ROW_BYTES, 2000 + PE_ROW_BYTES, 3000 + NOPE_ROW_BYTES, 4000 + PE_ROW_BYTES},
                1
            ) ||
            AdjacentMla({1000, 2000, 3000, 4000}, {1000, 2000, 3000, 4000}, 1) ||
            rl(2, 131072, 2) != 8 || rl(1, 262144, 1) != 16 || rl(0, 128, 1) != -1) {
            throw std::runtime_error("dec");
        }
        const U32 R = (mode == 3) ? 5U : 2U, P = (mode == 3) ? 10U : 6U;
        constexpr U32 C = 128, K = 2, NR = PAGE_ROWS * NOPE_ROW_BYTES, ER = PAGE_ROWS * PE_ROW_BYTES;
        std::vector<U8> sn(P * NR), spe(P * ER), cn(P * NR, 0xA5), cpe(P * ER, 0x5A), exp_cn = cn, exp_cpe = cpe;
        FillPlane(sn, 0x11U, NR, NOPE_ROW_BYTES);
        FillPlane(spe, 0x73U, ER, PE_ROW_BYTES);
        std::vector<int32_t> ids(R * C, 0), st(R * 8, 0), meta(R * 4, 0), tbl(R * K, 0);
        std::vector<U64> mb(P * 2, 0);
        if (mode == 3) {
            const U32 selPg[5] = {4, 5, 6, 7, 8};
            const U32 fullPg[5] = {1, 2, 3, 0, 9};
            for (U32 r = 0; r < R; ++r) {
                st[r * 8] = 4; st[r * 8 + 1] = 4;
                meta[r * 4] = static_cast<int32_t>(r); meta[r * 4 + 1] = 256;
                meta[r * 4 + 2] = static_cast<int32_t>(r * 4); meta[r * 4 + 3] = 4;
                tbl[r * K] = static_cast<int32_t>(selPg[r]);
                tbl[r * K + 1] = static_cast<int32_t>(fullPg[r]);
                ids[r * C] = 0; ids[r * C + 1] = 1; ids[r * C + 2] = 3; ids[r * C + 3] = 128;
                mb[selPg[r] * 2U] = uintptr_t(sn.data() + r * NR);
                mb[selPg[r] * 2U + 1U] = uintptr_t(spe.data() + r * ER);
            }
        } else {
            st = {4, 4, 7, 0, 0, 0, 0, 0, 2, 4, 6, 0, 0, 0, 0, 0};
            meta = {0, 256, 0, 4, 1, 256, 4, 4};
            tbl = {4, 1, 5, 2};
            ids[0] = 0; ids[1] = 1; ids[2] = 3; ids[3] = 128; ids[C] = 0; ids[C + 1] = 128;
            mb[8] = uintptr_t(sn.data()); mb[9] = uintptr_t(spe.data());
            mb[10] = uintptr_t(sn.data() + NR); mb[11] = uintptr_t(spe.data() + ER);
        }
        U32 reqs = (mode == 2) ? 1U : R;
        if (mode == 2) { st[1] = 999; }
        if (mode == 1) { mb[2] = 4096; mb[3] = 0; }
        regions.clear();
        queueDepth = 0;
        ([](auto&... a) { (Reg(a.data(), a.size() * sizeof(a[0])), ...); })(ids, st, meta, tbl, mb, cn, cpe, sn, spe);
        for (blockIndex = 0; blockIndex < 32U; ++blockIndex) {
            sparse_prefill_bf16_mla_inplace(
                P8(ids), P8(st), P8(mb), P8(meta), P8(tbl), cn.data(), cpe.data(),
                reqs, C, P, R, K, K, P, static_cast<U32>(gc_in)
            );
        }
        if (queueDepth != 0U) {
            throw std::runtime_error("lk");
        }
        if (mode == 0) {
            CopyRowRef(exp_cn, exp_cpe, sn, spe, 4, 0, 0, NR, ER);
            CopyRowRef(exp_cn, exp_cpe, sn, spe, 4, 0, 1, NR, ER);
            CopyRowRef(exp_cn, exp_cpe, sn, spe, 4, 0, 3, NR, ER);
            CopyRowRef(exp_cn, exp_cpe, sn, spe, 5, 1, 0, NR, ER);
        } else if (mode == 1) {
            CopyRowRef(exp_cn, exp_cpe, sn, spe, 5, 1, 0, NR, ER);
        } else if (mode == 3) {
            const U32 selPg[5] = {4, 5, 6, 7, 8};
            for (U32 r = 0; r < R; ++r) {
                for (U32 rw : {0U, 1U, 3U}) {
                    CopyRowRef(exp_cn, exp_cpe, sn, spe, selPg[r], r, rw, NR, ER);
                }
            }
        }
        if (cn != exp_cn || cpe != exp_cpe) {
            throw std::runtime_error("cp");
        }
    });
}

static void RunUnionStep(
    int32_t* topk, int32_t* out, int32_t* lens, uint32_t h, uint32_t q, uint32_t w, uint32_t cap, int aon,
    std::vector<int32_t>& ls, std::vector<int32_t>& coh, std::vector<float>& mem,
    std::vector<int32_t>& rc, std::vector<int32_t>& gc, std::vector<U32>& bm
) {
    regions.clear();
    Reg(topk, q * 8192U);
    Reg(out, cap * 4U);
    Reg(lens, 32U);
    ([](auto&... a) { (Reg(a.data(), a.size() * sizeof(a[0])), ...); })(ls, coh, mem, rc, gc, bm);
    for (blockIndex = 0; blockIndex < 32U; ++blockIndex) {
        sparse_prefill_adaptive_union(R8(topk), P8(ls), P8(coh), h, q);
    }
    g_sync.Reset();
    std::exception_ptr firstErr;
    std::mutex errMu;
    std::vector<std::thread> workers;
    workers.reserve(32);
    for (U32 tid = 0; tid < 32U; ++tid) {
        workers.emplace_back([&, tid] {
            blockIndex = tid;
            activePipes = 0;
            queueDepth = 0;
            assertOn = (aon != 0);
            cleanCacheLines.clear();
            try {
                sparse_prefill_partition_finish_fixed(
                    P8(ls), P8(coh), P8(mem), P8(rc), P8(gc), P8(bm), R8(out), R8(lens), h, q, w, cap
                );
            } catch (...) {
                g_sync.Abort();
                std::lock_guard<std::mutex> lk(errMu);
                if (!firstErr) {
                    firstErr = std::current_exception();
                }
            }
        });
    }
    for (auto& th : workers) {
        th.join();
    }
    if (firstErr) {
        std::rethrow_exception(firstErr);
    }
}

int test_dcache_guard() {
    return Guard(true, [&] {
        std::vector<int32_t> buf(32, 0);
        buf[0] = 42;
        buf[16] = 99;
        regions.clear();
        Reg(buf.data(), buf.size() * sizeof(int32_t));
        GlobalTensor<int32_t> g;
        g.SetGlobalBuffer(buf.data());
        auto expectStale = [&](Z idx) {
            bool threw = false;
            try {
                (void)g.GetValue(idx);
            } catch (const std::runtime_error& e) {
                threw = std::string(e.what()) == "stale_dcache";
            }
            if (!threw) {
                throw std::runtime_error("expected_stale_dcache");
            }
        };
        expectStale(0);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g[0]);
        if (g.GetValue(0) != 42) {
            throw std::runtime_error("dcache_val");
        }
        expectStale(16);
        {
            TPipe pipe;
            Buf srcBuf;
            srcBuf.v.assign(64, 0);
            auto loc = srcBuf.Get<int32_t>();
            loc.SetValue(0, 77);
            DataCopy(g, loc, 16U);
            expectStale(0);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g[0]);
            if (g.GetValue(0) != 77) {
                throw std::runtime_error("dcache_datacopy_val");
            }
            blockCount = 1U;
            SyncAll();
            blockCount = 32U;
            expectStale(0);
        }
    });
}

int test_merge_sorted_runs(uint32_t lc, uint32_t rc) {
    return Guard(true, [&] {
        TPipe pipe;
        Buf lb, rb, db;
        const U32 total = lc + rc;
        lb.v.assign((total + 8U) * 8U, 0);
        rb.v.assign((total + 8U) * 8U, 0);
        db.v.assign((total + 8U) * 8U, 0x5A);
        auto left = lb.Get<float>(), right = rb.Get<float>(), dst = db.Get<float>();
        std::vector<U32> expected;
        for (U32 side = 0; side < 2U; ++side) {
            auto src = side == 0U ? left : right;
            const U32 count = side == 0U ? lc : rc;
            for (U32 i = 0; i < count; ++i) {
                const U32 id = i * 2U + side;
                src.SetValue(i * 2U, -static_cast<float>(id));
                src.ReinterpretCast<U32>().SetValue(i * 2U + 1U, id);
                expected.push_back(id);
            }
        }
        std::sort(expected.begin(), expected.end());
        MergeSortedRuns(dst, left, right, lc, rc);
        for (U32 i = 0; i < total; ++i) {
            if (dst.GetValue(i * 2U) != -static_cast<float>(expected[i]) ||
                dst.ReinterpretCast<U32>().GetValue(i * 2U + 1U) != expected[i]) {
                throw std::runtime_error("merge_pair_mismatch");
            }
        }
        for (U32 i = total * 2U; i < (total + 8U) * 2U; ++i) {
            if (dst.ReinterpretCast<U32>().GetValue(i) != 0x5A5A5A5AU) {
                throw std::runtime_error("merge_tail_overwrite");
            }
        }
    });
}

int test_union_pipeline(int32_t* topk, int32_t* out, int32_t* lens, uint32_t h, uint32_t q, uint32_t cap, int aon) {
    return Guard(aon != 0, [&] {
        U32 ch = (q + 1U) / 2U, w = std::max<U32>(128U, ((h + 4095U) / 4096U) * 128U);
        std::vector<int32_t> ls(ch * COHORT_SIZE, -1), coh(ch * 8, 0), rc(256, 0), gc(256, 0);
        std::vector<float> mem(32 * w, 0);
        std::vector<U32> bm(32 * (((w + 31U) / 32U + 63U) / 64U * 64U), 0);
        RunUnionStep(topk, out, lens, h, q, w, cap, aon, ls, coh, mem, rc, gc, bm);
    });
}

int test_union_pipeline_reuse(
    const uint32_t* hs, const uint32_t* qs, const uint32_t* caps,
    int32_t** topks, int32_t** outs, int32_t** lenss, uint32_t steps, uint32_t max_h, uint32_t max_q, int aon
) {
    return Guard(aon != 0, [&] {
        U32 max_ch = (max_q + 1U) / 2U, max_w = std::max<U32>(128U, ((max_h + 4095U) / 4096U) * 128U);
        std::vector<int32_t> ls(max_ch * COHORT_SIZE, 0x33333333), coh(max_ch * 8, 0x44444444);
        std::vector<int32_t> rc(256, 0x55555555), gc(256, 0x66666666);
        std::vector<float> mem(32 * max_w, 1234.0F);
        std::vector<U32> bm(32 * (((max_w + 31U) / 32U + 63U) / 64U * 64U), 0xAAAAAAAAU);
        for (uint32_t s = 0; s < steps; ++s) {
            U32 w = std::max<U32>(128U, ((hs[s] + 4095U) / 4096U) * 128U);
            RunUnionStep(topks[s], outs[s], lenss[s], hs[s], qs[s], w, caps[s], aon, ls, coh, mem, rc, gc, bm);
        }
    });
}
}
"""


def _compile(tmp: Path) -> ctypes.CDLL:
    if (cxx := shutil.which("g++") or shutil.which("clang++")) is None:
        raise unittest.SkipTest("Host C++17 compiler required")
    li, nat = tmp / "lightning_indexer_vllm" / "op_kernel", tmp / "sparse_prefill_native"
    li.mkdir(parents=True)
    nat.mkdir(parents=True)
    for x in KDIR.glob("*"):
        shutil.copyfile(x, nat / x.name)
    for dst, txt in (
        (nat / "kernel_operator.h", HOST_STUB),
        (li / "lightning_indexer_common.h", "#pragma once\n"),
        (li / "lightning_indexer_vector.h", VEC_H),
        (nat / "h.cpp", CPP_TESTS),
    ):
        dst.write_text(txt)
    srcs = [str(nat / "h.cpp")]
    for name, fname in (
        ("m.cpp", "bf16_mla_inplace.cpp"),
        ("u.cpp", "sparse_prefill_union_pipeline.cpp"),
        ("f.cpp", "sparse_prefill_fused_fixed.cpp"),
    ):
        dst = nat / name
        dst.write_text((KDIR / fname).read_text().split('extern "C" int32_t launch')[0])
        srcs.append(str(dst))
    so = tmp / "libhost.so"
    cmd = [cxx, "-shared", "-fPIC", "-std=c++17", "-O2", "-UNDEBUG", "-DASCENDC_DUMP=1", "-pthread", f"-I{nat}"]
    res = subprocess.run([*cmd, "-Wall", "-Wextra", "-Werror", *srcs, "-o", str(so)], capture_output=True)
    assert res.returncode == 0, res.stderr.decode("utf-8", errors="replace")
    lib = ctypes.CDLL(str(so))
    lib.host_last_error.restype = ctypes.c_char_p
    lib.test_dcache_guard.argtypes = []
    lib.test_merge_sorted_runs.argtypes = [ctypes.c_uint32] * 2
    lib.test_mla_decode_and_copy.argtypes = [ctypes.c_int32] * 3
    lib.test_union_pipeline.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32] * 3 + [ctypes.c_int32]
    lib.test_union_pipeline_reuse.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 3 + [ctypes.c_int32]
    return lib


class SelectedKernelsHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._td = tempfile.TemporaryDirectory(prefix="sel-host-")
        cls.lib = _compile(Path(cls._td.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._td.cleanup()

    def test_mla_copy_coalesce_lanes_abort_and_slice_guard(self) -> None:
        err = lambda: self.lib.host_last_error().decode()
        # (a) Normal copy: gc=1U, gc=8U (R=2), and gc=8U (R=5 -> group 0 executes r=0 and r=4)
        for gc, mode in ((1, 0), (8, 0), (8, 3)):
            self.assertEqual(self.lib.test_mla_decode_and_copy(gc, mode, 1), 0, err())
        # (b) Invalid-row abort with assertions enabled -> throws device assert message
        self.assertEqual(self.lib.test_mla_decode_and_copy(1, 1, 1), 1)
        self.assertIn("Invalid selected row address", err())
        # (c) Assertions off: AbortCopy drains pending=2 via DiscardPending; r0 pages untouched, r1 writes row 0
        self.assertEqual(self.lib.test_mla_decode_and_copy(1, 1, 0), 0, err())
        # (d) LoadRequestSlice metadata/status mismatch abort (aon=1 and aon=0 leaving destination untouched)
        self.assertEqual(self.lib.test_mla_decode_and_copy(1, 2, 1), 1)
        self.assertIn("Invalid selected copy metadata or union status", err())
        self.assertEqual(self.lib.test_mla_decode_and_copy(1, 2, 0), 0, err())
        # (e) GlobalTensor scalar D$ invalidation guard
        self.assertEqual(self.lib.test_dcache_guard(), 0, err())

    def test_merge_source_length_split_on_either_side(self) -> None:
        # Each case exceeds the API source bound on one side only. Removing
        # that side's split must fail the independent MrgSort stub guard.
        for left, right in ((4096, 1), (1, 4096)):
            with self.subTest(left=left, right=right):
                self.assertEqual(self.lib.test_merge_sorted_runs(left, right), 0, self.lib.host_last_error().decode())

    def test_union_pipeline_status_adaptive_bounds_overflow_and_error(self) -> None:
        def run(h: int, q: int, cap: int, rows, aon: int = 1):
            tk = np.zeros((q, 2048), dtype=np.int32)
            for i, r in enumerate(rows):
                tk[i] = r[0] if len(r) == 1 else r
            out = np.full(cap, 0x5A5A5A5A, dtype=np.int32)
            ln = np.full(8, 0x77777777, dtype=np.int32)
            rc = self.lib.test_union_pipeline(tk.ctypes.data, out.ctypes.data, ln.ctypes.data, h, q, cap, aon)
            self.assertEqual(rc, 0, self.lib.host_last_error().decode())
            return ln.tolist(), out

        # 1. h=65537, q=5, cap=128 with duplicates and causal ids >= h ignored
        r0 = np.array([10, 0, 10, 65537, 65540, 256] + [0] * 2042, dtype=np.int32)
        ln, out = run(65537, 5, 128, [r0, [256], [65536], [65538], [10]])
        self.assertEqual(ln, [4, 5, 9, 0, 0, 0, 0, 0])
        self.assertEqual(out[:4].tolist(), [0, 10, 256, 65536])
        self.assertTrue(np.all(out[4:] == 0x5A5A5A5A))

        # 2. Adaptive q=130 > 128, cap=128
        rows_ad = [np.arange(i % 4, i % 4 + 2048, dtype=np.int32) % 64 for i in range(130)]
        ln, out = run(256, 130, 128, rows_ad)
        self.assertEqual(ln, [64, 130, 194, 0, 0, 0, 0, 0])
        self.assertEqual(out[:64].tolist(), list(range(64)))
        self.assertTrue(np.all(out[64:] == 0x5A5A5A5A))

        # 3. Exact-fit total == cap == 128 (ovf=0) & 4. Overflow total=129, cap=128 (ovf=1)
        ln, out = run(1024, 2, 128, [np.arange(128, dtype=np.int32).tolist() + [0] * 1920, [0]])
        self.assertEqual(ln, [128, 2, 130, 0, 0, 0, 0, 0])
        self.assertEqual(out.tolist(), list(range(128)))
        r_ovf = [list(range(128)) + [0] * 1920, [128] + [0] * 2047]
        ln, _ = run(1024, 2, 128, r_ovf)
        self.assertEqual(ln, [129, 2, 131, 0, 1, 0, 0, 0])

        # 5. Max history h=524288 with id=524287 (words_=512)
        ln, out = run(524288, 2, 256, [[524287], [0]])
        self.assertEqual(ln, [2, 2, 4, 0, 0, 0, 0, 0])
        self.assertEqual(out[:2].tolist(), [0, 524287])
        self.assertTrue(np.all(out[2:] == 0x5A5A5A5A))

        # 6. Zero history h=0, q=4, cap=128 -> total=0
        ln, out = run(0, 4, 128, [[0], [1], [2], [3]])
        self.assertEqual(ln, [0, 4, 4, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(out == 0x5A5A5A5A))

        # 7. Out-of-range bad=-2 and bad=h+q -> err=1
        for bad in (-2, 260):
            ln, _ = run(256, 4, 128, [[bad] + [1] * 2047, [2], [2], [2]])
            self.assertEqual(ln, [2, 4, 6, 1, 0, 0, 0, 0])

        # 8. Q > 128 merged-run > CHUNK (4096): h=8192, q=130, cap=8192 (rows 0..3 -> 6000 merged IDs)
        rows_chunk = [
            np.arange(0, 2048, dtype=np.int32),
            np.concatenate([np.arange(0, 1096, dtype=np.int32), np.arange(2048, 3000, dtype=np.int32)]),
            np.arange(3000, 5048, dtype=np.int32),
            np.concatenate([np.arange(3000, 4096, dtype=np.int32), np.arange(5048, 6000, dtype=np.int32)]),
        ] + [[0] for _ in range(126)]
        ln, out = run(8192, 130, 8192, rows_chunk)
        self.assertEqual(ln, [6000, 130, 6130, 0, 0, 0, 0, 0])
        self.assertEqual(out[:6000].tolist(), list(range(6000)))
        self.assertTrue(np.all(out[6000:] == 0x5A5A5A5A))

        # Two disjoint 4096-id cohorts exceed MERGE_IDS together. The caller
        # must handle capacity before merging; helper input (4096,4096) is
        # intentionally outside its reachable production precondition.
        rows_full = [np.arange(i * 2048, (i + 1) * 2048, dtype=np.int32) for i in range(4)]
        # Q=128 disables adaptive direct writes, so the second cohort must
        # flush the accumulator before the 4096+4096 merge would be attempted.
        rows_full += [[0] for _ in range(124)]
        ln, out = run(16384, 128, 8320, rows_full)
        self.assertEqual(ln, [8192, 128, 8320, 0, 0, 0, 0, 0])
        self.assertEqual(out[:8192].tolist(), list(range(8192)))
        self.assertTrue(np.all(out[8192:] == 0x5A5A5A5A))

        # 9. Q = 1026 (cohorts = 513 > COHORT_META_GROUPS = 512) multi-batch descriptor loop
        rows_multi = [[i % 16] for i in range(1024)] + [[777], [1500]]
        ln, out = run(2048, 1026, 256, rows_multi)
        self.assertEqual(ln, [18, 1026, 1044, 0, 0, 0, 0, 0])
        self.assertEqual(out[:18].tolist(), list(range(16)) + [777, 1500])
        self.assertTrue(np.all(out[18:] == 0x5A5A5A5A))

        # 10. Variable-history buffer reuse: h=65536 (words_=64) -> h=131072 (words_=128) -> h=65536 (words_=64)
        step_specs = [
            (65536, 4, 64, [[10], [65535], [10], [0]], [0, 10, 65535]),
            (131072, 4, 64, [[5], [65536], [131071], [5]], [5, 65536, 131071]),
            (65536, 4, 64, [[7], [65535], [65538], [7]], [7, 65535]),
        ]
        tks, outs, lns = [], [], []
        for _, q_s, cap_s, rows_s, _ in step_specs:
            tk_s = np.zeros((q_s, 2048), dtype=np.int32)
            for idx, r_s in enumerate(rows_s):
                tk_s[idx] = r_s[0] if len(r_s) == 1 else r_s
            tks.append(tk_s)
            outs.append(np.full(cap_s, 0x5A5A5A5A, dtype=np.int32))
            lns.append(np.full(8, 0x77777777, dtype=np.int32))
        hs = np.array([s[0] for s in step_specs], dtype=np.uint32)
        qs = np.array([s[1] for s in step_specs], dtype=np.uint32)
        caps = np.array([s[2] for s in step_specs], dtype=np.uint32)
        tk_ptrs = np.array([a.ctypes.data for a in tks], dtype=np.uintp)
        out_ptrs = np.array([a.ctypes.data for a in outs], dtype=np.uintp)
        ln_ptrs = np.array([a.ctypes.data for a in lns], dtype=np.uintp)
        rc = self.lib.test_union_pipeline_reuse(
            hs.ctypes.data,
            qs.ctypes.data,
            caps.ctypes.data,
            tk_ptrs.ctypes.data,
            out_ptrs.ctypes.data,
            ln_ptrs.ctypes.data,
            len(step_specs),
            131072,
            4,
            1,
        )
        self.assertEqual(rc, 0, self.lib.host_last_error().decode())
        for (_, q_s, _, _, exp), out_s, ln_s in zip(step_specs, outs, lns):
            self.assertEqual(ln_s.tolist(), [len(exp), q_s, len(exp) + q_s, 0, 0, 0, 0, 0])
            self.assertEqual(out_s[: len(exp)].tolist(), exp)
            self.assertTrue(np.all(out_s[len(exp) :] == 0x5A5A5A5A))


if __name__ == "__main__":
    unittest.main()
