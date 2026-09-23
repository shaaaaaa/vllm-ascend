// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
#include "launch.h"
namespace redesign {
constexpr uint32_t T = 256;
constexpr uint32_t W = 8192;
template<AscendC::HardEvent E> __aicore__ inline void Fence() {
    int32_t id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(E));
    AscendC::SetFlag<E>(id); AscendC::WaitFlag<E>(id);
}
__aicore__ inline void V() { AscendC::PipeBarrier<PIPE_V>(); }
template<typename X> __aicore__ inline X Fresh(AscendC::GlobalTensor<X>& x, uint64_t i) {
    AscendC::DataCacheCleanAndInvalid<X, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(x[i]);
    return x.GetValue(i);
}
__aicore__ inline void And(AscendC::LocalTensor<uint8_t> a,
                           AscendC::LocalTensor<uint8_t> b, uint32_t n) {
    AscendC::And(a.ReinterpretCast<uint16_t>(), a.ReinterpretCast<uint16_t>(),
                b.ReinterpretCast<uint16_t>(), n / 16); V();
}
__aicore__ inline void Range(AscendC::LocalTensor<uint8_t> mask,
    AscendC::LocalTensor<int32_t> x, AscendC::LocalTensor<int32_t> tmp,
    int32_t lo, int32_t hi, uint32_t n) {
    if (hi <= lo) { AscendC::Duplicate(mask.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), n / 16); V(); return; }
    AscendC::Maxs(tmp, x, lo, n); V();
    AscendC::Mins(tmp, tmp, hi - 1, n); V();
    AscendC::Compare(mask, tmp, x, AscendC::CMPMODE::EQ, n); V();
}
__aicore__ inline void Mod(AscendC::LocalTensor<int32_t> dst,
    AscendC::LocalTensor<int32_t> src, AscendC::LocalTensor<int32_t> tmp,
    uint32_t size, uint32_t n) {
    uint32_t bits = 0;
    for (uint32_t s = size; s > 1; s >>= 1) ++bits;
    AscendC::Maxs(dst, src, static_cast<int32_t>(0), n); V();
    AscendC::ShiftRight(tmp, dst, static_cast<int32_t>(bits), n); V();
    AscendC::Muls(tmp, tmp, static_cast<int32_t>(size), n); V();
    AscendC::Sub(dst, dst, tmp, n); V();
}
} // namespace redesign
