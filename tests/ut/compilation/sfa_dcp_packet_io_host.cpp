// SPDX-License-Identifier: Apache-2.0
// Run the production method bodies against bounded memory and synchronous DMA.
#include <algorithm>
#include <array>
#ifdef NDEBUG
  #undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#define SFA_DCP_PACKET
#define __aicore__
#include "sparse_flash_attention_packet_layout.h"
template <class T>
T Min(T a, T b) {
  return std::min(a, b);
}
template <class T>
T SFAAlign(T a, T b) {
  return (a + b - 1) / b * b;
}
struct Info {
  uint32_t qHeadNum, headDim = 512;
};
enum class HardEvent { MTE3_V, V_S, S_MTE3, S_V, V_MTE3 };
enum class RoundMode { CAST_NONE, CAST_RINT };
constexpr int PIPE_V = 0, SYNC_OUTPUT_BUF2_FLAG = 5, BYTE_BLOCK = 32;
template <HardEvent>
void WaitFlag(int) {}
template <HardEvent>
void SetFlag(int) {}
template <int>
void PipeBarrier() {}
template <class A, class B>
using IsSameType = std::is_same<A, B>;
// Only integer digits 0..255 are converted by the packet encoder.
struct half {
  uint16_t bits;
};
struct bfloat16_t {
  uint16_t bits;
};
uint32_t bits(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  return b;
}
template <class T>
uint16_t digit(unsigned x) {
  if constexpr (std::is_same_v<T, half>) return x ? uint16_t((bits(float(x)) >> 13) - (112 << 10)) : 0;
  return uint16_t(bits(float(x)) >> 16);
}
template <class T>
struct Tensor {
  uint8_t* p;
  size_t bytes;
  template <class U>
  Tensor<U> ReinterpretCast() const {
    return {p, bytes};
  }
  Tensor operator[](size_t i) const {
    assert(i * sizeof(T) <= bytes);
    return {p + i * sizeof(T), bytes - i * sizeof(T)};
  }
  T GetValue(size_t i) const {
    T v;
    assert((i + 1) * sizeof(T) <= bytes);
    std::memcpy(&v, p + i * sizeof(T), sizeof(T));
    return v;
  }
  void SetValue(size_t i, T v) {
    assert((i + 1) * sizeof(T) <= bytes);
    std::memcpy(p + i * sizeof(T), &v, sizeof(T));
  }
};
template <class T>
Tensor<T> view(std::vector<T>& v) {
  return {reinterpret_cast<uint8_t*>(v.data()), v.size() * sizeof(T)};
}
struct Buffer {
  std::array<uint8_t, 4096> mem{};
  template <class T>
  Tensor<T> Get() {
    return {mem.data(), mem.size()};
  }
};
template <class T>
void Duplicate(Tensor<T> a, T val, unsigned n) {
  for (unsigned i = 0; i < n; ++i) a.SetValue(i, val);
}
template <class T>
void DataCopy(Tensor<T> a, Tensor<T> b, unsigned n) {
  assert(a.bytes >= n * sizeof(T) && b.bytes >= n * sizeof(T));
  std::memcpy(a.p, b.p, n * sizeof(T));
}
struct DataCopyExtParams {
  uint16_t blockCount;
  uint32_t blockLen, srcStride, dstStride, rsv;
};
struct Nd2NzParams {
  unsigned ndNum, nValue, dValue, srcDValue, dstNzC0Stride, dstNzNStride, srcNdMatrixStride, dstNzMatrixStride;
};
template <class T>
void DataCopy(Tensor<T> dst, Tensor<T> src, Nd2NzParams p) {
  assert(p.ndNum == 1 && p.dstNzNStride == 1 && p.srcNdMatrixStride == 0 && p.dstNzMatrixStride == 0);
  for (unsigned h = 0; h < p.nValue; ++h)
    for (unsigned d = 0; d < p.dValue; ++d)
      dst.SetValue((d / 16 * p.dstNzC0Stride + h) * 16 + d % 16, src.GetValue(h * p.srcDValue + d));
}
template <class T>
void DataCopyPad(Tensor<T> a, Tensor<T> b, DataCopyExtParams d) {
  for (unsigned i = 0; i < d.blockCount; ++i)
    DataCopy(a[i * (d.blockLen + d.dstStride) / 2], b[i * (d.blockLen + 32 * d.srcStride) / 2], d.blockLen / 2);
}
template <class D, class S>
void Cast(Tensor<D> dst, Tensor<S> src, RoundMode, unsigned n) {
  for (unsigned i = 0; i < n; ++i) {
    auto s = src.GetValue(i);
    if constexpr (std::is_same_v<D, float>) {
      auto v = s.bits;
      dst.SetValue(i, v ? std::ldexp(1.f + float(v & 1023) / 1024, int(v >> 10) - 15) : 0.f);
    } else
      dst.SetValue(i, D{digit<D>(unsigned(s))});
  }
}
struct RunInfo {
  uint64_t attenOutOffset;
};
template <class OUT_T>
struct Kernel {
  Info constInfo;
  DcpPacketLayout packet;
  Tensor<OUT_T> queryGm, attentionOutGm;
  Tensor<float> softmaxMaxUb, softmaxSumUb;
  Buffer outputBuff2;
  void CopyGmToL1(Tensor<OUT_T> l1Tensor, Tensor<OUT_T> gmSrcTensor, unsigned srcN, unsigned srcD, unsigned srcDstride,
                  unsigned dstRows)
#include "gm_to_l1.inc"
      void read(Tensor<OUT_T> dst, uint64_t logical, uint32_t rows, uint32_t width, uint32_t feature)
#include "read.inc"
          void partial(RunInfo info, Tensor<OUT_T> attenOutUb, uint32_t wsMStart, uint32_t dealRowCount,
                       uint32_t columnCount, uint32_t actualColumnCount)
#include "partial.inc"
              void stats(uint64_t logical, uint32_t src, uint32_t rows)
#include "stats.inc"
                  void zero(uint64_t row)
#include "zero.inc"
};
template <class T>
void check(unsigned world, unsigned rows, unsigned heads) {
  const unsigned n = rows * heads, local = heads / world;
  std::vector<T> q(n * 576), wire(n * 528, T{0xdead}), p(n * 512);
  std::vector<float> m(n), s(n);
  Kernel<T> k{{heads}, {heads, rows, local, true}, view(q), view(wire), view(m), view(s)};
  for (unsigned t = 0; t < rows; ++t)
    for (unsigned h = 0; h < heads; ++h) {
      unsigned i = t * heads + h, idx = ((h / local) * rows + t) * local + h % local;
      m[i] = float(i) - 62.125f;
      s[i] = i % 3 ? float(i) + .375f : 0.f;
      for (unsigned d = 0; d < 576; ++d) q[idx * 576 + d].bits = uint16_t(i * 733 + d);
      for (unsigned d = 0; d < 512; ++d) p[i * 512 + d].bits = uint16_t(i * 509 + d);
    }
  for (unsigned i = 0; i < n;) {
    unsigned count = std::min(39u, n - i), aligned = (count + 15) / 16 * 16;
    for (unsigned feature : {0u, 256u, 512u}) {
      unsigned width = feature == 512 ? 64 : 256;
      std::vector<T> nz(aligned * width, T{0xdead});
      k.read(view(nz), i, count, width, feature);
      for (unsigned h = 0; h < count; ++h)
        for (unsigned d = 0; d < width; ++d)
          assert(nz[d / 16 * aligned * 16 + h * 16 + d % 16].bits == uint16_t((i + h) * 733 + feature + d));
    }
    unsigned rowOffset = i % 7;
    k.partial({(i - rowOffset) * 512}, view(p)[i * 512], rowOffset, count, 512, 512);
    k.stats(i, i, count);
    i += count;
  }
  for (unsigned pass = 0; pass < 2; ++pass) {
    if (pass) k.zero(rows - 1);
    for (unsigned t = 0; t < rows; ++t)
      for (unsigned h = 0; h < heads; ++h) {
        unsigned i = t * heads + h, off = (((h / local) * rows + t) * local + h % local) * 528;
        bool zero = pass && t == rows - 1;
        for (unsigned d = 0; d < 512; ++d) assert(wire[off + d].bits == (zero ? 0 : p[i * 512 + d].bits));
        uint32_t max = zero || s[i] == 0 ? 0xff800000u : bits(m[i]), sum = zero ? 0 : bits(s[i]);
        for (unsigned b = 0; b < 16; ++b) {
          unsigned x = b < 4 ? max : b < 8 ? sum : 0;
          assert(wire[off + 512 + b].bits == digit<T>(x >> (8 * (b % 4)) & 255));
        }
      }
  }
}
int main() {
  for (unsigned d : {2, 4, 8})
    for (unsigned t : {1, 2, 3, 8, 16})
      for (unsigned h : {8, 64}) {
        check<half>(d, t, h);
        check<bfloat16_t>(d, t, h);
      }
}
