// SPDX-License-Identifier: Apache-2.0
#include "common.h"
namespace redesign {
// Each block owns 256 directory cells. All writes stay inside that private
// aligned slice, including collision handling. No atomics or spin barriers.
class BuildTable {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    AscendC::GlobalTensor<int32_t> tags, ready, table;
    RedesignLaunch a;
public:
    __aicore__ inline void Init(RedesignLaunch args) {
        a = args; uint32_t n = a.queries * a.topk;
        tags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[2]));
        ready.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[4]));
        table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[7]));
        pipe.InitBuffer(buf, 7 * n * 4 + 2 * (n / 8) + T * 4);
    }
    __aicore__ inline void Process() {
        uint32_t n = a.queries * a.topk, slices = a.tableSize / T;
        auto x = buf.Get<int32_t>(); auto r = x[n]; auto bucket = r[n];
        auto tmp = bucket[n]; auto ids = tmp[n]; auto compact = ids[n];
        auto ones = compact[n];
        auto mask = ones[n].ReinterpretCast<uint8_t>(); auto other = mask[n / 8];
        auto out = other[n / 8].ReinterpretCast<int32_t>();
        for (uint32_t block = AscendC::GetBlockIdx(); block < a.requests * slices;
             block += AscendC::GetBlockNum()) {
            uint32_t request = block / slices, base = (block % slices) * T;
            AscendC::DataCopy(x, tags[static_cast<uint64_t>(request) * n], n);
            AscendC::DataCopy(r, ready[static_cast<uint64_t>(request) * n], n);
            Fence<AscendC::HardEvent::MTE2_V>();
            if (a.mode == 3) Mod(bucket, x, tmp, a.tableSize, n);
            else { AscendC::Adds(bucket, x, 0, n); V(); }
            Range(mask, bucket, tmp, base, base + T, n);
            Range(other, x, tmp, 0, a.universe, n); And(mask, other, n);
            AscendC::Duplicate(ones, static_cast<int32_t>(1), n); V();
            AscendC::Compare(other, r, ones, AscendC::CMPMODE::EQ, n); V(); And(mask, other, n);
            AscendC::CreateVecIndex(ids, static_cast<int32_t>(0), n); V();
            AscendC::Duplicate(out, static_cast<int32_t>(-1), T); V();
            AscendC::GatherMaskParams params;
            params.repeatTimes = 1; params.src0BlockStride = 1;
            params.src0RepeatStride = 8; params.src1RepeatStride = 8;
            uint64_t count = 0;
            AscendC::GatherMask(compact, ids, mask.ReinterpretCast<uint32_t>(),
                                true, n, params, count); V();
            Fence<AscendC::HardEvent::V_S>();
            // Deterministic last source slot wins a collision. A lost candidate
            // becomes a false miss; lookup always verifies token AND version.
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t source = static_cast<uint32_t>(compact.GetValue(i));
                out.SetValue(static_cast<uint32_t>(bucket.GetValue(source)) - base, source);
            }
            Fence<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopy(table[static_cast<uint64_t>(request) * a.tableSize + base], out, T);
            Fence<AscendC::HardEvent::MTE3_V>();
            Fence<AscendC::HardEvent::V_MTE2>();
        }
    }
};
}
extern "C" __global__ __aicore__ void resident_snapshot_build_redesign(RedesignLaunch a) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    redesign::BuildTable op; op.Init(a); op.Process();
}
void redesign_build(void* stream, const RedesignLaunch& a) {
    uint32_t blocks = a.requests * (a.tableSize / redesign::T);
    resident_snapshot_build_redesign<<<blocks < a.cores ? blocks : a.cores, nullptr, stream>>>(a);
}
