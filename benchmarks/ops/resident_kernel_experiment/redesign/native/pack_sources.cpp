// SPDX-License-Identifier: Apache-2.0
#include "common.h"
#include "pack_launch.h"
namespace redesign {
class PackSources {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    AscendC::GlobalTensor<int32_t> plan, tokens, boundary, selected, counts;
    AscendC::GlobalTensor<int64_t> oldSlots, newSlots, targets;
    PackLaunch a;
public:
    __aicore__ inline void Init(PackLaunch args) {
        a = args;
        plan.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[0]));
        tokens.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[1]));
        oldSlots.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(a.p[2]));
        newSlots.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(a.p[3]));
        boundary.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[4]));
        selected.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[5]));
        targets.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(a.p[6]));
        counts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.p[7]));
        pipe.InitBuffer(buf, 2*T*4 + T*8 + 3*T*4 + 3*T*8 + 3*16*4 + a.entries*8);
    }
    __aicore__ inline void Process() {
        uint32_t tiles = a.entries / T, total = a.requests * tiles;
        auto source = buf.Get<int32_t>(); auto raw = source[T];
        auto destination = raw[T].ReinterpretCast<int64_t>();
        auto ids = destination[T].ReinterpretCast<int32_t>();
        auto slots = ids[3*T].ReinterpretCast<int64_t>();
        auto num = slots[3*T].ReinterpretCast<int32_t>();
        auto oldMap = num[3*16].ReinterpretCast<int64_t>();
        for (uint32_t tile = AscendC::GetBlockIdx(); tile < total; tile += AscendC::GetBlockNum()) {
            uint32_t request = tile / tiles, base = tile * T;
            AscendC::DataCopy(source, plan[base], T);
            AscendC::DataCopy(raw, tokens[base], T);
            AscendC::DataCopy(destination, newSlots[base], T);
            // Mapping is updated between steps: DMA avoids stale scalar GM cache.
            AscendC::DataCopy(oldMap, oldSlots[static_cast<uint64_t>(request) * a.entries], a.entries);
            Fence<AscendC::HardEvent::MTE2_S>();
            int32_t split = Fresh(boundary, request);
            uint32_t sizes[3] = {0, 0, 0}; // host misses, resident hits, live tail
            for (uint32_t j = 0; j < T; ++j) {
                int32_t src = source.GetValue(j), token = raw.GetValue(j);
                if (src < -3 || src == -2 || src >= static_cast<int32_t>(a.entries)) continue;
                uint32_t kind = src == -1 ? 0 : (src == -3 ? 2 : 1);
                int32_t index = kind == 0 ? token : (kind == 2 ? token - split :
                    static_cast<int32_t>(oldMap.GetValue(src)));
                uint32_t offset = kind * T + sizes[kind]++;
                ids.SetValue(offset, index);
                slots.SetValue(offset, destination.GetValue(j));
            }
            for (uint32_t kind = 0; kind < 3; ++kind) {
                for (uint32_t i = 0; i < 16; ++i) num.SetValue(kind * 16 + i, i == 0 ? sizes[kind] : 0);
            }
            Fence<AscendC::HardEvent::S_MTE3>();
            for (uint32_t kind = 0; kind < 3; ++kind) {
                uint64_t row = static_cast<uint64_t>(kind) * total + tile;
                // Only counts[row,0] entries are consumed; padded descriptors are ignored.
                AscendC::DataCopy(selected[row*T], ids[kind*T], T);
                AscendC::DataCopy(targets[row*T], slots[kind*T], T);
                AscendC::DataCopy(counts[row*16], num[kind*16], 16);
            }
            Fence<AscendC::HardEvent::MTE3_S>();
            Fence<AscendC::HardEvent::MTE3_MTE2>();
        }
    }
};
}
extern "C" __global__ __aicore__ void resident_pack_sources_redesign(PackLaunch a) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    redesign::PackSources op; op.Init(a); op.Process();
}
void redesign_pack_sources(void* stream, const PackLaunch& a) {
    uint32_t blocks = a.requests * a.entries / redesign::T;
    PackLaunch launch = a;
    resident_pack_sources_redesign<<<blocks < a.cores ? blocks : a.cores, nullptr, stream>>>(&launch);
}
