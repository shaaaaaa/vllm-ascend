// SPDX-License-Identifier: Apache-2.0
#pragma once

// Logical [token, head] -> one contiguous block per head owner on the wire.
struct DcpPacketLayout {
    static constexpr uint32_t latent = 512, rope = 64, header = 16;
    static constexpr uint32_t queryWidth = latent + rope, sendWidth = latent + header;
    uint64_t qHeadNum, qSeqSize;
    uint32_t packetLocalHeads;
    bool packedQuery;
    __aicore__ inline uint64_t PacketRow(uint64_t logical) const
    {
        if (qSeqSize == 1) {
            return logical;
        }
        uint64_t head = logical % qHeadNum;
        return (head / packetLocalHeads * qSeqSize + logical / qHeadNum) * packetLocalHeads + head % packetLocalHeads;
    }
    __aicore__ inline uint32_t PacketRun(uint64_t logical, uint32_t remaining) const
    {
        uint32_t span = packetLocalHeads - uint32_t(logical % packetLocalHeads);
        return qSeqSize == 1 || remaining < span ? remaining : span;
    }
};
