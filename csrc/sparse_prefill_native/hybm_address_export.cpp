// SPDX-License-Identifier: Apache-2.0
// Host-only batching over the caller's already-loaded SDK address domain.
#include <cstdint>
#include <limits>

extern "C" int32_t dsa_export_hybm_components(
    uintptr_t mapper_address, const uint64_t* gvas, const uint64_t* sizes,
    uint64_t count, uint64_t offset, uint64_t bytes, uint64_t* output,
    uint64_t* failed_index, int32_t* mapper_status) {
    using Mapper = int32_t (*)(uint64_t, uint32_t, uint64_t*);
    constexpr uint32_t DEVICE_MEMORY = 1;
    constexpr uint64_t MAX_ADDRESS = std::numeric_limits<uint64_t>::max();
    if (!mapper_address || !failed_index || !mapper_status || !bytes ||
        (count && (!gvas || !sizes || !output))) return 1;
    auto mapper = reinterpret_cast<Mapper>(mapper_address);
    *mapper_status = 0;
    for (uint64_t i = 0; i < count; ++i) {
        *failed_index = i;
        const uint64_t base = gvas[i], size = sizes[i];
        if (!base || !size || base > MAX_ADDRESS - (size - 1) ||
            offset > size || bytes > size - offset) return 1;
        uint64_t begin = 0, end = 0;
        *mapper_status = mapper(base + offset, DEVICE_MEMORY, &begin);
        if (*mapper_status || !begin) return 2;
        *mapper_status = mapper(base + offset + bytes - 1, DEVICE_MEMORY, &end);
        if (*mapper_status || !end) return 2;
        if (begin > MAX_ADDRESS - (bytes - 1) || end != begin + bytes - 1) return 3;
        output[i] = begin;
    }
    return 0;
}
