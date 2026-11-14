# SPDX-License-Identifier: Apache-2.0
"""Run the exact copy kernel body with scalar CPU memory models, not NPU codegen."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class SelectedInplaceKernelTest(unittest.TestCase):
    def test_double_buffer_queue_and_original_page_bytes(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        kernel = ROOT / "csrc/sparse_prefill_native"
        with tempfile.TemporaryDirectory(prefix="selected-inplace-kernel-") as directory:
            build = Path(directory)
            (build / "kernel_operator.h").write_text('#include "selected_inplace_host_stub.h"\n')
            # The SDK launch expression is host-incompatible. Preserve the entire
            # real device body; only omit the host launch function after it.
            body = (
                (kernel / "packed_inplace.cpp")
                .read_text()
                .split('extern "C" int32_t launch_sparse_prefill_packed_inplace', 1)[0]
            )
            (build / "test.cpp").write_text(
                body
                + r"""
#include <cassert>
#include <string_view>
#include <vector>

uintptr_t metadataStart = 0;
size_t metadataBytes = 0;
size_t metadataRowReads = 0;

void CheckRequestStartsWithEmptyQueue(const void* data, size_t bytes, bool write) {
    const auto address = reinterpret_cast<uintptr_t>(data);
    if (!write && bytes == sizeof(int32_t) && address >= metadataStart &&
        address - metadataStart < metadataBytes && (address - metadataStart) % (4 * sizeof(int32_t)) == 0) {
        ++metadataRowReads;
        assert(AscendC::copyQueueDepth == 0);
    }
}

void CheckCopy(uint32_t count, uint32_t groupCores, uint32_t full, bool sparse,
               int error = 0, uint32_t placement = 0) {
    using namespace AscendC;
    regions.clear();
    copyQueuePeak = 0;
    const uint32_t h = 4093, columns = HistoryPages(h), stride = columns + 3;
    const uint32_t requests = 3, pages = requests * columns + 2, pageBytes = PAGE_ROWS * ROW_BYTES;
    const uint32_t capacity = std::max(128U, (count + 127U) / 128U * 128U);
    std::vector<int32_t> table(requests * stride, -1), ids(requests * capacity, -1);
    std::vector<int32_t> status(requests * 8, 0), metadata(requests * 4, 0);
    std::vector<uint8_t> source(pages * pageBytes), cache(source.size(), 0xa5), expected = cache;
    std::vector<uint64_t> bases(pages, 0);
    for (uint32_t page = 0; page < pages; ++page) {
        for (uint32_t byte = 0; byte < pageBytes; ++byte)
            source[page * pageBytes + byte] = (page * 31 + byte * 17 + byte / ROW_BYTES) % 251;
        // Independent source placement: logical/physical IDs cannot replace a DDR base lookup.
        bases[page] = reinterpret_cast<uintptr_t>(source.data() + (page * 17 % pages) * pageBytes);
    }
    for (uint32_t r = 0; r < requests; ++r) {
        const uint32_t row = (r * 2) % requests;
        for (uint32_t col = 0; col < columns; ++col) {
            const uint32_t page = 1 + row * columns + (placement == 2 ? col : col * 13 % columns);
            table[row * stride + col] = page;
            if (placement != 0)
                bases[page] = reinterpret_cast<uintptr_t>(source.data() + (row * columns + col) * pageBytes);
            if (full == 2 || (full == 1 && (col % 3 == 1 || col == columns - 1))) bases[page] = 0;
        }
        metadata[r * 4] = row;
        metadata[r * 4 + 1] = h;
        metadata[r * 4 + 2] = r * 125;
        metadata[r * 4 + 3] = 125;
        status[r * 8] = count;
        status[r * 8 + 1] = 125;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t id = sparse ? static_cast<uint64_t>(i) * h / count : i;
            ids[r * capacity + i] = id;
            const uint32_t page = table[row * stride + id / PAGE_ROWS];
            if (bases[page]) {
                const uint32_t offset = id % PAGE_ROWS * ROW_BYTES;
                std::memcpy(expected.data() + page * pageBytes + offset,
                    reinterpret_cast<void*>(bases[page] + offset), ROW_BYTES);
            }
        }
    }
    if (error == 1) ids[std::min(count - 1, 55U)] = h;
    if (error == 2) table[0] = pages;
    if (error == 3) bases[table[0]] = UINT64_MAX - (ROW_BYTES - 2U);
    if (error == 4) status[3] = 1;
    if (error == 5) metadata[3] = status[1] = MAX_BATCHED_TOKENS + 1;
    if (error == 6) metadata[2] = MAX_BATCHED_TOKENS - metadata[3] + 1;
    Register(table.data(), table.size() * 4); Register(ids.data(), ids.size() * 4);
    Register(status.data(), status.size() * 4); Register(metadata.data(), metadata.size() * 4);
    Register(bases.data(), bases.size() * 8); Register(source.data(), source.size());
    Register(cache.data(), cache.size());
    metadataStart = reinterpret_cast<uintptr_t>(metadata.data());
    metadataBytes = metadata.size() * sizeof(int32_t);
    metadataRowReads = 0;
    observeGlobalAccess = CheckRequestStartsWithEmptyQueue;
    bool rejected = false;
    try {
        for (blockIndex = 0; blockIndex < 32; ++blockIndex)
            sparse_prefill_packed_inplace(reinterpret_cast<uint8_t*>(ids.data()),
                reinterpret_cast<uint8_t*>(status.data()), reinterpret_cast<uint8_t*>(bases.data()),
                reinterpret_cast<uint8_t*>(metadata.data()), reinterpret_cast<uint8_t*>(table.data()),
                cache.data(), requests, capacity, pages, requests, columns, stride, pages, groupCores);
    } catch (const std::runtime_error& exception) {
        constexpr const char* errors[] = {"", "Selected copy token outside history",
            "Selected copy page outside source or destination domain", "Invalid selected row address",
            "Invalid original-page copy metadata or union status",
            "Invalid original-page copy metadata or union status",
            "Invalid original-page copy metadata or union status"};
        assert(error > 0 && std::string_view(exception.what()) == errors[error]);
        rejected = true;
    }
    observeGlobalAccess = nullptr;
    assert(rejected == (error != 0));
    if (!error) {
        assert(metadataRowReads == requests * groupCores);
        assert(cache == expected);
        if (count == 0 || full == 2) assert(copyQueuePeak == 0);
        if (placement == 2 && full == 0 && count == 49 * groupCores) assert(copyQueuePeak == 1);
        if (placement == 2 && full == 0 && count == 50 * groupCores) assert(copyQueuePeak == 2);
        // At least two payloads must coexist; otherwise this missed the queue path.
        if (count >= 2048 && full == 0) assert(copyQueuePeak == 2);
    }
}

void CheckInvalidLookaheadWithPendingRun() {
    using namespace AscendC;
    regions.clear();
    copyQueuePeak = 0;
    constexpr uint32_t capacity = 128, h = 128, pages = 2, pageBytes = PAGE_ROWS * ROW_BYTES;
    std::vector<uint8_t> source(pageBytes, 0x42), cache(pages * pageBytes, 0xa5);
    int32_t table[] = {1};
    int32_t ids[capacity] = {0, 2, h};
    int32_t status[8] = {24, 4097};
    int32_t metadata[4] = {0, h, 8192 - 4097, 4097};
    uint64_t bases[] = {0, reinterpret_cast<uintptr_t>(source.data())};
    Register(ids, sizeof(ids)); Register(status, sizeof(status)); Register(metadata, sizeof(metadata));
    Register(table, sizeof(table)); Register(bases, sizeof(bases));
    Register(source.data(), source.size()); Register(cache.data(), cache.size());
    blockIndex = 0;
    bool rejected = false;
    try {
        sparse_prefill_packed_inplace(reinterpret_cast<uint8_t*>(ids),
            reinterpret_cast<uint8_t*>(status), reinterpret_cast<uint8_t*>(bases),
            reinterpret_cast<uint8_t*>(metadata), reinterpret_cast<uint8_t*>(table),
            cache.data(), 1, capacity, pages, 1, 1, 1, pages, 8);
    } catch (const std::runtime_error& exception) {
        assert(std::string_view(exception.what()) == "Selected copy token outside history");
        rejected = true;
    }
    assert(rejected && copyQueuePeak == 1);
    assert(std::all_of(cache.begin(), cache.end(), [](uint8_t byte) { return byte == 0xa5; }));
}

void CheckQueueOwnership() {
    using namespace AscendC;
    {
        CopyQueue queue;
        queue.Init(2, 32);
        auto first = queue.AllocTensor<uint8_t>();
        auto second = queue.AllocTensor<uint8_t>();
        first.SetValue(0, 11); second.SetValue(0, 22);
        queue.EnQue(first); queue.EnQue(second);
        bool rejected = false;
        try { queue.FreeTensor(first); } catch (const std::logic_error&) { rejected = true; }
        assert(rejected);
        rejected = false;
        try { queue.AllocTensor<uint8_t>(); } catch (const std::logic_error&) { rejected = true; }
        assert(rejected);
        auto output = queue.DeQue<uint8_t>(); assert(output.GetValue(0) == 11); queue.FreeTensor(output);
        output = queue.DeQue<uint8_t>(); assert(output.GetValue(0) == 22); queue.FreeTensor(output);
    }
}

int main() {
    using namespace AscendC;
    for(uint32_t h : {65537U, 65661U, 65663U, 524287U}) {
        for(int32_t q : {1, 125, 127, 1023, 1025, 2048, 4097, 7984, 8192}) {
            regions.clear();
            const uint32_t columns = HistoryPages(h), pages = columns + 3;
            std::vector<int32_t> table(columns);
            for(uint32_t i=0;i<columns;++i) table[i]=i;
            // Highest physical page is the partial history/append boundary.
            table.back()=pages-1;
            std::vector<uint8_t> source(128*ROW_BYTES, 0x35), cache(pages*128*ROW_BYTES, 0xa5);
            std::vector<uint64_t> bases(pages, 0);
            bases[0]=reinterpret_cast<uintptr_t>(source.data());
            int32_t ids[128]={0,127,static_cast<int32_t>(h-1)};
            int32_t status[8]={3,q,3+q,0,0,0,0,0};
            int32_t meta[4]={0,static_cast<int32_t>(h),8192-q,q};
            Register(ids,sizeof(ids)); Register(status,sizeof(status)); Register(meta,sizeof(meta));
            Register(table.data(),table.size()*4); Register(bases.data(),bases.size()*8);
            Register(source.data(),source.size()); Register(cache.data(),cache.size());
            for(blockIndex=0;blockIndex<32;++blockIndex)
                sparse_prefill_packed_inplace(reinterpret_cast<uint8_t*>(ids),reinterpret_cast<uint8_t*>(status),
                    reinterpret_cast<uint8_t*>(bases.data()),reinterpret_cast<uint8_t*>(meta),
                    reinterpret_cast<uint8_t*>(table.data()),cache.data(),1,128,pages,1,columns,columns,pages,8);
            for(size_t i=0;i<cache.size();++i) {
                bool selected = i<ROW_BYTES || (i>=127*ROW_BYTES && i<128*ROW_BYTES);
                assert(cache[i]==(selected?0x35:0xa5));
            }
        }
    }
    CheckQueueOwnership();
    for (uint32_t cores : {8U, 16U, 32U}) {
        for (uint32_t perLane : {0U, 1U, 49U, 50U, 98U})
            CheckCopy(perLane * cores, cores, 0, false, 0, 2);
        CheckCopy(2048, cores, 1, false);
        CheckCopy(2048, cores, 2, false);
        CheckCopy(2048, cores, 0, true);
    }
    CheckCopy(3000, 8, 0, false, 0, 1);  // DDR adjacent, HBM fragmented: do not merge across pages.
    CheckCopy(3000, 8, 0, false, 0, 2);  // Both adjacent: allow exact cross-page runs.
    CheckCopy(3136, 8, 1, false);        // A FULL page between selected runs within one lane.
    for (int error = 1; error <= 6; ++error) CheckCopy(2048, 32, 0, false, error);
    CheckInvalidLookaheadWithPendingRun();
}
"""
            )
            executable = build / "copy"
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-DASCENDC_DUMP=1",
                    f"-I{build}",
                    f"-I{kernel}",
                    f"-I{Path(__file__).parent}",
                    str(build / "test.cpp"),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
