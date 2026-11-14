# SPDX-License-Identifier: Apache-2.0
"""Exercise the LI page gather kernel with a host model.

The model checks address ranges and both supported destination layouts. CANN
code generation, device barriers and DMA behavior require an NPU build and run.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "csrc/sparse_prefill_native"


class SelectedLIGatherTest(unittest.TestCase):
    def compiler(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        return compiler

    def test_real_copy_body_supports_native_and_packed_page_layouts(self):
        compiler = self.compiler()
        with tempfile.TemporaryDirectory(prefix="selected-li-body-") as directory:
            build = Path(directory)
            (build / "kernel_operator.h").write_text(
                '#include "selected_bitmap_host_stub.h"\n'
                "#define __global__\n#define KERNEL_TASK_TYPE_DEFAULT(...)\n"
                "namespace AscendC { using TPosition=QuePosition; constexpr int PIPE_ALL=1; "
                "inline uint32_t GetBlockNum(){return 8;} "
                "template<class T> using DataCopyPadExtParams=PaddingParams; }\n"
            )
            # Remove only the SDK launch syntax; keep the device body unchanged.
            body = (
                (SOURCE / "li_page_gather.cpp")
                .read_text()
                .split('\nnamespace {\nint32_t Launch', 1)[0]
            )
            (build / "body.cpp").write_text(
                body
                + r"""
#include <algorithm>
#include <cassert>
#include <vector>

void Check(uint32_t domain, uint32_t mainOffset, uint32_t dataOffset,
           uint32_t scaleOffset, uint32_t dataBytes, uint32_t scaleBytes,
           uint32_t destinationStrideBytes, bool allFull=false) {
    using namespace AscendC;
    regions.clear();
    const uint32_t pages=domain+3;
    const uint64_t objectBytes=std::max<uint64_t>({uint64_t(mainOffset)+1,
        uint64_t(dataOffset)+dataBytes,uint64_t(scaleOffset)+scaleBytes})+32;
    std::vector<uint8_t> objects(17*objectBytes),
        k(uint64_t(pages)*destinationStrideBytes,0xa5),
        scale(uint64_t(pages)*scaleBytes,0x73), expectedK=k, expectedScale=scale;
    for(size_t i=0;i<objects.size();++i) objects[i]=(i*19+i/objectBytes*7)%251;
    std::vector<uint64_t> bases(domain,0);
    for(uint32_t p=0;p<domain;++p) {
        // Native/shared/partial/append boundary pages have zero source entries.
        if(allFull || p%3==1 || p==domain-1) continue;
        auto* object=objects.data()+(p*7%17)*objectBytes;
        bases[p]=reinterpret_cast<uintptr_t>(object)+mainOffset;
        std::memcpy(expectedK.data()+uint64_t(p)*destinationStrideBytes,
                    object+dataOffset,dataBytes);
        std::memcpy(expectedScale.data()+uint64_t(p)*scaleBytes,
                    object+scaleOffset,scaleBytes);
    }
    Register(objects.data(),objects.size()); Register(bases.data(),bases.size()*8);
    Register(k.data(),k.size()); Register(scale.data(),scale.size());
    for(blockIndex=0;blockIndex<8;++blockIndex)
        dsa_selected_li_pages(reinterpret_cast<uint8_t*>(bases.data()),nullptr,k.data(),scale.data(),
                           domain,domain,mainOffset,dataOffset,scaleOffset,
                           dataBytes,scaleBytes,destinationStrideBytes);
    assert(k==expectedK && scale==expectedScale);
}

// Two full packed logical pages: page 1 must land at 32768 for K and 512 for
// scale, and neither destination may write past the requested page count.
void CheckPackedTwoPages() {
    using namespace AscendC;
    regions.clear(); blockIndex=0;
    constexpr uint32_t stride=32768, dataBytes=32768, scaleBytes=512;
    constexpr uint32_t mainOffset=64, dataOffset=128;
    constexpr uint32_t scaleOffset=dataOffset+dataBytes;
    std::vector<uint8_t> first(scaleOffset+scaleBytes,0), second=first;
    std::fill(first.begin()+dataOffset,first.begin()+dataOffset+dataBytes,0xa1);
    std::fill(first.begin()+scaleOffset,first.end(),0xb2);
    std::fill(second.begin()+dataOffset,second.begin()+dataOffset+dataBytes,0xc3);
    std::fill(second.begin()+scaleOffset,second.end(),0xd4);
    std::vector<uint64_t> bases{reinterpret_cast<uintptr_t>(first.data())+mainOffset,
                                reinterpret_cast<uintptr_t>(second.data())+mainOffset};
    std::vector<uint8_t> k(3*stride,0x5a), scale(3*scaleBytes,0x6b);
    Register(first.data(),first.size()); Register(second.data(),second.size());
    Register(bases.data(),bases.size()*8);
    Register(k.data(),k.size()); Register(scale.data(),scale.size());
    for(blockIndex=0;blockIndex<8;++blockIndex)
        dsa_selected_li_pages(reinterpret_cast<uint8_t*>(bases.data()),nullptr,k.data(),scale.data(),
                          2,2,mainOffset,dataOffset,scaleOffset,dataBytes,scaleBytes,stride);
    assert(std::all_of(k.begin(),k.begin()+stride,[](auto v){return v==0xa1;}));
    assert(std::all_of(k.begin()+stride,k.begin()+2*stride,[](auto v){return v==0xc3;}));
    assert(k[2*stride]==0x5a);
    assert(std::all_of(scale.begin(),scale.begin()+scaleBytes,[](auto v){return v==0xb2;}));
    assert(std::all_of(scale.begin()+scaleBytes,scale.begin()+2*scaleBytes,
                       [](auto v){return v==0xd4;}));
    assert(scale[2*scaleBytes]==0x6b);
}


// Explicit source -> destination mapping. Source IDs stay dense while the
// destination LI blocks are a separate domain.
void CheckMapped(uint32_t domain, uint32_t mainOffset, uint32_t dataOffset,
                 uint32_t scaleOffset, uint32_t dataBytes, uint32_t scaleBytes,
                 uint32_t destinationStrideBytes, uint32_t destinationPages) {
    using namespace AscendC;
    regions.clear();
    const uint64_t objectBytes=std::max<uint64_t>({uint64_t(mainOffset)+1,
        uint64_t(dataOffset)+dataBytes,uint64_t(scaleOffset)+scaleBytes})+32;
    std::vector<uint8_t> objects(17*objectBytes),
        k(uint64_t(destinationPages)*destinationStrideBytes,0xa5),
        scale(uint64_t(destinationPages)*scaleBytes,0x73), expectedK=k, expectedScale=scale;
    for(size_t i=0;i<objects.size();++i) objects[i]=(i*29+i/objectBytes*11)%251;
    std::vector<uint64_t> bases(domain,0);
    std::vector<uint32_t> destinations(domain);
    for(uint32_t p=0;p<domain;++p) {
        destinations[p]=(p*5+1)%destinationPages;
        // Native/shared/partial/append boundary pages have zero source entries.
        if(p%3==1) continue;
        auto* object=objects.data()+(p*7%17)*objectBytes;
        bases[p]=reinterpret_cast<uintptr_t>(object)+mainOffset;
        std::memcpy(expectedK.data()+uint64_t(destinations[p])*destinationStrideBytes,
                    object+dataOffset,dataBytes);
        std::memcpy(expectedScale.data()+uint64_t(destinations[p])*scaleBytes,
                    object+scaleOffset,scaleBytes);
    }
    Register(objects.data(),objects.size()); Register(bases.data(),bases.size()*8);
    Register(destinations.data(),destinations.size()*4);
    Register(k.data(),k.size()); Register(scale.data(),scale.size());
    for(blockIndex=0;blockIndex<8;++blockIndex)
        dsa_selected_li_pages(reinterpret_cast<uint8_t*>(bases.data()),reinterpret_cast<uint8_t*>(destinations.data()),
                           k.data(),scale.data(),domain,destinationPages,mainOffset,
                           dataOffset,scaleOffset,dataBytes,scaleBytes,destinationStrideBytes);
    assert(k==expectedK && scale==expectedScale);
}

// Packed layout: source main page IDs and destination LI block IDs are
// different domains. Source 0 lands on LI block 2 and source 1 on LI block 1;
// the second destination page starts at 32768 B / 512 B.
void CheckPackedMapped() {
    using namespace AscendC;
    regions.clear(); blockIndex=0;
    constexpr uint32_t stride=32768, dataBytes=32768, scaleBytes=512;
    constexpr uint32_t mainOffset=64, dataOffset=128;
    constexpr uint32_t scaleOffset=dataOffset+dataBytes;
    std::vector<uint8_t> first(scaleOffset+scaleBytes,0), second=first;
    std::fill(first.begin()+dataOffset,first.begin()+dataOffset+dataBytes,0xa1);
    std::fill(first.begin()+scaleOffset,first.end(),0xb2);
    std::fill(second.begin()+dataOffset,second.begin()+dataOffset+dataBytes,0xc3);
    std::fill(second.begin()+scaleOffset,second.end(),0xd4);
    std::vector<uint64_t> bases{reinterpret_cast<uintptr_t>(first.data())+mainOffset,
                                reinterpret_cast<uintptr_t>(second.data())+mainOffset};
    std::vector<uint32_t> destinations{2,1};
    std::vector<uint8_t> k(4*stride,0x5a), scale(4*scaleBytes,0x6b);
    Register(first.data(),first.size()); Register(second.data(),second.size());
    Register(bases.data(),bases.size()*8);
    Register(destinations.data(),destinations.size()*4);
    Register(k.data(),k.size()); Register(scale.data(),scale.size());
    for(blockIndex=0;blockIndex<8;++blockIndex)
        dsa_selected_li_pages(reinterpret_cast<uint8_t*>(bases.data()),reinterpret_cast<uint8_t*>(destinations.data()),
                           k.data(),scale.data(),2,3,mainOffset,dataOffset,scaleOffset,
                           dataBytes,scaleBytes,stride);
    assert(std::all_of(k.begin()+2*stride,k.begin()+3*stride,[](auto v){return v==0xa1;}));
    assert(std::all_of(k.begin()+stride,k.begin()+2*stride,[](auto v){return v==0xc3;}));
    assert(std::all_of(scale.begin()+2*scaleBytes,scale.begin()+3*scaleBytes,
                       [](auto v){return v==0xb2;}));
    assert(std::all_of(scale.begin()+scaleBytes,scale.begin()+2*scaleBytes,
                       [](auto v){return v==0xd4;}));
    assert(k[0]==0x5a && k[3*stride]==0x5a);
    assert(scale[0]==0x6b && scale[3*scaleBytes]==0x6b);
}

// A destination ID outside the target slab is rejected before any write.
void RejectMapped() {
    using namespace AscendC;
    regions.clear(); blockIndex=0;
    constexpr uint32_t stride=32768, dataBytes=32768, scaleBytes=512;
    constexpr uint32_t mainOffset=64, dataOffset=128;
    constexpr uint32_t scaleOffset=dataOffset+dataBytes;
    std::vector<uint8_t> source(scaleOffset+scaleBytes,0x11);
    std::vector<uint64_t> bases{reinterpret_cast<uintptr_t>(source.data())+mainOffset};
    std::vector<uint32_t> destinations{2};
    std::vector<uint8_t> k(3*stride,0xa5), scale(3*scaleBytes,0x73);
    Register(source.data(),source.size()); Register(bases.data(),bases.size()*8);
    Register(destinations.data(),destinations.size()*4);
    Register(k.data(),k.size()); Register(scale.data(),scale.size());
    bool rejected=false;
    try {
        dsa_selected_li_pages(reinterpret_cast<uint8_t*>(bases.data()),reinterpret_cast<uint8_t*>(destinations.data()),
                            k.data(),scale.data(),1,2,mainOffset,dataOffset,scaleOffset,
                            dataBytes,scaleBytes,stride);
    } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
    assert(std::all_of(k.begin(),k.end(),[](auto v){return v==0xa5;}));
    assert(std::all_of(scale.begin(),scale.end(),[](auto v){return v==0x73;}));
}

void Reject(uint64_t source, uint32_t mainOffset, uint32_t dataOffset,
            uint32_t scaleOffset, uint32_t dataBytes, uint32_t scaleBytes,
            uint32_t destinationStrideBytes) {
    using namespace AscendC;
    regions.clear(); blockIndex=0;
    std::vector<uint8_t> k(destinationStrideBytes,0xa5), scale(scaleBytes,0x73);
    Register(&source,sizeof(source)); Register(k.data(),k.size()); Register(scale.data(),scale.size());
    bool rejected=false;
    try {
        dsa_selected_li_pages(reinterpret_cast<uint8_t*>(&source),nullptr,k.data(),scale.data(),
                            1,1,mainOffset,dataOffset,scaleOffset,
                            dataBytes,scaleBytes,destinationStrideBytes);
    } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
    assert(std::all_of(k.begin(),k.end(),[](auto v){return v==0xa5;}));
    assert(std::all_of(scale.begin(),scale.end(),[](auto v){return v==0x73;}));
}

int main() {
    // M3v19 native 16 KiB data / 256 B scale / 16 KiB stride regression.
    for(uint32_t domain:{1U,7U,8U,9U,33U,4097U}) {
        Check(domain,8192,256,17408,16384,256,16384);
        Check(domain,8192,256,17408,16384,256,16384,true);
    }
    // Target packed 32 KiB data / 512 B scale / 32 KiB stride.
    for(uint32_t domain:{1U,7U,8U,9U,33U}) {
        Check(domain,8192,256,17408,32768,512,32768);
        Check(domain,8192,256,17408,32768,512,32768,true);
    }
    CheckPackedTwoPages();
    CheckMapped(7,8192,256,17408,16384,256,16384,11);
    CheckMapped(9,8192,256,17408,32768,512,32768,13);
    CheckPackedMapped();
    RejectMapped();
    Check(37,0,512,18000,16384,256,16384);  // Main component is first in its registered object.
    Check(37,32768,256,16896,16384,256,16384);  // LI components can precede the main component.
    Check(37,32768,256,16896,32768,512,32768,true);
    Reject(1,4096,8192,24576,16384,256,16384);
    Reject(4096,4096,8192,24576,16384,256,16384);  // A null whole-object base is not a valid lease.
    Reject(UINT64_MAX,0,0,32768,16384,256,16384);
    Reject(UINT64_MAX-32768-256+1,0,0,32768,16384,256,16384);  // Scale end alone overflows.
    Reject(UINT64_MAX-32768-512+1,0,0,32768,32768,512,32768);  // Packed scale end overflows.
}
"""
            )
            executable = build / "body"
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-DASCENDC_DUMP=1",
                    f"-I{build}",
                    f"-I{SOURCE}",
                    f"-I{Path(__file__).parent}",
                    str(build / "body.cpp"),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(executable)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
