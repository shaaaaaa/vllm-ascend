"""Emit one explicitly named AIV entry point per AscendC translation unit.

Keep kernel bodies/classes verbatim from the resident source. Only entry-point
names and the compile-time fast-path flag differ between old/new builds.
"""
import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "csrc/kernels/resident_sorted_cache.cpp"
SHARDED_SOURCE = SOURCE.with_name("resident_sorted_cache_coordinated.cpp")
KERNELS = {
    "union": "dsa_resident_sharded_union_kernel",
    "finalize": "dsa_resident_sorted_finalize_kernel",
    "update": "dsa_resident_sorted_update_kernel",
}
MEMBERSHIP_KERNELS = {
    "union": KERNELS["union"],
    "update": "dsa_resident_sorted_membership_update_kernel",
}
BUILD_PROFILES = ("full", "membership", "dcp")
POINTERS = dict(zip((
    "topkIndices", "splitBoundary", "rowReqIndices", "shardPacked", "shardMapping",
    "shardCounts", "requestStateIndices", "requestStateGenerations", "stateTokens",
    "stateSlots", "stateCounts", "stateGenerations", "priorSlots", "shardMissTokens",
    "shardMissPositions", "shardEvictableSlots", "missTokens", "missCounts",
    "targetSlots", "requestBlockTable",
), range(20), strict=True))
POINTERS["debugInfo"] = POINTERS["missCounts"]  # unused with debugStage=0
POINTERS.update(liveOrigins=20, selectedCounts=21)
SCALARS = {
    "requestCount": "a.requests", "stateRowCount": "a.stateRows", "dummyStateBase": "a.dummyBase",
    "rowsPerRequest": "a.mtp", "rowWidth": "2048", "shardCount": "a.shards",
    "shardCapacity": "a.capacity", "capacity": "a.capacity", "shardCountStride": "16",
    "shardCountRequestStride": "a.shards * 16", "generationStride": "8",
    "missCountStride": "16", "blockTableWidth": "a.blockTableWidth",
    "blockSize": "a.blockSize", "debugStage": "0",
    "dcpBits": "a.dcpBits", "dcpRank": "a.dcpRank",
}


def free_slot_first(source: str) -> str:
    """Change only standalone finalize policy; fail closed on source drift."""
    replacements = (
        ("""        uint32_t remainingEvictions = totalMissCount < totalEvictableCount
            ? totalMissCount
            : totalEvictableCount;""", """        ASSERT(totalOldCount <= capacity_);
        const uint32_t unused = capacity_ - totalOldCount;
        const uint32_t freshCount = totalMissCount < unused ? totalMissCount : unused;
        const uint32_t evictionCount = totalMissCount - freshCount;
        ASSERT(evictionCount <= totalEvictableCount);
        uint32_t remainingEvictions = evictionCount;"""),
        ("""        const uint32_t totalSelectedEvictCount =
            totalMissCount < totalEvictableCount
            ? totalMissCount
            : totalEvictableCount;""", ""),
        ("if (globalMiss < totalSelectedEvictCount)", "if (globalMiss >= freshCount)"),
        ("""                        // Resident allocation preserves a dense occupied
                        // prefix. Every old entry not present in the current
                        // union is emitted above as an evictable slot; after
                        // pooling all shards, any remaining miss grows the
                        // prefix from totalOldCount. Therefore the global
                        // candidate count is always capacity - hitCount.
                        slot = static_cast<int16_t>(
                            totalOldCount
                            + globalMiss
                            - totalSelectedEvictCount);""",
         """                        // Consume the unused dense-prefix tail before replacing
                        // old tokens. Hit slots and miss ordering are unchanged.
                        slot = static_cast<int16_t>(totalOldCount + globalMiss);"""),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("Standalone free-slot policy no longer matches production finalize")
        source = source.replace(old, new)
    return source


def generate(source: str, header: Path, *, variants=(("baseline", 0), ("optimized", 1)), kernels=None,
             compact=False, sharded=False, vector_union=False, union_stop=0, vector_intersection=False,
             vector_state=False, query_membership=False, dcp=False, free_first=False) -> dict[str, str]:
    if free_first:
        if not dcp:
            raise ValueError("free-slot-first is an isolated DCP experiment")
        source = free_slot_first(source)
    marker = 'extern "C" __global__ __aicore__ void\n'
    classes, separator, _ = source.partition(marker)
    if not separator:
        raise ValueError("Resident source entry-point boundary changed")
    output = {}
    for stage, name in (KERNELS if kernels is None else kernels).items():
        pattern = re.compile(re.escape(marker + name) + r"\((.*?)\)\n\{\n(.*?)\n\}", re.S)
        matches = list(pattern.finditer(source))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one definition of {name}")
        match = matches[0]
        arguments = []
        parameters = re.sub(r"(?ms)^#ifdef RESIDENT_DCP_SHARDING\n(.*?)^#endif\n",
                            r"\1" if dcp else "", match[1])
        for parameter in parameters.split(","):
            pointer = re.fullmatch(r"\s*__gm__\s+(int(?:16|32|64)_t)\*\s+(\w+)\s*", parameter)
            if pointer:
                arguments.append(f"static_cast<{pointer[1]}*>(a.tensors[{POINTERS[pointer[2]]}])")
            else:
                scalar = re.fullmatch(r"\s*uint32_t\s+(\w+)\s*", parameter)
                if scalar is None or scalar[1] not in SCALARS:
                    raise ValueError(f"Unsupported kernel parameter: {parameter}")
                arguments.append(SCALARS[scalar[1]])
        for variant, enabled in variants:
            kernel_name = f"{name}_{variant}"
            if union_stop:
                kernel_name += "_" + stage.removeprefix("union_")
            entry = match[0].replace(name, kernel_name, 1)
            logical_blocks = "a.requests" if stage == "finalize" and not sharded else "a.requests * a.shards"
            if query_membership and stage == "update":
                logical_blocks = "a.requests * (a.shards + a.mtp)"
            launch = (
                f'\n}}  // namespace\n#include "{header.resolve().as_posix()}"\n\n'
                f"void resident_experiment_{variant}_{stage}(void* stream, const ResidentLaunch& a)\n{{\n"
                f"    const uint32_t logicalBlocks = {logical_blocks};\n"
                f"    {kernel_name}<<<logicalBlocks < a.cores ? logicalBlocks : a.cores, nullptr, stream>>>(\n"
                + ",\n".join(f"        {arg}" for arg in arguments) + ");\n}\n"
            )
            output[f"{variant}_{stage}.cpp"] = (
                "// Generated; do not edit. Kernel algorithm is copied verbatim.\n"
                f"#define RESIDENT_EXPERIMENT_SKIP_UNCHANGED {enabled}\n"
                f"#define RESIDENT_EXPERIMENT_COMPACT_REMAP {int(compact)}\n"
                f"#define RESIDENT_EXPERIMENT_VECTOR_UNION {int(vector_union)}\n"
                f"#define RESIDENT_EXPERIMENT_VECTOR_INTERSECTION {int(vector_intersection)}\n"
                f"#define RESIDENT_EXPERIMENT_VECTOR_STATE {int(vector_state)}\n"
                f"#define RESIDENT_EXPERIMENT_UNION_STOP {union_stop}\n"
                f"#define RESIDENT_EXPERIMENT_QUERY_MEMBERSHIP {int(query_membership)}\n"
                + ("#define RESIDENT_DCP_SHARDING 1\n" if dcp else "")
                + classes + entry + launch
            )
    return output


def build_sources(profile: str) -> dict[str, str]:
    if profile not in BUILD_PROFILES:
        raise ValueError(f"Unsupported resident build profile: {profile}")
    source = SOURCE.read_text(encoding="utf-8")
    header = HERE / "launch.h"
    if profile == "dcp":
        flags = dict(dcp=True, query_membership=True, vector_intersection=True, vector_state=True)
        output = generate(source, header, variants=(("dcp", 0),), kernels=MEMBERSHIP_KERNELS, **flags)
        for variant, free_first in (("dcp_old", False), ("dcp_new", True)):
            output.update(generate(source, header, variants=((variant, 0),),
                                   kernels={"finalize": KERNELS["finalize"]}, free_first=free_first, **flags))
        return output
    output = generate(source, header, variants=(("baseline", 0),),
                      kernels={"finalize": KERNELS["finalize"]})
    output.update(generate(source, header, variants=(("intersection", 0),),
                           kernels={"union": KERNELS["union"]}, vector_intersection=True))
    output.update(generate(source, header, variants=(("state", 0),),
                           kernels={"update": KERNELS["update"]}, vector_state=True))
    output.update(generate(source, header, variants=(("membership", 0),),
                           kernels=MEMBERSHIP_KERNELS, vector_intersection=True,
                           vector_state=True, query_membership=True))
    if profile == "membership":
        return output

    output.update(generate(source, header))
    output.update(generate(source, HERE / "launch.h", variants=(("compact", 0),),
                           kernels={"update": KERNELS["update"]}, compact=True))
    output.update(generate(SHARDED_SOURCE.read_text(encoding="utf-8"), HERE / "launch.h",
                           variants=(("sharded", 0),), sharded=True,
                           kernels={"finalize": "dsa_resident_sharded_finalize_worker_kernel"}))
    output.update(generate(source, HERE / "launch.h", variants=(("vector", 0),),
                           kernels={"union": KERNELS["union"]}, vector_union=True))
    for variant in ('baseline', 'state'):
        output.update(generate(source, HERE / 'launch.h', variants=((variant, 0),),
                               kernels={'state_update': 'dsa_resident_sorted_state_update_kernel'},
                               vector_state=variant == 'state'))
    output.update(generate(source, HERE / 'launch.h', variants=(('baseline', 0),),
                           kernels={'remap': 'dsa_resident_sorted_remap_kernel'}))
    for variant in ("baseline", "vector"):
        for phase, stop in (("sort", 1), ("dedup", 2)):
            output.update(generate(source, HERE / "launch.h", variants=((variant, 0),),
                                   kernels={f"union_{phase}": KERNELS["union"]},
                                   vector_union=variant == "vector", union_stop=stop))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=BUILD_PROFILES, default="full")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = build_sources(args.profile)
    for name, content in output.items():
        path = args.output / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
