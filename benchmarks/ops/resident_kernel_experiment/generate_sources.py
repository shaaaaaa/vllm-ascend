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
POINTERS = dict(zip((
    "topkIndices", "splitBoundary", "rowReqIndices", "shardPacked", "shardMapping",
    "shardCounts", "requestStateIndices", "requestStateGenerations", "stateTokens",
    "stateSlots", "stateCounts", "stateGenerations", "priorSlots", "shardMissTokens",
    "shardMissPositions", "shardEvictableSlots", "missTokens", "missCounts",
    "targetSlots", "requestBlockTable",
), range(20), strict=True))
POINTERS["debugInfo"] = POINTERS["missCounts"]  # unused with debugStage=0
SCALARS = {
    "requestCount": "a.requests", "stateRowCount": "a.stateRows", "dummyStateBase": "a.dummyBase",
    "rowsPerRequest": "a.mtp", "rowWidth": "2048", "shardCount": "a.shards",
    "shardCapacity": "a.capacity", "capacity": "a.capacity", "shardCountStride": "16",
    "shardCountRequestStride": "a.shards * 16", "generationStride": "8",
    "missCountStride": "16", "blockTableWidth": "a.blockTableWidth",
    "blockSize": "a.blockSize", "debugStage": "0",
}


def generate(source: str, header: Path, *, variants=(("baseline", 0), ("optimized", 1)), kernels=None,
             compact=False, sharded=False) -> dict[str, str]:
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
        for parameter in match[1].split(","):
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
            entry = match[0].replace(name, kernel_name, 1)
            logical_blocks = "a.requests" if stage == "finalize" and not sharded else "a.requests * a.shards"
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
                + classes + entry + launch
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    output = generate(source, HERE / "launch.h")
    output.update(generate(source, HERE / "launch.h", variants=(("compact", 0),),
                           kernels={"update": KERNELS["update"]}, compact=True))
    output.update(generate(SHARDED_SOURCE.read_text(encoding="utf-8"), HERE / "launch.h",
                           variants=(("sharded", 0),), sharded=True,
                           kernels={"finalize": "dsa_resident_sharded_finalize_worker_kernel"}))
    for name, content in output.items():
        path = args.output / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
