"""Host checks for the standalone packaging; no AscendC compiler required."""
import re
from pathlib import Path

import pytest
from generate_sources import HERE, KERNELS, SHARDED_SOURCE, SOURCE, generate


@pytest.fixture
def sources():
    return generate(SOURCE.read_text(encoding="utf-8"), HERE / "launch.h")


def test_each_translation_unit_has_one_explicit_aiv_entry(sources):
    assert len(sources) == 6
    original = SOURCE.read_text(encoding="utf-8")
    marker = 'extern "C" __global__ __aicore__ void\n'
    classes = original.partition(marker)[0]
    for filename, content in sources.items():
        variant, stage = Path(filename).stem.split("_")
        name = KERNELS[stage]
        assert content.count(marker) == 1
        assert f"{marker}{name}_{variant}(" in content
        assert classes in content  # helpers and algorithm bodies unchanged
        definition = re.search(re.escape(marker + name) + r"\(.*?\n\}", original, re.S)[0]
        assert definition.replace(name, f"{name}_{variant}", 1) in content
        assert "#define dsa_resident_" not in content
        assert '#include "baseline.cpp"' not in content
        assert f"#define RESIDENT_EXPERIMENT_SKIP_UNCHANGED {int(variant == 'optimized')}\n" in content


@pytest.mark.parametrize("stage,indices", [
    ("union", list(range(16))),
    ("finalize", [3, 5, 12, 13, 14, 15, 16, 17, 18, 19, 17]),
    ("update", [0, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11]),
])
def test_generated_launch_keeps_existing_pointer_order(sources, stage, indices):
    for variant in ("baseline", "optimized"):
        content = sources[f"{variant}_{stage}.cpp"]
        launch = content.split(f"void resident_experiment_{variant}_{stage}", 1)[1]
        assert list(map(int, re.findall(r"a\.tensors\[(\d+)\]", launch))) == indices
        logical = "a.requests" if stage == "finalize" else "a.requests * a.shards"
        assert f"logicalBlocks = {logical};" in launch


def test_generator_refuses_unknown_kernel_arguments():
    source = SOURCE.read_text(encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        generate(source.replace("__gm__ int32_t* topkIndices", "__gm__ int32_t* unknownBuffer"), HERE / "launch.h")


def test_new_variants_have_separate_single_entry_sources():
    compact = generate(SOURCE.read_text(encoding="utf-8"), HERE / "launch.h",
                       variants=(("compact", 0),), kernels={"update": KERNELS["update"]}, compact=True)
    sharded = generate(SHARDED_SOURCE.read_text(encoding="utf-8"), HERE / "launch.h",
                       variants=(("sharded", 0),), sharded=True,
                       kernels={"finalize": "dsa_resident_sharded_finalize_worker_kernel"})
    assert set(compact) == {"compact_update.cpp"}
    assert set(sharded) == {"sharded_finalize.cpp"}
    for source in (*compact.values(), *sharded.values()):
        assert source.count('extern "C" __global__ __aicore__ void') == 1
        assert "#define RESIDENT_EXPERIMENT_SKIP_UNCHANGED 0" in source
        assert "logicalBlocks = a.requests * a.shards" in source
    assert "#define RESIDENT_EXPERIMENT_COMPACT_REMAP 1" in compact["compact_update.cpp"]


def test_vector_union_and_probes_have_distinct_entry_names():
    source = SOURCE.read_text(encoding="utf-8")
    symbols = []
    for variant in ("baseline", "vector"):
        for stage, stop in (("union", 0), ("union_sort", 1), ("union_dedup", 2)):
            generated = generate(source, HERE / "launch.h", variants=((variant, 0),),
                                 kernels={stage: KERNELS["union"]}, vector_union=variant == "vector", union_stop=stop)
            assert len(generated) == 1
            content = next(iter(generated.values()))
            entries = re.findall(r'extern "C" __global__ __aicore__ void\n(\w+)\(', content)
            assert len(entries) == 1
            assert f"{entries[0]}<<<" in content
            assert f"#define RESIDENT_EXPERIMENT_UNION_STOP {stop}" in content
            assert f"#define RESIDENT_EXPERIMENT_VECTOR_UNION {int(variant == 'vector')}" in content
            symbols += entries
    assert len(set(symbols)) == 6
