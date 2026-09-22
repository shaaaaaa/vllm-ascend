"""Host checks for the standalone packaging; no AscendC compiler required."""
import re
from pathlib import Path

import pytest
from generate_sources import HERE, KERNELS, SOURCE, generate


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
