# SPDX-License-Identifier: Apache-2.0
"""CPU-only path/reference checks for reusable OFF archives."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools/layerwise_prefill_correctness_baseline.py"
SPEC = importlib.util.spec_from_file_location("correctness_baseline_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def _case(path, *, case="off", explicit_case=True):
    path.mkdir(parents=True)
    result = {"completed": True}
    if explicit_case:
        result["case"] = case
    (path / "result.json").write_text(json.dumps(result))
    (path / "environment.json").write_text(
        json.dumps(
            {
                "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": "false" if case == "off" else "true",
                "LMCACHE_STORE_ASYNC": "false" if case == "off" else "true",
            }
        )
    )
    return path.resolve()


@pytest.mark.parametrize("explicit_case", [True, False])
def test_accept_original_root_case_and_renamed_case(tmp_path, explicit_case):
    off = _case(tmp_path / "old" / "off", explicit_case=explicit_case)
    assert baseline.normalize_off_directory(off.parent) == off
    assert baseline.normalize_off_directory(off) == off
    renamed = off.with_name("reference_saved")
    off.rename(renamed)
    assert baseline.normalize_off_directory(renamed) == renamed


@pytest.mark.parametrize("explicit_case", [True, False])
def test_on_case_cannot_be_selected_as_off_even_after_rename(tmp_path, explicit_case):
    on = _case(tmp_path / "off", case="on", explicit_case=explicit_case)
    with pytest.raises(ValueError, match="not OFF|OFF environment"):
        baseline.normalize_off_directory(on)


def test_missing_archive_fails_and_default_resolver_does_not_mutate(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        baseline.normalize_off_directory(tmp_path / "absent")
    assert baseline.resolve_off_directory(tmp_path) == tmp_path / "off"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "reference",
    [
        [],
        {"schema": 2, "off_dir": "ignored"},
        {"schema": True, "off_dir": "ignored"},
        {"schema": 1, "off_dir": "relative/off"},
        {"schema": 1, "off_dir": None},
        {"schema": 1, "off_dir": "", "extra": 1},
    ],
)
def test_invalid_reference_schema_fails_closed(tmp_path, reference):
    (tmp_path / "off_reference.json").write_text(json.dumps(reference))
    with pytest.raises(ValueError):
        baseline.resolve_off_directory(tmp_path)


def test_reference_requires_actual_case_and_rejects_chained_runs(tmp_path):
    off = _case(tmp_path / "old" / "off")
    current = tmp_path / "current"
    current.mkdir()
    reference_path = current / "off_reference.json"
    reference_path.write_text(json.dumps({"schema": 1, "off_dir": str(off.parent)}))
    with pytest.raises(ValueError, match="case itself"):
        baseline.resolve_off_directory(current)
    reference_path.write_text(json.dumps({"schema": 1, "off_dir": str(off)}))
    assert baseline.resolve_off_directory(current) == off
    with pytest.raises(ValueError, match="original archive"):
        baseline.normalize_off_directory(current)
    (off.parent / "off_reference.json").write_text(json.dumps({"schema": 1, "off_dir": str(off)}))
    with pytest.raises(ValueError, match="Recursive"):
        baseline.resolve_off_directory(current)
