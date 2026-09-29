# SPDX-License-Identifier: Apache-2.0
"""Read-only resolution of an existing OFF tensor archive."""

from __future__ import annotations

import json
from pathlib import Path

OFF_ENV_FIELDS = ("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "LMCACHE_STORE_ASYNC")


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def normalize_off_directory(user_path: str | Path) -> Path:
    """Accept an original run root or OFF case, without following references.

    The case directory may have been renamed. Older runs lack ``result.case``;
    both explicit OFF environment flags are required regardless of that field.
    Full rank/layer coverage validation is performed by ``validate_off_baseline``.
    """
    directory = Path(user_path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"OFF directory does not exist: {directory}")
    if (directory / "off_reference.json").exists():
        raise ValueError("An OFF reference must point to the original archive, not another referenced run")
    case = directory if (directory / "result.json").is_file() else directory / "off"
    if not case.is_dir():
        raise ValueError(f"No OFF case directory found in {directory}")
    if (case / "off_reference.json").exists() or (case.parent / "off_reference.json").exists():
        raise ValueError("Recursive OFF references are not supported; select the original OFF archive")
    result = _read_object(case / "result.json")
    if "case" in result and result["case"] != "off":
        raise ValueError(f"Selected case is not OFF: {case}")
    environment = _read_object(case / "environment.json")
    if any(environment.get(field) != "false" for field in OFF_ENV_FIELDS):
        raise ValueError(f"Selected case does not have both OFF environment flags: {case}")
    return case.resolve()


def resolve_off_directory(run_root: str | Path) -> Path:
    """Return the local OFF case or the exact external archive in a strict reference.

    No files are copied, linked, or modified. The default local path is returned
    even before OFF completes, so its normal coverage validator can report errors.
    """
    root = Path(run_root).resolve()
    reference_path = root / "off_reference.json"
    if not reference_path.exists():
        return root / "off"
    reference = _read_object(reference_path)
    if set(reference) != {"schema", "off_dir"} or type(reference.get("schema")) is not int or reference["schema"] != 1:
        raise ValueError("Invalid off_reference.json schema; expected exactly schema=1 and off_dir")
    target = reference["off_dir"]
    if not isinstance(target, str) or not target or not Path(target).is_absolute():
        raise ValueError("off_reference.json off_dir must be an absolute OFF case path")
    path = Path(target).resolve()
    case = normalize_off_directory(path)
    if case != path:
        raise ValueError("off_reference.json must name the OFF case itself, not a run root")
    return case
