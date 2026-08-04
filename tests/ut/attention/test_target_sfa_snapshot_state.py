import os
from pathlib import Path
from typing import Any

import pytest
import torch

_PREVIOUS_PRE_ENV = "VLLM_ASCEND_TARGET_SFA_PREVIOUS_PRE"
_FAILURE_INPUT_ENV = "VLLM_ASCEND_TARGET_SFA_REPLAY_FAILURE_INPUT"


def _load_snapshot(env_name: str, expected_suffix: str) -> tuple[Path, dict]:
    raw_path = os.getenv(env_name)
    if not raw_path:
        pytest.skip(f"set {env_name} to a target SFA snapshot")
    path = Path(raw_path)
    if not path.is_file() or not path.name.endswith(expected_suffix):
        pytest.fail(
            f"{env_name} must name an existing {expected_suffix} file: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        pytest.fail(f"target SFA snapshot is not a dictionary: {path}")
    return path, payload


def _load_snapshot_path(path: Path, expected_suffix: str) -> dict:
    if not path.is_file() or not path.name.endswith(expected_suffix):
        pytest.fail(f"target SFA snapshot is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        pytest.fail(f"target SFA snapshot is not a dictionary: {path}")
    return payload


def _require_tensor(snapshot: dict, name: str, source: Path) -> torch.Tensor:
    value = snapshot.get(name)
    if not isinstance(value, torch.Tensor):
        pytest.fail(f"resident state snapshot {source} lacks tensor {name}")
    return value


def _validated_resident_rows(
    snapshot: Any,
    source: Path,
) -> tuple[int, dict[int, tuple[int, dict[int, int]]]]:
    if not isinstance(snapshot, dict):
        pytest.fail(f"target SFA snapshot lacks resident state: {source}")

    row_indices = snapshot.get("row_indices")
    if not isinstance(row_indices, list) or not all(
        isinstance(row, int) for row in row_indices
    ):
        pytest.fail(f"resident state {source} has invalid row_indices")
    if len(row_indices) != len(set(row_indices)):
        pytest.fail(f"resident state {source} contains duplicate row indices")

    tokens = _require_tensor(snapshot, "tokens", source)
    slots = _require_tensor(snapshot, "slots", source)
    counts = _require_tensor(snapshot, "counts", source)
    generations = _require_tensor(snapshot, "generations", source)
    if tokens.ndim != 3 or slots.shape != tokens.shape:
        pytest.fail(
            f"resident token/slot shapes are invalid in {source}: "
            f"tokens={tuple(tokens.shape)} slots={tuple(slots.shape)}"
        )
    row_count, shard_count, capacity = tokens.shape
    if row_count != len(row_indices):
        pytest.fail(
            f"resident row count does not match row_indices in {source}: "
            f"rows={row_count} indices={len(row_indices)}"
        )
    if (
        counts.ndim != 3
        or tuple(counts.shape[:2]) != (row_count, shard_count)
        or counts.shape[2] < 1
        or generations.ndim != 2
        or generations.shape[0] != row_count
        or generations.shape[1] < 1
    ):
        pytest.fail(
            f"resident count/generation shapes are invalid in {source}: "
            f"counts={tuple(counts.shape)} "
            f"generations={tuple(generations.shape)}"
        )

    rows: dict[int, tuple[int, dict[int, int]]] = {}
    for local_row, state_row in enumerate(row_indices):
        generation = int(generations[local_row, 0])
        resident: dict[int, int] = {}
        occupied_slots: set[int] = set()
        total_count = 0
        for shard in range(shard_count):
            count = int(counts[local_row, shard, 0])
            if count < 0 or count > capacity:
                pytest.fail(
                    f"resident count is out of range in {source}: "
                    f"row={state_row} shard={shard} count={count} "
                    f"capacity={capacity}"
                )
            total_count += count
            shard_tokens = tokens[local_row, shard, :count].tolist()
            shard_slots = slots[local_row, shard, :count].tolist()
            if shard_tokens != sorted(shard_tokens):
                inversion = next(
                    index
                    for index in range(1, count)
                    if shard_tokens[index - 1] > shard_tokens[index]
                )
                begin = max(inversion - 4, 0)
                end = min(inversion + 5, count)
                pytest.fail(
                    f"resident tokens are not sorted in {source}: "
                    f"row={state_row} shard={shard} count={count} "
                    f"first_inversion={inversion - 1}->{inversion} "
                    f"tokens[{begin}:{end}]={shard_tokens[begin:end]} "
                    f"slots[{begin}:{end}]={shard_slots[begin:end]}"
                )
            if len(shard_tokens) != len(set(shard_tokens)):
                pytest.fail(
                    f"resident tokens are duplicated in {source}: "
                    f"row={state_row} shard={shard}"
                )
            for token, slot in zip(shard_tokens, shard_slots, strict=True):
                token = int(token)
                slot = int(slot)
                if token < 0 or token % shard_count != shard:
                    pytest.fail(
                        f"resident token is in the wrong shard in {source}: "
                        f"row={state_row} shard={shard} token={token}"
                    )
                if slot < 0 or slot >= capacity:
                    pytest.fail(
                        f"resident slot is out of range in {source}: "
                        f"row={state_row} shard={shard} slot={slot} "
                        f"capacity={capacity}"
                    )
                if slot in occupied_slots:
                    pytest.fail(
                        f"resident slot is duplicated in {source}: "
                        f"row={state_row} slot={slot}"
                    )
                occupied_slots.add(slot)
                resident[token] = slot
        if total_count > capacity:
            pytest.fail(
                f"resident row exceeds scratch capacity in {source}: "
                f"row={state_row} count={total_count} capacity={capacity}"
            )
        rows[state_row] = (generation, resident)
    return shard_count, rows


def _validate_union_workspace(pre: dict, source: Path) -> tuple[int, list[list[int]]]:
    workspace = pre.get("resident_workspace")
    fields = workspace.get("fields") if isinstance(workspace, dict) else None
    if not isinstance(fields, dict):
        pytest.fail(f"target SFA pre snapshot lacks resident workspace: {source}")
    packed = _require_tensor(fields, "shard_packed", source)
    counts = _require_tensor(fields, "shard_counts", source)
    if (
        packed.ndim != 3
        or counts.ndim != 3
        or tuple(counts.shape[:2]) != tuple(packed.shape[:2])
        or counts.shape[2] < 1
    ):
        pytest.fail(
            f"resident union workspace shapes are invalid in {source}: "
            f"packed={tuple(packed.shape)} counts={tuple(counts.shape)}"
        )
    request_count, shard_count, capacity = packed.shape
    request_counts: list[list[int]] = []
    for request in range(request_count):
        shard_counts: list[int] = []
        total_count = 0
        for shard in range(shard_count):
            count = int(counts[request, shard, 0])
            if count < 0 or count > capacity:
                pytest.fail(
                    f"resident union count is out of range in {source}: "
                    f"request={request} shard={shard} count={count} "
                    f"capacity={capacity}"
                )
            total_count += count
            shard_counts.append(count)
            shard_tokens = packed[request, shard, :count].tolist()
            if shard_tokens != sorted(shard_tokens):
                inversion = next(
                    index
                    for index in range(1, count)
                    if shard_tokens[index - 1] > shard_tokens[index]
                )
                begin = max(inversion - 4, 0)
                end = min(inversion + 5, count)
                pytest.fail(
                    f"resident union tokens are not sorted in {source}: "
                    f"request={request} shard={shard} count={count} "
                    f"first_inversion={inversion - 1}->{inversion} "
                    f"tokens[{begin}:{end}]={shard_tokens[begin:end]}"
                )
            if len(shard_tokens) != len(set(shard_tokens)):
                pytest.fail(
                    f"resident union tokens are duplicated in {source}: "
                    f"request={request} shard={shard}"
                )
            wrong_shard = next(
                (
                    int(token)
                    for token in shard_tokens
                    if int(token) < 0 or int(token) % shard_count != shard
                ),
                None,
            )
            if wrong_shard is not None:
                pytest.fail(
                    f"resident union token is in the wrong shard in {source}: "
                    f"request={request} shard={shard} token={wrong_shard}"
                )
        if total_count > capacity:
            pytest.fail(
                f"resident union exceeds scratch capacity in {source}: "
                f"request={request} count={total_count} capacity={capacity}"
            )
        request_counts.append(shard_counts)
    return shard_count, request_counts


def test_target_sfa_resident_state_is_continuous_before_failure():
    """Validate the saved state transition immediately before graph failure."""
    previous_path, previous = _load_snapshot(
        _PREVIOUS_PRE_ENV,
        "_pre.pt",
    )
    current_path, current = _load_snapshot(
        _FAILURE_INPUT_ENV,
        "_input.pt",
    )

    identity_mismatches = {
        name: (previous.get(name), current.get(name))
        for name in ("schema_version", "rank", "layer", "layer_name")
        if previous.get(name) != current.get(name)
    }
    if identity_mismatches:
        pytest.fail(
            "target SFA snapshots are not from the same rank/layer: "
            f"{identity_mismatches}"
        )
    previous_step = previous.get("step_id")
    current_step = current.get("step_id")
    if (
        not isinstance(previous_step, int)
        or not isinstance(current_step, int)
        or current_step != previous_step + 1
    ):
        pytest.fail(
            "target SFA snapshots are not consecutive steps: "
            f"previous={previous_step} current={current_step}"
        )

    previous_input_path = previous_path.with_name(
        previous_path.name.replace("_pre.pt", "_input.pt")
    )
    previous_input = _load_snapshot_path(previous_input_path, "_input.pt")
    previous_input_mismatches = {
        name: (previous_input.get(name), previous.get(name))
        for name in ("schema_version", "step_id", "rank", "layer", "layer_name")
        if previous_input.get(name) != previous.get(name)
    }
    if previous_input_mismatches:
        pytest.fail(
            "previous target SFA input/pre snapshots do not match: "
            f"{previous_input_mismatches}"
        )

    input_shards, input_rows = _validated_resident_rows(
        previous_input.get("resident_state_before"),
        previous_input_path,
    )
    print(
        "[target-sfa-state-transition] previous input state passed:"
        f" step={previous_step} shards={input_shards}"
        f" resident_counts="
        f"{{{', '.join(f'{row}: {len(state[1])}' for row, state in input_rows.items())}}}"
    )
    union_shards, union_counts = _validate_union_workspace(
        previous,
        previous_path,
    )
    print(
        "[target-sfa-state-transition] previous union workspace passed:"
        f" step={previous_step} shards={union_shards}"
        f" shard_counts={union_counts}"
    )

    previous_shards, previous_rows = _validated_resident_rows(
        previous.get("resident_state_after"),
        previous_path,
    )
    print(
        "[target-sfa-state-transition] previous output state passed:"
        f" step={previous_step} shards={previous_shards}"
    )
    current_shards, current_rows = _validated_resident_rows(
        current.get("resident_state_before"),
        current_path,
    )
    assert input_shards == previous_shards
    assert union_shards == previous_shards
    assert current_shards == previous_shards
    assert current_rows.keys() == previous_rows.keys()
    for state_row, previous_state in previous_rows.items():
        current_state = current_rows[state_row]
        assert current_state == previous_state, (
            "resident state changed between consecutive layer-0 graph calls: "
            f"row={state_row} previous_generation={previous_state[0]} "
            f"current_generation={current_state[0]} "
            f"previous_count={len(previous_state[1])} "
            f"current_count={len(current_state[1])}"
        )

    counts = {
        row: len(state[1]) for row, state in current_rows.items()
    }
    print(
        "[target-sfa-state-transition]"
        f" previous={previous_path} current={current_path}"
        f" rank={current.get('rank')} layer={current.get('layer')}"
        f" steps={previous_step}->{current_step}"
        f" shards={current_shards} resident_counts={counts}"
    )
