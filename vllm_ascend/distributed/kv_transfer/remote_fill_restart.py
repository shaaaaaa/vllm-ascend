# SPDX-License-Identifier: Apache-2.0
"""Fail-closed orchestration contract for a paired RemoteFill restart."""

# Standard
import json
from argparse import ArgumentParser
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from time import time_ns
from typing import Any, Protocol

IDENTITY_FIELDS = (
    "source_incarnation_id",
    "destination_incarnation_id",
    "source_engine_epoch",
    "destination_engine_epoch",
    "source_session",
    "destination_session",
    "global_te_registration_generation",
    "shared_cache_generation",
    "shared_cache_incarnation_id",
)
FRESHNESS_FIELDS = (
    "source_incarnation_id",
    "destination_incarnation_id",
    "source_engine_epoch",
    "destination_engine_epoch",
    "global_te_registration_generation",
    "shared_cache_incarnation_id",
)


class PairedRestartAdapter(Protocol):
    """Deployment supervisor operations for one source/destination engine pair."""

    def current_identity(self, pair_id: str) -> Mapping[str, Any]: ...

    def invalidate_proxy_placement(self, pair_id: str) -> None: ...

    def stop_admission(self, pair_id: str) -> None: ...

    def terminate_engine_groups(self, pair_id: str) -> None: ...

    def wait_engine_groups_stopped(self, pair_id: str, timeout_seconds: float) -> Mapping[str, bool]: ...

    def start_engine_groups(self, pair_id: str) -> None: ...

    def discover_identity(self, pair_id: str, timeout_seconds: float) -> Mapping[str, Any]: ...

    def publish_proxy_placement(self, pair_id: str, identity: Mapping[str, Any]) -> None: ...

    def restore_admission(self, pair_id: str) -> None: ...


def _identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    epochs = ("source_engine_epoch", "destination_engine_epoch")
    strings = (
        "source_incarnation_id",
        "destination_incarnation_id",
        "source_session",
        "destination_session",
        "shared_cache_incarnation_id",
    )
    invalid = [
        field
        for field in epochs
        if isinstance(identity.get(field), bool) or not isinstance(identity.get(field), int) or identity[field] <= 0
    ]
    invalid.extend(
        field
        for field in strings
        if not isinstance(identity.get(field), str) or not identity[field]
    )
    for field in ("global_te_registration_generation", "shared_cache_generation"):
        generation = identity.get(field)
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            invalid.append(field)
    if invalid:
        raise RuntimeError("paired restart identity is invalid: " + ", ".join(invalid))
    return {field: identity[field] for field in IDENTITY_FIELDS}


def restart_affected_pair(
    adapter: PairedRestartAdapter,
    pair_id: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Restart one complete P+D unit and publish only fresh placement identity."""

    if not pair_id or timeout_seconds <= 0:
        raise ValueError("pair_id and timeout_seconds must be valid")
    old = _identity(adapter.current_identity(pair_id))
    steps: list[dict[str, Any]] = []

    def run(name: str, operation) -> Any:
        result = operation()
        steps.append({"step": name, "wall_time_ns": time_ns()})
        return result

    run("proxy_placement_invalidated", lambda: adapter.invalidate_proxy_placement(pair_id))
    run("admission_stopped", lambda: adapter.stop_admission(pair_id))
    run("engine_groups_terminated", lambda: adapter.terminate_engine_groups(pair_id))
    stopped = run(
        "engine_groups_stopped",
        lambda: adapter.wait_engine_groups_stopped(pair_id, timeout_seconds),
    )
    stop_counts = (
        ("source_expected", "source_stopped"),
        ("destination_expected", "destination_stopped"),
    )
    if not isinstance(stopped, Mapping) or any(
        isinstance(stopped.get(expected), bool)
        or not isinstance(stopped.get(expected), int)
        or stopped[expected] <= 0
        or stopped.get(actual) != stopped[expected]
        for expected, actual in stop_counts
    ):
        raise RuntimeError("paired restart did not stop every P and D process")
    started = False
    try:
        run("engine_groups_started", lambda: adapter.start_engine_groups(pair_id))
        started = True
        new = _identity(
            run(
                "fresh_identity_discovered",
                lambda: adapter.discover_identity(pair_id, timeout_seconds),
            )
        )
        stale = [field for field in FRESHNESS_FIELDS if new[field] == old[field]]
        if stale:
            raise RuntimeError("paired restart reused identity: " + ", ".join(stale))
        run(
            "proxy_placement_published",
            lambda: adapter.publish_proxy_placement(pair_id, new),
        )
        run("admission_restored", lambda: adapter.restore_admission(pair_id))
    except BaseException:
        if started:
            run(
                "failed_restart_groups_terminated",
                lambda: adapter.terminate_engine_groups(pair_id),
            )
            run(
                "failed_restart_groups_stopped",
                lambda: adapter.wait_engine_groups_stopped(
                    pair_id, timeout_seconds
                ),
            )
        raise
    return {
        "schema": 1,
        "kind": "remote_fill_paired_restart_record",
        "pair_id": pair_id,
        "old_identity": old,
        "new_identity": new,
        "stopped": dict(stopped),
        "steps": steps,
    }


def main() -> None:
    """Run the state machine through a deployment-owned supervisor adapter."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="module:factory")
    parser.add_argument("--adapter-config", required=True, type=Path)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    module, separator, factory = args.adapter.partition(":")
    if not separator:
        raise ValueError("adapter must use module:factory syntax")
    config = json.loads(args.adapter_config.read_text(encoding="utf-8"))
    adapter = getattr(import_module(module), factory)(config)
    record = restart_affected_pair(
        adapter,
        args.pair_id,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
