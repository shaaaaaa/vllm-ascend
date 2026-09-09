# SPDX-License-Identifier: Apache-2.0
"""Pure remote-fill identity validation shared by decoder discovery wrappers.

Wrappers validate their rank binding and retain envelope, session and duplicate
handling. This helper borrows JSON fields, owns no state, and returns a fresh
placement or None when native push is disabled. It performs no I/O or cleanup.
"""

from typing import Any

_REMOTE_FILL_VERIFICATION_CAPABILITY_BYTES = 32


def validate_remote_fill_placement(
    remote_fill: dict[str, Any],
    dp_rank: int,
) -> dict[str, Any] | None:
    """Validate common fields after the caller has bound a nonnegative DP rank."""
    required_strings = (
        "destination_engine_id",
        "control_endpoint",
        "token_hash_algorithm",
        "descriptor_verification_capability",
    )
    if any(not isinstance(remote_fill.get(name), str) or not remote_fill[name].strip() for name in required_strings):
        raise ValueError("Decoder remote-fill string identity is invalid")
    epoch = remote_fill.get("destination_engine_epoch")
    generation = remote_fill.get("shared_cache_generation")
    tp_size = remote_fill.get("destination_tp_size")
    dp_size = remote_fill.get("destination_dp_size")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("Decoder remote-fill engine epoch is invalid")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("Decoder remote-fill shared-cache generation is invalid")
    if (
        isinstance(tp_size, bool)
        or not isinstance(tp_size, int)
        or tp_size <= 0
        or isinstance(dp_size, bool)
        or not isinstance(dp_size, int)
        or dp_size <= 0
        or dp_rank >= dp_size
    ):
        raise ValueError("Decoder remote-fill parallel topology is invalid")
    global_te_push = remote_fill.get("global_te_push")
    if not isinstance(global_te_push, bool):
        raise ValueError("Decoder remote-fill native capability is invalid")
    if not global_te_push:
        return None
    verification_capability = remote_fill["descriptor_verification_capability"]
    try:
        verification_key = bytes.fromhex(verification_capability)
    except ValueError as error:
        raise ValueError("Decoder remote-fill verification capability is invalid") from error
    if (
        len(verification_key) != _REMOTE_FILL_VERIFICATION_CAPABILITY_BYTES
        or verification_key.hex() != verification_capability
    ):
        raise ValueError("Decoder remote-fill verification capability is invalid")
    python_hash_seed = remote_fill.get("python_hash_seed", "")
    if not isinstance(python_hash_seed, str) or (
        remote_fill["token_hash_algorithm"] == "builtin" and not python_hash_seed
    ):
        raise ValueError("Decoder remote-fill hash identity is invalid")
    return {
        "destination_engine_id": remote_fill["destination_engine_id"].strip(),
        "destination_engine_epoch": epoch,
        "control_endpoint": remote_fill["control_endpoint"].strip(),
        "destination_dp_rank": dp_rank,
        "shared_cache_generation": generation,
        "destination_tp_size": tp_size,
        "destination_dp_size": dp_size,
        "global_te_push": global_te_push,
        "token_hash_algorithm": remote_fill["token_hash_algorithm"].strip(),
        "python_hash_seed": python_hash_seed,
        "descriptor_verification_capability": verification_capability,
    }
