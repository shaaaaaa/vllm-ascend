# SPDX-License-Identifier: Apache-2.0
"""Common discovery validation retains its skip and error-priority contract."""

import pytest

from examples.disaggregated_prefill_v1.remote_fill_placement import (
    validate_remote_fill_placement,
)


@pytest.fixture
def placement():
    return {
        "destination_engine_id": " decoder ",
        "control_endpoint": " tcp://decoder:19001 ",
        "token_hash_algorithm": "builtin",
        "descriptor_verification_capability": "ab" * 32,
        "destination_engine_epoch": 7,
        "shared_cache_generation": 0,
        "destination_tp_size": 8,
        "destination_dp_size": 2,
        "global_te_push": True,
        "python_hash_seed": "0",
    }


def test_common_placement_is_fresh_and_preserves_borrowed_input(placement):
    original = dict(placement)
    result = validate_remote_fill_placement(placement, 1)
    assert placement == original
    assert result["destination_engine_id"] == "decoder"
    assert result["control_endpoint"] == "tcp://decoder:19001"
    assert result["destination_dp_rank"] == 1
    result["destination_engine_epoch"] = 99
    assert placement["destination_engine_epoch"] == 7


def test_native_disabled_skips_verification_hash_validation(placement):
    placement.update(
        global_te_push=False,
        descriptor_verification_capability="not-hex",
        python_hash_seed=None,
    )
    assert validate_remote_fill_placement(placement, 0) is None


def test_native_disabled_still_checks_topology_first(placement):
    placement.update(global_te_push=False, destination_tp_size=True)
    with pytest.raises(ValueError, match="parallel topology is invalid"):
        validate_remote_fill_placement(placement, 0)


@pytest.mark.parametrize("key", ["AB" * 32, "ab" * 31, "ab " * 32, "gg" * 32])
def test_capability_must_be_canonical_hex(placement, key):
    placement["descriptor_verification_capability"] = key
    with pytest.raises(ValueError, match="verification capability is invalid"):
        validate_remote_fill_placement(placement, 0)


def test_verification_error_precedes_hash_error(placement):
    placement.update(descriptor_verification_capability="not-hex", python_hash_seed="")
    with pytest.raises(ValueError, match="verification capability is invalid"):
        validate_remote_fill_placement(placement, 0)
