# SPDX-License-Identifier: Apache-2.0
"""Eligibility policy for preparing a selected-prefill step."""

from __future__ import annotations

from collections.abc import Mapping


def selected_prefill_requested(additional_config: Mapping[str, object]) -> bool:
    """Read the explicit opt-in from Ascend's additional configuration."""
    return additional_config.get("enable_selected_prefill") is True


def selected_prefill_ineligibility_reason(
    *,
    requested: bool,
    layerwise_prefill_p_node: bool,
    enforce_eager: bool,
    tensor_parallel_size: int,
    kv_group_count: int,
    dsa_shared_pool: bool,
    is_lmcache_connector: bool = False,
    main_kv_layout_supported: bool = False,
    registered_source_ready: bool = False,
    bank_writer_ready: bool = False,
    reader_retirement_ready: bool = False,
) -> str | None:
    """Return the first reason to retain the native path, or allow preparation.

    The caller supplies verified connector and layout facts before starting
    any transfer. Layout support requires checking the actual main KV buffers.
    Indexer C8 is independent, and this target lacks M3v19 main C8 support.
    Source readiness includes support for the active connector world size.
    An eligible step still needs validated source coverage and bank ownership
    before any native load can be omitted.
    """
    if requested is not True:
        return "Selected prefill is disabled."
    if layerwise_prefill_p_node is not True:
        return "Selected prefill requires a layerwise prefill P node."
    if enforce_eager is not True:
        return "Selected prefill requires eager execution."
    if tensor_parallel_size != 1:
        return "Selected prefill currently requires tensor parallel size 1."
    if kv_group_count != 2 or dsa_shared_pool is not True:
        return "Selected prefill requires two KV groups in the DSA shared pool."
    if is_lmcache_connector is not True:
        return "Selected prefill requires an LMCache connector."
    if main_kv_layout_supported is not True:
        return "Selected prefill requires a verified main KV layout."
    if registered_source_ready is not True:
        return "Selected prefill requires registered LMCache sources."
    if bank_writer_ready is not True:
        return "Selected prefill requires the LMCache bank writer."
    if reader_retirement_ready is not True:
        return "Selected prefill requires LMCache reader retirement."
    return None


def packed_main_c8_enabled(kv_cache_spec: object) -> bool:
    """Return the packed-main C8 marker, which must be an explicit boolean.

    Ordinary specs omit the marker and default to disabled. Callers that fake
    a spec must set the marker explicitly instead of relying on mock children.
    """
    marker = getattr(kv_cache_spec, "cache_sparse_sfa_c8", False)
    if type(marker) is not bool:
        raise ValueError("Packed main C8 marker must be a boolean")
    return marker
