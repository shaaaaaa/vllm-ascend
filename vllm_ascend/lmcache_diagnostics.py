# SPDX-License-Identifier: Apache-2.0
"""Side-effect-free bridge for optional LMCache NPU diagnostics.

vLLM-Ascend model modules are imported while the platform and TorchDynamo are
still being initialized.  They must not import ``lmcache_ascend`` directly:
that package installs process-wide NPU compatibility patches at import time.
LMCache registers an immutable callback bundle here after its worker connector
has been initialized.  Until then, every entry point is a cheap no-op.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DiagnosticCallback = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class NPUContentDiagnosticCallbacks:
    """Callbacks owned by an initialized LMCache-Ascend connector."""

    begin_deferred_step: DiagnosticCallback
    flush_deferred: DiagnosticCallback
    fingerprint_compact_group1: DiagnosticCallback
    register_group1_source: DiagnosticCallback
    queue_group1_first_consume: DiagnosticCallback
    queue_cache_tail: DiagnosticCallback
    queue_selected_topk: DiagnosticCallback
    queue_staged_graph_stage: DiagnosticCallback


_CALLBACKS: NPUContentDiagnosticCallbacks | None = None


def install_npu_content_diagnostic_callbacks(
    callbacks: NPUContentDiagnosticCallbacks,
) -> None:
    """Atomically enable the callbacks before inference begins."""
    global _CALLBACKS
    _CALLBACKS = callbacks


def clear_npu_content_diagnostic_callbacks() -> None:
    """Disable all callbacks without importing or retaining LMCache."""
    global _CALLBACKS
    _CALLBACKS = None


def npu_content_diagnostics_enabled() -> bool:
    """Return whether LMCache installed the opt-in diagnostic callbacks."""
    return _CALLBACKS is not None


def begin_deferred_diagnostic_step() -> None:
    callbacks = _CALLBACKS
    if callbacks is not None:
        callbacks.begin_deferred_step()


def flush_deferred_diagnostics() -> None:
    callbacks = _CALLBACKS
    if callbacks is not None:
        callbacks.flush_deferred()


def fingerprint_compact_group1(**kwargs: Any) -> Any:
    callbacks = _CALLBACKS
    if callbacks is None:
        return None
    return callbacks.fingerprint_compact_group1(**kwargs)


def register_group1_source_fingerprint(*args: Any, **kwargs: Any) -> None:
    callbacks = _CALLBACKS
    if callbacks is not None:
        callbacks.register_group1_source(*args, **kwargs)


def queue_group1_first_consume(**kwargs: Any) -> None:
    callbacks = _CALLBACKS
    if callbacks is not None:
        callbacks.queue_group1_first_consume(**kwargs)


def queue_cache_tail_fingerprint(**kwargs: Any) -> None:
    callbacks = _CALLBACKS
    if callbacks is not None:
        callbacks.queue_cache_tail(**kwargs)


def queue_selected_topk_fingerprint(**kwargs: Any) -> None:
    callbacks = _CALLBACKS
    if callbacks is not None:
        callbacks.queue_selected_topk(**kwargs)


def queue_staged_graph_stage_fingerprint(**kwargs: Any) -> None:
    callbacks = _CALLBACKS
    if callbacks is not None:
        callbacks.queue_staged_graph_stage(**kwargs)
