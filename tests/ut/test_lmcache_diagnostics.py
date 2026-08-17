# SPDX-License-Identifier: Apache-2.0
"""Tests for the side-effect-free LMCache diagnostic callback bridge."""

import ast
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from vllm_ascend import lmcache_diagnostics as bridge


@pytest.fixture(autouse=True)
def reset_callbacks() -> Iterator[None]:
    bridge.clear_npu_content_diagnostic_callbacks()
    yield
    bridge.clear_npu_content_diagnostic_callbacks()


def test_bridge_is_disabled_and_noop_by_default() -> None:
    assert bridge.npu_content_diagnostics_enabled() is False
    assert bridge.fingerprint_compact_group1(value="unused") is None
    bridge.begin_deferred_diagnostic_step()
    bridge.flush_deferred_diagnostics()
    bridge.register_group1_source_fingerprint("request", {})
    bridge.queue_group1_first_consume(value="unused")
    bridge.queue_selected_topk_fingerprint(value="unused")


def test_installed_callback_bundle_routes_every_operation() -> None:
    events: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def callback(name: str, result: Any = None) -> Callable[..., Any]:
        def invoke(*args: Any, **kwargs: Any) -> Any:
            events.append((name, args, kwargs))
            return result

        return invoke

    bridge.install_npu_content_diagnostic_callbacks(
        bridge.NPUContentDiagnosticCallbacks(
            begin_deferred_step=callback("begin"),
            flush_deferred=callback("flush"),
            fingerprint_compact_group1=callback("fingerprint", "digest"),
            register_group1_source=callback("register"),
            queue_group1_first_consume=callback("consume"),
            queue_selected_topk=callback("topk"),
        )
    )

    assert bridge.npu_content_diagnostics_enabled() is True
    bridge.begin_deferred_diagnostic_step()
    assert bridge.fingerprint_compact_group1(req_id="request") == "digest"
    bridge.register_group1_source_fingerprint("request", {"hash": "digest"})
    bridge.queue_group1_first_consume(req_ids=["request"])
    bridge.queue_selected_topk_fingerprint(req_ids=["request"])
    bridge.flush_deferred_diagnostics()

    assert [event for event, _, _ in events] == [
        "begin",
        "fingerprint",
        "register",
        "consume",
        "topk",
        "flush",
    ]


@pytest.mark.parametrize(
    "relative_path",
    [
        "vllm_ascend/attention/sfa_v1.py",
        "vllm_ascend/worker/model_runner_v1.py",
        (
            "vllm_ascend/distributed/kv_transfer/kv_p2p/"
            "mooncake_connector.py"
        ),
    ],
)
def test_model_and_transfer_modules_do_not_import_lmcache_ascend(
    relative_path: str,
) -> None:
    """Guard the TorchDynamo startup order against eager plugin imports."""
    repository_root = Path(__file__).resolve().parents[2]
    tree = ast.parse((repository_root / relative_path).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        module == "lmcache_ascend" or module.startswith("lmcache_ascend.")
        for module in imported_modules
    )


def test_live_p2p_first_consume_diagnostic_is_not_persistent_load_gated() -> None:
    """Keep live P2P observable when the persistent index load is disabled."""
    repository_root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (repository_root / "vllm_ascend/attention/sfa_v1.py").read_text(
            encoding="utf-8"
        )
    )
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "queue_group1_first_consume"
    ]
    assert len(calls) == 1
    call = calls[0]
    ancestor = parents.get(call)
    while ancestor is not None:
        if isinstance(ancestor, ast.If):
            assert "index_lmcache_enabled" not in ast.unparse(ancestor.test)
        ancestor = parents.get(ancestor)

    keyword_names = {keyword.arg for keyword in call.keywords}
    assert {
        "seq_lens_cpu",
        "row_request_indices",
        "num_decode_tokens",
        "num_actual_tokens",
        "attn_state",
        "decode_valid_rows_all",
    } <= keyword_names
